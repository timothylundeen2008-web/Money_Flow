"""
storage_backend.py  (v1 — July 2026)   ── CLOSES GAP G0
──────────────────────────────────────────────────────────────────────────────
Durable storage for a platform that has none.

THE PROBLEM
  Streamlit Community Cloud has an EPHEMERAL filesystem. Anything written to
  disk is lost on redeploy, on container restart, and when the app sleeps after
  inactivity. Nothing in this codebase ever persisted state before now — every
  existing to_csv is a download button — so this failure mode had never bitten.

  It bites hardest on exactly the data you cannot recreate. The ETF flow poll
  (etf_flow_tracker.py) records shares outstanding, which free sources expose
  ONLY as a current snapshot. That history cannot be backfilled at any price.
  Polling into an ephemeral filesystem means accumulating six weeks of the one
  dataset you can never get back, then losing all of it on a redeploy.

THE FIX — three backends behind one interface, auto-selected
  github   Commits to your repo via the GitHub contents API. Durable,
           versioned, free, diffable. Requires GITHUB_TOKEN + GITHUB_REPO.
           ⭐ Use this for anything that accumulates over time.
  local    Plain file I/O. Correct on a VPS/laptop, ephemeral on Cloud.
  session  Streamlit session_state. Survives reruns, dies with the session.

  read()/write() work identically across all three, so callers never branch on
  backend. Failures degrade DOWN the list (github → local → session) and are
  reported, never silent.

CONFIGURATION (Streamlit secrets, or environment variables)
  GITHUB_TOKEN   fine-grained PAT, "Contents: read and write" on ONE repo
  GITHUB_REPO    "username/all-weather-dashboard"
  GITHUB_BRANCH  default "main"
  DATA_DIR       default "data"

SECURITY
  Scope the token to a single repo and Contents-only. It can then commit data
  files and nothing else — it cannot read other repos, touch Actions, or open
  pull requests. Never commit the token itself; use Streamlit secrets.
"""

from __future__ import annotations

import base64
import io
import json
import os
from datetime import datetime

import pandas as pd

try:
    import streamlit as st
    _HAS_ST = True
except Exception:                                    # usable outside Streamlit
    _HAS_ST = False

try:
    import requests
    _HAS_REQ = True
except Exception:
    _HAS_REQ = False


GITHUB_API = "https://api.github.com"


# ── Configuration ─────────────────────────────────────────────────────────────

def _cfg(key: str, default: str = "") -> str:
    """Streamlit secrets first, then environment. Never raises."""
    if _HAS_ST:
        try:
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
    return os.environ.get(key, default)


def backend_status() -> dict:
    """What storage is actually active, and why. Surfaced in the UI so the
    durability situation is never a surprise."""
    token, repo = _cfg("GITHUB_TOKEN"), _cfg("GITHUB_REPO")
    if token and repo and _HAS_REQ:
        return {"backend": "github", "durable": True, "target": repo,
                "detail": f"Commits to {repo}@{_cfg('GITHUB_BRANCH', 'main')} — "
                          f"durable and versioned."}
    if token and not repo:
        return {"backend": "local", "durable": False, "target": _cfg("DATA_DIR", "data"),
                "detail": "GITHUB_TOKEN set but GITHUB_REPO missing — falling back to "
                          "local files, which are EPHEMERAL on Streamlit Cloud."}
    return {"backend": "local", "durable": False, "target": _cfg("DATA_DIR", "data"),
            "detail": "No GitHub credentials. Local files are EPHEMERAL on Streamlit "
                      "Cloud — they vanish on redeploy, restart, or sleep. Set "
                      "GITHUB_TOKEN and GITHUB_REPO in secrets to make storage durable."}


# ── GitHub backend ────────────────────────────────────────────────────────────

def _gh_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def _gh_get(path: str) -> tuple[str | None, str | None]:
    """Returns (content, sha). sha is required to update an existing file."""
    token, repo = _cfg("GITHUB_TOKEN"), _cfg("GITHUB_REPO")
    branch = _cfg("GITHUB_BRANCH", "main")
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    try:
        r = requests.get(url, headers=_gh_headers(token),
                         params={"ref": branch}, timeout=30)
        if r.status_code == 404:
            return None, None                       # not an error: first write
        r.raise_for_status()
        j = r.json()
        content = base64.b64decode(j["content"]).decode("utf-8")
        return content, j["sha"]
    except Exception as e:
        print(f"[storage] github read failed for {path}: {e}")
        return None, None


def _gh_put(path: str, content: str, message: str) -> bool:
    token, repo = _cfg("GITHUB_TOKEN"), _cfg("GITHUB_REPO")
    branch = _cfg("GITHUB_BRANCH", "main")
    _, sha = _gh_get(path)                           # need current sha to update
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    body = {"message": message, "branch": branch,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii")}
    if sha:
        body["sha"] = sha
    try:
        r = requests.put(url, headers=_gh_headers(token), json=body, timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[storage] github write failed for {path}: {e}")
        return False


# ── Public interface ──────────────────────────────────────────────────────────

def read_text(name: str) -> str | None:
    """Read a stored file by logical name (e.g. 'checklist_log.csv')."""
    data_dir = _cfg("DATA_DIR", "data")
    path = f"{data_dir}/{name}"
    status = backend_status()

    if status["backend"] == "github":
        content, _ = _gh_get(path)
        if content is not None:
            return content

    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"[storage] local read failed: {e}")

    if _HAS_ST:
        return st.session_state.get(f"_store_{name}")
    return None


def write_text(name: str, content: str, message: str | None = None) -> dict:
    """
    Write through every available layer. Returns a result dict describing what
    actually succeeded, so the UI can tell the user whether the data is safe.
    """
    data_dir = _cfg("DATA_DIR", "data")
    path = f"{data_dir}/{name}"
    msg = message or f"data: update {name} ({datetime.now():%Y-%m-%d %H:%M})"
    res = {"github": False, "local": False, "session": False, "durable": False}

    if backend_status()["backend"] == "github":
        res["github"] = _gh_put(path, content, msg)
        res["durable"] = res["github"]

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        res["local"] = True
    except Exception as e:
        print(f"[storage] local write failed (expected on read-only FS): {e}")

    if _HAS_ST:
        st.session_state[f"_store_{name}"] = content
        res["session"] = True

    return res


def read_df(name: str) -> pd.DataFrame:
    """CSV convenience. Returns an EMPTY DataFrame when absent — never raises,
    so a first run and a failed read look the same to callers."""
    txt = read_text(name)
    if not txt:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(txt))
    except Exception as e:
        print(f"[storage] parse failed for {name}: {e}")
        return pd.DataFrame()


def write_df(name: str, df: pd.DataFrame, message: str | None = None) -> dict:
    return write_text(name, df.to_csv(index=False), message)


def append_row(name: str, row: dict, dedupe_on: list[str] | None = None) -> pd.DataFrame:
    """
    Append one row, optionally replacing any existing row matching on
    `dedupe_on`. Idempotent by design: a scheduled job that fires twice, or a
    user who re-runs a checklist, must not create duplicates.
    """
    df = read_df(name)
    new = pd.DataFrame([row])
    if not df.empty and dedupe_on and all(c in df.columns for c in dedupe_on):
        mask = pd.Series(True, index=df.index)
        for c in dedupe_on:
            mask &= (df[c].astype(str) == str(row.get(c)))
        df = df[~mask]
    out = pd.concat([df, new], ignore_index=True)
    if "date" in out.columns:
        out = out.sort_values("date")
    write_df(name, out)
    return out


def read_json(name: str) -> dict | None:
    txt = read_text(name)
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception as e:
        print(f"[storage] json parse failed for {name}: {e}")
        return None


def write_json(name: str, obj: dict, message: str | None = None) -> dict:
    return write_text(name, json.dumps(obj, indent=2, default=str), message)


def selftest() -> dict:
    """
    Verify the configured backend actually works, end to end. Run this once
    after deployment — a token with the wrong scope fails silently on write
    otherwise, and you would not find out until you needed the history.
    """
    name = "_storage_selftest.json"
    stamp = datetime.now().isoformat(timespec="seconds")
    w = write_json(name, {"stamp": stamp, "note": "storage_backend selftest"})
    back = read_json(name)
    ok = bool(back and back.get("stamp") == stamp)
    return {"status": backend_status(), "write_result": w, "roundtrip_ok": ok,
            "message": ("Storage verified — data will survive redeploys."
                        if ok and w.get("durable") else
                        "Round-trip OK but NOT durable — set GITHUB_TOKEN/GITHUB_REPO."
                        if ok else
                        "FAILED — check token scope (needs Contents: read and write) "
                        "and that GITHUB_REPO is 'owner/repo'.")}
