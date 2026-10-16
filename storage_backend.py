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
  GH_APP_ID, GH_APP_PRIVATE_KEY (+ optional GH_APP_INSTALLATION_ID)
                 preferred: GitHub App with "Contents: read and write"; never
                 expires (see GITHUB_APP_SETUP.md). Takes priority when set.
  GITHUB_TOKEN   fallback: fine-grained PAT, "Contents: read and write"
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

def _raw_cfg(key: str, default: str = "") -> str:
    """Streamlit secrets first, then environment. Never raises."""
    if _HAS_ST:
        try:
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
    return os.environ.get(key, default)


def _cfg(key: str, default: str = "") -> str:
    """Same as _raw_cfg, except GITHUB_TOKEN resolves through github_token().

    Every caller in this codebase (here, log_viewer, swing_tab, publish
    scripts) asks for _cfg("GITHUB_TOKEN"). Routing that one key through the
    GitHub App means none of them had to change.
    """
    if key == "GITHUB_TOKEN":
        return github_token() or default
    return _raw_cfg(key, default)


# ── GitHub App auth (v2 — Oct 2026) ───────────────────────────────────────────
# A fine-grained PAT expires (max one year) and every app silently drops to
# ephemeral storage the day it does. A GitHub App's private key never expires;
# it is used to mint a one-hour installation token on demand, cached here and
# refreshed five minutes before expiry.
#
# Secrets (Streamlit secrets or environment):
#   GH_APP_ID               numeric App ID from the app's settings page
#   GH_APP_PRIVATE_KEY      full contents of the downloaded .pem file
#   GH_APP_INSTALLATION_ID  optional; discovered from GITHUB_REPO when unset
#
# When the App secrets are absent, or minting fails, GITHUB_TOKEN (the old PAT)
# is used exactly as before. Nothing breaks if the App is half set up.

import threading
import time as _time

_APP_LOCK = threading.Lock()
_APP_CACHE: dict = {"token": "", "expires": 0.0, "installation": "",
                    "retry_after": 0.0}
_APP_STATE: dict = {"source": "", "error": ""}
_APP_REFRESH_S = 300          # refresh when under 5 minutes of life remains
_APP_RETRY_S = 60             # after a failed mint, wait before trying again


def _app_configured() -> bool:
    return bool(_raw_cfg("GH_APP_ID").strip() and _raw_cfg("GH_APP_PRIVATE_KEY").strip())


def _app_private_key() -> str:
    key = _raw_cfg("GH_APP_PRIVATE_KEY").strip()
    if "\\n" in key and "\n" not in key:          # pasted as one line with literal \n
        key = key.replace("\\n", "\n")
    return key


def _app_jwt() -> str:
    import jwt                                      # PyJWT; needs `cryptography`
    now = int(_time.time())
    payload = {"iat": now - 60, "exp": now + 540,  # GitHub allows at most 10 min
               "iss": _raw_cfg("GH_APP_ID").strip()}
    tok = jwt.encode(payload, _app_private_key(), algorithm="RS256")
    return tok.decode() if isinstance(tok, bytes) else tok


def _app_installation_id(app_jwt: str) -> str:
    inst = _raw_cfg("GH_APP_INSTALLATION_ID").strip()
    if inst:
        return inst
    if _APP_CACHE["installation"]:
        return _APP_CACHE["installation"]
    h = _gh_headers(app_jwt)
    repo = _raw_cfg("GITHUB_REPO").strip()
    if repo:
        r = requests.get(f"{GITHUB_API}/repos/{repo}/installation", headers=h, timeout=20)
        if r.status_code == 200:
            return str(r.json()["id"])
    r = requests.get(f"{GITHUB_API}/app/installations", headers=h, timeout=20)
    r.raise_for_status()
    insts = r.json()
    if len(insts) != 1:
        raise RuntimeError(f"app has {len(insts)} installations; set GH_APP_INSTALLATION_ID")
    return str(insts[0]["id"])


def _mint_app_token() -> tuple[str, float]:
    app_jwt = _app_jwt()
    inst = _app_installation_id(app_jwt)
    r = requests.post(f"{GITHUB_API}/app/installations/{inst}/access_tokens",
                      headers=_gh_headers(app_jwt), timeout=20)
    r.raise_for_status()
    j = r.json()
    exp = datetime.strptime(j["expires_at"], "%Y-%m-%dT%H:%M:%SZ")
    exp_epoch = (exp - datetime(1970, 1, 1)).total_seconds()
    _APP_CACHE["installation"] = inst
    return j["token"], exp_epoch


def github_token() -> str:
    """Token for the GitHub contents API: App installation token when the App
    is configured and minting works, otherwise the static GITHUB_TOKEN."""
    if _app_configured() and _HAS_REQ:
        with _APP_LOCK:
            now = _time.time()
            if _APP_CACHE["token"] and _APP_CACHE["expires"] - now > _APP_REFRESH_S:
                _APP_STATE.update(source="app", error="")
                return _APP_CACHE["token"]
            if now >= _APP_CACHE["retry_after"]:
                try:
                    tok, exp = _mint_app_token()
                    _APP_CACHE.update(token=tok, expires=exp, retry_after=0.0)
                    _APP_STATE.update(source="app", error="")
                    return tok
                except Exception as e:
                    _APP_CACHE["retry_after"] = now + _APP_RETRY_S
                    _APP_STATE["error"] = f"{type(e).__name__}: {e}"
                    print(f"[storage] GitHub App token mint failed: {_APP_STATE['error']}")
            if _APP_CACHE["token"] and _APP_CACHE["expires"] > now:
                _APP_STATE["source"] = "app"           # stale-but-valid beats nothing
                return _APP_CACHE["token"]
    pat = _raw_cfg("GITHUB_TOKEN")
    _APP_STATE["source"] = "pat" if pat else ""
    return pat


def auth_status() -> dict:
    """Which credential is in use, for the UI and the publish logs."""
    tok = github_token()
    src = _APP_STATE["source"]
    label = {"app": "GitHub App (auto-renewing)",
             "pat": "personal access token (expires)"}.get(src, "none")
    if _app_configured() and src != "app":
        label += " — GitHub App configured but failed: " + (_APP_STATE["error"] or "unknown")
    return {"source": src or "none", "label": label, "has_token": bool(tok),
            "app_configured": _app_configured(), "error": _APP_STATE["error"]}


def backend_status() -> dict:
    """What storage is actually active, and why. Surfaced in the UI so the
    durability situation is never a surprise."""
    token, repo = _cfg("GITHUB_TOKEN"), _cfg("GITHUB_REPO")
    if token and repo and _HAS_REQ:
        auth = auth_status()
        return {"backend": "github", "durable": True, "target": repo,
                "auth": auth["source"],
                "detail": f"Commits to {repo}@{_cfg('GITHUB_BRANCH', 'main')} — "
                          f"durable and versioned. Auth: {auth['label']}."}
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
                        "Round-trip OK but NOT durable — set the GitHub App secrets (or GITHUB_TOKEN) and GITHUB_REPO."
                        if ok else
                        "FAILED — check token scope (needs Contents: read and write) "
                        "and that GITHUB_REPO is 'owner/repo'.")}
