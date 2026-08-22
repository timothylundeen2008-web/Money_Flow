"""
fred_client.py  (v1 — July 2026)
──────────────────────────────────────────────────────────────────────────────
FRED access that works WITH or WITHOUT an API key.

THE PROBLEM THIS SOLVES
  regime_classifier._inline_fetch_fred() opens with:

      if requests is None or not api_key:
          return pd.Series(dtype=float)

  So with no key, every FRED series returns EMPTY before a request is even
  attempted — and because the entire SignalSet is built from FRED, the whole
  Level-1 layer goes dark at once: HY OAS, DFII10, 2s10s, the short real rate,
  CPI, the regime key, the Fed reaction flag, the repression score, and the
  KMLM stance. Every one of them renders "unavailable", which looks like a bug
  but is just an unset key.

  Worse, the key lives in a sidebar st.text_input, so it has to be re-typed
  EVERY session. Miss it and the dashboard silently downgrades to a chart
  viewer with no regime read at all.

THE FIX
  FRED publishes the same data through two endpoints:
    api.stlouisfed.org/fred/series/observations   requires a key, JSON
    fred.stlouisfed.org/graph/fredgraph.csv       NO KEY, CSV

  The CSV endpoint is what rep_data_fetcher.py has always used as its fallback,
  and what treasury_data.py uses for the curve signal. This module wraps both:
  API when a key is present (higher rate limits, revision metadata), keyless
  CSV otherwise — so the dashboard degrades to "slightly slower" instead of
  "completely blind".

  Signature matches _inline_fetch_fred exactly, so it drops straight into
  compute_signals(fetch_fred=...) / full_assessment(fetch_fred=...).

ON HARDCODING THE KEY
  Don't. This file is pushed to GitHub, and a committed key is a leaked key —
  git history keeps it even after you delete the line, and GitHub's secret
  scanner will flag it. Use Streamlit secrets instead: same result, no
  re-typing, no leak. See get_api_key() below and DEPLOYMENT.md.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd

try:
    import requests
except Exception:
    requests = None

try:
    import streamlit as st
    _cache = st.cache_data(ttl=3600, show_spinner=False)
    _HAS_ST = True
except Exception:
    _HAS_ST = False
    def _cache(fn):
        return fn

import os

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS = {"User-Agent": "AllWeatherDashboard/1.0 (research)"}


def get_api_key() -> str:
    """
    Resolve the FRED key WITHOUT requiring it to be typed each session.

    Order: Streamlit secrets → environment → "" (keyless CSV fallback).

    To set it once and forget it, add to Streamlit secrets:
        FRED_API_KEY = "your32charkey"

    Never hardcode it in a source file that gets committed.
    """
    if _HAS_ST:
        try:
            if "FRED_API_KEY" in st.secrets:
                return str(st.secrets["FRED_API_KEY"]).strip()
        except Exception:
            pass
    return os.environ.get("FRED_API_KEY", "").strip()


def _via_api(series_id: str, api_key: str, start: str) -> pd.Series:
    try:
        r = requests.get(FRED_API, timeout=20, headers=_HEADERS, params={
            "series_id": series_id, "api_key": api_key,
            "file_type": "json", "observation_start": start})
        r.raise_for_status()
        obs = r.json().get("observations", [])
        idx, vals = [], []
        for o in obs:
            v = o.get("value", ".")
            if v not in (".", "", None):
                idx.append(pd.Timestamp(o["date"]))
                vals.append(float(v))
        if not idx:
            return pd.Series(dtype=float)
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name=series_id)
    except Exception as e:
        print(f"[fred] API failed for {series_id}: {e}")
        return pd.Series(dtype=float)


def _via_csv(series_id: str, start: str) -> pd.Series:
    """Keyless. Same data, CSV transport."""
    try:
        r = requests.get(FRED_CSV, timeout=25, headers=_HEADERS,
                         params={"id": series_id, "cosd": start})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if df.shape[1] < 2:
            return pd.Series(dtype=float)
        dcol, vcol = df.columns[0], df.columns[1]
        df[vcol] = pd.to_numeric(df[vcol], errors="coerce")   # FRED uses "." for NA
        s = pd.Series(df[vcol].values, index=pd.to_datetime(df[dcol]),
                      name=series_id).dropna()
        return s
    except Exception as e:
        print(f"[fred] CSV failed for {series_id}: {e}")
        return pd.Series(dtype=float)


def _fetch_fred_uncached(series_id: str, api_key: str = "",
                         start: str = "2015-01-01") -> pd.Series:
    """
    Drop-in replacement for regime_classifier._inline_fetch_fred.

    Identical signature, so it can be injected directly:
        full_assessment(fred_api_key=key, fetch_fred=fred_client.fetch_fred)

    Difference: an empty key falls back to the keyless CSV endpoint instead of
    returning an empty Series. If the API path fails for any reason (bad key,
    rate limit, outage) it ALSO falls back rather than giving up — a wrong key
    should degrade to slower, not to blind.
    """
    if requests is None:
        return pd.Series(dtype=float)

    if api_key:
        s = _via_api(series_id, api_key, start)
        if not s.empty:
            return s
        print(f"[fred] API returned nothing for {series_id}; trying keyless CSV")

    return _via_csv(series_id, start)


# Cache only SUCCESSFUL fetches. A transient failure returns empty but is not
# cached, so the next rerun retries instead of serving a poisoned empty Series
# for the whole TTL — which would look exactly like "unavailable that won't fix
# itself even though the network recovered".
_fred_mem: dict = {}


@_cache
def _fetch_fred_cached(series_id: str, api_key: str, start: str) -> pd.Series:
    return _fetch_fred_uncached(series_id, api_key, start)


def fetch_fred(series_id: str, api_key: str = "",
               start: str = "2015-01-01") -> pd.Series:
    """
    Drop-in replacement for regime_classifier._inline_fetch_fred, with a
    non-poisoning cache: successful pulls are cached for the TTL, failures are
    not, so recovery is automatic on the next rerun.
    """
    key_present = bool(api_key)
    try:
        s = _fetch_fred_cached(series_id, api_key, start)
        if s is not None and not s.empty:
            return s
    except Exception as e:
        print(f"[fred] cached path error for {series_id}: {e}")
    # Cache returned/raised empty — retry uncached so a recovered network works
    return _fetch_fred_uncached(series_id, api_key, start)


def status(api_key: str = "") -> dict:
    """What mode are we in, and does it actually work? Surfaced in the UI."""
    key = api_key or get_api_key()
    probe = fetch_fred("DGS10", key, start="2025-01-01")
    ok = not probe.empty
    return {
        "mode": "api" if key else "keyless-csv",
        "key_present": bool(key),
        "working": ok,
        "rows": len(probe),
        "latest": (probe.index[-1].date().isoformat() if ok else None),
        "message": (
            f"FRED via authenticated API — {len(probe)} rows, latest "
            f"{probe.index[-1].date()}." if ok and key else
            f"FRED via KEYLESS CSV — {len(probe)} rows, latest "
            f"{probe.index[-1].date()}. Works fine; a free key at "
            f"fredaccount.stlouisfed.org raises rate limits." if ok else
            "FRED unreachable on BOTH endpoints. Check network egress — every "
            "Level-1 signal depends on this."),
    }
