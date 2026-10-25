"""
fred_client.py  (v1 — July 2026; v1.1 Oct 2026: retries, outage breaker, last-good memory)
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
from typing import Optional

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

import time as _time

# ── v1.1 resilience (Oct 2026) ─────────────────────────────────────────────
# On 2026-10-21 every FRED series came back empty in the nightly run and the
# regime classifier fell through to `neutral`. One transient failure used to
# mean one empty series: there was no retry. Now:
#   * each HTTP call retries on connection errors, timeouts, 429 and 5xx
#     (RETRY_WAITS seconds between attempts); other 4xx fail at once
#   * after OUTAGE_AFTER consecutive series fail on BOTH endpoints, FRED is
#     treated as down for OUTAGE_COOLDOWN seconds and calls return empty
#     immediately, so a full outage costs seconds, not minutes per series
#   * the last good pull of each series is kept in memory and returned
#     (and reported as stale via last_good_age()) when a refetch fails
# The classifier's own data-failure guard (regime_classifier v9) handles
# whatever still comes back empty.
RETRY_WAITS = (1.5, 4.0)
OUTAGE_AFTER = 4
OUTAGE_COOLDOWN = 300
_state = {"consecutive_failures": 0, "down_until": 0.0, "last_error": ""}
_last_good: dict = {}       # (series_id, start) -> (fetched_epoch, Series)


def _http_get(url: str, params: dict, timeout: int) -> "requests.Response":
    """GET with bounded retries on transient failures. Raises on final failure."""
    last = None
    for i in range(len(RETRY_WAITS) + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=_HEADERS, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code}")
            else:
                r.raise_for_status()          # other 4xx: not transient, no retry
                return r
        except requests.exceptions.HTTPError:
            raise
        except Exception as e:                # connection error, timeout, DNS
            last = e
        if i < len(RETRY_WAITS):
            _time.sleep(RETRY_WAITS[i])
    raise last if last else RuntimeError("request failed")


def outage_status() -> dict:
    """Is FRED currently treated as down? For the UI and the brief's data health."""
    now = _time.time()
    return {"down": now < _state["down_until"],
            "seconds_left": max(0, int(_state["down_until"] - now)),
            "consecutive_failures": _state["consecutive_failures"],
            "last_error": _state["last_error"]}


def last_good_age(series_id: str, start: str = "2015-01-01") -> Optional[float]:
    """Seconds since the last successful pull of this series in this process."""
    v = _last_good.get((series_id, start))
    return None if v is None else _time.time() - v[0]


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
        r = _http_get(FRED_API, {"series_id": series_id, "api_key": api_key,
                                 "file_type": "json", "observation_start": start}, 20)
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
        r = _http_get(FRED_CSV, {"id": series_id, "cosd": start}, 25)
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
    if _time.time() < _state["down_until"]:
        return _fallback_last_good(series_id, start)

    s = pd.Series(dtype=float)
    if api_key:
        s = _via_api(series_id, api_key, start)
        if s.empty:
            print(f"[fred] API returned nothing for {series_id}; trying keyless CSV")
    if s.empty:
        s = _via_csv(series_id, start)

    if not s.empty:
        _state["consecutive_failures"] = 0
        _last_good[(series_id, start)] = (_time.time(), s)
        return s
    _state["consecutive_failures"] += 1
    _state["last_error"] = f"{series_id}: both endpoints empty"
    if _state["consecutive_failures"] >= OUTAGE_AFTER:
        _state["down_until"] = _time.time() + OUTAGE_COOLDOWN
        print(f"[fred] {OUTAGE_AFTER} series in a row failed on both endpoints — treating FRED "
              f"as DOWN for {OUTAGE_COOLDOWN}s")
    return _fallback_last_good(series_id, start)


def _fallback_last_good(series_id: str, start: str) -> pd.Series:
    v = _last_good.get((series_id, start))
    if v is not None:
        print(f"[fred] serving last good {series_id} ({int(_time.time() - v[0])}s old)")
        return v[1]
    return pd.Series(dtype=float)


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
    # Cache returned/raised empty — retry uncached so a recovered network works,
    # unless FRED is already marked down (then don't spend another round trip).
    if _time.time() < _state["down_until"]:
        return _fallback_last_good(series_id, start)
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


def selftest() -> dict:
    """Retry, breaker and last-good behaviour with a fake transport (no network)."""
    import types
    f = []
    global requests
    real = requests
    calls = {"n": 0}

    class _Resp:
        def __init__(self, code, text=""):
            self.status_code, self.text = code, text
        def raise_for_status(self):
            if self.status_code >= 400:
                raise real.exceptions.HTTPError(f"HTTP {self.status_code}")
        def json(self):
            return {"observations": [{"date": "2026-10-20", "value": "1.0"}]}

    def fake(seq):
        it = iter(seq)
        def get(url, **k):
            calls["n"] += 1
            x = next(it)
            if isinstance(x, Exception):
                raise x
            return _Resp(x, "DATE,X\n2026-10-20,1.0\n")
        return types.SimpleNamespace(get=get, exceptions=real.exceptions)

    waits, _sleep = RETRY_WAITS, _time.sleep
    try:
        globals()["RETRY_WAITS"] = (0.0, 0.0)
        _time.sleep = lambda s: None
        _state.update(consecutive_failures=0, down_until=0.0)
        # transient 503 then success -> one series, no failure counted
        requests = fake([503, 200])
        s = _fetch_fred_uncached("TEST1", "", "2015-01-01")
        if s.empty or calls["n"] != 2:
            f.append(f"must retry a 503 and succeed: rows={len(s)} calls={calls['n']}")
        # 400 is not retried
        calls["n"] = 0
        requests = fake([400])
        try:
            _http_get("u", {}, 1); f.append("400 must raise")
        except Exception:
            pass
        if calls["n"] != 1:
            f.append(f"400 must not be retried (calls={calls['n']})")
        # last-good served when a refetch fails
        requests = fake([real.exceptions.ConnectionError("x")] * 3)
        lg = _fetch_fred_uncached("TEST1", "", "2015-01-01")
        if lg.empty:
            f.append("a failed refetch must serve the last good series")
        # breaker opens after OUTAGE_AFTER consecutive total failures
        _state.update(consecutive_failures=0, down_until=0.0)
        requests = fake([real.exceptions.ConnectionError("x")] * 3 * OUTAGE_AFTER)
        for i in range(OUTAGE_AFTER):
            _fetch_fred_uncached(f"NEW{i}", "", "2015-01-01")
        if not outage_status()["down"]:
            f.append("breaker must open after consecutive total failures")
        calls["n"] = 0
        _fetch_fred_uncached("NEWX", "", "2015-01-01")
        if calls["n"] != 0:
            f.append("while down, calls must return without touching the network")
    finally:
        requests = real
        globals()["RETRY_WAITS"] = waits
        _time.sleep = _sleep
        _state.update(consecutive_failures=0, down_until=0.0, last_error="")
        _last_good.clear()
    return {"ok": not f, "failures": f}


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(selftest(), indent=2))
