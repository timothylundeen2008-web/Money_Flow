"""
macro_flags.py  (v1 — Sept 2026)
──────────────────────────────────────────────────────────────────────────────
ONE source for the four regime inputs the classifier cannot fetch itself:

    fed_bs_expanding          LIVE  — FRED WALCL, 4-week average weekly change
    top20_concentration_pct   LIVE  — sum of the 20 largest weights in SPY's
                                      daily holdings file (State Street)
    cape                      LIVE  — Shiller CAPE from multpl.com
    deficit_gt_5pct_gdp       MANUAL — annual fiscal fact, dated below

Every LIVE input falls back to the dated MANUAL value when its fetch fails,
and says so. Every value carries its source, as-of date, age and a stale
flag, and get_flags() returns plain-language warnings for anything stale or
fallen back — so a regime call can never again rest on a hand-typed number
without the reader seeing how old it is.

WHY THIS EXISTS
  Until Sept 2026 these four were hand-typed constants, duplicated in
  app.py, auto_log.py (x3 repos) and checklist_tab.py. The valuation guard
  that routes a goldilocks read into `transition_ambiguous` was running on a
  CAPE typed in on 2026-08-10 and a concentration figure from 2026-08-07 —
  past the checklist's own 45-day staleness rule — while FED_BS_EXPANDING was
  hard-set True even though the H.4.1 tab computes the posture live.

  This file is IDENTICAL in all three repos. Change it in one, copy to all.

UPDATING THE MANUAL VALUES
  Edit MANUAL below: change `value` AND `asof` together. The stale flag is
  computed from `asof`, so forgetting the date is what gets flagged.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Callable, Optional

import pandas as pd

# ── Manual values (fallbacks for the live inputs; primary for the deficit) ──
MANUAL = {
    "cape": {
        "value": 42.0, "asof": "2026-08-10", "source": "multpl.com (manual entry)",
        "stale_after_days": 35,
    },
    "top20_concentration_pct": {
        "value": 50.8, "asof": "2026-08-07", "source": "JPMorgan, cited in 2026-08-07 review (manual entry)",
        "stale_after_days": 35,
    },
    "deficit_gt_5pct_gdp": {
        "value": True, "asof": "2026-08-13",
        "source": "FY2026 deficit $1.9tn = 5.8% of GDP (manual entry)",
        "stale_after_days": 120,
    },
    "fed_bs_expanding": {
        "value": True, "asof": "2026-08-13",
        "source": "reserve-management T-bill purchases (manual entry)",
        "stale_after_days": 21,
    },
}

# Fed posture thresholds — same bands as the Markets dashboard H.4.1 tab.
# Reserve-management growth counts as EXPANSION for the repression score
# (duration-free liquidity added); flat and QT do not.
RESERVE_MGMT_WEEKLY_BN = 5.0
QE_WEEKLY_BN = 20.0
QT_WEEKLY_BN = -5.0
WALCL_MAX_AGE_DAYS = 14          # weekly series; older than 2 prints = stale

SPY_HOLDINGS_URL = ("https://www.ssga.com/library-content/products/fund-data/"
                    "etfs/us/holdings-daily-us-en-spy.xlsx")
CAPE_URL = "https://www.multpl.com/shiller-pe"
CAPE_PLAUSIBLE = (5.0, 80.0)
TOP20_PLAUSIBLE = (15.0, 80.0)

CACHE_TTL_SECONDS = 6 * 3600
_CACHE: dict = {}

_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124 Safari/537.36"),
       "Accept-Language": "en-US,en;q=0.9"}


# ── helpers ─────────────────────────────────────────────────────────────────

def _today() -> date:
    try:
        import market_time as mt
        return mt.et_date()
    except Exception:
        return date.today()


def _age_days(asof: str | date | None, today: Optional[date] = None) -> Optional[int]:
    if not asof:
        return None
    d = pd.Timestamp(asof).date()
    return ((today or _today()) - d).days


def _manual(name: str, today: Optional[date] = None, why: str = "") -> dict:
    m = MANUAL[name]
    age = _age_days(m["asof"], today)
    return {"value": m["value"], "source": m["source"], "asof": m["asof"],
            "age_days": age, "stale": age is not None and age > m["stale_after_days"],
            "live": False, "detail": why}


def _get(url: str, timeout: int = 30):
    import requests
    r = requests.get(url, timeout=timeout, headers=_UA)
    r.raise_for_status()
    return r


# ── live inputs ─────────────────────────────────────────────────────────────

def fed_posture_from_walcl(walcl: pd.Series, today: Optional[date] = None) -> dict:
    """Classify the Fed balance sheet from WALCL ($ millions, weekly).

    Returns {"expanding", "state", "wk4_bn", "wk13_bn", "asof"}; raises
    ValueError when there is too little or too old data to judge."""
    s = pd.Series(walcl).dropna()
    if len(s) < 5:
        raise ValueError(f"WALCL has {len(s)} observations; need 5+ for a 4-week average")
    asof = pd.Timestamp(s.index[-1]).date()
    age = ((today or _today()) - asof).days
    if age > WALCL_MAX_AGE_DAYS:
        raise ValueError(f"WALCL last print {asof} is {age} days old")
    wk4 = (s.iloc[-1] - s.iloc[-5]) / 4 / 1000.0
    wk13 = (s.iloc[-1] - s.iloc[-14]) / 13 / 1000.0 if len(s) >= 14 else None
    if wk4 >= QE_WEEKLY_BN and (wk13 is None or wk13 > 0):
        state = "QE expansion"
    elif wk4 >= RESERVE_MGMT_WEEKLY_BN:
        state = "reserve-management growth"
    elif wk4 > QT_WEEKLY_BN:
        state = "flat / neutral"
    else:
        state = "QT runoff"
    return {"expanding": state in ("QE expansion", "reserve-management growth"),
            "state": state, "wk4_bn": round(float(wk4), 1),
            "wk13_bn": None if wk13 is None else round(float(wk13), 1),
            "asof": asof.isoformat()}


def top20_from_holdings(raw: bytes) -> tuple[float, int]:
    """Sum the 20 largest weights in an SSGA daily-holdings workbook.
    Returns (pct, n_holdings). Raises ValueError on an unrecognised layout."""
    import io
    grid = pd.read_excel(io.BytesIO(raw), header=None)
    hdr = None
    for i in range(min(len(grid), 40)):
        cells = [str(c).strip().lower() for c in grid.iloc[i].tolist()]
        if "weight" in cells and any(c in ("ticker", "name", "identifier") for c in cells):
            hdr = i
            break
    if hdr is None:
        raise ValueError("no header row with 'Weight' and 'Ticker'/'Name' in the first 40 rows")
    df = grid.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip().lower() for c in grid.iloc[hdr].tolist()]
    w = pd.to_numeric(df["weight"], errors="coerce").dropna()
    w = w[w > 0]
    if len(w) < 100:
        raise ValueError(f"only {len(w)} weighted holdings parsed — not the full SPY basket")
    total = float(w.sum())
    if total < 1.5:                       # weights given as fractions, not percent
        w = w * 100
    pct = float(w.sort_values(ascending=False).head(20).sum())
    return round(pct, 2), int(len(w))


def cape_from_multpl(html: str) -> float:
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    m = re.search(r"Current Shiller PE Ratio[^0-9]{0,40}([\d]{1,3}\.\d{1,2})", text, re.I)
    if not m:
        raise ValueError("no 'Current Shiller PE Ratio' value on page")
    v = float(m.group(1))
    if not (CAPE_PLAUSIBLE[0] <= v <= CAPE_PLAUSIBLE[1]):
        raise ValueError(f"implausible CAPE {v}")
    return v


# ── the one public entry point ──────────────────────────────────────────────

def get_flags(fetch_fred: Optional[Callable] = None, api_key: str = "",
              live: bool = True, today: Optional[date] = None,
              fetchers: Optional[dict] = None) -> dict:
    """All four regime inputs with provenance.

    fetch_fred(series_id, api_key, start) -> pd.Series   (fred_client.fetch_fred)
    fetchers: optional overrides for the selftest — {"walcl", "holdings", "cape"}
              each a zero-arg callable returning the raw input.

    Returns {
        "fed_bs_expanding", "deficit_gt_5pct_gdp", "cape", "top20_concentration_pct",
        "meta": {name: {value, source, asof, age_days, stale, live, detail}},
        "warnings": [str], "fed_state": str | None,
    }"""
    key = ("flags", bool(fetch_fred), live, str(today))
    hit = _CACHE.get(key)
    if hit and fetchers is None and time.time() - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]

    today = today or _today()
    f = fetchers or {}
    meta, warnings, fed_state = {}, [], None

    # Fed balance sheet
    try:
        if not live:
            raise RuntimeError("live fetch disabled")
        if "walcl" in f:
            walcl = f["walcl"]()
        elif fetch_fred is not None:
            walcl = fetch_fred("WALCL", api_key, (today - timedelta(days=200)).isoformat())
        else:
            raise RuntimeError("no FRED fetcher supplied")
        p = fed_posture_from_walcl(walcl, today)
        fed_state = p["state"]
        meta["fed_bs_expanding"] = {
            "value": p["expanding"], "source": "FRED WALCL (live)", "asof": p["asof"],
            "age_days": _age_days(p["asof"], today), "stale": False, "live": True,
            "detail": (f"{p['state']}: 4-wk avg {p['wk4_bn']:+.1f}B/wk"
                       + (f", 13-wk avg {p['wk13_bn']:+.1f}B/wk" if p["wk13_bn"] is not None else ""))}
    except Exception as e:
        meta["fed_bs_expanding"] = _manual("fed_bs_expanding", today, f"live WALCL failed: {e}")

    # Top-20 concentration
    try:
        if not live:
            raise RuntimeError("live fetch disabled")
        raw = f["holdings"]() if "holdings" in f else _get(SPY_HOLDINGS_URL).content
        pct, n = top20_from_holdings(raw)
        if not (TOP20_PLAUSIBLE[0] <= pct <= TOP20_PLAUSIBLE[1]):
            raise ValueError(f"implausible top-20 weight {pct}%")
        meta["top20_concentration_pct"] = {
            "value": pct, "source": f"SPY daily holdings, State Street (live, {n} names)",
            "asof": today.isoformat(), "age_days": 0, "stale": False, "live": True,
            "detail": "top 20 SPY weights summed"}
    except Exception as e:
        meta["top20_concentration_pct"] = _manual("top20_concentration_pct", today, f"live SPY holdings failed: {e}")

    # CAPE
    try:
        if not live:
            raise RuntimeError("live fetch disabled")
        html = f["cape"]() if "cape" in f else _get(CAPE_URL).text
        v = cape_from_multpl(html)
        meta["cape"] = {"value": v, "source": "multpl.com (live)", "asof": today.isoformat(),
                        "age_days": 0, "stale": False, "live": True, "detail": ""}
    except Exception as e:
        meta["cape"] = _manual("cape", today, f"live CAPE failed: {e}")

    # Deficit — manual by design
    meta["deficit_gt_5pct_gdp"] = _manual("deficit_gt_5pct_gdp", today)

    for name, m in meta.items():
        label = name.replace("_", " ")
        if m["stale"]:
            warnings.append(f"{label} = {m['value']} is STALE — manual value from {m['asof']} "
                            f"({m['age_days']}d old). {m['detail']}".strip())
        elif not m["live"] and name != "deficit_gt_5pct_gdp":
            warnings.append(f"{label} = {m['value']} is a manual fallback from {m['asof']} "
                            f"({m['age_days']}d old). {m['detail']}".strip())

    out = {"fed_bs_expanding": bool(meta["fed_bs_expanding"]["value"]),
           "deficit_gt_5pct_gdp": bool(meta["deficit_gt_5pct_gdp"]["value"]),
           "cape": float(meta["cape"]["value"]),
           "top20_concentration_pct": float(meta["top20_concentration_pct"]["value"]),
           "meta": meta, "warnings": warnings, "fed_state": fed_state}
    if fetchers is None:
        _CACHE[key] = (time.time(), out)
    return out


def classifier_kwargs(flags: dict) -> dict:
    """The four keyword arguments full_assessment()/render_regime_section() take."""
    return {k: flags[k] for k in ("fed_bs_expanding", "deficit_gt_5pct_gdp",
                                  "cape", "top20_concentration_pct")}


def summary_lines(flags: dict) -> list[str]:
    """One markdown line per input, for logs and UI captions."""
    L = []
    for name in ("cape", "top20_concentration_pct", "fed_bs_expanding", "deficit_gt_5pct_gdp"):
        m = flags["meta"][name]
        tag = "live" if m["live"] else ("⚠ STALE manual" if m["stale"] else "manual")
        age = f", {m['age_days']}d old" if not m["live"] and m["age_days"] is not None else ""
        extra = f" — {m['detail']}" if m["live"] and m["detail"] else ""
        L.append(f"{name.replace('_', ' ')}: **{m['value']}** ({tag}, as of {m['asof']}{age}){extra}")
    return L


# ── selftest (offline) ──────────────────────────────────────────────────────

def selftest() -> dict:
    import io
    fails = []
    today = date(2026, 9, 26)

    # WALCL: +$6B/wk for 20 weeks -> reserve-management growth, expanding
    idx = pd.date_range(end="2026-09-23", periods=20, freq="7D")
    walcl_up = pd.Series([6_700_000 + 6_000 * i for i in range(20)], index=idx, dtype=float)
    walcl_qt = pd.Series([6_900_000 - 10_000 * i for i in range(20)], index=idx, dtype=float)
    walcl_old = pd.Series(walcl_up.values, index=idx - pd.Timedelta(days=60))

    # SPY holdings: 503 names, top 20 = 40% by construction
    rows = [["SPDR S&P 500 ETF Trust", None, None, None], ["Holdings: As of 25-Sep-2026", None, None, None],
            [None, None, None, None], ["Name", "Ticker", "Identifier", "Weight"]]
    rows += [[f"Big{i}", f"B{i}", f"X{i}", 2.0] for i in range(20)]
    rows += [[f"Small{i}", f"S{i}", f"Y{i}", 60.0 / 483] for i in range(483)]
    buf = io.BytesIO(); pd.DataFrame(rows).to_excel(buf, header=False, index=False)
    cape_html = "<div id='current'><b>Current Shiller PE Ratio:</b> 41.37 <span>+0.12</span></div>"

    ok = get_flags(today=today, fetchers={"walcl": lambda: walcl_up, "holdings": lambda: buf.getvalue(),
                                           "cape": lambda: cape_html})
    if not (ok["fed_bs_expanding"] is True and ok["fed_state"] == "reserve-management growth"):
        fails.append(f"WALCL +6B/wk must be reserve-management growth / expanding: {ok['meta']['fed_bs_expanding']}")
    if abs(ok["top20_concentration_pct"] - 40.0) > 0.01:
        fails.append(f"top-20 should be 40.0: {ok['top20_concentration_pct']}")
    if ok["cape"] != 41.37:
        fails.append(f"CAPE should parse 41.37: {ok['cape']}")
    if ok["warnings"]:
        fails.append(f"all-live run must carry no warnings except the deficit (manual by design): {ok['warnings']}")

    qt = get_flags(today=today, fetchers={"walcl": lambda: walcl_qt, "holdings": lambda: buf.getvalue(),
                                           "cape": lambda: cape_html})
    if qt["fed_bs_expanding"] is not False or qt["fed_state"] != "QT runoff":
        fails.append("WALCL -10B/wk must be QT runoff / not expanding")

    def boom():
        raise RuntimeError("network down")
    fb = get_flags(today=today, fetchers={"walcl": lambda: walcl_old, "holdings": boom, "cape": boom})
    m = fb["meta"]
    if m["cape"]["live"] or m["cape"]["value"] != MANUAL["cape"]["value"] or not m["cape"]["stale"]:
        fails.append(f"failed CAPE fetch must fall back to the manual value and flag it stale: {m['cape']}")
    if m["fed_bs_expanding"]["live"] or "days old" not in m["fed_bs_expanding"]["detail"]:
        fails.append(f"a 60-day-old WALCL must be rejected and fall back: {m['fed_bs_expanding']}")
    if not any("cape" in w and "STALE" in w for w in fb["warnings"]):
        fails.append(f"stale CAPE must produce a warning: {fb['warnings']}")
    if set(classifier_kwargs(fb)) != {"fed_bs_expanding", "deficit_gt_5pct_gdp", "cape", "top20_concentration_pct"}:
        fails.append("classifier_kwargs keys wrong")
    if len(summary_lines(fb)) != 4:
        fails.append("summary_lines must return 4 lines")

    try:
        cape_from_multpl("<p>Current Shiller PE Ratio: 420.5</p>"); fails.append("implausible CAPE must raise")
    except ValueError:
        pass
    return {"ok": not fails, "failures": fails, "example": summary_lines(ok)}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, default=str))
