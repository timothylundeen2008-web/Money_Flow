"""
cot_fetcher.py  (v1 — July 2026)
──────────────────────────────────────────────────────────────────────────────
CFTC Commitments of Traders — Traders in Financial Futures (TFF).

EVIDENCE TIER: A (reported positioning).
  This is not a proxy. The TFF report breaks out, by name, how much of the
  open interest is held by ASSET MANAGERS (pensions, insurers, mutual funds)
  versus LEVERAGED FUNDS (hedge funds, CTAs, commodity pool operators). It is
  the closest thing to a direct read on institutional positioning available at
  zero cost, and it is the only dataset in this entire stack that identifies
  WHO holds a position rather than inferring it from tape behavior.

  Free, public domain, no API key. Data as of each Tuesday, published Friday
  3:30pm ET — so it is ALWAYS at least a 3-day-stale snapshot. It is regime
  and sizing context; it is never an entry trigger.

WHY IT MAPS ONTO THIS BOOK
  COT contracts line up almost one-to-one with the All-Weather macro sleeves:
      gold        → GLD / SLV / RING
      WTI crude   → XLE / XOP / USO
      10Y & 30Y   → TLT (and the DFII10 / TLT re-arm decision)
      E-mini S&P  → the growth sleeve
      DXY         → the Daily Step 4 dollar-divergence check

READ PERCENTILES, NOT LEVELS
  A net-long position means little in isolation; the same net long can be a
  3-year extreme or unremarkable depending on the contract. Percentile rank
  against trailing history is the signal. Positioning extremes are where
  regime calls get their asymmetry: Asset Managers at a 3-year low in gold
  with the repression thesis intact is a very different setup from an
  identical chart at a 3-year positioning high.

  The two cohorts behave differently ON PURPOSE — Leveraged Funds are fast and
  mean-reverting, Asset Managers are slow and trend-persistent. A divergence
  between them is itself informative and is surfaced below.

⚠ SCHEMA VERIFICATION REQUIRED ON FIRST RUN
  The Socrata dataset id and column names below are correct to the best of
  current knowledge but MUST be confirmed against publicreporting.cftc.gov
  before this is trusted for sizing — CFTC has renamed fields before. Call
  verify_schema() once after deployment; it reports exactly what it found.
  This module degrades to an explicit error, never to silent bad data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

try:
    import streamlit as st
    _cache = st.cache_data(ttl=21600, show_spinner=False)   # 6h; weekly data
except Exception:
    def _cache(fn):
        return fn


# ── Endpoint / schema constants (VERIFY ON FIRST RUN) ─────────────────────────
SOCRATA_BASE = "https://publicreporting.cftc.gov/resource"
TFF_DATASET  = "gpe5-46if"          # Traders in Financial Futures, futures only
LEGACY_DATASET = "6dca-aqww"        # Legacy futures only (commodities fallback)

DATE_FIELD = "report_date_as_yyyy_mm_dd"
NAME_FIELD = "market_and_exchange_names"

TFF_FIELDS = {
    "asset_mgr_long":  "asset_mgr_positions_long",
    "asset_mgr_short": "asset_mgr_positions_short",
    "lev_fund_long":   "lev_money_positions_long",
    "lev_fund_short":  "lev_money_positions_short",
    "open_interest":   "open_interest_all",
}
# Legacy report uses non-commercial ("large speculator") instead of the
# asset-manager / leveraged-fund split. Commodities live here, not in TFF.
LEGACY_FIELDS = {
    "noncomm_long":  "noncomm_positions_long_all",
    "noncomm_short": "noncomm_positions_short_all",
    "comm_long":     "comm_positions_long_all",
    "comm_short":    "comm_positions_short_all",
    "open_interest": "open_interest_all",
}

# Contract names as they appear in market_and_exchange_names.
# Financials → TFF; physical commodities → Legacy.
CONTRACTS = {
    "SP500":  ("E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",            "tff"),
    "UST10Y": ("10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",    "tff"),
    "UST30Y": ("U.S. TREASURY BONDS - CHICAGO BOARD OF TRADE",            "tff"),
    "UST2Y":  ("2-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE",     "tff"),
    "DXY":    ("U.S. DOLLAR INDEX - ICE FUTURES U.S.",                    "tff"),
    "GOLD":   ("GOLD - COMMODITY EXCHANGE INC.",                          "legacy"),
    "SILVER": ("SILVER - COMMODITY EXCHANGE INC.",                        "legacy"),
    "WTI":    ("CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",   "legacy"),
}

# Which portfolio sleeve each contract informs — used by the weekly review.
SLEEVE_MAP = {
    "GOLD": "GLD / SLV / RING", "SILVER": "SLV", "WTI": "XLE / XOP / PDBC",
    "UST10Y": "TLT", "UST30Y": "TLT", "UST2Y": "USFR / SGOV",
    "SP500": "VGT / QQQ / SMH", "DXY": "Daily Step 4 dollar check",
}

_HEADERS = {"User-Agent": "AllWeatherDashboard/1.0 (research; contact via repo)"}


# ── Fetch ─────────────────────────────────────────────────────────────────────

def _get(dataset: str, contract: str, limit: int = 200) -> pd.DataFrame:
    """One Socrata query. Returns empty DataFrame on any failure — never raises."""
    url = f"{SOCRATA_BASE}/{dataset}.json"
    params = {
        "$where": f"{NAME_FIELD}='{contract}'",
        "$order": f"{DATE_FIELD} DESC",
        "$limit": limit,
    }
    try:
        r = requests.get(url, params=params, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            print(f"[cot] no rows for {contract!r} in {dataset}")
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df[DATE_FIELD] = pd.to_datetime(df[DATE_FIELD])
        return df.sort_values(DATE_FIELD).set_index(DATE_FIELD)
    except Exception as e:
        print(f"[cot] fetch failed for {contract}: {type(e).__name__}: {e}")
        return pd.DataFrame()


def verify_schema() -> dict:
    """
    Run once after deployment. Confirms the dataset ids resolve and the
    expected columns exist, and reports precisely what is missing so the
    constants above can be corrected. Silent schema drift is the failure
    mode this guards against.
    """
    out = {}
    for label, ds, fields in (("tff", TFF_DATASET, TFF_FIELDS),
                              ("legacy", LEGACY_DATASET, LEGACY_FIELDS)):
        probe = "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE" if label == "tff" \
                else "GOLD - COMMODITY EXCHANGE INC."
        df = _get(ds, probe, limit=1)
        if df.empty:
            out[label] = {"ok": False, "reason": "no data returned"}
            continue
        missing = [v for v in fields.values() if v not in df.columns]
        out[label] = {"ok": not missing, "missing": missing,
                      "available_columns": sorted(df.columns)[:40]}
    return out


# ── Metrics ───────────────────────────────────────────────────────────────────

def _pct_rank(series: pd.Series, value: float) -> float:
    """Percentile rank of `value` within `series`, 0-100."""
    s = series.dropna()
    if len(s) < 26:            # need ~6 months of weeks to rank meaningfully
        return float("nan")
    return round(float((s < value).mean() * 100), 1)


def contract_positioning(key: str, years: int = 3) -> dict:
    """
    Net positioning plus percentile rank for one contract.

    For TFF contracts returns Asset Manager and Leveraged Fund nets separately.
    For Legacy contracts returns non-commercial ("large spec") and commercial
    ("hedger/producer") nets — commercials are the natural counterparty and
    frequently the more informative side at extremes.
    """
    if key not in CONTRACTS:
        return {"contract": key, "available": False, "reason": "unknown contract"}
    name, kind = CONTRACTS[key]
    weeks = years * 52
    df = _get(TFF_DATASET if kind == "tff" else LEGACY_DATASET, name, limit=weeks)
    if df.empty:
        return {"contract": key, "available": False, "reason": "no data"}

    fields = TFF_FIELDS if kind == "tff" else LEGACY_FIELDS
    missing = [v for v in fields.values() if v not in df.columns]
    if missing:
        return {"contract": key, "available": False,
                "reason": f"schema mismatch, missing {missing} — run verify_schema()"}

    num = df[list(fields.values())].apply(pd.to_numeric, errors="coerce")
    oi = num[fields["open_interest"]].replace(0, np.nan)

    out = {"contract": key, "available": True, "kind": kind,
           "sleeve": SLEEVE_MAP.get(key, ""),
           "report_date": df.index[-1].date().isoformat(),
           "weeks_history": len(df)}

    if kind == "tff":
        pairs = [("asset_mgr", "asset_mgr_long", "asset_mgr_short"),
                 ("lev_fund", "lev_fund_long", "lev_fund_short")]
    else:
        pairs = [("noncomm", "noncomm_long", "noncomm_short"),
                 ("comm", "comm_long", "comm_short")]

    for label, lk, sk in pairs:
        net = num[fields[lk]] - num[fields[sk]]
        net_pct_oi = (net / oi) * 100
        out[f"{label}_net"] = int(net.iloc[-1]) if pd.notna(net.iloc[-1]) else None
        out[f"{label}_net_pct_oi"] = (round(float(net_pct_oi.iloc[-1]), 2)
                                      if pd.notna(net_pct_oi.iloc[-1]) else None)
        out[f"{label}_pctile"] = _pct_rank(net_pct_oi.iloc[:-1], net_pct_oi.iloc[-1])
        out[f"{label}_chg_4w"] = (int(net.iloc[-1] - net.iloc[-5])
                                  if len(net) >= 5 and pd.notna(net.iloc[-5]) else None)

    out["flag"] = _positioning_flag(out, pairs[0][0], pairs[1][0])
    return out


def _positioning_flag(r: dict, slow_label: str, fast_label: str) -> str:
    """
    Extremes and cohort divergence — the two readings worth acting on.

    Cohort divergence matters because the slow cohort (Asset Managers /
    commercials) and the fast cohort (Leveraged Funds / large specs) taking
    opposite sides is a classic setup: the fast money is usually the one that
    has to unwind.
    """
    slow, fast = r.get(f"{slow_label}_pctile"), r.get(f"{fast_label}_pctile")
    if slow is None or fast is None or pd.isna(slow) or pd.isna(fast):
        return "insufficient history"
    notes = []
    for lbl, p in ((slow_label, slow), (fast_label, fast)):
        if p >= 90:
            notes.append(f"{lbl} crowded LONG ({p:.0f}th pctile)")
        elif p <= 10:
            notes.append(f"{lbl} crowded SHORT ({p:.0f}th pctile)")
    if abs(slow - fast) >= 50:
        notes.append(f"COHORT DIVERGENCE ({slow_label} {slow:.0f} vs {fast_label} {fast:.0f})")
    return " · ".join(notes) if notes else "neutral"


@_cache
def build_cot_table(years: int = 3) -> pd.DataFrame:
    """
    Positioning panel across every mapped contract, one row each.

    Weekly-review use: read the percentile columns, not the raw nets. Flag
    anything beyond the 90th or below the 10th, and any cohort divergence.
    """
    rows = []
    for key in CONTRACTS:
        r = contract_positioning(key, years=years)
        if r.get("available"):
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("contract")
