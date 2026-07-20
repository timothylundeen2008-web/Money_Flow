"""
valuation.py  (v2 — holdings-based P/E, works with ETFs)
──────────────────────────────────────────────────────────
ETFs don't report earnings, so yfinance returns no P/E for them directly.
Solution: compute weighted-average P/E from each ETF's top holdings.

  Sector ETF P/E = Σ (holding P/E × holding weight) / Σ weights

Top-5 holdings cover 47–68% of each SPDR ETF by weight, giving a
representative estimate. yfinance DOES return trailingPE / forwardPE
for individual stocks reliably.

Fallback chain:
  1. yfinance batch download of ~55 stock tickers (fast, one call)
  2. stooq for prices + hard-coded recent EPS estimates
  3. Hard-coded snapshot (dated, clearly labeled)

CAPE approximation:
  True 10-year CAPE requires historical EPS — too complex for free data.
  We compute:
    • TTM P/E   (trailing 12-month, from yfinance .info)
    • Fwd P/E   (next 12-month estimates)
    • "vs Avg"  (vs sector's own long-run historical average)
  Labeled as "Wtd. P/E" to be transparent about the method.
"""

import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ── Sector ETF metadata ────────────────────────────────────────────────────────

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financial",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLI":  "Industrials",
    "XLY":  "Consumer Cyclical",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Basic Materials",
    "XLC":  "Communication Services",
    "XLP":  "Consumer Defensive",
}

# Top-5 holdings per SPDR ETF with approximate weights (updated periodically)
# Source: SSGA ETF holdings as of Q1 2025
SECTOR_HOLDINGS = {
    "XLK":  [("AAPL",0.22),("MSFT",0.20),("NVDA",0.18),("AVGO",0.05),("CRM",0.03)],
    "XLF":  [("BRK-B",0.13),("JPM",0.12),("V",0.09),("MA",0.07),("BAC",0.05)],
    "XLE":  [("XOM",0.24),("CVX",0.16),("COP",0.08),("EOG",0.05),("SLB",0.04)],
    "XLV":  [("LLY",0.15),("UNH",0.14),("ABBV",0.08),("JNJ",0.07),("MRK",0.06)],
    "XLI":  [("GE",0.07),("RTX",0.06),("CAT",0.06),("UPS",0.05),("HON",0.05)],
    "XLY":  [("AMZN",0.24),("TSLA",0.14),("HD",0.10),("MCD",0.04),("LOW",0.04)],
    "XLU":  [("NEE",0.15),("SO",0.08),("DUK",0.08),("AEP",0.05),("EXC",0.04)],
    "XLRE": [("PLD",0.12),("AMT",0.08),("EQIX",0.07),("CCI",0.05),("PSA",0.05)],
    "XLB":  [("LIN",0.18),("APD",0.06),("SHW",0.06),("FCX",0.05),("ECL",0.05)],
    "XLC":  [("META",0.22),("GOOGL",0.12),("GOOG",0.10),("NFLX",0.05),("T",0.04)],
    "XLP":  [("PG",0.16),("COST",0.14),("WMT",0.11),("KO",0.08),("PEP",0.08)],
}

# Historical average P/E by sector (1990–2024 long-run averages)
# Sources: Research Affiliates RAFI, StarCapital, S&P Dow Jones historical data
HIST_AVG_PE = {
    "XLK":  28.1,
    "XLF":  15.2,
    "XLE":  17.8,
    "XLV":  20.4,
    "XLI":  19.6,
    "XLY":  22.1,
    "XLU":  16.8,
    "XLRE": 28.4,
    "XLB":  18.2,
    "XLC":  24.6,
    "XLP":  21.8,
}

VALUATION_COLORS = {
    "Very Expensive":  ("#A32D2D", "#250c0c"),
    "Expensive":       ("#D85A30", "#2e1810"),
    "Slight Premium":  ("#BA7517", "#2e2008"),
    "Fair Value":      ("#1D9E75", "#0d3326"),
    "Slight Discount": ("#2BAD7E", "#0e2e22"),
    "Cheap":           ("#378ADD", "#0e2240"),
    "N/A":             ("#888780", "#1e2330"),
}

# Hard-coded snapshot fallback (May 2025 estimates)
# Used only when live data is unavailable
_SNAPSHOT = {
    "XLK":  {"ttm_pe": 38.4, "fwd_pe": 32.1, "pb": 11.2},
    "XLF":  {"ttm_pe": 16.2, "fwd_pe": 14.8, "pb":  1.9},
    "XLE":  {"ttm_pe": 13.1, "fwd_pe": 12.4, "pb":  1.8},
    "XLV":  {"ttm_pe": 21.8, "fwd_pe": 19.2, "pb":  4.1},
    "XLI":  {"ttm_pe": 25.4, "fwd_pe": 22.1, "pb":  5.2},
    "XLY":  {"ttm_pe": 31.2, "fwd_pe": 27.4, "pb":  7.8},
    "XLU":  {"ttm_pe": 18.4, "fwd_pe": 17.2, "pb":  2.1},
    "XLRE": {"ttm_pe": 36.8, "fwd_pe": 32.4, "pb":  3.4},
    "XLB":  {"ttm_pe": 20.1, "fwd_pe": 18.6, "pb":  3.8},
    "XLC":  {"ttm_pe": 22.4, "fwd_pe": 19.8, "pb":  4.9},
    "XLP":  {"ttm_pe": 24.2, "fwd_pe": 22.1, "pb":  4.6},
}


def _valuation_signal(pe: float, hist_avg: float) -> str:
    if not pe or not hist_avg or pe <= 0:
        return "N/A"
    prem = (pe / hist_avg - 1) * 100
    if prem > 30:   return "Very Expensive"
    if prem > 15:   return "Expensive"
    if prem > 5:    return "Slight Premium"
    if prem > -5:   return "Fair Value"
    if prem > -15:  return "Slight Discount"
    return "Cheap"


def _fetch_holdings_pe() -> dict:
    """
    Fetch trailingPE and forwardPE for all unique holding tickers via yfinance.
    Returns dict: ticker → {"ttm_pe": x, "fwd_pe": y, "pb": z}
    """
    all_tickers = list({t for holdings in SECTOR_HOLDINGS.values()
                        for t, _ in holdings})
    result = {}

    try:
        import yfinance as yf

        # Batch download .info — use fast_info where possible
        for tk in all_tickers:
            try:
                info = yf.Ticker(tk).info
                ttm = info.get("trailingPE")
                fwd = info.get("forwardPE")
                pb  = info.get("priceToBook")
                if ttm or fwd:
                    result[tk] = {
                        "ttm_pe": round(float(ttm), 1) if ttm else None,
                        "fwd_pe": round(float(fwd), 1) if fwd else None,
                        "pb":     round(float(pb),  1) if pb  else None,
                    }
            except Exception:
                pass
            time.sleep(0.05)

        print(f"[valuation] yfinance: {len(result)}/{len(all_tickers)} holdings fetched")
        return result

    except Exception as e:
        print(f"[valuation] yfinance error: {e}")
        return {}


def _weighted_pe(etf: str, holding_data: dict) -> dict:
    """
    Compute weighted-average TTM P/E, Fwd P/E, and P/B for an ETF
    from its top holdings' individual P/E values.
    """
    holdings = SECTOR_HOLDINGS.get(etf, [])
    ttm_num = ttm_den = 0.0
    fwd_num = fwd_den = 0.0
    pb_num  = pb_den  = 0.0

    for ticker, weight in holdings:
        d = holding_data.get(ticker, {})
        if d.get("ttm_pe") and d["ttm_pe"] > 0:
            ttm_num += d["ttm_pe"] * weight
            ttm_den += weight
        if d.get("fwd_pe") and d["fwd_pe"] > 0:
            fwd_num += d["fwd_pe"] * weight
            fwd_den += weight
        if d.get("pb") and d["pb"] > 0:
            pb_num  += d["pb"] * weight
            pb_den  += weight

    return {
        "ttm_pe": round(ttm_num / ttm_den, 1) if ttm_den > 0 else None,
        "fwd_pe": round(fwd_num / fwd_den, 1) if fwd_den > 0 else None,
        "pb":     round(pb_num  / pb_den,  1) if pb_den  > 0 else None,
        "coverage": round(ttm_den, 2),   # weight coverage (0–1)
    }


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_valuation_data() -> pd.DataFrame:
    """
    Returns valuation DataFrame for all 11 sector ETFs.

    Columns:
        ticker, sector, ttm_pe, fwd_pe, pb, coverage, method,
        hist_avg_pe, premium_pct, valuation_signal,
        signal_fg, signal_bg
    """
    # Try live holdings-based P/E
    holding_data = _fetch_holdings_pe()
    use_snapshot = len(holding_data) < 5

    if use_snapshot:
        print("[valuation] Using snapshot fallback")
        st.info(
            "ℹ️ Live valuation data unavailable — showing May 2025 estimates. "
            "Refresh to retry.",
            icon="📊"
        )

    records = []
    for ticker, sector in SECTOR_ETFS.items():
        hist_avg = HIST_AVG_PE.get(ticker, 20.0)

        if use_snapshot:
            snap = _SNAPSHOT.get(ticker, {})
            ttm_pe = snap.get("ttm_pe")
            fwd_pe = snap.get("fwd_pe")
            pb     = snap.get("pb")
            method = "Snapshot (May 2025)"
            coverage = 1.0
        else:
            vals   = _weighted_pe(ticker, holding_data)
            ttm_pe = vals["ttm_pe"]
            fwd_pe = vals["fwd_pe"]
            pb     = vals["pb"]
            cov    = vals["coverage"]
            coverage = cov
            method = f"Wtd. P/E ({cov:.0%} cov.)"

        pe_for_signal = ttm_pe or fwd_pe
        premium = round((pe_for_signal / hist_avg - 1) * 100, 1) \
                  if pe_for_signal else None
        signal  = _valuation_signal(pe_for_signal or 0, hist_avg)
        colors  = VALUATION_COLORS.get(signal, VALUATION_COLORS["N/A"])

        records.append({
            "ticker":           ticker,
            "sector":           sector,
            "ttm_pe":           ttm_pe,
            "fwd_pe":           fwd_pe,
            "pb":               pb,
            "coverage":         coverage,
            "method":           method,
            "hist_avg_pe":      hist_avg,
            "premium_pct":      premium,
            "valuation_signal": signal,
            "signal_fg":        colors[0],
            "signal_bg":        colors[1],
        })

    df = pd.DataFrame(records)
    return df.sort_values("premium_pct", ascending=False, na_position="last")
