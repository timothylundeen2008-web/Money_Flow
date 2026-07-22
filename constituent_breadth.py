"""
constituent_breadth.py  (v1 — July 2026)
──────────────────────────────────────────────────────────────────────────────
THE SINGLE-STOCK LAYER — and a deliberate statement about how to use it.

WHY THIS MODULE EXISTS
  A sector ETF is a cap-weighted average, and averages hide two things that
  matter enormously for a flow thesis:

  1. INTRA-SECTOR ROTATION CANCELS OUT. If institutions buy integrated majors
     and sell shale, XLE barely moves. The flow is large, real, and completely
     invisible at the ETF level. Inter-sector rotation is what this dashboard
     was built to see; intra-sector rotation is arguably more common and was
     entirely undetectable before this module.

  2. CONCENTRATION LAUNDERS SINGLE-STOCK SIGNALS AS SECTOR SIGNALS. XLK's top
     ten holdings are ~61% of the fund; NVDA, AAPL and MSFT alone are roughly a
     third of it. "Technology is being accumulated" frequently means "NVDA is
     being accumulated, and 60 other names are noise." Acting on that as a
     SECTOR view is a category error — you are taking single-stock risk while
     believing you hold a diversified sector position.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
  It does NOT run the flow-scoring machinery on hundreds of individual stocks.
  That would produce hundreds of noisy signals and a false sense of precision,
  and — critically — single stocks have NO creations/redemptions data, so the
  best Tier-A flow signal available (etf_flow_tracker.py) does not exist for
  them. Per-stock flow scoring buys noise, not information.

  Instead, constituent data is used DIAGNOSTICALLY: to answer "is the ETF-level
  signal broad-based, or is it one stock wearing a sector's clothes?" That is
  the question single-stock data answers better than anything else, and it is
  the question that determines whether a sector-level position is the right
  instrument for the thesis.

EVIDENCE TIER
  Breadth on price     → Tier C (outcome)
  Breadth on CMF       → Tier B (directional pressure), the valuable one
  Neither is Tier A. This module validates flow signals; it does not source them.

HOLDINGS MAP OWNERSHIP
  SECTOR_CONSTITUENTS below is the CANONICAL holdings map for this repo.
  valuation.py currently carries its own shorter top-5 copy dated Q1 2025 whose
  weights have visibly drifted (it still shows AAPL 0.22 / MSFT 0.20 / NVDA 0.18
  for XLK). Migrate valuation.py to import top_holdings() from here so the two
  cannot diverge again — duplicated constants have already caused three
  documented defects in this codebase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import streamlit as st
    _cache = st.cache_data(ttl=3600, show_spinner=False)
except Exception:                                    # usable outside Streamlit
    def _cache(fn):
        return fn

from flow_metrics import chaikin_money_flow, MIN_BARS_CMF


# ── Canonical holdings map ────────────────────────────────────────────────────
# Approximate top-10 weights. REFRESH QUARTERLY from the issuer holdings files
# (SSGA publishes daily holdings per fund) and bump HOLDINGS_ASOF when you do.
# Weights need not sum to 1.0 — they are the top-N slice of each fund.
HOLDINGS_ASOF = "2026-07-01"
STALE_AFTER_DAYS = 120        # holdings drift slowly; ~1 quarter is the cadence

SECTOR_CONSTITUENTS: dict[str, list[tuple[str, float]]] = {
    "XLK":  [("NVDA",0.145),("AAPL",0.130),("MSFT",0.090),("AVGO",0.053),("MU",0.045),
             ("AMD",0.038),("PLTR",0.029),("CSCO",0.027),("LRCX",0.027),("AMAT",0.026)],
    "XLF":  [("BRK-B",0.125),("JPM",0.115),("V",0.085),("MA",0.070),("BAC",0.050),
             ("WFC",0.040),("GS",0.033),("SPGI",0.030),("AXP",0.028),("MS",0.027)],
    "XLE":  [("XOM",0.225),("CVX",0.160),("COP",0.080),("WMB",0.050),("EOG",0.045),
             ("SLB",0.040),("OKE",0.035),("MPC",0.033),("PSX",0.031),("KMI",0.030)],
    "XLV":  [("LLY",0.140),("UNH",0.110),("JNJ",0.075),("ABBV",0.070),("MRK",0.055),
             ("TMO",0.045),("ABT",0.042),("ISRG",0.038),("AMGN",0.033),("PFE",0.030)],
    "XLI":  [("GE",0.070),("CAT",0.060),("RTX",0.058),("UNP",0.045),("HON",0.043),
             ("BA",0.042),("UPS",0.035),("ETN",0.034),("DE",0.032),("LMT",0.030)],
    "XLY":  [("AMZN",0.230),("TSLA",0.135),("HD",0.095),("MCD",0.045),("BKNG",0.040),
             ("LOW",0.038),("TJX",0.033),("SBUX",0.028),("NKE",0.025),("CMG",0.022)],
    "XLU":  [("NEE",0.140),("SO",0.080),("DUK",0.078),("CEG",0.070),("AEP",0.050),
             ("SRE",0.042),("D",0.040),("EXC",0.036),("PCG",0.032),("XEL",0.030)],
    "XLRE": [("PLD",0.115),("AMT",0.085),("EQIX",0.075),("WELL",0.062),("CCI",0.050),
             ("PSA",0.048),("SPG",0.042),("O",0.040),("DLR",0.038),("CBRE",0.030)],
    "XLB":  [("LIN",0.175),("SHW",0.062),("APD",0.058),("FCX",0.055),("ECL",0.050),
             ("NEM",0.048),("NUE",0.035),("DOW",0.030),("PPG",0.028),("VMC",0.027)],
    "XLC":  [("META",0.215),("GOOGL",0.125),("GOOG",0.105),("NFLX",0.055),("DIS",0.045),
             ("TMUS",0.042),("VZ",0.038),("CMCSA",0.035),("T",0.033),("EA",0.020)],
    "XLP":  [("COST",0.150),("PG",0.145),("WMT",0.115),("KO",0.085),("PEP",0.075),
             ("PM",0.060),("MO",0.045),("MDLZ",0.038),("CL",0.033),("TGT",0.025)],
}


def top_holdings(etf: str, n: int = 5) -> list[tuple[str, float]]:
    """Canonical accessor. Import this from valuation.py instead of duplicating."""
    return SECTOR_CONSTITUENTS.get(etf, [])[:n]


def holdings_stale_days(today: pd.Timestamp | None = None) -> int:
    today = today or pd.Timestamp.today()
    return int((today.normalize() - pd.Timestamp(HOLDINGS_ASOF)).days)


# ── Data ──────────────────────────────────────────────────────────────────────

def all_constituent_tickers() -> list[str]:
    return sorted({t for hs in SECTOR_CONSTITUENTS.values() for t, _ in hs})


@_cache
def fetch_constituent_ohlcv(period: str = "400d") -> dict[str, pd.DataFrame]:
    """
    One batched OHLCV download for every constituent (~110 tickers).

    Returns {ticker: DataFrame[High, Low, Close, Volume]}. Tickers that fail
    are OMITTED rather than filled — a missing name must reduce the reported
    coverage, never silently shrink the denominator without the caller knowing.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[breadth] yfinance unavailable")
        return {}

    tickers = all_constituent_tickers()
    try:
        raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                          progress=False, threads=True, group_by="ticker", timeout=40)
    except Exception as e:
        print(f"[breadth] download failed: {e}")
        return {}
    if raw is None or raw.empty:
        return {}

    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        try:
            df = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df[["High", "Low", "Close", "Volume"]].dropna()
            if len(df) >= MIN_BARS_CMF:
                out[tk] = df
        except Exception:
            continue
    print(f"[breadth] fetched {len(out)}/{len(tickers)} constituents")
    return out


# ── Per-sector diagnostics ────────────────────────────────────────────────────

def _pct_above_ma(df: pd.DataFrame, window: int) -> bool | None:
    c = df["Close"].dropna()
    if len(c) < window:
        return None
    return bool(c.iloc[-1] > c.rolling(window).mean().iloc[-1])


def sector_breadth(etf: str, data: dict[str, pd.DataFrame],
                   lookback: int = 21) -> dict:
    """
    Breadth, dispersion and concentration diagnostics for one sector ETF.

    The headline output is `signal_quality`, which tells you whether an
    ETF-level flow reading deserves to be treated as a SECTOR view:

      BROAD        — participation is widespread; the sector signal is real
      CONCENTRATED — the move is the mega-caps; you are taking single-stock
                     risk in a sector wrapper. Consider expressing the view in
                     the specific names, or size as a single-stock position.
      NARROW       — few names participating and high dispersion; this is a
                     stock-picker's tape, and sector-level signals are weak.
      INSUFFICIENT — not enough constituent coverage to judge.
    """
    holds = SECTOR_CONSTITUENTS.get(etf, [])
    res = {"etf": etf, "n_constituents": 0, "coverage": 0.0,
           "pct_above_50dma": np.nan, "pct_above_200dma": np.nan,
           "pct_positive_cmf": np.nan, "pct_positive_return": np.nan,
           "dispersion": np.nan, "top3_weight": np.nan,
           "top3_return_share": np.nan, "signal_quality": "INSUFFICIENT"}
    if not holds:
        return res

    present = [(t, w) for t, w in holds if t in data]
    res["n_constituents"] = len(present)
    res["coverage"] = round(len(present) / len(holds), 2) if holds else 0.0
    if len(present) < 5:
        return res

    above50, above200, pos_cmf, rets, contribs = [], [], [], [], []
    for tk, w in present:
        df = data[tk]
        a50, a200 = _pct_above_ma(df, 50), _pct_above_ma(df, 200)
        if a50 is not None:
            above50.append(a50)
        if a200 is not None:
            above200.append(a200)

        cmf = chaikin_money_flow(df["High"], df["Low"], df["Close"], df["Volume"])
        if cmf.notna().any():
            pos_cmf.append(bool(cmf.dropna().iloc[-1] > 0))

        c = df["Close"].dropna()
        if len(c) > lookback:
            r = float(c.iloc[-1] / c.iloc[-lookback - 1] - 1)
            rets.append(r)
            contribs.append((tk, w, w * r))

    if not rets:
        return res

    res["pct_above_50dma"]     = round(float(np.mean(above50)), 3) if above50 else np.nan
    res["pct_above_200dma"]    = round(float(np.mean(above200)), 3) if above200 else np.nan
    res["pct_positive_cmf"]    = round(float(np.mean(pos_cmf)), 3) if pos_cmf else np.nan
    res["pct_positive_return"] = round(float(np.mean([r > 0 for r in rets])), 3)
    res["dispersion"]          = round(float(np.std(rets)), 4)

    # Concentration: how much of the weighted move came from the top 3 names
    contribs.sort(key=lambda x: -x[1])
    top3 = contribs[:3]
    res["top3_weight"] = round(float(sum(w for _, w, _ in top3)), 3)
    total_contrib = sum(abs(c) for _, _, c in contribs)
    res["top3_return_share"] = (round(float(sum(abs(c) for _, _, c in top3) / total_contrib), 3)
                                if total_contrib > 0 else np.nan)

    res["signal_quality"] = _quality_verdict(res)
    return res


def _quality_verdict(r: dict) -> str:
    """
    BROAD        ≥60% of names with positive CMF and ≥55% above their 50DMA
    CONCENTRATED top-3 drove >55% of the weighted move, or CMF breadth <40%
                 while the ETF itself is rising
    NARROW       <45% participation with elevated dispersion
    """
    cmf_b, ret_b = r.get("pct_positive_cmf"), r.get("pct_positive_return")
    a50, top3 = r.get("pct_above_50dma"), r.get("top3_return_share")
    disp = r.get("dispersion")
    if any(pd.isna(v) for v in (cmf_b, ret_b)):
        return "INSUFFICIENT"
    if not pd.isna(top3) and top3 > 0.55:
        return "CONCENTRATED"
    if cmf_b >= 0.60 and (pd.isna(a50) or a50 >= 0.55):
        return "BROAD"
    if cmf_b < 0.40:
        return "CONCENTRATED"
    if ret_b < 0.45 and not pd.isna(disp) and disp > 0.08:
        return "NARROW"
    return "BROAD" if cmf_b >= 0.50 else "NARROW"


@_cache
def build_breadth_table(lookback: int = 21) -> pd.DataFrame:
    """
    Full constituent-diagnostic table, one row per sector ETF.

    Intended use in the weekly review: read this ALONGSIDE the ETF-level flow
    reading, never instead of it. The pairing that matters:

      ETF accumulating + BROAD        → sector thesis confirmed, size normally
      ETF accumulating + CONCENTRATED → it is the mega-caps; take the position
                                        in the names or accept single-stock risk
      ETF flat + high dispersion      → real intra-sector rotation the ETF is
                                        cancelling out; look underneath
    """
    data = fetch_constituent_ohlcv()
    if not data:
        return pd.DataFrame()
    rows = [sector_breadth(etf, data, lookback) for etf in SECTOR_CONSTITUENTS]
    df = pd.DataFrame(rows)
    df["holdings_asof"] = HOLDINGS_ASOF
    df["holdings_stale"] = holdings_stale_days() > STALE_AFTER_DAYS
    return df


def intra_sector_rotation(etf: str, data: dict[str, pd.DataFrame],
                          lookback: int = 21, top_k: int = 3) -> dict:
    """
    Surface the rotation happening INSIDE a sector — the flow an ETF average
    structurally cancels out.

    Returns the strongest and weakest constituents by CMF, plus a spread. A
    wide CMF spread with a flat ETF is the signature of institutions rotating
    within the sector rather than into or out of it, and it is invisible to
    every ETF-level signal in this dashboard.
    """
    holds = [(t, w) for t, w in SECTOR_CONSTITUENTS.get(etf, []) if t in data]
    scores = []
    for tk, w in holds:
        df = data[tk]
        cmf = chaikin_money_flow(df["High"], df["Low"], df["Close"], df["Volume"])
        if cmf.notna().any():
            scores.append((tk, w, float(cmf.dropna().iloc[-1])))
    if len(scores) < 4:
        return {"etf": etf, "available": False}

    scores.sort(key=lambda x: -x[2])
    into = [(t, round(c, 3)) for t, _, c in scores[:top_k]]
    outof = [(t, round(c, 3)) for t, _, c in scores[-top_k:]]
    spread = round(scores[0][2] - scores[-1][2], 3)
    return {"etf": etf, "available": True, "rotating_into": into,
            "rotating_out_of": outof, "cmf_spread": spread,
            "interpretation": ("Wide intra-sector CMF spread — money is moving "
                               "WITHIN the sector; an ETF-level position may "
                               "net out the very flow you are trying to follow."
                               if spread > 0.20 else
                               "Narrow spread — constituents moving together; "
                               "the ETF is a fair expression of the flow.")}
