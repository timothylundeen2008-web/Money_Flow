"""
top_movers.py
─────────────
Fetches price + volume data for a curated universe of 40 sub-sector ETFs,
computes an Institutional Flow Score, and returns the top 10.

Why ETFs instead of individual stocks?
  • ETF flows ARE institutional money (BlackRock/Vanguard/State Street inflows)
  • 40 tickers fetch in ~15 seconds vs 500 S&P constituents in ~2.5 minutes
  • Volume spikes on ETFs = confirmed institutional conviction, not earnings noise
  • Covers equities, bonds, commodities, international in one unified view

Institutional Flow Score (0–100) weights four components:
  40% Momentum consistency  – average performance across 1W / 1M / 3M
  30% Acceleration          – 1M perf vs 3M run-rate (spread signal)
  20% Volume conviction     – relative volume vs 20-day average
  10% Cross-timeframe unity – all timeframes positive = full confirmation
"""

import time
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
import streamlit as st

# ── Curated ETF universe ───────────────────────────────────────────────────────
# Format: ticker → (display name, asset class / category)

ETF_UNIVERSE = {
    # ── Technology ──
    "SMH":  ("Semiconductors",         "Technology"),
    "SOXX": ("Semiconductors II",      "Technology"),
    "IGV":  ("Software",               "Technology"),
    "SKYY": ("Cloud Computing",        "Technology"),
    "HACK": ("Cybersecurity",          "Technology"),
    # ── Financial ──
    "KRE":  ("Regional Banks",         "Financial"),
    "KBE":  ("Banks Broad",            "Financial"),
    "IAI":  ("Broker-Dealers",         "Financial"),
    # ── Healthcare ──
    "IBB":  ("Biotech",                "Healthcare"),
    "XBI":  ("Biotech Small Cap",      "Healthcare"),
    "IHI":  ("Medical Devices",        "Healthcare"),
    "PPH":  ("Pharmaceuticals",        "Healthcare"),
    # ── Energy ──
    "XOP":  ("Oil & Gas E&P",          "Energy"),
    "OIH":  ("Oil Services",           "Energy"),
    "AMLP": ("Pipelines / MLP",        "Energy"),
    # ── Industrials ──
    "ITA":  ("Aerospace & Defense",    "Industrials"),
    "XTN":  ("Transportation",         "Industrials"),
    "PAVE": ("Infrastructure",         "Industrials"),
    # ── Consumer ──
    "XRT":  ("Retail",                 "Consumer Cyclical"),
    "XHB":  ("Homebuilders",           "Consumer Cyclical"),
    "PBJ":  ("Food & Beverage",        "Consumer Defensive"),
    # ── Broad Market ──
    "QQQ":  ("Nasdaq 100",             "Broad Market"),
    "IWM":  ("Russell 2000",           "Broad Market"),
    "IWO":  ("Russell 2000 Growth",    "Broad Market"),
    "MDY":  ("S&P MidCap 400",         "Broad Market"),
    # ── Fixed Income ──
    "TLT":  ("Long Bonds 20Y+",        "Fixed Income"),
    "HYG":  ("High Yield Corp",        "Fixed Income"),
    "LQD":  ("Investment Grade Corp",  "Fixed Income"),
    "EMB":  ("Emerging Mkt Bonds",     "Fixed Income"),
    "TIP":  ("TIPS / Inflation",       "Fixed Income"),
    # ── Commodities ──
    "GLD":  ("Gold",                   "Commodities"),
    "SLV":  ("Silver",                 "Commodities"),
    "PDBC": ("Commodities Broad",      "Commodities"),
    "USO":  ("Oil (WTI)",              "Commodities"),
    "DBA":  ("Agriculture",            "Commodities"),
    # ── International ──
    "EEM":  ("Emerging Markets",       "International"),
    "EFA":  ("Developed Intl (EAFE)",  "International"),
    "EWJ":  ("Japan",                  "International"),
    "FXI":  ("China Large Cap",        "International"),
    "INDA": ("India",                  "International"),
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ── Price fetching ─────────────────────────────────────────────────────────────

def _fetch_yfinance_bulk(tickers: list[str]) -> pd.DataFrame | None:
    """Download ~130 days of OHLCV via yfinance (works on Streamlit Cloud)."""
    try:
        import yfinance as yf
        raw = yf.download(
            tickers, period="130d", interval="1d",
            auto_adjust=True, progress=False, threads=True, timeout=25,
        )
        if raw.empty:
            return None
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        volumes = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else None
        return closes, volumes
    except Exception as e:
        print(f"[top_movers/yfinance] {e}")
        return None, None


def _fetch_stooq_single(ticker: str) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    """Fetch close + volume from stooq.com CSV for one ticker."""
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200 or len(r.text) < 100:
            return None, None
        df = pd.read_csv(StringIO(r.text), parse_dates=["Date"])
        df = df.sort_values("Date").set_index("Date").iloc[-130:]
        close = df["Close"].dropna() if "Close" in df.columns else None
        volume = df["Volume"].dropna() if "Volume" in df.columns else None
        return close, volume
    except Exception as e:
        print(f"[top_movers/stooq] {ticker}: {e}")
        return None, None


def _fetch_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (closes_df, volumes_df) with tickers as columns.
    Tries yfinance first, falls back to stooq.
    """
    tickers = list(ETF_UNIVERSE.keys())

    # Try yfinance
    closes, volumes = _fetch_yfinance_bulk(tickers)
    if closes is not None and not closes.empty:
        # Verify we got real data (not all NaN)
        if closes.notna().sum().sum() > len(tickers) * 10:
            print(f"[top_movers] yfinance OK: {closes.shape}")
            return closes, volumes if volumes is not None else pd.DataFrame()

    # Fall back to stooq (sequential)
    print("[top_movers] Falling back to stooq...")
    close_dict, vol_dict = {}, {}
    for tk in tickers:
        c, v = _fetch_stooq_single(tk)
        if c is not None:
            close_dict[tk] = c
        if v is not None:
            vol_dict[tk] = v
        time.sleep(0.2)

    if not close_dict:
        return pd.DataFrame(), pd.DataFrame()

    closes  = pd.DataFrame(close_dict).sort_index()
    volumes = pd.DataFrame(vol_dict).sort_index() if vol_dict else pd.DataFrame()
    print(f"[top_movers] stooq OK: {closes.shape}")
    return closes, volumes


# ── Performance computation ────────────────────────────────────────────────────

def _pct(series: pd.Series, days: int) -> float:
    s = series.dropna()
    if len(s) < days + 1:
        return float("nan")
    idx = max(0, len(s) - days - 1)
    start = s.iloc[idx]
    return (s.iloc[-1] / start - 1) * 100 if start != 0 else float("nan")


def _vol_ratio(vol_series: pd.Series) -> float:
    """Recent 5-day avg volume vs 20-day avg volume."""
    v = vol_series.dropna()
    if len(v) < 25:
        return 1.0
    recent = v.iloc[-5:].mean()
    baseline = v.iloc[-25:-5].mean()
    return round(recent / baseline, 2) if baseline != 0 else 1.0


# ── Institutional Flow Score ───────────────────────────────────────────────────

def _flow_score(row: pd.Series) -> float:
    """
    Composite score (higher = stronger institutional inflow signal).

    Components:
      40% Momentum consistency  = equal-weighted avg of 1W, 1M, 3M performance
      30% Acceleration          = 1M perf minus (3M perf ÷ 3) monthly run-rate
      20% Volume conviction     = relative volume ratio (recent vs baseline), capped at 3×
      10% Timeframe consistency = fraction of 1W / 1M / 3M that are positive
    """
    perfs = [row["perf_1w"], row["perf_1m"], row["perf_3m"]]

    # Skip if missing data
    if any(pd.isna(p) for p in perfs):
        return float("nan")

    momentum   = float(np.mean(perfs))
    accel      = float(row["perf_1m"] - (row["perf_3m"] / 3))
    vol_conf   = min(float(row.get("vol_ratio", 1.0)), 3.0)
    consistency = sum(1 for p in perfs if p > 0) / 3

    score = (
        0.40 * momentum
        + 0.30 * accel * 2          # scale accel to same range as momentum
        + 0.20 * vol_conf * 10      # scale vol_ratio (1–3) to ~10–30 range
        + 0.10 * consistency * 20   # scale 0–1 to 0–20 range
    )
    return round(score, 2)


def _signal_label(row: pd.Series) -> str:
    spread   = row["perf_1m"] - (row["perf_3m"] / 3)
    vol      = row.get("vol_ratio", 1.0)
    perf_1m  = row["perf_1m"]

    if spread > 1.5 and vol > 1.3:
        return "Strong Accumulation"
    if spread > 0.5 and vol > 1.1:
        return "Accumulation"
    if perf_1m > 0 and spread > 0:
        return "Inflow"
    if spread < -1.5 and vol < 0.9:
        return "Strong Distribution"
    if spread < -0.5:
        return "Distribution"
    if perf_1m < 0:
        return "Outflow"
    return "Neutral"


SIGNAL_COLORS = {
    "Strong Accumulation": ("#1D9E75", "#E1F5EE"),
    "Accumulation":        ("#2BAD7E", "#E8F8F2"),
    "Inflow":              ("#378ADD", "#E6F1FB"),
    "Neutral":             ("#888780", "#F1EFE8"),
    "Outflow":             ("#D85A30", "#FDF0EA"),
    "Distribution":        ("#D04020", "#FCEBEB"),
    "Strong Distribution": ("#A32D2D", "#F9DEDE"),
}


# ── Main entry point ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_top_movers(top_n: int = 10) -> pd.DataFrame:
    """
    Returns a DataFrame of the top N ETFs ranked by Institutional Flow Score.

    Columns:
        ticker, name, category, perf_1d, perf_1w, perf_1m, perf_3m,
        vol_ratio, spread, flow_score, signal, signal_fg, signal_bg
    """
    closes, volumes = _fetch_prices()

    if closes.empty:
        st.warning("⚠️ Could not fetch ETF universe data — top movers unavailable.", icon="📡")
        return pd.DataFrame()

    records = []
    for ticker, (name, category) in ETF_UNIVERSE.items():
        if ticker not in closes.columns:
            continue
        s = closes[ticker]
        v = volumes[ticker] if not volumes.empty and ticker in volumes.columns else pd.Series(dtype=float)

        rec = {
            "ticker":    ticker,
            "name":      name,
            "category":  category,
            "perf_1d":   round(_pct(s, 1),  2),
            "perf_1w":   round(_pct(s, 5),  2),
            "perf_1m":   round(_pct(s, 21), 2),
            "perf_3m":   round(_pct(s, 63), 2),
            "vol_ratio": _vol_ratio(v) if not v.empty else 1.0,
        }
        rec["spread"]     = round(rec["perf_1m"] - (rec["perf_3m"] / 3), 2)
        rec["flow_score"] = _flow_score(pd.Series(rec))
        rec["signal"]     = _signal_label(pd.Series(rec))
        sig_colors        = SIGNAL_COLORS.get(rec["signal"], ("#888780", "#F1EFE8"))
        rec["signal_fg"]  = sig_colors[0]
        rec["signal_bg"]  = sig_colors[1]
        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = (
        pd.DataFrame(records)
        .dropna(subset=["flow_score"])
        .sort_values("flow_score", ascending=False)
        .reset_index(drop=True)
    )
    return df.head(top_n)