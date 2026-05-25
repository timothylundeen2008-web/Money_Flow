"""
data_fetcher.py  (v2 – yfinance + stooq fallback)
──────────────────────────────────────────────────
Computes all sector performance timeframes from actual ETF price history.
Primary source : yfinance (Yahoo Finance) – works on Streamlit Cloud
Fallback source: stooq.com  CSV – no API key, no rate limits
Demo fallback  : hard-coded data shown only when both sources fail

Why not Finviz scraping?
  Finviz returns HTTP 403 to cloud/server IPs (Streamlit Cloud, GitHub Actions).
  Computing from ETF price history is more accurate anyway – Finviz rounds to 1dp.
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

SECTORS = {
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

TICKERS = list(SECTORS.keys())

_cache_timestamp: datetime | None = None


def get_cache_age_minutes() -> float:
    if _cache_timestamp is None:
        return 999.0
    return (datetime.now() - _cache_timestamp).total_seconds() / 60


# ── Performance helpers ────────────────────────────────────────────────────────

def _pct_change(series: pd.Series, days_back: int) -> float:
    """
    Return percentage change from `days_back` trading days ago to latest close.
    Looks back up to days_back * 1.5 calendar days to account for weekends/holidays.
    """
    if len(series) < 2:
        return 0.0
    recent = series.dropna()
    if len(recent) < 2:
        return 0.0
    end_price = recent.iloc[-1]
    # Use exact index position for speed
    start_idx = max(0, len(recent) - days_back - 1)
    start_price = recent.iloc[start_idx]
    if start_price == 0:
        return 0.0
    return (end_price / start_price - 1) * 100


def _ytd_change(series: pd.Series) -> float:
    """Return YTD % change: last trading day of prev year → latest."""
    if series.empty:
        return 0.0
    recent = series.dropna()
    year_start = datetime(datetime.now().year, 1, 1)
    before_year = recent[recent.index < str(year_start)]
    if before_year.empty:
        return 0.0
    start_price = before_year.iloc[-1]
    end_price = recent.iloc[-1]
    if start_price == 0:
        return 0.0
    return (end_price / start_price - 1) * 100


# ── Source 1: yfinance ─────────────────────────────────────────────────────────

def _fetch_yfinance(tickers: list[str]) -> pd.DataFrame | None:
    """
    Download ~400 days of daily closes via yfinance.
    Returns wide DataFrame with tickers as columns, dates as index.
    """
    try:
        import yfinance as yf
        raw = yf.download(
            tickers,
            period="400d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=20,
        )
        if raw.empty:
            return None
        # Handle both single and multi-ticker return shapes
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
        closes = closes.dropna(how="all")
        return closes if not closes.empty else None
    except Exception as e:
        print(f"[yfinance] Error: {e}")
        return None


# ── Source 2: stooq CSV ────────────────────────────────────────────────────────

def _fetch_stooq_single(ticker: str) -> pd.Series | None:
    """Fetch daily close prices from stooq.com for a single ticker."""
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200 or len(r.text) < 100:
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), parse_dates=["Date"])
        df = df.sort_values("Date").set_index("Date")
        if "Close" not in df.columns:
            return None
        s = df["Close"].dropna()
        # Keep last 400 trading days
        return s.iloc[-400:] if len(s) > 400 else s
    except Exception as e:
        print(f"[stooq] {ticker}: {e}")
        return None


def _fetch_stooq(tickers: list[str]) -> pd.DataFrame | None:
    """Fetch all tickers from stooq. Slower (sequential) but reliable."""
    result = {}
    for tk in tickers:
        s = _fetch_stooq_single(tk)
        if s is not None and not s.empty:
            result[tk] = s
        time.sleep(0.3)   # polite rate limiting
    if not result:
        return None
    df = pd.DataFrame(result)
    df = df.sort_index().dropna(how="all")
    return df if not df.empty else None


# ── Build sector DataFrame from price history ──────────────────────────────────

def _build_sector_df(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Given a price DataFrame (dates × tickers), compute all performance columns.

    Trading-day approximations:
      1D  =  1 trading day
      1W  =  5 trading days
      1M  = 21 trading days
      3M  = 63 trading days
      6M  = 126 trading days
      1Y  = 252 trading days
      YTD = calendar Jan 1 → today
    """
    records = []
    for ticker, sector_name in SECTORS.items():
        if ticker not in prices.columns:
            print(f"[data_fetcher] Missing price data for {ticker}")
            continue
        s = prices[ticker].dropna()
        if len(s) < 10:
            continue
        records.append({
            "sector":   sector_name,
            "ticker":   ticker,
            "perf_1d":  _pct_change(s, 1),
            "perf_1w":  _pct_change(s, 5),
            "perf_1m":  _pct_change(s, 21),
            "perf_3m":  _pct_change(s, 63),
            "perf_6m":  _pct_change(s, 126),
            "perf_1y":  _pct_change(s, 252),
            "perf_ytd": _ytd_change(s),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Round to 2dp for display
    perf_cols = ["perf_1d","perf_1w","perf_1m","perf_3m","perf_6m","perf_1y","perf_ytd"]
    df[perf_cols] = df[perf_cols].round(2)
    return df


# ── Main entry point ───────────────────────────────────────────────────────────

def fetch_sector_data() -> pd.DataFrame:
    """
    Fetch sector ETF prices and compute all performance timeframes.
    Tries yfinance first, falls back to stooq, then demo data.
    """
    global _cache_timestamp

    # ── Try yfinance ──
    print("[data_fetcher] Trying yfinance...")
    prices = _fetch_yfinance(TICKERS)

    if prices is not None and not prices.empty:
        df = _build_sector_df(prices)
        if not df.empty and df["perf_1m"].abs().sum() > 0.1:
            _cache_timestamp = datetime.now()
            print(f"[data_fetcher] yfinance OK – {len(df)} sectors")
            return df
        else:
            print("[data_fetcher] yfinance returned flat data, trying stooq...")

    # ── Try stooq ──
    print("[data_fetcher] Trying stooq...")
    prices = _fetch_stooq(TICKERS)

    if prices is not None and not prices.empty:
        df = _build_sector_df(prices)
        if not df.empty and df["perf_1m"].abs().sum() > 0.1:
            _cache_timestamp = datetime.now()
            print(f"[data_fetcher] stooq OK – {len(df)} sectors")
            return df

    # ── Final fallback: demo data ──
    print("[data_fetcher] All sources failed – using demo data")
    st.warning(
        "⚠️ Could not fetch live data (yfinance and stooq both unreachable). "
        "Displaying illustrative demo data. Try the **🔄 Refresh** button in a few minutes.",
        icon="📡",
    )
    return _demo_data()


# ── Demo / fallback data ───────────────────────────────────────────────────────

def _demo_data() -> pd.DataFrame:
    global _cache_timestamp
    _cache_timestamp = datetime.now()
    data = [
        {"sector":"Technology",             "ticker":"XLK",  "perf_1d": 1.2,  "perf_1w": 2.8,  "perf_1m": 4.1,  "perf_3m":11.2,"perf_6m":14.8,"perf_1y":28.4,"perf_ytd": 9.3},
        {"sector":"Financial",              "ticker":"XLF",  "perf_1d": 0.8,  "perf_1w": 1.9,  "perf_1m": 3.6,  "perf_3m": 8.4,"perf_6m":12.1,"perf_1y":21.3,"perf_ytd": 7.1},
        {"sector":"Energy",                 "ticker":"XLE",  "perf_1d":-0.4,  "perf_1w":-1.2,  "perf_1m":-2.8,  "perf_3m":-6.1,"perf_6m":-4.2,"perf_1y": 3.1,"perf_ytd":-4.8},
        {"sector":"Healthcare",             "ticker":"XLV",  "perf_1d": 0.3,  "perf_1w": 0.6,  "perf_1m": 1.2,  "perf_3m": 3.4,"perf_6m": 6.8,"perf_1y":12.1,"perf_ytd": 2.4},
        {"sector":"Industrials",            "ticker":"XLI",  "perf_1d": 0.6,  "perf_1w": 1.4,  "perf_1m": 2.9,  "perf_3m": 7.8,"perf_6m": 9.4,"perf_1y":18.6,"perf_ytd": 5.8},
        {"sector":"Consumer Cyclical",      "ticker":"XLY",  "perf_1d":-0.2,  "perf_1w":-0.8,  "perf_1m": 0.4,  "perf_3m":-2.1,"perf_6m": 1.2,"perf_1y": 8.4,"perf_ytd":-1.4},
        {"sector":"Utilities",              "ticker":"XLU",  "perf_1d": 0.1,  "perf_1w": 0.3,  "perf_1m": 0.8,  "perf_3m": 2.1,"perf_6m":-1.8,"perf_1y": 4.2,"perf_ytd": 1.1},
        {"sector":"Real Estate",            "ticker":"XLRE", "perf_1d":-0.6,  "perf_1w":-1.8,  "perf_1m":-3.4,  "perf_3m":-7.2,"perf_6m":-9.1,"perf_1y":-4.8,"perf_ytd":-5.2},
        {"sector":"Basic Materials",        "ticker":"XLB",  "perf_1d": 0.4,  "perf_1w": 1.1,  "perf_1m": 2.2,  "perf_3m": 5.6,"perf_6m": 7.2,"perf_1y":14.8,"perf_ytd": 4.1},
        {"sector":"Communication Services", "ticker":"XLC",  "perf_1d": 0.9,  "perf_1w": 2.1,  "perf_1m": 3.8,  "perf_3m": 9.6,"perf_6m":13.4,"perf_1y":22.8,"perf_ytd": 7.8},
        {"sector":"Consumer Defensive",     "ticker":"XLP",  "perf_1d":-0.1,  "perf_1w": 0.2,  "perf_1m": 0.6,  "perf_3m": 1.4,"perf_6m": 2.8,"perf_1y": 6.4,"perf_ytd": 0.8},
    ]
    return pd.DataFrame(data)