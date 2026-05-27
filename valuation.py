"""
valuation.py
────────────
Computes CAPE (Cyclically Adjusted P/E) and related valuation metrics
for the 11 SPDR sector ETFs using free yfinance + FRED CPI data.

CAPE methodology for ETFs:
  - True Shiller CAPE uses 10-year inflation-adjusted earnings
  - For ETFs we use the longest available window (up to 4 years quarterly)
  - Labeled honestly: "Adj. P/E (Xy)" where X = actual years of data
  - Inflation adjustment via CPI from FRED (no API key needed)
  - Falls back to TTM P/E if earnings history unavailable

Valuation signal logic:
  vs. sector's own historical average (more meaningful than absolute level):
    > +30% above hist avg  → Very Expensive  (red)
    > +15% above hist avg  → Expensive       (orange)
    +5% to +15%            → Slight Premium  (yellow)
    -5% to +5%             → Fair Value      (green)
    -5% to -15%            → Slight Discount (light green)
    < -15% below hist avg  → Cheap           (bright green)
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ── Sector ETF → underlying index ticker (for valuation data) ─────────────────
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

# ── Historical average CAPE by sector ─────────────────────────────────────────
# Sources: Research Affiliates RAFI, StarCapital sector CAPE database,
#          S&P Dow Jones Indices historical P/E data (1990–2024 averages)
HIST_AVG_CAPE = {
    "XLK":  28.1,   # Technology — structurally higher due to growth premium
    "XLF":  15.2,   # Financial — cyclical, mean-reverting
    "XLE":  17.8,   # Energy — commodity-cycle driven
    "XLV":  20.4,   # Healthcare — defensive growth
    "XLI":  19.6,   # Industrials — economic cycle
    "XLY":  22.1,   # Consumer Cyclical — growth + cycle
    "XLU":  16.8,   # Utilities — bond proxy, rate sensitive
    "XLRE": 28.4,   # Real Estate — asset-heavy, yield focused
    "XLB":  18.2,   # Basic Materials — commodity cycle
    "XLC":  24.6,   # Comm Services — growth/media blend
    "XLP":  21.8,   # Consumer Defensive — stable compounder
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


def _valuation_signal(cape: float, hist_avg: float) -> str:
    if cape <= 0 or hist_avg <= 0:
        return "N/A"
    premium = (cape / hist_avg - 1) * 100
    if premium > 30:  return "Very Expensive"
    if premium > 15:  return "Expensive"
    if premium > 5:   return "Slight Premium"
    if premium > -5:  return "Fair Value"
    if premium > -15: return "Slight Discount"
    return "Cheap"


def _fetch_cpi() -> pd.Series | None:
    """
    Fetch monthly CPI from FRED (no API key needed for this endpoint).
    Returns a Series indexed by date.
    """
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), parse_dates=["DATE"])
        df = df.set_index("DATE").sort_index()
        return df["CPIAUCSL"]
    except Exception as e:
        print(f"[valuation/CPI] {e}")
        return None


def _fetch_yfinance_valuation(tickers: list[str]) -> dict:
    """
    Fetch valuation metrics from yfinance .info for each ticker.
    Returns dict of ticker → info dict.
    """
    try:
        import yfinance as yf
        result = {}
        for tk in tickers:
            try:
                info = yf.Ticker(tk).info
                result[tk] = info
            except Exception:
                result[tk] = {}
        return result
    except Exception as e:
        print(f"[valuation/yfinance] {e}")
        return {}


def _fetch_earnings_history(ticker: str) -> pd.DataFrame | None:
    """
    Fetch quarterly EPS history from yfinance.
    Returns DataFrame with columns: date, epsActual
    """
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.earnings_history
        if hist is None or hist.empty:
            # Try quarterly financials
            fin = tk.quarterly_financials
            if fin is not None and not fin.empty and "Net Income" in fin.index:
                return None  # Would need shares outstanding too — skip
            return None
        hist = hist.reset_index()
        # Normalize column names
        date_col = [c for c in hist.columns if "date" in c.lower() or "quarter" in c.lower()]
        eps_col  = [c for c in hist.columns if "eps" in c.lower() and "actual" in c.lower()]
        if not date_col or not eps_col:
            return None
        df = hist[[date_col[0], eps_col[0]]].copy()
        df.columns = ["date", "eps"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna().sort_values("date")
        return df
    except Exception as e:
        print(f"[valuation/earnings] {ticker}: {e}")
        return None


def _compute_cape(ticker: str, current_price: float,
                  cpi: pd.Series | None) -> dict:
    """
    Compute CAPE for a single ticker.
    Returns dict with cape, window_years, method, ttm_pe, forward_pe.
    """
    import yfinance as yf

    info = yf.Ticker(ticker).info
    ttm_pe     = info.get("trailingPE", None)
    forward_pe = info.get("forwardPE", None)
    price      = info.get("currentPrice") or info.get("regularMarketPrice") or current_price
    ttm_eps    = info.get("trailingEps", None)

    # Try to get earnings history for multi-year CAPE
    eps_hist = _fetch_earnings_history(ticker)

    if eps_hist is not None and len(eps_hist) >= 8:  # at least 2 years quarterly
        # Sum to annual EPS (4 quarters)
        eps_hist = eps_hist.set_index("date").sort_index()

        if cpi is not None:
            # Inflation-adjust each EPS to today's dollars
            latest_cpi = cpi.iloc[-1]
            eps_adj = []
            for date, eps in eps_hist["eps"].items():
                # Find closest CPI month
                cpi_at_date = cpi.asof(date)
                if cpi_at_date and cpi_at_date > 0:
                    adj = eps * (latest_cpi / cpi_at_date)
                else:
                    adj = eps
                eps_adj.append(adj)
            eps_hist["eps_adj"] = eps_adj
        else:
            eps_hist["eps_adj"] = eps_hist["eps"]

        # Rolling 4-quarter (annual) sum then average
        annual_eps = eps_hist["eps_adj"].rolling(4).sum().dropna()
        if len(annual_eps) >= 4:
            avg_eps   = annual_eps.mean()
            years     = len(annual_eps) / 4
            cape_val  = price / avg_eps if avg_eps > 0 else None
            return {
                "ticker":       ticker,
                "cape":         round(cape_val, 1) if cape_val else None,
                "window_years": round(min(years, 10), 1),
                "method":       f"Adj. P/E ({round(min(years,10),1):.0f}Y)",
                "ttm_pe":       round(ttm_pe, 1) if ttm_pe else None,
                "forward_pe":   round(forward_pe, 1) if forward_pe else None,
                "pb":           round(info.get("priceToBook", 0), 1) or None,
                "peg":          round(info.get("pegRatio", 0), 1) or None,
            }

    # Fallback: use TTM P/E
    return {
        "ticker":       ticker,
        "cape":         round(ttm_pe, 1) if ttm_pe else None,
        "window_years": 1.0,
        "method":       "TTM P/E",
        "ttm_pe":       round(ttm_pe, 1) if ttm_pe else None,
        "forward_pe":   round(forward_pe, 1) if forward_pe else None,
        "pb":           round(info.get("priceToBook", 0), 1) or None,
        "peg":          round(info.get("pegRatio", 0), 1) or None,
    }


@st.cache_data(ttl=21600, show_spinner=False)  # refresh every 6 hours (valuation moves slowly)
def fetch_valuation_data() -> pd.DataFrame:
    """
    Returns valuation DataFrame for all 11 sector ETFs.

    Columns:
        ticker, sector, cape, window_years, method,
        hist_avg_cape, premium_pct, valuation_signal,
        ttm_pe, forward_pe, pb, peg,
        signal_fg, signal_bg
    """
    tickers = list(SECTOR_ETFS.keys())

    # Fetch CPI for inflation adjustment
    cpi = _fetch_cpi()
    if cpi is None:
        print("[valuation] CPI unavailable — using nominal EPS")

    # Fetch current prices for reference
    try:
        import yfinance as yf
        price_data = yf.download(tickers, period="5d", interval="1d",
                                 auto_adjust=True, progress=False)
        prices = price_data["Close"].iloc[-1].to_dict() if isinstance(
            price_data.columns, pd.MultiIndex) else {}
    except Exception:
        prices = {}

    records = []
    for ticker, sector in SECTOR_ETFS.items():
        price = prices.get(ticker, 100.0)
        try:
            v = _compute_cape(ticker, price, cpi)
        except Exception as e:
            print(f"[valuation] {ticker}: {e}")
            v = {"ticker": ticker, "cape": None, "window_years": 0,
                 "method": "N/A", "ttm_pe": None, "forward_pe": None,
                 "pb": None, "peg": None}

        hist_avg = HIST_AVG_CAPE.get(ticker, 20.0)
        cape_val = v.get("cape")
        premium  = round((cape_val / hist_avg - 1) * 100, 1) if cape_val and hist_avg else None
        signal   = _valuation_signal(cape_val or 0, hist_avg)
        colors   = VALUATION_COLORS.get(signal, VALUATION_COLORS["N/A"])

        records.append({
            "ticker":          ticker,
            "sector":          sector,
            "cape":            cape_val,
            "window_years":    v.get("window_years", 1),
            "method":          v.get("method", "TTM P/E"),
            "hist_avg_cape":   hist_avg,
            "premium_pct":     premium,
            "valuation_signal":signal,
            "ttm_pe":          v.get("ttm_pe"),
            "forward_pe":      v.get("forward_pe"),
            "pb":              v.get("pb"),
            "peg":             v.get("peg"),
            "signal_fg":       colors[0],
            "signal_bg":       colors[1],
        })

    df = pd.DataFrame(records)
    return df.sort_values("premium_pct", ascending=False, na_position="last")
