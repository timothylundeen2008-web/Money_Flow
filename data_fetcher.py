"""
data_fetcher.py
───────────────
Fetches sector ETF performance data from Finviz groups page (free tier).

Finviz's /groups.ashx?g=sector endpoint returns a table with columns:
  Name, Market Cap, P/E, Fwd P/E, PEG, P/S, P/B, P/C, P/FCF,
  EPS past 5Y, EPS next 5Y, Sales past 5Y, Float Short,
  Perf Week, Perf Month, Perf Quart, Perf Half, Perf Year, Perf YTD,
  Avg Volume, Rel Volume, Change, Volume

We scrape multiple timeframe views and merge them.
No API key required. Rate-limit friendly (1 req/timeframe).
"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Sector name → SPDR ETF ticker mapping
SECTOR_TICKERS = {
    "Technology":             "XLK",
    "Financial":              "XLF",
    "Energy":                 "XLE",
    "Healthcare":             "XLV",
    "Industrials":            "XLI",
    "Consumer Cyclical":      "XLY",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Basic Materials":        "XLB",
    "Communication Services": "XLC",
    "Consumer Defensive":     "XLP",
}

# Finviz group performance timeframe URL params
# o=perf_* controls sort; we always fetch all rows
FINVIZ_URL = "https://finviz.com/groups.ashx"
BASE_PARAMS = {"g": "sector", "v": "120"}  # v=120 = performance view

# Cache timestamp
_cache_time: datetime | None = None


def get_cache_age_minutes() -> float:
    if _cache_time is None:
        return 999.0
    return (datetime.now() - _cache_time).total_seconds() / 60


def _fetch_finviz_table(params: dict, retries: int = 3) -> pd.DataFrame | None:
    """
    Fetches Finviz groups table with given params.
    Returns a DataFrame of the parsed HTML table.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(
                FINVIZ_URL,
                params=params,
                headers=HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Find the groups table
            table = soup.find("table", {"class": "groups_table"})
            if table is None:
                # Try alternative table class names finviz uses
                table = soup.find("table", id="groups-table")
            if table is None:
                tables = soup.find_all("table")
                # The groups table is typically the largest table on the page
                table = max(tables, key=lambda t: len(t.find_all("tr")), default=None)

            if table is None:
                time.sleep(2 ** attempt)
                continue

            rows = table.find_all("tr")
            if len(rows) < 2:
                time.sleep(2 ** attempt)
                continue

            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            data = []
            for row in rows[1:]:
                cells = row.find_all("td")
                if cells:
                    data.append([c.get_text(strip=True) for c in cells])

            if not data:
                time.sleep(2 ** attempt)
                continue

            df = pd.DataFrame(data, columns=headers[: len(data[0])])
            return df

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[data_fetcher] Failed after {retries} attempts: {e}")
                return None

    return None


def _parse_pct(val: str) -> float:
    """Parse '3.45%' or '-1.2%' → float. Returns NaN on failure."""
    try:
        return float(val.replace("%", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return float("nan")


def _normalize_sector_name(name: str) -> str:
    """Normalize Finviz sector names to our canonical names."""
    mapping = {
        "financial":              "Financial",
        "financials":             "Financial",
        "technology":             "Technology",
        "energy":                 "Energy",
        "health care":            "Healthcare",
        "healthcare":             "Healthcare",
        "industrials":            "Industrials",
        "consumer cyclical":      "Consumer Cyclical",
        "consumer discretionary": "Consumer Cyclical",
        "utilities":              "Utilities",
        "real estate":            "Real Estate",
        "basic materials":        "Basic Materials",
        "materials":              "Basic Materials",
        "communication services": "Communication Services",
        "consumer defensive":     "Consumer Defensive",
        "consumer staples":       "Consumer Defensive",
    }
    return mapping.get(name.lower().strip(), name.strip())


def fetch_sector_data() -> pd.DataFrame | None:
    """
    Main entry point. Fetches sector performance for all timeframes
    from Finviz and returns a unified DataFrame.

    Columns returned:
        sector, ticker,
        perf_1d, perf_1w, perf_1m, perf_3m, perf_6m, perf_1y, perf_ytd
    """
    global _cache_time

    # Finviz performance view (v=120) has columns:
    # Name, Perf Week, Perf Month, Perf Quart, Perf Half, Perf Year, Perf YTD, Change (=1D)
    params = {**BASE_PARAMS}
    df_raw = _fetch_finviz_table(params)

    if df_raw is None or df_raw.empty:
        print("[data_fetcher] Primary fetch failed, trying fallback URL...")
        df_raw = _fetch_finviz_table({**BASE_PARAMS, "v": "110"})

    if df_raw is None or df_raw.empty:
        return _fallback_demo_data()

    # Map Finviz column names (they vary slightly by version)
    col_map = {
        # Finviz column name variations → our standard name
        "Name":        "sector_raw",
        "Perf Week":   "perf_1w",
        "Perf Month":  "perf_1m",
        "Perf Quart":  "perf_3m",
        "Perf Half":   "perf_6m",
        "Perf Year":   "perf_1y",
        "Perf YTD":    "perf_ytd",
        "Change":      "perf_1d",
        # Alternative column names
        "Performance (Week)":    "perf_1w",
        "Performance (Month)":   "perf_1m",
        "Performance (Quarter)": "perf_3m",
        "Performance (Half Y)":  "perf_6m",
        "Performance (Year)":    "perf_1y",
        "Performance (YTD)":     "perf_ytd",
    }

    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

    # Find the sector name column
    name_col = "sector_raw"
    if name_col not in df_raw.columns:
        # Try to find it
        for c in df_raw.columns:
            if c.lower() in ("name", "sector", "group"):
                df_raw = df_raw.rename(columns={c: name_col})
                break

    if name_col not in df_raw.columns:
        print("[data_fetcher] Could not identify sector name column")
        return _fallback_demo_data()

    # Keep only rows that match known sectors
    perf_cols = ["perf_1d", "perf_1w", "perf_1m", "perf_3m", "perf_6m", "perf_1y", "perf_ytd"]

    records = []
    for _, row in df_raw.iterrows():
        raw_name = str(row.get(name_col, "")).strip()
        if not raw_name or raw_name.lower() in ("", "name", "sector"):
            continue

        canonical = _normalize_sector_name(raw_name)
        ticker = SECTOR_TICKERS.get(canonical, "")
        if not ticker:
            continue

        record = {"sector": canonical, "ticker": ticker}
        for col in perf_cols:
            val = row.get(col, "0%")
            record[col] = _parse_pct(str(val))

        records.append(record)

    if not records:
        print("[data_fetcher] No matching sectors found in scraped data")
        return _fallback_demo_data()

    df = pd.DataFrame(records)

    # Fill any missing perf columns with NaN
    for col in perf_cols:
        if col not in df.columns:
            df[col] = float("nan")

    _cache_time = datetime.now()
    print(f"[data_fetcher] Successfully fetched {len(df)} sectors at {_cache_time}")
    return df


def _fallback_demo_data() -> pd.DataFrame:
    """
    Returns illustrative demo data when Finviz is unreachable.
    Clearly marked as demo data in the UI via a Streamlit warning.
    This matches the data used in the Claude widget.
    """
    import streamlit as st
    st.warning(
        "⚠️ Could not reach Finviz — displaying demo data. "
        "Real data will load when the connection is restored.",
        icon="📡",
    )

    demo = [
        {"sector": "Technology",             "ticker": "XLK",  "perf_1d": 1.2,  "perf_1w": 2.8,  "perf_1m": 4.1,  "perf_3m": 11.2, "perf_6m": 14.8, "perf_1y": 28.4, "perf_ytd": 9.3},
        {"sector": "Financial",              "ticker": "XLF",  "perf_1d": 0.8,  "perf_1w": 1.9,  "perf_1m": 3.6,  "perf_3m": 8.4,  "perf_6m": 12.1, "perf_1y": 21.3, "perf_ytd": 7.1},
        {"sector": "Energy",                 "ticker": "XLE",  "perf_1d": -0.4, "perf_1w": -1.2, "perf_1m": -2.8, "perf_3m": -6.1, "perf_6m": -4.2, "perf_1y": 3.1,  "perf_ytd": -4.8},
        {"sector": "Healthcare",             "ticker": "XLV",  "perf_1d": 0.3,  "perf_1w": 0.6,  "perf_1m": 1.2,  "perf_3m": 3.4,  "perf_6m": 6.8,  "perf_1y": 12.1, "perf_ytd": 2.4},
        {"sector": "Industrials",            "ticker": "XLI",  "perf_1d": 0.6,  "perf_1w": 1.4,  "perf_1m": 2.9,  "perf_3m": 7.8,  "perf_6m": 9.4,  "perf_1y": 18.6, "perf_ytd": 5.8},
        {"sector": "Consumer Cyclical",      "ticker": "XLY",  "perf_1d": -0.2, "perf_1w": -0.8, "perf_1m": 0.4,  "perf_3m": -2.1, "perf_6m": 1.2,  "perf_1y": 8.4,  "perf_ytd": -1.4},
        {"sector": "Utilities",              "ticker": "XLU",  "perf_1d": 0.1,  "perf_1w": 0.3,  "perf_1m": 0.8,  "perf_3m": 2.1,  "perf_6m": -1.8, "perf_1y": 4.2,  "perf_ytd": 1.1},
        {"sector": "Real Estate",            "ticker": "XLRE", "perf_1d": -0.6, "perf_1w": -1.8, "perf_1m": -3.4, "perf_3m": -7.2, "perf_6m": -9.1, "perf_1y": -4.8, "perf_ytd": -5.2},
        {"sector": "Basic Materials",        "ticker": "XLB",  "perf_1d": 0.4,  "perf_1w": 1.1,  "perf_1m": 2.2,  "perf_3m": 5.6,  "perf_6m": 7.2,  "perf_1y": 14.8, "perf_ytd": 4.1},
        {"sector": "Communication Services", "ticker": "XLC",  "perf_1d": 0.9,  "perf_1w": 2.1,  "perf_1m": 3.8,  "perf_3m": 9.6,  "perf_6m": 13.4, "perf_1y": 22.8, "perf_ytd": 7.8},
        {"sector": "Consumer Defensive",     "ticker": "XLP",  "perf_1d": -0.1, "perf_1w": 0.2,  "perf_1m": 0.6,  "perf_3m": 1.4,  "perf_6m": 2.8,  "perf_1y": 6.4,  "perf_ytd": 0.8},
    ]
    _cache_time = datetime.now()
    return pd.DataFrame(demo)