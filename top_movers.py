"""
top_movers.py  (v3 — tiered volume thresholds + rotation flow data)
────────────────────────────────────────────────────────────────────
Data-driven volume spike thresholds based on Average Daily Dollar Volume (ADDV).

THRESHOLD RESEARCH (sources: ValuEngine Oct 2024, SeekingAlpha, Morpheus Trading,
                             Oxford Academic RFS ETF Liquidity study):

  Tier 1  >$2B ADDV   → 1.25x threshold
    XLK ($2.74B), XLF, XLV ($7B), QQQ, SPY
    At $2B+ daily, institutions move $500M routinely. Only 1.25x+ = directional
    conviction. 1.5x on these = massive event (earnings, macro shock).

  Tier 2  $200M–$2B   → 1.50x threshold  (the classic institutional signal)
    Most SPDR sector ETFs (XLE, XLI, XLY, XLC, XLP, XLU, XLRE, XLB), IWM, GLD, TLT
    $200M–$2B ADV: 1.5x filters noise, catches real rotation flows.

  Tier 3  $50M–$200M  → 2.00x threshold
    Sub-sector ETFs: SMH, IBB, ITA, PAVE, KRE, EEM, HYG
    At this level a single large hedge fund trade = 1.5x. Need 2.0x for
    broad institutional confirmation.

  Tier 4  <$50M       → 3.00x threshold
    Niche/thematic: XBI, SKYY, HACK, AMLP, DBA
    Retail noise routinely spikes these 1.5–2x. 3.0x = real institutional entry.
    Treat as early-signal requiring next-day confirmation.
"""

import time
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime
import streamlit as st

# ── ETF Universe: ticker → (name, category, addv_tier) ────────────────────────
# addv_tier: 1=mega(>$2B), 2=high($200M-$2B), 3=moderate($50-200M), 4=lower(<$50M)

ETF_UNIVERSE = {
    # Technology
    "SMH":  ("Semiconductors",        "Technology",    3),
    "SOXX": ("Semiconductors II",     "Technology",    3),
    "IGV":  ("Software",              "Technology",    3),
    "SKYY": ("Cloud Computing",       "Technology",    4),
    "HACK": ("Cybersecurity",         "Technology",    4),
    # Financial
    "KRE":  ("Regional Banks",        "Financial",     3),
    "KBE":  ("Banks Broad",           "Financial",     3),
    "IAI":  ("Broker-Dealers",        "Financial",     3),
    # Healthcare
    "IBB":  ("Biotech",               "Healthcare",    3),
    "XBI":  ("Biotech Small Cap",     "Healthcare",    4),
    "IHI":  ("Medical Devices",       "Healthcare",    3),
    "PPH":  ("Pharmaceuticals",       "Healthcare",    3),
    # Energy
    "XOP":  ("Oil & Gas E&P",         "Energy",        3),
    "OIH":  ("Oil Services",          "Energy",        3),
    "AMLP": ("Pipelines / MLP",       "Energy",        4),
    # Industrials
    "ITA":  ("Aerospace & Defense",   "Industrials",   3),
    "XTN":  ("Transportation",        "Industrials",   3),
    "PAVE": ("Infrastructure",        "Industrials",   3),
    # Consumer
    "XRT":  ("Retail",                "Consumer Cyclical", 3),
    "XHB":  ("Homebuilders",          "Consumer Cyclical", 3),
    "PBJ":  ("Food & Beverage",       "Consumer Defensive", 4),
    # Broad Market
    "QQQ":  ("Nasdaq 100",            "Broad Market",  1),
    "IWM":  ("Russell 2000",          "Broad Market",  2),
    "IWO":  ("Russell 2000 Growth",   "Broad Market",  2),
    "MDY":  ("S&P MidCap 400",        "Broad Market",  2),
    # Fixed Income
    "TLT":  ("Long Bonds 20Y+",       "Fixed Income",  2),
    "HYG":  ("High Yield Corp",       "Fixed Income",  2),
    "LQD":  ("Investment Grade Corp", "Fixed Income",  2),
    "EMB":  ("Emerging Mkt Bonds",    "Fixed Income",  3),
    "TIP":  ("TIPS / Inflation",      "Fixed Income",  2),
    # Commodities
    "GLD":  ("Gold",                  "Commodities",   2),
    "SLV":  ("Silver",                "Commodities",   2),
    "PDBC": ("Commodities Broad",     "Commodities",   3),
    "USO":  ("Oil (WTI)",             "Commodities",   3),
    "DBA":  ("Agriculture",           "Commodities",   4),
    # International
    "EEM":  ("Emerging Markets",      "International", 2),
    "EFA":  ("Developed Intl EAFE",   "International", 2),
    "EWJ":  ("Japan",                 "International", 3),
    "FXI":  ("China Large Cap",       "International", 3),
    "INDA": ("India",                 "International", 3),
}

# ── Tiered thresholds ──────────────────────────────────────────────────────────
TIER_THRESHOLDS = {1: 1.25, 2: 1.50, 3: 2.00, 4: 3.00}
TIER_LABELS     = {
    1: ("Mega Liquid",    ">$2B ADV",     "1.25×"),
    2: ("High Liquid",    "$200M–$2B",    "1.50×"),
    3: ("Moderate Liq.",  "$50M–$200M",   "2.00×"),
    4: ("Lower Liquid",   "<$50M ADV",    "3.00×"),
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

SIGNAL_COLORS = {
    "Strong Accumulation": ("#1D9E75", "#0d3326"),
    "Accumulation":        ("#2BAD7E", "#0e2e22"),
    "Inflow":              ("#378ADD", "#0e2240"),
    "Neutral":             ("#888780", "#1e2330"),
    "Outflow":             ("#D85A30", "#2e1810"),
    "Distribution":        ("#D04020", "#2e1208"),
    "Strong Distribution": ("#A32D2D", "#250c0c"),
}

VOL_SPIKE_THRESHOLD = 1.5   # legacy default; per-ticker threshold now from TIER_THRESHOLDS


# ── Price + volume fetching ────────────────────────────────────────────────────

def _fetch_yfinance(tickers):
    try:
        import yfinance as yf
        raw = yf.download(tickers, period="130d", interval="1d",
                          auto_adjust=True, progress=False, threads=True, timeout=25)
        if raw.empty:
            return None, None
        closes  = raw["Close"]  if isinstance(raw.columns, pd.MultiIndex) else raw
        volumes = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else pd.DataFrame()
        return closes, volumes
    except Exception as e:
        print(f"[top_movers/yfinance] {e}")
        return None, None


def _fetch_stooq_single(ticker):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200 or len(r.text) < 100:
            return None, None
        df = pd.read_csv(StringIO(r.text), parse_dates=["Date"])
        df = df.sort_values("Date").set_index("Date").iloc[-130:]
        close  = df["Close"].dropna()  if "Close"  in df.columns else None
        volume = df["Volume"].dropna() if "Volume" in df.columns else None
        return close, volume
    except Exception as e:
        print(f"[top_movers/stooq] {ticker}: {e}")
        return None, None


def _fetch_prices():
    tickers = list(ETF_UNIVERSE.keys())
    closes, volumes = _fetch_yfinance(tickers)
    if closes is not None and closes.notna().sum().sum() > len(tickers) * 10:
        return closes, volumes if volumes is not None else pd.DataFrame()
    print("[top_movers] yfinance failed, trying stooq...")
    cd, vd = {}, {}
    for tk in tickers:
        c, v = _fetch_stooq_single(tk)
        if c is not None: cd[tk] = c
        if v is not None: vd[tk] = v
        time.sleep(0.2)
    if not cd:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(cd).sort_index(), pd.DataFrame(vd).sort_index() if vd else pd.DataFrame()


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _pct(s, days):
    s = s.dropna()
    if len(s) < days + 1: return float("nan")
    start = s.iloc[max(0, len(s)-days-1)]
    return (s.iloc[-1]/start - 1)*100 if start != 0 else float("nan")


def _vol_ratio(v):
    v = v.dropna()
    if len(v) < 25: return 1.0
    recent   = v.iloc[-5:].mean()
    baseline = v.iloc[-25:-5].mean()
    return round(recent/baseline, 2) if baseline != 0 else 1.0


def _addv_usd(c, v):
    """Estimate Average Daily Dollar Volume (millions) from last 20 days."""
    c, v = c.dropna(), v.dropna()
    n = min(len(c), len(v), 20)
    if n < 5: return 0.0
    return round(float((c.iloc[-n:].values * v.iloc[-n:].values).mean()) / 1e6, 1)


# ── Signal logic (tier-aware) ──────────────────────────────────────────────────

def _signal_label(row):
    spread    = row["perf_1m"] - (row["perf_3m"] / 3)
    vol       = row.get("vol_ratio", 1.0)
    perf_1m   = row["perf_1m"]
    threshold = row.get("spike_threshold", 1.5)
    spike     = vol >= threshold

    if spike and spread > 1.5 and perf_1m > 0: return "Strong Accumulation"
    if spike and spread > 0   and perf_1m > 0: return "Accumulation"
    if spike and perf_1m < 0  and spread < -1: return "Strong Distribution"
    if spike and perf_1m < 0:                  return "Distribution"
    if spread > 1.5 and vol > 1.1:             return "Accumulation"
    if perf_1m > 0  and spread > 0:            return "Inflow"
    if spread < -1.5 and vol < 0.9:            return "Distribution"
    if perf_1m < 0  and spread < -0.5:         return "Outflow"
    return "Neutral"


# ── Flow score ─────────────────────────────────────────────────────────────────

def _flow_score(row):
    perfs = [row["perf_1w"], row["perf_1m"], row["perf_3m"]]
    if any(pd.isna(p) for p in perfs): return float("nan")
    momentum    = float(np.mean(perfs))
    accel       = float(row["perf_1m"] - (row["perf_3m"] / 3))
    vol_conf    = min(float(row.get("vol_ratio", 1.0)), 3.0)
    consistency = sum(1 for p in perfs if p > 0) / 3
    score = (0.40*momentum + 0.30*accel*2 + 0.20*vol_conf*10 + 0.10*consistency*20)
    if row.get("vol_spike", False):
        score += 8.0    # bonus for crossing tier-appropriate threshold
    return round(score, 2)


# ── Main fetch ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_top_movers(top_n=10):
    closes, volumes = _fetch_prices()
    if closes.empty:
        st.warning("⚠️ Could not fetch ETF data — top movers unavailable.", icon="📡")
        return pd.DataFrame()

    records = []
    for ticker, (name, category, tier) in ETF_UNIVERSE.items():
        if ticker not in closes.columns: continue
        s = closes[ticker]
        v = volumes[ticker] if not volumes.empty and ticker in volumes.columns else pd.Series(dtype=float)

        threshold = TIER_THRESHOLDS[tier]
        vol_r     = _vol_ratio(v) if not v.empty else 1.0
        addv      = _addv_usd(s, v) if not v.empty else 0.0

        rec = {
            "ticker":          ticker,
            "name":            name,
            "category":        category,
            "tier":            tier,
            "tier_label":      TIER_LABELS[tier][0],
            "addv_range":      TIER_LABELS[tier][1],
            "spike_threshold": threshold,
            "spike_label":     TIER_LABELS[tier][2],
            "addv_M":          addv,
            "perf_1d":         round(_pct(s, 1),  2),
            "perf_1w":         round(_pct(s, 5),  2),
            "perf_1m":         round(_pct(s, 21), 2),
            "perf_3m":         round(_pct(s, 63), 2),
            "vol_ratio":       vol_r,
        }
        rec["spread"]     = round(rec["perf_1m"] - (rec["perf_3m"] / 3), 2)
        rec["vol_spike"]  = vol_r >= threshold
        rec["flow_score"] = _flow_score(pd.Series(rec))
        rec["signal"]     = _signal_label(pd.Series(rec))
        sc = SIGNAL_COLORS.get(rec["signal"], ("#888780", "#1e2330"))
        rec["signal_fg"]  = sc[0]
        rec["signal_bg"]  = sc[1]
        records.append(rec)

    if not records: return pd.DataFrame()
    df = (pd.DataFrame(records)
            .dropna(subset=["flow_score"])
            .sort_values("flow_score", ascending=False)
            .reset_index(drop=True))
    return df.head(top_n)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector_flow_data():
    """
    Returns all ETFs (not top-N) for the rotation flow visualization.
    Groups by category and computes aggregate flow score per sector.
    """
    closes, volumes = _fetch_prices()
    if closes.empty: return pd.DataFrame()

    records = []
    for ticker, (name, category, tier) in ETF_UNIVERSE.items():
        if ticker not in closes.columns: continue
        s = closes[ticker]
        v = volumes[ticker] if not volumes.empty and ticker in volumes.columns else pd.Series(dtype=float)
        threshold = TIER_THRESHOLDS[tier]
        vol_r     = _vol_ratio(v) if not v.empty else 1.0

        rec = {
            "ticker": ticker, "name": name, "category": category, "tier": tier,
            "spike_threshold": threshold,
            "perf_1d":  round(_pct(s, 1),  2),
            "perf_1w":  round(_pct(s, 5),  2),
            "perf_1m":  round(_pct(s, 21), 2),
            "perf_3m":  round(_pct(s, 63), 2),
            "vol_ratio": vol_r,
        }
        rec["spread"]    = round(rec["perf_1m"] - (rec["perf_3m"] / 3), 2)
        rec["vol_spike"] = vol_r >= threshold
        rec["flow_score"]= _flow_score(pd.Series(rec))
        records.append(rec)

    return pd.DataFrame(records).dropna(subset=["flow_score"]) if records else pd.DataFrame()