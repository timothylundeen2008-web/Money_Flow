# VERSION: v4-20260721
"""
top_movers.py  (v4 — directional volume via flow_metrics)
────────────────────────────────────────────────────────────────────
CHANGES v3 -> v4 (July 2026 flow-detection review):
  [CRITICAL] _flow_score REPLACED. The old composite awarded a FLAT +8 for
             crossing the volume-spike threshold while its entire momentum
             term spanned roughly ±2, so a single loud session outweighed any
             realistic momentum difference ~4x. Mediocre names that printed one
             big volume day outranked strong quiet accumulators — the exact
             INVERSION of this framework's stated thesis that loud volume is
             retail and institutions accumulate quietly. Three of its four
             terms were also transformations of the same three price numbers,
             making a two-factor blend look like a four-factor one.
             Now split into TWO scores that are reported side by side and
             NEVER summed, because they measure opposite phenomena:
               accumulation_score — quiet, sustained, directional buying
               event_score        — loud activity + which side it closed on
  [CRITICAL] Fetches OHLCV. High/Low were downloaded and discarded; they are
             the inputs to the money-flow multiplier, the only way to tell
             BUYING volume from SELLING volume.
  [CHANGED]  _signal_label now uses CMF (directional) rather than inferring
             direction from trailing price sign. A heavy-volume day closing on
             the LOW is now labeled distribution; previously it was labeled
             "Accumulation" whenever the trailing month happened to be green.
  [NOTE]     _vol_ratio is unchanged (5-day avg vs prior 20-day baseline) and
             is still UNDIRECTED — it says how much traded, never on which
             side. It is now a gate and a magnitude input, not a signal.
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
    "Buying Event":        ("#4FB286", "#0d2b21"),
    "Inflow":              ("#378ADD", "#0e2240"),
    "Neutral":             ("#888780", "#1e2330"),
    "Neutral (px only)":   ("#6E6D68", "#1a1e29"),
    "Inflow (px only)":    ("#5E7FA8", "#131d2e"),
    "Outflow (px only)":   ("#A86A50", "#241610"),
    "Outflow":             ("#D85A30", "#2e1810"),
    "Selling Event":       ("#C25A3A", "#2b1510"),
    "Distribution":        ("#D04020", "#2e1208"),
    "Strong Distribution": ("#A32D2D", "#250c0c"),
}

VOL_SPIKE_THRESHOLD = 1.5   # legacy default; per-ticker threshold now from TIER_THRESHOLDS


# ── Price + volume fetching ────────────────────────────────────────────────────

_LAST_OHLCV = {}


def _fetch_yfinance(tickers):
    """
    Download OHLCV. period bumped 130d -> 200d so the 63-bar minimum for the
    flow metrics survives holidays and partial listings with real margin.
    Populates _LAST_OHLCV alongside the legacy close/volume return.
    """
    global _LAST_OHLCV
    try:
        import yfinance as yf
        raw = yf.download(tickers, period="200d", interval="1d",
                          auto_adjust=True, progress=False, threads=True, timeout=30)
        if raw.empty:
            return None, None
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = set(raw.columns.get_level_values(0))
            if "Close" in lvl0:
                closes, volumes = raw["Close"], raw["Volume"]
                _LAST_OHLCV = {tk: pd.DataFrame({
                                  "High": raw["High"][tk], "Low": raw["Low"][tk],
                                  "Close": raw["Close"][tk], "Volume": raw["Volume"][tk]
                               }).dropna() for tk in closes.columns}
            else:
                closes = pd.DataFrame({tk: raw[tk]["Close"] for tk in lvl0})
                volumes = pd.DataFrame({tk: raw[tk]["Volume"] for tk in lvl0})
                _LAST_OHLCV = {tk: raw[tk][["High","Low","Close","Volume"]].dropna()
                               for tk in lvl0}
        else:
            closes, volumes = raw, pd.DataFrame()
            _LAST_OHLCV = {}
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
    """
    Direction now comes from CMF (where closes land inside their ranges),
    not from the sign of trailing price. Falls back to the legacy price-based
    rule ONLY when CMF is unavailable, and says so via the "(px only)" suffix
    so a Tier-C inference is never mistaken for a Tier-B measurement.
    """
    cmf   = row.get("cmf")
    spike = bool(row.get("vol_spike", False))
    quiet = not spike

    if cmf is None or pd.isna(cmf):
        perf_1m = row.get("perf_1m", 0)
        spread  = row.get("spread", 0)
        if perf_1m > 0 and spread > 0:  return "Inflow (px only)"
        if perf_1m < 0 and spread < 0:  return "Outflow (px only)"
        return "Neutral (px only)"

    if cmf >= 0.10:
        return "Strong Accumulation" if quiet else "Buying Event"
    if cmf >= 0.05:
        return "Accumulation" if quiet else "Inflow"
    if cmf <= -0.10:
        return "Strong Distribution" if quiet else "Selling Event"
    if cmf <= -0.05:
        return "Distribution" if quiet else "Outflow"
    return "Neutral"


# ── Scoring ────────────────────────────────────────────────────────────────────
# The v3 _flow_score is GONE. See the module changelog for why. Ranking now
# uses accumulation_score from flow_metrics (quiet, directional, Tier B), with
# event_score reported alongside and never added to it.

def _attach_flow_metrics(rec, ohlcv):
    """Attach the Tier-B panel, or explicit NaNs when history is short."""
    from flow_metrics import compute_all
    if ohlcv is None or len(ohlcv) < 63:
        rec.update({"cmf": np.nan, "cmf_persist": np.nan, "ad_divergence": np.nan,
                    "mfi": np.nan, "obv_trend": np.nan,
                    "stealth_label": "Insufficient history", "stealth_score": 0,
                    "accumulation_score": np.nan, "event_score": 0.0,
                    "event_direction": "None", "insufficient_history": True})
        return rec
    rec.update(compute_all(ohlcv, rec.get("vol_ratio", 1.0),
                           rec.get("spike_threshold", 1.5)))
    return rec


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
        # momentum acceleration — renamed from "spread"; it is NOT accumulation
        rec["momentum_accel"] = round(rec["perf_1m"] - (rec["perf_3m"] / 3), 2)
        rec["spread"]         = rec["momentum_accel"]   # back-compat alias
        rec["vol_spike"]      = vol_r >= threshold
        rec = _attach_flow_metrics(rec, _LAST_OHLCV.get(ticker))
        rec["signal"]         = _signal_label(pd.Series(rec))
        sc = SIGNAL_COLORS.get(rec["signal"], ("#888780", "#1e2330"))
        rec["signal_fg"], rec["signal_bg"] = sc[0], sc[1]
        records.append(rec)

    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    # Rank on QUIET accumulation. Names with too little history to compute a
    # directional score are shown last rather than dropped, so their absence
    # is visible instead of silent.
    df = (df.sort_values("accumulation_score", ascending=False, na_position="last")
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
        rec["momentum_accel"] = round(rec["perf_1m"] - (rec["perf_3m"] / 3), 2)
        rec["spread"]         = rec["momentum_accel"]
        rec["vol_spike"]      = vol_r >= threshold
        rec = _attach_flow_metrics(rec, _LAST_OHLCV.get(ticker))
        records.append(rec)

    return pd.DataFrame(records) if records else pd.DataFrame()
