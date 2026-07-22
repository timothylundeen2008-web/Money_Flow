"""
etf_flow_tracker.py  (v1 — July 2026)
──────────────────────────────────────────────────────────────────────────────
ETF net flow from creations and redemptions.

EVIDENCE TIER: A (money).
  This is the only module in the repo that measures CAPITAL rather than price
  or pressure.

      net flow (t) = Δ shares outstanding (t) × NAV/close (t)

  Why this is categorically different from volume: secondary-market trading
  between two investors moves price and prints volume but creates no shares.
  Shares outstanding change ONLY when an Authorized Participant — a large
  institution — transacts directly with the issuer, in creation units of
  typically ≥25,000 shares. So a change in shares outstanding is, by
  construction, evidence of institutional-scale net demand rather than an
  inference from tape behavior.

  This is as close to a clean institutional flow signal as free public data
  allows, and nothing else in this dashboard is a substitute for it.

★ START POLLING IMMEDIATELY — HISTORY CANNOT BE BACKFILLED FOR FREE
  Free sources expose shares outstanding as a CURRENT SNAPSHOT, not a time
  series. Commercial feeds (ETF Global via Nasdaq Data Link, ETF.com) carry
  the computed daily history but are paid. The free path is to snapshot daily
  and accumulate your own series — which means every day this module is not
  running is a day of history permanently lost. Within a quarter this becomes
  the most valuable dataset in the stack.

INTERPRETIVE CAUTIONS (build these into any UI that shows this)
  - Flow is not conviction. Creations happen for hedging, model-portfolio
    rebalancing, and index-tracking mandates, not only directional views.
  - Shares outstanding are reported with a lag and are revised; treat a single
    day as noise and read the 5- and 20-day sums.
  - Share splits break the delta. handle_split() below detects and neutralizes
    the obvious cases, but verify any single-day flow larger than ~15% of AUM.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import streamlit as st
    _cache = st.cache_data(ttl=3600, show_spinner=False)
except Exception:
    def _cache(fn):
        return fn


DEFAULT_STORE = os.environ.get("ETF_FLOW_STORE", "data/etf_shares_history.csv")

TRACKED = [
    # All-Weather sleeves
    "VGT", "SMH", "QQQ", "GLD", "SLV", "RING", "XLE", "PDBC", "SCHD",
    "XLV", "XLU", "SGOV", "USFR", "TLT", "KMLM",
    # Sector universe (rotation dashboard)
    "XLK", "XLF", "XLI", "XLY", "XLRE", "XLB", "XLC", "XLP",
    # Key sub-sectors
    "KRE", "IBB", "XOP", "ITA", "PAVE", "IWM", "HYG", "EEM",
]

# A single-day flow above this share of AUM is implausible and almost always a
# split, a reporting error, or a stale snapshot — flagged, not silently used.
IMPLAUSIBLE_DAILY_FLOW_PCT = 0.15


# ── Snapshot acquisition ──────────────────────────────────────────────────────

def _snapshot_one(ticker: str) -> dict | None:
    """
    Current shares outstanding + price for one ETF.

    Tries fast_info first (cheap, stable), then .info. Returns None rather
    than a guess when shares outstanding is unavailable — a missing snapshot
    must create a gap, never a fabricated data point.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        t = yf.Ticker(ticker)
        shares = price = None

        try:
            fi = t.fast_info
            shares = getattr(fi, "shares", None) or fi.get("shares")           # type: ignore
            price = getattr(fi, "last_price", None) or fi.get("lastPrice")     # type: ignore
        except Exception:
            pass

        if not shares or not price:
            info = t.info or {}
            shares = shares or info.get("sharesOutstanding")
            price = price or info.get("navPrice") or info.get("previousClose")

        if not shares or not price:
            return None
        return {"date": datetime.now().date().isoformat(), "ticker": ticker,
                "shares_outstanding": float(shares), "price": float(price)}
    except Exception as e:
        print(f"[etf_flow] {ticker}: {type(e).__name__}: {e}")
        return None


def snapshot_all(tickers: list[str] | None = None,
                 store: str = DEFAULT_STORE) -> pd.DataFrame:
    """
    Take today's snapshot for every tracked ETF and APPEND to the local store.

    Idempotent per day: re-running overwrites today's rows rather than
    duplicating them, so a scheduled job that fires twice is harmless.
    Run this ONCE PER TRADING DAY, after the close.
    """
    tickers = tickers or TRACKED
    rows = [r for r in (_snapshot_one(tk) for tk in tickers) if r]
    if not rows:
        print("[etf_flow] no snapshots captured")
        return pd.DataFrame()

    new = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(store) or ".", exist_ok=True)

    if os.path.exists(store):
        hist = pd.read_csv(store)
        today = new["date"].iloc[0]
        hist = hist[~((hist["date"] == today) & (hist["ticker"].isin(new["ticker"])))]
        out = pd.concat([hist, new], ignore_index=True)
    else:
        out = new

    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    out.to_csv(store, index=False)
    print(f"[etf_flow] stored {len(new)} snapshots; history now {len(out)} rows")
    return new


# ── Flow computation ──────────────────────────────────────────────────────────

def load_history(store: str = DEFAULT_STORE) -> pd.DataFrame:
    if not os.path.exists(store):
        return pd.DataFrame(columns=["date", "ticker", "shares_outstanding", "price"])
    df = pd.read_csv(store)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def _flag_splits(g: pd.DataFrame) -> pd.Series:
    """
    Detect share-count jumps consistent with a split rather than a flow.

    A split multiplies shares by a clean ratio while price divides by the same
    ratio, so shares×price (AUM) is roughly unchanged. Genuine flow changes
    AUM. Comparing the two separates them.
    """
    sh_ratio = g["shares_outstanding"] / g["shares_outstanding"].shift(1)
    px_ratio = g["price"] / g["price"].shift(1)
    product = sh_ratio * px_ratio
    return (sh_ratio.sub(1).abs() > 0.20) & (product.sub(1).abs() < 0.05)


def compute_flows(store: str = DEFAULT_STORE) -> pd.DataFrame:
    """
    Daily net flow per ticker, plus 5- and 20-day rolling sums.

    Columns: date, ticker, shares_outstanding, price, aum,
             net_flow, net_flow_5d, net_flow_20d,
             flow_pct_aum_20d, is_split, implausible
    """
    hist = load_history(store)
    if hist.empty or len(hist) < 2:
        return pd.DataFrame()

    frames = []
    for tk, g in hist.groupby("ticker"):
        g = g.sort_values("date").copy()
        if len(g) < 2:
            continue
        g["aum"] = g["shares_outstanding"] * g["price"]
        g["is_split"] = _flag_splits(g).fillna(False)

        d_shares = g["shares_outstanding"].diff()
        g["net_flow"] = (d_shares * g["price"]).where(~g["is_split"], np.nan)

        g["implausible"] = (g["net_flow"].abs() / g["aum"]) > IMPLAUSIBLE_DAILY_FLOW_PCT
        g.loc[g["implausible"], "net_flow"] = np.nan   # excluded, not silently used

        g["net_flow_5d"] = g["net_flow"].rolling(5, min_periods=2).sum()
        g["net_flow_20d"] = g["net_flow"].rolling(20, min_periods=5).sum()
        g["flow_pct_aum_20d"] = (g["net_flow_20d"] / g["aum"]) * 100
        frames.append(g)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def flow_vs_price_divergence(store: str = DEFAULT_STORE,
                             window: int = 20) -> pd.DataFrame:
    """
    ★ THE HIGHEST-VALUE OUTPUT IN THIS MODULE ★

    Compares price direction against MONEY direction over the same window and
    classifies the four combinations:

      CONFIRMED UPTREND   price ↑  flow ↑   real capital behind the move
      DISTRIBUTION        price ↑  flow ↓   rising on redemptions — money is
                                            LEAVING into strength. This is the
                                            pattern no price-only or volume-only
                                            signal in this dashboard can see,
                                            and the reason this module exists.
      ACCUMULATION        price ↓  flow ↑   money entering weakness — the
                                            classic institutional entry
      CONFIRMED DOWNTREND price ↓  flow ↓   capitulation, no support

    Returns one row per ticker with enough history.
    """
    fl = compute_flows(store)
    if fl.empty:
        return pd.DataFrame()

    rows = []
    for tk, g in fl.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < window + 1:
            continue
        px_chg = float(g["price"].iloc[-1] / g["price"].iloc[-window - 1] - 1) * 100
        flow = float(g["net_flow"].iloc[-window:].sum(skipna=True))
        aum = float(g["aum"].iloc[-1])
        flow_pct = (flow / aum * 100) if aum else np.nan

        if px_chg > 0 and flow > 0:
            verdict = "CONFIRMED UPTREND"
        elif px_chg > 0 and flow < 0:
            verdict = "DISTRIBUTION (price up, money out)"
        elif px_chg < 0 and flow > 0:
            verdict = "ACCUMULATION (price down, money in)"
        else:
            verdict = "CONFIRMED DOWNTREND"

        rows.append({"ticker": tk, "days": len(g),
                     "price_chg_pct": round(px_chg, 2),
                     "net_flow_usd": round(flow, 0),
                     "net_flow_pct_aum": round(flow_pct, 2) if flow_pct == flow_pct else np.nan,
                     "verdict": verdict,
                     "divergence": verdict.startswith(("DISTRIBUTION", "ACCUMULATION"))})

    df = pd.DataFrame(rows)
    return df.sort_values("net_flow_pct_aum", ascending=False) if not df.empty else df


def coverage_report(store: str = DEFAULT_STORE) -> dict:
    """
    How much usable history exists yet. Call this before trusting any output —
    the module is honest about being useless on day one and improving daily.
    """
    hist = load_history(store)
    if hist.empty:
        return {"tickers": 0, "days": 0, "ready": False,
                "message": "No history yet. Run snapshot_all() once per trading "
                           "day. Flow readings need ~20 sessions to be useful; "
                           "divergence detection needs ~40."}
    per = hist.groupby("ticker")["date"].count()
    days = int(per.max())
    return {"tickers": int(hist["ticker"].nunique()), "days": days,
            "median_days": int(per.median()),
            "first_date": str(hist["date"].min().date()),
            "ready": days >= 20,
            "message": ("Sufficient history for 20-day flow readings."
                        if days >= 20 else
                        f"Only {days} sessions stored — need ~20. Keep polling daily.")}
