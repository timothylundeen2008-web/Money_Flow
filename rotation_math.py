"""
rotation_math.py  (v2 – corrected window mapping)
──────────────────────────────────────────────────
RRG Mathematics using proper timeframe windows.

Key insight: the RRG toggle controls WHICH timeframe pair is used.
Wider windows (3M toggle) give more stable, meaningful quadrant distribution.
Shorter windows (1W toggle) show short-term momentum shifts.

Toggle → RS-Ratio window → RS-Momentum window
  1W   →   1M RS          →  1W RS  / 1M RS   (short-term pulse)
  1M   →   3M RS          →  1M RS  / 3M RS   (medium-term trend)
  3M   →   1Y RS          →  6M RS  / 1Y RS   (long-term cycle) ← DEFAULT

All RS values are relative to the equal-weighted sector average (proxy for SPX).
"""

import pandas as pd
import numpy as np


# ── Core helpers ───────────────────────────────────────────────────────────────

def _benchmark(df: pd.DataFrame, col: str) -> float:
    """Equal-weighted mean of all sectors for a given column."""
    return df[col].mean()


def _to_rs(sector_perf: float, benchmark_perf: float) -> float:
    """
    Relative Strength ratio, normalized to 100.
    RS = (1 + sector%) / (1 + benchmark%) × 100
    100 = exactly market performance; >100 = outperforming.
    """
    s = 1 + sector_perf / 100
    b = 1 + benchmark_perf / 100
    return (s / b) * 100 if b != 0 else 100.0


# ── Step 1: Compute raw RS values at each timeframe ───────────────────────────

def compute_rs_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RS values at 1W, 1M, 3M, 6M, and 1Y timeframes.
    These are the building blocks for all ratio/momentum combinations.

    Also sets the primary rs_ratio and per-toggle columns used by the UI.

    Toggle mapping (ratio column → momentum column):
      1W toggle: rs_ratio_1w = rs_1m,  rs_momentum_1w = rs_1w / rs_1m
      1M toggle: rs_ratio_1m = rs_3m,  rs_momentum_1m = rs_1m / rs_3m
      3M toggle: rs_ratio_3m = rs_1y,  rs_momentum_3m = rs_6m / rs_1y  ← default
    """
    df = df.copy()

    # Raw RS at each timeframe
    for tf, col in [
        ("1w", "perf_1w"),
        ("1m", "perf_1m"),
        ("3m", "perf_3m"),
        ("6m", "perf_6m"),
        ("1y", "perf_1y"),
    ]:
        bm = _benchmark(df, col)
        df[f"rs_{tf}"] = df[col].apply(lambda v: _to_rs(v, bm))

    # Per-toggle RS-Ratio columns (what the x-axis shows on each toggle)
    df["rs_ratio_1w"] = df["rs_1m"]   # 1W toggle x-axis = 1M RS
    df["rs_ratio_1m"] = df["rs_3m"]   # 1M toggle x-axis = 3M RS
    df["rs_ratio_3m"] = df["rs_1y"]   # 3M toggle x-axis = 1Y RS

    # Primary rs_ratio = 3M toggle (default view)
    df["rs_ratio"] = df["rs_ratio_3m"]

    return df


# ── Step 2: Compute RS-Momentum per toggle ────────────────────────────────────

def compute_rs_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    RS-Momentum = rate of change of RS-Ratio.
    Computed as shorter-window RS / longer-window RS × 100.

    >100 = relative strength is accelerating (improving)
    <100 = relative strength is decelerating (weakening)
    """
    df = df.copy()

    safe_div = lambda a, b: (a / b * 100) if b != 0 else 100.0

    # 1W toggle momentum: is this week's RS stronger than the 1M RS?
    df["rs_momentum_1w"] = df.apply(
        lambda r: safe_div(r["rs_1w"], r["rs_1m"]), axis=1
    )

    # 1M toggle momentum: is this month's RS stronger than the 3M RS?
    df["rs_momentum_1m"] = df.apply(
        lambda r: safe_div(r["rs_1m"], r["rs_3m"]), axis=1
    )

    # 3M toggle momentum: is this 6M RS stronger than the 1Y RS?
    df["rs_momentum_3m"] = df.apply(
        lambda r: safe_div(r["rs_6m"], r["rs_1y"]), axis=1
    )

    # Primary rs_momentum = 3M toggle (default view)
    df["rs_momentum"] = df["rs_momentum_3m"]

    return df


# ── Step 3: Accumulation/distribution spread ──────────────────────────────────

def compute_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spread = 1M performance − (6M performance ÷ 6)

    Normalizes 6M to a monthly run-rate and compares to actual 1M.
    Positive spread → recent month is running HOT vs trend = accumulation signal.
    Negative spread → recent month is running COLD vs trend = distribution signal.
    """
    df = df.copy()
    df["spread"] = df["perf_1m"] - (df["perf_6m"] / 6)
    return df


# ── Step 4: Quadrant classification ───────────────────────────────────────────

def classify_quadrant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign RRG quadrant based on primary rs_ratio and rs_momentum.

    Leading:   ratio ≥ 100, momentum ≥ 100  → outperforming & accelerating
    Weakening: ratio ≥ 100, momentum < 100  → outperforming but decelerating
    Lagging:   ratio < 100,  momentum < 100  → underperforming & decelerating
    Improving: ratio < 100,  momentum ≥ 100  → underperforming but accelerating
    """
    df = df.copy()

    def _quad(row):
        r = row["rs_ratio"]
        m = row["rs_momentum"]
        if r >= 100 and m >= 100: return "Leading"
        if r >= 100 and m <  100: return "Weakening"
        if r <  100 and m <  100: return "Lagging"
        return "Improving"

    df["quadrant"] = df.apply(_quad, axis=1)
    return df


# ── Step 5: Signal ranking ────────────────────────────────────────────────────

def rank_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank sectors by institutional signal priority.

    Scoring:
      Improving  = base 10  (early accumulation — highest alpha potential)
      Leading    = base 8   (confirmed momentum — ride it)
      Weakening  = base 5   (distribution alert — watch for exit)
      Lagging    = base 2   (avoid)

    + spread bonus (0–5 pts): magnitude of accumulation/distribution spread
    + momentum bonus (0–3 pts): how far momentum deviates from 100

    Lower signal_rank = higher priority.
    """
    df = df.copy()

    quad_base = {"Improving": 10, "Leading": 8, "Weakening": 5, "Lagging": 2}
    spread_max = df["spread"].abs().max() or 1
    mom_max    = (df["rs_momentum"] - 100).abs().max() or 1

    def _score(row):
        base         = quad_base.get(row["quadrant"], 0)
        spread_bonus = (abs(row["spread"]) / spread_max) * 5
        mom_bonus    = (abs(row["rs_momentum"] - 100) / mom_max) * 3
        if row["quadrant"] in ("Weakening", "Lagging") and row["spread"] < 0:
            spread_bonus = -spread_bonus
        return base + spread_bonus + mom_bonus

    df["signal_score"] = df.apply(_score, axis=1)
    df["signal_rank"]  = df["signal_score"].rank(ascending=False).astype(int)
    return df


# ── Summary helper ─────────────────────────────────────────────────────────────

def get_rotation_summary(df: pd.DataFrame) -> dict:
    quad_counts  = df["quadrant"].value_counts().to_dict()
    inflow  = quad_counts.get("Improving", 0) + quad_counts.get("Leading", 0)
    outflow = quad_counts.get("Weakening", 0) + quad_counts.get("Lagging", 0)
    bias = ("Bullish rotation" if inflow > outflow else
            "Bearish rotation" if outflow > inflow else "Neutral rotation")
    return {
        "quadrant_counts":    quad_counts,
        "bias":               bias,
        "top_accumulation":   df[df["quadrant"] == "Improving"].nlargest(3, "signal_score")["sector"].tolist(),
        "top_momentum":       df[df["quadrant"] == "Leading"].nlargest(3, "signal_score")["sector"].tolist(),
        "distribution_watch": df[df["quadrant"] == "Weakening"].nlargest(3, "signal_score")["sector"].tolist(),
        "avg_rs_ratio":       df["rs_ratio"].mean(),
        "avg_rs_momentum":    df["rs_momentum"].mean(),
    }