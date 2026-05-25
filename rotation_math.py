"""
rotation_math.py
────────────────
Implements the Relative Rotation Graph (RRG) mathematics.

Core concepts (Julius de Kempenaer methodology):
  RS-Ratio   = where a sector's relative strength IS vs the benchmark
  RS-Momentum = how fast RS-Ratio is CHANGING (acceleration)

Both are normalized to 100 as the neutral baseline.
Values above 100 → outperforming / accelerating
Values below 100 → underperforming / decelerating

The four quadrants:
  Leading   (ratio>100, mom>100) → strong AND accelerating
  Weakening (ratio>100, mom<100) → strong BUT decelerating
  Lagging   (ratio<100, mom<100) → weak AND decelerating
  Improving (ratio<100, mom>100) → weak BUT accelerating ← watch for accumulation

Signal ranking:
  Priority 1: Improving sectors with high spread → early accumulation
  Priority 2: Leading sectors holding momentum
  Priority 3: Weakening sectors (distribution alert)
  Priority 4: Lagging (avoid)
"""

import pandas as pd
import numpy as np


# ── Benchmark normalization ────────────────────────────────────────────────────
# We treat the equal-weighted average of all sectors as the "benchmark".
# This approximates SPX without needing a separate data fetch.
# A more precise version would subtract actual SPX performance.


def _benchmark(df: pd.DataFrame, col: str) -> float:
    """Equal-weighted average of all sectors for a given performance column."""
    return df[col].mean()


def _to_rs_scale(sector_perf: float, benchmark_perf: float) -> float:
    """
    Convert raw performance to RS scale centered at 100.
    RS = (1 + sector/100) / (1 + benchmark/100) * 100
    This gives a ratio where 100 = exactly market performance.
    """
    s = 1 + sector_perf / 100
    b = 1 + benchmark_perf / 100
    if b == 0:
        return 100.0
    return (s / b) * 100


def compute_rs_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RS-Ratio for 1W, 1M, and 3M timeframes.
    Also compute the primary rs_ratio used in the main quadrant classification
    (smoothed average of 1M and 3M, approximating the 10-week EMA used in RRG).

    Adds columns: rs_ratio_1w, rs_ratio_1m, rs_ratio_3m, rs_ratio
    """
    df = df.copy()

    for tf, col in [("1w", "perf_1w"), ("1m", "perf_1m"), ("3m", "perf_3m")]:
        bm = _benchmark(df, col)
        df[f"rs_ratio_{tf}"] = df[col].apply(lambda v: _to_rs_scale(v, bm))

    # Primary RS-Ratio: weight 1M more heavily (approximates medium-term trend)
    df["rs_ratio"] = df["rs_ratio_1m"] * 0.6 + df["rs_ratio_3m"] * 0.4

    return df


def compute_rs_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RS-Momentum = rate of change in RS-Ratio.
    Momentum = (short-term RS - long-term RS) normalized to 100 baseline.

    Specifically:
      RS-Mom = RS(1W) / RS(1M) * 100
      This measures whether recent relative performance is
      accelerating (+) or decelerating (-) vs the medium term.

    Also computes per-timeframe momentum columns for RRG toggle.

    Adds columns: rs_momentum_1w, rs_momentum_1m, rs_momentum_3m, rs_momentum
    """
    df = df.copy()

    # 1W timeframe: compare 1D vs 1W
    bm_1d = _benchmark(df, "perf_1d")
    bm_1w = _benchmark(df, "perf_1w")
    df["rs_1d"] = df["perf_1d"].apply(lambda v: _to_rs_scale(v, bm_1d))
    df["rs_momentum_1w"] = df.apply(
        lambda r: (r["rs_1d"] / r["rs_ratio_1w"] * 100) if r["rs_ratio_1w"] != 0 else 100,
        axis=1,
    )

    # 1M timeframe: compare 1W vs 1M
    df["rs_momentum_1m"] = df.apply(
        lambda r: (r["rs_ratio_1w"] / r["rs_ratio_1m"] * 100) if r["rs_ratio_1m"] != 0 else 100,
        axis=1,
    )

    # 3M timeframe: compare 1M vs 3M
    df["rs_momentum_3m"] = df.apply(
        lambda r: (r["rs_ratio_1m"] / r["rs_ratio_3m"] * 100) if r["rs_ratio_3m"] != 0 else 100,
        axis=1,
    )

    # Primary RS-Momentum: use 1M timeframe as default
    df["rs_momentum"] = df["rs_momentum_1m"]

    # Cleanup temp column
    df = df.drop(columns=["rs_1d"], errors="ignore")

    return df


def compute_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spread = 1M performance − (6M performance ÷ 6)

    This normalizes 6M to a monthly rate and compares it to the actual 1M.
    A positive spread means the sector is accelerating above its 6M run rate.
    This is the clearest free-data proxy for institutional accumulation.

    Adds column: spread
    """
    df = df.copy()
    df["spread"] = df["perf_1m"] - (df["perf_6m"] / 6)
    return df


def classify_quadrant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each sector to an RRG quadrant based on rs_ratio and rs_momentum.

    Quadrant rules (centered at 100):
      Leading:   ratio >= 100 AND momentum >= 100
      Weakening: ratio >= 100 AND momentum <  100
      Lagging:   ratio <  100 AND momentum <  100
      Improving: ratio <  100 AND momentum >= 100

    Adds column: quadrant
    """
    df = df.copy()

    def _quad(row):
        r = row["rs_ratio"]
        m = row["rs_momentum"]
        if r >= 100 and m >= 100:
            return "Leading"
        if r >= 100 and m < 100:
            return "Weakening"
        if r < 100 and m < 100:
            return "Lagging"
        return "Improving"

    df["quadrant"] = df.apply(_quad, axis=1)
    return df


def rank_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank sectors by institutional signal priority.

    Scoring logic:
      Base quadrant score:
        Improving  = 10  (early accumulation — highest alpha potential)
        Leading    = 8   (confirmed trend — ride the momentum)
        Weakening  = 5   (distribution alert — monitor for exit)
        Lagging    = 2   (avoid — lowest priority)

      Spread bonus: adds up to 5 points based on absolute spread magnitude
        (sectors with extreme spreads are more actionable)

      Momentum bonus: adds up to 3 points based on rs_momentum deviation from 100
        (stronger momentum = stronger signal)

    Lower signal_rank number = higher priority signal.

    Adds column: signal_rank, signal_score
    """
    df = df.copy()

    quad_base = {"Improving": 10, "Leading": 8, "Weakening": 5, "Lagging": 2}

    spread_max = df["spread"].abs().max() or 1
    mom_max = (df["rs_momentum"] - 100).abs().max() or 1

    def _score(row):
        base  = quad_base.get(row["quadrant"], 0)
        spread_bonus = (abs(row["spread"]) / spread_max) * 5
        mom_bonus    = (abs(row["rs_momentum"] - 100) / mom_max) * 3
        # For weakening/lagging, spread is negative bonus (penalize distribution)
        if row["quadrant"] in ("Weakening", "Lagging") and row["spread"] < 0:
            spread_bonus = -spread_bonus
        return base + spread_bonus + mom_bonus

    df["signal_score"] = df.apply(_score, axis=1)
    df["signal_rank"]  = df["signal_score"].rank(ascending=False).astype(int)

    return df


def get_rotation_summary(df: pd.DataFrame) -> dict:
    """
    Returns a summary dict of the current rotation state.
    Useful for generating the narrative interpretation.
    """
    quad_counts = df["quadrant"].value_counts().to_dict()
    top_improving = df[df["quadrant"] == "Improving"].sort_values("signal_score", ascending=False)
    top_leading   = df[df["quadrant"] == "Leading"].sort_values("signal_score", ascending=False)
    top_weakening = df[df["quadrant"] == "Weakening"].sort_values("signal_score", ascending=False)

    # Rotation bias: is money flowing in (improving+leading) or out (weakening+lagging)?
    inflow_sectors  = quad_counts.get("Improving", 0) + quad_counts.get("Leading", 0)
    outflow_sectors = quad_counts.get("Weakening", 0) + quad_counts.get("Lagging", 0)

    bias = "Bullish rotation" if inflow_sectors > outflow_sectors else \
           "Bearish rotation" if outflow_sectors > inflow_sectors else "Neutral rotation"

    return {
        "quadrant_counts": quad_counts,
        "bias": bias,
        "top_accumulation": top_improving.head(3)["sector"].tolist(),
        "top_momentum":     top_leading.head(3)["sector"].tolist(),
        "distribution_watch": top_weakening.head(3)["sector"].tolist(),
        "avg_rs_ratio":    df["rs_ratio"].mean(),
        "avg_rs_momentum": df["rs_momentum"].mean(),
    }