"""
rotation_math.py  (v3.2 — November 2026)
──────────────────────────────────────────────────────────────────
v3.2 (2026-11-01, fix_status N32): `stealth_signal` — shown on the dashboard
  and worth 0-2 points in signal_score — now comes from flow_metrics'
  stealth_label (Tier B: CMF pressure, persistence and A/D absorption, gated
  on QUIET volume). The old rule here used four price conditions and no
  volume at all; flow_metrics' own docstring records that it "fired on any
  ordinary uptrend". With no stealth_label available the flag is "None" —
  never the price-only fallback. The brief and signal_outcomes already used
  stealth_label, so the dashboard now agrees with them.

v3.1 (2026-11-01, fix_status N17): the quadrant vote is RRG BREADTH, not
  money. get_rotation_summary() used to call more Improving+Leading than
  Weakening+Lagging sectors "Bullish rotation" (and the app labelled the
  counts "inflow / outflow"). Quadrants are relative price strength (Tier C);
  creations and redemptions (Tier A) live in flow_map. rrg_breadth() states
  the count plainly; the "bias" key now carries that wording.

(v3 — unified spread formula + rotation history)
RRG Mathematics using proper timeframe windows.

CHANGE LOG v2 → v3:
  - Spread formula unified to 1M vs 3M÷3 (was 1M vs 6M÷6 — too slow)
  - Added compute_rotation_history() for directional arrow data
  - Added stealth_accumulation() rolling 5-day signal
  - compute_spread now consistent with top_movers.py

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
    RS = (1 + sector%) / (1 + benchmark%) x 100
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

    Toggle mapping (ratio column → momentum column):
      1W toggle: rs_ratio_1w = rs_1m,  rs_momentum_1w = rs_1w / rs_1m
      1M toggle: rs_ratio_1m = rs_3m,  rs_momentum_1m = rs_1m / rs_3m
      3M toggle: rs_ratio_3m = rs_1y,  rs_momentum_3m = rs_6m / rs_1y  (default)
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

    # Per-toggle RS-Ratio columns
    df["rs_ratio_1w"] = df["rs_1m"]
    df["rs_ratio_1m"] = df["rs_3m"]
    df["rs_ratio_3m"] = df["rs_1y"]

    # Primary rs_ratio = 3M toggle (default view)
    df["rs_ratio"] = df["rs_ratio_3m"]

    return df


# ── Step 2: Compute RS-Momentum per toggle ────────────────────────────────────

def compute_rs_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    RS-Momentum = rate of change of RS-Ratio.
    Computed as shorter-window RS / longer-window RS x 100.

    >100 = relative strength is accelerating (improving)
    <100 = relative strength is decelerating (weakening)
    """
    df = df.copy()

    safe_div = lambda a, b: (a / b * 100) if b != 0 else 100.0

    df["rs_momentum_1w"] = df.apply(
        lambda r: safe_div(r["rs_1w"], r["rs_1m"]), axis=1
    )
    df["rs_momentum_1m"] = df.apply(
        lambda r: safe_div(r["rs_1m"], r["rs_3m"]), axis=1
    )
    df["rs_momentum_3m"] = df.apply(
        lambda r: safe_div(r["rs_6m"], r["rs_1y"]), axis=1
    )

    # Primary rs_momentum = 3M toggle (default view)
    df["rs_momentum"] = df["rs_momentum_3m"]

    return df


# ── Step 3: Accumulation/distribution spread ──────────────────────────────────

def compute_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spread = 1M performance minus (3M performance / 3)

    CHANGED v2->v3: was (6M / 6), now (3M / 3).
    Reason: 3-month run-rate is more responsive to institutional rotation
    cycles (typically 1-3 months). The 6M baseline included stale history
    that masked current accumulation/distribution signals.

    Consistent with top_movers.py spread formula.

    Positive spread: recent month running HOT vs 3M trend = accumulation.
    Negative spread: recent month running COLD vs 3M trend = distribution.
    """
    df = df.copy()
    df["spread"] = df["perf_1m"] - (df["perf_3m"] / 3)
    return df


# ── Step 3b: Stealth accumulation (rolling multi-day signal) ──────────────────

STEALTH_LABELS = ("Stealth", "Strong Stealth")


def compute_stealth_accumulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stealth accumulation flag for the dashboard and signal_score, taken from
    flow_metrics.stealth_accumulation (stealth_label, computed per sector by
    data_fetcher from OHLCV): buying pressure (CMF > +0.05), sustained (CMF
    positive on >= 60% of the window) and absorbed (A/D rising faster than
    price), and only on QUIET volume — a volume spike disqualifies it.

    v3.2: the previous rule here was price-only (1W > 0, spread > 0, 1D > 0,
    1M > 1W) despite promising volume, and fired on most sectors in any
    uptrend. It is gone; there is deliberately no price-only fallback.
    """
    df = df.copy()
    if "stealth_label" in df.columns:
        lab = df["stealth_label"].astype(str)
        df["stealth_signal"] = lab.where(lab.isin(STEALTH_LABELS), "None")
    else:
        df["stealth_signal"] = "None"
    return df


# ── Step 3c: Rotation history (for directional arrows) ────────────────────────

def compute_rotation_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a synthetic 'previous position' for each sector to draw
    directional arrows on the RRG showing which way each dot is moving.

    Since we don't have actual prior-week data in a single fetch, we
    approximate the previous RS position using shorter timeframes:
      prev_rs_ratio    = rs_3m  (3M RS as a proxy for where we WERE)
      prev_rs_momentum = rs_3m / rs_6m * 100

    The arrow goes from (prev_ratio, prev_mom) to (current ratio, current mom).
    This gives a directional sense even without historical snapshots.
    """
    df = df.copy()

    # Previous-period approximation using medium-term RS
    safe_div = lambda a, b: (a / b * 100) if b != 0 else 100.0

    df["prev_rs_ratio"]    = df["rs_3m"]
    df["prev_rs_momentum"] = df.apply(
        lambda r: safe_div(r["rs_3m"], r.get("rs_6m", r["rs_3m"])), axis=1
    )

    # Delta vectors for arrow rendering
    df["delta_ratio"]    = df["rs_ratio"]    - df["prev_rs_ratio"]
    df["delta_momentum"] = df["rs_momentum"] - df["prev_rs_momentum"]

    # Rotation direction label
    def _direction(row):
        dr = row["delta_ratio"]
        dm = row["delta_momentum"]
        if dr > 0 and dm > 0: return "Strengthening"
        if dr > 0 and dm < 0: return "Topping"
        if dr < 0 and dm < 0: return "Weakening"
        if dr < 0 and dm > 0: return "Bottoming"
        return "Stable"

    df["rotation_direction"] = df.apply(_direction, axis=1)
    return df


# ── Step 4: Quadrant classification ───────────────────────────────────────────

def classify_quadrant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign RRG quadrant based on primary rs_ratio and rs_momentum.

    Leading:   ratio >= 100, momentum >= 100  -- outperforming & accelerating
    Weakening: ratio >= 100, momentum < 100   -- outperforming but decelerating
    Lagging:   ratio < 100,  momentum < 100   -- underperforming & decelerating
    Improving: ratio < 100,  momentum >= 100  -- underperforming but accelerating
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
      Improving  = base 10  (early accumulation -- highest alpha potential)
      Leading    = base 8   (confirmed momentum -- ride it)
      Weakening  = base 5   (distribution alert -- watch for exit)
      Lagging    = base 2   (avoid)

    + spread bonus (0-5 pts): magnitude of accumulation/distribution spread
    + momentum bonus (0-3 pts): how far momentum deviates from 100
    + stealth bonus (0-2 pts): stealth accumulation signal

    Lower signal_rank = higher priority.
    """
    df = df.copy()

    quad_base  = {"Improving": 10, "Leading": 8, "Weakening": 5, "Lagging": 2}
    spread_max = df["spread"].abs().max() or 1
    mom_max    = (df["rs_momentum"] - 100).abs().max() or 1

    def _score(row):
        base         = quad_base.get(row["quadrant"], 0)
        spread_bonus = (abs(row["spread"]) / spread_max) * 5
        mom_bonus    = (abs(row["rs_momentum"] - 100) / mom_max) * 3
        if row["quadrant"] in ("Weakening", "Lagging") and row["spread"] < 0:
            spread_bonus = -spread_bonus
        stealth_bonus = 2.0 if row.get("stealth_signal") == "Strong Stealth" else \
                        1.0 if row.get("stealth_signal") == "Stealth" else 0.0
        return base + spread_bonus + mom_bonus + stealth_bonus

    df["signal_score"] = df.apply(_score, axis=1)
    df["signal_rank"]  = df["signal_score"].rank(ascending=False).astype(int)
    return df


# ── Summary helper ─────────────────────────────────────────────────────────────

def rrg_breadth(quad_counts: dict) -> dict:
    """How many sectors sit in Improving or Leading, out of all classified.
    Relative strength vs the equal-weighted sector average — Tier C, never
    money: every sector can fall and some will still be 'Leading'."""
    up = int(quad_counts.get("Improving", 0)) + int(quad_counts.get("Leading", 0))
    total = int(sum(int(v) for k, v in quad_counts.items()
                    if k in ("Improving", "Leading", "Weakening", "Lagging")))
    return {"up": up, "total": total, "value": f"{up} of {total}" if total else "n/a",
            "label": f"RRG breadth: {up} of {total} sectors improving or leading" if total else "RRG breadth: n/a",
            "note": "relative strength vs the sector average — not money flow"}


def get_rotation_summary(df: pd.DataFrame) -> dict:
    quad_counts = df["quadrant"].value_counts().to_dict()
    br = rrg_breadth(quad_counts)
    return {
        "quadrant_counts":    quad_counts,
        "bias":               br["label"],        # v3.1: was "Bullish/Bearish rotation" (N17)
        "rrg_breadth":        br,
        "top_accumulation":   df[df["quadrant"] == "Improving"].nlargest(3, "signal_score")["sector"].tolist(),
        "top_momentum":       df[df["quadrant"] == "Leading"].nlargest(3, "signal_score")["sector"].tolist(),
        "distribution_watch": df[df["quadrant"] == "Weakening"].nlargest(3, "signal_score")["sector"].tolist(),
        "avg_rs_ratio":       df["rs_ratio"].mean(),
        "avg_rs_momentum":    df["rs_momentum"].mean(),
    }


# ── Step 6: The full pipeline — ONE definition, every caller ─────────────────

PIPELINE = (
    compute_rs_ratio,
    compute_rs_momentum,
    compute_spread,
    compute_stealth_accumulation,
    compute_rotation_history,
    classify_quadrant,
    rank_signals,
)


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw fetch_sector_data() output -> fully scored RRG table.

    This is the same seven steps app.py always ran before rendering. It
    lives here so that publish_to_checklist.py (the scheduled path that
    feeds rotation_bridge -> Portfolio-Tracker -> Swing Desk) runs the
    IDENTICAL pipeline. Before this function existed the publisher shipped
    the raw fetch with no `quadrant` column, so the Swing Desk's rotation
    confluence leg could never pass from scheduled data. One pipeline, two
    callers -- two copies would drift.
    """
    if df is None or df.empty:
        return df
    for step in PIPELINE:
        df = step(df)
    return df
