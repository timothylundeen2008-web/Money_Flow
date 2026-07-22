"""
flow_metrics.py  (v1 — July 2026)
──────────────────────────────────────────────────────────────────────────────
Directional volume analysis. This module exists because every pre-v1 signal in
the Rotation Dashboard was computed from close prices alone (or from close plus
an undirected share-volume ratio), which cannot distinguish buying pressure from
selling pressure.

THE CORE IDEA
  Share volume tells you HOW MUCH traded. It never tells you WHO INITIATED or on
  WHICH SIDE. The money-flow multiplier fixes this by weighting each session's
  volume by where the close landed inside that session's range:

      MFM = ((Close - Low) - (High - Close)) / (High - Low)      ∈ [-1, +1]

  Closing on the high with heavy volume = genuine accumulation (+1 × volume).
  Closing on the low with heavy volume  = genuine distribution  (-1 × volume).
  Closing mid-range                     = volume with no directional information.

  High and Low were ALREADY being downloaded by both fetchers and discarded.
  This module is the reason to stop discarding them.

EVIDENCE TIER: B (directional volume / pressure).
  Tier B measures buying and selling PRESSURE. It does not measure MONEY —
  that requires creations/redemptions (etf_flow_tracker.py) or reported
  positioning (cot_fetcher.py), which are Tier A. Do not describe a Tier-B
  signal as "institutional accumulation" without a Tier-A confirmation.

DESIGN RULES FOLLOWED HERE
  - Insufficient history returns NaN. Never silently substitutes a shorter
    window (the failure mode behind both the CPI bug and the perf_1y bug).
  - Every public function documents its minimum bar count.
  - No I/O, no network, no Streamlit. Pure functions, unit-testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Minimum bar requirements ──────────────────────────────────────────────────
MIN_BARS_CMF = 21
MIN_BARS_MFI = 15
MIN_BARS_STEALTH = 63


# ─────────────────────────────────────────────────────────────────────────────
#  PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def money_flow_multiplier(high: pd.Series, low: pd.Series,
                          close: pd.Series) -> pd.Series:
    """
    Per-session directional weight in [-1, +1].

    Sessions where High == Low (limit moves, halts, bad ticks) carry no
    positional information and are assigned 0.0 rather than NaN or inf —
    they contribute zero directional volume instead of poisoning the series.
    """
    rng = (high - low)
    mfm = ((close - low) - (high - close)) / rng.replace(0, np.nan)
    return mfm.fillna(0.0)


def money_flow_volume(high: pd.Series, low: pd.Series, close: pd.Series,
                      volume: pd.Series) -> pd.Series:
    """Per-session signed volume: the directional building block."""
    return money_flow_multiplier(high, low, close) * volume


def ad_line(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series) -> pd.Series:
    """
    Accumulation/Distribution Line — cumulative signed volume.

    This is the REAL A/D line. The Rotation Dashboard's pre-v1 "A/D spread"
    (perf_1m - perf_3m/3) contained no volume whatsoever and was a momentum
    residual; the weekly checklist's 3-of-3 confluence test has always
    required an A/D confirmation leg that did not exist in code until now.

    Level matters less than SLOPE and DIVERGENCE vs price.
    """
    return money_flow_volume(high, low, close, volume).cumsum()


def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series,
                       volume: pd.Series, period: int = MIN_BARS_CMF) -> pd.Series:
    """
    CMF = sum(money-flow volume, period) / sum(volume, period)   ∈ [-1, +1]

    Normalized, so comparable across tickers of different liquidity — unlike
    raw A/D level or raw share volume. Default 21 sessions ≈ one trading month.

    Interpretation used throughout this codebase:
        > +0.10  strong buying pressure
        > +0.05  buying pressure
        -0.05..+0.05  neutral / no edge
        < -0.05  selling pressure
        < -0.10  strong selling pressure
    """
    if len(close.dropna()) < period:
        return pd.Series(np.nan, index=close.index)
    mfv = money_flow_volume(high, low, close, volume)
    vol_sum = volume.rolling(period).sum()
    return (mfv.rolling(period).sum() / vol_sum.replace(0, np.nan))


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV — cumulative volume signed by close-to-close direction.

    Cruder than the A/D line (whole-session granularity, ignores intraday
    position) but robust when High/Low are unreliable, and a useful second
    opinion: A/D and OBV disagreeing is itself a caution flag.
    """
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, period: int = 14) -> pd.Series:
    """
    MFI — volume-weighted RSI on typical price, bounded 0–100.
    Above 80 = overbought on volume; below 20 = oversold. Useful as an
    exhaustion filter on entries, NOT as a flow measure in its own right.
    """
    if len(close.dropna()) < period + 1:
        return pd.Series(np.nan, index=close.index)
    typical = (high + low + close) / 3.0
    raw = typical * volume
    delta = typical.diff()
    pos = raw.where(delta > 0, 0.0).rolling(period).sum()
    neg = raw.where(delta < 0, 0.0).rolling(period).sum()
    ratio = pos / neg.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


# ─────────────────────────────────────────────────────────────────────────────
#  DERIVED DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

_TREND_CAP = 5.0        # ±5 daily-noise-units per bar is already an extreme trend
DIVERGENCE_SCALE = 2.0  # calibration constant for scoring; see accumulation_score


def _trend_strength(s: pd.Series, window: int) -> float:
    """
    Least-squares slope over `window` bars, normalized by the series' OWN
    daily-change volatility. Result is dimensionless: "bars of trend per unit
    of daily noise", directly comparable across series with incompatible units
    (the A/D line is in signed shares; price is in dollars).

    Why not normalize by level: the A/D line is cumulative, so its level
    depends on an arbitrary series start and drifts without bound — dividing
    by it produced unstable results and returned NaN for a perfectly neutral
    (all-zero) A/D line, which is a real market state, not missing data.

    Degenerate cases handled explicitly:
      flat series (no slope, no noise)   -> 0.0   (neutral, NOT NaN)
      perfectly linear (slope, no noise) -> ±cap  (maximal trend)
      insufficient history               -> NaN   (never a truncated window)
    """
    s = s.dropna()
    if len(s) < window:
        return float("nan")
    y = s.iloc[-window:].to_numpy(dtype=float)
    x = np.arange(window, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    noise = float(np.std(np.diff(y)))
    if noise == 0.0:
        # A constant series yields a ~1e-15 polyfit slope from float error, so
        # test against a scale-aware tolerance rather than exact zero —
        # otherwise a perfectly flat series reports a MAXIMAL trend.
        tol = 1e-9 * max(1.0, float(np.abs(y).mean()))
        if abs(slope) <= tol:
            return 0.0
        return float(np.sign(slope) * _TREND_CAP)
    return float(np.clip(slope / noise, -_TREND_CAP, _TREND_CAP))


def ad_price_divergence(high, low, close, volume, window: int = 21) -> float:
    """
    The single most valuable number in this module.

    Returns (normalized A/D slope) - (normalized price slope) over `window`.

    Both trends are expressed in daily-noise units (see _trend_strength), so
    the difference is meaningful across tickers.

        > 0  A/D trending up harder than price  → absorption / accumulation
        ≈ 0  volume confirms price              → ordinary trend
        < 0  price trending up harder than A/D  → distribution into strength,
                                                  or a low-conviction drift

    A sector making new highs on a FALLING A/D line is the classic
    distribution-into-strength pattern, and it is invisible to any
    price-only or undirected-volume signal.
    """
    ad = ad_line(high, low, close, volume)
    return _trend_strength(ad, window) - _trend_strength(close, window)


def cmf_persistence(cmf: pd.Series, window: int = 21) -> float:
    """
    Fraction of the last `window` sessions with positive CMF, in [0, 1].

    Persistence separates real absorption from a single strong session:
    institutions build across many sessions, so a 0.75 persistence with a
    modest CMF beats a 0.35 persistence with one spike.
    """
    c = cmf.dropna()
    if len(c) < window:
        return float("nan")
    return float((c.iloc[-window:] > 0).mean())


# ─────────────────────────────────────────────────────────────────────────────
#  STEALTH ACCUMULATION  (replaces the pre-v1 price-only signal)
# ─────────────────────────────────────────────────────────────────────────────

def stealth_accumulation(high, low, close, volume,
                         vol_ratio: float,
                         tier_threshold: float,
                         window: int = 21) -> dict:
    """
    Detect QUIET institutional absorption.

    WHY THIS WAS REWRITTEN
      The previous compute_stealth_accumulation() docstring promised
      "sustained above-average volume + positive price action" but its four
      implemented conditions were ALL price-only (perf_1w>0, spread>0,
      perf_1d>0, perf_1m>perf_1w). It fired on any ordinary uptrend and
      contained no volume term at all.

    WHAT STEALTH ACTUALLY LOOKS LIKE
      Someone absorbing supply without moving the tape leaves a specific
      fingerprint: directional volume accumulates while price stays quiet,
      and it does so WITHOUT a headline volume spike (a spike means the
      activity was loud, which is the opposite of stealth — and, per this
      framework's own thesis, more often retail or event-driven).

    QUIETNESS IS A PRECONDITION, NOT A POINT. If volume is at or above the
    tier spike threshold the activity was loud, which disqualifies it as
    stealth by definition regardless of how strong the pressure is — that
    name belongs on event_score() instead. (Scoring quietness as one of four
    additive points let a loud, heavy-volume buying event still be labeled
    "Stealth" at 3/4, which is exactly the mislabeling this rewrite exists
    to eliminate.)

    THREE SCORED CONDITIONS (all Tier B, all volume-aware), gated on quiet:
      1. Buying pressure present      — CMF > +0.05
      2. Sustained, not a one-day pop — CMF positive ≥60% of the window
      3. Absorption vs price          — A/D trending up harder than price

    Returns dict with label, score 0-3, and the component values so the UI
    can show WHY it fired rather than just that it did.
    """
    out = {"stealth_label": "None", "stealth_score": 0,
           "cmf": float("nan"), "cmf_persist": float("nan"),
           "ad_divergence": float("nan"), "is_quiet": None}

    if len(close.dropna()) < MIN_BARS_STEALTH:
        out["stealth_label"] = "Insufficient history"
        return out

    cmf = chaikin_money_flow(high, low, close, volume, period=window)
    cmf_now = float(cmf.dropna().iloc[-1]) if cmf.notna().any() else float("nan")
    persist = cmf_persistence(cmf, window)
    diverg = ad_price_divergence(high, low, close, volume, window)
    quiet = bool(vol_ratio < tier_threshold)

    out.update(cmf=cmf_now, cmf_persist=persist,
               ad_divergence=diverg, is_quiet=quiet)

    if any(pd.isna(v) for v in (cmf_now, persist, diverg)):
        out["stealth_label"] = "Insufficient history"
        return out

    if not quiet:
        out["stealth_score"] = 0
        out["stealth_label"] = "Loud — not stealth (see event score)"
        return out

    score = sum([cmf_now > 0.05, persist >= 0.60, diverg > 0.0])
    out["stealth_score"] = int(score)
    out["stealth_label"] = {3: "Strong Stealth", 2: "Stealth",
                            1: "Weak / Mixed"}.get(score, "None")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  SCORING  (split, deliberately — see rationale)
# ─────────────────────────────────────────────────────────────────────────────

def accumulation_score(high, low, close, volume,
                       vol_ratio: float, tier_threshold: float,
                       window: int = 21) -> float:
    """
    QUIET accumulation, 0-100. Rewards sustained directional buying pressure
    that has NOT announced itself.

    Components (independent — no term is a transformation of another):
      CMF level        ±35   directional pressure now
      Persistence      ±25   sustained, not a one-session artifact
      A/D divergence   ±25   absorption relative to price
      Quietness        ±15   below tier threshold credits; above it PENALIZES

    Range is roughly -100..+100. Quietness is signed rather than 0-15 so that
    loud names are actively pushed DOWN the quiet-accumulation ranking instead
    of merely failing to gain — they remain fully visible on event_score().

    WHY THIS REPLACES THE SINGLE flow_score
      The pre-v1 _flow_score awarded a FLAT +8 for crossing the volume-spike
      threshold, while its entire momentum term spanned roughly ±2. A spike
      therefore outweighed any realistic momentum difference ~4x, so mediocre
      names with one loud session outranked strong quiet accumulators — the
      exact inversion of this framework's stated thesis that loud volume is
      retail. Three of its four terms were also transformations of the same
      three price numbers, making a two-factor blend look like a four-factor one.

      Loudness has NOT been deleted — it moved to event_score(), which is
      reported ALONGSIDE this, never summed into it. They measure opposite
      phenomena and must not be added together.
    """
    if len(close.dropna()) < MIN_BARS_STEALTH:
        return float("nan")

    cmf = chaikin_money_flow(high, low, close, volume, period=window)
    if not cmf.notna().any():
        return float("nan")
    cmf_now = float(cmf.dropna().iloc[-1])
    persist = cmf_persistence(cmf, window)
    diverg = ad_price_divergence(high, low, close, volume, window)
    if any(pd.isna(v) for v in (persist, diverg)):
        return float("nan")

    # CMF in [-1,1] but realistically ±0.25; clip and scale to 0-35
    cmf_pts = float(np.clip(cmf_now / 0.20, -1, 1) * 35)
    # Persistence 0-1 → 0-25, centered so 0.5 (coin flip) scores 0
    persist_pts = float(np.clip((persist - 0.5) / 0.5, -1, 1) * 25)
    # Divergence is a difference of two capped trends, so it spans ±2*_TREND_CAP.
    # DIVERGENCE_SCALE is a CALIBRATION CONSTANT: retune it against live sector
    # data once a few weeks of readings exist (synthetic extremes saturate it).
    diverg_pts = float(np.clip(diverg / DIVERGENCE_SCALE, -1, 1) * 25)
    # Quietness: +15 below threshold, scaling to -15 as volume gets louder
    if vol_ratio < tier_threshold:
        quiet_pts = 15.0
    else:
        excess = (vol_ratio - tier_threshold) / max(tier_threshold, 1e-9)
        quiet_pts = float(-15.0 * np.clip(excess, 0, 1))

    return round(cmf_pts + persist_pts + diverg_pts + quiet_pts, 2)


def event_score(high, low, close, volume,
                vol_ratio: float, tier_threshold: float) -> dict:
    """
    LOUD activity, 0-100 — "something happened here, go find out what."

    This is explicitly NOT an accumulation signal. A volume spike says an
    event occurred; it does not say who acted or on which side. The direction
    field below is the useful part: a spike closing near the session low on
    heavy volume is distribution, not accumulation, and the pre-v1 signal
    labeled both as "Accumulation" whenever the trailing month happened to
    be positive.
    """
    out = {"event_score": 0.0, "event_direction": "None",
           "spike": False, "spike_mfm": float("nan")}
    if len(close.dropna()) < 2 or vol_ratio is None or pd.isna(vol_ratio):
        return out

    spike = bool(vol_ratio >= tier_threshold)
    out["spike"] = spike

    mfm = money_flow_multiplier(high, low, close)
    mfm_now = float(mfm.iloc[-1]) if len(mfm) else float("nan")
    out["spike_mfm"] = mfm_now

    magnitude = float(np.clip((vol_ratio / tier_threshold - 1) / 1.0, 0, 1) * 60)
    base = 40.0 if spike else 0.0
    out["event_score"] = round(base + magnitude, 2)

    if not spike:
        out["event_direction"] = "None"
    elif mfm_now > 0.3:
        out["event_direction"] = "Buying event"
    elif mfm_now < -0.3:
        out["event_direction"] = "Selling event"
    else:
        out["event_direction"] = "Two-sided / unresolved"
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(ohlcv: pd.DataFrame, vol_ratio: float,
                tier_threshold: float, window: int = 21) -> dict:
    """
    Compute the full Tier-B panel for one ticker.

    `ohlcv` must have columns High, Low, Close, Volume (case-insensitive).
    Returns a flat dict suitable for a DataFrame row. All values NaN and
    `insufficient_history=True` when there are not enough bars — this
    function never returns a number computed on a truncated window.
    """
    cols = {c.lower(): c for c in ohlcv.columns}
    try:
        h, l = ohlcv[cols["high"]], ohlcv[cols["low"]]
        c, v = ohlcv[cols["close"]], ohlcv[cols["volume"]]
    except KeyError as e:
        raise KeyError(f"flow_metrics.compute_all needs High/Low/Close/Volume; missing {e}")

    n = len(c.dropna())
    if n < MIN_BARS_STEALTH:
        return {"insufficient_history": True, "bars": n,
                "cmf": np.nan, "cmf_persist": np.nan, "ad_divergence": np.nan,
                "mfi": np.nan, "obv_trend": np.nan,
                "stealth_label": "Insufficient history", "stealth_score": 0,
                "accumulation_score": np.nan,
                "event_score": 0.0, "event_direction": "None", "spike": False}

    cmf = chaikin_money_flow(h, l, c, v, period=window)
    mfi = money_flow_index(h, l, c, v)
    st = stealth_accumulation(h, l, c, v, vol_ratio, tier_threshold, window)
    ev = event_score(h, l, c, v, vol_ratio, tier_threshold)

    return {
        "insufficient_history": False,
        "bars": n,
        "cmf": round(float(cmf.dropna().iloc[-1]), 4) if cmf.notna().any() else np.nan,
        "cmf_persist": round(st["cmf_persist"], 3) if not pd.isna(st["cmf_persist"]) else np.nan,
        "ad_divergence": round(st["ad_divergence"], 5) if not pd.isna(st["ad_divergence"]) else np.nan,
        "mfi": round(float(mfi.dropna().iloc[-1]), 1) if mfi.notna().any() else np.nan,
        "obv_trend": round(_trend_strength(on_balance_volume(c, v), window), 3),
        "stealth_label": st["stealth_label"],
        "stealth_score": st["stealth_score"],
        "accumulation_score": accumulation_score(h, l, c, v, vol_ratio, tier_threshold, window),
        "event_score": ev["event_score"],
        "event_direction": ev["event_direction"],
        "spike": ev["spike"],
    }
