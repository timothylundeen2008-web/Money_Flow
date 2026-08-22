"""
regime_bands.py  (v1 — July 2026)
=================================
The short-real-rate BAND and the equity-leadership GUARD.

WHY THIS MODULE EXISTS
----------------------
Every non-crisis branch in classify_regime() was gated on the same bare
inequality:

    short_real_rate < 0

On 2026-07-29 the short real policy rate read approximately +0.10%
(EFFR ~3.58% - CPI YoY 3.50%). That single sign flip made THREE regimes
unreachable simultaneously — inflationary_repression, hard_repression, and
stagflation — and the classifier fell through to `goldilocks`, whose overlay
is:

    VGT +4, QQQ +3, SMH +2, GLD -3, SLV -2, TLT -4, KMLM -2, SGOV +2

i.e. it instructed ADDING to semiconductors and growth tech on a day the
Nasdaq 100 was in a technical correction 11% off its June high, the fourth
consecutive down session for SMH, and the Dow's worst decline since April
2025. That is not a mislabel. That is actively harmful sizing advice
produced by a 10-basis-point move in a number derived from a CPI print that
was distorted -0.4% m/m by a one-month energy collapse.

TWO FIXES, BOTH DEFENSIVE
-------------------------
1. BAND, not sign. |short_real| < TRANSITION_BAND is its own state. A gauge
   sitting inside its own measurement noise should say "I don't know", not
   pick a side. Band default 0.25% ~= one month of CPI noise.

2. LEADERSHIP GUARD. `goldilocks` cannot be confirmed while the growth
   complex is in a >10% drawdown from its own 60-day high. Positive real
   rates plus tight credit are necessary for goldilocks; they are not
   sufficient, and equity internals are the missing third leg.

FAIL DIRECTION
--------------
Both fail CLOSED, following the precedent already set by _gold_trend_ok()
in regime_classifier.py ("Fail SAFE: any data problem returns False — no
momentum data -> no add"). Missing price data cannot confirm goldilocks, so
it returns the conservative transition state with an explicit degraded
driver. Failing OPEN here would mean a data outage silently re-enables the
most aggressive overlay in the book.

INTEGRATION
-----------
See PATCHES.md. Three call sites in regime_classifier.classify_regime()
plus one new entry in the REGIMES dict.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
#  Tunables — every one of these is a decision, so it is named and documented
# --------------------------------------------------------------------------- #

# Half-width of the ambiguous band around a zero short real policy rate, in
# percentage points. 0.25 is roughly one month of CPI noise: a single monthly
# CPI print of +/-0.2% m/m moves trailing YoY by a comparable amount, and the
# June 2026 print alone moved headline YoY by 70bp. Anything inside this band
# is not a signal about the world, it is a signal about the measurement.
TRANSITION_BAND = 0.25

# Leadership guard: drawdown from the trailing high, and the lookback for that
# high. QQQ is the proxy because the growth complex is what the goldilocks
# overlay ADDS to — the guard has to watch the thing the overlay would buy.
LEADERSHIP_TICKER = "QQQ"
LEADERSHIP_LOOKBACK_DAYS = 60
LEADERSHIP_MAX_DRAWDOWN = -10.0     # percent

# ── Valuation / concentration guard (v2, August 2026) ────────────────────────
# The leadership guard catches a crash IN PROGRESS. It does NOT catch
# expensive-and-euphoric, which is the live risk: on 2026-08-07 QQQ sat at
# record highs (guard passes cleanly) with Shiller CAPE at 42.19 — the
# highest since August 2000, against an all-time peak of 44.19 in November
# 1999 — and the top 20 S&P names at 50.8% of index weight, a level unseen
# in over half a century.
#
# The specific hazard: July CPI could push the short real policy rate above
# +0.25%, clearing the band into DECISIVELY POSITIVE. With HY OAS ~284bp the
# goldilocks branch would then fire and its overlay (VGT +4, QQQ +3, SMH +2)
# would instruct ADDING growth at the second-highest valuation in ~150 years
# — while the ETF flow layer is blind. Positive real rates plus tight credit
# plus intact leadership are still not sufficient for goldilocks when the
# index is this expensive and this concentrated.
#
# Thresholds are deliberately extreme-only. This is a circuit breaker for
# historic outliers, not a valuation timing model — CAPE predicts the next
# decade, not the next twelve months, so it must never be used to call a top.
CAPE_BLOCK_LEVEL = 40.0          # dot-com-peak territory
CONCENTRATION_BLOCK_PCT = 45.0   # top-20 share of index weight


# Band labels, exposed so the UI can render them without re-deriving.
BAND_NEGATIVE = "NEGATIVE"
BAND_AMBIGUOUS = "AMBIGUOUS"
BAND_POSITIVE = "POSITIVE"
BAND_UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------- #
#  The band
# --------------------------------------------------------------------------- #
def short_real_band(short_real: Optional[float],
                    band: float = TRANSITION_BAND) -> dict:
    """
    Classify the short real policy rate into a BAND rather than a sign.

    Returns a dict, always, with keys:
        state   BAND_NEGATIVE | BAND_AMBIGUOUS | BAND_POSITIVE | BAND_UNKNOWN
        value   the input, echoed (may be None)
        band    the half-width used
        detail  a display string explaining the classification

    The detail string is written for the dashboard, not for logs — it has to
    explain to a reader WHY the regime is ambiguous, or the transition state
    reads like a bug.
    """
    if short_real is None:
        return {"state": BAND_UNKNOWN, "value": None, "band": band,
                "detail": "Short real policy rate unavailable — regime "
                          "classification is degraded, not neutral."}

    if short_real < -band:
        return {"state": BAND_NEGATIVE, "value": short_real, "band": band,
                "detail": f"Short real policy rate {short_real:+.2f}% is "
                          f"decisively negative (beyond the ±{band:.2f}% "
                          f"transition band). Repression channel is open at "
                          f"the front end."}

    if short_real > band:
        return {"state": BAND_POSITIVE, "value": short_real, "band": band,
                "detail": f"Short real policy rate {short_real:+.2f}% is "
                          f"decisively positive (beyond the ±{band:.2f}% "
                          f"transition band). Savers are being paid a real "
                          f"return at the front end."}

    return {"state": BAND_AMBIGUOUS, "value": short_real, "band": band,
            "detail": f"Short real policy rate {short_real:+.2f}% sits INSIDE "
                      f"the ±{band:.2f}% transition band — within the noise of "
                      f"a single CPI print. The framework's primary gauge is "
                      f"not currently giving a directional reading, so no "
                      f"regime that depends on its sign can be confirmed."}


def is_negative(short_real: Optional[float], band: float = TRANSITION_BAND) -> bool:
    """Decisively negative. Replaces `short_real < 0` at repression call sites."""
    return short_real_band(short_real, band)["state"] == BAND_NEGATIVE


def is_positive(short_real: Optional[float], band: float = TRANSITION_BAND) -> bool:
    """Decisively positive. Replaces `short_real >= 0` at the goldilocks site."""
    return short_real_band(short_real, band)["state"] == BAND_POSITIVE


def is_ambiguous(short_real: Optional[float], band: float = TRANSITION_BAND) -> bool:
    return short_real_band(short_real, band)["state"] == BAND_AMBIGUOUS


# --------------------------------------------------------------------------- #
#  The leadership guard
# --------------------------------------------------------------------------- #
def leadership_drawdown(fetch_prices: Callable,
                        ticker: str = LEADERSHIP_TICKER,
                        lookback: int = LEADERSHIP_LOOKBACK_DAYS) -> dict:
    """
    Drawdown of the growth complex from its own trailing high.

    Returns a dict, always, with keys:
        drawdown_pct   float or None
        high           float or None
        last           float or None
        available      bool
        detail         display string

    Never raises. `available=False` is the degraded case and callers must
    treat it as "cannot confirm", not as "no drawdown".
    """
    out = {"drawdown_pct": None, "high": None, "last": None,
           "available": False, "ticker": ticker, "lookback": lookback,
           "detail": ""}
    try:
        px = fetch_prices(ticker, "1y")
        # yfinance can hand back a single-column DataFrame or MultiIndex
        # columns even for one ticker — squeeze defensively, same pattern as
        # _inline_fetch_prices / _gold_trend_ok in regime_classifier.
        px = pd.Series(px.squeeze() if hasattr(px, "squeeze") else px)
        px = px.astype(float).dropna()
        if len(px) < lookback + 1:
            out["detail"] = (f"{ticker}: only {len(px)} sessions available, "
                             f"need {lookback + 1}. Leadership guard "
                             f"unavailable.")
            return out

        window = px.iloc[-lookback:]
        high = float(window.max())
        last = float(px.iloc[-1])
        if high <= 0:
            out["detail"] = f"{ticker}: non-positive trailing high; guard skipped."
            return out

        dd = (last / high - 1.0) * 100.0
        out.update(drawdown_pct=dd, high=high, last=last, available=True)
        out["detail"] = (f"{ticker} is {dd:+.1f}% from its {lookback}-session "
                         f"high of {high:,.2f} (last {last:,.2f}).")
        return out
    except Exception as e:                                   # never raise
        out["detail"] = f"Leadership guard failed for {ticker}: {e}"
        return out


def valuation_ok(cape: Optional[float] = None,
                 top20_concentration_pct: Optional[float] = None,
                 cape_block: float = CAPE_BLOCK_LEVEL,
                 conc_block: float = CONCENTRATION_BLOCK_PCT) -> tuple[bool, str]:
    """
    Block a growth-ADDITIVE regime when the index is at a historic valuation
    or concentration extreme.

    FAILS OPEN, deliberately — the opposite of leadership_ok(). Rationale:
    these inputs are manual/periodic (CAPE is monthly, concentration is
    quarterly), not live feeds. Failing closed on a missing monthly input
    would suppress goldilocks permanently for reasons unrelated to markets,
    which is its own silent-wrong-output failure. Missing data therefore
    passes WITH a visible warning, rather than blocking invisibly.

    Both thresholds are extreme-only. CAPE at 30 does not block; CAPE at 42
    does.
    """
    notes = []
    blocked = False

    if cape is None:
        notes.append("⚠ CAPE not supplied — valuation guard not evaluated.")
    elif cape >= cape_block:
        blocked = True
        notes.append(f"Shiller CAPE {cape:.1f} ≥ {cape_block:.0f} — "
                     f"dot-com-peak territory.")
    else:
        notes.append(f"CAPE {cape:.1f} below the {cape_block:.0f} block level.")

    if top20_concentration_pct is None:
        notes.append("⚠ Concentration not supplied — not evaluated.")
    elif top20_concentration_pct >= conc_block:
        blocked = True
        notes.append(f"Top-20 concentration {top20_concentration_pct:.1f}% ≥ "
                     f"{conc_block:.0f}% — index-level single-theme risk.")
    else:
        notes.append(f"Top-20 concentration {top20_concentration_pct:.1f}% "
                     f"below the {conc_block:.0f}% block level.")

    if blocked:
        return False, ("Valuation guard BLOCKS a growth-additive regime: "
                       + " ".join(notes) +
                       " Adding to growth at a historic valuation extreme is "
                       "not what a confirmed benign regime should instruct. "
                       "Routing to the transition overlay instead.")
    return True, "Valuation guard clear: " + " ".join(notes)


def leadership_ok(fetch_prices: Callable,
                  ticker: str = LEADERSHIP_TICKER,
                  lookback: int = LEADERSHIP_LOOKBACK_DAYS,
                  max_dd: float = LEADERSHIP_MAX_DRAWDOWN) -> tuple[bool, str]:
    """
    True only when the growth complex is NOT in a >max_dd drawdown.

    FAILS CLOSED: unavailable data returns (False, reason). The goldilocks
    overlay adds +4 VGT / +3 QQQ / +2 SMH; a data outage must not be the
    thing that re-enables it.
    """
    dd = leadership_drawdown(fetch_prices, ticker, lookback)
    if not dd["available"]:
        return False, (f"⚠ Leadership guard unavailable ({dd['detail']}) — "
                       f"cannot confirm a growth-additive regime on missing "
                       f"price data.")
    if dd["drawdown_pct"] <= max_dd:
        return False, (f"Leadership guard BLOCKS: {dd['detail']} "
                       f"Threshold {max_dd:+.0f}%. A regime whose overlay adds "
                       f"to growth cannot be confirmed while the growth "
                       f"complex is in a correction.")
    return True, f"Leadership guard clear: {dd['detail']}"


# --------------------------------------------------------------------------- #
#  The overlay for the new regime — kept HERE so the tilt and the band that
#  triggers it live in one file and cannot drift apart.
# --------------------------------------------------------------------------- #
# Rationale, sleeve by sleeve. When the framework's primary gauge is inside
# its own noise, the correct action is not to guess which side it will resolve
# to — it is to stop expressing EITHER side and get paid to wait.
#
#   Growth trimmed  (-5)  Adding growth requires a CONFIRMED positive real
#                         rate; an unconfirmed one is not a reason to buy
#                         multiple expansion.
#   Metals trimmed  (-4)  Adding metals requires a CONFIRMED negative real
#                         rate; symmetric logic, opposite side.
#   TLT trimmed     (-3)  Long duration stays hostage to term premium
#                         regardless of where the FRONT end resolves. This is
#                         the one sleeve the ambiguity does not rescue.
#   Floaters  +4 / Bills +3  Carry with no duration risk. Paid to wait, and
#                         floaters specifically benefit if the ambiguity
#                         resolves toward hikes.
#   Trend     +3          Trend is regime-agnostic — it profits from whichever
#                         way the long end resolves, which is exactly the
#                         exposure you want when you cannot call the direction.
#   Defensives +2         Small tilt to cash-flow and healthcare; pricing
#                         power is robust to both resolutions.
#
# Sum: -12 funded / +12 deployed. MUST be sum-zero or regime_classifier's
# import-time assertion will fail — which is the intended behaviour.
TRANSITION_OVERLAY = {
    "VGT": -3, "QQQ": -1, "SMH": -1,
    "GLD": -2, "SLV": -1, "RING": -1,
    "TLT": -3,
    "USFR": +4, "SGOV": +3,
    "KMLM": +3,
    "SCHD": +1, "XLV": +1,
}

TRANSITION_LABEL = "Transition — Ambiguous (short real rate at zero)"

TRANSITION_BLURB = (
    "The short real policy rate is inside the ±{band:.2f}% transition band, so "
    "the framework's primary gauge is not giving a directional reading. No "
    "regime that depends on its sign can be confirmed. Hold near base weights, "
    "take carry at the front end with no duration risk, keep trend on (it is "
    "agnostic to which way this resolves), and express NEITHER the repression "
    "trade nor the reflation trade until the gauge clears the band."
).format(band=TRANSITION_BAND)


# Self-check at import, mirroring regime_classifier's FIX 1 guard.
_s = sum(TRANSITION_OVERLAY.values())
assert _s == 0, f"TRANSITION_OVERLAY sums to {_s:+d}; overlays must be sum-zero"


def selftest(fetch_prices: Callable | None = None) -> dict:
    """
    Verify band logic against the boundary cases that actually matter, and
    optionally exercise the leadership guard against live prices.

    Run this once after deployment. A band that is off by a sign is invisible
    until the day it flips your regime.
    """
    cases = [
        (-0.60, BAND_NEGATIVE),    # May 2026: EFFR 3.63 - CPI 4.20
        (-0.26, BAND_NEGATIVE),    # just outside the band
        (-0.25, BAND_AMBIGUOUS),   # boundary is inclusive of ambiguous
        (+0.08, BAND_AMBIGUOUS),   # 2026-07-29: EFFR 3.58 - CPI 3.50
        (+0.25, BAND_AMBIGUOUS),
        (+0.26, BAND_POSITIVE),
        (+1.50, BAND_POSITIVE),
        (None,  BAND_UNKNOWN),
    ]
    failures = []
    for val, expected in cases:
        got = short_real_band(val)["state"]
        if got != expected:
            failures.append({"input": val, "expected": expected, "got": got})

    out = {"band": TRANSITION_BAND,
           "cases_run": len(cases),
           "failures": failures,
           "ok": not failures,
           "overlay_sum": sum(TRANSITION_OVERLAY.values())}

    if fetch_prices is not None:
        ok, why = leadership_ok(fetch_prices)
        out["leadership_ok"] = ok
        out["leadership_detail"] = why

    return out


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, default=str))
