"""
regime_classifier.py  (v3 — July 2026 band/guard patch; v8 Oct 2026: growth_scare needs price confirmation)
=======================================================

v3 CHANGELOG (vs v2):
  FIX 6  BAND, not sign. Every non-crisis branch was gated on
         `short_real_rate < 0`. On 2026-07-29 the short real policy rate read
         +0.10% (EFFR ~3.58% - CPI 3.50%), which made inflationary_repression,
         hard_repression AND stagflation unreachable simultaneously and dumped
         the classifier into goldilocks -- whose overlay is VGT +4, QQQ +3,
         SMH +2. It instructed ADDING to semis on the fourth consecutive down
         session of a 12% SMH drawdown. |short_real| < 0.25% is now its own
         regime: transition_ambiguous. See regime_bands.py.
  FIX 7  LEADERSHIP GUARD on goldilocks. Positive real rates plus tight credit
         are NECESSARY for goldilocks, not SUFFICIENT. The branch is now
         blocked while the growth complex is >10% below its own 60-day high.
         Fails CLOSED, matching _gold_trend_ok()'s precedent.
  FIX 8  The gold momentum gate now covers EVERY regime that adds GLD.
         v2 gated inflationary_repression (+3) only, leaving hard_repression
         (+4) and stagflation (+4) ungated -- so two of the three regimes that
         buy gold could still buy it into a confirmed downtrend. As of
         2026-07-29 GLD sits ~10% below a FALLING 200d with an active death
         cross, so this is live, not theoretical.
  FIX 9  repression_score() reports top-weight points, the hollow flag, and the
         RAW 2s10s momentum. A 5 built entirely from fiscal/plumbing components
         is not the same state as a 5 that includes the two real-yield gauges,
         and the band label alone hides the difference.
  FIX 10 full_assessment() now FORWARDS fed_bs_expanding and
         deficit_gt_5pct_gdp to repression_score(), and fetch_prices to
         classify_regime(). Previously both flags were silently dropped, so the
         live score was structurally capped at 8/10 and permanently degraded.

v2 CHANGELOG (vs v1):
Shared macro-regime engine for the Repression Dashboard and the All-Weather
Portfolio Dashboard.

Core idea implemented here (per the crux correction):
  There are TWO different "real yields" and they must never be conflated.

    1. SHORT real policy rate  = EFFR (DFF) - trailing CPI YoY   -> repression gauge
    2. LONG  real market yield = DFII10 (10y TIPS yield)         -> duration friend/foe

  The SIGN of (1) and the DIRECTION (momentum) of (2), combined with HY credit
  spreads and the 60-day stock/bond correlation, place us in one of FIVE regime
  quadrants, each of which maps to a target portfolio tilt.

v2 CHANGELOG (vs v1):
  FIX 1  Overlays are now sum-zero by construction (asserted at import).
         v1's inflationary_repression overlay summed to -6, so renormalization
         silently scaled UNTOUCHED sleeves up ~6.4% — VGT rose 20%->21.3% in
         the one regime whose defining signal (rising long real yields)
         compresses growth multiples. Overlays now trim growth explicitly.
  FIX 2  The GLD tilt in inflationary_repression is MOMENTUM-GATED
         (Level-4 entry confirmation). Rising long real yields are gold's
         primary headwind; the regime must not mechanically add to a metal
         in a confirmed downtrend. Gate fails -> the +3 redirects to SGOV.
  FIX 3  New HARD_REPRESSION regime. v1 had no rule for (short real negative,
         long real FALLING, credit tight) — the max-metals, TLT-viable
         quadrant — so it fell through to 'neutral' with no tilts.
  FIX 4  USFR is no longer cut with SGOV. Floaters' coupons RISE with hikes;
         when the front-end risk is a hike (June 2026 SEP: 3.8% end-2026),
         USFR is a beneficiary, not dead cash.
  FIX 5  classify_regime surfaces missing DFII10 momentum instead of silently
         treating None as "not rising". Precedence comments added.
  NEW    repression_score(): the 0-10 stacking score from the written
         framework, now in code. The quadrant says WHAT; the score says HOW
         HARD to tilt.

Public API is unchanged and backward compatible:
  full_assessment, compute_signals, classify_regime, target_weights,
  kmlm_signal, fed_reaction_flag, REGIMES, BASE_WEIGHTS, SignalSet

FRED series used:
  DFF          Effective Federal Funds Rate (daily)
  CPIAUCNS     CPI, not seasonally adjusted (index; YoY computed here)
  DFII10       10y TIPS real yield
  T10YIE       10y breakeven inflation
  DGS10, DGS2  nominal 10y / 2y (for 2s10s)
  BAMLH0A0HYM2 ICE BofA US High Yield OAS
  BAMLC0A0CM   ICE BofA US Corporate (IG) OAS
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

try:
    import requests  # only needed if you use the inline FRED fetcher
except Exception:  # pragma: no cover
    requests = None

FED_TARGET_INFLATION = 2.0

# v3 FIX 6/7: band logic + leadership guard live in their own module so the
# tilt and the test that triggers it cannot drift apart.
import regime_bands as _rb

FRED_SERIES = {
    "eff_funds": "DFF",
    # v3.4 fix. Was CPIAUCSL (seasonally adjusted). BLS's own headline YoY
    # figure -- the "3.4%" quoted in every release and every news article --
    # is ALWAYS computed from the NOT-seasonally-adjusted series: "rose 3.4
    # percent over the last 12 months, not seasonally adjusted (NSA)"
    # (bls.gov/cpi/). CPIAUCSL is explicitly the seasonally-adjusted series
    # per FRED's own series description. SA and NSA 12-month changes commonly
    # diverge by ~0.1pp after seasonal-factor revisions -- this was masked
    # while the larger calendar-gap bug (v3.3) was overstating YoY by ~0.5pp;
    # fixing that one exposed this smaller, independent, structural mismatch.
    "cpi_index": "CPIAUCNS",
    # New in this pass: the SA companion series, used ONLY for the two
    # leading/current-read metrics below (cpi_3m_saar, cpi_mom_sa) -- never
    # for cpi_yoy or short_real_rate, which must stay NSA.
    "cpi_index_sa": "CPIAUCSL",
    "real_10y": "DFII10",
    "breakeven_10y": "T10YIE",
    # v5: the dollar. Added because the framework was structurally blind to
    # FX -- SignalSet had 14 fields covering rates/credit/curve/inflation and
    # ZERO covering the currency, so a rising-long-yields/falling-dollar
    # divergence (the classic term-premium / debt-confidence signature) was
    # literally unrepresentable in the regime read.
    "dxy": "DTWEXBGS",          # Broad USD index, daily, FRED
    "nom_10y": "DGS10",
    "nom_2y": "DGS2",
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",
}

# --------------------------------------------------------------------------- #
#  Regime definitions + target tilts
# --------------------------------------------------------------------------- #
# Weights are the *base* All-Weather sleeve. Each regime supplies an overlay
# that shifts weights. Overlays are additive deltas (in %) and MUST sum to
# zero (asserted below) so that a tilt means exactly what it says — v1's
# renormalization of non-zero-sum overlays silently distorted untouched
# sleeves.

BASE_WEIGHTS = {
    "VGT": 20, "SMH": 4, "QQQ": 4,           # growth / tech
    "GLD": 12, "SLV": 5, "RING": 5,          # precious metals
    "XLE": 5, "PDBC": 3,                     # commodities / energy
    "SCHD": 13, "XLV": 4, "XLU": 3,          # defensive equity
    "SGOV": 5, "USFR": 3,                    # cash
    "TLT": 10, "KMLM": 4,                    # duration (contingent) + trend
}

REGIMES = {
    "inflationary_repression": {
        "label": "Inflationary Repression",
        "blurb": (
            "Negative SHORT real rate + POSITIVE, RISING long real yield. Debt "
            "eroded via inflation overshoot at the front end; long end NOT "
            "suppressed. Real assets WITH MOMENTUM and trend win; long "
            "duration bleeds; rate-sensitive growth compresses."
        ),
        # sum-zero: -24 / +24
        "overlay": {
            # ── funded from ──
            "TLT": -10,                        # contingent duration OFF (bleeds)
            "VGT": -4, "SMH": -2, "QQQ": -1,   # rising real yields hit growth
            "SLV": -2, "RING": -3,             # high-beta metals: no trend, no add
            "SGOV": -2,                        # T-bills bleed in real terms
            # ── deployed to ──
            "KMLM": +6,                        # trend replaces failing bond hedge
            "XLE": +3, "PDBC": +2,             # real assets WITH momentum
            "SCHD": +4, "XLV": +2, "XLU": +1,  # defensive rotation
            "USFR": +3,                        # floaters win if the Fed hikes
            "GLD": +3,                         # MOMENTUM-GATED (see target_weights)
        },
    },
    "hard_repression": {
        "label": "Hard Repression",
        "blurb": (
            "Negative short real rate with long real yields suppressed or "
            "falling while credit stays calm (yield-curve-control signature). "
            "Peak debasement: metals and miners lead, duration works again, "
            "cash is the worst asset."
        ),
        # sum-zero: -11 / +11
        "overlay": {
            "GLD": +4, "RING": +2, "SLV": +1,
            "TLT": +2, "KMLM": +2,
            "SGOV": -4, "USFR": -2,
            "VGT": -3, "QQQ": -1, "SMH": -1,
        },
    },
    "liquidity_crisis": {
        "label": "Liquidity Crisis",
        "blurb": (
            "HY spreads blowing out, long real yields FALLING (flight to "
            "quality). Duration and cash are the shock absorbers; metals may "
            "sell off first before rallying."
        ),
        # sum-zero: -14 / +14 (unchanged from v1 — already balanced)
        "overlay": {
            "TLT": +6,            # switch/boost contingent duration
            "SGOV": +4, "USFR": +2,
            "KMLM": +2,
            "VGT": -6, "SMH": -2, "QQQ": -2,
            "SLV": -2, "RING": -2,
        },
    },
    "stagflation": {
        "label": "Stagflation",
        "blurb": (
            "Negative short real rate WITH growth rolling over (2s10s "
            "re-steepening from inversion). Gold, trend, and defensives; cut "
            "cyclical growth and energy demand risk."
        ),
        # sum-zero: -13 / +13 (v1 summed to -1; XLU +1 -> +2)
        "overlay": {
            "GLD": +4, "KMLM": +3,
            "SCHD": +2, "XLV": +2, "XLU": +2,
            "VGT": -5, "SMH": -2, "QQQ": -2, "XLE": -3,
            "TLT": -1,
        },
    },
    "goldilocks": {
        "label": "Goldilocks / Reflation",
        "blurb": (
            "Positive real rates, tight credit, stable inflation. Normalize "
            "toward growth; trim hedges and reduce trend."
        ),
        # sum-zero: -11 / +11 (unchanged from v1 — already balanced)
        "overlay": {
            "VGT": +4, "QQQ": +3, "SMH": +2,
            "KMLM": -2, "TLT": -4,
            "GLD": -3, "SLV": -2,
            "SGOV": +2,
        },
    },
    # v4: GROWTH SCARE. The first branch in this framework that does NOT key
    # off the short real rate's sign. Fires when the growth composite is
    # CONTRACTING (>= 3 of 4 series live and agreeing), regardless of where
    # real rates sit.
    #
    # WHY IT HAS TO EXIST: on 2026-08-14 payrolls were -23,000, retail sales
    # had fallen the most in over a year, and consumer sentiment was 51.0 --
    # an unambiguous growth scare that this classifier could not name,
    # because `stagflation` requires a decisively NEGATIVE short real rate
    # and the rate was +0.27%. A framework that cannot label a contracting
    # economy is not a macro framework.
    #
    # OVERLAY REASONING, sleeve by sleeve:
    #   Growth/cyclicals cut (-11): VGT, QQQ, SMH, XLE, PDBC. Earnings
    #       estimates fall in a contraction and high-multiple names de-rate
    #       hardest; energy and broad commodities are demand-sensitive.
    #   Defensives/carry up (+11): SCHD, XLV, XLU (cash-flow and inelastic
    #       demand), SGOV/USFR (paid to wait), KMLM (trend is the one sleeve
    #       agnostic to WHICH way this resolves).
    #   TLT deliberately UNCHANGED at 0. A growth scare argues for duration;
    #       a term-premium/fiscal scare argues violently against it, and the
    #       two look identical at the start. The HY>5% liquidity_crisis
    #       override already re-arms TLT for the genuinely deflationary case,
    #       which is the branch where duration reliably works. Adding
    #       duration here would be taking a side the data has not yet picked.
    "growth_scare": {
        "label": "Growth Scare — contraction signal",
        "blurb": (
            "The growth composite is CONTRACTING: labour and consumer data "
            "are deteriorating together, independent of where real rates "
            "sit. Cut cyclicals and high-multiple growth, rotate to "
            "cash-flow defensives and front-end carry, and keep trend on. "
            "Duration is deliberately NOT added here — a growth scare and a "
            "fiscal/term-premium scare look identical early, and only the "
            "credit override can tell them apart."
        ),
        "overlay": {
            "VGT": -4, "QQQ": -2, "SMH": -2, "XLE": -2, "PDBC": -1,
            "SCHD": +3, "XLV": +3, "XLU": +1, "SGOV": +2, "USFR": +1,
            "KMLM": +1,
        },
    },
    # v5: TERM PREMIUM REPRICING -- structurally the INVERSE of financial
    # repression, and the state this framework previously could not name.
    #
    # Repression is: negative front-end real rates, a SUPPRESSED long end,
    # savers quietly liquidated. This is the opposite on every axis: savers
    # PAID a positive real rate at the front end, the long end RISING because
    # buyers demand more compensation, and the currency WEAKENING even as
    # yields climb.
    #
    # That last part is the whole signal. Higher yields normally ATTRACT
    # foreign capital and support the currency. When yields rise and the
    # dollar falls simultaneously, the higher yield is not pricing better
    # growth or higher expected inflation -- it is pricing the RISK of
    # holding the paper at all: fiscal sustainability, supply, credibility.
    # indicators.score_dollar_divergence() has detected exactly this pattern
    # for some time, but it fed only the Repression Watch scorecard and never
    # reached this classifier -- the signal existed and the regime engine
    # could not hear it.
    #
    # OVERLAY REASONING:
    #   TLT -6   Long duration is the DIRECT loser. Term premium expansion is
    #            precisely a repricing of long-dated paper downward. This is
    #            the largest single tilt in the overlay for that reason.
    #   VGT -4, QQQ -2, SMH -1  Long-duration EQUITY. Rising discount rates
    #            hit the assets whose value sits furthest in the future,
    #            which is exactly high-multiple growth.
    #   GLD +4, SLV +2  The classic debasement hedge. Gold's whole case is a
    #            currency losing credibility while real assets hold it.
    #   PDBC +2, XLE +1  Commodities are priced in dollars; a weakening
    #            dollar is a mechanical tailwind.
    #   KMLM +2  Trend is agnostic to WHICH way this resolves, and this is
    #            precisely the kind of persistent directional move it
    #            captures.
    #   USFR +2  Floating rate. The front end pays a positive REAL rate here
    #            (unlike repression, where it does not) and the coupon RISES
    #            if the Fed responds to currency weakness by tightening.
    "term_premium_repricing": {
        "label": "Term Premium Repricing — fiscal/currency risk",
        "blurb": (
            "The INVERSE of financial repression. The short real rate is "
            "positive (savers are paid), but the long end is rising while "
            "the dollar FALLS — higher yields are failing to attract the "
            "capital they normally would. That combination prices "
            "fiscal/credibility risk, not growth or inflation expectations. "
            "Long duration is the direct loser; real assets and trend are "
            "the direct beneficiaries. Note this is NOT a credit event: "
            "spreads are calm, which is what separates it from "
            "liquidity_crisis. Watch whether credit eventually confirms what "
            "rates are already pricing."
        ),
        "overlay": {
            "TLT": -6, "VGT": -4, "QQQ": -2, "SMH": -1,
            "GLD": +4, "SLV": +2, "PDBC": +2, "XLE": +1,
            "KMLM": +2, "USFR": +2,
        },
    },
    # v6, Sept 2026: RESTRICTIVE TIGHTENING — the state the term-premium
    # branch's own comment describes ("rising yields WITH a firm dollar is
    # ordinary tightening") but that no branch returned. It fell through to
    # goldilocks, got blocked by the valuation guard, and was labelled
    # `transition_ambiguous`, whose overlay holds TLT near base weight while
    # long real yields rise — the opposite of what the checklist's own
    # "TLT 0% while DFII10 rising" rule instructs.
    #
    # Signature: short real rate decisively POSITIVE (savers paid), long real
    # yield RISING >= 0.20pp over 3 months, dollar NOT falling (otherwise it
    # is term_premium_repricing), credit calm (otherwise liquidity_crisis).
    # Live example 2026-09-26: short real +0.48%, DFII10 2.85% (+0.66pp/3mo),
    # UUP +2.1% over 1 month, HY 2.80%, after a 25bp September hike.
    #
    # OVERLAY REASONING (sum-zero, -19 / +19):
    #   TLT -8   Duration is the direct loser of rising real yields at the
    #            long end. Leaves 2%, not 0: a residual hedge for the day the
    #            tightening breaks something (the HY override re-arms it).
    #   VGT -3, QQQ -2, SMH -1  Long-duration equity; a rising discount rate
    #            compresses the multiples furthest out.
    #   XLU -2   Bond proxy; same mechanism as TLT.
    #   GLD -2, SLV -1  Rising REAL yields are gold's primary headwind (the
    #            momentum gate already refuses adds in this state).
    #   SGOV +6, USFR +5  The front end now pays a POSITIVE real rate. Cash
    #            is an asset in this regime, and floaters reset upward if
    #            the Fed keeps going.
    #   KMLM +4  Trend earns in persistent rate moves, and the stock/bond
    #            correlation is positive — the bond hedge is not working.
    #   SCHD +3, XLV +1  Cash-flow equity with lower duration than growth.
    "restrictive_tightening": {
        "label": "Restrictive Tightening — real rates rising",
        "blurb": (
            "Real rates are rising at BOTH ends: the Fed is holding or "
            "raising a policy rate above inflation, and the 10-year real "
            "yield is climbing, while the dollar is firm and credit is calm. "
            "Money is getting more expensive in an orderly way. Duration and "
            "long-duration growth are the losers; front-end cash (paid a "
            "positive real rate), trend and cash-flow defensives are the "
            "winners. Not a crisis: watch HY spreads for the moment "
            "tightening starts to break something."
        ),
        "overlay": {
            "TLT": -8, "VGT": -3, "QQQ": -2, "SMH": -1, "XLU": -2,
            "GLD": -2, "SLV": -1,
            "SGOV": +6, "USFR": +5, "KMLM": +4, "SCHD": +3, "XLV": +1,
        },
    },
    # v7: CREDIT STRESS. Every calm-credit branch above (hard repression,
    # term premium, restrictive tightening, goldilocks) required HY OAS < 3.5%,
    # and the only branch for wide spreads was the liquidity-crisis override
    # (> 5% AND +0.5pp in two weeks). Between 3.5% and that override nothing
    # matched, so a widening-credit tape fell through to "neutral" -- the
    # label that means "hold base weights". This is the missing state.
    "credit_stress": {
        "label": "Credit Stress — spreads widening",
        "blurb": (
            "High-yield spreads have left the calm zone (>= 3.5%) and are "
            "either widening fast or already wide (>= 4.5%), but have not hit "
            "the liquidity-crisis override. Credit leads equities: trim "
            "high-multiple growth, cyclicals and miners; lift front-end cash, "
            "trend and quality defensives. While long real yields are still "
            "RISING this keeps every restrictive-tightening defence (TLT ~2%) "
            "and adds a credit layer on top. Only once real yields turn DOWN "
            "(flight to quality) does duration come back, and then only "
            "through the entry gate."
        ),
        # v7.1 (Oct 2026) — ESCALATION-SAFE. Overlays apply to BASE, not to the
        # previous regime. v7's original overlay left TLT at base (10%), so
        # escalating from restrictive_tightening (TLT 2%) would have BOUGHT
        # 8pts of TLT, re-added XLU and gold through failed stops/gates, and
        # cut cash 5pts. This overlay = restrictive_tightening's overlay +
        # a sum-zero credit layer, so the escalation is strictly more
        # defensive. sum-zero: -27 / +27.
        "overlay": {
            "TLT": -8, "VGT": -6, "QQQ": -3, "SMH": -2, "XLU": -2, "GLD": -2,
            "SLV": -1, "XLE": -1, "PDBC": -1, "RING": -1,
            "SGOV": +10, "USFR": +6, "KMLM": +5, "SCHD": +3, "XLV": +3,
        },
        # Used instead when the 10y real yield's 3-month change is NEGATIVE
        # (flight to quality): duration and gold back to base, credit cuts
        # kept. Any re-add still has to pass entry_gate(). sum-zero: -15/+15.
        "overlay_flight": {
            "VGT": -6, "QQQ": -3, "SMH": -2, "XLE": -1, "PDBC": -1,
            "RING": -1, "SLV": -1,
            "SGOV": +6, "USFR": +3, "KMLM": +3, "XLV": +3,
        },
    },
    # v3 FIX 6. Fires when |short real rate| < regime_bands.TRANSITION_BAND.
    # The gauge is inside its own measurement noise, so express NEITHER the
    # repression trade nor the reflation trade and take carry while waiting.
    "transition_ambiguous": {
        "label": _rb.TRANSITION_LABEL,
        "blurb": _rb.TRANSITION_BLURB,
        "overlay": dict(_rb.TRANSITION_OVERLAY),
    },
    "neutral": {
        "label": "Neutral / Transition",
        "blurb": (
            "Signals are mixed or transitioning between quadrants. Hold the "
            "base allocation and wait for confirmation before rebalancing."
        ),
        "overlay": {},
    },
}

# FIX 1 guard: overlays must be sum-zero so tilts mean what they say.
for _k, _r in REGIMES.items():
    _s = sum(_r["overlay"].values())
    assert _s == 0, f"Overlay for '{_k}' sums to {_s:+d}; overlays must be sum-zero"


# --------------------------------------------------------------------------- #
#  Inline fetchers (fallbacks). Pass your own to override.
# --------------------------------------------------------------------------- #
def _inline_fetch_fred(series_id: str, api_key: str,
                       start: str = "2015-01-01") -> pd.Series:
    """Minimal FRED fetch mirroring the dashboard's existing pattern.
    Returns a float Series indexed by date; empty Series on any failure."""
    if requests is None or not api_key:
        return pd.Series(dtype=float)
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if not obs:
            return pd.Series(dtype=float)
        idx, vals = [], []
        for o in obs:
            v = o.get("value", ".")
            if v not in (".", "", None):
                idx.append(pd.Timestamp(o["date"]))
                vals.append(float(v))
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name=series_id)
    except Exception:
        return pd.Series(dtype=float)


def _inline_fetch_prices(ticker: str, period: str = "1y") -> pd.Series:
    """
    Always returns a 1-D Series. Recent yfinance versions can return a
    single-column DataFrame or MultiIndex columns even for one ticker, so we
    squeeze/flatten defensively.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False,
                         auto_adjust=True)
        if df is None or len(df) == 0:
            return pd.Series(dtype=float)
        # Prefer the Close column; handle MultiIndex and single-column frames.
        obj = df["Close"] if "Close" in df.columns else df
        if isinstance(obj, pd.DataFrame):
            # single column -> squeeze to Series; multi -> take first column
            obj = obj.iloc[:, 0] if obj.shape[1] >= 1 else pd.Series(dtype=float)
        return pd.Series(obj).astype(float).dropna()
    except Exception:
        return pd.Series(dtype=float)


# --------------------------------------------------------------------------- #
#  Signal computation
# --------------------------------------------------------------------------- #
# ── CPI label constants — imported everywhere a CPI figure is displayed ─────
# One source of truth so "CPI YoY (NSA)" can never drift into "CPI (NSA)" in
# one file and "YoY CPI, not seasonally adjusted" in another. Every display
# site — the dashboard header, the daily log line, the alert text — pulls
# these exact strings rather than writing its own.
CPI_YOY_LABEL   = "CPI YoY (NSA)"
CPI_SAAR_LABEL  = "CPI 3M SAAR"
CPI_MOM_LABEL   = "CPI MoM (SA)"

CPI_YOY_HELP  = ("Trailing 12-month change, NOT seasonally adjusted — the "
                 "figure BLS itself uses for its headline release and the "
                 "one every news article quotes. LAGGING by design: still "
                 "weighted by prints from up to 11 months ago. Feeds "
                 "short_real_rate (EFFR minus this).")
CPI_SAAR_HELP = ("Last 3 months, seasonally adjusted, compounded to an "
                 "annual rate: ((index_now/index_3mo_ago)^4 - 1) x 100. "
                 "LEADING: catches an inflation inflection months before "
                 "it shows up in the slower YoY figure. Context only — "
                 "does NOT feed the regime classifier.")
CPI_MOM_HELP  = ("Single latest month, seasonally adjusted. The rawest, "
                 "noisiest, most current read — the number markets react "
                 "to on release day. Context only.")


@dataclass
class SignalSet:
    short_real_rate: Optional[float] = None      # EFFR - CPI YoY
    long_real_yield: Optional[float] = None      # DFII10 level
    long_real_mom_3m: Optional[float] = None     # change over ~63 sessions
    breakeven_10y: Optional[float] = None
    cpi_yoy: Optional[float] = None               # NSA, 12mo — feeds short_real_rate
    cpi_3m_saar: Optional[float] = None            # SA, 3mo annualized — leading
    cpi_mom_sa: Optional[float] = None             # SA, latest single month
    eff_funds: Optional[float] = None
    spread_2s10s: Optional[float] = None
    spread_2s10s_mom_3m: Optional[float] = None
    hy_oas: Optional[float] = None
    hy_oas_mom_2w: Optional[float] = None
    ig_oas: Optional[float] = None
    stock_bond_corr_60d: Optional[float] = None
    dxy: Optional[float] = None                   # broad USD index level
    dxy_20d_change_pct: Optional[float] = None    # ~1-month % change
    asof: Optional[_dt.date] = None
    notes: list = field(default_factory=list)
    credit_cycle: Optional[dict] = None           # v7: spread-peak re-entry state

    def as_row(self) -> pd.DataFrame:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("notes", "asof", "credit_cycle")}
        return pd.DataFrame([d])


def _last(s: pd.Series):
    return None if s is None or s.empty else float(s.iloc[-1])


def _delta(s: pd.Series, sessions: int):
    if s is None or len(s.dropna()) <= sessions:
        return None
    s = s.dropna()
    return float(s.iloc[-1] - s.iloc[-1 - sessions])


def compute_signals(
    fred_api_key: str = "",
    fetch_fred: Callable = _inline_fetch_fred,
    fetch_prices: Callable = _inline_fetch_prices,
    start: str = "2015-01-01",
) -> SignalSet:
    """Pull the raw series and derive the two real yields + companions."""
    sig = SignalSet(asof=_dt.date.today())

    eff = fetch_fred(FRED_SERIES["eff_funds"], fred_api_key, start)
    cpi = fetch_fred(FRED_SERIES["cpi_index"], fred_api_key, start)
    cpi_sa = fetch_fred(FRED_SERIES["cpi_index_sa"], fred_api_key, start)
    r10 = fetch_fred(FRED_SERIES["real_10y"], fred_api_key, start)
    be10 = fetch_fred(FRED_SERIES["breakeven_10y"], fred_api_key, start)
    n10 = fetch_fred(FRED_SERIES["nom_10y"], fred_api_key, start)
    n2 = fetch_fred(FRED_SERIES["nom_2y"], fred_api_key, start)
    hy = fetch_fred(FRED_SERIES["hy_oas"], fred_api_key, start)
    ig = fetch_fred(FRED_SERIES["ig_oas"], fred_api_key, start)
    dxy_s = fetch_fred(FRED_SERIES["dxy"], fred_api_key, start)

    # --- CPI YoY from the index (resample to month-end; calendar-safe) ---
    #
    # v3.3 fix. Previously: resample("ME").last().dropna() BEFORE pct_change(12).
    # Oct and Nov 2025 CPI were never published (BLS shutdown-related
    # cancellation) -- CPIAUCSL has no observations for those two months, full
    # stop, permanently. resample("ME") correctly creates NaN placeholder rows
    # for them, spanning the true calendar gap -- but the immediate .dropna()
    # REMOVED those placeholder rows, compressing the index. pct_change(12) is
    # POSITION-based, not date-based: with the gap months physically removed,
    # "12 rows back" from any month after the gap no longer equals "12 calendar
    # months back" -- it lands roughly 2 months EARLIER than it should (e.g.
    # comparing July 2026 to May 2025 instead of July 2025).
    #
    # Reproduced exactly with synthetic data spanning the real gap: this
    # ordering overstated July 2026 YoY by +0.47pp against a same-inputs fixed
    # calculation -- the same direction and rough magnitude as the live
    # 3.54% (code) vs 3.4% (actual BLS July 2026 print, confirmed via news
    # search) discrepancy this fix was written to close.
    #
    # Fix: dropna() AFTER pct_change(12), not before. Keeping the NaN
    # placeholders through the position-based calculation preserves true
    # calendar spacing -- pct_change(12) then correctly produces NaN only for
    # the specific months whose OWN 12-months-back comparison touches a gap
    # month, while every other month's comparison (like today's, once the
    # window has moved past Oct/Nov 2025) stays calendar-accurate. Only strip
    # NaN from the OUTPUT, once alignment is no longer at risk.
    if cpi is not None and not cpi.empty:
        cpi_m = cpi.resample("ME").last()          # keep gap-month NaNs
        if len(cpi_m) > 12:
            yoy = (cpi_m.pct_change(12) * 100).dropna()   # drop only now
            if len(yoy):
                sig.cpi_yoy = _last(yoy)

    # --- CPI 3M SAAR (leading) and CPI MoM SA (most current) ---
    # Deliberately SA here, unlike cpi_yoy above -- see CPI_SAAR_HELP /
    # CPI_MOM_HELP. At a 3-month or 1-month window seasonal effects do NOT
    # cancel the way they do over a full 12-month comparison, so seasonal
    # adjustment is the theoretically correct choice for these two, exactly
    # the mirror image of why cpi_yoy above must be NSA. Same
    # dropna-after-pct_change discipline as the NSA fix, applied here too,
    # so a future data gap in the SA series can't reintroduce the same
    # calendar-alignment bug in a new place.
    if cpi_sa is not None and not cpi_sa.empty:
        cpi_sa_m = cpi_sa.resample("ME").last()          # keep gap NaNs
        if len(cpi_sa_m) > 3:
            mom = (cpi_sa_m.pct_change(1) * 100).dropna()
            if len(mom):
                sig.cpi_mom_sa = _last(mom)

            saar = ((cpi_sa_m.pct_change(3) + 1) ** 4 - 1) * 100
            saar = saar.dropna()
            if len(saar):
                sig.cpi_3m_saar = _last(saar)

    sig.eff_funds = _last(eff)

    # --- SIGNAL 1: SHORT real policy rate (the repression gauge) ---
    if sig.eff_funds is not None and sig.cpi_yoy is not None:
        sig.short_real_rate = round(sig.eff_funds - sig.cpi_yoy, 2)

    # --- SIGNAL 2: LONG real yield level + momentum (duration gauge) ---
    sig.long_real_yield = _last(r10)
    sig.long_real_mom_3m = _delta(r10, 63)   # ~3 months of trading days

    sig.breakeven_10y = _last(be10)

    # --- 2s10s + its momentum ---
    if n10 is not None and n2 is not None and not n10.empty and not n2.empty:
        curve = (n10 - n2).dropna()
        sig.spread_2s10s = _last(curve)
        sig.spread_2s10s_mom_3m = _delta(curve, 63)

    # --- Credit spreads ---
    sig.hy_oas = _last(hy)
    sig.hy_oas_mom_2w = _delta(hy, 10)
    try:
        sig.credit_cycle = credit_cycle_state(hy)
    except Exception as exc:                 # optional signal; never break the regime read
        sig.credit_cycle = {"state": "UNAVAILABLE", "action": f"credit cycle failed: {exc}"}
    sig.ig_oas = _last(ig)

    # v5: dollar level + ~1-month change. The CHANGE is what matters for the
    # term-premium divergence test -- a falling dollar WHILE long real yields
    # rise is the signature; the level alone says little.
    sig.dxy = _last(dxy_s)
    if dxy_s is not None and len(dxy_s.dropna()) > 21:
        _d = dxy_s.dropna()
        _prior = float(_d.iloc[-22])
        if _prior:
            sig.dxy_20d_change_pct = round(
                (float(_d.iloc[-1]) / _prior - 1) * 100, 2)

    # --- SIGNAL: stock/bond 60d correlation (the KMLM-sizing signal) ---
    try:
        spy = fetch_prices("SPY", "1y")
        tlt = fetch_prices("TLT", "1y")
        # Coerce anything (DataFrame/MultiIndex) down to a 1-D float Series.
        spy = pd.Series(spy.squeeze() if hasattr(spy, "squeeze") else spy).astype(float)
        tlt = pd.Series(tlt.squeeze() if hasattr(tlt, "squeeze") else tlt).astype(float)
        if len(spy) > 65 and len(tlt) > 65:
            rets = pd.concat(
                [spy.pct_change().rename("spy"),
                 tlt.pct_change().rename("tlt")],
                axis=1,
            ).dropna()
            if len(rets) > 60:
                sig.stock_bond_corr_60d = round(
                    float(rets["spy"].tail(60).corr(rets["tlt"].tail(60))), 2)
    except Exception as exc:  # never let this optional signal crash the app
        sig.notes.append(f"stock/bond corr unavailable: {exc}")

    return sig


# --------------------------------------------------------------------------- #
#  Fed reaction function: hard vs soft repression
# --------------------------------------------------------------------------- #
def fed_reaction_flag(sig: SignalSet) -> dict:
    """
    Soft repression  = Fed BEHIND the curve (short real rate <= +0.25%) while
                       inflation runs above target and the long real yield is
                       positive -- inflation quietly erodes the debt.
    Restrictive      = Fed holding policy ABOVE inflation (short real > +0.25%).
                       Savers are paid; that is tightening, not repression, even
                       with inflation above target. (Sept 2026 fix: the old test
                       ignored the short real rate, so a +0.48% policy rate read
                       "SOFT repression" beside a restrictive_tightening regime.)
    Hard repression  = inflation high AND long real yield suppressed toward or
                       below zero (yield-curve control).
    Display-only: this flag never sets a weight; classify_regime() governs.
    """
    inflation_hot = (sig.cpi_yoy or 0) > FED_TARGET_INFLATION + 0.5
    long_real = sig.long_real_yield
    long_real_pos = long_real is not None and long_real > 0.5
    short = sig.short_real_rate
    restrictive = short is not None and short > 0.25

    if inflation_hot and long_real is not None and long_real < 0.25:
        state = "HARD repression (yield suppression / YCC risk)"
        detail = ("Long real yields pinned low despite hot inflation — classic "
                  "financial-repression signature. Nominal bonds bleed slowly.")
    elif restrictive:
        state = "RESTRICTIVE (policy above inflation)"
        detail = (f"Short real rate {short:+.2f}% — the Fed is holding policy above "
                  f"inflation{' even with CPI above target' if inflation_hot else ''}. "
                  "Savers are paid: this is tightening, not repression. Long duration "
                  "is NOT safe while the long real yield is rising.")
    elif inflation_hot and long_real_pos:
        state = "SOFT repression (inflation overshoot)"
        detail = ("Fed behind the curve on above-target inflation while the long end "
                  "stays positive. Long duration is NOT safe here.")
    else:
        state = "Not repressive"
        detail = "Inflation near target or real yields unremarkable."
    return {"state": state, "detail": detail}


# --------------------------------------------------------------------------- #
#  Repression Proximity Score (0-10) — NEW in v2
# --------------------------------------------------------------------------- #
def repression_score(sig: SignalSet,
                     fed_bs_expanding: Optional[bool] = None,
                     deficit_gt_5pct_gdp: Optional[bool] = None) -> dict:
    """
    The stacking score from the written framework. The quadrant says WHAT
    regime we're in; this score says HOW HARD to tilt.

      Soft real policy rate negative .......... +2
      DFII10 below 1% ......................... +2
      2s10s positive and widening ............. +1
      Fed balance sheet expanding ............. +1   (pass fed_bs_expanding)
      HY spreads tight (<350bps) .............. +1
      10y breakeven above 2.5% ................ +1
      CPI above Fed target .................... +1
      Deficit > 5% GDP ........................ +1   (pass deficit_gt_5pct_gdp)

    Bands: 8-10 peak repression | 5-7 moderate | 2-4 tightening | 0-2 anti.
    Components whose inputs are unavailable score 0 and are listed in
    'missing' so a degraded score is never mistaken for a low score.
    """
    pts, reasons, missing = 0, [], []

    # v3 FIX 9: TOP-WEIGHT tracking. These two components carry 4 of the 10
    # points. A score of 5 with 0/4 top-weight points is a materially different
    # state from a 5 that includes them, and the band label alone hides it.
    top_earned, top_total = 0, 4

    # v3.1 fix: use the BAND, not a raw sign test, for this component too.
    # classify_regime() already treats |short_real| < 0.25% as its own
    # AMBIGUOUS state and holds the regime label stable through it. This
    # scoring component did not -- a live tick from +0.10% to -0.10% (noise
    # well inside the band, nothing real changed) flipped the score from
    # 4/10 "hollow" to 6/10 "Moderate repression" while the regime banner
    # stayed at transition_ambiguous the whole time. A score meant to answer
    # "how hard to tilt" should not swing on sub-basis-point noise while the
    # regime call it is supposed to be consistent with does not move.
    _srr_band = _rb.short_real_band(sig.short_real_rate)
    if sig.short_real_rate is None:
        missing.append("short real rate")
    elif _srr_band["state"] == _rb.BAND_NEGATIVE:
        pts += 2; top_earned += 2
        reasons.append(f"Short real rate {sig.short_real_rate:+.2f}% "
                       f"(decisively negative, beyond \u00b1{_srr_band['band']:.2f}%) (+2)")
    elif _srr_band["state"] == _rb.BAND_AMBIGUOUS:
        reasons.append(f"Short real rate {sig.short_real_rate:+.2f}% is INSIDE "
                       f"the \u00b1{_srr_band['band']:.2f}% transition band (+0) "
                       f"\u2014 no point either way; this is noise, not a signal")
    else:
        reasons.append(f"Short real rate {sig.short_real_rate:+.2f}% "
                       f"(decisively positive) NOT negative (+0) \u2014 primary "
                       f"repression gauge is OFF")

    if sig.long_real_yield is None:
        missing.append("DFII10 level")
    elif sig.long_real_yield < 1.0:
        pts += 2; top_earned += 2
        reasons.append(f"DFII10 {sig.long_real_yield:.2f}% < 1% (+2)")
    else:
        reasons.append(f"DFII10 {sig.long_real_yield:.2f}% is ABOVE 1% (+0) "
                       f"\u2014 the long end is not suppressed")

    # v3 FIX 9: report the RAW momentum value either way. This component alone
    # decides whether the band prints 4 ("Tightening cycle") or 5 ("Moderate
    # repression"), and on 2026-07-29 it sat inside noise -- the curve had
    # shifted up ~35-40bp roughly in parallel, leaving 2s10s near +43bp against
    # ~+48bp three months earlier. A band flip driven by 5bp must be visible.
    if sig.spread_2s10s is None or sig.spread_2s10s_mom_3m is None:
        missing.append("2s10s / momentum")
    elif sig.spread_2s10s > 0 and sig.spread_2s10s_mom_3m > 0:
        pts += 1
        reasons.append(f"2s10s {sig.spread_2s10s:+.2f}% positive and widening "
                       f"(3m mom {sig.spread_2s10s_mom_3m:+.3f}) (+1)")
    else:
        reasons.append(f"2s10s {sig.spread_2s10s:+.2f}%, 3m momentum "
                       f"{sig.spread_2s10s_mom_3m:+.3f} \u2014 not both positive "
                       f"and widening (+0)")

    if fed_bs_expanding is None:
        missing.append("Fed balance sheet direction (pass fed_bs_expanding)")
    elif fed_bs_expanding:
        pts += 1; reasons.append("Fed balance sheet expanding (+1)")

    if sig.hy_oas is None:
        missing.append("HY OAS")
    elif sig.hy_oas < 3.5:
        pts += 1; reasons.append(f"HY OAS {sig.hy_oas:.2f}% tight (+1)")

    if sig.breakeven_10y is None:
        missing.append("10y breakeven")
    elif sig.breakeven_10y > 2.5:
        pts += 1; reasons.append(f"Breakeven {sig.breakeven_10y:.2f}% > 2.5% (+1)")

    if sig.cpi_yoy is None:
        missing.append("CPI YoY")
    elif sig.cpi_yoy > FED_TARGET_INFLATION:
        pts += 1; reasons.append(f"CPI {sig.cpi_yoy:.1f}% above target (+1)")

    if deficit_gt_5pct_gdp is None:
        missing.append("deficit vs GDP (pass deficit_gt_5pct_gdp)")
    elif deficit_gt_5pct_gdp:
        pts += 1; reasons.append("Deficit > 5% of GDP (+1)")

    band = ("Peak repression" if pts >= 8 else
            "Moderate repression" if pts >= 5 else
            "Tightening cycle" if pts >= 2 else "Anti-repression")

    hollow = (top_earned == 0)
    caveat = ""
    if hollow:
        caveat = (f"HOLLOW {band}: 0 of {top_total} top-weight points earned. "
                  f"Every point comes from second-tier components (fiscal, "
                  f"liquidity, credit, CPI level) while BOTH primary gauges "
                  f"\u2014 the sign of the short real policy rate and DFII10 "
                  f"below 1% \u2014 are off. Treat the band as an upper bound "
                  f"on the strength of the repression read.")
    elif top_earned < top_total:
        caveat = (f"PARTIAL {band}: {top_earned} of {top_total} top-weight "
                  f"points earned. Directionally supported, not confirmed.")

    # Backward compatible: score/band/reasons/missing keep their meaning; the
    # rest are additive so existing consumers are untouched.
    return {"score": pts, "band": band, "reasons": reasons, "missing": missing,
            "top_weight_earned": top_earned, "top_weight_total": top_total,
            "top_weight_display": f"{top_earned}/{top_total}",
            "hollow": hollow, "caveat": caveat}


# --------------------------------------------------------------------------- #
#  The regime classifier (5 quadrants + neutral)
# --------------------------------------------------------------------------- #
def classify_regime(sig: SignalSet, fetch_prices: Callable = None,
                    cape: float = None,
                    top20_concentration_pct: float = None,
                    growth: dict = None) -> dict:
    """Return the regime key, label, blurb, and drivers list.

    Precedence (deliberate):
      1. Liquidity crisis overrides everything (credit leads equities).
      2. Inflationary repression beats stagflation when BOTH rising long
         real yields and curve re-steepening fire — duration risk is the
         more actionable signal.
      2b. Hard repression fills the v1 gap (neg short real + falling long
         real + calm credit previously fell through to 'neutral').
      3b. v3: the short real rate is tested as a BAND, not a sign. Inside the
         band no sign-dependent regime can be confirmed, so we return
         transition_ambiguous rather than falling through to goldilocks.
      4b. v3: goldilocks additionally requires that equity leadership is not
         in a correction. Pass fetch_prices to arm that guard.
    """
    drivers = []

    hy = sig.hy_oas
    hy_rising = (sig.hy_oas_mom_2w or 0) > 0.5
    long_mom = sig.long_real_mom_3m
    short_real = sig.short_real_rate
    curve_resteep = (sig.spread_2s10s_mom_3m or 0) > 0.15

    # v3 FIX 6: BAND, not sign.
    band = _rb.short_real_band(short_real)
    short_neg = band["state"] == _rb.BAND_NEGATIVE
    short_pos = band["state"] == _rb.BAND_POSITIVE
    if band["state"] == _rb.BAND_AMBIGUOUS:
        drivers.append(band["detail"])

    # FIX 5: surface a degraded classification instead of silently treating
    # a missing DFII10 momentum as "not rising".
    if long_mom is None:
        drivers.append("⚠ DFII10 momentum unavailable — classification degraded")

    # 1) Liquidity crisis OVERRIDES everything else.
    if hy is not None and hy > 5.0 and hy_rising:
        drivers.append(f"HY OAS {hy:.2f}% and widening (> 500 bps)")
        if long_mom is not None and long_mom < 0:
            drivers.append("Long real yield falling (flight to quality)")
        return _regime("liquidity_crisis", drivers)

    # 1b) GROWTH SCARE — v4. Placed SECOND, immediately after the liquidity
    #     override and BEFORE every real-rate branch, because a confirmed
    #     contraction dominates the repression question: it does not matter
    #     whether the front end is repressing savers if the economy is
    #     shrinking. Requires `confirmed` (>= 3 of 4 growth series live) so a
    #     single surviving noisy series can never move the regime on its own.
    g_state = (growth or {}).get("state")
    g_confirmed = bool((growth or {}).get("confirmed"))
    if growth and g_confirmed and g_state == "CONTRACTING":
        # v8 (Oct 2026): PRICE MUST AGREE. The composite is monthly, lagged,
        # labour/consumer-only and scored with a downside lean, so it can
        # read CONTRACTING while output and the tape are fine (Oct 2026:
        # composite 0 while GDPNow +3.7% on capex). Acting then would cut the
        # very sectors carrying growth. Only price can veto: unwired or
        # unavailable price data ACTS on the macro read (fail toward
        # protection — cuts are never gated elsewhere either).
        _gpc = growth_price_confirmation(fetch_prices)
        _gtxt = (f"Growth composite CONTRACTING "
                 f"({(growth or {}).get('score', 0):+d}) — "
                 f"{(growth or {}).get('detail', '')[:160]}")
        if _gpc["confirms"] is False:
            drivers.append(f"⚠ {_gtxt} — but PRICE DOES NOT CONFIRM "
                           f"({_gpc['detail']}). Warning only: not acting on "
                           f"economic data the tape disagrees with.")
        else:
            drivers.append(_gtxt)
            drivers.append(_gpc["detail"])
            return _regime("growth_scare", drivers)
    if growth and not g_confirmed and g_state == "CONTRACTING":
        drivers.append("⚠ Growth reads CONTRACTING but is UNCONFIRMED "
                       "(<3 of 4 series live) — not acting on it.")

    # 2) Inflationary repression: neg short real + rising long real.
    if short_neg and long_mom is not None and long_mom > 0:
        drivers.append(f"Short real rate {short_real:+.2f}% "
                       f"(decisively negative, beyond ±{band['band']:.2f}%)")
        drivers.append("Long real yield rising (duration headwind)")
        return _regime("inflationary_repression", drivers)

    # 2b) Hard repression: neg short real + long real FALLING/suppressed,
    #     credit calm. Yield-curve-control signature. (NEW in v2 — FIX 3)
    if (short_neg and long_mom is not None and long_mom < 0
            and hy is not None and hy < 3.5):
        drivers.append(f"Short real rate {short_real:+.2f}% "
                       f"(decisively negative, beyond ±{band['band']:.2f}%)")
        drivers.append("Long real yield falling (duration suppressed/rallying)")
        return _regime("hard_repression", drivers)

    # 3) Stagflation: neg short real + growth rolling over.
    if short_neg and curve_resteep:
        drivers.append(f"Short real rate {short_real:+.2f}% "
                       f"(decisively negative, beyond ±{band['band']:.2f}%)")
        drivers.append("2s10s re-steepening from inversion (growth risk)")
        return _regime("stagflation", drivers)

    # 3b) TERM PREMIUM REPRICING -- v5. Placed BEFORE goldilocks because it
    #     is a COMPETING EXPLANATION for the same underlying state: a
    #     positive short real rate with calm credit. Goldilocks reads that
    #     as benign. This branch asks whether the LONG end and the CURRENCY
    #     agree -- and if the long end is rising while the dollar falls,
    #     they do not. Higher yields that fail to attract capital are
    #     pricing risk, not growth, and calling that "goldilocks" would
    #     instruct adding growth into a fiscal/credibility repricing.
    #
    #     Requires ALL of:
    #       * short real rate decisively POSITIVE  (not repression)
    #       * long real yield RISING               (term premium expanding)
    #       * dollar FALLING                       (the divergence itself)
    #       * credit CALM                          (else liquidity_crisis)
    #
    #     The dollar condition is what makes this specific rather than just
    #     "rates went up". Rising yields WITH a firm dollar is ordinary
    #     tightening; rising yields with a FALLING dollar is the tell.
    DXY_FALLING_PCT = -1.0     # ~1-month change; a real move, not noise
    dxy_falling = (sig.dxy_20d_change_pct is not None
                   and sig.dxy_20d_change_pct <= DXY_FALLING_PCT)
    if (short_pos and long_mom is not None and long_mom > 0
            and dxy_falling and hy is not None and hy < 3.5):
        drivers.append(f"Short real rate {short_real:+.2f}% (positive — "
                       f"savers PAID, the inverse of repression)")
        drivers.append(f"Long real yield RISING ({long_mom:+.2f}pp/3mo) "
                       f"while the dollar FELL "
                       f"{sig.dxy_20d_change_pct:+.1f}% over ~1 month")
        drivers.append("Higher yields failing to attract capital price "
                       "FISCAL/CREDIBILITY risk, not growth or inflation "
                       "expectations")
        drivers.append(f"HY OAS {hy:.2f}% — credit CALM, so this is a "
                       f"repricing, not yet a credit event")
        return _regime("term_premium_repricing", drivers)

    # 3c) RESTRICTIVE TIGHTENING — v6. Same positive-short-real, calm-credit
    #     state as goldilocks, but the long end is rising fast and the dollar
    #     is NOT falling (the falling-dollar case returned above). Placed
    #     BEFORE goldilocks because rising real yields at both ends is
    #     restrictive, not benign — goldilocks' guards (valuation, growth,
    #     leadership) were catching this state for the wrong reason.
    TIGHTENING_LONG_MOM_PP = 0.20      # 3-month rise in DFII10, pp
    dxy_not_falling = (sig.dxy_20d_change_pct is None
                       or sig.dxy_20d_change_pct > DXY_FALLING_PCT)
    if (short_pos and long_mom is not None and long_mom >= TIGHTENING_LONG_MOM_PP
            and dxy_not_falling and hy is not None and hy < 3.5):
        drivers.append(f"Short real rate {short_real:+.2f}% (decisively "
                       f"positive, beyond ±{band['band']:.2f}%) — policy is "
                       f"above inflation")
        drivers.append(f"Long real yield RISING {long_mom:+.2f}pp over 3 months "
                       f"(threshold +{TIGHTENING_LONG_MOM_PP:.2f}pp)")
        if sig.dxy_20d_change_pct is None:
            drivers.append("⚠ Dollar 20-day change unavailable — cannot rule "
                           "out term-premium repricing; treated as not falling")
        else:
            drivers.append(f"Dollar {sig.dxy_20d_change_pct:+.1f}% over ~1 month "
                           f"— firm, so higher yields ARE attracting capital "
                           f"(ordinary tightening, not a credibility repricing)")
        drivers.append(f"HY OAS {hy:.2f}% — credit calm; tightening has not "
                       f"broken anything yet")
        if growth and g_confirmed and g_state == "DETERIORATING":
            drivers.append("⚠ Growth composite DETERIORATING — tightening into "
                           "a slowing economy raises the odds of a growth scare "
                           "next; watch for CONTRACTING")
        return _regime("restrictive_tightening", drivers)

    # 4) Goldilocks: DECISIVELY positive real + tight credit + leadership
    #    intact. v3 FIX 7 adds the third condition. Without it this branch fired
    #    on 2026-07-29 and its overlay (VGT +4, QQQ +3, SMH +2) instructed
    #    adding to the exact complex that was unwinding.
    if short_pos and hy is not None and hy < 3.5:
        # v3.2 FIX 11: valuation/concentration circuit breaker. The leadership
        # guard below catches a crash IN PROGRESS; it cannot catch
        # expensive-and-euphoric. On 2026-08-07 QQQ was at record highs (guard
        # passes) with CAPE 42.19 and the top 20 names at 50.8% of index
        # weight. A cool CPI print pushing the short real rate above +0.25%
        # would have fired goldilocks and instructed ADDING growth
        # (VGT +4, QQQ +3, SMH +2) at the second-highest valuation in ~150
        # years. Fails OPEN on missing inputs — see regime_bands.valuation_ok.
        # v4: growth is a THIRD goldilocks guard, alongside valuation and
        # leadership. "Positive real rates + tight credit" cannot be a
        # CONFIRMED benign regime while labour and consumer data are
        # deteriorating together — that combination is late-cycle, not
        # goldilocks. DETERIORATING blocks here; CONTRACTING never reaches
        # this branch (it returns growth_scare above).
        if growth and g_confirmed and g_state == "DETERIORATING":
            drivers.append(f"Short real rate {short_real:+.2f}% (positive)")
            drivers.append(f"HY OAS {hy:.2f}% (tight credit)")
            drivers.append(f"Growth guard BLOCKS: composite DETERIORATING "
                           f"({(growth or {}).get('score', 0):+d}). Rates and "
                           f"credit alone say goldilocks, but labour/consumer "
                           f"data are weakening together — that is late-cycle, "
                           f"not benign.")
            return _regime("transition_ambiguous", drivers,
                           reason="growth_guard")

        val_ok, val_why = _rb.valuation_ok(cape, top20_concentration_pct)
        if not val_ok:
            drivers.append(f"Short real rate {short_real:+.2f}% (positive)")
            drivers.append(f"HY OAS {hy:.2f}% (tight credit)")
            drivers.append(val_why)
            return _regime("transition_ambiguous", drivers,
                           reason="valuation_guard")
        drivers.append(val_why)

        if fetch_prices is not None:
            lead_ok, lead_why = _rb.leadership_ok(fetch_prices)
            if not lead_ok:
                drivers.append(f"Short real rate {short_real:+.2f}% (positive)")
                drivers.append(f"HY OAS {hy:.2f}% (tight credit)")
                drivers.append(lead_why)
                return _regime("transition_ambiguous", drivers,
                               reason="leadership_guard")
            drivers.append(lead_why)
        else:
            drivers.append("⚠ Leadership guard not wired (fetch_prices=None) "
                           "— goldilocks confirmed on rates and credit only.")
        drivers.append(f"Short real rate {short_real:+.2f}% "
                       f"(decisively positive, beyond ±{band['band']:.2f}%)")
        drivers.append(f"HY OAS {hy:.2f}% (tight credit)")
        return _regime("goldilocks", drivers)

    # 4b) CREDIT STRESS -- v7. Reached only when no calm-credit regime
    #     matched. HY OAS >= 3.5% AND either widening (>= +0.25pp over ~2
    #     weeks) or simply wide (>= 4.5%). A wide-but-stable 3.5-4.5% tape
    #     still falls through: level alone is not stress.
    CREDIT_CALM_MAX_PCT = 3.5
    CREDIT_WIDEN_2W_PP = 0.25
    CREDIT_WIDE_PCT = 4.5
    hy_widening = (sig.hy_oas_mom_2w or 0) >= CREDIT_WIDEN_2W_PP
    if hy is not None and hy >= CREDIT_CALM_MAX_PCT and (hy_widening or hy >= CREDIT_WIDE_PCT):
        drivers.append(f"HY OAS {hy:.2f}% — out of the calm zone "
                       f"(>= {CREDIT_CALM_MAX_PCT:.1f}%)")
        if hy_widening:
            drivers.append(f"Spreads widening {sig.hy_oas_mom_2w:+.2f}pp over ~2 weeks "
                           f"(threshold +{CREDIT_WIDEN_2W_PP:.2f}pp)")
        if hy >= CREDIT_WIDE_PCT:
            drivers.append(f"Spread level >= {CREDIT_WIDE_PCT:.1f}% — wide regardless of trend")
        if long_mom is not None:
            drivers.append(f"Long real yield {long_mom:+.2f}pp/3mo "
                           f"({'rising — rates and credit both tightening' if long_mom > 0 else 'falling — flight to quality building'})")
        if sig.spread_2s10s_mom_3m is not None and curve_resteep:
            drivers.append(f"2s10s steepening {sig.spread_2s10s_mom_3m:+.2f}pp/3mo")
        if short_real is not None:
            drivers.append(f"Short real rate {short_real:+.2f}%")
        drivers.append("Not yet a liquidity crisis (needs HY > 5.0% AND +0.5pp/2wk)")
        return _regime("credit_stress", drivers)

    # 5) v3 FIX 6: the gauge is inside its own noise. Distinct from 'neutral',
    #    which means the signals disagree; this means the main signal is silent.
    if band["state"] == _rb.BAND_AMBIGUOUS:
        return _regime("transition_ambiguous", drivers,
                       reason="band_ambiguous")

    drivers.append("Signals mixed / transitioning")
    return _regime("neutral", drivers)


# v3.6 fix. transition_ambiguous has THREE distinct entry paths -- band
# ambiguity, the valuation guard, and the leadership guard -- but until now
# all three shared one fixed label/blurb ("...short real rate at zero..."),
# because REGIMES[key]["label"/"blurb"] is static per key. That text is
# actively WRONG when the guard routed here: on 2026-08-13 the short real
# rate was a decisively positive +0.27%, not "at zero" -- the valuation
# guard (CAPE 42, concentration 50.8%) is what redirected here, and the
# banner told a different story than the "Why" line right below it.
#
# Fix keeps the single "transition_ambiguous" KEY -- and therefore every
# downstream consumer (target_weights, the quadrant table, the repression
# score, checklist_tab) is untouched -- and only makes the DISPLAYED
# label/blurb depend on WHY this call landed here. band_ambiguous keeps the
# exact original text as the default.
_TRANSITION_VARIANTS = {
    "band_ambiguous": {
        "label": "Transition — Ambiguous (short real rate at zero)",
        "blurb": ("The short real policy rate is inside the ±0.25% "
                 "transition band, so the framework's primary gauge is not "
                 "giving a directional reading. No regime that depends on "
                 "its sign can be confirmed. Hold near base weights, take "
                 "carry at the front end with no duration risk, keep trend "
                 "on (it is agnostic to which way this resolves), and "
                 "express NEITHER the repression trade nor the reflation "
                 "trade until the gauge clears the band."),
    },
    "growth_guard": {
        "label": "Transition — Growth Guard Active",
        "blurb": ("Rates and credit alone say Goldilocks: the short real "
                 "policy rate is decisively positive and HY spreads are "
                 "tight. But the growth composite is DETERIORATING — labour "
                 "and consumer data weakening together. That combination is "
                 "late-cycle, not benign, and confirming a growth-additive "
                 "regime into it would be adding risk exactly as the "
                 "economy slows. Hold near base weights; if growth "
                 "deteriorates further this becomes a Growth Scare."),
    },
    "valuation_guard": {
        "label": "Transition — Valuation Guard Active",
        "blurb": ("Rates and credit alone say Goldilocks: the short real "
                 "policy rate is decisively positive and HY spreads are "
                 "tight. But CAPE and/or index concentration are sitting at "
                 "a historic extreme, and adding growth on top of that is "
                 "not what a confirmed benign regime should instruct. Hold "
                 "near base weights and reassess once valuation or "
                 "concentration normalizes — this is a DIFFERENT reason "
                 "than a silent short-rate gauge, see the driver list."),
    },
    "leadership_guard": {
        "label": "Transition — Leadership Guard Active",
        "blurb": ("Rates and credit alone say Goldilocks: the short real "
                 "policy rate is decisively positive and HY spreads are "
                 "tight. But the growth/momentum complex is in a meaningful "
                 "drawdown from its own recent highs — confirming a regime "
                 "whose overlay ADDS to that same complex would be adding "
                 "into an active correction. Hold near base weights until "
                 "leadership stabilizes."),
    },
}


# --------------------------------------------------------------------------- #
#  v8: price confirmation for growth_scare
# --------------------------------------------------------------------------- #
GROWTH_PC_CYCLICALS = ("XLY", "XLI", "XLF")
GROWTH_PC_DEFENSIVES = ("XLP", "XLU", "XLV")
GROWTH_PC_SPREAD_PP = -3.0      # cyclicals vs defensives, 20 sessions
GROWTH_PC_LOOKBACK = 20


def growth_price_confirmation(fetch_prices: Callable = None) -> dict:
    """Does the tape agree that growth is contracting?

    Confirms when EITHER
      - SPY closes below its 200-day average (the broad trend has broken), or
      - cyclicals (XLY/XLI/XLF) trail defensives (XLP/XLU/XLV) by 3pp or more
        over 20 sessions (the market is pricing a slowdown by rotation).
    Returns {"confirms": True|False|None, "available": bool, "detail": str}.
    confirms=None means no usable price data -> the caller acts on the macro
    read (fail toward protection) and says so.
    """
    if fetch_prices is None:
        return {"confirms": None, "available": False,
                "detail": "⚠ Price confirmation not wired (fetch_prices=None) — "
                          "acting on the growth composite alone."}

    def _px(t):
        try:
            s = fetch_prices(t, "1y")
            s = pd.Series(s).astype(float).dropna() if s is not None else pd.Series(dtype=float)
            return s
        except Exception:
            return pd.Series(dtype=float)

    parts, votes = [], []
    spy = _px("SPY")
    if len(spy) >= 200:
        ma200 = float(spy.tail(200).mean())
        last = float(spy.iloc[-1])
        below = last < ma200
        votes.append(below)
        parts.append(f"SPY {last:.2f} {'below' if below else 'above'} 200d {ma200:.2f}")

    def _ret(t):
        s = _px(t)
        if len(s) <= GROWTH_PC_LOOKBACK:
            return None
        return (float(s.iloc[-1]) / float(s.iloc[-1 - GROWTH_PC_LOOKBACK]) - 1) * 100

    cyc = [r for r in (_ret(t) for t in GROWTH_PC_CYCLICALS) if r is not None]
    dfn = [r for r in (_ret(t) for t in GROWTH_PC_DEFENSIVES) if r is not None]
    if len(cyc) >= 2 and len(dfn) >= 2:
        spread = sum(cyc) / len(cyc) - sum(dfn) / len(dfn)
        rot = spread <= GROWTH_PC_SPREAD_PP
        votes.append(rot)
        parts.append(f"cyclicals vs defensives {spread:+.1f}pp over {GROWTH_PC_LOOKBACK}d "
                     f"(confirms at ≤ {GROWTH_PC_SPREAD_PP:+.0f}pp)")

    if not votes:
        return {"confirms": None, "available": False,
                "detail": "⚠ Price confirmation unavailable (no SPY/sector data) — "
                          "acting on the growth composite alone."}
    ok = any(votes)
    return {"confirms": ok, "available": True,
            "detail": ("Price confirms: " if ok else "") + "; ".join(parts)}


def _regime(key: str, drivers: list, reason: str | None = None) -> dict:
    r = REGIMES[key]
    label, blurb = r["label"], r["blurb"]
    if key == "transition_ambiguous":
        variant = _TRANSITION_VARIANTS.get(reason or "band_ambiguous",
                                           _TRANSITION_VARIANTS["band_ambiguous"])
        label, blurb = variant["label"], variant["blurb"]
    return {"key": key, "label": label, "blurb": blurb,
            "drivers": drivers, "transition_reason": reason}


# --------------------------------------------------------------------------- #
#  Momentum gate (Level-4 entry confirmation) — NEW in v2
# --------------------------------------------------------------------------- #
def _gold_trend_ok(fetch_prices: Callable = _inline_fetch_prices) -> bool:
    """True when GLD closes above a RISING 200-day MA.
    Fail SAFE: any data problem returns False (no momentum data -> no add)."""
    try:
        px = fetch_prices("GLD", "2y")
        px = pd.Series(px.squeeze() if hasattr(px, "squeeze") else px)
        px = px.astype(float).dropna()
        if len(px) < 221:
            return False
        ma200 = px.rolling(200).mean()
        return bool(px.iloc[-1] > ma200.iloc[-1]
                    and ma200.iloc[-1] > ma200.iloc[-21])
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#  Target weights for a regime
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  v7.1 ENTRY GATE — every ADD must clear the trend test (Oct 2026)
# --------------------------------------------------------------------------- #
# The regime decides WHAT to own. The entry gate decides WHEN an add happens.
# It is not market timing: it never moves the portfolio out of the regime's
# allocation; it only refuses to BUY INTO a downtrend and parks that weight
# in T-bills until the trend confirms.
#
# Same 200-day definition as trend_filter.py (200d SMA, slope over 21
# sessions) so the two never disagree; stricter for ENTRIES than for holds,
# by design: demand confirmation to add, give room to hold.
#
#   CONFIRMED  close > RISING 200d AND close > 50d  -> full add
#   PULLBACK   close > RISING 200d, below the 50d   -> half now, rest on a
#                                                       close back above the 50d
#   BLOCKED    below the 200d, OR the 200d falling   -> no add; parked in SGOV
#              ("above a falling 200d" is a bounce      until a close above a
#               inside a downtrend, not an entry)       flat-or-rising 200d
#   UNAVAILABLE no/short data                        -> treated as BLOCKED
#
# CUTS ARE NEVER GATED. Trend confirmation is required to buy, never to sell.
ENTRY_MA_LONG = 200
ENTRY_MA_SHORT = 50
ENTRY_SLOPE_LOOKBACK = 21
ENTRY_EXEMPT = {"SGOV", "USFR", "BIL", "SHV"}   # cash: where gated weight parks
ENTRY_PARK = "SGOV"
ENTRY_FRACTION = {"CONFIRMED": 1.0, "PULLBACK": 0.5, "BLOCKED": 0.0,
                  "UNAVAILABLE": 0.0, "EXEMPT": 1.0}
_ENTRY_CACHE: dict = {}

# v7.2 CRISIS EXCEPTION FOR DURATION. In a deflationary crash yields collapse
# and TLT rallies hard — but from a long downtrend it can take weeks to get
# back above a RISING 200-day, so the standard gate would hold the crash
# hedge in T-bills through the best of the move. In a crisis state ONLY,
# the duration add releases on shorter-term confirmation:
#   close > 50-day                                 -> full add
#   real yields falling (DFII10 3m < 0) AND
#     close > 20-day                               -> half add
#   otherwise                                      -> blocked until a close > 50-day
# Crisis states: liquidity_crisis, or credit_stress while real yields fall
# (its flight-to-quality variant). Everything else uses the standard gate.
CRISIS_DURATION = {"TLT"}
CRISIS_MA_FAST = 20


def _crisis_duration(ticker: str, regime_key: Optional[str], signals) -> bool:
    if ticker not in CRISIS_DURATION or not regime_key:
        return False
    if regime_key == "liquidity_crisis":
        return True
    mom = getattr(signals, "long_real_mom_3m", None) if signals is not None else None
    return regime_key == "credit_stress" and mom is not None and mom < 0


def entry_gate(ticker: str, fetch_prices: Callable = None,
               regime_key: Optional[str] = None, signals: "SignalSet" = None) -> dict:
    """Trend state for ADDING to `ticker`. Never raises; fails closed.
    regime_key/signals only matter for the duration crisis exception."""
    if ticker in ENTRY_EXEMPT:
        return {"ticker": ticker, "state": "EXEMPT", "fraction": 1.0,
                "detail": "cash — no trend gate (gated adds park here)", "trigger": None}
    crisis = _crisis_duration(ticker, regime_key, signals)
    if crisis:
        return _crisis_gate(ticker, fetch_prices, signals)
    fp = fetch_prices or _inline_fetch_prices
    # Cache only the built-in downloader (one fetch per ticker per day per
    # process). A caller-supplied fetcher memoises itself; keying on its id()
    # is unsafe because short-lived functions reuse addresses.
    key = (ticker, _dt.date.today()) if fetch_prices is None else None
    if key is not None and key in _ENTRY_CACHE:
        return dict(_ENTRY_CACHE[key])
    out = {"ticker": ticker, "state": "UNAVAILABLE", "fraction": 0.0, "close": None,
           "ma50": None, "ma200": None, "ma200_rising": None,
           "detail": "no price data — failing closed (no add)",
           "trigger": "price data available and a close above a rising 200-day"}
    try:
        px = fp(ticker, "2y")
        px = pd.Series(px.squeeze() if hasattr(px, "squeeze") else px).astype(float).dropna()
        need = ENTRY_MA_LONG + ENTRY_SLOPE_LOOKBACK
        if len(px) >= need:
            ma200 = px.rolling(ENTRY_MA_LONG).mean()
            ma50 = px.rolling(ENTRY_MA_SHORT).mean()
            c, m2, m5 = float(px.iloc[-1]), float(ma200.iloc[-1]), float(ma50.iloc[-1])
            rising = m2 >= float(ma200.iloc[-1 - ENTRY_SLOPE_LOOKBACK])
            gap2, gap5 = (c / m2 - 1) * 100, (c / m5 - 1) * 100
            out.update(close=round(c, 2), ma50=round(m5, 2), ma200=round(m2, 2),
                       ma200_rising=rising)
            if c > m2 and rising and c > m5:
                out.update(state="CONFIRMED", fraction=1.0, trigger=None,
                           detail=f"{gap2:+.1f}% vs rising 200d, {gap5:+.1f}% vs 50d — full add")
            elif c > m2 and rising:
                out.update(state="PULLBACK", fraction=0.5,
                           trigger=f"close back above the 50-day ({m5:,.2f})",
                           detail=f"{gap2:+.1f}% above a rising 200d but {gap5:+.1f}% below the 50d "
                                  "— pullback in an uptrend: half now")
            else:
                why = (f"{gap2:+.1f}% vs a {'rising' if rising else 'FALLING'} 200d "
                       f"({m2:,.2f})")
                out.update(state="BLOCKED", fraction=0.0,
                           trigger=f"close above a flat-or-rising 200-day ({m2:,.2f})",
                           detail=why + (" — bounce inside a downtrend" if c > m2 else " — downtrend")
                                  + "; add parked in SGOV")
        else:
            out["detail"] = f"only {len(px)} sessions (need {need}) — failing closed (no add)"
    except Exception as e:
        out["detail"] = f"trend check failed ({type(e).__name__}) — failing closed (no add)"
    if key is not None:
        _ENTRY_CACHE[key] = dict(out)
    return out


def _crisis_gate(ticker: str, fetch_prices: Callable = None, signals=None) -> dict:
    fp = fetch_prices or _inline_fetch_prices
    out = {"ticker": ticker, "state": "UNAVAILABLE", "fraction": 0.0, "close": None,
           "ma20": None, "ma50": None, "crisis_exception": True,
           "detail": "crisis exception: no price data — failing closed (no add)",
           "trigger": "a close above the 50-day"}
    try:
        px = fp(ticker, "2y")
        px = pd.Series(px.squeeze() if hasattr(px, "squeeze") else px).astype(float).dropna()
        if len(px) >= ENTRY_MA_SHORT:
            c = float(px.iloc[-1])
            m50 = float(px.rolling(ENTRY_MA_SHORT).mean().iloc[-1])
            m20 = float(px.rolling(CRISIS_MA_FAST).mean().iloc[-1])
            mom = getattr(signals, "long_real_mom_3m", None) if signals is not None else None
            out.update(close=round(c, 2), ma20=round(m20, 2), ma50=round(m50, 2))
            if c > m50:
                out.update(state="CONFIRMED", fraction=1.0, trigger=None,
                           detail=f"crisis exception: {(c / m50 - 1) * 100:+.1f}% above the 50-day — full add")
            elif mom is not None and mom < 0 and c > m20:
                out.update(state="PULLBACK", fraction=0.5,
                           trigger=f"a close above the 50-day ({m50:,.2f})",
                           detail=f"crisis exception: real yields falling ({mom:+.2f}pp/3mo) and above the "
                                  f"20-day — half now, rest above the 50-day")
            else:
                out.update(state="BLOCKED", fraction=0.0,
                           trigger=f"a close above the 50-day ({m50:,.2f})",
                           detail=f"crisis exception: {(c / m50 - 1) * 100:+.1f}% below the 50-day "
                                  "— still falling; add parked in SGOV")
        else:
            out["detail"] = f"crisis exception: only {len(px)} sessions — failing closed (no add)"
    except Exception as e:
        out["detail"] = f"crisis exception: check failed ({type(e).__name__}) — failing closed"
    return out


def regime_overlay(regime_key: str, signals: "SignalSet" = None) -> dict:
    """The overlay for a regime, resolving state-dependent variants.
    credit_stress uses overlay_flight only when the 10y real yield's 3-month
    change is known AND negative; anything else gets the defensive default."""
    r = REGIMES[regime_key]
    if (regime_key == "credit_stress" and signals is not None
            and getattr(signals, "long_real_mom_3m", None) is not None
            and signals.long_real_mom_3m < 0 and "overlay_flight" in r):
        return dict(r["overlay_flight"])
    return dict(r["overlay"])


def target_weights(regime_key: str,
                   fetch_prices: Callable = None,
                   signals: "SignalSet" = None,
                   gate: bool = True,
                   detail: bool = False):
    """Apply the regime overlay to the base sleeve and renormalize to 100%.

    v7.1: the old gold-only momentum gate is generalised — EVERY positive
    tilt (an add vs base) must clear entry_gate(); a BLOCKED add parks in
    SGOV, a PULLBACK add goes in at half with the rest parked. Negative
    tilts are never gated. gate=False returns the raw regime allocation
    (reference tables). detail=True returns (weights, gates).

    Backward compatible: target_weights('goldilocks') etc. work unchanged.
    """
    w = dict(BASE_WEIGHTS)
    overlay = regime_overlay(regime_key, signals)
    gates = {}
    if gate:
        for t, d in list(overlay.items()):
            if d <= 0 or t in ENTRY_EXEMPT:
                continue
            g = entry_gate(t, fetch_prices, regime_key, signals)
            gates[t] = g
            keep = round(d * g["fraction"], 2)
            if keep < d:
                overlay[ENTRY_PARK] = overlay.get(ENTRY_PARK, 0) + (d - keep)
                overlay[t] = keep
    for t, d in overlay.items():
        w[t] = max(0, w.get(t, 0) + d)
    total = sum(w.values())
    out = w if total <= 0 else {t: round(v * 100 / total, 1) for t, v in w.items()}
    return (out, gates) if detail else out


HEDGE_SLEEVES = {"KMLM"}


def transition_plan(from_key: str, to_key: str, fetch_prices: Callable = None,
                    signals: "SignalSet" = None, current: dict = None) -> dict:
    """The exact trades to move from one allocation to another, in protocol
    order (cuts, hedges, adds), with EVERY add re-checked against
    entry_gate() relative to where you are now — this catches weight that
    comes back through BASE (e.g. TLT 2% -> 10%), which a gate on the
    overlay alone cannot see.

    `current` = actual weights (e.g. from the ledger); defaults to
    from_key's own gated targets."""
    frm = dict(current) if current else target_weights(from_key, fetch_prices, signals)
    to = target_weights(to_key, fetch_prices, signals)
    rows, parked = [], 0.0
    for t in sorted(set(frm) | set(to)):
        a, b = float(frm.get(t, 0)), float(to.get(t, 0))
        d = round(b - a, 1)
        if abs(d) < 0.05:
            continue
        if d < 0:
            rows.append({"ticker": t, "from": a, "to": b, "change": d, "step": "1 cut",
                         "now": d, "pending": 0.0, "gate": "—", "trigger": None,
                         "detail": "cuts are never gated"})
            continue
        step = "2 hedge" if t in HEDGE_SLEEVES else "3 add"
        g = entry_gate(t, fetch_prices, to_key, signals)
        now = round(d * g["fraction"], 1)
        pend = round(d - now, 1)
        if t != ENTRY_PARK:
            parked += pend
        rows.append({"ticker": t, "from": a, "to": b, "change": d, "step": step,
                     "now": now, "pending": pend, "gate": g["state"],
                     "trigger": g.get("trigger"), "detail": g["detail"]})
    if parked > 0:
        for r in rows:
            if r["ticker"] == ENTRY_PARK:
                # net the parked weight against SGOV's own planned change
                r["now"] = round(r["change"] + parked, 1)
                r["pending"] = round(-parked, 1)     # released as the gated adds clear
                r["step"] = "3 add" if r["now"] > 0 else "1 cut"
                r["trigger"] = "drawn down as each blocked/half add clears its trigger"
                r["detail"] = f"planned {r['change']:+.1f}, +{parked:.1f}pt parked from gated adds"
                break
        else:
            rows.append({"ticker": ENTRY_PARK, "from": float(frm.get(ENTRY_PARK, 0)),
                         "to": float(to.get(ENTRY_PARK, 0)), "change": 0.0, "step": "3 add",
                         "now": round(parked, 1), "pending": round(-parked, 1), "gate": "EXEMPT",
                         "trigger": "drawn down as each blocked/half add clears its trigger",
                         "detail": f"+{parked:.1f}pt parked from gated adds"})
    rows.sort(key=lambda r: (r["step"], r["change"]))
    blocked = [r["ticker"] for r in rows if r["gate"] in ("BLOCKED", "UNAVAILABLE")]
    half = [r["ticker"] for r in rows if r["gate"] == "PULLBACK"]
    summary = (f"{from_key} -> {to_key}: {sum(1 for r in rows if r['change'] < 0)} cuts, "
               f"{sum(1 for r in rows if r['change'] > 0)} adds"
               + (f"; BLOCKED by trend (parked in {ENTRY_PARK}): {', '.join(blocked)}" if blocked else "")
               + (f"; half-size on pullback: {', '.join(half)}" if half else "")
               + (f"; {parked:.1f}pt parked" if parked > 0 else ""))
    return {"from": from_key, "to": to_key, "rows": rows, "parked": round(parked, 1),
            "blocked": blocked, "half": half, "summary": summary}


# --------------------------------------------------------------------------- #
#  KMLM sizing signal (explicit, for the portfolio app)
# --------------------------------------------------------------------------- #
def kmlm_signal(sig: SignalSet) -> dict:
    """
    Trend-following (KMLM) wants sustained cross-asset trends, especially
    inflationary ones. Its single best 'own more of me' tell is the stock/bond
    correlation flipping POSITIVE (60/40 breaks). Choppy/mean-reverting tape and
    V-reversals are its enemy.
    """
    score = 0
    reasons = []
    corr = sig.stock_bond_corr_60d
    if corr is not None:
        if corr > 0.2:
            score += 2
            reasons.append(f"Stock/bond corr {corr:+.2f} POSITIVE — 60/40 "
                           "breaking, trend earns its keep (INCREASE)")
        elif corr < -0.3:
            score -= 1
            reasons.append(f"Stock/bond corr {corr:+.2f} strongly negative — "
                           "diversification working, less need for trend")

    # v3 FIX 6 consumer: this was a FOURTH site using the bare sign test. It
    # now uses the same band, so KMLM sizing and the regime label can no longer
    # disagree about whether the short real rate is negative.
    if sig.cpi_yoy is not None and sig.short_real_rate is not None:
        if (sig.cpi_yoy > FED_TARGET_INFLATION
                and _rb.is_negative(sig.short_real_rate)):
            score += 1
            reasons.append("Inflation above target with decisively negative "
                           "short real rate \u2014 inflationary trend backdrop "
                           "(INCREASE)")
        elif (sig.cpi_yoy > FED_TARGET_INFLATION
                and _rb.is_ambiguous(sig.short_real_rate)):
            reasons.append(f"Inflation above target but the short real rate "
                           f"({sig.short_real_rate:+.2f}%) is inside the "
                           f"\u00b1{_rb.TRANSITION_BAND:.2f}% band \u2014 no "
                           f"inflationary-backdrop point awarded")

    if (sig.long_real_mom_3m or 0) > 0:
        score += 1
        reasons.append("Long real yields rising (bond downtrend) — trend "
                       "tailwind (INCREASE)")

    if score >= 3:
        stance = "INCREASE KMLM"
        funding = ("Fund from CASH first (SGOV — it bleeds negative real "
                   "return; keep USFR if hike risk is live), then "
                   "rate-sensitive growth (SMH/QQQ). Do NOT sell "
                   "metals/energy in this regime.")
    elif score <= 0:
        stance = "REDUCE KMLM"
        funding = ("Rotate proceeds back to growth (VGT/QQQ) or cash. Trend is "
                   "prone to whipsaw in this tape.")
    else:
        stance = "HOLD KMLM"
        funding = "No change warranted yet."

    return {"stance": stance, "score": score, "reasons": reasons,
            "funding": funding}


# --------------------------------------------------------------------------- #
#  One-call convenience for either app
# --------------------------------------------------------------------------- #
def full_assessment(fred_api_key: str = "",
                    fed_bs_expanding: Optional[bool] = None,
                    deficit_gt_5pct_gdp: Optional[bool] = None,
                    cape: Optional[float] = None,
                    top20_concentration_pct: Optional[float] = None,
                    growth: Optional[dict] = None,
                    **kw) -> dict:
    """
    v3 FIX 10. Two changes, both of which were silently degrading the output:

    1. fed_bs_expanding and deficit_gt_5pct_gdp are now FORWARDED to
       repression_score(). Previously they were never passed, so both always
       landed in missing[] -- the live score was structurally capped at 8/10
       and permanently reported as incomplete. Both are manual/derived flags
       the caller has to supply; as of July 2026 the Fed balance sheet is
       expanding (~+$150bn since January via reserve-management bill purchases)
       and the FY2026 deficit is 5.8% of GDP, so the correct call is
       full_assessment(key, fed_bs_expanding=True, deficit_gt_5pct_gdp=True).

    2. fetch_prices is forwarded to classify_regime() so the v3 leadership
       guard is armed. Without it the guard stays dormant and emits a visible
       "not wired" driver rather than silently passing.
    """
    fetch_prices = kw.get("fetch_prices")
    sig = compute_signals(fred_api_key, **kw)
    regime = classify_regime(sig, fetch_prices=fetch_prices, cape=cape,
                             top20_concentration_pct=top20_concentration_pct,
                             growth=growth)
    _tw = target_weights(regime["key"], fetch_prices=fetch_prices, signals=sig, detail=True)
    return {
        "signals": sig,
        "regime": regime,
        "fed": fed_reaction_flag(sig),
        "targets": _tw[0],
        "target_gates": _tw[1],
        "kmlm": kmlm_signal(sig),
        "credit_cycle": sig.credit_cycle or {"state": "UNAVAILABLE",
                                             "action": _CC_ACTION["UNAVAILABLE"]},
        "repression": repression_score(
            sig,
            fed_bs_expanding=fed_bs_expanding,
            deficit_gt_5pct_gdp=deficit_gt_5pct_gdp),
    }


# --------------------------------------------------------------------------- #
#  v7 — CREDIT CYCLE / SPREAD-PEAK RE-ENTRY RULE (Oct 2026)
# --------------------------------------------------------------------------- #
# The regime branches catch credit ESCALATION (credit_stress, liquidity_crisis)
# but nothing said when the episode is OVER. The profitable trade in a credit
# cycle is buying cyclical / deep value and small caps after spreads PEAK and
# turn -- not while they widen. This is a separate state machine on the HY OAS
# history; it never changes the regime key or the overlay weights. It gates
# which ADDS are eligible at the weekend review.
#
#   NO_EPISODE        HY never reached the stress line in the lookback
#   WIDENING          at/near the episode high, or the high is < 10 sessions old
#   PEAK_FORMING      off the high, but not enough retrace or still re-widening
#   RE_ENTRY_PENDING  peak-and-turn conditions met on 1 close
#   RE_ENTRY          met on 2 consecutive closes (the two-close protocol)
#
# Peak-and-turn = ALL of: episode high >= 3.5% in the last ~6 months, high is
# >= 10 sessions old, spreads have retraced >= max(0.50pp, 25% of the
# trough-to-peak widening), and the 2-week change is negative.
CREDIT_EPISODE_MIN_PCT = 3.5
CREDIT_LOOKBACK = 126           # ~6 months of sessions
CREDIT_PEAK_AGE_MIN = 10
CREDIT_RETRACE_MIN_PP = 0.50
CREDIT_RETRACE_FRAC = 0.25
CREDIT_NEAR_PEAK_PP = 0.10

_CC_ACTION = {
    "NO_EPISODE": "No credit episode in the last ~6 months — the re-entry rule is inactive.",
    "WIDENING": ("Spreads at or near the episode high. Do NOT add cyclical/deep value, small caps "
                 "or credit-sensitive equity yet; quality value only."),
    "PEAK_FORMING": ("Off the high but the turn is not confirmed. Watch; no cyclical adds."),
    "RE_ENTRY_PENDING": ("Peak-and-turn met on 1 close. Confirm on the next close before acting."),
    "RE_ENTRY": ("Spread peak confirmed. At the weekend review cyclical/deep value and small caps "
                 "become ELIGIBLE (each still needs Level 2 flow and a Level 4 entry); re-evaluate "
                 "duration bought for the bust as spreads normalise."),
    "UNAVAILABLE": "HY OAS history unavailable — re-entry rule cannot be evaluated.",
}


def _cc_core(s: pd.Series) -> dict:
    w = s.tail(CREDIT_LOOKBACK)
    vals = w.to_numpy(dtype=float)
    pos = int(vals.argmax())
    peak = float(vals[pos])
    trough = float(vals[: pos + 1].min())
    cur = float(vals[-1])
    age = len(vals) - 1 - pos
    mom = float(s.iloc[-1] - s.iloc[-11]) if len(s) > 10 else None
    need = max(CREDIT_RETRACE_MIN_PP, CREDIT_RETRACE_FRAC * (peak - trough))
    off = peak - cur
    if peak < CREDIT_EPISODE_MIN_PCT:
        state = "NO_EPISODE"
    elif off <= CREDIT_NEAR_PEAK_PP or age < CREDIT_PEAK_AGE_MIN:
        state = "WIDENING"
    elif off >= need and mom is not None and mom < 0:
        state = "TURN"
    else:
        state = "PEAK_FORMING"
    return {"state": state, "peak": round(peak, 2), "peak_date": str(w.index[pos])[:10],
            "peak_age": age, "trough": round(trough, 2), "current": round(cur, 2),
            "retrace_needed": round(need, 2), "off_peak": round(off, 2),
            "mom_2w": None if mom is None else round(mom, 2)}


def credit_cycle_state(hy: Optional[pd.Series]) -> dict:
    """Spread-peak re-entry state from the HY OAS history (percent units)."""
    s = None if hy is None else pd.Series(hy).dropna().astype(float)
    if s is None or len(s) < 30:
        return {"state": "UNAVAILABLE", "action": _CC_ACTION["UNAVAILABLE"]}
    today = _cc_core(s)
    if today["state"] == "TURN":
        prev = _cc_core(s.iloc[:-1])
        today["state"] = "RE_ENTRY" if prev["state"] == "TURN" else "RE_ENTRY_PENDING"
    today["action"] = _CC_ACTION[today["state"]]
    if today["state"] in ("PEAK_FORMING",):
        today["action"] += (f" Off the {today['peak']:.2f}% high by {today['off_peak']:.2f}pp of "
                            f"{today['retrace_needed']:.2f}pp needed; 2-week change "
                            f"{today['mom_2w']:+.2f}pp.") if today["mom_2w"] is not None else ""
    return today



# --------------------------------------------------------------------------- #
#  Selftest (offline) — v6, Sept 2026
# --------------------------------------------------------------------------- #
def selftest() -> dict:
    """Overlay invariants plus the branch boundaries around restrictive_tightening."""
    fails = []
    for k, r in REGIMES.items():
        for ok_ in ("overlay", "overlay_flight"):
            if ok_ not in r:
                continue
            s = sum((r.get(ok_) or {}).values())
            if s != 0:
                fails.append(f"{k} {ok_} sums to {s}, must be 0")
            unknown = set(r.get(ok_) or {}) - set(BASE_WEIGHTS)
            if unknown:
                fails.append(f"{k} {ok_} uses tickers not in BASE_WEIGHTS: {unknown}")

    # Deterministic price fixtures for the entry gate (no network in tests).
    import numpy as _np
    _n = 300
    _up = pd.Series(_np.linspace(100, 160, _n))                        # rising: CONFIRMED
    _pull = pd.Series(_np.r_[_np.linspace(100, 160, _n - 15),
                             _np.linspace(160, 150, 15)])              # dip under 50d: PULLBACK
    _down = pd.Series(_np.linspace(160, 100, _n))                      # falling: BLOCKED
    _bounce = pd.Series(_np.r_[_np.linspace(200, 100, _n - 10),
                               _np.linspace(100, 175, 10)])            # above a FALLING 200d: BLOCKED
    _fx = lambda series: (lambda t, p="2y": series)
    FP_UP, FP_DOWN = _fx(_up), _fx(_down)

    def sig(**kw):
        base = dict(short_real_rate=0.48, long_real_yield=2.85, long_real_mom_3m=0.66,
                    hy_oas=2.80, hy_oas_mom_2w=0.10, spread_2s10s=0.31,
                    spread_2s10s_mom_3m=0.0, dxy_20d_change_pct=2.1)
        base.update(kw)
        return SignalSet(**base)

    def key(s, **kw):
        return classify_regime(s, fetch_prices=None, **kw)["key"]

    # 2026-09-26 live inputs -> restrictive_tightening (was transition_ambiguous)
    if key(sig(), cape=41.48, top20_concentration_pct=49.87) != "restrictive_tightening":
        fails.append("today's inputs must classify as restrictive_tightening")
    # dollar FALLING with the same rates -> term premium repricing wins
    if key(sig(dxy_20d_change_pct=-1.5)) != "term_premium_repricing":
        fails.append("falling dollar must still route to term_premium_repricing")
    # long end NOT rising fast -> falls through to the goldilocks family
    if key(sig(long_real_mom_3m=0.10), cape=41.48, top20_concentration_pct=49.87) != "transition_ambiguous":
        fails.append("slow long-end rise with CAPE > 40 must stay in the valuation-guarded transition")
    if key(sig(long_real_mom_3m=0.10), cape=30.0, top20_concentration_pct=35.0) != "goldilocks":
        fails.append("slow long-end rise with valuation OK should reach goldilocks (leadership guard unwired)")
    # credit stress overrides
    if key(sig(hy_oas=5.5, hy_oas_mom_2w=0.8)) != "liquidity_crisis":
        fails.append("HY crisis must override tightening")
    # short real inside the band -> not tightening
    if key(sig(short_real_rate=0.10)) == "restrictive_tightening":
        fails.append("short real inside ±0.25% band must not confirm tightening")
    # missing dollar data -> still tightening, with a named caveat
    r = classify_regime(sig(dxy_20d_change_pct=None))
    if r["key"] != "restrictive_tightening" or not any("unavailable" in d for d in r["drivers"]):
        fails.append("missing DXY must classify tightening and say so in drivers")
    # growth contraction dominates
    if key(sig(), growth={"state": "CONTRACTING", "confirmed": True, "score": -4, "detail": "t"}) != "growth_scare":
        fails.append("confirmed contraction must override tightening")
    # v8: price must agree before growth_scare acts
    _G4 = {"state": "CONTRACTING", "confirmed": True, "score": -4, "detail": "t"}
    _idx = pd.bdate_range("2025-01-01", periods=260)

    def _fp_maker(spy_trend, cyc_20d, dfn_20d):
        def _fp(t, period="1y"):
            if t == "SPY":
                return pd.Series(np.linspace(100, 100 * (1 + spy_trend), 260), index=_idx)
            r = cyc_20d if t in GROWTH_PC_CYCLICALS else dfn_20d if t in GROWTH_PC_DEFENSIVES else 0.0
            base = np.full(260, 100.0)
            base[-21:] = np.linspace(100, 100 * (1 + r / 100), 21)
            return pd.Series(base, index=_idx)
        return _fp
    _tape_ok = _fp_maker(0.20, 1.0, 0.5)          # uptrend, cyclicals leading
    _tape_rot = _fp_maker(0.20, -3.0, 1.0)        # uptrend but -4pp rotation
    _tape_brk = _fp_maker(-0.20, 0.0, 0.0)        # SPY below 200d
    if classify_regime(sig(), fetch_prices=_tape_ok, growth=_G4)["key"] == "growth_scare":
        fails.append("contraction with a healthy tape must NOT act (warning only)")
    _r = classify_regime(sig(), fetch_prices=_tape_ok, growth=_G4)
    if not any("PRICE DOES NOT CONFIRM" in d for d in _r["drivers"]):
        fails.append("unconfirmed contraction must be surfaced as a warning driver")
    if classify_regime(sig(), fetch_prices=_tape_rot, growth=_G4)["key"] != "growth_scare":
        fails.append("defensive rotation of 4pp must confirm growth_scare")
    if classify_regime(sig(), fetch_prices=_tape_brk, growth=_G4)["key"] != "growth_scare":
        fails.append("SPY below 200d must confirm growth_scare")
    if classify_regime(sig(), fetch_prices=lambda t, p="1y": pd.Series(dtype=float), growth=_G4)["key"] != "growth_scare":
        fails.append("unavailable price data must fail toward protection (act)")
    if classify_regime(sig(hy_oas=3.9, hy_oas_mom_2w=0.4), fetch_prices=_tape_ok, growth=_G4)["key"] != "credit_stress":
        fails.append("vetoed growth_scare must fall through to credit_stress")

    # v7: credit stress fills the 3.5%..crisis hole (was: neutral)
    if key(sig(hy_oas=3.8, hy_oas_mom_2w=0.40)) != "credit_stress":
        fails.append("HY 3.8% widening +0.40pp must be credit_stress, not neutral")
    if key(sig(hy_oas=3.6, hy_oas_mom_2w=0.05), cape=30.0, top20_concentration_pct=35.0) != "neutral":
        fails.append("HY 3.6% flat is elevated but not stress: neutral")
    if key(sig(hy_oas=4.7, hy_oas_mom_2w=0.0)) != "credit_stress":
        fails.append("HY 4.7% (wide) must be credit_stress even if not widening")
    if key(sig(hy_oas=5.2, hy_oas_mom_2w=0.2)) != "credit_stress":
        fails.append("HY 5.2% but only +0.2pp/2wk misses the crisis override -> credit_stress")
    if key(sig(hy_oas=3.4, hy_oas_mom_2w=0.40)) != "restrictive_tightening":
        fails.append("HY 3.4% is still the calm zone")
    if key(sig(short_real_rate=0.10, hy_oas=3.9, hy_oas_mom_2w=0.4)) != "credit_stress":
        fails.append("credit stress must beat the ambiguity band")
    if key(sig(hy_oas=3.9, hy_oas_mom_2w=0.4), growth={"state": "CONTRACTING", "confirmed": True, "score": -4, "detail": "t"}) != "growth_scare":
        fails.append("confirmed contraction still outranks credit stress")
    if key(sig(short_real_rate=-0.6, hy_oas=3.9, hy_oas_mom_2w=0.4)) != "inflationary_repression":
        fails.append("negative short real + rising long real keeps repression precedence")
    _cs = target_weights("credit_stress", FP_UP)
    if abs(sum(_cs.values()) - 100) > 0.5 or _cs.get("SGOV", 0) <= target_weights("neutral", FP_UP).get("SGOV", 0):
        fails.append(f"credit_stress targets must sum to 100 and lift SGOV: {_cs}")

    # v7: spread-peak re-entry state machine
    import numpy as _np
    _ix = pd.bdate_range("2026-01-01", periods=200)
    def _hy(path):
        return pd.Series(_np.interp(_np.arange(200), [p[0] for p in path], [p[1] for p in path]), index=_ix)
    _cs = lambda path: credit_cycle_state(_hy(path))["state"]
    if _cs([(0, 2.7), (199, 2.95)]) != "NO_EPISODE":
        fails.append("calm HY must be NO_EPISODE")
    if _cs([(0, 2.7), (190, 4.2), (199, 4.25)]) != "WIDENING":
        fails.append("spreads at their high must be WIDENING")
    if _cs([(0, 2.7), (150, 4.5), (199, 4.2)]) != "PEAK_FORMING":
        fails.append("0.30pp off a 4.5% high (need 0.50) must be PEAK_FORMING")
    if _cs([(0, 2.7), (140, 5.0), (199, 4.1)]) != "RE_ENTRY":
        fails.append("0.9pp off a 5.0% high, falling, 2 closes must be RE_ENTRY")
    _p = _hy([(0, 2.7), (140, 5.0), (197, 4.5), (198, 4.5), (199, 4.35)])
    _p.iloc[-2] = 4.62      # yesterday failed the retrace test; today passes
    if credit_cycle_state(_p)["state"] != "RE_ENTRY_PENDING":
        fails.append(f"first qualifying close must be RE_ENTRY_PENDING: {credit_cycle_state(_p)}")
    if _cs([(0, 2.7), (140, 5.0), (180, 4.0), (199, 4.3)]) != "PEAK_FORMING":
        fails.append("re-widening after a retrace (2w change > 0) must not be RE_ENTRY")
    if credit_cycle_state(None)["state"] != "UNAVAILABLE":
        fails.append("missing HY history must be UNAVAILABLE")

    # v7.1: entry gate states
    _gs = {n: entry_gate("XLV", _fx(sr))["state"] for n, sr in
           (("up", _up), ("pull", _pull), ("down", _down), ("bounce", _bounce),
            ("short", pd.Series(_np.linspace(1, 2, 50))))}
    if _gs != {"up": "CONFIRMED", "pull": "PULLBACK", "down": "BLOCKED",
               "bounce": "BLOCKED", "short": "UNAVAILABLE"}:
        fails.append(f"entry gate states wrong: {_gs}")
    if entry_gate("SGOV", FP_DOWN)["state"] != "EXEMPT":
        fails.append("cash must be exempt from the entry gate")
    # v7.1: credit_stress is escalation-safe vs restrictive_tightening
    _rt = target_weights("restrictive_tightening", FP_UP)
    _cs = target_weights("credit_stress", FP_UP, sig())          # real yields rising
    _exp = {"VGT": 14, "SMH": 2, "QQQ": 1, "GLD": 10, "SLV": 4, "RING": 4, "XLE": 4,
            "PDBC": 2, "SCHD": 16, "XLV": 7, "XLU": 1, "SGOV": 15, "USFR": 9, "TLT": 2, "KMLM": 9}
    if any(abs(_cs[t] - v) > 0.05 for t, v in _exp.items()):
        fails.append(f"credit_stress (rising real yields) targets wrong: {_cs}")
    for t in ("TLT", "XLU", "GLD", "SLV"):
        if _cs[t] > _rt[t]:
            fails.append(f"escalation must never ADD {t}: {_rt[t]} -> {_cs[t]}")
    if (_cs["SGOV"] + _cs["USFR"]) < (_rt["SGOV"] + _rt["USFR"]) or _cs["KMLM"] < _rt["KMLM"]:
        fails.append("escalation must not cut cash or trend")
    if regime_overlay("credit_stress", sig(long_real_mom_3m=-0.3)) != REGIMES["credit_stress"]["overlay_flight"]:
        fails.append("falling real yields must select the flight-to-quality overlay")
    if regime_overlay("credit_stress", sig(long_real_mom_3m=None)) != REGIMES["credit_stress"]["overlay"]:
        fails.append("unknown real-yield direction must default to the defensive overlay")
    # gated adds park in SGOV; cuts never gated
    _g = target_weights("restrictive_tightening", FP_DOWN)
    if _g["KMLM"] > BASE_WEIGHTS["KMLM"] + 0.05 or _g["SGOV"] <= _rt["SGOV"]:
        fails.append(f"blocked adds must park in SGOV: {_g}")
    if _g["TLT"] != _rt["TLT"] or _g["VGT"] != _rt["VGT"]:
        fails.append("cuts must not change when the gate blocks")
    _raw = target_weights("restrictive_tightening", FP_DOWN, gate=False)
    if _raw != _rt:
        fails.append("gate=False must return the raw regime allocation")
    # transition plan: re-adds through BASE are gated too
    _tp = transition_plan("restrictive_tightening", "credit_stress", FP_DOWN, sig(long_real_mom_3m=-0.3))
    _tlt = next((r for r in _tp["rows"] if r["ticker"] == "TLT"), None)
    if _tlt is None or _tlt["now"] != 0 or _tlt["pending"] <= 0 or "TLT" not in _tp["blocked"]:
        fails.append(f"TLT re-add via base must be BLOCKED in a downtrend: {_tlt}")
    _sg = next((r for r in _tp["rows"] if r["ticker"] == "SGOV"), None)
    if _sg is None or abs(_sg["now"] - (_sg["change"] + _tp["parked"])) > 0.05:
        fails.append(f"blocked weight must be parked in SGOV: {_sg} parked {_tp['parked']}")
    if [r["step"] for r in _tp["rows"]] != sorted(r["step"] for r in _tp["rows"]):
        fails.append("plan must be ordered cuts, hedges, adds")
    _tp2 = transition_plan("restrictive_tightening", "credit_stress", FP_UP, sig())
    if any(r["ticker"] in ("TLT", "XLU", "GLD") and r["change"] > 0 for r in _tp2["rows"]):
        fails.append(f"RT -> credit_stress (rising) must not buy TLT/XLU/GLD: {_tp2['summary']}")

    # v7.2: duration crisis exception
    _tlt_rebound = pd.Series(_np.r_[_np.linspace(200, 100, _n - 30), _np.linspace(100, 125, 30)])  # below falling 200d, above 50d
    _tlt_turning = pd.Series(_np.r_[_np.linspace(200, 100, _n - 8), _np.linspace(100, 104, 8)])    # above 20d, below 50d
    if entry_gate("TLT", _fx(_tlt_rebound))["state"] != "BLOCKED":
        fails.append("outside a crisis TLT below a falling 200d must stay BLOCKED")
    if entry_gate("TLT", _fx(_tlt_rebound), "liquidity_crisis")["state"] != "CONFIRMED":
        fails.append("liquidity_crisis: TLT above its 50-day must release in full")
    if entry_gate("TLT", _fx(_tlt_turning), "liquidity_crisis", sig(long_real_mom_3m=-0.4))["state"] != "PULLBACK":
        fails.append("liquidity_crisis: falling real yields + above 20d must release half")
    if entry_gate("TLT", _fx(_tlt_turning), "liquidity_crisis", sig(long_real_mom_3m=0.3))["state"] != "BLOCKED":
        fails.append("liquidity_crisis: below 50d with real yields RISING must stay blocked")
    if entry_gate("TLT", _fx(_tlt_rebound), "credit_stress", sig(long_real_mom_3m=0.3))["state"] != "BLOCKED":
        fails.append("credit_stress with rising real yields gets NO crisis exception")
    if entry_gate("TLT", _fx(_tlt_rebound), "credit_stress", sig(long_real_mom_3m=-0.3))["state"] != "CONFIRMED":
        fails.append("credit_stress flight variant gets the crisis exception")
    if entry_gate("XLV", _fx(_tlt_rebound), "liquidity_crisis")["state"] != "BLOCKED":
        fails.append("the crisis exception is for duration only")
    _lc = target_weights("liquidity_crisis", _fx(_tlt_rebound), sig(long_real_mom_3m=-0.4))
    if _lc["TLT"] < 15.5:
        fails.append(f"liquidity_crisis TLT re-arm must execute on the crisis exception: {_lc['TLT']}")

    tw = target_weights("restrictive_tightening", FP_UP)
    if abs(sum(tw.values()) - 100) > 0.5:
        fails.append(f"restrictive_tightening targets sum to {sum(tw.values())}")
    if tw.get("TLT", 99) > 3 or tw.get("SGOV", 0) < 10:
        fails.append(f"tightening targets should cut TLT to ~2 and lift SGOV: {tw}")
    # fed_reaction_flag must agree with the short real rate (Sept 2026 fix)
    _f = lambda **k: fed_reaction_flag(SignalSet(**k))["state"]
    if not _f(short_real_rate=0.48, long_real_yield=2.85, cpi_yoy=3.4).startswith("RESTRICTIVE"):
        fails.append("positive short real rate must read RESTRICTIVE, not repression")
    if not _f(short_real_rate=-0.5, long_real_yield=1.5, cpi_yoy=3.4).startswith("SOFT"):
        fails.append("negative short real + positive long real + hot CPI must read SOFT repression")
    if not _f(short_real_rate=-1.5, long_real_yield=0.0, cpi_yoy=4.0).startswith("HARD"):
        fails.append("suppressed long real + hot CPI must read HARD repression")
    return {"ok": not fails, "failures": fails, "targets_restrictive_tightening": tw}


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(selftest(), indent=2, default=str))
