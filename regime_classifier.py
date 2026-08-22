"""
regime_classifier.py  (v3 — July 2026 band/guard patch)
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
    asof: Optional[_dt.date] = None
    notes: list = field(default_factory=list)

    def as_row(self) -> pd.DataFrame:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("notes", "asof")}
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
    sig.ig_oas = _last(ig)

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
    Soft repression  = Fed HOLDING/HIKING into above-target inflation, long real
                       yield positive (inflation overshoot erodes debt).
    Hard repression  = Fed CUTTING/CAPPING while inflation high AND long real
                       yield suppressed toward/below zero (yield-curve control).
    """
    inflation_hot = (sig.cpi_yoy or 0) > FED_TARGET_INFLATION + 0.5
    long_real = sig.long_real_yield
    long_real_pos = long_real is not None and long_real > 0.5

    if inflation_hot and long_real_pos:
        state = "SOFT repression (inflation overshoot)"
        detail = ("Fed tolerating / fighting above-target inflation while the "
                  "long end stays positive. Long duration is NOT safe here.")
    elif inflation_hot and long_real is not None and long_real < 0.25:
        state = "HARD repression (yield suppression / YCC risk)"
        detail = ("Long real yields pinned low despite hot inflation — classic "
                  "financial-repression signature. Nominal bonds bleed slowly.")
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
        drivers.append(f"Growth composite CONTRACTING "
                       f"({(growth or {}).get('score', 0):+d}) — "
                       f"{(growth or {}).get('detail', '')[:160]}")
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
def target_weights(regime_key: str,
                   fetch_prices: Callable = None) -> dict:
    """Apply the regime overlay to the base sleeve and renormalize to 100%.

    v2: in inflationary_repression the GLD tilt is momentum-gated (FIX 2).
    Rising long real yields — the regime's defining signal — are gold's
    primary headwind, so the metal add requires trend confirmation; when the
    gate fails, the tilt redirects to SGOV until GLD reclaims a rising
    200-day MA (the Lagging->Improving hook, in RRG terms).

    Backward compatible: target_weights('goldilocks') etc. work unchanged.
    """
    w = dict(BASE_WEIGHTS)
    overlay = dict(REGIMES[regime_key]["overlay"])

    # v3 FIX 8: the gate now covers EVERY regime that ADDS gold, not just
    # inflationary_repression. v2 gated that one (+3) and left hard_repression
    # (+4) and stagflation (+4) ungated -- two of the three regimes that buy
    # gold could still buy it into a confirmed downtrend. As of 2026-07-29 GLD
    # sits ~10% below a FALLING 200d with an active death cross, so the
    # difference is live rather than theoretical.
    #
    # Note the asymmetry: a NEGATIVE GLD tilt (goldilocks, -3) is never gated.
    # Trend confirmation is required to ADD to a metal, not to trim one.
    if overlay.get("GLD", 0) > 0:
        if not _gold_trend_ok(fetch_prices or _inline_fetch_prices):
            overlay["SGOV"] = overlay.get("SGOV", 0) + overlay["GLD"]
            overlay["GLD"] = 0

    for t, d in overlay.items():
        w[t] = max(0, w.get(t, 0) + d)
    total = sum(w.values())
    if total <= 0:
        return w
    return {t: round(v * 100 / total, 1) for t, v in w.items()}


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
    return {
        "signals": sig,
        "regime": regime,
        "fed": fed_reaction_flag(sig),
        "targets": target_weights(regime["key"], fetch_prices=fetch_prices),
        "kmlm": kmlm_signal(sig),
        "repression": repression_score(
            sig,
            fed_bs_expanding=fed_bs_expanding,
            deficit_gt_5pct_gdp=deficit_gt_5pct_gdp),
    }
