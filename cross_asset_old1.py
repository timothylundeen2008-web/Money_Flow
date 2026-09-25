"""
cross_asset.py  (v1 — July 2026)
================================
The four-way confirmation panel: CREDIT, RATES, VOL, FX.

WHY THIS IS ASSEMBLY, NOT NEW PLUMBING
--------------------------------------
Almost every input already exists. data_fetcher.fetch_all_indicators() already
returns hy_spread, treasury_10y, tips_real_yield, breakeven, dxy_current and
dxy_20d_change_pct, and indicators.py already scores dollar_divergence,
gold_momentum_gate and auction_demand. They are scored INDIVIDUALLY and never
compared to each other.

That gap is what made 2026-07-29 unreadable from the dashboard:

    credit  HY OAS 2.77%              -> CALM
    rates   30y 5.19% (2007 high),
            DFII10 2.44% (18y high)   -> STRESS
    vol     VIX ~18-19                -> CALM
    fx      DXY ~101.3, top of range  -> CALM-ish

Three of four said "no crisis". One said "repricing". THE DIVERGENCE IS THE
SIGNAL, and no panel in the stack could show it because the four readings never
met.

LEAD-LAG WEIGHTING
------------------
A lone dissenter is not equally meaningful in each lane. Credit leads equity
drawdowns by roughly 2-4 weeks because credit dealers reprice ahead of
volatility-targeting strategies; FX and vol are closer to coincident. So a
solitary CREDIT dissent is a warning, while a solitary FX dissent is usually
noise. The panel weights accordingly and says which lane is dissenting rather
than just reporting a count.

Only new series required: VIX (^VIX via the existing yfinance fetcher).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

CALM, WATCH, STRESS, UNKNOWN = "CALM", "WATCH", "STRESS", "UNKNOWN"

# How much a lone dissent in each lane should worry you. Derived from lead-lag,
# not from conviction.
LANE_WEIGHT = {"credit": 1.00, "rates": 0.80, "vol": 0.55, "fx": 0.40}

LANE_LEAD = {
    "credit": "Leads equity drawdowns by ~2-4 weeks. A lone credit dissent is "
              "the single most actionable divergence on this panel.",
    "rates":  "Leads multiple compression. Distinguishes a term-premium "
              "repricing from an inflation-expectations shock — check whether "
              "breakevens are moving WITH real yields or against them.",
    "vol":    "Roughly coincident. Useful for sizing convexity, weak as a "
              "leading indicator.",
    "fx":     "Roughly coincident, and noisy. Matters mainly as a transmission "
              "channel — dollar strength tightens global conditions and is a "
              "direct headwind to metals.",
}


def _vote(state: str, reading: str, note: str = "") -> dict:
    return {"state": state, "reading": reading, "note": note}


# --------------------------------------------------------------------------- #
#  Lanes
# --------------------------------------------------------------------------- #
def credit_lane(hy_oas: Optional[float],
                hy_mom_2w: Optional[float] = None) -> dict:
    if hy_oas is None:
        return _vote(UNKNOWN, "n/a", "HY OAS unavailable — the credit lane is "
                                     "dark, which is not the same as calm.")
    widening = (hy_mom_2w or 0) > 0.5
    if hy_oas > 5.0:
        return _vote(STRESS, f"{hy_oas:.2f}%",
                     "Beyond the 500bp liquidity-crisis line."
                     + (" And widening fast." if widening else ""))
    if hy_oas > 3.5:
        return _vote(WATCH, f"{hy_oas:.2f}%",
                     "Above the 350bp complacency line — risk is being repriced."
                     + (" Widening." if widening else ""))
    if widening:
        return _vote(WATCH, f"{hy_oas:.2f}%",
                     "Level is tight but the 2-week momentum is widening. "
                     "Direction beats level in this lane.")
    return _vote(CALM, f"{hy_oas:.2f}%",
                 "Below 350bp. Read this as late-cycle risk COMPRESSION, not "
                 "safety — it leaves little room to tighten and a great deal "
                 "to widen.")


def rates_lane(long_real: Optional[float],
               long_real_mom_3m: Optional[float],
               breakeven: Optional[float] = None,
               nom_30y: Optional[float] = None) -> dict:
    """
    Stress in this lane means "the long end is repricing against risk assets".

    The diagnostic that matters is the JOINT move: real yields up with
    breakevens DOWN is a term-premium / fiscal-credibility event, not an
    inflation event, and the two call for opposite hedges.
    """
    if long_real is None:
        return _vote(UNKNOWN, "n/a", "DFII10 unavailable.")
    if long_real_mom_3m is None:
        return _vote(WATCH, f"{long_real:.2f}%",
                     "⚠ DFII10 momentum unavailable — direction unknown, so "
                     "this lane cannot confirm or deny.")

    rising = long_real_mom_3m > 0
    reading = f"{long_real:.2f}%" + (f" / 30y {nom_30y:.2f}%" if nom_30y else "")

    if rising and long_real > 2.0:
        diag = ("Long real yields rising from an already-high level: duration "
                "headwind and growth-multiple compression.")
        if breakeven is not None:
            if breakeven < 2.35:
                diag += (f" Breakevens {breakeven:.2f}% are LOW and not "
                         f"confirming an inflation story — this is TERM PREMIUM "
                         f"(supply / fiscal credibility), not inflation "
                         f"expectations. TIPS over nominals; short duration "
                         f"over long.")
            else:
                diag += (f" Breakevens {breakeven:.2f}% are also elevated — "
                         f"inflation compensation is contributing, so real "
                         f"assets carry more of the hedge.")
        return _vote(STRESS, reading, diag)

    if rising:
        return _vote(WATCH, reading,
                     "Long real yields rising but still moderate. Duration is a "
                     "foe; not yet a repricing event.")
    if long_real < 1.0:
        return _vote(CALM, reading,
                     "Long real yields low and not rising — the yield-curve-"
                     "control signature. Duration flips from foe to friend here.")
    return _vote(CALM, reading,
                 "Long real yields high but no longer rising. The repricing may "
                 "be done; confirm with auction demand before adding duration.")


def vol_lane(vix: Optional[float]) -> dict:
    if vix is None:
        return _vote(UNKNOWN, "n/a", "VIX unavailable.")
    if vix >= 30:
        return _vote(STRESS, f"{vix:.1f}",
                     "Acute. Protection is expensive; reduce exposure directly "
                     "rather than buying convexity here.")
    if vix >= 21:
        return _vote(WATCH, f"{vix:.1f}",
                     "Above the 21 line that shifts the risk regime toward "
                     "defensive positioning.")
    return _vote(CALM, f"{vix:.1f}",
                 "Derivatives are not pricing acute stress. If equities are "
                 "falling anyway, that combination favours the "
                 "positioning-unwind reading over the macro-break reading — "
                 "and it makes convexity comparatively cheap.")


def fx_lane(dxy: Optional[float], dxy_20d_change_pct: Optional[float],
            dxy_52w_high: Optional[float] = None) -> dict:
    if dxy is None:
        return _vote(UNKNOWN, "n/a", "DXY unavailable.")
    reading = f"{dxy:.1f}"
    near_high = (dxy_52w_high is not None and dxy_52w_high > 0
                 and dxy >= 0.98 * dxy_52w_high)
    chg = dxy_20d_change_pct

    if chg is not None and chg >= 2.5:
        return _vote(STRESS, reading,
                     f"Dollar up {chg:+.1f}% in 20 sessions — a global "
                     f"tightening impulse and a direct headwind to metals and "
                     f"non-US equity.")
    if near_high or (chg is not None and chg >= 1.0):
        return _vote(WATCH, reading,
                     "Dollar firm"
                     + (" and near its 52-week high" if near_high else "")
                     + ". Hedged international exposure is preferable to "
                       "unhedged here, and it is a standing headwind to gold.")
    return _vote(CALM, reading, "Dollar is not a tightening force at present.")


# --------------------------------------------------------------------------- #
#  The panel
# --------------------------------------------------------------------------- #
def divergence(hy_oas=None, hy_mom_2w=None,
               long_real=None, long_real_mom_3m=None, breakeven=None,
               nom_30y=None, vix=None,
               dxy=None, dxy_20d_change_pct=None, dxy_52w_high=None) -> dict:
    """
    Build all four lanes and diagnose their agreement.

    Returns a dict with:
        lanes           {lane: vote}
        agreement_pct   0-100, share of KNOWN lanes voting with the majority
        majority        CALM | WATCH | STRESS | UNKNOWN
        dissenters      [lane names]
        flag            "" when aligned, otherwise the divergence warning
        headline        one-line summary for the UI
    """
    lanes = {
        "credit": credit_lane(hy_oas, hy_mom_2w),
        "rates": rates_lane(long_real, long_real_mom_3m, breakeven, nom_30y),
        "vol": vol_lane(vix),
        "fx": fx_lane(dxy, dxy_20d_change_pct, dxy_52w_high),
    }

    known = {k: v for k, v in lanes.items() if v["state"] != UNKNOWN}
    out = {"lanes": lanes, "n_known": len(known), "n_dark": 4 - len(known)}

    if not known:
        out.update(agreement_pct=None, majority=UNKNOWN, dissenters=[],
                   flag="All four lanes are dark. No cross-asset read is "
                        "possible; the regime call rests on a single input.",
                   headline="Cross-asset: no data")
        return out

    counts: dict[str, int] = {}
    for v in known.values():
        counts[v["state"]] = counts.get(v["state"], 0) + 1
    majority = max(counts, key=lambda k: counts[k])
    agreement = 100.0 * counts[majority] / len(known)
    dissenters = [k for k, v in known.items() if v["state"] != majority]

    out.update(agreement_pct=round(agreement, 1), majority=majority,
               dissenters=dissenters)

    if not dissenters:
        out["flag"] = ""
        out["headline"] = (f"Cross-asset ALIGNED: all {len(known)} live lanes "
                           f"read {majority}.")
        return out

    worst = max(dissenters, key=lambda k: LANE_WEIGHT.get(k, 0.5))
    weight = LANE_WEIGHT.get(worst, 0.5)
    severity = ("HIGH" if weight >= 0.9 else
                "MODERATE" if weight >= 0.7 else "LOW")

    out["flag"] = (
        f"DIVERGENCE ({severity}): {len(known) - len(dissenters)} of "
        f"{len(known)} lanes read {majority}, but "
        f"{', '.join(f'{d} ({lanes[d]['state']})' for d in dissenters)} "
        f"disagree. Highest-weight dissenter is {worst.upper()}. "
        f"{LANE_LEAD.get(worst, '')}"
    )
    out["headline"] = (f"Cross-asset SPLIT {agreement:.0f}%: majority "
                       f"{majority}, {worst} dissenting")
    out["dissent_severity"] = severity
    out["dissent_weight"] = weight
    return out


def from_raw(raw: dict, sig=None, vix: Optional[float] = None,
             dxy_52w_high: Optional[float] = None) -> dict:
    """
    Adapter for the existing data shapes.

    `raw` is fetch_all_indicators()'s dict; `sig` is a SignalSet (or anything
    with the same attribute names). Uses getattr/get defensively so a schema
    change degrades one lane instead of raising — the same pattern as
    checklist_tab._get().
    """
    def g(name, default=None):
        if sig is None:
            return default
        if isinstance(sig, dict):
            return sig.get(name, default)
        return getattr(sig, name, default)

    return divergence(
        hy_oas=g("hy_oas", raw.get("hy_spread")),
        hy_mom_2w=g("hy_oas_mom_2w"),
        long_real=g("long_real_yield", raw.get("tips_real_yield")),
        long_real_mom_3m=g("long_real_mom_3m"),
        breakeven=g("breakeven_10y", raw.get("breakeven")),
        nom_30y=raw.get("treasury_30y"),
        vix=vix,
        dxy=raw.get("dxy_current"),
        dxy_20d_change_pct=raw.get("dxy_20d_change_pct"),
        dxy_52w_high=dxy_52w_high,
    )


def selftest() -> dict:
    """Reproduce 2026-07-29 and confirm the divergence is flagged."""
    d = divergence(hy_oas=2.77, hy_mom_2w=0.0,
                   long_real=2.44, long_real_mom_3m=0.48, breakeven=2.26,
                   nom_30y=5.19, vix=18.5,
                   dxy=101.3, dxy_20d_change_pct=0.4, dxy_52w_high=101.80)
    failures = []
    if not d["flag"]:
        failures.append("No divergence flagged on the 2026-07-29 input set")
    if d["lanes"]["rates"]["state"] != STRESS:
        failures.append(f"Rates lane read {d['lanes']['rates']['state']}, "
                        f"expected STRESS")
    if d["lanes"]["credit"]["state"] != CALM:
        failures.append(f"Credit lane read {d['lanes']['credit']['state']}, "
                        f"expected CALM")
    if "term premium" not in d["lanes"]["rates"]["note"].lower():
        failures.append("Rates lane failed to diagnose term premium from the "
                        "real-yields-up / breakevens-down combination")
    return {"ok": not failures, "failures": failures,
            "headline": d["headline"], "flag": d["flag"],
            "agreement_pct": d["agreement_pct"],
            "lanes": {k: v["state"] for k, v in d["lanes"].items()}}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, default=str))
