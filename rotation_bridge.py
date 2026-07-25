"""
rotation_bridge.py  (v1 — July 2026)   ── CLOSES GAPS G4 and G7
──────────────────────────────────────────────────────────────────────────────
Lets the All-Weather app read the Rotation app's Tier-A and Tier-B signals.

THE PROBLEM
  Every rotation, flow, COT and breadth check lives in a SEPARATE repo deployed
  as a separate Streamlit app. This app cannot import from it. So the entire
  evidence layer v5 was written to add — the Tier-A money data and Tier-B
  directional-volume data — was manual re-typing in the checklist tab.

  That friction is not cosmetic. Seven of the checklist's manual items come
  from this one architectural fact, and manual steps are precisely the ones
  that get skipped on a busy weekend. The confluence test's Tier-A leg is the
  one that gates FULL SIZE — leaving it as re-typing means the most consequential
  check in the system is the least likely to be completed.

THE FIX — publish/consume via the shared GitHub repo
  Producer (Rotation app, or its GitHub Action):  publish_summary(...)
  Consumer (this app):                            read_summary()

  A small JSON blob written to the same repo the storage backend already uses.
  Loosest possible coupling, works TODAY without merging repos, and the
  staleness question it introduces is handled explicitly rather than ignored:
  every read reports the age of the data and refuses to present a stale
  snapshot as current.

WHY NOT JUST MERGE THE REPOS
  Merging is the better long-term answer and GAPS.md still recommends it — it
  would also fix the duplicated regime_classifier.py and
  repression_regime_section.py files that have already caused three defects.
  This bridge is the version that works this week, without a migration.

G7 — COT SCHEMA
  cot_status() surfaces whether the producer verified the CFTC schema, so a
  silent field rename upstream shows up here as a warning rather than as
  quietly missing positioning data.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from storage_backend import read_json, write_json

SUMMARY_FILE = "rotation_summary.json"
STALE_HOURS = 36          # a weekend-run summary read on Monday is still fine
VERY_STALE_HOURS = 168    # a week — beyond this it is not evidence


# ── Producer side (runs in the Rotation app / its Action) ─────────────────────

def publish_summary(sector_df: pd.DataFrame | None = None,
                    flow_divergence: pd.DataFrame | None = None,
                    cot_table: pd.DataFrame | None = None,
                    breadth_table: pd.DataFrame | None = None,
                    cot_schema_ok: bool | None = None,
                    notes: str = "") -> dict:
    """
    Call from the Rotation app after its data refresh.

    Publishes only what the All-Weather checklist actually consumes — this is
    a signal summary, not a data mirror.
    """
    payload = {
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
        "cot_schema_ok": cot_schema_ok,
        "sectors": [], "flow_divergences": [], "cot": [], "breadth": [],
    }

    if sector_df is not None and not sector_df.empty:
        keep = [c for c in ("ticker", "sector", "quadrant", "accumulation_score",
                            "event_score", "stealth_label", "cmf", "vol_ratio",
                            "signal", "momentum_accel") if c in sector_df.columns]
        d = sector_df[keep].copy()
        if "accumulation_score" in d.columns:
            d = d.sort_values("accumulation_score", ascending=False, na_position="last")
        payload["sectors"] = d.head(15).to_dict("records")

    if flow_divergence is not None and not flow_divergence.empty:
        keep = [c for c in ("ticker", "price_chg_pct", "net_flow_pct_aum",
                            "verdict", "divergence") if c in flow_divergence.columns]
        payload["flow_divergences"] = flow_divergence[keep].to_dict("records")

    if cot_table is not None and not cot_table.empty:
        t = cot_table.reset_index() if cot_table.index.name else cot_table
        keep = [c for c in t.columns
                if c in ("contract", "sleeve", "report_date", "flag")
                or c.endswith(("_pctile", "_net", "_net_pct_oi"))]
        payload["cot"] = t[keep].to_dict("records")

    if breadth_table is not None and not breadth_table.empty:
        keep = [c for c in ("etf", "signal_quality", "pct_positive_cmf",
                            "top3_return_share", "dispersion") if c in breadth_table.columns]
        payload["breadth"] = breadth_table[keep].to_dict("records")

    res = write_json(SUMMARY_FILE, payload,
                     message=f"rotation: publish summary "
                             f"({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)")
    return {"ok": bool(res.get("github") or res.get("local")),
            "durable": res.get("durable", False), "storage": res,
            "counts": {k: len(payload[k]) for k in
                       ("sectors", "flow_divergences", "cot", "breadth")}}


# ── Consumer side (this app) ──────────────────────────────────────────────────

def read_summary() -> dict:
    """
    Read the published summary with an explicit staleness verdict.

    A stale snapshot presented as current is worse than no data — it is the
    same failure the 45-day valuation stale-flag rule exists to prevent, so it
    is handled the same way here.
    """
    data = read_json(SUMMARY_FILE)
    if not data:
        return {"available": False, "stale": True,
                "message": "No rotation summary published yet. Run publish_summary() "
                           "from the Rotation app (or its GitHub Action). Until then, "
                           "Tier-A and Tier-B legs must be entered manually — and per "
                           "the provenance rule, an unevidenced leg scores NOT MET."}

    try:
        pub = datetime.fromisoformat(data["published_at"])
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - pub).total_seconds() / 3600
    except Exception:
        age_h = float("inf")

    very_stale = age_h > VERY_STALE_HOURS
    stale = age_h > STALE_HOURS

    return {
        "available": True, "age_hours": round(age_h, 1),
        "stale": stale, "very_stale": very_stale,
        "published_at": data.get("published_at"),
        "sectors": data.get("sectors", []),
        "flow_divergences": data.get("flow_divergences", []),
        "cot": data.get("cot", []),
        "breadth": data.get("breadth", []),
        "cot_schema_ok": data.get("cot_schema_ok"),
        "message": (f"⚠ VERY STALE ({age_h/24:.1f} days). Do not treat as evidence — "
                    f"re-run the Rotation app. Per the provenance rule, an "
                    f"unevidenced leg scores NOT MET, not 'probably fine'."
                    if very_stale else
                    f"⚠ Stale ({age_h:.0f}h old) — usable for context, but re-publish "
                    f"before sizing off it." if stale else
                    f"Fresh ({age_h:.0f}h old)."),
    }


def confluence_inputs(ticker: str) -> dict:
    """
    Pull the Tier-B and Tier-A legs for one candidate, for Weekly Step 5.

    Returns each leg as met / not_met / unavailable. UNAVAILABLE IS NOT MET —
    the provenance rule is enforced in the return value so a caller cannot
    accidentally score a missing leg as passing.
    """
    s = read_summary()
    base = {"ticker": ticker, "tier_b": "unavailable", "tier_a": "unavailable",
            "tier_b_detail": "", "tier_a_detail": "", "max_size": "watchlist"}
    if not s.get("available") or s.get("very_stale"):
        base["note"] = s.get("message")
        return base

    for row in s.get("sectors", []):
        if str(row.get("ticker", "")).upper() == ticker.upper():
            cmf = row.get("cmf")
            acc = row.get("accumulation_score")
            if cmf is not None and acc is not None:
                met = (cmf > 0.05) and (acc > 10)
                base["tier_b"] = "met" if met else "not_met"
                base["tier_b_detail"] = f"CMF {cmf}, accumulation score {acc}"
            break

    for row in s.get("flow_divergences", []):
        if str(row.get("ticker", "")).upper() == ticker.upper():
            flow = row.get("net_flow_pct_aum")
            if flow is not None:
                base["tier_a"] = "met" if flow > 0 else "not_met"
                base["tier_a_detail"] = (f"4w net flow {flow:+.2f}% of AUM "
                                         f"({row.get('verdict', '')})")
            break

    if base["tier_a"] == "met" and base["tier_b"] == "met":
        base["max_size"] = "full (pending Tier-C rotation leg)"
    elif base["tier_b"] == "met":
        base["max_size"] = "half (no Tier-A money confirmation)"
    return base


def cot_status() -> dict:
    """G7: has the producer verified the CFTC schema?"""
    s = read_summary()
    if not s.get("available"):
        return {"ok": False, "message": "No summary published."}
    flag = s.get("cot_schema_ok")
    if flag is None:
        return {"ok": False,
                "message": "COT schema never verified. Run cot_fetcher.verify_schema() "
                           "in the Rotation app and pass the result to "
                           "publish_summary(cot_schema_ok=...). CFTC has renamed "
                           "Socrata fields before — unverified positioning data "
                           "should not be sized off."}
    return {"ok": bool(flag), "rows": len(s.get("cot", [])),
            "message": "COT schema verified." if flag else
                       "⚠ COT schema check FAILED upstream — positioning data may be "
                       "wrong or missing. Fix in the Rotation app before use."}
