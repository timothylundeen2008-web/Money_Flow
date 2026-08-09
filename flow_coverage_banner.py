"""
flow_coverage_banner.py  (v1 — August 2026)
────────────────────────────────────────────────────────────────────────────
Renders the DATA READINESS state of the three flow layers, prominently.

WHY THIS EXISTS
───────────────
etf_flow_tracker.coverage_report() already computes exactly the right thing:

    {"tickers": n, "days": d, "median_days": m, "ready": d >= 20,
     "message": "Only N sessions stored — need ~20. Keep polling daily."}

Nothing in app.py renders it. That is the specific failure mode this module
closes: with `data/etf_shares_history.csv` absent, every downstream flow
function returns an empty frame, and an empty frame renders as an empty
table — which reads as "no institutional flows detected" when the truth is
"no data has ever been collected."

Those two states demand opposite actions. The first is a market observation.
The second is a broken pipeline. A dashboard that cannot tell them apart is
the silent-wrong-output failure this whole system is built to avoid, so the
banner FAILS LOUD: absent data is rendered as an error, not as a quiet zero.

THE THREE LAYERS DEGRADE INDEPENDENTLY
──────────────────────────────────────
  COT        weekly, public history, works on day one, ALWAYS >=3 days stale
  ETF flow   daily, self-accumulated, needs ~20 sessions, CANNOT be backfilled
  Breadth    daily, computed from price/volume, works on day one

So the banner reports each separately rather than emitting one composite
"healthy/unhealthy" light. A composite would hide the fact that COT is fully
readable today while ETF flow is months from useful.

USAGE
─────
    import flow_coverage_banner as fcb
    fcb.render(st)                      # top of the Flow tab
    state = fcb.assess()                # or headless, for publish_to_checklist
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

# Sessions of ETF share-count history required before flow_vs_price_divergence
# and the 20-day sums mean anything. Matches etf_flow_tracker's own threshold.
READY_THRESHOLD = 20

# A 5-day sum is marginally readable well before 20, so it gets its own tier.
PARTIAL_THRESHOLD = 5

# COT is published Fridays 3:30pm ET for the prior Tuesday. Anything beyond
# ~10 days means the fetch is failing, not merely that data is inherently lagged.
COT_STALE_DAYS = 10

STATUS_READY = "READY"
STATUS_PARTIAL = "PARTIAL"
STATUS_COLLECTING = "COLLECTING"
STATUS_ABSENT = "ABSENT"
STATUS_ERROR = "ERROR"


def _etf_state(store: str | None = None) -> dict:
    """Coverage of the self-accumulated ETF share-count series (Tier A)."""
    out = {"layer": "ETF creations/redemptions", "tier": "A (capital)",
           "status": STATUS_ABSENT, "days": 0, "tickers": 0,
           "days_remaining": READY_THRESHOLD, "detail": "", "action": ""}
    try:
        import etf_flow_tracker as eft
        store = store or eft.DEFAULT_STORE

        if not os.path.exists(store):
            out["detail"] = (
                f"No flow store at `{store}`. Shares-outstanding history has "
                f"never been written, so every flow panel below is EMPTY BY "
                f"ABSENCE OF DATA — not because institutions are inactive.")
            out["action"] = (
                "Run the 'Daily Poll & Publish' workflow (Actions tab → Run "
                "workflow). Free sources expose shares outstanding only as a "
                "CURRENT SNAPSHOT, so every day not polled is permanently "
                "lost history.")
            return out

        rep = eft.coverage_report(store)
        days = int(rep.get("days", 0) or 0)
        out.update(days=days, tickers=int(rep.get("tickers", 0) or 0),
                   days_remaining=max(0, READY_THRESHOLD - days))

        if days >= READY_THRESHOLD:
            out["status"] = STATUS_READY
            out["detail"] = (f"{days} sessions across {out['tickers']} tickers. "
                             f"20-day sums and flow-vs-price divergence are "
                             f"valid.")
            out["action"] = "Read the 20-day divergence table as primary."
        elif days >= PARTIAL_THRESHOLD:
            out["status"] = STATUS_PARTIAL
            out["detail"] = (f"{days} sessions stored. 5-day sums are "
                             f"marginally readable; 20-day divergence is NOT "
                             f"yet valid.")
            out["action"] = (f"{out['days_remaining']} more sessions "
                             f"(~{out['days_remaining'] // 5 + 1} weeks) until "
                             f"divergence analysis unlocks. Keep polling.")
        else:
            out["status"] = STATUS_COLLECTING
            out["detail"] = (f"Only {days} session(s) stored. Nothing in this "
                             f"layer is readable yet.")
            out["action"] = (f"{out['days_remaining']} more sessions needed. "
                             f"Do not interpret flow panels until then.")
        return out
    except Exception as e:
        out["status"] = STATUS_ERROR
        out["detail"] = f"Coverage check failed: {e}"
        out["action"] = "Fix the fetcher before reading any flow panel."
        return out


def _cot_state() -> dict:
    """COT is public history — readable on day one, but always lagged."""
    out = {"layer": "COT futures positioning", "tier": "A (positioning)",
           "status": STATUS_READY, "days": None, "detail": "", "action": ""}
    try:
        import cot_fetcher  # noqa: F401
        out["detail"] = (
            "Public CFTC history — no accumulation required, readable today. "
            "STRUCTURALLY LAGGED: data as of each Tuesday, published Friday "
            "3:30pm ET, so it is ALWAYS at least 3 days stale by construction.")
        out["action"] = (
            "With the ETF layer still filling, this is currently your best "
            "genuine institutional signal. Read the PERCENTILE columns, not "
            "raw net positions.")
        return out
    except Exception as e:
        out["status"] = STATUS_ERROR
        out["detail"] = f"cot_fetcher unavailable: {e}"
        return out


def _breadth_state() -> dict:
    """Breadth/AD/CMF compute from price+volume — no accumulation needed."""
    out = {"layer": "Constituent breadth & A/D", "tier": "B (pressure)",
           "status": STATUS_READY, "days": None, "detail": "", "action": ""}
    try:
        import constituent_breadth  # noqa: F401
        import flow_metrics  # noqa: F401
        out["detail"] = (
            "Computed from price and volume on demand — readable today. This "
            "measures PRESSURE, not capital: it cannot distinguish two "
            "investors trading with each other from new money entering.")
        out["action"] = (
            "Use as a VALIDATOR, never as the primary flow read. Its job is "
            "answering whether an ETF-level signal is sector-wide or one "
            "mega-cap dragging the index.")
        return out
    except Exception as e:
        out["status"] = STATUS_ERROR
        out["detail"] = f"breadth modules unavailable: {e}"
        return out


def assess(store: str | None = None) -> dict:
    """Headless state for all three layers. Never raises."""
    layers = [_cot_state(), _etf_state(store), _breadth_state()]
    etf = layers[1]

    if etf["status"] in (STATUS_ABSENT, STATUS_ERROR):
        overall = ("Tier A CAPITAL layer is DARK. Positioning (COT) and "
                   "pressure (breadth) are readable; money flow is not.")
    elif etf["status"] == STATUS_READY:
        overall = "All three layers readable. Full multi-horizon read available."
    else:
        overall = (f"ETF capital layer still accumulating "
                   f"({etf['days']}/{READY_THRESHOLD} sessions). Lead with COT "
                   f"and breadth until it fills.")

    return {"layers": layers, "etf": etf, "overall": overall,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def render(st, store: str | None = None) -> dict:
    """Render the banner. Returns the assessment so callers can gate panels."""
    state = assess(store)
    etf = state["etf"]

    st.markdown("#### Data coverage — read this before the panels below")

    if etf["status"] == STATUS_ABSENT:
        st.error(f"**ETF FLOW LAYER: NO DATA.** {etf['detail']}\n\n"
                 f"**Action:** {etf['action']}")
    elif etf["status"] == STATUS_ERROR:
        st.error(f"**ETF FLOW LAYER: ERROR.** {etf['detail']}")
    elif etf["status"] == STATUS_READY:
        st.success(f"**ETF flow layer ready.** {etf['detail']}")
    else:
        pct = min(1.0, etf["days"] / READY_THRESHOLD)
        st.warning(f"**ETF flow layer accumulating.** {etf['detail']}\n\n"
                   f"{etf['action']}")
        try:
            st.progress(pct, text=f"{etf['days']}/{READY_THRESHOLD} sessions")
        except Exception:
            st.progress(pct)

    cols = st.columns(3)
    for col, layer in zip(cols, state["layers"]):
        icon = {"READY": "🟢", "PARTIAL": "🟡", "COLLECTING": "🟠",
                "ABSENT": "🔴", "ERROR": "🔴"}.get(layer["status"], "⚪")
        with col:
            st.markdown(f"{icon} **{layer['layer']}**  \n"
                        f"<span style='color:#9ca3af;font-size:0.85rem;'>"
                        f"Tier {layer['tier']} · {layer['status']}</span>",
                        unsafe_allow_html=True)
            st.caption(layer["detail"])

    st.info(f"**Overall:** {state['overall']}")
    return state


def gate(st, state: dict, layer: str = "etf") -> bool:
    """
    Guard a panel. Returns False and prints a placeholder when the layer is
    not readable, so an empty table is never rendered as if it were a result.

        state = fcb.render(st)
        if fcb.gate(st, state):
            render_divergence_table()
    """
    if layer == "etf":
        s = state["etf"]["status"]
        if s in (STATUS_READY, STATUS_PARTIAL):
            return True
        st.caption("— Panel suppressed: insufficient flow history. An empty "
                   "table here would read as 'no flows', which is not what the "
                   "data says. It says nothing yet.")
        return False
    return True


if __name__ == "__main__":
    import json
    print(json.dumps(assess(), indent=2, default=str))
