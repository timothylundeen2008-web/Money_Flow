"""
dashboard_links.py  (v1 — August 2026)
──────────────────────────────────────
Cross-dashboard navigation. Identical copy in all three repos.

THE THREE-DASHBOARD ARCHITECTURE
────────────────────────────────
Each answers a different question, in a deliberate order:

    1. MONEY FLOW      "Where is capital actually going?"
                       Evidence. COT positioning, ETF creations/redemptions,
                       sector rotation, breadth. Tier A (capital) separated
                       from Tier B (pressure), because volume is not flow.

    2. MARKETS         "What environment is this, and what does the detail
                       say?" Regime classification across four axes — rates,
                       credit, growth, valuation — plus relative strength,
                       trend state, curve shape, factor exposure.

    3. ALL-WEATHER     "Given both of the above, what should I hold?"
                       Current state and suggested changes, with capital
                       preservation as the anchor rather than an afterthought.

The reading ORDER matters and is not arbitrary: flow is evidence, markets is
interpretation, portfolio is action. Reading them backwards — deciding an
allocation and then looking for flow to justify it — is the failure mode this
ordering exists to prevent.

CONFIGURATION
─────────────
URLs live in ONE place. Update this file, copy to all three repos. Anything
left as a placeholder renders as disabled rather than as a broken link — a
dead link is worse than a visibly missing one.
"""

from __future__ import annotations

# ── The three deployments ───────────────────────────────────────────────────
# ⚠ Replace any placeholder below with the real Streamlit URL, then copy this
# file to all three repos so every dashboard shows the same nav.
DASHBOARDS = {
    "money_flow": {
        "name": "Institutional Money Flow",
        "icon": "💧",
        "url": "https://moneyflow-p893texzgslivee8jg7djb.streamlit.app",
        "role": "Evidence — where capital is actually moving",
        "order": 1,
    },
    "markets": {
        "name": "Markets Dashboard",
        "icon": "🌡️",
        "url": "https://mechanically-architected-jqmdefv8di4ee8n7epkpqs.streamlit.app",
        "role": "Macro regime + detail — what environment is this",
        "order": 2,
    },
    "all_weather": {
        "name": "All-Weather Portfolio",
        "icon": "🧭",
        "url": "https://portfolio-tracker-bvou6bvseejz4gygmlcm9y.streamlit.app",
        "role": "Action — current state and suggested changes",
        "order": 3,
    },
}

PLACEHOLDER_PREFIX = "REPLACE_WITH"


def is_configured(key: str) -> bool:
    url = DASHBOARDS.get(key, {}).get("url", "")
    return bool(url) and not url.startswith(PLACEHOLDER_PREFIX)


def render_nav(st, current: str):
    """
    Render the cross-dashboard nav bar. `current` is this app's key.

    Placed at the very top of each app so the three are usable together —
    open all three in tabs and move between them without hunting for URLs.
    """
    items = sorted(DASHBOARDS.items(), key=lambda kv: kv[1]["order"])
    cols = st.columns(len(items))

    for col, (key, d) in zip(cols, items):
        with col:
            if key == current:
                col.markdown(
                    f"<div style='background:#1a2332;border:1px solid #2d4263;"
                    f"border-radius:8px;padding:.5rem .75rem;text-align:center;'>"
                    f"<span style='font-size:.85rem;font-weight:700;"
                    f"color:#7aa2e3;'>{d['icon']} {d['name']}</span><br>"
                    f"<span style='font-size:.68rem;color:#5c6475;'>"
                    f"you are here</span></div>",
                    unsafe_allow_html=True)
            elif is_configured(key):
                col.markdown(
                    f"<a href='{d['url']}' target='_blank' "
                    f"style='text-decoration:none;'>"
                    f"<div style='background:#13161b;border:1px solid #242830;"
                    f"border-radius:8px;padding:.5rem .75rem;text-align:center;'>"
                    f"<span style='font-size:.85rem;font-weight:600;"
                    f"color:#9aa3b2;'>{d['icon']} {d['name']} ↗</span><br>"
                    f"<span style='font-size:.68rem;color:#5c6475;'>"
                    f"{d['role']}</span></div></a>",
                    unsafe_allow_html=True)
            else:
                col.markdown(
                    f"<div style='background:#13161b;border:1px dashed #3a2f2f;"
                    f"border-radius:8px;padding:.5rem .75rem;text-align:center;"
                    f"opacity:.55;'>"
                    f"<span style='font-size:.85rem;color:#7a8394;'>"
                    f"{d['icon']} {d['name']}</span><br>"
                    f"<span style='font-size:.68rem;color:#a06767;'>"
                    f"URL not set in dashboard_links.py</span></div>",
                    unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:.7rem;color:#5c6475;text-align:center;"
        "margin:.35rem 0 .8rem;'>"
        "Read in order: <b>flow is evidence</b> → <b>markets is "
        "interpretation</b> → <b>portfolio is action</b>. Deciding an "
        "allocation first and then looking for flow to justify it is the "
        "failure mode this ordering prevents."
        "</div>", unsafe_allow_html=True)


def selftest() -> dict:
    failures = []
    if len(DASHBOARDS) != 3:
        failures.append(f"expected 3 dashboards, got {len(DASHBOARDS)}")
    orders = sorted(d["order"] for d in DASHBOARDS.values())
    if orders != [1, 2, 3]:
        failures.append(f"order values are {orders}, expected 1,2,3")
    unconfigured = [k for k in DASHBOARDS if not is_configured(k)]
    return {"ok": not failures, "failures": failures,
            "configured": [k for k in DASHBOARDS if is_configured(k)],
            "needs_url": unconfigured}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, default=str))
