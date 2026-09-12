"""
publish_to_checklist.py  (v1 — July 2026)
──────────────────────────────────────────────────────────────────────────────
DEPLOYMENT.md Step 5, implemented: computes every Tier-A/B/C signal this repo
produces and publishes them to the shared bridge the All-Weather checklist
reads (rotation_bridge.publish_summary → storage_backend → GitHub).

Run this two ways:
  1. Scheduled, headless — .github/workflows/daily_publish.yml (below). This is
     the durable path: it runs whether or not anyone opens the Streamlit app.
  2. On demand — a "🔄 Publish to Checklist" button in the app sidebar, for
     immediate freshness before a weekly review.

WHY A SEPARATE SCRIPT RATHER THAN INLINING INTO app.py
  The scheduled Action runs this with NO Streamlit runtime at all (bare
  `python publish_to_checklist.py`). Every function called here must work
  headless — fetch_sector_data(), build_cot_table(), etc. already do (their
  @st.cache_data decorators degrade to no-ops outside Streamlit, per each
  module's own guard). This script is the thing that proves it.

WHAT GETS PUBLISHED
  sector_df        fetch_sector_data() — quadrant, momentum, CMF, accumulation/
                    event scores (flow_metrics, wired in via data_fetcher v3)
  flow_divergence  etf_flow_tracker.flow_vs_price_divergence() — needs the
                    daily shares-outstanding poll to have run first (see below)
  cot_table        cot_fetcher.build_cot_table() — Asset Manager / Leveraged
                    Fund positioning percentiles
  breadth_table    constituent_breadth.build_breadth_table() — BROAD /
                    CONCENTRATED / NARROW per sector
  cot_schema_ok    cot_fetcher.verify_schema() — CFTC has renamed fields
                    before; publish_summary refuses to imply positioning data
                    is trustworthy when this hasn't passed

ORDERING MATTERS
  The ETF shares snapshot must run BEFORE flow_vs_price_divergence() — it
  reads accumulated history, not a live feed. The combined workflow below
  runs the snapshot first, in the same job, before this script.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone


def _safe(label: str, fn, *a, **kw):
    """Never let one failing source blank the whole publish. A partial summary
    with 3 of 4 sources beats no summary at all — and the caller can see
    exactly what failed from the printed traceback."""
    try:
        return fn(*a, **kw)
    except Exception:
        print(f"[publish] {label} FAILED:", file=sys.stderr)
        traceback.print_exc()
        return None


def main() -> int:
    print(f"[publish] starting {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")

    from data_fetcher import fetch_sector_data
    from etf_flow_tracker import flow_vs_price_divergence, coverage_report
    from cot_fetcher import build_cot_table, verify_schema
    from constituent_breadth import build_breadth_table
    from rotation_bridge import publish_summary

    sector_df = _safe("sector_df", fetch_sector_data)
    print(f"[publish] sector_df: {0 if sector_df is None else len(sector_df)} rows")

    cov = _safe("etf flow coverage check", coverage_report) or {}
    print(f"[publish] ETF flow coverage: {cov.get('message', 'unknown')}")
    flow_div = _safe("flow_divergence", flow_vs_price_divergence)
    print(f"[publish] flow_divergence: {0 if flow_div is None else len(flow_div)} rows")

    schema = _safe("COT schema check", verify_schema) or {}
    schema_ok = bool(schema) and all(v.get("ok") for v in schema.values())
    if schema and not schema_ok:
        print(f"[publish] ⚠ COT schema check FAILED: {schema}", file=sys.stderr)
    cot_table = _safe("cot_table", build_cot_table)
    print(f"[publish] cot_table: {0 if cot_table is None else len(cot_table)} rows "
          f"(schema_ok={schema_ok})")

    breadth = _safe("breadth_table", build_breadth_table)
    print(f"[publish] breadth_table: {0 if breadth is None else len(breadth)} rows")

    notes = (f"published by publish_to_checklist.py; "
             f"etf_flow coverage: {cov.get('days', 0)}d history, "
             f"ready={cov.get('ready', False)}")

    result = publish_summary(
        sector_df=sector_df, flow_divergence=flow_div,
        cot_table=cot_table, breadth_table=breadth,
        cot_schema_ok=schema_ok if schema else None,
        notes=notes,
    )

    print(f"[publish] result: ok={result['ok']} durable={result['durable']} "
          f"counts={result['counts']}")

    if not result["ok"]:
        print("[publish] FAILED TO WRITE — check GITHUB_TOKEN/GITHUB_REPO "
              "(env vars in the Action, or Streamlit secrets when run from the "
              "app). See storage_backend.backend_status().", file=sys.stderr)
        return 1
    if not result["durable"]:
        print("[publish] ⚠ wrote to a NON-durable backend (local/session only) "
              "— this will be lost. Configure GITHUB_TOKEN + GITHUB_REPO.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
