"""
auto_log.py  (v1 — August 2026)
───────────────────────────────
Runs the checklist automatically on a schedule, persists the results, and
writes a plain-language daily/weekly analysis.

WHAT IT DOES
────────────
    daily   Steps 0-9 of Checklist v4 Part 1. Captures the log line the
            checklist has always asked for by hand — HY OAS, VIX, DFII10,
            2s10s, DXY, short real rate, plus regime, score and gate states.
    weekly  Steps 1-10 of Part 2, plus drift vs target weights and the
            institutional-flow layer.

Both append to a CSV history AND write a dated markdown summary, because the
two serve different purposes: the CSV is what makes trends measurable over
months, the markdown is what you actually read.

DESIGN RULES
────────────
1. NEVER FABRICATE. Every field that cannot be fetched is recorded as null
   with the reason. A log full of nulls is honest and fixable; a log of
   silently-defaulted zeros is the failure this whole project exists to
   prevent.
2. NEVER TRADE. This produces observations, not orders. The checklist's
   daily action rule stands: only the HY crisis override, a stop execution,
   and the three mechanical options rules authorize a daily trade.
3. ET-KEYED. Rows are keyed by Eastern trading date via market_time, never
   by UTC date. A 22:30 UTC run is the same ET day in summer and the
   PREVIOUS ET day in winter.
4. IDEMPOTENT. Re-running for the same date overwrites that date's row
   rather than duplicating it, so a retried job is harmless.
5. SKIP NON-TRADING DAYS. Writing a "log" for a holiday manufactures a fake
   session and corrupts every rolling window computed from the history.

SCHEDULING — WHY 6:30 PM ET AND NOT 5:00 PM
───────────────────────────────────────────
One hour after the close is too early for the inputs this log depends on:

    Treasury H.15 (DGS10, DFII10)  released ~4:15pm ET, reaches FRED later
    FRED daily series               generally settled by early evening
    ICE BofA HY OAS via FRED        ~1 day lag regardless of run time
    ETF issuer shares files         typically posted 6-9pm ET

A 5:00pm run would routinely capture the PREVIOUS day's Treasury values while
stamping them with today's date — a silent off-by-one that would be nearly
invisible in the CSV and would corrupt every momentum calculation built on
it. 6:30pm ET (2.5 hours after close) clears all four sources with margin
while still landing the same evening.

Weekly at Sunday 6:00am ET is well chosen and unchanged: it falls after the
COT release (Friday 3:30pm ET) and before the week begins, which is exactly
what Checklist v4 Weekly Step 5 requires.
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime

import market_time as mt

# Manual macro/valuation flags — same set, same "why", as checklist_tab.py
# and app.py. Kept in sync by hand; review dates whenever the underlying
# facts change. CAPE and concentration specifically arm the goldilocks
# valuation guard — without them it fails open silently, which is exactly
# how a scheduled run could log a goldilocks regime through a stretch where
# CAPE was already above the guard's own 40.0 block level.
FED_BS_EXPANDING = True
DEFICIT_GT_5PCT_GDP = True
CAPE_CURRENT = 42.0                  # multpl.com, 2026-08-10
TOP20_CONCENTRATION_PCT = 50.8       # JPMorgan, cited 2026-08-07 review
MACRO_FLAGS_REVIEWED = "2026-08-13"

DAILY_CSV = "logs/daily_log.csv"
WEEKLY_CSV = "logs/weekly_log.csv"
SUMMARY_DIR = "logs/summaries"


# ── Safe accessors ───────────────────────────────────────────────────────────
def _get(obj, name, default=None):
    """Attribute-or-key accessor. full_assessment() returns dataclasses in
    some paths and dicts in others; this tolerates both."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _try(fn, label: str, errors: list):
    """Run fn, capture any failure as a recorded error rather than a crash.

    A logger that dies on one bad feed loses the whole session — including
    the fields that WERE available. Partial data with named gaps beats none.
    """
    try:
        return fn()
    except Exception as e:
        errors.append(f"{label}: {type(e).__name__}: {e}")
        return None


# ── Daily ────────────────────────────────────────────────────────────────────
def run_daily(fred_key: str = "", force: bool = False) -> dict:
    """Execute Checklist v4 Part 1 and return the captured row."""
    status = mt.market_status()
    et_day = mt.et_date()
    errors: list[str] = []

    if not force and not mt.is_trading_day(et_day):
        return {"skipped": True, "reason": f"{et_day} is not a trading day "
                                           f"({status['status']})",
                "et_date": et_day.isoformat()}

    row: dict = {
        "et_date": et_day.isoformat(),
        "run_utc": mt.now_utc().isoformat(timespec="seconds"),
        "run_et": mt.fmt_et(),
        "minutes_since_close": status["minutes_since_close"],
    }

    # Step 0 — coverage gate. Runs FIRST: every flow reading depends on it.
    def _coverage():
        # flow_integrity.py audits the ETF creations/redemptions store, which
        # only exists in the Money_Flow repo. Running this logger from
        # Portfolio-Tracker or the Repression repo, the module is correctly
        # absent — that used to surface as a raw ModuleNotFoundError in the
        # errors field, which read like a bug rather than an expected,
        # repo-specific gap. Distinguish the two explicitly so a future log
        # review doesn't chase a non-issue.
        try:
            import flow_integrity as fi
        except ImportError:
            return {"flow_status": "N/A",
                    "flow_detail": "flow_integrity not applicable in this "
                                   "repo (ETF flow data lives in Money_Flow "
                                   "only)."}
        rep = fi.full_report()
        return {"flow_status": rep["status"],
                "flow_detail": rep["shares_movement"]["detail"],
                "flow_missing_days": len(rep["continuity"].get("missing", []))}
    row.update(_try(_coverage, "Step 0 coverage", errors) or
               {"flow_status": "UNKNOWN"})

    # Steps 1-6 — the macro log line, plus regime.
    def _regime():
        import regime_classifier as rc
        from fred_client import fetch_fred
        out = rc.full_assessment(fred_key, fetch_fred=fetch_fred,
                                 fed_bs_expanding=FED_BS_EXPANDING,
                                 deficit_gt_5pct_gdp=DEFICIT_GT_5PCT_GDP,
                                 cape=CAPE_CURRENT,
                                 top20_concentration_pct=TOP20_CONCENTRATION_PCT)
        sig, reg = out["signals"], out["regime"]
        sc, km = out["repression"], out["kmlm"]
        return {
            "regime": _get(reg, "key"),
            "regime_label": _get(reg, "label"),
            "short_real_rate": _get(sig, "short_real_rate"),
            "long_real_yield": _get(sig, "long_real_yield"),
            "long_real_mom_3m": _get(sig, "long_real_mom_3m"),
            "spread_2s10s": _get(sig, "spread_2s10s"),
            "hy_oas": _get(sig, "hy_oas"),
            "hy_oas_mom_2w": _get(sig, "hy_oas_mom_2w"),
            "breakeven_10y": _get(sig, "breakeven_10y"),
            "cpi_yoy": _get(sig, "cpi_yoy"),           # NSA, 12mo — feeds short_real_rate
            "cpi_3m_saar": _get(sig, "cpi_3m_saar"),    # SA, 3mo annualized — leading
            "cpi_mom_sa": _get(sig, "cpi_mom_sa"),      # SA, latest month
            "stock_bond_corr_60d": _get(sig, "stock_bond_corr_60d"),
            "repression_score": _get(sc, "score"),
            "repression_band": _get(sc, "band"),
            "top_weight": _get(sc, "top_weight_display"),
            "hollow": _get(sc, "hollow"),
            "kmlm_stance": _get(km, "stance"),
            "missing": ";".join(_get(sc, "missing", []) or []),
            "drivers": " | ".join(_get(reg, "drivers", []) or [])[:600],
        }
    row.update(_try(_regime, "Steps 1-6 regime", errors) or {})

    # Step 2 — VIX. Separate so a VIX failure doesn't lose the regime block.
    def _vix():
        import yfinance as yf
        h = yf.Ticker("^VIX").history(period="5d")
        return {"vix": round(float(h["Close"].iloc[-1]), 2)} if not h.empty else {}
    row.update(_try(_vix, "Step 2 VIX", errors) or {})

    # Step 3 — gold gate (Level 4).
    def _gold():
        import regime_classifier as rc
        ok = rc._gold_trend_ok(rc._inline_fetch_prices)
        return {"gold_gate": "PASS" if ok else "FAIL"}
    row.update(_try(_gold, "Step 3 gold gate", errors) or {})

    # Broad market context — what the framework's own signals are silent
    # about. Kept separate from the regime block so a price-fetch failure
    # cannot lose the macro capture.
    def _context():
        import market_context as mc
        ctx = mc.snapshot()
        # Sept 2026: snapshot() names its own gaps (stale bar, missing
        # tickers) but they never reached the log's errors field, so a
        # Monday row repeating Friday's tape read as clean.
        errors.extend(f"Market context: {e}" for e in ctx.get("errors", []))
        spy = ctx.get("indices", {}).get("SPY", {})
        qqq = ctx.get("indices", {}).get("QQQ", {})
        return {"_ctx": ctx,
                "spy_1d": spy.get("d1"), "spy_1w": spy.get("d5"),
                "qqq_1d": qqq.get("d1"), "qqq_1w": qqq.get("d5"),
                "sector_dispersion_1w": ctx.get("sector_dispersion_1w"),
                "breadth_divergence_1w": ctx.get("breadth_divergence_1w")}
    row.update(_try(_context, "Market context", errors) or {})

    # Exception alerts. An EMPTY list is the normal, expected outcome — the
    # daily job is not meant to produce narrative, only to surface a
    # pre-committed threshold crossing. See market_context.alerts().
    def _alerts():
        import market_context as mc
        fired = mc.alerts(row.get("_ctx") or {}, row)
        return {"alerts_fired": len(fired),
                "alert_max_level": (fired[0]["level"] if fired else "NONE"),
                "alert_rules": ";".join(a["rule"] for a in fired),
                "_alerts": fired}
    row.update(_try(_alerts, "Alerts", errors) or {})

    row["errors"] = " || ".join(errors) if errors else ""
    row["error_count"] = len(errors)

    # Private keys (leading underscore) are for the summary writer only and
    # must not bloat the CSV history.
    _csv_row = {k: v for k, v in row.items() if not k.startswith("_")}
    _append_csv(DAILY_CSV, _csv_row, key="et_date")
    _write_summary(row, kind="daily")
    return row


# ── Weekly ───────────────────────────────────────────────────────────────────
def run_weekly(fred_key: str = "") -> dict:
    """Execute Checklist v4 Part 2. Includes everything daily captures, plus
    drift, correlation, and the institutional-flow layer."""
    errors: list[str] = []
    row = run_daily(fred_key, force=True)
    if row.get("skipped"):
        row = {"et_date": mt.et_date().isoformat()}
    row["run_et"] = mt.fmt_et()
    row["week_of"] = mt.et_date().isoformat()

    # Step 9 — drift vs targets.
    def _targets():
        import regime_classifier as rc
        reg = row.get("regime") or "transition_ambiguous"
        w = rc.target_weights(reg, fetch_prices=rc._inline_fetch_prices)
        return {"targets_json": json.dumps({k: round(v, 1) for k, v in w.items()}),
                "growth_complex_pct": round(w.get("VGT", 0) + w.get("QQQ", 0)
                                            + w.get("SMH", 0), 1),
                "cash_like_pct": round(w.get("SGOV", 0) + w.get("USFR", 0), 1)}
    row.update(_try(_targets, "Step 9 targets", errors) or {})

    # Cross-asset divergence — the four-lane confirmation panel.
    def _xasset():
        import cross_asset as ca
        d = ca.divergence(
            hy_oas=row.get("hy_oas"), hy_mom_2w=row.get("hy_oas_mom_2w"),
            long_real=row.get("long_real_yield"),
            long_real_mom_3m=row.get("long_real_mom_3m"),
            breakeven=row.get("breakeven_10y"), vix=row.get("vix"))
        return {"xasset_majority": d.get("majority"),
                "xasset_agreement_pct": d.get("agreement_pct"),
                "xasset_dissenters": ",".join(d.get("dissenters", [])),
                "xasset_flag": (d.get("flag") or "")[:400]}
    row.update(_try(_xasset, "Cross-asset", errors) or {})

    # Step 5 — institutional flow. Gated on integrity, not just coverage.
    def _flow():
        import flow_integrity as fi
        rep = fi.full_report()
        if rep["status"] != "OK":
            return {"flow_readable": False,
                    "flow_note": f"{rep['status']}: {rep['headline']}"}
        return {"flow_readable": True, "flow_note": "integrity checks passed"}
    row.update(_try(_flow, "Step 5 flow", errors) or {})

    row["errors"] = " || ".join(errors) if errors else row.get("errors", "")
    row["error_count"] = len(errors)

    _append_csv(WEEKLY_CSV, row, key="week_of")
    _write_summary(row, kind="weekly")
    return row


# ── Persistence ──────────────────────────────────────────────────────────────
def _append_csv(path: str, row: dict, key: str):
    """Append idempotently — re-running a date replaces that row."""
    import pandas as pd
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(path):
        try:
            hist = pd.read_csv(path)
            if key in hist.columns:
                hist = hist[hist[key] != row.get(key)]
            new = pd.concat([hist, new], ignore_index=True)
        except Exception as e:
            print(f"[auto_log] could not merge {path}: {e}")
    if key in new.columns:
        new = new.sort_values(key)
    new.to_csv(path, index=False)
    print(f"[auto_log] wrote {path} ({len(new)} rows)")


def _fmt(v, suffix="%", dp=2):
    return "n/a" if v is None or v == "" else f"{float(v):+.{dp}f}{suffix}"


def _write_summary(row: dict, kind: str):
    """
    Write the human-readable analysis alongside the CSV row, plus a small
    JSON sidecar recording WHEN this run actually happened.

    Why the sidecar exists: the scheduled workflow's gate step decides
    whether to run again by checking "does today's report already exist" —
    but a bare file-existence check can't tell a proper 6:30pm end-of-day
    capture apart from an ad-hoc manual test run at, say, 12:35pm testing an
    unrelated code fix. On 2026-08-13 exactly that happened: a mid-day
    manual verification run wrote a report, the gate saw it and concluded
    "already done today," and the REAL 6:30pm scheduled run silently stood
    down for the rest of the day -- with no crash, no error, just nothing.
    The sidecar's et_hour lets the gate distinguish "a report exists" from
    "a report exists from within the proper window," so an early manual test
    can no longer suppress the evening capture.
    """
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    d = row.get("et_date", mt.et_date().isoformat())
    path = os.path.join(SUMMARY_DIR, f"{d}_{kind}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_summary(row, kind))
    print(f"[auto_log] wrote {path}")

    meta_path = os.path.join(SUMMARY_DIR, f"{d}_{kind}.meta.json")
    now = mt.now_et()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"et_date": d, "et_hour": now.hour,
                  "run_et": mt.fmt_et(now)}, f)
    print(f"[auto_log] wrote {meta_path}")


def build_summary(row: dict, kind: str = "daily") -> str:
    """
    Compose the analysis. Interpretation is rule-based and states its own
    caveats — it never asserts more confidence than the data supports.
    """
    L = []
    L.append(f"# {kind.title()} market log — {row.get('et_date','?')}")
    L.append("")
    L.append(f"*Generated {row.get('run_et','?')} · automated; observations "
             f"only, not trade instructions.*")
    L.append("")

    if row.get("skipped"):
        L.append(f"**Skipped:** {row.get('reason')}")
        return "\n".join(L)

    # Regime
    L.append("## Regime (Level 1)")
    reg, band = row.get("regime"), row.get("repression_band")
    score, tw = row.get("repression_score"), row.get("top_weight")
    hollow = row.get("hollow")
    L.append(f"- **Regime:** `{reg}` — {row.get('regime_label','')}")
    L.append(f"- **Score:** {score}/10 ({band}) · top-weight {tw}"
             + (" · **HOLLOW**" if hollow else ""))
    if hollow:
        L.append("  - A hollow score means every point comes from second-tier "
                 "components while both primary real-yield gauges are off. "
                 "Treat the band as an upper bound on the repression read.")
    if row.get("missing"):
        L.append(f"- ⚠ **Missing inputs:** {row['missing']} — this is an "
                 f"INCOMPLETE score, not necessarily a low one.")
    L.append("")

    # The log line
    L.append("## Daily log line")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Short real policy rate | {_fmt(row.get('short_real_rate'))} |")
    L.append(f"| DFII10 (long real) | {_fmt(row.get('long_real_yield'))} |")
    L.append(f"| DFII10 3m momentum | {_fmt(row.get('long_real_mom_3m'),'',3)} |")
    L.append(f"| 2s10s | {_fmt(row.get('spread_2s10s'))} |")
    L.append(f"| HY OAS | {_fmt(row.get('hy_oas'))} |")
    L.append(f"| VIX | {row.get('vix','n/a')} |")
    L.append(f"| CPI YoY (NSA) | {_fmt(row.get('cpi_yoy'))} |")
    L.append(f"| CPI 3M SAAR | {_fmt(row.get('cpi_3m_saar'))} |")
    L.append(f"| CPI MoM (SA) | {_fmt(row.get('cpi_mom_sa'), dp=2)} |")
    L.append(f"| Stock/bond 60d corr | {_fmt(row.get('stock_bond_corr_60d'),'',2)} |")
    L.append(f"| Gold gate | {row.get('gold_gate','n/a')} |")
    L.append(f"| KMLM stance | {row.get('kmlm_stance','n/a')} |")
    L.append("")

    # Daily = EXCEPTION report. Weekly = full narrative. This split is
    # deliberate: Checklist v4 makes daily observation and weekly decision,
    # and a daily essay manufactures significance on days that contain none.
    if kind == "daily":
        try:
            import market_context as mc
            L.append(mc.alert_block(row.get("_alerts") or []))
        except Exception:
            pass
    else:
        try:
            import market_context as mc
            if row.get("_ctx"):
                L.append(mc.context_block(row["_ctx"]))
        except Exception:
            pass

    # Rule-based interpretation
    L.append("## Interpretation")
    notes = []

    srr = row.get("short_real_rate")
    if srr is not None and srr != "":
        srr = float(srr)
        if abs(srr) < 0.25:
            notes.append(f"Short real rate {srr:+.2f}% is INSIDE the ±0.25% "
                         f"band — the primary gauge is silent, so no "
                         f"sign-dependent regime can be confirmed. Express "
                         f"neither the repression nor the reflation trade.")
        elif srr < 0:
            notes.append(f"Short real rate {srr:+.2f}% is decisively negative "
                         f"— the front-end repression channel is OPEN.")
        else:
            notes.append(f"Short real rate {srr:+.2f}% is decisively positive "
                         f"— savers are paid; the repression gauge is OFF.")

    hy = row.get("hy_oas")
    if hy not in (None, ""):
        hy = float(hy)
        if hy > 5.0:
            notes.append(f"🔴 **HY OAS {hy:.2f}% is beyond the 5.00% crisis "
                         f"line.** This is the one signal authorizing a "
                         f"same-day de-risk under the daily action rule.")
        elif hy > 3.5:
            notes.append(f"⚠ HY OAS {hy:.2f}% is above the 3.50% complacency "
                         f"line — credit is repricing risk.")
        else:
            notes.append(f"HY OAS {hy:.2f}% is below 3.50%. Read this as "
                         f"late-cycle risk COMPRESSION, not safety — it "
                         f"leaves little room to tighten and much to widen.")

    vix = row.get("vix")
    if vix not in (None, ""):
        vix = float(vix)
        if vix >= 35:
            notes.append(f"VIX {vix:.1f} ≥ 35 — trend-whipsaw zone. PAUSE "
                         f"KMLM adds; do not sell the hedge mid-crisis.")
        elif vix >= 30:
            notes.append(f"VIX {vix:.1f} ≥ 30 — de-risk equities.")
        elif vix < 20:
            notes.append(f"VIX {vix:.1f} — derivatives are not pricing stress, "
                         f"which also makes convexity comparatively cheap.")

    lr = row.get("long_real_yield")
    if lr not in (None, ""):
        lr = float(lr)
        if lr > 2.5:
            notes.append(f"DFII10 {lr:.2f}% is above 2.50% — long-duration "
                         f"equity multiples historically cannot ignore this.")

    if row.get("kmlm_stance") == "INCREASE KMLM":
        notes.append("KMLM signal reads INCREASE — the stock/bond hedge is "
                     "impaired. Fund from SGOV first, then growth; never from "
                     "metals or energy.")

    if row.get("flow_status") and row["flow_status"] != "OK":
        notes.append(f"🔴 **Flow layer {row['flow_status']}** — "
                     f"{row.get('flow_detail','')[:200]} Institutional flow "
                     f"panels are NOT readable; lead with COT and breadth.")

    if row.get("error_count"):
        notes.append(f"⚠ {row['error_count']} data error(s) this run — the "
                     f"log is partial. See the errors column.")

    for n in notes:
        L.append(f"- {n}")
    L.append("")

    if kind == "weekly":
        L.append("## Weekly additions")
        L.append(f"- **Target growth complex:** "
                 f"{row.get('growth_complex_pct','n/a')}% · "
                 f"**cash-like:** {row.get('cash_like_pct','n/a')}%")
        if row.get("xasset_majority"):
            L.append(f"- **Cross-asset:** majority {row['xasset_majority']}, "
                     f"agreement {row.get('xasset_agreement_pct')}%"
                     + (f", dissenting: {row['xasset_dissenters']}"
                        if row.get("xasset_dissenters") else ""))
        if row.get("xasset_flag"):
            L.append(f"  - {row['xasset_flag']}")
        L.append(f"- **Flow layer readable:** {row.get('flow_readable')} — "
                 f"{row.get('flow_note','')}")
        L.append("")
        L.append("**Manual steps this log cannot perform:** verify stops and "
                 "portfolio heat (Step 8), write each position's invalidation "
                 "sentence, compare live weights against the targets above "
                 "(Step 9), and run the stress tab (Step 10).")
        L.append("")

    L.append("---")
    L.append("")
    L.append("**Daily action rule:** the only trades authorized are the HY "
             "crisis override, a stop execution, and the three mechanical "
             "options rules. Everything else waits for the weekly review.")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    key = os.environ.get("FRED_API_KEY", "")
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"

    # LOG_FORCE lets a manual workflow_dispatch (which the YAML gate already
    # decided SHOULD run regardless of day) reach this script and override the
    # internal trading-day check too. Without it, `python auto_log.py daily` on
    # a weekend returns {"skipped": True} having written NOTHING — which then
    # makes the workflow's commit step fail on a logs/ directory that was never
    # created. run_weekly() already forces internally, so only daily was
    # affected. A scheduled cron never sets this, so weekend/holiday skipping
    # is unchanged in production.
    force = os.environ.get("LOG_FORCE", "").strip().lower() in ("1", "true", "yes")

    result = run_weekly(key) if mode == "weekly" else run_daily(key, force=force)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("targets_json", "drivers")},
                     indent=2, default=str))
