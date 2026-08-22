"""
log_viewer.py  (v1 — August 2026)
─────────────────────────────────
The REVIEW surface for auto_log.py's output.

WHY THIS EXISTS
───────────────
auto_log.py captures and writes; it had no reader. The reports landed in
three places, none of which is where you actually look:

    logs/summaries/YYYY-MM-DD_daily.md   in the repo
    logs/daily_log.csv                    machine-readable history
    the Actions run page                  buried, and only for that one run

Archival is not review. This module renders the same reports inside the
dashboard, so the daily/weekly read happens where every other signal already
lives.

THE STALENESS PROBLEM, AND WHY IT READS FROM GITHUB FIRST
─────────────────────────────────────────────────────────
Streamlit Community Cloud serves a checkout of the repo taken at deploy time.
GitHub Actions commits new logs AFTER that checkout, so a running app can be
hours or days behind — and would silently render an old report as though it
were today's. That is the same class of failure as a stale timestamp or a
frozen catalyst: plausible, wrong, and invisible.

So the read order is deliberately GITHUB FIRST, local second:

    1. GitHub Contents API  — always current, needs GITHUB_TOKEN/GITHUB_REPO
    2. local filesystem     — the deploy-time checkout, possibly stale
    3. nothing              — say so loudly

Whichever source answers, the UI states WHICH one and how old the report is.
A report with no provenance is a report you cannot trust.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

LOG_DIR = "logs"
SUMMARY_DIR = "logs/summaries"
DAILY_CSV = "logs/daily_log.csv"
WEEKLY_CSV = "logs/weekly_log.csv"

SRC_GITHUB = "GitHub (current)"
SRC_LOCAL = "local checkout (may be stale)"
SRC_NONE = "unavailable"


# ── Sourcing ─────────────────────────────────────────────────────────────────
def _gh_list(path: str) -> list[str] | None:
    """Filenames in a repo directory via the Contents API. None if unavailable."""
    try:
        import requests
        import storage_backend as sb
        token, repo = sb._cfg("GITHUB_TOKEN"), sb._cfg("GITHUB_REPO")
        branch = sb._cfg("GITHUB_BRANCH", "main")
        if not token or not repo:
            return None
        r = requests.get(
            f"https://api.github.com/repos/{repo}/contents/{path}",
            headers=sb._gh_headers(token), params={"ref": branch}, timeout=20)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return [i["name"] for i in r.json() if i.get("type") == "file"]
    except Exception as e:
        print(f"[log_viewer] github list failed: {e}")
        return None


def _gh_read(path: str) -> str | None:
    try:
        import storage_backend as sb
        content, _ = sb._gh_get(path)
        return content
    except Exception as e:
        print(f"[log_viewer] github read failed: {e}")
        return None


def list_reports() -> tuple[list[str], str]:
    """(filenames, source). GitHub first so a stale checkout can't masquerade."""
    names = _gh_list(SUMMARY_DIR)
    if names is not None and names:
        return sorted(names, reverse=True), SRC_GITHUB
    if os.path.isdir(SUMMARY_DIR):
        local = sorted(os.listdir(SUMMARY_DIR), reverse=True)
        if local:
            return [n for n in local if n.endswith(".md")], SRC_LOCAL
    return [], SRC_NONE


def read_report(filename: str) -> tuple[str | None, str]:
    path = f"{SUMMARY_DIR}/{filename}"
    txt = _gh_read(path)
    if txt:
        return txt, SRC_GITHUB
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read(), SRC_LOCAL
    except Exception as e:
        print(f"[log_viewer] local read failed: {e}")
    return None, SRC_NONE


def read_history(kind: str = "daily"):
    """The CSV history as a DataFrame. None when unavailable."""
    import io
    import pandas as pd
    path = DAILY_CSV if kind == "daily" else WEEKLY_CSV
    txt = _gh_read(path)
    if txt:
        try:
            return pd.read_csv(io.StringIO(txt)), SRC_GITHUB
        except Exception:
            pass
    try:
        if os.path.exists(path):
            return pd.read_csv(path), SRC_LOCAL
    except Exception as e:
        print(f"[log_viewer] history read failed: {e}")
    return None, SRC_NONE


def _age_days(filename: str) -> int | None:
    m = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
    if not m:
        return None
    try:
        import market_time as mt
        today = mt.et_date()
    except Exception:
        today = datetime.now().date()
    return (today - datetime.strptime(m.group(1), "%Y-%m-%d").date()).days


# ── Render ───────────────────────────────────────────────────────────────────
def render(st):
    """Render the Logs & Analysis tab."""
    st.markdown("## 📋 Daily & Weekly Analysis")
    st.caption(
        "Reports generated automatically by the scheduled checklist run — "
        "weekdays 6:30 PM ET (daily) and Sundays 6:00 AM ET (weekly). "
        "Daily is an EXCEPTION report: no alerts is the expected outcome."
    )

    reports, source = list_reports()

    if source == SRC_NONE:
        st.error(
            "**No reports found.** The scheduled log has not committed to "
            "`logs/summaries/` yet.\n\n"
            "Check: Actions → *Scheduled Checklist Logs* → has it run "
            "successfully? A run can succeed while writing nothing if it "
            "landed on a non-trading day without `LOG_FORCE`."
        )
        return

    if source == SRC_LOCAL:
        st.warning(
            "⚠ Reading from the **deploy-time checkout**, not live GitHub. "
            "Reports committed since this app last redeployed will NOT appear. "
            "Set `GITHUB_TOKEN` and `GITHUB_REPO` in Streamlit secrets to read "
            "current logs."
        )

    daily = [r for r in reports if r.endswith("_daily.md")]
    weekly = [r for r in reports if r.endswith("_weekly.md")]

    view = st.radio("View", ["Latest daily", "Latest weekly", "Browse history",
                             "Trends"], horizontal=True, label_visibility="collapsed")

    if view == "Latest daily":
        _render_one(st, daily, source, "daily")
    elif view == "Latest weekly":
        _render_one(st, weekly, source, "weekly")
    elif view == "Browse history":
        pick = st.selectbox("Select a report", reports)
        if pick:
            txt, src = read_report(pick)
            age = _age_days(pick)
            st.caption(f"Source: {src}"
                       + (f" · {age} day(s) old" if age is not None else ""))
            st.markdown(txt or "*Could not read this report.*")
    else:
        _render_trends(st)


def _render_one(st, files: list[str], source: str, kind: str):
    if not files:
        st.info(f"No {kind} report yet. The first one will appear after the "
                f"next scheduled {kind} run.")
        return
    latest = files[0]
    txt, src = read_report(latest)
    age = _age_days(latest)

    # Staleness is stated, never implied. A report rendered without its age
    # reads as current regardless of when it was written.
    if age is not None:
        if kind == "daily" and age > 3:
            st.warning(f"⚠ Latest daily report is **{age} days old** "
                       f"({latest}). The scheduled run may have stopped — "
                       f"check the Actions history.")
        elif kind == "weekly" and age > 9:
            st.warning(f"⚠ Latest weekly report is **{age} days old** "
                       f"({latest}).")
        else:
            st.caption(f"Source: {src} · {latest} · "
                       + ("today" if age == 0 else f"{age} day(s) ago"))

    st.markdown(txt or "*Could not read this report.*")

    if txt:
        st.download_button("⬇ Download this report", txt, file_name=latest,
                           mime="text/markdown")


def _render_trends(st):
    """Chart the captured history — the reason the CSV exists at all."""
    df, src = read_history("daily")
    if df is None or df.empty:
        st.info("No daily history yet. Trends appear once the scheduled run "
                "has captured a few sessions.")
        return

    st.caption(f"Source: {src} · {len(df)} session(s) captured")

    if len(df) < 3:
        st.info(f"Only {len(df)} session(s) captured. Trend charts need a few "
                f"more before they mean anything — a two-point line is not a "
                f"trend.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    plot_cols = [c for c in ["hy_oas", "vix", "long_real_yield",
                             "short_real_rate", "repression_score",
                             "spread_2s10s"] if c in df.columns]
    if not plot_cols:
        st.warning("No plottable columns found in the history.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    pick = st.multiselect("Series", plot_cols,
                          default=[c for c in ("hy_oas", "vix") if c in plot_cols])
    if pick and "et_date" in df.columns:
        import pandas as pd
        d = df.copy()
        d["et_date"] = pd.to_datetime(d["et_date"], errors="coerce")
        d = d.dropna(subset=["et_date"]).set_index("et_date")
        st.line_chart(d[pick])

    # Alert history is the highest-signal column: it shows which tripwires
    # actually fired over time. A rule that has NEVER fired is mis-specified,
    # not selective — the same diagnostic used for the volume tiers.
    if "alerts_fired" in df.columns:
        total = int(df["alerts_fired"].fillna(0).sum())
        st.markdown(f"**Alerts fired across {len(df)} sessions: {total}**")
        if "alert_rules" in df.columns:
            rules = [r for s in df["alert_rules"].dropna()
                     for r in str(s).split(";") if r]
            if rules:
                import collections
                counts = collections.Counter(rules)
                st.markdown("Most frequent: " + " · ".join(
                    f"`{k}` ×{v}" for k, v in counts.most_common(5)))
            elif len(df) >= 20:
                st.caption(
                    "No alert has fired in 20+ sessions. That is plausible in "
                    "a calm tape, but worth re-reading the thresholds in "
                    "`market_context.ALERT_RULES` — a rule that never fires is "
                    "mis-specified, not selective."
                )

    with st.expander("Raw history"):
        st.dataframe(df, use_container_width=True, hide_index=True)


def selftest() -> dict:
    reports, source = list_reports()
    return {"ok": True, "reports_found": len(reports), "source": source,
            "latest": reports[0] if reports else None}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, default=str))
