"""
flow_integrity.py  (v1 — August 2026)
─────────────────────────────────────
Detects the STALE SHARES OUTSTANDING failure before it becomes a full
history of fabricated zeros.

THE FAILURE THIS CATCHES
────────────────────────
On 2026-08-07 the ETF flow store held two sessions (Aug 5, Aug 7) across 20
tickers. Comparing them:

    prices changed:              20 of 20   ✓
    shares outstanding changed:   0 of 20   ✗

Zero of twenty, including QQQ, XLF (883M shares) and GLD. The probability
that no creation or redemption occurred across twenty major ETFs over two
sessions is effectively nil.

Root cause: _snapshot_one() sources shares outstanding from yfinance's
fast_info/.info. That field is NOT a daily series — yfinance has never
exposed shares outstanding at better than annual granularity. So the
module's founding assumption — that daily deltas in shares outstanding back
out creations and redemptions — is false for that data source.

WHY A DETECTOR AND NOT JUST A FIX
─────────────────────────────────
Because the failure is SILENT and gets HARDER to catch over time. Left
running, the store accumulates 20 sessions of d_shares == 0 for every
ticker. Every flow panel renders zeros. Zeros read as "no institutional
flows detected." coverage_report() flips to ready=True and CERTIFIES the
garbage. The "still collecting" excuse — the thing that made this visible
on day two — disappears.

A replacement fetcher can be written (see etf_flow_tracker.SHARES_SOURCES),
but no fetcher should be trusted without a standing check that the field it
returns actually MOVES. This module is that check. It is designed to run on
every poll and every dashboard render, permanently — not as a one-off
diagnostic.

DESIGN: FAIL LOUD, NEVER SILENT
───────────────────────────────
Every function returns a verdict dict with an explicit `status`. The
degenerate case (no movement at all) returns status=BROKEN, never a
reassuring zero. Callers must gate flow panels on status == OK.
"""

from __future__ import annotations

import pandas as pd

STATUS_OK = "OK"
STATUS_SUSPECT = "SUSPECT"
STATUS_BROKEN = "BROKEN"
STATUS_INSUFFICIENT = "INSUFFICIENT"

# If fewer than this share of tickers ever show a share-count change across
# the whole history, the field is not a live series. Real ETF share counts
# move constantly: over any multi-day window a large majority of a 20-ticker
# universe will have had at least one creation or redemption.
MIN_MOVERS_FRACTION = 0.25

# A single pair of sessions is noisy; below this many distinct dates we
# report INSUFFICIENT rather than guessing.
MIN_DATES = 2


def trustworthy_tickers(store: str = "data/etf_shares_history.csv",
                        min_sessions: int = 2) -> dict:
    """
    WHICH specific tickers have confirmed, moving shares_outstanding data —
    not just the aggregate count check_shares_move() reports.

    Added Sept 2026: check_shares_move()'s per-ticker loop already computes
    this internally to build its aggregate movers/total count, but never
    exposed the actual ticker-level result. That gap meant a genuinely
    strict, correct all-or-nothing gate (_tier_a_readable() in app.py) had
    no way to be anything OTHER than all-or-nothing — there was no per-
    ticker signal to build a graduated version from. This is that signal.

    Returns {"confirmed": {tickers with real movement}, "unconfirmed": {...},
             "never_seen": {tracked tickers absent from the store entirely}}.
    A ticker with genuinely ZERO rows ever is a DIFFERENT state from one
    that's present but static — conflating them was part of why 11 of 31
    tracked tickers could go missing from every diagnostic without a single
    logged reason.
    """
    # v2, Sept 2026: distinguish "too little history to judge" from
    # "genuinely static across enough sessions to be suspicious". A ticker
    # with exactly ONE recorded row will always show nunique<=1 trivially --
    # a single value cannot be "not unique" relative to itself -- so without
    # this check a brand-new source's first-ever successful row looked
    # identical to a confirmed-broken one. Caught directly from a real
    # dashboard screenshot where 11 tickers each showed "all 1 recorded
    # values identical", which is not evidence of staleness at all.
    out = {"confirmed": set(), "unconfirmed": set(), "insufficient": set(),
          "never_seen": set()}
    try:
        df = pd.read_csv(store)
    except Exception:
        return out

    seen = set(df["ticker"].unique()) if "ticker" in df.columns else set()
    try:
        from etf_flow_tracker import TRACKED
        out["never_seen"] = set(TRACKED) - seen
    except Exception:
        pass

    for tk in seen:
        g = df[df["ticker"] == tk].sort_values("date")
        if len(g) < min_sessions:
            out["insufficient"].add(tk)
        elif g["shares_outstanding"].nunique(dropna=True) > 1:
            out["confirmed"].add(tk)
        else:
            out["unconfirmed"].add(tk)
    return out


def check_shares_move(store: str = "data/etf_shares_history.csv") -> dict:
    """
    Verify shares_outstanding actually varies over time.

    This is the core integrity test. Returns a verdict dict, never raises.

        status              OK | SUSPECT | BROKEN | INSUFFICIENT
        movers / total      tickers whose share count ever changed
        price_movers        control group — proves the poll itself works
        detail / action     display strings
    """
    out = {"status": STATUS_INSUFFICIENT, "movers": 0, "total": 0,
           "price_movers": 0, "dates": 0, "detail": "", "action": ""}
    try:
        df = pd.read_csv(store)
    except Exception as e:
        out["detail"] = f"Cannot read {store}: {e}"
        out["action"] = "Confirm the poller has committed data."
        return out

    need = {"date", "ticker", "shares_outstanding", "price"}
    if not need.issubset(df.columns):
        out["detail"] = f"{store} missing columns: {need - set(df.columns)}"
        return out

    dates = sorted(df["date"].unique())
    out["dates"] = len(dates)
    tickers = sorted(df["ticker"].unique())
    out["total"] = len(tickers)

    if len(dates) < MIN_DATES:
        out["detail"] = (f"Only {len(dates)} session(s) stored — need at least "
                         f"{MIN_DATES} to test whether shares outstanding "
                         f"moves at all.")
        out["action"] = "Re-run this check after the next poll."
        return out

    movers = price_movers = 0
    for tk in tickers:
        g = df[df["ticker"] == tk].sort_values("date")
        if g["shares_outstanding"].nunique(dropna=True) > 1:
            movers += 1
        if g["price"].nunique(dropna=True) > 1:
            price_movers += 1
    out["movers"], out["price_movers"] = movers, price_movers

    frac = movers / len(tickers) if tickers else 0.0
    price_frac = price_movers / len(tickers) if tickers else 0.0

    # The control group matters: if prices move and shares don't, the poll
    # itself is healthy and the SHARES FIELD specifically is dead. That
    # distinction is what points at the data source rather than the pipeline.
    if movers == 0 and price_movers > 0:
        out["status"] = STATUS_BROKEN
        out["detail"] = (
            f"BROKEN: across {len(dates)} sessions, prices changed for "
            f"{price_movers}/{len(tickers)} tickers but shares outstanding "
            f"changed for {movers}/{len(tickers)}. The poll is working; the "
            f"shares_outstanding FIELD is static.")
        out["action"] = (
            "Do NOT wait for 20 sessions — the history would be all zeros and "
            "would render as 'no institutional flows'. Replace the shares "
            "source (see etf_flow_tracker.SHARES_SOURCES) before collecting "
            "further.")
        return out

    if movers == 0 and price_movers == 0:
        out["status"] = STATUS_BROKEN
        out["detail"] = (f"BROKEN: nothing moved across {len(dates)} sessions — "
                         f"neither price nor shares. The poll is returning a "
                         f"cached or duplicated snapshot.")
        out["action"] = "Check the poller and the data source together."
        return out

    if frac < MIN_MOVERS_FRACTION:
        out["status"] = STATUS_SUSPECT
        out["detail"] = (
            f"SUSPECT: only {movers}/{len(tickers)} tickers "
            f"({frac:.0%}) ever showed a share-count change across "
            f"{len(dates)} sessions, vs {price_frac:.0%} for price. Expect a "
            f"large majority over any multi-day window.")
        out["action"] = ("Verify the shares source updates daily before "
                         "trusting any flow reading.")
        return out

    out["status"] = STATUS_OK
    out["detail"] = (f"OK: {movers}/{len(tickers)} tickers ({frac:.0%}) show "
                     f"share-count movement across {len(dates)} sessions. The "
                     f"field is live.")
    return out


def check_poll_continuity(store: str = "data/etf_shares_history.csv",
                          expected_weekdays: bool = True) -> dict:
    """
    Detect missed polls. Free sources expose shares outstanding as a CURRENT
    snapshot, so a missed weekday is permanently lost history — it cannot be
    backfilled later.
    """
    out = {"status": STATUS_INSUFFICIENT, "dates": 0, "missing": [],
           "detail": "", "action": ""}
    try:
        df = pd.read_csv(store)
        dates = sorted(pd.to_datetime(df["date"].unique()))
    except Exception as e:
        out["detail"] = f"Cannot read {store}: {e}"
        return out

    out["dates"] = len(dates)
    if len(dates) < 2:
        out["detail"] = "Need at least 2 sessions to check continuity."
        return out

    span = pd.bdate_range(dates[0], dates[-1])
    have = {d.date() for d in dates}
    missing = [d.date().isoformat() for d in span if d.date() not in have]
    out["missing"] = missing

    if missing:
        out["status"] = STATUS_SUSPECT
        out["detail"] = (f"{len(missing)} weekday(s) missing between "
                         f"{dates[0]:%Y-%m-%d} and {dates[-1]:%Y-%m-%d}: "
                         f"{', '.join(missing[:8])}"
                         f"{'…' if len(missing) > 8 else ''}")
        out["action"] = ("Check the Actions run history for failed polls. "
                         "Missed days CANNOT be backfilled — the source only "
                         "exposes a current snapshot.")
    else:
        out["status"] = STATUS_OK
        out["detail"] = (f"No gaps: all {len(span)} weekdays present between "
                         f"{dates[0]:%Y-%m-%d} and {dates[-1]:%Y-%m-%d}.")
    return out


def full_report(store: str = "data/etf_shares_history.csv") -> dict:
    """Both checks plus a single overall verdict for the coverage banner."""
    moves = check_shares_move(store)
    cont = check_poll_continuity(store)

    if moves["status"] == STATUS_BROKEN:
        overall, headline = STATUS_BROKEN, (
            "Flow layer BROKEN — shares outstanding is not a live series. "
            "Every flow reading would be a fabricated zero.")
    elif STATUS_SUSPECT in (moves["status"], cont["status"]):
        overall, headline = STATUS_SUSPECT, (
            "Flow layer SUSPECT — verify the data source before acting on "
            "any flow signal.")
    elif moves["status"] == STATUS_INSUFFICIENT:
        overall, headline = STATUS_INSUFFICIENT, (
            "Not enough sessions to validate the flow layer yet.")
    else:
        overall, headline = STATUS_OK, "Flow layer integrity checks passed."

    return {"status": overall, "headline": headline,
            "shares_movement": moves, "continuity": cont}


def gate_ok(store: str = "data/etf_shares_history.csv") -> bool:
    """True only when flow panels are safe to render."""
    return full_report(store)["status"] == STATUS_OK


def render(st, store: str = "data/etf_shares_history.csv") -> dict:
    """Render the integrity verdict. Import-safe; `st` is passed in."""
    rep = full_report(store)
    m, c = rep["shares_movement"], rep["continuity"]

    if rep["status"] == STATUS_BROKEN:
        st.error(f"**{rep['headline']}**\n\n{m['detail']}\n\n"
                 f"**Action:** {m['action']}")
    elif rep["status"] == STATUS_SUSPECT:
        # v3, Sept 2026: ATTRIBUTE the verdict to the specific sub-check
        # that actually triggered it. full_report() combines TWO independent
        # checks (shares movement, poll continuity) into one overall status
        # via `SUSPECT in (moves.status, cont.status)` -- either one alone
        # can produce the SUSPECT headline. Previously this box always
        # displayed the SHARES sub-check's own detail text regardless of
        # WHICH check actually failed, so a case where shares movement was
        # genuinely OK (above its own 25% threshold) but continuity had
        # gaps rendered as "SUSPECT" next to text that literally said "OK:
        # 20/31 tickers... show share-count movement" -- correct in
        # isolation, deeply confusing in context. Both sub-checks are now
        # shown separately, each labeled with its OWN status.
        st.warning(f"**{rep['headline']}**")
        _m_ok = m["status"] != STATUS_SUSPECT and m["status"] != STATUS_BROKEN
        _c_ok = c["status"] != STATUS_SUSPECT
        st.markdown(f"{'✅' if _m_ok else '⚠️'} **Shares movement:** {m['detail']}")
        if c.get("detail"):
            st.markdown(f"{'✅' if _c_ok else '⚠️'} **Poll continuity:** {c['detail']}")
        if _m_ok and not _c_ok:
            st.caption(
                "The shares-movement check PASSED on its own — this SUSPECT "
                "verdict is driven entirely by missing polling days, not by "
                "data quality. A 65% movement rate above this check's 25% "
                "threshold is a genuine pass, separate from whether the "
                "poll ran every day."
            )
        elif not _m_ok and _c_ok:
            st.caption(
                "Polling ran on schedule — this SUSPECT verdict is driven "
                "entirely by too few tickers showing real movement, not by "
                "missing days."
            )

        # ── WHICH tickers, by name — added Sept 2026 ─────────────────────────
        # The aggregate "20/31 (65%)" figure alone can't answer the question
        # that actually matters: is 20/31 a STABLE, explicable state (some
        # funds' data genuinely isn't available a given way) or a bug? Two
        # consecutive fresh runs landing on the identical 20/31 is itself
        # informative -- a bug from network flakiness would likely vary run
        # to run; a fixed data-availability limit per ticker would not. This
        # makes that distinguishable by showing WHICH ones, not just how many.
        trust = trustworthy_tickers(store)
        if trust["unconfirmed"] or trust["never_seen"] or trust["insufficient"]:
            with st.expander(
                f"Which tickers, specifically ({len(trust['unconfirmed'])} "
                f"unconfirmed, {len(trust['insufficient'])} too new to judge, "
                f"{len(trust['never_seen'])} never seen)",
                expanded=True):
                if trust["insufficient"]:
                    st.markdown("**Too new to judge yet (1 session only):**")
                    st.caption(
                        "A single data point cannot be 'static' -- there is "
                        "nothing yet to compare it against. This is NOT "
                        "evidence of a problem; it means the next poll run "
                        "will be the first real test for these tickers."
                    )
                    for tk in sorted(trust["insufficient"]):
                        st.caption(f"○ {tk}")
                errs = {}
                try:
                    from etf_flow_tracker import load_last_run_errors
                    errs = load_last_run_errors(store).get("errors", {})
                except Exception:
                    pass

                if trust["unconfirmed"]:
                    st.markdown("**Present but not moving:**")
                    try:
                        raw = pd.read_csv(store)
                    except Exception:
                        raw = None
                    for tk in sorted(trust["unconfirmed"]):
                        reason = errs.get(tk)
                        if reason:
                            st.caption(f"○ {tk} — {reason[-1]}")
                        elif raw is not None:
                            # No logged error means the source likely
                            # returned a value WITHOUT raising -- show what
                            # it actually returned and from where, since
                            # "no error" and "silently stale" look identical
                            # otherwise. If shares_source is aum_implied and
                            # the value truly never varies, that is very
                            # likely the SAME staleness problem one layer
                            # down: totalAssets may ALSO only update
                            # periodically for these specific funds, not
                            # daily -- worth confirming directly rather than
                            # assuming this source is fixed just because it
                            # returns something without erroring.
                            g = raw[raw["ticker"] == tk].sort_values("date")
                            if len(g):
                                src = g["shares_source"].iloc[-1] if "shares_source" in g else "?"
                                vals = g["shares_outstanding"]
                                n_unique = vals.nunique(dropna=True)
                                if n_unique == 1:
                                    summary = (f"all {len(vals)} recorded values "
                                              f"identical: {vals.iloc[-1]:,.0f}")
                                else:
                                    summary = (f"{n_unique} distinct values across "
                                              f"{len(vals)} sessions, latest: "
                                              f"{vals.iloc[-1]:,.0f}")
                                st.caption(
                                    f"○ {tk} — no error logged, source: "
                                    f"**{src}**. {summary}")
                            else:
                                st.caption(f"○ {tk} — no error logged, no rows found.")
                        else:
                            st.caption(f"○ {tk} — no error logged; the value "
                                     f"returned is simply identical every "
                                     f"session so far.")
                if trust["never_seen"]:
                    st.markdown("**Never appeared in the store at all:**")
                    for tk in sorted(trust["never_seen"]):
                        reason = errs.get(tk)
                        st.caption(f"○ {tk}" + (f" — {reason[-1]}" if reason else ""))
                st.caption(
                    "If this SAME list repeats across multiple fresh runs "
                    "(not just multiple dashboard reloads of one run), that "
                    "is a stable per-ticker data-availability limit, not a "
                    "transient bug — worth confirming before spending more "
                    "effort chasing it as one."
                )
    elif rep["status"] == STATUS_OK:
        st.success(f"**{rep['headline']}** {m['detail']}")
    else:
        st.info(f"{rep['headline']} {m['detail']}")
    return rep


if __name__ == "__main__":
    import json
    print(json.dumps(full_report(), indent=2, default=str))
