"""
market_context.py  (v1 — August 2026)
─────────────────────────────────────
Broad market context the three dashboards do NOT cover, plus the
exception-based alert detector that decides whether a given day is worth
your attention at all.

WHY THIS IS SEPARATE FROM auto_log.py
─────────────────────────────────────
auto_log.py captures the FRAMEWORK's own signals — regime, repression score,
gates, the checklist log line. That is the system observing itself.

This module captures what the framework is silent about: index levels, sector
dispersion, breadth, and the cross-asset complex. A regime read with no idea
what the tape actually did is missing the thing every human analyst starts
with.

THE CADENCE ARGUMENT, ENCODED
─────────────────────────────
Checklist v4 is explicit that daily is OBSERVATION and weekly is DECISION,
and that the daily action rule authorizes only three trades. A daily
narrative summary would work against that: it manufactures the feeling of
significance on days that contain none, and creates pressure to act when the
framework has already decided you shouldn't.

So this module does NOT write a daily essay. It runs `alerts()`, which
returns an empty list on a normal day. Silence is the intended output.
Something appearing means a threshold you pre-committed to was crossed.

Weekly is where the full narrative belongs, and `context_block()` feeds it.

THRESHOLDS ARE PRE-COMMITMENTS
──────────────────────────────
Every threshold below is set in advance, in code, where it can be reviewed
when calm. That is the entire point — deciding after the number arrives is
narrative-fitting. Each one is traceable to a rule already in the checklist
rather than invented here.
"""

from __future__ import annotations

from typing import Optional

import market_time as mt

# ── Universe ─────────────────────────────────────────────────────────────────
INDICES = {
    "SPY": "S&P 500", "QQQ": "Nasdaq-100", "IWM": "Russell 2000",
    "DIA": "Dow 30", "RSP": "S&P 500 equal-weight",
}
SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Healthcare", "XLI": "Industrials", "XLY": "Cons. Disc.",
    "XLP": "Cons. Staples", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communications",
}
CROSS_ASSET = {
    "GLD": "Gold", "USO": "Oil (WTI)", "TLT": "20y+ Treasuries",
    "UUP": "US Dollar", "HYG": "High Yield credit", "SMH": "Semiconductors",
}

# ── Alert thresholds — each traceable to an existing checklist rule ──────────
ALERT_RULES = {
    # Checklist Daily Step 1. The ONLY signal authorizing a same-day de-risk.
    "hy_oas_crisis": 5.00,
    "hy_oas_watch": 3.50,
    "hy_oas_velocity_2w": 0.50,      # widening >50bp/2wk
    # Daily Step 2. 30 = de-risk; 35 = trend-whipsaw, pause KMLM adds.
    "vix_derisk": 30.0,
    "vix_whipsaw": 35.0,
    # Daily Step 3.
    "dfii10_level": 2.50,
    # Daily Step 4 — the band boundary itself.
    "short_real_band": 0.25,
    # New: CPI 3M SAAR (leading) vs CPI YoY NSA (lagging) divergence. Not a
    # daily-actionable trigger -- this is an EARLY-WARNING context flag,
    # same severity tier as the band-crossing INFO alert below. 1.5pp is
    # deliberately wide: MoM/SAAR noise is real, and the point is to catch
    # a genuine multi-month inflection, not chase a single hot or cool print.
    "cpi_saar_divergence": 1.5,
    # An index move large enough that "nothing happened" is false.
    "index_move_1d": 2.0,
    # Sector dispersion: rotation vs uniform move.
    "sector_dispersion": 4.0,
    # Breadth: cap-weight vs equal-weight divergence over a week.
    "breadth_divergence_1w": 2.0,
}


def _fetch(tickers: list[str], period: str = "3mo"):
    """Adjusted closes for a ticker list. Returns None on any failure —
    never a partial frame silently treated as complete."""
    try:
        import yfinance as yf
        df = yf.download(tickers, period=period, progress=False,
                         auto_adjust=True)["Close"]
        if hasattr(df, "to_frame"):
            df = df.to_frame() if len(tickers) == 1 else df
        return df.dropna(how="all")
    except Exception as e:
        print(f"[market_context] fetch failed: {e}")
        return None


def _pct(series, days: int) -> Optional[float]:
    try:
        s = series.dropna()
        if len(s) < days + 1:
            return None
        return round(float(s.iloc[-1] / s.iloc[-1 - days] - 1) * 100, 2)
    except Exception:
        return None


def snapshot() -> dict:
    """
    Current market context. Every field is None-safe; a failed fetch produces
    a named gap rather than a fabricated zero.
    """
    out = {"et_date": mt.et_date().isoformat(), "asof": mt.fmt_et(),
           "indices": {}, "sectors": {}, "cross_asset": {}, "errors": []}

    all_tk = list(INDICES) + list(SECTORS) + list(CROSS_ASSET)
    df = _fetch(all_tk)
    if df is None or df.empty:
        out["errors"].append("price fetch returned nothing")
        return out

    for group, universe in (("indices", INDICES), ("sectors", SECTORS),
                            ("cross_asset", CROSS_ASSET)):
        for tk, name in universe.items():
            if tk not in df.columns:
                out["errors"].append(f"{tk} missing from fetch")
                continue
            s = df[tk]
            out[group][tk] = {
                "name": name,
                "last": round(float(s.dropna().iloc[-1]), 2) if len(s.dropna()) else None,
                "d1": _pct(s, 1), "d5": _pct(s, 5),
                "d21": _pct(s, 21), "d63": _pct(s, 63),
            }

    # Derived: sector dispersion (rotation vs uniform move) and breadth.
    week = [v["d5"] for v in out["sectors"].values() if v.get("d5") is not None]
    if week:
        out["sector_dispersion_1w"] = round(max(week) - min(week), 2)
        best = max(out["sectors"].items(), key=lambda kv: kv[1].get("d5") or -1e9)
        worst = min(out["sectors"].items(), key=lambda kv: kv[1].get("d5") or 1e9)
        out["sector_best_1w"] = {"ticker": best[0], "name": best[1]["name"],
                                 "d5": best[1]["d5"]}
        out["sector_worst_1w"] = {"ticker": worst[0], "name": worst[1]["name"],
                                  "d5": worst[1]["d5"]}

    spy, rsp = out["indices"].get("SPY", {}), out["indices"].get("RSP", {})
    if spy.get("d5") is not None and rsp.get("d5") is not None:
        out["breadth_divergence_1w"] = round(spy["d5"] - rsp["d5"], 2)
    return out


def alerts(ctx: dict, signals: Optional[dict] = None) -> list[dict]:
    """
    Return ONLY threshold crossings. An empty list is the normal, expected
    output — it means the day contained nothing your pre-committed rules
    care about.

    `signals` is an optional dict of framework values (hy_oas, vix,
    long_real_yield, short_real_rate) from auto_log's daily row, so the
    macro tripwires can be evaluated alongside the tape.
    """
    fired: list[dict] = []
    s = signals or {}
    R = ALERT_RULES

    def add(level, rule, msg, action=""):
        fired.append({"level": level, "rule": rule, "message": msg,
                      "action": action})

    # ── Credit: the one daily-actionable tripwire ───────────────────────────
    hy = s.get("hy_oas")
    if hy is not None:
        hy = float(hy)
        vel = float(s.get("hy_oas_mom_2w") or 0)
        if hy > R["hy_oas_crisis"] and vel > R["hy_oas_velocity_2w"]:
            add("CRITICAL", "hy_oas_crisis",
                f"HY OAS {hy:.2f}% is beyond the {R['hy_oas_crisis']:.2f}% "
                f"crisis line AND widening {vel:+.2f}%/2wk.",
                "CRISIS OVERRIDE — this is the one signal authorizing a "
                "same-day de-risk: trim growth, switch TLT on, raise cash. "
                "Beats the inflation regime.")
        elif hy > R["hy_oas_crisis"]:
            add("CRITICAL", "hy_oas_crisis",
                f"HY OAS {hy:.2f}% is beyond the {R['hy_oas_crisis']:.2f}% "
                f"crisis line.", "Verify velocity, then apply the override.")
        elif hy > R["hy_oas_watch"]:
            add("WARNING", "hy_oas_watch",
                f"HY OAS {hy:.2f}% crossed the {R['hy_oas_watch']:.2f}% "
                f"complacency line — credit is repricing risk.",
                "Not same-day actionable. Flag for the weekly review.")

    # ── Volatility ──────────────────────────────────────────────────────────
    vix = s.get("vix") or (ctx.get("cross_asset", {}).get("VIX", {}) or {}).get("last")
    if vix is not None:
        vix = float(vix)
        if vix >= R["vix_whipsaw"]:
            add("WARNING", "vix_whipsaw",
                f"VIX {vix:.1f} ≥ {R['vix_whipsaw']:.0f} — trend-whipsaw zone.",
                "PAUSE KMLM adds. Do NOT sell the hedge mid-crisis.")
        elif vix >= R["vix_derisk"]:
            add("WARNING", "vix_derisk",
                f"VIX {vix:.1f} ≥ {R['vix_derisk']:.0f}.", "De-risk equities.")

    # ── Long real yield ─────────────────────────────────────────────────────
    lr = s.get("long_real_yield")
    if lr is not None and float(lr) > R["dfii10_level"]:
        add("WARNING", "dfii10_level",
            f"DFII10 {float(lr):.2f}% above {R['dfii10_level']:.2f}%.",
            "Long-duration equity multiples historically cannot ignore this. "
            "Weekly review item.")

    # ── The band boundary — a genuine regime change, not noise ──────────────
    srr = s.get("short_real_rate")
    if srr is not None:
        srr = float(srr)
        b = R["short_real_band"]
        if abs(srr) > b:
            side = "POSITIVE" if srr > 0 else "NEGATIVE"
            add("INFO", "short_real_band",
                f"Short real rate {srr:+.2f}% has CLEARED the ±{b:.2f}% band "
                f"({side}).",
                "A confirmed band crossing is a real signal. Per the "
                "regime-transition protocol, require TWO consecutive daily "
                "closes before executing, then cuts first, hedges second, "
                "adds last.")

    # ── CPI 3M SAAR vs YoY divergence — inflation inflection early-warning ──
    # SAAR is the leading (SA, 3mo annualized) read; YoY (NSA) is the
    # lagging one that actually feeds short_real_rate. A wide gap between
    # them means recent months are running meaningfully hotter or cooler
    # than the trailing 12-month figure reflects -- exactly the situation
    # where the slower number is about to start moving.
    saar, yoy = s.get("cpi_3m_saar"), s.get("cpi_yoy")
    if saar is not None and yoy is not None:
        saar, yoy = float(saar), float(yoy)
        gap = saar - yoy
        thresh = R["cpi_saar_divergence"]
        if abs(gap) >= thresh:
            direction = "hotter" if gap > 0 else "cooler"
            add("INFO", "cpi_saar_divergence",
                f"CPI 3M SAAR ({saar:+.2f}%) is running {abs(gap):.1f}pp "
                f"{direction} than CPI YoY NSA ({yoy:+.2f}%).",
                f"The trailing YoY figure hasn't caught up to the recent "
                f"pace yet. {'Watch for the short real rate to move toward decisively negative as this feeds through.' if gap > 0 else 'Watch for the short real rate to move toward decisively positive as this feeds through.'} "
                f"Not same-day actionable — flag for the weekly review.")

    # ── Tape: only when the move is large enough that "nothing happened"
    #    would be false ────────────────────────────────────────────────────
    spy = ctx.get("indices", {}).get("SPY", {})
    if spy.get("d1") is not None and abs(spy["d1"]) >= R["index_move_1d"]:
        add("INFO", "index_move_1d",
            f"S&P 500 moved {spy['d1']:+.2f}% today.",
            "Context only — no action authorized by a price move alone.")

    disp = ctx.get("sector_dispersion_1w")
    if disp is not None and disp >= R["sector_dispersion"]:
        b, w = ctx.get("sector_best_1w", {}), ctx.get("sector_worst_1w", {})
        add("INFO", "sector_dispersion",
            f"Sector dispersion {disp:.1f}pts this week "
            f"({b.get('name')} {b.get('d5'):+.1f}% vs "
            f"{w.get('name')} {w.get('d5'):+.1f}%).",
            "Wide dispersion = genuine rotation rather than a uniform move. "
            "Cross-check against COT before treating it as capital flow.")

    bd = ctx.get("breadth_divergence_1w")
    if bd is not None and abs(bd) >= R["breadth_divergence_1w"]:
        if bd > 0:
            add("WARNING", "breadth_divergence",
                f"Cap-weight beat equal-weight by {bd:.1f}pts this week — "
                f"the index is being carried by its largest names.",
                "Narrow breadth. With top-20 concentration already at "
                "historic highs, this compounds index-level single-theme "
                "risk.")
        else:
            add("INFO", "breadth_divergence",
                f"Equal-weight beat cap-weight by {abs(bd):.1f}pts — "
                f"broadening participation.")

    return fired


def context_block(ctx: dict) -> str:
    """Markdown market-context section for the WEEKLY analysis."""
    if not ctx.get("indices"):
        return "## Market context\n\n*Unavailable — price fetch failed.*\n"

    L = ["## Market context", ""]
    L.append("| | 1D | 1W | 1M | 3M |")
    L.append("|---|---|---|---|---|")
    for tk, v in ctx["indices"].items():
        f = lambda x: "n/a" if x is None else f"{x:+.1f}%"
        L.append(f"| **{v['name']}** | {f(v['d1'])} | {f(v['d5'])} | "
                 f"{f(v['d21'])} | {f(v['d63'])} |")
    L.append("")

    ranked = sorted(
        [(v["name"], v["d5"]) for v in ctx["sectors"].values()
         if v.get("d5") is not None],
        key=lambda x: -x[1])
    if ranked:
        L.append("**Sector leadership (1W):** "
                 + " · ".join(f"{n} {p:+.1f}%" for n, p in ranked[:3])
                 + "  →  lagging: "
                 + " · ".join(f"{n} {p:+.1f}%" for n, p in ranked[-3:]))
        L.append("")

    if ctx.get("breadth_divergence_1w") is not None:
        bd = ctx["breadth_divergence_1w"]
        L.append(f"**Breadth:** cap-weight vs equal-weight {bd:+.1f}pts this "
                 f"week — " + ("narrow, carried by the largest names."
                               if bd > 0 else "broadening participation."))
        L.append("")
    return "\n".join(L)


def alert_block(fired: list[dict]) -> str:
    """Markdown for the DAILY exception report. Silence is the normal case."""
    if not fired:
        return ("## Daily alerts\n\n**None.** No pre-committed threshold was "
                "crossed today. Per the daily action rule, no action is "
                "authorized — this is the expected outcome on most days.\n")
    icon = {"CRITICAL": "🔴", "WARNING": "⚠️", "INFO": "ℹ️"}
    L = [f"## Daily alerts — {len(fired)} fired", ""]
    for a in sorted(fired, key=lambda x: ["CRITICAL", "WARNING", "INFO"]
                    .index(x["level"])):
        L.append(f"### {icon.get(a['level'],'')} {a['level']} — {a['rule']}")
        L.append(a["message"])
        if a["action"]:
            L.append(f"\n**Action:** {a['action']}")
        L.append("")
    return "\n".join(L)


def selftest() -> dict:
    """Verify alerts fire on crossings and stay SILENT on a normal day."""
    failures = []
    calm_ctx = {"indices": {"SPY": {"name": "S&P 500", "d1": 0.3, "d5": 1.1}},
                "sectors": {}, "sector_dispersion_1w": 1.2,
                "breadth_divergence_1w": 0.4}
    calm_sig = {"hy_oas": 2.84, "vix": 14.9, "long_real_yield": 2.41,
                "short_real_rate": 0.08}
    if alerts(calm_ctx, calm_sig):
        failures.append("Alerts fired on a calm day — should be silent")

    crisis = dict(calm_sig, hy_oas=5.4, hy_oas_mom_2w=0.7, vix=36.0)
    got = alerts(calm_ctx, crisis)
    if not any(a["level"] == "CRITICAL" for a in got):
        failures.append("HY crisis did not raise CRITICAL")
    if not any(a["rule"] == "vix_whipsaw" for a in got):
        failures.append("VIX whipsaw not detected")

    band = dict(calm_sig, short_real_rate=0.48)
    if not any(a["rule"] == "short_real_band" for a in alerts(calm_ctx, band)):
        failures.append("Band crossing not detected")

    return {"ok": not failures, "failures": failures,
            "calm_day_alerts": len(alerts(calm_ctx, calm_sig)),
            "crisis_day_alerts": len(got)}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, default=str))
