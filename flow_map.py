"""
flow_map.py  (v1.1 — October 2026)
──────────────────────────────────
v1.1 (2026-10-31): asset-class roll-up (ASSET_CLASS: US equity, international,
  Treasuries, credit, cash-like, gold & commodities, alternatives), a rule-based
  verdict on whether money left stocks and where it went (money_verdict), and
  bridge_summary(), which publish_to_checklist writes as flow_map_summary.json
  for the Portfolio-Tracker brief. Same measurability rules as the panel.
  A class with under half its funds measured is flagged thin (indicative only).

"Where is money leaving, and where is it arriving?" — one panel at the top of
the Money Flow dashboard that answers that in a single read, built ONLY from
verified ETF creation/redemption data (Tier A).

WHAT THIS CAN AND CANNOT SAY
────────────────────────────
ETF share data shows net creations (money arriving) and redemptions (money
leaving) per fund. It does NOT show that a dollar redeemed from XLP was the
same dollar that created XLK shares. So this panel shows SOURCES and
DESTINATIONS, never traced pairs — no Sankey, no arrows between funds. The
headline says "left X / arrived in Y", which is what the data supports.

WINDOWS (sessions, not calendar days)
  5   ~1 week   — catches turns early; one large model-portfolio trade can
                  dominate it. Use for "is something changing?"
  20  ~1 month  — matches institutional rebalancing cadence and the
                  integrity test's own window. The default.
  60  ~1 quarter— the trend; smooths month-end and option-expiry noise.
  A fund appears in a window only when its share series is trustworthy
  (etf_flow_tracker.ticker_quality) AND it has measured flow on >= 80% of the
  window's sessions. Everything else is listed as "building history" with
  its session count — never estimated, never back-filled with zeros.

UNITS
  Dollars answer "how much"; % of AUM answers "how strongly". Dollar bars are
  dominated by the giants (QQQ, GLD, TLT, SPY); % of AUM lets a $40M move in
  a $400M fund register. The chart toggles bar length between the two and
  always labels both.

GROUPS
  Every tracked fund belongs to exactly one theme, and every theme carries a
  risk tag (risk-on / risk-off / neutral) so the panel can also say whether
  money is rotating WITHIN risk or LEAVING it.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

WINDOWS = {5: "5 sessions (~1 week)", 20: "20 sessions (~1 month)",
           60: "60 sessions (~1 quarter)"}
DEFAULT_WINDOW = 20
MIN_COVERAGE = 0.80
MAX_STALE_SESSIONS = 3        # a fund whose last row lags the store by more is excluded

RISK_ON, RISK_OFF, NEUTRAL = "risk-on", "risk-off", "neutral"

# theme -> (risk tag, tickers). Every TRACKED ticker must appear exactly once
# (selftest enforces it, so a new fund can never silently miss the map).
GROUPS = {
    "Tech & growth":        (RISK_ON,  ["XLK", "VGT", "QQQ", "SMH", "SOXX", "IGV", "SKYY", "HACK"]),
    "Communication":        (RISK_ON,  ["XLC"]),
    "Consumer cyclical":    (RISK_ON,  ["XLY", "XRT", "XHB"]),
    "Financials":           (RISK_ON,  ["XLF", "KRE", "KBE", "IAI"]),
    "Industrials":          (RISK_ON,  ["XLI", "ITA", "XTN", "PAVE"]),
    "Energy & materials":   (RISK_ON,  ["XLE", "XOP", "OIH", "AMLP", "XLB"]),
    "Biotech":              (RISK_ON,  ["IBB", "XBI"]),
    "Small & mid caps":     (RISK_ON,  ["IWM", "IWO", "MDY"]),
    "Broad US market":      (RISK_ON,  ["SPY"]),
    "International":        (RISK_ON,  ["EEM", "EFA", "EWJ", "FXI", "INDA"]),
    "High-yield & EM credit": (RISK_ON, ["HYG", "EMB"]),
    "Defensives":           (RISK_OFF, ["XLP", "XLU", "XLV", "PBJ", "PPH", "IHI", "SCHD"]),
    "Real estate":          (NEUTRAL,  ["XLRE"]),
    "IG credit":            (NEUTRAL,  ["LQD"]),
    "Treasuries & TIPS":    (RISK_OFF, ["TLT", "TIP"]),
    "Cash-like":            (RISK_OFF, ["SGOV", "USFR"]),
    "Gold & metals":        (RISK_OFF, ["GLD", "SLV", "RING"]),
    "Commodities":          (NEUTRAL,  ["PDBC", "USO", "DBA"]),
    "Trend / managed futures": (NEUTRAL, ["KMLM"]),
}
TICKER_GROUP = {t: g for g, (_, ts) in GROUPS.items() for t in ts}

# Funds with heavy short interest / used as hedging vehicles. Their share
# counts move with short-selling and market-maker hedging (creations to cover
# borrow), not only investor demand — a big creation in KRE can be shorts
# being built, not buyers arriving. Marked with † wherever they appear.
HEDGE_VEHICLES = {"KRE", "KBE", "XRT", "XHB", "XBI", "IWM", "XOP", "SMH", "SPY", "QQQ", "HYG"}
GROUP_RISK = {g: r for g, (r, _) in GROUPS.items()}


def _names() -> dict:
    try:
        from top_movers import ETF_UNIVERSE
        n = {t: v[0] for t, v in ETF_UNIVERSE.items()}
    except Exception:
        n = {}
    n.update({"SPY": "S&P 500", "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
              "XLV": "Health Care", "XLI": "Industrials", "XLY": "Consumer Discretionary",
              "XLP": "Consumer Staples", "XLU": "Utilities", "XLRE": "Real Estate",
              "XLB": "Materials", "XLC": "Communication", "VGT": "Info Tech (Vanguard)",
              "SCHD": "US Dividend", "SGOV": "0-3M T-Bills", "USFR": "Floating-rate Treasuries",
              "RING": "Gold Miners", "KMLM": "Managed Futures"})
    return n


def fmt_usd(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    a = abs(x)
    s = f"${a / 1e9:.2f}B" if a >= 1e9 else f"${a / 1e6:.0f}M" if a >= 1e6 else f"${a / 1e3:.0f}K"
    return ("−" if x < 0 else "+") + s


# ── computation ─────────────────────────────────────────────────────────────

def fund_flows(flows: pd.DataFrame, window: int, quality_fn=None) -> pd.DataFrame:
    """One row per tracked fund for this window. `flows` is
    etf_flow_tracker.compute_flows() output; quality_fn is ticker_quality."""
    if quality_fn is None:
        from etf_flow_tracker import ticker_quality as quality_fn
    names = _names()
    cols = ["ticker", "name", "group", "risk", "flow_usd", "aum", "aum_start", "pct_aum", "price_chg_pct",
            "agreement", "sessions", "measurable", "status"]
    if flows is None or flows.empty:
        return pd.DataFrame(columns=cols)
    store_dates = sorted(pd.to_datetime(flows["date"]).unique())
    need = math.ceil(MIN_COVERAGE * window)
    rows = []
    for tk, g in flows.groupby("ticker"):
        g = g.sort_values("date")
        grp = TICKER_GROUP.get(tk, "Unmapped")
        base = {"ticker": tk, "name": names.get(tk, tk), "group": grp,
                "risk": GROUP_RISK.get(grp, NEUTRAL), "flow_usd": np.nan, "aum": np.nan,
                "aum_start": np.nan,
                "pct_aum": np.nan, "price_chg_pct": np.nan, "agreement": "",
                "sessions": 0, "measurable": False, "status": ""}
        q = quality_fn(g)
        last = pd.Timestamp(g["date"].iloc[-1])
        lag = sum(1 for d in store_dates if d > last)
        tail = g.tail(window)
        obs = int(tail["net_flow"].notna().sum())
        base["sessions"] = obs
        if not q.get("trustworthy"):
            base["status"] = f"not verified — {q.get('reason', '')}"
        elif lag > MAX_STALE_SESSIONS:
            base["status"] = f"stale — last data {last.date()} ({lag} sessions behind)"
        elif obs < need:
            base["status"] = f"building history — {obs} of {window} sessions measured (needs {need})"
        else:
            flow = float(tail["net_flow"].sum(skipna=True))
            aum = float(g["aum"].iloc[-1])
            # % of AUM at the START of the window: dividing by today's AUM
            # overstates a fund that shrank (or grew) during the window.
            aum0 = float(g["aum"].iloc[-window - 1]) if len(g) > window else float(g["aum"].iloc[0])
            ref = g["price"].iloc[-window - 1] if len(g) > window else g["price"].iloc[0]
            px = float(g["price"].iloc[-1] / ref - 1) * 100 if ref else np.nan
            if flow >= 0 and px >= 0:
                agree = "confirmed ↑ (price up, money in)"
            elif flow < 0 and px < 0:
                agree = "confirmed ↓ (price down, money out)"
            elif flow >= 0:
                agree = "accumulation (price down, money in)"
            else:
                agree = "distribution (price up, money out)"
            base.update(flow_usd=flow, aum=aum, aum_start=aum0,
                        pct_aum=flow / aum0 * 100 if aum0 else np.nan,
                        price_chg_pct=px, agreement=agree, measurable=True,
                        status=f"measured ({q.get('source')})")
        rows.append(base)
    return pd.DataFrame(rows, columns=cols)


def group_flows(funds: pd.DataFrame) -> pd.DataFrame:
    cols = ["group", "risk", "flow_usd", "aum", "pct_aum", "measured", "total", "tickers"]
    if funds.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for grp, g in funds.groupby("group"):
        m = g[g["measurable"]]
        if m.empty:
            continue
        flow, aum = float(m["flow_usd"].sum()), float(m["aum_start"].sum())
        out.append({"group": grp, "risk": GROUP_RISK.get(grp, NEUTRAL), "flow_usd": flow,
                    "aum": aum, "pct_aum": flow / aum * 100 if aum else np.nan,
                    "measured": len(m), "total": len(GROUPS.get(grp, (None, g["ticker"].tolist()))[1]),
                    "tickers": ", ".join(m.sort_values("flow_usd")["ticker"])})
    return pd.DataFrame(out, columns=cols).sort_values("flow_usd")


def risk_split(groups: pd.DataFrame) -> dict:
    return {r: float(groups.loc[groups["risk"] == r, "flow_usd"].sum()) if not groups.empty else 0.0
            for r in (RISK_ON, RISK_OFF, NEUTRAL)}


def headline(groups: pd.DataFrame, funds: pd.DataFrame, window: int) -> str:
    meas, tot = int(funds["measurable"].sum()) if not funds.empty else 0, len(funds)
    if groups.empty:
        return (f"No fund has enough verified flow history for the {window}-session window yet "
                f"({meas} of {tot} measurable). Try a shorter window.")
    out_g = groups[groups["flow_usd"] < 0].sort_values("flow_usd").head(3)
    in_g = groups[groups["flow_usd"] > 0].sort_values("flow_usd", ascending=False).head(3)
    left = ", ".join(f"{r.group} ({fmt_usd(r.flow_usd)})" for r in out_g.itertuples()) or "nothing material"
    came = ", ".join(f"{r.group} ({fmt_usd(r.flow_usd)})" for r in in_g.itertuples()) or "nothing material"
    rs = risk_split(groups)
    net = sum(rs.values())
    tone = _tone(rs)
    blind = [g for g in GROUPS if g not in set(groups["group"])]
    blind_txt = (f" Not yet measurable: {', '.join(blind)}." if blind else "")
    return (f"Last {window} sessions: money left **{left}**; money arrived in **{came}**. "
            f"Net across measured funds {fmt_usd(net)} — {tone}. "
            f"Coverage: {meas} of {tot} tracked funds.{blind_txt}")


# ── asset classes (v1.1): "did money leave stocks, and where did it go?" ────
# Themes roll up to asset classes so the question is answered at the level it
# is asked. Every theme maps to exactly one class (selftest enforces it).
ASSET_CLASS = {
    "Tech & growth": "US equity", "Communication": "US equity", "Consumer cyclical": "US equity",
    "Financials": "US equity", "Industrials": "US equity", "Energy & materials": "US equity",
    "Biotech": "US equity", "Small & mid caps": "US equity", "Broad US market": "US equity",
    "Defensives": "US equity", "Real estate": "US equity",
    "International": "International equity",
    "Treasuries & TIPS": "Treasuries",
    "High-yield & EM credit": "Credit", "IG credit": "Credit",
    "Cash-like": "Cash-like",
    "Gold & metals": "Gold & commodities", "Commodities": "Gold & commodities",
    "Trend / managed futures": "Alternatives",
}
ASSET_ORDER = ["US equity", "International equity", "Treasuries", "Credit", "Cash-like",
               "Gold & commodities", "Alternatives"]
STOCK_CLASSES = ("US equity", "International equity")
SAFE_CLASSES = ("Treasuries", "Cash-like")
MATERIAL_PCT = 0.25     # a bucket's net flow must be at least this % of its measured assets to count as in/out
SUMMARY_FILE = "flow_map_summary.json"
THIN_COVERAGE = 0.5     # fewer than half a class's funds measured → its reading is flagged thin (indicative only)


def asset_class_flows(funds: pd.DataFrame) -> pd.DataFrame:
    """One row per asset class: net flow and start-of-window assets of the
    MEASURABLE funds, with how many of the class's funds were measurable and
    which were not (never estimated, never zero-filled)."""
    cols = ["asset_class", "flow_usd", "aum", "pct_aum", "measured", "total", "missing", "direction", "thin"]
    rows = []
    for ac in ASSET_ORDER:
        themes = [g for g, c in ASSET_CLASS.items() if c == ac]
        all_t = [t for g in themes for t in GROUPS[g][1]]
        sub = funds[funds["ticker"].isin(all_t)] if not funds.empty else funds
        m = sub[sub["measurable"].astype(bool)] if not sub.empty else sub
        flow = float(m["flow_usd"].sum()) if not m.empty else 0.0
        aum = float(m["aum_start"].sum()) if not m.empty else 0.0
        pct = flow / aum * 100 if aum else np.nan
        direction = ("n/a" if m.empty else "in" if pct >= MATERIAL_PCT else
                     "out" if pct <= -MATERIAL_PCT else "flat")
        missing = sorted(set(all_t) - set(m["ticker"])) if not sub.empty else sorted(all_t)
        rows.append({"asset_class": ac, "flow_usd": flow if not m.empty else np.nan, "aum": aum,
                     "pct_aum": pct, "measured": len(m), "total": len(all_t),
                     "missing": missing, "direction": direction,
                     "thin": bool(len(m) and len(m) < THIN_COVERAGE * len(all_t))})
    return pd.DataFrame(rows, columns=cols)


def _side(classes: pd.DataFrame, names) -> tuple:
    sub = classes[classes["asset_class"].isin(names) & (classes["measured"] > 0)]
    flow, aum = float(sub["flow_usd"].sum()), float(sub["aum"].sum())
    pct = flow / aum * 100 if aum else None
    d = "n/a" if pct is None else "in" if pct >= MATERIAL_PCT else "out" if pct <= -MATERIAL_PCT else "flat"
    return flow, pct, d


def money_verdict(classes: pd.DataFrame, groups: pd.DataFrame, funds: pd.DataFrame, window: int) -> dict:
    """Verdict / evidence / rule / what would change it, from the class table."""
    s_flow, s_pct, s_dir = _side(classes, STOCK_CLASSES)
    f_flow, f_pct, f_dir = _side(classes, SAFE_CLASSES)
    rule = (f"Net {window}-session ETF creations minus redemptions, from trusted share counts only "
            f"(integrity test passed and at least {MIN_COVERAGE:.0%} of sessions measured). A bucket counts as "
            f"in or out when its net flow is at least {MATERIAL_PCT}% of its measured assets. Stocks = US + "
            f"international equity; safety = Treasuries + cash-like. ETF data shows where money left and where "
            f"it arrived, not which dollar went where.")
    if s_dir == "n/a" or f_dir == "n/a":
        return {"kind": "unavailable", "verdict": "Not enough measured funds to compare stocks with safe assets",
                "evidence": [], "rule": rule, "changes_if": "More funds passing the integrity test."}
    s_txt = f"stocks {fmt_usd(s_flow)} ({s_pct:+.2f}% of measured assets)"
    f_txt = f"Treasuries + cash {fmt_usd(f_flow)} ({f_pct:+.2f}%)"
    if s_dir == "out" and f_dir == "in":
        kind = "out_of_stocks"
        share = f_flow / abs(s_flow) if s_flow else 0
        verdict = (f"Money left stocks for safety: {s_txt}; {f_txt}, about {share:.0%} of the stock outflow")
        changes = "Stock funds turning net positive, or safe assets turning net negative."
    elif s_dir == "in" and f_dir == "out":
        kind = "into_stocks"
        verdict = f"Money moved from safety into stocks: {s_txt}; {f_txt}"
        changes = "Stock funds turning net negative while safe assets take money in."
    elif s_dir == "in" and f_dir in ("in", "flat"):
        kind = "new_money"
        verdict = (f"New money, not a move out of stocks: {s_txt}; {f_txt}")
        changes = f"Stock funds net outflow of {MATERIAL_PCT}% or more of their assets."
    elif s_dir == "out" and f_dir in ("out", "flat"):
        kind = "leaving_both"
        verdict = (f"Money left stocks without arriving in Treasuries or cash funds: {s_txt}; {f_txt} — "
                   f"it went to funds not measured here, or out of ETFs altogether")
        changes = "Treasuries or cash-like funds taking money in."
    else:
        kind = "stocks_flat"
        verdict = f"No material net move in stocks: {s_txt}; {f_txt}"
        changes = f"Stock funds net flow of ±{MATERIAL_PCT}% or more of their assets."
    ev = []
    for r in classes.itertuples():
        if r.measured == 0:
            ev.append(f"{r.asset_class}: no measured fund ({r.total} tracked)")
            continue
        gap = f"; not measured: {', '.join(r.missing)}" if r.missing else ""
        thin = (f" — THIN: under half its funds measured, so this reading rests on "
                f"{'one fund' if r.measured == 1 else f'{r.measured} funds'} and is indicative only") if r.thin else ""
        ev.append(f"{r.asset_class}: {fmt_usd(r.flow_usd)} ({r.pct_aum:+.2f}%, {r.direction}) — "
                  f"{r.measured} of {r.total} funds{gap}{thin}")
    eq = groups[groups["group"].map(lambda g: ASSET_CLASS.get(g) in STOCK_CLASSES)] if not groups.empty else groups
    if not eq.empty:
        outs = eq[eq["flow_usd"] < 0].sort_values("flow_usd").head(2)
        ins = eq[eq["flow_usd"] > 0].sort_values("flow_usd", ascending=False).head(3)
        if len(outs) and len(ins):
            ev.append("Within stocks: out of " + ", ".join(f"{r.group} ({fmt_usd(r.flow_usd)})" for r in outs.itertuples())
                      + "; into " + ", ".join(f"{r.group} ({fmt_usd(r.flow_usd)})" for r in ins.itertuples()))
    if not funds.empty:
        big = funds[funds["measurable"].astype(bool)].copy()
        big["abs"] = big["flow_usd"].abs()
        hv = [t for t in big.sort_values("abs", ascending=False).head(3)["ticker"] if t in HEDGE_VEHICLES]
        if hv:
            ev.append(f"† {', '.join(hv)} {'is' if len(hv) == 1 else 'are'} among the largest moves and "
                      f"{'is a' if len(hv) == 1 else 'are'} hedging vehicle{'s' if len(hv) > 1 else ''}: share counts "
                      f"also move with short selling and market-maker hedging, not only investor money")
    return {"kind": kind, "verdict": verdict, "evidence": ev, "rule": rule, "changes_if": changes}


def bridge_summary(flows: pd.DataFrame, window: int = DEFAULT_WINDOW, quality_fn=None,
                   now: Optional[str] = None) -> dict:
    """JSON-safe summary published for the Portfolio-Tracker brief."""
    from datetime import datetime, timezone
    m = build(flows, window, quality_fn)
    funds, groups = m["funds"], m["groups"]
    classes = asset_class_flows(funds)
    v = money_verdict(classes, groups, funds, window)
    clean = lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else x
    last = str(pd.to_datetime(flows["date"]).max().date()) if flows is not None and not flows.empty else None
    return {
        "published_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": window, "data_through": last,
        "coverage": {"measured": int(funds["measurable"].sum()) if not funds.empty else 0, "total": len(funds)},
        "asset_classes": [{k: (clean(r[k]) if k != "missing" else r[k]) for k in classes.columns}
                          for _, r in classes.iterrows()],
        "themes": [{"group": r.group, "asset_class": ASSET_CLASS.get(r.group), "risk": r.risk,
                    "flow_usd": clean(r.flow_usd), "pct_aum": clean(r.pct_aum), "measured": int(r.measured),
                    "total": int(r.total), "tickers": r.tickers} for r in groups.itertuples()] if not groups.empty else [],
        "risk_split": {k: float(x) for k, x in m["risk"].items()},
        "verdict": v,
        "excluded": {r.ticker: r.status for r in funds.itertuples() if not r.measurable} if not funds.empty else {},
    }


def _tone(rs: dict) -> str:
    on, off = rs[RISK_ON], rs[RISK_OFF]
    if on < 0 and off > 0:
        share = off / abs(on)
        if share >= 0.5:
            return "a rotation from risk-on into risk-off funds"
        return (f"money is leaving risk-on funds, and only {share:.0%} of it shows up in measured "
                "risk-off funds — the rest went to cash or to funds not measured here")
    if on > 0 and off < 0:
        share = on / abs(off)
        if share >= 0.5:
            return "a rotation from risk-off into risk-on funds"
        return (f"money is leaving risk-off funds, and only {share:.0%} of it shows up in measured "
                "risk-on funds")
    if on > 0 and off > 0:
        return "both risk-on and risk-off funds are taking money in (new money, not rotation)"
    if on < 0 and off < 0:
        return "money is leaving both risk-on and risk-off funds"
    return "no clear risk direction"


def build(flows: pd.DataFrame, window: int, quality_fn=None) -> dict:
    funds = fund_flows(flows, window, quality_fn)
    groups = group_flows(funds)
    return {"window": window, "funds": funds, "groups": groups,
            "risk": risk_split(groups), "headline": headline(groups, funds, window)}


# ── rendering ───────────────────────────────────────────────────────────────

def _bar_chart(df: pd.DataFrame, label_col: str, length: str, title: str):
    import plotly.graph_objects as go
    df = df.sort_values("flow_usd" if length == "usd" else "pct_aum")
    x = df["flow_usd"] if length == "usd" else df["pct_aum"]
    colors = ["#e05252" if v < 0 else "#3fb950" for v in x]
    text = [f"{fmt_usd(f)} · {p:+.2f}% AUM" for f, p in zip(df["flow_usd"], df["pct_aum"])]
    fig = go.Figure(go.Bar(
        x=x, y=df[label_col], orientation="h", marker_color=colors, text=text,
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>"))
    fig.add_vline(x=0, line_color="#888", line_width=1)
    lo, hi = float(min(x.min(), 0)), float(max(x.max(), 0))
    span = (hi - lo) or 1.0
    pad = 0.45 * span                      # room for the outside value labels
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=max(260, 30 * len(df) + 90), margin=dict(l=10, r=20, t=40, b=40),
        paper_bgcolor="rgba(14,17,23,0)", plot_bgcolor="rgba(14,17,23,0)",
        font=dict(color="#c0c4d6"), showlegend=False,
        xaxis=dict(title="Net flow ($)" if length == "usd" else "Net flow (% of fund AUM)",
                   gridcolor="#2a2d3e", zeroline=False,
                   range=[lo - (pad if lo < 0 else 0.05 * span), hi + (pad if hi > 0 else 0.05 * span)],
                   tickprefix="$" if length == "usd" else "", ticksuffix="" if length == "usd" else "%"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True, ticklabelstandoff=8))
    return fig


def render(st, store: str = "data/etf_shares_history.csv", ext: Optional[dict] = None) -> Optional[dict]:
    """`ext` = extension.lookup() output; adds an Overbought/oversold column
    to the fund table when present."""
    st.markdown("## 🧭 Flow Map — where money is leaving and arriving")
    try:
        from etf_flow_tracker import compute_flows, ticker_quality
        flows = compute_flows(store)
    except Exception as e:
        st.warning(f"Flow map unavailable: {type(e).__name__}: {e}")
        return None
    c1, c2, c3 = st.columns([2, 1, 1])
    window = c1.radio("Window", list(WINDOWS), index=list(WINDOWS).index(DEFAULT_WINDOW),
                      format_func=lambda w: WINDOWS[w], horizontal=True, key="fm_window")
    view = c2.radio("View", ["By theme", "By fund"], horizontal=True, key="fm_view")
    length = c3.radio("Bar length", ["$", "% of AUM"], horizontal=True, key="fm_len")
    m = build(flows, window, ticker_quality)
    # Escape "$": Streamlit markdown treats a pair of dollar signs as LaTeX, which
    # turned "-$1.91B ... +$645M" into italic math (caught Oct 2026).
    st.markdown(m["headline"].replace("$", "\\$"))

    rs = m["risk"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Risk-on funds", fmt_usd(rs[RISK_ON]),
              help="Tech, cyclicals, financials, industrials, energy/materials, biotech, "
                   "small/mid, broad US, international, HY & EM credit")
    k2.metric("Risk-off funds", fmt_usd(rs[RISK_OFF]),
              help="Defensives, Treasuries & TIPS, cash-like, gold & metals")
    k3.metric("Neutral", fmt_usd(rs[NEUTRAL]),
              help="Real estate, IG credit, broad commodities, managed futures")

    ln = "usd" if length == "$" else "pct"
    if view == "By theme":
        if not m["groups"].empty:
            g = m["groups"].copy()
            g["label"] = [f"{r.group} ({r.measured}/{r.total})" for r in g.itertuples()]
            st.plotly_chart(_bar_chart(g, "label", ln, "Net flow by theme — red = leaving, green = arriving"),
                            use_container_width=True)
            st.caption("(n/m) = funds measured of funds in the theme. % of AUM = net flow over the "
                       "combined AUM of the measured funds at the START of the window.")
    else:
        f = m["funds"][m["funds"]["measurable"]].copy()
        if not f.empty:
            f["label"] = [f"{t}{'†' if t in HEDGE_VEHICLES else ''} · {n[:22]}"
                          for t, n in zip(f["ticker"], f["name"])]
            st.plotly_chart(_bar_chart(f, "label", ln, "Net flow by fund — red = leaving, green = arriving"),
                            use_container_width=True)

    meas = m["funds"][m["funds"]["measurable"]]
    if not meas.empty:
        with st.expander("Fund detail — flow, price, and whether they agree", expanded=False):
            t = meas.sort_values("flow_usd").assign(
                **{"Net flow": lambda d: d["flow_usd"].map(fmt_usd),
                   "% AUM": lambda d: d["pct_aum"].map(lambda v: f"{v:+.2f}%"),
                   "Price": lambda d: d["price_chg_pct"].map(lambda v: f"{v:+.1f}%"),
                   "AUM": lambda d: d["aum"].map(lambda v: fmt_usd(v).lstrip("+"))})
            _cols = ["ticker", "name", "group", "Net flow", "% AUM", "Price", "agreement"]
            if ext:
                try:
                    from extension import badge_label
                    t["Overbought/oversold"] = [badge_label(ext.get(x)) for x in t["ticker"]]
                    _cols.append("Overbought/oversold")
                except Exception:
                    pass
            t["ticker"] = [f"{x}†" if x in HEDGE_VEHICLES else x for x in t["ticker"]]
            st.dataframe(t[_cols + ["AUM", "sessions"]].rename(columns={
                                "ticker": "Fund", "name": "Name", "group": "Theme",
                                "agreement": "Price vs money", "sessions": "Sessions measured"}),
                         use_container_width=True, hide_index=True)
    pending = m["funds"][~m["funds"]["measurable"]]
    if not pending.empty:
        with st.expander(f"Not measurable in this window ({len(pending)} funds) — and why", expanded=False):
            st.dataframe(pending[["ticker", "group", "status"]].rename(columns={
                "ticker": "Fund", "group": "Theme", "status": "Reason"}).sort_values("Theme"),
                use_container_width=True, hide_index=True)
    st.caption(
        "Sources and destinations, not traced pairs: ETF data shows money leaving one set of funds "
        "and arriving in another, not that the same dollars moved between them. Tier A only — "
        "verified share-count changes (issuer or reported data). † = heavily shorted / hedging "
        "vehicle: its creations and redemptions also reflect short-selling and market-maker hedging, "
        "not only investors. % of AUM uses AUM at the start of the window. Funds without enough verified "
        "history for the window are listed, never estimated. Share data is polled once daily after "
        "the close, so the latest session can lag by one day."
    )
    return m


# ── selftest ────────────────────────────────────────────────────────────────

def selftest() -> dict:
    f = []
    try:
        from etf_flow_tracker import TRACKED
        tracked = set(TRACKED)
        mapped = set(TICKER_GROUP)
        if tracked - mapped:
            f.append(f"tracked funds missing from GROUPS: {sorted(tracked - mapped)}")
        dup = [t for t in mapped if sum(t in ts for _, ts in GROUPS.values()) > 1]
        if dup:
            f.append(f"funds in more than one theme: {dup}")
    except Exception as e:
        f.append(f"TRACKED import failed: {e}")

    dates = pd.bdate_range("2026-07-01", periods=70)

    def series(tk, flows, aum=1e9):
        return pd.DataFrame({"date": dates[-len(flows):], "ticker": tk,
                             "net_flow": flows, "aum": aum, "price": np.linspace(100, 105, len(flows)),
                             "shares_source": "issuer_spdr", "shares_outstanding": 1e7})
    ok_q = lambda g: {"trustworthy": True, "source": "issuer_spdr"}
    df = pd.concat([
        series("XLK", [np.nan] + [10e6] * 69),            # in, risk-on
        series("XLP", [np.nan] + [-5e6] * 69),             # out, risk-off
        series("TLT", [np.nan] + [3e6] * 69, aum=5e9),     # in, risk-off
        series("GLD", [np.nan, 1e6, 1e6, 1e6]),            # only 3 sessions
    ], ignore_index=True)
    m = build(df, 20, ok_q)
    fu = m["funds"].set_index("ticker")
    if abs(fu.loc["XLK", "flow_usd"] - 200e6) > 1:
        f.append(f"20-session sum wrong: {fu.loc['XLK', 'flow_usd']}")
    shrink = series("KRE", [np.nan] + [-10e6] * 69)
    shrink["aum"] = np.linspace(2e9, 1e9, 70)
    k = build(pd.concat([df, shrink]), 20, ok_q)["funds"].set_index("ticker").loc["KRE"]
    if abs(k["pct_aum"] - (-200e6 / k["aum_start"] * 100)) > 1e-6 or k["aum_start"] <= k["aum"]:
        f.append(f"% AUM must use start-of-window AUM: {k['pct_aum']} start {k['aum_start']} now {k['aum']}")
    if abs(fu.loc["XLK", "pct_aum"] - 20.0) > 1e-6:
        f.append(f"% AUM wrong: {fu.loc['XLK', 'pct_aum']}")
    if fu.loc["GLD", "measurable"] or "building history" not in fu.loc["GLD", "status"]:
        f.append(f"3-session fund must be 'building history' in a 20 window: {fu.loc['GLD', 'status']}")
    if not fu.loc["XLP", "agreement"].startswith("distribution"):
        f.append(f"price up + money out must be distribution: {fu.loc['XLP', 'agreement']}")
    if "Tech & growth" not in m["headline"] or "Defensives" not in m["headline"]:
        f.append(f"headline must name sources and destinations: {m['headline']}")
    if "only" not in _tone({RISK_ON: -1e9, RISK_OFF: 1e8, NEUTRAL: 0}) or \
            "rotation from risk-on" not in _tone({RISK_ON: -1e9, RISK_OFF: 8e8, NEUTRAL: 0}):
        f.append("tone must distinguish a real rotation from money leaving the measured set")
    if "Not yet measurable" not in m["headline"]:
        f.append("headline must name the blind themes")
    if abs(m["risk"][RISK_ON] - 200e6) > 1 or abs(m["risk"][RISK_OFF] - (-100e6 + 60e6)) > 1:
        f.append(f"risk split wrong: {m['risk']}")
    m5 = build(df, 5, ok_q)
    if m5["funds"].set_index("ticker").loc["GLD", "measurable"]:
        f.append("3 of 5 sessions (< 80%) must not be measurable")
    bad_q = lambda g: {"trustworthy": False, "reason": "stale-AUM artifact"}
    mb = build(df, 20, bad_q)
    if mb["funds"]["measurable"].any() or "No fund" not in mb["headline"]:
        f.append("untrusted series must never be measured")
    stale = df.copy()
    stale = stale[~((stale["ticker"] == "XLP") & (stale["date"] > dates[-6]))]
    ms = build(stale, 20, ok_q).get("funds").set_index("ticker")
    if ms.loc["XLP", "measurable"] or "stale" not in ms.loc["XLP", "status"]:
        f.append(f"a fund 5 sessions behind the store must be stale: {ms.loc['XLP', 'status']}")
    if fmt_usd(-1.234e9) != "−$1.23B" or fmt_usd(45e6) != "+$45M":
        f.append(f"fmt_usd: {fmt_usd(-1.234e9)} {fmt_usd(45e6)}")

    # ── v1.1 asset classes and the money verdict ──
    if set(ASSET_CLASS) != set(GROUPS):
        f.append(f"every theme must map to exactly one asset class: "
                 f"missing {sorted(set(GROUPS) - set(ASSET_CLASS))}, extra {sorted(set(ASSET_CLASS) - set(GROUPS))}")
    if set(ASSET_CLASS.values()) - set(ASSET_ORDER):
        f.append("asset class outside ASSET_ORDER")

    def fl(tk, per_day, aum=1e9):
        return series(tk, [np.nan] + [per_day] * 69, aum=aum)

    def verdict_for(parts):
        d = pd.concat(parts, ignore_index=True)
        mm = build(d, 20, ok_q)
        cl = asset_class_flows(mm["funds"])
        return cl, money_verdict(cl, mm["groups"], mm["funds"], 20)
    # stocks out 20 x -1M on 1B = -2%; Treasuries in +1%: out of stocks into safety
    cl, v = verdict_for([fl("SPY", -1e6), fl("TLT", 0.5e6), fl("SGOV", 0.1e6)])
    if v["kind"] != "out_of_stocks":
        f.append(f"stocks out + safety in must read as out_of_stocks: {v['kind']} {v['verdict']}")
    ceq = cl.set_index("asset_class")
    if abs(ceq.loc["US equity", "flow_usd"] + 20e6) > 1 or ceq.loc["US equity", "direction"] != "out":
        f.append(f"US equity class sum wrong: {ceq.loc['US equity'].to_dict()}")
    if "SPY" not in " ".join(v["evidence"]) or "†" not in " ".join(v["evidence"]):
        f.append("a hedging vehicle among the largest moves must be flagged")
    for parts, want in (([fl("XLK", 1e6), fl("TLT", -0.5e6)], "into_stocks"),
                        ([fl("XLK", 1e6), fl("TLT", 0.5e6)], "new_money"),
                        ([fl("XLK", -1e6), fl("TLT", -0.5e6)], "leaving_both"),
                        ([fl("XLK", 0.01e6), fl("TLT", 0.5e6)], "stocks_flat")):
        k_ = verdict_for(parts)[1]["kind"]
        if k_ != want:
            f.append(f"money_verdict must be {want}: {k_}")
    # class sums equal the theme sums; an unmeasured fund is listed, never zero-filled
    d = pd.concat([fl("XLK", 1e6), fl("XLP", -0.5e6), fl("TLT", 0.2e6),
                   series("SGOV", [np.nan, 1e6, 1e6])], ignore_index=True)
    mm = build(d, 20, ok_q)
    cl = asset_class_flows(mm["funds"]).set_index("asset_class")
    us_themes = mm["groups"][mm["groups"]["group"].map(ASSET_CLASS) == "US equity"]["flow_usd"].sum()
    if abs(cl.loc["US equity", "flow_usd"] - us_themes) > 1:
        f.append("US equity class must equal the sum of its themes")
    if cl.loc["Cash-like", "measured"] != 0 or "SGOV" not in cl.loc["Cash-like", "missing"] \
            or cl.loc["Cash-like", "direction"] != "n/a":
        f.append(f"an unmeasured cash fund must be listed as missing, not counted: {cl.loc['Cash-like'].to_dict()}")
    # a class's direction uses the same materiality line as the verdict: +0.02% is flat, not "in"
    cl_f, _ = verdict_for([fl("XLK", 0.01e6), fl("TLT", 0.5e6)])
    if cl_f.set_index("asset_class").loc["US equity", "direction"] != "flat":
        f.append(f"a +0.02% class flow must read flat: {cl_f.set_index('asset_class').loc['US equity'].to_dict()}")
    # credit is not safety: stocks out with only credit taking money in is NOT out_of_stocks
    if verdict_for([fl("XLK", -1e6), fl("HYG", 1e6), fl("TLT", 0.0)])[1]["kind"] != "leaving_both":
        f.append("stocks out + credit in (Treasuries flat) must read leaving_both — credit is not a safe asset")
    # a class with under half its funds measured is flagged thin, in the table and the evidence
    cl_t, v_t = verdict_for([fl("EFA", 1e6), fl("TLT", 0.5e6), fl("TIP", 0.5e6)])
    ct = cl_t.set_index("asset_class")
    if not ct.loc["International equity", "thin"] or ct.loc["Treasuries", "thin"] or ct.loc["Cash-like", "thin"]:
        f.append(f"thin flag: Intl 1-of-{ct.loc['International equity', 'total']} must be thin, "
                 f"Treasuries 2-of-2 and unmeasured Cash-like must not: {ct['thin'].to_dict()}")
    if not any(e.startswith("International equity") and "THIN" in e and "one fund" in e for e in v_t["evidence"]) \
            or any(e.startswith("Treasuries") and "THIN" in e for e in v_t["evidence"]):
        f.append(f"evidence must mark only the thin class: {v_t['evidence']}")
    # with no measured stock or safe fund the verdict refuses rather than guessing
    v_none = money_verdict(asset_class_flows(build(fl("GLD", 1e6), 20, ok_q)["funds"]), pd.DataFrame(
        columns=["group", "flow_usd"]), pd.DataFrame(columns=["ticker", "measurable", "flow_usd"]), 20)
    if v_none["kind"] != "unavailable":
        f.append(f"no measured stock or safe fund must give 'unavailable': {v_none['kind']}")
    # the published summary is JSON-safe and carries every part
    import json as _json
    bs = bridge_summary(d, 20, ok_q, now="2026-10-31T22:00:00+00:00")
    try:
        _json.loads(_json.dumps(bs))
    except Exception as e:
        f.append(f"bridge summary must be JSON-safe: {e}")
    for k in ("asset_classes", "themes", "verdict", "coverage", "excluded", "data_through"):
        if k not in bs:
            f.append(f"bridge summary missing {k}")
    if [r["asset_class"] for r in bs["asset_classes"]] != ASSET_ORDER:
        f.append("bridge summary must list every asset class in order")
    if any(type(r.get("thin")) is not bool for r in bs["asset_classes"]):
        f.append("bridge summary must publish thin as a plain true/false")
    if any(isinstance(r["flow_usd"], float) and math.isnan(r["flow_usd"]) for r in bs["asset_classes"]):
        f.append("bridge summary must publish None, not NaN")
    return {"ok": not f, "failures": f}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2))
