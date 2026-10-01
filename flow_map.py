"""
flow_map.py  (v1 — October 2026)
────────────────────────────────
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


def render(st, store: str = "data/etf_shares_history.csv") -> Optional[dict]:
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
    st.markdown(m["headline"])

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
            t["ticker"] = [f"{x}†" if x in HEDGE_VEHICLES else x for x in t["ticker"]]
            st.dataframe(t[["ticker", "name", "group", "Net flow", "% AUM", "Price", "agreement",
                            "AUM", "sessions"]].rename(columns={
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
    return {"ok": not f, "failures": f}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2))
