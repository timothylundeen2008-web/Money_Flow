"""
extension.py  (v1 — October 2026)
─────────────────────────────────
"Which funds are overbought / oversold — and does the money agree?"

WHY THIS IS NOT ON THE RRG ALONE
────────────────────────────────
The RRG measures strength RELATIVE TO SPY and the trend of that strength. A
fund can sit deep in Leading because the market is weak, without being
stretched at all; in a melt-up every fund can be stretched while the RRG
shows nothing unusual. Overbought is a different question: how far has the
fund run from ITS OWN recent prices? So it lives in its own column/panel and
only DECORATES the RRG (a ring), never moves a dot.

THREE TESTS (each fund vs its own history, never vs other funds)
  1. RSI(14), Wilder smoothing          ≥ 70 high   ≤ 30 low
  2. Z-score vs 20-day mean             ≥ +2 high   ≤ −2 low   (= outside the
     (close − SMA20) / stdev20                                  Bollinger band)
  3. % from the 50-day average, ranked against the fund's OWN last 252
     sessions of that same measure      ≥ 90th pct  ≤ 10th pct
     A fixed "+8% above the 50d" would flag SMH every week and TLT never;
     ranking against its own year adapts to each fund's volatility.

BADGE
  OVERBOUGHT  ≥ 2 of 3 high tests     EXTENDED ↑  exactly 1 high test
  OVERSOLD    ≥ 2 of 3 low tests      EXTENDED ↓  exactly 1 low test
  NEUTRAL     none
  Two-of-three is deliberate: RSI alone stays > 70 for weeks in a healthy
  trend, which is why single-indicator overbought screens are mostly noise.

FLOW CROSS-READ (the part worth acting on)
  Money direction comes from, in order of trust:
    A  verified ETF creations/redemptions, 20 sessions (flow_map / Tier A):
       in ≥ +0.5% of AUM, out ≤ −0.5% of AUM, else flat
    B  Chaikin Money Flow(21) from daily OHLCV when Tier A isn't measurable:
       in > +0.05, out < −0.05, else flat. CMF is a VOLUME proxy, not
       verified dollars — labelled as such everywhere it appears.
  OVERBOUGHT + money in   → "Strong — hold, don't chase"
  OVERBOUGHT + money out  → "⚠ Warning — stretched while money leaves"
  OVERSOLD   + money in   → "Watch — possible accumulation (run the 3-of-3)"
  OVERSOLD   + money out  → "Falling knife — money still leaving"
  This is display-only. Nothing here feeds the regime, the model portfolio
  or any brief.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

RSI_N, Z_N, MA_N, RANK_N = 14, 20, 50, 252
RSI_HI, RSI_LO = 70.0, 30.0
Z_HI, Z_LO = 2.0, -2.0
PCTL_HI, PCTL_LO = 90.0, 10.0
MIN_ROWS = MA_N + 60          # below this the 50d-rank test is unavailable

TIER_A_BAND = 0.5             # % of AUM over 20 sessions
CMF_BAND = 0.05
FLOW_WINDOW = 20

OVERBOUGHT, EXT_UP, NEUTRAL, EXT_DN, OVERSOLD = (
    "OVERBOUGHT", "EXTENDED ↑", "NEUTRAL", "EXTENDED ↓", "OVERSOLD")
BADGE_ORDER = {OVERBOUGHT: 0, EXT_UP: 1, NEUTRAL: 2, EXT_DN: 3, OVERSOLD: 4}
BADGE_COLOR = {OVERBOUGHT: "#E24B4A", EXT_UP: "#BA7517", NEUTRAL: "#888780",
               EXT_DN: "#378ADD", OVERSOLD: "#1D9E75"}
BADGE_ICON = {OVERBOUGHT: "🔴", EXT_UP: "🟠", NEUTRAL: "", EXT_DN: "🔵", OVERSOLD: "🟢"}


# ── indicators ──────────────────────────────────────────────────────────────

def rsi(close: pd.Series, n: int = RSI_N) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna())


def _f(x) -> float:
    try:
        x = float(x)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def metrics(close: pd.Series) -> dict:
    """The three tests for one fund. NaN where history is too short."""
    c = pd.Series(close).dropna().astype(float)
    out = {"rsi14": np.nan, "z20": np.nan, "pct_vs_50d": np.nan,
           "pct_50d_rank": np.nan, "rows": int(len(c)), "last": np.nan}
    if len(c) < RSI_N + 1:
        return out
    out["last"] = float(c.iloc[-1])
    out["rsi14"] = _f(rsi(c).iloc[-1])
    if len(c) >= Z_N:
        w = c.tail(Z_N)
        sd = float(w.std())
        out["z20"] = (float(c.iloc[-1]) - float(w.mean())) / sd if sd > 0 else np.nan
    if len(c) >= MA_N:
        dist = (c / c.rolling(MA_N).mean() - 1) * 100
        out["pct_vs_50d"] = _f(dist.iloc[-1])
        if len(c) >= MIN_ROWS:
            hist = dist.dropna().tail(RANK_N)
            out["pct_50d_rank"] = float((hist <= hist.iloc[-1]).mean() * 100)
    return out


def classify(rsi14: float, z20: float, pct_rank: float) -> tuple[str, int, int]:
    """(badge, high-test hits, low-test hits). NaN tests simply don't vote."""
    def ok(v):
        return v is not None and not (isinstance(v, float) and math.isnan(v))
    hi = sum([ok(rsi14) and rsi14 >= RSI_HI, ok(z20) and z20 >= Z_HI,
              ok(pct_rank) and pct_rank >= PCTL_HI])
    lo = sum([ok(rsi14) and rsi14 <= RSI_LO, ok(z20) and z20 <= Z_LO,
              ok(pct_rank) and pct_rank <= PCTL_LO])
    if hi >= 2:
        return OVERBOUGHT, hi, lo
    if lo >= 2:
        return OVERSOLD, hi, lo
    if hi == 1 and lo == 0:
        return EXT_UP, hi, lo
    if lo == 1 and hi == 0:
        return EXT_DN, hi, lo
    return NEUTRAL, hi, lo


def flow_direction(tier_a_pct_aum: Optional[float], cmf: Optional[float]) -> tuple[str, str]:
    """('in'|'out'|'flat'|'unknown', source label)."""
    if tier_a_pct_aum is not None and not math.isnan(_f(tier_a_pct_aum)):
        v = float(tier_a_pct_aum)
        d = "in" if v >= TIER_A_BAND else "out" if v <= -TIER_A_BAND else "flat"
        return d, f"verified {v:+.1f}% AUM"
    if cmf is not None and not math.isnan(_f(cmf)):
        v = float(cmf)
        d = "in" if v > CMF_BAND else "out" if v < -CMF_BAND else "flat"
        return d, f"volume CMF {v:+.2f}"
    return "unknown", "no flow data"


def flow_read(badge: str, direction: str) -> str:
    if badge == OVERBOUGHT:
        return {"in": "Strong — hold, don't chase",
                "out": "⚠ Warning — stretched while money leaves",
                "flat": "Stretched — no flow confirmation",
                }.get(direction, "Stretched — flow unknown")
    if badge == OVERSOLD:
        return {"in": "Watch — possible accumulation (run the 3-of-3)",
                "out": "Falling knife — money still leaving",
                "flat": "Washed out — no buyer yet",
                }.get(direction, "Washed out — flow unknown")
    if badge == EXT_UP:
        return "Getting stretched — money leaving" if direction == "out" else "Getting stretched"
    if badge == EXT_DN:
        return "Pulling back — money arriving" if direction == "in" else "Pulling back"
    return ""


def _cmf(ohlcv: pd.DataFrame) -> float:
    try:
        from flow_metrics import chaikin_money_flow
        s = chaikin_money_flow(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], ohlcv["Volume"],
                               period=21)
        s = s.dropna()
        return float(s.iloc[-1]) if len(s) else float("nan")
    except Exception:
        return float("nan")


# ── table ───────────────────────────────────────────────────────────────────

def build_table(ohlcv: dict, tier_a: Optional[pd.DataFrame] = None,
                names: Optional[dict] = None, groups: Optional[dict] = None) -> pd.DataFrame:
    """One row per fund. `ohlcv` = {ticker: DataFrame[High, Low, Close, Volume]};
    `tier_a` = flow_map.fund_flows(..., 20) output (only measurable rows used)."""
    names, groups = names or {}, groups or {}
    ta = {}
    if tier_a is not None and not tier_a.empty:
        for r in tier_a[tier_a["measurable"]].itertuples():
            ta[r.ticker] = r.pct_aum
    rows = []
    for tk, d in sorted(ohlcv.items()):
        if d is None or d.empty or "Close" not in d:
            continue
        m = metrics(d["Close"])
        badge, hi, lo = classify(m["rsi14"], m["z20"], m["pct_50d_rank"])
        cmf = _cmf(d) if {"High", "Low", "Volume"} <= set(d.columns) else float("nan")
        direction, src = flow_direction(ta.get(tk), cmf)
        rows.append({"ticker": tk, "name": names.get(tk, tk), "group": groups.get(tk, ""),
                     "badge": badge, "tests_hi": hi, "tests_lo": lo,
                     "rsi14": m["rsi14"], "z20": m["z20"], "pct_vs_50d": m["pct_vs_50d"],
                     "pct_50d_rank": m["pct_50d_rank"], "flow_dir": direction,
                     "flow_src": src, "read": flow_read(badge, direction),
                     "asof": pd.Timestamp(d.index[-1]).date() if len(d) else None,
                     "rows": m["rows"]})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["_o"] = t["badge"].map(BADGE_ORDER)
    return t.sort_values(["_o", "rsi14"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def badge_label(row) -> str:
    """Compact text for other tables: '🔴 OVERBOUGHT (RSI 74)'."""
    if row is None:
        return "—"
    b = row["badge"]
    r = row["rsi14"]
    rs = f" (RSI {r:.0f})" if not math.isnan(_f(r)) else ""
    return f"{BADGE_ICON.get(b, '')} {b}{rs}".strip()


def lookup(table: Optional[pd.DataFrame]) -> dict:
    if table is None or table.empty:
        return {}
    return {r["ticker"]: r for _, r in table.iterrows()}


# ── data ────────────────────────────────────────────────────────────────────

def fetch_ohlcv(tickers: tuple) -> dict:
    """~500 calendar days of daily OHLCV (≈ 340 sessions: enough for the
    252-session rank of a 50-day distance). Missing tickers are skipped."""
    import yfinance as yf
    raw = yf.download(list(tickers), period="500d", interval="1d", auto_adjust=True,
                      progress=False, threads=True, timeout=20, group_by="ticker")
    out = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        for tk in tickers:
            if tk in lvl0:
                d = raw[tk][["High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            elif "Close" in lvl0 and tk in raw["Close"].columns:
                d = pd.DataFrame({f: raw[f][tk] for f in ("High", "Low", "Close", "Volume")}).dropna(subset=["Close"])
            else:
                continue
            if not d.empty:
                out[tk] = d
    elif len(tickers) == 1:
        out[tickers[0]] = raw[["High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    return out


def universe() -> tuple:
    tks = set()
    try:
        from data_fetcher import TICKERS
        tks |= set(TICKERS)
    except Exception:
        pass
    try:
        from flow_map import TICKER_GROUP
        tks |= set(TICKER_GROUP)
    except Exception:
        pass
    return tuple(sorted(tks))


def compute(st=None) -> Optional[pd.DataFrame]:
    """Full table for the dashboard (cached 1h when Streamlit is present)."""
    tickers = universe()
    if st is not None:
        fetch = st.cache_data(ttl=3600, show_spinner=False)(fetch_ohlcv)
    else:
        fetch = fetch_ohlcv
    try:
        ohlcv = fetch(tickers)
    except Exception as e:
        print(f"[extension] price download failed: {type(e).__name__}: {e}")
        ohlcv = {}
    if not ohlcv:
        # Same bars the sector table used, if that fetch succeeded (sectors only).
        try:
            from data_fetcher import get_last_ohlcv
            ohlcv = dict(get_last_ohlcv())
        except Exception:
            ohlcv = {}
    tier_a, names, groups = None, {}, {}
    try:
        import flow_map as fm
        from etf_flow_tracker import compute_flows, ticker_quality
        tier_a = fm.fund_flows(compute_flows("data/etf_shares_history.csv"), FLOW_WINDOW, ticker_quality)
        names, groups = fm._names(), dict(fm.TICKER_GROUP)
    except Exception:
        pass
    try:
        from data_fetcher import SECTORS
        for k, v in SECTORS.items():
            names.setdefault(k, v)
    except Exception:
        pass
    return build_table(ohlcv, tier_a, names, groups)


# ── render ──────────────────────────────────────────────────────────────────

def _fmt(v, f):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f.format(v)


def _ordinal(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    n = int(round(float(v)))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def render(st, table: Optional[pd.DataFrame]) -> None:
    st.markdown("## 🌡️ Overbought / oversold — and does the money agree?")
    if table is None or table.empty:
        st.warning("Extension table unavailable (price download failed). Try Refresh in a few minutes.")
        return
    n = table["badge"].value_counts().to_dict()
    c = st.columns(5)
    for col, b in zip(c, [OVERBOUGHT, EXT_UP, NEUTRAL, EXT_DN, OVERSOLD]):
        col.metric(f"{BADGE_ICON.get(b, '')} {b}".strip(), n.get(b, 0))

    flagged = table[table["badge"].isin([OVERBOUGHT, OVERSOLD])]
    warn = table[(table["badge"] == OVERBOUGHT) & (table["flow_dir"] == "out")]
    if not warn.empty:
        st.error("**Stretched while money leaves:** " + ", ".join(
            f"{r.ticker} (RSI {r.rsi14:.0f}, {r.flow_src})" for r in warn.itertuples()))
    watch = table[(table["badge"] == OVERSOLD) & (table["flow_dir"] == "in")]
    if not watch.empty:
        st.success("**Oversold with money arriving — run the 3-of-3:** " + ", ".join(
            f"{r.ticker} (RSI {r.rsi14:.0f}, {r.flow_src})" for r in watch.itertuples()))
    if flagged.empty:
        st.info("No fund is overbought or oversold on 2-of-3 tests right now.")

    def _view(t):
        return pd.DataFrame({
            "Fund": t["ticker"], "Name": t["name"], "Theme": t["group"],
            "Status": [f"{BADGE_ICON.get(b, '')} {b}".strip() for b in t["badge"]],
            "RSI(14)": t["rsi14"].map(lambda v: _fmt(v, "{:.0f}")),
            "Z vs 20d": t["z20"].map(lambda v: _fmt(v, "{:+.1f}")),
            "% vs 50d": t["pct_vs_50d"].map(lambda v: _fmt(v, "{:+.1f}%")),
            "50d rank (own 1y)": t["pct_50d_rank"].map(_ordinal),
            "Money": [f"{d} · {s}" for d, s in zip(t["flow_dir"], t["flow_src"])],
            "Read": t["read"], "As of": t["asof"]})

    if not flagged.empty:
        st.dataframe(_view(flagged), use_container_width=True, hide_index=True)
    with st.expander(f"All funds ({len(table)}) — sorted overbought → oversold", expanded=False):
        st.dataframe(_view(table), use_container_width=True, hide_index=True)
    st.caption(
        "Each fund is judged against its OWN history, not against other funds or SPY. "
        "Three tests: RSI(14) ≥ 70 / ≤ 30; price ≥ 2 standard deviations above / below its 20-day "
        "average (outside the Bollinger band); distance from the 50-day average in the top / bottom "
        "10% of the fund's own last year. OVERBOUGHT / OVERSOLD = 2 of 3; EXTENDED = 1 of 3. "
        "Money: 'verified' = ETF creations/redemptions over 20 sessions (±0.5% of AUM); 'volume CMF' = "
        "Chaikin Money Flow(21), a volume-based estimate used only where verified data isn't measurable. "
        "Overbought with money arriving is strength (don't chase, don't sell on it alone); overbought "
        "with money leaving is the warning. Display only — nothing here changes the model portfolio."
    )


# ── selftest ────────────────────────────────────────────────────────────────

def selftest() -> dict:
    f = []
    idx = pd.bdate_range("2025-01-01", periods=340)
    rng = np.random.default_rng(7)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 340)))

    # 1. parabolic last 15 sessions → OVERBOUGHT
    up = base.copy()
    up[-15:] = up[-16] * np.exp(np.cumsum(np.full(15, 0.012)))
    m = metrics(pd.Series(up, idx))
    b, hi, _ = classify(m["rsi14"], m["z20"], m["pct_50d_rank"])
    if b != OVERBOUGHT:
        f.append(f"parabolic run must be OVERBOUGHT, got {b} {m}")
    # 2. crash last 15 → OVERSOLD
    dn = base.copy()
    dn[-15:] = dn[-16] * np.exp(np.cumsum(np.full(15, -0.012)))
    m = metrics(pd.Series(dn, idx))
    b, _, lo = classify(m["rsi14"], m["z20"], m["pct_50d_rank"])
    if b != OVERSOLD:
        f.append(f"crash must be OVERSOLD, got {b} {m}")
    # 3. RSI alone does not make OVERBOUGHT
    if classify(75, 0.5, 60)[0] != EXT_UP:
        f.append("RSI-only must be EXTENDED ↑")
    if classify(np.nan, np.nan, np.nan)[0] != NEUTRAL:
        f.append("all-NaN must be NEUTRAL")
    # 4. short history: rank unavailable, no crash
    m = metrics(pd.Series(base[:80], idx[:80]))
    if not math.isnan(m["pct_50d_rank"]):
        f.append("rank must be NaN under MIN_ROWS")
    # 5. RSI sanity: monotone rise → 100
    r = rsi(pd.Series(np.arange(1, 40, dtype=float)))
    if abs(r.iloc[-1] - 100) > 1e-9:
        f.append(f"monotone RSI must be 100, got {r.iloc[-1]}")
    # 6. flow precedence & reads
    if flow_direction(-1.2, 0.3)[0] != "out":
        f.append("Tier A must override CMF")
    if flow_direction(np.nan, 0.10)[0] != "in" or "CMF" not in flow_direction(np.nan, 0.10)[1]:
        f.append("CMF fallback must label itself")
    if not flow_read(OVERBOUGHT, "out").startswith("⚠"):
        f.append("overbought + out must warn")
    if "3-of-3" not in flow_read(OVERSOLD, "in"):
        f.append("oversold + in must point to 3-of-3")
    # 7. build_table end-to-end with Tier A override
    vol = pd.Series(1e6, idx)
    ohlcv = {"AAA": pd.DataFrame({"High": up * 1.005, "Low": up * 0.995, "Close": up, "Volume": vol}, idx),
             "BBB": pd.DataFrame({"High": dn * 1.005, "Low": dn * 0.995, "Close": dn, "Volume": vol}, idx)}
    ta = pd.DataFrame({"ticker": ["AAA"], "pct_aum": [-2.0], "measurable": [True]})
    t = build_table(ohlcv, ta)
    a = t.set_index("ticker").loc["AAA"]
    if a["badge"] != OVERBOUGHT or a["flow_dir"] != "out" or not a["read"].startswith("⚠"):
        f.append(f"AAA should be overbought+out warning: {a.to_dict()}")
    if t.iloc[0]["ticker"] != "AAA" or t.iloc[-1]["ticker"] != "BBB":
        f.append("sort must be overbought → oversold")
    if "verified" not in a["flow_src"]:
        f.append("Tier A source must say verified")
    if [_ordinal(x) for x in (1, 2, 3, 11, 12, 22, 0, 100)] != \
            ["1st", "2nd", "3rd", "11th", "12th", "22nd", "0th", "100th"]:
        f.append("ordinal formatting")
    return {"ok": not f, "failures": f}


if __name__ == "__main__":
    r = selftest()
    print("extension selftest:", "PASS" if r["ok"] else "FAIL")
    for x in r["failures"]:
        print("  -", x)
