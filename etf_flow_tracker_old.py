"""
etf_flow_tracker.py  (v2 — September 2026)
──────────────────────────────────────────────────────────────────────────────
⚠ HONEST STATUS, READ BEFORE TRUSTING ANY OUTPUT ⚠

The sources below were implemented WITHOUT live network access to their
endpoints — this sandbox cannot reach ssga.com, ishares.com, or
stockanalysis.com to verify a single request. Every function is a genuine,
best-effort implementation using each vendor's documented/historically-stable
public data pattern, but "best-effort without live testing" is not the same
as "verified working". Run verify_new_source() (bottom of this file)
IMMEDIATELY after deploying, and again after 2-3 real trading days — do NOT
wait for a 20-session accumulation before checking, the way the original
yfinance bug went undetected for weeks. See flow_integrity.py for the
detector this connects to.

WHAT CHANGED FROM v1
─────────────────────
v1 had exactly one implemented source (yfinance's `sharesOutstanding`),
proven broken by this repo's own data: across 20 sessions, prices moved for
20/20 tracked tickers while shares outstanding moved for 0/20. That field is
an annual-granularity snapshot on Yahoo's backend — no polling frequency
fixes a source with no daily resolution to give.

v2 adds FOUR independent sources with genuine redundancy, prioritized by
issuer coverage of this repo's own TRACKED universe:

    1. issuer_spdr      State Street. Covers 14 of 31 tracked tickers (all
                        sector SPDRs + KRE/XOP/ITA) — the single highest-
                        leverage source to get right.
    2. issuer_spdr_gold GLD specifically — sponsored via a SEPARATE site
                        (spdrgoldshares.com) from the other SPDR funds.
    3. issuer_ishares   BlackRock. Covers 8 more tickers.
    4. aum_implied      shares = totalAssets / price, from yfinance's
                        totalAssets field — genuinely DIFFERENT from
                        sharesOutstanding on Yahoo's backend, and worth
                        testing independently since it may carry a different
                        (better) update cadence. Also serves as a
                        cross-check against whichever primary source is used:
                        a large divergence between the two is itself a data-
                        quality signal.
    5. yfinance         sharesOutstanding. Retained ONLY as a last resort so
                        polling never goes fully empty. Known broken — see
                        above. flow_integrity will correctly flag a store
                        built primarily on this.

EVIDENCE TIER: A (money).
  net flow (t) = Δ shares outstanding (t) × NAV/close (t)
  Shares outstanding change ONLY when an Authorized Participant transacts
  directly with the issuer, in creation units of typically ≥25,000 shares —
  by construction, evidence of institutional-scale net demand rather than an
  inference from tape behavior.

★ THE ACTUAL RECOMMENDED FIX, IF YOU WANT THIS DONE RIGHT RATHER THAN
  BEST-EFFORT: ETF Global's "ETF Daily Fund Flows – US Listed" dataset on AWS
  Data Exchange. It carries shares outstanding, NAV, AND net daily flow
  DIRECTLY — no differencing required — with history back to 2017, so it
  also solves the "20 sessions of accumulated history" problem entirely.
  Requires subscribing via the AWS Marketplace console (a step this code
  cannot do on your behalf). See aws_data_exchange_stub() at the bottom of
  this file for the integration shape once subscribed.

INTERPRETIVE CAUTIONS (build these into any UI that shows this)
  - Flow is not conviction. Creations happen for hedging, model-portfolio
    rebalancing, and index-tracking mandates, not only directional views.
  - Shares outstanding are reported with a lag and are revised; treat a
    single day as noise and read the 5- and 20-day sums.
  - Share splits break the delta. handle_split() below detects and
    neutralizes the obvious cases, but verify any single-day flow larger
    than ~15% of AUM.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import streamlit as st
    _cache = st.cache_data(ttl=3600, show_spinner=False)
except Exception:
    def _cache(fn):
        return fn


DEFAULT_STORE = os.environ.get("ETF_FLOW_STORE", "data/etf_shares_history.csv")

# The scheduled snapshot runs as a GitHub Action, a SEPARATE process from the
# live Streamlit app. print() statements during that run go to the Action's
# own log, which the app cannot see. This registry is written to a small
# JSON sidecar alongside the CSV store so the app CAN read exactly what
# failed, in the same place it already shows verification status — no trip
# to "Manage app" logs required.
_RUN_ERRORS: dict[str, list[str]] = {}


def _log_error(ticker: str, source: str, exc: Exception) -> None:
    msg = f"{source}: {type(exc).__name__}: {exc}"
    print(f"[etf_flow][{source}] {ticker}: {type(exc).__name__}: {exc}")
    _RUN_ERRORS.setdefault(ticker, []).append(msg)


def _errors_path(store: str) -> str:
    base, _ = os.path.splitext(store)
    return base + "_last_run_errors.json"


def load_last_run_errors(store: str = DEFAULT_STORE) -> dict:
    """
    What failed on the MOST RECENT snapshot_all() run, per ticker. Written
    by that run as a sidecar file next to the CSV store, since the run
    itself happens in a separate process (a scheduled Action) from whatever
    reads this — usually the live dashboard.

    Returns {"run_at": iso timestamp, "errors": {ticker: [messages]}} or a
    clearly-empty dict if the sidecar doesn't exist yet (e.g. no run has
    happened since this feature was added).
    """
    path = _errors_path(store)
    if not os.path.exists(path):
        return {"run_at": None, "errors": {}}
    try:
        import json
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"run_at": None, "errors": {}, "load_error": str(e)}

TRACKED = [
    # All-Weather sleeves
    "VGT", "SMH", "QQQ", "GLD", "SLV", "RING", "XLE", "PDBC", "SCHD",
    "XLV", "XLU", "SGOV", "USFR", "TLT", "KMLM",
    # Sector universe (rotation dashboard)
    "XLK", "XLF", "XLI", "XLY", "XLRE", "XLB", "XLC", "XLP",
    # Key sub-sectors
    "KRE", "IBB", "XOP", "ITA", "PAVE", "IWM", "HYG", "EEM",
]

# ── Issuer routing table ────────────────────────────────────────────────────
ISSUER_SPDR = {"XLK", "XLF", "XLI", "XLY", "XLRE", "XLB", "XLC", "XLP",
               "XLE", "XLV", "XLU", "KRE", "XOP", "ITA"}
ISSUER_SPDR_GOLD = {"GLD"}
ISSUER_ISHARES = {"SLV", "RING", "SGOV", "TLT", "IBB", "IWM", "HYG", "EEM"}

IMPLAUSIBLE_DAILY_FLOW_PCT = 0.15
CROSS_CHECK_DIVERGENCE_PCT = 0.05


# ── Individual source implementations ───────────────────────────────────────

def _shares_from_spdr(ticker: str) -> tuple[float | None, float | None]:
    """
    State Street's daily per-fund data file. Covers 14 of 31 tracked tickers
    — the single highest-leverage source in this module. UNVERIFIED against
    the live endpoint — SSGA has changed this site's structure before and
    may again; confirm on first real run.
    """
    try:
        import requests
        url = (f"https://www.ssga.com/us/en/individual/etfs/library-content/"
              f"products/fund-data/etfs/us/fund-data-{ticker.lower()}-us-en.json")
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        shares = (data.get("sharesOutstanding") or data.get("fundShares")
                  or data.get("shares_outstanding"))
        price = (data.get("nav") or data.get("navPrice")
                 or data.get("closePrice"))
        if shares and price:
            return float(shares), float(price)
        _log_error(ticker, "spdr", RuntimeError(
            f"response parsed but no shares/price found in expected keys "
            f"(got keys: {list(data.keys())[:8]})"))
        return None, None
    except Exception as e:
        _log_error(ticker, "spdr", e)
        return None, None


def _shares_from_spdr_gold(ticker: str) -> tuple[float | None, float | None]:
    """
    GLD specifically, via spdrgoldshares.com — a separate site from the
    other SPDR funds. UNVERIFIED against the live endpoint.
    """
    if ticker != "GLD":
        return None, None
    try:
        import requests
        url = "https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_ajax.json"
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        shares = data.get("sharesOutstanding") or data.get("shares")
        price = data.get("navPerShare") or data.get("closePrice")
        if shares and price:
            return float(shares), float(price)
        _log_error(ticker, "spdr_gold", RuntimeError(
            f"response parsed but no shares/price found in expected keys "
            f"(got keys: {list(data.keys())[:8]})"))
        return None, None
    except Exception as e:
        _log_error(ticker, "spdr_gold", e)
        return None, None


def _shares_from_ishares(ticker: str) -> tuple[float | None, float | None]:
    """
    BlackRock iShares daily fund data. Covers 8 more tracked tickers.
    UNVERIFIED against the live endpoint — iShares' fund IDs are numeric,
    not ticker-based, so this tries their ticker-search API first; confirm
    it resolves correctly on first real run.
    """
    try:
        import requests
        search_url = f"https://www.ishares.com/us/product-screener/product-screener-v3.jsn?tickers={ticker}"
        r = requests.get(search_url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        rows = data.get("data", {}).get("tableData", {}).get("data", [])
        if not rows:
            return None, None
        row = rows[0]
        shares = row.get("sharesOutstanding")
        price = row.get("navAmount") or row.get("closePrice")
        if shares and price:
            return float(shares), float(price)
        _log_error(ticker, "ishares", RuntimeError(
            f"response parsed but no matching row/shares found"))
        return None, None
    except Exception as e:
        _log_error(ticker, "ishares", e)
        return None, None


def _shares_from_aum_implied(ticker: str) -> tuple[float | None, float | None]:
    """
    shares ≈ totalAssets / price. Uses yfinance's totalAssets field — a
    GENUINELY DIFFERENT statistic from sharesOutstanding on Yahoo's own
    backend, potentially updated on a different cadence. Also doubles as a
    cross-check against whatever the primary issuer source returned.

    Still an ESTIMATE, not a reported figure — AUM tracks NAV, which can
    differ from market price by the fund's tracking spread, usually small
    for liquid ETFs but not zero. Treat as a cross-check, not primary,
    until it has its own track record of moving day to day.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        aum = info.get("totalAssets")
        price = info.get("navPrice") or info.get("previousClose")
        if aum and price:
            return float(aum) / float(price), float(price)
        _log_error(ticker, "aum_implied", RuntimeError(
            f"totalAssets or navPrice/previousClose missing from "
            f".info (aum={aum!r}, price={price!r})"))
        return None, None
    except Exception as e:
        _log_error(ticker, "aum_implied", e)
        return None, None


def _shares_from_yfinance(ticker: str) -> tuple[float | None, float | None]:
    """
    LAST RESORT. Known to return a static, non-daily value for
    sharesOutstanding specifically — proven by this repo's own 20-session
    data (0/20 tickers moved). Retained so polling never records nothing at
    all. flow_integrity will correctly flag a store built primarily on this.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, None
    try:
        t = yf.Ticker(ticker)
        shares = price = None
        try:
            fi = t.fast_info
            shares = getattr(fi, "shares", None) or fi.get("shares")
            price = getattr(fi, "last_price", None) or fi.get("lastPrice")
        except Exception:
            pass
        if not shares or not price:
            info = t.info or {}
            shares = shares or info.get("sharesOutstanding")
            price = price or info.get("navPrice") or info.get("previousClose")
        return shares, price
    except Exception as e:
        print(f"[etf_flow][yfinance] {ticker}: {type(e).__name__}: {e}")
        return None, None


# ── Routing ──────────────────────────────────────────────────────────────────

def _issuer_source_for(ticker: str):
    if ticker in ISSUER_SPDR_GOLD:
        return _shares_from_spdr_gold
    if ticker in ISSUER_SPDR:
        return _shares_from_spdr
    if ticker in ISSUER_ISHARES:
        return _shares_from_ishares
    return None


def _snapshot_one(ticker: str) -> dict | None:
    """
    Tries, in order: the ticker's routed issuer source, then AUM-implied,
    then yfinance sharesOutstanding as the last resort. Records BOTH the
    primary result and the AUM-implied cross-check when available, so a
    large divergence between two independently-derived numbers is visible
    in the store rather than silently discarded.

    Returns None rather than a guess when every source fails.
    """
    shares = price = None
    used = None

    primary = _issuer_source_for(ticker)
    if primary is not None:
        s, p = primary(ticker)
        if s and p:
            shares, price, used = s, p, primary.__name__.replace("_shares_from_", "")

    aum_shares, aum_price = _shares_from_aum_implied(ticker)

    if shares is None and aum_shares:
        shares, price, used = aum_shares, (price or aum_price), "aum_implied"

    if shares is None:
        s, p = _shares_from_yfinance(ticker)
        if s and p:
            shares, price, used = s, p, "yfinance"

    if not price:
        _, p = _shares_from_yfinance(ticker)
        price = price or p

    if not shares or not price:
        return None

    row = {"date": datetime.now().date().isoformat(), "ticker": ticker,
           "shares_outstanding": float(shares), "price": float(price),
           "shares_source": used or "unknown"}

    if aum_shares and used != "aum_implied":
        divergence = abs(aum_shares - shares) / shares
        row["aum_implied_shares"] = round(float(aum_shares), 0)
        row["cross_check_divergence_pct"] = round(divergence * 100, 2)
        if divergence > CROSS_CHECK_DIVERGENCE_PCT:
            print(f"[etf_flow] {ticker}: {used} vs aum_implied diverge "
                 f"{divergence*100:.1f}% — worth a manual look.")

    return row


def snapshot_all(tickers: list[str] | None = None,
                 store: str = DEFAULT_STORE) -> pd.DataFrame:
    """Once-per-trading-day snapshot for every tracked ETF."""
    global _RUN_ERRORS
    _RUN_ERRORS = {}   # reset — this run's errors only, not accumulated forever

    tickers = tickers or TRACKED
    rows = [r for r in (_snapshot_one(tk) for tk in tickers) if r]

    # Write the sidecar REGARDLESS of whether any rows were captured — an
    # all-failure run is exactly the case this exists to diagnose, and it
    # must not be the one case that produces no diagnostic file.
    try:
        import json
        os.makedirs(os.path.dirname(store) or ".", exist_ok=True)
        with open(_errors_path(store), "w") as f:
            json.dump({"run_at": datetime.now().isoformat(),
                      "errors": _RUN_ERRORS}, f, indent=2)
    except Exception as e:
        print(f"[etf_flow] could not write error sidecar: {e}")

    if not rows:
        print("[etf_flow] no snapshots captured")
        return pd.DataFrame()

    new = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(store) or ".", exist_ok=True)

    if os.path.exists(store):
        hist = pd.read_csv(store)
        today = new["date"].iloc[0]
        hist = hist[~((hist["date"] == today) & (hist["ticker"].isin(new["ticker"])))]
        out = pd.concat([hist, new], ignore_index=True)
    else:
        out = new

    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    out.to_csv(store, index=False)

    sources_used = new["shares_source"].value_counts().to_dict()
    print(f"[etf_flow] stored {len(new)} snapshots; history now {len(out)} rows. "
         f"Sources used today: {sources_used}")
    return new


# ── Flow computation (unchanged from v1) ────────────────────────────────────

def load_history(store: str = DEFAULT_STORE) -> pd.DataFrame:
    if not os.path.exists(store):
        return pd.DataFrame(columns=["date", "ticker", "shares_outstanding", "price"])
    df = pd.read_csv(store)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


def _flag_splits(g: pd.DataFrame) -> pd.Series:
    sh_ratio = g["shares_outstanding"] / g["shares_outstanding"].shift(1)
    px_ratio = g["price"] / g["price"].shift(1)
    product = sh_ratio * px_ratio
    return (sh_ratio.sub(1).abs() > 0.20) & (product.sub(1).abs() < 0.05)


def compute_flows(store: str = DEFAULT_STORE) -> pd.DataFrame:
    hist = load_history(store)
    if hist.empty or len(hist) < 2:
        return pd.DataFrame()

    frames = []
    for tk, g in hist.groupby("ticker"):
        g = g.sort_values("date").copy()
        if len(g) < 2:
            continue
        g["aum"] = g["shares_outstanding"] * g["price"]
        g["is_split"] = _flag_splits(g).fillna(False)

        d_shares = g["shares_outstanding"].diff()
        g["net_flow"] = (d_shares * g["price"]).where(~g["is_split"], np.nan)

        g["implausible"] = (g["net_flow"].abs() / g["aum"]) > IMPLAUSIBLE_DAILY_FLOW_PCT
        g.loc[g["implausible"], "net_flow"] = np.nan

        g["net_flow_5d"] = g["net_flow"].rolling(5, min_periods=2).sum()
        g["net_flow_20d"] = g["net_flow"].rolling(20, min_periods=5).sum()
        g["flow_pct_aum_20d"] = (g["net_flow_20d"] / g["aum"]) * 100
        frames.append(g)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def flow_vs_price_divergence(store: str = DEFAULT_STORE,
                             window: int = 20) -> pd.DataFrame:
    fl = compute_flows(store)
    if fl.empty:
        return pd.DataFrame()

    rows = []
    for tk, g in fl.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < window + 1:
            continue
        px_chg = float(g["price"].iloc[-1] / g["price"].iloc[-window - 1] - 1) * 100
        flow = float(g["net_flow"].iloc[-window:].sum(skipna=True))
        aum = float(g["aum"].iloc[-1])
        flow_pct = (flow / aum * 100) if aum else np.nan

        if px_chg > 0 and flow > 0:
            verdict = "CONFIRMED UPTREND"
        elif px_chg > 0 and flow < 0:
            verdict = "DISTRIBUTION (price up, money out)"
        elif px_chg < 0 and flow > 0:
            verdict = "ACCUMULATION (price down, money in)"
        else:
            verdict = "CONFIRMED DOWNTREND"

        rows.append({"ticker": tk, "days": len(g),
                     "price_chg_pct": round(px_chg, 2),
                     "net_flow_usd": round(flow, 0),
                     "net_flow_pct_aum": round(flow_pct, 2) if flow_pct == flow_pct else np.nan,
                     "verdict": verdict,
                     "divergence": verdict.startswith(("DISTRIBUTION", "ACCUMULATION"))})

    df = pd.DataFrame(rows)
    return df.sort_values("net_flow_pct_aum", ascending=False) if not df.empty else df


def coverage_report(store: str = DEFAULT_STORE) -> dict:
    hist = load_history(store)
    if hist.empty:
        return {"tickers": 0, "days": 0, "ready": False,
                "message": "No history yet. Run snapshot_all() once per trading "
                           "day. Flow readings need ~20 sessions to be useful; "
                           "divergence detection needs ~40."}
    per = hist.groupby("ticker")["date"].count()
    days = int(per.max())
    return {"tickers": int(hist["ticker"].nunique()), "days": days,
            "median_days": int(per.median()),
            "first_date": str(hist["date"].min().date()),
            "ready": days >= 20,
            "message": ("Sufficient history for 20-day flow readings."
                        if days >= 20 else
                        f"Only {days} sessions stored — need ~20. Keep polling daily.")}


# ── Fast verification — the piece that was missing last time ───────────────

def verify_new_source(store: str = DEFAULT_STORE, min_sessions: int = 2) -> dict:
    """
    Did shares_outstanding actually MOVE, checked after just a couple of
    sessions rather than waiting for 20. This is the check that should have
    existed from day one of v1 — the original bug ran for weeks before
    anyone looked closely enough to notice every delta was exactly zero.

    Run this after every deploy of a new/changed source, and again after the
    first 2-3 real trading days. Do not trust snapshot_all() output before
    this returns ok=True.
    """
    hist = load_history(store)
    if hist.empty:
        return {"ok": False, "checked": 0,
                "message": "No history yet — nothing to verify."}

    out = {"ok": None, "by_source": {}, "moved": 0, "static": 0,
          "insufficient": 0, "detail": []}

    for tk, g in hist.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < min_sessions:
            out["insufficient"] += 1
            continue
        src = g["shares_source"].iloc[-1]
        out["by_source"].setdefault(src, {"moved": 0, "static": 0})
        changed = g["shares_outstanding"].diff().abs().gt(0).any()
        if changed:
            out["moved"] += 1
            out["by_source"][src]["moved"] += 1
        else:
            out["static"] += 1
            out["by_source"][src]["static"] += 1
            out["detail"].append(f"{tk} ({src}): shares_outstanding "
                                 f"unchanged across {len(g)} sessions")

    total_checked = out["moved"] + out["static"]
    out["checked"] = total_checked
    if total_checked == 0:
        out["ok"] = False
        out["message"] = (f"{out['insufficient']} ticker(s) have fewer than "
                          f"{min_sessions} sessions — too early to verify. "
                          f"Check again after {min_sessions} sessions.")
    elif out["static"] == total_checked:
        out["ok"] = False
        out["message"] = (f"STILL BROKEN. All {total_checked} tickers with "
                          f"enough history show zero movement. The new "
                          f"source(s) are not resolving, or are ALSO "
                          f"static. Check the [etf_flow] log lines from "
                          f"snapshot_all() for per-source errors.")
    elif out["static"] > 0:
        out["ok"] = "partial"
        out["message"] = (f"PARTIALLY WORKING: {out['moved']}/{total_checked} "
                          f"tickers show real movement, {out['static']} "
                          f"still static. Check by_source breakdown — a "
                          f"specific issuer parser is likely still broken "
                          f"while others work.")
    else:
        out["ok"] = True
        out["message"] = (f"WORKING: all {total_checked} tickers with enough "
                          f"history show real day-to-day movement.")
    return out


# ── The actual recommended fix, if you want this done right ────────────────

def aws_data_exchange_stub(dataset_arn: str = "", region: str = "us-east-1"):
    """
    Integration shape for ETF Global's "ETF Daily Fund Flows – US Listed"
    dataset via AWS Data Exchange, once subscribed via the AWS Marketplace
    console (a step this code cannot do on your behalf).

    This dataset carries shares outstanding, NAV, AND net daily flow
    DIRECTLY — no differencing, no per-issuer scraping, no 20-session wait,
    history back to 2017. It is the actual recommended fix; everything
    above is the free-tier best effort in its absence.

    NOT IMPLEMENTED — genuinely need the ARN and confirmed response shape
    from an active subscription to build this correctly. Once subscribed,
    share the console's export job output/schema and this becomes a
    straightforward boto3 `dataexchange` client call, replacing every
    function above.
    """
    raise NotImplementedError(
        "Subscribe to ETF Global's dataset via AWS Data Exchange first "
        "(search 'ETF Daily Fund Flows' in the AWS Marketplace console). "
        "Once subscribed, share the export job's response shape and this "
        "becomes a real, working integration."
    )
