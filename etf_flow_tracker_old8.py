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
from datetime import datetime, timedelta

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
    # v3, Sept 2026: expanded to match top_movers.ETF_UNIVERSE exactly.
    # Before this, only 14 of the 40 tickers the "Where's the Money" ranking
    # could display were ever polled for Tier A capital data -- the other 26
    # were structurally incapable of showing a Tier A badge, not because
    # their flow data was worse, but because they were never in this list at
    # all. This closes that gap; every ranked ticker is now Tier-A-eligible.
    "SOXX", "IGV", "SKYY", "HACK",              # Technology
    "KBE", "IAI",                                 # Financial
    "XBI", "IHI", "PPH",                          # Healthcare
    "OIH", "AMLP",                                # Energy
    "XTN",                                         # Industrials
    "XRT", "XHB", "PBJ",                          # Consumer
    "IWO", "MDY",                                  # Broad Market
    "LQD", "EMB", "TIP",                          # Fixed Income
    "USO", "DBA",                                  # Commodities
    "EFA", "EWJ", "FXI", "INDA",                  # International
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
        # v3: the /us/en/individual/etfs/ prefix 404'd via a server redirect
        # to this shorter path (confirmed from the actual captured error on
        # XLE's first live run) -- trying the redirect target directly.
        # Still unverified whether the FILE exists at this path; only the
        # path PREFIX is corrected from real evidence.
        url = (f"https://www.ssga.com/library-content/"
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
        if not shares or not price:
            _log_error(ticker, "yfinance", RuntimeError(
                "fast_info and .info both returned no usable shares/price"))
        return shares, price
    except Exception as e:
        _log_error(ticker, "yfinance", e)
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

    # v3 REORDER, Sept 2026: aum_implied tried FIRST. The first real run
    # against live data confirmed it (20/31 tickers moving) while every
    # issuer-direct guess so far returned 404/500 -- see the specific
    # errors captured in load_last_run_errors(). Empirical evidence beats
    # a plausible-sounding guess; issuer-direct stays as a second attempt
    # in case a fix lands or a vendor's site changes, but no longer gates
    # the primary path.
    aum_shares, aum_price = _shares_from_aum_implied(ticker)
    if aum_shares:
        shares, price, used = aum_shares, aum_price, "aum_implied"

    if shares is None:
        primary = _issuer_source_for(ticker)
        if primary is not None:
            s, p = primary(ticker)
            if s and p:
                shares, price, used = s, p, primary.__name__.replace("_shares_from_", "")

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


# ── Stale-AUM guard (Sept 2026) ──────────────────────────────────────────────
# aum_implied computes shares = totalAssets / price. When yfinance's
# totalAssets does not update, the implied share count moves EXACTLY inverse
# to price: a falling price reads as an inflow, a rising price as an outflow.
# Measured on the live store (2026-09-24): implied AUM was unchanged on 57%
# of day-over-day observations, and for 35 of 57 tickers corr(Δln shares,
# Δln price) was -1.00 across every stored session. The old integrity test
# (shares.nunique() > 1) passed all of them, because a stale AUM divided by
# a moving price always "moves".
#
# Rules:
#   * An aum_implied row whose implied AUM equals the prior row's carries NO
#     flow information. Its flow is NaN, never a number.
#   * Flow is measured between consecutive AUM UPDATES only, so a real
#     change that arrives after several stale days is counted once, in full.
#   * A ticker is trustworthy only if its source is an issuer feed, or if
#     implied AUM updated on >= MIN_AUM_UPDATE_RATE of sessions AND its share
#     changes are not a mirror image of price (corr > PRICE_MIRROR_CORR).
AUM_UNCHANGED_TOL = 1e-6        # relative change below this = "not updated"
MIN_AUM_UPDATE_RATE = 0.40      # of the last 20 sessions on the current source
PRICE_MIRROR_CORR = -0.80
MIN_QUALITY_SESSIONS = 5        # below this, too little history to judge
MAX_SESSIONS_SINCE_UPDATE = 3   # AUM unchanged longer than this = feed frozen


def _stale_aum_rows(g: pd.DataFrame) -> pd.Series:
    """True for aum_implied rows whose implied AUM didn't change vs the prior row."""
    aum = g["shares_outstanding"] * g["price"]
    unchanged = aum.pct_change().abs() < AUM_UNCHANGED_TOL
    if "shares_source" in g.columns:
        unchanged &= g["shares_source"].eq("aum_implied")
    return unchanged.fillna(False)


def ticker_quality(g: pd.DataFrame, window: int = 20) -> dict:
    """Is this ticker's share series real capital data, or a price echo?

    Judged on the CURRENT source only (rows since the source last changed),
    and on the most recent `window` sessions of it — a feed that updated in
    early September and froze afterwards is not trustworthy today.
    yfinance sharesOutstanding is never trusted (annual granularity; see
    principles). Returns {"sessions", "source", "aum_update_rate",
    "last_update", "price_mirror_corr", "moving", "artifact", "trustworthy",
    "reason"}."""
    g = g.sort_values("date")
    src = g["shares_source"].iloc[-1] if "shares_source" in g.columns and len(g) else None
    if "shares_source" in g.columns and len(g):
        run_id = g["shares_source"].ne(g["shares_source"].shift()).cumsum()
        g = g[run_id == run_id.iloc[-1]]            # current-source tail only
    n = len(g)
    out = {"sessions": n, "source": src, "aum_update_rate": None, "last_update": None,
           "price_mirror_corr": None, "moving": False, "artifact": False,
           "trustworthy": False, "reason": ""}
    if n < 2:
        out["reason"] = f"fewer than 2 sessions on current source ({src})"
        return out
    out["moving"] = bool(g["shares_outstanding"].nunique(dropna=True) > 1)
    if src == "yfinance":
        out["reason"] = "yfinance sharesOutstanding is annual-granularity — never capital data"
        return out
    if src != "aum_implied":
        out["trustworthy"] = out["moving"]
        out["reason"] = f"issuer source ({src})" if out["moving"] else f"{src}: shares never changed"
        return out

    stale_all = _stale_aum_rows(g)
    upd_dates = g.loc[~stale_all, "date"].iloc[1:] if n > 1 else pd.Series(dtype=object)
    out["last_update"] = str(pd.to_datetime(upd_dates.max()).date()) if len(upd_dates) else None
    stale = stale_all.iloc[1:].iloc[-window:]
    out["aum_update_rate"] = round(float(1 - stale.mean()), 3)
    # Sessions since the last real AUM update: a feed that froze recently is
    # dead today even if its 20-session rate still looks fine.
    tail_stale = stale_all.iloc[1:][::-1]
    out["sessions_since_update"] = int(tail_stale.cumprod().sum()) if len(tail_stale) else 0
    dls = np.log(g["shares_outstanding"]).diff()
    dlp = np.log(g["price"]).diff()
    mask = dlp.abs() > 1e-6
    if mask.sum() >= 3 and dls[mask].std() > 0:
        out["price_mirror_corr"] = round(float(dls[mask].corr(dlp[mask])), 3)
    mirror = out["price_mirror_corr"] is not None and out["price_mirror_corr"] <= PRICE_MIRROR_CORR
    slow = out["aum_update_rate"] < MIN_AUM_UPDATE_RATE
    frozen = out["sessions_since_update"] > MAX_SESSIONS_SINCE_UPDATE
    out["artifact"] = bool(mirror or slow or frozen)
    if n < MIN_QUALITY_SESSIONS:
        out["reason"] = f"only {n} sessions — too few to trust an aum_implied series"
    elif out["artifact"]:
        why = []
        if slow:
            why.append(f"AUM updated on only {out['aum_update_rate']:.0%} of the last "
                       f"{min(window, n - 1)} sessions (last update {out['last_update'] or 'never'})")
        if frozen:
            why.append(f"AUM frozen for the last {out['sessions_since_update']} sessions")
        if mirror:
            why.append(f"share changes mirror price (corr {out['price_mirror_corr']:+.2f})")
        out["reason"] = "stale-AUM artifact: " + "; ".join(why)
    else:
        out["trustworthy"] = out["moving"]
        out["reason"] = (f"aum_implied, AUM updated {out['aum_update_rate']:.0%} of sessions"
                         if out["moving"] else "aum_implied: shares never changed")
    return out


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
        g["stale_aum"] = _stale_aum_rows(g)

        # Measure share change between consecutive INFORMATIVE rows only
        # (issuer rows, or aum_implied rows where AUM actually updated).
        # Stale rows get NaN: no information, not zero and not a price echo.
        informative = ~g["stale_aum"]
        d_shares = g["shares_outstanding"].where(informative).ffill().diff()
        d_shares = d_shares.where(informative)
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
        q = ticker_quality(g)
        if not q["trustworthy"]:
            # Sept 2026: a stale-AUM series says nothing about flow. Report it
            # as unreliable instead of letting price-inverse noise become an
            # ACCUMULATION / DISTRIBUTION verdict.
            rows.append({"ticker": tk, "days": len(g),
                         "price_chg_pct": round(px_chg, 2),
                         "net_flow_usd": np.nan, "net_flow_pct_aum": np.nan,
                         "verdict": f"UNRELIABLE — {q['reason']}",
                         "divergence": False})
            continue
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
          "insufficient": 0, "insufficient_tickers": [],
          "never_seen_tickers": [], "detail": []}

    seen = set(hist["ticker"].unique())
    out["never_seen_tickers"] = sorted(set(TRACKED) - seen)

    for tk, g in hist.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < min_sessions:
            out["insufficient"] += 1
            out["insufficient_tickers"].append(tk)
            continue
        src = g["shares_source"].iloc[-1]
        out["by_source"].setdefault(src, {"moved": 0, "static": 0})
        # Sept 2026: "moved" now means genuinely moved — a stale-AUM series
        # that only echoes price counts as static, with the reason named.
        q = ticker_quality(g)
        if q["trustworthy"]:
            out["moved"] += 1
            out["by_source"][src]["moved"] += 1
        else:
            out["static"] += 1
            out["by_source"][src]["static"] += 1
            out["detail"].append(f"{tk} ({src}): {q['reason'] or 'shares_outstanding unchanged'} "
                                 f"across {len(g)} sessions")

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

def backfill_shares_history(ticker: str, days: int = 60) -> "pd.Series | None":
    """
    Historical shares outstanding via yfinance's get_shares_full() -- a
    genuinely different endpoint from everything else in this file. Every
    OTHER source here (.info's sharesOutstanding, .info's totalAssets)
    is a LIVE SNAPSHOT with no memory: asking it "what was this on August
    15th" is not a question it can answer, which is exactly why a brand-new
    ticker with one successful row has to wait for tomorrow's poll to prove
    anything -- there is nothing to compare it against yet.

    get_shares_full() is different: it is Yahoo's own historical chart-data
    endpoint, commonly used to reconstruct historical market cap for STOCKS.
    Whether it has real per-day resolution for ETFS SPECIFICALLY is
    UNVERIFIED -- untested against live data, same honest caveat as every
    other source in this file. If it works, this turns "wait 2-3 real
    trading days to find out" into "check right now" for exactly the
    tickers currently stuck in the insufficient/too-new-to-judge bucket.

    Returns None (never raises) if the method is unavailable, returns
    nothing, or every value is identical (which would mean this source has
    the SAME snapshot-only limitation as the others, just discovered
    immediately instead of after a multi-day wait).
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        s = t.get_shares_full(start=start)
        if s is None or len(s) == 0:
            return None
        return s
    except Exception as e:
        print(f"[etf_flow][backfill] {ticker}: {type(e).__name__}: {e}")
        return None


def verify_via_backfill(tickers: list[str], days: int = 60) -> dict:
    """
    Run backfill_shares_history() against a specific list of tickers (meant
    for exactly the ones currently stuck as "too new to judge") and report,
    per ticker, whether REAL historical variation shows up -- an immediate
    answer instead of a multi-day wait.
    """
    out = {"available": {}, "unavailable": [], "checked_at": datetime.now().isoformat()}
    for tk in tickers:
        hist = backfill_shares_history(tk, days=days)
        if hist is None or len(hist) < 2:
            out["unavailable"].append(tk)
            continue
        n_unique = hist.nunique(dropna=True)
        out["available"][tk] = {
            "sessions": len(hist), "unique_values": int(n_unique),
            "varies": n_unique > 1,
            "first": float(hist.iloc[0]), "last": float(hist.iloc[-1]),
        }
    return out


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


# ── Selftest: stale-AUM guard (offline, synthetic store) ────────────────────

def selftest() -> dict:
    import tempfile
    f = []
    dates = pd.bdate_range("2026-08-03", periods=30)
    rng = np.random.default_rng(1)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))

    rows = []
    # REAL: true shares grow 0.5%/day (inflow); AUM = shares x price, updates daily.
    sh = 1_000_000 * np.cumprod(np.full(len(dates), 1.005))
    for d, p, s in zip(dates, px, sh):
        rows.append({"date": d.date().isoformat(), "ticker": "REAL", "shares_outstanding": s,
                     "price": p, "shares_source": "aum_implied"})
    # ECHO: AUM frozen at 100M the whole time -> implied shares = 100M / price.
    for d, p in zip(dates, px):
        rows.append({"date": d.date().isoformat(), "ticker": "ECHO", "shares_outstanding": 100e6 / p,
                     "price": p, "shares_source": "aum_implied"})
    # LAPSED: real for 20 sessions, then AUM freezes for the last 10.
    frozen_aum = None
    for i, (d, p, s) in enumerate(zip(dates, px, sh)):
        if i < 20:
            shares = s
        else:
            frozen_aum = frozen_aum or sh[19] * px[19]
            shares = frozen_aum / p
        rows.append({"date": d.date().isoformat(), "ticker": "LAPSED", "shares_outstanding": shares,
                     "price": p, "shares_source": "aum_implied"})
    # ISSUER: direct share counts, static then one creation.
    for i, (d, p) in enumerate(zip(dates, px)):
        rows.append({"date": d.date().isoformat(), "ticker": "ISSUER",
                     "shares_outstanding": 5e6 if i < 15 else 5.2e6, "price": p,
                     "shares_source": "issuer_spdr"})
    store = os.path.join(tempfile.mkdtemp(), "hist.csv")
    pd.DataFrame(rows).to_csv(store, index=False)
    hist = load_history(store)
    q = {tk: ticker_quality(g) for tk, g in hist.groupby("ticker")}

    if not q["REAL"]["trustworthy"]:
        f.append(f"REAL should be trustworthy: {q['REAL']}")
    if q["ECHO"]["trustworthy"] or not q["ECHO"]["artifact"]:
        f.append(f"ECHO (frozen AUM) must be an artifact: {q['ECHO']}")
    if q["LAPSED"]["trustworthy"]:
        f.append(f"LAPSED (froze 10 of last 20 sessions... ) should fail the recent-update test: {q['LAPSED']}")
    if not q["ISSUER"]["trustworthy"]:
        f.append("issuer source with a real creation must be trustworthy")

    fl = compute_flows(store)
    echo = fl[fl["ticker"] == "ECHO"]
    if echo["net_flow"].notna().any():
        f.append("frozen-AUM rows must carry NaN flow, never a number")
    real = fl[fl["ticker"] == "REAL"]
    if not (real["net_flow"].dropna() > 0).all():
        f.append("REAL inflows must be positive every day")

    div = flow_vs_price_divergence(store, window=20)
    v = dict(zip(div["ticker"], div["verdict"]))
    if not str(v.get("ECHO", "")).startswith("UNRELIABLE"):
        f.append(f"ECHO divergence must read UNRELIABLE: {v.get('ECHO')}")
    if str(v.get("REAL", "")).startswith("UNRELIABLE"):
        f.append(f"REAL divergence must produce a verdict: {v.get('REAL')}")
    if bool(div.loc[div["ticker"] == "ECHO", "divergence"].any()):
        f.append("an unreliable ticker must never be flagged as a divergence")
    return {"ok": not f, "failures": f,
            "quality": {k: (x["trustworthy"], x["aum_update_rate"], x["price_mirror_corr"]) for k, x in q.items()}}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, default=str))
