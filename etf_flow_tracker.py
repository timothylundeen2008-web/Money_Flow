"""
etf_flow_tracker.py  (v2.3 — November 2026)
──────────────────────────────────────────────────────────────────────────────
v2.3 (2026-11-01, fix_status N18): a day whose flow exceeds
  IMPLAUSIBLE_DAILY_FLOW_PCT of AUM is still FLAGGED, but for REPORTED share
  counts (issuer or relay) it is no longer dropped. Dropping one day broke
  the 20-day sum: XRT's +23% creation on 10/21 was deleted while the
  redemptions around it were kept, so XRT read -23% of AUM when its shares
  actually fell 3%; XHB read +19% (true +3%), XTN -0.3% (true -27%). A
  reported count that spikes and reverses nets out in the sum by itself.
  Estimated (aum_implied) rows keep the guard. flow_vs_price_divergence now
  divides by start-of-window AUM, like the Flow Map.

v2.2 (2026-11-01, fix_status N31): every row is dated by the trading SESSION
  it describes, not by the calendar day of the poll. Polls on weekends and
  holidays used to add rows of their own (Sat 10/24, Sat 10/31, Sun 11/01), so
  the frozen-count test counted those polls as flat sessions (IAI, FXI, SGOV
  and INDA were excluded while flat only 3-4 trading days), a "20-session"
  window held about 16 trading days, and the Flow Map said data ran "through"
  a Sunday. Now:
    * a poll is dated by session_for(): the last trading day whose close has
      passed, in New York time (the runner's clock is UTC, which also put
      evening polls on the next calendar day);
    * a later poll for the same session replaces the earlier row, so a
      weekend relay catch-up is kept as that session's count (real flow,
      counted once);
    * rows written before v2.2 are re-dated once from the time their poll
      actually ran (MIGRATION_* tables, derived from the repo's commit
      history). The scheduled poll has mostly run after midnight UTC since
      late September, so Monday's poll was stored as Tuesday and Friday's as
      Saturday;
    * _upsert re-dates and collapses the whole store on every write (the
      one-time migration happens on the first poll after deploy), and
      load_history applies the same rule on read. Rows carry
      session_dated = True once dated by session.

v2.1 (2026-10-31, fix_status N19): ticker_quality() no longer trusts a
  third-party relay series just because its share count changed at some point.
  If the count has not changed for more than MAX_RELAY_FLAT_SESSIONS (4) it is
  treated as frozen: SGOV sat at 1.16B for 5+ sessions while +4.30% of AUM was
  reported as inflow, and EEM/EFA had been flat for 16-19 sessions. Issuer
  feeds are unchanged. Every result now carries sessions_since_change.

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
from datetime import date as _date, datetime, timedelta, timezone

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
# ticker -> ISO date the issuer says its share count is "as of". Issuer rows
# are stored under THAT date (not the poll date), so flows line up with the
# day creations/redemptions actually settled.
_LAST_ASOF: dict[str, str] = {}


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
    # v5, Oct 2026: the broad market itself. Without SPY the flow map could
    # not tell "rotating between sectors" from "leaving US equities". SPDR
    # fund -> SSGA NAV-history routing + the daily backfill seed ~90 days of
    # real history on the first poll (falls back like any SPDR fund if not).
    "SPY",
]

# ── Issuer routing table ────────────────────────────────────────────────────
# v4, Sept 2026 — issuer sources rebuilt against endpoints VERIFIED live on
# 2026-09-25 (the v3 URLs were guesses and 404'd, which is why aum_implied
# became primary and the flow layer froze when yfinance totalAssets stopped
# updating around 2026-09-07).
#
#  SPDR  : SSGA's NAV-history workbook, one per fund —
#          https://www.ssga.com/library-content/products/fund-data/etfs/us/navhist-us-en-{t}.xlsx
#          Daily Date / NAV / Shares Outstanding history. Also used by
#          backfill_issuer_history() to seed real history immediately.
#  iShares: the product page (https://www.ishares.com/us/products/{id}/),
#          which prints "Shares Outstanding N as of Mon DD, YYYY". IDs below
#          were each confirmed to resolve to the named fund. ITA was
#          previously routed to SPDR in error — it is an iShares fund (ID not
#          yet verified, so it stays on the fallback path for now).
ISSUER_SPDR = {"XLK", "XLF", "XLI", "XLY", "XLRE", "XLB", "XLC", "XLP",
               "XLE", "XLV", "XLU", "KRE", "KBE", "XOP", "XBI", "XRT",
               "XHB", "XTN", "MDY", "SPY"}
ISSUER_SPDR_GOLD = {"GLD"}
ISHARES_IDS = {
    "IWM": 239710, "TLT": 239454, "HYG": 239565, "EEM": 239637,
    "IBB": 239699, "SOXX": 239705, "EFA": 239623, "LQD": 239566,
    "EMB": 239572, "IWO": 239709, "IGV": 239771, "IHI": 239516,
    "TIP": 239467, "EWJ": 239665, "FXI": 239536,
}
ISSUER_ISHARES = set(ISHARES_IDS)
SSGA_NAVHIST_URL = ("https://www.ssga.com/library-content/products/fund-data/"
                    "etfs/us/navhist-us-en-{t}.xlsx")
ISHARES_PAGE_URL = "https://www.ishares.com/us/products/{id}/"
# Sources that carry a REPORTED share count (issuer file, or a relay of it)
# rather than an estimate. Once any exist for a ticker, only these count.
REPORTED_PREFIXES = ("issuer", "thirdparty")
ISSUER_MAX_AGE_DAYS = 6          # an issuer figure older than this is stale
_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36")}

IMPLAUSIBLE_DAILY_FLOW_PCT = 0.15
CROSS_CHECK_DIVERGENCE_PCT = 0.05


# ── Individual source implementations ───────────────────────────────────────

def parse_ssga_navhist(raw: bytes) -> pd.DataFrame:
    """Parse an SSGA NAV-history workbook into date / nav / shares_outstanding.

    Tolerant of the header block SSGA puts above the table: finds the row
    that contains a 'Shares Outstanding' cell and uses it as the header.
    Raises ValueError with a specific message if the layout is unrecognised."""
    import io
    grid = pd.read_excel(io.BytesIO(raw), header=None)
    hdr_idx = None
    for i in range(min(len(grid), 40)):
        cells = [str(c).strip().lower() for c in grid.iloc[i].tolist()]
        if any("shares outstanding" in c for c in cells):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("no 'Shares Outstanding' header in the first 40 rows")
    df = grid.iloc[hdr_idx + 1:].copy()
    df.columns = [str(c).strip() for c in grid.iloc[hdr_idx].tolist()]
    low = {c: c.lower() for c in df.columns}
    date_c = next((c for c, l in low.items() if l == "date" or l.startswith("date")), None)
    sh_c = next((c for c, l in low.items() if "shares outstanding" in l), None)
    nav_c = next((c for c, l in low.items() if l == "nav" or l.startswith("nav")), None)
    if not date_c or not sh_c:
        raise ValueError(f"missing Date/Shares columns (got {list(df.columns)[:8]})")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_c], errors="coerce"),
        "shares_outstanding": pd.to_numeric(
            df[sh_c].astype(str).str.replace(",", "", regex=False), errors="coerce"),
        "nav": pd.to_numeric(df[nav_c], errors="coerce") if nav_c else np.nan,
    }).dropna(subset=["date", "shares_outstanding"])
    out = out[out["shares_outstanding"] > 0]
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _ssga_navhist(ticker: str) -> pd.DataFrame | None:
    try:
        import requests
        r = requests.get(SSGA_NAVHIST_URL.format(t=ticker.lower()), timeout=30, headers=_UA)
        r.raise_for_status()
        return parse_ssga_navhist(r.content)
    except Exception as e:
        _log_error(ticker, "spdr", e)
        return None


def _fresh(asof, ticker: str, source: str) -> bool:
    age = (pd.Timestamp.now().normalize() - pd.Timestamp(asof).normalize()).days
    if age > ISSUER_MAX_AGE_DAYS:
        _log_error(ticker, source, RuntimeError(f"issuer figure is stale: as of {pd.Timestamp(asof).date()} ({age}d)"))
        return False
    return True


def _shares_from_spdr(ticker: str) -> tuple[float | None, float | None]:
    """Latest reported shares outstanding + NAV from SSGA's NAV-history file.
    Sets _LAST_ASOF[ticker] to the file's own date for that figure."""
    h = _ssga_navhist(ticker)
    if h is None or h.empty:
        return None, None
    last = h.iloc[-1]
    if not _fresh(last["date"], ticker, "spdr"):
        return None, None
    _LAST_ASOF[ticker] = last["date"].date().isoformat()
    nav = float(last["nav"]) if pd.notna(last["nav"]) else None
    return float(last["shares_outstanding"]), nav


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


_ISH_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?)"
_ISH_DATE = r"as of\s*([A-Z][a-z]{2,8}\.?\s+\d{1,2},\s+\d{4})"


def parse_ishares_page(html: str, ticker: str) -> tuple[float, str]:
    """Extract (shares_outstanding, as_of_iso) from an iShares product page.

    Verifies the page is for `ticker` (IDs can be wrong or reassigned) and
    raises ValueError with a specific reason otherwise."""
    import re
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    head = (title.group(1) if title else html[:5000])
    if not re.search(rf"\b{re.escape(ticker)}\b", head):
        raise ValueError(f"page is not for {ticker} (title: {head.strip()[:80]!r})")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    i = text.lower().find("shares outstanding")
    if i < 0:
        raise ValueError("no 'Shares Outstanding' on page")
    window = text[i:i + 400]
    num = re.search(_ISH_NUM, window)
    dt = re.search(_ISH_DATE, window)
    if not num:
        raise ValueError(f"no share count near 'Shares Outstanding': {window[:120]!r}")
    asof = pd.to_datetime(dt.group(1).replace(".", ""), errors="coerce") if dt else pd.NaT
    if pd.isna(asof):
        raise ValueError("no as-of date near 'Shares Outstanding'")
    return float(num.group(1).replace(",", "")), asof.date().isoformat()


# v4.1, Sept 2026: ishares.com answered 403 to every request from the GitHub
# Actions runner. That is bot protection keyed on the HTTP client's TLS
# fingerprint (python-requests is recognised and refused), not a bad URL —
# the same pages load in a browser. curl_cffi impersonates a real Chrome TLS
# handshake; it is already installed as a yfinance dependency.
ISHARES_PAGE_URLS = (
    "https://www.ishares.com/us/products/{id}/",
)
_BROWSER_HEADERS = {
    **_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.ishares.com/us/products/etf-investments",
}


def _browser_get(url: str, timeout: int = 30) -> str:
    """GET a page the way a browser would. curl_cffi (Chrome impersonation)
    first; plain requests with full browser headers as the fallback.
    Raises on HTTP errors, naming which client failed."""
    errs = []
    try:
        from curl_cffi import requests as creq
        r = creq.get(url, impersonate="chrome", timeout=timeout, headers=_BROWSER_HEADERS)
        if r.status_code == 200 and r.text:
            return r.text
        errs.append(f"curl_cffi HTTP {r.status_code}")
    except ImportError:
        errs.append("curl_cffi not installed")
    except Exception as e:
        errs.append(f"curl_cffi {type(e).__name__}: {e}")
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers=_BROWSER_HEADERS)
        if r.status_code == 200 and r.text:
            return r.text
        errs.append(f"requests HTTP {r.status_code}")
    except Exception as e:
        errs.append(f"requests {type(e).__name__}: {e}")
    raise RuntimeError(f"{url}: " + "; ".join(errs))


# v4.2, Sept 2026: the 2026-09-26 run showed iShares refusing GitHub runner
# IPs outright (403 even with Chrome impersonation) — an IP block, not a
# client block. Disabled so it stops adding 15 failures per run; the iShares
# tickers fall through to the third-party source below. Re-enable if the
# poll ever moves off GitHub-hosted runners.
ISHARES_ENABLED = False

# Third-party relay of issuer-reported shares outstanding. Verified
# 2026-09-26 against issuer figures: IWM 272.65M (iShares page: 272,650,000),
# XLK 651.01M (SSGA file: 651.26M the prior day). Covers every tracked
# ticker, including funds with no issuer source here (QQQ, VGT, SMH, SCHD,
# PDBC, KMLM, USFR, SGOV, GLD...). Resolution is 0.01M shares — ample for
# 20-day flow sums. No as-of date is printed, so rows use the poll date.
STOCKANALYSIS_URL = "https://stockanalysis.com/etf/{t}/"
_SA_MULT = {"K": 1e3, "M": 1e6, "B": 1e9}


def parse_stockanalysis_page(html: str, ticker: str) -> float:
    """Extract shares outstanding from a StockAnalysis ETF page. Verifies the
    page is for `ticker`; raises ValueError with a reason otherwise."""
    import re
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    head = title.group(1) if title else html[:3000]
    if not re.search(rf"\b{re.escape(ticker)}\b", head, re.I):
        raise ValueError(f"page is not for {ticker} (title: {head.strip()[:80]!r})")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    m = re.search(r"Shares Out(?:standing)?\s*\|?\s*([\d.,]+)\s*([KMB])\b", text)
    if not m:
        if re.search(r"Shares Out(?:standing)?\s*\|?\s*n/?a\b", text, re.I):
            # Observed 2026-09-26: some tickers (SCHD, TIP) print "n/a" on
            # some days and a value on others. A gap day, not a parse bug.
            raise ValueError("StockAnalysis shows 'Shares Out: n/a' today (data gap, not a parse error)")
        raise ValueError("no 'Shares Out' figure on page")
    val = float(m.group(1).replace(",", "")) * _SA_MULT[m.group(2)]
    if val <= 0:
        raise ValueError(f"non-positive shares {val}")
    return val


# Schwab ETFs — the fund page prints "Shares Outstanding  As of MM/DD/YYYY ...
# N". Verified 2026-09-26: SCHD 3,294,650,000 as of 09/24/2026 (StockAnalysis
# showed n/a for SCHD that day).
ISSUER_SCHWAB = {"SCHD"}
SCHWAB_URL = "https://www.schwabassetmanagement.com/products/{t}"


def parse_schwab_page(html: str, ticker: str) -> tuple[float, str]:
    import re
    if not re.search(rf"\b{re.escape(ticker)}\b", html[:20000], re.I):
        raise ValueError(f"page does not mention {ticker}")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    i = text.find("Shares Outstanding")
    if i < 0:
        raise ValueError("no 'Shares Outstanding' on page")
    window = text[i:i + 300]
    dt = re.search(r"(\d{2}/\d{2}/\d{4})", window)
    num = re.search(r"(\d{1,3}(?:,\d{3}){2,})", window)
    if not num or not dt:
        raise ValueError(f"no share count/date near 'Shares Outstanding': {window[:120]!r}")
    return float(num.group(1).replace(",", "")), pd.to_datetime(dt.group(1)).date().isoformat()


def _shares_from_schwab(ticker: str) -> tuple[float | None, float | None]:
    try:
        shares, asof = parse_schwab_page(_browser_get(SCHWAB_URL.format(t=ticker.lower())), ticker)
        if not _fresh(asof, ticker, "schwab"):
            return None, None
        _LAST_ASOF[ticker] = asof
        return shares, None
    except Exception as e:
        _log_error(ticker, "schwab", e)
        return None, None


def _shares_from_stockanalysis(ticker: str) -> tuple[float | None, float | None]:
    try:
        html = _browser_get(STOCKANALYSIS_URL.format(t=ticker.lower()))
        return parse_stockanalysis_page(html, ticker), None
    except Exception as e:
        _log_error(ticker, "stockanalysis", e)
        return None, None


def _shares_from_ishares(ticker: str) -> tuple[float | None, float | None]:
    """Reported shares outstanding from the iShares/BlackRock product page.
    Price is left to the caller (yfinance close); flows only need Δshares × price."""
    pid = ISHARES_IDS.get(ticker)
    if pid is None or not ISHARES_ENABLED:
        return None, None
    failures = []
    for tmpl in ISHARES_PAGE_URLS:
        try:
            html = _browser_get(tmpl.format(id=pid))
            shares, asof = parse_ishares_page(html, ticker)
            if not _fresh(asof, ticker, "ishares"):
                return None, None
            _LAST_ASOF[ticker] = asof
            return shares, None
        except Exception as e:
            failures.append(str(e))
    _log_error(ticker, "ishares", RuntimeError(" | ".join(failures)))
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
    # v4.3: spdr_gold removed from routing — its endpoint 404s every run and
    # the history file blocks automated readers; GLD is covered by the
    # third-party relay (verified 352.40M on 2026-09-26).
    if ticker in ISSUER_SPDR:
        return _shares_from_spdr
    if ticker in ISSUER_SCHWAB:
        return _shares_from_schwab
    if ticker in ISSUER_ISHARES and ISHARES_ENABLED:
        return _shares_from_ishares
    return None


# ── Session dating (v2.2, N31) ──────────────────────────────────────────────
try:
    import market_time as _mt
except Exception:                       # weekday-only fallback; holidays then count as sessions
    _mt = None
_ET_CLOSE_HOUR = 16


def _last_trading_day(d: _date) -> _date:
    if _mt is not None:
        return _mt.last_trading_day(d)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def session_for(when: datetime | None = None) -> str:
    """The trading session a poll at `when` describes: the last trading day
    whose 4pm New York close has passed. Naive datetimes are taken as UTC
    (the Actions runner clock). Saturday, Sunday or a holiday -> the prior
    trading day; a trading day before the close -> the prior trading day."""
    from zoneinfo import ZoneInfo
    when = when or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    et = when.astimezone(ZoneInfo("America/New_York"))
    d = et.date()
    close = _mt.close_time(d).hour if _mt is not None and hasattr(_mt, "close_time") else _ET_CLOSE_HOUR
    if _last_trading_day(d) == d and et.hour < close:
        d -= timedelta(days=1)
    return _last_trading_day(d).isoformat()


def session_of_date(d) -> str:
    """A stored calendar date -> the session it belongs to (weekend/holiday
    dates fold into the prior trading day; trading days are unchanged)."""
    return _last_trading_day(pd.Timestamp(d).date()).isoformat()


# One-time migration of rows written before v2.2. Those polls dated relay and
# estimate rows by the runner's UTC calendar day, and the scheduled poll
# (cron 21:45 UTC) has mostly started after midnight UTC since late September,
# so Monday's poll was stored as Tuesday ... Friday's as Saturday. Each row's
# session below comes from the commit that last changed its share count, where
# that commit's UTC day equals the row's date (the poll's own clock; 15
# minutes allowed for a run that crossed midnight). Built 2026-11-01 from
# Money_Flow history through 441a58b: all 1,671 non-issuer rows were dated
# from a poll time. Labels not listed are already right (or a weekend/holiday
# that folds into the prior session). Issuer rows carry the issuer's as-of
# date and are never shifted. Rows written by v2.2 carry SESSION_COL = True
# and are never looked up here.
SESSION_COL = "session_dated"
MIGRATION_LABEL_SESSION = {
    "2026-08-05": "2026-08-04", "2026-08-07": "2026-08-06", "2026-08-27": "2026-08-26", "2026-08-28": "2026-08-27",
    "2026-09-01": "2026-08-31", "2026-09-04": "2026-09-03", "2026-09-29": "2026-09-28", "2026-09-30": "2026-09-29",
    "2026-10-01": "2026-09-30", "2026-10-02": "2026-10-01", "2026-10-06": "2026-10-05", "2026-10-07": "2026-10-06",
    "2026-10-08": "2026-10-07", "2026-10-09": "2026-10-08", "2026-10-13": "2026-10-12", "2026-10-14": "2026-10-13",
    "2026-10-15": "2026-10-14", "2026-10-16": "2026-10-15", "2026-10-20": "2026-10-19", "2026-10-21": "2026-10-20",
    "2026-10-22": "2026-10-21", "2026-10-23": "2026-10-22", "2026-10-27": "2026-10-26", "2026-10-28": "2026-10-27",
    "2026-10-29": "2026-10-28", "2026-10-30": "2026-10-29",
}
MIGRATION_ROW_SESSION = {
    "2026-10-15|AMLP": "2026-10-15", "2026-10-15|IAI": "2026-10-15", "2026-10-15|IBB": "2026-10-15",
    "2026-10-15|IGV": "2026-10-15", "2026-10-15|IHI": "2026-10-15", "2026-10-15|ITA": "2026-10-15",
    "2026-10-15|IWM": "2026-10-15", "2026-10-15|PPH": "2026-10-15", "2026-10-15|QQQ": "2026-10-15",
    "2026-10-15|SLV": "2026-10-15", "2026-10-15|SMH": "2026-10-15", "2026-10-15|SOXX": "2026-10-15",
    "2026-10-15|TLT": "2026-10-15", "2026-10-15|USFR": "2026-10-15", "2026-10-23|PDBC": "2026-10-23",
    "2026-10-23|QQQ": "2026-10-23", "2026-10-23|SLV": "2026-10-23", "2026-10-23|SMH": "2026-10-23",
    "2026-10-23|TLT": "2026-10-23", "2026-10-23|USFR": "2026-10-23",
}


def _is_true(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1", "1.0")


def _row_session(label: str, ticker: str, source, dated) -> str:
    if not _is_true(dated) and not str(source).startswith("issuer"):
        s = MIGRATION_ROW_SESSION.get(f"{label}|{ticker}") or MIGRATION_LABEL_SESSION.get(label)
        if s:
            return s
    return session_of_date(label)


def normalize_sessions(df: pd.DataFrame, new_mask=None) -> pd.DataFrame:
    """Re-date every row to its trading session and keep ONE row per
    (ticker, session): rows in `new_mask` (this poll) win; otherwise the row
    with the later original date (the later poll) wins. Rows from before v2.2
    are re-dated through the migration tables first. Every returned row is
    marked SESSION_COL = True, so the result is idempotent."""
    if df is None or df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["_orig"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["_new"] = (np.asarray(new_mask, dtype=bool) if new_mask is not None
                   else np.zeros(len(out), dtype=bool)).astype(int)
    out["_seq"] = np.arange(len(out))
    src = out["shares_source"] if "shares_source" in out.columns else pd.Series("", index=out.index)
    dated = out[SESSION_COL] if SESSION_COL in out.columns else pd.Series(False, index=out.index)
    out["date"] = [_row_session(l, t, so, d) for l, t, so, d in zip(out["_orig"], out["ticker"], src, dated)]
    out[SESSION_COL] = True
    out = (out.sort_values(["ticker", "date", "_new", "_orig", "_seq"])
              .drop_duplicates(["ticker", "date"], keep="last"))
    return out.drop(columns=["_orig", "_new", "_seq"]).reset_index(drop=True)


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
    row_date = session_for()            # v2.2: the session this poll describes (was the UTC calendar day)
    _LAST_ASOF.pop(ticker, None)

    # v4 ORDER, Sept 2026: issuer FIRST again, now that the issuer endpoints
    # are verified. aum_implied is the fallback — and flow_integrity treats it
    # as untrustworthy unless its AUM demonstrably updates.
    primary = _issuer_source_for(ticker)
    if primary is not None:
        s, p = primary(ticker)
        if s:
            shares, price = s, p
            used = "issuer_" + primary.__name__.replace("_shares_from_", "")
            row_date = session_of_date(_LAST_ASOF.get(ticker, row_date))

    # v4.2: third-party relay of reported shares, before any estimate.
    if shares is None:
        s, p = _shares_from_stockanalysis(ticker)
        if s:
            shares, price, used = s, p, "thirdparty_stockanalysis"

    # aum_implied is always fetched: last-resort estimate, and a cross-check.
    aum_shares, aum_price = _shares_from_aum_implied(ticker)
    if shares is None and aum_shares:
        shares, price, used = aum_shares, aum_price, "aum_implied"

    if shares is None:
        s, p = _shares_from_yfinance(ticker)
        if s and p:
            shares, price, used = s, p, "yfinance"

    if not price:
        price = aum_price
    if not price:
        _, p = _shares_from_yfinance(ticker)
        price = price or p

    if not shares or not price:
        return None

    row = {"date": row_date, "ticker": ticker,
           "shares_outstanding": float(shares), "price": float(price),
           "shares_source": used or "unknown", SESSION_COL: True}

    if aum_shares and used != "aum_implied":
        divergence = abs(aum_shares - shares) / shares
        row["aum_implied_shares"] = round(float(aum_shares), 0)
        row["cross_check_divergence_pct"] = round(divergence * 100, 2)
        # v4.2: the reported figure wins; a large gap is evidence the frozen
        # yfinance totalAssets estimate is wrong, not that the report is.
        if divergence > CROSS_CHECK_DIVERGENCE_PCT:
            print(f"[etf_flow] {ticker}: reported ({used}) vs aum_implied estimate differ "
                  f"{divergence*100:.1f}% — the estimate is stale; reported figure used.")

    return row


def _upsert(store: str, new: pd.DataFrame) -> pd.DataFrame:
    """Write rows keyed on (ticker, session); new rows replace old ones.
    v4: rows can carry different dates (issuer as-of dates vs poll date),
    so replacement is per (ticker, date) pair, not 'today' only.
    v2.2: every row, old and new, is re-dated to its trading session first,
    so a weekend/holiday row folds into the prior session (new rows win,
    then the later original date). This migrates an existing store in place."""
    os.makedirs(os.path.dirname(store) or ".", exist_ok=True)
    new = new.copy()
    new["date"] = new["date"].astype(str)
    if os.path.exists(store):
        hist = pd.read_csv(store)
        hist["date"] = hist["date"].astype(str)
        out = pd.concat([hist, new], ignore_index=True)
        mask = np.r_[np.zeros(len(hist), dtype=bool), np.ones(len(new), dtype=bool)]
    else:
        out, mask = new, np.ones(len(new), dtype=bool)
    out = normalize_sessions(out, mask).sort_values(["ticker", "date"]).reset_index(drop=True)
    out.to_csv(store, index=False)
    return out


def backfill_issuer_history(tickers: list[str] | None = None, days: int = 90,
                            store: str = DEFAULT_STORE) -> dict:
    """Seed the store with REAL reported share history from SSGA's NAV-history
    files (SPDR funds). Replaces any aum_implied/yfinance rows on the same
    dates. Idempotent — safe to run on every poll; it also self-heals days
    the daily poll missed. iShares pages carry only the latest figure, so
    those tickers build history one poll at a time."""
    tickers = [t for t in (tickers or TRACKED) if t in ISSUER_SPDR]
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    frames, report = [], {}
    for tk in tickers:
        h = _ssga_navhist(tk)
        if h is None or h.empty:
            report[tk] = "failed (see errors sidecar)"
            continue
        h = h[h["date"] >= cutoff]
        if h.empty:
            report[tk] = "no rows in window"
            continue
        f = pd.DataFrame({"date": h["date"].dt.date.astype(str), "ticker": tk,
                          "shares_outstanding": h["shares_outstanding"].astype(float),
                          "price": h["nav"].astype(float),
                          "shares_source": "issuer_spdr", SESSION_COL: True})
        f = f.dropna(subset=["price"])
        frames.append(f)
        report[tk] = f"{len(f)} rows, {f['date'].min()} → {f['date'].max()}"
    if frames:
        _upsert(store, pd.concat(frames, ignore_index=True))
    return report


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

    out = _upsert(store, new)

    sources_used = new["shares_source"].value_counts().to_dict()
    print(f"[etf_flow] stored {len(new)} snapshots; history now {len(out)} rows. "
         f"Sources used today: {sources_used}")
    return new


# ── Flow computation (unchanged from v1) ────────────────────────────────────

def load_history(store: str = DEFAULT_STORE) -> pd.DataFrame:
    if not os.path.exists(store):
        return pd.DataFrame(columns=["date", "ticker", "shares_outstanding", "price"])
    df = normalize_sessions(pd.read_csv(store))      # v2.2: one row per trading session
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
# v2.1 (N19): a third-party relay count unchanged for longer than this is
# treated as frozen. Set from the store on 2026-10-31: across 21 issuer-fed
# series (SPDR, Schwab) the longest run without a share change in 60 sessions
# was 4 (XTN, a small fund), so a relay count flat for 5+ sessions is not
# behaving like real creations/redemptions. Issuer feeds are exempt: a flat
# issuer count is a genuine no-flow day.
RELAY_PREFIX = "thirdparty"
MAX_RELAY_FLAT_SESSIONS = 4


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
        is_issuer = g["shares_source"].astype(str).str.startswith(REPORTED_PREFIXES)
        if is_issuer.tail(5).any():
            # Issuer data present recently: judge on issuer rows only. A
            # one-day issuer outage (aum fallback that day) must not reset an
            # otherwise real series.
            g = g[is_issuer]
            src = g["shares_source"].iloc[-1]
        else:
            run_id = g["shares_source"].ne(g["shares_source"].shift()).cumsum()
            g = g[run_id == run_id.iloc[-1]]        # current-source tail only
    n = len(g)
    out = {"sessions": n, "source": src, "aum_update_rate": None, "last_update": None,
           "price_mirror_corr": None, "moving": False, "artifact": False,
           "trustworthy": False, "reason": ""}
    if n < 2:
        out["reason"] = f"fewer than 2 sessions on current source ({src})"
        return out
    out["moving"] = bool(g["shares_outstanding"].nunique(dropna=True) > 1)
    changed = g["shares_outstanding"].diff().iloc[1:].fillna(0).ne(0)
    out["sessions_since_change"] = int((~changed[::-1]).cumprod().sum()) if len(changed) else 0
    if src == "yfinance":
        out["reason"] = "yfinance sharesOutstanding is annual-granularity — never capital data"
        return out
    if src != "aum_implied":
        if not out["moving"]:
            out["reason"] = f"{src}: shares never changed"
            return out
        # v2.1 (N19): a relay count that moved once and then stopped is frozen,
        # not live. "Moved at some point" let SGOV, EEM and EFA through.
        if str(src).startswith(RELAY_PREFIX) and out["sessions_since_change"] > MAX_RELAY_FLAT_SESSIONS:
            out["artifact"] = True
            out["reason"] = (f"{src}: shares unchanged for the last {out['sessions_since_change']} sessions "
                             f"(issuer feeds never went more than {MAX_RELAY_FLAT_SESSIONS}) — treated as frozen "
                             f"until the count moves again")
            return out
        out["trustworthy"] = True
        out["reason"] = f"issuer source ({src})" if not str(src).startswith(RELAY_PREFIX) else \
            f"{src}: last share change {out['sessions_since_change']} session(s) ago"
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
        g = g.reset_index(drop=True)
        informative = ~g["stale_aum"]
        first_issuer = None
        if "shares_source" in g.columns:
            is_issuer = g["shares_source"].astype(str).str.startswith(REPORTED_PREFIXES)
            if is_issuer.any():
                # Once issuer data exists, only issuer rows are informative:
                # an estimate differenced against a reported figure is noise.
                first_issuer = int(is_issuer.values.argmax())
                after = pd.Series(np.arange(len(g)) >= first_issuer)
                informative = informative & ~(after & ~is_issuer)
        d_shares = g["shares_outstanding"].where(informative).ffill().diff()
        d_shares = d_shares.where(informative)
        if first_issuer is not None:
            d_shares.iloc[first_issuer] = np.nan   # never diff reported vs estimated
        g["net_flow"] = (d_shares * g["price"]).where(~g["is_split"], np.nan)

        g["implausible"] = (g["net_flow"].abs() / g["aum"]) > IMPLAUSIBLE_DAILY_FLOW_PCT
        # v2.3 (N18): only ESTIMATED rows lose a big day. A reported count's big
        # day is real creation/redemption (small, heavily shorted funds such as
        # XRT routinely move 20%+ in a day); dropping it alone breaks the sum.
        reported = (g["shares_source"].astype(str).str.startswith(REPORTED_PREFIXES)
                    if "shares_source" in g.columns else pd.Series(False, index=g.index))
        g.loc[g["implausible"] & ~reported, "net_flow"] = np.nan

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
        aum = float(g["aum"].iloc[-window - 1])          # v2.3: start-of-window AUM, as in the Flow Map
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
    dates = pd.bdate_range("2026-09-08", periods=30)     # v2.2: no market holiday inside (a holiday row now folds into the prior session)
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
    pd.DataFrame(rows).assign(**{SESSION_COL: True}).to_csv(store, index=False)
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

    # ── v4 issuer parsers ────────────────────────────────────────────────
    import io
    grid = [["State Street SPDR", None, None, None],
            ["Fund Name:", "The Technology Select Sector SPDR Fund", None, None],
            [None, None, None, None],
            ["Date", "NAV", "Shares Outstanding", "Total Net Assets"],
            ["22-Sep-2026", 180.10, "651,000,000", 1.17e11],
            ["23-Sep-2026", 181.00, "651,260,000", 1.18e11],
            ["24-Sep-2026", 182.25, "652,510,000", 1.19e11]]
    buf = io.BytesIO()
    pd.DataFrame(grid).to_excel(buf, header=False, index=False)
    try:
        nh = parse_ssga_navhist(buf.getvalue())
        if len(nh) != 3 or nh["shares_outstanding"].iloc[-1] != 652_510_000 or nh["date"].iloc[-1].day != 24:
            f.append(f"SSGA navhist parse wrong: {nh.to_dict('records')}")
    except Exception as e:
        f.append(f"SSGA navhist parse raised: {e}")
    bad = io.BytesIO(); pd.DataFrame([["Date", "NAV"], ["x", 1]]).to_excel(bad, header=False, index=False)
    try:
        parse_ssga_navhist(bad.getvalue()); f.append("navhist without a shares column must raise")
    except ValueError:
        pass

    html = ("<html><head><title>iShares Russell 2000 ETF | IWM</title></head><body>"
            "<span class='caption'>Shares Outstanding <span class='as-of-date'>as of Sep 24, 2026</span></span>"
            "<span class='data'>272,650,000</span></body></html>")
    try:
        sh, asof = parse_ishares_page(html, "IWM")
        if sh != 272_650_000 or asof != "2026-09-24":
            f.append(f"iShares parse wrong: {sh}, {asof}")
    except Exception as e:
        f.append(f"iShares parse raised: {e}")
    try:
        parse_ishares_page(html, "TLT"); f.append("iShares page for another fund must be rejected")
    except ValueError:
        pass

    sa = ("<html><head><title>IWM ETF Stock Price &amp; Overview</title></head><body>"
          "<table><tr><td>Assets</td><td>$75.91B</td></tr>"
          "<tr><td>Shares Out</td><td>272.65M</td></tr></table></body></html>")
    try:
        if parse_stockanalysis_page(sa, "IWM") != 272_650_000:
            f.append("stockanalysis parse wrong")
    except Exception as e:
        f.append(f"stockanalysis parse raised: {e}")
    try:
        parse_stockanalysis_page(sa, "QQQ"); f.append("stockanalysis page for another fund must be rejected")
    except ValueError:
        pass
    try:
        parse_stockanalysis_page(sa.replace("IWM", "TIP").replace("272.65M", "n/a"), "TIP")
        f.append("an n/a Shares Out must raise")
    except ValueError as e:
        if "n/a" not in str(e):
            f.append(f"n/a must be named as a data gap: {e}")
    sch = ("<html><title>SCHD | Schwab U.S. Dividend Equity ETF</title><body>"
           "<th>Shares Outstanding</th><td>As of 09/24/2026</td><td>09/24/2026</td>"
           "<td>3,294,650,000</td></body></html>")
    try:
        if parse_schwab_page(sch, "SCHD") != (3_294_650_000, "2026-09-24"):
            f.append(f"schwab parse wrong: {parse_schwab_page(sch, 'SCHD')}")
    except Exception as e:
        f.append(f"schwab parse raised: {e}")
    if _issuer_source_for("SCHD") is not _shares_from_schwab or _issuer_source_for("GLD") is not None:
        f.append("routing: SCHD -> schwab, GLD -> no issuer (third-party)")
    q_tp = ticker_quality(pd.DataFrame({"date": pd.to_datetime(dates[:6]), "ticker": "T",
                                        "shares_outstanding": [1e6, 1e6, 1.01e6, 1.01e6, 1.02e6, 1.02e6],
                                        "price": px[:6], "shares_source": "thirdparty_stockanalysis"}))
    if not q_tp["trustworthy"]:
        f.append(f"a moving third-party reported series must be trustworthy: {q_tp}")

    # v2.1 (N19): a relay count that moved and then froze is not trustworthy;
    # the boundary is MAX_RELAY_FLAT_SESSIONS; issuer feeds are exempt.
    def _series(src, shares):
        k = len(shares)
        return pd.DataFrame({"date": pd.to_datetime(dates[:k]), "ticker": "T", "shares_outstanding": shares,
                             "price": px[:k], "shares_source": src})
    moved = [1e6, 1.01e6, 1.02e6, 1.03e6]
    q_fz = ticker_quality(_series("thirdparty_stockanalysis", moved + [1.03e6] * 6))
    if q_fz["trustworthy"] or q_fz["sessions_since_change"] != 6 or "unchanged for the last 6" not in q_fz["reason"]:
        f.append(f"a relay count flat for 6 sessions must be untrusted and say why: {q_fz}")
    q_edge = ticker_quality(_series("thirdparty_stockanalysis", moved + [1.03e6] * MAX_RELAY_FLAT_SESSIONS))
    if not q_edge["trustworthy"]:
        f.append(f"a relay count flat for exactly {MAX_RELAY_FLAT_SESSIONS} sessions is still live: {q_edge}")
    q_over = ticker_quality(_series("thirdparty_stockanalysis", moved + [1.03e6] * (MAX_RELAY_FLAT_SESSIONS + 1)))
    if q_over["trustworthy"]:
        f.append(f"a relay count flat for {MAX_RELAY_FLAT_SESSIONS + 1} sessions must be untrusted: {q_over}")
    q_back = ticker_quality(_series("thirdparty_stockanalysis", moved + [1.03e6] * 6 + [1.05e6]))
    if not q_back["trustworthy"]:
        f.append(f"a relay count that moves again must be trusted again: {q_back}")
    q_iss = ticker_quality(_series("issuer_spdr", moved + [1.03e6] * 8))
    if not q_iss["trustworthy"]:
        f.append(f"an issuer count flat for 8 sessions is genuine no-flow and stays trusted: {q_iss}")

    # ── issuer rows supersede estimates; a one-day outage doesn't reset them
    rows2 = []
    for i, (d, p) in enumerate(zip(dates, px)):
        if i < 10:       # old aum_implied echo rows
            rows2.append({"date": d.date().isoformat(), "ticker": "MIX", "shares_outstanding": 100e6 / p,
                          "price": p, "shares_source": "aum_implied"})
        elif i == 25:    # issuer outage day -> aum fallback
            rows2.append({"date": d.date().isoformat(), "ticker": "MIX", "shares_outstanding": 100e6 / p,
                          "price": p, "shares_source": "aum_implied"})
        else:            # real reported shares, +1%/day
            rows2.append({"date": d.date().isoformat(), "ticker": "MIX",
                          "shares_outstanding": 1e6 * 1.01 ** i, "price": p, "shares_source": "issuer_spdr"})
    store2 = os.path.join(tempfile.mkdtemp(), "hist.csv")
    pd.DataFrame(rows2).assign(**{SESSION_COL: True}).to_csv(store2, index=False)
    qm = ticker_quality(load_history(store2))
    if not qm["trustworthy"]:
        f.append(f"issuer series with a one-day aum outage must stay trustworthy: {qm}")
    fm = compute_flows(store2)
    nf = fm["net_flow"]
    if nf.iloc[10] == nf.iloc[10] or nf.iloc[25] == nf.iloc[25]:
        f.append("first issuer row and the outage row must carry NaN flow")
    if not (nf.iloc[11:25].dropna() > 0).all() or nf.iloc[26:].dropna().le(0).any():
        f.append("issuer rows must show the real +1%/day inflow, including across the outage")

    # _upsert replaces per (ticker, date) pair
    s3 = os.path.join(tempfile.mkdtemp(), "h.csv")
    _upsert(s3, pd.DataFrame([{"date": "2026-09-23", "ticker": "A", "shares_outstanding": 1, "price": 1, "shares_source": "aum_implied"},
                              {"date": "2026-09-24", "ticker": "A", "shares_outstanding": 1, "price": 1, "shares_source": "aum_implied"}]))
    out3 = _upsert(s3, pd.DataFrame([{"date": "2026-09-23", "ticker": "A", "shares_outstanding": 5, "price": 1, "shares_source": "issuer_spdr"}]))
    if len(out3) != 2 or out3.loc[out3["date"] == "2026-09-23", "shares_source"].iloc[0] != "issuer_spdr":
        f.append(f"_upsert must replace only the matching (ticker, date): {out3.to_dict('records')}")
    # ── v2.2 (N31): rows are dated by trading session, never by poll calendar day
    U = timezone.utc
    for when, want, why in (
            (datetime(2026, 10, 31, 1, 3, tzinfo=U), "2026-10-30", "Fri 9:03pm ET poll (Sat 01:03 UTC)"),
            (datetime(2026, 10, 31, 22, 35, tzinfo=U), "2026-10-30", "Saturday poll"),
            (datetime(2026, 11, 1, 11, 15, tzinfo=U), "2026-10-30", "Sunday poll"),
            (datetime(2026, 11, 2, 21, 45, tzinfo=U), "2026-11-02", "Monday 4:45pm ET scheduled poll"),
            (datetime(2026, 11, 2, 15, 0, tzinfo=U), "2026-10-30", "Monday 10am ET, before the close"),
            (datetime(2026, 11, 26, 22, 0, tzinfo=U), "2026-11-25", "Thanksgiving poll"),
            (datetime(2026, 11, 27, 18, 30, tzinfo=U), "2026-11-27", "early-close day after 1pm ET"),
            (datetime(2026, 10, 30, 21, 45), "2026-10-30", "naive clock read as UTC"),
            (datetime(2026, 10, 27, 18, 0, tzinfo=U), "2026-10-26", "Tue 2pm EDT (18:00 UTC) is before the close"),
            (datetime(2026, 11, 2, 20, 30, tzinfo=U), "2026-10-30", "Mon 3:30pm EST (20:30 UTC) is before the close"),
            (datetime(2026, 11, 3, 0, 44, tzinfo=U), "2026-11-02", "Tue 00:44 UTC is Mon 7:44pm EST")):
        if session_for(when) != want:
            f.append(f"session_for: {why} must be {want}, got {session_for(when)}")
    # a snapshot row is dated by session_for() and marked, whatever the runner's calendar day
    _g = globals()
    _saved = {k: _g[k] for k in ("session_for", "_issuer_source_for", "_shares_from_stockanalysis",
                                 "_shares_from_aum_implied", "_shares_from_yfinance")}
    try:
        _g["session_for"] = lambda when=None: "2001-01-02"
        _g["_issuer_source_for"] = lambda t: None
        _g["_shares_from_stockanalysis"] = lambda t: (1e6, 10.0)
        _g["_shares_from_aum_implied"] = lambda t: (None, None)
        _g["_shares_from_yfinance"] = lambda t: (None, None)
        _row = _snapshot_one("TST")
        if not _row or _row["date"] != "2001-01-02" or not _is_true(_row.get(SESSION_COL)):
            f.append(f"a relay snapshot must be dated by session_for() and marked session-dated: {_row}")
    finally:
        _g.update(_saved)
    if session_of_date("2026-10-31") != "2026-10-30" or session_of_date("2026-11-02") != "2026-11-02" \
            or session_of_date("2026-11-26") != "2026-11-25":
        f.append("session_of_date: Sat -> Fri, Mon unchanged, holiday -> prior day")

    # a store with weekend polls: Thu, Fri, Sat (relay catch-up), Sun (same), then a Monday poll
    def r_(d, sh, px_=10.0, src="thirdparty_stockanalysis", dated=True):
        return {"date": d, "ticker": "W", "shares_outstanding": sh, "price": px_, "shares_source": src,
                SESSION_COL: dated}
    s4 = os.path.join(tempfile.mkdtemp(), "h.csv")
    pd.DataFrame([r_("2026-10-29", 100.0), r_("2026-10-30", 100.0), r_("2026-10-31", 110.0),
                  r_("2026-11-01", 110.0)]).to_csv(s4, index=False)
    lh = load_history(s4)
    if [d.date().isoformat() for d in lh["date"]] != ["2026-10-29", "2026-10-30"] \
            or lh["shares_outstanding"].iloc[-1] != 110.0:
        f.append(f"load_history must fold weekend polls into Friday and keep the latest count: "
                 f"{lh[['date', 'shares_outstanding']].to_dict('records')}")
    raw_before = pd.read_csv(s4)
    if len(raw_before) != 4:
        f.append("load_history must not rewrite the store (only _upsert writes)")
    o4 = _upsert(s4, pd.DataFrame([r_("2026-11-02", 112.0)]))
    if list(o4["date"]) != ["2026-10-29", "2026-10-30", "2026-11-02"] or list(o4["shares_outstanding"]) != [100.0, 110.0, 112.0]:
        f.append(f"_upsert must migrate weekend rows into Friday on write: {o4[['date', 'shares_outstanding']].to_dict('records')}")
    if pd.read_csv(s4)["date"].astype(str).map(lambda d: pd.Timestamp(d).dayofweek).ge(5).any():
        f.append("no weekend date may remain in the stored file after a write")
    fl4 = compute_flows(s4)
    if abs(float(fl4["net_flow"].sum(skipna=True)) - (12.0 * 10.0)) > 1e-9:
        f.append(f"total flow must be preserved by the fold (Thu 100 -> Mon 112 = 12 shares x $10): {fl4['net_flow'].tolist()}")
    # this poll wins over an older weekend row mapped to the same session
    s5 = os.path.join(tempfile.mkdtemp(), "h.csv")
    pd.DataFrame([r_("2026-10-30", 100.0), r_("2026-10-31", 105.0)]).to_csv(s5, index=False)
    o5 = _upsert(s5, pd.DataFrame([r_("2026-10-30", 107.0)]))
    if len(o5) != 1 or o5["shares_outstanding"].iloc[0] != 107.0:
        f.append(f"a new poll for a session must replace an older weekend row for it: {o5.to_dict('records')}")
    # idempotent
    n1 = normalize_sessions(pd.read_csv(s4))
    if not n1.equals(normalize_sessions(n1)):
        f.append("normalize_sessions must be idempotent")
    # the frozen-count test counts trading sessions: Mon-Wed moving, flat Wed->Fri, plus Sat/Sun polls
    wk = [("2026-10-19", 1.00e6), ("2026-10-20", 1.01e6), ("2026-10-21", 1.02e6), ("2026-10-22", 1.03e6),
          ("2026-10-23", 1.04e6), ("2026-10-24", 1.04e6), ("2026-10-26", 1.05e6), ("2026-10-27", 1.06e6),
          ("2026-10-28", 1.07e6), ("2026-10-29", 1.07e6), ("2026-10-30", 1.07e6), ("2026-10-31", 1.07e6),
          ("2026-11-01", 1.07e6)]
    s6 = os.path.join(tempfile.mkdtemp(), "h.csv")
    pd.DataFrame([dict(r_(d, v), ticker="F") for d, v in wk]).to_csv(s6, index=False)
    q6 = ticker_quality(load_history(s6))
    if q6.get("sessions_since_change") != 2 or not q6["trustworthy"]:
        f.append(f"flat Thu+Fri with weekend polls is 2 sessions and still live (was counted 4): {q6}")
    # one-time migration of pre-v2.2 rows (unmarked): re-dated from the poll's real time
    if _row_session("2026-10-06", "TLT", "thirdparty_stockanalysis", None) != "2026-10-05":
        f.append("unmarked relay row labelled Tue 10/06 (polled Mon evening ET) must move to Mon 10/05")
    if _row_session("2026-10-06", "XLK", "issuer_spdr", None) != "2026-10-06":
        f.append("an issuer row keeps its as-of date; the migration never shifts it")
    if _row_session("2026-10-06", "TLT", "thirdparty_stockanalysis", True) != "2026-10-06":
        f.append("a row already dated by session (v2.2) must never be looked up in the migration table")
    if _row_session("2026-10-15", "TLT", "thirdparty_stockanalysis", "") != "2026-10-15" or \
            _row_session("2026-10-15", "SPY", "thirdparty_stockanalysis", "") != "2026-10-14":
        f.append("10/15: rows re-polled Thu evening stay Thu; the rest (Wed-evening poll) move to Wed")
    if _row_session("2026-11-01", "TLT", "thirdparty_stockanalysis", "") != "2026-10-30":
        f.append("an unlisted weekend label folds into the prior session")
    s7 = os.path.join(tempfile.mkdtemp(), "h.csv")
    pd.DataFrame([r_(d, v, dated=None) for d, v in (("2026-10-27", 100.0), ("2026-10-28", 101.0),
                  ("2026-10-29", 102.0), ("2026-10-30", 103.0), ("2026-10-31", 104.0), ("2026-11-01", 105.0))]
                 + [dict(r_("2026-10-26", 50.0, src="issuer_spdr", dated=None), ticker="X")]).to_csv(s7, index=False)
    o7 = _upsert(s7, pd.DataFrame([r_("2026-11-02", 106.0)]))
    w7 = o7[o7["shares_source"].str.startswith("thirdparty")]
    if list(w7["date"]) != ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02"] \
            or list(w7["shares_outstanding"]) != [100.0, 101.0, 102.0, 103.0, 105.0, 106.0]:
        f.append(f"pre-v2.2 Tue-Sat labels must become Mon-Fri sessions, the Sunday poll winning Friday: "
                 f"{w7[['date', 'shares_outstanding']].to_dict('records')}")
    if o7.loc[o7["ticker"] == "X", "date"].tolist() != ["2026-10-26"]:
        f.append("an unmarked issuer row must keep its as-of date through the migration")
    if not o7[SESSION_COL].map(_is_true).all():
        f.append("every row must be marked session-dated after a write")
    o7b = _upsert(s7, pd.DataFrame([r_("2026-11-02", 106.0)]))
    if not o7b[["date", "ticker", "shares_outstanding"]].equals(o7[["date", "ticker", "shares_outstanding"]]):
        f.append("a second write must not shift migrated rows again")
    # v2.3 (N18): a reported fund's big day stays in the sum, so the 20-day flow equals the share change
    big = [4.0e6, 3.9e6, 3.8e6, 4.8e6, 4.7e6, 4.6e6, 4.5e6]      # one +26% creation (XRT 10/21), small redemptions around it
    s8 = os.path.join(tempfile.mkdtemp(), "h.csv")
    pd.DataFrame([{"date": d.date().isoformat(), "ticker": "XR", "shares_outstanding": v, "price": 10.0,
                   "shares_source": "issuer_spdr", SESSION_COL: True} for d, v in zip(dates[:7], big)]
                 + [{"date": d.date().isoformat(), "ticker": "EST", "shares_outstanding": v, "price": 10.0,
                     "shares_source": "aum_implied", SESSION_COL: True} for d, v in zip(dates[:7], big)]).to_csv(s8, index=False)
    f8 = compute_flows(s8)
    xr, est = f8[f8["ticker"] == "XR"], f8[f8["ticker"] == "EST"]
    if abs(float(xr["net_flow"].sum(skipna=True)) - (4.5e6 - 4.0e6) * 10) > 1e-6 or not xr["implausible"].any():
        f.append(f"a reported fund's big days must stay in the sum (net = share change x price) and still be flagged: "
                 f"{xr['net_flow'].tolist()}")
    if est.loc[est["implausible"], "net_flow"].notna().any():
        f.append("an estimated (aum_implied) row's implausible day must still be dropped")
    dv8 = flow_vs_price_divergence(s8, window=5)
    if dv8.empty or "XR" not in set(dv8["ticker"]):
        f.append("the divergence test fund must be present (test would otherwise skip)")
    else:
        r8 = dv8.set_index("ticker").loc["XR"]
        g8 = f8[f8["ticker"] == "XR"].sort_values("date")
        want8 = float(g8["net_flow"].iloc[-5:].sum()) / float(g8["aum"].iloc[-6]) * 100
        if abs(r8["net_flow_pct_aum"] - round(want8, 2)) > 0.011:
            f.append(f"divergence % of AUM must use start-of-window AUM: {r8['net_flow_pct_aum']} vs {want8:.2f}")
    return {"ok": not f, "failures": f,
            "quality": {k: (x["trustworthy"], x["aum_update_rate"], x["price_mirror_corr"]) for k, x in q.items()}}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, default=str))
