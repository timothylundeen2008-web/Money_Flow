"""
issuer_probe.py  (v1 — November 2026)
──────────────────────────────────────────────────────────────────────────────
fix_status N30, step 1 of 2: MEASURE before wiring.

Some funds' share counts come only from a third-party relay that updates
irregularly (SGOV, FXI, INDA, PBJ, AMLP, RING, OIH, HACK) or not at all
(GLD, USO, DBA, EFA, EEM, VGT), so they drop out of the Tier A flow mid-week
or never count. The durable fix is each issuer's own published figure. But
iShares has refused GitHub's runner IPs since 2026-09-26 and spdrgoldshares
404s, and this repo's development workspace cannot reach any issuer site, so
no parser here is written blind.

This probe runs in GitHub Actions (manual workflow "Issuer Probe"). For each
problem fund it requests the candidate official pages and records:
  * HTTP outcome (status, bytes, which client got through, final URL)
  * whether the page is for that fund (title check)
  * every "shares outstanding" figure found, with any as-of date beside it
  * the ratio to the relay's latest stored count (the sanity check)
  * a short excerpt of the real response (logs/issuer_probe/samples/)
It never writes to the share store. Step 2 builds parsers only for sources
that answered, named the right fund, and agreed with the relay — and tests
them against these saved real excerpts.

Candidate URLs are best-known public product pages, NOT verified; a wrong
URL or product ID simply shows up as "not this fund" or an HTTP error.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

OUT_DIR = "logs/issuer_probe"
SAMPLE_DIR = os.path.join(OUT_DIR, "samples")
AGREE_TOL = 0.05            # a found count within ±5% of the relay's latest = agrees
EXCERPT_CHARS = 700         # characters kept either side of each "shares outstanding" hit

_ISH = "https://www.ishares.com/us/products/{id}/"
CANDIDATES = {
    # iShares — blocked from runner IPs on 2026-09-26; re-checked because runner IPs rotate.
    "SGOV": [("ishares", _ISH.format(id=314116))],
    "EFA":  [("ishares", _ISH.format(id=239623))],
    "EEM":  [("ishares", _ISH.format(id=239637))],
    "FXI":  [("ishares", _ISH.format(id=239536))],
    "INDA": [("ishares", _ISH.format(id=239659))],
    "RING": [("ishares", _ISH.format(id=239658))],
    "EWJ":  [("ishares", _ISH.format(id=239665))],
    "TIP":  [("ishares", _ISH.format(id=239467))],
    # Others
    "GLD":  [("ssga", "https://www.ssga.com/us/en/intermediary/etfs/spdr-gold-shares-gld"),
             ("spdrgold", "https://www.spdrgoldshares.com/usa/")],
    "VGT":  [("vanguard", "https://investor.vanguard.com/investment-products/etfs/profile/vgt"),
             ("vanguard_adv", "https://advisors.vanguard.com/investments/products/vgt/vanguard-information-technology-etf")],
    "PBJ":  [("invesco", "https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=Investor&ticker=PBJ")],
    "PDBC": [("invesco", "https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=Investor&ticker=PDBC")],
    "DBA":  [("invesco", "https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=Investor&ticker=DBA")],
    "USO":  [("uscf", "https://www.uscfinvestments.com/uso")],
    "AMLP": [("alps", "https://www.alpsfunds.com/products/etf/amlp")],
    "KMLM": [("kraneshares", "https://kraneshares.com/kmlm/")],
    "USFR": [("wisdomtree", "https://www.wisdomtree.com/investments/etfs/fixed-income/usfr")],
    "OIH":  [("vaneck", "https://www.vaneck.com/us/en/investments/oil-services-etf-oih/")],
    "HACK": [("amplify", "https://amplifyetfs.com/hack/")],
}

_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(K|M|B|thousand|million|billion)?\b"
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}
_DATES = (r"as of\s*(\d{1,2}/\d{1,2}/\d{2,4})", r"as of\s*([A-Z][a-z]{2,8}\.?\s+\d{1,2},?\s+\d{4})",
          r"(\d{1,2}/\d{1,2}/\d{4})", r"([A-Z][a-z]{2,8}\.?\s+\d{1,2},\s+\d{4})")


def _text(html: str) -> str:
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&nbsp;|&#160;", " ", t)
    return re.sub(r"\s+", " ", t)


def title_of(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    return re.sub(r"\s+", " ", m.group(1)).strip()[:200] if m else ""


def is_for(html: str, ticker: str) -> bool:
    head = title_of(html) or html[:4000]
    return bool(re.search(rf"\b{re.escape(ticker)}\b", head, re.I))


def find_share_counts(html: str) -> list:
    """Every plausible share count printed after 'shares outstanding' (in
    text, or under a JSON key containing both words), with a nearby as-of
    date when one is printed. Counts under 10,000 are ignored (prices,
    percentages, years)."""
    out = []
    text = _text(html)
    for m in re.finditer(r"shares\s+outstanding", text, re.I):
        win = text[m.end(): m.end() + 220]
        v = None
        for n in re.finditer(_NUM, win):          # first number big enough: skips an as-of day/year in between
            x = float(n.group(1).replace(",", "")) * _MULT.get((n.group(2) or "").lower(), 1.0)
            if x >= 1e4:
                v = x
                break
        if v is None:
            continue
        around = text[max(0, m.start() - 200): m.end() + 260]
        asof = None
        for pat in _DATES:
            d = re.search(pat, around)
            if d:
                asof = d.group(1)
                break
        out.append({"value": v, "asof": asof, "context": text[max(0, m.start() - 60): m.end() + 120]})
    for k, v in re.findall(r'"([A-Za-z_]*[Ss]hares[A-Za-z_]*[Oo]utstanding[A-Za-z_]*)"\s*:\s*"?([\d,.]+)', html):
        try:
            x = float(v.replace(",", ""))
        except ValueError:
            continue
        if x >= 1e4:
            out.append({"value": x, "asof": None, "context": f"JSON key {k}"})
    return out


def excerpt(html: str) -> str:
    text = _text(html)
    parts = [f"TITLE: {title_of(html)}"]
    for m in list(re.finditer(r"shares\s+outstanding", text, re.I))[:3]:
        parts.append(text[max(0, m.start() - EXCERPT_CHARS): m.end() + EXCERPT_CHARS])
    if len(parts) == 1:
        parts.append(text[:1500])
    return "\n---\n".join(parts)


def fetch(url: str, timeout: int = 30) -> dict:
    """curl_cffi (Chrome TLS impersonation) first, then plain requests."""
    from etf_flow_tracker import _BROWSER_HEADERS
    res = {"url": url, "attempts": []}
    t0 = time.time()
    try:
        from curl_cffi import requests as creq
        r = creq.get(url, impersonate="chrome", timeout=timeout, headers=_BROWSER_HEADERS, allow_redirects=True)
        res["attempts"].append(f"curl_cffi HTTP {r.status_code} {len(r.text or '')}B")
        if r.status_code == 200 and r.text:
            res.update(ok=True, client="curl_cffi", status=200, html=r.text, final_url=str(r.url))
    except Exception as e:
        res["attempts"].append(f"curl_cffi {type(e).__name__}: {str(e)[:120]}")
    if not res.get("ok"):
        try:
            import requests
            r = requests.get(url, timeout=timeout, headers=_BROWSER_HEADERS, allow_redirects=True)
            res["attempts"].append(f"requests HTTP {r.status_code} {len(r.text or '')}B")
            if r.status_code == 200 and r.text:
                res.update(ok=True, client="requests", status=200, html=r.text, final_url=str(r.url))
        except Exception as e:
            res["attempts"].append(f"requests {type(e).__name__}: {str(e)[:120]}")
    res["seconds"] = round(time.time() - t0, 1)
    return res


def relay_latest() -> dict:
    """Latest stored share count per ticker (what the probe compares against)."""
    try:
        from etf_flow_tracker import load_history
        h = load_history()
        last = h.sort_values("date").groupby("ticker").tail(1)
        return {r.ticker: {"shares": float(r.shares_outstanding), "date": str(r.date.date()),
                           "source": str(r.shares_source)} for r in last.itertuples()}
    except Exception as e:
        print(f"[probe] could not read the share store: {e}")
        return {}


def judge(ticker: str, page: dict, relay: dict) -> dict:
    """One candidate's verdict: usable only if it answered, named the fund,
    and printed a count within AGREE_TOL of the relay's latest."""
    out = {k: page.get(k) for k in ("url", "client", "status", "final_url", "seconds", "attempts")}
    if not page.get("ok"):
        out["verdict"] = "BLOCKED / ERROR"
        return out
    html = page["html"]
    out["title"] = title_of(html)
    out["bytes"] = len(html)
    out["for_ticker"] = is_for(html, ticker)
    counts = find_share_counts(html)
    out["counts"] = counts[:5]
    ref = (relay.get(ticker) or {}).get("shares")
    best = None
    if counts and ref:
        best = min(counts, key=lambda c: abs(c["value"] / ref - 1))
        out["ratio_to_relay"] = round(best["value"] / ref, 4)
    if not out["for_ticker"]:
        out["verdict"] = "WRONG PAGE"
    elif not counts:
        out["verdict"] = "NO COUNT ON PAGE (likely rendered by JavaScript)"
    elif best is None:
        out["verdict"] = "COUNT FOUND, NO RELAY TO COMPARE"
    elif abs(best["value"] / ref - 1) <= AGREE_TOL:
        out["verdict"] = "USABLE" + ("" if best.get("asof") else " (no as-of date printed)")
    else:
        out["verdict"] = f"COUNT DISAGREES WITH RELAY ({out['ratio_to_relay']:.2f}x)"
    return out


def run(candidates: dict = None, fetcher=fetch) -> dict:
    candidates = candidates or CANDIDATES
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    relay = relay_latest()
    report = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "agree_tol": AGREE_TOL,
              "funds": {}}
    for tk, cands in candidates.items():
        rows = []
        for issuer, url in cands:
            page = fetcher(url)
            j = judge(tk, page, relay)
            j["issuer"] = issuer
            if page.get("ok"):
                path = os.path.join(SAMPLE_DIR, f"{tk}_{issuer}.txt")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(excerpt(page["html"]))
                j["sample"] = path
            rows.append(j)
            print(f"[probe] {tk:5} {issuer:13} {j['verdict']:52} {'; '.join(j.get('attempts') or [])[:90]}")
        report["funds"][tk] = {"relay": relay.get(tk), "candidates": rows}
    day = report["run_at"][:10]
    with open(os.path.join(OUT_DIR, f"{day}.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    usable = sorted(tk for tk, v in report["funds"].items() if any(c["verdict"].startswith("USABLE") for c in v["candidates"]))
    print(f"[probe] usable official sources: {len(usable)} of {len(candidates)} funds: {', '.join(usable) or 'none'}")
    return report


def selftest() -> dict:
    import tempfile
    f = []
    page = ("<html><head><title>iShares 0-3 Month Treasury Bond ETF | SGOV</title></head><body>"
            "<div>Shares Outstanding <span>as of Oct 30, 2026</span> 1,163,400,000</div>"
            "<script>var x = 'Shares Outstanding 5';</script></body></html>")
    c = find_share_counts(page)
    if not c or c[0]["value"] != 1_163_400_000 or c[0]["asof"] != "Oct 30, 2026":
        f.append(f"find_share_counts must read the count and its as-of date: {c}")
    if find_share_counts("<p>Shares Outstanding as of 10/30/2026 4,550,226</p>")[0]["value"] != 4_550_226:
        f.append("a numeric as-of date between label and count must be skipped")
    if find_share_counts("<p>Shares Outstanding 12.5</p>"):
        f.append("a number under 10,000 is not a share count")
    if find_share_counts("<p>Shares Outstanding 352.40M</p>")[0]["value"] != 352_400_000:
        f.append("an M suffix must scale to millions")
    if find_share_counts('{"sharesOutstanding": "4,550,226"}')[0]["value"] != 4_550_226:
        f.append("a JSON sharesOutstanding key must be read")
    if not is_for(page, "SGOV") or is_for(page, "TLT"):
        f.append("is_for must check the page title for the ticker")
    relay = {"SGOV": {"shares": 1.16e9}}
    ok = judge("SGOV", {"ok": True, "html": page}, relay)
    if not ok["verdict"].startswith("USABLE") or abs(ok["ratio_to_relay"] - 1.0029) > 1e-3:
        f.append(f"a matching count on the right page must be USABLE: {ok}")
    if not judge("TLT", {"ok": True, "html": page}, {"TLT": {"shares": 1.16e9}})["verdict"] == "WRONG PAGE":
        f.append("a page for another fund must be WRONG PAGE")
    if not judge("SGOV", {"ok": True, "html": page}, {"SGOV": {"shares": 2.0e9}})["verdict"].startswith("COUNT DISAGREES"):
        f.append("a count far from the relay must be flagged as disagreeing")
    if judge("SGOV", {"ok": False, "attempts": ["curl_cffi HTTP 403"]}, relay)["verdict"] != "BLOCKED / ERROR":
        f.append("a refused request must be BLOCKED / ERROR")
    shell = "<html><head><title>Vanguard VGT</title></head><body><div id='root'></div></body></html>"
    if not judge("VGT", {"ok": True, "html": shell}, {"VGT": {"shares": 1e8}})["verdict"].startswith("NO COUNT"):
        f.append("a JavaScript shell with no count must say so")
    # run() end to end with a fake fetcher, in a temp dir; never touches the network or the share store
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        global relay_latest
        saved = relay_latest
        relay_latest = lambda: relay
        try:
            rep = run({"SGOV": [("ishares", "x")], "TLT": [("ishares", "y")]},
                      fetcher=lambda u: {"ok": True, "html": page} if u == "x" else {"ok": False, "attempts": ["HTTP 403"]})
        finally:
            relay_latest = saved
        if rep["funds"]["SGOV"]["candidates"][0]["verdict"][:6] != "USABLE" or \
                rep["funds"]["TLT"]["candidates"][0]["verdict"] != "BLOCKED / ERROR":
            f.append(f"run() verdicts: {rep['funds']}")
        if not os.path.exists(os.path.join(SAMPLE_DIR, "SGOV_ishares.txt")) or \
                not [p for p in os.listdir(OUT_DIR) if p.endswith(".json")]:
            f.append("run() must save the excerpt and the dated report")
        if os.path.exists("data/etf_shares_history.csv"):
            f.append("the probe must never write the share store")
    finally:
        os.chdir(cwd)
    if set(CANDIDATES) - {"SGOV", "EFA", "EEM", "FXI", "INDA", "RING", "EWJ", "TIP", "GLD", "VGT", "PBJ", "PDBC",
                          "DBA", "USO", "AMLP", "KMLM", "USFR", "OIH", "HACK"}:
        f.append("unexpected fund in CANDIDATES")
    return {"ok": not f, "failures": f}


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = selftest()
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)
    run()
