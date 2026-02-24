# ai_trading_mvp/app/validator.py
#
# Validates that the data shown in our scanner matches FinViz's live quote pages.
# Any discrepancy — however small — is flagged. Run before market open and
# multiple times in the last hour before close.

import re
import time
import datetime
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

_FV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _scrape_finviz_quote(symbol: str) -> dict | None:
    """
    Scrapes the FinViz quote page for a single stock.
    Returns a dict with the raw values FinViz displays.
    Returns None on fetch failure.
    """
    url = f"https://finviz.com/quote.ashx?t={symbol}&p=d"
    try:
        resp = requests.get(url, headers=_FV_HEADERS, timeout=12)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # FinViz quote fundamentals are in a table with alternating label/value <td> cells.
        # Walk every table and build a flat label→value dict.
        data = {}
        for table in soup.find_all("table"):
            cells = table.find_all("td")
            for i in range(len(cells) - 1):
                label = cells[i].get_text(strip=True)
                value = cells[i + 1].get_text(strip=True)
                if label and value:
                    data[label] = value

        def _parse_float(s):
            if not s or s in ("-", "N/A", ""):
                return None
            s = s.replace(",", "").replace("%", "").strip()
            try:
                return float(s)
            except Exception:
                return None

        def _parse_shorthand(s):
            """Convert FinViz shorthand like '2.45M', '150K', '1.2B' to an integer."""
            if not s or s in ("-", "N/A", ""):
                return None
            s = s.strip()
            multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
            for suffix, mult in multipliers.items():
                if s.upper().endswith(suffix):
                    try:
                        return int(float(s[:-1]) * mult)
                    except Exception:
                        return None
            try:
                return int(float(s.replace(",", "")))
            except Exception:
                return None

        # "Change" on quote page is like "+12.34%" or "-5.67%"
        change_raw = data.get("Change", "")
        change_pct = _parse_float(change_raw)

        return {
            "price":              _parse_float(data.get("Price")),
            "change_pct":         change_pct,
            "rel_volume":         _parse_float(data.get("Rel Volume")),
            "shares_outstanding": _parse_shorthand(data.get("Shs Outstand")),
            "float_shares":       _parse_shorthand(data.get("Shs Float")),
            "raw": {k: data.get(k) for k in
                    ("Price", "Change", "Rel Volume", "Shs Outstand", "Shs Float")},
        }
    except Exception as e:
        print(f"VALIDATOR: error scraping {symbol}: {e}")
        return None


def _compare(metric, finviz_val, our_val, real_time=False):
    """
    Build a comparison entry. Any non-zero difference → match=False.
    For real_time metrics (price) we log but never mark as a failure.
    """
    if finviz_val is None and our_val is None:
        return None  # nothing to compare

    if finviz_val is None or our_val is None:
        return {
            "metric":   metric,
            "finviz":   finviz_val,
            "ours":     our_val,
            "diff":     None,
            "diff_pct": None,
            "match":    real_time,  # treat missing as OK only for real-time fields
            "real_time": real_time,
        }

    diff = finviz_val - our_val
    diff_pct = (abs(diff) / abs(finviz_val) * 100) if finviz_val != 0 else 0.0
    match = real_time or (abs(diff) == 0)   # zero tolerance on non-real-time fields

    return {
        "metric":    metric,
        "finviz":    finviz_val,
        "ours":      our_val,
        "diff":      round(diff, 4),
        "diff_pct":  round(diff_pct, 2),
        "match":     match,
        "real_time": real_time,
    }


def validate_scan_results(scan_results: list, max_symbols: int = 15) -> dict:
    """
    Cross-checks our top scan results against live FinViz quote pages.
    ANY difference (other than real-time price) is flagged as a mismatch.

    Returns a report dict with per-symbol details and an overall summary.
    """
    # Focus on the highest-scoring stocks — those are most likely to be traded.
    top = sorted(scan_results, key=lambda x: x.get("score", 0), reverse=True)[:max_symbols]

    rows = []

    def _check_one(stock):
        symbol = stock["symbol"]
        print(f"VALIDATOR: checking {symbol}...")
        fv = _scrape_finviz_quote(symbol)
        time.sleep(0.3)   # be polite to FinViz

        if fv is None:
            return {
                "symbol":  symbol,
                "score":   stock.get("score"),
                "status":  "ERROR",
                "error":   "Could not fetch FinViz quote page",
                "checks":  [],
                "fv_raw":  {},
            }

        checklist  = stock.get("checklist", {})
        our_rv     = checklist.get("relative_volume")
        our_price  = stock.get("price")
        our_daily  = stock.get("daily_return_pct")
        our_shares = checklist.get("shares_outstanding")
        our_float  = checklist.get("float_shares")

        checks = []

        # Relative Volume — most critical; we now pull directly from FinViz
        # screener so these MUST match.
        c = _compare("Rel Volume", fv["rel_volume"], our_rv)
        if c:
            checks.append(c)

        # Price — informational (real-time, will always differ)
        c = _compare("Price", fv["price"], our_price, real_time=True)
        if c:
            checks.append(c)

        # Daily Change % — should match screener filter value closely
        c = _compare("Daily Change %", fv["change_pct"], our_daily)
        if c:
            checks.append(c)

        # Shares Outstanding — static, should match yfinance exactly modulo
        # FinViz's shorthand rounding (e.g. "2.45M" = 2,450,000).
        c = _compare("Shares Outstanding", fv["shares_outstanding"], our_shares)
        if c:
            checks.append(c)

        # Float Shares — secondary; also static
        c = _compare("Float Shares", fv["float_shares"], our_float)
        if c:
            checks.append(c)

        mismatches = [ch for ch in checks if not ch["match"] and not ch.get("real_time")]
        status = "WARN" if mismatches else "OK"

        return {
            "symbol":  symbol,
            "score":   stock.get("score"),
            "status":  status,
            "checks":  checks,
            "fv_raw":  fv.get("raw", {}),
        }

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_check_one, s): s for s in top}
        for future in as_completed(futures):
            result = future.result()
            if result:
                rows.append(result)

    rows.sort(key=lambda x: x.get("score", 0), reverse=True)

    warn_count  = sum(1 for r in rows if r["status"] == "WARN")
    error_count = sum(1 for r in rows if r["status"] == "ERROR")

    report = {
        "run_at":          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S ET"),
        "symbols_checked": len(rows),
        "ok_count":        len(rows) - warn_count - error_count,
        "warn_count":      warn_count,
        "error_count":     error_count,
        "rows":            rows,
    }

    # Print a terse summary to server logs
    print(
        f"VALIDATOR: {len(rows)} checked — "
        f"{report['ok_count']} OK, {warn_count} WARN, {error_count} ERROR"
    )
    for r in rows:
        if r["status"] != "OK":
            bad = [c for c in r["checks"] if not c.get("match") and not c.get("real_time")]
            for b in bad:
                print(
                    f"  MISMATCH {r['symbol']} [{b['metric']}] "
                    f"FinViz={b['finviz']}  Ours={b['ours']}  diff={b.get('diff')}"
                )

    return report
