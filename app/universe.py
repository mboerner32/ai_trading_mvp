"""
Universe fetchers for live scanning and historical backfill.

- get_finviz_universe():   strict live-scan filters (used by scanner.py)
- fetch_backfill_universe(): loose filters for training data — up to 500 US micro/small-cap
  tickers priced under $10. The backfill engine itself applies the strict criteria
  (price < $5, daily gain 10-100%, rel vol > 10x) when sliding the historical window.
"""
import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup


# ---- Live scan screener (strict) ----
_LIVE_URL = (
    "https://finviz.com/screener.ashx?"
    "v=111&f=geo_usa,sh_curvol_o5000,sh_price_u5,sh_relvol_o10,ta_perf_d10o&ft=4"
)

# ---- Extended backfill screener (loose — backfill filters the rest) ----
# price < $5, avg daily vol > 200K, US stocks, sorted by volume descending
_BACKFILL_URL = (
    "https://finviz.com/screener.ashx?"
    "v=111&f=geo_usa,sh_price_u5,sh_avgvol_o200&o=-volume"
)

# ---- Mid-cap backfill screener ($5–$15) — stocks that may have been < $5 historically ----
_BACKFILL_URL_MID = (
    "https://finviz.com/screener.ashx?"
    "v=111&f=geo_usa,sh_price_u15,sh_price_o5,sh_avgvol_o300&o=-volume"
)

_CACHE_FILE = "/tmp/backfill_universe.json"
_CACHE_TTL  = 86400  # 24 hours


# ---------------------------------------------------
# Shared page fetcher (TS-block parser — same approach as scanner.py)
# ---------------------------------------------------
def _fetch_page(base_url: str, row_start: int) -> list:
    """Fetch one page of 20 Finviz screener results via the fast TS block."""
    url = f"{base_url}&r={row_start}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        match = re.search(r"<!-- TS(.*?)TE -->", resp.text, re.DOTALL)
        if not match:
            return []
        block = match.group(1).strip()
        tickers = []
        for line in block.split("\n"):
            parts = line.strip().split("|")
            if parts and parts[0].strip():
                tickers.append(parts[0].strip())
        return tickers
    except Exception as e:
        print(f"Universe fetch error (row {row_start}): {e}")
        return []


# ---------------------------------------------------
# Live-scan universe (strict filters — used by scanner.py legacy path)
# ---------------------------------------------------
def get_finviz_universe():
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(_LIVE_URL, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    tickers = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "quote.ashx?t=" in href:
            ticker = href.split("t=")[1].split("&")[0]
            if ticker.isupper() and len(ticker) <= 5:
                tickers.append(ticker)

    tickers = list(set(tickers))
    print("FOUND TICKERS:", tickers)
    return tickers


# ---------------------------------------------------
# Extended backfill universe (loose filters + 24h cache)
# ---------------------------------------------------
def fetch_backfill_universe(max_tickers: int = 1000) -> list:
    """
    Fetch a broad list of US micro/small-cap tickers for historical backfill.
    Pulls two Finviz passes:
      1. price < $5, avg daily vol > 200K (primary targets)
      2. price $5–$15, avg daily vol > 300K (may have been < $5 historically)
    Returns up to max_tickers deduplicated tickers.
    Caches result for 24 hours to respect Finviz rate limits.
    """
    # --- Check cache ---
    try:
        if os.path.exists(_CACHE_FILE):
            if time.time() - os.path.getmtime(_CACHE_FILE) < _CACHE_TTL:
                with open(_CACHE_FILE) as f:
                    cached = json.load(f)
                print(f"Universe: loaded {len(cached)} cached backfill tickers")
                return cached[:max_tickers]
    except Exception:
        pass

    # --- Pass 1: price < $5 ---
    tickers = []
    for row_start in range(1, max_tickers + 1, 20):
        page = _fetch_page(_BACKFILL_URL, row_start)
        tickers.extend(page)
        if len(page) < 20:
            break
        time.sleep(0.4)

    # --- Pass 2: price $5–$15 (historically may have been < $5) ---
    mid_tickers = []
    for row_start in range(1, 501, 20):
        page = _fetch_page(_BACKFILL_URL_MID, row_start)
        mid_tickers.extend(page)
        if len(page) < 20:
            break
        time.sleep(0.4)
    tickers.extend(mid_tickers)
    print(f"Universe: fetched {len(tickers)} tickers before dedup "
          f"(pass2 added {len(mid_tickers)})")

    # Deduplicate, preserve order
    seen = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    # --- Persist cache ---
    try:
        with open(_CACHE_FILE, "w") as f:
            json.dump(unique, f)
    except Exception:
        pass

    print(f"Universe: fetched {len(unique)} backfill candidates from Finviz")
    return unique[:max_tickers]
