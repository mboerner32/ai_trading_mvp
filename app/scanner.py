# ai_trading_mvp/app/scanner.py

import requests
import re
import time
import datetime
import pytz
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.scoring_engine import score_stock, score_stock_squeeze, DEFAULT_SQUEEZE_WEIGHTS
from app.database import get_squeeze_weights


def _tod_factor() -> float:
    """
    Returns the factor that projects partial-day volume to end-of-day,
    matching FinViz's time-of-day-adjusted Rel Volume calculation.

    Example: if 30% of the trading session has elapsed, the current
    volume is ~30% of the final volume, so factor = 1 / 0.30 ≈ 3.33.
    Returns 1.0 before market open or after close (volume is final).
    """
    et = pytz.timezone("America/New_York")
    now = datetime.datetime.now(et)

    # NYSE regular session: 9:30 AM – 4:00 PM ET, Mon–Fri
    if now.weekday() >= 5:
        return 1.0

    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)

    if now <= market_open or now >= market_close:
        return 1.0   # before open or after close — full-day volume already in

    total_secs   = (market_close - market_open).total_seconds()   # 23,400 s
    elapsed_secs = (now - market_open).total_seconds()
    fraction     = elapsed_secs / total_secs

    return 1.0 / fraction if fraction > 0 else 1.0


# ---------------------------------------------------
# FINVIZ FILTERS
# v=161 = Performance view — includes Rel Volume column
# geo:usa, current volume > 5M, price < $5, relvol > 10, today > +10%
# ---------------------------------------------------
BASE_URL = (
    "https://finviz.com/screener.ashx?"
    "v=161&f=geo_usa,sh_curvol_o5000,"
    "sh_price_u5,sh_relvol_o10,ta_perf_d10o"
)
BASE_URL_PREMARKET = (
    "https://finviz.com/screener.ashx?"
    "v=161&f=geo_usa,sh_curvol_o500,"
    "sh_price_u5,sh_relvol_o5,ta_perf_d5o"
)

_FV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------
# GET FINVIZ TICKERS + RELATIVE VOLUMES
# Uses v=161 (Performance view) to pull FinViz's own Rel Volume value.
# Returns (tickers_list, relvol_dict  {symbol: float})
# ---------------------------------------------------
def get_finviz_tickers(premarket: bool = False):
    """
    Fetch tickers + FinViz Rel Volume from the Performance screener view.
    Falls back to ticker-only if HTML parsing fails.
    """
    base = BASE_URL_PREMARKET if premarket else BASE_URL

    tickers  = []
    relvol   = {}
    page     = 1

    print(f"Pulling FinViz candidates{'  (pre-market)' if premarket else ''}...")

    while True:
        url      = f"{base}&r={(page - 1) * 20 + 1}"
        response = requests.get(url, headers=_FV_HEADERS, timeout=15)

        if response.status_code != 200:
            break

        html = response.text
        page_tickers, page_relvol = _parse_finviz_page(html)

        if not page_tickers:
            break

        tickers.extend(page_tickers)
        relvol.update(page_relvol)

        if len(page_tickers) < 20:
            break

        page += 1

    print(f"Found {len(tickers)} FinViz stocks "
          f"({len(relvol)} with Rel Volume)")
    return tickers, relvol


def _parse_finviz_page(html: str):
    """
    Parse one page of FinViz screener HTML.
    Returns (tickers, relvol_dict) — relvol_dict may be empty if Rel Volume
    column is not present in the current view.

    Strategy:
      1. PRIMARY — TS block (always present, always reliable for tickers).
      2. SECONDARY — find the results table that actually contains stock data
         rows (identified by having a numeric "No." cell in the first column)
         and extract Rel Volume from it if the column exists.
    """
    tickers = []
    relvol  = {}

    # --- Step 1: tickers from TS block -----------------------------------
    match = re.search(r"<!-- TS(.*?)TE -->", html, re.DOTALL)
    if match:
        for line in match.group(1).strip().split("\n"):
            parts = line.strip().split("|")
            if parts and parts[0].strip():
                tickers.append(parts[0].strip())

    if not tickers:
        return tickers, relvol

    # --- Step 2: Rel Volume from the results table -----------------------
    # The real results table has rows whose first <td> is a row number ("1", "2", …)
    # and whose <th> row includes "Ticker". Several UI tables on the page also
    # have "Ticker" in their <th> cells (filter dropdowns etc.) — we skip those
    # by requiring that at least one data row starts with a numeric cell.
    soup = BeautifulSoup(html, "html.parser")

    for t in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if "Ticker" not in headers:
            continue

        # Verify this is a data table (first non-header row starts with a digit)
        data_rows = t.find_all("tr")[1:]
        is_results_table = False
        for row in data_rows:
            cells = row.find_all("td")
            if cells and cells[0].get_text(strip=True).isdigit():
                is_results_table = True
                break
        if not is_results_table:
            continue

        ticker_idx = next((i for i, h in enumerate(headers) if h == "Ticker"), None)
        rv_idx     = next((i for i, h in enumerate(headers)
                           if h in ("Rel Volume", "Rel Vol")), None)

        if ticker_idx is None or rv_idx is None:
            break   # found the results table but no Rel Volume column

        for row in data_rows:
            cells = row.find_all("td")
            if len(cells) <= max(ticker_idx, rv_idx):
                continue
            ticker = cells[ticker_idx].get_text(strip=True)
            if not ticker or ticker not in tickers:
                continue
            try:
                relvol[ticker] = float(cells[rv_idx].get_text(strip=True))
            except (ValueError, AttributeError):
                pass
        break   # stop after the first valid results table

    return tickers, relvol


# ---------------------------------------------------
# LIVE QUOTES FROM FINVIZ QUOTE PAGES
# Pulls all available live metrics in a single request per ticker.
# ---------------------------------------------------
def get_finviz_quotes(tickers: list, max_workers: int = 5) -> dict:
    """
    Scrapes individual FinViz quote pages to get live metrics for each ticker.
    Returns {symbol: dict} where each dict contains:
        price, change_pct, rel_volume, volume,
        shares_outstanding, float_shares, institution_pct
    Only keys with non-None values are included.
    """

    def _parse_float(s):
        if not s or s in ("-", "N/A", ""):
            return None
        try:
            return float(s.replace(",", "").replace("%", "").strip())
        except Exception:
            return None

    def _parse_shorthand(s):
        """Convert FinViz shorthand like '2.45M', '150K', '1.2B' to int."""
        if not s or s in ("-", "N/A", ""):
            return None
        s = s.strip()
        for suffix, mult in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
            if s.upper().endswith(suffix):
                try:
                    return int(float(s[:-1]) * mult)
                except Exception:
                    return None
        try:
            return int(float(s.replace(",", "")))
        except Exception:
            return None

    def _fetch_one(symbol):
        url = f"https://finviz.com/quote.ashx?t={symbol}&p=d"
        try:
            resp = requests.get(url, headers=_FV_HEADERS, timeout=12)
            if resp.status_code != 200:
                return symbol, None
            soup = BeautifulSoup(resp.text, "html.parser")

            # Build label→value dict from alternating td cells across all tables
            data = {}
            for table in soup.find_all("table"):
                cells = table.find_all("td")
                for i in range(len(cells) - 1):
                    label = cells[i].get_text(strip=True)
                    value = cells[i + 1].get_text(strip=True)
                    if label and value:
                        data[label] = value

            result = {}

            price = _parse_float(data.get("Price"))
            if price is not None:
                result["price"] = price

            # "Change" is like "+12.34%" — store as percentage (12.34)
            change_pct = _parse_float(data.get("Change"))
            if change_pct is not None:
                result["change_pct"] = change_pct

            rel_vol = _parse_float(data.get("Rel Volume"))
            if rel_vol is not None:
                result["rel_volume"] = rel_vol

            volume = _parse_shorthand(data.get("Volume"))
            if volume is not None:
                result["volume"] = volume

            shares = _parse_shorthand(data.get("Shs Outstand"))
            if shares is not None:
                result["shares_outstanding"] = shares

            float_sh = _parse_shorthand(data.get("Shs Float"))
            if float_sh is not None:
                result["float_shares"] = float_sh

            # "Inst Own" shown as "63.10%" — store as ratio (0.631)
            inst_own = _parse_float(data.get("Inst Own"))
            if inst_own is not None:
                result["institution_pct"] = inst_own / 100

            return symbol, result if result else None
        except Exception as e:
            print(f"FinViz quote fetch error for {symbol}: {e}")
            return symbol, None

    print(f"Fetching live quotes from FinViz for {len(tickers)} symbol(s)...")
    quotes = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sym, q in executor.map(_fetch_one, tickers):
            if q:
                quotes[sym] = q

    print(f"Got FinViz quotes for {len(quotes)}/{len(tickers)} symbols")
    return quotes


# ---------------------------------------------------
# FUNDAMENTALS (Numeric Values)
# ---------------------------------------------------
def get_fundamentals(symbol):

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Check for recent news (last 3 days) — news-driven moves are less clean
        recent_news_present = False
        news_headlines = []
        try:
            news = ticker.news
            if news:
                cutoff_ts = time.time() - (3 * 24 * 3600)
                recent_news_present = any(
                    a.get("providerPublishTime", 0) > cutoff_ts
                    for a in news[:5]
                )
                # Extract top 3 headlines with human-readable timestamps
                for item in news[:5]:
                    # Handle both old flat format and newer nested content format
                    title = (item.get("title")
                             or item.get("content", {}).get("title"))
                    ts = item.get("providerPublishTime") or 0
                    if not ts:
                        pub_date = item.get("content", {}).get("pubDate", "")
                        if pub_date:
                            try:
                                ts = int(datetime.datetime.fromisoformat(
                                    pub_date.replace("Z", "+00:00")
                                ).timestamp())
                            except Exception:
                                pass
                    if not title:
                        continue
                    # Format relative time like FinViz: "Today, 8:58 AM"
                    time_label = ""
                    if ts:
                        try:
                            dt = datetime.datetime.fromtimestamp(ts)
                            now = datetime.datetime.now()
                            delta_days = (now.date() - dt.date()).days
                            hour = dt.strftime("%I").lstrip("0") or "12"
                            mins = dt.strftime("%M")
                            ampm = dt.strftime("%p")
                            if delta_days == 0:
                                time_label = f"Today, {hour}:{mins} {ampm}"
                            elif delta_days == 1:
                                time_label = "Yesterday"
                            elif delta_days < 7:
                                time_label = f"{delta_days} days ago"
                            else:
                                time_label = dt.strftime("%b %-d")
                        except Exception:
                            pass
                    news_headlines.append({"title": title, "when": time_label})
                    if len(news_headlines) >= 3:
                        break
        except Exception:
            pass

        # Try both field name variants across yfinance versions
        institution_pct = (
            info.get("institutionsPercentHeld") or
            info.get("heldPercentInstitutions")
        )

        # Fallback: parse major_holders DataFrame
        if institution_pct is None:
            try:
                mh = ticker.major_holders
                if mh is not None and not mh.empty:
                    for _, row in mh.iterrows():
                        breakdown = str(row.get("Breakdown", "")).lower()
                        if "held by institutions" in breakdown and "float" not in breakdown:
                            val = str(row.get("Value", "")).replace("%", "").strip()
                            institution_pct = float(val) / 100
                            break
            except Exception:
                pass

        return {
            "shares_outstanding": info.get("sharesOutstanding"),
            "float_shares": info.get("floatShares"),
            "total_cash": info.get("totalCash"),
            "institution_pct": institution_pct,
            "recent_news_present": recent_news_present,
            "news_headlines": news_headlines,
        }

    except Exception:
        return None


# ---------------------------------------------------
# DATA ENGINEERING
# ---------------------------------------------------
def prepare_dataframe(df):

    # Newer yfinance versions return MultiIndex columns.
    # The level order varies: (Price, Ticker) or (Ticker, Price) depending on version.
    # Detect which level contains standard price field names and use that.
    if isinstance(df.columns, pd.MultiIndex):
        _price_fields = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
        lvl0 = set(df.columns.get_level_values(0))
        if lvl0 & _price_fields:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)

    # Drop any duplicate column names (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    df = df.dropna()

    # RETURNS
    df["daily_return"] = df["close"].pct_change()
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)

    # 10-day compression range
    df["range_10d"] = (
        (df["high"].rolling(10).max()
         - df["low"].rolling(10).min())
        / df["close"]
    )

    # 63-day relative volume — time-of-day adjusted to match FinViz.
    # FinViz projects today's partial volume to end-of-day before dividing
    # by the 63-day average.  We apply the same factor to the latest bar only.
    rolling_mean = df["volume"].rolling(63).mean()
    df["relative_volume"] = df["volume"] / rolling_mean
    factor = _tod_factor()
    if factor != 1.0:
        df.loc[df.index[-1], "relative_volume"] = (
            df["volume"].iloc[-1] * factor / rolling_mean.iloc[-1]
        )

    return df


# ---------------------------------------------------
# RUN SCAN (SHOW ALL FINVIZ STOCKS)
# ---------------------------------------------------
def run_scan(mode="standard", premarket: bool = False):

    tickers, _ = get_finviz_tickers(premarket=premarket)

    # Fetch all live metrics from individual FinViz quote pages
    live_quotes = get_finviz_quotes(tickers)

    results = []
    summary = {"total_scanned": 0, "qualified": 0}

    if not tickers:
        return {"results": results, "summary": summary}

    if mode == "squeeze":
        weights_data = get_squeeze_weights()
        active_weights = weights_data["weights"] if weights_data else DEFAULT_SQUEEZE_WEIGHTS
        scorer = lambda symbol, df, fundamentals=None: score_stock_squeeze(
            symbol, df, fundamentals, weights=active_weights
        )
    else:
        scorer = score_stock

    # --- Step 1: batch-download all symbols in a single call ----------------
    # Calling yf.download() concurrently from multiple threads corrupts its
    # internal state in yfinance 1.x — all threads end up with the same data.
    # One batch call is safe and also faster (single HTTP round-trip).
    print(f"Downloading price data for {len(tickers)} symbol(s)...")
    try:
        raw = yf.download(
            tickers if len(tickers) > 1 else tickers[0],
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
        )
    except Exception as e:
        print(f"Batch download failed: {e}")
        return {"results": results, "summary": summary}

    # Extract a clean per-symbol DataFrame from the batch result
    dfs = {}
    if len(tickers) == 1:
        if not raw.empty:
            dfs[tickers[0]] = raw.copy()
    else:
        for sym in tickers:
            try:
                sym_df = raw[sym].copy()
                if not sym_df.empty:
                    dfs[sym] = sym_df
            except Exception:
                pass

    # --- Step 2: fetch fundamentals in parallel (yfinance Ticker.info, safe) -
    def _fetch_fund(symbol):
        return symbol, get_fundamentals(symbol)

    fundamentals_map = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for sym, fund in executor.map(_fetch_fund, tickers):
            fundamentals_map[sym] = fund

    # --- Step 3: score each symbol sequentially (pure CPU, no I/O) ----------
    for symbol in tickers:
        summary["total_scanned"] += 1
        if symbol not in dfs:
            continue
        try:
            df = prepare_dataframe(dfs[symbol])
            fundamentals = fundamentals_map.get(symbol) or {}
            # Inject live FinViz values — these override yfinance-calculated metrics
            if symbol in live_quotes:
                q = live_quotes[symbol]
                if q.get("rel_volume") is not None:
                    fundamentals["finviz_relvol"] = q["rel_volume"]
                if q.get("price") is not None:
                    fundamentals["finviz_price"] = q["price"]
                if q.get("change_pct") is not None:
                    fundamentals["finviz_change_pct"] = q["change_pct"]
                if q.get("volume") is not None:
                    fundamentals["finviz_volume"] = q["volume"]
                if q.get("shares_outstanding") is not None:
                    fundamentals["shares_outstanding"] = q["shares_outstanding"]
                if q.get("float_shares") is not None:
                    fundamentals["float_shares"] = q["float_shares"]
                if q.get("institution_pct") is not None:
                    fundamentals["institution_pct"] = q["institution_pct"]
            result = scorer(symbol, df, fundamentals=fundamentals)
            if result:
                results.append(result)
                summary["qualified"] += 1
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {"results": results, "summary": summary}
