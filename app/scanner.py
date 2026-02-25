# ai_trading_mvp/app/scanner.py

import requests
import re
import time
import datetime
import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.scoring_engine import score_stock, score_stock_squeeze, DEFAULT_SQUEEZE_WEIGHTS
from app.database import get_squeeze_weights


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

    # 63-day relative volume (FinViz accurate)
    df["relative_volume"] = (
        df["volume"] / df["volume"].rolling(63).mean()
    )

    return df


# ---------------------------------------------------
# RUN SCAN (SHOW ALL FINVIZ STOCKS)
# ---------------------------------------------------
def run_scan(mode="standard", premarket: bool = False):

    tickers, fv_relvol = get_finviz_tickers(premarket=premarket)

    results = []
    summary = {
        "total_scanned": 0,
        "qualified": 0
    }

    if mode == "squeeze":
        weights_data = get_squeeze_weights()
        active_weights = weights_data["weights"] if weights_data else DEFAULT_SQUEEZE_WEIGHTS
        scorer = lambda symbol, df, fundamentals=None: score_stock_squeeze(
            symbol, df, fundamentals, weights=active_weights
        )
    else:
        scorer = score_stock

    def _scan_one(symbol):
        try:
            df = yf.download(
                symbol,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=False
            )
            if df.empty:
                return None
            df = prepare_dataframe(df)
            fundamentals = get_fundamentals(symbol)
            # Inject FinViz's own Rel Volume so scorers use the accurate value
            if fundamentals is None:
                fundamentals = {}
            if symbol in fv_relvol:
                fundamentals["finviz_relvol"] = fv_relvol[symbol]
            return scorer(symbol, df, fundamentals=fundamentals)
        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_scan_one, sym): sym for sym in tickers}
        for future in as_completed(futures):
            summary["total_scanned"] += 1
            result = future.result()
            if result:
                results.append(result)
                summary["qualified"] += 1

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {
        "results": results,
        "summary": summary
    }
