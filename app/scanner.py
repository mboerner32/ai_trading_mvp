# ai_trading_mvp/app/scanner.py

import requests
import re
import time
import yfinance as yf
import pandas as pd
from app.scoring_engine import score_stock, score_stock_squeeze, DEFAULT_SQUEEZE_WEIGHTS
from app.database import get_squeeze_weights


# ---------------------------------------------------
# FINVIZ FILTER (Exact Replica)
# geo:usa
# current volume > 5M
# price < $5
# relative volume > 10
# performance today > 10%
# ---------------------------------------------------
BASE_URL = (
    "https://finviz.com/screener.ashx?"
    "v=111&f=geo_usa,sh_curvol_o5000,"
    "sh_price_u5,sh_relvol_o10,ta_perf_d10o"
)


# ---------------------------------------------------
# GET FINVIZ TICKERS (FAST TS BLOCK PARSER)
# ---------------------------------------------------
def get_finviz_tickers():

    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = []
    page = 1

    print("Pulling FinViz candidates...")

    while True:

        url = f"{BASE_URL}&r={(page-1)*20+1}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            break

        html = response.text

        match = re.search(r"<!-- TS(.*?)TE -->", html, re.DOTALL)

        if not match:
            break

        block = match.group(1).strip()
        lines = block.split("\n")

        page_tickers = []

        for line in lines:
            parts = line.strip().split("|")
            if len(parts) >= 1:
                ticker = parts[0].strip()
                if ticker:
                    page_tickers.append(ticker)

        if not page_tickers:
            break

        tickers.extend(page_tickers)

        if len(page_tickers) < 20:
            break

        page += 1

    print(f"Found {len(tickers)} FinViz stocks")
    return tickers


# ---------------------------------------------------
# FUNDAMENTALS (Numeric Values)
# ---------------------------------------------------
def get_fundamentals(symbol):

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Check for recent news (last 3 days) — news-driven moves are less clean
        recent_news_present = False
        try:
            news = ticker.news
            if news:
                cutoff_ts = time.time() - (3 * 24 * 3600)
                recent_news_present = any(
                    a.get("providerPublishTime", 0) > cutoff_ts
                    for a in news[:5]
                )
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
        }

    except Exception:
        return None


# ---------------------------------------------------
# DATA ENGINEERING
# ---------------------------------------------------
def prepare_dataframe(df):

    # Newer yfinance versions return MultiIndex columns like ('Close', 'AAPL').
    # Flatten to single-level so downstream code can access 'close', 'volume', etc.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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
def run_scan(mode="standard"):

    tickers = get_finviz_tickers()

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

    for symbol in tickers:

        try:
            df = yf.download(
                symbol,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=False
            )

            if df.empty:
                continue

            df = prepare_dataframe(df)

            fundamentals = get_fundamentals(symbol)

            summary["total_scanned"] += 1

            result = scorer(symbol, df, fundamentals=fundamentals)

            if result:
                results.append(result)
                summary["qualified"] += 1

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            continue

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {
        "results": results,
        "summary": summary
    }
