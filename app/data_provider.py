# app/data_provider.py

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

DB_NAME = "scan_history.db"


def get_stock_data(ticker: str) -> dict | None:
    """
    Fetch enriched stock data required for:
    - Hard filters
    - Scoring engine
    - Technical analysis
    """

    try:
        stock = yf.Ticker(ticker)

        # ---------- PRICE + SHORT HISTORY ----------
        df = stock.history(period="10d")

        if df.empty or len(df) < 3:
            return None

        price = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]

        today_return = ((price - prev_close) / prev_close) * 100
        current_volume = df["Volume"].iloc[-1]

        # ---------- 30 DAY VOLUME ----------
        df_30 = stock.history(period="30d")

        if df_30.empty:
            return None

        avg_volume = df_30["Volume"].mean()

        relative_volume = (
            current_volume / avg_volume if avg_volume > 0 else 0
        )

        # ---------- FUNDAMENTALS ----------
        info = stock.info

        shares_outstanding = info.get("sharesOutstanding", 0)
        cash_per_share = info.get("totalCashPerShare", 0)
        market_cap = info.get("marketCap", 0)

        # ---------- NEWS CHECK (last 48h) ----------
        news_recent = False
        try:
            news_items = stock.news
            if news_items:
                now = datetime.utcnow()
                for item in news_items[:5]:
                    pub_time = datetime.utcfromtimestamp(
                        item["providerPublishTime"]
                    )
                    if now - pub_time <= timedelta(hours=48):
                        news_recent = True
                        break
        except Exception:
            news_recent = False

        return {
            "price": float(price),
            "today_return": float(today_return),
            "relative_volume": float(relative_volume),
            "current_volume": int(current_volume),
            "avg_volume": float(avg_volume),
            "shares_outstanding": int(shares_outstanding) if shares_outstanding else 0,
            "cash_per_share": float(cash_per_share) if cash_per_share else 0,
            "market_cap": int(market_cap) if market_cap else 0,
            "news_recent": news_recent,
            "history": df,
        }

    except Exception as e:
        print(f"Data error for {ticker}: {e}")
        return None


# ---------------------------------------------------------
# SCORE BUCKET ANALYTICS (SEPARATE FUNCTION)
# ---------------------------------------------------------

def get_score_buckets():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT score, next_day_return
        FROM scans
        WHERE next_day_return IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    buckets = {
        "80-100": [],
        "60-79": [],
        "40-59": [],
        "0-39": []
    }

    for score, ret in rows:
        if score >= 80:
            buckets["80-100"].append(ret)
        elif score >= 60:
            buckets["60-79"].append(ret)
        elif score >= 40:
            buckets["40-59"].append(ret)
        else:
            buckets["0-39"].append(ret)

    results = {}

    for bucket, values in buckets.items():
        if values:
            wins = [v for v in values if v > 0]
            results[bucket] = {
                "trades": len(values),
                "avg_return": round(sum(values) / len(values), 2),
                "win_rate": round((len(wins) / len(values)) * 100, 2)
            }
        else:
            results[bucket] = {
                "trades": 0,
                "avg_return": 0,
                "win_rate": 0
            }

    return results