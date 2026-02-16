import sqlite3
from datetime import datetime, timedelta
import yfinance as yf

DB_NAME = "scan_history.db"


# -----------------------------------
# INIT DATABASE
# -----------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            score INTEGER,
            recommendation TEXT,
            mode TEXT,
            relative_volume REAL,
            today_return REAL,
            shares_outstanding INTEGER,
            news_recent INTEGER,
            next_day_return REAL,
            three_day_return REAL
        )
    """)

    conn.commit()
    conn.close()


# -----------------------------------
# LOG SCAN
# -----------------------------------
def log_scan(results, mode):
    if not results:
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    timestamp = datetime.utcnow().isoformat()

    for stock in results:
        cursor.execute("""
            INSERT INTO scans (
                timestamp,
                symbol,
                score,
                recommendation,
                mode,
                relative_volume,
                today_return,
                shares_outstanding,
                news_recent,
                next_day_return,
                three_day_return
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """, (
            timestamp,
            stock["symbol"],
            stock["score"],
            stock["recommendation"],
            mode,
            stock["breakdown"]["relative_volume"]["value"],
            stock["breakdown"]["performance"]["value"],
            stock["breakdown"]["shares_outstanding"]["value"],
            int(stock["breakdown"]["news"]["value"])
        ))

    conn.commit()
    conn.close()


# -----------------------------------
# UPDATE RETURNS
# -----------------------------------
def update_returns():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, symbol, timestamp
        FROM scans
        WHERE next_day_return IS NULL
    """)

    rows = cursor.fetchall()
    today = datetime.utcnow().date()

    for row_id, symbol, timestamp in rows:
        try:
            scan_datetime = datetime.fromisoformat(timestamp)
            scan_date = scan_datetime.date()

            if scan_date >= today:
                continue

            start_date = scan_date
            end_date = start_date + timedelta(days=5)

            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False
            )

            if data.empty or len(data) < 3:
                continue

            open_price = data["Open"].iloc[1]
            close_day1 = data["Close"].iloc[1]
            close_day3 = data["Close"].iloc[2]

            next_ret = ((close_day1 - open_price) / open_price) * 100
            three_ret = ((close_day3 - open_price) / open_price) * 100

            cursor.execute("""
                UPDATE scans
                SET next_day_return = ?, three_day_return = ?
                WHERE id = ?
            """, (float(next_ret), float(three_ret), row_id))

        except Exception:
            continue

    conn.commit()
    conn.close()


# -----------------------------------
# HOLDING PERFORMANCE
# -----------------------------------
def get_holding_performance():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT next_day_return, three_day_return
        FROM scans
        WHERE next_day_return IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    day1 = [r[0] for r in rows if r[0] is not None]
    day3 = [r[1] for r in rows if r[1] is not None]

    avg_day1 = round(sum(day1) / len(day1), 2) if day1 else 0
    avg_day3 = round(sum(day3) / len(day3), 2) if day3 else 0

    return {
        "avg_day1": avg_day1,
        "avg_day3": avg_day3
    }


# -----------------------------------
# EQUITY CURVE
# -----------------------------------
def get_equity_curve(starting_capital=10000):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT next_day_return
        FROM scans
        WHERE next_day_return IS NOT NULL
        ORDER BY id ASC
    """)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    capital = starting_capital
    equity = []

    for (ret,) in rows:
        capital *= (1 + ret / 100)
        equity.append(round(capital, 2))

    return equity