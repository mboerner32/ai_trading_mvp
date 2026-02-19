import sqlite3
from datetime import datetime, timedelta
import yfinance as yf
from passlib.context import CryptContext

DB_NAME = "scan_history.db"

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


# ---------------- INIT DB ----------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # ---------------- SCANS TABLE ----------------
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

    # ---------------- CANDIDATES TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE,
            last_updated TEXT
        )
    """)

    # ---------------- USERS TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            hashed_password TEXT
        )
    """)

    conn.commit()
    conn.close()


# ---------------- USER AUTH ----------------
def create_user(username: str, password: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    hashed = pwd_context.hash(password)

    cursor.execute("""
        INSERT OR IGNORE INTO users (username, hashed_password)
        VALUES (?, ?)
    """, (username, hashed))

    conn.commit()
    conn.close()


def seed_users():
    """Ensure default users exist — safe to call on every startup."""
    create_user("BobbyAxelrod", "Billions")


def authenticate_user(username: str, password: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT hashed_password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False

    return pwd_context.verify(password, row[0])


# ---------------- HOLDING PERFORMANCE ----------------
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

    return {
        "avg_day1": round(sum(day1)/len(day1), 2) if day1 else 0,
        "avg_day3": round(sum(day3)/len(day3), 2) if day3 else 0
    }


# ---------------- EQUITY CURVE ----------------
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


# ---------------- SCORE BUCKETS ----------------
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
                "avg_return": round(sum(values)/len(values), 2),
                "win_rate": round((len(wins)/len(values))*100, 2)
            }
        else:
            results[bucket] = {
                "trades": 0,
                "avg_return": 0,
                "win_rate": 0
            }

    return results
# ---------------- RETURN UPDATES ----------------
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

    for scan_id, symbol, timestamp in rows:
        try:
            scan_date = datetime.fromisoformat(timestamp).date()

            # Only process scans at least 3 days old
            if (today - scan_date).days < 3:
                continue

            # Never request future data
            end_date = min(scan_date + timedelta(days=5), today)

            data = yf.download(
                symbol,
                start=scan_date,
                end=end_date,
                progress=False
            )

            if data.empty or len(data) < 2:
                continue

            # Flatten MultiIndex columns from newer yfinance versions
            if hasattr(data.columns, "levels"):
                data.columns = data.columns.get_level_values(0)

            closes = data["Close"].tolist()

            next_day = None
            three_day = None

            if len(closes) >= 2:
                next_day = ((closes[1] - closes[0]) / closes[0]) * 100

            if len(closes) >= 4:
                three_day = ((closes[3] - closes[0]) / closes[0]) * 100

            cursor.execute("""
                UPDATE scans
                SET next_day_return = ?, three_day_return = ?
                WHERE id = ?
            """, (
                round(next_day, 2) if next_day is not None else None,
                round(three_day, 2) if three_day is not None else None,
                scan_id
            ))

        except Exception:
            continue  # Skip failures silently

    conn.commit()
    conn.close()


# ---------------- SAVE SCAN ----------------
def save_scan(results: list, mode: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    timestamp = datetime.utcnow().isoformat()

    for r in results:
        checklist = r.get("checklist", {})
        cursor.execute("""
            INSERT INTO scans (
                timestamp, symbol, score, recommendation, mode,
                relative_volume, today_return, shares_outstanding,
                news_recent, next_day_return, three_day_return
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            r.get("symbol"),
            r.get("score"),
            r.get("recommendation"),
            mode,
            checklist.get("relative_volume"),
            r.get("daily_return_pct"),
            checklist.get("shares_outstanding"),
            0,
            None,
            None
        ))

    conn.commit()
    conn.close()
