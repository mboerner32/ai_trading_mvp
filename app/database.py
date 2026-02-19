import sqlite3
import json
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

    # ---------------- TRADES TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT    NOT NULL,
            entry_price   REAL    NOT NULL,
            shares        REAL    NOT NULL,
            position_size REAL    NOT NULL,
            status        TEXT    NOT NULL DEFAULT 'open',
            exit_price    REAL,
            realized_pnl  REAL,
            opened_at     TEXT    NOT NULL,
            closed_at     TEXT
        )
    """)

    # ---------------- PORTFOLIO TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id   INTEGER PRIMARY KEY,
            cash REAL    NOT NULL DEFAULT 10000.0
        )
    """)

    cursor.execute("""
        INSERT OR IGNORE INTO portfolio (id, cash) VALUES (1, 10000.0)
    """)

    # ---------------- FEEDBACK TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at     TEXT NOT NULL,
            symbol         TEXT,
            user_text      TEXT,
            chart_analysis TEXT
        )
    """)

    # ---------------- SETTINGS TABLE ----------------
    # Generic key/value store for persisting AI-generated outputs.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at TEXT
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
    create_user("admin", "admin123")
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
    # Fetch pending rows then close connection before slow yf.download calls
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, symbol, timestamp
        FROM scans
        WHERE next_day_return IS NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    today = datetime.utcnow().date()
    updates = []

    for scan_id, symbol, timestamp in rows:
        try:
            scan_date = datetime.fromisoformat(timestamp).date()

            if (today - scan_date).days < 3:
                continue

            end_date = min(scan_date + timedelta(days=5), today)

            data = yf.download(
                symbol,
                start=scan_date,
                end=end_date,
                progress=False
            )

            if data.empty or len(data) < 2:
                continue

            if hasattr(data.columns, "levels"):
                data.columns = data.columns.get_level_values(0)

            closes = data["Close"].tolist()

            next_day = None
            three_day = None

            if len(closes) >= 2:
                next_day = ((closes[1] - closes[0]) / closes[0]) * 100

            if len(closes) >= 4:
                three_day = ((closes[3] - closes[0]) / closes[0]) * 100

            updates.append((
                round(next_day, 2) if next_day is not None else None,
                round(three_day, 2) if three_day is not None else None,
                scan_id
            ))

        except Exception:
            continue

    if updates:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        for (nd, td, sid) in updates:
            cursor.execute("""
                UPDATE scans
                SET next_day_return = ?, three_day_return = ?
                WHERE id = ?
            """, (nd, td, sid))
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


# ---------------- PAPER TRADING ----------------
POSITION_SIZE = 1000.0


def open_trade(symbol: str, price: float, position_size: float = POSITION_SIZE):
    if price <= 0 or position_size <= 0:
        return None

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT cash FROM portfolio WHERE id = 1")
    row = cursor.fetchone()
    cash = row[0] if row else 0.0

    if cash < position_size:
        conn.close()
        return None

    shares = position_size / price
    opened_at = datetime.utcnow().isoformat()

    cursor.execute(
        "UPDATE portfolio SET cash = cash - ? WHERE id = 1",
        (position_size,)
    )
    cursor.execute("""
        INSERT INTO trades (symbol, entry_price, shares, position_size, status, opened_at)
        VALUES (?, ?, ?, ?, 'open', ?)
    """, (symbol, round(price, 4), round(shares, 6), position_size, opened_at))

    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"trade_id": trade_id, "shares": shares, "entry_price": price}


def get_trade_by_id(trade_id: int):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT symbol, entry_price FROM trades WHERE id = ? AND status = 'open'",
        (trade_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {"symbol": row[0], "entry_price": row[1]}


def close_trade(trade_id: int, exit_price: float):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT shares, entry_price, status FROM trades WHERE id = ?",
        (trade_id,)
    )
    row = cursor.fetchone()

    if not row or row[2] != 'open':
        conn.close()
        return None

    shares, entry_price, _ = row
    proceeds = shares * exit_price
    realized_pnl = proceeds - (shares * entry_price)
    closed_at = datetime.utcnow().isoformat()

    cursor.execute("""
        UPDATE trades
        SET status = 'closed', exit_price = ?, realized_pnl = ?, closed_at = ?
        WHERE id = ?
    """, (round(exit_price, 4), round(realized_pnl, 4), closed_at, trade_id))
    cursor.execute(
        "UPDATE portfolio SET cash = cash + ? WHERE id = 1",
        (round(proceeds, 4),)
    )

    conn.commit()
    conn.close()
    return {"realized_pnl": realized_pnl, "proceeds": proceeds}


def get_open_positions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, symbol, entry_price, shares, position_size, opened_at
        FROM trades WHERE status = 'open'
        ORDER BY opened_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    positions = []
    for row in rows:
        trade_id, symbol, entry_price, shares, position_size, opened_at = row
        positions.append({
            "trade_id": trade_id,
            "symbol": symbol,
            "entry_price": entry_price,
            "shares": shares,
            "position_size": position_size,
            "opened_at": opened_at,
        })
    return positions


def get_trade_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, symbol, entry_price, exit_price, shares,
               position_size, realized_pnl, opened_at, closed_at
        FROM trades WHERE status = 'closed'
        ORDER BY closed_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        (trade_id, symbol, entry_price, exit_price, shares,
         position_size, realized_pnl, opened_at, closed_at) = row
        pnl_pct = (realized_pnl / position_size * 100) if position_size else 0
        history.append({
            "trade_id": trade_id,
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": round(shares, 2),
            "position_size": position_size,
            "realized_pnl": round(realized_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "opened_at": opened_at,
            "closed_at": closed_at,
        })
    return history


def get_portfolio_summary():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT cash FROM portfolio WHERE id = 1")
    row = cursor.fetchone()
    cash = row[0] if row else 10000.0

    cursor.execute("""
        SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE status = 'closed'
    """)
    realized_total = cursor.fetchone()[0]

    conn.close()
    return {
        "cash": round(cash, 2),
        "realized_pnl": round(realized_total, 2),
    }


# ---------------- WEIGHT OPTIMIZATION DATA ----------------
def get_optimization_data():
    """
    Buckets historical scan signals by range and computes win rate + avg return
    per bucket. Used by the AI weight optimizer.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT relative_volume, today_return, shares_outstanding, next_day_return
        FROM scans
        WHERE next_day_return IS NOT NULL
        ORDER BY id DESC LIMIT 500
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    def bucket_stats(values):
        if not values:
            return {"count": 0, "avg_return": 0.0, "win_rate": 0.0}
        wins = [v for v in values if v > 0]
        return {
            "count": len(values),
            "avg_return": round(sum(values) / len(values), 2),
            "win_rate": round(len(wins) / len(values) * 100, 1),
        }

    relvol  = {">=50x": [], "25-50x": [], "10-25x": [], "<10x": []}
    gain    = {"20-40%": [], "10-20%": [], "40-100%": [], "<10%": []}
    shares  = {"<1M": [], "1-5M": [], "5-10M": [], "10-30M": [], "30M+": []}

    for rv, tr, so, nd in rows:
        if rv is not None:
            if rv >= 50:   relvol[">=50x"].append(nd)
            elif rv >= 25: relvol["25-50x"].append(nd)
            elif rv >= 10: relvol["10-25x"].append(nd)
            else:          relvol["<10x"].append(nd)

        if tr is not None:
            if 20 <= tr <= 40:        gain["20-40%"].append(nd)
            elif 10 <= tr < 20:       gain["10-20%"].append(nd)
            elif 40 < tr <= 100:      gain["40-100%"].append(nd)
            elif tr < 10:             gain["<10%"].append(nd)

        if so is not None:
            if so < 1_000_000:        shares["<1M"].append(nd)
            elif so < 5_000_000:      shares["1-5M"].append(nd)
            elif so < 10_000_000:     shares["5-10M"].append(nd)
            elif so < 30_000_000:     shares["10-30M"].append(nd)
            else:                     shares["30M+"].append(nd)

    return {
        "total_trades": len(rows),
        "relative_volume": {k: bucket_stats(v) for k, v in relvol.items()},
        "daily_gain":      {k: bucket_stats(v) for k, v in gain.items()},
        "shares_outstanding": {k: bucket_stats(v) for k, v in shares.items()},
    }


# ---------------- FEEDBACK ----------------
def save_feedback(symbol: str, user_text: str, chart_analysis: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (created_at, symbol, user_text, chart_analysis)
        VALUES (?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), symbol or "", user_text or "", chart_analysis or ""))
    conn.commit()
    conn.close()


def get_recent_feedback(limit: int = 10):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, created_at, symbol, user_text, chart_analysis
        FROM feedback
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": r[0], "created_at": r[1], "symbol": r[2],
         "user_text": r[3], "chart_analysis": r[4]}
        for r in rows
    ]


def get_all_feedback():
    """Returns all feedback entries for hypothesis synthesis."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, created_at, symbol, user_text, chart_analysis
        FROM feedback
        ORDER BY id ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": r[0], "created_at": r[1], "symbol": r[2],
         "user_text": r[3], "chart_analysis": r[4]}
        for r in rows
    ]


# ---------------- HYPOTHESIS STORAGE ----------------
def save_hypothesis(content: str, feedback_count: int):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at)
        VALUES ('hypothesis', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                       updated_at = excluded.updated_at
    """, (content, datetime.utcnow().isoformat()))
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at)
        VALUES ('hypothesis_feedback_count', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                       updated_at = excluded.updated_at
    """, (str(feedback_count), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_hypothesis():
    """Returns (content, feedback_count, updated_at) or None if no hypothesis stored."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT value, updated_at FROM settings WHERE key = 'hypothesis'
    """)
    row = cursor.fetchone()
    cursor.execute("""
        SELECT value FROM settings WHERE key = 'hypothesis_feedback_count'
    """)
    count_row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "content": row[0],
        "feedback_count": int(count_row[0]) if count_row else 0,
        "updated_at": row[1],
    }


# ---------------- COMPLEX + AI WEIGHT STORAGE ----------------
def save_squeeze_weights(weights: dict, rationale: str = "",
                         suggestions: list = None, summary: str = ""):
    """Persist AI-optimized squeeze weights to the settings table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    for key, value in [
        ("squeeze_weights",             json.dumps(weights)),
        ("squeeze_weights_rationale",   rationale),
        ("squeeze_weights_suggestions", json.dumps(suggestions or [])),
        ("squeeze_weights_summary",     summary),
    ]:
        cursor.execute("""
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
        """, (key, value, now))

    conn.commit()
    conn.close()


def get_squeeze_weights():
    """
    Returns dict with AI-optimized weights, rationale, suggestions, summary,
    and updated_at. Returns None if no weights have been stored yet.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    keys = [
        "squeeze_weights",
        "squeeze_weights_rationale",
        "squeeze_weights_suggestions",
        "squeeze_weights_summary",
    ]
    result = {}
    for key in keys:
        cursor.execute("SELECT value, updated_at FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            result[key] = row[0]
            result["updated_at"] = row[1]

    conn.close()

    if "squeeze_weights" not in result:
        return None

    return {
        "weights":     json.loads(result["squeeze_weights"]),
        "rationale":   result.get("squeeze_weights_rationale", ""),
        "suggestions": json.loads(result.get("squeeze_weights_suggestions", "[]")),
        "summary":     result.get("squeeze_weights_summary", ""),
        "updated_at":  result.get("updated_at", ""),
    }
