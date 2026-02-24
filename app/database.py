import os
import sqlite3
import json
from datetime import datetime, timedelta
import yfinance as yf
from passlib.context import CryptContext

# On Render: mount a persistent disk at /data and set DB_PATH=/data/scan_history.db
# Locally: falls back to scan_history.db in the project root
DB_NAME = os.environ.get("DB_PATH", "scan_history.db")

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

    # ---------------- WATCHLIST TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol         TEXT NOT NULL UNIQUE,
            added_at       TEXT NOT NULL,
            price_at_add   REAL
        )
    """)

    # ---------------- WEIGHT CHANGELOG TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weight_changelog (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            updated_at  TEXT NOT NULL,
            summary     TEXT,
            rationale   TEXT,
            weights     TEXT
        )
    """)

    conn.commit()

    # Migration: add notes column to trades if not present
    try:
        cursor.execute("ALTER TABLE trades ADD COLUMN notes TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: add days_to_20pct — how many trading days until stock first hit +20%
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN days_to_20pct INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: per-trade take-profit target (default 20%)
    try:
        cursor.execute("ALTER TABLE trades ADD COLUMN take_profit_pct REAL DEFAULT 20.0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

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
        SELECT score, next_day_return, days_to_20pct
        FROM scans
        WHERE next_day_return IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    buckets = {"80-100": [], "60-79": [], "40-59": [], "0-39": []}

    for score, ret, d20 in rows:
        item = (ret, d20)
        if score >= 80:   buckets["80-100"].append(item)
        elif score >= 60: buckets["60-79"].append(item)
        elif score >= 40: buckets["40-59"].append(item)
        else:             buckets["0-39"].append(item)

    results = {}

    for bucket, items in buckets.items():
        if items:
            values  = [v for v, d in items]
            wins    = [v for v in values if v > 0]
            hits_20 = [(v, d) for v, d in items if v >= 20]
            valid_days = [d for v, d in hits_20 if d is not None]
            avg_days = round(sum(valid_days) / len(valid_days), 1) if valid_days else None
            results[bucket] = {
                "trades":            len(items),
                "avg_return":        round(sum(values)/len(values), 2),
                "win_rate":          round(len(wins)/len(items)*100, 2),
                "hit_20pct":         round(len(hits_20)/len(items)*100, 2),
                "avg_days_to_20pct": avg_days,
            }
        else:
            results[bucket] = {
                "trades": 0, "avg_return": 0, "win_rate": 0,
                "hit_20pct": 0, "avg_days_to_20pct": None,
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

            # Fetch 14 calendar days to cover 7 trading days
            end_date = min(scan_date + timedelta(days=14), today)

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
            base   = closes[0]

            next_day  = None
            three_day = None
            if len(closes) >= 2:
                next_day  = (closes[1] - base) / base * 100
            if len(closes) >= 4:
                three_day = (closes[3] - base) / base * 100

            # First trading day within 7 where stock closed up ≥20% from scan close
            days_to_20pct = None
            for d in range(1, min(8, len(closes))):
                if (closes[d] / base - 1) >= 0.20:
                    days_to_20pct = d
                    break

            updates.append((
                round(next_day,  2) if next_day  is not None else None,
                round(three_day, 2) if three_day is not None else None,
                days_to_20pct,
                scan_id,
            ))

        except Exception:
            continue

    if updates:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        for (nd, td, d20, sid) in updates:
            cursor.execute("""
                UPDATE scans
                SET next_day_return = ?, three_day_return = ?, days_to_20pct = ?
                WHERE id = ?
            """, (nd, td, d20, sid))
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


def open_trade(symbol: str, price: float, position_size: float = POSITION_SIZE,
               notes: str = "", take_profit_pct: float = 20.0):
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
        INSERT INTO trades (symbol, entry_price, shares, position_size, status,
                            opened_at, notes, take_profit_pct)
        VALUES (?, ?, ?, ?, 'open', ?, ?, ?)
    """, (symbol, round(price, 4), round(shares, 6), position_size,
          opened_at, notes or "", take_profit_pct))

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
        SELECT id, symbol, entry_price, shares, position_size, opened_at,
               COALESCE(notes, ''), COALESCE(take_profit_pct, 20.0)
        FROM trades WHERE status = 'open'
        ORDER BY opened_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    positions = []
    for row in rows:
        trade_id, symbol, entry_price, shares, position_size, opened_at, notes, take_profit_pct = row
        positions.append({
            "trade_id": trade_id,
            "symbol": symbol,
            "entry_price": entry_price,
            "shares": shares,
            "position_size": position_size,
            "opened_at": opened_at,
            "notes": notes,
            "take_profit_pct": take_profit_pct,
        })
    return positions


def get_trade_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, symbol, entry_price, exit_price, shares,
               position_size, realized_pnl, opened_at, closed_at,
               COALESCE(notes, '')
        FROM trades WHERE status = 'closed'
        ORDER BY closed_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        (trade_id, symbol, entry_price, exit_price, shares,
         position_size, realized_pnl, opened_at, closed_at, notes) = row
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
            "notes": notes,
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
        SELECT relative_volume, today_return, shares_outstanding,
               next_day_return, days_to_20pct
        FROM scans
        WHERE next_day_return IS NOT NULL
        ORDER BY id DESC LIMIT 500
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    def bucket_stats(items):
        """items = list of (next_day_return, days_to_20pct)"""
        if not items:
            return {"count": 0, "avg_return": 0.0, "win_rate": 0.0,
                    "hit_20pct": 0.0, "avg_days_to_20pct": None}
        values  = [nd for nd, d in items]
        wins    = [nd for nd in values if nd > 0]
        hits_20 = [(nd, d) for nd, d in items if nd >= 20]
        valid_days = [d for nd, d in hits_20 if d is not None]
        avg_days = round(sum(valid_days) / len(valid_days), 1) if valid_days else None
        return {
            "count":             len(items),
            "avg_return":        round(sum(values) / len(values), 2),
            "win_rate":          round(len(wins)    / len(items) * 100, 1),
            "hit_20pct":         round(len(hits_20) / len(items) * 100, 1),
            "avg_days_to_20pct": avg_days,
        }

    relvol  = {">=50x": [], "25-50x": [], "10-25x": [], "<10x": []}
    gain    = {"20-40%": [], "10-20%": [], "40-100%": [], "<10%": []}
    shares  = {"<1M": [], "1-5M": [], "5-10M": [], "10-30M": [], "30M+": []}

    for rv, tr, so, nd, d20 in rows:
        item = (nd, d20)
        if rv is not None:
            if rv >= 50:   relvol[">=50x"].append(item)
            elif rv >= 25: relvol["25-50x"].append(item)
            elif rv >= 10: relvol["10-25x"].append(item)
            else:          relvol["<10x"].append(item)

        if tr is not None:
            if 20 <= tr <= 40:   gain["20-40%"].append(item)
            elif 10 <= tr < 20:  gain["10-20%"].append(item)
            elif 40 < tr <= 100: gain["40-100%"].append(item)
            elif tr < 10:        gain["<10%"].append(item)

        if so is not None:
            if so < 1_000_000:   shares["<1M"].append(item)
            elif so < 5_000_000: shares["1-5M"].append(item)
            elif so < 10_000_000:shares["5-10M"].append(item)
            elif so < 30_000_000:shares["10-30M"].append(item)
            else:                shares["30M+"].append(item)

    return {
        "total_trades":       len(rows),
        "relative_volume":    {k: bucket_stats(v) for k, v in relvol.items()},
        "daily_gain":         {k: bucket_stats(v) for k, v in gain.items()},
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


def update_feedback_analysis(feedback_id: int, new_analysis: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE feedback SET chart_analysis = ? WHERE id = ?",
        (new_analysis, feedback_id)
    )
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


# ---------------- SCAN CACHE ----------------
def save_scan_cache(mode: str, results: list, summary: dict):
    """Persist the last scan results so repeat dashboard loads are instant."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    value = json.dumps({"results": results, "summary": summary, "cached_at": now})
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (f"scan_cache_{mode}", value, now))
    conn.commit()
    conn.close()


def get_scan_cache(mode: str, max_age_minutes: int = 15):
    """
    Returns cached scan data if younger than max_age_minutes, else None.
    Result includes 'results', 'summary', and 'cache_age_minutes'.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (f"scan_cache_{mode}",))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    data = json.loads(row[0])
    cached_at = datetime.fromisoformat(data["cached_at"])
    age_minutes = (datetime.utcnow() - cached_at).total_seconds() / 60
    if age_minutes > max_age_minutes:
        return None
    data["cache_age_minutes"] = round(age_minutes)
    return data


# ---------------- WEIGHT CHANGELOG ----------------
def save_weight_changelog(summary: str, rationale: str, weights: dict):
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO weight_changelog (updated_at, summary, rationale, weights) VALUES (?, ?, ?, ?)",
        (now, summary, rationale, json.dumps(weights))
    )
    conn.commit()
    conn.close()


def get_weight_changelog(limit: int = 20) -> list:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT updated_at, summary, rationale, weights FROM weight_changelog ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "updated_at": r[0],
            "summary":    r[1],
            "rationale":  r[2],
            "weights":    json.loads(r[3]) if r[3] else {},
        }
        for r in rows
    ]


# ---------------- WATCHLIST ----------------
def add_to_watchlist(symbol: str, price: float = None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    cursor.execute(
        "INSERT OR IGNORE INTO watchlist (symbol, added_at, price_at_add) VALUES (?, ?, ?)",
        (symbol.upper().strip(), now, price)
    )
    conn.commit()
    conn.close()


def remove_from_watchlist(symbol: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper().strip(),))
    conn.commit()
    conn.close()


def get_watchlist() -> list:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, added_at, price_at_add FROM watchlist ORDER BY added_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"symbol": r[0], "added_at": r[1], "price_at_add": r[2]} for r in rows]


# ---------------- HISTORICAL BACKFILL ----------------
def save_historical_scans(examples: list) -> int:
    """
    Insert labeled historical scan examples (mode='historical').
    Clears existing historical rows first to keep the table idempotent.
    """
    if not examples:
        return 0
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scans WHERE mode = 'historical'")
    for ex in examples:
        cursor.execute("""
            INSERT INTO scans (
                timestamp, symbol, score, recommendation, mode,
                relative_volume, today_return, shares_outstanding,
                news_recent, next_day_return, three_day_return, days_to_20pct
            ) VALUES (?, ?, ?, ?, 'historical', ?, ?, ?, 0, ?, ?, ?)
        """, (
            ex["timestamp"],
            ex["symbol"],
            ex["score"],
            ex["recommendation"],
            ex.get("relative_volume"),
            ex.get("today_return"),
            ex.get("shares_outstanding"),
            ex.get("next_day_return"),
            ex.get("three_day_return"),
            ex.get("days_to_20pct"),
        ))
    conn.commit()
    conn.close()
    return len(examples)


def set_backfill_status(status: str, processed: int = 0,
                        total: int = 0, saved: int = 0):
    value = json.dumps({
        "status": status,
        "processed": processed,
        "total": total,
        "saved": saved,
        "updated_at": datetime.utcnow().isoformat(),
    })
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES ('backfill_status', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (value, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_backfill_status() -> dict:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = 'backfill_status'")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {"status": "idle", "processed": 0, "total": 0, "saved": 0}
    return json.loads(row[0])


def get_historical_count() -> int:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM scans WHERE mode = 'historical'")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0


def get_historical_examples(limit: int = 20) -> list:
    """Fetches top historical scan examples for display on the Feedback tab."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT symbol, score, today_return, relative_volume, shares_outstanding,
               next_day_return, timestamp, days_to_20pct
        FROM scans
        WHERE mode = 'historical' AND next_day_return IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "symbol":             r[0],
            "score":              r[1],
            "today_return":       round(r[2], 1) if r[2] is not None else None,
            "relative_volume":    round(r[3], 1) if r[3] is not None else None,
            "shares_outstanding": r[4],
            "next_day_return":    round(r[5], 2) if r[5] is not None else None,
            "timestamp":          (r[6] or "")[:10],
            "days_to_20pct":      r[7],
        }
        for r in rows
    ]


# ---------------- SIZING CALIBRATION STATS ----------------
def get_sizing_stats() -> dict | None:
    """
    Returns historical performance stats grouped by score bucket for use
    as quantitative calibration in position sizing decisions.
    Covers all labeled scans (historical + live with known returns).
    Returns None if fewer than 10 labeled rows exist.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT score, relative_volume, shares_outstanding,
               today_return, next_day_return, days_to_20pct
        FROM scans
        WHERE next_day_return IS NOT NULL
        ORDER BY id DESC LIMIT 1000
    """)
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 10:
        return None

    def bucket_stats(items):
        """items = list of (next_day_return, days_to_20pct)"""
        if not items:
            return None
        returns = [nd for nd, d in items]
        # "win" = hit the 20% take-profit target within 7 trading days
        hits_20    = [(nd, d) for nd, d in items if nd >= 20]
        any_pos    = sum(1 for nd in returns if nd > 0)
        valid_days = [d for nd, d in hits_20 if d is not None]
        avg_days   = round(sum(valid_days) / len(valid_days), 1) if valid_days else None
        sorted_r   = sorted(returns)
        median     = sorted_r[len(sorted_r) // 2]
        return {
            "count":             len(items),
            "win_rate":          round(len(hits_20) / len(items) * 100, 1),  # 20%+ hit rate
            "any_pos":           round(any_pos      / len(items) * 100, 1),
            "avg_return":        round(sum(returns) / len(returns), 2),
            "median":            round(median, 2),
            "avg_days_to_20pct": avg_days,
        }

    # Score buckets
    score_buckets = {"90-100": [], "75-89": [], "50-74": [], "<50": []}
    # Relative volume buckets
    rv_buckets = {">=50x": [], "25-50x": [], "10-25x": [], "<10x": []}
    # Float buckets
    float_buckets = {"<10M": [], "10-30M": [], "30M+": []}
    # Daily gain buckets
    gain_buckets = {"20-40%": [], "10-20%": [], "40-100%": [], "<10%": []}

    for score, rv, shares, today_ret, nd, d20 in rows:
        if nd is None:
            continue
        item = (nd, d20)

        # Score
        if score >= 90:        score_buckets["90-100"].append(item)
        elif score >= 75:      score_buckets["75-89"].append(item)
        elif score >= 50:      score_buckets["50-74"].append(item)
        else:                  score_buckets["<50"].append(item)

        # Relative volume
        if rv is not None:
            if rv >= 50:       rv_buckets[">=50x"].append(item)
            elif rv >= 25:     rv_buckets["25-50x"].append(item)
            elif rv >= 10:     rv_buckets["10-25x"].append(item)
            else:              rv_buckets["<10x"].append(item)

        # Float
        if shares is not None:
            if shares < 10_000_000:    float_buckets["<10M"].append(item)
            elif shares < 30_000_000:  float_buckets["10-30M"].append(item)
            else:                      float_buckets["30M+"].append(item)

        # Daily gain
        if today_ret is not None:
            if 20 <= today_ret <= 40:      gain_buckets["20-40%"].append(item)
            elif 10 <= today_ret < 20:     gain_buckets["10-20%"].append(item)
            elif 40 < today_ret <= 100:    gain_buckets["40-100%"].append(item)
            elif today_ret < 10:           gain_buckets["<10%"].append(item)

    result = {
        "total": len(rows),
        "by_score":    {k: bucket_stats(v) for k, v in score_buckets.items() if v},
        "by_relvol":   {k: bucket_stats(v) for k, v in rv_buckets.items() if v},
        "by_float":    {k: bucket_stats(v) for k, v in float_buckets.items() if v},
        "by_gain":     {k: bucket_stats(v) for k, v in gain_buckets.items() if v},
    }

    # Only return if at least one bucket has data
    if not any(result[k] for k in ("by_score", "by_relvol", "by_float", "by_gain")):
        return None
    return result


# ---------------- VALIDATION REPORTS ----------------
def save_validation_report(report: dict):
    """Persist a validation report. Keeps the last 50 runs."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()

    # Load existing list
    cursor.execute("SELECT value FROM settings WHERE key = 'validation_reports'")
    row = cursor.fetchone()
    reports = json.loads(row[0]) if row else []

    reports.insert(0, report)      # newest first
    reports = reports[:50]          # cap at 50

    cursor.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES ('validation_reports', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (json.dumps(reports), now))
    conn.commit()
    conn.close()


def get_validation_reports(limit: int = 10) -> list:
    """Return the most recent validation reports."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = 'validation_reports'")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return []
    reports = json.loads(row[0])
    return reports[:limit]


# ---------------- RISK METRICS ----------------
def get_risk_metrics() -> dict:
    """Compute Sharpe ratio, max drawdown, and win rate from closed trades."""
    import math
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT realized_pnl, position_size
        FROM trades WHERE status = 'closed' AND position_size > 0
    """)
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 2:
        return {"sharpe": None, "max_drawdown": None, "win_rate": None, "total_closed": len(rows)}

    returns = [r[0] / r[1] * 100 for r in rows]   # pnl % per trade
    wins = sum(1 for r in returns if r > 0)
    win_rate = round(wins / len(returns) * 100, 1)

    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0
    sharpe = round(mean_r / std_r, 2) if std_r > 0 else None

    # Max drawdown from equity curve
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    for r in returns:
        equity *= (1 + r / 100)
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        "sharpe":        sharpe,
        "max_drawdown":  round(max_dd, 2),
        "win_rate":      win_rate,
        "total_closed":  len(rows),
    }
