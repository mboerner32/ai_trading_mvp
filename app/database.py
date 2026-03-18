import os
import sqlite3
import json
import time
import threading
from datetime import datetime, timedelta
import yfinance as yf
from passlib.context import CryptContext

# Resolve DB path: explicit env var > auto-detect Render persistent disk > local fallback.
# This ensures data always lands on the persistent volume even if DB_PATH is not set.
def _resolve_db_path() -> str:
    if os.environ.get("DB_PATH"):
        return os.environ["DB_PATH"]
    if os.path.isdir("/data"):
        return "/data/scan_history.db"
    return "scan_history.db"

DB_NAME = _resolve_db_path()

# ---------------------------------------------------------------------------
# Simple thread-safe TTL cache for expensive read-only analytics aggregations.
# Cache is keyed by (function_name, args_hash); entries expire after `ttl` seconds.
# Call analytics_cache_clear() after writes that would change aggregated results.
# ---------------------------------------------------------------------------
_ANALYTICS_CACHE: dict = {}
_ANALYTICS_CACHE_LOCK = threading.Lock()


def _cache_get(key: str):
    with _ANALYTICS_CACHE_LOCK:
        entry = _ANALYTICS_CACHE.get(key)
        if entry and time.monotonic() < entry["expires"]:
            return entry["value"], True
        return None, False


def _cache_set(key: str, value, ttl_seconds: int):
    with _ANALYTICS_CACHE_LOCK:
        _ANALYTICS_CACHE[key] = {"value": value, "expires": time.monotonic() + ttl_seconds}


def analytics_cache_clear(*keys):
    """Clear specific cache keys, or all analytics cache entries if no keys given."""
    with _ANALYTICS_CACHE_LOCK:
        if keys:
            for k in keys:
                _ANALYTICS_CACHE.pop(k, None)
        else:
            _ANALYTICS_CACHE.clear()

# Feedback backup lives alongside the DB so it persists on Render's disk too.
# Download it periodically via /feedback/export for an off-disk copy.
_DB_DIR = os.path.dirname(os.path.abspath(DB_NAME)) or "."
FEEDBACK_BACKUP_PATH = os.path.join(_DB_DIR, "feedback_backup.json")

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def _connect() -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode and a generous busy timeout."""
    conn = sqlite3.connect(DB_NAME, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


# ---------------- INIT DB ----------------
def init_db():
    conn = sqlite3.connect(DB_NAME, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")   # allow concurrent reads alongside writes
    conn.execute("PRAGMA busy_timeout=10000") # wait up to 10s before raising "database is locked"
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

    # Migration: stop loss + trailing stop support
    for col in [
        "ALTER TABLE trades ADD COLUMN stop_loss_pct REAL DEFAULT NULL",
        "ALTER TABLE trades ADD COLUMN trade_mode TEXT DEFAULT NULL",
        "ALTER TABLE trades ADD COLUMN high_watermark REAL DEFAULT NULL",
        "ALTER TABLE trades ADD COLUMN close_reason TEXT DEFAULT NULL",
    ]:
        try:
            cursor.execute(col)
            conn.commit()
        except sqlite3.OperationalError:
            pass

    # Migration: AI trade recommendation stored per scan
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN ai_trade_rec TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Migration: outcome tag for manual feedback entries
    try:
        cursor.execute("ALTER TABLE feedback ADD COLUMN outcome_tag TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Migration: outcome tag for closed paper trades
    try:
        cursor.execute("ALTER TABLE trades ADD COLUMN outcome_tag TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Migration: sideways compression value per scan (used by LSTM + AI calibration)
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN range_10d REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Migration: was yesterday green (boolean) per scan
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN yesterday_green INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Migration: float shares per scan (better than shares_outstanding for float analysis)
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN float_shares INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Migration: scan_price — price at alert time (used as base for days_to_20pct)
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN scan_price REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Migration: reset days_to_20pct=0 (same-day hits) for live scans — they used the
    # full-day high which may predate the alert. update_returns() will re-label from d=1.
    try:
        cursor.execute("""
            UPDATE scans SET days_to_20pct = NULL
            WHERE days_to_20pct = 0
              AND mode NOT IN ('historical', 'fivemin_bt')
        """)
        conn.commit()
    except Exception:
        pass

    # Migration: watchlist near-miss tracking columns
    try:
        cursor.execute("ALTER TABLE watchlist ADD COLUMN score INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE watchlist ADD COLUMN alerted INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE watchlist ADD COLUMN date TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE watchlist ADD COLUMN last_checked TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # is_admin flag on users (1 = admin, 0 = regular trader)
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        conn.commit()
        # Make the 'admin' account an admin
        cursor.execute("UPDATE users SET is_admin = 1 WHERE username = 'admin'")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # ---------------- TELEGRAM RECIPIENTS TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telegram_recipients (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id  TEXT NOT NULL UNIQUE,
            label    TEXT NOT NULL DEFAULT '',
            added_at TEXT NOT NULL
        )
    """)
    conn.commit()

    # ---------------- TELEGRAM LOG TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telegram_log (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            sent_at  TEXT NOT NULL,
            chat_id  TEXT NOT NULL,
            message  TEXT NOT NULL
        )
    """)
    conn.commit()

    # ---------------- HYPOTHESIS RULES TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hypothesis_rules (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_text  TEXT NOT NULL,
            source     TEXT NOT NULL DEFAULT '',
            status     TEXT NOT NULL DEFAULT 'pending',
            generation INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()

    # Migration: tag each scan with active rule IDs at AI-enrich time
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN active_rule_ids TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # Migration: confidence score and auto-applied flag on hypothesis rules
    try:
        cursor.execute("ALTER TABLE hypothesis_rules ADD COLUMN confidence_score INTEGER DEFAULT NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE hypothesis_rules ADD COLUMN auto_applied INTEGER NOT NULL DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    # Migration: projection_json — stores AI-validated projected impact for bundle rules
    try:
        cursor.execute("ALTER TABLE hypothesis_rules ADD COLUMN projection_json TEXT DEFAULT NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    # ---------------- AUTOAI LOG TABLE ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS autoai_log (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            ran_at                    TEXT NOT NULL,
            trigger_reason            TEXT,
            trades_evaluated          INTEGER,
            hypotheses_added          INTEGER,
            hypotheses_auto_activated INTEGER,
            hypotheses_to_pending     INTEGER,
            weights_auto_applied      INTEGER,
            weight_confidence         INTEGER,
            summary                   TEXT,
            full_response             TEXT
        )
    """)

    # Chat suggestions: non-admin users can queue action suggestions for admin review
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_suggestions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            suggested_by TEXT NOT NULL,
            suggested_at TEXT NOT NULL,
            action_json  TEXT NOT NULL,
            note         TEXT,
            status       TEXT NOT NULL DEFAULT 'pending'
        )
    """)

    # Weight versions: every weight change gets a versioned snapshot for A/B tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weight_versions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT    NOT NULL,
            weights    TEXT    NOT NULL,
            summary    TEXT,
            rationale  TEXT,
            source     TEXT,
            goal       TEXT,
            created_at TEXT    NOT NULL
        )
    """)
    conn.commit()

    # Migration: per-signal fired state for backtest tracking
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN signals_json TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: link each scan to the weight version active at scan time
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN weights_version_id INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: LSTM hit-probability at scan time (0.0–1.0, NULL if not run)
    try:
        cursor.execute("ALTER TABLE scans ADD COLUMN lstm_prob REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    # Migration: tighten time_stop_days 10 → 5 (97.1% of winners hit by day 5, 2026-03-14)
    try:
        row = cursor.execute(
            "SELECT value FROM settings WHERE key='stop_loss_params_daily'"
        ).fetchone()
        if row:
            import json as _j
            p = _j.loads(row[0])
            if p.get("time_stop_days", 10) >= 10:
                p["time_stop_days"] = 5
                p["stale_days"] = min(p.get("stale_days", 7), 5)
                cursor.execute(
                    "UPDATE settings SET value=? WHERE key='stop_loss_params_daily'",
                    (_j.dumps(p),)
                )
                conn.commit()
    except Exception:
        pass

    # ---------------- INDEXES ----------------
    # These are CREATE IF NOT EXISTS so safe to run on every startup.
    # Cover the most frequent query patterns in analytics and scan saving.
    for idx_sql in [
        # Analytics aggregations: mode + outcome filter (most queries)
        "CREATE INDEX IF NOT EXISTS idx_scans_mode_ndr ON scans(mode, next_day_return)",
        # Per-signal stats: signals_json present + outcome known
        "CREATE INDEX IF NOT EXISTS idx_scans_signals_ndr ON scans(next_day_return, signals_json) WHERE signals_json IS NOT NULL",
        # Model comparison: mode + ai_trade_rec + outcome
        "CREATE INDEX IF NOT EXISTS idx_scans_mode_ai ON scans(mode, ai_trade_rec, next_day_return)",
        # LSTM gate validation: lstm_prob + outcome
        "CREATE INDEX IF NOT EXISTS idx_scans_lstm_ndr ON scans(lstm_prob, next_day_return, days_to_20pct)",
        # Symbol + timestamp lookups (ticker history, dedup)
        "CREATE INDEX IF NOT EXISTS idx_scans_symbol_ts ON scans(symbol, timestamp)",
        # Hypothesis rule tag lookups (active_rule_ids)
        "CREATE INDEX IF NOT EXISTS idx_scans_rule_ids ON scans(active_rule_ids) WHERE active_rule_ids IS NOT NULL",
        # Trades: status filter (open positions)
        "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
    ]:
        try:
            cursor.execute(idx_sql)
        except Exception:
            pass
    conn.commit()

    conn.close()
    _seed_user_observations()


# ---------------- SEED USER OBSERVATIONS ----------------
# Observations submitted by users that the AI should test against backtest data.
# Stored as 'pending' hypothesis rules so they appear in the registry and get
# picked up by autonomous_optimize() for evidence-based validation.
_USER_OBSERVATIONS = [
    {
        "text": (
            "HYPOTHESIS: Stocks with ≥20M cumulative volume in the first trading hour (9:30–10:30 ET) "
            "are significantly more likely to hit 20%+ intraday. "
            "Observed in all manually analyzed 5m spike winners. "
            "Signal tracked as first_hour_vol_20m in fired_signals — validate against backtest data."
        ),
        "source": "user_observation",
    },
]

def _seed_user_observations():
    """Insert user-submitted observations as pending hypothesis rules if they don't already exist."""
    conn = sqlite3.connect(DB_NAME, timeout=10)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    for obs in _USER_OBSERVATIONS:
        cursor.execute(
            "SELECT COUNT(*) FROM hypothesis_rules WHERE source = ? AND rule_text LIKE ?",
            (obs["source"], obs["text"][:60] + "%")
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                """INSERT INTO hypothesis_rules
                   (rule_text, source, status, generation, created_at, confidence_score, auto_applied)
                   VALUES (?, ?, 'pending', 0, ?, 0, 0)""",
                (obs["text"], obs["source"], now)
            )
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


def is_user_admin(username: str) -> bool:
    """Return True if the given username has is_admin=1 in the DB."""
    if not username:
        return False
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(is_admin, 0) FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return bool(row and row[0])


def get_all_users() -> list:
    """Return list of all users (id, username, is_admin). Passwords excluded."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, COALESCE(is_admin, 0) FROM users ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "username": r[1], "is_admin": bool(r[2])} for r in rows]


def delete_user(username: str) -> bool:
    """Delete a user by username. Returns True if deleted, False if not found.
    The 'admin' account cannot be deleted."""
    if username == "admin":
        return False
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def seed_users():
    """
    Create default + env-configured accounts on startup if they do not already exist.
    Uses ON CONFLICT DO NOTHING so passwords changed via /account are never overwritten.

    Seeding sources (all use ON CONFLICT DO NOTHING):
      1. Hardcoded defaults: admin (ADMIN_PASSWORD env var) + BobbyAxelrod (BOB_PASSWORD env var)
      2. SEED_USERS env var — JSON array of {"username": "...", "password": "..."} objects.
         Add every web-created user here on Render so they survive DB resets:
         SEED_USERS=[{"username":"kelly","password":"her_password"},{"username":"rob","password":"his_password"}]
    """
    import os as _os
    admin_pw = _os.environ.get("ADMIN_PASSWORD")
    bob_pw   = _os.environ.get("BOB_PASSWORD")
    if not admin_pw:
        print("STARTUP WARNING: ADMIN_PASSWORD env var not set — admin account will NOT be seeded. "
              "Set ADMIN_PASSWORD in Render environment variables.")
    if not bob_pw:
        print("STARTUP WARNING: BOB_PASSWORD env var not set — BobbyAxelrod account will NOT be seeded.")
    defaults = [
        (u, p) for u, p in [
            ("admin",        admin_pw),
            ("BobbyAxelrod", bob_pw),
        ] if p  # skip accounts with no password configured
    ]

    # Load extra users from SEED_USERS env var (JSON array)
    seed_users_json = _os.environ.get("SEED_USERS", "")
    extra_users = []
    if seed_users_json:
        try:
            parsed = json.loads(seed_users_json)
            if isinstance(parsed, list):
                for u in parsed:
                    if isinstance(u, dict) and u.get("username") and u.get("password"):
                        extra_users.append((u["username"], u["password"]))
        except Exception as e:
            print(f"SEED_USERS parse error (skipping extra users): {e}")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    for username, password in defaults + extra_users:
        hashed = pwd_context.hash(password)
        cursor.execute("""
            INSERT INTO users (username, hashed_password) VALUES (?, ?)
            ON CONFLICT(username) DO NOTHING
        """, (username, hashed))
    if extra_users:
        # Ensure all SEED_USERS accounts have is_admin=0 (DO NOTHING on conflict preserves existing role)
        for username, _ in extra_users:
            cursor.execute(
                "UPDATE users SET is_admin = COALESCE(is_admin, 0) WHERE username = ?",
                (username,)
            )
    conn.commit()
    conn.close()
    if extra_users:
        print(f"SEED_USERS: seeded {len(extra_users)} extra user(s): {[u for u, _ in extra_users]}")


def authenticate_user(username: str, password: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT hashed_password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False

    return pwd_context.verify(password, row[0])


def update_password(username: str, new_password: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    hashed = pwd_context.hash(new_password)
    cursor.execute(
        "UPDATE users SET hashed_password = ? WHERE username = ?",
        (hashed, username)
    )
    conn.commit()
    conn.close()


# ---------------- HOLDING PERFORMANCE ----------------
def _mode_clause(modes):
    """Return (sql_snippet, params) for an optional mode IN filter."""
    if modes:
        placeholders = ",".join("?" * len(modes))
        return f" AND mode IN ({placeholders})", list(modes)
    return "", []


def get_holding_performance(modes=None):
    _ck = f"holding_perf:{modes}"
    _cached, _hit = _cache_get(_ck)
    if _hit:
        return _cached
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    mc, mp = _mode_clause(modes)
    cursor.execute(f"""
        SELECT next_day_return, three_day_return
        FROM scans
        WHERE next_day_return IS NOT NULL{mc}
    """, mp)
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return None
    day1 = [r[0] for r in rows if r[0] is not None]
    day3 = [r[1] for r in rows if r[1] is not None]
    _result = {
        "avg_day1": round(sum(day1)/len(day1), 2) if day1 else 0,
        "avg_day3": round(sum(day3)/len(day3), 2) if day3 else 0
    }
    _cache_set(_ck, _result, 3600)
    return _result


# ---------------- EQUITY CURVE ----------------
def get_equity_curve(starting_capital=10000, modes=None):
    _ck = f"equity_curve:{starting_capital}:{modes}"
    _cached, _hit = _cache_get(_ck)
    if _hit:
        return _cached
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    mc, mp = _mode_clause(modes)
    cursor.execute(f"""
        SELECT next_day_return FROM scans
        WHERE next_day_return IS NOT NULL{mc}
        ORDER BY id ASC
    """, mp)
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return None
    capital = starting_capital
    equity = []
    for (ret,) in rows:
        capital *= (1 + ret / 100)
        equity.append(round(capital, 2))
    _cache_set(_ck, equity, 7200)
    return equity


# ---------------- SCORE BUCKETS ----------------
def get_score_buckets(modes=None):
    _ck = f"score_buckets:{modes}"
    _cached, _hit = _cache_get(_ck)
    if _hit:
        return _cached
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    mc, mp = _mode_clause(modes)
    cursor.execute(f"""
        SELECT score, next_day_return, days_to_20pct
        FROM scans
        WHERE next_day_return IS NOT NULL{mc}
    """, mp)
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
            hits_20 = [(v, d) for v, d in items if d is not None]  # intraday HIGH touched +20%
            valid_days = [d for v, d in hits_20]  # all have d non-None by definition
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

    _cache_set(_ck, results, 3600)
    return results


def get_model_comparison_stats() -> dict:
    """
    Returns per-model (autoai vs squeeze) TRADE call performance broken down by score bucket.
    Cached for 1800s (30 min); invalidate via analytics_cache_clear('model_comparison').

    For each model × score bucket, tracks:
      - trade_calls:    number of AI TRADE calls with known outcomes
      - win_rate:       % of TRADE calls with positive next-day return
      - avg_next_day:   average next-day return %
      - avg_3day:       average 3-day return %  (medium-term, ~1 week)
      - hit_20pct:      % of TRADE calls that touched +20% within 10 trading days (~2 weeks)
      - avg_days_to_20: average trading days to hit the +20% target
      - no_trade_calls: number of NO_TRADE calls (for comparison)

    Uses json_extract() to parse the ai_trade_rec JSON column.
    Only includes scans with next_day_return IS NOT NULL (outcomes known).
    Excludes 'historical' mode scans.
    """
    _ck = "model_comparison"
    _cached, _hit = _cache_get(_ck)
    if _hit:
        return _cached
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    MODELS = {"autoai": "Auto AI", "squeeze": "Complex + AI"}
    BUCKET_DEFS = [
        ("80-100", 80, 101),
        ("75-79",  75, 80),
        ("60-74",  60, 75),
        ("0-59",    0, 60),
    ]

    result = {}

    for mode_key, mode_label in MODELS.items():
        # All TRADE/NO_TRADE calls with known returns for this mode.
        # We fetch timestamp so we can separate closed-window rows from pending ones.
        # 10 trading days ≈ 14 calendar days. A row is "window closed" when either:
        #   (a) days_to_20pct IS NOT NULL  — hit 20%, confirmed
        #   (b) timestamp < now - 14 days  — enough time passed; stock didn't hit
        # Rows where window is still open (< 14 days old, d20 IS NULL) are "outstanding"
        # and are excluded from hit_20pct to avoid deflating the rate with pending trades.
        cursor.execute("""
            SELECT score,
                   next_day_return,
                   three_day_return,
                   days_to_20pct,
                   json_extract(ai_trade_rec, '$.decision') as decision,
                   timestamp
            FROM scans
            WHERE next_day_return IS NOT NULL
              AND mode = ?
              AND ai_trade_rec IS NOT NULL
        """, (mode_key,))
        rows = cursor.fetchall()

        def _window_closed(d20, ts):
            """True if the 10-day outcome window has definitively closed."""
            if d20 is not None:
                return True  # already hit 20%
            try:
                from datetime import datetime as _dt, timezone as _tz
                scan_dt = _dt.fromisoformat(ts.replace("Z", "+00:00"))
                age_days = (_dt.now(_tz.utc) - scan_dt.replace(tzinfo=_tz.utc)
                            if scan_dt.tzinfo is None
                            else _dt.now(_tz.utc) - scan_dt).days
                return age_days >= 14
            except Exception:
                return True  # assume closed if we can't parse

        buckets = {}
        for bucket_label, lo, hi in BUCKET_DEFS:
            # All TRADE rows in this bucket (for win_rate / avg_return calcs)
            trade_rows = [
                (nd, td, d20) for score, nd, td, d20, dec, ts in rows
                if lo <= score < hi and dec == "TRADE"
            ]
            # Only closed-window rows for hit_20pct (excludes pending)
            closed_trade_rows = [
                (nd, td, d20) for score, nd, td, d20, dec, ts in rows
                if lo <= score < hi and dec == "TRADE" and _window_closed(d20, ts)
            ]
            outstanding_count = len(trade_rows) - len(closed_trade_rows)
            no_trade_count = sum(
                1 for score, nd, td, d20, dec, ts in rows
                if lo <= score < hi and dec == "NO_TRADE"
            )
            total_scored = sum(1 for score, *_ in rows if lo <= score < hi)

            if trade_rows:
                nd_vals   = [nd for nd, td, d20 in trade_rows if nd is not None]
                td_vals   = [td for nd, td, d20 in trade_rows if td is not None]
                wins      = [v for v in nd_vals if v > 0]
                hits_20   = [d20 for nd, td, d20 in closed_trade_rows if d20 is not None]
                denom_20  = len(closed_trade_rows)
                avg_days  = round(sum(hits_20) / len(hits_20), 1) if hits_20 else None
                nd_sorted = sorted(nd_vals)
                td_sorted = sorted(td_vals)
                buckets[bucket_label] = {
                    "trade_calls":      len(trade_rows),
                    "no_trade_calls":   no_trade_count,
                    "total_scored":     total_scored,
                    "outstanding":      outstanding_count,
                    "win_rate":         round(len(wins) / len(nd_vals) * 100, 1) if nd_vals else None,
                    "avg_next_day":     round(sum(nd_vals) / len(nd_vals), 2) if nd_vals else None,
                    "median_next_day":  round(nd_sorted[len(nd_sorted) // 2], 2) if nd_sorted else None,
                    "avg_3day":         round(sum(td_vals) / len(td_vals), 2) if td_vals else None,
                    "median_3day":      round(td_sorted[len(td_sorted) // 2], 2) if td_sorted else None,
                    "hit_20pct":        round(len(hits_20) / denom_20 * 100, 1) if denom_20 else None,
                    "avg_days_to_20":   avg_days,
                }
            else:
                buckets[bucket_label] = {
                    "trade_calls":      0,
                    "no_trade_calls":   no_trade_count,
                    "total_scored":     total_scored,
                    "outstanding":      0,
                    "win_rate":         None,
                    "avg_next_day":     None,
                    "median_next_day":  None,
                    "avg_3day":         None,
                    "median_3day":      None,
                    "hit_20pct":        None,
                    "avg_days_to_20":   None,
                }

        # Overall totals for this model
        all_trade_rows = [
            (nd, td, d20) for score, nd, td, d20, dec, ts in rows if dec == "TRADE"
        ]
        all_closed_trade_rows = [
            (nd, td, d20) for score, nd, td, d20, dec, ts in rows
            if dec == "TRADE" and _window_closed(d20, ts)
        ]
        outstanding_total = len(all_trade_rows) - len(all_closed_trade_rows)
        nd_all  = [nd for nd, td, d20 in all_trade_rows if nd is not None]
        td_all  = [td for nd, td, d20 in all_trade_rows if td is not None]
        h20_all = [d20 for nd, td, d20 in all_closed_trade_rows if d20 is not None]
        denom_20_all = len(all_closed_trade_rows)
        nd_all_sorted = sorted(nd_all)
        td_all_sorted = sorted(td_all)

        result[mode_key] = {
            "label":   mode_label,
            "buckets": buckets,
            "overall": {
                "trade_calls":         len(all_trade_rows),
                "outstanding":         outstanding_total,
                "no_trade_calls":      sum(1 for *_, dec, ts in rows if dec == "NO_TRADE"),
                "total_with_outcomes": len(rows),
                "win_rate":            round(sum(1 for v in nd_all if v > 0) / len(nd_all) * 100, 1) if nd_all else None,
                "avg_next_day":        round(sum(nd_all) / len(nd_all), 2) if nd_all else None,
                "median_next_day":     round(nd_all_sorted[len(nd_all_sorted) // 2], 2) if nd_all_sorted else None,
                "avg_3day":            round(sum(td_all) / len(td_all), 2) if td_all else None,
                "median_3day":         round(td_all_sorted[len(td_all_sorted) // 2], 2) if td_all_sorted else None,
                "hit_20pct":           round(len(h20_all) / denom_20_all * 100, 1) if denom_20_all else None,
                "avg_days_to_20":      round(sum(h20_all) / len(h20_all), 1) if h20_all else None,
            },
        }

    conn.close()
    _cache_set(_ck, result, 1800)
    return result
# ---------------- RETURN UPDATES ----------------
def update_returns():
    # Fetch pending rows then close connection before slow yf.download calls
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Also pick up rows where next_day_return is set but days_to_20pct is NULL
    # (happens after resetting stale d=0 same-day hits)
    cursor.execute("""
        SELECT id, symbol, timestamp, scan_price
        FROM scans
        WHERE next_day_return IS NULL
           OR (next_day_return IS NOT NULL AND days_to_20pct IS NULL
               AND mode NOT IN ('historical', 'fivemin_bt'))
    """)
    rows = cursor.fetchall()
    conn.close()

    today = datetime.utcnow().date()
    updates = []

    for scan_id, symbol, timestamp, scan_price in rows:
        try:
            scan_date = datetime.fromisoformat(timestamp).date()

            if (today - scan_date).days < 2:
                continue

            # Fetch 21 calendar days to cover 10 trading days (2 weeks), incl. holiday-heavy weeks
            end_date = min(scan_date + timedelta(days=21), today)

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
            highs  = data["High"].tolist()

            # Use scan_price (alert price) as base if available — measures from actual entry point.
            # Falls back to scan-day close for older rows that predate this column.
            base = float(scan_price) if scan_price else closes[0]

            next_day  = None
            three_day = None
            if len(closes) >= 2:
                next_day  = (closes[1] - base) / base * 100
            if len(closes) >= 4:
                three_day = (closes[3] - base) / base * 100

            # Discard implausible returns — scan_price/yfinance mismatch
            # (e.g. overnight scan captured pre-squeeze price vs. split-adjusted close).
            # A legitimate next-day move from alert price won't exceed ±500%.
            if next_day  is not None and abs(next_day)  > 500: next_day  = None
            if three_day is not None and abs(three_day) > 500: three_day = None

            # Check next 10 trading days (d=1..10) for intraday HIGH ≥20% above alert price.
            # d=0 (same-day) excluded: the day's high may predate the alert, making it
            # unreachable — entry price is the scan_price at alert time, not the open.
            days_to_20pct = None
            for d in range(1, min(11, len(highs))):
                if (highs[d] / base - 1) >= 0.20:
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
        # New outcome labels → analytics aggregations are stale
        analytics_cache_clear()


# ---------------- SAVE SCAN ----------------
def save_scan(results: list, mode: str) -> dict:
    """Saves scan results and returns {symbol: scan_id} for AI rec persistence."""
    # Determine which model type's version to stamp (fivemin uses squeeze weights)
    model_type = "autoai" if mode == "autoai" else "squeeze"
    version_id = get_current_version_id(model_type)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.utcnow().isoformat()
    scan_ids = {}

    for r in results:
        checklist = r.get("checklist", {})
        fired = checklist.get("fired_signals", {})
        signals_json_str = json.dumps(fired) if fired else None
        cursor.execute("""
            INSERT INTO scans (
                timestamp, symbol, score, recommendation, mode,
                relative_volume, today_return, shares_outstanding,
                news_recent, next_day_return, three_day_return, scan_price,
                signals_json, weights_version_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            None,
            r.get("price"),
            signals_json_str,
            version_id,
        ))
        scan_ids[r.get("symbol")] = cursor.lastrowid

    conn.commit()
    conn.close()
    return scan_ids


def save_scan_candidates(candidates: list, mode: str):
    """
    Save ALL stocks seen by the FinViz screener (even unscored / low-score ones)
    with mode='candidate_<mode>' so outcomes can be labeled later and the AI can
    test hypotheses against the full observation pool, not just high-scoring results.
    Skips any symbol already saved in the last 6 hours to avoid duplication.
    """
    if not candidates:
        return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.utcnow().isoformat()
    candidate_mode = f"candidate_{mode}"
    cutoff = (datetime.utcnow() - timedelta(hours=6)).isoformat()

    for c in candidates:
        symbol = c.get("symbol")
        if not symbol:
            continue
        # Skip if already saved recently (duplicate suppression)
        cursor.execute(
            "SELECT COUNT(*) FROM scans WHERE symbol=? AND mode=? AND timestamp > ?",
            (symbol, candidate_mode, cutoff)
        )
        if cursor.fetchone()[0] > 0:
            continue
        fired = c.get("fired_signals", {})
        signals_json_str = json.dumps(fired) if fired else None
        cursor.execute("""
            INSERT INTO scans (
                timestamp, symbol, score, recommendation, mode,
                relative_volume, today_return, shares_outstanding,
                news_recent, next_day_return, three_day_return, scan_price,
                signals_json, weights_version_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            symbol,
            c.get("score", 0),
            c.get("recommendation", "CANDIDATE"),
            candidate_mode,
            c.get("relative_volume"),
            c.get("daily_return_pct"),
            c.get("shares_outstanding"),
            0, None, None,
            c.get("price"),
            signals_json_str,
            None,
        ))
    conn.commit()
    conn.close()


def update_scan_ai_rec(scan_id: int, decision: str, confidence: str, rationale: str,
                       lstm_prob: float = None):
    """Persists the AI trade recommendation and LSTM probability to the scans table."""
    import json
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE scans SET ai_trade_rec = ?, lstm_prob = ? WHERE id = ?",
        (json.dumps({"decision": decision, "confidence": confidence, "rationale": rationale}),
         lstm_prob, scan_id)
    )
    conn.commit()
    conn.close()


def get_ticker_scan_history(symbol: str, limit: int = 5) -> list:
    """Returns recent scan records for a ticker, used as context for AI trade calls."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, score, recommendation, relative_volume,
               today_return, next_day_return, ai_trade_rec
        FROM scans WHERE symbol = ? AND mode != 'historical'
        ORDER BY timestamp DESC LIMIT ?
    """, (symbol, limit))
    rows = cursor.fetchall()
    conn.close()
    return [
        {"timestamp": r[0][:10] if r[0] else "", "score": r[1], "rec": r[2],
         "relvol": r[3], "return_pct": r[4], "next_day": r[5], "ai_call": r[6]}
        for r in rows
    ]


def tag_feedback_outcome(feedback_id: int, tag):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE feedback SET outcome_tag = ? WHERE id = ?", (tag, feedback_id))
    conn.commit()
    conn.close()


def tag_trade_outcome(trade_id: int, tag):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE trades SET outcome_tag = ? WHERE id = ?", (tag, trade_id))
    conn.commit()
    conn.close()


# ---------------- PAPER TRADING ----------------
POSITION_SIZE = 1000.0


def open_trade(symbol: str, price: float, position_size: float = POSITION_SIZE,
               notes: str = "", take_profit_pct: float = 20.0,
               stop_loss_pct: float = None, trade_mode: str = None):
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
                            opened_at, notes, take_profit_pct,
                            stop_loss_pct, trade_mode, high_watermark)
        VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?)
    """, (symbol, round(price, 4), round(shares, 6), position_size,
          opened_at, notes or "", take_profit_pct,
          stop_loss_pct, trade_mode, round(price, 4)))

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


def close_trade(trade_id: int, exit_price: float, close_reason: str = None):
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
        SET status = 'closed', exit_price = ?, realized_pnl = ?, closed_at = ?,
            close_reason = ?
        WHERE id = ?
    """, (round(exit_price, 4), round(realized_pnl, 4), closed_at, close_reason, trade_id))
    cursor.execute(
        "UPDATE portfolio SET cash = cash + ? WHERE id = 1",
        (round(proceeds, 4),)
    )

    conn.commit()
    conn.close()
    return {"realized_pnl": realized_pnl, "proceeds": proceeds}


def update_high_watermark(trade_id: int, price: float):
    """Update the high_watermark for a trade if price is a new peak."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE trades SET high_watermark = ? WHERE id = ? AND (high_watermark IS NULL OR ? > high_watermark)",
            (round(price, 4), trade_id, round(price, 4))
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def count_auto_trades_today(date_str: str) -> int:
    """Count auto-trades (open + closed) opened on date_str (YYYY-MM-DD)."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM trades
            WHERE notes LIKE 'auto-trade%'
              AND DATE(opened_at) = ?
        """, (date_str,))
        count = cursor.fetchone()[0] or 0
        conn.close()
        return count
    except Exception:
        return 0


def get_alerted_symbols_today(date_str: str) -> set:
    """Return set of symbols that had a TRADE alert saved today (for dedup across restarts)."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT symbol FROM scans
            WHERE ai_trade_rec LIKE '%TRADE%'
              AND DATE(scan_date) = ?
        """, (date_str,))
        symbols = {row[0] for row in cursor.fetchall()}
        conn.close()
        return symbols
    except Exception:
        return set()


def get_open_positions():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, symbol, entry_price, shares, position_size, opened_at,
               COALESCE(notes, ''), COALESCE(take_profit_pct, 20.0),
               stop_loss_pct, trade_mode, high_watermark
        FROM trades WHERE status = 'open'
        ORDER BY opened_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    positions = []
    for row in rows:
        (trade_id, symbol, entry_price, shares, position_size, opened_at,
         notes, take_profit_pct, stop_loss_pct, trade_mode, high_watermark) = row
        positions.append({
            "trade_id":        trade_id,
            "symbol":          symbol,
            "entry_price":     entry_price,
            "shares":          shares,
            "position_size":   position_size,
            "opened_at":       opened_at,
            "notes":           notes,
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct":   stop_loss_pct,
            "trade_mode":      trade_mode,
            "high_watermark":  high_watermark or entry_price,
        })
    return positions


def get_trade_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, symbol, entry_price, exit_price, shares,
               position_size, realized_pnl, opened_at, closed_at,
               COALESCE(notes, ''), COALESCE(outcome_tag, '')
        FROM trades WHERE status = 'closed'
        ORDER BY closed_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        (trade_id, symbol, entry_price, exit_price, shares,
         position_size, realized_pnl, opened_at, closed_at, notes, outcome_tag) = row
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
            "outcome_tag": outcome_tag,
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
    # Deduped base: one row per symbol per day, daily modes only, confirmed outcomes.
    cursor.execute("""
        SELECT MAX(relative_volume), MAX(today_return), MIN(shares_outstanding),
               MAX(next_day_return), MIN(days_to_20pct), MIN(timestamp)
        FROM scans
        WHERE mode NOT IN ('fivemin','fivemin_bt','candidate_fivemin','standard','strict')
          AND (days_to_20pct IS NOT NULL
               OR (julianday('now') - julianday(timestamp)) >= 14)
        GROUP BY symbol, DATE(timestamp)
        ORDER BY MIN(timestamp) DESC LIMIT 500
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
        values     = [nd for nd, d in items if nd is not None]
        wins       = [nd for nd in values if nd > 0]
        hits_20    = [(nd, d) for nd, d in items if d is not None]
        valid_days = [d for nd, d in hits_20]
        avg_days   = round(sum(valid_days) / len(valid_days), 1) if valid_days else None
        return {
            "count":             len(items),
            "avg_return":        round(sum(values) / len(values), 2) if values else 0.0,
            "win_rate":          round(len(wins) / len(values) * 100, 1) if values else 0.0,
            "hit_20pct":         round(len(hits_20) / len(items) * 100, 1),
            "avg_days_to_20pct": avg_days,
        }

    relvol  = {"500x+": [], "100-499x": [], "50-99x": [], "25-49x": [], "10-24x": [], "<10x": []}
    gain    = {">80%": [], "50-80%": [], "30-50%": [], "20-30%": [], "10-20%": [], "<10%": []}
    shares  = {"<10M": [], "10-30M": [], "30-100M": [], "100M+": []}
    dow     = {"Monday": [], "Tuesday": [], "Wednesday": [], "Thursday": [], "Friday": []}
    dow_map = {"1": "Monday", "2": "Tuesday", "3": "Wednesday", "4": "Thursday", "5": "Friday"}

    for rv, tr, so, nd, d20, ts in rows:
        item = (nd, d20)
        if rv is not None:
            if rv >= 500:  relvol["500x+"].append(item)
            elif rv >= 100: relvol["100-499x"].append(item)
            elif rv >= 50:  relvol["50-99x"].append(item)
            elif rv >= 25:  relvol["25-49x"].append(item)
            elif rv >= 10:  relvol["10-24x"].append(item)
            else:           relvol["<10x"].append(item)

        if tr is not None:
            if tr >= 80:         gain[">80%"].append(item)
            elif tr >= 50:       gain["50-80%"].append(item)
            elif tr >= 30:       gain["30-50%"].append(item)
            elif tr >= 20:       gain["20-30%"].append(item)
            elif tr >= 10:       gain["10-20%"].append(item)
            else:                gain["<10%"].append(item)

        if so is not None:
            if so < 10_000_000:  shares["<10M"].append(item)
            elif so < 30_000_000:shares["10-30M"].append(item)
            elif so < 100_000_000:shares["30-100M"].append(item)
            else:                 shares["100M+"].append(item)

        if ts:
            import datetime as _dt
            try:
                d = _dt.datetime.fromisoformat(str(ts)[:19])
                day_name = dow_map.get(str(d.weekday() + 1 if d.weekday() < 6 else d.weekday()))
                if day_name:
                    dow[day_name].append(item)
            except Exception:
                pass

    return {
        "total_trades":       len(rows),
        "relative_volume":    {k: bucket_stats(v) for k, v in relvol.items()},
        "daily_gain":         {k: bucket_stats(v) for k, v in gain.items()},
        "shares_outstanding": {k: bucket_stats(v) for k, v in shares.items()},
        "day_of_week":        {k: bucket_stats(v) for k, v in dow.items()},
        "per_signal_stats":   get_per_signal_stats(),
    }


# ---------------- PER-SIGNAL BACKTEST ----------------

_ALL_SIGNAL_KEYS = [
    "rel_vol_500x", "rel_vol_100x",
    "rel_vol_50x", "rel_vol_25x", "rel_vol_10x", "rel_vol_5x",
    "daily_sweet_20_40", "daily_ok_10_20", "daily_ok_40_100",
    "sideways_chop", "momentum_continuation", "yesterday_green",
    "shares_lt10m", "shares_lt30m", "shares_lt100m",
    "no_news_bonus", "high_cash_bonus",
    "institution_moderate", "institution_strong", "sector_biotech_bonus",
    "rsi_momentum_bonus", "macd_positive_bonus", "bb_upper_breakout",
    "consecutive_green_bonus", "low_float_ratio_bonus",
    "first_hour_vol_20m",
]


def get_per_signal_stats(modes=None) -> dict:
    """
    For each of the 22 signal keys, compute performance stats across scans
    where that signal fired AND outcomes are known (next_day_return IS NOT NULL).
    Optional modes filter restricts to specific scan modes (e.g. ['fivemin']).
    Cached 3600s; call analytics_cache_clear('per_signal_stats') to invalidate.
    """
    _ck = f"per_signal_stats:{modes}"
    _cached, _hit = _cache_get(_ck)
    if _hit:
        return _cached
    conn = _connect()
    cursor = conn.cursor()
    mc, mp = _mode_clause(modes)

    def _stats(rows):
        if not rows:
            return {"count": 0, "win_rate": 0.0, "hit_20pct": 0.0, "avg_return": 0.0}
        nd_vals = [nd for nd, d20 in rows if nd is not None]
        wins    = [nd for nd in nd_vals if nd > 0]
        hits    = [d20 for nd, d20 in rows if d20 is not None]
        return {
            "count":      len(rows),
            "win_rate":   round(len(wins) / len(nd_vals) * 100, 1) if nd_vals else 0.0,
            "hit_20pct":  round(len(hits) / len(rows) * 100, 1),
            "avg_return": round(sum(nd_vals) / len(nd_vals), 2) if nd_vals else 0.0,
        }

    # Deduped base: one row per symbol per day, daily modes only, confirmed outcomes.
    # MAX(next_day_return) keeps win_rate metric; MIN(days_to_20pct) is the hit metric.
    _excl = "mode NOT IN ('fivemin','fivemin_bt','candidate_fivemin','standard','strict')"
    _outcome = "(days_to_20pct IS NOT NULL OR (julianday('now') - julianday(timestamp)) >= 14)"
    _mode_extra = f" AND mode IN ({','.join('?'*len(modes))})" if modes else ""
    _dedup_cte = f"""
        WITH deduped AS (
          SELECT symbol, DATE(timestamp) as sd,
                 MIN(days_to_20pct)   AS best_d20,
                 MAX(next_day_return) AS best_ndr,
                 MAX(signals_json)    AS signals_json
          FROM scans
          WHERE {_excl} AND {_outcome}
            AND signals_json IS NOT NULL AND signals_json != '{{}}'
            {_mode_extra}
          GROUP BY symbol, DATE(timestamp)
        )
    """
    _mp = list(modes) if modes else []

    cursor.execute(
        _dedup_cte + "SELECT best_ndr, best_d20 FROM deduped", _mp
    )
    baseline_rows = cursor.fetchall()

    if not baseline_rows:
        conn.close()
        return {"baseline": None, "signals": []}

    baseline = _stats(baseline_rows)

    signal_results = []
    for key in _ALL_SIGNAL_KEYS:
        cursor.execute(
            _dedup_cte +
            "SELECT best_ndr, best_d20 FROM deduped "
            "WHERE json_extract(signals_json, '$.' || ?) = 1",
            _mp + [key]
        )
        rows = cursor.fetchall()
        s = _stats(rows)
        s["key"]             = key
        s["vs_baseline_hit"] = round(s["hit_20pct"] - baseline["hit_20pct"], 1)
        signal_results.append(s)

    conn.close()
    signal_results.sort(key=lambda x: x["hit_20pct"], reverse=True)
    _result = {"baseline": baseline, "signals": signal_results}
    _cache_set(_ck, _result, 3600)
    return _result


# ---------------- TRADE SIGNAL AUTOPSY ----------------

def get_trade_signal_autopsy() -> dict:
    """
    For every AI TRADE call with a known outcome (next_day_return IS NOT NULL),
    compute per-signal and signal-combination win/loss rates.

    Returns:
      {
        "total_trades": int,   # AI TRADE calls with outcomes
        "wins": int,
        "losses": int,
        "overall_win_rate": float,
        "individual": [         # per-signal breakdown, sorted by win_rate asc (worst first)
          {"key": str, "fires": int, "wins": int, "losses": int, "win_rate": float,
           "vs_overall": float}  # pp delta vs overall TRADE win rate
          ...
        ],
        "combos": [             # top co-firing combos on LOSING trades (n>=3), sorted by loss_count desc
          {"signals": [str, ...], "loss_count": int, "win_count": int, "win_rate": float}
          ...
        ]
      }
    Cached 3600s.
    """
    _ck = "trade_signal_autopsy"
    _cached, _hit = _cache_get(_ck)
    if _hit:
        return _cached

    conn = _connect()
    cursor = conn.cursor()

    # Deduped TRADE calls: one per symbol per day, daily modes, confirmed outcomes.
    # A symbol+day counts as a TRADE call if any scan that day called TRADE.
    cursor.execute("""
        SELECT MIN(days_to_20pct) as best_d20, MAX(signals_json) as signals_json
        FROM scans
        WHERE mode NOT IN ('fivemin','fivemin_bt','candidate_fivemin','standard','strict')
          AND signals_json IS NOT NULL AND signals_json != '{}'
          AND ai_trade_rec LIKE '%"decision": "TRADE"%'
          AND (days_to_20pct IS NOT NULL
               OR (julianday('now') - julianday(timestamp)) >= 14)
        GROUP BY symbol, DATE(timestamp)
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        result = {"total_trades": 0, "wins": 0, "losses": 0,
                  "overall_win_rate": 0.0, "individual": [], "combos": []}
        _cache_set(_ck, result, 3600)
        return result

    total = len(rows)
    wins_total = sum(1 for d20, _ in rows if d20 is not None)
    losses_total = total - wins_total
    overall_wr = round(wins_total / total * 100, 1) if total else 0.0

    # --- Individual signal breakdown ---
    signal_stats: dict[str, dict] = {}
    for d20, sj in rows:
        try:
            fired = json.loads(sj) if sj else {}
        except (json.JSONDecodeError, TypeError):
            fired = {}
        is_win = d20 is not None
        for key in fired:
            if key not in signal_stats:
                signal_stats[key] = {"wins": 0, "losses": 0}
            if is_win:
                signal_stats[key]["wins"] += 1
            else:
                signal_stats[key]["losses"] += 1

    individual = []
    for key, s in signal_stats.items():
        fires = s["wins"] + s["losses"]
        if fires < 3:
            continue
        wr = round(s["wins"] / fires * 100, 1)
        individual.append({
            "key": key,
            "fires": fires,
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": wr,
            "vs_overall": round(wr - overall_wr, 1),
        })
    individual.sort(key=lambda x: x["win_rate"])  # worst first

    # --- Signal combination breakdown on LOSING trades ---
    # Find pairs/triples that co-fire frequently on losses
    from itertools import combinations as _combos
    combo_counts: dict[tuple, dict] = {}
    for d20, sj in rows:
        try:
            fired = json.loads(sj) if sj else {}
        except (json.JSONDecodeError, TypeError):
            fired = {}
        is_win = d20 is not None
        keys = sorted(fired.keys())
        # Only pairs (triples get noisy with small n)
        for pair in _combos(keys, 2):
            if pair not in combo_counts:
                combo_counts[pair] = {"wins": 0, "losses": 0}
            if is_win:
                combo_counts[pair]["wins"] += 1
            else:
                combo_counts[pair]["losses"] += 1

    combos = []
    for pair, s in combo_counts.items():
        total_c = s["wins"] + s["losses"]
        if total_c < 3:
            continue
        wr = round(s["wins"] / total_c * 100, 1)
        combos.append({
            "signals": list(pair),
            "total": total_c,
            "loss_count": s["losses"],
            "win_count": s["wins"],
            "win_rate": wr,
        })
    # Sort: lowest win rate first (worst combos), then by total fires desc for ties
    combos.sort(key=lambda x: (x["win_rate"], -x["total"]))

    result = {
        "total_trades": total,
        "wins": wins_total,
        "losses": losses_total,
        "overall_win_rate": overall_wr,
        "individual": individual,
        "combos": combos[:20],  # top 20 worst combos
    }
    _cache_set(_ck, result, 3600)
    return result


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
    _write_feedback_backup()  # keep backup file in sync after every save


def _write_feedback_backup():
    """
    Write all feedback entries to a JSON file alongside the database.
    Called automatically after every save/import so the backup stays current.
    Silently skips on any I/O error (never blocks the main request).
    """
    try:
        entries = get_all_feedback()
        with open(FEEDBACK_BACKUP_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def import_feedback_from_backup(entries: list) -> int:
    """
    Insert feedback entries from a backup list (e.g. an uploaded JSON file).
    Deduplicates by (created_at, symbol) to avoid double-inserts.
    Returns the number of new entries inserted.
    """
    if not entries:
        return 0
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Build dedup set from existing rows
    cursor.execute("SELECT created_at, symbol FROM feedback")
    existing = {(r[0], r[1]) for r in cursor.fetchall()}
    inserted = 0
    for entry in entries:
        key = (entry.get("created_at", ""), entry.get("symbol", ""))
        if key in existing:
            continue
        cursor.execute("""
            INSERT INTO feedback (created_at, symbol, user_text, chart_analysis, outcome_tag)
            VALUES (?, ?, ?, ?, ?)
        """, (
            entry.get("created_at") or datetime.utcnow().isoformat(),
            entry.get("symbol", ""),
            entry.get("user_text", ""),
            entry.get("chart_analysis", ""),
            entry.get("outcome_tag") or None,
        ))
        existing.add(key)
        inserted += 1
    conn.commit()
    conn.close()
    if inserted > 0:
        _write_feedback_backup()  # update backup to include restored entries
    return inserted


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
        SELECT id, created_at, symbol, user_text, chart_analysis,
               COALESCE(outcome_tag, '')
        FROM feedback
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": r[0], "created_at": r[1], "symbol": r[2],
         "user_text": r[3], "chart_analysis": r[4], "outcome_tag": r[5]}
        for r in rows
    ]


def get_all_feedback():
    """Returns all feedback entries for hypothesis synthesis."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, created_at, symbol, user_text, chart_analysis,
               COALESCE(outcome_tag, '')
        FROM feedback
        ORDER BY id ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": r[0], "created_at": r[1], "symbol": r[2],
         "user_text": r[3], "chart_analysis": r[4], "outcome_tag": r[5]}
        for r in rows
    ]


# ---------------- HYPOTHESIS STORAGE ----------------
def save_hypothesis(content: str, feedback_count: int):
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at)
        VALUES ('hypothesis', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                       updated_at = excluded.updated_at
    """, (content, now))
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at)
        VALUES ('hypothesis_feedback_count', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                       updated_at = excluded.updated_at
    """, (str(feedback_count), now))
    # Extract and cache the Agent Context block separately (used by active-rules hypothesis builder)
    agent_ctx = ""
    if "## Agent Context" in content:
        agent_ctx = content.split("## Agent Context")[1].strip()
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at)
        VALUES ('hypothesis_agent_context', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                       updated_at = excluded.updated_at
    """, (agent_ctx, now))
    conn.commit()
    conn.close()


def save_autoai_hypothesis_blob(content: str, feedback_count: int):
    """
    Saves Auto AI's synthesized hypothesis text under its own settings key
    ('autoai_hypothesis') so it never overwrites the admin-curated Complex+AI blob.
    """
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at)
        VALUES ('autoai_hypothesis', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                       updated_at = excluded.updated_at
    """, (content, now))
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at)
        VALUES ('autoai_hypothesis_feedback_count', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                       updated_at = excluded.updated_at
    """, (str(feedback_count), now))
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


# ---------------- HYPOTHESIS RULES ----------------
def save_hypothesis_rules(rules: list):
    """
    Insert parsed hypothesis rules from a new synthesis run.
    All new rules start as 'pending' (not yet reviewed).
    Assigns a generation number = current max + 1.
    """
    if not rules:
        return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(generation), 0) FROM hypothesis_rules")
    next_gen = cursor.fetchone()[0] + 1
    now = datetime.utcnow().isoformat()
    for r in rules:
        cursor.execute("""
            INSERT INTO hypothesis_rules (rule_text, source, status, generation, created_at)
            VALUES (?, ?, 'pending', ?, ?)
        """, (r["text"], r.get("source", ""), next_gen, now))
    conn.commit()
    conn.close()


def get_hypothesis_rules() -> list:
    """
    Returns all hypothesis rules sorted by projected impact (bundles with highest
    hit_20pct delta first), then pending, then by generation.
    Includes live win/loss stats and projection_json for bundle rules.
    Uses a single pre-aggregation pass instead of N+1 per-rule queries.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, rule_text, source, status, generation, created_at,
               confidence_score, auto_applied, projection_json
        FROM hypothesis_rules
        ORDER BY generation DESC, id ASC
    """)
    rows = cursor.fetchall()

    # Pre-aggregate rule stats in one query: fetch all labeled scans with rule IDs,
    # then count in Python — avoids N separate LIKE queries (one per rule).
    cursor.execute("""
        SELECT active_rule_ids,
               days_to_20pct IS NOT NULL as won
        FROM scans
        WHERE next_day_return IS NOT NULL
          AND active_rule_ids IS NOT NULL
          AND active_rule_ids != ''
    """)
    rule_totals: dict = {}   # {rule_id_int: {"total": int, "wins": int}}
    for (rule_ids_str, won) in cursor.fetchall():
        try:
            # active_rule_ids stored as comma-separated ints or JSON array
            if rule_ids_str.startswith("["):
                ids = json.loads(rule_ids_str)
            else:
                ids = [int(x.strip()) for x in rule_ids_str.split(",") if x.strip().isdigit()]
        except Exception:
            continue
        for rid_int in ids:
            entry = rule_totals.setdefault(rid_int, {"total": 0, "wins": 0})
            entry["total"] += 1
            if won:
                entry["wins"] += 1

    conn.close()

    results = []
    for (rid, rule_text, source, status, generation, created_at,
         confidence_score, auto_applied, projection_json_str) in rows:
        stat = rule_totals.get(rid, {"total": 0, "wins": 0})
        total    = stat["total"]
        wins     = stat["wins"]
        win_rate = round(wins / total * 100, 1) if total > 0 else None

        projection = None
        proj_delta_hit = 0.0
        if projection_json_str:
            try:
                projection = json.loads(projection_json_str)
                proj_delta_hit = projection.get("delta", {}).get("hit_20pct", 0.0) or 0.0
            except Exception:
                pass

        results.append({
            "id":               rid,
            "rule_text":        rule_text,
            "source":           source or "",
            "status":           status,
            "generation":       generation,
            "created_at":       (created_at or "")[:10],
            "trades":           total,
            "wins":             wins,
            "win_rate":         win_rate,
            "confidence_score": confidence_score,
            "auto_applied":     auto_applied or 0,
            "projection":       projection,
            "_proj_delta_hit":  proj_delta_hit,
        })

    # Sort: bundles with projection by delta desc, then rest by status (pending first) + id
    def sort_key(r):
        has_proj = r["projection"] is not None
        return (0 if has_proj else 1, -r["_proj_delta_hit"], r["id"])

    results.sort(key=sort_key)
    return results


def refresh_bundle_projections() -> int:
    """
    Recompute projection_json for all bundle rules (those that already have
    projection_json set). Called after signals backfill so projections reflect
    real per-signal data instead of the empty-DB values stored at creation time.
    Returns the number of rules updated.
    """
    per_sig = get_per_signal_stats()
    baseline = (per_sig or {}).get("baseline") or {}
    if baseline.get("count", 0) < 5:
        return 0  # not enough data yet

    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, rule_text, projection_json
        FROM hypothesis_rules
        WHERE projection_json IS NOT NULL
    """)
    rows = cursor.fetchall()
    updated = 0
    for (rid, rule_text, _) in rows:
        # Parse signal keys from the first line: "[BUNDLE — label] key1 + key2 + ..."
        try:
            first_line = rule_text.split("\n")[0]
            after_bracket = first_line.split("] ", 1)[1] if "] " in first_line else ""
            signal_keys = [k.strip() for k in after_bracket.split(" + ") if k.strip()]
        except Exception:
            signal_keys = []

        if not signal_keys:
            # Fallback: extract from "Signals adjusted: key=val, ..." line
            try:
                for line in rule_text.split("\n"):
                    if line.startswith("Signals adjusted:"):
                        pairs = line.split(":", 1)[1].strip().split(", ")
                        signal_keys = [p.split("=")[0].strip() for p in pairs if "=" in p]
                        break
            except Exception:
                pass

        if not signal_keys:
            continue

        projection = project_bundle_impact(signal_keys, {"per_signal_stats": per_sig})
        cursor.execute(
            "UPDATE hypothesis_rules SET projection_json = ? WHERE id = ?",
            (json.dumps(projection), rid)
        )
        updated += 1

    conn.commit()
    conn.close()
    return updated


def get_pending_rule_count() -> int:
    """Returns number of rules awaiting admin review."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM hypothesis_rules WHERE status = 'pending'")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0


def project_bundle_impact(signal_keys: list, opt_data: dict | None = None) -> dict:
    """
    Given a list of signal keys (e.g. ["rel_vol_50x", "shares_lt10m"]), compute
    the projected impact of prioritising those signals, validated against the
    per-signal backtest data.

    Returns:
      {
        "baseline":  {hit_20pct, win_rate, avg_return, avg_days_to_20pct, count},
        "projected": {hit_20pct, win_rate, avg_return, avg_days_to_20pct},
        "delta":     {hit_20pct, win_rate, avg_return, avg_days_to_20pct},
        "signals":   [{key, count, hit_20pct, win_rate, avg_return, avg_days_to_20pct, vs_baseline_hit},...],
        "coverage":  int,   # number of signals with enough data
        "validated": bool,  # True if >=1 signal had sufficient backtest data
      }
    """
    if opt_data is None:
        opt_data = get_optimization_data() or {}

    per_sig = opt_data.get("per_signal_stats") or get_per_signal_stats()
    baseline = (per_sig or {}).get("baseline") or {}
    all_signals = (per_sig or {}).get("signals") or []

    sig_map = {s["key"]: s for s in all_signals if s.get("count", 0) >= 5}
    baseline_hit  = baseline.get("hit_20pct", 0) or 0
    baseline_win  = baseline.get("win_rate",  0) or 0
    baseline_ret  = baseline.get("avg_return", 0) or 0
    baseline_days = baseline.get("avg_days_to_20pct") or None
    baseline_cnt  = baseline.get("count", 0) or 0

    matched = []
    for key in signal_keys:
        if key in sig_map:
            matched.append(sig_map[key])

    if not matched:
        return {
            "baseline":  {"hit_20pct": baseline_hit, "win_rate": baseline_win,
                          "avg_return": baseline_ret, "avg_days_to_20pct": baseline_days,
                          "count": baseline_cnt},
            "projected": {"hit_20pct": baseline_hit, "win_rate": baseline_win,
                          "avg_return": baseline_ret, "avg_days_to_20pct": baseline_days},
            "delta":     {"hit_20pct": 0, "win_rate": 0, "avg_return": 0, "avg_days_to_20pct": 0},
            "signals":   [],
            "coverage":  0,
            "validated": False,
        }

    # Weighted average by fire count
    total_weight = sum(s["count"] for s in matched)
    def wavg(field):
        vals = [(s[field], s["count"]) for s in matched if s.get(field) is not None]
        if not vals:
            return None
        return sum(v * w for v, w in vals) / sum(w for _, w in vals)

    proj_hit  = round(wavg("hit_20pct") or baseline_hit, 1)
    proj_win  = round(wavg("win_rate")  or baseline_win,  1)
    proj_ret  = round(wavg("avg_return") or baseline_ret,  2)
    proj_days_raw = wavg("avg_days_to_20pct")
    proj_days = round(proj_days_raw, 1) if proj_days_raw is not None else baseline_days

    delta_days = None
    if proj_days is not None and baseline_days is not None:
        delta_days = round(proj_days - baseline_days, 1)

    return {
        "baseline":  {"hit_20pct": baseline_hit, "win_rate": baseline_win,
                      "avg_return": baseline_ret, "avg_days_to_20pct": baseline_days,
                      "count": baseline_cnt},
        "projected": {"hit_20pct": proj_hit, "win_rate": proj_win,
                      "avg_return": proj_ret, "avg_days_to_20pct": proj_days},
        "delta":     {
            "hit_20pct":       round(proj_hit - baseline_hit, 1),
            "win_rate":        round(proj_win - baseline_win,  1),
            "avg_return":      round(proj_ret - baseline_ret,  2),
            "avg_days_to_20pct": delta_days,
        },
        "signals":   [{"key": s["key"], "count": s["count"],
                       "hit_20pct": s["hit_20pct"], "win_rate": s["win_rate"],
                       "avg_return": s["avg_return"],
                       "avg_days_to_20pct": s.get("avg_days_to_20pct"),
                       "vs_baseline_hit": s["vs_baseline_hit"]} for s in matched],
        "coverage":  len(matched),
        "validated": len(matched) >= 1,
    }


def save_bundle_as_rule(goal: str, weights: dict, rationale: str,
                        summary: str, opt_data: dict | None = None) -> int:
    """
    Save an AI-suggested weight bundle as a hypothesis rule with AI-validated
    projected impact metrics. The rule is created as 'active' (bundle already applied).
    Returns the new rule id.
    """
    # Derive signal keys from weight keys (they map 1:1 to per-signal keys)
    signal_keys = list(weights.keys())
    projection = project_bundle_impact(signal_keys, opt_data)

    goal_labels = {
        "combined": "Win Rate + Speed + Upside",
        "win_rate": "Win Rate",
        "speed":    "Speed to Target",
        "upside":   "Max Upside",
    }
    goal_label = goal_labels.get(goal, goal.title())
    signals_str = " + ".join(signal_keys)

    rule_text = (
        f"[BUNDLE — {goal_label}] {signals_str}\n"
        f"{rationale}\n"
        f"Signals adjusted: {', '.join(f'{k}={v}' for k, v in weights.items())}"
    )

    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO hypothesis_rules
            (rule_text, source, status, generation, created_at,
             confidence_score, auto_applied, projection_json)
        VALUES (?, 'ai_chat_bundle', 'active', 0, ?, 85, 1, ?)
    """, (rule_text, now, json.dumps(projection)))
    rule_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rule_id


def update_rule_status(rule_id: int, status: str) -> bool:
    """Set a rule's status to 'active', 'rejected', or 'pending'. Returns True on success."""
    if status not in ("active", "rejected", "pending"):
        return False
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE hypothesis_rules SET status = ? WHERE id = ?", (status, rule_id))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return ok


def get_active_rule_ids() -> list:
    """Returns list of active rule IDs for tagging scan records."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM hypothesis_rules WHERE status = 'active'")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def save_scan_active_rules(scan_id: int, rule_ids: list):
    """Tag a scan with the IDs of hypothesis rules that were active at enrichment time."""
    if not rule_ids:
        return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE scans SET active_rule_ids = ? WHERE id = ?",
        (json.dumps(rule_ids), scan_id)
    )
    conn.commit()
    conn.close()


def get_active_hypothesis_text(mode: str = None) -> str | None:
    """
    Returns the hypothesis text that should be fed into AI calls.

    mode="autoai"  → includes all active rules (admin-approved + Auto AI auto-applied)
    mode=anything else (default) → includes only admin-approved rules (auto_applied=0),
                                   keeping Complex+AI fully isolated from Auto AI rules.

    Behaviour:
    - If no qualifying rules exist OR all are still pending: fall back to the full
      synthesized blob (preserves existing behaviour).
    - If at least one rule has been reviewed: build a compact hypothesis from only
      the active rules + cached Agent Context.
    """
    autoai_mode = (mode == "autoai")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if autoai_mode:
        cursor.execute("SELECT COUNT(*) FROM hypothesis_rules")
    else:
        cursor.execute("SELECT COUNT(*) FROM hypothesis_rules WHERE auto_applied = 0")
    total = cursor.fetchone()[0]

    if total == 0:
        # Auto AI falls back to its own blob first, then admin blob
        fallback_key = 'autoai_hypothesis' if autoai_mode else 'hypothesis'
        cursor.execute("SELECT value FROM settings WHERE key = ?", (fallback_key,))
        row = cursor.fetchone()
        if not row and autoai_mode:
            cursor.execute("SELECT value FROM settings WHERE key = 'hypothesis'")
            row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    if autoai_mode:
        cursor.execute("SELECT COUNT(*) FROM hypothesis_rules WHERE status != 'pending'")
    else:
        cursor.execute("SELECT COUNT(*) FROM hypothesis_rules WHERE status != 'pending' AND auto_applied = 0")
    reviewed = cursor.fetchone()[0]

    if reviewed == 0:
        fallback_key = 'autoai_hypothesis' if autoai_mode else 'hypothesis'
        cursor.execute("SELECT value FROM settings WHERE key = ?", (fallback_key,))
        row = cursor.fetchone()
        if not row and autoai_mode:
            cursor.execute("SELECT value FROM settings WHERE key = 'hypothesis'")
            row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    if autoai_mode:
        cursor.execute(
            "SELECT id, rule_text FROM hypothesis_rules WHERE status = 'active' ORDER BY id ASC"
        )
    else:
        cursor.execute(
            "SELECT id, rule_text FROM hypothesis_rules WHERE status = 'active' AND auto_applied = 0 ORDER BY id ASC"
        )
    active = cursor.fetchall()

    if not active:
        conn.close()
        return None

    lines = ["## Active Hypothesis Rules\n"]
    for i, (_, rule_text) in enumerate(active, 1):
        lines.append(f"{i}. {rule_text}\n")

    cursor.execute("SELECT value FROM settings WHERE key = 'hypothesis_agent_context'")
    ctx_row = cursor.fetchone()
    if ctx_row and ctx_row[0]:
        lines.append(f"\n## Agent Context\n{ctx_row[0]}")

    conn.close()
    return "\n".join(lines)


# ---------------- COMPLEX + AI WEIGHT STORAGE ----------------
def rescore_historical_from_signals(weights: dict) -> int:
    """
    Recompute the score column for every row that has signals_json using the
    provided weights dict. Returns the number of rows updated.

    Uses only the signals that fired (signals_json keys with value=True/1).
    Applies the same max_score normalisation as score_stock_squeeze().
    Rows without signals_json are left unchanged.
    """
    # Compute max_score denominator from current weights (mirrors scoring_engine.py)
    w = weights
    max_score = (
        max(w.get("rel_vol_500x", 40), w.get("rel_vol_100x", 35),
            w.get("rel_vol_50x", 30), w.get("rel_vol_25x", 22),
            w.get("rel_vol_10x", 15), w.get("rel_vol_5x", 7), 0) +
        max(w.get("daily_sweet_20_40", 10), w.get("daily_ok_10_20", -5),
            w.get("daily_ok_40_100", 7), 0) +
        w.get("sideways_chop", 8) +
        w.get("yesterday_green", 9) +
        max(w.get("shares_lt10m", 18), w.get("shares_lt30m", 28),
            w.get("shares_lt100m", 8), 0) +
        w.get("high_cash_bonus", 5) +
        max(w.get("institution_strong", 5), w.get("institution_moderate", 2), 0) +
        w.get("sector_biotech_bonus", 5) +
        w.get("no_news_bonus", 5) +
        w.get("rsi_momentum_bonus", 0) +
        w.get("macd_positive_bonus", 0) +
        w.get("bb_upper_breakout", 0) +
        w.get("consecutive_green_bonus", 0) +
        w.get("low_float_ratio_bonus", 0)
    )
    if max_score <= 0:
        max_score = 100

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT id, signals_json FROM scans WHERE signals_json IS NOT NULL AND signals_json != '{}'"
    ).fetchall()

    updated = 0
    for row_id, sj in rows:
        try:
            fired = json.loads(sj)
            raw = sum(w.get(k, 0) for k, v in fired.items() if v)
            new_score = max(0, min(100, round(raw / max_score * 100)))
            cursor.execute("UPDATE scans SET score=? WHERE id=?", (new_score, row_id))
            updated += 1
        except Exception:
            continue

    conn.commit()
    conn.close()
    print(f"RESCORE: updated {updated} rows with new weights")
    return updated


def save_squeeze_weights(weights: dict, rationale: str = "",
                         suggestions: list = None, summary: str = "",
                         source: str = "manual", goal: str = ""):
    """Persist AI-optimized squeeze weights to the settings table and create a version snapshot.
    Also rescores all historical rows that have signals_json so per-signal and bucket
    stats always reflect the current model rather than a mix of weight versions."""
    save_weight_version("squeeze", weights, summary=summary, rationale=rationale,
                        source=source, goal=goal)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    for key, value in [
        ("squeeze_weights",             json.dumps(weights)),
        ("squeeze_weights_rationale",   rationale),
        ("squeeze_weights_suggestions", json.dumps(suggestions or [])),
        ("squeeze_weights_summary",     summary),
        ("squeeze_weights_last_changed", now),
    ]:
        cursor.execute("""
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
        """, (key, value, now))
    conn.commit()
    conn.close()
    # Rescore all historical rows so stats are always current-model-consistent
    rescore_historical_from_signals(weights)


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
    # Strip any non-serializable fields (e.g. _df pandas DataFrames) before caching
    clean = [{k: v for k, v in r.items() if k != "_df"} for r in results]
    value = json.dumps({"results": clean, "summary": summary, "cached_at": now})
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


def save_weight_version(model_type: str, weights: dict, summary: str = "",
                        rationale: str = "", source: str = "manual",
                        goal: str = "") -> int:
    """Insert a versioned weight snapshot and update the active version setting.
    Returns the new version_id."""
    now = datetime.utcnow().isoformat()
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO weight_versions (model_type, weights, summary, rationale, source, goal, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (model_type, json.dumps(weights), summary or "", rationale or "",
         source or "manual", goal or "", now)
    )
    version_id = cursor.lastrowid
    # Track the active version_id in settings so save_scan() can stamp it
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (f"current_{model_type}_version_id", str(version_id), now))
    conn.commit()
    conn.close()
    return version_id


def get_current_version_id(model_type: str) -> int | None:
    """Return the current active weight version ID for a model type, or None."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?",
                   (f"current_{model_type}_version_id",))
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        try:
            return int(row[0])
        except (ValueError, TypeError):
            return None
    return None


def get_version_performance_stats(model_type: str = None) -> list:
    """
    Returns performance metrics per weight version, ordered newest first.
    Each entry has the version metadata plus win_rate, hit_20pct, avg_return,
    avg_days_to_20pct, and scan_count computed from scans linked to that version.
    Only versions with at least 1 scan with known outcomes are included.
    """
    conn = _connect()
    cursor = conn.cursor()

    type_filter = "AND wv.model_type = ?" if model_type else ""
    params = [model_type] if model_type else []

    cursor.execute(f"""
        SELECT
            wv.id,
            wv.model_type,
            wv.summary,
            wv.source,
            wv.goal,
            wv.created_at,
            COUNT(s.id)                                                    AS scan_count,
            AVG(s.next_day_return)                                         AS avg_return,
            100.0 * SUM(CASE WHEN s.next_day_return > 0 THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(s.id), 0)                                 AS win_rate,
            100.0 * SUM(CASE WHEN s.days_to_20pct IS NOT NULL THEN 1 ELSE 0 END)
                  / NULLIF(COUNT(s.id), 0)                                 AS hit_20pct,
            AVG(CASE WHEN s.days_to_20pct IS NOT NULL THEN s.days_to_20pct END) AS avg_days_to_20pct
        FROM weight_versions wv
        LEFT JOIN scans s ON s.weights_version_id = wv.id
                          AND s.next_day_return IS NOT NULL
        WHERE 1=1 {type_filter}
        GROUP BY wv.id
        ORDER BY wv.id DESC
        LIMIT 30
    """, params)

    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id":               r[0],
            "model_type":       r[1],
            "summary":          r[2] or "",
            "source":           r[3] or "manual",
            "goal":             r[4] or "",
            "created_at":       (r[5] or "")[:16].replace("T", " "),
            "scan_count":       r[6] or 0,
            "avg_return":       round(r[7], 2) if r[7] is not None else None,
            "win_rate":         round(r[8], 1) if r[8] is not None else None,
            "hit_20pct":        round(r[9], 1) if r[9] is not None else None,
            "avg_days_to_20pct": round(r[10], 1) if r[10] is not None else None,
        })
    return results


def get_weight_changelog(limit: int = 20) -> list:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, updated_at, summary, rationale, weights FROM weight_changelog ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id":         r[0],
            "updated_at": r[1],
            "summary":    r[2],
            "rationale":  r[3],
            "weights":    json.loads(r[4]) if r[4] else {},
        }
        for r in rows
    ]


# ---------------- WATCHLIST ----------------
def add_to_watchlist(symbol: str, price: float = None):
    """Manually add a symbol to the watchlist (from the trades page form)."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    today = datetime.utcnow().date().isoformat()
    cursor.execute(
        """INSERT OR IGNORE INTO watchlist (symbol, added_at, price_at_add, alerted, date)
           VALUES (?, ?, ?, 0, ?)""",
        (symbol.upper().strip(), now, price, today)
    )
    conn.commit()
    conn.close()


def add_near_miss_to_watchlist(symbol: str, score: int, price: float = None):
    """
    Upsert a near-miss stock (score 40-74) into the watchlist for intraday re-checking.
    - Same day: updates the score (in case it improved) but keeps the alerted flag.
    - New day: resets alerted flag and updates all fields.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now   = datetime.utcnow().isoformat()
    today = datetime.utcnow().date().isoformat()

    cursor.execute("SELECT date, alerted FROM watchlist WHERE symbol = ?",
                   (symbol.upper().strip(),))
    row = cursor.fetchone()

    if row:
        existing_date, existing_alerted = row
        if existing_date == today:
            # Same day — update score only; preserve alerted flag
            cursor.execute(
                "UPDATE watchlist SET score = ?, last_checked = ? WHERE symbol = ?",
                (score, now, symbol.upper().strip())
            )
        else:
            # New day — reset alerted, update everything
            cursor.execute(
                """UPDATE watchlist SET score = ?, price_at_add = ?, added_at = ?,
                   date = ?, alerted = 0, last_checked = ? WHERE symbol = ?""",
                (score, price, now, today, now, symbol.upper().strip())
            )
    else:
        cursor.execute(
            """INSERT INTO watchlist (symbol, added_at, price_at_add, score, alerted, date, last_checked)
               VALUES (?, ?, ?, ?, 0, ?, ?)""",
            (symbol.upper().strip(), now, price, score, today, now)
        )

    conn.commit()
    conn.close()


def get_active_watchlist(today_only: bool = True) -> list:
    """
    Returns unalerted watchlist items.
    today_only=True (default): only today's near-misses (for intraday re-checks).
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.utcnow().date().isoformat()

    if today_only:
        cursor.execute(
            """SELECT symbol, added_at, price_at_add, score, alerted, date
               FROM watchlist WHERE alerted = 0 AND date = ? ORDER BY score DESC""",
            (today,)
        )
    else:
        cursor.execute(
            """SELECT symbol, added_at, price_at_add, score, alerted, date
               FROM watchlist WHERE alerted = 0 ORDER BY score DESC"""
        )

    rows = cursor.fetchall()
    conn.close()
    return [
        {"symbol": r[0], "added_at": r[1], "price_at_add": r[2],
         "score": r[3], "alerted": r[4], "date": r[5]}
        for r in rows
    ]


def mark_watchlist_alerted(symbol: str):
    """Mark a watchlist symbol as alerted so we don't send duplicate alerts."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE watchlist SET alerted = 1 WHERE symbol = ?",
                   (symbol.upper().strip(),))
    conn.commit()
    conn.close()


def clear_old_watchlist(days_old: int = 2):
    """Remove watchlist entries older than N days to keep the table clean."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cutoff = (datetime.utcnow() - timedelta(days=days_old)).date().isoformat()
    cursor.execute("DELETE FROM watchlist WHERE date < ?", (cutoff,))
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
    cursor.execute(
        """SELECT symbol, added_at, price_at_add, COALESCE(score, 0), COALESCE(alerted, 0), date
           FROM watchlist ORDER BY added_at DESC"""
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"symbol": r[0], "added_at": r[1], "price_at_add": r[2],
         "score": r[3], "alerted": r[4], "date": r[5]}
        for r in rows
    ]


# ---------------- HISTORICAL BACKFILL ----------------
def save_historical_scans(examples: list, clear_first: bool = True) -> int:
    """
    Insert labeled historical scan examples (mode='historical').
    Clears existing historical rows first (clear_first=True) to keep the table idempotent.
    Pass clear_first=False for subsequent batch calls.
    """
    if not examples:
        return 0
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if clear_first:
        cursor.execute("DELETE FROM scans WHERE mode = 'historical'")
    for ex in examples:
        cursor.execute("""
            INSERT INTO scans (
                timestamp, symbol, score, recommendation, mode,
                relative_volume, today_return, shares_outstanding,
                news_recent, next_day_return, three_day_return, days_to_20pct,
                range_10d, yesterday_green, float_shares
            ) VALUES (?, ?, ?, ?, 'historical', ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
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
            ex.get("range_10d"),
            ex.get("yesterday_green"),
            ex.get("float_shares"),
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
        # "win" = intraday HIGH touched +20% take-profit within 10 trading days
        hits_20    = [(nd, d) for nd, d in items if d is not None]
        any_pos    = sum(1 for nd in returns if nd > 0)
        valid_days = [d for nd, d in hits_20]
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
    _ck = "risk_metrics"
    _cached, _hit = _cache_get(_ck)
    if _hit:
        return _cached
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

    _result = {
        "sharpe":        sharpe,
        "max_drawdown":  round(max_dd, 2),
        "win_rate":      win_rate,
        "total_closed":  len(rows),
    }
    _cache_set(_ck, _result, 3600)
    return _result


# ---------------- MODEL VALIDATION ----------------
def get_model_validation_stats() -> dict | None:
    """
    Validate AI model predictions against actual outcomes.
    Groups live labeled scans by AI decision, confidence, and score bucket.
    Returns None if fewer than 10 labeled live scans with AI calls exist.
    Cached 3600s; invalidate via analytics_cache_clear('model_validation').
    """
    _ck = "model_validation"
    _cached, _hit = _cache_get(_ck)
    if _hit:
        return _cached
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT score, ai_trade_rec, next_day_return, days_to_20pct
        FROM scans
        WHERE mode != 'historical'
          AND next_day_return IS NOT NULL
          AND ai_trade_rec IS NOT NULL
        ORDER BY id DESC LIMIT 1000
    """)
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 10:
        return None

    by_decision   = {"TRADE": [], "NO_TRADE": []}
    by_confidence = {"HIGH": [], "MEDIUM": [], "LOW": []}
    by_score      = {"90-100": [], "75-89": [], "50-74": [], "<50": []}

    for score, ai_rec_json, nd, d20 in rows:
        item = (nd, d20)
        try:
            ai_rec     = json.loads(ai_rec_json)
            decision   = ai_rec.get("decision",   "NO_TRADE").upper()
            confidence = ai_rec.get("confidence", "LOW").upper()
        except Exception:
            continue

        if decision in by_decision:
            by_decision[decision].append(item)
        if confidence in by_confidence:
            by_confidence[confidence].append(item)

        if score >= 90:        by_score["90-100"].append(item)
        elif score >= 75:      by_score["75-89"].append(item)
        elif score >= 50:      by_score["50-74"].append(item)
        else:                  by_score["<50"].append(item)

    def _stats(items):
        if not items:
            return None
        returns  = [nd for nd, d in items]
        hits_20  = [(nd, d) for nd, d in items if d is not None]
        wins     = [r for r in returns if r > 0]
        valid_days = [d for nd, d in hits_20]
        avg_days = round(sum(valid_days) / len(valid_days), 1) if valid_days else None
        return {
            "count":       len(items),
            "hit_20pct":   round(len(hits_20)  / len(items) * 100, 1),
            "win_rate":    round(len(wins)      / len(items) * 100, 1),
            "avg_return":  round(sum(returns)   / len(returns), 2),
            "avg_days_to_20": avg_days,
        }

    _result = {
        "total":         len(rows),
        "by_decision":   {k: _stats(v) for k, v in by_decision.items()   if v},
        "by_confidence": {k: _stats(v) for k, v in by_confidence.items() if v},
        "by_score":      {k: _stats(v) for k, v in by_score.items()      if v},
    }
    _cache_set(_ck, _result, 3600)
    return _result


# ---------------- TELEGRAM RECIPIENTS ----------------
def get_telegram_recipients() -> list:
    """Returns all saved Telegram recipients as list of {chat_id, label, added_at}."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT chat_id, label, added_at FROM telegram_recipients ORDER BY added_at ASC")
    rows = cursor.fetchall()
    conn.close()
    return [{"chat_id": r[0], "label": r[1], "added_at": r[2]} for r in rows]


def add_telegram_recipient(chat_id: str, label: str = "") -> bool:
    """Adds a Telegram recipient. Returns True if added, False if already exists."""
    chat_id = chat_id.strip()
    if not chat_id:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO telegram_recipients (chat_id, label, added_at) VALUES (?, ?, ?)",
            (chat_id, label.strip(), datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def delete_telegram_recipient(chat_id: str):
    """Removes a Telegram recipient by chat_id."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM telegram_recipients WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()


def get_ai_decision_accuracy() -> dict | None:
    """
    Computes AI TRADE/NO_TRADE accuracy against resolved outcomes.
    Resolved = next_day_return IS NOT NULL (backfilled 2+ days after scan).
    Returns None if fewer than 5 resolved AI calls exist.
    Used to inject self-calibration context into future recommend_trade() calls.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ai_trade_rec, days_to_20pct
        FROM scans
        WHERE mode != 'historical'
          AND next_day_return IS NOT NULL
          AND ai_trade_rec IS NOT NULL
        ORDER BY id DESC LIMIT 500
    """)
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 5:
        return None

    trade_calls    = []
    no_trade_calls = []

    for ai_rec_json, d20 in rows:
        try:
            decision = json.loads(ai_rec_json).get("decision", "").upper()
        except Exception:
            continue
        hit = d20 is not None
        if decision == "TRADE":
            trade_calls.append(hit)
        elif decision == "NO_TRADE":
            no_trade_calls.append(hit)

    result = {"total_resolved": len(rows)}

    if trade_calls:
        hits = sum(trade_calls)
        result["trade"] = {
            "count":     len(trade_calls),
            "hit_20pct": round(hits / len(trade_calls) * 100, 1),
            "missed":    round((len(trade_calls) - hits) / len(trade_calls) * 100, 1),
        }
    if no_trade_calls:
        missed_moves = sum(no_trade_calls)
        result["no_trade"] = {
            "count":        len(no_trade_calls),
            "correct":      round((len(no_trade_calls) - missed_moves) / len(no_trade_calls) * 100, 1),
            "missed_moves": round(missed_moves / len(no_trade_calls) * 100, 1),
        }

    return result


# ---------------- SELF-LEARNING LOOP ----------------
def get_live_scan_stats(modes=None) -> dict:
    """Stats on live scan outcomes for the self-learning tracker on the analytics page.
    Optional modes filter restricts to specific scan modes (e.g. ['fivemin']).
    Without a filter, excludes only 'historical' mode scans."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if modes:
        placeholders = ",".join("?" * len(modes))
        base_clause = f"mode IN ({placeholders})"
        base_params = list(modes)
    else:
        base_clause = "mode != 'historical'"
        base_params = []

    three_days_ago = (datetime.utcnow() - timedelta(days=2)).isoformat()

    cursor.execute(f"SELECT COUNT(*) FROM scans WHERE {base_clause}", base_params)
    total = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM scans WHERE {base_clause} AND next_day_return IS NOT NULL", base_params)
    labeled = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM scans WHERE {base_clause} AND next_day_return IS NULL AND timestamp >= ?",
                   base_params + [three_days_ago])
    pending = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM scans WHERE {base_clause} AND next_day_return IS NULL AND timestamp < ?",
                   base_params + [three_days_ago])
    awaiting = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM scans WHERE {base_clause} AND days_to_20pct IS NOT NULL", base_params)
    hit_20 = cursor.fetchone()[0]
    cursor.execute(f"SELECT AVG(days_to_20pct) FROM scans WHERE {base_clause} AND days_to_20pct IS NOT NULL", base_params)
    avg_days_row = cursor.fetchone()
    avg_days = round(avg_days_row[0], 1) if avg_days_row[0] else None

    cursor.execute("SELECT updated_at FROM settings WHERE key = 'auto_learn_labeled_count'")
    last_learn_row = cursor.fetchone()
    last_learn = last_learn_row[0] if last_learn_row else None

    conn.close()
    hit_rate = round(hit_20 / labeled * 100, 1) if labeled > 0 else 0
    return {
        "total":           total,
        "labeled":         labeled,
        "pending":         pending,
        "awaiting":        awaiting,
        "hit_20_count":    hit_20,
        "hit_rate":        hit_rate,
        "avg_days_to_20":  avg_days,
        "last_auto_learn": last_learn,
    }


def get_auto_learn_count() -> int:
    """Returns the labeled scan count recorded at the last auto-learn run."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = 'auto_learn_labeled_count'")
    row = cursor.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def save_auto_learn_count(count: int):
    """Persist the labeled scan count after a successful auto-learn run."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES ('auto_learn_labeled_count', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (str(count), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


# ---------------- AUTO AI ----------------

def save_autoai_log(entry: dict):
    """Insert a row into the autoai_log audit table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO autoai_log
            (ran_at, trigger_reason, trades_evaluated, hypotheses_added,
             hypotheses_auto_activated, hypotheses_to_pending,
             weights_auto_applied, weight_confidence, summary, full_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        entry.get("ran_at", datetime.utcnow().isoformat()),
        entry.get("trigger_reason", ""),
        entry.get("trades_evaluated", 0),
        entry.get("hypotheses_added", 0),
        entry.get("hypotheses_auto_activated", 0),
        entry.get("hypotheses_to_pending", 0),
        entry.get("weights_auto_applied", 0),
        entry.get("weight_confidence", 0),
        entry.get("summary", ""),
        entry.get("full_response", ""),
    ))
    conn.commit()
    conn.close()


def get_autoai_log(limit: int = 20) -> list:
    """Returns recent Auto AI optimization run records, newest first."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, ran_at, trigger_reason, trades_evaluated, hypotheses_added,
               hypotheses_auto_activated, hypotheses_to_pending,
               weights_auto_applied, weight_confidence, summary, full_response
        FROM autoai_log
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id":                        r[0],
            "ran_at":                    r[1],
            "trigger_reason":            r[2],
            "trades_evaluated":          r[3],
            "hypotheses_added":          r[4],
            "hypotheses_auto_activated": r[5],
            "hypotheses_to_pending":     r[6],
            "weights_auto_applied":      r[7],
            "weight_confidence":         r[8],
            "summary":                   r[9],
            "full_response":             r[10],
        }
        for r in rows
    ]


def get_last_autoai_run_count() -> int:
    """Returns the outcome-labeled scan count recorded at the last Auto AI optimization run."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = 'autoai_last_trade_count'")
    row = cursor.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def save_last_autoai_run_count(count: int):
    """Persist the outcome-labeled scan count after a successful Auto AI optimization run."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES ('autoai_last_trade_count', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (str(count), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def save_hypothesis_rules_with_confidence(rules: list):
    """
    Insert hypothesis rules with confidence scores from an Auto AI optimization run.
    Rules with confidence >= 80 are auto-activated (status='active', auto_applied=1).
    Rules with confidence 50-79 go to pending for admin review.
    Assigns a generation number = current max + 1.
    """
    if not rules:
        return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(generation), 0) FROM hypothesis_rules")
    next_gen = cursor.fetchone()[0] + 1
    now = datetime.utcnow().isoformat()
    for r in rules:
        cursor.execute("""
            INSERT INTO hypothesis_rules
                (rule_text, source, status, generation, created_at, confidence_score, auto_applied)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            r["text"],
            r.get("source", ""),
            r.get("status", "pending"),
            next_gen,
            now,
            r.get("confidence_score"),
            r.get("auto_applied", 0),
        ))
    conn.commit()
    conn.close()


def get_autoai_weights():
    """
    Returns dict with Auto AI autonomous weights, rationale, suggestions, summary,
    and updated_at. Returns None if no Auto AI weights have been stored yet.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    keys = [
        "autoai_weights",
        "autoai_weights_rationale",
        "autoai_weights_suggestions",
        "autoai_weights_summary",
    ]
    result = {}
    for key in keys:
        cursor.execute("SELECT value, updated_at FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            result[key] = row[0]
            result["updated_at"] = row[1]
    conn.close()
    if "autoai_weights" not in result:
        return None
    return {
        "weights":     json.loads(result["autoai_weights"]),
        "rationale":   result.get("autoai_weights_rationale", ""),
        "suggestions": json.loads(result.get("autoai_weights_suggestions", "[]")),
        "summary":     result.get("autoai_weights_summary", ""),
        "updated_at":  result.get("updated_at", ""),
    }


def save_autoai_weights(weights: dict, rationale: str = "",
                        suggestions: list = None, summary: str = "",
                        source: str = "autoai", goal: str = ""):
    """Persist Auto AI autonomous weights to the settings table and create a version snapshot."""
    save_weight_version("autoai", weights, summary=summary, rationale=rationale,
                        source=source, goal=goal)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    for key, value in [
        ("autoai_weights",             json.dumps(weights)),
        ("autoai_weights_rationale",   rationale),
        ("autoai_weights_suggestions", json.dumps(suggestions or [])),
        ("autoai_weights_summary",     summary),
    ]:
        cursor.execute("""
            INSERT INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
        """, (key, value, now))
    conn.commit()
    conn.close()


# ---------------- CHAT SUGGESTIONS ----------------
def save_chat_suggestion(suggested_by: str, action: dict, note: str = "") -> int:
    """Save a non-admin user's suggested action for admin review. Returns new row id."""
    import json as _json
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_suggestions (suggested_by, suggested_at, action_json, note, status)
        VALUES (?, ?, ?, ?, 'pending')
    """, (suggested_by, datetime.utcnow().isoformat(), _json.dumps(action), note or ""))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_chat_suggestions(status: str = "pending") -> list:
    """Return chat suggestions filtered by status, newest first."""
    import json as _json
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, suggested_by, suggested_at, action_json, note, status
        FROM chat_suggestions WHERE status = ? ORDER BY id DESC
    """, (status,))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        try:
            action = _json.loads(r["action_json"])
        except Exception:
            action = {}
        result.append({
            "id": r["id"],
            "suggested_by": r["suggested_by"],
            "suggested_at": r["suggested_at"],
            "action": action,
            "note": r["note"],
            "status": r["status"],
        })
    return result


def dismiss_chat_suggestion(suggestion_id: int, status: str = "dismissed") -> bool:
    """Mark a suggestion as dismissed or approved."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE chat_suggestions SET status = ? WHERE id = ?", (status, suggestion_id))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return ok
