"""
System health checks — run periodically to verify all major subsystems
are functioning correctly. Results stored in settings table as 'health_status'.

Checks:
  1.  db_connectivity      — Can we read/write the scans table?
  2.  scanner_activity     — Was a scan recorded in the last expected window?
  3.  return_labeling      — Are outcomes being filled in by update_returns?
  4.  scan_price_recorded  — Are scan prices being saved (needed for return calc)?
  5.  fivemin_scanner      — Is the 5m scanner active during market hours?
  6.  finviz_connectivity  — Can we reach the FinViz screener?
  7.  telegram_bot         — Is the Telegram bot token valid and reachable?
  8.  paper_trading        — Is Alpaca configured and the account active?
  9.  ai_optimization      — Has the Auto AI optimizer run recently?
  10. weight_versioning    — Are model weight changes being tracked?
  11. feedback_uploads     — Is manual feedback being recorded?
  12. lstm_model           — Does the trained LSTM model file exist?
"""

import json
import os
import sqlite3
import datetime

import pytz
import requests

from app.database import DB_NAME


# ── helpers ──────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.datetime.utcnow().isoformat()


def _check(id_: str, name: str, status: str, message: str, detail: str = "") -> dict:
    return {
        "id":      id_,
        "name":    name,
        "status":  status,   # "ok" | "warning" | "error" | "skipped"
        "message": message,
        "detail":  detail,
    }


def _is_market_hours() -> bool:
    et = pytz.timezone("America/New_York")
    now = datetime.datetime.now(et)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return datetime.time(9, 30) <= t <= datetime.time(16, 0)


def _is_weekday_et() -> bool:
    et = pytz.timezone("America/New_York")
    return datetime.datetime.now(et).weekday() < 5


def _hours_since(ts_iso: str) -> float | None:
    """Return hours elapsed since an ISO timestamp string (UTC assumed)."""
    if not ts_iso:
        return None
    try:
        # strip timezone info if present for comparison
        ts = ts_iso[:19]
        dt = datetime.datetime.fromisoformat(ts)
        delta = datetime.datetime.utcnow() - dt
        return delta.total_seconds() / 3600
    except Exception:
        return None


# ── individual checks ─────────────────────────────────────────────────────────

def _check_db_connectivity() -> dict:
    name = "DB Connectivity"
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM scans")
        count = cursor.fetchone()[0]
        conn.close()
        return _check("db_connectivity", name, "ok",
                      f"DB readable — {count:,} total scan rows")
    except Exception as e:
        return _check("db_connectivity", name, "error",
                      "Cannot read scans table", str(e))


def _check_scanner_activity() -> dict:
    name = "Scanner Recording"
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(timestamp), COUNT(*) FROM scans
            WHERE mode NOT IN ('historical', 'fivemin_bt')
            AND timestamp >= datetime('now', '-7 days')
        """)
        row = cursor.fetchone()
        conn.close()
        last_ts, recent_count = row[0], row[1]

        hours = _hours_since(last_ts)

        if hours is None:
            return _check("scanner_activity", name, "warning",
                          "No scans recorded in the past 7 days",
                          "Run a manual scan or check FinViz connectivity")

        if _is_weekday_et():
            if hours <= 8:
                return _check("scanner_activity", name, "ok",
                              f"Last scan {hours:.1f}h ago — {recent_count} scans in 7d")
            elif hours <= 26:
                return _check("scanner_activity", name, "warning",
                              f"Last scan {hours:.1f}h ago — expected within 8h on weekdays",
                              f"{recent_count} scans in past 7 days")
            else:
                return _check("scanner_activity", name, "error",
                              f"No scan in {hours:.0f}h — scanner may be down",
                              f"{recent_count} scans in past 7 days")
        else:
            # Weekend — expect last scan to be from Friday (≤72h)
            if hours <= 72:
                return _check("scanner_activity", name, "ok",
                              f"Weekend — last scan {hours:.0f}h ago (expected)")
            else:
                return _check("scanner_activity", name, "warning",
                              f"Last scan {hours:.0f}h ago",
                              f"{recent_count} scans in past 7 days")
    except Exception as e:
        return _check("scanner_activity", name, "error", "Check failed", str(e))


def _check_return_labeling() -> dict:
    name = "Outcome Labeling"
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # Scans older than 3 days that still lack a next_day_return
        cursor.execute("""
            SELECT COUNT(*) FROM scans
            WHERE next_day_return IS NULL
            AND mode NOT IN ('historical', 'fivemin_bt')
            AND timestamp < datetime('now', '-3 days')
        """)
        unlabeled = cursor.fetchone()[0]

        # Total labeled in last 30 days
        cursor.execute("""
            SELECT COUNT(*) FROM scans
            WHERE next_day_return IS NOT NULL
            AND mode NOT IN ('historical', 'fivemin_bt')
            AND timestamp >= datetime('now', '-30 days')
        """)
        labeled_30d = cursor.fetchone()[0]
        conn.close()

        if unlabeled == 0:
            return _check("return_labeling", name, "ok",
                          f"All outcomes labeled — {labeled_30d} labeled in 30d")
        elif unlabeled <= 5:
            return _check("return_labeling", name, "warning",
                          f"{unlabeled} scan(s) >3 days old without outcome",
                          "update_returns() may not have run; check scheduler")
        else:
            return _check("return_labeling", name, "error",
                          f"{unlabeled} scans missing outcomes (>3 days old)",
                          "update_returns() appears to not be running")
    except Exception as e:
        return _check("return_labeling", name, "error", "Check failed", str(e))


def _check_scan_price_recorded() -> dict:
    name = "Scan Price Recording"
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # Recent live scans — do they have scan_price?
        cursor.execute("""
            SELECT COUNT(*), SUM(CASE WHEN scan_price IS NULL THEN 1 ELSE 0 END)
            FROM scans
            WHERE mode NOT IN ('historical', 'fivemin_bt')
            AND timestamp >= datetime('now', '-7 days')
        """)
        row = cursor.fetchone()
        conn.close()
        total, missing = row[0] or 0, row[1] or 0
        if total == 0:
            return _check("scan_price_recorded", name, "skipped",
                          "No live scans in past 7 days to check")
        pct_missing = missing / total * 100
        if pct_missing == 0:
            return _check("scan_price_recorded", name, "ok",
                          f"All {total} recent scans have price recorded")
        elif pct_missing < 20:
            return _check("scan_price_recorded", name, "warning",
                          f"{missing}/{total} recent scans missing scan_price ({pct_missing:.0f}%)",
                          "Partial data — return calculation will be impaired")
        else:
            return _check("scan_price_recorded", name, "error",
                          f"{pct_missing:.0f}% of recent scans missing scan_price",
                          "Return tracking will be unreliable — investigate save_scan()")
    except Exception as e:
        return _check("scan_price_recorded", name, "error", "Check failed", str(e))


def _check_fivemin_scanner() -> dict:
    name = "5m Scanner Activity"
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(timestamp), COUNT(*) FROM scans
            WHERE mode = 'fivemin'
            AND timestamp >= datetime('now', '-2 days')
        """)
        row = cursor.fetchone()
        conn.close()
        last_ts, count_2d = row[0], row[1]
        hours = _hours_since(last_ts)

        if _is_market_hours():
            if hours is not None and hours <= 0.25:  # within 15 min
                return _check("fivemin_scanner", name, "ok",
                              f"Active — last 5m scan {int(hours*60)}min ago")
            elif hours is not None and hours <= 1:
                return _check("fivemin_scanner", name, "warning",
                              f"5m scanner last ran {int(hours*60)}min ago (expected <15min)",
                              "May have missed a scheduled run")
            else:
                return _check("fivemin_scanner", name, "error",
                              "5m scanner has not run during current market session",
                              "Check scheduler — _fivemin_spike_scan should run every 5min 10:00–15:30 ET")
        else:
            if hours is not None and hours <= 72:
                return _check("fivemin_scanner", name, "ok",
                              f"Market closed — last 5m scan {hours:.0f}h ago, {count_2d} in 2d")
            else:
                return _check("fivemin_scanner", name, "warning",
                              "No 5m scans in past 2 days",
                              f"Last scan: {last_ts or 'never'}")
    except Exception as e:
        return _check("fivemin_scanner", name, "error", "Check failed", str(e))


def _check_finviz_connectivity() -> dict:
    name = "FinViz Connectivity"
    url = (
        "https://finviz.com/screener.ashx?"
        "v=161&f=sh_curvol_o1000,sh_price_u5,sh_relvol_o10,ta_perf_d10o&ft=4"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            # Quick sanity: page should contain "Total:" or screener table
            if "Total:" in resp.text or "screener-table" in resp.text or "ticker" in resp.text.lower():
                return _check("finviz_connectivity", name, "ok",
                              f"FinViz reachable — HTTP {resp.status_code}")
            else:
                return _check("finviz_connectivity", name, "warning",
                              "FinViz returned 200 but response looks unexpected",
                              "May be returning a CAPTCHA or error page")
        elif resp.status_code == 429:
            return _check("finviz_connectivity", name, "warning",
                          "FinViz rate-limiting us (HTTP 429)",
                          "Scans may fail during heavy usage periods")
        else:
            return _check("finviz_connectivity", name, "error",
                          f"FinViz returned HTTP {resp.status_code}",
                          resp.text[:200])
    except requests.Timeout:
        return _check("finviz_connectivity", name, "error",
                      "FinViz request timed out (>10s)")
    except Exception as e:
        return _check("finviz_connectivity", name, "error",
                      "Cannot reach FinViz", str(e))


def _check_telegram_bot() -> dict:
    name = "Telegram Alerts"
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return _check("telegram_bot", name, "warning",
                      "TELEGRAM_BOT_TOKEN not set — alerts disabled",
                      "Set the env var to enable Telegram notifications")
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{token}/getMe",
            timeout=8,
        )
        data = resp.json()
        if data.get("ok"):
            bot_name = data.get("result", {}).get("username", "?")
            # Check if any recipients are configured
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM telegram_recipients WHERE active = 1")
                recipients = cursor.fetchone()[0]
                conn.close()
            except Exception:
                recipients = 0
            if recipients > 0:
                return _check("telegram_bot", name, "ok",
                              f"Bot @{bot_name} active — {recipients} recipient(s) configured")
            else:
                return _check("telegram_bot", name, "warning",
                              f"Bot @{bot_name} valid but no active recipients configured",
                              "Add recipients via the Alerts settings page")
        else:
            return _check("telegram_bot", name, "error",
                          "Telegram API rejected the bot token",
                          data.get("description", ""))
    except Exception as e:
        return _check("telegram_bot", name, "error",
                      "Cannot reach Telegram API", str(e))


def _check_paper_trading() -> dict:
    name = "Paper Trading (Alpaca)"
    from app import broker
    if not broker.is_configured():
        return _check("paper_trading", name, "skipped",
                      "Alpaca API keys not set — paper trading disabled",
                      "Set ALPACA_API_KEY and ALPACA_SECRET_KEY to enable")
    try:
        account = broker.get_account()
        if account is None:
            return _check("paper_trading", name, "error",
                          "Alpaca keys set but account call failed",
                          "Check ALPACA_API_KEY / ALPACA_SECRET_KEY are valid paper keys")
        status = account.get("status", "?")
        equity = account.get("equity", "?")
        buying_power = account.get("buying_power", "?")
        if status == "ACTIVE":
            # Count open paper trades in our DB
            try:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'")
                open_trades = cursor.fetchone()[0]
                conn.close()
            except Exception:
                open_trades = "?"
            return _check("paper_trading", name, "ok",
                          f"Account ACTIVE — equity ${float(equity):,.2f}, "
                          f"{open_trades} open trade(s)",
                          f"Buying power: ${float(buying_power):,.2f}")
        else:
            return _check("paper_trading", name, "warning",
                          f"Alpaca account status: {status}",
                          "May need to activate the paper trading account")
    except Exception as e:
        return _check("paper_trading", name, "error", "Alpaca check failed", str(e))


def _check_ai_optimization() -> dict:
    name = "AI Auto-Optimizer"
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(ran_at), COUNT(*) FROM autoai_log")
        row = cursor.fetchone()
        conn.close()
        last_ran, total_runs = row[0], row[1]
        hours = _hours_since(last_ran)

        if total_runs == 0:
            return _check("ai_optimization", name, "warning",
                          "No Auto AI optimization runs recorded yet",
                          "Trigger a run from Analytics → Daily Models")
        if hours is None:
            return _check("ai_optimization", name, "warning",
                          "Cannot parse last run timestamp")
        if hours <= 48:
            return _check("ai_optimization", name, "ok",
                          f"Last run {hours:.0f}h ago — {total_runs} total runs")
        elif hours <= 168:  # 7 days
            return _check("ai_optimization", name, "warning",
                          f"Last run {hours / 24:.0f} days ago",
                          "Consider running Auto AI more frequently for faster model improvement")
        else:
            return _check("ai_optimization", name, "error",
                          f"No optimization in {hours / 24:.0f} days",
                          "Auto AI has not run recently — trigger from Analytics")
    except Exception as e:
        return _check("ai_optimization", name, "error", "Check failed", str(e))


def _check_weight_versioning() -> dict:
    name = "Weight Version Tracking"
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), MAX(created_at) FROM weight_versions")
        row = cursor.fetchone()
        conn.close()
        count, last_at = row[0], row[1]
        if count == 0:
            return _check("weight_versioning", name, "warning",
                          "No weight versions recorded yet",
                          "Save or optimize weights to start tracking model evolution")
        hours = _hours_since(last_at)
        ago = f"{hours:.0f}h ago" if hours and hours < 48 else (
            f"{hours/24:.0f} days ago" if hours else "unknown"
        )
        return _check("weight_versioning", name, "ok",
                      f"{count} version(s) tracked — last update {ago}")
    except sqlite3.OperationalError:
        return _check("weight_versioning", name, "warning",
                      "weight_versions table not found",
                      "Run init_db() to apply schema migrations")
    except Exception as e:
        return _check("weight_versioning", name, "error", "Check failed", str(e))


def _check_feedback_uploads() -> dict:
    name = "Manual Feedback & Uploads"
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), MAX(created_at) FROM feedback")
        row = cursor.fetchone()
        conn.close()
        count, last_at = row[0], row[1]
        if count == 0:
            return _check("feedback_uploads", name, "warning",
                          "No feedback entries yet",
                          "Upload chart notes via the Feedback page to improve AI hypotheses")
        hours = _hours_since(last_at)
        ago = f"{hours:.0f}h ago" if hours and hours < 48 else (
            f"{int(hours/24)} days ago" if hours else "unknown"
        )
        return _check("feedback_uploads", name, "ok",
                      f"{count} feedback entries — last upload {ago}")
    except Exception as e:
        return _check("feedback_uploads", name, "error", "Check failed", str(e))


def _check_lstm_model() -> dict:
    name = "LSTM Model File"
    model_paths = [
        "lstm_model.pkl",
        "app/lstm_model.pkl",
        os.path.join(os.path.dirname(DB_NAME), "lstm_model.pkl"),
    ]
    for path in model_paths:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            mtime = datetime.datetime.utcfromtimestamp(os.path.getmtime(path))
            hours = (datetime.datetime.utcnow() - mtime).total_seconds() / 3600
            ago = f"{hours:.0f}h ago" if hours < 48 else f"{hours/24:.0f}d ago"
            return _check("lstm_model", name, "ok",
                          f"Model file found ({size_kb:.0f} KB) — last trained {ago}",
                          path)
    # Check if lstm_sequences.npz exists (training data but no model yet)
    for path in ["lstm_sequences.npz", "app/lstm_sequences.npz"]:
        if os.path.exists(path):
            return _check("lstm_model", name, "warning",
                          "Training sequences exist but no trained model found",
                          "Run 'Retrain LSTM' from Analytics to generate the model")
    return _check("lstm_model", name, "warning",
                  "No LSTM model or training data found",
                  "Run a historical backfill to generate training data, then train")


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_health_checks() -> dict:
    """
    Run all system health checks and return a structured summary.
    Also persists the result to the settings table.
    """
    checks = [
        _check_db_connectivity(),
        _check_scanner_activity(),
        _check_return_labeling(),
        _check_scan_price_recorded(),
        _check_fivemin_scanner(),
        _check_finviz_connectivity(),
        _check_telegram_bot(),
        _check_paper_trading(),
        _check_ai_optimization(),
        _check_weight_versioning(),
        _check_feedback_uploads(),
        _check_lstm_model(),
    ]

    # Overall status: worst of all non-skipped checks
    statuses = [c["status"] for c in checks if c["status"] != "skipped"]
    if "error" in statuses:
        overall = "error"
    elif "warning" in statuses:
        overall = "warning"
    else:
        overall = "ok"

    counts = {
        "ok":      sum(1 for c in checks if c["status"] == "ok"),
        "warning": sum(1 for c in checks if c["status"] == "warning"),
        "error":   sum(1 for c in checks if c["status"] == "error"),
        "skipped": sum(1 for c in checks if c["status"] == "skipped"),
    }

    result = {
        "checked_at": _utcnow(),
        "overall":    overall,
        "counts":     counts,
        "checks":     checks,
    }

    _save_health_status(result)
    return result


def _save_health_status(result: dict):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO settings (key, value, updated_at)
            VALUES ('health_status', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """, (json.dumps(result), _utcnow()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Health: failed to save status — {e}")


def get_health_status() -> dict | None:
    """Return the most recently stored health check result, or None."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = 'health_status'")
        row = cursor.fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None
