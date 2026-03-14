"""
Stop-Loss Parameter Optimizer

Backtests stop-loss parameter candidates against labeled scan history, then
validates proposed changes against two hard constraints before auto-applying:

  1. Winner preservation >= WINNER_PRESERVE_MIN (default 95%)
     "Don't kill trades that would have hit +20%"
  2. Average loss reduction >= LOSS_REDUCE_MIN (default 5%)
     "Meaningfully reduce downside on losing trades"

Daily model:  5 tunable parameters
  stop_loss_pct      — hard stop: exit if close < entry * (1 - X%)
  trail_activate_pct — trailing stop armed once gain exceeds X%
  trail_pullback_pp  — trailing stop fires if close drops X pp below peak close
  time_stop_days     — hard time exit if negative after X trading days
  stale_days         — flatline exit if gain < stale_gain_pct after X trading days
  stale_gain_pct     — flatline threshold (%)

5m model:  3 tunable parameters
  stop_loss_pct      — hard stop: -X% intraday (checked every 5m bar)
  trail_activate_pct — trailing stop armed once +X% intraday
  trail_pullback_pp  — trailing stop fires if price drops X pp below 5m peak

Simulation uses the same price-check logic as the live _autoclose_take_profit():
  - Take-profit triggered on the day's HIGH (intraday can touch +20%)
  - Hard stop checked on daily CLOSE (avoids false triggers on penny-stock noise)
  - Trailing / time / staleness stops checked on daily CLOSE
  - For 5m: every bar's HIGH (for trail peak) and LOW (for hard stop) are used
"""

import json
import sqlite3
import time
from datetime import date, datetime, timedelta

from app.database import DB_NAME

# ─── Validation thresholds ────────────────────────────────────────────────────
WINNER_PRESERVE_MIN = 0.95   # at least 95% of winners must still win
LOSS_REDUCE_MIN     = 0.05   # average loss must shrink by at least 5%
MIN_BACKTEST_ROWS   = 30     # refuse to optimize with fewer labeled scans

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DAILY_PARAMS = {
    "stop_loss_pct":      20.0,
    "trail_activate_pct": 10.0,
    "trail_pullback_pp":  12.0,
    "time_stop_days":     5,    # tightened from 10: 97.1% of winners hit by day 5 (n=2418, 2026-03-14)
    "stale_days":         5,    # aligned with time_stop
    "stale_gain_pct":     3.0,
}

DEFAULT_5M_PARAMS = {
    "stop_loss_pct":      15.0,
    "trail_activate_pct": 10.0,
    "trail_pullback_pp":   8.0,
}

# ─── Settings persistence ─────────────────────────────────────────────────────
_DAILY_KEY  = "stop_loss_params_daily"
_5M_KEY     = "stop_loss_params_fivemin"


def get_stop_loss_params(mode: str) -> dict:
    """Load stop-loss params from settings table, or return defaults."""
    key = _DAILY_KEY if mode == "daily" else _5M_KEY
    defaults = DEFAULT_DAILY_PARAMS if mode == "daily" else DEFAULT_5M_PARAMS
    try:
        conn = sqlite3.connect(DB_NAME)
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        if row:
            return {**defaults, **json.loads(row[0])}
    except Exception:
        pass
    return defaults.copy()


def save_stop_loss_params(mode: str, params: dict):
    """Persist stop-loss params to settings table."""
    key = _DAILY_KEY if mode == "daily" else _5M_KEY
    now = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
    """, (key, json.dumps(params), now))
    conn.commit()
    conn.close()


# ─── Price path fetching ──────────────────────────────────────────────────────

def _fetch_daily_path(symbol: str, start_date: date, n_days: int = 12) -> list:
    """
    Return up to n_days of daily OHLCV dicts starting from start_date.
    Each dict: {open, high, low, close, date}
    Returns [] on any error.
    """
    try:
        import yfinance as yf
        end = start_date + timedelta(days=n_days * 2)   # enough calendar days
        df = yf.download(symbol,
                         start=start_date.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            return []
        # Flatten multi-index if present
        if hasattr(df.columns, "get_level_values"):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass
        rows = []
        for ts, row in df.iterrows():
            try:
                rows.append({
                    "date":  ts.date() if hasattr(ts, "date") else ts,
                    "open":  float(row.get("Open",  row.get("open",  0)) or 0),
                    "high":  float(row.get("High",  row.get("high",  0)) or 0),
                    "low":   float(row.get("Low",   row.get("low",   0)) or 0),
                    "close": float(row.get("Close", row.get("close", 0)) or 0),
                })
            except Exception:
                continue
            if len(rows) >= n_days:
                break
        return rows
    except Exception:
        return []


def _fetch_5m_path(symbol: str, trade_date: date) -> list:
    """
    Return 5-minute bars for symbol on trade_date.
    Each dict: {ts (datetime), high, low, close, minutes_from_open}
    Returns [] if data unavailable (yfinance only has ~60 days of 5m history).
    """
    try:
        import yfinance as yf
        import pytz
        et = pytz.timezone("America/New_York")
        start = trade_date
        end   = trade_date + timedelta(days=1)
        df = yf.download(symbol,
                         start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         interval="5m", progress=False, auto_adjust=False)
        if df.empty:
            return []
        if hasattr(df.columns, "get_level_values"):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass

        market_open = et.localize(datetime.combine(trade_date, datetime.min.time()).replace(hour=9, minute=30))
        rows = []
        for ts, row in df.iterrows():
            try:
                ts_et = ts.tz_convert(et) if ts.tzinfo else et.localize(ts)
                if ts_et < market_open:
                    continue
                mins = int((ts_et - market_open).total_seconds() / 60)
                rows.append({
                    "ts":               ts_et,
                    "minutes_from_open": mins,
                    "high":  float(row.get("High",  row.get("high",  0)) or 0),
                    "low":   float(row.get("Low",   row.get("low",   0)) or 0),
                    "close": float(row.get("Close", row.get("close", 0)) or 0),
                })
            except Exception:
                continue
        return rows
    except Exception:
        return []


# ─── Simulation engines ───────────────────────────────────────────────────────

def _simulate_daily(entry: float, path: list, params: dict) -> dict:
    """
    Simulate a daily trade through `path` using `params`.
    Returns {exit_price, close_reason, day, hit_20pct}
    Matching live _autoclose_take_profit() logic exactly:
      - Take-profit on HIGH (intraday can touch it)
      - All stops on CLOSE (avoids penny-stock noise)
    """
    sl  = params["stop_loss_pct"]
    ta  = params["trail_activate_pct"]
    tp  = params["trail_pullback_pp"]
    tsd = int(params["time_stop_days"])
    sd  = int(params["stale_days"])
    sgp = params["stale_gain_pct"]

    peak_close = entry   # tracks peak CLOSE for trailing stop

    for day, bar in enumerate(path, start=1):
        high  = bar["high"]
        close = bar["close"]
        if close <= 0:
            continue

        # Take-profit: can touch intraday high
        if high > 0 and (high / entry - 1) * 100 >= 20.0:
            return {"exit_price": entry * 1.20, "close_reason": "take_profit",
                    "day": day, "hit_20pct": True}

        pnl_pct  = (close / entry - 1) * 100
        if close > peak_close:
            peak_close = close
        peak_pct = (peak_close / entry - 1) * 100

        # Hard stop (close)
        if pnl_pct <= -sl:
            return {"exit_price": close, "close_reason": "stop_loss",
                    "day": day, "hit_20pct": False}

        # Trailing stop (close)
        if peak_pct >= ta and pnl_pct <= (peak_pct - tp):
            return {"exit_price": close, "close_reason": "trailing_stop",
                    "day": day, "hit_20pct": False}

        # Time stop
        if day >= tsd and pnl_pct < 0:
            return {"exit_price": close, "close_reason": "time_stop",
                    "day": day, "hit_20pct": False}

        # Staleness stop
        if day >= sd and pnl_pct < sgp:
            return {"exit_price": close, "close_reason": "staleness_stop",
                    "day": day, "hit_20pct": False}

    # Held through all bars — final price
    last = path[-1]["close"] if path else entry
    hit  = last > 0 and (last / entry - 1) * 100 >= 20.0
    return {"exit_price": last, "close_reason": "expired",
            "day": len(path), "hit_20pct": hit}


def _simulate_5m(entry: float, bars: list, params: dict) -> dict:
    """
    Simulate a 5m trade through intraday `bars`.
    Rules:
      - Take-profit: HIGH >= entry * 1.20
      - Hard stop:   LOW  <= entry * (1 - stop_loss_pct/100)
      - Trailing:    peak HIGH tracked; exit if close < peak * (1 - trail_pullback_pp/100)
                     (only armed once gain hits trail_activate_pct%)
      - Time stop:   > 90 min elapsed AND gain < +5%
      - EOD:         bar at or after 3:50 PM ET
    """
    sl  = params["stop_loss_pct"]
    ta  = params["trail_activate_pct"]
    tp  = params["trail_pullback_pp"]

    peak_high = entry

    for bar in bars:
        mins  = bar["minutes_from_open"]
        high  = bar["high"]
        low   = bar["low"]
        close = bar["close"]
        if close <= 0:
            continue

        # EOD force-close (3:50 PM = 230 min from 9:30 open)
        if mins >= 230:
            return {"exit_price": close, "close_reason": "eod",
                    "mins": mins, "hit_20pct": (close / entry - 1) * 100 >= 20.0}

        if high > peak_high:
            peak_high = high
        peak_pct = (peak_high / entry - 1) * 100
        pnl_pct  = (close    / entry - 1) * 100

        # Take-profit on intraday high
        if high > 0 and (high / entry - 1) * 100 >= 20.0:
            return {"exit_price": entry * 1.20, "close_reason": "take_profit",
                    "mins": mins, "hit_20pct": True}

        # Hard stop on intraday low
        if low > 0 and (low / entry - 1) * 100 <= -sl:
            return {"exit_price": entry * (1 - sl / 100), "close_reason": "stop_loss",
                    "mins": mins, "hit_20pct": False}

        # Trailing stop (close price, only armed after gain >= trail_activate_pct)
        if peak_pct >= ta and pnl_pct <= (peak_pct - tp):
            return {"exit_price": close, "close_reason": "trailing_stop",
                    "mins": mins, "hit_20pct": False}

        # Time stop: >90 min and still < +5%
        if mins > 90 and pnl_pct < 5.0:
            return {"exit_price": close, "close_reason": "time_stop",
                    "mins": mins, "hit_20pct": False}

    last = bars[-1]["close"] if bars else entry
    hit  = last > 0 and (last / entry - 1) * 100 >= 20.0
    return {"exit_price": last, "close_reason": "expired",
            "mins": bars[-1]["minutes_from_open"] if bars else 0, "hit_20pct": hit}


# ─── Backtest runner ──────────────────────────────────────────────────────────

def _load_backtest_scans(mode: str, limit: int = 400) -> list:
    """
    Load labeled scans suitable for stop-loss backtesting.
    Returns list of {symbol, scan_date, entry_price, actual_winner (bool)}.
    """
    db_mode_filter = "AND mode = 'fivemin'" if mode == "5m" else "AND mode NOT IN ('fivemin', 'fivemin_bt')"
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    # Use ALL labeled scans including historical/backfill — they have real outcome data
    rows = conn.execute(f"""
        SELECT symbol, DATE(timestamp) as scan_date,
               scan_price, days_to_20pct
        FROM scans
        WHERE scan_price IS NOT NULL AND scan_price > 0
          AND next_day_return IS NOT NULL
          {db_mode_filter}
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [
        {
            "symbol":         r["symbol"],
            "scan_date":      date.fromisoformat(r["scan_date"]),
            "entry_price":    float(r["scan_price"]),
            "actual_winner":  r["days_to_20pct"] is not None,
        }
        for r in rows
        if r["scan_price"] and float(r["scan_price"]) > 0
    ]


def backtest_params(params: dict, mode: str = "daily",
                    max_scans: int = 200) -> dict | None:
    """
    Run stop-loss simulation over labeled scans, fetching price paths from yfinance.

    Returns metrics dict:
    {
        n_scans, n_winners, n_losers,
        winner_preservation,   (fraction of actual winners that hit 20% under params)
        avg_loss_pct,          (average % loss on losing simulations)
        premature_stops,       (count of winners stopped before hitting 20%)
        close_reason_counts,   (dict)
    }
    Returns None if not enough data.
    """
    scans = _load_backtest_scans(mode, limit=max_scans)
    if len(scans) < MIN_BACKTEST_ROWS:
        return None

    n_winners = sum(1 for s in scans if s["actual_winner"])
    if n_winners < 5:
        return None

    results = []
    for scan in scans:
        symbol     = scan["symbol"]
        entry      = scan["entry_price"]
        scan_date  = scan["scan_date"]
        is_winner  = scan["actual_winner"]

        if mode == "5m":
            path = _fetch_5m_path(symbol, scan_date)
            if not path:
                continue
            sim = _simulate_5m(entry, path, params)
        else:
            path = _fetch_daily_path(symbol, scan_date, n_days=12)
            if not path:
                continue
            sim = _simulate_daily(entry, path, params)

        results.append({
            "is_winner":     is_winner,
            "hit_20pct":     sim["hit_20pct"],
            "exit_price":    sim["exit_price"],
            "close_reason":  sim["close_reason"],
            "pnl_pct":       (sim["exit_price"] / entry - 1) * 100,
        })
        time.sleep(0.15)   # rate-limit yfinance

    if not results:
        return None

    actual_winners    = [r for r in results if r["is_winner"]]
    preserved_winners = [r for r in actual_winners if r["hit_20pct"]]
    losers            = [r for r in results if r["pnl_pct"] < 0]
    premature         = [r for r in actual_winners if not r["hit_20pct"]]
    reasons           = {}
    for r in results:
        reasons[r["close_reason"]] = reasons.get(r["close_reason"], 0) + 1

    return {
        "n_scans":              len(results),
        "n_winners":            len(actual_winners),
        "n_losers":             len(losers),
        "winner_preservation":  len(preserved_winners) / len(actual_winners) if actual_winners else 1.0,
        "avg_loss_pct":         sum(r["pnl_pct"] for r in losers) / len(losers) if losers else 0.0,
        "premature_stops":      len(premature),
        "close_reason_counts":  reasons,
    }


def validate_proposed_params(current_params: dict, proposed_params: dict,
                               mode: str = "daily") -> dict:
    """
    Backtest both param sets and check the two hard constraints.

    Returns {
        valid:              bool,
        winner_preservation: float,   (proposed)
        loss_reduction:      float,   (fractional improvement; positive = better)
        current_metrics:    dict,
        proposed_metrics:   dict,
        failure_reason:     str | None,
    }
    """
    print(f"STOP-LOSS OPT: backtesting current params ({mode})...")
    cur = backtest_params(current_params, mode=mode)
    if cur is None:
        return {"valid": False, "failure_reason": "insufficient backtest data for current params"}

    print(f"STOP-LOSS OPT: backtesting proposed params ({mode})...")
    prop = backtest_params(proposed_params, mode=mode)
    if prop is None:
        return {"valid": False, "failure_reason": "insufficient backtest data for proposed params"}

    wp = prop["winner_preservation"]
    cur_loss  = cur["avg_loss_pct"]    # negative number
    prop_loss = prop["avg_loss_pct"]   # negative number

    # Loss reduction: positive = proposed has smaller losses
    # e.g. current=-15%, proposed=-12% → reduction = (-12 - -15) / abs(-15) = 0.20 → 20% better
    loss_reduction = (prop_loss - cur_loss) / abs(cur_loss) if cur_loss != 0 else 0.0

    meets_winner = wp >= WINNER_PRESERVE_MIN
    meets_loss   = loss_reduction >= LOSS_REDUCE_MIN

    failure = None
    if not meets_winner:
        failure = (f"winner preservation {wp:.1%} < required {WINNER_PRESERVE_MIN:.0%} "
                   f"({prop['premature_stops']} winners killed)")
    elif not meets_loss:
        failure = (f"loss reduction {loss_reduction:.1%} < required {LOSS_REDUCE_MIN:.0%} "
                   f"(current avg loss {cur_loss:.1f}%, proposed {prop_loss:.1f}%)")

    return {
        "valid":               meets_winner and meets_loss,
        "winner_preservation": wp,
        "loss_reduction":      loss_reduction,
        "current_metrics":     cur,
        "proposed_metrics":    prop,
        "failure_reason":      failure,
    }
