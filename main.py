# ai_trading_mvp/main.py

import asyncio
import datetime
import io
import json
import os as _os
import re as _re
import threading
import zipfile
import pytz
import requests
import yfinance as yf

from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import Response, FileResponse
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.broker import submit_market_order, close_position as broker_close, is_configured as broker_configured, get_account as broker_get_account
from app.scanner import run_scan, run_scan_5m, get_finviz_quotes, prepare_dataframe
from app.validator import validate_scan_results
from app.alerts import send_scan_alert, send_take_profit_alert, send_watchlist_alert, _send_telegram_admin, send_invite_email, send_weekly_report_email, send_exit_alert
from app.backfill import build_historical_dataset, backfill_signals_for_historical
from app.backfill_5m import backfill_5m_history, get_5m_backfill_status, set_5m_backfill_status
from app.health import run_health_checks, get_health_status
from app.lstm_model import train_lstm, predict_hit_probability, get_lstm_status, get_sequence_stats
from app.lstm_5m import (
    train_lstm_5m, predict_5m_hit_probability,
    get_lstm_5m_status, get_5m_sequence_stats, build_5m_sequences,
)
from app.ai_agent import (
    recommend_position_size,
    predict_price_target,
    recommend_trade,
    get_stock_news,
    analyze_and_optimize_weights,
    analyze_chart_feedback,
    reprocess_chart_analysis,
    synthesize_feedback_hypotheses,
    synthesize_historical_hypothesis,
    synthesize_combined_hypothesis,
    optimize_complex_ai_weights,
    parse_rules_from_synthesis,
    autonomous_optimize,
    chat_with_model,
)
from app.scoring_engine import DEFAULT_SQUEEZE_WEIGHTS, score_stock_squeeze
from app.database import (
    init_db,
    seed_users,
    get_all_users,
    delete_user,
    save_scan,
    update_returns,
    get_score_buckets,
    get_holding_performance,
    get_equity_curve,
    create_user,
    authenticate_user,
    open_trade,
    close_trade,
    get_trade_by_id,
    get_open_positions,
    get_trade_history,
    get_portfolio_summary,
    get_optimization_data,
    save_feedback,
    update_feedback_analysis,
    get_recent_feedback,
    get_all_feedback,
    save_hypothesis,
    save_autoai_hypothesis_blob,
    get_hypothesis,
    update_password,
    save_squeeze_weights,
    get_squeeze_weights,
    save_scan_cache,
    get_scan_cache,
    save_weight_changelog,
    get_weight_changelog,
    add_to_watchlist,
    add_near_miss_to_watchlist,
    get_active_watchlist,
    mark_watchlist_alerted,
    clear_old_watchlist,
    remove_from_watchlist,
    get_watchlist,
    get_risk_metrics,
    save_historical_scans,
    set_backfill_status,
    get_backfill_status,
    get_historical_count,
    get_historical_examples,
    get_sizing_stats,
    save_validation_report,
    get_validation_reports,
    get_live_scan_stats,
    get_model_validation_stats,
    get_ai_decision_accuracy,
    get_telegram_recipients,
    add_telegram_recipient,
    delete_telegram_recipient,
    get_auto_learn_count,
    save_auto_learn_count,
    update_scan_ai_rec,
    get_ticker_scan_history,
    tag_feedback_outcome,
    tag_trade_outcome,
    save_hypothesis_rules,
    get_hypothesis_rules,
    get_pending_rule_count,
    update_rule_status,
    get_active_rule_ids,
    save_scan_active_rules,
    get_active_hypothesis_text,
    import_feedback_from_backup,
    FEEDBACK_BACKUP_PATH,
    save_autoai_log,
    get_autoai_log,
    get_last_autoai_run_count,
    save_last_autoai_run_count,
    save_hypothesis_rules_with_confidence,
    get_autoai_weights,
    save_autoai_weights,
    get_model_comparison_stats,
    save_chat_suggestion,
    get_chat_suggestions,
    dismiss_chat_suggestion,
    get_per_signal_stats,
    refresh_bundle_projections,
    save_weight_version,
    get_current_version_id,
    get_version_performance_stats,
    save_bundle_as_rule,
    save_scan_candidates,
    rescore_historical_from_signals,
    count_auto_trades_today,
    get_alerted_symbols_today,
    update_high_watermark,
    is_user_admin,
)
from app.ml_optimizer import train_xgb_weights, get_xgb_status

app = FastAPI()

# Session middleware — secret must be set in env (SESSION_SECRET); falls back to
# a random value so the app still starts, but sessions won't survive restarts.
_session_secret = _os.environ.get("SESSION_SECRET") or _os.urandom(32).hex()
if not _os.environ.get("SESSION_SECRET"):
    print("STARTUP: WARNING — SESSION_SECRET env var not set. Sessions will not "
          "persist across restarts. Set SESSION_SECRET in Render environment variables.")
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret,
    https_only=True,      # cookie only sent over HTTPS (Render always uses HTTPS)
    same_site="strict",   # mitigates CSRF
    max_age=3600,         # sessions expire after 1 hour of inactivity
)

# Security headers on every response
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"]           = "DENY"
    response.headers["X-Content-Type-Options"]    = "nosniff"
    response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"]        = "geolocation=(), microphone=(), camera=()"
    return response

# In-memory login rate limiter: max 10 attempts per IP per 15-minute window
import time as _time
_login_attempts: dict = {}   # {ip: [timestamp, ...]}
_LOGIN_MAX        = 10
_LOGIN_WINDOW_SEC = 900      # 15 minutes

# Initialize database and seed users
init_db()
seed_users()

# In-process flag so the backfill guard is reliable even if DB status got stuck
_backfill_running = False
_signals_backfill_running = False
_lstm_training    = False

# Alert deduplication — seeded from DB on startup so restarts don't re-alert
def _init_alerted_today() -> set:
    import datetime, pytz
    today = datetime.datetime.now(pytz.timezone("America/New_York")).date().isoformat()
    return get_alerted_symbols_today(today)

_alerted_today: set = _init_alerted_today()
_alerted_date:  str = datetime.datetime.now(pytz.timezone("America/New_York")).date().isoformat()

# (no daily trade cap — available cash is the natural gate)

# ---------------- SCHEDULER ----------------
_MODEL_LABELS = {
    "squeeze":  "Daily Auto",
    "autoai":   "Daily Complex + AI",
    "fivemin":  "5 Min Spike",
    "standard": "Daily Standard",
}

# Daily model exit rules — applied to squeeze and autoai positions only
_DAILY_STOP_LOSS_PCT      = 20.0   # hard cut: -20% from entry
_DAILY_TRAIL_ACTIVATE_PCT = 10.0   # trailing stop activates once up +10%
_DAILY_TRAIL_PULLBACK_PP  = 12.0   # exit if price drops 12pp below peak
_DAILY_TIME_STOP_DAYS     = 10     # close after 10 trading days if still negative
_DAILY_STALE_DAYS         = 7      # close after 7 trading days if gain < +3%
_DAILY_STALE_GAIN_PCT     = 3.0    # flatline threshold for staleness stop


def _trading_days_since(opened_at_iso: str) -> int:
    """Count trading days (Mon–Fri) elapsed since the trade was opened."""
    import pytz
    from datetime import timedelta as _td
    try:
        opened = datetime.datetime.fromisoformat(opened_at_iso).date()
        today  = datetime.datetime.now(pytz.timezone("America/New_York")).date()
        days   = 0
        cur    = opened
        while cur < today:
            cur += _td(days=1)
            if cur.weekday() < 5:   # Mon=0 … Fri=4
                days += 1
        return days
    except Exception as _e:
        print(f"TRADING-DAYS: failed to compute days since {opened_at_iso}: {_e}")
        return 0


def _autoclose_take_profit() -> list:
    """
    Money management for all open positions.

    Take-profit (+20%): all positions.
    Risk exits (stop-loss, trailing, time, staleness): daily model positions only
    (trade_mode in squeeze / autoai).  5m positions are managed separately.
    """
    positions    = get_open_positions()
    closed       = []
    take_profits = []
    risk_exits   = []

    for pos in positions:
        symbol   = pos["symbol"]
        entry    = pos["entry_price"]
        trade_id = pos["trade_id"]
        mode     = pos.get("trade_mode") or ""

        # Guard: malformed trade record — skip rather than crash
        if not entry or entry <= 0:
            print(f"MONEY-MGMT: skipping {symbol} trade_id={trade_id} — invalid entry_price {entry!r}")
            continue

        price    = _fetch_current_price(symbol, entry)
        if not price or price <= 0:
            continue

        pnl_pct  = (price - entry) / entry * 100
        tp_target = pos.get("take_profit_pct", 20.0)

        # ── Take-profit (all positions) ──────────────────────────────────
        if pnl_pct >= tp_target:
            result = close_trade(trade_id, price, close_reason="take_profit")
            if result:
                broker_close(symbol)
                data = {
                    "symbol":       symbol,
                    "entry_price":  entry,
                    "exit_price":   round(price, 4),
                    "realized_pnl": result["realized_pnl"],
                    "pnl_pct":      round(pnl_pct, 1),
                }
                closed.append(data)
                take_profits.append(data)
            continue   # no further checks once closed

        # ── Daily model risk exits (squeeze / autoai only) ───────────────
        if mode not in ("squeeze", "autoai"):
            continue

        model_label    = _MODEL_LABELS.get(mode, mode)
        high_watermark = pos.get("high_watermark") or entry
        # entry already validated > 0 above
        peak_pct       = (high_watermark - entry) / entry * 100
        trading_days   = _trading_days_since(pos["opened_at"])
        close_reason   = None

        # Update high watermark using the fetched (closing) price only.
        # Daily model checks use closing prices — intraday lows/highs create false
        # triggers on penny stocks that routinely swing ±15% within a session.
        if price > high_watermark:
            update_high_watermark(trade_id, price)
            peak_pct = pnl_pct

        # 1. Hard stop loss: -20% on closing price
        if pnl_pct <= -_DAILY_STOP_LOSS_PCT:
            close_reason = "stop_loss"

        # 2. Trailing stop: activates at +10% close, exits if close pulls back 12pp from peak close
        elif peak_pct >= _DAILY_TRAIL_ACTIVATE_PCT and pnl_pct <= (peak_pct - _DAILY_TRAIL_PULLBACK_PP):
            close_reason = "trailing_stop"

        # 3. Time stop: 10 trading days and still negative
        elif trading_days >= _DAILY_TIME_STOP_DAYS and pnl_pct < 0:
            close_reason = "time_stop"

        # 4. Staleness stop: 7 trading days and gain < +3% (momentum gone)
        elif trading_days >= _DAILY_STALE_DAYS and pnl_pct < _DAILY_STALE_GAIN_PCT:
            close_reason = "staleness_stop"

        if close_reason:
            result = close_trade(trade_id, price, close_reason=close_reason)
            if result:
                broker_close(symbol)
                data = {
                    "symbol":       symbol,
                    "entry_price":  entry,
                    "exit_price":   round(price, 4),
                    "realized_pnl": result["realized_pnl"],
                    "pnl_pct":      round(pnl_pct, 1),
                    "close_reason": close_reason,
                    "model_label":  model_label,
                }
                closed.append(data)
                risk_exits.append(data)

    if take_profits:
        send_take_profit_alert(take_profits)
    if risk_exits:
        send_exit_alert(risk_exits)

    return closed


def _auto_learn():
    """Re-synthesize hypothesis + weights if new scan outcomes were labeled since last run.
    Called automatically at the end of each morning scan (_scheduled_scan)."""
    try:
        opt_data = get_optimization_data()
        if not opt_data or opt_data["total_trades"] < 5:
            return  # not enough labeled data yet

        current_count = opt_data["total_trades"]
        last_count = get_auto_learn_count()

        if current_count <= last_count:
            print(f"AUTO-LEARN: no new labeled scans ({current_count} total) — skipping")
            return

        print(f"AUTO-LEARN: {current_count - last_count} new labeled scan(s) → updating hypothesis + weights")
        _run_hypothesis_and_weights(get_all_feedback())
        save_auto_learn_count(current_count)
        print("AUTO-LEARN: complete")
    except Exception as e:
        import traceback
        msg = f"AUTO-LEARN: failed — {e}\n{traceback.format_exc()}"
        print(msg)
        _send_telegram_admin(f"<b>⚠️ AUTO-LEARN ERROR</b>\n<code>{str(e)[:300]}</code>")


AUTOAI_CONFIDENCE_HYPOTHESIS = 80   # >= auto-activate hypothesis
AUTOAI_CONFIDENCE_WEIGHTS    = 75   # >= auto-apply weights
AUTOAI_MIN_CLOSED_TRADES     = 10   # minimum closed trades before weights auto-apply
AUTOAI_MIN_NEW_OUTCOMES      = 5    # new outcome-labeled scans needed to trigger a run
AUTOAI_MAX_WEIGHT_DRIFT      = 0.40 # cap any single weight to ±40% of its default

# --- Chat rate limiting (in-process, per user) ---
import collections as _collections
_chat_rate: dict[str, _collections.deque] = {}
_CHAT_RATE_WINDOW = 60      # seconds
_CHAT_RATE_MAX    = 20      # messages per window

# Global Claude API rate limiter — caps total API calls across all users/sources
# to prevent runaway spend from multiple sessions or scheduled tasks firing simultaneously.
_global_claude_calls: _collections.deque = _collections.deque()
_GLOBAL_CLAUDE_MAX    = 60   # max Claude API calls per minute across all users
_GLOBAL_CLAUDE_WINDOW = 60   # seconds

def _global_claude_allowed() -> bool:
    """Returns False if the global Claude API rate limit is exceeded."""
    now = _time.monotonic()
    while _global_claude_calls and now - _global_claude_calls[0] > _GLOBAL_CLAUDE_WINDOW:
        _global_claude_calls.popleft()
    if len(_global_claude_calls) >= _GLOBAL_CLAUDE_MAX:
        print(f"CLAUDE RATE LIMIT: global cap of {_GLOBAL_CLAUDE_MAX} calls/min reached — request dropped")
        return False
    _global_claude_calls.append(now)
    return True

def _chat_allowed(username: str) -> bool:
    now = _time.monotonic()
    q = _chat_rate.setdefault(username, _collections.deque())
    while q and now - q[0] > _CHAT_RATE_WINDOW:
        q.popleft()
    if len(q) >= _CHAT_RATE_MAX:
        return False
    q.append(now)
    return True


def _build_hypothesis_text_from_autoai(result: dict) -> str:
    """Convert autonomous_optimize() JSON hypotheses list into the existing ## Hypotheses text format."""
    hyps = result.get("hypotheses", [])
    if not hyps:
        return ""
    lines = ["## Hypotheses (Auto AI)\n"]
    for i, h in enumerate(hyps, 1):
        conf = h.get("confidence", 0)
        lines.append(f"{i}. [{conf}% confidence] {h['text']}\n")
    rationale = result.get("rationale", "")
    if rationale:
        lines.append(f"\n## Agent Context\n{rationale}")
    return "\n".join(lines)


def _auto_ai_optimize():
    """
    Autonomous self-improvement loop for Auto AI mode.
    Triggered after each morning scan. Requires >= 5 new outcome-labeled scans since last run.
    Auto-activates hypotheses >= 80 confidence.
    Auto-applies weights if confidence >= 75 AND >= 10 closed trades.
    Logs everything to autoai_log.
    """
    try:
        opt_data = get_optimization_data()
        if not opt_data or opt_data.get("total_trades", 0) < 5:
            print("AUTO-AI: insufficient data — skipping")
            return

        current_count = opt_data["total_trades"]
        last_count    = get_last_autoai_run_count()
        new_outcomes  = current_count - last_count

        if new_outcomes < AUTOAI_MIN_NEW_OUTCOMES:
            print(f"AUTO-AI: only {new_outcomes} new outcomes — skipping (need {AUTOAI_MIN_NEW_OUTCOMES})")
            return

        print(f"AUTO-AI: {new_outcomes} new outcomes since last run — starting autonomous optimization")

        all_feedback    = get_all_feedback()
        # Include all active rules (admin-approved + Auto AI auto-applied) as prior context.
        prior_text      = get_active_hypothesis_text(mode="autoai")
        autoai_w_data   = get_autoai_weights()
        current_weights = autoai_w_data["weights"] if autoai_w_data else DEFAULT_SQUEEZE_WEIGHTS.copy()
        all_trades      = get_trade_history()
        closed_trades   = [t for t in all_trades if t.get("exit_price") is not None]

        all_rules = get_hypothesis_rules()
        result = autonomous_optimize(
            opt_data, all_feedback, current_weights,
            prior_text, closed_trades[-20:],
            hypothesis_rules=all_rules,
        )

        if "error" in result and not result.get("hypotheses"):
            print(f"AUTO-AI: optimization failed — {result['error']}")
            return

        # XGBoost auto-transition: if enough labeled live data exists, use ML weights
        # instead of Claude's weight suggestions (Claude still runs for hypotheses above)
        xgb_weights = train_xgb_weights()
        if xgb_weights:
            print(f"AUTO-AI: XGBoost model active — using ML-derived weights")
            new_weights = xgb_weights
        else:
            # Guardrail: allow full disabling (floor=0), cap upward drift at default+40%
            new_weights = result.get("weights", current_weights)
        for key, default_val in DEFAULT_SQUEEZE_WEIGHTS.items():
            if key not in new_weights:
                # Optional criteria (default=0) stay 0 if AI didn't mention them
                new_weights[key] = default_val
            if default_val == 0:
                # Optional criterion: AI can enable up to 20 pts, or leave at 0
                new_weights[key] = max(0, min(20, int(new_weights[key])))
            else:
                # Existing criterion: allow full disable (0) or cap at default+40%
                max_val = int(default_val * (1 + AUTOAI_MAX_WEIGHT_DRIFT))
                new_weights[key] = max(0, min(max_val, int(new_weights[key])))

        # Safety guardrail: ensure at least one rel_vol tier and one shares tier stay > 0
        # (zeroing all of them would make the scorer useless for momentum detection)
        rel_vol_keys = ["rel_vol_50x", "rel_vol_25x", "rel_vol_10x", "rel_vol_5x"]
        shares_keys  = ["shares_lt10m", "shares_lt30m", "shares_lt100m"]
        if max(new_weights.get(k, 0) for k in rel_vol_keys) == 0:
            new_weights["rel_vol_10x"] = DEFAULT_SQUEEZE_WEIGHTS["rel_vol_10x"]
            print("AUTO-AI: guardrail restored rel_vol_10x (all rel_vol tiers were zeroed)")
        if max(new_weights.get(k, 0) for k in shares_keys) == 0:
            new_weights["shares_lt10m"] = DEFAULT_SQUEEZE_WEIGHTS["shares_lt10m"]
            print("AUTO-AI: guardrail restored shares_lt10m (all shares tiers were zeroed)")

        hypotheses_auto    = 0
        hypotheses_pending = 0

        # Save hypotheses with confidence; auto-activate high-confidence ones
        raw_hyps = result.get("hypotheses", [])
        if raw_hyps:
            # Save Auto AI's synthesized hypothesis to its OWN key so it never
            # overwrites the admin-curated Complex+AI hypothesis blob.
            hyp_text = _build_hypothesis_text_from_autoai(result)
            if hyp_text:
                save_autoai_hypothesis_blob(hyp_text, len(all_feedback) + current_count)

            rules_to_save = []
            for h in raw_hyps:
                conf = h.get("confidence", 0)
                if conf < 50:
                    continue
                auto = conf >= AUTOAI_CONFIDENCE_HYPOTHESIS
                rules_to_save.append({
                    "text":             h["text"],
                    "source":           h.get("source", ""),
                    "confidence_score": conf,
                    "auto_applied":     1 if auto else 0,
                    "status":           "active" if auto else "pending",
                })
                if auto:
                    hypotheses_auto += 1
                else:
                    hypotheses_pending += 1

            if rules_to_save:
                save_hypothesis_rules_with_confidence(rules_to_save)

        # Auto-apply weights if confidence threshold met
        weights_auto_applied = 0
        weight_conf          = result.get("weight_confidence", 0)
        num_closed           = len(closed_trades)

        if weight_conf >= AUTOAI_CONFIDENCE_WEIGHTS and num_closed >= AUTOAI_MIN_CLOSED_TRADES:
            save_autoai_weights(
                new_weights,
                result.get("rationale", ""),
                result.get("suggestions", []),
                f"[Auto AI] {result.get('summary', '')}",
            )
            save_weight_changelog(
                f"[Auto AI] {result.get('summary', '')}",
                result.get("rationale", ""),
                new_weights,
            )
            weights_auto_applied = 1
            print(f"AUTO-AI: weights auto-applied (confidence={weight_conf}, closed_trades={num_closed})")
        else:
            print(f"AUTO-AI: weights NOT applied — confidence={weight_conf}, closed_trades={num_closed}")

        save_autoai_log({
            "ran_at":                    datetime.datetime.utcnow().isoformat(),
            "trigger_reason":            f"{new_outcomes} new outcome-labeled scans",
            "trades_evaluated":          current_count,
            "hypotheses_added":          len(raw_hyps),
            "hypotheses_auto_activated": hypotheses_auto,
            "hypotheses_to_pending":     hypotheses_pending,
            "weights_auto_applied":      weights_auto_applied,
            "weight_confidence":         weight_conf,
            "summary":                   result.get("summary", ""),
            "full_response":             __import__("json").dumps(result),
        })
        save_last_autoai_run_count(current_count)

        summary_msg = result.get("summary", "autonomous optimization cycle complete")
        _send_telegram_admin(
            f"<b>Auto AI Optimization</b>\n"
            f"{hypotheses_auto} rules auto-activated · {hypotheses_pending} to review · "
            f"Weights: {'applied' if weights_auto_applied else 'skipped'}\n"
            f"<i>{summary_msg}</i>"
        )
        print(f"AUTO-AI: complete — {hypotheses_auto} auto-activated, "
              f"{hypotheses_pending} pending, weights={'applied' if weights_auto_applied else 'skipped'}")
    except Exception as e:
        import traceback
        msg = f"AUTO-AI: failed — {e}\n{traceback.format_exc()}"
        print(msg)
        _send_telegram_admin(f"<b>⚠️ AUTO-AI ERROR</b>\n<code>{str(e)[:300]}</code>")


def _enrich_high_scorers(results: list, mode: str = None, scan_id_map: dict = None) -> list:
    """
    For each stock scoring >= 50, add lstm_prob and ai_trade_call to the result dict.
    Called during scheduled and intraday scans so alerts carry full AI context.
    Runs in parallel (max 4 workers); gracefully skips on error.
    mode is passed through to get_active_hypothesis_text() so Auto AI uses its own
    rules and Complex+AI is fully isolated from Auto AI auto-applied rules.
    scan_id_map: {symbol: scan_id} from save_scan() — when provided, persists
    lstm_prob and ai_trade_rec to the DB so outcomes can be back-analyzed later.
    """
    from concurrent.futures import ThreadPoolExecutor

    hypothesis_text  = get_active_hypothesis_text(mode=mode)
    sizing_stats     = get_sizing_stats()
    ai_accuracy      = get_ai_decision_accuracy()
    _modes           = _FIVEMIN_MODES if mode == "fivemin" else _DAILY_MODES
    _per_signal      = get_per_signal_stats(modes=_modes)
    high_scorers     = [r for r in results if r.get("score", 0) >= 50]

    def _enrich_one(stock):
        try:
            checklist = stock.get("checklist", {})
            # LSTM: fivemin gated at 75; daily models gated at 50 (matches AI enrichment threshold)
            lstm_prob = None
            lstm_threshold = 75 if mode == "fivemin" else 50
            if stock.get("score", 0) >= lstm_threshold:
                if mode == "fivemin":
                    lstm_prob = predict_5m_hit_probability(
                        stock["symbol"],
                        shares_outstanding=checklist.get("shares_outstanding"),
                        sector=checklist.get("sector"),
                    )
                else:
                    lstm_prob = predict_hit_probability(
                        stock["symbol"],
                        shares_outstanding=checklist.get("shares_outstanding"),
                        sector=checklist.get("sector"),
                    )
            stock["lstm_prob"] = lstm_prob
            ticker_history = get_ticker_scan_history(stock["symbol"])
            news = get_stock_news(stock["symbol"])
            stock["ai_trade_call"] = recommend_trade(
                stock, hypothesis_text, sizing_stats, ticker_history,
                lstm_prob=lstm_prob, news_headlines=news,
                ai_accuracy=ai_accuracy, per_signal_stats=_per_signal
            )
            # Persist lstm_prob + ai_trade_rec to DB so outcomes can be back-analyzed
            if scan_id_map:
                scan_id = scan_id_map.get(stock.get("symbol"))
                if scan_id:
                    try:
                        tc = stock["ai_trade_call"]
                        update_scan_ai_rec(scan_id, tc["decision"], tc["confidence"],
                                           tc["rationale"], lstm_prob=lstm_prob)
                    except Exception as _db_e:
                        print(f"ENRICH: failed to persist ai_rec for {stock['symbol']}: {_db_e}")
        except Exception as e:
            print(f"ENRICH: {stock['symbol']} failed — {e}")

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(_enrich_one, high_scorers))

    return results


_LSTM_GATE = 0.55   # minimum LSTM probability for alerts and paper trades
_LSTM_NONE_MIN_SCORE = 75  # if LSTM unavailable, require this score floor instead


def _passes_lstm_gate(r: dict) -> bool:
    """
    Returns True if the stock passes the LSTM quality gate:
      - lstm_prob >= 55%  (model agrees the setup is likely to hit)
      - OR lstm_prob is None AND score >= 75  (no model data but very high scorer)
    Stocks with lstm_prob < 55% or (None + score < 75) are filtered out.
    """
    lstm = r.get("lstm_prob")
    score = r.get("score", 0)
    if lstm is None:
        return score >= _LSTM_NONE_MIN_SCORE
    return lstm >= _LSTM_GATE


def _auto_paper_trade(results: list, today_str: str, mode: str = "squeeze") -> set:
    """
    Open paper trades for TRADE calls (HIGH or MEDIUM confidence) found in enriched scan results.
    No daily cap — available cash is the natural gate ($500 min per trade).
    Skips symbols already in open positions.
    Mirrors each trade to Alpaca paper account if configured.
    Sends a Telegram notification for each auto-trade opened.
    Returns set of symbols where a trade was actually opened (for accurate alert badges).
    """
    traded = set()

    try:
        open_symbols = {p["symbol"] for p in get_open_positions()}
    except Exception as e:
        print(f"AUTO-TRADE: could not fetch open positions — skipping to avoid duplicates: {e}")
        return traded
    try:
        portfolio = get_portfolio_summary()
        available = portfolio["cash"]
    except Exception as e:
        print(f"AUTO-TRADE: could not fetch portfolio — skipping: {e}")
        return traded

    for r in results:
        tc = r.get("ai_trade_call") or {}
        if tc.get("decision") != "TRADE" or tc.get("confidence") not in ("HIGH", "MEDIUM"):
            continue
        if not _passes_lstm_gate(r):
            print(f"AUTO-TRADE: {r.get('symbol')} skipped — LSTM {r.get('lstm_prob')} below 55% gate")
            continue
        symbol = r["symbol"]
        if symbol in open_symbols:
            continue
        if available < 500:
            continue

        position_size = r.get("ai_rec", {}).get("amount", 1000)
        entry_price   = _fetch_current_price(symbol, r.get("price", 0))
        confidence    = tc.get("confidence", "")
        result = open_trade(
            symbol, entry_price, position_size,
            f"auto-trade: {mode} | {confidence} confidence AI call",
            take_profit_pct=20.0,
            stop_loss_pct=_DAILY_STOP_LOSS_PCT,
            trade_mode=mode,
        )
        if result:
            traded.add(symbol)
            open_symbols.add(symbol)
            available -= position_size

            # Mirror to Alpaca paper account (pass price for whole-share fallback)
            alpaca_order = submit_market_order(symbol, position_size, price=entry_price)
            if alpaca_order:
                qty_info = alpaca_order.get("qty") or alpaca_order.get("notional", "")
                alpaca_str = f"Alpaca ✓ ({qty_info})"
            elif broker_configured():
                alpaca_str = "Alpaca rejected — check Render logs for reason"
            else:
                alpaca_str = "Paper only (Alpaca not configured)"

            _send_telegram_admin(
                f"<b>Auto-Trade Opened: {symbol}</b>\n"
                f"Model: {_MODEL_LABELS.get(mode, mode)} · Entry: ${entry_price:.4f} · Score: {r['score']}/100 · ${position_size} · Target +20%\n"
                f"{alpaca_str}\n"
                f"<i>{tc.get('rationale', '')}</i>"
            )
            print(f"AUTO-TRADE: opened {symbol} at ${entry_price:.4f} "
                  f"(score={r['score']}, conf={confidence}, size=${position_size}, {alpaca_str})")

    return traded


def _scheduled_scan():
    """Auto-run all scan modes at 9:45am ET Mon–Fri and refresh cache."""
    global _alerted_today, _alerted_date
    print("SCHEDULER: Running scheduled morning scan...")

    et_now    = datetime.datetime.now(pytz.timezone("America/New_York"))
    today_str = et_now.date().isoformat()

    # Reset daily dedup set on each new trading day — re-seed from DB to survive restarts
    if today_str != _alerted_date:
        _alerted_today = get_alerted_symbols_today(today_str)
        _alerted_date  = today_str

    for mode in ["autoai", "squeeze", "strict", "standard"]:
        try:
            data = run_scan(mode=mode)
            scan_ids = save_scan(data["results"], mode)
            save_scan_candidates(data["results"], mode)
            save_scan_cache(mode, data["results"], data["summary"])
            print(f"SCHEDULER: {mode} scan complete ({len(data['results'])} results)")
            if mode in ("autoai", "squeeze"):
                label = "Auto AI" if mode == "autoai" else "Complex + AI"
                # Enrich score >= 50 with AI calls + LSTM before alerting; persist to DB
                _enrich_high_scorers(data["results"], mode=mode, scan_id_map=scan_ids)
                # Auto paper-trade FIRST so Alpaca order is queued before Telegram fires
                traded = _auto_paper_trade(data["results"], today_str)
                # Alert TRADE calls that pass LSTM quality gate (55%+ or None+score>=75)
                gated_alerts = [r for r in data["results"] if _passes_lstm_gate(r)]
                send_scan_alert(gated_alerts, label, min_score=0, ai_trade_only=True, traded_symbols=traded)
                for r in gated_alerts:
                    if (r.get("ai_trade_call") or {}).get("decision") == "TRADE":
                        _alerted_today.add(r.get("symbol"))
                # Log near-misses (40-74) to watchlist for intraday re-checking
                for r in data["results"]:
                    score = r.get("score", 0)
                    if 40 <= score < 75:
                        add_near_miss_to_watchlist(r["symbol"], score, r.get("price"))
        except Exception as e:
            print(f"SCHEDULER: {mode} scan failed — {e}")
            _send_telegram_admin(f"⚠️ <b>Morning scan failed ({mode})</b>\n{e}")
    _autoclose_take_profit()
    update_returns()
    _auto_learn()
    _auto_ai_optimize()


def _premarket_scan():
    """Pre-market scan at 8:30am ET — squeeze mode with relaxed Finviz filters."""
    print("SCHEDULER: Running pre-market scan...")
    today_str = datetime.datetime.now(pytz.timezone("America/New_York")).date().isoformat()
    try:
        data = run_scan(mode="squeeze", premarket=True)
        scan_ids = save_scan(data["results"], "squeeze")   # persist for hypothesis testing
        save_scan_cache("squeeze", data["results"], data["summary"])
        _enrich_high_scorers(data["results"], mode="squeeze", scan_id_map=scan_ids)
        gated = [r for r in data["results"] if _passes_lstm_gate(r)]
        traded = _auto_paper_trade(gated, today_str)
        send_scan_alert(gated, "Complex + AI (pre-market)", min_score=0, ai_trade_only=True, traded_symbols=traded)
        print(f"SCHEDULER: pre-market scan complete ({len(data['results'])} results)")
    except Exception as e:
        print(f"SCHEDULER: pre-market scan failed — {e}")
        _send_telegram_admin(f"⚠️ <b>Pre-market scan failed</b>\n{e}")


def _run_validation():
    """
    Validate that our scanner data matches FinViz live quote pages.
    Runs before market open (9:00 AM ET) and every 15 min in the last
    hour before close (3:00–3:45 PM ET).  Any discrepancy is flagged.
    """
    print("VALIDATOR: Starting data validation run...")
    try:
        # Use the squeeze scan cache (most recent scan) as the source of truth
        # for what our system currently shows.
        cached = get_scan_cache("squeeze", max_age_minutes=120)
        if not cached or not cached.get("results"):
            print("VALIDATOR: No recent scan cache found — skipping.")
            return
        report = validate_scan_results(cached["results"], max_symbols=15)
        save_validation_report(report)
        print(
            f"VALIDATOR: Done — {report['warn_count']} mismatches "
            f"out of {report['symbols_checked']} symbols."
        )
    except Exception as e:
        print(f"VALIDATOR: run failed — {e}")


def _daily_backfill():
    """
    Runs automatically at 6:00 AM ET every weekday.
    Incremental backfill: picks up any new tickers from yesterday's scans,
    retrains LSTM on the growing dataset, then re-syncs hypothesis + weights.
    Skipped if a manual backfill is already running.
    """
    global _backfill_running
    if _backfill_running:
        print("DAILY BACKFILL: skipped — manual run already in progress")
        return
    _backfill_running = True
    print("DAILY BACKFILL: starting scheduled run")
    try:
        clear_old_watchlist(days_old=2)   # prune stale near-miss entries
        weights_data = get_squeeze_weights()
        weights = weights_data["weights"] if weights_data else None
        build_historical_dataset(weights=weights)
        try:
            lstm_stats = train_lstm()
            print(f"DAILY BACKFILL: LSTM retrained — {lstm_stats}")
        except Exception as e:
            print(f"DAILY BACKFILL: LSTM training skipped — {e}")
            _send_telegram_admin(f"⚠️ <b>LSTM retrain failed</b>\n{e}")
        _run_hypothesis_and_weights(get_all_feedback())
        print("DAILY BACKFILL: complete")
    except Exception as e:
        print(f"DAILY BACKFILL: error — {e}")
        _send_telegram_admin(f"⚠️ <b>Daily backfill failed</b>\n{e}")
    finally:
        _backfill_running = False


def _intraday_scan():
    """
    Squeeze scan every 30 min during market hours (10:00–15:30 ET Mon–Fri).
    Alerts only on NEW high-scorers not already alerted today.
    Near-misses (40-74) are added to the watchlist for 15-min re-checking.
    """
    global _alerted_today, _alerted_date

    et_now    = datetime.datetime.now(pytz.timezone("America/New_York"))
    today_str = et_now.date().isoformat()

    # Safety: re-seed from DB if we somehow missed the morning reset
    if today_str != _alerted_date:
        _alerted_today = get_alerted_symbols_today(today_str)
        _alerted_date  = today_str

    print(f"INTRADAY SCAN: running at {et_now.strftime('%H:%M')} ET...")
    try:
        data = run_scan(mode="squeeze")
        scan_ids = save_scan(data["results"], "squeeze")
        save_scan_cache("squeeze", data["results"], data["summary"])

        # Also save Auto AI cache (uses its own weights; same raw results for intraday)
        try:
            data_autoai = run_scan(mode="autoai")
            save_scan(data_autoai["results"], "autoai")
            save_scan_cache("autoai", data_autoai["results"], data_autoai["summary"])
        except Exception as ae:
            print(f"INTRADAY SCAN: autoai cache update failed — {ae}")

        # Enrich score >= 50 stocks not yet alerted; persist lstm_prob + ai_trade_rec to DB
        new_candidates = [
            r for r in data["results"]
            if r.get("symbol") not in _alerted_today
        ]
        if new_candidates:
            _enrich_high_scorers(data["results"], mode="squeeze", scan_id_map=scan_ids)
        # Alert AI TRADE calls that pass the LSTM quality gate
        new_alerts = [
            r for r in data["results"]
            if (r.get("ai_trade_call") or {}).get("decision") == "TRADE"
            and r.get("symbol") not in _alerted_today
            and _passes_lstm_gate(r)
        ]
        if new_alerts:
            traded = _auto_paper_trade(new_alerts, today_str)
            send_scan_alert(new_alerts, f"Intraday {et_now.strftime('%H:%M')}",
                            min_score=0, ai_trade_only=True, traded_symbols=traded)
            for r in new_alerts:
                _alerted_today.add(r.get("symbol"))

        # Check take-profit on open positions every intraday cycle
        _autoclose_take_profit()

        # Add near-misses to watchlist
        for r in data["results"]:
            score = r.get("score", 0)
            if 40 <= score < 75:
                add_near_miss_to_watchlist(r["symbol"], score, r.get("price"))

        print(
            f"INTRADAY SCAN: {len(data['results'])} results, "
            f"{len(new_alerts)} new alert(s)"
        )
    except Exception as e:
        print(f"INTRADAY SCAN: failed — {e}")
        _send_telegram_admin(f"⚠️ <b>Intraday scan failed</b>\n{e}")


def _fivemin_spike_scan():
    """
    5m Spike scan — runs every 5 minutes 10:00–15:30 ET Mon–Fri.
    Detects stocks with current 5-min bar volume >= 10x their 10-day avg at
    this time slot. Alerts + auto paper-trades new high scorers (>= 75).
    """
    global _alerted_today, _alerted_date

    et_now    = datetime.datetime.now(pytz.timezone("America/New_York"))
    today_str = et_now.date().isoformat()

    # Hard stop at 15:30 ET
    if et_now.hour > 15 or (et_now.hour == 15 and et_now.minute > 30):
        return

    if today_str != _alerted_date:
        _alerted_today = get_alerted_symbols_today(today_str)
        _alerted_date  = today_str

    print(f"5M SPIKE SCAN: running at {et_now.strftime('%H:%M')} ET...")
    try:
        data = run_scan_5m()
        scan_ids = save_scan(data["results"], "fivemin")
        # Save ALL screened candidates (including low-score) for hypothesis observation pool
        save_scan_candidates(data["results"], "fivemin")
        save_scan_cache("fivemin", data["results"], data["summary"])

        new_alerts = [
            r for r in data["results"]
            if r.get("score", 0) >= 75 and r.get("symbol") not in _alerted_today
        ]
        if new_alerts:
            _enrich_high_scorers(new_alerts, mode="fivemin", scan_id_map=scan_ids)
            gated_5m = [r for r in new_alerts if _passes_lstm_gate(r)]
            traded = _auto_paper_trade(gated_5m, today_str, mode="fivemin")
            send_scan_alert(gated_5m, f"5m Spike {et_now.strftime('%H:%M')}", traded_symbols=traded)
            for r in new_alerts:  # mark all as alerted regardless of gate
                _alerted_today.add(r.get("symbol"))

        # Check take-profit on open positions every 5m cycle
        _autoclose_take_profit()

        print(
            f"5M SPIKE SCAN: {len(data['results'])} results, "
            f"{len(new_alerts)} new alert(s)"
        )
    except Exception as e:
        print(f"5M SPIKE SCAN: failed — {e}")
        _send_telegram_admin(f"⚠️ <b>5m spike scan failed</b>\n{e}")


def _check_watchlist():
    """
    Re-score today's near-miss watchlist symbols every 15 min.
    Sends an alert (email + Telegram) if any symbol rises to score >= 75.
    """
    active = get_active_watchlist(today_only=True)
    if not active:
        return

    symbols      = [w["symbol"] for w in active]
    symbol_scores = {w["symbol"]: w.get("score", 0) for w in active}
    print(f"WATCHLIST CHECK: re-scoring {len(symbols)} near-miss symbol(s)...")

    try:
        live_quotes = get_finviz_quotes(symbols, max_workers=5)

        raw = yf.download(
            symbols if len(symbols) > 1 else symbols[0],
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
        )

        dfs = {}
        if len(symbols) == 1:
            if not raw.empty:
                dfs[symbols[0]] = raw.copy()
        else:
            for sym in symbols:
                try:
                    sym_df = raw[sym].copy()
                    if not sym_df.empty:
                        dfs[sym] = sym_df
                except Exception as _e:
                    print(f"WATCHLIST: failed to extract data for {sym}: {_e}")

        weights_data   = get_squeeze_weights()
        active_weights = weights_data["weights"] if weights_data else DEFAULT_SQUEEZE_WEIGHTS

        for symbol in symbols:
            if symbol not in dfs:
                continue
            try:
                df           = prepare_dataframe(dfs[symbol])
                fundamentals = {}
                q            = live_quotes.get(symbol, {})
                if q.get("rel_volume") is not None:
                    fundamentals["finviz_relvol"] = q["rel_volume"]
                if q.get("price") is not None:
                    fundamentals["finviz_price"] = q["price"]
                if q.get("change_pct") is not None:
                    fundamentals["finviz_change_pct"] = q["change_pct"]
                if q.get("volume") is not None:
                    fundamentals["finviz_volume"] = q["volume"]
                if q.get("shares_outstanding") is not None:
                    fundamentals["shares_outstanding"] = q["shares_outstanding"]
                if q.get("float_shares") is not None:
                    fundamentals["float_shares"] = q["float_shares"]
                if q.get("institution_pct") is not None:
                    fundamentals["institution_pct"] = q["institution_pct"]

                result = score_stock_squeeze(symbol, df, fundamentals, weights=active_weights)
                if result and result.get("score", 0) >= 75:
                    old_score = symbol_scores.get(symbol, 0)
                    send_watchlist_alert(
                        symbol, result["score"], result.get("price"), old_score
                    )
                    mark_watchlist_alerted(symbol)
                    _alerted_today.add(symbol)
                    print(
                        f"WATCHLIST: {symbol} rose to {result['score']} "
                        f"(from {old_score}) — alert sent"
                    )
            except Exception as e:
                print(f"WATCHLIST CHECK: error scoring {symbol} — {e}")

    except Exception as e:
        print(f"WATCHLIST CHECK: failed — {e}")


def _weekly_analysis():
    """
    Runs every Monday at 7:00 AM ET.
    Queries the DB for the 8 standard weekly metrics, formats a Telegram report,
    and sends proposed weight changes for human approval — no auto-apply.
    """
    import sqlite3 as _sq
    db_path = _os.environ.get("DB_PATH", "scan_history.db")
    if not _os.path.exists(db_path):
        print("WEEKLY ANALYSIS: DB not found, skipping")
        return
    try:
        conn = _sq.connect(db_path)
        c = conn.cursor()

        # Baseline
        baseline_n, baseline_hit = c.execute(
            "SELECT COUNT(*), ROUND(100.0*SUM(CASE WHEN next_day_return>=20 THEN 1 ELSE 0 END)/COUNT(*),1) "
            "FROM scans WHERE next_day_return IS NOT NULL"
        ).fetchone()

        # Score buckets
        score_rows = c.execute(
            "SELECT CASE WHEN score<45 THEN '0-44' WHEN score<55 THEN '45-54' "
            "WHEN score<65 THEN '55-64' WHEN score<75 THEN '65-74' "
            "WHEN score<85 THEN '75-84' ELSE '85+' END as b, "
            "COUNT(*), ROUND(100.0*SUM(CASE WHEN next_day_return>=20 THEN 1 ELSE 0 END)/COUNT(*),1) "
            "FROM scans WHERE next_day_return IS NOT NULL GROUP BY b ORDER BY b"
        ).fetchall()

        # AI TRADE precision
        ai_row = c.execute(
            "SELECT COUNT(*), SUM(CASE WHEN next_day_return>=20 THEN 1 ELSE 0 END) "
            "FROM scans WHERE next_day_return IS NOT NULL "
            "AND ai_trade_rec LIKE '%TRADE%' AND ai_trade_rec NOT LIKE '%NO_TRADE%'"
        ).fetchone()
        ai_total, ai_hits = ai_row
        ai_pct = round(100.0 * ai_hits / ai_total, 1) if ai_total else 0

        # Relvol tiers
        rv_rows = c.execute(
            "SELECT CASE WHEN relative_volume>=500 THEN '500x+' WHEN relative_volume>=100 THEN '100-499x' "
            "WHEN relative_volume>=50 THEN '50-99x' WHEN relative_volume>=25 THEN '25-49x' "
            "WHEN relative_volume>=10 THEN '10-24x' ELSE '<10x' END as t, "
            "COUNT(*), ROUND(100.0*SUM(CASE WHEN next_day_return>=20 THEN 1 ELSE 0 END)/COUNT(*),1) "
            "FROM scans WHERE next_day_return IS NOT NULL AND relative_volume IS NOT NULL "
            "GROUP BY t ORDER BY MIN(relative_volume) DESC"
        ).fetchall()

        # Daily gain buckets
        gain_rows = c.execute(
            "SELECT CASE WHEN today_return>50 THEN '>50%' WHEN today_return>30 THEN '30-50%' "
            "WHEN today_return>20 THEN '20-30%' WHEN today_return>10 THEN '10-20%' ELSE '<10%' END as g, "
            "COUNT(*), ROUND(100.0*SUM(CASE WHEN next_day_return>=20 THEN 1 ELSE 0 END)/COUNT(*),1) "
            "FROM scans WHERE next_day_return IS NOT NULL AND today_return IS NOT NULL "
            "GROUP BY g ORDER BY MIN(today_return) DESC"
        ).fetchall()

        # Day of week
        dow_rows = c.execute(
            "SELECT CASE strftime('%w',timestamp) WHEN '1' THEN 'Mon' WHEN '2' THEN 'Tue' "
            "WHEN '3' THEN 'Wed' WHEN '4' THEN 'Thu' WHEN '5' THEN 'Fri' END as d, "
            "COUNT(*), ROUND(100.0*SUM(CASE WHEN next_day_return>=20 THEN 1 ELSE 0 END)/COUNT(*),1) "
            "FROM scans WHERE next_day_return IS NOT NULL GROUP BY d ORDER BY d"
        ).fetchall()

        # Per-signal stats (top 5 + bottom 5, min 5 fires)
        sig_rows = c.execute(
            "SELECT key, COUNT(*) as n, "
            "ROUND(100.0*SUM(CASE WHEN next_day_return>=20 THEN 1 ELSE 0 END)/COUNT(*),1) as hit "
            "FROM (SELECT s.next_day_return, je.key FROM scans s, json_each(s.signals_json) je "
            "WHERE s.next_day_return IS NOT NULL AND s.signals_json IS NOT NULL "
            "AND s.signals_json!='{\"\":\"\"}' AND je.value=1) "
            "GROUP BY key HAVING n>=5 ORDER BY hit DESC"
        ).fetchall()

        # XGBoost eligibility
        xgb_n = c.execute(
            "SELECT COUNT(*) FROM scans WHERE next_day_return IS NOT NULL "
            "AND signals_json IS NOT NULL AND signals_json!='{}'"
        ).fetchone()[0]

        # New labeled rows this week
        new_labeled = c.execute(
            "SELECT COUNT(*) FROM scans WHERE next_day_return IS NOT NULL "
            "AND timestamp >= date('now','-7 days')"
        ).fetchone()[0]

        conn.close()

        # --- Format Telegram message ---
        lines = ["&#128200; <b>Weekly Model Analysis</b>",
                 f"Labeled: {baseline_n} rows | Baseline hit rate: {baseline_hit}%",
                 f"XGBoost eligible: {xgb_n}/500 | New labeled this week: {new_labeled}",
                 ""]

        lines.append("<b>Score Buckets</b>")
        for b, n, hit in score_rows:
            conf = "H" if n >= 30 else ("M" if n >= 10 else "L")
            lines.append(f"  {b}: {hit}% ({n} rows, {conf})")

        lines.append(f"\n<b>AI TRADE Precision</b>: {ai_pct}% ({ai_hits}/{ai_total} calls)")

        lines.append("\n<b>Relvol Tiers</b>")
        for t, n, hit in rv_rows:
            conf = "H" if n >= 30 else ("M" if n >= 10 else "L")
            lines.append(f"  {t}: {hit}% (n={n}, {conf})")

        lines.append("\n<b>Daily Gain Buckets</b>")
        for g, n, hit in gain_rows:
            conf = "H" if n >= 30 else ("M" if n >= 10 else "L")
            lines.append(f"  {g}: {hit}% (n={n}, {conf})")

        lines.append("\n<b>Day of Week</b>")
        for d, n, hit in dow_rows:
            if d:
                lines.append(f"  {d}: {hit}% (n={n})")

        if sig_rows:
            top5 = [r for r in sig_rows[:5]]
            bot5 = [r for r in sig_rows[-5:] if r not in top5]
            lines.append("\n<b>Top Signals</b>")
            for k, n, hit in top5:
                vs = round(hit - baseline_hit, 1)
                sign = "+" if vs >= 0 else ""
                lines.append(f"  {k}: {hit}% ({sign}{vs}pp, n={n})")
            lines.append("<b>Bottom Signals</b>")
            for k, n, hit in bot5:
                vs = round(hit - baseline_hit, 1)
                lines.append(f"  {k}: {hit}% ({vs}pp, n={n})")

        # Check cooldown: no proposals within 7 days of last weight change
        last_changed_row = conn.execute(
            "SELECT value FROM settings WHERE key='squeeze_weights_last_changed'"
        ).fetchone() if False else None  # conn already closed above — re-open
        _conn2 = _sq.connect(db_path)
        last_changed_str = (_conn2.execute(
            "SELECT value FROM settings WHERE key='squeeze_weights_last_changed'"
        ).fetchone() or [None])[0]
        _conn2.close()

        cooldown_active = False
        if last_changed_str:
            try:
                last_dt = datetime.datetime.fromisoformat(last_changed_str)
                days_since = (datetime.datetime.utcnow() - last_dt).days
                if days_since < 7:
                    cooldown_active = True
                    lines.append(f"\n&#128274; Weight change cooldown active ({days_since}d since last change — need 7d). No proposals this week.")
            except Exception as _e:
                print(f"WEEKLY: failed to compute cooldown: {_e}")

        # Proposed changes: require n≥100 AND no cooldown AND ≥5pp deviation
        proposals = []
        if not cooldown_active:
            for k, n, hit in sig_rows:
                if n >= 100:
                    diff = hit - baseline_hit
                    if diff <= -5:
                        proposals.append(f"  REDUCE weight: {k} ({hit}% hit, {diff:+.1f}pp vs baseline, n={n})")
                    elif diff >= 5:
                        proposals.append(f"  BOOST weight: {k} ({hit}% hit, {diff:+.1f}pp vs baseline, n={n})")
            if proposals:
                lines.append("\n&#9888;&#65039; <b>Proposed Changes (awaiting your approval)</b>")
                lines.append("  Rules: n≥100 labeled rows per signal, ≥5pp vs baseline, 7-day cooldown since last change.")
                lines += proposals
                lines.append("Reply 'approve weekly changes' to apply, or review analytics page.")
            else:
                lines.append("\nNo high-confidence weight changes proposed this week (n≥100 threshold not met or deviation <5pp).")

        tg_text = "\n".join(lines)
        _send_telegram_admin(tg_text)
        print("WEEKLY ANALYSIS: report sent to Telegram")

        # Build HTML email (plain text with minimal formatting)
        plain = tg_text.replace("<b>", "").replace("</b>", "")
        html_rows = []
        for ln in plain.split("\n"):
            ln_stripped = ln.strip()
            if ln_stripped.startswith("Weekly Model Analysis"):
                html_rows.append(f"<h2>{ln_stripped}</h2>")
            elif ln_stripped.startswith("Proposed Changes"):
                html_rows.append(f"<h3 style='color:#e67e22'>{ln_stripped}</h3>")
            elif ln_stripped.startswith("REDUCE") or ln_stripped.startswith("BOOST"):
                color = "#c0392b" if ln_stripped.startswith("REDUCE") else "#27ae60"
                html_rows.append(f"<p style='color:{color};margin:2px 0'>{ln_stripped}</p>")
            elif ln_stripped == "":
                html_rows.append("<br>")
            else:
                html_rows.append(f"<p style='margin:2px 0'>{ln_stripped}</p>")
        html_body = (
            "<html><body style='font-family:monospace;font-size:13px;max-width:700px;margin:auto'>"
            + "\n".join(html_rows)
            + "</body></html>"
        )
        send_weekly_report_email(
            subject=f"AI Trading Model — Weekly Analysis {datetime.datetime.now().strftime('%Y-%m-%d')}",
            html_body=html_body,
        )

    except Exception as e:
        print(f"WEEKLY ANALYSIS: failed — {e}")


_scheduler = BackgroundScheduler(timezone=pytz.timezone("America/New_York"))
_scheduler.add_job(_scheduled_scan,  "cron", day_of_week="mon-fri", hour=9,  minute=45)
_scheduler.add_job(_premarket_scan,  "cron", day_of_week="mon-fri", hour=8,  minute=30)
# Validation: 9:00 AM ET (before market open) and every 15 min 3:00–3:45 PM ET
_scheduler.add_job(_run_validation,  "cron", day_of_week="mon-fri", hour=9,  minute=0)
_scheduler.add_job(_run_validation,  "cron", day_of_week="mon-fri", hour=15, minute=0)
_scheduler.add_job(_run_validation,  "cron", day_of_week="mon-fri", hour=15, minute=15)
_scheduler.add_job(_run_validation,  "cron", day_of_week="mon-fri", hour=15, minute=30)
_scheduler.add_job(_run_validation,  "cron", day_of_week="mon-fri", hour=15, minute=45)
# Daily backfill + LSTM retrain at 6:00 AM ET — before market open, after prior day outcomes settle
_scheduler.add_job(_daily_backfill,  "cron", day_of_week="mon-fri", hour=6,  minute=0)
# Intraday squeeze scan every 30 min during market hours (10:00–15:30 ET)
_scheduler.add_job(_intraday_scan,   "cron", day_of_week="mon-fri",
                   hour="10-15", minute="0,30")
# Watchlist near-miss re-check at the midpoints (:15 and :45) of each intraday hour
_scheduler.add_job(_check_watchlist, "cron", day_of_week="mon-fri",
                   hour="10-15", minute="15,45")
# 5m Spike scan every 5 minutes during market hours (10:00–15:30 ET)
_scheduler.add_job(_fivemin_spike_scan, "cron", day_of_week="mon-fri",
                   hour="10-15", minute="0,5,10,15,20,25,30,35,40,45,50,55")
# System health check every 30 minutes, all days
_scheduler.add_job(run_health_checks, "interval", minutes=30)
# Weekly analysis report — every Saturday at 8:00 AM ET → Telegram + email
_scheduler.add_job(_weekly_analysis, "cron", day_of_week="sat", hour=8, minute=0)
_scheduler.start()

# Warm up yfinance crumb on startup so the first scheduled scan doesn't hit a 401.
# Yahoo Finance rotates crumbs periodically; this forces a fresh fetch at boot time.
try:
    yf.download("SPY", period="1d", interval="1d", progress=False, auto_adjust=False)
    print("STARTUP: yfinance crumb refreshed ✓")
except Exception as _yf_err:
    print(f"STARTUP: yfinance warm-up failed (non-fatal) — {_yf_err}")

# Log API key status at startup so it's visible in Render logs
if _os.environ.get("ANTHROPIC_API_KEY"):
    print("STARTUP: ANTHROPIC_API_KEY is set ✓")
else:
    print("STARTUP: WARNING — ANTHROPIC_API_KEY is NOT set. AI features will fail.")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates", autoescape=True)


def _fetch_current_price(symbol: str, fallback: float) -> float:
    try:
        fi = yf.Ticker(symbol).fast_info
        price = fi.last_price
        if price and price > 0:
            return float(price)
        price = fi.previous_close
        if price and price > 0:
            return float(price)
    except Exception as _e:
        print(f"PRICE FETCH: {symbol} live price failed ({_e}) — trying scan price fallback")
    if fallback and fallback > 0:
        print(f"PRICE FETCH: {symbol} using scan price fallback ${fallback:.4f}")
        return float(fallback)
    print(f"PRICE FETCH: {symbol} no valid price — paper trade will be skipped")
    return 0.0


# Stock ticker symbols: 1–10 uppercase letters/digits, optional .A/.B suffix (e.g. BRK.A)
_SYMBOL_RE = _re.compile(r'^[A-Z0-9]{1,10}(\.[A-Z]{1,2})?$')


def _validate_symbol(raw: str) -> str | None:
    """Return sanitised uppercase symbol if valid, None if it looks malicious."""
    cleaned = raw.strip().upper()
    return cleaned if _SYMBOL_RE.match(cleaned) else None


# ---------------- HEALTH CHECK ----------------
_health_calls: dict = {}   # {ip: [timestamp, ...]}
_HEALTH_MAX        = 60    # max calls per window
_HEALTH_WINDOW_SEC = 60    # 1-minute window

@app.get("/health")
@app.head("/health")
def health(request: Request):
    ip  = request.client.host if request.client else "unknown"
    now = _time.time()
    q   = _health_calls.setdefault(ip, [])
    # Evict timestamps outside the window
    while q and now - q[0] > _HEALTH_WINDOW_SEC:
        q.pop(0)
    if len(q) >= _HEALTH_MAX:
        return Response(status_code=429)
    q.append(now)
    return Response(status_code=200)


# ---------------- ROOT ----------------
@app.get("/")
@app.head("/")
def root():
    return RedirectResponse("/login")


# ---------------- INVITE ----------------
@app.get("/invite", response_class=HTMLResponse)
def invite_page(request: Request):
    """Public invite page — no login required. Shows account creation link + Telegram QR."""
    import requests as _req
    bot_username = ""
    token = _os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        try:
            r = _req.get(f"https://api.telegram.org/bot{token}/getMe", timeout=5)
            if r.ok:
                bot_username = r.json().get("result", {}).get("username", "")
        except Exception:
            pass
    return templates.TemplateResponse(
        "invite.html",
        {"request": request, "bot_username": bot_username}
    )


# ---------------- LOGIN ----------------
@app.head("/login")
def login_head():
    return Response(status_code=200)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    # Rate-limit by IP address
    ip = request.client.host if request.client else "unknown"
    now = _time.time()
    attempts = _login_attempts.get(ip, [])
    attempts = [t for t in attempts if now - t < _LOGIN_WINDOW_SEC]
    if len(attempts) >= _LOGIN_MAX:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Too many login attempts. Try again in 15 minutes."}
        )

    if not authenticate_user(username, password):
        attempts.append(now)
        _login_attempts[ip] = attempts
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid credentials"}
        )

    # Successful login — clear attempts and regenerate session
    _login_attempts.pop(ip, None)
    request.session.clear()
    request.session["user"] = username
    return RedirectResponse("/dashboard", status_code=303)


# ---------------- DASHBOARD ----------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, mode: str = "squeeze", trade_error: str = "",
              refresh: str = ""):

    if "user" not in request.session:
        return RedirectResponse("/login")

    # Serve from cache (15-min TTL) unless user clicked Refresh
    cached = None if refresh else get_scan_cache(mode)
    scan_id_map = {}
    if cached:
        scan_data = {"results": cached["results"], "summary": cached["summary"]}
        cache_age = cached["cache_age_minutes"]
    else:
        scan_data = run_scan_5m() if mode == "fivemin" else run_scan(mode=mode)
        scan_id_map = save_scan(scan_data["results"], mode)
        update_returns()
        save_scan_cache(mode, scan_data["results"], scan_data["summary"])
        cache_age = 0

    available_cash  = get_portfolio_summary()["cash"]
    hypothesis_text = get_active_hypothesis_text(mode=mode)
    sizing_stats    = get_sizing_stats()
    ai_accuracy     = get_ai_decision_accuracy()
    active_rule_ids = get_active_rule_ids()
    _modes          = _FIVEMIN_MODES if mode == "fivemin" else _DAILY_MODES
    _per_signal     = get_per_signal_stats(modes=_modes)

    # Parallel AI enrichment — position sizing + price target prediction (score >= 75) + AI trade call
    def _ai_enrich(stock):
        stock["ai_rec"] = recommend_position_size(
            stock, available_cash, hypothesis_text, sizing_stats
        )
        if stock["score"] >= 75:
            stock["ai_target"] = predict_price_target(stock, sizing_stats, hypothesis_text)
        else:
            stock["ai_target"] = {"target_pct": 20, "rationale": ""}
        # Local LSTM inference — daily models run at 50 (matches AI enrichment threshold)
        lstm_prob = None
        if stock["score"] >= 50:
            checklist = stock.get("checklist", {})
            lstm_prob = predict_hit_probability(
                stock["symbol"],
                shares_outstanding=checklist.get("shares_outstanding"),
                sector=checklist.get("sector"),
            )
        stock["lstm_prob"] = lstm_prob

        ticker_history = get_ticker_scan_history(stock["symbol"])
        news = get_stock_news(stock["symbol"])
        stock["ai_trade_call"] = recommend_trade(
            stock, hypothesis_text, sizing_stats, ticker_history,
            lstm_prob=lstm_prob, news_headlines=news,
            ai_accuracy=ai_accuracy, per_signal_stats=_per_signal
        )
        scan_id = scan_id_map.get(stock.get("symbol"))
        if scan_id:
            try:
                tc = stock["ai_trade_call"]
                update_scan_ai_rec(scan_id, tc["decision"], tc["confidence"], tc["rationale"],
                                   lstm_prob=lstm_prob)
                # Tag which hypothesis rules were active — enables per-rule win rate tracking
                if active_rule_ids:
                    save_scan_active_rules(scan_id, active_rule_ids)
            except Exception as _e:
                print(f"AI-ENRICH: failed to save AI rec for scan_id={scan_id}: {_e}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(_ai_enrich, scan_data["results"]))

    ai_weights_info = get_squeeze_weights() if mode == "squeeze" else None
    autoai_weights_info = get_autoai_weights() if mode == "autoai" else None
    last_autoai_run = None
    if mode == "autoai":
        log = get_autoai_log(limit=1)
        last_autoai_run = log[0] if log else None

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": scan_data["results"],
            "summary": scan_data["summary"],
            "mode": mode,
            "trade_error": trade_error,
            "user": request.session["user"],
            "ai_weights_info": ai_weights_info,
            "autoai_weights_info": autoai_weights_info,
            "last_autoai_run": last_autoai_run,
            "cache_age": cache_age,
            "sizing_calibrated": sizing_stats is not None,
            "sizing_total": sizing_stats["total"] if sizing_stats else 0,
            "near_misses": get_active_watchlist(today_only=True),
            "health_status": get_health_status(),
        }
    )


# ---------------- ANALYTICS ----------------
_DAILY_MODES = ["squeeze", "autoai", "strict", "standard"]
_FIVEMIN_MODES = ["fivemin", "fivemin_bt"]
_5m_backfill_running = False

@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request):

    if "user" not in request.session:
        return RedirectResponse("/login")

    active_tab = request.query_params.get("tab", "daily")

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "active_tab": active_tab,
            "user": request.session["user"],
            # --- shared / tab-independent ---
            "weight_changelog": get_weight_changelog(),
            "risk_metrics": get_risk_metrics(),
            "historical_count": get_historical_count(),
            "backfill_status": get_backfill_status(),
            "signals_backfill_running": _signals_backfill_running,
            "complex_ai_weights": get_squeeze_weights(),
            "hypothesis": get_hypothesis(),
            "hypothesis_rules": get_hypothesis_rules(),
            "pending_rule_count": get_pending_rule_count(),
            "lstm_status": get_lstm_status(),
            "seq_stats": get_sequence_stats(),
            "xgb_status": get_xgb_status(),
            "model_validation": get_model_validation_stats(),
            "autoai_log": get_autoai_log(limit=20),
            "autoai_weights": get_autoai_weights(),
            "model_comparison": get_model_comparison_stats(),
            # --- Tab 1: Daily Models ---
            "score_buckets":    get_score_buckets(modes=_DAILY_MODES),
            "holding_perf":     get_holding_performance(modes=_DAILY_MODES),
            "equity_curve":     get_equity_curve(modes=_DAILY_MODES),
            "live_scan_stats":  get_live_scan_stats(modes=_DAILY_MODES),
            "per_signal_stats": get_per_signal_stats(modes=_DAILY_MODES),
            # --- Tab 2: 5m Spike ---
            "fm_score_buckets":    get_score_buckets(modes=_FIVEMIN_MODES),
            "fm_holding_perf":     get_holding_performance(modes=_FIVEMIN_MODES),
            "fm_equity_curve":     get_equity_curve(modes=_FIVEMIN_MODES),
            "fm_live_scan_stats":  get_live_scan_stats(modes=_FIVEMIN_MODES),
            "fm_per_signal_stats": get_per_signal_stats(modes=_FIVEMIN_MODES),
            "fivemin_backfill_status": get_5m_backfill_status(),
            "lstm_5m_status": get_lstm_5m_status(),
            "seq_5m_stats": get_5m_sequence_stats(),
            # --- Tab 3: Version Tracker ---
            "version_stats": get_version_performance_stats(),
        }
    )


# ---------------- TRADES ----------------
TARGET_GAIN = 0.20  # 20% profit target — auto-sell when hit


@app.get("/trades", response_class=HTMLResponse)
def trades_page(request: Request):

    if "user" not in request.session:
        return RedirectResponse("/login")

    # --- Auto-close any positions that have hit their per-trade take-profit target ---
    auto_closed = []
    now = datetime.datetime.now()
    for pos in get_open_positions():
        # Grace period: never auto-close a trade opened less than 5 minutes ago
        try:
            opened_dt = datetime.datetime.fromisoformat(pos["opened_at"].replace(" ", "T"))
            if (now - opened_dt).total_seconds() < 300:
                continue
        except Exception as _e:
            print(f"TRADES: failed to parse opened_at for {pos.get('symbol','?')}: {_e}")
        take_profit_target = pos.get("take_profit_pct", 20.0)
        target_price = pos["entry_price"] * (1 + take_profit_target / 100)
        current_price = _fetch_current_price(pos["symbol"], pos["entry_price"])
        if current_price >= target_price:
            result = close_trade(pos["trade_id"], current_price)
            if result:
                broker_close(pos["symbol"])
                auto_closed.append({
                    "symbol": pos["symbol"],
                    "exit_price": round(current_price, 4),
                    "realized_pnl": round(result["realized_pnl"], 2),
                })

    # Reload after any auto-closes
    positions = get_open_positions()
    history = get_trade_history()
    summary = get_portfolio_summary()

    open_value = 0.0
    for pos in positions:
        current_price = _fetch_current_price(pos["symbol"], pos["entry_price"])
        unrealized_pnl = (current_price - pos["entry_price"]) * pos["shares"]
        current_value = current_price * pos["shares"]
        open_value += current_value

        take_profit_pct = pos.get("take_profit_pct", 20.0)
        pos["current_price"] = round(current_price, 4)
        pos["target_price"] = round(pos["entry_price"] * (1 + take_profit_pct / 100), 4)
        pos["take_profit_pct"] = take_profit_pct
        pos["unrealized_pnl"] = round(unrealized_pnl, 2)
        pos["current_value"] = round(current_value, 2)
        pos["pnl_pct"] = round(
            (current_price - pos["entry_price"]) / pos["entry_price"] * 100, 2
        )
        pos["to_target_pct"] = round(
            (pos["target_price"] - current_price) / current_price * 100, 2
        )

    total_value = summary["cash"] + open_value

    return templates.TemplateResponse(
        "trades.html",
        {
            "request": request,
            "user": request.session["user"],
            "positions": positions,
            "history": history,
            "auto_closed": auto_closed,
            "watchlist": get_watchlist(),
            "summary": {
                **summary,
                "open_value": round(open_value, 2),
                "total_value": round(total_value, 2),
                "total_pnl": round(total_value - 10000.0, 2),
            }
        }
    )


@app.post("/trade/buy")
def trade_buy(
    request: Request,
    symbol: str = Form(...),
    price: float = Form(...),
    position_size: float = Form(1000.0),
    notes: str = Form(""),
    take_profit_pct: float = Form(20.0),
):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    symbol = _validate_symbol(symbol) or ""
    if not symbol:
        return RedirectResponse("/dashboard?trade_error=invalid_symbol", status_code=303)

    # Always use the live market price as entry — the form price comes from
    # a potentially stale scan cache and would cause immediate auto-closes.
    entry_price = _fetch_current_price(symbol, price)

    result = open_trade(symbol, entry_price, position_size, notes, take_profit_pct)

    if result is None:
        return RedirectResponse("/dashboard?trade_error=insufficient_funds", status_code=303)

    return RedirectResponse("/trades", status_code=303)


@app.post("/trade/sell/{trade_id}")
def trade_sell(request: Request, trade_id: int):

    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    trade = get_trade_by_id(trade_id)
    if trade is None:
        return RedirectResponse("/trades", status_code=303)

    current_price = _fetch_current_price(trade["symbol"], trade["entry_price"])
    close_trade(trade_id, current_price)
    broker_close(trade["symbol"])

    return RedirectResponse("/trades", status_code=303)


@app.post("/trade/{trade_id}/tag")
def tag_trade_route(request: Request, trade_id: int, outcome_tag: str = Form(...)):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if outcome_tag == "clear":
        tag_trade_outcome(trade_id, None)
    elif outcome_tag in ("win", "loss"):
        tag_trade_outcome(trade_id, outcome_tag)
    return RedirectResponse("/trades", status_code=303)


# ---------------- WEIGHT OPTIMIZER ----------------
@app.post("/optimize", response_class=HTMLResponse)
def optimize_weights(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if not _require_admin(request):
        return RedirectResponse("/analytics", status_code=303)

    opt_data = get_optimization_data()
    if not opt_data or opt_data["total_trades"] < 5:
        analysis = (
            "Not enough backtested data yet — need at least 5 scans with known returns. "
            "Returns are filled in automatically once scans are 3+ trading days old."
        )
    else:
        analysis = analyze_and_optimize_weights(opt_data)

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "weight_changelog": get_weight_changelog(),
            "risk_metrics": get_risk_metrics(),
            "historical_count": get_historical_count(),
            "backfill_status": get_backfill_status(),
            "signals_backfill_running": _signals_backfill_running,
            "user": request.session["user"],
            "active_tab": "daily",
            "weight_analysis": analysis,
            "opt_data": opt_data,
            "complex_ai_weights": get_squeeze_weights(),
            "hypothesis": get_hypothesis(),
            "lstm_status": get_lstm_status(),
            "seq_stats": get_sequence_stats(),
            "xgb_status": get_xgb_status(),
            "model_validation": get_model_validation_stats(),
            "model_comparison": get_model_comparison_stats(),
            "autoai_log":          get_autoai_log(limit=20),
            "autoai_weights":      get_autoai_weights(),
            "hypothesis_rules":    get_hypothesis_rules(),
            "pending_rule_count":  get_pending_rule_count(),
            "score_buckets":       get_score_buckets(modes=_DAILY_MODES),
            "holding_perf":        get_holding_performance(modes=_DAILY_MODES),
            "equity_curve":        get_equity_curve(modes=_DAILY_MODES),
            "live_scan_stats":     get_live_scan_stats(modes=_DAILY_MODES),
            "per_signal_stats":    get_per_signal_stats(modes=_DAILY_MODES),
            "fm_score_buckets":    get_score_buckets(modes=_FIVEMIN_MODES),
            "fm_holding_perf":     get_holding_performance(modes=_FIVEMIN_MODES),
            "fm_equity_curve":     get_equity_curve(modes=_FIVEMIN_MODES),
            "fm_live_scan_stats":  get_live_scan_stats(modes=_FIVEMIN_MODES),
            "fm_per_signal_stats": get_per_signal_stats(modes=_FIVEMIN_MODES),
            "fivemin_backfill_status": get_5m_backfill_status(),
            "lstm_5m_status": get_lstm_5m_status(),
            "seq_5m_stats": get_5m_sequence_stats(),
            "version_stats":       get_version_performance_stats(),
        }
    )


# ---------------- FEEDBACK ----------------
@app.get("/feedback", response_class=HTMLResponse)
def feedback_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login")

    manual_count = len(get_all_feedback())
    hist_count   = get_historical_count()
    return templates.TemplateResponse(
        "feedback.html",
        {
            "request": request,
            "user": request.session["user"],
            "recent_feedback": get_recent_feedback(limit=10),
            "hypothesis": get_hypothesis(),
            "weights_info": get_squeeze_weights(),
            "historical_count": hist_count,
            "manual_count": manual_count,
            "historical_examples": get_historical_examples(limit=20),
            "reanalyzing": request.query_params.get("reanalyzing") == "1",
            "syncing": request.query_params.get("syncing") == "1",
        }
    )




@app.post("/sync-all-sources")
def sync_all_sources(request: Request, background_tasks: BackgroundTasks):
    """Synthesize hypothesis from both manual submissions and historical scan data."""
    if "user" not in request.session:
        return RedirectResponse("/login")

    def _run():
        all_feedback = get_all_feedback()
        opt_data = get_optimization_data()
        hist_count = get_historical_count()
        prior = get_hypothesis()
        prior_text = prior["content"] if prior else None
        text = synthesize_combined_hypothesis(all_feedback, opt_data, hist_count, prior_text)
        total = len(all_feedback) + hist_count
        if text:
            save_hypothesis(text, total)
            rules = parse_rules_from_synthesis(text)
            if rules:
                save_hypothesis_rules(rules)

    background_tasks.add_task(_run)
    return RedirectResponse("/feedback?syncing=1", status_code=303)


@app.post("/reanalyze-feedback")
def reanalyze_feedback(request: Request, background_tasks: BackgroundTasks):
    if "user" not in request.session:
        return RedirectResponse("/login")

    def _run():
        entries = get_all_feedback()
        for entry in entries:
            if not entry.get("chart_analysis"):
                continue
            new_analysis = reprocess_chart_analysis(
                entry["chart_analysis"],
                entry.get("user_text", ""),
                entry.get("symbol")
            )
            update_feedback_analysis(entry["id"], new_analysis)
        # Re-synthesize hypothesis with updated analyses
        all_feedback = get_all_feedback()
        if all_feedback:
            text = synthesize_feedback_hypotheses(all_feedback)
            if text:
                save_hypothesis(text, len(all_feedback))
                rules = parse_rules_from_synthesis(text)
                if rules:
                    save_hypothesis_rules(rules)

    background_tasks.add_task(_run)
    return RedirectResponse("/feedback?reanalyzing=1", status_code=303)


def _run_hypothesis_and_weights(all_feedback: list):
    """Background task: synthesize combined hypothesis from all sources then optimize weights."""
    try:
        opt_data = get_optimization_data()
        hist_count = get_historical_count()
        prior = get_hypothesis()
        prior_text = prior["content"] if prior else None
        hypothesis_text = synthesize_combined_hypothesis(all_feedback, opt_data, hist_count, prior_text)
        hypothesis_content = None
        if hypothesis_text:
            save_hypothesis(hypothesis_text, len(all_feedback) + hist_count)
            hypothesis_content = hypothesis_text
            # Parse discrete rules and save as pending for admin review
            rules = parse_rules_from_synthesis(hypothesis_text)
            if rules:
                save_hypothesis_rules(rules)
                print(f"HYPOTHESIS: {len(rules)} rule(s) added as pending — review on Analytics page")

        weights_data = get_squeeze_weights()
        current_weights = weights_data["weights"] if weights_data else DEFAULT_SQUEEZE_WEIGHTS.copy()
        opt_result = optimize_complex_ai_weights(opt_data, all_feedback, current_weights, hypothesis_content)
        if "error" not in opt_result:
            save_squeeze_weights(
                opt_result["weights"],
                opt_result["rationale"],
                opt_result["suggestions"],
                opt_result["summary"],
            )
            save_weight_changelog(
                opt_result.get("summary", ""),
                opt_result.get("rationale", ""),
                opt_result["weights"],
            )
    except Exception as e:
        import traceback
        print(f"HYPOTHESIS/WEIGHTS background task failed — {e}\n{traceback.format_exc()}")
        _send_telegram_admin(f"<b>⚠️ HYPOTHESIS UPDATE ERROR</b>\n<code>{str(e)[:300]}</code>")


@app.post("/feedback", response_class=HTMLResponse)
async def submit_feedback(
    request: Request,
    background_tasks: BackgroundTasks,
):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    symbol = form.get("symbol", "")
    user_text = form.get("user_text", "")
    charts = form.getlist("charts")

    chart_analysis = ""
    valid_charts = [c for c in charts if hasattr(c, "filename") and c.filename]
    if valid_charts:
        # Read all file bytes first
        chart_data = []
        for chart in valid_charts:
            image_bytes = await chart.read()
            fn = chart.filename.lower()
            if fn.endswith(".png"):
                media_type = "image/png"
            elif fn.endswith(".gif"):
                media_type = "image/gif"
            elif fn.endswith(".webp"):
                media_type = "image/webp"
            else:
                media_type = "image/jpeg"
            chart_data.append((image_bytes, media_type))

        # Analyze all charts in parallel
        loop = asyncio.get_event_loop()
        sym = symbol.strip() or None
        tasks = [
            loop.run_in_executor(None, analyze_chart_feedback, img, mt, user_text, sym)
            for img, mt in chart_data
        ]
        results = await asyncio.gather(*tasks)

        if len(valid_charts) > 1:
            analyses = [f"--- Chart {i+1} of {len(results)} ---\n{r}" for i, r in enumerate(results)]
        else:
            analyses = list(results)
        chart_analysis = "\n\n".join(analyses)

    save_feedback(symbol.upper().strip(), user_text, chart_analysis)

    # Hypothesis synthesis + weight optimization run in background
    # save_feedback already committed, so get_all_feedback() includes this submission
    all_feedback = get_all_feedback()
    background_tasks.add_task(_run_hypothesis_and_weights, all_feedback)

    hypothesis_data = get_hypothesis()

    manual_count = len(all_feedback)
    hist_count   = get_historical_count()
    return templates.TemplateResponse(
        "feedback.html",
        {
            "request": request,
            "user": request.session["user"],
            "recent_feedback": get_recent_feedback(limit=10),
            "hypothesis": hypothesis_data,
            "weights_info": get_squeeze_weights(),
            "historical_count": hist_count,
            "manual_count": manual_count,
            "historical_examples": get_historical_examples(limit=20),
            "submitted": True,
            "chart_analysis": chart_analysis,
        }
    )


_ZIP_IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_ZIP_MAX_IMAGES   = 30      # cap per upload to avoid very long API waits
_ZIP_MAX_BYTES    = 100 * 1024 * 1024  # 100 MB raw zip limit


@app.post("/feedback/upload-zip")
async def feedback_upload_zip(request: Request, background_tasks: BackgroundTasks):
    """
    Accept a zip file (e.g. downloaded from Google Photos), extract every image,
    run each through Claude chart analysis, and save as individual feedback entries.
    """
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    form       = await request.form()
    zip_upload = form.get("zip_file")
    symbol     = (form.get("symbol") or "").strip().upper()
    user_text  = (form.get("user_text") or "").strip()

    if not zip_upload or not hasattr(zip_upload, "read"):
        return RedirectResponse("/feedback?zip_error=no_file", status_code=303)

    raw = await zip_upload.read()
    if len(raw) > _ZIP_MAX_BYTES:
        return RedirectResponse("/feedback?zip_error=too_large", status_code=303)

    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return RedirectResponse("/feedback?zip_error=invalid_zip", status_code=303)

    # Collect image entries — skip macOS metadata sidecars and folders
    image_names = [
        n for n in zf.namelist()
        if not n.startswith("__MACOSX")
        and not n.endswith("/")
        and _os.path.splitext(n.lower())[1] in _ZIP_IMAGE_EXTS
    ]

    if not image_names:
        return RedirectResponse("/feedback?zip_error=no_images", status_code=303)

    # Cap and sort so order is deterministic (Google Photos names are date-stamped)
    image_names = sorted(image_names)[:_ZIP_MAX_IMAGES]

    # Read all image bytes up front (sync, fast)
    chart_data = []
    for name in image_names:
        ext = _os.path.splitext(name.lower())[1]
        media_type = {
            ".png":  "image/png",
            ".gif":  "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
        chart_data.append((_os.path.basename(name), zf.read(name), media_type))

    # Analyze every image in parallel (same approach as multi-chart upload)
    loop   = asyncio.get_event_loop()
    sym    = symbol or None
    tasks  = [
        loop.run_in_executor(None, analyze_chart_feedback, img_bytes, mt, user_text, sym)
        for _, img_bytes, mt in chart_data
    ]
    analyses = await asyncio.gather(*tasks)

    # Save one feedback entry per image
    for (fname, _, _), analysis in zip(chart_data, analyses):
        note = f"[From zip: {fname}]"
        combined_text = f"{note}\n{user_text}" if user_text else note
        save_feedback(symbol, combined_text, analysis)

    saved = len(chart_data)
    background_tasks.add_task(_run_hypothesis_and_weights, get_all_feedback())
    return RedirectResponse(f"/feedback?zip_imported={saved}", status_code=303)


@app.post("/feedback/{feedback_id}/tag")
def tag_feedback_route(request: Request, feedback_id: int, outcome_tag: str = Form(...)):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if outcome_tag == "clear":
        tag_feedback_outcome(feedback_id, None)
    elif outcome_tag in ("win", "loss"):
        tag_feedback_outcome(feedback_id, outcome_tag)
    return RedirectResponse("/feedback", status_code=303)


# ---------------- COMPLEX + AI OPTIMIZER ----------------
@app.post("/optimize-complex", response_class=HTMLResponse)
def optimize_complex(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if not _require_admin(request):
        return RedirectResponse("/analytics", status_code=303)

    opt_data = get_optimization_data()
    all_feedback = get_all_feedback()
    hypothesis_data = get_hypothesis()

    weights_data = get_squeeze_weights()
    current_weights = weights_data["weights"] if weights_data else DEFAULT_SQUEEZE_WEIGHTS.copy()

    opt_result = optimize_complex_ai_weights(
        opt_data, all_feedback, current_weights,
        hypothesis_data["content"] if hypothesis_data else None
    )

    if "error" not in opt_result:
        save_squeeze_weights(
            opt_result["weights"],
            opt_result["rationale"],
            opt_result["suggestions"],
            opt_result["summary"],
        )
        save_weight_changelog(
            opt_result.get("summary", ""),
            opt_result.get("rationale", ""),
            opt_result["weights"],
        )

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "weight_changelog": get_weight_changelog(),
            "risk_metrics": get_risk_metrics(),
            "historical_count": get_historical_count(),
            "backfill_status": get_backfill_status(),
            "signals_backfill_running": _signals_backfill_running,
            "user": request.session["user"],
            "active_tab": "daily",
            "complex_ai_result": opt_result,
            "complex_ai_weights": get_squeeze_weights(),
            "hypothesis": hypothesis_data,
            "lstm_status": get_lstm_status(),
            "seq_stats": get_sequence_stats(),
            "xgb_status": get_xgb_status(),
            "model_validation": get_model_validation_stats(),
            "model_comparison": get_model_comparison_stats(),
            "autoai_log":          get_autoai_log(limit=20),
            "autoai_weights":      get_autoai_weights(),
            "hypothesis_rules":    get_hypothesis_rules(),
            "pending_rule_count":  get_pending_rule_count(),
            "score_buckets":       get_score_buckets(modes=_DAILY_MODES),
            "holding_perf":        get_holding_performance(modes=_DAILY_MODES),
            "equity_curve":        get_equity_curve(modes=_DAILY_MODES),
            "live_scan_stats":     get_live_scan_stats(modes=_DAILY_MODES),
            "per_signal_stats":    get_per_signal_stats(modes=_DAILY_MODES),
            "fm_score_buckets":    get_score_buckets(modes=_FIVEMIN_MODES),
            "fm_holding_perf":     get_holding_performance(modes=_FIVEMIN_MODES),
            "fm_equity_curve":     get_equity_curve(modes=_FIVEMIN_MODES),
            "fm_live_scan_stats":  get_live_scan_stats(modes=_FIVEMIN_MODES),
            "fm_per_signal_stats": get_per_signal_stats(modes=_FIVEMIN_MODES),
            "fivemin_backfill_status": get_5m_backfill_status(),
            "lstm_5m_status": get_lstm_5m_status(),
            "seq_5m_stats": get_5m_sequence_stats(),
            "version_stats":       get_version_performance_stats(),
        }
    )


# ---------------- APPLY MODEL UPDATE ----------------
@app.post("/revert-weights/{changelog_id}")
def revert_weights(request: Request, changelog_id: int):
    if "user" not in request.session or not _require_admin(request):
        return RedirectResponse("/login", status_code=303)
    changelog = get_weight_changelog(limit=100)
    entry = next((e for e in changelog if e["id"] == changelog_id), None)
    if not entry or not entry["weights"]:
        return RedirectResponse("/analytics?revert=not_found", status_code=303)
    save_squeeze_weights(
        entry["weights"],
        rationale=f"Reverted to: {entry['summary'] or entry['updated_at'][:10]}",
        summary=f"Revert to {entry['updated_at'][:10]}",
        source="manual",
        goal="",
    )
    return RedirectResponse("/analytics?revert=ok", status_code=303)


@app.post("/apply-model-update")
def apply_model_update(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    all_feedback = get_all_feedback()
    hypothesis_data = get_hypothesis()
    opt_data = get_optimization_data()
    weights_data = get_squeeze_weights()
    current_weights = weights_data["weights"] if weights_data else DEFAULT_SQUEEZE_WEIGHTS.copy()

    opt_result = optimize_complex_ai_weights(
        opt_data, all_feedback, current_weights,
        hypothesis_data["content"] if hypothesis_data else None
    )

    if "error" not in opt_result:
        save_squeeze_weights(
            opt_result["weights"],
            opt_result["rationale"],
            opt_result["suggestions"],
            opt_result["summary"],
        )
        save_weight_changelog(
            opt_result.get("summary", ""),
            opt_result.get("rationale", ""),
            opt_result["weights"],
        )

    return RedirectResponse("/analytics", status_code=303)


# ---------------- LIVE PRICE API ----------------
@app.get("/api/positions/prices")
def api_position_prices(request: Request):
    if "user" not in request.session:
        return Response(status_code=401)
    positions = get_open_positions()
    data = {}
    for pos in positions:
        current = _fetch_current_price(pos["symbol"], pos["entry_price"])
        unrealized = round((current - pos["entry_price"]) * pos["shares"], 2)
        current_value = round(current * pos["shares"], 2)
        pnl_pct = round((current - pos["entry_price"]) / pos["entry_price"] * 100, 2)
        take_profit_pct = pos.get("take_profit_pct", 20.0)
        target = round(pos["entry_price"] * (1 + take_profit_pct / 100), 4)
        to_target_pct = round((target - current) / current * 100, 2)
        data[pos["trade_id"]] = {
            "current_price":  round(current, 4),
            "unrealized_pnl": unrealized,
            "current_value":  current_value,
            "pnl_pct":        pnl_pct,
            "to_target_pct":  to_target_pct,
        }
    return data


# ---------------- WATCHLIST ----------------
@app.post("/watchlist/add")
def watchlist_add(request: Request, symbol: str = Form(...), price: float = Form(0.0)):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    symbol = _validate_symbol(symbol) or ""
    if not symbol:
        return RedirectResponse("/trades", status_code=303)
    add_to_watchlist(symbol, price or None)
    return RedirectResponse("/trades", status_code=303)


@app.post("/watchlist/remove/{symbol}")
def watchlist_remove(request: Request, symbol: str):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    remove_from_watchlist(symbol)
    return RedirectResponse("/trades", status_code=303)


# ---------------- HISTORICAL BACKFILL ----------------
@app.post("/backfill-history")
def start_backfill(request: Request):
    global _backfill_running
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    if _backfill_running:
        return RedirectResponse("/analytics?backfill=already_running", status_code=303)

    weights_data = get_squeeze_weights()
    weights = weights_data["weights"] if weights_data else None

    def _run(**kwargs):
        global _backfill_running
        _backfill_running = True
        try:
            build_historical_dataset(**kwargs)   # also builds lstm_sequences.npz
            # Train LSTM on the freshly built sequences
            print("BACKFILL: training LSTM model...")
            try:
                lstm_stats = train_lstm()
                print(f"BACKFILL: LSTM trained — {lstm_stats}")
            except Exception as e:
                print(f"BACKFILL: LSTM training failed — {e}")
            # Auto-activate Claude hypothesis + weights — no extra click needed
            print("BACKFILL: auto-running hypothesis + weight synthesis")
            _run_hypothesis_and_weights(get_all_feedback())
            print("BACKFILL: AI model activated")
        finally:
            _backfill_running = False

    threading.Thread(target=_run, kwargs={"weights": weights}, daemon=True).start()

    return RedirectResponse("/analytics?backfill=started", status_code=303)


@app.post("/apply-default-weights")
def apply_default_weights(request: Request):
    """Save DEFAULT_SQUEEZE_WEIGHTS to the settings table and rescore all history.
    Use after any code-level weight change to make it take effect in the live scorer."""
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    save_squeeze_weights(
        DEFAULT_SQUEEZE_WEIGHTS,
        rationale="Manually applied code-level DEFAULT_SQUEEZE_WEIGHTS",
        source="manual",
        summary="Reset to current DEFAULT_SQUEEZE_WEIGHTS (2026-03-10 data-driven rebalance)",
    )
    return RedirectResponse("/analytics?rescore=done&weights=defaults_applied", status_code=303)


@app.post("/rescore-history")
def rescore_history(request: Request):
    """Recompute score for all rows with signals_json using current weights.
    Run this after any manual weight change to keep stats version-consistent."""
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    weights_data = get_squeeze_weights()
    weights = weights_data["weights"] if weights_data else DEFAULT_SQUEEZE_WEIGHTS
    n = rescore_historical_from_signals(weights)
    return RedirectResponse(f"/analytics?rescore=done&rows={n}", status_code=303)


@app.post("/backfill-signals")
def start_signals_backfill(request: Request):
    """Re-score all historical rows to populate signals_json for XGBoost training."""
    global _signals_backfill_running
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if _signals_backfill_running:
        return RedirectResponse("/analytics?signals_backfill=already_running", status_code=303)

    def _run():
        global _signals_backfill_running
        _signals_backfill_running = True
        try:
            n = backfill_signals_for_historical(max_workers=2)
            print(f"SIGNALS BACKFILL: complete — {n} rows updated")
        finally:
            _signals_backfill_running = False

    threading.Thread(target=_run, daemon=True).start()
    return RedirectResponse("/analytics?signals_backfill=started", status_code=303)


@app.get("/api/backfill-status")
def api_backfill_status(request: Request):
    if "user" not in request.session:
        return Response(status_code=401)
    return get_backfill_status()


@app.post("/refresh-projections")
def refresh_projections(request: Request):
    """Recompute projection_json for all bundle rules using current per-signal data."""
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    try:
        n = refresh_bundle_projections()
        print(f"PROJECTIONS: refreshed {n} bundle rules")
    except Exception as e:
        print(f"PROJECTIONS: refresh failed — {e}")
    return RedirectResponse("/analytics#rules-registry", status_code=303)


# ---------------- 5m SPIKE BACKFILL ----------------
@app.post("/backfill-5m")
def start_5m_backfill(request: Request):
    global _5m_backfill_running
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    if _5m_backfill_running:
        return RedirectResponse("/analytics?tab=fivemin&backfill5m=already_running", status_code=303)

    def _run():
        global _5m_backfill_running
        _5m_backfill_running = True
        try:
            backfill_5m_history(max_tickers=150)
        finally:
            _5m_backfill_running = False

    threading.Thread(target=_run, daemon=True).start()
    return RedirectResponse("/analytics?tab=fivemin&backfill5m=started", status_code=303)


@app.get("/api/backfill-5m-status")
def api_5m_backfill_status(request: Request):
    if "user" not in request.session:
        return Response(status_code=401)
    return get_5m_backfill_status()


# ---------------- SYSTEM HEALTH ----------------
@app.get("/health", response_class=HTMLResponse)
def health_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login")
    return templates.TemplateResponse("health.html", {
        "request": request,
        "user": request.session["user"],
        "health": get_health_status(),
    })


@app.post("/run-health-check")
def run_health_check_now(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if not _require_admin(request):
        return RedirectResponse("/health", status_code=303)
    threading.Thread(target=run_health_checks, daemon=True).start()
    return RedirectResponse("/health?ran=1", status_code=303)


@app.get("/api/health-status")
def api_health_status(request: Request):
    if "user" not in request.session:
        return Response(status_code=401)
    return get_health_status() or {}


@app.post("/retrain-lstm")
def retrain_lstm(request: Request):
    """Re-train the LSTM on existing lstm_sequences.npz without re-running the full backfill."""
    global _lstm_training
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if _lstm_training:
        return RedirectResponse("/analytics?lstm=already_training", status_code=303)

    def _run():
        global _lstm_training
        _lstm_training = True
        try:
            stats = train_lstm()
            print(f"RETRAIN LSTM: complete — {stats}")
        except Exception as e:
            print(f"RETRAIN LSTM: failed — {e}")
        finally:
            _lstm_training = False

    threading.Thread(target=_run, daemon=True).start()
    return RedirectResponse("/analytics?lstm=training", status_code=303)


_lstm_5m_training = False

@app.post("/build-5m-sequences")
def build_5m_sequences_endpoint(request: Request):
    """Build 5m training sequences from 60d historical 5m data, then train the 5m LSTM."""
    global _lstm_5m_training
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if _lstm_5m_training:
        return RedirectResponse("/analytics?tab=fivemin&lstm5m=already_training", status_code=303)

    def _run():
        global _lstm_5m_training
        _lstm_5m_training = True
        try:
            from app.backfill import SEED_TICKERS
            from app.universe import fetch_backfill_universe
            try:
                extra = fetch_backfill_universe(max_tickers=500)
            except Exception:
                extra = []
            tickers = list(dict.fromkeys(SEED_TICKERS + extra))
            n = build_5m_sequences(tickers)
            print(f"5m SEQUENCES: built {n} sequences")
            if n >= 30:
                stats = train_lstm_5m()
                print(f"5m LSTM: trained — {stats}")
        except Exception as e:
            print(f"5m SEQUENCES: failed — {e}")
        finally:
            _lstm_5m_training = False

    threading.Thread(target=_run, daemon=True).start()
    return RedirectResponse("/analytics?tab=fivemin&lstm5m=building", status_code=303)


@app.post("/retrain-lstm-5m")
def retrain_lstm_5m_endpoint(request: Request):
    """Re-train the 5m LSTM on existing lstm_sequences_5m.npz."""
    global _lstm_5m_training
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if _lstm_5m_training:
        return RedirectResponse("/analytics?tab=fivemin&lstm5m=already_training", status_code=303)

    def _run():
        global _lstm_5m_training
        _lstm_5m_training = True
        try:
            stats = train_lstm_5m()
            print(f"RETRAIN 5m LSTM: complete — {stats}")
        except Exception as e:
            print(f"RETRAIN 5m LSTM: failed — {e}")
        finally:
            _lstm_5m_training = False

    threading.Thread(target=_run, daemon=True).start()
    return RedirectResponse("/analytics?tab=fivemin&lstm5m=training", status_code=303)


@app.get("/api/hypothesis-status")
def api_hypothesis_status(request: Request):
    """Polling endpoint — returns hypothesis updated_at so the frontend
    knows when a fresh synthesis has been saved."""
    if "user" not in request.session:
        return Response(status_code=401)
    h = get_hypothesis()
    if h:
        return {"ready": True, "updated_at": h["updated_at"]}
    return {"ready": False}


# ---------------- VALIDATION ----------------
@app.get("/validation", response_class=HTMLResponse)
def validation_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login")
    return templates.TemplateResponse(
        "validation.html",
        {
            "request": request,
            "user": request.session["user"],
            "reports": get_validation_reports(limit=10),
        }
    )


@app.post("/validation/run")
def validation_run_now(request: Request, background_tasks: BackgroundTasks):
    """Manually trigger a validation run against live FinViz data."""
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    background_tasks.add_task(_run_validation)
    return RedirectResponse("/validation?running=1", status_code=303)


# ---------------- ACCOUNT ----------------
@app.get("/account", response_class=HTMLResponse)
def account_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login")
    return templates.TemplateResponse(
        "account.html",
        {"request": request, "user": request.session["user"]}
    )


@app.post("/account/change-password", response_class=HTMLResponse)
def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    username = request.session["user"]

    if not authenticate_user(username, current_password):
        return templates.TemplateResponse(
            "account.html",
            {"request": request, "user": username,
             "error": "Current password is incorrect."}
        )

    if new_password != confirm_password:
        return templates.TemplateResponse(
            "account.html",
            {"request": request, "user": username,
             "error": "New passwords do not match."}
        )

    if len(new_password) < 8:
        return templates.TemplateResponse(
            "account.html",
            {"request": request, "user": username,
             "error": "New password must be at least 8 characters."}
        )

    if not _re.search(r'[A-Za-z]', new_password) or not _re.search(r'[0-9]', new_password):
        return templates.TemplateResponse(
            "account.html",
            {"request": request, "user": username,
             "error": "New password must contain at least one letter and one number."}
        )

    update_password(username, new_password)
    return templates.TemplateResponse(
        "account.html",
        {"request": request, "user": username,
         "success": "Password updated successfully."}
    )


# ---------------- BROKER STATUS ----------------
@app.get("/api/alpaca-test")
def alpaca_test_order(request: Request, symbol: str = "LGVN", amount: float = 100):
    """
    Test endpoint: attempts a real Alpaca paper order for `symbol` at `amount` dollars.
    Returns the full Alpaca response (success or error) so you can diagnose issues.
    Admin only. Use: /api/alpaca-test?symbol=LGVN&amount=100
    """
    if "user" not in request.session:
        return Response(status_code=401)
    if not broker_configured():
        return {"error": "Alpaca not configured — check ALPACA_API_KEY and ALPACA_SECRET_KEY env vars"}

    import requests as _req, math as _math
    import os as _os_
    headers = {
        "APCA-API-KEY-ID":     _os_.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": _os_.environ.get("ALPACA_SECRET_KEY", ""),
        "accept":              "application/json",
        "content-type":        "application/json",
    }
    base = "https://paper-api.alpaca.markets"

    # Step 1: account check
    acct = broker_get_account()
    if not acct:
        return {"error": "Alpaca account unreachable"}

    # Step 2: asset check
    asset_resp = _req.get(f"{base}/v2/assets/{symbol}", headers=headers, timeout=10)
    asset = asset_resp.json() if asset_resp.ok else {"error": asset_resp.text}

    # Step 3: try notional order
    notional_resp = _req.post(f"{base}/v2/orders", headers=headers, timeout=10, json={
        "symbol": symbol, "notional": str(round(amount, 2)),
        "side": "buy", "type": "market", "time_in_force": "day",
    })
    notional_result = {"status": notional_resp.status_code, "body": notional_resp.json() if notional_resp.content else {}}

    # Cancel if submitted so we don't leave test orders open
    if notional_resp.ok:
        order_id = notional_resp.json().get("id")
        if order_id:
            _req.delete(f"{base}/v2/orders/{order_id}", headers=headers, timeout=10)
        return {
            "result": "notional_order_succeeded",
            "account": {"buying_power": acct.get("buying_power"), "status": acct.get("status")},
            "asset": {"fractionable": asset.get("fractionable"), "tradable": asset.get("tradable"), "exchange": asset.get("exchange")},
            "order": notional_result,
        }

    # Step 4: try qty fallback with real live price
    import yfinance as yf
    try:
        fi = yf.Ticker(symbol).fast_info
        ticker_price = fi.last_price  # attribute access, not dict .get()
        if not ticker_price or ticker_price <= 0:
            ticker_price = None
    except Exception as _e:
        print(f"PRICE FETCH: yfinance error for {symbol}: {_e}")
        ticker_price = None

    if ticker_price is None:
        return {
            "result": "qty_fallback_skipped",
            "reason": f"Could not fetch live price for {symbol} — try during market hours",
            "notional_attempt": notional_result,
            "asset": asset,
        }

    qty = _math.floor(amount / ticker_price)
    qty_resp = _req.post(f"{base}/v2/orders", headers=headers, timeout=10, json={
        "symbol": symbol, "qty": str(qty),
        "side": "buy", "type": "market", "time_in_force": "day",
    }) if qty >= 1 else None

    qty_result = None
    if qty_resp:
        qty_result = {"status": qty_resp.status_code, "body": qty_resp.json() if qty_resp.content else {}}
        if qty_resp.ok:
            order_id = qty_resp.json().get("id")
            if order_id:
                _req.delete(f"{base}/v2/orders/{order_id}", headers=headers, timeout=10)

    return {
        "result": "qty_fallback_succeeded" if (qty_resp and qty_resp.ok) else "both_failed",
        "account": {"buying_power": acct.get("buying_power"), "status": acct.get("status")},
        "asset": {"fractionable": asset.get("fractionable"), "tradable": asset.get("tradable"), "exchange": asset.get("exchange")},
        "notional_attempt": notional_result,
        "qty_fallback": qty_result,
        "qty_used": qty,
        "price_used": ticker_price,
    }


@app.get("/api/broker-status")
def broker_status(request: Request):
    if "user" not in request.session:
        return Response(status_code=401)
    if not broker_configured():
        return {"configured": False}
    acct = broker_get_account()
    if acct:
        return {
            "configured":   True,
            "status":       acct.get("status"),
            "buying_power": acct.get("buying_power"),
            "equity":       acct.get("equity"),
        }
    return {"configured": True, "status": "error"}


@app.get("/api/autoai-status")
def autoai_status(request: Request):
    if "user" not in request.session:
        return Response(status_code=401)
    log = get_autoai_log(limit=1)
    last_run = log[0] if log else None
    rules = get_hypothesis_rules()
    auto_active = sum(1 for r in rules if r.get("auto_applied") and r["status"] == "active")
    return {"last_run": last_run, "auto_active_rules": auto_active}


@app.post("/api/chat")
async def api_chat(request: Request):
    """AI model advisor chat endpoint. All logged-in users."""
    if "user" not in request.session:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    username = request.session["user"]
    if not _chat_allowed(username):
        return JSONResponse({"error": "Rate limit exceeded — please wait a moment."}, status_code=429)
    if not _global_claude_allowed():
        return JSONResponse({"error": "Server AI rate limit reached — try again in a minute."}, status_code=429)
    body = await request.json()
    message = (body.get("message") or "").strip()
    history = body.get("history") or []
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    context = {
        "hypothesis_rules": get_hypothesis_rules(),
        "squeeze_weights":  get_squeeze_weights(),
        "autoai_weights":   get_autoai_weights(),
        "opt_data":         get_optimization_data(),
        "feedback":         get_all_feedback()[-8:],
        "closed_trades":    [t for t in get_trade_history() if t.get("exit_price")][-10:],
    }
    result = chat_with_model(message, history, context)
    return JSONResponse(result)


@app.post("/api/chat/execute-action")
async def api_chat_execute(request: Request):
    """Execute an action suggested by the AI advisor. Admin only."""
    if "user" not in request.session:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    if not _require_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    body = await request.json()
    action = body.get("action")

    if action == "activate_rule":
        ok = update_rule_status(int(body["rule_id"]), "active")
        return JSONResponse({"ok": ok, "message": "Rule activated." if ok else "Rule not found."})

    elif action == "reject_rule":
        ok = update_rule_status(int(body["rule_id"]), "rejected")
        return JSONResponse({"ok": ok, "message": "Rule rejected." if ok else "Rule not found."})

    elif action == "add_rule":
        text = (body.get("text") or "").strip()
        if not text:
            return JSONResponse({"ok": False, "message": "Rule text is empty."})
        save_hypothesis_rules([{"text": text, "source": body.get("source", "chat")}])
        return JSONResponse({"ok": True, "message": "Rule added as pending — review it on the Analytics page."})

    elif action == "update_weights":
        model  = body.get("model", "complex")
        weights    = body.get("weights") or {}
        rationale  = body.get("rationale", "")
        summary    = body.get("summary", "Updated via AI chat")
        if not weights:
            return JSONResponse({"ok": False, "message": "No weights provided."})
        if model == "autoai":
            # Merge with existing Auto AI weights so only changed keys are updated
            existing = get_autoai_weights()
            base = existing["weights"].copy() if existing else DEFAULT_SQUEEZE_WEIGHTS.copy()
            base.update({k: int(v) for k, v in weights.items()})
            save_autoai_weights(base, rationale, [], summary, source="ai_chat")
        else:
            existing = get_squeeze_weights()
            base = existing["weights"].copy() if existing else DEFAULT_SQUEEZE_WEIGHTS.copy()
            base.update({k: int(v) for k, v in weights.items()})
            save_squeeze_weights(base, rationale, [], summary, source="ai_chat")
            save_weight_changelog(summary, rationale, base)
        model_label = "Auto AI" if model == "autoai" else "Complex+AI"
        return JSONResponse({"ok": True, "message": f"{model_label} weights updated."})

    elif action == "update_weights_bundle":
        model     = body.get("model", "complex")
        weights   = body.get("weights") or {}
        rationale = body.get("rationale", "")
        summary   = body.get("summary", "Bundle update via AI chat")
        goal      = body.get("goal", "")
        if not weights:
            return JSONResponse({"ok": False, "message": "No weights in bundle."})
        if model == "autoai":
            existing = get_autoai_weights()
            base = existing["weights"].copy() if existing else DEFAULT_SQUEEZE_WEIGHTS.copy()
            base.update({k: int(v) for k, v in weights.items()})
            save_autoai_weights(base, rationale, [], summary, source="ai_chat", goal=goal)
        else:
            existing = get_squeeze_weights()
            base = existing["weights"].copy() if existing else DEFAULT_SQUEEZE_WEIGHTS.copy()
            base.update({k: int(v) for k, v in weights.items()})
            save_squeeze_weights(base, rationale, [], summary, source="ai_chat", goal=goal)
            save_weight_changelog(summary, rationale, base)
        opt_data = get_optimization_data()
        save_bundle_as_rule(goal, weights, rationale, summary, opt_data)
        goal_label = {"combined": "Win Rate + Speed + Upside", "win_rate": "Win Rate", "speed": "Speed-to-Target", "upside": "Max Upside"}.get(goal, goal or "bundle")
        model_label = "Auto AI" if model == "autoai" else "Complex+AI"
        return JSONResponse({"ok": True, "message": f"{model_label} {goal_label} bundle applied — {len(weights)} signals updated."})

    return JSONResponse({"ok": False, "message": f"Unknown action: {action}"}, status_code=400)


@app.post("/api/chat/suggest-action")
async def api_chat_suggest(request: Request):
    """Non-admin: queue an AI-suggested action for admin review."""
    if "user" not in request.session:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    body = await request.json()
    action = body.get("action") or {}
    note = body.get("note", "")
    if not action:
        return JSONResponse({"ok": False, "message": "No action provided."})
    suggestion_id = save_chat_suggestion(request.session["user"], action, note)
    return JSONResponse({"ok": True, "message": "Suggestion sent to admin for review.", "id": suggestion_id})


@app.get("/api/chat/suggestions")
def api_get_suggestions(request: Request):
    """Admin: get pending suggestions from non-admin users."""
    if "user" not in request.session:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    if not _require_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    return JSONResponse({"suggestions": get_chat_suggestions("pending")})


@app.post("/api/chat/suggestions/{suggestion_id}/approve")
async def api_approve_suggestion(request: Request, suggestion_id: int):
    """Admin: approve a suggestion — executes the action then marks it approved."""
    if "user" not in request.session or not _require_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    suggestions = get_chat_suggestions("pending")
    suggestion = next((s for s in suggestions if s["id"] == suggestion_id), None)
    if not suggestion:
        return JSONResponse({"ok": False, "message": "Suggestion not found."}, status_code=404)
    # Execute the action (reuse same logic as execute-action)
    action_body = suggestion["action"]
    fake_request = request  # borrow session; action logic uses body not session
    body = action_body
    action = body.get("action")
    ok_msg = "Approved and executed."
    if action == "activate_rule":
        update_rule_status(int(body["rule_id"]), "active")
    elif action == "reject_rule":
        update_rule_status(int(body["rule_id"]), "rejected")
    elif action == "add_rule":
        save_hypothesis_rules([{"text": body.get("text",""), "source": body.get("source","chat")}])
    elif action in ("update_weights", "update_weights_bundle"):
        model = body.get("model","complex")
        weights = body.get("weights") or {}
        rationale = body.get("rationale","")
        summary = body.get("summary","Approved via chat suggestion")
        goal = body.get("goal", "")
        if model == "autoai":
            existing = get_autoai_weights()
            base = existing["weights"].copy() if existing else DEFAULT_SQUEEZE_WEIGHTS.copy()
            base.update({k: int(v) for k, v in weights.items()})
            save_autoai_weights(base, rationale, [], summary)
        else:
            existing = get_squeeze_weights()
            base = existing["weights"].copy() if existing else DEFAULT_SQUEEZE_WEIGHTS.copy()
            base.update({k: int(v) for k, v in weights.items()})
            save_squeeze_weights(base, rationale, [], summary)
            save_weight_changelog(summary, rationale, base)
        model_label = "Auto AI" if model == "autoai" else "Complex+AI"
        if action == "update_weights_bundle":
            goal_label = {"combined": "Win Rate + Speed + Upside", "win_rate": "Win Rate", "speed": "Speed-to-Target", "upside": "Max Upside"}.get(goal, goal or "bundle")
            opt_data = get_optimization_data()
            save_bundle_as_rule(goal, weights, rationale, summary, opt_data)
            ok_msg = f"{model_label} {goal_label} bundle applied — {len(weights)} signals updated."
        else:
            ok_msg = f"{model_label} weights updated."
    else:
        ok_msg = f"Unknown action type: {action}"
    dismiss_chat_suggestion(suggestion_id, "approved")
    return JSONResponse({"ok": True, "message": ok_msg})


@app.post("/api/chat/suggestions/{suggestion_id}/dismiss")
async def api_dismiss_suggestion(request: Request, suggestion_id: int):
    """Admin: dismiss a suggestion without executing."""
    if "user" not in request.session or not _require_admin(request):
        return JSONResponse({"error": "Admin only"}, status_code=403)
    dismiss_chat_suggestion(suggestion_id, "dismissed")
    return JSONResponse({"ok": True, "message": "Suggestion dismissed."})


@app.post("/admin/test-autoai")
def test_autoai(request: Request):
    """Admin-only: manually trigger the Auto AI optimization loop for testing."""
    if "user" not in request.session:
        return Response(status_code=401)
    if not _require_admin(request):
        return Response(status_code=403)
    try:
        _auto_ai_optimize()
        return {"ok": True, "message": "Auto AI optimization triggered — check autoai_log table"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------- TEST TELEGRAM ----------------
@app.post("/api/test-telegram")
def test_telegram(request: Request):
    if "user" not in request.session:
        return Response(status_code=401)
    from app.alerts import _send_telegram
    _send_telegram("<b>Reno Robs Trading Bot</b>\n\nTelegram alerts are working!")
    return {"ok": True}


@app.get("/api/telegram/bot-info")
def telegram_bot_info(request: Request):
    """Return bot username for QR code generation."""
    if "user" not in request.session:
        return Response(status_code=401)
    token = _os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not configured"}
    try:
        resp = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=5)
        data = resp.json()
        if data.get("ok"):
            return {
                "ok": True,
                "username": data["result"]["username"],
                "first_name": data["result"]["first_name"],
            }
        return {"ok": False, "error": data.get("description", "Unknown error")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/telegram/pending-users")
def telegram_pending_users(request: Request):
    """
    Calls Telegram getUpdates and returns everyone who has messaged the bot,
    flagging which ones are already in the DB.
    """
    if "user" not in request.session:
        return Response(status_code=401)
    token = _os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return {"ok": False, "error": "TELEGRAM_BOT_TOKEN not configured", "users": []}
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params={"limit": 100},
            timeout=10,
        )
        data = resp.json()
        if not data.get("ok"):
            return {"ok": False, "error": data.get("description", "Unknown error"), "users": []}

        existing = {r["chat_id"] for r in get_telegram_recipients()}
        seen_ids = set()
        users = []
        for update in data.get("result", []):
            chat = None
            if "message" in update:
                chat = update["message"].get("chat")
            elif "my_chat_member" in update:
                chat = update["my_chat_member"].get("chat")
            if not chat:
                continue
            chat_id = str(chat.get("id", ""))
            if not chat_id or chat_id in seen_ids:
                continue
            seen_ids.add(chat_id)
            first = chat.get("first_name", "")
            last = chat.get("last_name", "")
            uname = chat.get("username", "")
            name = f"{first} {last}".strip() or uname or chat_id
            users.append({
                "chat_id": chat_id,
                "name": name,
                "username": uname,
                "already_added": chat_id in existing,
            })
        return {"ok": True, "users": users}
    except Exception as e:
        return {"ok": False, "error": str(e), "users": []}


@app.post("/api/telegram/add-recipient")
async def telegram_add_recipient_ajax(request: Request):
    """JSON endpoint for one-click Add from the pending-users list."""
    if "user" not in request.session or not _require_admin(request):
        return {"ok": False, "error": "Unauthorized"}
    body = await request.json()
    chat_id = str(body.get("chat_id", "")).strip()
    label = str(body.get("label", "")).strip()
    if not chat_id:
        return {"ok": False, "error": "chat_id required"}
    added = add_telegram_recipient(chat_id, label)
    if added:
        return {"ok": True}
    return {"ok": False, "error": "already_added"}


# ---------------- USER MANAGEMENT (admin only) ----------------
def _require_admin(request: Request) -> bool:
    """Returns True if the current session user has is_admin=1 in the DB."""
    username = request.session.get("user")
    if not username:
        return False
    return is_user_admin(username)


@app.get("/users", response_class=HTMLResponse)
def users_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login")
    if not _require_admin(request):
        return RedirectResponse("/dashboard")
    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
            "user": request.session["user"],
            "users": get_all_users(),
            "telegram_recipients": get_telegram_recipients(),
        }
    )


@app.post("/users/create")
def users_create(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if "user" not in request.session or not _require_admin(request):
        return RedirectResponse("/login", status_code=303)
    username = username.strip()
    if not username or len(password) < 8:
        return RedirectResponse("/users?error=invalid", status_code=303)
    create_user(username, password)
    return RedirectResponse("/users?created=1", status_code=303)


@app.post("/users/delete/{username}")
def users_delete(request: Request, username: str):
    if "user" not in request.session or not _require_admin(request):
        return RedirectResponse("/login", status_code=303)
    delete_user(username)
    return RedirectResponse("/users", status_code=303)


@app.post("/users/telegram/add")
def telegram_add(
    request: Request,
    chat_id: str = Form(...),
    label: str = Form(""),
):
    if "user" not in request.session or not _require_admin(request):
        return RedirectResponse("/login", status_code=303)
    chat_id = chat_id.strip()
    if not chat_id:
        return RedirectResponse("/users?tg_error=empty", status_code=303)
    added = add_telegram_recipient(chat_id, label)
    if added:
        return RedirectResponse("/users?tg_added=1", status_code=303)
    return RedirectResponse("/users?tg_error=exists", status_code=303)


@app.post("/users/telegram/delete/{chat_id}")
def telegram_delete(request: Request, chat_id: str):
    if "user" not in request.session or not _require_admin(request):
        return RedirectResponse("/login", status_code=303)
    delete_telegram_recipient(chat_id)
    return RedirectResponse("/users", status_code=303)


# ---------------- HYPOTHESIS RULE MANAGEMENT ----------------
@app.post("/hypothesis-rules/{rule_id}/activate")
def hypothesis_rule_activate(request: Request, rule_id: int):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if not _require_admin(request):
        return RedirectResponse("/analytics", status_code=303)
    update_rule_status(rule_id, "active")
    return RedirectResponse("/analytics", status_code=303)


@app.post("/hypothesis-rules/{rule_id}/reject")
def hypothesis_rule_reject(request: Request, rule_id: int):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    if not _require_admin(request):
        return RedirectResponse("/analytics", status_code=303)
    update_rule_status(rule_id, "rejected")
    return RedirectResponse("/analytics", status_code=303)


# ---------------- FEEDBACK BACKUP / RESTORE ----------------
@app.get("/feedback/export")
def feedback_export(request: Request):
    """Download all feedback entries as a JSON backup file."""
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    entries = get_all_feedback()
    content = json.dumps(entries, indent=2, ensure_ascii=False)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=feedback_backup.json"},
    )


@app.get("/export-db")
def export_db(request: Request, token: str = ""):
    """Download the live SQLite database for offline analysis."""
    import os as _os
    _export_token = _os.environ.get("EXPORT_TOKEN") or _os.environ.get("Export_Token", "")
    authed = "user" in request.session or (
        _export_token and token == _export_token
    )
    if not authed:
        return RedirectResponse("/login", status_code=303)
    db_path = _os.environ.get("DB_PATH", "scan_history.db")
    if not _os.path.exists(db_path):
        return Response("Database file not found", status_code=404)
    from datetime import date
    filename = f"scan_history_{date.today().isoformat()}.db"
    return FileResponse(
        path=db_path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/feedback/import-backup")
async def feedback_import_backup(request: Request):
    """Restore feedback entries from an uploaded JSON backup file."""
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    try:
        form = await request.form()
        backup_file = form.get("backup_file")
        if not backup_file or not hasattr(backup_file, "read"):
            return RedirectResponse("/feedback?import_error=no_file", status_code=303)
        content = await backup_file.read()
        entries = json.loads(content)
        if not isinstance(entries, list):
            return RedirectResponse("/feedback?import_error=invalid_format", status_code=303)
        count = import_feedback_from_backup(entries)
        return RedirectResponse(f"/feedback?imported={count}", status_code=303)
    except Exception as _e:
        print(f"FEEDBACK IMPORT: failed — {_e}")
        return RedirectResponse("/feedback?import_error=parse_error", status_code=303)


# ---------------- LOGOUT ----------------
@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")
