# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app locally
uvicorn main:app --host 0.0.0.0 --port 5000 --reload

# Download latest DB from Render (requires EXPORT_TOKEN env var)
make sync          # or: EXPORT_TOKEN=<token> ./scripts/sync_db.sh

# Quick DB summary after sync
make analyze

# Backfill LSTM historical scores (uses historical price windows, not today's data)
venv/bin/python3 -u backfill_lstm.py
venv/bin/python3 -u backfill_lstm.py --reset   # clears existing values first
```

There is no test suite — validation is done via live FinViz cross-checks (`/validation/run`).

## Architecture

**FastAPI app** (`main.py`, ~3.5K lines) with APScheduler background jobs, SQLite persistence, and three ML models.

### Data Flow

```
FinViz screener → score_stock_squeeze() → LSTM inference → Claude AI → Telegram/Email alert → Alpaca order
                                               ↓
                                        save_scan() → DB (scans table)
                                               ↓
                          _autoclose_take_profit() runs every second → close_trade()
```

### Key Modules

| File | Purpose |
|------|---------|
| `main.py` | All FastAPI routes, scheduler, stop-loss enforcement, weekly report |
| `app/database.py` | All SQLite CRUD — scans, trades, settings, signals, weights |
| `app/scanner.py` | FinViz screener + yfinance OHLCV fetcher |
| `app/scoring_engine.py` | Squeeze signal scoring (0–100 pts); `DEFAULT_SQUEEZE_WEIGHTS` |
| `app/ai_agent.py` | Claude API calls: trade rec, price target, weight optimization, stop-loss review |
| `app/lstm_model.py` | Daily LSTM (20 days × 12 features → P(hit +20% in 10 days)) |
| `app/lstm_5m.py` | 5-minute intraday LSTM (12 bars × 7 features) |
| `app/ml_optimizer.py` | XGBoost on `signals_json` flags; auto-activates at 500 labeled rows |
| `app/stop_loss_optimizer.py` | Backtest exit rules; validate proposed param changes |
| `app/backfill.py` | Historical OHLCV scoring + signals; seeds 130+ tickers × 6 years |
| `app/alerts.py` | Telegram + email delivery (`send_scan_alert`, `send_weekly_report_email`) |
| `app/broker.py` | Alpaca paper trading API wrapper |
| `app/validator.py` | FinViz live quote cross-check for scan accuracy audit |

### Scoring Pipeline

`score_stock_squeeze()` returns `{"score": int, "checklist": dict, "fired_signals": dict}`. The `fired_signals` dict is saved as `signals_json TEXT` in the scans table — this is the XGBoost feature input and the per-signal analytics feed.

Scoring weights live in two places: `DEFAULT_SQUEEZE_WEIGHTS` (code default) and the `settings` table key `squeeze_weights` (overrides when present). The XGBoost optimizer can replace these weights entirely once 500+ labeled rows exist.

### ML Model Gates

| Model | Gate | Purpose |
|-------|------|---------|
| Daily LSTM | score >= 50 | P(hit +20% within 10 trading days); trained on `days_to_20pct IS NOT NULL` label |
| 5m LSTM | score >= 75 | P(intraday high hits +20% before 3:50 PM) |
| XGBoost | 500 labeled rows | Auto-replaces `DEFAULT_SQUEEZE_WEIGHTS`; trained on `signals_json` binary flags |

**Critical:** LSTM is validated against `days_to_20pct IS NOT NULL` (10-day window), NOT `next_day_return >= 20`. The gate threshold 55% provides +10.3pp lift; going above 55% adds no additional lift.

### Daily vs 5m Model Split

Daily modes: `squeeze`, `autoai`, `historical`. 5m mode: `fivemin`.

- Alert thresholds, LSTM gating, and stop-loss rules differ by mode — **never apply daily changes to fivemin without explicit confirmation**.
- Stop-loss params are stored per-mode in `settings` table as JSON (`stop_loss_params_daily`, `stop_loss_params_fivemin`).
- Stop-loss changes always queue as pending proposals; **never auto-apply** (unlike weight optimization which can auto-apply at high confidence).

### Scheduler (APScheduler, America/New_York)

| Time | Job |
|------|-----|
| Mon–Fri 6:00 AM | `_daily_backfill()` — LSTM retrain |
| Mon–Fri 8:30 AM | `_premarket_scan()` |
| Mon–Fri 9:45 AM | `_scheduled_scan()` — primary scan + AI + alerts |
| Mon–Fri 10:00–15:30 | `_intraday_scan()` every 5 min |
| Sat 8:00 AM | `_weekly_analysis()` — Telegram + email report |

### Database (SQLite, WAL mode)

Key tables: `scans`, `trades`, `settings`, `users`, `telegram_recipients`, `hypothesis_rules`, `weight_changelog`, `autoai_log`.

The `settings` table is a key/value store used for: current weights, autoai weights, hypothesis rules, stop-loss params + proposals, LSTM threshold config.

Persistent disk on Render at `/data/` — models save to `/data/lstm_model.pt`, `/data/lstm_model_5m.pt`, `/data/ml_model.pkl`, and DB at `/data/scan_history.db`.

**5m LSTM is not persisted on Render** (ephemeral filesystem issue) — must rebuild after each deploy via `/build-5m-sequences`.

### Required Environment Variables

```
ANTHROPIC_API_KEY           Claude API
ALPACA_API_KEY / SECRET_KEY Alpaca paper trading
GMAIL_USER / GMAIL_APP_PASSWORD  Email alerts
ALERT_EMAIL_TO              Alert recipient
WEEKLY_REPORT_RECIPIENTS    Comma-separated: kvgiebel@gmail.com,mboerner32@gmail.com
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  Telegram alerts
SESSION_SECRET              Session encryption (must persist across Render restarts)
DB_PATH                     Default: scan_history.db; Render: /data/scan_history.db
EXPORT_TOKEN                Auth token for /export-db download endpoint
```

### Weekly Analysis Protocol

Trigger phrase: "weekly review", "weekly analysis", or "analyze the DB".

1. User runs `make sync` to download fresh DB
2. Run these queries on `scan_history.db`:
   - Score bucket hit rates (0-44, 45-54, 55-64, 65-74, 75-84, 85+)
   - AI TRADE precision: % that hit +20% within 10 trading days
   - Relvol tier hit rates, daily gain bucket hit rates, day-of-week hit rates
   - Per-signal hit rates from `signals_json` (top 5 / bottom 5)
   - Float/shares bucket hit rates
   - New labeled rows since last review
   - LSTM gate validation: gate ≥55% → % with `days_to_20pct IS NOT NULL` vs baseline
   - Stop-loss status: current daily + 5m params, pending proposals, backtest metrics
   - Trade progress: open positions, closed trades P&L, AI TRADE win rate

Confidence rubric: Low = <10 samples, Medium = 10–30, High = 30+.
