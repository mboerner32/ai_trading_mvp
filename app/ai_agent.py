# ai_trading_mvp/app/ai_agent.py

import json
import os
import re
import base64
import anthropic
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically


def _msg_text(message) -> str:
    """Safely extract text from an Anthropic API response, guarding against empty content."""
    if not message.content:
        raise ValueError("Claude API returned empty content")
    return message.content[0].text.strip()


def get_stock_news(symbol: str, max_headlines: int = 5) -> list[str]:
    """
    Fetch recent news headlines for a symbol via yfinance (free, no API key).
    Returns a list of headline strings (empty list on any error).
    """
    try:
        news = yf.Ticker(symbol).news or []
        headlines = []
        for item in news[:max_headlines]:
            title = item.get("title") or ""
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def _get_lstm_tier_context() -> str:
    """
    Query the live DB for LSTM gate tier hit rates and return a formatted
    context string for injection into the trade prompt. Cached for 1 hour.
    Returns empty string if insufficient data (<30 LSTM-scored rows).
    """
    import sqlite3 as _sq
    import time as _time
    cache = _get_lstm_tier_context.__dict__
    if cache.get("_ts") and _time.monotonic() - cache["_ts"] < 3600 and cache.get("_val") is not None:
        return cache["_val"]
    try:
        from app.database import DB_NAME
        conn = _sq.connect(DB_NAME)
        rows = conn.execute("""
            SELECT gate, COUNT(*) as n,
              ROUND(100.0*SUM(CASE WHEN days_to_20pct IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1) as hit
            FROM (
              SELECT days_to_20pct,
                CASE WHEN lstm_prob>=0.65 THEN '>=65%'
                     WHEN lstm_prob>=0.55 THEN '55-64%'
                     WHEN lstm_prob>=0.45 THEN '45-54%'
                     ELSE '<45%' END as gate
              FROM scans WHERE next_day_return IS NOT NULL AND lstm_prob IS NOT NULL
            ) GROUP BY gate ORDER BY MIN(gate) DESC
        """).fetchall()
        baseline = conn.execute("""
            SELECT COUNT(*),
              ROUND(100.0*SUM(CASE WHEN days_to_20pct IS NOT NULL THEN 1 ELSE 0 END)/COUNT(*),1)
            FROM scans WHERE next_day_return IS NOT NULL AND lstm_prob IS NOT NULL
        """).fetchone()
        conn.close()
        total = sum(r[1] for r in rows)
        if total < 30 or not baseline or not baseline[0]:
            cache["_ts"] = _time.monotonic()
            cache["_val"] = ""
            return ""
        lines = [f"\nLSTM backtested tier performance ({baseline[0]} labeled scans, baseline {baseline[1]}%):"]
        for gate, n, hit in rows:
            if n >= 5:
                vs = round((hit or 0) - (baseline[1] or 0), 1)
                lines.append(f"  LSTM {gate}: {hit}% ({'+' if vs>=0 else ''}{vs}pp vs baseline, n={n})")
        result = "\n".join(lines)
        cache["_ts"] = _time.monotonic()
        cache["_val"] = result
        return result
    except Exception:
        return ""


def parse_rules_from_synthesis(text: str) -> list:
    """
    Extract individual hypothesis rules from a synthesis output string.

    The synthesis format contains a '## Hypotheses' section with a numbered list.
    Each rule is extracted as its own dict: {"text": str, "source": str}.

    Source is inferred from keywords in the rule body:
      "both"       — mentions historical AND feedback/manual
      "historical" — mentions historical data only
      "feedback"   — mentions feedback/manual only
      ""           — unclassified
    """
    if "## Hypotheses" not in text:
        return []

    # Isolate the Hypotheses section (stop at ## Agent Context or end)
    section = text.split("## Hypotheses")[1]
    if "## Agent Context" in section:
        section = section.split("## Agent Context")[0]

    # Find the start position of each numbered item (e.g. "1. ", "2. ")
    pattern = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
    positions = [m.start() for m in pattern.finditer(section)]

    rules = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(section)
        rule_text = section[pos:end].strip()
        # Strip leading "N. "
        rule_text = re.sub(r'^\d+\.\s+', '', rule_text).strip()
        if not rule_text:
            continue
        lower = rule_text.lower()
        has_hist = "historical" in lower
        has_feed = "feedback" in lower or "manual" in lower or "submitted" in lower
        if has_hist and has_feed:
            source = "both"
        elif has_hist:
            source = "historical"
        elif has_feed:
            source = "feedback"
        else:
            source = ""
        rules.append({"text": rule_text, "source": source})

    return rules


def recommend_position_size(stock: dict, available_cash: float,
                            hypothesis: str = None,
                            sizing_stats: dict = None) -> dict:
    """
    Calls Claude to recommend a position size for a given stock.
    Returns {"amount": int, "rationale": str}.
    Falls back to {"amount": 1000, "rationale": ""} on any error.

    hypothesis:    synthesized pattern hypotheses from accumulated user feedback.
    sizing_stats:  quantitative historical performance stats from get_sizing_stats().
                   When provided, the model uses actual win rates/returns to calibrate size.
    """
    checklist = stock.get("checklist", {})

    shares_outstanding = checklist.get("shares_outstanding")
    shares_str = f"{shares_outstanding:,}" if shares_outstanding else "N/A"

    # ---- Historical calibration section ----
    calibration_section = ""
    if sizing_stats and sizing_stats.get("total", 0) >= 10:
        lines = [f"\nHistorical calibration from {sizing_stats['total']} labeled scans:"]

        for label, key in [("Score", "by_score"), ("Rel Vol", "by_relvol"),
                           ("Float", "by_float"), ("Daily gain", "by_gain")]:
            buckets = sizing_stats.get(key, {})
            if buckets:
                parts = []
                for bkt, s in buckets.items():
                    parts.append(f"{bkt}→{s['win_rate']}%hit20pct/{s['avg_return']:+.1f}%avg({s['count']})")
                lines.append(f"  {label}: {' | '.join(parts)}")

        lines.append("Use these stats to calibrate size: higher 20%+ hit rate + positive avg return → larger size.")
        calibration_section = "\n".join(lines)

    # ---- Qualitative hypothesis section ----
    feedback_section = ""
    if hypothesis:
        summary = hypothesis[:400].strip()
        feedback_section = (
            f"\n\nLearned patterns from winning trades:\n{summary}"
        )

    prompt = f"""You are a trading risk manager for a paper trading simulator.
Strategy: buy low-float momentum stocks and target a 20%+ spike the next day.

Stock: {stock['symbol']}
Score: {stock['score']}/100
Recommendation: {stock['recommendation']}
Price: ${stock['price']}
Available cash: ${available_cash:.0f}

Checklist signals:
- Relative Volume: {checklist.get('relative_volume')}
- Shares Outstanding: {shares_str}
- Daily gain today: {checklist.get('five_day_return_pct', 'N/A')}%  (squeeze sweet spot: 20-40%)
- Sideways Compression: {checklist.get('sideways_chop')}
- Yesterday Green: {checklist.get('yesterday_green')}
- Institutional Ownership: {str(checklist.get('institution_pct')) + '%' if checklist.get('institution_pct') is not None else 'N/A'}
- No News Catalyst (organic move): {checklist.get('no_news_catalyst')}{calibration_section}{feedback_section}

Backtested signal insights (544 labeled live scans, 2026-03-09):
- SIZE UP: shares 10–30M + rel_vol ≥50x → 27% hit rate (2.2x baseline) — max position
- SIZE UP: shares 10–30M + daily gain 40–100% → 24.6% hit rate — large position
- SIZE DOWN: shares <10M → 8.1% hit rate (below baseline) — volatile, unreliable
- SIZE DOWN: daily gain 10–20% → 5.9% hit rate — modest gainers rarely follow through

Recommend a position size in dollars. Choose from: $250, $500, $750, $1000, $1500, $2000.
Never recommend more than ${available_cash:.0f} available cash.
Base your sizing on the 20%+ hit rate in the historical calibration stats — higher hit rate = larger position.

Respond in exactly this format (no other text):
AMOUNT: <dollar amount>
RATIONALE: <one sentence, 15 words or less>"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}]
        )
        text = _msg_text(message)

        amount_line = next((l for l in text.split('\n') if l.startswith('AMOUNT:')), None)
        rationale_line = next((l for l in text.split('\n') if l.startswith('RATIONALE:')), None)

        if not amount_line:
            return {"amount": 1000, "rationale": ""}

        amount = int(
            amount_line.replace('AMOUNT:', '').strip()
            .replace('$', '').replace(',', '')
        )
        rationale = rationale_line.replace('RATIONALE:', '').strip() if rationale_line else ""

        # Snap to nearest valid tier
        valid = [250, 500, 750, 1000, 1500, 2000]
        amount = min(valid, key=lambda x: abs(x - amount))

        # Never exceed available cash
        amount = min(amount, int(available_cash))
        amount = max(amount, 0)

        return {"amount": amount, "rationale": rationale}

    except Exception:
        return {"amount": 1000, "rationale": ""}


def predict_price_target(stock: dict, sizing_stats: dict = None,
                         hypothesis: str = None) -> dict:
    """
    Predicts a take-profit % target for high-scoring stocks (called for score >= 75).
    Returns {"target_pct": int, "rationale": str}.
    Falls back to {"target_pct": 20, "rationale": ""} on any error.
    """
    checklist = stock.get("checklist", {})
    shares_outstanding = checklist.get("shares_outstanding")
    shares_str = f"{shares_outstanding:,}" if shares_outstanding else "N/A"

    calibration_section = ""
    if sizing_stats and sizing_stats.get("total", 0) >= 10:
        lines = [f"\nHistorical calibration ({sizing_stats['total']} labeled scans):"]
        for label, key in [("Score", "by_score"), ("Rel Vol", "by_relvol"),
                           ("Float", "by_float"), ("Daily gain", "by_gain")]:
            buckets = sizing_stats.get(key, {})
            if buckets:
                parts = [
                    f"{bkt}→{s['win_rate']}%hit20pct/{s['avg_return']:+.1f}%avg({s['count']})"
                    for bkt, s in buckets.items()
                ]
                lines.append(f"  {label}: {' | '.join(parts)}")
        calibration_section = "\n".join(lines)

    hypothesis_section = ""
    if hypothesis:
        hypothesis_section = (
            f"\n\nLearned patterns from winning trades:\n{hypothesis[:300].strip()}"
        )

    prompt = f"""You are a momentum trading analyst for a paper trading simulator.
Strategy: buy low-float stocks spiking on unusual volume and hold for a 1-2 day pump.

Stock: {stock['symbol']}
Score: {stock['score']}/100
Recommendation: {stock['recommendation']}
Price: ${stock['price']}
Relative Volume: {checklist.get('relative_volume')}
Shares Outstanding: {shares_str}
Daily gain today: {stock.get('daily_return_pct', 'N/A')}%
Sideways Compression: {checklist.get('sideways_chop')}
Relvol Tier: {checklist.get('relvol_tier', 'N/A')}
Float Tier: {checklist.get('float_tier', 'N/A')}{calibration_section}{hypothesis_section}

Backtested avg winner return by combo (544 labeled live scans, 2026-03-09):
- shares 10–30M + daily gain 40–100%: avg winner return 54.8% → target 35–50%
- shares 10–30M + rel_vol ≥50x: avg winner return 42.8% → target 30–40%
- shares 10–30M alone: avg winner return 47.2% → target 30–35%
- daily gain 10–20%: avg winner return 43.9% but only 5.9% ever hit 20% — don't size up
- shares <10M: avg winner return 53.3% but unreliable hit rate (8.1%) — keep target conservative

Predict the best take-profit target for this trade.
Higher relative volume + smaller float + barcoded compression = more room to run.
Choose one from: 20%, 25%, 30%, 35%, 40%, 50%

Respond in exactly this format (no other text):
TARGET: <number only>
RATIONALE: <one sentence, 15 words or less>"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}]
        )
        text = _msg_text(message)
        target_line    = next((l for l in text.split('\n') if l.startswith('TARGET:')), None)
        rationale_line = next((l for l in text.split('\n') if l.startswith('RATIONALE:')), None)
        if not target_line:
            return {"target_pct": 20, "rationale": ""}
        target_pct = int(
            target_line.replace('TARGET:', '').strip().replace('%', '').replace(',', '')
        )
        rationale = rationale_line.replace('RATIONALE:', '').strip() if rationale_line else ""
        valid = [20, 25, 30, 35, 40, 50]
        target_pct = min(valid, key=lambda x: abs(x - target_pct))
        return {"target_pct": target_pct, "rationale": rationale}
    except Exception:
        return {"target_pct": 20, "rationale": ""}


def recommend_trade(stock: dict, hypothesis: str = None,
                    sizing_stats: dict = None,
                    ticker_history: list = None,
                    lstm_prob: float = None,
                    news_headlines: list = None,
                    ai_accuracy: dict = None,
                    per_signal_stats: dict = None) -> dict:
    """
    Makes TRADE/NO_TRADE call using five context sources:
    hypothesis, market context (day-of-week), ticker history, historical calibration,
    and local LSTM model probability.
    Returns {"decision": "TRADE"|"NO_TRADE", "confidence": "HIGH"|"MEDIUM"|"LOW", "rationale": str}
    Falls back to {"decision": "NO_TRADE", "confidence": "LOW", "rationale": ""} on error.
    """
    checklist = stock.get("checklist", {})
    shares_outstanding = checklist.get("shares_outstanding")
    shares_str = f"{shares_outstanding:,}" if shares_outstanding else "N/A"

    # Context 1: historical calibration
    calibration_section = ""
    if sizing_stats and sizing_stats.get("total", 0) >= 10:
        lines = [f"\nHistorical calibration ({sizing_stats['total']} labeled scans):"]
        for label, key in [("Score", "by_score"), ("Rel Vol", "by_relvol"),
                           ("Float", "by_float"), ("Daily gain", "by_gain")]:
            buckets = sizing_stats.get(key, {})
            if buckets:
                parts = [f"{bkt}→{s['win_rate']}%hit20pct/{s['avg_return']:+.1f}%avg({s['count']})"
                         for bkt, s in buckets.items()]
                lines.append(f"  {label}: {' | '.join(parts)}")
        calibration_section = "\n".join(lines)

    # Context 2: hypothesis patterns
    hypothesis_section = ""
    if hypothesis:
        hypothesis_section = f"\nLearned patterns:\n{hypothesis[:700].strip()}"

    # Context 3: recent scan history for this ticker
    history_section = ""
    if ticker_history:
        lines = ["\nPrior scans for this ticker (most recent first):"]
        for h in ticker_history[:3]:
            nd = f" → next_day: {h['next_day']:+.1f}%" if h.get("next_day") else ""
            relvol = h.get("relvol") or 0
            lines.append(f"  {h['timestamp']}: score={h['score']}, relvol={relvol:.1f}x{nd}")
        history_section = "\n".join(lines)

    # Context 4: market context (day of week) — clean deduped data, days_to_20pct metric, 2026-03-17
    # Baseline: 49.4% (n=2432 unique symbol-days, daily modes only, confirmed outcomes)
    day_name = datetime.now().strftime("%A")
    _dow_data = {
        "Monday":    ("51.5%", "+2.1pp", "normal scrutiny"),
        "Tuesday":   ("46.5%", "-2.9pp", "slightly below baseline — apply mild additional scrutiny"),
        "Wednesday": ("43.7%", "-5.7pp", "ELEVATED SCRUTINY — historically weakest day, 7pp below Friday"),
        "Thursday":  ("51.9%", "+2.5pp", "normal scrutiny — above baseline, good day for setups"),
        "Friday":    ("53.2%", "+3.9pp", "normal scrutiny — historically strongest day"),
    }
    _dow = _dow_data.get(day_name, ("N/A", "N/A", "normal scrutiny"))
    market_section = (f"\nMarket context: Today is {day_name} "
                      f"(hit rate: {_dow[0]}, {_dow[1]} vs 49.4% baseline — {_dow[2]}). "
                      f"Only Wednesday shows meaningful underperformance historically.")

    # Context 5: LSTM model prediction + live backtested tier performance
    _lstm_tier_context = _get_lstm_tier_context()
    if lstm_prob is not None:
        if lstm_prob >= 0.65:
            _lstm_interp = "STRONG signal — historically the highest-performing tier."
        elif lstm_prob >= 0.55:
            _lstm_interp = "POSITIVE signal — above baseline, meaningful edge."
        elif lstm_prob >= 0.45:
            _lstm_interp = "WEAK signal — near baseline, limited predictive edge."
        else:
            _lstm_interp = "NEGATIVE signal — below baseline. Require 3+ strong confirming signals to override."
        lstm_section = (
            f"\nLSTM model: {lstm_prob:.0%} probability this stock hits +20% within 10 trading days. "
            f"{_lstm_interp}"
            f"{_lstm_tier_context}"
        )
    else:
        lstm_section = (
            "\nLSTM model: unavailable (insufficient price history for this symbol). "
            "Treat as a missing signal — do NOT assume the setup is strong. "
            f"Require 3+ strong confirming signals to call TRADE without LSTM."
            f"{_lstm_tier_context}"
        )

    # Context 6: recent news headlines
    news_section = ""
    if news_headlines:
        lines = ["\nRecent news headlines:"]
        for h in news_headlines[:5]:
            lines.append(f"  • {h}")
        news_section = "\n".join(lines)

    # Context 7: per-signal performance for the signals that fired on THIS stock
    signal_perf_section = ""
    if per_signal_stats and per_signal_stats.get("baseline"):
        baseline = per_signal_stats["baseline"]
        if baseline.get("count", 0) >= 5:
            fired = checklist.get("fired_signals", {})
            signal_rows = {s["key"]: s for s in per_signal_stats.get("signals", [])}
            lines = []
            for key in fired:
                if key in signal_rows:
                    s = signal_rows[key]
                    if s.get("count", 0) >= 5:
                        vs = s["hit_20pct"] - baseline["hit_20pct"]
                        sign = "+" if vs >= 0 else ""
                        lines.append(
                            f"  {key}: {s['hit_20pct']}% hit 20%+ "
                            f"({sign}{vs:.0f}pp vs {baseline['hit_20pct']}% baseline, n={s['count']})"
                        )
            if lines:
                signal_perf_section = "\nFired-signal backtest performance:\n" + "\n".join(lines)

    # Context 8: AI self-calibration — own historical accuracy
    accuracy_section = ""
    if ai_accuracy and ai_accuracy.get("total_resolved", 0) >= 5:
        parts = [f"\nYour past AI call accuracy ({ai_accuracy['total_resolved']} resolved calls):"]
        if "trade" in ai_accuracy:
            t = ai_accuracy["trade"]
            parts.append(
                f"  TRADE calls: {t['hit_20pct']}% hit +20% within 10 trading days  |  "
                f"{t['missed']}% missed  ({t['count']} calls)"
            )
        if "no_trade" in ai_accuracy:
            nt = ai_accuracy["no_trade"]
            parts.append(
                f"  NO_TRADE calls: {nt['correct']}% correctly avoided  |  "
                f"{nt['missed_moves']}% missed a real move  ({nt['count']} calls)"
            )
        if "trade" in ai_accuracy:
            hit_rate = ai_accuracy["trade"]["hit_20pct"]
            if hit_rate < 50:
                parts.append(
                    f"  → TRADE precision is {hit_rate:.0f}% — target is 80%+. "
                    "Only call TRADE when multiple strong signals clearly align. "
                    "Be selective: a high-quality NO_TRADE is better than a low-conviction TRADE."
                )
            elif hit_rate < 80:
                parts.append(
                    f"  → TRADE precision is {hit_rate:.0f}% — approaching the 80% target. "
                    "Stay selective and keep raising the bar."
                )
            else:
                parts.append(
                    f"  → TRADE precision is {hit_rate:.0f}% — at or above the 80% target. "
                    "Maintain this standard."
                )
        accuracy_section = "\n".join(parts)

    relvol_raw = checklist.get('relative_volume')
    relvol_tier_label = (
        "Extreme (≥500x)" if relvol_raw and relvol_raw >= 500 else
        "Exceptional (≥100x)" if relvol_raw and relvol_raw >= 100 else
        "Very High (≥50x)" if relvol_raw and relvol_raw >= 50 else
        "Ideal (≥25x)" if relvol_raw and relvol_raw >= 25 else
        "Good (≥10x)" if relvol_raw and relvol_raw >= 10 else
        "Adequate (≥5x)" if relvol_raw and relvol_raw >= 5 else "Low"
    )

    prompt = f"""You are a momentum trading AI making a TRADE or NO_TRADE call. Treat this as if you are trading with your own money — every TRADE call costs real capital, and losses come out of your pocket. Be selective. Be precise.

Strategy: identify low-float microcaps with unusual volume that will hit +20% above alert price within 5 trading days (77% of winners hit Day 1, 97% by Day 5). Target precision: 60%+ of TRADE calls hit +20% (baseline is 49.4% — you need a clear edge above that). Be selective but not paralyzed — 2–5 quality TRADE calls per day is the goal.

Stock: {stock['symbol']} | Score: {stock['score']}/100 | Price: ${stock['price']}
Signals:
- Relative Volume: {relvol_raw}x [{relvol_tier_label}]
- Shares Outstanding: {shares_str}
- Daily gain today: {stock.get('daily_return_pct', 'N/A')}%
- Sideways Compression: {checklist.get('sideways_chop')}
- Yesterday Green: {checklist.get('yesterday_green')}
- Institutional Ownership: {str(checklist.get('institution_pct')) + '%' if checklist.get('institution_pct') is not None else 'N/A'}
- Sector/Industry: {checklist.get('sector') or 'N/A'} / {checklist.get('industry') or 'N/A'}{signal_perf_section}{calibration_section}{hypothesis_section}{history_section}{market_section}{lstm_section}{news_section}{accuracy_section}

BACKTESTED PERFORMANCE (2,438 deduped symbol-days, clean methodology, 2026-03-17):
Baseline: 49.4% hit +20% within 10 trading days.
Relvol tiers:
  ≥500x:    100% (n=18)  — extreme event, very strong signal
  100–499x: 100% (n=39)  — exceptional event, strong signal
  50–99x:   45.5% (-3.9pp, n=297) — slightly below baseline; 1–2 confirming signals needed
  25–49x:   49.5% (-0.4pp, n=687) — at baseline; standard conviction required
  10–24x:   48.2% (-1.7pp, n=1374) — near baseline; standard conviction required

Float/shares tiers:
  10–30M:   53.8% (+3.9pp, n=515) — strongest tier, positive signal
  <10M:     48.9% (-1.0pp, n=427) — at baseline, not a meaningful edge alone
  30–100M:  48.3% (-1.7pp, n=717) — at baseline
  >100M:    filtered out

Daily gain tiers:
  40–100%:  56.7% (+6.8pp, n=723) — strongest daily gain signal
  20–40%:   49.4% (-0.5pp, n=952) — at baseline
  10–20%:   44.0% (-5.9pp, n=754) — below baseline, weaker setup

Key findings:
- institution_strong (40%+ ownership) is HARMFUL — do NOT treat high institutional ownership as positive.
- 77% of winners hit within Day 1, 97% by Day 5. Ask: will this stock move +20% within 5 trading days?
- 10–30M float is the sweet spot. <10M is not inherently better than larger float.
- Daily gain 40–100% (+6.8pp) is a meaningful positive. 20–40% is neutral.

RELVOL SCRUTINY TIERS:
- ≥100x relvol: clear edge — standard conviction required.
- 50–99x relvol: meaningful signal — require 2+ confirming signals (float <30M, sideways compression, biotech catalyst, LSTM ≥55%).
- 25–49x relvol: HIGH SCRUTINY — win rate drops significantly here. Require ALL of: float <30M + LSTM ≥60% + 1 other strong signal. Do not call TRADE on relvol alone.
- 10–24x relvol: HIGH SCRUTINY — weak setup. Require LSTM ≥65% + float <30M + at least 2 other strong signals. Very few setups justify TRADE at this tier.

FLOAT/SHARES SCRUTINY:
- <30M shares: acceptable — standard scrutiny applies.
- 30–100M shares: HIGH SCRUTINY — win rate drops significantly. Require relvol ≥100x OR LSTM ≥65% to compensate. Do not call TRADE on moderate relvol + large float alone.

TRADE calls require genuine conviction. Ask: what specific edge does this stock have over the baseline 49.4% hit rate?

Confidence criteria:
- HIGH: relvol ≥100x + float <30M + 1+ other signals aligned, no red flags
- MEDIUM: relvol 50–99x with 2+ confirming signals, OR relvol 25–49x with LSTM ≥60% + float <30M
- LOW: weak relvol or large float with limited supporting signals — usually NO_TRADE territory

Respond EXACTLY (no other text):
DECISION: TRADE or NO_TRADE
CONFIDENCE: HIGH or MEDIUM or LOW
RATIONALE: <one sentence, 15 words or less>"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}]
        )
        text = _msg_text(message)
        lines_map = {}
        for line in text.split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                lines_map[key.strip()] = val.strip()
        decision = lines_map.get("DECISION", "NO_TRADE").upper()
        confidence = lines_map.get("CONFIDENCE", "LOW").upper()
        rationale = lines_map.get("RATIONALE", "")
        if decision not in ("TRADE", "NO_TRADE"):
            decision = "NO_TRADE"
        if confidence not in ("HIGH", "MEDIUM", "LOW"):
            confidence = "LOW"
        return {"decision": decision, "confidence": confidence, "rationale": rationale}
    except Exception as _e:
        print(f"AI make_trade_call error for {stock.get('symbol', '?')}: {_e}")
        return {"decision": "NO_TRADE", "confidence": "LOW", "rationale": f"AI error: {_e}"}


def synthesize_combined_hypothesis(feedback_entries: list,
                                   opt_data: dict,
                                   historical_count: int,
                                   prior_hypothesis: str = None) -> str:
    """
    Unified synthesis combining qualitative manual feedback AND quantitative
    historical scan data. Evolves the prior hypothesis rather than replacing it.
    """
    has_feedback   = bool(feedback_entries)
    has_historical = bool(opt_data and historical_count > 0)
    if not has_feedback and not has_historical:
        return ""

    # Build feedback section
    if has_feedback:
        entries_text = ""
        for i, fb in enumerate(feedback_entries, 1):
            sym = fb.get("symbol") or "Unknown"
            user_text = (fb.get("user_text") or "").strip()
            chart_analysis = (fb.get("chart_analysis") or "").strip()
            date = (fb.get("created_at") or "")[:10]
            entries_text += f"\n--- Entry {i} ({date}) — {sym} ---\n"
            if user_text:
                entries_text += f"User notes: {user_text}\n"
            if chart_analysis:
                entries_text += f"Chart analysis: {chart_analysis[:400]}\n"
        feedback_section = (
            f"## Manual Submissions ({len(feedback_entries)} winning trades)\n"
            + entries_text
        )
    else:
        feedback_section = "## Manual Submissions\nNone submitted yet."

    # Build historical section
    if has_historical:
        def fmt_bucket(name, stats):
            if stats["count"] == 0:
                return None
            hit = stats.get("hit_20pct", "?")
            days = stats.get("avg_days_to_20pct")
            days_str = f" (avg {days}d to target)" if days is not None else ""
            return (f"  {name}: {stats['count']} trades, "
                    f"{hit}% hit 20%+ target{days_str}, "
                    f"{stats['win_rate']}% any-gain, "
                    f"{stats['avg_return']:+.1f}% avg next-day return")

        relvol_lines = [fmt_bucket(k, v) for k, v in opt_data.get("relative_volume", {}).items()]
        gain_lines   = [fmt_bucket(k, v) for k, v in opt_data.get("daily_gain", {}).items()]
        shares_lines = [fmt_bucket(k, v) for k, v in opt_data.get("shares_outstanding", {}).items()]

        relvol_text  = "\n".join(l for l in relvol_lines  if l) or "  No data"
        gain_text    = "\n".join(l for l in gain_lines    if l) or "  No data"
        shares_text  = "\n".join(l for l in shares_lines  if l) or "  No data"

        historical_section = (
            f"## Historical Scan Data ({historical_count} labeled examples)\n"
            f"Relative Volume buckets:\n{relvol_text}\n\n"
            f"Daily gain buckets:\n{gain_text}\n\n"
            f"Shares outstanding buckets:\n{shares_text}"
        )
    else:
        historical_section = "## Historical Scan Data\nNone available."

    prior_section = ""
    if prior_hypothesis and prior_hypothesis.strip():
        prior_section = f"""## Prior Hypothesis (evolve this — retain confirmed patterns, upgrade confidence on new evidence, add new findings)
{prior_hypothesis[:1200].strip()}

"""

    prompt = f"""You are evolving a set of trading pattern hypotheses over time using new evidence.
Sources:
1. {len(feedback_entries)} manually submitted winning trades (qualitative)
2. {historical_count} labeled historical scan examples (quantitative)

GOAL: Identify which setup characteristics most reliably predict a 20%+ intraday price spike within 7 trading days.
Setups that hit 20%+ FASTER are higher quality. Build on prior findings — do not discard confirmed patterns.

{prior_section}{feedback_section}

{historical_section}

Instructions:
- KEEP confirmed patterns from the prior hypothesis; upgrade confidence if new data supports them
- ADD new patterns found in this data that aren't already captured
- REMOVE patterns that new evidence contradicts
- For each hypothesis: state the pattern, source (manual/historical/both), avg days-to-20% if known,
  confidence (HIGH/EMERGING/PRELIMINARY), and position sizing implication
- Aim for 5-8 total hypotheses (quality over quantity)

Then write an AGENT CONTEXT section: a compact 3-4 sentence summary of the strongest signals,
formatted for direct injection into a trading AI agent prompt.

Format exactly:
## Hypotheses

[numbered list]

## Agent Context
[compact paragraph]"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}]
        )
        return _msg_text(message)
    except Exception as e:
        return f"Synthesis unavailable: {str(e)}"


def synthesize_feedback_hypotheses(feedback_entries: list) -> str:
    """
    Reads all accumulated user feedback and synthesizes cross-entry patterns
    into testable hypotheses. Returns hypothesis text to be stored and reused.
    Uses claude-sonnet-4-6 for pattern recognition across multiple entries.
    """
    if not feedback_entries:
        return ""

    # Build a structured summary of all submissions
    entries_text = ""
    for i, fb in enumerate(feedback_entries, 1):
        sym = fb.get("symbol") or "Unknown"
        user_text = (fb.get("user_text") or "").strip()
        chart_analysis = (fb.get("chart_analysis") or "").strip()
        date = (fb.get("created_at") or "")[:10]
        entries_text += f"\n--- Entry {i} ({date}) — {sym} ---\n"
        if user_text:
            entries_text += f"User notes: {user_text}\n"
        if chart_analysis:
            entries_text += f"Chart analysis: {chart_analysis}\n"

    n = len(feedback_entries)

    prompt = f"""You are analyzing {n} winning trade submissions from a trader to identify recurring patterns and form testable hypotheses.

{entries_text}

Based on ALL submissions above, identify patterns that appear across multiple entries (not just one). For each hypothesis:
- State what the pattern is
- Note how many entries support it (even if indirectly)
- Rate confidence: HIGH (3+ entries), EMERGING (2 entries), PRELIMINARY (1 entry but notable)
- State how this should influence position sizing (size up, size down, or no effect)

Then write a AGENT CONTEXT section: a compact 3-4 sentence summary of the highest-confidence hypotheses formatted for injection into a trading AI agent.

Format exactly:
## Hypotheses

[numbered list]

## Agent Context
[compact paragraph]"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return _msg_text(message)
    except Exception as e:
        return f"Synthesis unavailable: {str(e)}"


def synthesize_historical_hypothesis(opt_data: dict, historical_count: int) -> str:
    """
    Generates a hypothesis from quantitative historical backtest data (win rates by bucket).
    Stores the result in the same format as synthesize_feedback_hypotheses.
    """
    if not opt_data:
        return ""

    def fmt_bucket(name, stats):
        if stats["count"] == 0:
            return None
        hit = stats.get("hit_20pct", "?")
        days = stats.get("avg_days_to_20pct")
        days_str = f" (avg {days}d to target)" if days is not None else ""
        return (f"  {name}: {stats['count']} trades, "
                f"{hit}% hit 20%+ target{days_str}, "
                f"{stats['win_rate']}% any-gain, "
                f"{stats['avg_return']:+.1f}% avg next-day return")

    relvol_lines  = [fmt_bucket(k, v) for k, v in opt_data.get("relative_volume", {}).items()]
    gain_lines    = [fmt_bucket(k, v) for k, v in opt_data.get("daily_gain", {}).items()]
    shares_lines  = [fmt_bucket(k, v) for k, v in opt_data.get("shares_outstanding", {}).items()]

    relvol_text  = "\n".join(l for l in relvol_lines  if l)
    gain_text    = "\n".join(l for l in gain_lines    if l)
    shares_text  = "\n".join(l for l in shares_lines  if l)

    prompt = f"""You are analyzing {historical_count} historical momentum/squeeze trades to find which signals predict a 20%+ price spike within 7 trading days.

GOAL: The strategy buys after a scan signal and targets a 20%+ gain. Setups that hit 20%+ in fewer days are the strongest signals — avg_days_to_20pct lower = higher quality setup.
Identify which bucket characteristics most reliably produce that outcome AND hit it fastest.

Relative Volume buckets (next-day return stats after scan day):
{relvol_text}

Today's gain buckets:
{gain_text}

Shares outstanding buckets:
{shares_text}

Identify the strongest statistical patterns predicting 20%+ spikes. For each hypothesis:
- State what the signal predicts about the 20%+ outcome
- Reference the supporting hit rate, avg return, and avg days-to-target data
- Lower avg_days_to_20pct = faster to target = stronger signal quality
- Rate confidence: HIGH (many trades, clear edge), EMERGING (positive pattern, moderate sample), PRELIMINARY (limited data)
- State how this should influence position sizing (size up when 20%+ hit rate is highest AND days-to-target is lowest)

Then write an AGENT CONTEXT section: a compact 3-4 sentence summary specifically about
which signals predict 20%+ next-day spikes, for injection into a trading AI agent.

Format exactly:
## Hypotheses

[numbered list]

## Agent Context
[compact paragraph]"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        return _msg_text(message)
    except Exception as e:
        return f"Synthesis unavailable: {str(e)}"


def analyze_and_optimize_weights(opt_data: dict) -> str:
    """
    Analyzes per-bucket backtest performance and recommends scoring weight
    adjustments to maximize 20%+ next-day spike rate.
    Uses claude-sonnet-4-6 for richer reasoning.
    """
    def fmt(stats):
        if not stats or stats["count"] == 0:
            return "No data"
        hit = stats.get("hit_20pct", 0)
        days = stats.get("avg_days_to_20pct")
        days_str = f" (avg {days}d to target)" if days is not None else ""
        return (f"{stats['count']} trades | "
                f"{hit}% hit 20%+{days_str} | "
                f"{stats['win_rate']}% any-gain | "
                f"{stats['avg_return']:+.1f}% avg return")

    rv = opt_data["relative_volume"]
    dg = opt_data["daily_gain"]
    so = opt_data["shares_outstanding"]

    prompt = f"""You are a quantitative trading system optimizer.
Goal: maximize the rate at which top-scored stocks spike 20%+ the next day (our take-profit target).
Analyze this backtested signal performance and recommend specific weight adjustments.

TOTAL BACKTESTED TRADES: {opt_data['total_trades']}

RELATIVE VOLUME (current scoring: ≥29x → +2pts, ≥10x → +1pt):
- ≥50x:    {fmt(rv.get(">=50x"))}
- 25–50x:  {fmt(rv.get("25-50x"))}
- 10–25x:  {fmt(rv.get("10-25x"))}
- <10x:    {fmt(rv.get("<10x"))}

DAILY GAIN TODAY (current scoring: 10–40% → +2pts, else → 0pts):
- 20–40%:   {fmt(dg.get("20-40%"))}
- 10–20%:   {fmt(dg.get("10-20%"))}
- 40–100%:  {fmt(dg.get("40-100%"))}
- <10%:     {fmt(dg.get("<10%"))}

SHARES OUTSTANDING (current scoring: <5M → +2pts, else → 0pts):
- <1M:    {fmt(so.get("<1M"))}
- 1–5M:   {fmt(so.get("1-5M"))}
- 5–10M:  {fmt(so.get("5-10M"))}
- 10–30M: {fmt(so.get("10-30M"))}
- 30M+:   {fmt(so.get("30M+"))}

For each signal, state whether to increase, decrease, split, or keep the current weights.
Focus primarily on the "hit 20%+" column AND avg days-to-target — lower days = faster to target = higher quality signal.
Upweight signals with both high hit rate AND fast time-to-target.
Also suggest 3 new criteria not currently tracked that could improve 20%+ prediction speed and reliability.

Format your response with these exact section headers:
## Relative Volume Weights
## Daily Gain Weights
## Shares Outstanding Weights
## Top 3 Changes to Implement Now
## New Criteria to Track"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}]
        )
        return _msg_text(message)
    except Exception as e:
        return f"Analysis unavailable: {str(e)}"


def optimize_complex_ai_weights(
    opt_data,
    all_feedback: list,
    current_weights: dict,
    hypothesis: str = None,
) -> dict:
    """
    Analyzes backtesting signal performance + synthesized hypothesis to produce
    AI-optimized scoring weights for Complex + AI Mode.

    The hypothesis (from synthesize_combined_hypothesis) is the PRIMARY driver:
    weight changes should reflect patterns the hypothesis identified as predictive.

    Returns:
        {
            "weights":     dict of new weight values,
            "rationale":   str explanation,
            "suggestions": list of new signal ideas,
            "summary":     one-liner for display,
        }
    Falls back to current_weights on any error (adds "error" key).
    """
    import json

    REQUIRED_KEYS = [
        "rel_vol_500x", "rel_vol_100x",
        "rel_vol_50x", "rel_vol_25x", "rel_vol_10x", "rel_vol_5x",
        "daily_sweet_20_40", "daily_ok_10_20", "daily_ok_40_100",
        "sideways_chop", "yesterday_green",
        "shares_lt10m", "shares_lt30m", "shares_lt100m",
        "no_news_bonus", "high_cash_bonus",
        "institution_moderate", "institution_strong",
    ]

    # ---- Format backtest section ----
    if opt_data and opt_data.get("total_trades", 0) >= 5:
        def fmt(stats):
            if not stats or stats["count"] == 0:
                return "No data"
            hit = stats.get("hit_20pct", 0)
            days = stats.get("avg_days_to_20pct")
            days_str = f" (avg {days}d to target)" if days is not None else ""
            return (f"{stats['count']} trades | "
                    f"{hit}% hit 20%+ target{days_str} | "
                    f"{stats['win_rate']}% any-gain | "
                    f"{stats['avg_return']:+.1f}% avg return")

        rv = opt_data["relative_volume"]
        dg = opt_data["daily_gain"]
        so = opt_data["shares_outstanding"]

        backtest_section = f"""BACKTESTED SIGNAL PERFORMANCE ({opt_data['total_trades']} trades):
Goal: predict which setups will spike 20%+ the next day.

Relative Volume:
- >=50x:   {fmt(rv.get(">=50x"))}
- 25-50x:  {fmt(rv.get("25-50x"))}
- 10-25x:  {fmt(rv.get("10-25x"))}
- <10x:    {fmt(rv.get("<10x"))}

Daily Gain:
- 20-40%:  {fmt(dg.get("20-40%"))}
- 10-20%:  {fmt(dg.get("10-20%"))}
- 40-100%: {fmt(dg.get("40-100%"))}
- <10%:    {fmt(dg.get("<10%"))}

Shares Outstanding:
- <1M:     {fmt(so.get("<1M"))}
- 1-5M:    {fmt(so.get("1-5M"))}
- 5-10M:   {fmt(so.get("5-10M"))}
- 10-30M:  {fmt(so.get("10-30M"))}
- 30M+:    {fmt(so.get("30M+"))}"""
    else:
        backtest_section = (
            "BACKTESTED SIGNAL PERFORMANCE: Insufficient data "
            "(<5 trades with known outcomes). Base adjustments on hypothesis patterns."
        )

    # ---- Format hypothesis / feedback section ----
    if hypothesis:
        # Pull out Agent Context block for a compact summary if present
        if "## Agent Context" in hypothesis:
            agent_ctx = hypothesis.split("## Agent Context")[1].strip()[:500]
            feedback_section = (
                f"SYNTHESIZED HYPOTHESIS (PRIMARY — use this to drive weight changes):\n"
                f"{hypothesis[:1000]}\n\n"
                f"AGENT CONTEXT SUMMARY:\n{agent_ctx}"
            )
        else:
            feedback_section = (
                f"SYNTHESIZED HYPOTHESIS (PRIMARY — use this to drive weight changes):\n"
                f"{hypothesis[:1400]}"
            )
    elif all_feedback:
        lines = ""
        for fb in all_feedback[-6:]:
            sym = fb.get("symbol") or "?"
            txt = (fb.get("user_text") or "")[:180]
            lines += f"\n- {sym}: {txt}"
        feedback_section = f"TRADER FEEDBACK (last {min(6, len(all_feedback))} submissions):{lines}"
    else:
        feedback_section = "TRADER FEEDBACK: None submitted yet."

    # Show current weights for context (only the required keys)
    current_display = {k: current_weights.get(k, 0) for k in REQUIRED_KEYS}

    prompt = f"""You are optimizing scoring weights for a Complex + AI stock scanner targeting short-squeeze momentum setups in low-float microcaps.

## Current Active Weights (integer scale 0–50; the scorer normalises output to 0–100 automatically)
{json.dumps(current_display, indent=2)}

Weight definitions — tiered signals (only the highest matching tier scores per signal):
Relative Volume tiers:
  rel_vol_500x:  pts when rel vol >= 500x (current default: 40) — extreme event
  rel_vol_100x:  pts when rel vol >= 100x (current default: 35)
  rel_vol_50x:   pts when rel vol >= 50x (current default: 30)
  rel_vol_25x:   pts when rel vol >= 25x (current default: 22)
  rel_vol_10x:   pts when rel vol >= 10x (current default: 15)
  rel_vol_5x:    pts when rel vol >=  5x (current default: 7)
Daily Gain tiers:
  daily_sweet_20_40: pts for gain 20–40% sweet spot (default: 10)
  daily_ok_10_20:    pts for gain 10–20%            (default: 5)
  daily_ok_40_100:   pts for gain 40–100%           (default: 5)
Consolidation / trend:
  sideways_chop:   pts for 10-day range < 20% tight compression (default: 8)
  yesterday_green: pts when prior day closed positive            (default: 7)
Float / share count tiers:
  shares_lt10m:  pts when shares < 10M  (default: 30)
  shares_lt30m:  pts when shares 10–30M (default: 18)
  shares_lt100m: pts when shares 30–100M (default: 8)
Nice-to-have bonuses:
  no_news_bonus:        pts when no news catalyst (organic move, default: 5)
  high_cash_bonus:      pts when cash/share > price (default: 5)
  institution_moderate: pts when institutional ownership >= 20% (default: 3)
  institution_strong:   pts when institutional ownership >= 50% (default: 5)

## Evidence
{backtest_section}

{feedback_section}

## Task
The goal is to predict which setups will spike 20%+ the next day (take-profit target).

IMPORTANT: If a synthesized hypothesis is provided above, treat it as the PRIMARY driver:
- Increase weights for signals the hypothesis identifies as strong 20%+ predictors
- Decrease weights for signals the hypothesis identifies as weak or unreliable
- Suggestions should include new metrics explicitly mentioned in the hypothesis

Constraints:
- All weights must be integers 0–50
- Tier ordering must be preserved: rel_vol_500x >= rel_vol_100x >= rel_vol_50x >= rel_vol_25x >= rel_vol_10x >= rel_vol_5x
- Tier ordering must be preserved: shares_lt10m >= shares_lt30m >= shares_lt100m
- Upweight signals where hit-20%+ rate is highest AND avg-days-to-target is lowest

Also suggest 3–5 new scoring criteria/metrics to add (drawn from hypothesis patterns or backtest gaps).

Respond in EXACTLY this format with no other text:

WEIGHTS_JSON:
{{"rel_vol_500x": 0, "rel_vol_100x": 0, "rel_vol_50x": 0, "rel_vol_25x": 0, "rel_vol_10x": 0, "rel_vol_5x": 0, "daily_sweet_20_40": 0, "daily_ok_10_20": 0, "daily_ok_40_100": 0, "sideways_chop": 0, "yesterday_green": 0, "shares_lt10m": 0, "shares_lt30m": 0, "shares_lt100m": 0, "no_news_bonus": 0, "high_cash_bonus": 0, "institution_moderate": 0, "institution_strong": 0}}

RATIONALE:
[2-3 sentences explaining key weight changes — explicitly reference which hypothesis patterns drove them]

SUGGESTIONS:
- [new scoring signal/metric idea 1]
- [new scoring signal/metric idea 2]
- [new scoring signal/metric idea 3]

SUMMARY:
[one sentence, max 20 words, for dashboard display]"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        text = _msg_text(message)

        # Parse WEIGHTS_JSON block
        if "WEIGHTS_JSON:" not in text:
            raise ValueError("No WEIGHTS_JSON section in response")
        json_part = text.split("WEIGHTS_JSON:")[1].split("RATIONALE:")[0].strip()
        new_weights = json.loads(json_part)

        # Validate and clamp all required keys to int 0-50
        for key in REQUIRED_KEYS:
            new_weights[key] = max(0, min(50, int(new_weights.get(key, current_weights.get(key, 5)))))

        # Enforce tier ordering constraints
        new_weights["rel_vol_100x"] = min(new_weights.get("rel_vol_100x", 35), new_weights.get("rel_vol_500x", 40))
        new_weights["rel_vol_50x"]  = min(new_weights["rel_vol_50x"],  new_weights.get("rel_vol_100x", 35))
        new_weights["rel_vol_25x"]  = min(new_weights["rel_vol_25x"],  new_weights["rel_vol_50x"])
        new_weights["rel_vol_10x"]  = min(new_weights["rel_vol_10x"],  new_weights["rel_vol_25x"])
        new_weights["rel_vol_5x"]   = min(new_weights["rel_vol_5x"],   new_weights["rel_vol_10x"])
        new_weights["shares_lt30m"] = min(new_weights["shares_lt30m"], new_weights["shares_lt10m"])
        new_weights["shares_lt100m"]= min(new_weights["shares_lt100m"],new_weights["shares_lt30m"])

        # Parse RATIONALE
        rationale = ""
        if "RATIONALE:" in text:
            rationale = text.split("RATIONALE:")[1].split("SUGGESTIONS:")[0].strip()

        # Parse SUGGESTIONS
        suggestions = []
        if "SUGGESTIONS:" in text:
            sug_block = text.split("SUGGESTIONS:")[1].split("SUMMARY:")[0].strip()
            for line in sug_block.split("\n"):
                line = line.strip().lstrip("-*. ").strip()
                if line:
                    suggestions.append(line)

        # Parse SUMMARY
        summary = ""
        if "SUMMARY:" in text:
            summary = text.split("SUMMARY:")[1].strip()

        return {
            "weights":     new_weights,
            "rationale":   rationale,
            "suggestions": suggestions[:5],
            "summary":     summary,
        }

    except Exception as e:
        return {
            "weights":     current_weights,
            "rationale":   "",
            "suggestions": [],
            "summary":     "",
            "error":       str(e),
        }


def autonomous_optimize(
    opt_data: dict,
    all_feedback: list,
    current_weights: dict,
    prior_hypothesis: str = None,
    closed_trades: list = None,
    hypothesis_rules: list = None,
) -> dict:
    """
    Single Claude Sonnet call that combines hypothesis synthesis + weight optimization
    for the Auto AI autonomous self-improving model.

    Returns structured JSON with per-hypothesis confidence scores and a weight
    confidence score. Callers use these to decide what to auto-activate vs. send
    to pending for admin review.

    Returns:
        {
            "hypotheses": [{"text": str, "source": str, "confidence": int}],
            "weights": dict,           # all 16 required keys as ints
            "weight_confidence": int,  # 0-100
            "rationale": str,
            "summary": str,
            "suggestions": list,
        }
    On error adds "error" key and falls back to current_weights.
    """
    import json

    REQUIRED_KEYS = [
        "rel_vol_500x", "rel_vol_100x",
        "rel_vol_50x", "rel_vol_25x", "rel_vol_10x", "rel_vol_5x",
        "daily_sweet_20_40", "daily_ok_10_20", "daily_ok_40_100",
        "sideways_chop", "yesterday_green",
        "shares_lt10m", "shares_lt30m", "shares_lt100m",
        "no_news_bonus", "high_cash_bonus",
        "institution_moderate", "institution_strong",
        "sector_biotech_bonus",
    ]
    OPTIONAL_KEYS = [
        "rsi_momentum_bonus", "macd_positive_bonus", "bb_upper_breakout",
        "consecutive_green_bonus", "low_float_ratio_bonus",
    ]
    ALL_KEYS = REQUIRED_KEYS + OPTIONAL_KEYS

    # ---- Format backtest section (reuse same logic as optimize_complex_ai_weights) ----
    if opt_data and opt_data.get("total_trades", 0) >= 5:
        def fmt(stats):
            if not stats or stats["count"] == 0:
                return "No data"
            hit = stats.get("hit_20pct", 0)
            days = stats.get("avg_days_to_20pct")
            days_str = f" (avg {days}d to target)" if days is not None else ""
            return (f"{stats['count']} trades | "
                    f"{hit}% hit 20%+{days_str} | "
                    f"{stats['win_rate']}% any-gain | "
                    f"{stats['avg_return']:+.1f}% avg return")
        rv = opt_data["relative_volume"]
        dg = opt_data["daily_gain"]
        so = opt_data["shares_outstanding"]
        dw = opt_data.get("day_of_week", {})
        backtest_section = f"""BACKTESTED SIGNAL PERFORMANCE ({opt_data['total_trades']} trades with known outcomes):
Relative Volume:
  500x+:    {fmt(rv.get("500x+"))}
  100-499x: {fmt(rv.get("100-499x"))}
  50-99x:   {fmt(rv.get("50-99x"))}
  25-49x:   {fmt(rv.get("25-49x"))}
  10-24x:   {fmt(rv.get("10-24x"))}
Daily Gain at scan time:
  >80%:   {fmt(dg.get(">80%"))}
  50-80%: {fmt(dg.get("50-80%"))}
  30-50%: {fmt(dg.get("30-50%"))}
  20-30%: {fmt(dg.get("20-30%"))}
  10-20%: {fmt(dg.get("10-20%"))}
  <10%:   {fmt(dg.get("<10%"))}
Float (shares outstanding):
  <10M:    {fmt(so.get("<10M"))}
  10-30M:  {fmt(so.get("10-30M"))}
  30-100M: {fmt(so.get("30-100M"))}
  100M+:   {fmt(so.get("100M+"))}
Day of Week:
  Monday: {fmt(dw.get("Monday"))} | Tuesday: {fmt(dw.get("Tuesday"))} | Wednesday: {fmt(dw.get("Wednesday"))} | Thursday: {fmt(dw.get("Thursday"))} | Friday: {fmt(dw.get("Friday"))}"""
    else:
        backtest_section = "BACKTESTED SIGNAL PERFORMANCE: Insufficient data (<5 trades with known outcomes)."

    # ---- Format per-signal stats section ----
    per_sig = (opt_data or {}).get("per_signal_stats") if opt_data else None
    if (per_sig and per_sig.get("baseline") and
            per_sig["baseline"]["count"] >= 5 and per_sig.get("signals")):
        bl = per_sig["baseline"]
        lines = [
            f"PER-SIGNAL BACKTEST ({bl['count']} scans · baseline: "
            f"{bl['hit_20pct']}% hit 20%+, {bl['win_rate']}% any-gain):"
        ]
        qualified = [s for s in per_sig["signals"] if s["count"] >= 5]
        for s in qualified[:10]:
            arrow = "+" if s["vs_baseline_hit"] >= 0 else ""
            lines.append(
                f"  {s['key']}: {s['count']} fires | "
                f"{s['hit_20pct']}% hit20+ ({arrow}{s['vs_baseline_hit']}pp vs baseline) | "
                f"{s['win_rate']}% win | {s['avg_return']:+.1f}% avg"
            )
        bottom = [s for s in reversed(qualified) if s["count"] >= 5][:5]
        if bottom:
            lines.append("  [LOWEST PERFORMING:]")
            for s in bottom:
                arrow = "+" if s["vs_baseline_hit"] >= 0 else ""
                lines.append(
                    f"  {s['key']}: {s['count']} fires | "
                    f"{s['hit_20pct']}% hit20+ ({arrow}{s['vs_baseline_hit']}pp vs baseline)"
                )
        per_signal_section = "\n".join(lines)
    else:
        per_signal_section = (
            "PER-SIGNAL BACKTEST: Insufficient data "
            "(need 5+ scans with signals_json AND known outcomes)."
        )

    # ---- Format trade signal autopsy section ----
    try:
        from app.database import get_trade_signal_autopsy
        autopsy = get_trade_signal_autopsy()
    except Exception:
        autopsy = None

    if autopsy and autopsy.get("total_trades", 0) >= 5:
        at = autopsy
        alines = [
            f"TRADE SIGNAL AUTOPSY ({at['total_trades']} AI TRADE calls · "
            f"{at['wins']}W / {at['losses']}L · overall win rate: {at['overall_win_rate']}%):"
        ]
        # Individual signals — show worst (false positive generators) and best
        indiv = [s for s in at.get("individual", []) if s["fires"] >= 3]
        if indiv:
            worst = indiv[:5]  # already sorted worst-first
            best  = list(reversed(indiv))[:5]
            alines.append("  [INDIVIDUAL — FALSE POSITIVE SIGNALS (worst win rate on TRADE calls):]")
            for s in worst:
                arrow = "+" if s["vs_overall"] >= 0 else ""
                alines.append(
                    f"    {s['key']}: {s['fires']} fires | "
                    f"{s['wins']}W/{s['losses']}L | {s['win_rate']}% win rate "
                    f"({arrow}{s['vs_overall']}pp vs {at['overall_win_rate']}% overall)"
                )
            alines.append("  [INDIVIDUAL — STRONGEST SIGNALS (best win rate on TRADE calls):]")
            for s in best:
                arrow = "+" if s["vs_overall"] >= 0 else ""
                alines.append(
                    f"    {s['key']}: {s['fires']} fires | "
                    f"{s['wins']}W/{s['losses']}L | {s['win_rate']}% win rate "
                    f"({arrow}{s['vs_overall']}pp vs {at['overall_win_rate']}% overall)"
                )
        # Signal combos — worst combinations on losses
        combos = [c for c in at.get("combos", []) if c["total"] >= 3]
        if combos:
            alines.append("  [SIGNAL COMBOS — WORST CO-FIRING PAIRS ON TRADE CALLS (lowest win rate):]")
            for c in combos[:10]:
                alines.append(
                    f"    {' + '.join(c['signals'])}: "
                    f"{c['total']} fires | {c['win_count']}W/{c['loss_count']}L | "
                    f"{c['win_rate']}% win rate"
                )
            alines.append("  [SIGNAL COMBOS — STRONGEST CO-FIRING PAIRS ON TRADE CALLS (highest win rate):]")
            best_combos = sorted(combos, key=lambda x: -x["win_rate"])[:5]
            for c in best_combos:
                alines.append(
                    f"    {' + '.join(c['signals'])}: "
                    f"{c['total']} fires | {c['win_count']}W/{c['loss_count']}L | "
                    f"{c['win_rate']}% win rate"
                )
        autopsy_section = "\n".join(alines)
    else:
        autopsy_section = (
            "TRADE SIGNAL AUTOPSY: Insufficient data "
            "(need 5+ AI TRADE calls with known outcomes)."
        )

    # ---- Format hypotheses-to-test section ----
    # Includes user observations + any pending rules waiting for AI validation
    to_test = []
    if hypothesis_rules:
        for r in hypothesis_rules:
            if r.get("status") in ("pending", "active") and r.get("source") in (
                "user_observation", "user", "feedback", "historical"
            ):
                to_test.append(r)
    if to_test:
        lines = [
            "HYPOTHESES TO TEST (validate each with signal/backtest evidence — be explicit):"
        ]
        for i, r in enumerate(to_test[:8], 1):
            status = r.get("status", "pending").upper()
            lines.append(f"  H{i} [{status}] {r['rule_text'][:300]}")
        lines.append(
            "For each hypothesis: cite the relevant signal key (e.g. first_hour_vol_20m), "
            "its hit_20pct and vs_baseline from PER-SIGNAL BACKTEST above, "
            "and state VALIDATED / REJECTED / INSUFFICIENT_DATA with reasoning."
        )
        hypotheses_to_test_section = "\n".join(lines)
    else:
        hypotheses_to_test_section = (
            "HYPOTHESES TO TEST: None pending — generate new ones from the evidence above."
        )

    # ---- Format prior hypothesis section ----
    if prior_hypothesis:
        hyp_section = f"PRIOR HYPOTHESIS / LEARNED PATTERNS (evolve, do not discard):\n{prior_hypothesis[:1200]}"
    else:
        hyp_section = "PRIOR HYPOTHESIS: None yet — derive patterns fresh from evidence."

    # ---- Format feedback section ----
    if all_feedback:
        lines = ""
        for fb in all_feedback[-8:]:
            sym = fb.get("symbol") or "?"
            txt = (fb.get("user_text") or "")[:160]
            tag = fb.get("outcome_tag") or ""
            lines += f"\n- {sym} [{tag}]: {txt}"
        feedback_section = f"MANUAL FEEDBACK (last {min(8, len(all_feedback))} submissions):{lines}"
    else:
        feedback_section = "MANUAL FEEDBACK: None submitted yet."

    # ---- Format closed trades section ----
    if closed_trades:
        trade_lines = ""
        for t in closed_trades[-20:]:
            sym = t.get("symbol", "?")
            entry = t.get("entry_price", 0)
            exit_ = t.get("exit_price") or 0
            pnl   = t.get("pnl_pct") or t.get("realized_pnl") or 0
            tag   = t.get("outcome_tag") or ("win" if pnl > 0 else "loss")
            trade_lines += f"\n- {sym}: entry ${entry:.2f} → exit ${exit_:.2f} ({pnl:+.1f}%) [{tag}]"
        trades_section = f"CLOSED PAPER TRADES (last {min(20, len(closed_trades))} trades):{trade_lines}"
    else:
        trades_section = "CLOSED PAPER TRADES: None yet."

    current_display = {k: current_weights.get(k, 0) for k in ALL_KEYS}

    prompt = f"""You are autonomously evolving a self-improving stock trading model for low-float microcap momentum setups.

OPTIMIZATION GOALS (in priority order):
1. Hit rate: maximize % of TRADE calls where stock touches +20% above alert price within 10 trading days — TARGET 80%+
2. Win rate: maximize % of scans with any positive next-day return
3. Speed: minimize avg days to hit the +20% target (faster = better risk/reward)
4. Magnitude: maximize avg return (feeds future take-profit ceiling decisions)

Only raise weight_confidence >= 75 when the evidence clearly supports a change on ALL four metrics — or at minimum improves hit rate without harming the others.

You have THREE tasks:

TASK 1 — HYPOTHESIS TESTING: Validate or reject each hypothesis listed under "HYPOTHESES TO TEST" using the per-signal backtest data. Generate new hypotheses from patterns you observe in the data that are NOT already listed.
TASK 2 — HYPOTHESIS EVOLUTION: Combine tested + new hypotheses into updated pattern rules. Assign a confidence score (0-100) to each.
TASK 3 — CRITERIA & WEIGHT OPTIMIZATION: Decide which scoring criteria to keep, remove, or add based on validated evidence. Set weight=0 to disable, 1-50 to enable.

## Current Auto AI Weights (0 = disabled, 1–50 = enabled; scorer normalises to 0–100 automatically)
{json.dumps(current_display, indent=2)}

## Scoring Criteria Control
EXISTING CRITERIA — set to 0 to remove if evidence shows it reduces win rate or % return:
  rel_vol_500x, rel_vol_100x, rel_vol_50x, rel_vol_25x, rel_vol_10x, rel_vol_5x  [relative volume tiers, keep descending order]
  daily_sweet_20_40, daily_ok_10_20, daily_ok_40_100  [daily gain tiers]
  shares_lt10m, shares_lt30m, shares_lt100m            [share count tiers, keep descending order]
  sideways_chop          avg abs daily return last 5 days <10%: quiet chop-to-ignition setup (57% of winners)
  momentum_continuation  avg abs daily return last 5 days >=10%: already running, catching the next leg (43% of winners)
  [NOTE: sideways_chop and momentum_continuation are mutually exclusive — a stock fires exactly one or neither]
  yesterday_green  Prior day closed positive
  no_news_bonus    No recent news: organic move, not headline-chasing
  high_cash_bonus  Cash per share > stock price: balance sheet safety
  institution_moderate  15-39% institutional ownership
  institution_strong    40%+ institutional ownership: holds the floor
  sector_biotech_bonus  Healthcare / Biotech sector

OPTIONAL CRITERIA — disabled by default (weight=0), set 1-20 to enable if evidence supports:
  rsi_momentum_bonus      RSI 50-70: healthy uptrend, not yet overbought
  macd_positive_bonus     Positive MACD: bullish momentum crossover confirmed
  bb_upper_breakout       Bollinger %B > 0.85: price breaking above upper Bollinger Band
  consecutive_green_bonus 2+ consecutive green days: sustained buying pressure
  low_float_ratio_bonus   Float < 40% of shares outstanding: tight float amplifies moves

## Hypotheses to Validate
{hypotheses_to_test_section}

## Evidence
{backtest_section}

{per_signal_section}

{autopsy_section}

{hyp_section}

{feedback_section}

{trades_section}

## Confidence Guidelines
- Hypothesis confidence >= 80: will be AUTO-ACTIVATED immediately (no admin review)
- Hypothesis confidence 50-79: goes to pending for admin review
- Hypothesis confidence < 50: omit entirely
- Weight confidence >= 75: weights will be AUTO-APPLIED (if >= 10 closed trades exist)
- Weight confidence < 75: weights will be logged but NOT applied

## Constraints
- All weight values must be integers 0-50 (optional criteria: max 20)
- Tier ordering: rel_vol_500x >= rel_vol_100x >= rel_vol_50x >= rel_vol_25x >= rel_vol_10x >= rel_vol_5x
- Tier ordering: shares_lt10m >= shares_lt30m >= shares_lt100m
- Include ALL 24 keys in the output weights dict (set unwanted ones to 0)

## Data Integrity Rules — STRICT
- DO NOT hallucinate signal performance. Every weight change and hypothesis MUST be grounded in a specific number from the Evidence section above (hit rate, sample size, pp vs baseline).
- DO NOT invent hit rates, sample sizes, or patterns not present in the data provided.
- If a signal has insufficient data (n < 10), state INSUFFICIENT_DATA — do not speculate about its direction.
- Confidence rubric: Low = n<10, Medium = n 10–30, High = n 30+. Only raise weight_confidence >= 75 for High confidence signals with >= 5pp effect size.
- Every rationale sentence must cite: signal name + hit rate % + sample size + pp vs baseline. No vague claims.
- Double-check tier ordering constraints before outputting weights. Verify all 24 keys are present.

## Required Output Format
Respond with ONLY a valid JSON object (no markdown, no code fences, no other text):

{{"hypotheses": [{{"text": "specific pattern rule", "source": "historical|feedback|both", "confidence": 85}}], "weights": {{"rel_vol_500x": 40, "rel_vol_100x": 35, "rel_vol_50x": 30, "rel_vol_25x": 22, "rel_vol_10x": 15, "rel_vol_5x": 7, "daily_sweet_20_40": 10, "daily_ok_10_20": 5, "daily_ok_40_100": 5, "sideways_chop": 8, "yesterday_green": 7, "shares_lt10m": 30, "shares_lt30m": 18, "shares_lt100m": 8, "no_news_bonus": 5, "high_cash_bonus": 5, "institution_moderate": 2, "institution_strong": 5, "sector_biotech_bonus": 5, "rsi_momentum_bonus": 0, "macd_positive_bonus": 0, "bb_upper_breakout": 0, "consecutive_green_bonus": 0, "low_float_ratio_bonus": 0}}, "weight_confidence": 82, "rationale": "2-3 sentences citing specific evidence that drove changes", "summary": "one sentence max 20 words", "suggestions": ["new signal idea 1", "new signal idea 2"]}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = _msg_text(message)

        # Try to parse as JSON directly; fallback: extract from code fence
        result = None
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            import re
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                result = json.loads(m.group(1))
            else:
                # Last resort: find first { ... } block
                m2 = re.search(r"\{.*\}", text, re.DOTALL)
                if m2:
                    result = json.loads(m2.group(0))
                else:
                    raise ValueError(f"Could not parse JSON from response: {text[:300]}")

        # Validate and clamp all weight keys
        weights = result.get("weights", {})
        for key in REQUIRED_KEYS:
            weights[key] = max(0, min(50, int(weights.get(key, current_weights.get(key, 5)))))
        for key in OPTIONAL_KEYS:
            weights[key] = max(0, min(20, int(weights.get(key, 0))))

        # Enforce tier ordering constraints
        weights["rel_vol_100x"] = min(weights.get("rel_vol_100x", 35), weights.get("rel_vol_500x", 40))
        weights["rel_vol_50x"]  = min(weights["rel_vol_50x"],  weights.get("rel_vol_100x", 35))
        weights["rel_vol_25x"]  = min(weights["rel_vol_25x"],  weights["rel_vol_50x"])
        weights["rel_vol_10x"]  = min(weights["rel_vol_10x"],  weights["rel_vol_25x"])
        weights["rel_vol_5x"]   = min(weights["rel_vol_5x"],   weights["rel_vol_10x"])
        weights["shares_lt30m"] = min(weights["shares_lt30m"], weights["shares_lt10m"])
        weights["shares_lt100m"]= min(weights["shares_lt100m"],weights["shares_lt30m"])
        result["weights"] = weights

        # Ensure weight_confidence is an int 0-100
        result["weight_confidence"] = max(0, min(100, int(result.get("weight_confidence", 0))))

        # Ensure hypotheses is a list of dicts with required keys
        hyps = result.get("hypotheses", [])
        clean_hyps = []
        for h in hyps:
            if isinstance(h, dict) and h.get("text") and isinstance(h.get("confidence"), (int, float)):
                clean_hyps.append({
                    "text":       str(h["text"])[:400],
                    "source":     str(h.get("source", ""))[:20],
                    "confidence": max(0, min(100, int(h["confidence"]))),
                })
        result["hypotheses"] = clean_hyps

        return result

    except Exception as e:
        return {
            "hypotheses":       [],
            "weights":          current_weights,
            "weight_confidence": 0,
            "rationale":        "",
            "summary":          "",
            "suggestions":      [],
            "error":            str(e),
        }


def chat_with_model(message: str, history: list, context: dict) -> dict:
    """
    Multi-turn chat with the AI model advisor.

    history: list of {"role": "user"|"assistant", "content": str}
    context: dict with hypothesis_rules, squeeze_weights, autoai_weights, opt_data, feedback, closed_trades

    Returns {"reply": str, "actions": list[dict]}
    Each action dict has at minimum: "action" key + whatever fields that action needs.
    """
    import json as _json

    # ---- Format context sections ----
    rules = context.get("hypothesis_rules", [])
    if rules:
        rule_lines = []
        for r in rules:
            status_tag = f"[{r.get('status','?').upper()}]"
            ai_tag = " [AUTO-AI]" if r.get("auto_applied") else ""
            conf = f" conf={r['confidence_score']}%" if r.get("confidence_score") else ""
            rule_lines.append(f"  #{r['id']} {status_tag}{ai_tag}{conf}: {r.get('rule_text','')}")
        rules_section = "HYPOTHESIS RULES:\n" + "\n".join(rule_lines)
    else:
        rules_section = "HYPOTHESIS RULES: None yet."

    sq_w = context.get("squeeze_weights")
    ai_w = context.get("autoai_weights")
    weights_section = ""
    if sq_w:
        weights_section += f"COMPLEX+AI WEIGHTS (last updated {sq_w.get('updated_at','?')[:10]}):\n{_json.dumps(sq_w.get('weights', {}))}\n"
    if ai_w:
        weights_section += f"AUTO AI WEIGHTS (last updated {ai_w.get('updated_at','?')[:10]}):\n{_json.dumps(ai_w.get('weights', {}))}\n"
    if not weights_section:
        weights_section = "WEIGHTS: Using defaults (no custom weights saved yet)."

    opt = context.get("opt_data") or {}
    if opt.get("total_trades", 0) >= 5:
        def _fmt(stats):
            if not stats or stats["count"] == 0:
                return "No data"
            hit  = stats.get("hit_20pct", 0)
            days = stats.get("avg_days_to_20pct")
            days_str = f" (avg {days}d to target)" if days is not None else ""
            return (f"{stats['count']} trades | "
                    f"{hit}% hit 20%+{days_str} | "
                    f"{stats['win_rate']}% any-gain | "
                    f"{stats['avg_return']:+.1f}% avg return")
        rv = opt.get("relative_volume", {})
        dg = opt.get("daily_gain", {})
        so = opt.get("shares_outstanding", {})
        perf_section = f"""BACKTESTED SIGNAL PERFORMANCE ({opt['total_trades']} trades with known outcomes):
Relative Volume: >=50x: {_fmt(rv.get(">=50x"))} | 25-50x: {_fmt(rv.get("25-50x"))} | 10-25x: {_fmt(rv.get("10-25x"))} | <10x: {_fmt(rv.get("<10x"))}
Daily Gain: 20-40%: {_fmt(dg.get("20-40%"))} | 10-20%: {_fmt(dg.get("10-20%"))} | 40-100%: {_fmt(dg.get("40-100%"))} | <10%: {_fmt(dg.get("<10%"))}
Shares: <1M: {_fmt(so.get("<1M"))} | 1-5M: {_fmt(so.get("1-5M"))} | 5-10M: {_fmt(so.get("5-10M"))} | 10-30M: {_fmt(so.get("10-30M"))} | 30M+: {_fmt(so.get("30M+"))}"""

        per_sig = opt.get("per_signal_stats")
        if (per_sig and per_sig.get("baseline") and
                per_sig["baseline"]["count"] >= 5 and per_sig.get("signals")):
            bl = per_sig["baseline"]
            sig_lines = [
                f"PER-SIGNAL BACKTEST ({bl['count']} scans · baseline: "
                f"{bl['hit_20pct']}% hit 20%+, {bl['win_rate']}% any-gain):"
            ]
            qualified = [s for s in per_sig["signals"] if s["count"] >= 5]
            for s in qualified[:10]:
                arrow = "+" if s["vs_baseline_hit"] >= 0 else ""
                sig_lines.append(
                    f"  {s['key']}: {s['count']} fires | "
                    f"{s['hit_20pct']}% hit20+ ({arrow}{s['vs_baseline_hit']}pp vs baseline) | "
                    f"{s['win_rate']}% win | {s['avg_return']:+.1f}% avg"
                )
            perf_section += "\n" + "\n".join(sig_lines)
    else:
        perf_section = "BACKTEST: Insufficient data (<5 trades with known outcomes)."

    fb = context.get("feedback", [])
    if fb:
        fb_lines = [f"  - {f.get('symbol','?')} [{f.get('outcome_tag','')}]: {(f.get('user_text') or '')[:100]}" for f in fb]
        feedback_section = "RECENT FEEDBACK:\n" + "\n".join(fb_lines)
    else:
        feedback_section = "RECENT FEEDBACK: None."

    trades = context.get("closed_trades", [])
    if trades:
        t_lines = [f"  - {t.get('symbol','?')}: {t.get('pnl_pct') or t.get('realized_pnl') or 0:+.1f}%" for t in trades]
        trades_section = "RECENT CLOSED TRADES:\n" + "\n".join(t_lines)
    else:
        trades_section = "RECENT CLOSED TRADES: None yet."

    system_prompt = f"""You are an AI trading model advisor for a microcap momentum scanner targeting 20%+ intraday spikes.
Your role: help the admin explore hypotheses, understand what's working, and decide how to update the model.
Be concise, direct, and evidence-based.

When you want to suggest an executable action, put it on its own line as valid JSON with an "action" key:
{{"action": "activate_rule", "rule_id": 12, "label": "Activate: 'RSI 50-70 outperforms'"}}
{{"action": "reject_rule", "rule_id": 8, "label": "Reject: 'no_news_bonus shows no edge'"}}
{{"action": "add_rule", "text": "Specific pattern rule text here", "source": "discussion", "label": "Add new rule: brief description"}}
{{"action": "update_weights", "model": "complex", "weights": {{}}, "rationale": "...", "summary": "one sentence", "label": "Update Complex+AI weights"}}
{{"action": "update_weights", "model": "autoai", "weights": {{}}, "rationale": "...", "summary": "one sentence", "label": "Update Auto AI weights"}}

When multiple weight changes all serve the same optimization goal, bundle them into a SINGLE update_weights_bundle action rather than listing them separately:
{{"action": "update_weights_bundle", "goal": "win_rate", "model": "complex", "weights": {{"rel_vol_50x": 30, "rsi_momentum_bonus": 20}}, "rationale": "Both signals show strong positive next-day returns — boosting together improves the odds a trade closes green.", "summary": "Win Rate bundle: rel_vol_50x + rsi_momentum_bonus", "label": "Bundle (Win Rate): rel_vol_50x \u2191 + rsi_momentum_bonus \u2191"}}
{{"action": "update_weights_bundle", "goal": "speed", "model": "complex", "weights": {{"rel_vol_50x": 30, "daily_sweet_20_40": 25}}, "rationale": "These signals correlate with the fastest time-to-20% target based on avg_days_to_20pct.", "summary": "Speed bundle: highest-velocity signal combo", "label": "Bundle (Speed-to-Target): rel_vol_50x + daily_sweet_20_40"}}
{{"action": "update_weights_bundle", "goal": "upside", "model": "complex", "weights": {{"shares_lt10m": 35, "no_news_bonus": 15}}, "rationale": "Low float + no-news setups show the highest hit_20pct, suggesting take-profit can be set above 20%.", "summary": "Upside bundle: maximize hit rate above 20%", "label": "Bundle (Max Upside): shares_lt10m + no_news_bonus"}}

Optimization goals — in strict priority order:
1. win_rate (PRIMARY): maximize the probability a trade ends positive (next-day or multi-day win rate)
2. speed: minimize avg_days_to_20pct — fastest time to hit the +20% target
3. upside: maximize hit_20pct — setups most likely to reach or exceed +20% (enables higher take-profit)

Bundling strategy — win_rate is the hard constraint, speed and upside are bonuses:
The single rule: no change may materially hurt win_rate. Within that constraint, maximize gains across all three goals.

Evaluate each candidate signal change against all three metrics before including it:
- INCLUDE if win_rate is neutral or improves — regardless of what it does to speed/upside.
- ACTIVELY SEEK signals that also improve speed or upside — these are bonuses, not requirements.
- EXCLUDE only if the change would materially hurt win_rate (signal's win_rate notably below baseline).

Bundling rules:
- goal="combined": bundle improves win_rate AND also improves speed and/or upside.
- goal="win_rate": win_rate improves but speed and upside are unaffected or mixed.
- goal="speed": speed improves while win_rate is intact (neutral or better).
- goal="upside": upside improves while win_rate is intact (neutral or better).
- Never produce a bundle that degrades win_rate, even if speed or upside would benefit greatly.

Always include your constraint reasoning in the rationale: note each signal's effect on all three metrics and whether it passed the win_rate safety check.

Bundle examples:
{{"action": "update_weights_bundle", "goal": "combined", "model": "complex", "weights": {{"rel_vol_50x": 35, "shares_lt10m": 30}}, "rationale": "rel_vol_50x: strong win_rate (+18pp vs baseline), speed neutral (avg 2.1 days vs 2.3 baseline), upside positive (+12pp hit_20pct). shares_lt10m: win_rate +14pp, speed neutral, upside +9pp. Both pass constraint check.", "summary": "Win rate up, speed/upside preserved", "label": "Bundle (Win Rate + Speed + Upside): rel_vol_50x \u2191 + shares_lt10m \u2191"}}
{{"action": "update_weights_bundle", "goal": "win_rate", "model": "complex", "weights": {{"yesterday_green": 20}}, "rationale": "yesterday_green: win_rate +11pp but avg_days_to_20pct jumps to 4.8 vs 2.3 baseline — fails speed constraint. Isolated to win_rate bundle only.", "summary": "Win rate improvement, speed trade-off noted", "label": "Bundle (Win Rate): yesterday_green \u2191"}}

## Current System State
{rules_section}

{weights_section}
{perf_section}

{feedback_section}

{trades_section}"""

    messages = []
    for h in history:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=system_prompt,
            messages=messages,
        )
        raw = _msg_text(response)

        # Parse out action JSON lines from the reply
        reply_lines = []
        actions = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    obj = _json.loads(stripped)
                    if "action" in obj:
                        actions.append(obj)
                        continue
                except Exception:
                    pass
            reply_lines.append(line)

        reply = "\n".join(reply_lines).strip()
        return {"reply": reply, "actions": actions}

    except Exception as e:
        return {"reply": f"Sorry, I encountered an error: {e}", "actions": []}


def _fetch_historical_factors(symbol: str, move_date: date) -> str | None:
    """
    Fetches actual yfinance data for symbol around move_date and returns
    a formatted string of the scoring model's key numeric factors.
    Returns None if data is unavailable.
    """
    try:
        start = move_date - timedelta(days=100)
        end   = move_date + timedelta(days=5)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1d")

        if df.empty or len(df) < 5:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index).date
        available = df.index.tolist()

        closest = min(available, key=lambda d: abs((d - move_date).days))
        if abs((closest - move_date).days) > 5:
            return None

        idx   = available.index(closest)
        row   = df.iloc[idx]
        close = float(row["Close"])

        # Daily return
        daily_return = None
        if idx > 0:
            prev = float(df.iloc[idx - 1]["Close"])
            if prev > 0:
                daily_return = (close - prev) / prev * 100

        # Relative volume (63-day avg)
        relative_volume = None
        if idx >= 63:
            avg_vol = float(df["Volume"].iloc[idx - 63:idx].mean())
            if avg_vol > 0:
                relative_volume = float(row["Volume"]) / avg_vol

        # 10-day range compression
        range_10d = None
        if idx >= 9:
            high_10 = float(df["High"].iloc[idx - 9:idx + 1].max())
            low_10  = float(df["Low"].iloc[idx - 9:idx + 1].min())
            if close > 0:
                range_10d = (high_10 - low_10) / close * 100

        # Yesterday green
        yesterday_green = daily_return is not None and daily_return > 0
        if idx >= 2:
            prev2 = float(df.iloc[idx - 2]["Close"])
            prev1 = float(df.iloc[idx - 1]["Close"])
            yesterday_green = prev1 > prev2

        # Fundamentals
        info = ticker.info
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        institution_pct = (
            info.get("institutionsPercentHeld") or
            info.get("heldPercentInstitutions")
        )

        lines = [f"Actual data for {symbol} on {closest}:"]
        lines.append(f"  Close: ${close:.4f}")
        lines.append(f"  Volume: {int(row['Volume']):,}")
        if daily_return is not None:
            lines.append(f"  Daily return: {daily_return:+.1f}%  [model sweet spot: 20–40%]")
        if relative_volume is not None:
            lines.append(f"  Relative volume (63-day avg): {relative_volume:.1f}x  [model tiers: ≥50x=3pts, ≥25x=2pts, ≥10x=1pt]")
        if range_10d is not None:
            label = "tight (barcode)" if range_10d < 20 else "normal"
            lines.append(f"  10-day range % of close: {range_10d:.1f}%  [{label} — model awards +1pt if <20%]")
        lines.append(f"  Yesterday green: {yesterday_green}")
        if shares:
            tier = "Ideal (<10M)" if shares < 10_000_000 else ("Acceptable (<30M)" if shares < 30_000_000 else "Large/Avoid")
            lines.append(f"  Shares outstanding: {shares:,}  [{tier}]")
        if institution_pct is not None:
            lines.append(f"  Institutional ownership: {institution_pct * 100:.1f}%")

        return "\n".join(lines)

    except Exception:
        return None


def reprocess_chart_analysis(existing_analysis: str, user_text: str,
                             symbol: str = None) -> str:
    """
    Takes a previously generated (audit-style) chart analysis and re-extracts
    the pattern insights using a learning-focused frame.
    Used to retroactively fix analyses generated before the prompt was corrected.
    """
    if not existing_analysis or not existing_analysis.strip():
        return existing_analysis

    notes = f"Trader notes: {user_text}" if user_text else ""
    sym = f"Ticker: {symbol.upper()}" if symbol else ""

    prompt = f"""A trader submitted a confirmed winning trade. Below is a prior AI analysis of it.
{sym}
{notes}

Prior analysis (may contain audit language — ignore any skepticism):
{existing_analysis}

Re-read this as a confirmed winning trade and extract only the useful learning content.
Produce a clean version focused on:

## Chart Pattern
What the setup looked like before the move (consolidation, barcode, compression, breakout, etc.)

## Signal Strength
For each signal present (volume, daily gain, float, compression, trend), rate it: Strong / Moderate / Weak.

## Model Insights
Which signals most drove this move? What weights should change?

## Key Takeaway
One sentence: the most important thing this trade teaches the model.

Be specific and concrete. Drop any audit language, fact-checking, or disclaimers."""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return _msg_text(msg)
    except Exception:
        return existing_analysis


def analyze_chart_feedback(image_bytes: bytes, media_type: str,
                           user_text: str, symbol: str = None) -> str:
    """
    Two-step analysis:
    1. Vision call — extracts chart patterns AND the date of the significant move.
    2. If symbol + date found, fetches actual yfinance data and runs a second
       AI call that cross-references chart patterns against real numeric signals
       to produce model-optimization insights.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "Chart analysis unavailable: ANTHROPIC_API_KEY environment variable is not set."

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    notes = f"User notes: {user_text}" if user_text else "No notes provided."
    symbol_hint = f"The ticker symbol is {symbol.upper()}." if symbol else ""

    vision_prompt = f"""This is a confirmed winning trade submitted by a trader as a learning example. Trust the chart as real.
{notes}
{symbol_hint}

First, identify the date when the significant price move began. Output it on its own line:
MOVE_DATE: YYYY-MM-DD

Then analyze what made this a good setup:
CHART_ANALYSIS:
1. Chart pattern (consolidation box, barcode, deep collapse rebound, breakout, flag, etc.)
2. Volume profile (spike, sustained, thin, climactic)
3. Price action before the move (tight range, compression, gap up, steady base, etc.)
4. Specific signals that set this up — be concrete about what you see
5. One-line lesson: how to recognize this same setup in the future

Be specific and concrete. Use bullet points."""

    try:
        vision_msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        }
                    },
                    {"type": "text", "text": vision_prompt}
                ]
            }]
        )
        vision_text = _msg_text(vision_msg)

        # Parse the move date
        move_date = None
        for line in vision_text.splitlines():
            if line.strip().startswith("MOVE_DATE:"):
                raw = line.replace("MOVE_DATE:", "").strip()
                try:
                    move_date = datetime.strptime(raw, "%Y-%m-%d").date()
                except ValueError:
                    pass
                break

        # Step 2 — fetch real data and cross-reference if possible
        if symbol and move_date:
            historical = _fetch_historical_factors(symbol.upper(), move_date)
        else:
            historical = None

        if historical:
            enrich_prompt = f"""You are learning from a confirmed winning trade to improve a short-squeeze momentum scoring model.
The trader already made money on this trade. Accept it as a valid real example.

--- What the chart shows ---
{vision_text}

--- Numeric data from that date ---
{historical}

Use the numeric data to measure how strong each signal was on that specific day. Produce:

## Signal Strength on This Trade
For each model factor (relative volume, daily gain %, 10-day range, float size, yesterday green,
institutional ownership), state the actual level and rate it: Strong / Moderate / Weak / Absent.

## Model Weight Suggestions
Based on which signals were strongest in this confirmed winner, suggest specific weight increases
or decreases to help the model find more setups like this.

## Patterns Not Yet Tracked
List any chart or data patterns visible here that our model doesn't currently score
but appear predictive of the move.

## Key Takeaway
One sentence: the single most important thing this trade teaches the model.

Be specific with numbers. Use bullet points."""

            enrich_msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=700,
                messages=[{"role": "user", "content": enrich_prompt}]
            )
            return _msg_text(enrich_msg)

        # Return chart-only analysis if no historical data available
        return vision_text

    except Exception as e:
        return f"Could not analyze chart: {str(e)}"


def optimize_stop_loss(model_type: str, current_params: dict,
                       current_metrics: dict, proposed_params: dict,
                       validation: dict) -> dict:
    """
    Ask Claude to evaluate a proposed stop-loss parameter change and return
    a structured decision: approve / reject / suggest alternative.

    model_type: "daily" or "5m"
    current_params / proposed_params: parameter dicts from stop_loss_optimizer
    current_metrics / validation: backtest results from validate_proposed_params()

    Returns:
    {
        "decision":    "approve" | "reject" | "suggest",
        "confidence":  0–100,
        "rationale":   str,
        "params":      dict   (approved or suggested params — may differ from proposed),
    }
    """
    cm = current_metrics
    pm = validation.get("proposed_metrics", {})

    cur_params_str  = "\n".join(f"  {k}: {v}" for k, v in current_params.items())
    prop_params_str = "\n".join(f"  {k}: {v}" for k, v in proposed_params.items())

    prompt = f"""You are a quantitative risk manager reviewing a proposed stop-loss parameter change
for a momentum stock scanner trading system ({model_type} model).

## Current Parameters
{cur_params_str}

## Proposed Parameters
{prop_params_str}

## Backtest Results — Current Params
  Scans tested         : {cm.get('n_scans', 0)}
  Actual winners (hit 20%+): {cm.get('n_winners', 0)} ({cm.get('n_winners',0)/max(cm.get('n_scans',1),1)*100:.1f}%)
  Avg loss on losers   : {cm.get('avg_loss_pct', 0):.1f}%
  Exit reason breakdown: {cm.get('close_reason_counts', {})}

## Backtest Results — Proposed Params
  Winner preservation  : {validation.get('winner_preservation', 0):.1%}
    (% of actual winners that still hit 20% — must stay ≥ 95%)
  Premature stops      : {pm.get('premature_stops', '?')} winners killed early
  Avg loss on losers   : {pm.get('avg_loss_pct', 0):.1f}%
  Loss reduction       : {validation.get('loss_reduction', 0)*100:.1f}% improvement
  Exit reason breakdown: {pm.get('close_reason_counts', {})}

## Validation Status
  Passes winner constraint: {'YES' if validation.get('winner_preservation', 0) >= 0.95 else 'NO'}
  Passes loss constraint  : {'YES' if validation.get('loss_reduction', 0) >= 0.05 else 'NO'}
  {'Failure reason: ' + validation['failure_reason'] if validation.get('failure_reason') else ''}

## Your Mandate
Approve only if BOTH constraints are met AND the change is conservative.
Reject if either constraint fails with no good reason to override.
Suggest alternative params if the direction is right but magnitude is off.

Key principles:
- Never tighten stop_loss_pct below 10% for daily (too many false stops on volatile names)
- Never tighten stop_loss_pct below 8% for 5m (intraday noise is high on low-float stocks)
- A 5–15% loss reduction while preserving 97%+ of winners is ideal
- Time/staleness stops should not be shortened unless there is clear evidence of faster decay
- When in doubt, reject — the current params stay until evidence is strong

Respond with ONLY valid JSON (no markdown):
{{
  "decision": "approve" | "reject" | "suggest",
  "confidence": <integer 0-100>,
  "rationale": "<2-3 sentence explanation>",
  "params": {{ <final parameter dict — use proposed if approve, current if reject, or your alternative if suggest> }}
}}"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _msg_text(msg).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        # Ensure params key always present
        if "params" not in result:
            result["params"] = current_params
        return result
    except Exception as e:
        return {
            "decision":   "reject",
            "confidence": 0,
            "rationale":  f"Claude call failed: {e}",
            "params":     current_params,
        }


def generate_weekly_insights(data: dict) -> str:
    """
    Passes structured weekly analysis data to Claude and returns a formatted
    Key Findings + Ranked Recommendations section for the weekly report.
    Falls back to empty string on any error so the report still sends.
    """
    def _fmt_rows(rows, cols):
        return "\n".join("  " + " | ".join(str(v) for v in row) for row in rows) or "  (no data)"

    baseline_hit = data.get("baseline_hit", 0)

    sig_rows = data.get("sig_rows", [])
    top_sigs  = "\n".join(f"  {k}: {hit}% ({hit - baseline_hit:+.1f}pp, n={n})" for k, n, hit in sig_rows[:8])
    bot_sigs  = "\n".join(f"  {k}: {hit}% ({hit - baseline_hit:+.1f}pp, n={n})" for k, n, hit in sig_rows[-8:])

    score_section = "\n".join(
        f"  {b}: {hit}% (n={n})" for b, n, hit in data.get("score_rows", [])
    )
    rv_section = "\n".join(
        f"  {t}: {hit}% (n={n})" for t, n, hit in data.get("rv_rows", [])
    )
    lstm_section = "\n".join(
        f"  LSTM {gate}: {hit}% ({(hit or 0) - baseline_hit:+.1f}pp, n={n})"
        for gate, n, hit in data.get("lstm_gate_rows", [])
    )
    lstm_score_section = "\n".join(
        f"  score {b}: raw={hit_all}% (n={n_all}) | +LSTM>=55%: {hit_gated}% (n={n_gated})"
        for b, n_all, hit_all, n_gated, hit_gated in data.get("lstm_score_rows", [])
    )
    dow_section = "\n".join(
        f"  {d}: raw={hit_all}% (n={n_all}) | +LSTM>=55%: {hit_gated}% (n={n_gated})"
        for d, n_all, hit_all, n_gated, hit_gated in data.get("dow_lstm_rows", [])
        if d
    )
    speed_rows = data.get("speed_rows", [])
    total_hits = sum(r[1] for r in speed_rows) or 1
    cumulative = 0
    speed_lines = []
    for day, n in speed_rows:
        cumulative += n
        speed_lines.append(f"  Day {day}: {n} hits (cumulative {round(100*cumulative/total_hits)}%)")
    speed_section = "\n".join(speed_lines)

    prompt = f"""You are analyzing weekly performance data for an AI stock scanner that identifies squeeze plays — stocks likely to hit +20% within 10 trading days. The scanner uses a weighted scoring model (0–100 pts) and an LSTM probability gate (currently >=55%).

Baseline hit rate: {baseline_hit}% (n={data.get("baseline_n", 0)})
AI TRADE precision: {data.get("ai_pct", 0)}% ({data.get("ai_hits", 0)}/{data.get("ai_total", 0)} calls)

SCORE BUCKETS:
{score_section}

RELVOL TIERS:
{rv_section}

DAY OF WEEK (raw hit rate / with LSTM>=55% gate):
{dow_section or "  (no LSTM data yet)"}

LSTM GATE BUCKETS:
{lstm_section or "  (no data yet)"}

SCORE x LSTM>=55% CROSS-TAB:
{lstm_score_section or "  (no data yet)"}

HIT SPEED — days until 20%+ achieved:
{speed_section or "  (no data yet)"}

TOP SIGNALS (vs baseline {baseline_hit}%):
{top_sigs or "  (no data)"}

BOTTOM SIGNALS:
{bot_sigs or "  (no data)"}

Your task: identify the 3–5 most important findings from this data and produce ranked, actionable recommendations. Focus on:
1. Signals with negative vs-baseline performance (zero or reduce weight)
2. Score bucket anomalies (high buckets underperforming lower ones)
3. LSTM gate — where raising the threshold stops helping vs hurts volume
4. Day-of-week patterns (gaps >8pp between days)
5. Relvol cliffs (where hit rate drops sharply between adjacent tiers)
6. Hit speed — whether the take-profit window or time stop could be tightened
7. LSTM+score combos that outperform either signal alone

Confidence: High = n>=30, Med = n>=10, Low = n<10. Only recommend changes with Medium or High confidence.

Format EXACTLY as:

📊 KEY FINDINGS
1. [One-sentence insight — lead with the conclusion, then supporting numbers in parens]
2. [...]
...

🎯 RANKED RECOMMENDATIONS
1. [High/Med confidence] Change: [specific change] | Evidence: [numbers] | Expected lift: [estimate]
2. [...]
...

Keep total under 700 words. No preamble or closing remarks."""

    try:
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return _msg_text(msg)
    except Exception as e:
        return f"(Weekly insights unavailable: {e})"
