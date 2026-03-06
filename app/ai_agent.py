# ai_trading_mvp/app/ai_agent.py

import os
import re
import base64
import anthropic
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically


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
        text = message.content[0].text.strip()

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
        text = message.content[0].text.strip()
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
                    ai_accuracy: dict = None) -> dict:
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
        hypothesis_section = f"\nLearned patterns:\n{hypothesis[:400].strip()}"

    # Context 3: recent scan history for this ticker
    history_section = ""
    if ticker_history:
        lines = ["\nPrior scans for this ticker (most recent first):"]
        for h in ticker_history[:3]:
            nd = f" → next_day: {h['next_day']:+.1f}%" if h.get("next_day") else ""
            relvol = h.get("relvol") or 0
            lines.append(f"  {h['timestamp']}: score={h['score']}, relvol={relvol:.1f}x{nd}")
        history_section = "\n".join(lines)

    # Context 4: market context (day of week)
    day_name = datetime.now().strftime("%A")
    market_section = f"\nMarket context: Today is {day_name}."

    # Context 5: LSTM model prediction
    lstm_section = ""
    if lstm_prob is not None:
        lstm_section = (
            f"\nLSTM model (trained on historical setups): "
            f"{lstm_prob:.0%} probability this stock hits +20% intraday within 7 days."
        )

    # Context 6: recent news headlines
    news_section = ""
    if news_headlines:
        lines = ["\nRecent news headlines:"]
        for h in news_headlines[:5]:
            lines.append(f"  • {h}")
        news_section = "\n".join(lines)

    # Context 7: AI self-calibration — own historical accuracy
    accuracy_section = ""
    if ai_accuracy and ai_accuracy.get("total_resolved", 0) >= 5:
        parts = [f"\nYour past AI call accuracy ({ai_accuracy['total_resolved']} resolved calls):"]
        if "trade" in ai_accuracy:
            t = ai_accuracy["trade"]
            parts.append(
                f"  TRADE calls: {t['hit_20pct']}% hit 20% target  |  "
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
            if hit_rate < 40:
                parts.append(
                    "  → Your TRADE calls are underperforming. "
                    "Raise your bar — only call TRADE on the highest-conviction setups."
                )
            elif hit_rate > 65:
                parts.append(
                    "  → Your TRADE calls are performing well. "
                    "Confidence is justified on quality setups."
                )
        accuracy_section = "\n".join(parts)

    prompt = f"""You are a momentum trading AI making a TRADE or NO_TRADE call for a low-float microcap.
Strategy: buy stocks showing unusual volume compression setups targeting 20%+ next-day spike.

Stock: {stock['symbol']} | Score: {stock['score']}/100 | Price: ${stock['price']}
Signals:
- Relative Volume: {checklist.get('relative_volume')}x
- Shares Outstanding: {shares_str}
- Daily gain today: {stock.get('daily_return_pct', 'N/A')}%
- Sideways Compression: {checklist.get('sideways_chop')}
- Yesterday Green: {checklist.get('yesterday_green')}
- Institutional Ownership: {str(checklist.get('institution_pct')) + '%' if checklist.get('institution_pct') is not None else 'N/A'} (40%+ = strong floor, lowers downside risk)
- Sector/Industry: {checklist.get('sector') or 'N/A'} / {checklist.get('industry') or 'N/A'} (Biotech/Healthcare historically outperforms for this setup){calibration_section}{hypothesis_section}{history_section}{market_section}{lstm_section}{news_section}{accuracy_section}

Make a TRADE or NO_TRADE call. Does this setup match learned patterns? Is the score/signal quality sufficient?

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
        text = message.content[0].text.strip()
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
    except Exception:
        return {"decision": "NO_TRADE", "confidence": "LOW", "rationale": ""}


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
        return message.content[0].text.strip()
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
        return message.content[0].text.strip()
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
        return message.content[0].text.strip()
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
        return message.content[0].text.strip()
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
- Tier ordering must be preserved: rel_vol_50x >= rel_vol_25x >= rel_vol_10x >= rel_vol_5x
- Tier ordering must be preserved: shares_lt10m >= shares_lt30m >= shares_lt100m
- Upweight signals where hit-20%+ rate is highest AND avg-days-to-target is lowest

Also suggest 3–5 new scoring criteria/metrics to add (drawn from hypothesis patterns or backtest gaps).

Respond in EXACTLY this format with no other text:

WEIGHTS_JSON:
{{"rel_vol_50x": 0, "rel_vol_25x": 0, "rel_vol_10x": 0, "rel_vol_5x": 0, "daily_sweet_20_40": 0, "daily_ok_10_20": 0, "daily_ok_40_100": 0, "sideways_chop": 0, "yesterday_green": 0, "shares_lt10m": 0, "shares_lt30m": 0, "shares_lt100m": 0, "no_news_bonus": 0, "high_cash_bonus": 0, "institution_moderate": 0, "institution_strong": 0}}

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
        text = message.content[0].text.strip()

        # Parse WEIGHTS_JSON block
        if "WEIGHTS_JSON:" not in text:
            raise ValueError("No WEIGHTS_JSON section in response")
        json_part = text.split("WEIGHTS_JSON:")[1].split("RATIONALE:")[0].strip()
        new_weights = json.loads(json_part)

        # Validate and clamp all required keys to int 0-50
        for key in REQUIRED_KEYS:
            new_weights[key] = max(0, min(50, int(new_weights.get(key, current_weights.get(key, 5)))))

        # Enforce tier ordering constraints
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
                    f"{hit}% hit 20%+ target{days_str} | "
                    f"{stats['win_rate']}% any-gain | "
                    f"{stats['avg_return']:+.1f}% avg return")
        rv = opt_data["relative_volume"]
        dg = opt_data["daily_gain"]
        so = opt_data["shares_outstanding"]
        backtest_section = f"""BACKTESTED SIGNAL PERFORMANCE ({opt_data['total_trades']} trades):
Relative Volume: >=50x: {fmt(rv.get(">=50x"))} | 25-50x: {fmt(rv.get("25-50x"))} | 10-25x: {fmt(rv.get("10-25x"))} | <10x: {fmt(rv.get("<10x"))}
Daily Gain: 20-40%: {fmt(dg.get("20-40%"))} | 10-20%: {fmt(dg.get("10-20%"))} | 40-100%: {fmt(dg.get("40-100%"))} | <10%: {fmt(dg.get("<10%"))}
Shares: <1M: {fmt(so.get("<1M"))} | 1-5M: {fmt(so.get("1-5M"))} | 5-10M: {fmt(so.get("5-10M"))} | 10-30M: {fmt(so.get("10-30M"))} | 30M+: {fmt(so.get("30M+"))}"""
    else:
        backtest_section = "BACKTESTED SIGNAL PERFORMANCE: Insufficient data (<5 trades with known outcomes)."

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

    prompt = f"""You are autonomously evolving a self-improving stock trading model for low-float microcap momentum setups targeting 20%+ intraday spikes.

You have TWO tasks:

TASK 1 — HYPOTHESIS EVOLUTION: Generate pattern rules supported by the evidence. Assign a confidence score (0-100) to each.
TASK 2 — CRITERIA & WEIGHT OPTIMIZATION: Decide which scoring criteria to keep, remove, or add. Set weight=0 to disable a criterion, 1-50 to enable it.

## Current Auto AI Weights (0 = disabled, 1–50 = enabled; scorer normalises to 0–100 automatically)
{json.dumps(current_display, indent=2)}

## Scoring Criteria Control
EXISTING CRITERIA — set to 0 to remove if evidence shows it reduces win rate or % return:
  rel_vol_50x, rel_vol_25x, rel_vol_10x, rel_vol_5x  [relative volume tiers, keep descending order]
  daily_sweet_20_40, daily_ok_10_20, daily_ok_40_100  [daily gain tiers]
  shares_lt10m, shares_lt30m, shares_lt100m            [share count tiers, keep descending order]
  sideways_chop    10-day price range <20%: consolidation before breakout
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

## Evidence
{backtest_section}

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
- Tier ordering: rel_vol_50x >= rel_vol_25x >= rel_vol_10x >= rel_vol_5x
- Tier ordering: shares_lt10m >= shares_lt30m >= shares_lt100m
- Include ALL 22 keys in the output weights dict (set unwanted ones to 0)

## Required Output Format
Respond with ONLY a valid JSON object (no markdown, no code fences, no other text):

{{"hypotheses": [{{"text": "specific pattern rule", "source": "historical|feedback|both", "confidence": 85}}], "weights": {{"rel_vol_50x": 30, "rel_vol_25x": 22, "rel_vol_10x": 15, "rel_vol_5x": 7, "daily_sweet_20_40": 10, "daily_ok_10_20": 5, "daily_ok_40_100": 5, "sideways_chop": 8, "yesterday_green": 7, "shares_lt10m": 30, "shares_lt30m": 18, "shares_lt100m": 8, "no_news_bonus": 5, "high_cash_bonus": 5, "institution_moderate": 2, "institution_strong": 5, "sector_biotech_bonus": 5, "rsi_momentum_bonus": 0, "macd_positive_bonus": 0, "bb_upper_breakout": 0, "consecutive_green_bonus": 0, "low_float_ratio_bonus": 0}}, "weight_confidence": 82, "rationale": "2-3 sentences citing specific evidence that drove changes", "summary": "one sentence max 20 words", "suggestions": ["new signal idea 1", "new signal idea 2"]}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text.strip()

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
        return msg.content[0].text.strip()
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
        vision_text = vision_msg.content[0].text.strip()

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
            return enrich_msg.content[0].text.strip()

        # Return chart-only analysis if no historical data available
        return vision_text

    except Exception as e:
        return f"Could not analyze chart: {str(e)}"
