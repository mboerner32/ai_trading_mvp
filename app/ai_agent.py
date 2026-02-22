# ai_trading_mvp/app/ai_agent.py

import os
import base64
import anthropic
import yfinance as yf
import pandas as pd
from datetime import datetime, date, timedelta

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically


def recommend_position_size(stock: dict, available_cash: float,
                            hypothesis: str = None) -> dict:
    """
    Calls Claude to recommend a position size for a given stock.
    Returns {"amount": int, "rationale": str}.
    Falls back to {"amount": 1000, "rationale": ""} on any error.

    hypothesis: synthesized pattern hypotheses from accumulated user feedback.
                If provided, injected as learned context to inform sizing.
    """
    checklist = stock.get("checklist", {})

    shares_outstanding = checklist.get("shares_outstanding")
    shares_str = f"{shares_outstanding:,}" if shares_outstanding else "N/A"

    # Inject synthesized hypothesis context (concise — first 500 chars)
    feedback_section = ""
    if hypothesis:
        summary = hypothesis[:500].strip()
        feedback_section = (
            f"\n\nHypotheses derived from user-submitted winning trade patterns:\n{summary}"
            "\nUse these hypotheses to adjust sizing confidence."
        )

    prompt = f"""You are a trading risk manager for a paper trading simulator.

Stock: {stock['symbol']}
Score: {stock['score']}/100
Recommendation: {stock['recommendation']}
Price: ${stock['price']}
Available cash: ${available_cash:.0f}

Checklist signals:
- Relative Volume: {checklist.get('relative_volume')}
- Small Float (<5M shares): {checklist.get('small_float')} ({shares_str} shares)
- High Cash per Share: {checklist.get('high_cash')}
- Sweet Spot (10-40% today): {checklist.get('sweet_spot_10_40')}
- Overheated (>100% today): {checklist.get('over_100_percent')}
- Sideways Compression: {checklist.get('sideways_chop')}
- Yesterday Green: {checklist.get('yesterday_green')}
- Recent Decline: {checklist.get('recent_decline')}
- 5-Day Return: {checklist.get('five_day_return_pct')}%
- Institutional Ownership: {str(checklist.get('institution_pct')) + '%' if checklist.get('institution_pct') is not None else 'N/A'}
- No News Catalyst (organic move): {checklist.get('no_news_catalyst')}{feedback_section}

Recommend a position size in dollars. Choose from: $250, $500, $750, $1000, $1500, $2000.
Never recommend more than ${available_cash:.0f} available cash.

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


def synthesize_combined_hypothesis(feedback_entries: list,
                                   opt_data: dict,
                                   historical_count: int) -> str:
    """
    Unified synthesis combining qualitative manual feedback AND quantitative
    historical scan data. Produces a single hypothesis for both sources.
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
            return (f"  {name}: {stats['count']} trades, "
                    f"{stats['win_rate']}% win rate, "
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

    prompt = f"""You are synthesizing trading pattern hypotheses from two sources:
1. {len(feedback_entries)} manually submitted winning trades (qualitative: chart screenshots + trader notes)
2. {historical_count} labeled historical scan examples (quantitative: next-day return statistics)

{feedback_section}

{historical_section}

Identify patterns supported by either or both sources. For each hypothesis:
- State what the pattern is
- Note whether evidence is from manual submissions, historical stats, or both
- Rate confidence: HIGH (strong evidence), EMERGING (moderate), PRELIMINARY (limited)
- State how it should influence position sizing (size up, size down, no effect)

Then write an AGENT CONTEXT section: a compact 3-4 sentence summary of the highest-confidence hypotheses for injection into a trading AI agent.

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
        return (f"  {name}: {stats['count']} trades, "
                f"{stats['win_rate']}% win rate, "
                f"{stats['avg_return']:+.1f}% avg next-day return")

    relvol_lines  = [fmt_bucket(k, v) for k, v in opt_data.get("relative_volume", {}).items()]
    gain_lines    = [fmt_bucket(k, v) for k, v in opt_data.get("daily_gain", {}).items()]
    shares_lines  = [fmt_bucket(k, v) for k, v in opt_data.get("shares_outstanding", {}).items()]

    relvol_text  = "\n".join(l for l in relvol_lines  if l)
    gain_text    = "\n".join(l for l in gain_lines    if l)
    shares_text  = "\n".join(l for l in shares_lines  if l)

    prompt = f"""You are analyzing {historical_count} historical momentum/squeeze trades to find which signals predict next-day gains.

Relative Volume buckets (next-day return after scan day):
{relvol_text}

Today's gain buckets:
{gain_text}

Shares outstanding buckets:
{shares_text}

Identify the strongest statistical patterns. For each hypothesis:
- State what the signal predicts
- Reference the supporting win rate and avg return
- Rate confidence: HIGH (many trades, clear edge), EMERGING (positive pattern, moderate sample), PRELIMINARY (limited data)
- State how this should influence position sizing

Then write an AGENT CONTEXT section: a compact 3-4 sentence summary for injection into a trading AI agent.

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
    adjustments to maximize win rate × avg return.
    Uses claude-sonnet-4-6 for richer reasoning.
    """
    def fmt(stats):
        if not stats or stats["count"] == 0:
            return "No data"
        return (f"{stats['count']} trades | "
                f"{stats['win_rate']}% win rate | "
                f"{stats['avg_return']:+.1f}% avg 1-day return")

    rv = opt_data["relative_volume"]
    dg = opt_data["daily_gain"]
    so = opt_data["shares_outstanding"]

    prompt = f"""You are a quantitative trading system optimizer.
Analyze this backtested signal performance and recommend specific integer weight adjustments.

TOTAL BACKTESTED TRADES: {opt_data['total_trades']}

RELATIVE VOLUME (current scoring: ≥29x → +2pts, ≥10x → +1pt, max 10pts total):
- ≥50x:    {fmt(rv.get(">=50x"))}
- 25–50x:  {fmt(rv.get("25-50x"))}
- 10–25x:  {fmt(rv.get("10-25x"))}
- <10x:    {fmt(rv.get("<10x"))}

DAILY GAIN (current scoring: 10–40% → +2pts, else → 0pts):
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

For each signal, state whether to increase, decrease, split, or keep the current weights and by how much. Focus on the combination of win rate AND average return — a bucket with 70% win rate and +5% avg return is better than 60% win rate and +8% avg return.

Format your response with these exact section headers:
## Relative Volume Weights
## Daily Gain Weights
## Shares Outstanding Weights
## Top 3 Changes to Implement Now"""

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
    Analyzes backtesting signal performance + trader feedback to produce
    AI-optimized scoring weights for Complex + AI Mode.

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
        "rel_vol_50x", "rel_vol_25x", "rel_vol_10x",
        "daily_sweet_20_40", "daily_ok_10_20", "daily_ok_40_100",
        "sideways_chop", "yesterday_green",
        "shares_lt10m", "shares_lt30m", "shares_gte100m_penalty",
        "no_news_bonus",
    ]

    # ---- Format backtest section ----
    if opt_data and opt_data.get("total_trades", 0) >= 5:
        def fmt(stats):
            if not stats or stats["count"] == 0:
                return "No data"
            return (f"{stats['count']} trades | "
                    f"{stats['win_rate']}% win rate | "
                    f"{stats['avg_return']:+.1f}% avg return")

        rv = opt_data["relative_volume"]
        dg = opt_data["daily_gain"]
        so = opt_data["shares_outstanding"]

        backtest_section = f"""BACKTESTED SIGNAL PERFORMANCE ({opt_data['total_trades']} trades with known next-day returns):

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
            "(<5 trades with known outcomes). Base adjustments on feedback patterns."
        )

    # ---- Format feedback/hypothesis section ----
    if hypothesis:
        feedback_section = f"TRADER FEEDBACK & HYPOTHESES:\n{hypothesis[:800]}"
    elif all_feedback:
        lines = ""
        for fb in all_feedback[-6:]:
            sym = fb.get("symbol") or "?"
            txt = (fb.get("user_text") or "")[:180]
            lines += f"\n- {sym}: {txt}"
        feedback_section = f"TRADER FEEDBACK (last {min(6, len(all_feedback))} submissions):{lines}"
    else:
        feedback_section = "TRADER FEEDBACK: None submitted yet."

    prompt = f"""You are optimizing the scoring weights for a Complex + AI stock scanner targeting short-squeeze momentum setups in low-float microcaps.

## Current Active Weights
{json.dumps(current_weights, indent=2)}

Weight definitions:
- rel_vol_50x:            points awarded when relative volume >= 50x
- rel_vol_25x:            points awarded when relative volume >= 25x (exclusive)
- rel_vol_10x:            points awarded when relative volume >= 10x (exclusive)
- daily_sweet_20_40:      points for daily gain 20-40% (momentum sweet spot)
- daily_ok_10_20:         points for daily gain 10-20% (early stage)
- daily_ok_40_100:        points for daily gain 40-100% (extended but not blown off)
- sideways_chop:          points for 10-day price range < 20% (tight consolidation)
- yesterday_green:        points when previous trading day closed positive
- shares_lt10m:           points when shares outstanding < 10M (ideal float)
- shares_lt30m:           points when shares outstanding 10-30M (acceptable float)
- shares_gte100m_penalty: points DEDUCTED when shares >= 100M (too diluted to squeeze)
- no_news_bonus:          points awarded when no news catalyst detected (organic move preferred)

## Evidence
{backtest_section}

{feedback_section}

## Task
Produce optimal integer weights (0-5 each) to maximize win_rate x avg_return for top-scored stocks.
- Upweight signals that show both high win rate AND positive avg return
- Zero-out or downweight signals that do not predict outcomes
- The penalty weight should reflect how strongly large floats hurt outcomes
- Also suggest 3-5 new data points to collect to further improve the model

Respond in EXACTLY this format with no other text:

WEIGHTS_JSON:
{{"rel_vol_50x": 0, "rel_vol_25x": 0, "rel_vol_10x": 0, "daily_sweet_20_40": 0, "daily_ok_10_20": 0, "daily_ok_40_100": 0, "sideways_chop": 0, "yesterday_green": 0, "shares_lt10m": 0, "shares_lt30m": 0, "shares_gte100m_penalty": 0, "no_news_bonus": 0}}

RATIONALE:
[2-3 sentences explaining the key weight changes and why]

SUGGESTIONS:
- [new signal idea 1]
- [new signal idea 2]
- [new signal idea 3]

SUMMARY:
[one sentence, max 20 words, for dashboard display]"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text.strip()

        # Parse WEIGHTS_JSON block
        if "WEIGHTS_JSON:" not in text:
            raise ValueError("No WEIGHTS_JSON section in response")
        json_part = text.split("WEIGHTS_JSON:")[1].split("RATIONALE:")[0].strip()
        new_weights = json.loads(json_part)

        # Validate and clamp all required keys to int 0-5
        for key in REQUIRED_KEYS:
            new_weights[key] = max(0, min(5, int(new_weights.get(key, current_weights.get(key, 1)))))

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
