# ai_trading_mvp/app/ai_agent.py

import os
import base64
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


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
- Institutional Ownership: {str(checklist.get('institution_pct')) + '%' if checklist.get('institution_pct') is not None else 'N/A'}{feedback_section}

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
{{"rel_vol_50x": 0, "rel_vol_25x": 0, "rel_vol_10x": 0, "daily_sweet_20_40": 0, "daily_ok_10_20": 0, "daily_ok_40_100": 0, "sideways_chop": 0, "yesterday_green": 0, "shares_lt10m": 0, "shares_lt30m": 0, "shares_gte100m_penalty": 0}}

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


def analyze_chart_feedback(image_bytes: bytes, media_type: str, user_text: str) -> str:
    """
    Uses Claude vision to extract chart patterns and trade signals from a
    user-submitted screenshot. Returns a bullet-point analysis string.
    """
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    notes = f"User notes: {user_text}" if user_text else "No notes provided."

    prompt = f"""Analyze this trading chart screenshot submitted as a winning trade example.
{notes}

Extract and list:
1. Chart pattern visible (consolidation box, barcode, deep collapse rebound, breakout, etc.)
2. Volume profile (spike, sustained, thin)
3. Price action characteristics (tight range before move, gap up, steady climb, etc.)
4. Key signals that made this a good setup
5. One-line lesson for identifying similar setups in the future

Be specific and concise. Use bullet points."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=450,
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
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        return message.content[0].text.strip()
    except Exception as e:
        return f"Could not analyze chart: {str(e)}"
