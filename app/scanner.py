from app.universe import get_finviz_universe
from app.data_provider import get_stock_data
from app.database import log_scan


def calculate_technical_signals(history):
    sideways = False
    yesterday_green = False
    recent_decline = False
    tech_score = 0

    if history is None or len(history) < 4:
        return sideways, yesterday_green, recent_decline, tech_score

    closes = history["Close"]
    highs = history["High"]
    lows = history["Low"]
    opens = history["Open"]

    recent_high = highs.tail(4).max()
    recent_low = lows.tail(4).min()

    range_pct = ((recent_high - recent_low) / recent_low) * 100

    if range_pct < 15:
        sideways = True
        tech_score += 10

    if closes.iloc[-2] > opens.iloc[-2]:
        yesterday_green = True
        tech_score += 5

    three_day_drop = ((closes.iloc[-4] - closes.iloc[-1]) / closes.iloc[-4]) * 100

    if three_day_drop > 20:
        recent_decline = True
        tech_score -= 20

    return sideways, yesterday_green, recent_decline, tech_score


def run_scan(strict: bool = False):
    tickers = get_finviz_universe()
    results = []

    for ticker in tickers:
        try:
            data = get_stock_data(ticker)
            if not data:
                continue

            price = data["price"]
            today_return = data["today_return"]
            relative_volume = data["relative_volume"]
            current_volume = data["current_volume"]
            shares = data.get("shares_outstanding", 0)
            cash_per_share = data.get("cash_per_share", 0)
            market_cap = data.get("market_cap", 0)
            news_recent = data.get("news_recent", False)
            history = data.get("history")

            # Hard Filters
            if price >= 5:
                continue

            if today_return < 10:
                continue

            if today_return > 100:
                continue

            if current_volume < 5_000_000:
                continue

            score = 0
            breakdown = {}

            # Relative Volume
            if relative_volume >= 29:
                rv_score = 25
                rv_label = "Sweet Spot"
            elif relative_volume >= 10:
                rv_score = 10
                rv_label = "High"
            else:
                rv_score = 0
                rv_label = "Low"

            score += rv_score
            breakdown["relative_volume"] = {
                "value": round(relative_volume, 2),
                "score": rv_score,
                "label": rv_label,
            }

            # Shares Outstanding
            if shares and shares < 5_000_000:
                float_score = 20
                float_label = "Low Float"
            elif shares and shares < 10_000_000:
                float_score = 10
                float_label = "Moderate Float"
            else:
                float_score = 0
                float_label = "Large Float"

            score += float_score
            breakdown["shares_outstanding"] = {
                "value": shares,
                "score": float_score,
                "label": float_label,
            }

            # Cash Strength
            cash_strength = cash_per_share * shares if shares else 0

            if market_cap and cash_strength > market_cap * 0.5:
                cash_score = 15
                cash_label = "Strong"
            elif market_cap and cash_strength > market_cap * 0.25:
                cash_score = 8
                cash_label = "Moderate"
            else:
                cash_score = 0
                cash_label = "Weak"

            score += cash_score
            breakdown["cash_strength"] = {
                "value": round(cash_strength, 2),
                "score": cash_score,
                "label": cash_label,
            }

            # News
            if news_recent:
                news_score = -15
                news_label = "Recent Catalyst"
            else:
                news_score = 10
                news_label = "No News"

            score += news_score
            breakdown["news"] = {
                "value": news_recent,
                "score": news_score,
                "label": news_label,
            }

            # Performance
            if 10 <= today_return <= 40:
                perf_score = 15
                perf_label = "Sweet Spot"
            elif 40 < today_return <= 100:
                perf_score = 5
                perf_label = "Extended"
            else:
                perf_score = 0
                perf_label = "Weak"

            score += perf_score
            breakdown["performance"] = {
                "value": round(today_return, 2),
                "score": perf_score,
                "label": perf_label,
            }

            # Technical
            sideways, yesterday_green, recent_decline, tech_score = \
                calculate_technical_signals(history)

            score += tech_score

            breakdown["technical"] = {
                "sideways": sideways,
                "yesterday_green": yesterday_green,
                "recent_decline": recent_decline,
                "score": tech_score,
            }

            # Strict Mode Enforcement
            if strict:
                if relative_volume < 29:
                    continue
                if shares >= 5_000_000:
                    continue
                if score < 60:
                    continue

            # Recommendation
            if score >= 80:
                recommendation = "🔥 A+ Setup"
            elif score >= 55:
                recommendation = "⚡ Strong Setup"
            elif score >= 35:
                recommendation = "📈 Watchlist"
            else:
                recommendation = "👀 Speculative"

            results.append({
                "symbol": ticker,
                "price": round(price, 2),
                "score": score,
                "recommendation": recommendation,
                "breakdown": breakdown,
            })

        except Exception:
            continue

    # Sort results
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    total_scanned = len(tickers)
    total_passed = len(results)

    avg_score = round(
        sum(r["score"] for r in results) / total_passed, 1
    ) if total_passed > 0 else 0

    summary = {
        "total_scanned": total_scanned,
        "total_passed": total_passed,
        "avg_score": avg_score,
    }

    # Log scan results for backtesting
    log_scan(results, "strict" if strict else "standard")

    return {
        "results": results,
        "summary": summary,
    }
