print("Latest row:", latest.to_dict())
import yfinance as yf


def score_stock(symbol: str, df):
    """
    Applies your full trading checklist logic
    Returns structured result with score + recommendation
    """

    if df is None or len(df) < 20:
        return None

    latest = df.iloc[-1]

    try:
        price = float(latest["close"])
        relative_volume = float(latest["relative_volume"])
        daily_return = float(latest["daily_return"])
        volume = float(latest["volume"])
        return_3d = float(latest["return_3d"])
        return_5d = float(latest["return_5d"])
        range_10d = float(latest["range_10d"])
    except Exception:
        return None

    score = 0
    notes = []

    # --------------------------------------------------
    # HARD FILTERS (Auto reject)
    # --------------------------------------------------

    if price >= 5:
        return None  # Must be under $5

    if volume < 5_000_000:
        return None  # Must trade >5M shares today

    if daily_return < 0.10:
        return None  # Must be up at least +10%

    if daily_return > 1.00:
        return None  # Reject parabolic >100%

    if relative_volume < 10:
        return None  # Must have high relative volume

    # --------------------------------------------------
    # SCORING SECTION
    # --------------------------------------------------

    # Relative Volume Sweet Spot
    if relative_volume >= 29:
        score += 2
        notes.append("🔥 RVOL Sweet Spot (29+)")

    elif relative_volume >= 10:
        score += 1
        notes.append("High Relative Volume")

    # Premium tier trigger
    if relative_volume >= 100:
        score += 2
        notes.append("🚀 EXTREME RVOL (100+)")

    # Ideal daily performance range
    if 0.10 <= daily_return <= 0.40:
        score += 2
        notes.append("Ideal Performance Range (10–40%)")

    # Sideways chop setup
    if range_10d < 0.20:
        score += 1
        notes.append("Sideways Compression")

    # Reject recent dump pattern
    if return_3d < -0.20 or return_5d < -0.30:
        notes.append("Recent Selloff Detected")
        score -= 2

    # --------------------------------------------------
    # FLOAT / FUNDAMENTAL DATA
    # --------------------------------------------------

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        shares_outstanding = info.get("sharesOutstanding", None)
        cash = info.get("totalCash", None)

        if shares_outstanding and shares_outstanding < 5_000_000:
            score += 2
            notes.append("Low Float (<5M shares)")

        if shares_outstanding and cash:
            cash_per_share = cash / shares_outstanding
            if cash_per_share > price:
                score += 1
                notes.append("Strong Cash Position")

    except Exception:
        pass

    # --------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------

    recommendation = "DO NOT TRADE"

    if score >= 4:
        recommendation = "WATCH"
    if score >= 6:
        recommendation = "TRADE"

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "relative_volume": round(relative_volume, 2),
        "daily_return_pct": round(daily_return * 100, 2),
        "volume": int(volume),
        "score": score,
        "recommendation": recommendation,
        "notes": notes,
    }


