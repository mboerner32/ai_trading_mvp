# ai_trading_mvp/app/scoring_engine.py

def score_stock(symbol: str, df, fundamentals=None):

    # --------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------

    if df is None or len(df) < 20:
        return None  # only fail if completely unusable

    latest = df.iloc[-1]

    # Safe numeric conversion
    def safe_float(value):
        try:
            return float(value)
        except Exception:
            return None

    price = safe_float(latest.get("close"))
    volume = safe_float(latest.get("volume"))
    relative_volume = safe_float(latest.get("relative_volume"))
    daily_return = safe_float(latest.get("daily_return"))
    return_3d = safe_float(latest.get("return_3d"))
    return_5d = safe_float(latest.get("return_5d"))
    range_10d = safe_float(latest.get("range_10d"))

    # If price is missing, we cannot score
    if price is None:
        return None

    score = 0

    # --------------------------------------------------
    # CHECKLIST LOGIC (NO HARD FILTERING)
    # --------------------------------------------------

    sweet_spot = False
    overheated = False
    sideways_chop = False
    recent_decline = False
    yesterday_green = False

    # Relative Volume
    if relative_volume is not None:
        if relative_volume >= 29:
            score += 2
        elif relative_volume >= 10:
            score += 1

    # Performance Sweet Spot (10–40%)
    if daily_return is not None:
        sweet_spot = 0.10 <= daily_return <= 0.40
        overheated = daily_return > 1.00

        if sweet_spot:
            score += 2

    # Sideways Compression
    if range_10d is not None:
        sideways_chop = range_10d < 0.20
        if sideways_chop:
            score += 1

    # Recent Selloff Penalty
    if return_3d is not None and return_5d is not None:
        recent_decline = return_3d < -0.20 or return_5d < -0.30
        if recent_decline:
            score -= 2

    # Yesterday Green
    if len(df) >= 2:
        prev_return = safe_float(df.iloc[-2].get("daily_return"))
        if prev_return is not None and prev_return > 0:
            yesterday_green = True

    # --------------------------------------------------
    # FUNDAMENTALS (Injected from scanner)
    # --------------------------------------------------

    shares_outstanding = None
    float_shares = None
    total_cash = None
    cash_per_share = None
    small_float = False
    high_cash = False
    institution_pct = None

    if fundamentals:

        shares_outstanding = fundamentals.get("shares_outstanding")
        float_shares = fundamentals.get("float_shares")
        total_cash = fundamentals.get("total_cash")
        institution_pct = fundamentals.get("institution_pct")

        # Small Float Benchmark (<5M)
        if shares_outstanding and shares_outstanding < 5_000_000:
            small_float = True
            score += 2

        # Cash Per Share > Price
        if shares_outstanding and total_cash:
            try:
                cash_per_share = total_cash / shares_outstanding
                if cash_per_share > price:
                    high_cash = True
                    score += 1
            except Exception:
                cash_per_share = None

        # Institutional Ownership (max +2)
        # Strong backing >= 50%: +2, moderate >= 20%: +1
        if institution_pct is not None:
            if institution_pct >= 0.50:
                score += 2
            elif institution_pct >= 0.20:
                score += 1

    # --------------------------------------------------
    # SCALE TO 0–100
    # Max raw score: 2 (rel vol) + 2 (sweet spot) + 1 (sideways)
    #              + 2 (small float) + 1 (high cash) + 2 (inst. ownership) = 10
    # --------------------------------------------------

    score = max(0, round((score / 10) * 100))

    # --------------------------------------------------
    # RECOMMENDATION TIERS
    # --------------------------------------------------

    recommendation = "SPECULATIVE"

    if score >= 50:
        recommendation = "WATCH"

    if score >= 75:
        recommendation = "TRADE"

    # --------------------------------------------------
    # RETURN STRUCTURE (MATCHES DASHBOARD)
    # --------------------------------------------------

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "volume": int(volume) if volume else None,
        "relative_volume": round(relative_volume, 2) if relative_volume else None,
        "daily_return_pct": round(daily_return * 100, 2) if daily_return else None,
        "score": score,
        "recommendation": recommendation,
        "checklist": {
            "relative_volume": round(relative_volume, 2) if relative_volume else None,

            # Float
            "shares_outstanding": shares_outstanding,
            "float_shares": float_shares,
            "small_float": small_float,

            # Cash
            "total_cash": total_cash,
            "cash_per_share": round(cash_per_share, 2) if cash_per_share else None,
            "high_cash": high_cash,

            # Institutional Ownership
            "institution_pct": round(institution_pct * 100, 1) if institution_pct is not None else None,

            # Performance
            "sweet_spot_10_40": sweet_spot,
            "over_100_percent": overheated,
            "five_day_return_pct": round(return_5d * 100, 2) if return_5d else None,

            # Technical
            "sideways_chop": sideways_chop,
            "yesterday_green": yesterday_green,
            "recent_decline": recent_decline,
        }
    }
