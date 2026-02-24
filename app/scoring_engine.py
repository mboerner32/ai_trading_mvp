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

    # Prefer FinViz's own Rel Volume (accurate, time-of-day adjusted)
    if fundamentals and fundamentals.get("finviz_relvol"):
        relative_volume = fundamentals["finviz_relvol"]

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

    # No News Catalyst — organic move preferred over news-driven pump (+1)
    no_news_catalyst = not (fundamentals or {}).get("recent_news_present", False)
    if no_news_catalyst:
        score += 1

    # --------------------------------------------------
    # SCALE TO 0–100
    # Max raw score: 2 (rel vol) + 2 (sweet spot) + 1 (sideways)
    #              + 2 (small float) + 1 (high cash) + 2 (inst. ownership) + 1 (no news) = 11
    # --------------------------------------------------

    score = max(0, round((score / 11) * 100))

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
        "news_headlines": (fundamentals or {}).get("news_headlines", []),
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

            # News
            "no_news_catalyst": no_news_catalyst,

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


# --------------------------------------------------
# SQUEEZE MODE SCORER
# Relative Volume Short-Squeeze Momentum System
# Target: +20% moves in low-float microcaps with explosive volume
# --------------------------------------------------

# Default weights — overridden by AI optimizer when stored in DB
DEFAULT_SQUEEZE_WEIGHTS = {
    "rel_vol_50x":            3,   # relative volume >= 50x
    "rel_vol_25x":            2,   # relative volume >= 25x (exclusive)
    "rel_vol_10x":            1,   # relative volume >= 10x (exclusive)
    "daily_sweet_20_40":      2,   # daily gain 20–40% (sweet spot)
    "daily_ok_10_20":         1,   # daily gain 10–20%
    "daily_ok_40_100":        1,   # daily gain 40–100%
    "sideways_chop":          1,   # 10-day range < 20% (consolidation)
    "yesterday_green":        1,   # previous day positive
    "shares_lt10m":           3,   # shares outstanding < 10M
    "shares_lt30m":           1,   # shares outstanding 10M–30M
    "shares_gte100m_penalty": 2,   # deducted when shares >= 100M (stored positive)
    "no_news_bonus":          1,   # organic move with no news catalyst
}


def score_stock_squeeze(symbol: str, df, fundamentals=None, weights=None):

    if df is None or len(df) < 20:
        return None

    latest = df.iloc[-1]

    def safe_float(value):
        try:
            return float(value)
        except Exception:
            return None

    price          = safe_float(latest.get("close"))
    volume         = safe_float(latest.get("volume"))
    relative_volume = safe_float(latest.get("relative_volume"))
    daily_return   = safe_float(latest.get("daily_return"))
    range_10d      = safe_float(latest.get("range_10d"))

    # Prefer FinViz's own Rel Volume (accurate, time-of-day adjusted)
    if fundamentals and fundamentals.get("finviz_relvol"):
        relative_volume = fundamentals["finviz_relvol"]

    if price is None:
        return None

    # Hard exclusion: blow-off phase (>100% gain today)
    if daily_return is not None and daily_return > 1.00:
        return None

    # Merge provided weights with defaults so any missing key falls back safely
    w = {**DEFAULT_SQUEEZE_WEIGHTS, **(weights or {})}

    score = 0

    # --- Relative Volume ---
    relvol_tier = "Below Threshold"
    if relative_volume is not None:
        if relative_volume >= 50:
            score += w["rel_vol_50x"]
            relvol_tier = "Exceptional (≥50x)"
        elif relative_volume >= 25:
            score += w["rel_vol_25x"]
            relvol_tier = "Ideal (≥25x)"
        elif relative_volume >= 10:
            score += w["rel_vol_10x"]
            relvol_tier = "Minimum (≥10x)"

    # --- Daily Gain ---
    sweet_spot_squeeze = False
    if daily_return is not None:
        sweet_spot_squeeze = 0.20 <= daily_return <= 0.40
        if sweet_spot_squeeze:
            score += w["daily_sweet_20_40"]
        elif 0.10 <= daily_return < 0.20:
            score += w["daily_ok_10_20"]
        elif 0.40 < daily_return <= 1.00:
            score += w["daily_ok_40_100"]

    # --- Sideways Consolidation / Barcoding ---
    sideways_chop = False
    if range_10d is not None:
        sideways_chop = range_10d < 0.20
        if sideways_chop:
            score += w["sideways_chop"]

    # --- Yesterday Green ---
    yesterday_green = False
    if len(df) >= 2:
        prev_return = safe_float(df.iloc[-2].get("daily_return"))
        if prev_return is not None and prev_return > 0:
            yesterday_green = True
            score += w["yesterday_green"]

    # --- Shares Outstanding ---
    shares_outstanding = None
    float_shares       = None
    institution_pct    = None
    float_tier         = "Unknown"

    if fundamentals:
        shares_outstanding = fundamentals.get("shares_outstanding")
        float_shares       = fundamentals.get("float_shares")
        institution_pct    = fundamentals.get("institution_pct")

        if shares_outstanding is not None:
            if shares_outstanding < 10_000_000:
                score += w["shares_lt10m"]
                float_tier = "Ideal (<10M)"
            elif shares_outstanding < 30_000_000:
                score += w["shares_lt30m"]
                float_tier = "Acceptable (<30M)"
            elif shares_outstanding < 100_000_000:
                float_tier = "Large (30M–100M)"
            else:
                score -= w["shares_gte100m_penalty"]
                float_tier = "Avoid (≥100M)"

    # --- No News Catalyst ---
    no_news_catalyst = not (fundamentals or {}).get("recent_news_present", False)
    if no_news_catalyst:
        score += w.get("no_news_bonus", 0)

    # Compute max achievable score with current weights (no penalties)
    max_score = (
        max(w["rel_vol_50x"], w["rel_vol_25x"], w["rel_vol_10x"], 0) +
        max(w["daily_sweet_20_40"], w["daily_ok_10_20"], w["daily_ok_40_100"], 0) +
        w["sideways_chop"] +
        w["yesterday_green"] +
        max(w["shares_lt10m"], w["shares_lt30m"], 0) +
        w.get("no_news_bonus", 0)
    )
    if max_score <= 0:
        max_score = 10

    score = max(0, round((score / max_score) * 100))

    recommendation = "SPECULATIVE"
    if score >= 50:
        recommendation = "WATCH"
    if score >= 75:
        recommendation = "TRADE"

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "volume": int(volume) if volume else None,
        "relative_volume": round(relative_volume, 2) if relative_volume else None,
        "daily_return_pct": round(daily_return * 100, 2) if daily_return else None,
        "score": score,
        "recommendation": recommendation,
        "news_headlines": (fundamentals or {}).get("news_headlines", []),
        "checklist": {
            "relative_volume": round(relative_volume, 2) if relative_volume else None,
            "relvol_tier": relvol_tier,

            # Float
            "shares_outstanding": shares_outstanding,
            "float_shares": float_shares,
            "float_tier": float_tier,
            "small_float": shares_outstanding < 10_000_000 if shares_outstanding else False,

            # Performance
            "sweet_spot_squeeze": sweet_spot_squeeze,
            "daily_return_pct": round(daily_return * 100, 2) if daily_return else None,

            # Technical
            "sideways_chop": sideways_chop,
            "yesterday_green": yesterday_green,

            # Institutional Ownership
            "institution_pct": round(institution_pct * 100, 1) if institution_pct is not None else None,

            # News
            "no_news_catalyst": no_news_catalyst,

            # Placeholders for template compatibility
            "high_cash": False,
            "cash_per_share": None,
            "five_day_return_pct": None,
            "recent_decline": False,
            "over_100_percent": False,
            "sweet_spot_10_40": sweet_spot_squeeze,
        }
    }
