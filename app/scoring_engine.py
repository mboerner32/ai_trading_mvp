# ai_trading_mvp/app/scoring_engine.py
#
# Scoring out of 100 natively — no scaling factor needed.
#
# Point budget (max 100):
#   Relative volume   30 pts  (primary signal)
#   Share count       30 pts  (primary signal)
#   Daily gain        10 pts
#   Sideways chop      8 pts
#   Yesterday green    7 pts
#   High cash/share    5 pts  (nice-to-have)
#   Institution own    5 pts  (nice-to-have)
#   No news            5 pts  (nice-to-have)
#   ─────────────────────────
#   Max               100 pts


def score_stock(symbol: str, df, fundamentals=None):

    if df is None or len(df) < 20:
        return None

    latest = df.iloc[-1]

    def safe_float(value):
        try:
            return float(value)
        except Exception:
            return None

    price           = safe_float(latest.get("close"))
    volume          = safe_float(latest.get("volume"))
    relative_volume = safe_float(latest.get("relative_volume"))
    daily_return    = safe_float(latest.get("daily_return"))
    return_3d       = safe_float(latest.get("return_3d"))
    return_5d       = safe_float(latest.get("return_5d"))
    range_10d       = safe_float(latest.get("range_10d"))

    # Prefer live FinViz values over yfinance-calculated ones where available
    if fundamentals:
        if fundamentals.get("finviz_price") is not None:
            price = fundamentals["finviz_price"]
        if fundamentals.get("finviz_volume") is not None:
            volume = float(fundamentals["finviz_volume"])
        if fundamentals.get("finviz_change_pct") is not None:
            daily_return = fundamentals["finviz_change_pct"] / 100
        if fundamentals.get("finviz_relvol") is not None:
            relative_volume = fundamentals["finviz_relvol"]

    if price is None:
        return None

    shares_outstanding = (fundamentals or {}).get("shares_outstanding")

    # ------------------------------------------------------------------
    # HARD REQUIREMENTS
    # ------------------------------------------------------------------
    if relative_volume is not None and relative_volume < 5:
        return None
    if shares_outstanding is not None and shares_outstanding >= 100_000_000:
        return None

    score = 0

    # ------------------------------------------------------------------
    # RELATIVE VOLUME — 30 pts max (primary signal)
    # ------------------------------------------------------------------
    relvol_tier = "Below Threshold"
    if relative_volume is not None:
        if relative_volume >= 50:
            score += 30
            relvol_tier = "Exceptional (≥50x)"
        elif relative_volume >= 25:
            score += 22
            relvol_tier = "Ideal (≥25x)"
        elif relative_volume >= 10:
            score += 15
            relvol_tier = "Good (≥10x)"
        else:
            score += 7
            relvol_tier = "Adequate (≥5x)"

    # ------------------------------------------------------------------
    # SHARE COUNT — 30 pts max (primary signal)
    # ------------------------------------------------------------------
    float_shares    = None
    total_cash      = None
    cash_per_share  = None
    small_float     = False
    high_cash       = False
    institution_pct = None
    float_tier      = "Unknown"

    if fundamentals:
        float_shares    = fundamentals.get("float_shares")
        total_cash      = fundamentals.get("total_cash")
        institution_pct = fundamentals.get("institution_pct")

        if shares_outstanding is not None:
            if shares_outstanding < 10_000_000:
                score += 30
                small_float = True
                float_tier  = "Ideal (<10M)"
            elif shares_outstanding < 30_000_000:
                score += 18
                small_float = True
                float_tier  = "Acceptable (<30M)"
            else:
                score += 8
                float_tier  = "Large (30M–100M)"

    # ------------------------------------------------------------------
    # DAILY PERFORMANCE — 10 pts max
    # ------------------------------------------------------------------
    sweet_spot = False
    overheated = False
    if daily_return is not None:
        sweet_spot = 0.10 <= daily_return <= 0.40
        overheated = daily_return > 1.00
        if sweet_spot:
            score += 10

    # ------------------------------------------------------------------
    # SIDEWAYS COMPRESSION — 8 pts
    # ------------------------------------------------------------------
    sideways_chop = False
    if range_10d is not None:
        sideways_chop = range_10d < 0.20
        if sideways_chop:
            score += 8

    # ------------------------------------------------------------------
    # YESTERDAY GREEN — 7 pts
    # ------------------------------------------------------------------
    yesterday_green = False
    if len(df) >= 2:
        prev_return = safe_float(df.iloc[-2].get("daily_return"))
        if prev_return is not None and prev_return > 0:
            yesterday_green = True
            score += 7

    # ------------------------------------------------------------------
    # RECENT DECLINE PENALTY
    # ------------------------------------------------------------------
    recent_decline = False
    if return_3d is not None and return_5d is not None:
        recent_decline = return_3d < -0.20 or return_5d < -0.30
        if recent_decline:
            score -= 10

    # ------------------------------------------------------------------
    # NICE-TO-HAVE BONUSES (15 pts max)
    # ------------------------------------------------------------------
    if fundamentals:
        # High Cash per Share — 5 pts
        if shares_outstanding and total_cash:
            try:
                cash_per_share = total_cash / shares_outstanding
                if cash_per_share > price:
                    high_cash = True
                    score += 5
            except Exception:
                cash_per_share = None

        # Institutional Ownership — 5 pts max
        if institution_pct is not None:
            if institution_pct >= 0.50:
                score += 5
            elif institution_pct >= 0.20:
                score += 3

    # No News Catalyst — 5 pts
    no_news_catalyst = not (fundamentals or {}).get("recent_news_present", False)
    if no_news_catalyst:
        score += 5

    # Clamp to 0–100
    score = max(0, min(100, score))

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
            "shares_outstanding": shares_outstanding,
            "float_shares": float_shares,
            "float_tier": float_tier,
            "small_float": small_float,
            "total_cash": total_cash,
            "cash_per_share": round(cash_per_share, 2) if cash_per_share else None,
            "high_cash": high_cash,
            "institution_pct": round(institution_pct * 100, 1) if institution_pct is not None else None,
            "no_news_catalyst": no_news_catalyst,
            "sweet_spot_10_40": sweet_spot,
            "over_100_percent": overheated,
            "five_day_return_pct": round(return_5d * 100, 2) if return_5d else None,
            "sideways_chop": sideways_chop,
            "yesterday_green": yesterday_green,
            "recent_decline": recent_decline,
        }
    }


# ------------------------------------------------------------------
# SQUEEZE MODE SCORER
# ------------------------------------------------------------------
#
# DEFAULT_SQUEEZE_WEIGHTS are also set to sum to 100 so the raw score
# IS the final score. When the AI optimizer adjusts weights the
# max_score formula re-normalises so output stays 0–100.
#
DEFAULT_SQUEEZE_WEIGHTS = {
    "rel_vol_50x":          30,   # primary signal
    "rel_vol_25x":          22,
    "rel_vol_10x":          15,
    "rel_vol_5x":           7,
    "daily_sweet_20_40":    10,
    "daily_ok_10_20":       5,
    "daily_ok_40_100":      5,
    "sideways_chop":        8,
    "yesterday_green":      7,
    "shares_lt10m":         30,   # primary signal
    "shares_lt30m":         18,
    "shares_lt100m":        8,
    "no_news_bonus":        5,    # nice-to-have
    "high_cash_bonus":      5,    # nice-to-have
    "institution_moderate": 3,    # nice-to-have
    "institution_strong":   5,    # nice-to-have
}
# Max with defaults: 30+10+8+7+30+5+5+5 = 100


def score_stock_squeeze(symbol: str, df, fundamentals=None, weights=None):

    if df is None or len(df) < 20:
        return None

    latest = df.iloc[-1]

    def safe_float(value):
        try:
            return float(value)
        except Exception:
            return None

    price           = safe_float(latest.get("close"))
    volume          = safe_float(latest.get("volume"))
    relative_volume = safe_float(latest.get("relative_volume"))
    daily_return    = safe_float(latest.get("daily_return"))
    range_10d       = safe_float(latest.get("range_10d"))

    if fundamentals:
        if fundamentals.get("finviz_price") is not None:
            price = fundamentals["finviz_price"]
        if fundamentals.get("finviz_volume") is not None:
            volume = float(fundamentals["finviz_volume"])
        if fundamentals.get("finviz_change_pct") is not None:
            daily_return = fundamentals["finviz_change_pct"] / 100
        if fundamentals.get("finviz_relvol") is not None:
            relative_volume = fundamentals["finviz_relvol"]

    if price is None:
        return None

    shares_outstanding = (fundamentals or {}).get("shares_outstanding")

    # ------------------------------------------------------------------
    # HARD REQUIREMENTS
    # ------------------------------------------------------------------
    if relative_volume is not None and relative_volume < 5:
        return None
    if shares_outstanding is not None and shares_outstanding >= 100_000_000:
        return None
    if daily_return is not None and daily_return > 1.00:
        return None

    w = {**DEFAULT_SQUEEZE_WEIGHTS, **(weights or {})}

    score = 0

    # ------------------------------------------------------------------
    # RELATIVE VOLUME — primary signal
    # ------------------------------------------------------------------
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
            relvol_tier = "Good (≥10x)"
        else:
            score += w.get("rel_vol_5x", 7)
            relvol_tier = "Adequate (≥5x)"

    # ------------------------------------------------------------------
    # DAILY PERFORMANCE
    # ------------------------------------------------------------------
    sweet_spot_squeeze = False
    if daily_return is not None:
        sweet_spot_squeeze = 0.20 <= daily_return <= 0.40
        if sweet_spot_squeeze:
            score += w["daily_sweet_20_40"]
        elif 0.10 <= daily_return < 0.20:
            score += w["daily_ok_10_20"]
        elif 0.40 < daily_return <= 1.00:
            score += w["daily_ok_40_100"]

    # ------------------------------------------------------------------
    # SIDEWAYS CONSOLIDATION
    # ------------------------------------------------------------------
    sideways_chop = False
    if range_10d is not None:
        sideways_chop = range_10d < 0.20
        if sideways_chop:
            score += w["sideways_chop"]

    # Yesterday Green
    yesterday_green = False
    if len(df) >= 2:
        prev_return = safe_float(df.iloc[-2].get("daily_return"))
        if prev_return is not None and prev_return > 0:
            yesterday_green = True
            score += w["yesterday_green"]

    # ------------------------------------------------------------------
    # SHARE COUNT — primary signal
    # ------------------------------------------------------------------
    float_shares    = None
    float_tier      = "Unknown"
    institution_pct = None
    total_cash      = None
    cash_per_share  = None
    high_cash       = False

    if fundamentals:
        float_shares    = fundamentals.get("float_shares")
        institution_pct = fundamentals.get("institution_pct")
        total_cash      = fundamentals.get("total_cash")

        if shares_outstanding is not None:
            if shares_outstanding < 10_000_000:
                score += w["shares_lt10m"]
                float_tier = "Ideal (<10M)"
            elif shares_outstanding < 30_000_000:
                score += w["shares_lt30m"]
                float_tier = "Acceptable (<30M)"
            else:
                score += w.get("shares_lt100m", 8)
                float_tier = "Large (30M–100M)"

        # High Cash per Share — nice to have
        if shares_outstanding and total_cash and price:
            try:
                cash_per_share = total_cash / shares_outstanding
                if cash_per_share > price:
                    high_cash = True
                    score += w.get("high_cash_bonus", 5)
            except Exception:
                cash_per_share = None

        # Institutional Ownership — nice to have
        if institution_pct is not None:
            if institution_pct >= 0.50:
                score += w.get("institution_strong", 5)
            elif institution_pct >= 0.20:
                score += w.get("institution_moderate", 3)

    # No News Catalyst — nice to have
    no_news_catalyst = not (fundamentals or {}).get("recent_news_present", False)
    if no_news_catalyst:
        score += w.get("no_news_bonus", 5)

    # ------------------------------------------------------------------
    # NORMALISE TO 0–100
    # With default weights max_score = 100, so no change.
    # When AI adjusts weights, max_score re-normalises automatically.
    # ------------------------------------------------------------------
    max_score = (
        max(w["rel_vol_50x"], w["rel_vol_25x"], w["rel_vol_10x"],
            w.get("rel_vol_5x", 7), 0) +
        max(w["daily_sweet_20_40"], w["daily_ok_10_20"],
            w["daily_ok_40_100"], 0) +
        w["sideways_chop"] +
        w["yesterday_green"] +
        max(w["shares_lt10m"], w["shares_lt30m"],
            w.get("shares_lt100m", 8), 0) +
        w.get("high_cash_bonus", 5) +
        max(w.get("institution_strong", 5),
            w.get("institution_moderate", 3), 0) +
        w.get("no_news_bonus", 5)
    )
    if max_score <= 0:
        max_score = 100

    score = max(0, min(100, round(score / max_score * 100)))

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
            "shares_outstanding": shares_outstanding,
            "float_shares": float_shares,
            "float_tier": float_tier,
            "small_float": shares_outstanding < 10_000_000 if shares_outstanding else False,
            "total_cash": total_cash,
            "cash_per_share": round(cash_per_share, 2) if cash_per_share else None,
            "high_cash": high_cash,
            "sweet_spot_squeeze": sweet_spot_squeeze,
            "daily_return_pct": round(daily_return * 100, 2) if daily_return else None,
            "sideways_chop": sideways_chop,
            "yesterday_green": yesterday_green,
            "institution_pct": round(institution_pct * 100, 1) if institution_pct is not None else None,
            "no_news_catalyst": no_news_catalyst,
            # template compatibility aliases
            "high_cash": high_cash,
            "cash_per_share": round(cash_per_share, 2) if cash_per_share else None,
            "five_day_return_pct": None,
            "recent_decline": False,
            "over_100_percent": False,
            "sweet_spot_10_40": sweet_spot_squeeze,
        }
    }
