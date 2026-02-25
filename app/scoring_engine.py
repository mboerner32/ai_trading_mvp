# ai_trading_mvp/app/scoring_engine.py


def score_stock(symbol: str, df, fundamentals=None):

    # --------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------

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
        # FinViz change_pct is a percentage (12.34); convert to ratio (0.1234)
        if fundamentals.get("finviz_change_pct") is not None:
            daily_return = fundamentals["finviz_change_pct"] / 100
        if fundamentals.get("finviz_relvol") is not None:
            relative_volume = fundamentals["finviz_relvol"]

    if price is None:
        return None

    # Extract shares early — needed for hard filter below
    shares_outstanding = (fundamentals or {}).get("shares_outstanding")

    # ==================================================================
    # HARD REQUIREMENTS — stocks failing these are excluded entirely
    # ==================================================================

    # Must have meaningful relative volume (≥5x)
    if relative_volume is not None and relative_volume < 5:
        return None

    # Must have a small share count (<100M) when data is available
    if shares_outstanding is not None and shares_outstanding >= 100_000_000:
        return None

    score = 0

    # ==================================================================
    # RELATIVE VOLUME — primary signal, highest weight
    # ==================================================================

    relvol_tier = "Below Threshold"
    if relative_volume is not None:
        if relative_volume >= 50:
            score += 4
            relvol_tier = "Exceptional (≥50x)"
        elif relative_volume >= 25:
            score += 3
            relvol_tier = "Ideal (≥25x)"
        elif relative_volume >= 10:
            score += 2
            relvol_tier = "Good (≥10x)"
        else:  # 5–10
            score += 1
            relvol_tier = "Adequate (≥5x)"

    # ==================================================================
    # DAILY PERFORMANCE
    # ==================================================================

    sweet_spot = False
    overheated = False
    if daily_return is not None:
        sweet_spot = 0.10 <= daily_return <= 0.40
        overheated = daily_return > 1.00
        if sweet_spot:
            score += 2

    # ==================================================================
    # SIDEWAYS COMPRESSION — pre-breakout consolidation
    # ==================================================================

    sideways_chop = False
    if range_10d is not None:
        sideways_chop = range_10d < 0.20
        if sideways_chop:
            score += 2

    # Yesterday Green
    yesterday_green = False
    if len(df) >= 2:
        prev_return = safe_float(df.iloc[-2].get("daily_return"))
        if prev_return is not None and prev_return > 0:
            yesterday_green = True
            score += 1

    # Recent Selloff Penalty
    recent_decline = False
    if return_3d is not None and return_5d is not None:
        recent_decline = return_3d < -0.20 or return_5d < -0.30
        if recent_decline:
            score -= 2

    # ==================================================================
    # FUNDAMENTALS
    # ==================================================================

    float_shares   = None
    total_cash     = None
    cash_per_share = None
    small_float    = False
    high_cash      = False
    institution_pct = None
    float_tier     = "Unknown"

    if fundamentals:
        float_shares    = fundamentals.get("float_shares")
        total_cash      = fundamentals.get("total_cash")
        institution_pct = fundamentals.get("institution_pct")

        # Share count tiers — primary signal, high weight
        if shares_outstanding is not None:
            if shares_outstanding < 10_000_000:
                score += 4
                small_float = True
                float_tier  = "Ideal (<10M)"
            elif shares_outstanding < 30_000_000:
                score += 2
                small_float = True
                float_tier  = "Acceptable (<30M)"
            else:
                score += 1
                float_tier  = "Large (30M–100M)"

        # High Cash per Share — nice to have
        if shares_outstanding and total_cash:
            try:
                cash_per_share = total_cash / shares_outstanding
                if cash_per_share > price:
                    high_cash = True
                    score += 1
            except Exception:
                cash_per_share = None

        # Institutional Ownership — nice to have
        if institution_pct is not None:
            if institution_pct >= 0.50:
                score += 2
            elif institution_pct >= 0.20:
                score += 1

    # No News Catalyst — nice to have
    no_news_catalyst = not (fundamentals or {}).get("recent_news_present", False)
    if no_news_catalyst:
        score += 1

    # ==================================================================
    # SCALE TO 0–100
    # Max raw: 4 (relvol) + 2 (sweet spot) + 2 (sideways) + 1 (yesterday)
    #        + 4 (shares <10M) + 1 (cash) + 2 (institution) + 1 (no news) = 17
    # ==================================================================

    score = max(0, round((score / 17) * 100))

    # --------------------------------------------------
    # RECOMMENDATION TIERS
    # --------------------------------------------------

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
    "rel_vol_50x":          4,   # relative volume >= 50x (primary signal)
    "rel_vol_25x":          3,   # relative volume >= 25x (primary signal)
    "rel_vol_10x":          2,   # relative volume >= 10x
    "rel_vol_5x":           1,   # relative volume >= 5x (minimum)
    "daily_sweet_20_40":    2,   # daily gain 20–40% (sweet spot)
    "daily_ok_10_20":       1,   # daily gain 10–20%
    "daily_ok_40_100":      1,   # daily gain 40–100%
    "sideways_chop":        2,   # 10-day range < 20% (consolidation)
    "yesterday_green":      1,   # previous day positive
    "shares_lt10m":         4,   # shares outstanding < 10M (primary signal)
    "shares_lt30m":         2,   # shares outstanding 10M–30M
    "shares_lt100m":        1,   # shares outstanding 30M–100M
    "no_news_bonus":        1,   # organic move — nice to have
    "high_cash_bonus":      1,   # cash per share > price — nice to have
    "institution_moderate": 1,   # institutional ownership 20–50% — nice to have
    "institution_strong":   2,   # institutional ownership >= 50% — nice to have
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

    price           = safe_float(latest.get("close"))
    volume          = safe_float(latest.get("volume"))
    relative_volume = safe_float(latest.get("relative_volume"))
    daily_return    = safe_float(latest.get("daily_return"))
    range_10d       = safe_float(latest.get("range_10d"))

    # Prefer live FinViz values over yfinance-calculated ones where available
    if fundamentals:
        if fundamentals.get("finviz_price") is not None:
            price = fundamentals["finviz_price"]
        if fundamentals.get("finviz_volume") is not None:
            volume = float(fundamentals["finviz_volume"])
        # FinViz change_pct is a percentage (12.34); convert to ratio (0.1234)
        if fundamentals.get("finviz_change_pct") is not None:
            daily_return = fundamentals["finviz_change_pct"] / 100
        if fundamentals.get("finviz_relvol") is not None:
            relative_volume = fundamentals["finviz_relvol"]

    if price is None:
        return None

    # Extract shares early — needed for hard filter
    shares_outstanding = (fundamentals or {}).get("shares_outstanding")

    # ==================================================================
    # HARD REQUIREMENTS — stocks failing these are excluded entirely
    # ==================================================================

    # Must have meaningful relative volume (≥5x)
    if relative_volume is not None and relative_volume < 5:
        return None

    # Must have a small share count (<100M) when data is available
    if shares_outstanding is not None and shares_outstanding >= 100_000_000:
        return None

    # Blow-off phase exclusion (>100% gain today — already extended)
    if daily_return is not None and daily_return > 1.00:
        return None

    # Merge provided weights with defaults so any missing key falls back safely
    w = {**DEFAULT_SQUEEZE_WEIGHTS, **(weights or {})}

    score = 0

    # ==================================================================
    # RELATIVE VOLUME — primary signal, highest weight
    # ==================================================================

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
        else:  # 5–10
            score += w.get("rel_vol_5x", 1)
            relvol_tier = "Adequate (≥5x)"

    # ==================================================================
    # DAILY PERFORMANCE
    # ==================================================================

    sweet_spot_squeeze = False
    if daily_return is not None:
        sweet_spot_squeeze = 0.20 <= daily_return <= 0.40
        if sweet_spot_squeeze:
            score += w["daily_sweet_20_40"]
        elif 0.10 <= daily_return < 0.20:
            score += w["daily_ok_10_20"]
        elif 0.40 < daily_return <= 1.00:
            score += w["daily_ok_40_100"]

    # ==================================================================
    # SIDEWAYS CONSOLIDATION / BARCODING
    # ==================================================================

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

    # ==================================================================
    # SHARE COUNT — primary signal, high weight
    # ==================================================================

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
                score += w.get("shares_lt100m", 1)
                float_tier = "Large (30M–100M)"

        # High Cash per Share — nice to have
        if shares_outstanding and total_cash and price:
            try:
                cash_per_share = total_cash / shares_outstanding
                if cash_per_share > price:
                    high_cash = True
                    score += w.get("high_cash_bonus", 1)
            except Exception:
                cash_per_share = None

        # Institutional Ownership — nice to have
        if institution_pct is not None:
            if institution_pct >= 0.50:
                score += w.get("institution_strong", 2)
            elif institution_pct >= 0.20:
                score += w.get("institution_moderate", 1)

    # No News Catalyst — nice to have
    no_news_catalyst = not (fundamentals or {}).get("recent_news_present", False)
    if no_news_catalyst:
        score += w.get("no_news_bonus", 0)

    # ==================================================================
    # SCALE TO 0–100
    # Max raw (default weights):
    #   4 (relvol) + 2 (daily) + 2 (sideways) + 1 (yesterday)
    #   + 4 (shares) + 1 (cash) + 2 (institution) + 1 (no news) = 17
    # ==================================================================

    max_score = (
        max(w["rel_vol_50x"], w["rel_vol_25x"], w["rel_vol_10x"],
            w.get("rel_vol_5x", 1), 0) +
        max(w["daily_sweet_20_40"], w["daily_ok_10_20"],
            w["daily_ok_40_100"], 0) +
        w["sideways_chop"] +
        w["yesterday_green"] +
        max(w["shares_lt10m"], w["shares_lt30m"],
            w.get("shares_lt100m", 1), 0) +
        w.get("high_cash_bonus", 1) +
        max(w.get("institution_strong", 2),
            w.get("institution_moderate", 1), 0) +
        w.get("no_news_bonus", 0)
    )
    if max_score <= 0:
        max_score = 17

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

            # Cash
            "total_cash": total_cash,
            "cash_per_share": round(cash_per_share, 2) if cash_per_share else None,
            "high_cash": high_cash,

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
            "high_cash": high_cash,
            "cash_per_share": round(cash_per_share, 2) if cash_per_share else None,
            "five_day_return_pct": None,
            "recent_decline": False,
            "over_100_percent": False,
            "sweet_spot_10_40": sweet_spot_squeeze,
        }
    }
