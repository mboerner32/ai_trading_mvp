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

    sector   = None
    industry = None
    if fundamentals:
        float_shares    = fundamentals.get("float_shares")
        total_cash      = fundamentals.get("total_cash")
        institution_pct = fundamentals.get("institution_pct")
        sector          = fundamentals.get("sector")
        industry        = fundamentals.get("industry")

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
    # CANDLESTICK PATTERN — 8 pts (mutually exclusive archetypes)
    # sideways_chop:        avg absolute daily return last 5d < 10% → quiet consolidation before ignition
    # momentum_continuation: avg absolute daily return last 5d >= 10% → already in motion, catching next leg
    # ------------------------------------------------------------------
    sideways_chop = False
    momentum_continuation = False
    if len(df) >= 5:
        last5_returns = [safe_float(df.iloc[i].get("daily_return")) for i in range(-5, 0)]
        last5_returns = [r for r in last5_returns if r is not None]
        if len(last5_returns) >= 3:
            avg_abs_5d = sum(abs(r) for r in last5_returns) / len(last5_returns)
            if avg_abs_5d < 0.10:
                sideways_chop = True
                score += 8
            else:
                momentum_continuation = True
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
        # 40%+ is the benchmark: institutions hold the floor, lower risk
        if institution_pct is not None:
            if institution_pct >= 0.40:
                score += 5
            elif institution_pct >= 0.15:
                score += 2

        # Sector bonus — Biotech/Healthcare historically outperforms for this strategy
        if sector == "Healthcare" or (industry and "biotech" in industry.lower()):
            score += 5

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
            "sector": sector,
            "industry": industry,
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
    "rel_vol_500x":         45,   # extreme event (≥500x) — 66.7% hit rate (+35.9pp, n=135, 2026-03-14)
    "rel_vol_100x":         40,   # very high (100–499x) — 59.9% hit rate (+29.1pp, n=364, 2026-03-14)
    "rel_vol_50x":          8,    # 50–99x — 45.5% hit rate (-4.5pp, n=297, 2026-03-17 clean) — REDUCED
    "rel_vol_25x":          14,   # 25–49x — 49.5% hit rate (-0.4pp, n=687, 2026-03-17 clean)
    "rel_vol_10x":          8,    # 10–24x — 48.2% hit rate (-1.7pp, n=1374, 2026-03-17 clean)
    "rel_vol_5x":           3,
    "daily_sweet_20_40":    3,    # 20–40% — 49.4% hit rate (-0.5pp, n=952, 2026-03-17 clean) — at baseline
    "daily_ok_10_20":      -5,   # PENALTY: 44.0% hit rate (-5.9pp, n=754, 2026-03-17 clean)
    "daily_ok_40_100":      10,   # 40–100% — 56.7% hit rate (+6.8pp, n=723, 2026-03-17 clean) — BOOSTED
    "sideways_chop":          8,  # chop-to-ignition: avg_abs_5d < 10% (57% of winners, validated 2026-03-16)
    "momentum_continuation":  6,  # already running: avg_abs_5d >= 10% (43% of winners, validated 2026-03-16)
    "yesterday_green":        9,  # boosted: 14.4% hit rate (n=263, 2026-03-09)
    "shares_lt10m":         8,    # <10M — 48.9% hit rate (-1.0pp, n=427, 2026-03-17 clean) — REDUCED
    "shares_lt30m":         28,   # 10–30M — 53.8% hit rate (+3.9pp, n=515, 2026-03-17 clean) — PRIMARY
    "shares_lt100m":        0,    # 30–100M — 48.3% hit rate (-1.7pp, n=717, 2026-03-17 clean) — ZEROED
    "no_news_bonus":        5,    # nice-to-have
    "high_cash_bonus":      5,    # nice-to-have
    "institution_moderate": 2,    # 15–39% institutional ownership
    "institution_strong":   0,    # ZEROED: 13.6% hit rate (-17.2pp, n=103, 2026-03-14) — actively harmful
    "sector_biotech_bonus": 5,    # Healthcare/Biotech historically outperforms
    # Optional criteria — disabled (0) by default, Auto AI can enable
    # NOTE: first_hour_vol_20m is tracked in fired_signals for hypothesis testing
    # but NOT applied as a weight until AI validates it from backtesting evidence
    "rsi_momentum_bonus":     0,  # RSI 50–70: healthy uptrend, not overbought
    "macd_positive_bonus":    0,  # Positive MACD: bullish momentum crossover
    "bb_upper_breakout":      0,  # Bollinger %B > 0.85: breaking above upper band
    "consecutive_green_bonus": 0, # 2+ consecutive green days: sustained buying
    "low_float_ratio_bonus":  0,  # Float < 40% of shares: tighter float, faster moves
}
# Max with defaults: rv50x=30+sweet=10+chop=8+yg=9+lt30m=28+no_news=5+high_cash=5+inst_strong=5+biotech=5 = 105 → normalised to 100
# Optional criteria default to 0 so they don't affect the denominator until enabled


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
        if relative_volume >= 500:
            score += w.get("rel_vol_500x", 40)
            relvol_tier = "Extreme (≥500x)"
        elif relative_volume >= 100:
            score += w.get("rel_vol_100x", 35)
            relvol_tier = "Exceptional (≥100x)"
        elif relative_volume >= 50:
            score += w["rel_vol_50x"]
            relvol_tier = "Very High (≥50x)"
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
    # CANDLESTICK PATTERN — mutually exclusive archetypes
    # sideways_chop:         avg abs daily return last 5d < 10% → chop-to-ignition setup
    # momentum_continuation: avg abs daily return last 5d >= 10% → already running, next leg
    # Validated on 91/158 winners (chop) and 67/158 winners (momentum) — both archetypes score points
    # ------------------------------------------------------------------
    sideways_chop = False
    momentum_continuation = False
    if len(df) >= 5:
        last5_returns = [safe_float(df.iloc[i].get("daily_return")) for i in range(-5, 0)]
        last5_returns = [r for r in last5_returns if r is not None]
        if len(last5_returns) >= 3:
            avg_abs_5d = sum(abs(r) for r in last5_returns) / len(last5_returns)
            if avg_abs_5d < 0.10:
                sideways_chop = True
                score += w["sideways_chop"]
            else:
                momentum_continuation = True
                score += w.get("momentum_continuation", 6)

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

    sector   = None
    industry = None
    if fundamentals:
        float_shares    = fundamentals.get("float_shares")
        institution_pct = fundamentals.get("institution_pct")
        total_cash      = fundamentals.get("total_cash")
        sector          = fundamentals.get("sector")
        industry        = fundamentals.get("industry")

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

        # Institutional Ownership — 40%+ is the benchmark (holds the floor)
        if institution_pct is not None:
            if institution_pct >= 0.40:
                score += w.get("institution_strong", 5)
            elif institution_pct >= 0.15:
                score += w.get("institution_moderate", 2)

        # Sector bonus — Biotech/Healthcare historically outperforms for this strategy
        if sector == "Healthcare" or (industry and "biotech" in industry.lower()):
            score += w.get("sector_biotech_bonus", 5)

    # No News Catalyst — nice to have
    no_news_catalyst = not (fundamentals or {}).get("recent_news_present", False)
    if no_news_catalyst:
        score += w.get("no_news_bonus", 5)

    # First-hour volume: data collected for hypothesis testing, not scored yet
    first_hour_vol     = (fundamentals or {}).get("first_hour_vol")
    first_hour_vol_20m = first_hour_vol is not None and first_hour_vol >= 20_000_000
    # (score not applied — AI must validate this hypothesis first)

    # ------------------------------------------------------------------
    # OPTIONAL CRITERIA — disabled by default (weight=0), Auto AI can enable
    # ------------------------------------------------------------------
    if w.get("rsi_momentum_bonus", 0) > 0:
        rsi_val = safe_float(latest.get("rsi_14"))
        if rsi_val is not None and 50 < rsi_val <= 70:
            score += w["rsi_momentum_bonus"]

    if w.get("macd_positive_bonus", 0) > 0:
        macd_val = safe_float(latest.get("macd_norm"))
        if macd_val is not None and macd_val > 0:
            score += w["macd_positive_bonus"]

    if w.get("bb_upper_breakout", 0) > 0:
        bb_pct_val = safe_float(latest.get("bb_pct"))
        if bb_pct_val is not None and bb_pct_val > 0.85:
            score += w["bb_upper_breakout"]

    if w.get("consecutive_green_bonus", 0) > 0 and len(df) >= 2:
        today_ret = safe_float(latest.get("daily_return"))
        prev_ret  = safe_float(df.iloc[-2].get("daily_return"))
        if today_ret is not None and prev_ret is not None and today_ret > 0 and prev_ret > 0:
            score += w["consecutive_green_bonus"]

    if w.get("low_float_ratio_bonus", 0) > 0 and fundamentals:
        f_shares = fundamentals.get("float_shares")
        s_out    = fundamentals.get("shares_outstanding")
        if f_shares and s_out and s_out > 0 and (f_shares / s_out) < 0.40:
            score += w["low_float_ratio_bonus"]

    # ------------------------------------------------------------------
    # FIRED SIGNALS — record which criteria contributed points
    # Only keys that fired (True) are stored; omitted = did not fire.
    # Stored as signals_json in the scans table for per-signal backtesting.
    # ------------------------------------------------------------------
    fired_signals = {}

    if relative_volume is not None:
        if relative_volume >= 500:    fired_signals["rel_vol_500x"]     = True
        elif relative_volume >= 100:  fired_signals["rel_vol_100x"]     = True
        elif relative_volume >= 50:   fired_signals["rel_vol_50x"]      = True
        elif relative_volume >= 25:   fired_signals["rel_vol_25x"]      = True
        elif relative_volume >= 10:   fired_signals["rel_vol_10x"]      = True
        else:                         fired_signals["rel_vol_5x"]       = True

    if daily_return is not None:
        if 0.20 <= daily_return <= 0.40:    fired_signals["daily_sweet_20_40"] = True
        elif 0.10 <= daily_return < 0.20:   fired_signals["daily_ok_10_20"]    = True
        elif 0.40 < daily_return <= 1.00:   fired_signals["daily_ok_40_100"]   = True

    if fundamentals and shares_outstanding is not None:
        if shares_outstanding < 10_000_000:   fired_signals["shares_lt10m"]  = True
        elif shares_outstanding < 30_000_000: fired_signals["shares_lt30m"]  = True
        else:                                 fired_signals["shares_lt100m"] = True

    if sideways_chop:           fired_signals["sideways_chop"]          = True
    if momentum_continuation:   fired_signals["momentum_continuation"]  = True
    if yesterday_green:         fired_signals["yesterday_green"]        = True
    if no_news_catalyst: fired_signals["no_news_bonus"]    = True
    if high_cash:        fired_signals["high_cash_bonus"]  = True

    if institution_pct is not None:
        if institution_pct >= 0.40:   fired_signals["institution_strong"]   = True
        elif institution_pct >= 0.15: fired_signals["institution_moderate"] = True

    if sector == "Healthcare" or (industry and "biotech" in industry.lower()):
        fired_signals["sector_biotech_bonus"] = True

    if w.get("rsi_momentum_bonus", 0) > 0:
        rsi_v = safe_float(latest.get("rsi_14"))
        if rsi_v is not None and 50 < rsi_v <= 70:
            fired_signals["rsi_momentum_bonus"] = True

    if w.get("macd_positive_bonus", 0) > 0:
        macd_v = safe_float(latest.get("macd_norm"))
        if macd_v is not None and macd_v > 0:
            fired_signals["macd_positive_bonus"] = True

    if w.get("bb_upper_breakout", 0) > 0:
        bb_v = safe_float(latest.get("bb_pct"))
        if bb_v is not None and bb_v > 0.85:
            fired_signals["bb_upper_breakout"] = True

    if w.get("consecutive_green_bonus", 0) > 0 and len(df) >= 2:
        t_r = safe_float(latest.get("daily_return"))
        p_r = safe_float(df.iloc[-2].get("daily_return"))
        if t_r and p_r and t_r > 0 and p_r > 0:
            fired_signals["consecutive_green_bonus"] = True

    if w.get("low_float_ratio_bonus", 0) > 0 and fundamentals:
        fs = fundamentals.get("float_shares")
        so = shares_outstanding
        if fs and so and so > 0 and (fs / so) < 0.40:
            fired_signals["low_float_ratio_bonus"] = True

    if first_hour_vol_20m:
        fired_signals["first_hour_vol_20m"] = True

    # ------------------------------------------------------------------
    # NORMALISE TO 0–100
    # With default weights max_score = 100, so no change.
    # When AI adjusts weights, max_score re-normalises automatically.
    # ------------------------------------------------------------------
    max_score = (
        max(w.get("rel_vol_500x", 40), w.get("rel_vol_100x", 35),
            w["rel_vol_50x"], w["rel_vol_25x"], w["rel_vol_10x"],
            w.get("rel_vol_5x", 7), 0) +
        max(w["daily_sweet_20_40"], w["daily_ok_10_20"],
            w["daily_ok_40_100"], 0) +
        max(w["sideways_chop"], w.get("momentum_continuation", 6)) +
        w["yesterday_green"] +
        max(w["shares_lt10m"], w["shares_lt30m"],
            w.get("shares_lt100m", 8), 0) +
        w.get("high_cash_bonus", 5) +
        max(w.get("institution_strong", 5),
            w.get("institution_moderate", 2), 0) +
        w.get("sector_biotech_bonus", 5) +
        w.get("no_news_bonus", 5) +
        # Optional criteria (contribute 0 when disabled)
        w.get("rsi_momentum_bonus", 0) +
        w.get("macd_positive_bonus", 0) +
        w.get("bb_upper_breakout", 0) +
        w.get("consecutive_green_bonus", 0) +
        w.get("low_float_ratio_bonus", 0)
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
            "momentum_continuation": momentum_continuation,
            "yesterday_green": yesterday_green,
            "institution_pct": round(institution_pct * 100, 1) if institution_pct is not None else None,
            "sector": sector,
            "industry": industry,
            "no_news_catalyst": no_news_catalyst,
            "first_hour_vol": first_hour_vol,
            "first_hour_vol_20m": first_hour_vol_20m,
            # template compatibility aliases
            "high_cash": high_cash,
            "cash_per_share": round(cash_per_share, 2) if cash_per_share else None,
            "five_day_return_pct": None,
            "recent_decline": False,
            "over_100_percent": False,
            "sweet_spot_10_40": sweet_spot_squeeze,
            "fired_signals": fired_signals,
        }
    }
