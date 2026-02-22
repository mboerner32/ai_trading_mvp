"""
Historical backfill: applies squeeze scoring to past OHLCV data
to generate labeled training examples (qualifying scan day → next-day return).
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yfinance as yf


class _TimeoutSession(requests.Session):
    """Requests session that enforces a default timeout on every request."""
    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", 15)
        return super().request(method, url, **kwargs)

_SESSION = _TimeoutSession()

from app.database import DB_NAME
from app.scanner import prepare_dataframe
from app.scoring_engine import DEFAULT_SQUEEZE_WEIGHTS

# Known historical low-float / momentum tickers (2022–2024)
SEED_TICKERS = [
    "BBBY", "NAKD", "SNDL", "CTRM", "IDEX", "KOSS", "EXPR", "ATER",
    "SPRT", "PROG", "BBIG", "WISH", "GOEV", "XELA", "RIDE", "NKLA",
    "WKHS", "OCGN", "CLOV", "MVIS", "BLNK", "FCEL", "GNUS", "IMPP",
    "MULN", "FFIE", "HPNN", "MINE", "MARK", "PHUN", "DPRO", "AUVI",
    "VERB", "ILUS", "PIXY", "INPX", "MOBQ", "WRAP", "WISA", "SHOT",
    "TLRY", "HEXO", "BNGO", "DARE", "AVXL", "AYTU", "BCRX", "NEPT",
    "SESN", "GSAT", "SONN", "KULR", "AULT", "EFSH", "UPXI", "TRKA",
    "VISL", "AMPE", "BPTH", "ATXI", "LODE", "HMBL", "MNMD", "KERN",
    "IZEA", "COCP", "GTII", "ADXS", "ATOS", "MEGL", "BFRI", "CFRX",
    "DPLO", "BBAI", "ZEST", "LIDR", "LTRY", "RETO", "TPST", "GERN",
    "OCEA", "NOVN", "NRXP", "ISIG", "MTTR", "TBIO", "EBON", "DCFC",
    "DGLY", "HYMC", "IPIX", "GRPN", "FGEN", "LUMO", "SNPX", "RKDA",
    "XCUR", "SYTA", "NURO", "VVPR", "VLDR", "HYLN", "SOLO", "LKFN",
]


def _safe(val):
    try:
        v = float(val)
        return None if v != v else v  # NaN → None
    except Exception:
        return None


def _score(relative_volume, daily_return, range_10d,
           yesterday_green, shares_outstanding, weights):
    """Score one historical candidate using squeeze logic (no live fundamentals)."""
    w = {**DEFAULT_SQUEEZE_WEIGHTS, **(weights or {})}
    score = 0

    if relative_volume is not None:
        if relative_volume >= 50:
            score += w["rel_vol_50x"]
        elif relative_volume >= 25:
            score += w["rel_vol_25x"]
        elif relative_volume >= 10:
            score += w["rel_vol_10x"]

    if daily_return is not None:
        if 0.20 <= daily_return <= 0.40:
            score += w["daily_sweet_20_40"]
        elif 0.10 <= daily_return < 0.20:
            score += w["daily_ok_10_20"]
        elif 0.40 < daily_return <= 1.00:
            score += w["daily_ok_40_100"]

    if range_10d is not None and range_10d < 0.20:
        score += w["sideways_chop"]

    if yesterday_green:
        score += w["yesterday_green"]

    if shares_outstanding is not None:
        if shares_outstanding < 10_000_000:
            score += w["shares_lt10m"]
        elif shares_outstanding < 30_000_000:
            score += w["shares_lt30m"]
        elif shares_outstanding >= 100_000_000:
            score -= w["shares_gte100m_penalty"]

    # Treat historical moves as organic (no news signal available)
    score += w.get("no_news_bonus", 0)

    max_score = (
        max(w["rel_vol_50x"], w["rel_vol_25x"], w["rel_vol_10x"], 0)
        + max(w["daily_sweet_20_40"], w["daily_ok_10_20"], w["daily_ok_40_100"], 0)
        + w["sideways_chop"] + w["yesterday_green"]
        + max(w["shares_lt10m"], w["shares_lt30m"], 0)
        + w.get("no_news_bonus", 0)
    )
    max_score = max(max_score, 1)
    score = max(0, round((score / max_score) * 100))
    rec = "TRADE" if score >= 75 else ("WATCH" if score >= 50 else "SPECULATIVE")
    return score, rec


def _process_ticker(symbol, weights=None):
    """
    Download 2 years of OHLCV for `symbol`.
    Slide a window to find days matching live scan criteria:
        price < $5, daily gain 10–100%, relative volume ≥ 10.
    Returns a list of labeled example dicts with next_day_return filled in.
    """
    try:
        df = yf.download(
            symbol, period="2y", interval="1d",
            progress=False, auto_adjust=False,
            session=_SESSION
        )
        if df.empty or len(df) < 70:
            return []

        df = prepare_dataframe(df)

        # Fetch current shares outstanding (best available proxy for historical)
        shares_outstanding = None
        try:
            shares_outstanding = yf.Ticker(symbol, session=_SESSION).info.get("sharesOutstanding")
        except Exception:
            pass

        examples = []
        n = len(df)

        for i in range(70, n - 1):
            row = df.iloc[i]

            price           = _safe(row.get("close"))
            daily_return    = _safe(row.get("daily_return"))
            relative_volume = _safe(row.get("relative_volume"))
            range_10d       = _safe(row.get("range_10d"))

            # Hard screen criteria (mirror live scanner)
            if price is None or price <= 0 or price >= 5.0:
                continue
            if daily_return is None or daily_return < 0.10 or daily_return > 1.0:
                continue
            if relative_volume is None or relative_volume < 10:
                continue

            yesterday_green = False
            if i > 0:
                prev_ret = _safe(df.iloc[i - 1].get("daily_return"))
                yesterday_green = prev_ret is not None and prev_ret > 0

            score, rec = _score(
                relative_volume, daily_return, range_10d,
                yesterday_green, shares_outstanding, weights
            )

            close_today = price
            close_next  = _safe(df.iloc[i + 1].get("close"))
            if close_next is None or close_next <= 0:
                continue

            next_day_return = round((close_next - close_today) / close_today * 100, 2)

            three_day_return = None
            if i + 3 < n:
                c3 = _safe(df.iloc[i + 3].get("close"))
                if c3 and c3 > 0:
                    three_day_return = round((c3 - close_today) / close_today * 100, 2)

            scan_date = df.index[i]
            timestamp = (
                scan_date.isoformat()
                if hasattr(scan_date, "isoformat")
                else str(scan_date)
            )

            examples.append({
                "timestamp":          timestamp,
                "symbol":             symbol,
                "score":              score,
                "recommendation":     rec,
                "relative_volume":    round(relative_volume, 2),
                "today_return":       round(daily_return * 100, 2),
                "shares_outstanding": shares_outstanding,
                "next_day_return":    next_day_return,
                "three_day_return":   three_day_return,
            })

        return examples

    except Exception as e:
        print(f"Backfill: error on {symbol} — {e}")
        return []


def _get_db_tickers():
    """Return unique tickers already scanned (excluding historical mode)."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT symbol FROM scans "
            "WHERE mode != 'historical' ORDER BY id DESC LIMIT 200"
        )
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def build_historical_dataset(max_workers=6, weights=None):
    """
    Process all seed + previously seen tickers in parallel.
    Saves qualifying labeled examples into the scans table (mode='historical').
    Returns count of examples saved.
    """
    from app.database import save_historical_scans, set_backfill_status

    db_tickers = _get_db_tickers()

    # Merge seed + known tickers, deduplicated, seed list first
    seen = set()
    all_tickers = []
    for t in SEED_TICKERS + db_tickers:
        if t not in seen:
            seen.add(t)
            all_tickers.append(t)

    total = len(all_tickers)
    set_backfill_status("running", 0, total, 0)
    print(f"Backfill: starting — {total} tickers to process")

    all_examples = []
    processed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_ticker, sym, weights): sym
            for sym in all_tickers
        }
        for future in as_completed(futures):
            processed += 1
            try:
                result = future.result(timeout=30)
            except Exception:
                result = []
            if result:
                all_examples.extend(result)
            if processed % 10 == 0 or processed == total:
                set_backfill_status("running", processed, total, len(all_examples))

    saved = save_historical_scans(all_examples)
    set_backfill_status("complete", processed, total, saved)
    print(f"Backfill: complete — {saved} examples from {processed} tickers")
    return saved
