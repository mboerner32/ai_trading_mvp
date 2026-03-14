"""
Historical backfill: applies squeeze scoring to past OHLCV data
to generate labeled training examples (qualifying scan day → next-day return).
"""

import gc
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

# yfinance >=0.2.54 manages its own curl_cffi session internally.
# Do NOT pass a custom session — it will be rejected if it's not curl_cffi.

from app.database import DB_NAME
from app.scanner import prepare_dataframe
from app.scoring_engine import DEFAULT_SQUEEZE_WEIGHTS

# Known historical low-float / momentum tickers
# Updated 2025: removed confirmed-delisted/bankrupt symbols, added 2023–2025 active plays
SEED_TICKERS = [
    # --- Momentum era 2020–2022 (still useful historical data) ---
    "SNDL", "CTRM", "IDEX", "KOSS", "ATER",
    "PROG", "XELA", "WKHS", "OCGN", "CLOV",
    "MVIS", "BLNK", "FCEL", "GNUS", "IMPP",
    "MULN", "FFIE", "MARK", "PHUN", "DPRO",
    "AUVI", "VERB", "ILUS", "PIXY", "MOBQ",
    "WRAP", "WISA", "SHOT", "TLRY", "BNGO",
    "DARE", "AVXL", "AYTU", "BCRX", "NEPT",
    "SESN", "GSAT", "SONN", "KULR", "AULT",
    "EFSH", "UPXI", "TRKA", "VISL", "AMPE",
    "BPTH", "ATXI", "LODE", "MNMD", "KERN",
    "IZEA", "COCP", "ADXS", "ATOS", "MEGL",
    "BFRI", "BBAI", "ZEST", "TPST", "GERN",
    "OCEA", "NOVN", "NRXP", "ISIG", "MTTR",
    "DGLY", "HYMC", "IPIX", "GRPN", "FGEN",
    "RKDA", "NURO", "VVPR", "HYLN",

    # --- Active 2023–2025 momentum plays ---
    "SOUN", "RGTI", "QBTS", "IONQ", "ASTS",
    "LUNR", "DJT",  "ABAT", "GFAI", "ENVX",
    "MVST", "HOLO", "SRM",  "OUST", "GREE",
    "CLSK", "MMAT", "LAZR", "AIXI", "CODA",
    "CTXR", "KAVL", "PALI", "MIGI", "COCH",
    "FFAI", "CRKN", "ONCO", "BTBT", "PRST",

    # --- Pre-2022 penny/small-cap momentum (rich 5-year history) ---
    "ZOM",  "BOXL", "AGRX", "PTE",  "CPHI",
    "BHAT", "YCBD", "XBIO", "VVUS", "LPTX",
    "SIGA", "CTIC", "NKTR", "CPRX", "ARCT",
    "AGTC", "CRBP", "INVA", "XXII", "PIRS",
    "OBSV", "ADMA", "GNPX", "VBLT", "GLYC",
    "TPIC", "UAVS", "LAKE", "LXRX", "CDXS",
    "NVAX", "SRNE", "INMD", "CODX", "TXMD",
    "ABUS", "VXRT", "INO",  "OCUL", "ADAP",

    # --- 2022–2025 additions: crypto miners ---
    "MARA", "RIOT", "CIFR", "HUT",  "BITF",
    "CLSK", "IREN", "BTBT", "WULF", "MIGI",

    # --- 2022–2025 additions: biotech/small-cap spikes ---
    "CERO", "NKGN", "ATNF", "CRVS", "IMVT",
    "LASE", "FATH", "RVNC", "YMAB", "LIFW",
    "BURU", "ABTS", "LFLY", "CLRB", "CYTO",
    "ETON", "CASI", "APRE", "MIST", "SRTS",
    "KNSA", "PRTA", "AQST", "IPHA", "MTNB",
    "VTYX", "ACCD", "HOOK", "ADTX", "WINT",
    "TZOO", "NCPL", "SNPX", "PHAS", "FWBI",
    "CRMD", "NLSP", "ATNM", "HARP", "SPGX",

    # --- Additional historical low-float plays ---
    "NVOS", "NILE", "GOVX", "APCX", "PRPB",
    "LTRY", "SRTX", "OBLG", "AEYE", "IMAQ",
    "BACK", "HGEN", "CYCC", "AGLE", "MFON",
    "CLPS", "TAOP", "ABVC", "NXTD", "MIMO",
    "SEAC", "CREX", "SMFL", "EDTK", "XTLB",
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
        if relative_volume >= 500:
            score += w.get("rel_vol_500x", 40)
        elif relative_volume >= 100:
            score += w.get("rel_vol_100x", 35)
        elif relative_volume >= 50:
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
            score -= w.get("shares_gte100m_penalty", 0)

    # Treat historical moves as organic (no news signal available)
    score += w.get("no_news_bonus", 0)

    max_score = (
        max(w.get("rel_vol_500x", 40), w.get("rel_vol_100x", 35),
            w["rel_vol_50x"], w["rel_vol_25x"], w["rel_vol_10x"], 0)
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
    Download up to 10 years of OHLCV for `symbol`.
    Slide a window to find days matching live scan criteria:
        price < $5, daily gain 10–100%, relative volume ≥ 10.
    Returns a list of labeled example dicts with next_day_return filled in.
    """
    try:
        df = yf.download(
            symbol, start="2018-01-01", interval="1d",
            progress=False, auto_adjust=False
        )
        if df.empty or len(df) < 70:
            return []

        df = prepare_dataframe(df)

        # Fetch shares outstanding and sector (best available proxy for historical)
        shares_outstanding = None
        float_shares = None
        sector = None
        try:
            info = yf.Ticker(symbol).fast_info
            shares_outstanding = getattr(info, "shares", None)
        except Exception:
            pass
        try:
            sector = yf.Ticker(symbol).info.get("sector")
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

            # First trading day (1–10) where intraday HIGH hit +20% from scan-day close
            days_to_20pct = None
            for days_ahead in range(1, 11):
                if i + days_ahead < n:
                    h = _safe(df.iloc[i + days_ahead].get("high"))
                    if h and h > 0 and close_today > 0:
                        if (h / close_today - 1) >= 0.20:
                            days_to_20pct = days_ahead
                            break

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
                "float_shares":       None,  # skipped in backfill to save memory
                "next_day_return":    next_day_return,
                "three_day_return":   three_day_return,
                "days_to_20pct":      days_to_20pct,
                "range_10d":          round(range_10d, 4) if range_10d is not None else None,
                "yesterday_green":    int(yesterday_green),
                "sector":             sector,
            })

        return examples

    except Exception as e:
        print(f"Backfill: error on {symbol} — {e}")
        return []
    finally:
        # Explicitly free dataframe memory after each ticker
        try:
            del df
        except NameError:
            pass
        gc.collect()


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


def _load_us_tickers(path: str = "us_tickers.txt", max_count: int = 2000) -> list:
    """Load tickers from flat text file. Filters to 1-5 char alpha-only tickers."""
    try:
        with open(path) as f:
            raw = [line.strip().upper() for line in f if line.strip()]
        return [t for t in raw if 1 <= len(t) <= 5 and t.isalpha()][:max_count]
    except Exception:
        return []


def backfill_signals_for_historical(max_workers: int = 2) -> int:
    """
    For each historical scan row without signals_json:
      1. Download full OHLCV for the symbol (yfinance max period)
      2. Re-run score_stock_squeeze() at the matching scan date
      3. Persist fired_signals as signals_json

    This gives XGBoost 3,500+ labeled training rows with full signal features.
    Skips rows that already have signals_json set (safe to re-run).
    Returns number of rows updated.
    """
    import json
    from app.scoring_engine import score_stock_squeeze

    # Unique symbols that still need signals_json (all modes except 5m)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT symbol FROM scans
        WHERE mode NOT IN ('fivemin', 'fivemin_bt')
          AND (signals_json IS NULL OR signals_json = '{}')
    """)
    symbols = [r[0] for r in cursor.fetchall()]
    conn.close()

    if not symbols:
        print("Signal backfill: all historical rows already have signals_json")
        return 0

    print(f"Signal backfill: {len(symbols)} symbols to re-score")
    updated_total = 0

    def _process_one(symbol):
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp FROM scans
                WHERE mode NOT IN ('fivemin', 'fivemin_bt')
                  AND (signals_json IS NULL OR signals_json = '{}')
                  AND symbol = ?
            """, (symbol,))
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                return 0

            df = yf.download(symbol, start="2018-01-01", interval="1d",
                             progress=False, auto_adjust=False)
            if df.empty or len(df) < 30:
                return 0
            df = prepare_dataframe(df)

            fundamentals = {}
            try:
                info = yf.Ticker(symbol).fast_info
                fundamentals["shares_outstanding"] = getattr(info, "shares", None)
            except Exception:
                pass
            try:
                full_info = yf.Ticker(symbol).info
                fundamentals["sector"]       = full_info.get("sector")
                fundamentals["industry"]     = full_info.get("industry")
                fundamentals["float_shares"] = full_info.get("floatShares")
                fundamentals["institution_pct"] = full_info.get("heldPercentInstitutions")
            except Exception:
                pass

            date_index = {str(idx)[:10]: i for i, idx in enumerate(df.index)}

            updates = []
            for scan_id, timestamp in rows:
                date_str = str(timestamp)[:10]
                idx = date_index.get(date_str)
                if idx is None or idx < 1:
                    continue
                df_slice = df.iloc[:idx + 1]
                result = score_stock_squeeze(symbol, df_slice, fundamentals)
                if result:
                    fired = result.get("checklist", {}).get("fired_signals", {})
                    updates.append((json.dumps(fired) if fired else "{}", scan_id))

            if updates:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.executemany(
                    "UPDATE scans SET signals_json = ? WHERE id = ?", updates
                )
                conn.commit()
                conn.close()
                return len(updates)
            return 0

        except Exception as e:
            print(f"Signal backfill: error on {symbol} — {e}")
            return 0
        finally:
            try:
                del df
            except NameError:
                pass
            gc.collect()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, sym): sym for sym in symbols}
        for i, future in enumerate(as_completed(futures), 1):
            updated_total += future.result()
            if i % 20 == 0 or i == len(symbols):
                print(f"Signal backfill: {i}/{len(symbols)} symbols done, "
                      f"{updated_total} rows updated")

    print(f"Signal backfill: complete — {updated_total} rows updated")

    # Refresh bundle rule projections now that signals_json is populated
    try:
        from app.database import refresh_bundle_projections
        n_proj = refresh_bundle_projections()
        if n_proj > 0:
            print(f"Signal backfill: refreshed projections for {n_proj} bundle rules")
    except Exception as e:
        print(f"Signal backfill: projection refresh failed — {e}")

    return updated_total


def build_historical_dataset(max_workers=2, weights=None):
    """
    Process seed + Finviz + us_tickers.txt + previously seen tickers in parallel.
    Saves qualifying labeled examples into the scans table (mode='historical').
    Returns count of examples saved.

    Memory-efficient: batch-saves every 50 tickers (clears accumulator),
    keeps only lightweight (symbol, timestamp, days_to_20pct) tuples for LSTM.
    """
    from app.database import save_historical_scans, set_backfill_status
    from app.universe import fetch_backfill_universe

    # Purge pre-2018 historical rows — they predate reliable yfinance coverage
    # and can't be signal-backfilled, so they only dilute training quality
    try:
        conn = sqlite3.connect(DB_NAME)
        deleted = conn.execute(
            "DELETE FROM scans WHERE mode='historical' AND timestamp < '2018-01-01'"
        ).rowcount
        conn.commit()
        conn.close()
        if deleted:
            print(f"Backfill: purged {deleted} pre-2018 historical rows")
    except Exception as e:
        print(f"Backfill: pre-2018 purge failed — {e}")

    db_tickers = _get_db_tickers()

    # Fetch dynamic universe from Finviz (price < $10, avg vol > 200K)
    try:
        dynamic_tickers = fetch_backfill_universe(max_tickers=1000)
    except Exception as e:
        print(f"Backfill: dynamic universe fetch failed — {e}")
        dynamic_tickers = []

    # Load broad US ticker universe for extra coverage
    us_tickers = _load_us_tickers()

    # Merge seed + dynamic + us_tickers + known tickers, deduplicated, seed list first
    seen = set()
    all_tickers = []
    for t in SEED_TICKERS + dynamic_tickers + us_tickers + db_tickers:
        if t not in seen:
            seen.add(t)
            all_tickers.append(t)

    total = len(all_tickers)
    set_backfill_status("running", 0, total, 0)
    print(f"Backfill: starting — {total} tickers "
          f"(seed={len(SEED_TICKERS)}, finviz={len(dynamic_tickers)}, "
          f"us_tickers={len(us_tickers)}, db={len(db_tickers)})")

    batch = []          # full example dicts — flushed every 50 tickers
    seq_inputs = []     # lightweight tuples for LSTM — kept for full run
    total_saved = 0
    processed = 0
    first_batch = True

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
                batch.extend(result)
                for ex in result:
                    seq_inputs.append({
                        "symbol":            ex["symbol"],
                        "timestamp":         ex["timestamp"],
                        "days_to_20pct":     ex.get("days_to_20pct"),
                        "shares_outstanding": ex.get("shares_outstanding"),
                        "sector":            ex.get("sector"),
                    })

            # Update status on every ticker so progress bar stays live
            set_backfill_status("running", processed, total, total_saved + len(batch))

            # Flush batch to DB every 50 tickers to keep RAM low
            if processed % 50 == 0 or processed == total:
                if batch:
                    saved_now = save_historical_scans(batch, clear_first=first_batch)
                    total_saved += saved_now
                    batch.clear()
                    first_batch = False

    set_backfill_status("complete", processed, total, total_saved)
    print(f"Backfill: complete — {total_saved} examples from {processed} tickers")

    # Build LSTM sequences from lightweight inputs (batch list is already cleared)
    try:
        from app.lstm_model import build_sequences_from_backfill
        print("Backfill: building LSTM sequences...")
        n_seq = build_sequences_from_backfill(seq_inputs)
        print(f"Backfill: {n_seq} LSTM sequences saved")
    except Exception as e:
        print(f"Backfill: LSTM sequence build failed — {e}")

    return total_saved
