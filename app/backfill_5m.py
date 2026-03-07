"""
Historical 5m Spike backfill: finds past 5-minute volume spikes (>=10x relvol)
within yfinance's 60-day 5m data window and saves them as mode='fivemin_bt' scans
with pre-computed returns so analytics are immediately available.
"""

import gc
import json
import sqlite3
import datetime

import pandas as pd
import pytz
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.database import DB_NAME
from app.scoring_engine import DEFAULT_SQUEEZE_WEIGHTS


# Tickers known to produce 5m spikes — subset of the daily backfill seed list
_5M_SEED_TICKERS = [
    "SNDL", "CTRM", "IDEX", "ATER", "PROG", "XELA", "OCGN", "CLOV",
    "MVIS", "FCEL", "GNUS", "MULN", "FFIE", "MARK", "AUVI", "VERB",
    "SHOT", "TLRY", "BNGO", "DARE", "AVXL", "SESN", "GSAT", "SONN",
    "KULR", "AULT", "EFSH", "UPXI", "TRKA", "VISL", "AMPE", "BPTH",
    "LODE", "IZEA", "ATOS", "MEGL", "BFRI", "BBAI", "ZEST", "GERN",
    "NOVN", "NRXP", "DGLY", "HYMC", "IPIX", "GRPN", "NURO", "VVPR",
    "SOUN", "RGTI", "QBTS", "IONQ", "ASTS", "LUNR", "DJT", "ABAT",
    "GFAI", "ENVX", "MVST", "HOLO", "SRM", "OUST", "GREE", "CLSK",
    "MMAT", "LAZR", "AIXI", "CODA", "CTXR", "KAVL", "PALI", "MIGI",
    "CRKN", "ONCO", "BTBT", "PRST", "ZOM", "BHAT", "VVUS", "SIGA",
    "NKTR", "CPRX", "ARCT", "ADMA", "MARA", "RIOT", "CIFR", "HUT",
    "BITF", "IREN", "WULF", "CERO", "NKGN", "LASE", "FATH", "BURU",
    "ABTS", "CLRB", "CYTO", "ETON", "CASI", "MIST", "SRTS", "CRMD",
]


def _safe(v):
    try:
        f = float(v)
        return None if f != f else f
    except Exception:
        return None


def _get_fivemin_db_tickers() -> list:
    """Return unique tickers seen in fivemin live scans."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT symbol FROM scans "
            "WHERE mode = 'fivemin' ORDER BY id DESC LIMIT 200"
        )
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def _normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to the price-field level."""
    if isinstance(df.columns, pd.MultiIndex):
        price_fields = {"Open", "High", "Low", "Close", "Volume"}
        lvl0 = set(df.columns.get_level_values(0))
        df.columns = (
            df.columns.get_level_values(0) if lvl0 & price_fields
            else df.columns.get_level_values(-1)
        )
    return df.loc[:, ~df.columns.duplicated()]


def _process_ticker_5m(symbol: str, et) -> list:
    """
    Download 60d 5m data + 6mo daily data for `symbol`.
    For each past trading day, find the first 5m bar with >=10x relvol at that slot.
    Score it and compute returns from daily data.
    Returns list of labeled example dicts.
    """
    try:
        # --- 5m OHLCV ---
        data_5m = yf.download(
            symbol, period="60d", interval="5m",
            progress=False, auto_adjust=False
        )
        if data_5m is None or data_5m.empty:
            return []

        data_5m = _normalize_df_columns(data_5m)
        if "Volume" not in data_5m.columns or "Close" not in data_5m.columns:
            return []

        # Convert to ET
        if data_5m.index.tzinfo is None:
            data_5m.index = data_5m.index.tz_localize("UTC").tz_convert(et)
        else:
            data_5m.index = data_5m.index.tz_convert(et)

        # --- Daily OHLCV (for scoring + return computation) ---
        data_1d = yf.download(
            symbol, period="6mo", interval="1d",
            progress=False, auto_adjust=False
        )
        if data_1d is None or data_1d.empty or len(data_1d) < 5:
            return []

        data_1d = _normalize_df_columns(data_1d)

        # Make daily index tz-naive for date comparison
        if hasattr(data_1d.index, "tz") and data_1d.index.tz is not None:
            data_1d.index = data_1d.index.tz_convert(None)

        today_et = datetime.datetime.now(et).date()

        # Build list of past trading days present in the 5m window
        trading_days = sorted({ts.date() for ts in data_5m.index if ts.date() < today_et})
        if not trading_days:
            return []

        # Pre-build a date→row_position map for daily data
        daily_dates = [t.date() for t in data_1d.index]

        # Current shares outstanding (proxy — same approach as backfill.py)
        shares_outstanding = None
        try:
            info = yf.Ticker(symbol).fast_info
            shares_outstanding = getattr(info, "shares", None)
        except Exception:
            pass

        examples = []

        for trade_date in trading_days:
            # 5m bars for this day during regular hours (9:30–15:55 ET)
            day_mask = (
                (data_5m.index.date == trade_date) &
                (
                    (data_5m.index.hour > 9) |
                    ((data_5m.index.hour == 9) & (data_5m.index.minute >= 30))
                ) &
                (data_5m.index.hour < 16)
            )
            day_bars = data_5m[day_mask]
            if day_bars.empty:
                continue

            # Find the FIRST bar of the day with relvol >= 10x
            first_spike = None
            for bar_ts in day_bars.index:
                slot_h, slot_m = bar_ts.hour, bar_ts.minute

                # Prior bars at same time slot (days before trade_date)
                slot_mask = (
                    (data_5m.index.hour == slot_h) &
                    (data_5m.index.minute == slot_m) &
                    (data_5m.index.date < trade_date)
                )
                prior_vols = data_5m[slot_mask]["Volume"].dropna()

                if len(prior_vols) < 3:
                    continue

                hist_avg = float(prior_vols.tail(10).mean())
                if hist_avg <= 0:
                    continue

                cur_vol = _safe(day_bars.loc[bar_ts, "Volume"])
                if not cur_vol or cur_vol == 0:
                    continue

                relvol = cur_vol / hist_avg
                if relvol >= 10:
                    bar_close = _safe(day_bars.loc[bar_ts, "Close"])
                    if bar_close and bar_close > 0:
                        first_spike = {
                            "bar_ts": bar_ts,
                            "relvol": round(relvol, 2),
                            "scan_price": bar_close,
                        }
                    break  # only first spike per ticker per day

            if not first_spike:
                continue

            # Find daily row for trade_date
            try:
                row_pos = daily_dates.index(trade_date)
            except ValueError:
                continue

            row = data_1d.iloc[row_pos]
            close = _safe(row.get("Close"))
            open_ = _safe(row.get("Open"))
            if not close or close <= 0 or close >= 5.0:
                continue

            # Daily return
            if open_ and open_ > 0:
                daily_return = (close - open_) / open_
            elif row_pos > 0:
                prev_c = _safe(data_1d.iloc[row_pos - 1].get("Close"))
                daily_return = (close - prev_c) / prev_c if prev_c and prev_c > 0 else None
            else:
                daily_return = None

            if daily_return is None or daily_return < 0.05:
                continue

            # 10-day range
            range_10d = None
            if row_pos >= 10:
                window = data_1d.iloc[row_pos - 10:row_pos]
                highs = [_safe(window.iloc[i].get("High")) for i in range(len(window))]
                lows  = [_safe(window.iloc[i].get("Low"))  for i in range(len(window))]
                highs = [h for h in highs if h]
                lows  = [l for l in lows  if l]
                if highs and lows and close > 0:
                    range_10d = (max(highs) - min(lows)) / close

            # yesterday_green
            yesterday_green = False
            if row_pos > 0:
                prev = data_1d.iloc[row_pos - 1]
                pc, po = _safe(prev.get("Close")), _safe(prev.get("Open"))
                if pc and po and pc > po:
                    yesterday_green = True

            # Scoring
            w = DEFAULT_SQUEEZE_WEIGHTS
            rv = first_spike["relvol"]
            score_val = 0

            if rv >= 50:   score_val += w["rel_vol_50x"]
            elif rv >= 25: score_val += w["rel_vol_25x"]
            elif rv >= 10: score_val += w["rel_vol_10x"]

            if 0.20 <= daily_return <= 0.40:
                score_val += w["daily_sweet_20_40"]
            elif 0.10 <= daily_return < 0.20:
                score_val += w["daily_ok_10_20"]
            elif 0.40 < daily_return <= 1.00:
                score_val += w["daily_ok_40_100"]

            if range_10d is not None and range_10d < 0.20:
                score_val += w["sideways_chop"]
            if yesterday_green:
                score_val += w["yesterday_green"]

            if shares_outstanding is not None:
                if shares_outstanding < 10_000_000:
                    score_val += w["shares_lt10m"]
                elif shares_outstanding < 30_000_000:
                    score_val += w["shares_lt30m"]

            max_s = (
                max(w["rel_vol_50x"], w["rel_vol_25x"], w["rel_vol_10x"], 0)
                + max(w["daily_sweet_20_40"], w["daily_ok_10_20"], w["daily_ok_40_100"], 0)
                + w["sideways_chop"] + w["yesterday_green"]
                + max(w["shares_lt10m"], w["shares_lt30m"], 0)
                + w.get("no_news_bonus", 0)
            )
            max_s = max(max_s, 1)
            score = max(0, round((score_val / max_s) * 100))
            rec = "TRADE" if score >= 75 else ("WATCH" if score >= 50 else "SPECULATIVE")

            # Returns (computed directly — we have future daily data)
            scan_price = first_spike["scan_price"]
            n = len(data_1d)

            next_day_return = None
            if row_pos + 1 < n:
                c1 = _safe(data_1d.iloc[row_pos + 1].get("Close"))
                if c1 and c1 > 0:
                    next_day_return = round((c1 - scan_price) / scan_price * 100, 2)

            three_day_return = None
            if row_pos + 3 < n:
                c3 = _safe(data_1d.iloc[row_pos + 3].get("Close"))
                if c3 and c3 > 0:
                    three_day_return = round((c3 - scan_price) / scan_price * 100, 2)

            days_to_20pct = None
            for days_ahead in range(1, 11):
                if row_pos + days_ahead < n:
                    h = _safe(data_1d.iloc[row_pos + days_ahead].get("High"))
                    if h and h > 0 and (h / scan_price - 1) >= 0.20:
                        days_to_20pct = days_ahead
                        break

            examples.append({
                "timestamp":       first_spike["bar_ts"].isoformat(),
                "symbol":          symbol,
                "score":           score,
                "recommendation":  rec,
                "relative_volume": rv,
                "today_return":    round(daily_return * 100, 2),
                "scan_price":      scan_price,
                "shares_outstanding": shares_outstanding,
                "next_day_return": next_day_return,
                "three_day_return": three_day_return,
                "days_to_20pct":   days_to_20pct,
                "range_10d":       round(range_10d, 4) if range_10d is not None else None,
                "yesterday_green": int(yesterday_green),
            })

        return examples

    except Exception as e:
        print(f"5m Backfill: error on {symbol} — {e}")
        return []
    finally:
        try:
            del data_5m
        except NameError:
            pass
        try:
            del data_1d
        except NameError:
            pass
        gc.collect()


def _save_fivemin_bt_scans(examples: list, clear_first: bool = False) -> int:
    """
    Insert 5m backfill examples with mode='fivemin_bt'.
    When clear_first=True, wipes all existing fivemin_bt rows first (re-run case).
    """
    if not examples:
        return 0
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if clear_first:
        cursor.execute("DELETE FROM scans WHERE mode = 'fivemin_bt'")
    count = 0
    for ex in examples:
        try:
            cursor.execute("""
                INSERT INTO scans (
                    timestamp, symbol, score, recommendation, mode,
                    relative_volume, today_return, shares_outstanding,
                    scan_price, news_recent,
                    next_day_return, three_day_return, days_to_20pct,
                    range_10d, yesterday_green
                ) VALUES (?, ?, ?, ?, 'fivemin_bt', ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
            """, (
                ex["timestamp"],
                ex["symbol"],
                ex["score"],
                ex["recommendation"],
                ex.get("relative_volume"),
                ex.get("today_return"),
                ex.get("shares_outstanding"),
                ex.get("scan_price"),
                ex.get("next_day_return"),
                ex.get("three_day_return"),
                ex.get("days_to_20pct"),
                ex.get("range_10d"),
                ex.get("yesterday_green"),
            ))
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    return count


def set_5m_backfill_status(status: str, processed: int = 0,
                            total: int = 0, saved: int = 0):
    value = json.dumps({
        "status": status,
        "processed": processed,
        "total": total,
        "saved": saved,
        "updated_at": datetime.datetime.utcnow().isoformat(),
    })
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES ('fivemin_backfill_status', ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
    """, (value, datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_5m_backfill_status() -> dict:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = 'fivemin_backfill_status'")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {"status": "idle", "processed": 0, "total": 0, "saved": 0}
    return json.loads(row[0])


def backfill_5m_history(max_tickers: int = 150) -> int:
    """
    Scan 60 days of 5m history across `max_tickers` tickers to find past
    5-minute volume spikes (>=10x relvol) and save them as mode='fivemin_bt'
    scans with pre-computed returns.

    Uses default squeeze weights for scoring (no live fundamentals available
    for historical dates). Returns count of examples saved.
    """
    et = pytz.timezone("America/New_York")

    # Build ticker list: DB fivemin tickers first, then seed list, capped
    db_tickers = _get_fivemin_db_tickers()
    seen = set()
    all_tickers = []
    for t in db_tickers + _5M_SEED_TICKERS:
        if t not in seen:
            seen.add(t)
            all_tickers.append(t)
        if len(all_tickers) >= max_tickers:
            break

    total = len(all_tickers)
    set_5m_backfill_status("running", 0, total, 0)
    print(f"5m Backfill: starting — {total} tickers, 60-day window")

    batch = []
    total_saved = 0
    processed = 0
    first_batch = True

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_process_ticker_5m, sym, et): sym
            for sym in all_tickers
        }
        for future in as_completed(futures):
            processed += 1
            try:
                result = future.result(timeout=90)
            except Exception as e:
                print(f"5m Backfill: future error — {e}")
                result = []
            if result:
                batch.extend(result)

            set_5m_backfill_status("running", processed, total, total_saved + len(batch))

            # Flush every 25 tickers to keep memory low
            if processed % 25 == 0 or processed == total:
                if batch:
                    saved_now = _save_fivemin_bt_scans(batch, clear_first=first_batch)
                    total_saved += saved_now
                    batch.clear()
                    first_batch = False

    set_5m_backfill_status("complete", processed, total, total_saved)
    print(f"5m Backfill: complete — {total_saved} examples from {processed} tickers")
    return total_saved
