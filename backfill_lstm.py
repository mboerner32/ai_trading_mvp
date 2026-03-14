"""
backfill_lstm.py — Backfill lstm_prob for all historical scans using the price
window that was actually available on the scan date.

Uses predict_hit_probability_as_of(symbol, scan_date) which fetches the 20
trading-day window ending the day before scan_date — exactly matching what
the live scanner would have produced.

Run with:
    caffeinate python3 backfill_lstm.py
    caffeinate python3 backfill_lstm.py --reset   # clears existing values first

Progress is checkpointed: safe to interrupt and re-run.
"""

import sqlite3
import time
import os
import sys
from datetime import datetime

DB_NAME = os.environ.get("DB_PATH", "scan_history.db")
RESET = "--reset" in sys.argv


def _connect():
    conn = sqlite3.connect(DB_NAME, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


def main():
    from app.lstm_model import predict_hit_probability_as_of, load_lstm

    model = load_lstm()
    if model is None:
        print("✗ LSTM model not found (lstm_model.pt required).")
        return

    conn = _connect()

    if RESET:
        print("→ Resetting all existing lstm_prob values...")
        conn.execute("UPDATE scans SET lstm_prob = NULL")
        conn.commit()
        print("  Done. All lstm_prob set to NULL.\n")

    # Group by (symbol, scan_date) to make one yfinance call per unique pair.
    # Take the first shares_outstanding seen for that pair (doesn't change intraday).
    rows = conn.execute("""
        SELECT
            symbol,
            DATE(timestamp)          AS scan_date,
            MIN(shares_outstanding)  AS shares_outstanding,
            COUNT(*)                 AS n_scans,
            GROUP_CONCAT(id)         AS scan_ids
        FROM scans
        WHERE lstm_prob IS NULL
          AND DATE(timestamp) > '2000-01-01'
        GROUP BY symbol, DATE(timestamp)
        ORDER BY scan_date ASC
    """).fetchall()
    conn.close()

    total = len(rows)
    if total == 0:
        print("✓ Nothing to backfill — all scans already have lstm_prob.")
    else:
        print(f"→ {total} unique (symbol, date) pairs to score")
        print("  ~3–5 s per pair via yfinance. Interrupt any time; re-run to continue.\n")

    done = skipped = errors = 0

    for i, row in enumerate(rows):
        symbol    = row["symbol"]
        scan_date = row["scan_date"]
        shares    = row["shares_outstanding"]
        ids       = [int(x) for x in row["scan_ids"].split(",")]

        try:
            prob = predict_hit_probability_as_of(symbol, scan_date,
                                                  shares_outstanding=shares,
                                                  sector=None)
        except Exception as e:
            print(f"  [{i+1}/{total}] {symbol:8s} {scan_date}  ERROR: {e}")
            errors += 1
            time.sleep(0.5)
            continue

        if prob is None:
            skipped += 1
            # Don't print every skip — too noisy for old/delisted symbols
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] ... ({skipped} skipped so far)")
        else:
            conn2 = _connect()
            placeholders = ",".join("?" * len(ids))
            conn2.execute(
                f"UPDATE scans SET lstm_prob = ? WHERE id IN ({placeholders})",
                [prob] + ids
            )
            conn2.commit()
            conn2.close()
            print(f"  [{i+1}/{total}] {symbol:8s} {scan_date}  → {prob:.1%}  ({len(ids)} rows)")
            done += 1

        time.sleep(0.25)   # gentle rate-limit

    print(f"\n✓ Done. {done} pairs scored, {skipped} skipped, {errors} errors.")

    conn3 = _connect()
    r = conn3.execute("""
        SELECT
            COUNT(*)                                                    AS total,
            COUNT(lstm_prob)                                            AS has_lstm,
            ROUND(AVG(CASE WHEN lstm_prob IS NOT NULL THEN lstm_prob END) * 100, 1) AS avg_pct,
            COUNT(CASE WHEN lstm_prob >= 0.65 THEN 1 END)              AS above_65,
            COUNT(CASE WHEN lstm_prob >= 0.55 AND lstm_prob < 0.65 THEN 1 END) AS btw_55_65,
            COUNT(CASE WHEN lstm_prob < 0.55  AND lstm_prob IS NOT NULL THEN 1 END) AS below_55
        FROM scans
    """).fetchone()
    conn3.close()

    print(f"\nDB summary:")
    print(f"  Total scans    : {r['total']}")
    print(f"  With lstm_prob : {r['has_lstm']}")
    print(f"  Avg LSTM prob  : {r['avg_pct']}%")
    print(f"  ≥65%           : {r['above_65']}")
    print(f"  55–64%         : {r['btw_55_65']}")
    print(f"  <55%           : {r['below_55']}")


if __name__ == "__main__":
    main()
