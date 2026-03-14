#!/usr/bin/env python3
"""
Telegram monitor — run locally to inspect bot activity and send test messages.

Usage:
  venv/bin/python3 telegram_monitor.py            # show recent updates + last 20 sent alerts
  venv/bin/python3 telegram_monitor.py --test     # send a test alert to verify formatting
  venv/bin/python3 telegram_monitor.py --tail     # poll continuously (ctrl+c to stop)
"""
import os, sys, time, json, sqlite3, argparse
import requests

TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "8637688005:AAG3Vx_wZzGK-YcD9CTwTyFqkxIuBPBuYxg")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID",   "7888205192")
BASE    = f"https://api.telegram.org/bot{TOKEN}"
DB_PATH = os.environ.get("DB_PATH", "scan_history.db")


def get_updates(offset=None, limit=20):
    params = {"limit": limit, "timeout": 0}
    if offset:
        params["offset"] = offset
    r = requests.get(f"{BASE}/getUpdates", params=params, timeout=10)
    r.raise_for_status()
    return r.json().get("result", [])


def send_message(text, parse_mode="HTML"):
    r = requests.post(f"{BASE}/sendMessage", json={
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
    }, timeout=10)
    r.raise_for_status()
    return r.json()


def show_recent_alerts(limit=20):
    """Pull last N outgoing alerts from the telegram_log table if it exists."""
    if not os.path.exists(DB_PATH):
        print(f"DB not found at {DB_PATH}")
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT sent_at, chat_id, message FROM telegram_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        if not rows:
            print("No logged alerts yet (telegram_log table may be empty or not exist).")
            return
        print(f"\n=== Last {len(rows)} sent alerts ===")
        for sent_at, chat, msg in rows:
            print(f"\n[{sent_at}] → {chat}")
            print(msg[:500] + ("..." if len(msg) > 500 else ""))
    except sqlite3.OperationalError:
        print("telegram_log table does not exist yet.")
    finally:
        conn.close()


def show_incoming(limit=20):
    updates = get_updates(limit=limit)
    if not updates:
        print("No incoming messages.")
        return
    print(f"\n=== Last {len(updates)} incoming messages ===")
    for u in updates:
        msg = u.get("message") or u.get("callback_query", {}).get("message", {})
        if msg:
            from_user = msg.get("from", {}).get("username", "?")
            text = msg.get("text", "")
            date = msg.get("date", "")
            print(f"  [{date}] @{from_user}: {text}")


def send_test_alert():
    sample = (
        "<b>1 High-Score Alert — Daily Complex + AI</b>\n\n"
        "<b>TSLA</b>  85/100  $250.00  RV:12x"
        " ✅ <b>TRADE</b> (HIGH · Daily Complex + AI — position opened)"
        " · LSTM: 72%\n"
        "  <i>Strong squeeze with elevated relative volume. Float &lt;10M, "
        "gap up on news catalyst. High LSTM confidence.</i>"
    )
    result = send_message(sample)
    print(f"Test alert sent. Message ID: {result.get('result', {}).get('message_id')}")


def tail_updates():
    print("Tailing incoming messages (ctrl+c to stop)...")
    offset = None
    while True:
        try:
            updates = get_updates(offset=offset)
            for u in updates:
                offset = u["update_id"] + 1
                msg = u.get("message", {})
                if msg:
                    from_user = msg.get("from", {}).get("username", "?")
                    text = msg.get("text", "")
                    print(f"  @{from_user}: {text}")
            time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true", help="Send a test alert")
    parser.add_argument("--tail",  action="store_true", help="Poll for incoming messages")
    parser.add_argument("--alerts", action="store_true", help="Show recent sent alerts from DB log")
    args = parser.parse_args()

    if args.test:
        send_test_alert()
    elif args.tail:
        tail_updates()
    elif args.alerts:
        show_recent_alerts()
    else:
        show_incoming()
        show_recent_alerts()
