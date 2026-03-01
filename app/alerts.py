"""
Email + Telegram alert system for trade notifications.

Email requires env vars: GMAIL_USER, GMAIL_APP_PASSWORD, ALERT_EMAIL_TO
Telegram requires env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
All functions are silent no-ops if the relevant env vars are not set.
"""

import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def _send_email(subject: str, body: str):
    """Send an email via Gmail SMTP SSL. Silent no-op if env vars not configured."""
    user = os.environ.get("GMAIL_USER")
    pwd  = os.environ.get("GMAIL_APP_PASSWORD")
    to   = os.environ.get("ALERT_EMAIL_TO")
    if not all([user, pwd, to]):
        return
    msg = MIMEMultipart()
    msg["From"]    = user
    msg["To"]      = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(user, pwd)
            s.sendmail(user, to, msg.as_string())
        print(f"Alert email sent: {subject}")
    except Exception as e:
        print(f"Alert email failed: {e}")


def _send_telegram(message: str):
    """
    Send a Telegram push notification.
    Silent no-op if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.

    Setup:
      1. Create a bot via @BotFather, copy the token → TELEGRAM_BOT_TOKEN env var
      2. Start a chat with your bot, then visit:
         https://api.telegram.org/bot<TOKEN>/getUpdates
         to find your chat_id → TELEGRAM_CHAT_ID env var
    """
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if resp.ok:
            print("Telegram alert sent")
        else:
            print(f"Telegram alert failed: {resp.status_code} {resp.text[:120]}")
    except Exception as e:
        print(f"Telegram alert error: {e}")


def send_scan_alert(results: list, mode: str, min_score: int = 75):
    """
    Email + Telegram top-scoring stocks from a scan.
    Only sends if at least one stock scores >= min_score.
    Includes AI call, confidence, LSTM prob, and rationale when available.
    """
    top = [r for r in results if r.get("score", 0) >= min_score]
    if not top:
        return

    email_lines = [f"{len(top)} stock(s) scored {min_score}+ in {mode} scan\n"]
    tg_lines    = [f"<b>{len(top)} High-Score Alert — {mode}</b>\n"]

    for r in top[:10]:
        chk        = r.get("checklist", {})
        price      = r.get("price", "?")
        rv         = chk.get("relative_volume", "?")
        tc         = r.get("ai_trade_call") or {}
        decision   = tc.get("decision", "")
        confidence = tc.get("confidence", "")
        rationale  = tc.get("rationale", "")
        lstm_prob  = r.get("lstm_prob")

        # Email line
        ai_str   = f"  AI:{decision}/{confidence}" if decision else ""
        lstm_str = f"  LSTM:{lstm_prob:.0%}" if lstm_prob is not None else ""
        email_lines.append(
            f"  {r['symbol']:6s}  Score:{r['score']}/100  {r.get('recommendation','')}"
            f"  ${price}  RV:{rv}x{ai_str}{lstm_str}"
        )
        if rationale:
            email_lines.append(f"         {rationale}")

        # Telegram line
        if decision == "TRADE":
            ai_badge = f" ✅ <b>{decision}</b> ({confidence})"
        elif decision == "NO_TRADE":
            ai_badge = f" ❌ {decision} ({confidence})"
        else:
            ai_badge = ""
        lstm_badge = f" · LSTM: {lstm_prob:.0%}" if lstm_prob is not None else ""
        tg_lines.append(
            f"<b>{r['symbol']}</b>  {r['score']}/100  ${price}  RV:{rv}x{ai_badge}{lstm_badge}"
        )
        if rationale and decision == "TRADE":
            tg_lines.append(f"  <i>{rationale}</i>")

    _send_email(
        subject=f"[Trading] {len(top)} High-Score Alert — {mode}",
        body="\n".join(email_lines),
    )
    _send_telegram("\n".join(tg_lines))


def send_watchlist_alert(symbol: str, score: int, price: float = None,
                         old_score: int = None, lstm_prob: float = None,
                         rationale: str = None):
    """
    Alert when a near-miss watchlist stock rises to score >= 75.
    old_score is the score when it was first added to the watchlist.
    """
    price_str  = f"  ${price:.2f}" if price else ""
    prev_str   = f" (was {old_score})" if old_score else ""
    lstm_str   = f"  LSTM: {lstm_prob:.0%}" if lstm_prob is not None else ""
    subject    = f"[Trading] Watchlist Breakout: {symbol} hit {score}/100"
    body       = f"WATCHLIST BREAKOUT: {symbol} now scoring {score}/100{prev_str}{price_str}{lstm_str}"
    if rationale:
        body  += f"\n{rationale}"
    tg_msg     = (
        f"<b>Watchlist Breakout: {symbol}</b>\n"
        f"Score: {score}/100{prev_str}{price_str}{lstm_str}"
    )
    if rationale:
        tg_msg += f"\n<i>{rationale}</i>"
    _send_email(subject=subject, body=body)
    _send_telegram(tg_msg)


def send_take_profit_alert(closed_trades: list):
    """
    Email + Telegram when one or more positions are auto-closed at the take-profit target.
    closed_trades: list of dicts with symbol, entry_price, exit_price, realized_pnl, pnl_pct
    """
    if not closed_trades:
        return

    email_lines = [f"{len(closed_trades)} position(s) auto-closed at take-profit target\n"]
    tg_lines    = [f"<b>{len(closed_trades)} Take-Profit Hit!</b>\n"]

    for t in closed_trades:
        email_lines.append(
            f"  {t['symbol']:6s}  +{t['pnl_pct']:.1f}%"
            f"  Entry:${t['entry_price']}  Close:${t['exit_price']:.4f}"
            f"  P&L:+${t['realized_pnl']:.2f}"
        )
        tg_lines.append(
            f"<b>{t['symbol']}</b>  +{t['pnl_pct']:.1f}%  P&L: +${t['realized_pnl']:.2f}"
        )

    _send_email(
        subject=f"[Trading] \U0001f3af {len(closed_trades)} Take-Profit Hit!",
        body="\n".join(email_lines),
    )
    _send_telegram("\n".join(tg_lines))
