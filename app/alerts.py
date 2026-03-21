"""
Email + Telegram alert system for trade notifications.

Email requires env vars: GMAIL_USER, GMAIL_APP_PASSWORD, ALERT_EMAIL_TO
Telegram requires env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
All functions are silent no-ops if the relevant env vars are not set.
"""

import os
import sqlite3
import smtplib
import requests
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from app.database import get_telegram_recipients


def _log_telegram(chat_id: str, message: str):
    """Persist outgoing Telegram message to telegram_log table for monitoring."""
    db_path = os.environ.get("DB_PATH", "scan_history.db")
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO telegram_log (sent_at, chat_id, message) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), chat_id, message),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"telegram_log write failed: {e}")


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


def send_weekly_report_email(subject: str, html_body: str):
    """
    Send the weekly analysis report to recipients configured via WEEKLY_REPORT_RECIPIENTS
    env var (comma-separated email addresses).
    Requires GMAIL_USER, GMAIL_APP_PASSWORD, and WEEKLY_REPORT_RECIPIENTS env vars.
    """
    user = os.environ.get("GMAIL_USER")
    pwd  = os.environ.get("GMAIL_APP_PASSWORD")
    recipients_raw = os.environ.get("WEEKLY_REPORT_RECIPIENTS", "")
    if not all([user, pwd]):
        print("WEEKLY EMAIL: GMAIL_USER or GMAIL_APP_PASSWORD not set — skipping")
        return
    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    if not recipients:
        print("WEEKLY EMAIL: WEEKLY_REPORT_RECIPIENTS env var not set — skipping")
        return
    msg = MIMEMultipart("alternative")
    msg["From"]    = user
    msg["To"]      = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(user, pwd)
            s.sendmail(user, recipients, msg.as_string())
        print(f"Weekly report emailed to {recipients}")
    except Exception as e:
        print(f"WEEKLY EMAIL: failed — {e}")


def send_invite_email(to_email: str, name: str, invite_url: str, bot_username: str = "") -> bool:
    """Send an HTML invite email to a new user. Returns True on success."""
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pwd  = os.environ.get("GMAIL_APP_PASSWORD")
    if not all([gmail_user, gmail_pwd]):
        return False

    tg_section = ""
    if bot_username:
        tg_section = f"""
        <div style="margin-top:28px; padding-top:28px; border-top:1px solid #f3f4f6;">
          <p style="font-size:15px; font-weight:600; margin:0 0 6px;">Step 2 — Sign up for Telegram alerts</p>
          <p style="font-size:13px; color:#6b7280; margin:0 0 16px;">
            Get instant alerts when a high-score stock is detected.
            Open <a href="https://t.me/{bot_username}" style="color:#0088cc;">@{bot_username}</a>
            on Telegram, tap <strong>Start</strong>, and send any message to activate.
          </p>
          <div style="text-align:center; margin:16px 0;">
            <img src="https://api.qrserver.com/v1/create-qr-code/?data=https://t.me/{bot_username}&size=140x140&margin=2"
                 width="140" height="140"
                 style="border-radius:10px; border:1px solid #e5e7eb;"
                 alt="Telegram QR code">
            <p style="font-size:11px; color:#9ca3af; margin-top:6px;">Scan with your phone camera</p>
          </div>
          <a href="https://t.me/{bot_username}"
             style="display:block; background:#0088cc; color:white; text-decoration:none;
                    text-align:center; padding:12px; border-radius:8px; font-weight:600; font-size:14px;">
            Open @{bot_username} on Telegram →
          </a>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="margin:0; padding:24px 16px; background:#f7f9fc;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;">
  <div style="max-width:480px; margin:0 auto; background:white; border-radius:16px;
              box-shadow:0 4px 24px rgba(0,0,0,0.08); overflow:hidden;">

    <div style="background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);
                color:white; padding:32px; text-align:center;">
      <div style="font-size:28px; margin-bottom:10px;">📈</div>
      <h1 style="margin:0 0 8px; font-size:22px; font-weight:700;">You're Invited</h1>
      <p style="margin:0; font-size:14px; opacity:0.85; line-height:1.5;">
        Hi {name}, you've been invited to join<br><strong>Reno Robs Trading</strong> —
        AI-powered stock alerts and real-time scan results.
      </p>
    </div>

    <div style="padding:28px 32px;">
      <p style="font-size:15px; font-weight:600; margin:0 0 6px;">Step 1 — Create your account</p>
      <p style="font-size:13px; color:#6b7280; margin:0 0 16px;">
        Click below to set up your login and access the live dashboard.
      </p>
      <a href="{invite_url}"
         style="display:block; background:#4f46e5; color:white; text-decoration:none;
                text-align:center; padding:13px; border-radius:8px;
                font-weight:700; font-size:15px;">
        Create Account →
      </a>
      {tg_section}
    </div>

    <div style="text-align:center; padding:0 32px 24px;
                font-size:12px; color:#9ca3af; line-height:1.6;">
      Questions? Reply to this email.<br>
      Reno Robs Trading · Powered by AI
    </div>
  </div>
</body>
</html>"""

    msg = MIMEMultipart("alternative")
    msg["From"]    = gmail_user
    msg["To"]      = to_email
    msg["Subject"] = "You're invited to Reno Robs Trading"
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(gmail_user, gmail_pwd)
            s.sendmail(gmail_user, to_email, msg.as_string())
        print(f"Invite email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Invite email failed: {e}")
        return False


def _send_telegram_admin(message: str):
    """Send to the primary TELEGRAM_CHAT_ID only (admin). Skips DB recipients."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").split(",")[0].strip()
    if not token or not chat_id:
        return
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if resp.ok:
            _log_telegram(chat_id, message)
    except Exception as e:
        print(f"Telegram admin alert error: {e}")


def _send_telegram(message: str):
    """
    Send a Telegram push notification to one or more recipients.
    Silent no-op if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.

    Setup:
      1. Create a bot via @BotFather, copy the token → TELEGRAM_BOT_TOKEN env var
      2. Each recipient must /start a chat with the bot, then visit:
         https://api.telegram.org/bot<TOKEN>/getUpdates
         to find their chat_id.
      3. Set TELEGRAM_CHAT_ID to a single ID or a comma-separated list for
         multiple recipients, e.g.: 123456789,987654321
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return

    # Merge env var recipients with DB-managed recipients (deduped)
    seen = set()
    chat_ids = []
    for cid in os.environ.get("TELEGRAM_CHAT_ID", "").split(","):
        cid = cid.strip()
        if cid and cid not in seen:
            chat_ids.append(cid)
            seen.add(cid)
    try:
        for r in get_telegram_recipients():
            cid = r["chat_id"].strip()
            if cid and cid not in seen:
                chat_ids.append(cid)
                seen.add(cid)
    except Exception as e:
        print(f"Telegram recipient fetch failed: {e}")

    if not chat_ids:
        return
    for chat_id in chat_ids:
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.ok:
                print(f"Telegram alert sent → {chat_id}")
                _log_telegram(chat_id, message)
            else:
                print(f"Telegram alert failed → {chat_id}: {resp.status_code} {resp.text[:120]}")
        except Exception as e:
            print(f"Telegram alert error → {chat_id}: {e}")


def send_scan_alert(results: list, mode: str, min_score: int = 75,
                    ai_trade_only: bool = False, traded_symbols: set = None,
                    scan_time_label: str = None):
    """
    Email + Telegram top-scoring stocks from a scan.
    Only sends if at least one stock scores >= min_score.
    If ai_trade_only=True, only stocks where AI called TRADE are included.
    traded_symbols: set of symbols where _auto_paper_trade actually opened a position.
    scan_time_label: optional HH:MM string appended to label (e.g. intraday scans).
    Includes AI call, confidence, LSTM prob, and rationale when available.
    """
    top = [r for r in results if r.get("score", 0) >= min_score]
    if ai_trade_only:
        top = [r for r in top if (r.get("ai_trade_call") or {}).get("decision") == "TRADE"]
    if not top:
        return

    _model_labels = {
        "squeeze":  "Daily Auto",
        "autoai":   "Daily Complex + AI",
        "fivemin":  "5 Min Spike",
        "standard": "Daily Standard",
    }
    _model_short = {
        "squeeze":  "Auto AI",
        "autoai":   "Complex+AI",
        "fivemin":  "5m Model",
        "standard": "Standard",
    }
    model_label = _model_labels.get(mode, mode)
    model_short = _model_short.get(mode, mode)
    if scan_time_label:
        model_label = f"{model_label} · Intraday {scan_time_label}"

    email_lines = [f"{len(top)} stock(s) scored {min_score}+ in {model_label} scan\n"]
    tg_lines    = [f"<b>{len(top)} High-Score Alert — {model_label}</b>\n"]

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
        symbol = r["symbol"]
        if decision == "TRADE":
            if confidence in ("HIGH", "MEDIUM"):
                if traded_symbols and symbol in traded_symbols:
                    ai_badge = f" ✅ <b>{decision}</b> ({confidence} · Model: {model_short} — position opened)"
                else:
                    ai_badge = f" ✅ <b>{decision}</b> ({confidence} · Model: {model_short})"
            else:
                ai_badge = f" ⚠️ <b>{decision}</b> ({confidence} · Model: {model_short} — no position, LOW conf)"
        elif decision == "NO_TRADE":
            ai_badge = f" ❌ {decision} ({confidence} · Model: {model_short})"
        else:
            ai_badge = ""
        lstm_badge = f" · LSTM: {lstm_prob:.0%}" if lstm_prob is not None else ""
        tg_lines.append(
            f"<b>{r['symbol']}</b>  {r['score']}/100  ${price}  RV:{rv}x{ai_badge}{lstm_badge}"
        )
        if rationale and decision == "TRADE":
            tg_lines.append(f"  <i>{rationale}</i>")

    _send_email(
        subject=f"[Trading] {len(top)} High-Score Alert — {model_label}",
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

    email_lines = [f"{len(closed_trades)} position(s) closed at take-profit target\n"]
    tg_msgs = []

    for t in closed_trades:
        email_lines.append(
            f"  {t['symbol']:6s}  +{t['pnl_pct']:.1f}%"
            f"  Entry: ${t['entry_price']:.4f}  Exit: ${t['exit_price']:.4f}"
            f"  P&L: +${t['realized_pnl']:.2f}"
        )
        tg_msgs.append(
            f"&#127919; <b>Take-Profit Hit: {t['symbol']}</b>\n"
            f"Entry: ${t['entry_price']:.4f} &#8594; Exit: ${t['exit_price']:.4f}\n"
            f"Gain: <b>+{t['pnl_pct']:.1f}%</b>  |  Profit: <b>+${t['realized_pnl']:.2f}</b>"
        )

    _send_email(
        subject=f"[Trading] \U0001f3af {len(closed_trades)} Take-Profit Hit!",
        body="\n".join(email_lines),
    )
    for msg in tg_msgs:
        _send_telegram(msg)


def send_exit_alert(closed_trades: list):
    """
    Telegram alert for risk-management exits (stop-loss, trailing stop, time stop, staleness).
    closed_trades: list of dicts with symbol, entry_price, exit_price, realized_pnl,
                   pnl_pct, close_reason, model_label
    """
    if not closed_trades:
        return

    _reason_labels = {
        "stop_loss":      ("🛑", "Stop Loss"),
        "trailing_stop":  ("📉", "Trailing Stop"),
        "time_stop":      ("⏱️", "Time Stop (2wk)"),
        "staleness_stop": ("😴", "Stale — Momentum Gone"),
    }

    for t in closed_trades:
        reason = t.get("close_reason", "exit")
        emoji, label = _reason_labels.get(reason, ("🔔", reason.replace("_", " ").title()))
        pnl_str = f"+${t['realized_pnl']:.2f}" if t['realized_pnl'] >= 0 else f"-${abs(t['realized_pnl']):.2f}"
        pct_str = f"+{t['pnl_pct']:.1f}%" if t['pnl_pct'] >= 0 else f"{t['pnl_pct']:.1f}%"
        model   = t.get("model_label", "")
        model_str = f" · {model}" if model else ""
        msg = (
            f"{emoji} <b>{label}: {t['symbol']}</b>{model_str}\n"
            f"Entry: ${t['entry_price']:.4f} &#8594; Exit: ${t['exit_price']:.4f}\n"
            f"P&L: <b>{pct_str}</b>  |  {pnl_str}"
        )
        _send_telegram(msg)
