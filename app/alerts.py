"""
Email alert system for trade notifications.
Requires env vars: GMAIL_USER, GMAIL_APP_PASSWORD, ALERT_EMAIL_TO
All functions are no-ops if env vars are not set.
"""

import os
import smtplib
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
        print(f"Alert sent: {subject}")
    except Exception as e:
        print(f"Alert email failed: {e}")


def send_scan_alert(results: list, mode: str, min_score: int = 75):
    """
    Email top-scoring stocks from a scan.
    Only sends if at least one stock scores >= min_score.
    """
    top = [r for r in results if r.get("score", 0) >= min_score]
    if not top:
        return

    lines = [f"{len(top)} stock(s) scored {min_score}+ in {mode} scan\n"]
    for r in top[:10]:
        chk  = r.get("checklist", {})
        price = r.get("price", "?")
        rv    = chk.get("relative_volume", "?")
        lines.append(
            f"  {r['symbol']:6s}  Score:{r['score']}/100  {r.get('recommendation','')}"
            f"  ${price}  RV:{rv}x"
        )

    _send_email(
        subject=f"[Trading] {len(top)} High-Score Alert — {mode}",
        body="\n".join(lines),
    )


def send_take_profit_alert(closed_trades: list):
    """
    Email when one or more positions are auto-closed at the 20% take-profit target.
    closed_trades: list of dicts with symbol, entry_price, exit_price, realized_pnl, pnl_pct
    """
    if not closed_trades:
        return

    lines = [f"{len(closed_trades)} position(s) auto-closed at take-profit target\n"]
    for t in closed_trades:
        lines.append(
            f"  {t['symbol']:6s}  +{t['pnl_pct']:.1f}%"
            f"  Entry:${t['entry_price']}  Close:${t['exit_price']:.4f}"
            f"  P&L:+${t['realized_pnl']:.2f}"
        )

    _send_email(
        subject=f"[Trading] \U0001f3af {len(closed_trades)} Take-Profit Hit!",
        body="\n".join(lines),
    )
