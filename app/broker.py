"""
Alpaca paper trading integration.

Env vars required:
  ALPACA_API_KEY    — paper trading API key ID
  ALPACA_SECRET_KEY — paper trading secret key

All functions are silent no-ops if env vars are not set.
Paper trading base URL is hardcoded so real money is never at risk.
"""

import os
import requests

ALPACA_BASE = "https://paper-api.alpaca.markets"


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID":     os.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
        "accept":              "application/json",
        "content-type":        "application/json",
    }


def is_configured() -> bool:
    return bool(os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"))


def get_account() -> dict | None:
    """Returns Alpaca account info (buying_power, equity, status) or None."""
    if not is_configured():
        return None
    try:
        resp = requests.get(f"{ALPACA_BASE}/v2/account", headers=_headers(), timeout=10)
        return resp.json() if resp.ok else None
    except Exception:
        return None


def submit_market_order(symbol: str, notional: float) -> dict | None:
    """
    Buy `notional` dollars of `symbol` at market using fractional shares.
    Returns the Alpaca order dict on success, None on failure.
    Logs a warning if the symbol is not tradeable on Alpaca (e.g. OTC stocks).
    """
    if not is_configured():
        return None
    try:
        resp = requests.post(
            f"{ALPACA_BASE}/v2/orders",
            headers=_headers(),
            json={
                "symbol":        symbol,
                "notional":      str(round(notional, 2)),
                "side":          "buy",
                "type":          "market",
                "time_in_force": "day",
            },
            timeout=10,
        )
        if resp.ok:
            order = resp.json()
            print(f"ALPACA ORDER SUBMITTED [{symbol}]: ${notional} — order_id={order.get('id','?')}")
            return order
        print(f"ALPACA ORDER FAILED [{symbol}]: {resp.status_code} {resp.text[:200]}")
        return None
    except Exception as e:
        print(f"ALPACA ORDER ERROR [{symbol}]: {e}")
        return None


def close_position(symbol: str) -> dict | None:
    """
    Close the entire Alpaca position for `symbol`.
    Returns the Alpaca response on success, None if no position exists or on error.
    """
    if not is_configured():
        return None
    try:
        resp = requests.delete(
            f"{ALPACA_BASE}/v2/positions/{symbol}",
            headers=_headers(),
            timeout=10,
        )
        if resp.ok:
            print(f"ALPACA POSITION CLOSED [{symbol}]")
            return resp.json()
        if resp.status_code != 404:
            print(f"ALPACA CLOSE FAILED [{symbol}]: {resp.status_code} {resp.text[:200]}")
        return None
    except Exception as e:
        print(f"ALPACA CLOSE ERROR [{symbol}]: {e}")
        return None
