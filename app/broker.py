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


def submit_market_order(symbol: str, notional: float, price: float = None) -> dict | None:
    """
    Buy `notional` dollars of `symbol` at market.
    Strategy:
      1. Try a fractional/notional order (works for fractionable stocks).
      2. If Alpaca rejects it (stock not fractionable), fall back to a
         whole-share qty order using floor(notional / price).
    Returns the Alpaca order dict on success, None on failure.
    `price` is only needed for the whole-share fallback; if omitted the
    fallback is skipped.
    """
    if not is_configured():
        return None
    try:
        # Attempt 1: notional (fractional)
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
            print(f"ALPACA ORDER SUBMITTED [{symbol}]: ${notional} notional — order_id={order.get('id','?')}")
            return order

        err_text = resp.text[:300]
        print(f"ALPACA NOTIONAL FAILED [{symbol}]: {resp.status_code} {err_text}")

        # Attempt 2: whole-share qty fallback for non-fractionable stocks
        if price and price > 0:
            import math
            qty = math.floor(notional / price)
            if qty < 1:
                print(f"ALPACA FALLBACK SKIPPED [{symbol}]: price ${price} too high for ${notional} budget")
                return None
            resp2 = requests.post(
                f"{ALPACA_BASE}/v2/orders",
                headers=_headers(),
                json={
                    "symbol":        symbol,
                    "qty":           str(qty),
                    "side":          "buy",
                    "type":          "market",
                    "time_in_force": "day",
                },
                timeout=10,
            )
            if resp2.ok:
                order = resp2.json()
                print(f"ALPACA ORDER SUBMITTED [{symbol}]: {qty} shares (whole-share fallback) — order_id={order.get('id','?')}")
                return order
            print(f"ALPACA FALLBACK FAILED [{symbol}]: {resp2.status_code} {resp2.text[:200]}")

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
