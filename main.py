# ai_trading_mvp/main.py

import yfinance as yf

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.scanner import run_scan
from app.database import (
    init_db,
    seed_users,
    save_scan,
    update_returns,
    get_score_buckets,
    get_holding_performance,
    get_equity_curve,
    create_user,
    authenticate_user,
    open_trade,
    close_trade,
    get_trade_by_id,
    get_open_positions,
    get_trade_history,
    get_portfolio_summary,
)

app = FastAPI()

# Session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key="CHANGE_THIS_SECRET_KEY"
)

# Initialize database and seed users
init_db()
seed_users()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def _fetch_current_price(symbol: str, fallback: float) -> float:
    try:
        fi = yf.Ticker(symbol).fast_info
        price = fi.last_price
        if price and price > 0:
            return float(price)
        price = fi.previous_close
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    return fallback


# ---------------- ROOT ----------------
@app.get("/")
def root():
    return RedirectResponse("/login")


# ---------------- LOGIN ----------------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):

    if not authenticate_user(username, password):
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Invalid credentials"
            }
        )

    request.session["user"] = username
    return RedirectResponse("/dashboard", status_code=303)


# ---------------- DASHBOARD ----------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, mode: str = "standard", trade_error: str = ""):

    if "user" not in request.session:
        return RedirectResponse("/login")

    strict_mode = mode == "strict"

    scan_data = run_scan(strict=strict_mode)
    save_scan(scan_data["results"], mode)
    update_returns()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": scan_data["results"],
            "summary": scan_data["summary"],
            "mode": mode,
            "trade_error": trade_error,
            "user": request.session["user"]
        }
    )


# ---------------- ANALYTICS ----------------
@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request):

    if "user" not in request.session:
        return RedirectResponse("/login")

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "score_buckets": get_score_buckets(),
            "holding_perf": get_holding_performance(),
            "equity_curve": get_equity_curve(),
            "user": request.session["user"]
        }
    )


# ---------------- TRADES ----------------
@app.get("/trades", response_class=HTMLResponse)
def trades_page(request: Request):

    if "user" not in request.session:
        return RedirectResponse("/login")

    positions = get_open_positions()
    history = get_trade_history()
    summary = get_portfolio_summary()

    open_value = 0.0
    for pos in positions:
        current_price = _fetch_current_price(pos["symbol"], pos["entry_price"])
        unrealized_pnl = (current_price - pos["entry_price"]) * pos["shares"]
        current_value = current_price * pos["shares"]
        open_value += current_value

        pos["current_price"] = round(current_price, 4)
        pos["unrealized_pnl"] = round(unrealized_pnl, 2)
        pos["current_value"] = round(current_value, 2)
        pos["pnl_pct"] = round(
            (current_price - pos["entry_price"]) / pos["entry_price"] * 100, 2
        )

    total_value = summary["cash"] + open_value

    return templates.TemplateResponse(
        "trades.html",
        {
            "request": request,
            "user": request.session["user"],
            "positions": positions,
            "history": history,
            "summary": {
                **summary,
                "open_value": round(open_value, 2),
                "total_value": round(total_value, 2),
                "total_pnl": round(total_value - 10000.0, 2),
            }
        }
    )


@app.post("/trade/buy")
def trade_buy(request: Request, symbol: str = Form(...), price: float = Form(...)):

    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    result = open_trade(symbol, price)

    if result is None:
        return RedirectResponse("/dashboard?trade_error=insufficient_funds", status_code=303)

    return RedirectResponse("/trades", status_code=303)


@app.post("/trade/sell/{trade_id}")
def trade_sell(request: Request, trade_id: int):

    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    trade = get_trade_by_id(trade_id)
    if trade is None:
        return RedirectResponse("/trades", status_code=303)

    current_price = _fetch_current_price(trade["symbol"], trade["entry_price"])
    close_trade(trade_id, current_price)

    return RedirectResponse("/trades", status_code=303)


# ---------------- LOGOUT ----------------
@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")
