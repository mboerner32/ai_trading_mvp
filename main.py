# ai_trading_mvp/main.py

import asyncio
import threading
import pytz
import yfinance as yf

from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.scanner import run_scan
from app.alerts import send_scan_alert, send_take_profit_alert
from app.backfill import build_historical_dataset
from app.ai_agent import (
    recommend_position_size,
    analyze_and_optimize_weights,
    analyze_chart_feedback,
    reprocess_chart_analysis,
    synthesize_feedback_hypotheses,
    synthesize_historical_hypothesis,
    synthesize_combined_hypothesis,
    optimize_complex_ai_weights,
)
from app.scoring_engine import DEFAULT_SQUEEZE_WEIGHTS
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
    get_optimization_data,
    save_feedback,
    update_feedback_analysis,
    get_recent_feedback,
    get_all_feedback,
    save_hypothesis,
    get_hypothesis,
    save_squeeze_weights,
    get_squeeze_weights,
    save_scan_cache,
    get_scan_cache,
    save_weight_changelog,
    get_weight_changelog,
    add_to_watchlist,
    remove_from_watchlist,
    get_watchlist,
    get_risk_metrics,
    save_historical_scans,
    set_backfill_status,
    get_backfill_status,
    get_historical_count,
    get_historical_examples,
    get_sizing_stats,
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

# In-process flag so the backfill guard is reliable even if DB status got stuck
_backfill_running = False

# ---------------- SCHEDULER ----------------
def _autoclose_take_profit() -> list:
    """Close open positions that have hit the 20% take-profit target."""
    positions = get_open_positions()
    closed = []
    for pos in positions:
        price = _fetch_current_price(pos["symbol"], pos["entry_price"])
        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
        if pnl_pct >= 20:
            result = close_trade(pos["trade_id"], price)
            if result:
                closed.append({
                    "symbol":       pos["symbol"],
                    "entry_price":  pos["entry_price"],
                    "exit_price":   round(price, 4),
                    "realized_pnl": result["realized_pnl"],
                    "pnl_pct":      round(pnl_pct, 1),
                })
    if closed:
        send_take_profit_alert(closed)
    return closed


def _scheduled_scan():
    """Auto-run all scan modes at 9:45am ET Mon–Fri and refresh cache."""
    print("SCHEDULER: Running scheduled morning scan...")
    for mode in ["squeeze", "strict", "standard"]:
        try:
            data = run_scan(mode=mode)
            save_scan(data["results"], mode)
            save_scan_cache(mode, data["results"], data["summary"])
            print(f"SCHEDULER: {mode} scan complete ({len(data['results'])} results)")
            if mode == "squeeze":
                send_scan_alert(data["results"], "Complex + AI")
        except Exception as e:
            print(f"SCHEDULER: {mode} scan failed — {e}")
    _autoclose_take_profit()
    update_returns()


def _premarket_scan():
    """Pre-market scan at 8:30am ET — squeeze mode with relaxed Finviz filters."""
    print("SCHEDULER: Running pre-market scan...")
    try:
        data = run_scan(mode="squeeze", premarket=True)
        save_scan_cache("squeeze", data["results"], data["summary"])
        send_scan_alert(data["results"], "Complex + AI (pre-market)")
        print(f"SCHEDULER: pre-market scan complete ({len(data['results'])} results)")
    except Exception as e:
        print(f"SCHEDULER: pre-market scan failed — {e}")


_scheduler = BackgroundScheduler(timezone=pytz.timezone("America/New_York"))
_scheduler.add_job(_scheduled_scan, "cron", day_of_week="mon-fri", hour=9, minute=45)
_scheduler.add_job(_premarket_scan,  "cron", day_of_week="mon-fri", hour=8, minute=30)
_scheduler.start()

# Log API key status at startup so it's visible in Render logs
import os as _os
if _os.environ.get("ANTHROPIC_API_KEY"):
    print("STARTUP: ANTHROPIC_API_KEY is set ✓")
else:
    print("STARTUP: WARNING — ANTHROPIC_API_KEY is NOT set. AI features will fail.")

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


# ---------------- HEALTH CHECK ----------------
@app.get("/health")
@app.head("/health")
def health():
    return Response(status_code=200)


# ---------------- ROOT ----------------
@app.get("/")
@app.head("/")
def root():
    return RedirectResponse("/login")


# ---------------- LOGIN ----------------
@app.head("/login")
def login_head():
    return Response(status_code=200)


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
def dashboard(request: Request, mode: str = "squeeze", trade_error: str = "",
              refresh: str = ""):

    if "user" not in request.session:
        return RedirectResponse("/login")

    # Serve from cache (15-min TTL) unless user clicked Refresh
    cached = None if refresh else get_scan_cache(mode)
    if cached:
        scan_data = {"results": cached["results"], "summary": cached["summary"]}
        cache_age = cached["cache_age_minutes"]
    else:
        scan_data = run_scan(mode=mode)
        save_scan(scan_data["results"], mode)
        update_returns()
        save_scan_cache(mode, scan_data["results"], scan_data["summary"])
        cache_age = 0

    available_cash = get_portfolio_summary()["cash"]
    hypothesis_data = get_hypothesis()
    hypothesis_text = hypothesis_data["content"] if hypothesis_data else None
    sizing_stats    = get_sizing_stats()

    # Parallel AI position sizing — each stock gets historical calibration injected
    def _size(stock):
        stock["ai_rec"] = recommend_position_size(
            stock, available_cash, hypothesis_text, sizing_stats
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(_size, scan_data["results"]))

    ai_weights_info = get_squeeze_weights() if mode == "squeeze" else None

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": scan_data["results"],
            "summary": scan_data["summary"],
            "mode": mode,
            "trade_error": trade_error,
            "user": request.session["user"],
            "ai_weights_info": ai_weights_info,
            "cache_age": cache_age,
            "sizing_calibrated": sizing_stats is not None,
            "sizing_total": sizing_stats["total"] if sizing_stats else 0,
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
            "weight_changelog": get_weight_changelog(),
            "risk_metrics": get_risk_metrics(),
            "historical_count": get_historical_count(),
            "backfill_status": get_backfill_status(),
            "user": request.session["user"],
        }
    )


# ---------------- TRADES ----------------
TARGET_GAIN = 0.20  # 20% profit target — auto-sell when hit


@app.get("/trades", response_class=HTMLResponse)
def trades_page(request: Request):

    if "user" not in request.session:
        return RedirectResponse("/login")

    # --- Auto-close any positions that have hit the 20% target ---
    auto_closed = []
    for pos in get_open_positions():
        target_price = pos["entry_price"] * (1 + TARGET_GAIN)
        current_price = _fetch_current_price(pos["symbol"], pos["entry_price"])
        if current_price >= target_price:
            result = close_trade(pos["trade_id"], current_price)
            if result:
                auto_closed.append({
                    "symbol": pos["symbol"],
                    "exit_price": round(current_price, 4),
                    "realized_pnl": round(result["realized_pnl"], 2),
                })

    # Reload after any auto-closes
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
        pos["target_price"] = round(pos["entry_price"] * (1 + TARGET_GAIN), 4)
        pos["unrealized_pnl"] = round(unrealized_pnl, 2)
        pos["current_value"] = round(current_value, 2)
        pos["pnl_pct"] = round(
            (current_price - pos["entry_price"]) / pos["entry_price"] * 100, 2
        )
        pos["to_target_pct"] = round(
            (pos["target_price"] - current_price) / current_price * 100, 2
        )

    total_value = summary["cash"] + open_value

    return templates.TemplateResponse(
        "trades.html",
        {
            "request": request,
            "user": request.session["user"],
            "positions": positions,
            "history": history,
            "auto_closed": auto_closed,
            "watchlist": get_watchlist(),
            "summary": {
                **summary,
                "open_value": round(open_value, 2),
                "total_value": round(total_value, 2),
                "total_pnl": round(total_value - 10000.0, 2),
            }
        }
    )


@app.post("/trade/buy")
def trade_buy(
    request: Request,
    symbol: str = Form(...),
    price: float = Form(...),
    position_size: float = Form(1000.0),
    notes: str = Form(""),
):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    result = open_trade(symbol, price, position_size, notes)

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


# ---------------- WEIGHT OPTIMIZER ----------------
@app.post("/optimize", response_class=HTMLResponse)
def optimize_weights(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    opt_data = get_optimization_data()
    if not opt_data or opt_data["total_trades"] < 5:
        analysis = (
            "Not enough backtested data yet — need at least 5 scans with known returns. "
            "Returns are filled in automatically once scans are 3+ trading days old."
        )
    else:
        analysis = analyze_and_optimize_weights(opt_data)

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "score_buckets": get_score_buckets(),
            "holding_perf": get_holding_performance(),
            "equity_curve": get_equity_curve(),
            "weight_changelog": get_weight_changelog(),
            "risk_metrics": get_risk_metrics(),
            "historical_count": get_historical_count(),
            "backfill_status": get_backfill_status(),
            "user": request.session["user"],
            "weight_analysis": analysis,
            "opt_data": opt_data,
        }
    )


# ---------------- FEEDBACK ----------------
@app.get("/feedback", response_class=HTMLResponse)
def feedback_page(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login")

    manual_count = len(get_all_feedback())
    hist_count   = get_historical_count()
    return templates.TemplateResponse(
        "feedback.html",
        {
            "request": request,
            "user": request.session["user"],
            "recent_feedback": get_recent_feedback(limit=10),
            "hypothesis": get_hypothesis(),
            "weights_info": get_squeeze_weights(),
            "historical_count": hist_count,
            "manual_count": manual_count,
            "historical_examples": get_historical_examples(limit=20),
            "reanalyzing": request.query_params.get("reanalyzing") == "1",
            "syncing": request.query_params.get("syncing") == "1",
        }
    )




@app.post("/sync-all-sources")
def sync_all_sources(request: Request, background_tasks: BackgroundTasks):
    """Synthesize hypothesis from both manual submissions and historical scan data."""
    if "user" not in request.session:
        return RedirectResponse("/login")

    def _run():
        all_feedback = get_all_feedback()
        opt_data = get_optimization_data()
        hist_count = get_historical_count()
        text = synthesize_combined_hypothesis(all_feedback, opt_data, hist_count)
        total = len(all_feedback) + hist_count
        if text:
            save_hypothesis(text, total)

    background_tasks.add_task(_run)
    return RedirectResponse("/feedback?syncing=1", status_code=303)


@app.post("/reanalyze-feedback")
def reanalyze_feedback(request: Request, background_tasks: BackgroundTasks):
    if "user" not in request.session:
        return RedirectResponse("/login")

    def _run():
        entries = get_all_feedback()
        for entry in entries:
            if not entry.get("chart_analysis"):
                continue
            new_analysis = reprocess_chart_analysis(
                entry["chart_analysis"],
                entry.get("user_text", ""),
                entry.get("symbol")
            )
            update_feedback_analysis(entry["id"], new_analysis)
        # Re-synthesize hypothesis with updated analyses
        all_feedback = get_all_feedback()
        if all_feedback:
            text = synthesize_feedback_hypotheses(all_feedback)
            if text:
                save_hypothesis(text, len(all_feedback))

    background_tasks.add_task(_run)
    return RedirectResponse("/feedback?reanalyzing=1", status_code=303)


def _run_hypothesis_and_weights(all_feedback: list):
    """Background task: synthesize combined hypothesis from all sources then optimize weights."""
    try:
        opt_data = get_optimization_data()
        hist_count = get_historical_count()
        hypothesis_text = synthesize_combined_hypothesis(all_feedback, opt_data, hist_count)
        hypothesis_content = None
        if hypothesis_text:
            save_hypothesis(hypothesis_text, len(all_feedback) + hist_count)
            hypothesis_content = hypothesis_text

        weights_data = get_squeeze_weights()
        current_weights = weights_data["weights"] if weights_data else DEFAULT_SQUEEZE_WEIGHTS.copy()
        opt_result = optimize_complex_ai_weights(opt_data, all_feedback, current_weights, hypothesis_content)
        if "error" not in opt_result:
            save_squeeze_weights(
                opt_result["weights"],
                opt_result["rationale"],
                opt_result["suggestions"],
                opt_result["summary"],
            )
    except Exception:
        pass


@app.post("/feedback", response_class=HTMLResponse)
async def submit_feedback(
    request: Request,
    background_tasks: BackgroundTasks,
):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    form = await request.form()
    symbol = form.get("symbol", "")
    user_text = form.get("user_text", "")
    charts = form.getlist("charts")

    chart_analysis = ""
    valid_charts = [c for c in charts if hasattr(c, "filename") and c.filename]
    if valid_charts:
        # Read all file bytes first
        chart_data = []
        for chart in valid_charts:
            image_bytes = await chart.read()
            fn = chart.filename.lower()
            if fn.endswith(".png"):
                media_type = "image/png"
            elif fn.endswith(".gif"):
                media_type = "image/gif"
            elif fn.endswith(".webp"):
                media_type = "image/webp"
            else:
                media_type = "image/jpeg"
            chart_data.append((image_bytes, media_type))

        # Analyze all charts in parallel
        loop = asyncio.get_event_loop()
        sym = symbol.strip() or None
        tasks = [
            loop.run_in_executor(None, analyze_chart_feedback, img, mt, user_text, sym)
            for img, mt in chart_data
        ]
        results = await asyncio.gather(*tasks)

        if len(valid_charts) > 1:
            analyses = [f"--- Chart {i+1} of {len(results)} ---\n{r}" for i, r in enumerate(results)]
        else:
            analyses = list(results)
        chart_analysis = "\n\n".join(analyses)

    save_feedback(symbol.upper().strip(), user_text, chart_analysis)

    # Hypothesis synthesis + weight optimization run in background
    # save_feedback already committed, so get_all_feedback() includes this submission
    all_feedback = get_all_feedback()
    background_tasks.add_task(_run_hypothesis_and_weights, all_feedback)

    hypothesis_data = get_hypothesis()

    manual_count = len(all_feedback)
    hist_count   = get_historical_count()
    return templates.TemplateResponse(
        "feedback.html",
        {
            "request": request,
            "user": request.session["user"],
            "recent_feedback": get_recent_feedback(limit=10),
            "hypothesis": hypothesis_data,
            "weights_info": get_squeeze_weights(),
            "historical_count": hist_count,
            "manual_count": manual_count,
            "historical_examples": get_historical_examples(limit=20),
            "submitted": True,
            "chart_analysis": chart_analysis,
        }
    )


# ---------------- COMPLEX + AI OPTIMIZER ----------------
@app.post("/optimize-complex", response_class=HTMLResponse)
def optimize_complex(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    opt_data = get_optimization_data()
    all_feedback = get_all_feedback()
    hypothesis_data = get_hypothesis()

    weights_data = get_squeeze_weights()
    current_weights = weights_data["weights"] if weights_data else DEFAULT_SQUEEZE_WEIGHTS.copy()

    opt_result = optimize_complex_ai_weights(
        opt_data, all_feedback, current_weights,
        hypothesis_data["content"] if hypothesis_data else None
    )

    if "error" not in opt_result:
        save_squeeze_weights(
            opt_result["weights"],
            opt_result["rationale"],
            opt_result["suggestions"],
            opt_result["summary"],
        )
        save_weight_changelog(
            opt_result.get("summary", ""),
            opt_result.get("rationale", ""),
            opt_result["weights"],
        )

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "score_buckets": get_score_buckets(),
            "holding_perf": get_holding_performance(),
            "equity_curve": get_equity_curve(),
            "weight_changelog": get_weight_changelog(),
            "risk_metrics": get_risk_metrics(),
            "historical_count": get_historical_count(),
            "backfill_status": get_backfill_status(),
            "user": request.session["user"],
            "complex_ai_result": opt_result,
            "complex_ai_weights": get_squeeze_weights(),
        }
    )


# ---------------- APPLY MODEL UPDATE ----------------
@app.post("/apply-model-update")
def apply_model_update(request: Request):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    all_feedback = get_all_feedback()
    hypothesis_data = get_hypothesis()
    opt_data = get_optimization_data()
    weights_data = get_squeeze_weights()
    current_weights = weights_data["weights"] if weights_data else DEFAULT_SQUEEZE_WEIGHTS.copy()

    opt_result = optimize_complex_ai_weights(
        opt_data, all_feedback, current_weights,
        hypothesis_data["content"] if hypothesis_data else None
    )

    if "error" not in opt_result:
        save_squeeze_weights(
            opt_result["weights"],
            opt_result["rationale"],
            opt_result["suggestions"],
            opt_result["summary"],
        )
        save_weight_changelog(
            opt_result.get("summary", ""),
            opt_result.get("rationale", ""),
            opt_result["weights"],
        )

    return RedirectResponse("/analytics", status_code=303)


# ---------------- LIVE PRICE API ----------------
@app.get("/api/positions/prices")
def api_position_prices(request: Request):
    if "user" not in request.session:
        return Response(status_code=401)
    positions = get_open_positions()
    TARGET_GAIN = 0.20
    data = {}
    for pos in positions:
        current = _fetch_current_price(pos["symbol"], pos["entry_price"])
        unrealized = round((current - pos["entry_price"]) * pos["shares"], 2)
        current_value = round(current * pos["shares"], 2)
        pnl_pct = round((current - pos["entry_price"]) / pos["entry_price"] * 100, 2)
        target = round(pos["entry_price"] * (1 + TARGET_GAIN), 4)
        to_target_pct = round((target - current) / current * 100, 2)
        data[pos["trade_id"]] = {
            "current_price":  round(current, 4),
            "unrealized_pnl": unrealized,
            "current_value":  current_value,
            "pnl_pct":        pnl_pct,
            "to_target_pct":  to_target_pct,
        }
    return data


# ---------------- WATCHLIST ----------------
@app.post("/watchlist/add")
def watchlist_add(request: Request, symbol: str = Form(...), price: float = Form(0.0)):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    add_to_watchlist(symbol, price or None)
    return RedirectResponse("/trades", status_code=303)


@app.post("/watchlist/remove/{symbol}")
def watchlist_remove(request: Request, symbol: str):
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)
    remove_from_watchlist(symbol)
    return RedirectResponse("/trades", status_code=303)


# ---------------- HISTORICAL BACKFILL ----------------
@app.post("/backfill-history")
def start_backfill(request: Request):
    global _backfill_running
    if "user" not in request.session:
        return RedirectResponse("/login", status_code=303)

    if _backfill_running:
        return RedirectResponse("/analytics?backfill=already_running", status_code=303)

    weights_data = get_squeeze_weights()
    weights = weights_data["weights"] if weights_data else None

    def _run(**kwargs):
        global _backfill_running
        _backfill_running = True
        try:
            build_historical_dataset(**kwargs)
        finally:
            _backfill_running = False

    threading.Thread(target=_run, kwargs={"weights": weights}, daemon=True).start()

    return RedirectResponse("/analytics?backfill=started", status_code=303)


@app.get("/api/backfill-status")
def api_backfill_status(request: Request):
    if "user" not in request.session:
        return Response(status_code=401)
    return get_backfill_status()


# ---------------- LOGOUT ----------------
@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")
