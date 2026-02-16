from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.scanner import run_scan
from app.database import (
    init_db,
    update_returns,
    get_holding_performance,
    get_equity_curve
)

from app.data_provider import get_score_buckets

app = FastAPI()

# Initialize SQLite database
init_db()

templates = Jinja2Templates(directory="templates")


# ===============================
# HOME ROUTE
# ===============================
@app.get("/", response_class=HTMLResponse)
def home(request: Request, mode: str = "standard"):

    strict_mode = True if mode == "strict" else False

    scan_data = run_scan(strict=strict_mode)

    # Update historical returns
    update_returns()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": scan_data["results"],
            "summary": scan_data["summary"],
            "mode": mode,
        }
    )


# ===============================
# ANALYTICS ROUTE
# ===============================
@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request):

    score_buckets = get_score_buckets()
    holding_perf = get_holding_performance()
    equity_curve = get_equity_curve()

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "score_buckets": score_buckets,
            "holding_perf": holding_perf,
            "equity_curve": equity_curve,
        }
    )