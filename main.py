rom fastapi import Form
from fastapi.responses import RedirectResponse
from app.database import (
    init_db,
    update_returns,
    get_score_buckets,
    get_holding_performance,
    get_equity_curve,
    create_user,
    authenticate_user
)
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.scanner import run_scan
from app.database import (
    init_db,
    update_returns,
    get_holding_performance,
    get_equity_curve