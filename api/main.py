from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import lifespan
from api.models import APIResponse

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PredictIQ",
    description="Agentic AI platform for industrial predictive maintenance.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",      # Streamlit dashboard
        "http://localhost:3000",      # frontend development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------------------------
# Request timing and logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_and_time_requests(request: Request, call_next):
    """Log every request with method, path, status and duration."""
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} "
        f"({duration_ms:.2f}ms)"
    )

    return response

# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception and return it in our standard envelope."""
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "data": None,
            "error": f"{type(exc).__name__}: {str(exc)}",
            "version": "1.0.0",
        },
    )

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from api.endpoints import health, classifier, rul, fleet

app.include_router(health.router,     prefix="/api/v1", tags=["Health"])
app.include_router(classifier.router, prefix="/api/v1", tags=["Classifier"])
app.include_router(rul.router,        prefix="/api/v1", tags=["RUL"])
app.include_router(fleet.router,      prefix="/api/v1", tags=["Fleet"])