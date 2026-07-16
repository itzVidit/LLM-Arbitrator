"""
api/app.py
----------
FastAPI application factory for the LLM Output Arbitration Service.

Usage
-----
Start the server (from project root, venv active):

    uvicorn api.app:app --reload --port 8000

Or via Python:

    from api.app import create_app
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)

Endpoints
---------
  POST   /v1/arbitrate               Evaluate a single LLM response
  POST   /v1/arbitrate/batch         Evaluate up to 10 responses concurrently
  GET    /v1/arbitrations/{id}       Retrieve a stored verdict by UUID
  GET    /v1/arbitrations            Paginated verdict list (filterable)
  GET    /v1/analytics               System-wide critic behaviour analytics
  GET    /health                     Health check (DB connectivity)
  GET    /docs                       Swagger UI (auto-generated)
  GET    /redoc                      ReDoc UI (auto-generated)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import configure_logging, settings

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Lifespan — DB init / teardown
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Initialises the SQLite database on startup and closes it on shutdown.
    """
    configure_logging()
    log.info("Starting LLM Output Arbitration Service...")

    from storage.database import get_db
    db = await get_db()
    await db.init()
    log.info("Database initialised at %s", settings.storage_path)

    # Pre-compile the LangGraph graph so first request is fast
    try:
        from orchestration.graph import get_graph
        get_graph()
        log.info("LangGraph arbitration graph compiled and cached")
    except Exception as exc:
        log.warning("Graph pre-compile failed (will compile on first request): %s", exc)

    yield  # Application runs here

    await db.close()
    log.info("Database closed. Shutdown complete.")


# ─────────────────────────────────────────────────────────────────────
#  Exception handlers
# ─────────────────────────────────────────────────────────────────────

async def _http_exception_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


async def _validation_exception_handler(request: Request, exc) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": exc.errors(),
            "status_code": 422,
        },
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "status_code": 500},
    )


# ─────────────────────────────────────────────────────────────────────
#  Request logging middleware
# ─────────────────────────────────────────────────────────────────────

async def _logging_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "%s %s  status=%d  latency=%.0fms",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


# ─────────────────────────────────────────────────────────────────────
#  Application factory
# ─────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Registers:
      - CORS middleware (allow all origins by default; restrict in production)
      - Request-logging middleware
      - HTTPException and ValidationError handlers
      - API router mounted at /v1
      - /health endpoint
    """
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    app = FastAPI(
        title="LLM Output Arbitration API",
        description=(
            "A multi-agent AI evaluation pipeline that routes any LLM response "
            "through five independent specialist critics and synthesises their "
            "findings into a single confidence-scored verdict.\n\n"
            "## Quick start\n"
            "```bash\n"
            "curl -X POST http://localhost:8000/v1/arbitrate \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"original_prompt\": \"What is 2+2?\", "
            "\"llm_response\": \"4\"}'\n"
            "```\n\n"
            "## Critics\n"
            "| Critic | Model | Evaluates |\n"
            "|--------|-------|-----------|\n"
            "| Accuracy | Gemini 2.5 Flash | Facts, hallucinations |\n"
            "| Logic | DeepSeek R1 70B | Reasoning chains |\n"
            "| Completeness | Gemma 3 (local) | Coverage, omissions |\n"
            "| Safety | Qwen3 32B | Harm, prompt injection |\n"
            "| Style | Phi-4 Mini (local) | Grammar, clarity |\n"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging ───────────────────────────────────────────────
    app.middleware("http")(_logging_middleware)

    # ── Exception handlers ────────────────────────────────────────────
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    # ── Routes ────────────────────────────────────────────────────────
    from api.routes import router
    app.include_router(router, prefix="/v1")

    # ── /health ───────────────────────────────────────────────────────
    @app.get(
        "/health",
        summary="Health check",
        description="Returns service health and basic database statistics.",
        tags=["System"],
        response_model=dict,
    )
    async def health() -> dict[str, Any]:
        from storage.database import get_db
        db = await get_db()
        try:
            total = await db.count_verdicts()
            db_ok = True
        except Exception as exc:
            log.warning("Health check DB query failed: %s", exc)
            total = -1
            db_ok = False

        return {
            "status": "ok" if db_ok else "degraded",
            "database": "ok" if db_ok else "error",
            "total_arbitrations": total,
            "version": "1.0.0",
        }

    log.info("FastAPI application created  routes=%d", len(app.routes))
    return app


# ─────────────────────────────────────────────────────────────────────
#  Module-level app instance (for `uvicorn api.app:app`)
# ─────────────────────────────────────────────────────────────────────

app = create_app()
