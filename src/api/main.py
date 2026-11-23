"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import benchmark, consensus, metrics
from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.storage.database import init_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )

    # Initialize database
    await init_db()
    logger.info("database_initialized")

    yield

    logger.info("application_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Multi-LLM orchestration system with consensus mechanism "
            "for reliable answers and objective benchmarking."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(consensus.router, prefix="/api/v1", tags=["consensus"])
    app.include_router(benchmark.router, prefix="/api/v1", tags=["benchmark"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])

    @app.get("/", tags=["health"])
    async def root() -> dict[str, str]:
        return {"status": "ok", "app": settings.app_name, "version": settings.app_version}

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    return app


app = create_app()
