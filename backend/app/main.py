"""
VARUNA AI — FastAPI application entry point.

Wires together configuration, middleware, and the versioned API router.
Run locally with:  uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown hooks.

    Later phases will load trained model artifacts and open DB pools here,
    so that models are loaded once per process rather than per request.
    """
    configure_logging()
    logger.info("VARUNA AI backend starting", extra={"env": settings.ENVIRONMENT})
    yield
    logger.info("VARUNA AI backend shutting down")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Explainable AI-based extreme rainfall intelligence and early warning "
        "system using satellite data. Built for SIH problem statement SIH260006 (ISRO)."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: the React dashboard runs on a separate origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_PREFIX)
