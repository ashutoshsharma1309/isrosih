"""
Database session management (SQLAlchemy + PostgreSQL/PostGIS).

Engine creation is lazy so the API can start (and unit tests can run)
without a database — endpoints that need the DB acquire a session via
the `get_db` dependency.
"""

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


@lru_cache
def _engine():
    return create_engine(settings.database_url, pool_pre_ping=True)


@lru_cache
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=_engine(), autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped DB session."""
    db = _session_factory()()
    try:
        yield db
    finally:
        db.close()
