"""SQLite engine, session factory, and table initialization."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.models import Base

_engine: Engine | None = None
_engine_url: str | None = None


def get_engine() -> Engine:
    """
    Return a singleton engine for the current SQLite URL.
    Recreates the engine if settings change (e.g., tests patching DATA_DIR).
    """
    global _engine, _engine_url
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{settings.sqlite_path.as_posix()}"
    if _engine is None or _engine_url != url:
        _engine_url = url
        _engine = create_engine(url, connect_args={"check_same_thread": False}, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
