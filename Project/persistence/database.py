"""The database engine, table creation and session handling."""

import os

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from config import config
from persistence.models import Base


def create_database_directory(directory_path=None):
    """Create the folder the database file lives in."""
    folder = config.DATABASE_DIRECTORY if directory_path is None else directory_path
    os.makedirs(folder, exist_ok=True)
    return folder


create_database_directory()
engine = create_engine(config.APPLICATION_DATABASE_URL,
                       connect_args={"check_same_thread": False})
SessionFactory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def bind_database(database_url):
    """Point every session at another database, for an evaluation or a test sandbox.

    The engine and session factory are module globals that every caller reaches
    through open_database_session, so rebinding them here is enough.
    """
    global engine, SessionFactory
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    SessionFactory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    _TABLES_READY["done"] = False
    create_all_tables(engine)
    return engine


def create_all_tables(target_engine=None):
    """Create every table; idempotent. Returns the table names ensured."""
    target = engine if target_engine is None else target_engine
    Base.metadata.create_all(bind=target)
    present = set(inspect(target).get_table_names())
    return [name for name in Base.metadata.tables if name in present]


_TABLES_READY = {"done": False}


def open_database_session():
    """A session the caller must close, normally in a finally block."""
    if not _TABLES_READY["done"]:
        create_all_tables(engine)
        _TABLES_READY["done"] = True
    return SessionFactory()


def get_database_session():
    """One session per web request, closed afterwards whatever happened."""
    session = open_database_session()
    try:
        yield session
    finally:
        session.close()
