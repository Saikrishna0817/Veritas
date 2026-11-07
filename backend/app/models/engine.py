"""Database connections for SQLite (dev, tests, single-box installs) and PostgreSQL (production).

Queries are written once with ``?`` placeholders and standard SQL (``ON CONFLICT`` upserts, ``RETURNING``);
this module only translates the placeholder style and gives both drivers the same small interface:
``execute(sql, params) -> cursor`` (rows index by name and convert with ``dict()``), ``commit``, ``rollback``.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Sequence


class Database:
    """One live connection plus the dialect it speaks. Not shared between threads."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.dialect = "postgres" if url.startswith(("postgres://", "postgresql://")) else "sqlite"
        self._conn = self._connect()

    # -- connection -------------------------------------------------------------------------------
    def _connect(self):
        if self.dialect == "postgres":
            import psycopg
            from psycopg.rows import dict_row
            return psycopg.connect(self.url, row_factory=dict_row, autocommit=False, connect_timeout=10)
        path = self.url.removeprefix("sqlite:///")
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    @property
    def closed(self) -> bool:
        if self.dialect == "postgres":
            return self._conn.closed or self._conn.broken
        try:
            self._conn.execute("SELECT 1")
            return False
        except sqlite3.ProgrammingError:
            return True

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # -- queries ----------------------------------------------------------------------------------
    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.dialect == "postgres" else sql

    def execute(self, sql: str, params: Sequence[Any] = ()):
        try:
            return self._conn.execute(self._sql(sql), tuple(params))
        except Exception:
            if self.dialect == "postgres":
                self._conn.rollback()      # a failed statement poisons a Postgres transaction
            raise

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    # -- introspection (used by migrations) -----------------------------------------------------
    def columns(self, table: str) -> set[str]:
        if self.dialect == "postgres":
            rows = self.execute("SELECT column_name AS name FROM information_schema.columns WHERE table_name = ?", (table,)).fetchall()
        else:
            rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r["name"] for r in rows}
