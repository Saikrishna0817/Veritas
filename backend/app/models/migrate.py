"""Versioned SQL migrations: ``migrations/NNNN_name.sql`` applied in order, once each, recorded in schema_migrations.

Files hold standard SQL (valid on SQLite and PostgreSQL), statements separated by ``;`` at the end of a line.
``ALTER TABLE t ADD COLUMN c ...`` is skipped when the column already exists, so a database created before
migrations were introduced upgrades cleanly.

Run by hand (or at container start):  python -m app.models.migrate
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

DIR = Path(__file__).parent / "migrations"
_ADD_COLUMN = re.compile(r"^\s*ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(\w+)", re.I)


def _statements(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";\n") if s.strip().rstrip(";")]


def migrate(db) -> list[str]:
    """Apply pending migrations; returns the versions applied."""
    db.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
    db.commit()
    done = {r["version"] for r in db.execute("SELECT version FROM schema_migrations").fetchall()}
    applied = []
    for path in sorted(DIR.glob("*.sql")):
        version = path.stem
        if version in done:
            continue
        for stmt in _statements(path.read_text()):
            m = _ADD_COLUMN.match(stmt)
            if m and m.group(2) in db.columns(m.group(1)):
                continue
            db.execute(stmt.rstrip(";"))
        db.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)", (version, datetime.now(timezone.utc).isoformat()))
        db.commit()
        applied.append(version)
    return applied


if __name__ == "__main__":
    from app.models import database
    print("applied:", migrate(database.connection()) or "nothing (up to date)")
