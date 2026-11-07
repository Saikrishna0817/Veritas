"""
SQLite Persistence Layer — AI Trust Forensics Platform v2.2
Stores all analysis results so they survive server restarts.
Thread-safe, uses WAL mode for concurrent reads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.config import settings

# DB lives next to the backend package
DB_PATH = Path(settings.sqlite_path) if settings.sqlite_path else Path(__file__).parent.parent.parent / "forensics_results.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local connection (SQLite is not thread-safe across threads)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS analysis_results (
            id          TEXT PRIMARY KEY,
            source      TEXT NOT NULL,          -- 'demo' | 'upload' | 'model_scan'
            filename    TEXT,
            verdict     TEXT,
            score       REAL,
            attack_type TEXT,
            detection_mode TEXT,
            n_samples   INTEGER,
            elapsed_ms  REAL,
            full_json   TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_source ON analysis_results(source);
        CREATE INDEX IF NOT EXISTS idx_created ON analysis_results(created_at DESC);

        CREATE TABLE IF NOT EXISTS model_scans (
            id              TEXT PRIMARY KEY,
            model_filename  TEXT NOT NULL,
            dataset_filename TEXT,
            model_type      TEXT,
            verdict         TEXT,
            score           REAL,
            attack_type     TEXT,
            n_samples       INTEGER,
            full_json       TEXT NOT NULL,
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_model_created ON model_scans(created_at DESC);

        CREATE TABLE IF NOT EXISTS audit_events (
            id           TEXT PRIMARY KEY,
            actor_id     TEXT NOT NULL,
            action       TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id  TEXT,
            details_json TEXT NOT NULL,
            created_at   TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_events(actor_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS users (
            id              TEXT PRIMARY KEY,
            username        TEXT NOT NULL UNIQUE,
            password_hash   TEXT NOT NULL,
            role            TEXT NOT NULL,
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

        CREATE TABLE IF NOT EXISTS defense_actions (
            action_id        TEXT PRIMARY KEY,
            action           TEXT NOT NULL,
            samples_affected INTEGER,
            suspicion_score  REAL,
            reason           TEXT,
            details_json     TEXT NOT NULL,
            created_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_defense_created ON defense_actions(created_at DESC);

        CREATE TABLE IF NOT EXISTS hitl_cases (
            case_id          TEXT PRIMARY KEY,
            suspicion_score  REAL,
            n_samples        INTEGER,
            status           TEXT NOT NULL,
            details_json     TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            deadline         TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_hitl_created ON hitl_cases(created_at DESC);

        CREATE TABLE IF NOT EXISTS hitl_decisions (
            id               TEXT PRIMARY KEY,
            case_id          TEXT NOT NULL,
            decision         TEXT NOT NULL,
            reviewer         TEXT NOT NULL,
            decided_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_hitl_decided ON hitl_decisions(decided_at DESC);
    """
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS usage_daily (
            owner_id TEXT NOT NULL, day TEXT NOT NULL, kind TEXT NOT NULL, amount INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (owner_id, day, kind))"""
    )
    for table in ("analysis_results", "model_scans", "defense_actions", "hitl_cases"):
        _add_owner_column(conn, table)
    conn.commit()
    _ensure_bootstrap_admin(conn)


# Reads take an explicit owner so a forgotten filter is a TypeError, not a leak. Rows saved before owner_id existed
# have owner NULL and are visible only with ALL_OWNERS (admins).
ALL_OWNERS = "*"


def _add_owner_column(conn, table: str) -> None:
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if "owner_id" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN owner_id TEXT")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_owner ON {table}(owner_id)")


def _scope(owner_id: str) -> tuple[str, tuple]:
    """(' AND owner_id = ?', (owner,)) or ('', ()) for ALL_OWNERS."""
    return ("", ()) if owner_id == ALL_OWNERS else (" AND owner_id = ?", (owner_id,))


def save_result(result: Dict[str, Any], source: str, filename: str = None, owner_id: str | None = None) -> str:
    """Persist an analysis result. Returns the stored ID."""
    rid = result.get("job_id") or result.get("dataset_id") or (result.get("dataset_info") or {}).get("dataset_id") or str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO analysis_results
            (id, source, filename, verdict, score, attack_type, detection_mode,
             n_samples, elapsed_ms, full_json, created_at, owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            rid,
            source,
            filename or result.get("dataset_info", {}).get("filename"),
            result.get("verdict"),
            result.get("overall_suspicion_score"),
            result.get("attack_classification", {}).get("attack_type"),
            result.get("detection_mode"),
            result.get("n_samples"),
            result.get("elapsed_ms"),
            json.dumps({**result, "owner_id": owner_id}),
            datetime.now(timezone.utc).isoformat(),
            owner_id,
        ),
    )
    conn.commit()
    return rid


def save_model_scan(scan: Dict[str, Any], owner_id: str | None = None) -> str:
    """Persist a model scan result."""
    rid = scan.get("scan_id") or str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO model_scans
            (id, model_filename, dataset_filename, model_type, verdict, score,
             attack_type, n_samples, full_json, created_at, owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            rid,
            scan.get("model_filename", "unknown"),
            scan.get("dataset_filename"),
            scan.get("model_type"),
            scan.get("verdict"),
            scan.get("overall_suspicion_score"),
            scan.get("attack_classification", {}).get("attack_type"),
            scan.get("n_samples"),
            json.dumps({**scan, "owner_id": owner_id}),
            datetime.now(timezone.utc).isoformat(),
            owner_id,
        ),
    )
    conn.commit()
    return rid


def get_result(rid: str, owner_id: str) -> Optional[Dict]:
    """Fetch a single result by ID, only if ``owner_id`` owns it (or is ALL_OWNERS)."""
    clause, args = _scope(owner_id)
    row = _get_conn().execute(f"SELECT full_json FROM analysis_results WHERE id = ?{clause}", (rid, *args)).fetchone()
    return json.loads(row["full_json"]) if row else None


def get_latest(owner_id: str, source: str = None) -> Optional[Dict]:
    """Fetch the owner's most recent result, optionally filtered by source."""
    clause, args = _scope(owner_id)
    if source:
        row = _get_conn().execute(
            f"SELECT full_json FROM analysis_results WHERE source=?{clause} ORDER BY created_at DESC LIMIT 1", (source, *args)).fetchone()
    else:
        row = _get_conn().execute(
            f"SELECT full_json FROM analysis_results WHERE 1=1{clause} ORDER BY created_at DESC LIMIT 1", args).fetchone()
    return json.loads(row["full_json"]) if row else None


def get_latest_real(owner_id: str) -> Optional[Dict]:
    """Owner's most recent non-demo analysis (upload or real_dataset)."""
    clause, args = _scope(owner_id)
    row = _get_conn().execute(
        f"SELECT full_json FROM analysis_results WHERE source IN ('upload', 'real_dataset'){clause} ORDER BY created_at DESC LIMIT 1", args
    ).fetchone()
    return json.loads(row["full_json"]) if row else None


def get_history(owner_id: str, source: str = None, limit: int = 20) -> List[Dict]:
    """Recent results as lightweight summary rows."""
    clause, args = _scope(owner_id)
    where, params = ("source=?" + clause, (source, *args)) if source else ("1=1" + clause, args)
    rows = _get_conn().execute(
        f"""SELECT id, source, filename, verdict, score, attack_type, detection_mode, n_samples, elapsed_ms, created_at, owner_id
            FROM analysis_results WHERE {where} ORDER BY created_at DESC LIMIT ?""", (*params, limit)).fetchall()
    return [dict(r) for r in rows]


def get_model_scan_history(owner_id: str, limit: int = 20) -> List[Dict]:
    """Recent model scan summaries."""
    clause, args = _scope(owner_id)
    rows = _get_conn().execute(
        f"""SELECT id, model_filename, dataset_filename, model_type, verdict, score, attack_type, n_samples, created_at, owner_id
            FROM model_scans WHERE 1=1{clause} ORDER BY created_at DESC LIMIT ?""", (*args, limit)).fetchall()
    return [dict(r) for r in rows]


def get_model_scan(rid: str, owner_id: str) -> Optional[Dict]:
    clause, args = _scope(owner_id)
    row = _get_conn().execute(f"SELECT full_json FROM model_scans WHERE id = ?{clause}", (rid, *args)).fetchone()
    return json.loads(row["full_json"]) if row else None


def get_stats(owner_id: str) -> Dict:
    """Aggregate statistics over the owner's stored results."""
    conn = _get_conn()
    clause, args = _scope(owner_id)
    total = conn.execute(f"SELECT COUNT(*) FROM analysis_results WHERE 1=1{clause}", args).fetchone()[0]
    by_source = conn.execute(f"SELECT source, COUNT(*) as n FROM analysis_results WHERE 1=1{clause} GROUP BY source", args).fetchall()
    by_verdict = conn.execute(f"SELECT verdict, COUNT(*) as n FROM analysis_results WHERE 1=1{clause} GROUP BY verdict", args).fetchall()
    model_scans = conn.execute(f"SELECT COUNT(*) FROM model_scans WHERE 1=1{clause}", args).fetchone()[0]
    return {
        "total_analyses": total,
        "model_scans": model_scans,
        "by_source": {r["source"]: r["n"] for r in by_source},
        "by_verdict": {r["verdict"]: r["n"] for r in by_verdict},
    }


def log_audit_event(
    actor_id: str,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    details: Dict[str, Any] | None = None,
) -> str:
    """Persist a compact, non-secret audit record for user-visible actions."""
    event_id = str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO audit_events (id, actor_id, action, resource_type, resource_id, details_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            actor_id,
            action,
            resource_type,
            resource_id,
            json.dumps(details or {}),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return event_id


def get_audit_events(actor_id: str | None = None, limit: int = 50) -> List[Dict]:
    """Return recent audit records without exposing unbounded history."""
    limit = max(1, min(limit, 200))
    conn = _get_conn()
    if actor_id:
        rows = conn.execute(
            "SELECT * FROM audit_events WHERE actor_id=? ORDER BY created_at DESC LIMIT ?",
            (actor_id, limit),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM audit_events ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    events = []
    for row in rows:
        event = dict(row)
        event["details"] = json.loads(event.pop("details_json"))
        events.append(event)
    return events


def _ensure_bootstrap_admin(conn: sqlite3.Connection) -> None:
    """Seed the env-configured administrator if missing; purge the legacy default user."""
    from app.core.security import hash_password, verify_password

    # Older versions seeded user/user123. Remove it if it still has that password.
    legacy = conn.execute("SELECT password_hash FROM users WHERE id = 'user-regular-001'").fetchone()
    if legacy and verify_password("user123", legacy["password_hash"]):
        conn.execute("DELETE FROM users WHERE id = 'user-regular-001'")

    if settings.admin_username and settings.admin_password:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (settings.admin_username,)).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    settings.admin_username,
                    settings.admin_username,
                    hash_password(settings.admin_password),
                    "admin",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    conn.commit()


def get_user_by_username(username: str) -> Optional[Dict]:
    conn = _get_conn()
    row = conn.execute("SELECT id, username, password_hash, role, created_at FROM users WHERE username=?", (username,)).fetchone()
    return dict(row) if row else None


def create_user(username: str, password_hash: str, role: str = "analyst") -> str:
    user_id = str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO users (id, username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, username, password_hash, role, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return user_id


def list_users(limit: int = 50) -> List[Dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY created_at DESC LIMIT ?",
        (max(1, min(limit, 200)),),
    ).fetchall()
    return [dict(row) for row in rows]


def delete_user(user_id: str) -> bool:
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    return cursor.rowcount > 0


def delete_result(result_id: str, owner_id: str) -> bool:
    conn = _get_conn()
    clause, args = _scope(owner_id)
    cur1 = conn.execute(f"DELETE FROM analysis_results WHERE id = ?{clause}", (result_id, *args))
    cur2 = conn.execute(f"DELETE FROM model_scans WHERE id = ?{clause}", (result_id, *args))
    conn.commit()
    return (cur1.rowcount + cur2.rowcount) > 0




# ── Defense & HITL Persistence ────────────────────────────────────────────────

def save_defense_action(action: Dict[str, Any]) -> None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO defense_actions
        (action_id, action, samples_affected, suspicion_score, reason, details_json, created_at, owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            action.get("action_id", str(uuid.uuid4())),
            action.get("action", "monitor"),
            action.get("samples_affected", 0),
            action.get("suspicion_score", 0.0),
            action.get("reason", ""),
            json.dumps(action),
            action.get("timestamp", datetime.now(timezone.utc).isoformat()),
            action.get("owner_id"),
        ),
    )
    conn.commit()


def get_defense_actions(owner_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    clause, args = _scope(owner_id)
    rows = _get_conn().execute(
        f"SELECT details_json FROM defense_actions WHERE 1=1{clause} ORDER BY created_at DESC LIMIT ?",
        (*args, max(1, min(limit, 200))),
    ).fetchall()
    return [json.loads(row["details_json"]) for row in rows]


def save_hitl_case(case: Dict[str, Any]) -> None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO hitl_cases
        (case_id, suspicion_score, n_samples, status, details_json, created_at, deadline, owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case["case_id"],
            case.get("suspicion_score", 0.0),
            case.get("n_samples", 0),
            case.get("status", "pending"),
            json.dumps(case),
            case.get("created_at", datetime.now(timezone.utc).isoformat()),
            case.get("deadline", ""),
            case.get("owner_id"),
        ),
    )
    conn.commit()


def get_pending_hitl_cases(owner_id: str) -> List[Dict[str, Any]]:
    clause, args = _scope(owner_id)
    rows = _get_conn().execute(
        f"SELECT details_json FROM hitl_cases WHERE status = 'pending'{clause} ORDER BY created_at DESC", args
    ).fetchall()
    return [json.loads(row["details_json"]) for row in rows]


def save_hitl_decision(decision: Dict[str, Any]) -> None:
    conn = _get_conn()
    decision_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO hitl_decisions (id, case_id, decision, reviewer, decided_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            decision["case_id"],
            decision["decision"],
            decision.get("reviewer", "analyst"),
            decision.get("decided_at", datetime.now(timezone.utc).isoformat()),
        ),
    )
    # Also update case status in hitl_cases table
    conn.execute(
        "UPDATE hitl_cases SET status = 'resolved' WHERE case_id = ?",
        (decision["case_id"],),
    )
    conn.commit()


def get_hitl_decisions(owner_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    clause, args = _scope(owner_id)
    rows = _get_conn().execute(
        f"""SELECT d.case_id, d.decision, d.reviewer, d.decided_at FROM hitl_decisions d
            JOIN hitl_cases c ON c.case_id = d.case_id WHERE 1=1{clause.replace('owner_id', 'c.owner_id')}
            ORDER BY d.decided_at DESC LIMIT ?""",
        (*args, max(1, min(limit, 200))),
    ).fetchall()
    return [dict(row) for row in rows]


# ── Usage quotas ──────────────────────────────────────────────────────────────

def charge_usage(owner_id: str, kind: str, amount: int, limit: int) -> tuple[bool, int]:
    """Add ``amount`` to today's counter if that keeps it within ``limit`` (0 = unlimited).
    Returns (allowed, used_today). The check and the increment are one transaction."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = _get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT amount FROM usage_daily WHERE owner_id=? AND day=? AND kind=?", (owner_id, day, kind)).fetchone()
        used = row["amount"] if row else 0
        if limit and used + amount > limit:
            conn.execute("ROLLBACK")
            return False, used
        conn.execute(
            """INSERT INTO usage_daily (owner_id, day, kind, amount) VALUES (?, ?, ?, ?)
               ON CONFLICT(owner_id, day, kind) DO UPDATE SET amount = amount + excluded.amount""",
            (owner_id, day, kind, amount),
        )
        conn.execute("COMMIT")
        return True, used + amount
    except Exception:
        conn.execute("ROLLBACK")
        raise


def get_usage(owner_id: str) -> Dict[str, int]:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = _get_conn().execute("SELECT kind, amount FROM usage_daily WHERE owner_id=? AND day=?", (owner_id, day)).fetchall()
    return {r["kind"]: r["amount"] for r in rows}
