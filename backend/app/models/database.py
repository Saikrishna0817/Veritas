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

        CREATE TABLE IF NOT EXISTS redteam_simulations (
            simulation_id    TEXT PRIMARY KEY,
            attack_type      TEXT NOT NULL,
            detected         INTEGER NOT NULL,
            verdict          TEXT NOT NULL,
            suspicion_score  REAL NOT NULL,
            resilience_score REAL NOT NULL,
            details_json     TEXT NOT NULL,
            created_at       TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_redteam_created ON redteam_simulations(created_at DESC);
    """
    )
    conn.commit()
    _ensure_bootstrap_admin(conn)


def save_result(result: Dict[str, Any], source: str, filename: str = None) -> str:
    """Persist an analysis result. Returns the stored ID."""
    rid = result.get("job_id") or result.get("dataset_id") or str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO analysis_results
            (id, source, filename, verdict, score, attack_type, detection_mode,
             n_samples, elapsed_ms, full_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(result),
            datetime.now(timezone.utc).isoformat() + "Z",
        ),
    )
    conn.commit()
    return rid


def save_model_scan(scan: Dict[str, Any]) -> str:
    """Persist a model scan result."""
    rid = scan.get("scan_id") or str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO model_scans
            (id, model_filename, dataset_filename, model_type, verdict, score,
             attack_type, n_samples, full_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(scan),
            datetime.now(timezone.utc).isoformat() + "Z",
        ),
    )
    conn.commit()
    return rid


def get_result(rid: str) -> Optional[Dict]:
    """Fetch a single result by ID."""
    conn = _get_conn()
    row = conn.execute("SELECT full_json FROM analysis_results WHERE id = ?", (rid,)).fetchone()
    return json.loads(row["full_json"]) if row else None


def get_latest(source: str = None) -> Optional[Dict]:
    """Fetch the most recent result, optionally filtered by source."""
    conn = _get_conn()
    if source:
        row = conn.execute(
            "SELECT full_json FROM analysis_results WHERE source=? ORDER BY created_at DESC LIMIT 1",
            (source,),
        ).fetchone()
    else:
        row = conn.execute("SELECT full_json FROM analysis_results ORDER BY created_at DESC LIMIT 1").fetchone()
    return json.loads(row["full_json"]) if row else None


def get_history(source: str = None, limit: int = 20) -> List[Dict]:
    """Fetch recent results as lightweight summary rows."""
    conn = _get_conn()
    if source:
        rows = conn.execute(
            """
            SELECT id, source, filename, verdict, score, attack_type,
                   detection_mode, n_samples, elapsed_ms, created_at
            FROM analysis_results WHERE source=?
            ORDER BY created_at DESC LIMIT ?
        """,
            (source, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, source, filename, verdict, score, attack_type,
                   detection_mode, n_samples, elapsed_ms, created_at
            FROM analysis_results ORDER BY created_at DESC LIMIT ?
        """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_model_scan_history(limit: int = 20) -> List[Dict]:
    """Fetch recent model scan summaries."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT id, model_filename, dataset_filename, model_type, verdict,
               score, attack_type, n_samples, created_at
        FROM model_scans ORDER BY created_at DESC LIMIT ?
    """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_model_scan(rid: str) -> Optional[Dict]:
    """Fetch a single model scan by ID."""
    conn = _get_conn()
    row = conn.execute("SELECT full_json FROM model_scans WHERE id = ?", (rid,)).fetchone()
    return json.loads(row["full_json"]) if row else None


def get_stats() -> Dict:
    """Return aggregate statistics across all stored results."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0]
    by_source = conn.execute("SELECT source, COUNT(*) as n FROM analysis_results GROUP BY source").fetchall()
    by_verdict = conn.execute("SELECT verdict, COUNT(*) as n FROM analysis_results GROUP BY verdict").fetchall()
    model_scans = conn.execute("SELECT COUNT(*) FROM model_scans").fetchone()[0]
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
            datetime.now(timezone.utc).isoformat() + "Z",
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
    """Seed the configured administrator into SQLite when no users exist."""
    if not settings.admin_username or not settings.admin_password:
        return
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count:
        return
    from app.core.security import hash_password

    conn.execute(
        """
        INSERT INTO users (id, username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            settings.admin_username,
            settings.admin_username,
            hash_password(settings.admin_password),
            "admin",
            datetime.now(timezone.utc).isoformat() + "Z",
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
        (user_id, username, password_hash, role, datetime.now(timezone.utc).isoformat() + "Z"),
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


# ── Defense & HITL Persistence ────────────────────────────────────────────────

def save_defense_action(action: Dict[str, Any]) -> None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO defense_actions
        (action_id, action, samples_affected, suspicion_score, reason, details_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            action.get("action_id", str(uuid.uuid4())),
            action.get("action", "monitor"),
            action.get("samples_affected", 0),
            action.get("suspicion_score", 0.0),
            action.get("reason", ""),
            json.dumps(action),
            action.get("timestamp", datetime.now(timezone.utc).isoformat() + "Z"),
        ),
    )
    conn.commit()


def get_defense_actions(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT details_json FROM defense_actions ORDER BY created_at DESC LIMIT ?",
        (max(1, min(limit, 200)),),
    ).fetchall()
    return [json.loads(row["details_json"]) for row in rows]


def save_hitl_case(case: Dict[str, Any]) -> None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO hitl_cases
        (case_id, suspicion_score, n_samples, status, details_json, created_at, deadline)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case["case_id"],
            case.get("suspicion_score", 0.0),
            case.get("n_samples", 0),
            case.get("status", "pending"),
            json.dumps(case),
            case.get("created_at", datetime.now(timezone.utc).isoformat() + "Z"),
            case.get("deadline", ""),
        ),
    )
    conn.commit()


def get_pending_hitl_cases() -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT details_json FROM hitl_cases WHERE status = 'pending' ORDER BY created_at DESC"
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
            decision.get("decided_at", datetime.now(timezone.utc).isoformat() + "Z"),
        ),
    )
    # Also update case status in hitl_cases table
    conn.execute(
        "UPDATE hitl_cases SET status = 'resolved' WHERE case_id = ?",
        (decision["case_id"],),
    )
    conn.commit()


def get_hitl_decisions(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT case_id, decision, reviewer, decided_at FROM hitl_decisions ORDER BY decided_at DESC LIMIT ?",
        (max(1, min(limit, 200)),),
    ).fetchall()
    return [dict(row) for row in rows]


def save_redteam_simulation(sim: Dict[str, Any]) -> None:
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO redteam_simulations
        (simulation_id, attack_type, detected, verdict, suspicion_score, resilience_score, details_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sim["simulation_id"],
            sim["attack_type"],
            1 if sim.get("detected") else 0,
            sim.get("verdict", "CLEAN"),
            sim.get("suspicion_score", 0.0),
            sim.get("resilience_score", 0.0),
            json.dumps(sim),
            sim.get("timestamp", datetime.now(timezone.utc).isoformat() + "Z"),
        ),
    )
    conn.commit()


def get_redteam_simulations(limit: int = 50) -> List[Dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT details_json FROM redteam_simulations ORDER BY created_at DESC LIMIT ?",
        (max(1, min(limit, 200)),),
    ).fetchall()
    return [json.loads(row["details_json"]) for row in rows]


