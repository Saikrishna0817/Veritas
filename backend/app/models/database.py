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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# DB lives next to the backend package
DB_PATH = Path(__file__).parent.parent.parent / "forensics_results.db"

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
    """
    )
    conn.commit()


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
            datetime.utcnow().isoformat(),
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
            datetime.utcnow().isoformat(),
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


