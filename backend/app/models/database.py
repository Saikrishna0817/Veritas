"""
Persistence layer — AI Trust Forensics Platform.

SQLite for development, tests and single-box installs; PostgreSQL in production (set VERITAS_DATABASE_URL).
Queries use ``?`` placeholders and standard SQL; app.models.engine adapts them per dialect. The schema comes from
the versioned files in app/models/migrations (app.models.migrate).

Reads of user-owned rows take an explicit owner so a forgotten filter is a TypeError, not a leak. Rows saved before
owner_id existed have owner NULL and are visible only with ALL_OWNERS (admins).
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.models.engine import Database

# SQLite file next to the backend package unless configured; PostgreSQL when VERITAS_DATABASE_URL is set.
DB_PATH = Path(settings.sqlite_path) if settings.sqlite_path else Path(__file__).parent.parent.parent / "forensics_results.db"
_url: str = settings.database_url or f"sqlite:///{DB_PATH}"

_local = threading.local()


def configure(url: str) -> None:
    """Point this process at another database (tests, tools). Drops this thread's connection."""
    global _url
    _url = url
    reset_connection()


def current_url() -> str:
    return _url


def reset_connection() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
    _local.conn = None


def connection() -> Database:
    """This thread's connection, reconnecting if the server dropped it."""
    conn = getattr(_local, "conn", None)
    if conn is None or conn.closed or conn.url != _url:
        if conn is not None:
            conn.close()
        conn = _local.conn = Database(_url)
    return conn


_get_conn = connection      # historical name used throughout this module and in tests


def init_db() -> None:
    """Bring the schema up to date and make sure the configured administrator exists."""
    from app.models.migrate import migrate
    conn = _get_conn()
    migrate(conn)
    _ensure_bootstrap_admin(conn)


ALL_OWNERS = "*"


def _scope(owner_id: str) -> tuple[str, tuple]:
    """(' AND owner_id = ?', (owner,)) or ('', ()) for ALL_OWNERS."""
    return ("", ()) if owner_id == ALL_OWNERS else (" AND owner_id = ?", (owner_id,))


def save_result(result: Dict[str, Any], source: str, filename: str = None, owner_id: str | None = None) -> str:
    """Persist an analysis result. Returns the stored ID."""
    rid = result.get("job_id") or result.get("dataset_id") or (result.get("dataset_info") or {}).get("dataset_id") or str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO analysis_results
            (id, source, filename, verdict, score, attack_type, detection_mode,
             n_samples, elapsed_ms, full_json, created_at, owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET source=excluded.source, filename=excluded.filename, verdict=excluded.verdict,
            score=excluded.score, attack_type=excluded.attack_type, detection_mode=excluded.detection_mode,
            n_samples=excluded.n_samples, elapsed_ms=excluded.elapsed_ms, full_json=excluded.full_json,
            created_at=excluded.created_at, owner_id=excluded.owner_id
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
        INSERT INTO model_scans
            (id, model_filename, dataset_filename, model_type, verdict, score,
             attack_type, n_samples, full_json, created_at, owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET model_filename=excluded.model_filename, dataset_filename=excluded.dataset_filename,
            model_type=excluded.model_type, verdict=excluded.verdict, score=excluded.score, attack_type=excluded.attack_type,
            n_samples=excluded.n_samples, full_json=excluded.full_json, created_at=excluded.created_at, owner_id=excluded.owner_id
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


def _before(before: str | None) -> tuple[str, tuple]:
    """Paging cursor: (' AND created_at < ?', (before,)) or nothing. ``before`` is a created_at from the previous page."""
    return (" AND created_at < ?", (before,)) if before else ("", ())


def get_history(owner_id: str, source: str = None, limit: int = 20, before: str | None = None) -> List[Dict]:
    """Recent results as lightweight summary rows, newest first; ``before`` continues from an earlier page."""
    clause, args = _scope(owner_id)
    page, page_args = _before(before)
    clause, args = clause + page, (*args, *page_args)
    where, params = ("source=?" + clause, (source, *args)) if source else ("1=1" + clause, args)
    rows = _get_conn().execute(
        f"""SELECT id, source, filename, verdict, score, attack_type, detection_mode, n_samples, elapsed_ms, created_at, owner_id
            FROM analysis_results WHERE {where} ORDER BY created_at DESC LIMIT ?""", (*params, limit)).fetchall()
    return [dict(r) for r in rows]


def get_model_scan_history(owner_id: str, limit: int = 20, before: str | None = None) -> List[Dict]:
    """Recent model scan summaries, newest first."""
    clause, args = _scope(owner_id)
    page, page_args = _before(before)
    clause, args = clause + page, (*args, *page_args)
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
    total = conn.execute(f"SELECT COUNT(*) AS n FROM analysis_results WHERE 1=1{clause}", args).fetchone()["n"]
    by_source = conn.execute(f"SELECT source, COUNT(*) as n FROM analysis_results WHERE 1=1{clause} GROUP BY source", args).fetchall()
    by_verdict = conn.execute(f"SELECT verdict, COUNT(*) as n FROM analysis_results WHERE 1=1{clause} GROUP BY verdict", args).fetchall()
    model_scans = conn.execute(f"SELECT COUNT(*) AS n FROM model_scans WHERE 1=1{clause}", args).fetchone()["n"]
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


def get_audit_events(actor_id: str | None = None, limit: int = 50, before: str | None = None) -> List[Dict]:
    """Return recent audit records without exposing unbounded history; ``before`` continues from an earlier page."""
    limit = max(1, min(limit, 200))
    conn = _get_conn()
    page, page_args = _before(before)
    if actor_id:
        rows = conn.execute(
            f"SELECT * FROM audit_events WHERE actor_id=?{page} ORDER BY created_at DESC LIMIT ?",
            (actor_id, *page_args, limit),
        ).fetchall()
    else:
        rows = conn.execute(f"SELECT * FROM audit_events WHERE 1=1{page} ORDER BY created_at DESC LIMIT ?", (*page_args, limit)).fetchall()
    events = []
    for row in rows:
        event = dict(row)
        event["details"] = json.loads(event.pop("details_json"))
        events.append(event)
    return events


def _ensure_bootstrap_admin(conn: Database) -> None:
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


_USER_COLS = "id, username, password_hash, role, created_at, email, email_verified, token_version, disabled_at"


def get_user_by_username(username: str) -> Optional[Dict]:
    row = _get_conn().execute(f"SELECT {_USER_COLS} FROM users WHERE username=?", (username,)).fetchone()
    return dict(row) if row else None


def username_taken(username: str) -> bool:
    """True if any account has this name in any letter case: 'Alice' and 'alice' must not be two people."""
    return _get_conn().execute("SELECT 1 FROM users WHERE lower(username) = lower(?)", (username,)).fetchone() is not None


def get_user_by_email(email: str) -> Optional[Dict]:
    row = _get_conn().execute(f"SELECT {_USER_COLS} FROM users WHERE email=?", (email,)).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[Dict]:
    row = _get_conn().execute(f"SELECT {_USER_COLS} FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(row) if row else None


def create_user(username: str, password_hash: str, role: str = "analyst", email: str | None = None, email_verified: bool = True) -> str:
    user_id = str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO users (id, username, password_hash, role, created_at, email, email_verified)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, username, password_hash, role, datetime.now(timezone.utc).isoformat(), email, 1 if email_verified else 0),
    )
    conn.commit()
    return user_id


def set_email_verified(user_id: str) -> None:
    conn = _get_conn()
    conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user_id,))
    conn.commit()


def end_all_sessions(user_id: str) -> None:
    """Every token issued so far for this account stops working (security.decode_access_token checks the version)."""
    conn = _get_conn()
    conn.execute("UPDATE users SET token_version = token_version + 1 WHERE id = ?", (user_id,))
    _revoke_refresh(conn, user_id=user_id)
    conn.commit()


def set_password_hash(user_id: str, password_hash: str) -> None:
    conn = _get_conn()
    # a new password ends every session: tokens carrying the old version are rejected (security.decode_access_token)
    conn.execute("UPDATE users SET password_hash = ?, token_version = token_version + 1 WHERE id = ?", (password_hash, user_id))
    _revoke_refresh(conn, user_id=user_id)
    conn.commit()


# ── Refresh tokens (P3): hashed, rotated on every use, grouped in families ───

def _hash_token(raw: str) -> str:
    import hashlib
    return hashlib.sha256(raw.encode()).hexdigest()


def _revoke_refresh(conn, user_id: str | None = None, family_id: str | None = None) -> None:
    col, val = ("user_id", user_id) if user_id else ("family_id", family_id)
    conn.execute(f"UPDATE refresh_tokens SET revoked_at = ? WHERE {col} = ? AND revoked_at IS NULL",
                 (datetime.now(timezone.utc).isoformat(), val))


def create_refresh_token(user_id: str, family_id: str | None = None) -> str:
    """Return a new random refresh token (only its SHA-256 is stored). A new sign-in starts a new family."""
    import secrets
    raw = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    conn = _get_conn()
    conn.execute(
        "INSERT INTO refresh_tokens (id, user_id, family_id, token_hash, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), user_id, family_id or str(uuid.uuid4()), _hash_token(raw), now.isoformat(),
         (now + timedelta(days=settings.refresh_expiry_days)).isoformat()),
    )
    conn.commit()
    return raw


def use_refresh_token(raw: str) -> tuple[str, Dict | None]:
    """Spend a refresh token. Returns (outcome, row): "ok" (row is the spent token, its family continues),
    "reuse" (it was already spent or revoked: the whole family is now revoked), or "invalid" (unknown or expired)."""
    conn = _get_conn()
    h = _hash_token(raw)
    row = conn.execute("SELECT * FROM refresh_tokens WHERE token_hash = ?", (h,)).fetchone()
    if row is None:
        return "invalid", None
    row = dict(row)
    now = datetime.now(timezone.utc).isoformat()
    if row["revoked_at"]:
        return "invalid", row
    if row["expires_at"] <= now:
        return "invalid", row
    claimed = conn.execute("UPDATE refresh_tokens SET used_at = ? WHERE id = ? AND used_at IS NULL RETURNING id", (now, row["id"])).fetchone()
    if claimed is None:       # someone spent it first: a stolen copy and the real one are both in play, end the family
        _revoke_refresh(conn, family_id=row["family_id"])
        conn.commit()
        return "reuse", row
    conn.commit()
    return "ok", row


def revoke_refresh_family(raw: str) -> None:
    """Sign-out of this browser: end the family the presented token belongs to (unknown tokens are ignored)."""
    conn = _get_conn()
    row = conn.execute("SELECT family_id FROM refresh_tokens WHERE token_hash = ?", (_hash_token(raw),)).fetchone()
    if row:
        _revoke_refresh(conn, family_id=row["family_id"])
        conn.commit()


# ── One-time tokens (email verification, password reset) ─────────────────────

def create_auth_token(user_id: str, purpose: str, ttl_minutes: int) -> str:
    """Return a new random token; only its SHA-256 is stored. Older unused tokens for the same purpose stop working."""
    import hashlib
    import secrets
    raw = secrets.token_urlsafe(32)
    conn = _get_conn()
    now = datetime.now(timezone.utc)
    conn.execute("UPDATE auth_tokens SET used_at = ? WHERE user_id = ? AND purpose = ? AND used_at IS NULL", (now.isoformat(), user_id, purpose))
    conn.execute(
        "INSERT INTO auth_tokens (token_hash, user_id, purpose, expires_at) VALUES (?, ?, ?, ?)",
        (hashlib.sha256(raw.encode()).hexdigest(), user_id, purpose, (now + timedelta(minutes=ttl_minutes)).isoformat()),
    )
    conn.commit()
    return raw


def consume_auth_token(raw: str, purpose: str) -> str | None:
    """Return the user id if the token is valid, unexpired and unused, and mark it used (single use)."""
    import hashlib
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    row = conn.execute(
        """UPDATE auth_tokens SET used_at = ? WHERE token_hash = ? AND purpose = ? AND used_at IS NULL AND expires_at > ?
           RETURNING user_id""",
        (now, hashlib.sha256(raw.encode()).hexdigest(), purpose, now),
    ).fetchone()
    conn.commit()
    return row["user_id"] if row else None


# ── API keys ─────────────────────────────────────────────────────────────────

def create_api_key(owner_id: str, name: str) -> tuple[str, Dict]:
    """Returns (raw_key, row). The raw key is shown once; only its hash is stored."""
    import hashlib
    import secrets
    raw = "vk_" + secrets.token_urlsafe(32)
    key_id = str(uuid.uuid4())
    conn = _get_conn()
    conn.execute(
        "INSERT INTO api_keys (id, owner_id, name, prefix, key_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (key_id, owner_id, name, raw[:10], hashlib.sha256(raw.encode()).hexdigest(), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return raw, {"id": key_id, "name": name, "prefix": raw[:10]}


def get_user_by_api_key(raw: str) -> Optional[Dict]:
    import hashlib
    conn = _get_conn()
    row = conn.execute(
        """SELECT u.id, u.username, u.role, k.id AS key_id FROM api_keys k JOIN users u ON u.id = k.owner_id
           WHERE k.key_hash = ? AND k.revoked_at IS NULL AND u.disabled_at IS NULL""",
        (hashlib.sha256(raw.encode()).hexdigest(),),
    ).fetchone()
    if row:
        conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (datetime.now(timezone.utc).isoformat(), row["key_id"]))
        conn.commit()
    return dict(row) if row else None


def list_api_keys(owner_id: str) -> List[Dict]:
    rows = _get_conn().execute(
        "SELECT id, name, prefix, created_at, last_used_at, revoked_at FROM api_keys WHERE owner_id = ? ORDER BY created_at DESC", (owner_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def revoke_api_key(owner_id: str, key_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("UPDATE api_keys SET revoked_at = ? WHERE id = ? AND owner_id = ? AND revoked_at IS NULL",
                       (datetime.now(timezone.utc).isoformat(), key_id, owner_id))
    conn.commit()
    return cur.rowcount > 0


def list_users(limit: int = 50, before: str | None = None) -> List[Dict]:
    conn = _get_conn()
    page, page_args = _before(before)
    rows = conn.execute(
        f"SELECT id, username, role, created_at, disabled_at FROM users WHERE 1=1{page} ORDER BY created_at DESC LIMIT ?",
        (*page_args, max(1, min(limit, 200))),
    ).fetchall()
    return [dict(row) for row in rows]


def set_user_role(user_id: str, role: str) -> bool:
    """Change an account's role. Takes effect on the next request: sessions read the current role (security.py)."""
    conn = _get_conn()
    cursor = conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()
    return cursor.rowcount > 0


def set_user_disabled(user_id: str, disabled: bool) -> bool:
    """Disable (or re-enable) an account. Disabling also ends its sessions; its data and API keys are kept, the
    keys simply stop authenticating while the account is disabled."""
    conn = _get_conn()
    if disabled:
        cur = conn.execute("UPDATE users SET disabled_at = ?, token_version = token_version + 1 WHERE id = ?",
                           (datetime.now(timezone.utc).isoformat(), user_id))
        _revoke_refresh(conn, user_id=user_id)
    else:
        cur = conn.execute("UPDATE users SET disabled_at = NULL WHERE id = ?", (user_id,))
    conn.commit()
    return cur.rowcount > 0


def delete_user(user_id: str) -> bool:
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    return cursor.rowcount > 0


def delete_account(user_id: str) -> Dict[str, int]:
    """Erase the user and everything keyed to them, audit events included (right to erasure)."""
    conn = _get_conn()
    try:                                  # one transaction: a failure part-way leaves the account and its data whole
        counts = _delete_owner_rows(conn, user_id)
        for table, col in (("api_keys", "owner_id"), ("auth_tokens", "user_id"), ("refresh_tokens", "user_id"), ("jobs", "owner_id"),
                           ("audit_events", "actor_id"), ("users", "id")):
            counts[table] = conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (user_id,)).rowcount
        conn.execute("INSERT INTO deleted_accounts (user_id, deleted_at) VALUES (?, ?) ON CONFLICT (user_id) DO NOTHING",
                     (user_id, datetime.now(timezone.utc).isoformat()))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return counts


def is_deleted_account(user_id: str) -> bool:
    return _get_conn().execute("SELECT 1 FROM deleted_accounts WHERE user_id = ?", (user_id,)).fetchone() is not None


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
        INSERT INTO defense_actions
        (action_id, action, samples_affected, suspicion_score, reason, details_json, created_at, owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (action_id) DO UPDATE SET action=excluded.action, samples_affected=excluded.samples_affected,
            suspicion_score=excluded.suspicion_score, reason=excluded.reason, details_json=excluded.details_json,
            created_at=excluded.created_at, owner_id=excluded.owner_id
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
        INSERT INTO hitl_cases
        (case_id, suspicion_score, n_samples, status, details_json, created_at, deadline, owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (case_id) DO UPDATE SET suspicion_score=excluded.suspicion_score, n_samples=excluded.n_samples,
            status=excluded.status, details_json=excluded.details_json, created_at=excluded.created_at,
            deadline=excluded.deadline, owner_id=excluded.owner_id
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


def save_hitl_decision(decision: Dict[str, Any]) -> bool:
    """Resolve a pending case and record the decision, in one transaction. False when the case was not pending any
    more (another decision got there first), and then nothing is recorded."""
    conn = _get_conn()
    try:
        claimed = conn.execute("UPDATE hitl_cases SET status = 'resolved' WHERE case_id = ? AND status = 'pending'",
                               (decision["case_id"],)).rowcount
        if not claimed:
            conn.rollback()
            return False
        conn.execute(
            "INSERT INTO hitl_decisions (id, case_id, decision, reviewer, decided_at) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), decision["case_id"], decision["decision"], decision.get("reviewer", "analyst"),
             decision.get("decided_at", datetime.now(timezone.utc).isoformat())),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return True


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
    Returns (allowed, used_today). One conditional upsert, so concurrent requests cannot overshoot the limit."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = _get_conn()
    if limit and amount > limit:
        return False, get_usage(owner_id).get(kind, 0)
    row = conn.execute(
        """INSERT INTO usage_daily (owner_id, day, kind, amount) VALUES (?, ?, ?, ?)
           ON CONFLICT (owner_id, day, kind) DO UPDATE SET amount = usage_daily.amount + excluded.amount
           WHERE ? = 0 OR usage_daily.amount + excluded.amount <= ?
           RETURNING amount""",
        (owner_id, day, kind, amount, limit, limit),
    ).fetchone()
    conn.commit()
    if row is None:
        return False, get_usage(owner_id).get(kind, 0)
    return True, row["amount"]


def refund_usage(owner_id: str, day: str, kind: str, amount: int) -> None:
    """Give back ``amount`` of a charge made on ``day`` (never below zero)."""
    conn = _get_conn()
    conn.execute("UPDATE usage_daily SET amount = CASE WHEN amount > ? THEN amount - ? ELSE 0 END WHERE owner_id=? AND day=? AND kind=?",
                 (amount, amount, owner_id, day, kind))
    conn.commit()


def get_usage(owner_id: str) -> Dict[str, int]:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = _get_conn().execute("SELECT kind, amount FROM usage_daily WHERE owner_id=? AND day=?", (owner_id, day)).fetchall()
    return {r["kind"]: r["amount"] for r in rows}


# ── Data deletion and retention ───────────────────────────────────────────────

_OWNED_TABLES = ("analysis_results", "model_scans", "defense_actions", "hitl_cases", "pending_uploads")


def delete_owner_data(owner_id: str) -> Dict[str, int]:
    """Erase everything stored for one user (the delete-my-data action). Audit events stay: they hold no results."""
    conn = _get_conn()
    try:
        counts = _delete_owner_rows(conn, owner_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return counts


def _delete_owner_rows(conn, owner_id: str) -> Dict[str, int]:
    """The deletes behind delete_owner_data, without committing (callers own the transaction)."""
    counts = {}
    conn.execute("DELETE FROM hitl_decisions WHERE case_id IN (SELECT case_id FROM hitl_cases WHERE owner_id = ?)", (owner_id,))
    for table in _OWNED_TABLES:
        counts[table] = conn.execute(f"DELETE FROM {table} WHERE owner_id = ?", (owner_id,)).rowcount
    counts["usage_daily"] = conn.execute("DELETE FROM usage_daily WHERE owner_id = ?", (owner_id,)).rowcount
    return counts


def purge_older_than(days: int) -> Dict[str, int]:
    """Retention: drop results, scans, defense actions and cases older than ``days`` (0 keeps everything)."""
    if days <= 0:
        return {}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = _get_conn()
    counts = {}
    conn.execute("DELETE FROM hitl_decisions WHERE case_id IN (SELECT case_id FROM hitl_cases WHERE created_at < ?)", (cutoff,))
    for table in _OWNED_TABLES:
        counts[table] = conn.execute(f"DELETE FROM {table} WHERE created_at < ?", (cutoff,)).rowcount
    conn.commit()
    return counts


# ── Direct uploads ────────────────────────────────────────────────────────────

def create_pending_upload(upload_id: str, owner_id: str, role: str, filename: str, size: int, storage_key: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO pending_uploads (id, owner_id, role, filename, size, storage_key, used, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        (upload_id, owner_id, role, filename, size, storage_key, datetime.now(timezone.utc).isoformat()))
    conn.commit()


def count_open_uploads(owner_id: str, since_iso: str) -> int:
    return _get_conn().execute(
        "SELECT COUNT(*) AS n FROM pending_uploads WHERE owner_id = ? AND used = 0 AND created_at >= ?", (owner_id, since_iso)).fetchone()["n"]


def claim_pending_upload(upload_id: str, owner_id: str) -> Optional[Dict]:
    """The owner's unused upload row, marked used in the same step so a presigned upload can start only one job."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM pending_uploads WHERE id = ? AND owner_id = ?", (upload_id, owner_id)).fetchone()
    if not row:
        return None
    claimed = conn.execute("UPDATE pending_uploads SET used = 1 WHERE id = ? AND owner_id = ? AND used = 0", (upload_id, owner_id)).rowcount
    conn.commit()
    return dict(row) if claimed else None


# ── Background jobs ───────────────────────────────────────────────────────────

def create_job(job_id: str, job_type: str, owner_id: str, payload: Dict[str, Any]) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO jobs (job_id, job_type, owner_id, status, payload_json, created_at) VALUES (?, ?, ?, 'queued', ?, ?)",
        (job_id, job_type, owner_id, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def update_job(job_id: str, **fields: Any) -> None:
    allowed = {"status", "result_json", "error", "started_at", "completed_at"}
    cols = [k for k in fields if k in allowed]
    conn = _get_conn()
    conn.execute(f"UPDATE jobs SET {', '.join(c + ' = ?' for c in cols)} WHERE job_id = ?", (*[fields[c] for c in cols], job_id))
    conn.commit()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    row = _get_conn().execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        return None
    job = dict(row)
    job["payload"] = json.loads(job.pop("payload_json") or "{}")
    result = job.pop("result_json")
    job["result"] = json.loads(result) if result else None
    return job
