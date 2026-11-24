from app.models.engine import Database
from app.models.migrate import DIR, migrate, _statements


def test_pre_migration_database_upgrades_in_place_and_is_idempotent(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'old.db'}")
    for stmt in _statements((DIR / "0001_initial.sql").read_text()):   # an old install: tables exist, no schema_migrations row
        db.execute(stmt)
    db.execute("INSERT INTO users (id, username, password_hash, role, created_at) VALUES ('u','legacy','x','user','2026-01-01')")
    db.commit()
    first = migrate(db)
    assert first[0].startswith("0001") and len(first) == len(list(DIR.glob("*.sql")))
    assert "owner_id" in db.columns("analysis_results") and "email_verified" in db.columns("users")
    assert db.execute("SELECT email_verified FROM users WHERE id='u'").fetchone()["email_verified"] == 1   # legacy users stay verified
    assert migrate(db) == []


def test_0009_renames_case_duplicates_then_index_rejects_variants(tmp_path):
    import pytest
    db = Database(f"sqlite:///{tmp_path / 'dup.db'}")
    db.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)")
    for path in sorted(DIR.glob("*.sql")):
        if path.stem >= "0009":
            break
        for stmt in _statements(path.read_text()):
            db.execute(stmt)
        db.execute("INSERT INTO schema_migrations VALUES (?, 'x')", (path.stem,))
    for uid, name, ts in [("aaaaaa111", "Bob", "2026-01-02"), ("bbbbbb222", "bob", "2026-01-01"),
                          ("cccccc333", "BOB", "2026-01-03"), ("dddddd444", "alice", "2026-01-01")]:
        db.execute("INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?, ?, 'x', 'user', ?)", (uid, name, ts))
    db.commit()
    assert migrate(db)[0] == "0009_username_case_insensitive"   # later migrations may follow
    names = {r["id"]: r["username"] for r in db.execute("SELECT id, username FROM users").fetchall()}
    assert names == {"bbbbbb222": "bob", "aaaaaa111": "Bob-aaaaaa", "cccccc333": "BOB-cccccc", "dddddd444": "alice"}
    with pytest.raises(Exception):
        db.execute("INSERT INTO users (id, username, password_hash, role, created_at) VALUES ('e', 'ALICE', 'x', 'user', '2026-02-01')")
