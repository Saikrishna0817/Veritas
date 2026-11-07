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
