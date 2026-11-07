-- Self-serve accounts, API keys, one-time tokens (email verification, password reset) and a durable job table.
ALTER TABLE users ADD COLUMN email TEXT;
ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 1;
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT NOT NULL,
    name        TEXT NOT NULL,
    prefix      TEXT NOT NULL,
    key_hash    TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_owner ON api_keys(owner_id);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    purpose     TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    used_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens(user_id, purpose);
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    job_type     TEXT NOT NULL,
    owner_id     TEXT NOT NULL,
    status       TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    result_json  TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner_id, created_at DESC);
