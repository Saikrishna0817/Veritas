-- Original schema (before owner_id and accounts). IF NOT EXISTS so pre-migration SQLite files adopt it as-is.
CREATE TABLE IF NOT EXISTS analysis_results (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    filename    TEXT,
    verdict     TEXT,
    score       DOUBLE PRECISION,
    attack_type TEXT,
    detection_mode TEXT,
    n_samples   INTEGER,
    elapsed_ms  DOUBLE PRECISION,
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
    score           DOUBLE PRECISION,
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
    suspicion_score  DOUBLE PRECISION,
    reason           TEXT,
    details_json     TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_defense_created ON defense_actions(created_at DESC);
CREATE TABLE IF NOT EXISTS hitl_cases (
    case_id          TEXT PRIMARY KEY,
    suspicion_score  DOUBLE PRECISION,
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
