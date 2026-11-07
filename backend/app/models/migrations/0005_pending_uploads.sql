-- Direct-to-storage uploads: a presigned URL is only honoured for the owner who asked for it, once, at the declared size.
CREATE TABLE IF NOT EXISTS pending_uploads (
    id         TEXT PRIMARY KEY,
    owner_id   TEXT NOT NULL,
    role       TEXT NOT NULL,
    filename   TEXT NOT NULL,
    size       BIGINT NOT NULL,
    storage_key TEXT NOT NULL,
    used       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_uploads_owner ON pending_uploads(owner_id, created_at DESC);
