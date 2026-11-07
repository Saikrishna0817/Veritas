-- Opaque ids of erased accounts, so a still-valid login token for one stops working. No personal data.
CREATE TABLE IF NOT EXISTS deleted_accounts (
    user_id    TEXT PRIMARY KEY,
    deleted_at TEXT NOT NULL
);
