-- Trial feedback: a person marks a flagged item of one of their results as a real problem ('bad') or 'fine'.
CREATE TABLE IF NOT EXISTS item_marks (
    id         TEXT PRIMARY KEY,
    owner_id   TEXT NOT NULL,
    result_id  TEXT NOT NULL,
    item       TEXT NOT NULL,
    rank       INTEGER,
    verdict    TEXT NOT NULL CHECK (verdict IN ('bad', 'fine')),
    created_at TEXT,
    updated_at TEXT,
    UNIQUE (owner_id, result_id, item)
);
CREATE INDEX IF NOT EXISTS idx_item_marks_result ON item_marks(result_id);
