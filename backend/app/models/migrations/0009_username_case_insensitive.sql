-- Usernames are unique case-insensitively. Later duplicates (by created_at, then id) are renamed first so the index
-- cannot fail on existing data: the oldest account keeps its name, the others become "<name>-<first 6 chars of id>".
UPDATE users SET username = username || '-' || substr(id, 1, 6)
WHERE EXISTS (SELECT 1 FROM users o WHERE lower(o.username) = lower(users.username)
              AND (o.created_at < users.created_at OR (o.created_at = users.created_at AND o.id < users.id)));
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_lower ON users (lower(username));
