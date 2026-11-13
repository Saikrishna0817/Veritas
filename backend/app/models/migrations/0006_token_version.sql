-- Bumped on password reset: login tokens carry it, so tokens issued before the reset stop working.
ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0;
