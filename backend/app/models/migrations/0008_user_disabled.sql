-- Deactivation: a disabled account keeps its data but cannot sign in, and its sessions and API keys stop working.
ALTER TABLE users ADD COLUMN disabled_at TEXT;
