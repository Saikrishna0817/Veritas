-- Tenant isolation: every user-owned row carries owner_id. Per-user daily usage counters for quotas.
ALTER TABLE analysis_results ADD COLUMN owner_id TEXT;
ALTER TABLE model_scans ADD COLUMN owner_id TEXT;
ALTER TABLE defense_actions ADD COLUMN owner_id TEXT;
ALTER TABLE hitl_cases ADD COLUMN owner_id TEXT;
CREATE INDEX IF NOT EXISTS idx_analysis_results_owner ON analysis_results(owner_id);
CREATE INDEX IF NOT EXISTS idx_model_scans_owner ON model_scans(owner_id);
CREATE INDEX IF NOT EXISTS idx_defense_actions_owner ON defense_actions(owner_id);
CREATE INDEX IF NOT EXISTS idx_hitl_cases_owner ON hitl_cases(owner_id);
CREATE TABLE IF NOT EXISTS usage_daily (
    owner_id TEXT NOT NULL,
    day      TEXT NOT NULL,
    kind     TEXT NOT NULL,
    amount   BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (owner_id, day, kind)
);
