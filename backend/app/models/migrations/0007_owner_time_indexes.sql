-- Owner-scoped lists filter on owner_id and sort by created_at: one composite index serves both, and the paging
-- cursor (created_at < ?) uses it too. The single-column owner indexes from 0002 become redundant but stay.
CREATE INDEX IF NOT EXISTS idx_analysis_results_owner_created ON analysis_results(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_scans_owner_created ON model_scans(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_defense_actions_owner_created ON defense_actions(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_hitl_cases_owner_created ON hitl_cases(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_created ON users(created_at DESC);
