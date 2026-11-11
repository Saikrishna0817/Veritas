# Veritas pages

Every page below is wired to the real backend through `src/services/api.js`; none reads fixture or mock data. The only illustrative content is labelled as such: the 3D specimen before a batch is examined ("illustrative specimen") and the landing page's "Example run" panels. Benchmark figures come from `docs/model-card.md`.

## Routes

| Route | Page | Who | Endpoints (via `api.*`) |
|---|---|---|---|
| `/` (signed out), `/welcome` | `LandingPage` | anyone | none |
| `/login` | `LoginPage` | anyone | `POST /auth/token`, `GET /auth/me` |
| `/admin/login` | `AdminLoginPage` | anyone | `POST /auth/token` (role admin) |
| `/signup`, `/verify-email`, `/forgot-password`, `/reset-password`, `/resend-verification` | `AccountFlowPage` | anyone | `/auth/signup`, `/auth/verify-email`, `/auth/password-reset/*`, `/auth/resend-verification` |
| `/` (signed in) | `Dashboard` | signed in | `GET /analyze/upload/latest`, `GET /history`, `GET /account/usage`, `POST /demo/run`, WebSocket `/ws/v1/detection-stream` |
| `/upload` | `UploadPage` | signed in | `POST /analyze/upload` (inline or `?background=true`), `POST /uploads/presign` + `/uploads/complete` when direct uploads are on, `GET /jobs/{id}` |
| `/model-scan` | `ModelScanPage` | signed in | `POST /analyze/model`, `GET /jobs/{id}`, `GET /analyze/model/history` |
| `/real-datasets` | `RealDatasetsPage` | signed in | `GET /datasets/real`, `POST /datasets/real/{id}/analyze`, `GET /datasets/real/{id}/download` |
| `/history` | `HistoryPage` | signed in | `GET /history`, `GET /history/{id}`, `GET /analyze/model/{id}` |
| `/reports` | `ReportsPage` | signed in | `POST /reports/generate?source=` |
| `/forensics` | `ForensicsPage` (preview) | signed in | `GET /forensics/latest?source=` |
| `/blue-team` | `BlueTeamPage` (preview) | signed in | `GET /blueteam/status`, `/blueteam/incidents`, `/blueteam/playbooks`, `/blueteam/playbook/{type}`, `POST /defense/hitl/decide` |
| `/account` | `AccountPage` | signed in | `GET /account/usage`, `/account/api-keys` (list, create, revoke), `DELETE /account/data`, `POST /account/delete` |
| `/admin` | `AdminDashboard` | admin | `/users` (list, create, delete), `GET /history` and `DELETE /history/{id}` (all accounts), `GET /audit/events`, `GET /defense/hitl/pending`, `GET /defense/status`, `POST /defense/quarantine`, `POST /defense/hitl/decide` |
| anything else, signed in | `NotFoundPage` | signed in | none |
| anything else, signed out | redirects to `/login` | — | — |

## States every data page handles

| State | How it looks | Rule |
|---|---|---|
| Loading | `Loading` line; stat cards and counts show "—" | Never a 0 before the number arrives. |
| Empty | `Empty` card that says why and links to the action that fixes it | Only on a successful empty answer or a 404 from a "latest" endpoint. |
| Failed load, nothing to show | `ErrorState` with Try again | Wording from the status (`services/errors.js`); the server's message is kept. |
| Server can't do it here | `Unavailable` (grey) | 501/503, or the server is not answering. |
| Failed refresh, data on screen | Data stays; `ErrorNote` above it | — |
| Failed action | `ErrorNote` next to the control | — |
| Page crashed while drawing | `PageErrorBoundary` | The shell keeps working. |

Missing fields render as "—" (`orDash`, `percent`), and verdict codes the interface does not know render neutrally with the raw code (`verdictInfo`), never as another verdict.

## Data the pages read

Field names are the backend's; these are the ones the interface depends on.

- **Analysis result** (upload, reference dataset, sample batch, history detail): `verdict` (`CONFIRMED_POISONED` · `SUSPICIOUS` · `LOW_RISK` · `CLEAN`), `n_samples`, `elapsed_ms`, `detection_mode`, `dataset_info { filename, n_rows, has_labels, label_column }`, `layer_results { label_check, shift_check { evidence, alarm, ran }, row_flags { row_scores, flagged_indices_incoming } }`, `poison_note`.
- **Model scan**: `verdict` (`DANGEROUS` · `REVIEW` · `NO_CODE_EXECUTION_FOUND` · `SAFE_FORMAT`), `model_filename`, `scan { findings[] { severity, ... } }`.
- **History list**: `results[] { id, source, verdict, filename | model_filename, n_samples, created_at }`, `stats { total_analyses, model_scans, by_verdict }`.
- **Usage**: `used { analyses, rows, scan_mb }`, `limits { analyses, rows, scan_mb }` (a missing limit means none).
- **Report**: `report_id`, `generated_at`, `title`, `data_source`, `dataset_info`, `executive_summary { verdict, attack_type, confidence }`, `layer_scores`, `attack_narrative`, `defense_actions[]`, `limitations`.
- **Forensics**: `verdict`, `source`, `dataset_info`, `attack_classification { attack_type, confidence, severity, description, probabilities }`, `injection_pattern { n_poisoned_samples, narrative, ... }`, `blast_radius`.
- **Response desk status**: `threat_level` (`CRITICAL` · `ELEVATED` · `GUARDED` · `NOMINAL`), `current_verdict`, `suspicion_score`, `updated_at`, `total_quarantined`, `n_defense_actions`, `defense_mode`, `hitl_queue_depth`, `pending_cases[]`.
