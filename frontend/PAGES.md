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
| `/history` | `HistoryPage` | signed in | `GET /history?source=&limit=` (50 → 100 → 200 with Show more), `GET /history/{id}`, `GET /analyze/model/{id}` |
| `/reports` | `ReportsPage` | signed in | `POST /reports/generate?source=auto\|upload\|real_dataset\|demo` |
| `/forensics` | `ForensicsPage` (preview) | signed in | `GET /forensics/latest?source=auto\|upload\|real_dataset\|demo` |
| `/blue-team` | `BlueTeamPage` (preview) | signed in | `GET /blueteam/status`, `/blueteam/incidents`, `/blueteam/playbooks`, `/blueteam/playbook/{type}`, `GET /defense/hitl/pending` (full queue; the status embeds only 5), `GET /history?limit=1` (to tell "nothing analysed" from a clean batch), `POST /defense/hitl/decide` |
| `/account` | `AccountPage` | signed in | `GET /account/usage`, `/account/api-keys` (list, create, revoke), `GET /audit/events` (own activity), `DELETE /account/data`, `POST /account/delete` |
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

- **Analysis result** (upload, reference dataset, sample batch, history detail): `verdict` (`CONFIRMED_POISONED` · `SUSPICIOUS` · `LOW_RISK` · `CLEAN`), `n_samples`, `elapsed_ms`, `dataset_info { filename, n_rows, has_labels, label_column, detection_mode }`, `layer_results { label_check, shift_check { evidence, alarm, ran }, row_flags { row_scores, flagged_indices_incoming } }`, `poison_note`.
- **Model scan**: `verdict` (`DANGEROUS` · `REVIEW` · `NO_CODE_EXECUTION_FOUND` · `SAFE_FORMAT`), `model_filename`, `scan { format, findings[] { severity, ... } }`. History: `{ scans[] }`.
- **Audit event**: `id`, `action` (for example `analysis.uploaded`, `apikey.created`), `actor_id`, `resource_type`, `resource_id`, `details`, `created_at`.
- **Background job**: `status` (`queued` · `running` · `completed` · `failed`), `result` when completed, `error` when failed.
- **History list**: `results[] { id, source, verdict, filename | model_filename, n_samples, created_at }`, `stats { total_analyses, model_scans, by_verdict }`.
- **Usage**: `used { analyses, rows, scan_mb }`, `limits { analyses, rows, scan_mb }` (a missing limit means none).
- **Report**: `report_id`, `generated_at`, `title`, `data_source`, `dataset_info`, `executive_summary { verdict, attack_type, confidence }`, `layer_scores`, `attack_narrative`, `defense_actions[]`, `limitations`.
- **Forensics**: `verdict`, `source`, `dataset_info`, `attack_classification { attack_type, confidence, severity, description, probabilities }`, `injection_pattern { n_poisoned_samples, narrative, ... }`, `blast_radius`.
- **Response desk status**: `threat_level` (`CRITICAL` · `ELEVATED` · `GUARDED` · `NOMINAL`), `current_verdict`, `suspicion_score`, `updated_at`, `total_quarantined`, `n_defense_actions`, `defense_mode`, `hitl_queue_depth`, `pending_cases[]`.

## Connection

- **Base URL:** `VITE_API_BASE_URL` at build time, default `/api/v1`. Sessions are a same-origin httpOnly cookie with a CSRF double-submit, so the API must be served from the same origin as the app: the Vite dev server proxies `/api` and `/ws` to `localhost:8001`, and `nginx.conf` does the same in production. A cross-origin base URL would not carry the session.
- **Auth:** `POST /auth/token` sets the cookies; every state-changing call echoes `veritas_csrf` in `X-CSRF-Token`. A 401 sends the visitor to `/login` (or `/admin/login` under `/admin`). Admin-only routes sit behind `ProtectedRoute requiredRole="admin"`; the server enforces the same.
- **Timeouts:** 30 s for reads, 180 s for analyses that run inside the request (small uploads, model scans, reference datasets, sample batches, reports), 120 s for CSV downloads. A timed-out request says so (`timeout`), separately from an unreachable server (`offline`).
- **Large files:** 10 MB and up go to a background job, straight to S3/R2 when the server has it (`/uploads/presign` 501 means it does not), otherwise through `?background=true`. `GET /jobs/{id}` is polled every 2 s; up to 5 failed polls in a row from the network or a 5xx are retried with backoff, a 404 or 403 ends it, and watching stops after 30 minutes (the job keeps running and its result lands in History). A job left running when the page closes is picked up again on return.
- **Live events:** `/ws/v1/detection-stream`, authenticated by the session cookie. Frames are `{ event, data }`; control frames are ignored. It reconnects with jittered backoff up to 30 s, and the rail shows "Event stream reconnecting" meanwhile.

## Not supported by the backend

These are documented, not worked around; none was added to the backend.

| Feature | Missing capability | Frontend impact | Suggested backend change |
|---|---|---|---|
| History beyond the newest 200 runs | `GET /history` has `limit` (≤ 200) but no `offset` or cursor | Show more stops at 200 and says so; older runs cannot be reached in the app | An `offset` or cursor parameter, plus a total count |
| Searching history | No filename or date filter on `GET /history` | No search box; only the source filter | `q`, `from`, `to` query parameters |
| "Nothing analysed yet" on the Response desk | `GET /blueteam/status` returns `NOMINAL`, `CLEAN` and 0 when there is no analysis, the same as a clean batch | One extra `GET /history?limit=1` per refresh to tell them apart | A `has_analysis` flag, as `GET /trust/score` already has |
| Missing report confidence | `POST /reports/generate` sends `confidence: 0` and `attack_type: "unknown"` when the classification is missing | A missing confidence shows as 0.0% | Send `null` when there is no value |
| Activity for reference datasets and sample batches | `POST /datasets/real/{name}/analyze` and `POST /demo/run` write no audit event | Those runs are absent from Account › Recent activity (they are in History) | Log an audit event for both |
| An admin's own activity | `GET /audit/events` returns everyone's events to an admin, newest 50 by default, with no actor filter | The Account page filters to the admin's own events client-side, so older ones of theirs can be missing | An `actor=me` parameter |
| Editing an account | `/users` has list, create and delete only | Admins cannot change a role or reset a password; they delete and recreate | `PATCH /users/{id}` |
| One "latest result" for the Overview | `GET /analyze/upload/latest` returns the server's in-memory latest of an upload or reference dataset (then the newest stored upload), never a sample batch | After a sample batch the Overview shows the result `POST /demo/run` returned, labelled "Synthetic sample batch · just run"; after a reload it shows the latest upload or reference run again | A `source` parameter, or a latest endpoint across all sources |
| Cancelling a background job | No cancel endpoint | There is no Cancel control; leaving the page only stops watching, and the job still finishes and counts against the quota | `DELETE /jobs/{id}` |

Endpoints deliberately not used: `/forensics/narrative` and `/blast-radius/latest` (already in `/forensics/latest`); `/forensics/timeline`, `/datasets/demo`, `/datasets/demo/samples` and `/detect/*` (synthetic demo data, not the user's); `/trust/score` (letter grades derived from one heuristic score, which would overstate what is measured); `/analyze/upload/{id}` (`/history/{id}` returns the same stored result).
