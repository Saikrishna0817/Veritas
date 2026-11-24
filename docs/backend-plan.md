# Backend plan and running task list

Started 2026-11-19 from a review of `backend/` on main at `3e84659`. This file keeps the task list current: what is
done, what is pending, bugs found and bugs fixed. Update it in the same commit as the change.

## 0. Current status (2026-11-20, branch `backend/continuation`)

Branches, oldest first; each builds on the one before:

- `backend/review-fixes` (PR #42): fixes 1 to 4.
- `backend/plan-features` (PR #43): P1 to P10. CI passed on SQLite, PostgreSQL and e2e.
- `backend/continuation`: the third batch (section 3b), with main merged in at `fb3ab71`.

Full suite: 255 passed, 6 skipped, on SQLite on Windows. The 6 skipped tests need torchaudio and ffmpeg.

| Area | Status | Missing before "done" |
|---|---|---|
| Sign-in, sessions, CSRF, API keys | Complete | Refresh tokens (P3) and TOTP 2FA (P12) done on `backend/auth-refresh-2fa` |
| Sign-up, email verification, password reset | Complete | none known |
| Account self-service (usage, erase data, delete, keys) | Complete | none known (erasure is now one transaction) |
| Admin user management (list, create, role, delete, disable) | Complete in the API and UI | P11 |
| Analyses (CSV, images, audio, video, model scan, demo, real datasets) | Complete | detector quality is out of scope here (see model card) |
| Direct-to-storage upload | Complete | quota refund fixed (B7) |
| Background jobs | Complete | no automatic retry (analyses are deterministic, so a retry would fail the same way); stale jobs fail after 30 min |
| History, reports, audit log | Complete | paging on every list |
| Defense and HITL review | Complete | double-decision race fixed (B8) |
| Live detection stream (WebSocket) | Complete | revocation fixed (B5) |
| Observability | Complete | `/ready`, metrics, sign-in alert rule; no load test of production volume |
| Error shape and layering | Needs refactoring | decisions 1 and 2 (section 4) |

"Complete" means: endpoint, validation, authentication, owner scoping, error answers and tests are all there, and
there are no known bugs. The routes and the data layer were reviewed in depth. The detection engines were not
reviewed, because their code and numbers are out of scope.

## 1. What the backend is today

| Area | What is there |
|---|---|
| Stack | Python, FastAPI 0.162, Starlette 1.10, Pydantic 2, uvicorn; numpy / pandas / scikit-learn for detection |
| Database | SQLite (dev, tests) or PostgreSQL (`VERITAS_DATABASE_URL`) through `app/models/engine.py`; versioned SQL migrations in `app/models/migrations` (0001–0006), applied at start by `app/models/migrate.py` |
| Data access | `app/models/database.py`: parameterised queries only; every owner-scoped read takes an explicit owner (`ALL_OWNERS` for admins) |
| Auth | bcrypt hashes; HS256 JWT (60 min) in an httpOnly cookie with a double-submit CSRF token, or a bearer token, or a `vk_` API key; roles `user` / `analyst` / `admin`; `token_version` ends sessions when a password changes |
| Accounts | sign-up with email verification, password reset, resend, self-service data erasure and account deletion, API keys (max 10) |
| Abuse controls | sliding-window rate limits (Redis-backed when configured), daily quotas per user, Cloudflare Turnstile on public forms |
| Jobs | API threads, or RQ queues `veritas` and `veritas-untrusted` (media) when Redis is set; job rows in the `jobs` table |
| Files | multipart upload, or direct-to-storage (`/uploads/presign` + `/uploads/complete`) on local disk or S3 |
| Observability | request IDs (`X-Request-ID`), JSON logging, Sentry when configured, Prometheus `/metrics`, `/health` |
| Security headers | CSP, nosniff, frame DENY, referrer and permissions policy; CORS limited to configured origins |
| Tests | 222 pytest tests (SQLite and PostgreSQL in CI), Playwright e2e, CodeQL, dependency audit |

Layering: routes call `database.py` and the engines directly; `app/api/dependencies.py` and `analysis_helpers.py`
hold the shared logic. There is no separate service layer.

## 2. Bugs found and fixed (branch `backend/review-fixes`)

Each fix has a test in `backend/tests/test_backend_review_2026_11.py`; every test fails on the code before the fix.

| # | Severity | Problem | Root cause | Fix |
|---|---|---|---|---|
| 1 | High | When an admin deleted an account, the person stayed signed in until their token expired (up to 60 min), with the role in the token. Their results, scans, API keys and jobs were left in the database. | `DELETE /users/{id}` called `delete_user()`, which removes only the `users` row. `decode_access_token` trusts the claims of an id it does not know unless the id is in `deleted_accounts`, and only self-service deletion wrote there. | Admin deletion now uses the same `delete_account()` erasure as self-service deletion (records the id, removes the person's rows) and clears their cached results. 404 is checked before anything is deleted. |
| 2 | Medium | `PATCH /users/{id}` (role change) failed from a cross-origin frontend. | CORS `allow_methods` listed GET, POST, DELETE, OPTIONS; the preflight for PATCH was refused. | PATCH added. Same-origin deployments (nginx proxy) were not affected. |
| 3 | Medium | No readiness probe; the `/health` docstring pointed to a `/health/ready` route that does not exist. `/health` always answers 200, even with the database down. | Never added. | `GET /ready`: 200 when the database (and Redis, when configured) answers, 503 otherwise. The cause goes to the log, not the response. |
| 4 | Low | Admins could create usernames with spaces, tabs, `;` or markup, which sign-up refuses. | `CreateUserRequest.username` had no pattern. | Same pattern as sign-up (`^[A-Za-z0-9._-]+$`); 422 otherwise. |

Full suite after the fixes: 222 passed, 8 skipped (SQLite, on Windows). Audio and video tests skip without
torchaudio / ffmpeg on this machine; CI runs them.

## 3. Second batch (branch `backend/plan-features`, built on `backend/review-fixes`)

| # | Status | Problem | What was done | Test |
|---|---|---|---|---|
| P1 | Done | A signed token for an id in neither `users` nor `deleted_accounts` was accepted with the role in its claims, so a leaked `VERITAS_JWT_SECRET` could mint an admin. | `decode_access_token` refuses tokens whose account does not exist. Tests that minted tokens for unstored ids use `tests/auth_helpers.token_for`, which creates the account first. | `test_auth_api.py::test_a_token_for_an_account_that_does_not_exist_is_refused` |
| P2 | Partly done | Sign-out clears the cookies, but a copied token stays valid until it expires. | `POST /auth/logout-all` ends every session of the account on every device (bumps `token_version`); API keys keep working. Plain `/auth/logout` is unchanged (this browser only). | `test_sessions_and_login_audit.py` |
| P3 | Done | No refresh tokens: sessions ended hard after 60 minutes. | `POST /auth/refresh`: refresh token in its own httpOnly, SameSite=strict cookie (`veritas_refresh`, path `/api/v1/auth` so plain logout can end its own family), stored as SHA-256 (migration 0010), rotated on every use, one family per sign-in. Reusing a spent token revokes the family and logs `auth.refresh_reuse_detected`. Logout, logout-all, password change, disable and delete revoke refresh tokens. 14 days (`VERITAS_REFRESH_EXPIRY_DAYS`). `apiFetch` renews once on a 401 (single-flight) and retries. Two tabs refreshing at the same instant look like reuse and end the session (no grace window: the safer choice). | `test_refresh_tokens.py`, `frontend/src/services/refresh.test.js` |
| P4 | Done | `DELETE /history/{id}` left the result in the in-memory caches, so `/forensics/latest` could still show it. | `dependencies.forget_result()` drops it from both caches. | `test_cache_and_usernames.py` |
| P5 | Done | No paging past `limit` on `/history`, `/audit/events`, `/users`. | `?before=<created_at>` cursor on all three; `next_before` in the body (`/history`, `/audit/events`) or the `X-Next-Before` header (`/users`, whose body stays a list). Bad cursors are 422. Calls without `before` answer exactly as before. | `test_paging.py` |
| P6 | Done | Owner-scoped lists filter on `owner_id` and sort by `created_at` with separate single-column indexes. | Migration 0007: `(owner_id, created_at DESC)` on `analysis_results`, `model_scans`, `defense_actions`, `hitl_cases`; `users(created_at)`. Not yet measured on PostgreSQL with production volume. | `test_paging.py::test_owner_and_time_indexes_exist` |
| P7 | Done | Admin-created duplicate usernames answered 400; sign-up 409. | 409 for both. The frontend shows the message and does not read the code. | `test_cache_and_usernames.py` |
| P8 | Done (check only) | `Alice` and `alice` could both exist. | Sign-up and admin creation refuse a name that matches an existing one in any letter case. No unique index on `lower(username)` yet: existing duplicates (if any) would make that migration fail, so check production data first. Sign-in stays exact-match. | `test_cache_and_usernames.py` |
| P9 | Done | Failed sign-ins were not in the audit log. | `auth.login_failed` with the IP: as the account's own event when the name exists (its owner sees it), as `anonymous` with no name when it does not (people type passwords into the name field). Also logged as warnings. Alerting on bursts is still to do. | `test_sessions_and_login_audit.py` |
| P10 | Done | No way to suspend an account without erasing it. | `POST /users/{id}/disable` and `/enable` (admin, not your own account, audited). Disabled: sign-in refused (403, only after the right password), sessions ended, API keys stop working, data kept. Migration 0008; `GET /users` adds `disabled_at`. Admin UI in P11. | `test_account_disable.py` |

Full suite after the second batch: 237 passed, 8 skipped (SQLite, Windows).

New endpoints the frontend does not use yet: `POST /auth/logout-all`, `POST /users/{id}/disable`, `POST /users/{id}/enable`,
and the `before` cursor. Existing response shapes are unchanged; the only changed status code is P7 (400 → 409).

## 3b. Third batch (branch `backend/continuation`)

Each fix has a test that fails on the code before it. The tests are in `test_ws_session_revocation.py`,
`test_backend_review_2026_11_20.py` and `test_login_metrics.py`.

| # | Severity | Problem | Root cause | Fix |
|---|---|---|---|---|
| B5 | Medium (security) | After sign-out everywhere, disabling or deleting an account, an open live stream kept receiving that person's events for up to 60 min. | `/ws/v1/detection-stream` checked only the token's expiry. | The stream re-checks the session every 30 s and on every frame, and closes it (1008). Broadcasts skip and close a socket whose session has ended. |
| B6 | Medium | `/analyze/model/history?limit=-1` returned every scan on SQLite and a 500 on PostgreSQL. `/datasets/demo/samples` took any limit and offset. | Plain `int` parameters with no bounds. | `limit` is 1 to 200 with the `before` cursor, as on `/history`. Demo samples take a limit of 1 to 500 and an offset of 0 or more. |
| B6b | Low | A `next_before` pasted into a URL without encoding was refused: its `+00:00` arrived as ` 00:00`. | `+` means a space in a query string. | Cursors now end in `Z`, and a `+` that arrived as a space is still read correctly. |
| B7 | Medium (data integrity) | Account erasure could stop half-way: the data was gone but the account remained. On SQLite, the uncommitted deletes stayed on the shared connection for the next unrelated commit to write. | Two separate commits, and no rollback on error. | One transaction with a rollback; `delete_owner_data` rolls back the same way. |
| B8 | Low | When queueing a direct upload failed, the files were deleted but the day's quota was still charged. | No refund in the error path. | `quota.refund` runs in the error path. |
| B9 | Medium (data integrity) | Two concurrent HITL decisions on one case were both recorded, and both side effects ran (for example, a quarantine after the case had been marked safe). | Read-then-write without a condition. | The case is claimed with `UPDATE ... WHERE status = 'pending'` in the same transaction; the second request gets 404. |
| P9b | Feature | No alert on bursts of failed sign-ins. | — | `veritas_auth_login_failures_total{outcome}` counts wrong passwords, unknown names and rate-limited attempts, with no names or IPs. The runbook has the alert rule. |

Checked and found fine: job status is scoped to its owner (404 for other people's jobs); presigned uploads are
single-use and size-checked; WebSocket origin checking.

### Still pending

| # | Item | Notes |
|---|---|---|
| P8b | Unique index on `lower(username)` | First run `SELECT lower(username), count(*) FROM users GROUP BY 1 HAVING count(*) > 1` on production. If it returns no rows, the index can be added as migration 0009; if it does, rename those accounts first. |
| P13 | Load test at production volume | `/loadtest` exists. Run it against staging with PostgreSQL to measure the 0007 indexes. |
| P14 | Delete abandoned direct-upload blobs | The `pending_uploads` rows go with the account, but the S3 objects rely on a bucket lifecycle rule. Check that the rule exists in production. |
| P11 | Admin UI for disable/enable, sign out everywhere, and paging | Done 2026-11-24. Disable / Enable with inline confirm, "Sign out on every device" on the Account page, "Load more" on users, audit log, stored results, History and model scans. |
| P12 | Admin 2FA (TOTP) | Done: see section 3c. Not yet done: QR code in the UI (no QR library installed), admin UI to reset another person's 2FA, requiring 2FA on the data endpoints an admin reaches through `require_user` (only `require_admin` endpoints are gated). |

## 3c. Fourth batch (branch `backend/auth-refresh-2fa`)

- Auth service extracted (decision 2, auth part): `app/services/auth.py` holds sign-in checks, session issue and end, sign-up and mail flows; routes call it.
- P3 refresh tokens: see the P3 row above. Migration 0010.
- P12 TOTP 2FA (RFC 6238, standard library only, RFC vectors tested). Migration 0011. `POST /auth/2fa/setup` returns the secret and
  an `otpauth://` URI once; `/enable` confirms with a code and returns 8 one-time recovery codes (stored hashed); `/disable` needs a
  current code. With 2FA on, `POST /auth/token` answers `{"two_factor_required": true, "challenge": ...}` (5 minutes, not usable as
  a session) and `POST /auth/2fa/verify` takes the challenge plus an authenticator or recovery code. Verify is limited per IP (10 per
  5 min) and per account (5 per 5 min); each 30 s step works once. The secret is stored encrypted with a key derived from
  `VERITAS_JWT_SECRET`: changing that secret makes people re-enroll. `VERITAS_REQUIRE_ADMIN_2FA=true` makes `require_admin`
  endpoints answer 403 until the admin has enrolled. `GET /auth/me` adds `two_factor_enabled`.

## 4. Brief items that need a decision before work starts

1. **Standard response envelope** (`{"success", "data", "message"}` / `{"success": false, "error": {...}}`). Every
   endpoint and every caller changes shape at once: the frontend (`services/api.js`), the CLI and the e2e tests.
   Options: (a) keep the current shapes, and only standardise errors to `{"detail", "code"}` additively; (b) add it as
   `/api/v2` and move clients over; (c) switch in place in one coordinated change with the frontend.
   Recommendation: (a) now, (b) only if a public API is planned.
2. **Service and repository layers.** Auth part done (`app/services/auth.py`, 2026-11-24); other modules still to do. Routes call `database.py` directly. Extracting services is worth doing module by
   module when a module changes (auth first), not as one big refactor of a codebase someone else is also working on.
3. **Payments, push notifications, OAuth sign-in, 2FA, feature flags.** Nothing in the product needs payments today.
   2FA for admins is the one item worth planning.
4. **Ownership.** A collaborator works on the backend. Large changes should be agreed with them first to avoid conflicts.

## 5. Feature plan (phased)

**Phase 1: foundation.** Mostly in place: config from env, migrations, central request IDs, logging, `/health`,
`/metrics`, `/api/v1`, CI. Remaining: `/ready` (done above), consistent error codes (decision 1), and a written
environment-variable reference (`.env.example` exists).

**Phase 2: core.** Done: P1, P2 (sign out everywhere), P3, P5, P6, P10, P12.

**Phase 3: security and stability.** Done: P4, P7, P8 (check), P9, P9b and B5 to B9. The backup and restore
procedure is already in `docs/runbook.md`. Remaining: P8b, P13 load test, P14.

**Phase 4: advanced.** Only what the product needs: notification preferences for finished jobs, data export, then the
items under decision 3.

## 6. Log

- 2026-11-19: review of main at `3e84659`; fixes 1–4 with tests on `backend/review-fixes`; pending list P1–P10.
- 2026-11-19: second batch on `backend/plan-features`: P1, P2 (logout-all), P4–P10, one commit each with tests.
- 2026-11-20: PR #43 CI green. Third batch on `backend/continuation` (main merged in): B5–B9 fixed, P9b done.
- 2026-11-24: fourth batch on `backend/auth-refresh-2fa`: auth service extracted, P3 refresh tokens, P12 admin 2FA.
- 2026-11-24: P11 on `frontend/admin-p11`: disable/enable, sign out everywhere, Load more on paged lists.
