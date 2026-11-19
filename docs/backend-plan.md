# Backend plan and running task list

Started 2026-11-19 from a review of `backend/` on main at `3e84659`. This file keeps the task list current: what is
done, what is pending, bugs found and bugs fixed. Update it in the same commit as the change.

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
| P3 | Pending | No refresh tokens: sessions end hard after 60 minutes. | Proposed: refresh token in its own httpOnly cookie, rotated on use, reuse detection ends the family. Needs a frontend change to renew silently. | — |
| P4 | Done | `DELETE /history/{id}` left the result in the in-memory caches, so `/forensics/latest` could still show it. | `dependencies.forget_result()` drops it from both caches. | `test_cache_and_usernames.py` |
| P5 | Done | No paging past `limit` on `/history`, `/audit/events`, `/users`. | `?before=<created_at>` cursor on all three; `next_before` in the body (`/history`, `/audit/events`) or the `X-Next-Before` header (`/users`, whose body stays a list). Bad cursors are 422. Calls without `before` answer exactly as before. | `test_paging.py` |
| P6 | Done | Owner-scoped lists filter on `owner_id` and sort by `created_at` with separate single-column indexes. | Migration 0007: `(owner_id, created_at DESC)` on `analysis_results`, `model_scans`, `defense_actions`, `hitl_cases`; `users(created_at)`. Not yet measured on PostgreSQL with production volume. | `test_paging.py::test_owner_and_time_indexes_exist` |
| P7 | Done | Admin-created duplicate usernames answered 400; sign-up 409. | 409 for both. The frontend shows the message and does not read the code. | `test_cache_and_usernames.py` |
| P8 | Done (check only) | `Alice` and `alice` could both exist. | Sign-up and admin creation refuse a name that matches an existing one in any letter case. No unique index on `lower(username)` yet: existing duplicates (if any) would make that migration fail, so check production data first. Sign-in stays exact-match. | `test_cache_and_usernames.py` |
| P9 | Done | Failed sign-ins were not in the audit log. | `auth.login_failed` with the IP: as the account's own event when the name exists (its owner sees it), as `anonymous` with no name when it does not (people type passwords into the name field). Also logged as warnings. Alerting on bursts is still to do. | `test_sessions_and_login_audit.py` |
| P10 | Done | No way to suspend an account without erasing it. | `POST /users/{id}/disable` and `/enable` (admin, not your own account, audited). Disabled: sign-in refused (403, only after the right password), sessions ended, API keys stop working, data kept. Migration 0008; `GET /users` adds `disabled_at`. No admin UI yet. | `test_account_disable.py` |

Full suite after the second batch: 237 passed, 8 skipped (SQLite, Windows).

New endpoints the frontend does not use yet: `POST /auth/logout-all`, `POST /users/{id}/disable`, `POST /users/{id}/enable`,
and the `before` cursor. Existing response shapes are unchanged; the only changed status code is P7 (400 → 409).

### Still pending

| # | Item | Notes |
|---|---|---|
| P3 | Refresh tokens | Design above; needs the frontend. |
| P8b | Unique index on `lower(username)` | After checking existing data for case duplicates. |
| P9b | Alert on bursts of failed sign-ins | Prometheus counter + alert rule, or Sentry. |
| P11 | Admin UI for disable/enable, sign out everywhere, and paging | Frontend work. |
| P12 | Admin 2FA (TOTP) | Planned under decision 3. |

## 4. Brief items that need a decision before work starts

1. **Standard response envelope** (`{"success", "data", "message"}` / `{"success": false, "error": {...}}`). Every
   endpoint and every caller changes shape at once: the frontend (`services/api.js`), the CLI and the e2e tests.
   Options: (a) keep the current shapes, and only standardise errors to `{"detail", "code"}` additively; (b) add it as
   `/api/v2` and move clients over; (c) switch in place in one coordinated change with the frontend.
   Recommendation: (a) now, (b) only if a public API is planned.
2. **Service and repository layers.** Routes call `database.py` directly. Extracting services is worth doing module by
   module when a module changes (auth first), not as one big refactor of a codebase someone else is also working on.
3. **Payments, push notifications, OAuth sign-in, 2FA, feature flags.** Nothing in the product needs payments today.
   2FA for admins is the one item worth planning.
4. **Ownership.** A collaborator works on the backend. Large changes should be agreed with them first to avoid conflicts.

## 5. Feature plan (phased)

**Phase 1: foundation.** Mostly in place: config from env, migrations, central request IDs, logging, `/health`,
`/metrics`, `/api/v1`, CI. Remaining: `/ready` (done above), consistent error codes (decision 1), and a written
environment-variable reference (`.env.example` exists).

**Phase 2: core.** Done: P1, P2 (sign out everywhere), P5, P6, P10. Remaining: P3 refresh tokens, admin 2FA.

**Phase 3: security and stability.** Done: P4, P7, P8 (check), P9. Remaining: P8b, P9b alerting, load test of the upload and analysis endpoints (`/loadtest` exists), backup and restore procedure in `docs/runbook.md`.

**Phase 4: advanced.** Only what the product needs: notification preferences for finished jobs, data export, then the
items under decision 3.

## 6. Log

- 2026-11-19: review of main at `3e84659`; fixes 1–4 with tests on `backend/review-fixes`; pending list P1–P10.
- 2026-11-19: second batch on `backend/plan-features`: P1, P2 (logout-all), P4–P10, one commit each with tests.
