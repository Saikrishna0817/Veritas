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

## 3. Found, not fixed yet (pending)

| # | Severity | Problem | Proposed fix | Why it waits |
|---|---|---|---|---|
| P1 | Medium | A signed token for an id that is in neither `users` nor `deleted_accounts` is accepted with the role in its claims. Only someone holding `VERITAS_JWT_SECRET` can mint one, so the risk is a leaked secret. | Refuse tokens whose user does not exist (fail closed). | About 15 test files mint tokens for ids that are not in the database; they need a helper that creates the user first. Mechanical but touches many files. |
| P2 | Medium | Sign-out clears the cookies, but the token itself stays valid until it expires. A copied token keeps working. | Short access token plus a refresh token, or a server-side session id checked per request; or bump `token_version` on sign-out ("sign out everywhere"). | Design decision (one device vs all devices); changes the auth contract. |
| P3 | Medium | No refresh tokens: sessions end hard after 60 minutes. | Refresh token in its own httpOnly cookie with rotation and reuse detection. | Same as P2; frontend change needed. |
| P4 | Low | `DELETE /history/{id}` leaves the result in the in-memory "latest" cache, so `/forensics/latest` can show a deleted result until restart or eviction. | Clear matching cache entries on delete. | Small; next batch. |
| P5 | Low | List endpoints have `limit` but no `offset` / cursor (`/history`, `/audit/events`, `/users`); `/users` returns at most 200. | Cursor pagination (`before=<created_at,id>`), additive query parameters. | Additive API change; frontend needs it to page. |
| P6 | Low | Owner-scoped lists filter on `owner_id` and sort by `created_at`, but the indexes are separate single-column ones. | Migration 0007: composite `(owner_id, created_at DESC)` indexes on `analysis_results`, `model_scans`, `defense_actions`, `hitl_cases`. | Needs a check on PostgreSQL with realistic volume. |
| P7 | Low | Admin-created duplicate usernames answer 400; sign-up answers 409. | 409 everywhere. | Changes a status code the frontend may read. |
| P8 | Low | Usernames are unique case-sensitively (`Alice` and `alice` can both exist), while the login rate limit keys on the lower-cased name. | Unique index on `lower(username)` plus a check at creation. | Needs a migration and a decision about existing duplicates. |
| P9 | Info | Failed sign-ins are rate-limited but not written to the audit log. | Audit `auth.login_failed` with the IP (not the password), and alert on bursts. | Small; next batch. |
| P10 | Info | No account deactivation (suspend without erasing). | `users.disabled_at`, checked at sign-in and per request; admin endpoint. | New feature. |

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

**Phase 2: core.** Auth is in place; remaining: P1, P2/P3, P10, admin 2FA. Pagination and indexes: P5, P6.

**Phase 3: security and stability.** Audit of failed sign-ins (P9), cache consistency (P4), username rules (P8),
load test of the upload and analysis endpoints (`/loadtest` exists), backup and restore procedure in `docs/runbook.md`.

**Phase 4: advanced.** Only what the product needs: notification preferences for finished jobs, data export, then the
items under decision 3.

## 6. Log

- 2026-11-19: review of main at `3e84659`; fixes 1–4 with tests on `backend/review-fixes`; pending list P1–P10.
