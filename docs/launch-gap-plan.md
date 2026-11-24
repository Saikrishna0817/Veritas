# Launch gap plan

Started 2026-11-25 on branch `launch/gap-plan`, cut from `origin/main` at `0878b79`. This file holds the final
requirements, what the code does today, and the work to close the gap. Update it in the same commit as each change.

## 1. Sources and conflicts

Read: `docs/PRODUCT.md` (2026-11-16), `docs/roadmap-to-launch.md` (2026-11-07), `docs/backend-plan.md` (2026-11-24),
`docs/trust-plan.md` (2026-11-22), `docs/decisions.md` (2026-11-07), `README.md` (2026-11-24), `docs/HANDOVER.md`.
Where two disagree, the newer one wins.

| # | Conflict | Older says | Newer says | Taken |
|---|---|---|---|---|
| K1 | `DESIGN.md` | The brief asks to read it | No `DESIGN.md` exists in the repository. The design tokens live in `frontend/src/styles/argilla.css` and the shared code in `frontend/shared/argilla` | Treated `argilla.css` as the design reference |
| K2 | Brand | `PRODUCT.md`: identity "being rethought"; anti-references include decorative 3D | The ARGILLA look (warm void, keycaps, one 3D scene that shows real state) became the app's identity in PR #35 (2026-11-17); a `PRODUCT.md` edit was proposed there | ARGILLA kept, as the brief requires. `PRODUCT.md` still lists decorative 3D as an anti-reference: decision D4 |
| K3 | Scope | Roadmap §1: tables and model files only; images, audio, video out of v1 | `HANDOVER.md`, `trust-plan.md`, `README.md`: all four data types | All four data types |
| K4 | Sessions | `decisions.md` 2026-09-02: JWT in `localStorage` | `backend-plan.md`: httpOnly cookie + CSRF, refresh tokens | Cookie sessions (what the code does). `decisions.md` gets a superseding entry (task D2) |
| K5 | Isolation | `decisions.md` 2026-11-06: results shared by every user | `backend-plan.md`, README: owner-scoped, admins see everyone's | Owner-scoped (what the code does); same superseding entry |
| K6 | Verdict names | Roadmap §1: drop `CONFIRMED_POISONED` until the benchmark supports it | Model card and README: the verdict is a calibrated false-alarm level (1 in 10,000) | Kept as in the newer docs |

## 2. What the finished product has to deliver

**Features.** Check a CSV, or a zip of images, audio or video, with or without a trusted reference; verdict as a
false-alarm level, evidence per check, the items to review (line numbers, cells, files), attack profile for tables,
limits shown beside the result. Model-file scan that never loads the file. History, reports, CSV export of the list,
review queue. Accounts: sign-up with email confirmation, password reset, sessions in httpOnly cookies, refresh, sign
out everywhere, API keys, CLI, usage and quotas, erase my data, delete my account. Admin: users (create, role,
disable, delete), audit log, defence actions, admin 2FA.

**Quality bar.** Every number from a detector measured on held-out data (`docs/model-card.md`, `make claims` in CI).
Nothing simulated. Evidence before verdict; limits beside the claim; calm wording. Accessible (WCAG 2.1 AA), 360 px to
1440 px+, real loading, empty and error states.

**Security.** Owner isolation with an automated test; CSRF; rate limits and quotas; Turnstile on public forms; uploads
size-checked, untrusted files never unpickled in the web process; scanner in a separate process; security headers and
CSP; admin 2FA available, optionally required; audit log of sign-ins and admin actions.

**Performance and operations.** Queue workers apart from the API, PostgreSQL, backups with a tested restore,
monitoring and an alert, staging, load test.

**Docs.** README, model card and benchmark findings, environment reference, runbook, legal drafts reviewed by a person.

**Launch criteria** (roadmap §0): measurably works; safe for strangers; stays up; honest; legally deployable.

## 3. Status before this branch (2026-11-25)

Test runs on `0878b79`, this machine (Windows, Python 3.13 venv outside the repository, Node 24):

- Backend: `pytest` 250 passed, 18 skipped (skips need torch, moto, Redis, PostgreSQL or ffmpeg, none installed here).
- Frontend: `eslint` clean; `vite build` passes; unit tests 18 of 18 pass when run as
  `node --test src/services/*.test.js`, but `npm test` fails on Node 24 (see C1).
- e2e: Playwright 7 of 7 passed against a local API (SQLite) and the Vite dev server, Chrome as the browser.
- CI on `0878b79`: backend (SQLite, PostgreSQL), frontend, e2e, security and CodeQL all green.

| Requirement | Status | Evidence |
|---|---|---|
| Table, image, audio, video checks with verdict, evidence, items, limits | DONE | `api/routes/upload.py`, `components/results.jsx`, e2e test 1; image/audio/video tests skip locally, pass in CI |
| Reference choices (file, cutoff column, opt-in split, screening) | DONE | README; `test_upload*`; Check data page |
| Model-file scan, never loads the file, separate process | DONE | `scanner/model_scan.py` (`scan_sandboxed`), `test_model_scan.py`, `docs/scanner-benchmark.md` |
| Benchmark-backed numbers, checkable claims | DONE | `test_claims.py` in CI, Benchmarks page |
| Nothing simulated | DONE | No simulated pages left; one stale `// Federated` comment in `services/api.js` (NEEDS POLISH, P1) |
| History, reports, CSV export of the review list, review queue | DONE | `HistoryPage`, `ReportsPage`, `results.jsx` export, admin Defence tab, e2e test 7 |
| Sign-up, verification, reset, cookies, CSRF, refresh, sign out everywhere, API keys | DONE | `services/auth.py`, `test_refresh_tokens.py`, `test_auth_api.py`, e2e tests 1–6 |
| Owner isolation | DONE | `owner_scope`, `test_tenant_isolation.py` (two users, by id, latest, history, reports, review queue) |
| Admin 2FA, required for admins when `VERITAS_REQUIRE_ADMIN_2FA=true` | PARTLY DONE | Only `require_admin` routes check enrollment. An admin without 2FA still reads everyone's results, scans and the full audit log through `owner_scope`, `_owned` and `/audit/events` (S1) |
| Admin can reset another person's 2FA (lost phone and codes) | MISSING | No endpoint; today it takes a database edit (F1) |
| 2FA enrollment shows a QR code | MISSING | Needs a QR library (no new dependencies allowed): proposal only (F3) |
| Retention (`VERITAS_RETENTION_DAYS`, promised in the privacy policy) | PARTLY DONE | `purge_older_than` runs once per process start (`api/dependencies.py:38`); a server that stays up never purges again (S2) |
| Self-service copy of my data | MISSING | Privacy policy says "email us"; no endpoint (F2) |
| Frontend unit tests run in CI | MISSING | `ci.yml` runs lint and build only; `npm test` breaks on Node 24 (C1) |
| Quotas, rate limits, Turnstile | DONE | `core/quota.py`, `core/rate_limit.py`, `test_accounts.py`; Turnstile only when a site key is set |
| Queue, PostgreSQL, backups, staging, load test | DONE in code | `docker-compose*.yml`, `scripts/`, `docs/staging.md` (P13 run); production deploy NOT VERIFIED (no secrets here) |
| Monitoring and alert | DONE in code | `/metrics`, `/ready`, Sentry hook, runbook alert rule; not running anywhere (NOT VERIFIED) |
| Abandoned direct-upload blobs (P14) | PARTLY DONE | Rule in `docs/runbook.md`; applying it needs the production bucket (NOT VERIFIED) |
| Terms and privacy reviewed by a person | MISSING | `docs/legal/*` are drafts with placeholders: needs a person (decision D1) |
| Accessibility, 360–1440 px | NEEDS POLISH | Checked by hand in earlier PRs; no automated a11y check (no new dependencies): spot check in this branch |
| README API table | NEEDS POLISH | Missing `/auth/refresh`, `/auth/logout-all`, `/auth/2fa/*`, `/users/{id}` PATCH, disable, enable, `/ready` (D1 docs task) |
| `decisions.md` | NEEDS POLISH | Two entries contradict the code (K4, K5) |

## 4. Plan

Order: security and data integrity, broken core flows, launch blockers, missing features, polish, docs.

| # | Task | Scope and files | Test | Size | Status |
|---|---|---|---|---|---|
| S1 | Admin 2FA requirement also covers reading other people's data | `api/dependencies.py` (`owner_scope`), `api/routes/upload.py` (`_owned`), `api/routes/reports.py` (`/audit/events`): an admin who must enroll and has not sees only their own rows | new `backend/tests/test_admin_2fa_scope.py` (fails before: 200 instead of 404) | S | done |
| S2 | Retention purge runs on a schedule, not only at start | `api/dependencies.py`, `main.py` lifespan: purge when due (every 6 h) | new `backend/tests/test_retention_schedule.py` (3 tests, all fail before) | S | done; also empties the latest-result caches when a purge deletes rows |
| C1 | Frontend unit tests run, and run in CI | `frontend/package.json` test script, `.github/workflows/ci.yml` | `npm test` passes on Node 24 here (18 of 18; it failed before: "Cannot find module src\services"); CI runs it on Node 22 | S | done |
| B1 | Launch blockers that need accounts or a person | deploy secrets, SMTP, Turnstile, Sentry, R2, domain, legal review, P14 in production | — | — | NOT DOABLE HERE: decisions D1–D3 |
| F1 | Admin resets another person's 2FA | `services/auth.py`, `api/routes/users.py` (new `POST /users/{id}/2fa/reset`, admin, not self, audited), `models/database.py`, `pages/admin/UsersTab.jsx`, `services/api.js` | `backend/tests/test_admin_2fa_reset.py` (2 tests, fail before: 404); browser check 1440 and 390 px | M | done. Found while checking: an open confirm sat inline beside the row buttons and pushed the accounts table wider than its frame (sideways scroll, text cut off at 1440 px); it now opens under the buttons (`argilla.css`), 0 px overflow at both widths |
| F2 | Download a copy of my data | new `GET /account/export` (JSON attachment: account, analyses and scans with full results, defence actions, review cases, my audit events, key names and prefixes, today's usage; columns named one by one so no hash or secret can appear), audited `account.data_exported`; "Download my data" on the Account page | `backend/tests/test_account_export.py` (2 tests, fail before: 404); browser check 1440 and 390 px: file saved, holds the user's analyses, no `password_hash` / `totp_secret` / `key_hash` | M | done |
| F3 | QR code for 2FA enrollment | needs a QR library | — | — | PROPOSAL (D5) |
| P1 | Remove the stale `// Federated` comment | `services/api.js` | lint | S | done |
| D1 | README API table brought up to date | `README.md` (also `/account/export`, `/users/{id}/2fa/reset`); `docs/legal/privacy-policy.md` names the download; `docs/backend-plan.md` P12 row | — | S | done |
| D2 | Superseding entries in `decisions.md` for K4, K5 | `docs/decisions.md` | — | S | done |

## 5. Decisions left for the owner

- **D1 Legal.** `docs/legal/*` need a person to fill placeholders and review them before public launch.
- **D2 Accounts and deploy.** Hosting, domain, `DEPLOY_*` secrets, SMTP, Turnstile, Sentry and R2/S3 credentials;
  then apply the P14 lifecycle rule and run the restore drill on the real stack.
- **D3 Require admin 2FA in production?** `VERITAS_REQUIRE_ADMIN_2FA` defaults to `false`.
- **D4 `PRODUCT.md` brand section.** It still says the identity is being rethought and lists decorative 3D as an
  anti-reference, while ARGILLA (with its 3D scene) is the identity. Update `PRODUCT.md` or confirm the exception.
- **D5 QR code for 2FA.** Needs a small QR library (for example `qrcode`); today people type the secret by hand.

## 6. Log

- 2026-11-25: audit of `0878b79`; plan written.
