# Launch plan: invite-only beta (2026-11-25)

The goal from the council review: Veritas is **closed as an invite-only beta** when it is deployed and monitored, its
legal pages are reviewed and match what the app does, a 3-5 person trial has a result against a bar set in advance, and
the model card's numbers come from the full rerun (or say how much was rerun). Everything not on this list is out of
scope until then (see "Not doing" at the end).

Work through the tasks in order; one task at a time, each closed by its "Done when" line. **Owner**: *You* = needs your
accounts, money, identity or judgement; *Claude* = can be done in the repo or on a machine you give access to.

## Stage 0. Decide before starting (you, 30 min)

| # | Task | Owner | Done when |
|---|---|---|---|
| 0.1 | Set the trial bar: how many people, which dataset types, and what counts as a pass (proposal: at least 3 of 5 people say the top 20 flagged items were worth reviewing, and at least one dataset where they confirm real problems). | You | Written into this file under "Trial bar". |
| 0.2 | Set the stop rule: what happens if the trial fails (proposal: narrow the claim in the model card and app to what passed, or present Veritas as a benchmark plus research demo; no new detector research). | You | Written under "Stop rule". |
| 0.3 | Set a deadline for the whole plan (proposal: 3 weeks from the start of stage 1). | You | Date written below. |

Trial bar: _to fill_ · Stop rule: _to fill_ · Deadline: _to fill_

## Stage 1. Accounts (you, 1-2 h)

All free tier. Turn on 2FA everywhere; store credentials in a password manager, never in the repo or chat.

| # | Task | Owner | Done when |
|---|---|---|---|
| 1.1 | Domain (any registrar; Cloudflare Registrar is at cost) and a Cloudflare account with the domain added. | You | DNS for the domain is served by Cloudflare. |
| 1.2 | Oracle Cloud Always Free: one Ampere A1 VM (4 OCPU, 24 GB, Ubuntu 24.04), SSH key login only, ports 22/80/443 open. | You | `ssh` into the VM works. |
| 1.3 | Cloudflare R2 bucket for uploads and one for backups, plus an API token scoped to them. | You | Bucket names and token saved in the password manager. |
| 1.4 | Email sender (Resend or Brevo), domain verified (SPF/DKIM records in Cloudflare). | You | The provider shows the domain verified. |
| 1.5 | Sentry (errors), UptimeRobot (uptime), optional Grafana Cloud (metrics). Spending alerts on Oracle and Cloudflare at $1. | You | Accounts exist; spend alerts set. |
| 1.6 | Decide the database: PostgreSQL in compose on the VM (default, simplest, backed up to R2) or Neon. | You | Choice noted here. |

## Stage 2. Make the code deployable (Claude, in the repo, can start now)

| # | Task | Owner | Done when |
|---|---|---|---|
| 2.1 | ~~ARM images~~ (done 2026-11-25): build backend and frontend for `linux/arm64` (the Oracle VM is ARM) in the tag workflow; check torch CPU wheels and ffmpeg on arm64. | Claude | A tagged build pushes arm64 images to GHCR; a local `docker buildx` arm64 build starts the API. |
| 2.2 | ~~Production compose overlay~~ (done 2026-11-25): Caddy or Cloudflare origin certificate for HTTPS, HSTS, `VERITAS_COOKIE_SECURE=true`, memory caps sized for the VM, log rotation. | Claude | `docker-compose.prod.yml` plus a runbook section; config validates. |
| 2.3 | ~~Invite-only mode~~ (done 2026-11-25): `VERITAS_SIGNUP_ENABLED=false` hides sign-up in the UI, and the admin creates accounts (already in the API); a short "request access" text instead of the form. | Claude | e2e test: sign-up page shows the invite text when disabled. |
| 2.4 | ~~Beta notice~~ (done 2026-11-25): a visible "beta, no warranty" line in the app, and the flagged-list precision stated where results are shown (from the model card: on real label errors 2-16% of listed items were confirmed). | Claude | Shown on the results page; e2e checks it. |
| 2.5 | Kill switch and abuse handling: a documented one-line way to take the site offline (maintenance page in nginx), how to disable an account and delete an upload, and an upload content policy in the terms. | Claude | Runbook "Incident: take it offline" section; tested on staging. |
| 2.6 | ~~Trial feedback~~ (done 2026-11-25): check the results view lets a person mark each flagged item "really bad / fine" and that the marks are stored per item with consent; build the missing part, plus an admin export of the marks. | Claude | e2e: mark an item, export shows it. |

## Stage 3. First deployment (you + Claude, about a week with bug fixing)

| # | Task | Owner | Done when |
|---|---|---|---|
| 3.1 | Give Claude a deploy user on the VM (SSH key), or run the commands Claude writes. Put secrets in `/opt/veritas/.env` on the VM yourself. | You | Claude can run `docker compose ps` on the VM, or you can paste its output. |
| 3.2 | Bring the stack up on the VM (migrations run at start), point the domain at it through Cloudflare (proxied, Full (strict) TLS). | Claude | `https://<domain>/health` is OK from outside. |
| 3.3 | Real email: sign up a test account, receive the verification mail, reset a password. | Claude + You | Both mails arrive (inbox, not spam). |
| 3.4 | Direct uploads to R2: CORS on the bucket, lifecycle rule for abandoned uploads (runbook P14 section), one image and one CSV analysis end to end. | Claude | Both analyses finish; `get-bucket-lifecycle-configuration` shows the rule. |
| 3.5 | Migration 0009 (case-insensitive usernames) has run on production; check for renamed accounts. | Claude | `schema_migrations` lists 0009-0011; renamed list empty or reviewed. |
| 3.6 | Backups: nightly `scripts/backup.sh` to the R2 backup bucket, then one real restore drill into a scratch database. | Claude | Restore drill output with matching row counts in the runbook log. |
| 3.7 | Monitoring: UptimeRobot on `/health`, Sentry DSN set (backend and frontend), failed-sign-in alert, an alert that reaches your phone. | Claude + You | A forced test error appears in Sentry; stopping the API pages you. |
| 3.8 | CI deploy: set the SSH secrets in GitHub, tag `v1.0.0-beta.1`, check the deploy job runs and takes a backup. | You + Claude | The tag deploys without manual steps. |
| 3.9 | Turn invite-only on, create your and your collaborator's accounts; give the collaborator access to the runbook and the deploy (bus factor). | You | Collaborator can deploy a tag. |

## Stage 4. Legal (you, with Claude drafting, 1-2 days)

| # | Task | Owner | Done when |
|---|---|---|---|
| 4.1 | Make the privacy policy match the app: what is stored (accounts, uploads, results, audit log), where (VM region, R2, Sentry, email provider), retention (uploads after N days, results until deleted), deletion (the delete-account flow). | Claude | Diff against `frontend/src/content/legal/` lists every data flow with its retention. |
| 4.2 | Terms: beta, no warranty, results are screening aids not verdicts, upload content policy, MIT licence for the code, governing law. | Claude | Draft ready for review. |
| 4.3 | One human review: a lawyer, or a vetted template service, against the drafts. | You | Reviewed versions published; date on both pages. |

## Stage 5. Trial (you, 1-2 weeks of waiting)

| # | Task | Owner | Done when |
|---|---|---|---|
| 5.1 | Recruit 3-5 people with their own data (classmates, colleagues, ML communities); short consent text: what they upload, how long it is kept, that marks are used to measure precision. | You | Names and consent recorded. |
| 5.2 | Onboarding: one-page guide (what to upload, how to read the list, how to mark items), and a feedback channel (a form or email). | Claude | Guide linked from the app for trial accounts. |
| 5.3 | Run: each person checks one dataset and marks the top of the list. Ask what they expected Veritas to find and it did not. | You | Marks exported. |
| 5.4 | Score against the trial bar; write `docs/trial-results.md`; every reported miss becomes a named limit or a new attack in the benchmark. | Claude | Results file and model card updated. |
| 5.5 | Apply the stop rule if the bar is missed. | You | Decision written in this file. |

## Stage 6. Numbers (Claude, unattended, any time the laptop is free)

| # | Task | Owner | Done when |
|---|---|---|---|
| 6.1 | Resume the full rerun from tables-breadth (memory note has the command); or run it on the Oracle VM before the app goes live there. | Claude | `make claims --fresh` passes for all ten benchmarks. |
| 6.2 | If it is not finished by the deadline: state in the model card which benchmarks were rerun and which still quote the original runs. | Claude | Model card says so. |

## Stage 7. Close (Claude + you, half a day)

| # | Task | Owner | Done when |
|---|---|---|---|
| 7.1 | Update the model card, trust plan and README with the trial and rerun results; tag `v1.0.0-beta`. | Claude | Tag pushed and deployed. |
| 7.2 | Maintenance plan: who watches alerts, Dependabot cadence (monthly batch), what to do if the free VM is reclaimed (restore from the R2 backup to a new VM, tested in 3.6). | You + Claude | Section in the runbook. |
| 7.3 | Mark this plan done. | You | — |

## Not doing (until after the beta)

2FA QR code, admin UI to reset 2FA (use a database command), thumbnails and audio playback for flagged files, admin
console restyle, more service-layer refactoring, a docs site, three.js 0.187 and Python 3.15 upgrades, new detector
research or tuning for the known weaknesses (they stay listed in the model card), production-scale load tests (the k6
runs cover a beta), a public launch post.
