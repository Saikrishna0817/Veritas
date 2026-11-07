# Runbook

Stack: `docker-compose.yml` = Postgres, Redis, API (`backend`), analysis worker (`worker`), nginx + SPA (`frontend`).
Put TLS/CDN (Cloudflare, Caddy) in front; the API and Postgres/Redis ports are not published.

## First deploy
1. VM with Docker (Oracle Always Free ARM, any VPS). `git clone`, `cp .env.example .env`, fill the required values
   (`POSTGRES_PASSWORD`, `VERITAS_JWT_SECRET`, `VERITAS_ADMIN_*`) and for a public host `VERITAS_PUBLIC_URL`,
   `CORS_ALLOW_ORIGINS`, `VERITAS_COOKIE_SECURE=true`, SMTP settings, `VERITAS_TURNSTILE_SECRET`.
2. `docker compose up -d --build`. The API container applies database migrations before it starts.
3. Check `docker compose ps` (all healthy), sign in as the admin, run one upload and one model scan.
4. Cron: `0 3 * * * cd /opt/veritas && BACKUP_S3_URL=s3://bucket/veritas BACKUP_S3_ENDPOINT=... scripts/backup.sh`.

## Deploy / roll back
- Tag `vX.Y.Z`; CI builds, pushes images and (if secrets are set) deploys over SSH, then takes a backup.
- Manual: `git fetch --tags && git checkout vX.Y.Z && docker compose up -d --build`.
- Rollback: check out the previous tag and `up -d --build`. Migrations only add tables, columns and indexes, so the previous
  version runs on the newer schema. A destructive migration needs its own plan; never ship one without a tested restore first.

## Staging

Staging is the same compose file under another project name: separate network, Postgres volume and upload volume, and a
different published port. No override file is needed (`docker compose config` shows `veritas-staging_pgdata` etc.).

```bash
cp .env .env.staging        # then change WEB_PORT=5174, VERITAS_PUBLIC_URL, CORS_ALLOW_ORIGINS, a different JWT secret and DB password
docker compose -p veritas-staging --env-file .env.staging up -d --build
```

Point a second hostname (for example `staging.your.domain`) at port 5174 in Cloudflare/Caddy. Use test SMTP and Turnstile
keys there. Deploy a tag to staging first, run `e2e/` against it
(`BASE_URL=https://staging.your.domain E2E_USER=... E2E_PASS=... npx playwright test`), then deploy production.

## Backups and restore
- `scripts/backup.sh` writes `backups/veritas-<utc>.dump` (pg_dump custom format), keeps the newest 14, optionally uploads to S3/R2.
- `scripts/restore-drill.sh [dump]` restores into a scratch database and compares row counts. Run it monthly and after changing
  the backup. **A backup that has not been restored is not a backup.**
- Real restore: stop `backend` and `worker`, `docker compose exec -T postgres pg_restore -U veritas -d veritas --clean --if-exists < dump`, start them.

## Monitoring and alerts
- Uptime: point UptimeRobot / Better Stack at `https://<host>/health` (alert on non-200 or `"status":"degraded"`).
- Metrics: set `VERITAS_METRICS_TOKEN`; Prometheus / Grafana Agent scrapes `/metrics` with `Authorization: Bearer <token>`
  (hidden without a token). Series: `veritas_http_requests_total`, `veritas_http_request_seconds`, `veritas_jobs_total`.
  Suggested alerts: 5xx rate > 2% for 10 min; p95 latency > 3 s; `veritas_jobs_total{status="failed"}` rising; queue depth.
- Errors: set `SENTRY_DSN` (backend). Logs: `docker compose logs -f backend worker`.
- Queue depth: `docker compose exec redis redis-cli llen rq:queue:veritas`.

## Common incidents
| Symptom | Check | Fix |
|---|---|---|
| Uploads accepted but jobs stay `queued` | `docker compose ps worker`, `logs worker` | restart worker; confirm `VERITAS_REDIS_URL` and shared `uploads` volume / S3 settings match the API |
| Everyone gets 429 | rate limiter / quotas | `VERITAS_QUOTA_*`, limiter windows in `app/core/rate_limit.py`; Redis down falls back to per-process limits (see logs) |
| Logins fail with 403 "Verify your email" | SMTP | set `VERITAS_SMTP_*`; without SMTP mails are only in the API log; admin can mark verified in SQL (`UPDATE users SET email_verified = 1 WHERE username = ...`) |
| Browser shows CSRF 403s | cookie `veritas_csrf` missing | `VERITAS_COOKIE_SECURE=true` over plain HTTP drops the cookies; use HTTPS or set false for local only |
| WebSocket closes immediately | `Origin` header | origin must be listed in `CORS_ALLOW_ORIGINS` |
| Disk filling | `uploads` volume, `pgdata` | uploads are deleted after each job; results older than `VERITAS_RETENTION_DAYS` are purged at API start |
| Hostile model file suspected | scanner worker | scans run in a `python -I` subprocess with CPU/memory limits; run the worker container without network access for defence in depth |

## Known limits (single-box design)
- One API process (WebSocket registry and a few small caches are in-process); scale analysis with more `worker` replicas.
- Live events reach only the user's own sockets on the API process that ran the request; with background jobs the UI polls the job.
- Postgres, Redis and uploads share the box: a host loss costs everything since the last off-box backup.
