# Runbook

Stack: `docker-compose.yml` = Postgres, Redis, API (`backend`), CSV worker (`worker`, queue `veritas`), isolated media
worker (`worker-media`, queue `veritas-untrusted`: images, audio, video and model files), nginx + SPA (`frontend`).

`worker-media` is the only container that parses untrusted binary files (ffmpeg, image decoders, pickle scanning). It is
on the `internal` network only (Postgres, Redis, the uploads volume; no internet), with a read-only root filesystem, a
2 GB `/tmp`, no capabilities, `no-new-privileges`, and memory / CPU / process caps (`MEDIA_WORKER_MEMORY`, default 6g;
`MEDIA_WORKER_CPUS`, default 2). ffmpeg and the scanner add per-process limits inside it. With S3/R2 storage it must
fetch uploads: add `egress` to its networks. The backend image is about 3.5 GB (CPU torch, torchaudio, ffmpeg and the
ResNet-18 / HuBERT weights); `--build-arg WITH_IMAGES=0` builds a CSV-only image.
Put TLS/CDN (Cloudflare, Caddy) in front; the API and Postgres/Redis ports are not published.

## First deploy
1. VM with Docker (Oracle Always Free ARM, any VPS). `git clone`, `cp .env.example .env`, fill the required values
   (`POSTGRES_PASSWORD`, `VERITAS_JWT_SECRET`, `VERITAS_ADMIN_*`) and for a public host `VERITAS_PUBLIC_URL`,
   `CORS_ALLOW_ORIGINS`, `VERITAS_COOKIE_SECURE=true`, SMTP settings, `VERITAS_TURNSTILE_SECRET`.
2. `docker compose up -d --build`. The API container applies database migrations before it starts. On an ARM VM
   (Oracle Ampere A1) this builds arm64 images natively; every dependency, CPU torch included, has arm64 wheels. Tagged
   releases also publish amd64 + arm64 images to `ghcr.io/saikrishna0817/veritas/{backend,frontend}:<tag>`.
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

## Direct uploads (S3 / Cloudflare R2)

With `VERITAS_STORAGE_URL=s3://bucket` the CLI sends files straight to the bucket (`POST /uploads/presign`, PUT, `POST /uploads/complete`)
so the API never carries the bytes. With local storage `/uploads/presign` answers 501 and clients fall back to `/analyze/upload`.

Bucket setup, once:
- **R2:** set `VERITAS_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com`, `AWS_DEFAULT_REGION=auto`, and an R2 API token as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. Presigned URLs use SigV4 with the size signed in.
- **Lifecycle rule:** expire objects under `upload/` and `model/` after 1 day (exact rule and check command below). The worker deletes uploads it processes; this removes ones that were presigned and never completed.
- **CORS (only when a browser uploads):** allow `PUT` from your site origin with header `Content-Length`. The CLI needs no CORS.
- The bucket must not be public. The API checks that an uploaded object exists, belongs to the caller, is used once and matches the declared size before queuing it; it never trusts the client's word.

Not verified against real S3 or R2 (tests use a mocked bucket): check that a PUT with a different size than declared is refused by the bucket.

### Lifecycle rule for abandoned uploads

Keys are `upload/<uuid>` and `model/<uuid>` (`app/core/storage.py: new_key`). Presigned objects that are never completed
stay in the bucket unless a rule expires them. Save as `lifecycle.json` (R2 expiry is whole days, so 1 is the minimum):

```json
{"Rules": [
  {"ID": "expire-abandoned-uploads", "Status": "Enabled", "Filter": {"Prefix": "upload/"}, "Expiration": {"Days": 1}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}},
  {"ID": "expire-abandoned-models",  "Status": "Enabled", "Filter": {"Prefix": "model/"},  "Expiration": {"Days": 1}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}}
]}
```

Apply (omit `--endpoint-url` for AWS S3; for R2 use `https://<account>.r2.cloudflarestorage.com`):

```bash
aws s3api put-bucket-lifecycle-configuration --bucket BUCKET --endpoint-url ENDPOINT --lifecycle-configuration file://lifecycle.json
```

Check that it exists (should list both rules, status `Enabled`; `NoSuchLifecycleConfiguration` means it is missing):

```bash
aws s3api get-bucket-lifecycle-configuration --bucket BUCKET --endpoint-url ENDPOINT --query 'Rules[].[ID,Status,Filter.Prefix,Expiration.Days]' --output table
```

R2 ignores `AbortIncompleteMultipartUpload` (single PUTs only), which is harmless. Run the check after each bucket change.

## Backups and restore
- `scripts/backup.sh` writes `backups/veritas-<utc>.dump` (pg_dump custom format), keeps the newest 14, optionally uploads to S3/R2.
- `scripts/restore-drill.sh [dump]` restores into a scratch database and compares row counts. Run it monthly and after changing
  the backup. **A backup that has not been restored is not a backup.**
- Real restore: stop `backend` and `worker`, `docker compose exec -T postgres pg_restore -U veritas -d veritas --clean --if-exists < dump`, start them.

## Monitoring and alerts
- Uptime: point UptimeRobot / Better Stack at `https://<host>/health` (alert on non-200 or `"status":"degraded"`).
- Metrics: set `VERITAS_METRICS_TOKEN`; Prometheus / Grafana Agent scrapes `/metrics` with `Authorization: Bearer <token>`
  (hidden without a token). Series: `veritas_http_requests_total`, `veritas_http_request_seconds`, `veritas_jobs_total`,
  `veritas_auth_login_failures_total{outcome="wrong_password"|"unknown_account"|"rate_limited"}`.
  Suggested alerts: 5xx rate > 2% for 10 min; p95 latency > 3 s; `veritas_jobs_total{status="failed"}` rising; queue depth.
- Readiness: `/ready` answers 503 when the database (or Redis, when configured) does not answer; use it for the load
  balancer or orchestrator, and `/health` for liveness.
- Failed sign-ins: who and from where is in the audit log (`auth.login_failed`); the counter is for alerting. Prometheus rule:

  ```yaml
  - alert: VeritasSignInBurst
    expr: sum(increase(veritas_auth_login_failures_total[10m])) > 50
    for: 5m
    annotations: { summary: "Many refused sign-ins: check the audit log for auth.login_failed by IP" }
  ```

  Tune the threshold to normal traffic. A steady rise of `rate_limited` means the limits are holding an attack back;
  a rise of `unknown_account` alone usually means name guessing.
- Errors: set `SENTRY_DSN` (backend). Logs: `docker compose logs -f backend worker`.
- Queue depth: `docker compose exec redis redis-cli llen rq:queue:veritas`.

## Common incidents
| Symptom | Check | Fix |
|---|---|---|
| Uploads accepted but jobs stay `queued` | `docker compose ps worker worker-media`, `logs worker-media` | CSV jobs need `worker`; images, audio, video and model scans need `worker-media`. Restart it; confirm `VERITAS_REDIS_URL` and the shared `uploads` volume / S3 settings (and `egress` for S3) |
| Everyone gets 429 | rate limiter / quotas | `VERITAS_QUOTA_*`, limiter windows in `app/core/rate_limit.py`; Redis down falls back to per-process limits (see logs) |
| Logins fail with 403 "Verify your email" | SMTP | set `VERITAS_SMTP_*`; without SMTP mails are only in the API log; admin can mark verified in SQL (`UPDATE users SET email_verified = 1 WHERE username = ...`) |
| Browser shows CSRF 403s | cookie `veritas_csrf` missing | `VERITAS_COOKIE_SECURE=true` over plain HTTP drops the cookies; use HTTPS or set false for local only |
| WebSocket closes immediately | `Origin` header | origin must be listed in `CORS_ALLOW_ORIGINS` |
| Disk filling | `uploads` volume, `pgdata` | uploads are deleted after each job; results older than `VERITAS_RETENTION_DAYS` are purged at API start |
| Hostile model or media file suspected | `logs worker-media` | it ran in `worker-media` (no internet, read-only, capped); scans run in a `python -I` subprocess and ffmpeg under rlimits. Delete the job's upload and the user's result if needed |

## Known limits (single-box design)
- One API process (WebSocket registry and a few small caches are in-process); scale analysis with more `worker` replicas.
- Live events reach only the user's own sockets on the API process that ran the request; with background jobs the UI polls the job.
- Postgres, Redis and uploads share the box: a host loss costs everything since the last off-box backup.
