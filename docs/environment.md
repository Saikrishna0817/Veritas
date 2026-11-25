# Environment variable reference

Backend variables are read from the process environment; a `.env` in `backend/` or the repository root is also loaded
(real environment variables win). `docker-compose.yml` passes most of them through, see [`.env.example`](../.env.example).
Booleans are true only for the text `true` (any case).

## Backend: required

The server refuses to start unless these are set (`validate_security_settings`).

| Name | Default | What it does |
|---|---|---|
| `VERITAS_JWT_SECRET` | none | Signs session tokens. Random, at least 32 characters. |
| `VERITAS_ADMIN_USERNAME` | none | Username of the admin account created at startup. |
| `VERITAS_ADMIN_PASSWORD` | none | Its password. Not a known default, at least 8 characters. |

## Backend: optional

| Name | Default | What it does |
|---|---|---|
| `VERITAS_ENV` | `production` | Environment label sent to Sentry. |
| `VERITAS_DATABASE_URL` | empty | `postgresql://user:pass@host/db`. Empty uses the SQLite file. |
| `FORENSICS_SQLITE_PATH` | empty (built-in path) | SQLite file location when no database URL is set. |
| `VERITAS_JWT_EXPIRY_MINUTES` | `60` | Session token lifetime. |
| `CORS_ALLOW_ORIGINS` | `http://localhost:5173` | Comma-separated origins allowed to call the API. |
| `VERITAS_PUBLIC_URL` | `http://localhost:5173` | Base URL used in emailed links. |
| `VERITAS_COOKIE_SECURE` | `true` | Secure flag on cookies. `false` only for plain-HTTP localhost. |
| `VERITAS_SIGNUP_ENABLED` | `true` | `false` closes public sign-up (invite-only beta: the admin creates accounts); the sign-up page then explains how to get one. |
| `VERITAS_ACCESS_CONTACT` | (empty) | While sign-up is closed, the email address the sign-up page gives for account requests. |
| `VERITAS_REQUIRE_EMAIL_VERIFICATION` | `true` | New accounts must confirm their email before signing in. |
| `VERITAS_TURNSTILE_SECRET` | empty | Cloudflare Turnstile secret key; empty turns the human check off. |
| `VERITAS_SMTP_HOST` | empty | SMTP server. Empty: mails are only written to the log (development). |
| `VERITAS_SMTP_PORT` | `587` | SMTP port (STARTTLS). |
| `VERITAS_SMTP_USER` | empty | SMTP login, if the provider needs one. |
| `VERITAS_SMTP_PASSWORD` | empty | SMTP password or API key. |
| `VERITAS_MAIL_FROM` | `Veritas <no-reply@localhost>` | Sender of outgoing mail. |
| `VERITAS_REDIS_URL` | empty | Shared rate limits and job queue. Empty keeps both in-process (development, tests). |
| `VERITAS_STORAGE_URL` | empty (local directory) | `file:///path` or `s3://bucket` (S3, Cloudflare R2, MinIO). S3 enables direct uploads. |
| `VERITAS_UPLOAD_DIR` | `<tmp>/veritas-uploads` | Local upload directory when the storage URL is empty. |
| `VERITAS_S3_ENDPOINT` | empty | Custom S3 endpoint (R2: `https://<account>.r2.cloudflarestorage.com`); empty for AWS. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | none | Bucket credentials (read by boto3, S3 only). |
| `AWS_DEFAULT_REGION` | none | Set to `auto` for R2 (read by boto3). |
| `VERITAS_RETENTION_DAYS` | `30` | Days results and uploads are kept before cleanup. |
| `VERITAS_QUOTA_ANALYSES_PER_DAY` | `50` | Analyses per user per UTC day. `0` disables. Admins are exempt. |
| `VERITAS_QUOTA_ROWS_PER_DAY` | `2000000` | Table rows analysed per user per UTC day. `0` disables. |
| `VERITAS_QUOTA_SCAN_MB_PER_DAY` | `1000` | Model-file megabytes scanned per user per UTC day. `0` disables. |
| `MAX_ANALYSIS_WORKERS` | `4` | Size of the in-process analysis thread pool. |
| `VERITAS_WORKER_QUEUES` | the default queue | Comma-separated queues a worker process listens to (the media worker uses its own). |
| `VERITAS_METRICS_TOKEN` | empty | Bearer token required for `/metrics`; empty hides the endpoint. |
| `SENTRY_DSN` | empty | Error reporting; empty turns it off. |
| `SENTRY_TRACES_SAMPLE_RATE` | `0` | Fraction of requests traced by Sentry. |
| `LOG_LEVEL` | `INFO` | Log level. |
| `LOG_FORMAT` | `text` | `json` for structured logs. |

## Docker Compose only

| Name | Default | What it does |
|---|---|---|
| `POSTGRES_PASSWORD` | none (required) | Database password; compose stops if empty. |
| `WEB_PORT` | `5173` | Host port nginx is published on. |
| `VERITAS_DOMAIN` | (none; required with `docker-compose.prod.yml`) | Public host name. Caddy gets its certificate for it; sets `VERITAS_PUBLIC_URL` and `CORS_ALLOW_ORIGINS` to `https://<domain>` and secure cookies. |
| `ACME_EMAIL` | (none; required with `docker-compose.prod.yml`) | Address for certificate expiry notices. |
| `MEDIA_WORKER_MEMORY` | `6g` | Memory cap of the media worker. |
| `MEDIA_WORKER_CPUS` | `2` | CPU cap of the media worker. |

## Frontend (build time, Vite)

| Name | Default | What it does |
|---|---|---|
| `VITE_API_BASE_URL` | `/api/v1` | API base URL baked into the build. |
| `VITE_TURNSTILE_SITE_KEY` | empty | Turnstile public site key; empty hides the widget. |
| `VITE_API_PROXY` | `http://localhost:8001` | Dev server only: where `/api` and the WebSocket are proxied. |
| `VITE_LANDING_PROXY` | `http://localhost:5180` | Dev server only: where the landing page is proxied. |
