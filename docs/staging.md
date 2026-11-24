# Staging on one machine

The full production stack from `docker-compose.yml` (PostgreSQL, Redis, API, worker, isolated media worker, nginx),
under its own project name, port and volumes, with memory caps from `docker-compose.staging.yml` so it can run next to
other work. It does not touch a stack started with plain `docker compose up`.

```bash
# once: secrets in .env.staging (git-ignored); see .env.example for the variables
python3 - <<'EOF' > .env.staging
import secrets as s
print(f"POSTGRES_PASSWORD={s.token_urlsafe(24)}\nVERITAS_JWT_SECRET={s.token_urlsafe(48)}")
print(f"VERITAS_ADMIN_USERNAME=staging-admin\nVERITAS_ADMIN_PASSWORD={s.token_urlsafe(18)}")
print("WEB_PORT=5190\nVERITAS_PUBLIC_URL=http://localhost:5190\nCORS_ALLOW_ORIGINS=http://localhost:5190\nVERITAS_COOKIE_SECURE=false")
EOF

alias stg='docker compose -p veritas-staging --env-file .env.staging -f docker-compose.yml -f docker-compose.staging.yml'
stg up -d --build                       # or name services: stg up -d postgres redis backend worker frontend
stg exec backend python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8001/ready').read())"
open http://localhost:5190
```

Load test (k6 through nginx, one account per virtual user):

```bash
stg exec -T backend python - 50 < loadtest/seed_users.py > /tmp/keys.json
k6 run -e BASE=http://127.0.0.1:5190 -e KEYS=/tmp/keys.json -e VUS=50 -e DURATION=3m loadtest/k6.js
```

Stop with `stg down` (keeps the data) or `stg down -v` (deletes the staging database and uploads).

## Results, 2026-11-24 (backend plan P13)

One laptop, shared with a training job (load about 20 on 12 cores), stack capped as in `docker-compose.staging.yml`,
one CSV worker, the media worker not started. k6 through nginx, one account per virtual user, 3 minutes each.

| Users | Requests | Failed | Reads p95 | Enqueue p95 | Upload to result, median | Analyses / s |
|---|---|---|---|---|---|---|
| 20 | 3,241 | 0 | 718 ms | 1.26 s | 54 s | 0.26 |
| 50 | 9,455 | 0 | 484 ms | 290 ms | 101 s | 0.45 |

- Nothing failed. The one worker is the limit: waits grow with users, as in the single-process test. Add workers
  (`stg up -d --scale worker=N`) before adding anything else. Latencies are noisy here because of the other job.
- Index 0007 (`analysis_results(owner_id, created_at DESC)`): with 500,000 rows over 2,000 accounts, a person's newest
  50 results take 0.08 ms (index scan, 53 buffers), and the next page with a `before` cursor 0.12 ms.
- Found while measuring: a request that only read left its PostgreSQL connection "idle in transaction", which blocked a
  `DROP INDEX` (and would block a migration) until that connection was used again. Fixed in `app/models/engine.py`:
  a plain read that opened the transaction is committed at once (`tests/test_pg_idle_transactions.py`).
