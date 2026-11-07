#!/usr/bin/env bash
# Daily Postgres backup for the compose stack. Cron it:  0 3 * * *  cd /opt/veritas && scripts/backup.sh
# Keeps the newest $KEEP dumps locally; if BACKUP_S3_URL is set (s3://bucket/prefix, e.g. a Cloudflare R2 bucket) the dump
# is also uploaded with the aws CLI (set AWS_* and BACKUP_S3_ENDPOINT).
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE=(docker compose ${COMPOSE_PROJECT:+-p "$COMPOSE_PROJECT"} ${COMPOSE_ENV_FILE:+--env-file "$COMPOSE_ENV_FILE"})
DIR="${BACKUP_DIR:-backups}"; KEEP="${KEEP:-14}"
mkdir -p "$DIR"
FILE="$DIR/veritas-$(date -u +%Y%m%dT%H%M%SZ).dump"
"${COMPOSE[@]}" exec -T postgres pg_dump -U veritas -Fc veritas > "$FILE"
[ -s "$FILE" ] || { echo "backup is empty" >&2; rm -f "$FILE"; exit 1; }
if [ -n "${BACKUP_S3_URL:-}" ]; then
  aws s3 cp "$FILE" "${BACKUP_S3_URL%/}/$(basename "$FILE")" ${BACKUP_S3_ENDPOINT:+--endpoint-url "$BACKUP_S3_ENDPOINT"}
fi
ls -1t "$DIR"/veritas-*.dump | tail -n +"$((KEEP + 1))" | xargs -r rm -f
echo "$FILE"
