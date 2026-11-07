#!/usr/bin/env bash
# Prove a backup restores: load the newest dump (or $1) into a scratch database and compare row counts of the key tables
# with the live database. Exit 1 on any mismatch. Run it after every backup change and on a schedule (monthly).
set -euo pipefail
cd "$(dirname "$0")/.."
COMPOSE=(docker compose ${COMPOSE_PROJECT:+-p "$COMPOSE_PROJECT"} ${COMPOSE_ENV_FILE:+--env-file "$COMPOSE_ENV_FILE"})
DUMP="${1:-$(ls -1t "${BACKUP_DIR:-backups}"/veritas-*.dump | head -1)}"
SCRATCH="restore_drill_$$"
psql_() { "${COMPOSE[@]}" exec -T postgres psql -U veritas -d "$1" -tAc "$2"; }
trap '"${COMPOSE[@]}" exec -T postgres psql -U veritas -d postgres -c "DROP DATABASE IF EXISTS $SCRATCH" >/dev/null' EXIT
psql_ postgres "CREATE DATABASE $SCRATCH" >/dev/null
"${COMPOSE[@]}" exec -T postgres pg_restore -U veritas -d "$SCRATCH" --no-owner < "$DUMP"
status=0
for t in users analysis_results model_scans jobs api_keys usage_daily schema_migrations; do
  live=$(psql_ veritas "SELECT count(*) FROM $t"); back=$(psql_ "$SCRATCH" "SELECT count(*) FROM $t")
  # the live DB may have grown since the dump; the restored copy must never have MORE rows than live or be empty when live is not
  if [ "$back" -gt "$live" ] || { [ "$live" -gt 0 ] && [ "$back" -eq 0 ]; }; then echo "MISMATCH $t: live=$live restored=$back"; status=1; else echo "ok $t: live=$live restored=$back"; fi
done
[ "$status" -eq 0 ] && echo "restore drill passed: $DUMP" || echo "restore drill FAILED: $DUMP"
exit $status
