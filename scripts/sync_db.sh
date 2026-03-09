#!/usr/bin/env bash
# sync_db.sh — Download the live scan_history.db from Render
# Usage: ./scripts/sync_db.sh
# Or:    make sync
#
# Set EXPORT_TOKEN env var to match the EXPORT_TOKEN set in Render env vars.
# Example: EXPORT_TOKEN=mysecret ./scripts/sync_db.sh
# Or add to your shell profile: export EXPORT_TOKEN=mysecret

set -euo pipefail

RENDER_URL="${RENDER_URL:-https://ai-trading-mvp.onrender.com}"
TOKEN="${EXPORT_TOKEN:-}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="${LOCAL_DIR}/db_backups"
DEST="${LOCAL_DIR}/scan_history.db"

if [[ -z "$TOKEN" ]]; then
    echo "✗ EXPORT_TOKEN is not set."
    echo "  Add it to your shell profile: export EXPORT_TOKEN=<your_token>"
    echo "  Then set the same value as EXPORT_TOKEN in Render Environment Variables."
    exit 1
fi

EXPORT_ENDPOINT="${RENDER_URL}/export-db?token=${TOKEN}"

mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/scan_history_${TIMESTAMP}.db"

echo "→ Downloading scan_history.db from Render..."

MAX_TRIES=3
ATTEMPT=0
HTTP_STATUS="000"

while [[ $ATTEMPT -lt $MAX_TRIES ]]; do
    HTTP_STATUS=$(curl -s -o "$BACKUP_FILE" -w "%{http_code}" --max-time 90 "$EXPORT_ENDPOINT")
    if [[ "$HTTP_STATUS" == "200" ]]; then
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if [[ $ATTEMPT -lt $MAX_TRIES ]]; then
        echo "  HTTP $HTTP_STATUS — retrying in 20s (attempt $ATTEMPT/$MAX_TRIES)..."
        sleep 20
    fi
done

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "✗ Failed after $MAX_TRIES attempts: HTTP $HTTP_STATUS"
    rm -f "$BACKUP_FILE"
    exit 1
fi

FILE_SIZE=$(du -sh "$BACKUP_FILE" | cut -f1)
ROW_COUNT=$(sqlite3 "$BACKUP_FILE" "SELECT COUNT(*) FROM scans;" 2>/dev/null || echo "?")
LABELED=$(sqlite3 "$BACKUP_FILE" "SELECT COUNT(*) FROM scans WHERE next_day_return IS NOT NULL;" 2>/dev/null || echo "?")
LAST_SCAN=$(sqlite3 "$BACKUP_FILE" "SELECT MAX(timestamp) FROM scans;" 2>/dev/null || echo "?")

# Replace local DB
cp "$BACKUP_FILE" "$DEST"

echo ""
echo "✓ Downloaded: $FILE_SIZE"
echo "  Scans total  : $ROW_COUNT"
echo "  Labeled rows : $LABELED (next_day_return known)"
echo "  Last scan    : $LAST_SCAN"
echo "  Saved to     : $DEST"
echo "  Backup at    : $BACKUP_FILE"
echo ""
echo "Ready for analysis."
