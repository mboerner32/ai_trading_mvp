#!/usr/bin/env bash
# sync_db_auto.sh — Automated nightly sync of Render DB to local
# Runs via crontab; logs to db_backups/sync.log

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_DIR/db_backups/sync.log"

mkdir -p "$PROJECT_DIR/db_backups"

echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$LOG_FILE"

export EXPORT_TOKEN="ff93480wjoaiergh03443"
export RENDER_URL="https://ai-trading-mvp.onrender.com"

cd "$PROJECT_DIR" && bash "$SCRIPT_DIR/sync_db.sh" >> "$LOG_FILE" 2>&1
echo "" >> "$LOG_FILE"
