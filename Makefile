.PHONY: sync analyze

# Download latest scan_history.db from Render
sync:
	@./scripts/sync_db.sh

# Sync then open a quick summary in Python
analyze: sync
	@python3 -c "\
import sqlite3, json, collections; \
c = sqlite3.connect('scan_history.db').cursor(); \
total = c.execute('SELECT COUNT(*) FROM scans').fetchone()[0]; \
labeled = c.execute('SELECT COUNT(*) FROM scans WHERE next_day_return IS NOT NULL').fetchone()[0]; \
hit20 = c.execute('SELECT COUNT(*) FROM scans WHERE next_day_return >= 20').fetchone()[0]; \
trade_calls = c.execute(\"SELECT COUNT(*) FROM scans WHERE ai_json LIKE '%TRADE%'\").fetchone()[0]; \
print(f'Scans: {total} | Labeled: {labeled} | Hit 20%+: {hit20} ({hit20*100//max(labeled,1)}%) | AI TRADE calls: {trade_calls}') \
"
