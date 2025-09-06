#!/bin/bash
# Script to backup DuckDB metadata database

set -e

# Configuration
DB_PATH="data/gdb_metadata.duckdb"
BACKUP_DIR="backups/metadata"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="gdb_metadata_${DATE}.duckdb"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create backup
echo "Creating backup: $BACKUP_NAME"
cp "$DB_PATH" "$BACKUP_DIR/$BACKUP_NAME"

# Compress backup
gzip "$BACKUP_DIR/$BACKUP_NAME"

echo "Backup created: $BACKUP_DIR/${BACKUP_NAME}.gz"

# Clean old backups (keep last 30)
echo "Cleaning old backups..."
cd "$BACKUP_DIR"
ls -t *.gz | tail -n +31 | xargs -r rm

echo "Backup completed successfully"