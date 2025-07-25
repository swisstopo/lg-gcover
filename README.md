# lg-gcover

**A Python library and CLI tool for working with Swiss GeoCover geological vector data**

lg-gcover simplifies the processing and analysis of geological vector datasets from the Swiss national Geological Survey (swisstopo). Built on modern geospatial Python tools like GeoPandas and Shapely, it provides both programmatic APIs and command-line utilities for geological data workflows.

## Key Features
- **CLI Interface**: Easy-to-use `gcover` command for batch processing
- **GeoPandas Integration**: Seamless integration with the Python geospatial ecosystem
- **ESRI Compatibility**: Optional support for ArcGIS Pro workflows via arcpy
- **Rich Output**: Beautiful terminal output with progress indicators and structured logging
- **Flexible Data Handling**: Support for various geological vector formats and projections

Perfect for geologists, GIS analysts, and researchers working with Swiss geological datasets who need efficient, reproducible data processing workflows.


## Usage

### GDB Asset Management - Usage

#### Quick Start

```bash
# Initialize the system
gcover gdb --env dev init

# Scan for GDB files
gcover gdb --env dev scan

# Sync to S3 (dry run first)
gcover gdb --env dev sync --dry-run
gcover gdb --env dev sync

# Check status
gcover gdb --env dev status
```

#### Core Commands

##### System Management
```bash
gcover gdb init                    # Initialize database and check connections
gcover gdb scan                    # Scan filesystem for GDB assets
gcover gdb sync                    # Upload new assets to S3 and update database
gcover gdb sync --dry-run          # Preview what would be synced
gcover gdb status                  # Show system statistics and health
```

##### Asset Discovery
```bash
gcover gdb list                    # List recent assets (default: 20)
gcover gdb list --limit 50         # List more assets
gcover gdb list --type backup_daily # Filter by asset type
gcover gdb list --rc RC1           # Filter by release candidate
gcover gdb list --since 2025-07-01 # Filter by date

gcover gdb search "2025"           # Search assets by term
gcover gdb search "topology" --download # Search and download
```

##### Processing
```bash
gcover gdb process /path/to/specific.gdb    # Process single asset
```

#### Environment Management

```bash
# Development environment (default)
gcover gdb --env dev scan
gcover gdb --env development scan

# Production environment  
gcover gdb --env prod sync
gcover gdb --env production sync

# With custom config
gcover gdb --config ./my-config.yaml --env prod sync

# With environment variables
export GDB_ENV=production
export GDB_S3_BUCKET=my-prod-bucket
gcover gdb sync
```

#### Asset Types

The system manages three types of GDB assets:

| Type | Description | Example |
|------|-------------|---------|
| **backup_daily** | Daily backups | `20250718_0310_2016-12-31.gdb` |
| **backup_weekly** | Weekly backups | `20250715_0330_2030-12-31.gdb` |
| **backup_monthly** | Monthly backups | `20250630_2200_2016-12-31.gdb` |
| **verification_tqa** | Technical Quality Assurance | `issue.gdb` in TQA folders |
| **verification_topology** | Topology verification | `issue.gdb` in Topology folders |
| **increment** | Data increments | `20250721_GCOVERP_2016-12-31.gdb` |

#### Database Queries

##### Direct DuckDB queries
```bash
# Connect to database
duckdb data/dev_gdb_metadata.duckdb

# Basic queries
SELECT COUNT(*) FROM gdb_assets;
SELECT DISTINCT asset_type FROM gdb_assets;
SELECT * FROM gdb_assets ORDER BY timestamp DESC LIMIT 10;

# Export to CSV
COPY (SELECT * FROM gdb_assets) TO 'assets_export.csv' (HEADER, DELIMITER ',');
```

##### Python queries

```python
import duckdb

with duckdb.connect("data/dev_gdb_metadata.duckdb") as conn:
    # Assets by type
    result = conn.execute("""
        SELECT asset_type, COUNT(*), SUM(file_size) / 1024 / 1024 / 1024 as gb
        FROM gdb_assets 
        GROUP BY asset_type
    """).fetchall()
    
    # Recent assets
    recent = conn.execute("""
        SELECT path, timestamp FROM gdb_assets 
        WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY timestamp DESC
    """).fetchall()
```

#### Configuration

##### Environment Variables
```bash
GDB_ENV=development|production      # Environment selection
GDB_S3_BUCKET=my-bucket            # Override S3 bucket
GDB_S3_PROFILE=my-profile          # AWS profile
GDB_DB_PATH=/path/to/db.duckdb     # Database path
GDB_LOG_LEVEL=DEBUG|INFO|WARNING   # Logging level
```

###### Config File (`config/gdb_config.yaml`)
```yaml
base_paths:
  backup: "/path/to/GCOVER"
  verification: "/path/to/Verifications"
  increment: "/path/to/Increment"

s3:
  bucket: "your-gdb-bucket"
  
database:
  path: "data/gdb_metadata.duckdb"
```

#### Maintenance

```bash
# Backup database
cp data/dev_gdb_metadata.duckdb data/backup_$(date +%Y%m%d).duckdb

# Clean temporary files
rm -rf /tmp/gdb_zips/*

# Database maintenance
duckdb data/dev_gdb_metadata.duckdb 'VACUUM;'
```

#### Troubleshooting

```bash
# Verbose output for debugging
gcover gdb --env dev --verbose scan

# Check configuration
gcover gdb --env dev --verbose status

# Verify AWS credentials
aws s3 ls s3://your-bucket/

# Check database
duckdb data/dev_gdb_metadata.duckdb 'SELECT COUNT(*) FROM gdb_assets;'
```

#### Common Workflows

##### Daily Sync
```bash
#!/bin/bash
# Daily sync script
export GDB_ENV=production
gcover gdb sync 2>&1 | logger -t gdb-sync
```

##### Weekly Report
```bash
# Generate weekly report
gcover gdb list --since $(date -d '7 days ago' +%Y-%m-%d) > weekly_report.txt
```

##### Find Missing Backups
```sql
-- SQL query to find gaps in daily backups
WITH daily_backups AS (
    SELECT DISTINCT DATE(timestamp) as backup_date
    FROM gdb_assets 
    WHERE asset_type = 'backup_daily'
      AND timestamp >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT generate_series(
    CURRENT_DATE - INTERVAL '30 days',
    CURRENT_DATE,
    INTERVAL '1 day'
) as missing_date
WHERE missing_date NOT IN (SELECT backup_date FROM daily_backups);
```

#### Data Flow

```
📁 Local GDB Files → 📦 ZIP Creation → ☁️ S3 Upload → 💾 Local Database Update
```

- **GDB assets** are compressed and uploaded to S3
- **Metadata** is stored in local DuckDB for fast querying  
- **Database** serves as a catalog of your S3 assets
- **No cloud database costs** - everything runs locally



### SDE connection

    # Voir vos versions utilisateur
    gcover sde user-versions

    # Lister toutes les versions de GCOVERP
    gcover sde versions -i GCOVERP

    # Test de connexion interactif
    gcover sde connect -i GCOVERP --interactive

    # Export JSON des versions
    gcover sde versions -f json > versions.json

    # Nettoyer les connexions
    gcover sde connections --cleanup