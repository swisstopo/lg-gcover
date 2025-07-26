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





### Quality Assurance (QA) Commands

Process ESRI FileGDB verification results, convert to web formats, and generate statistics for monitoring data quality issues.

#### Overview

The QA commands handle FileGDB files containing topology and technical quality verification results (~30k features per file). They:

- **Convert** spatial layers to web formats (GeoParquet/GeoJSON) 
- **Upload** converted files to S3 with organized structure
- **Generate** statistics and summaries for dashboard display
- **Handle** complex geometries from topology validation

#### Quick Start

```bash
# Process a single verification FileGDB
gcover qa process /path/to/issue.gdb

# Batch process weekly verification results  
gcover qa batch /media/marco/SANDISK/Verifications

# View recent statistics
gcover qa stats --days-back 7

# Generate HTML dashboard
gcover qa dashboard
```

#### Commands

##### `gcover qa process`

Process a single verification FileGDB to web formats.

**Basic Usage:**
```bash
gcover qa process /path/to/issue.gdb
```

**Advanced Options:**
```bash
# With geometry simplification for complex polygons
gcover qa process /path/to/issue.gdb --simplify-tolerance 1.0

# Output both GeoParquet and GeoJSON formats
gcover qa process /path/to/issue.gdb --format both

# Local processing only (no S3 upload)
gcover qa process /path/to/issue.gdb --no-upload

# Verbose logging for debugging
gcover qa process /path/to/issue.gdb --verbose
```

##### `gcover qa batch`

Process multiple FileGDBs in a directory.

```bash
# Process all issue.gdb files
gcover qa batch /media/marco/SANDISK/Verifications

# Apply geometry simplification to all files
gcover qa batch /path/to/verifications --simplify-tolerance 1.0

# Dry run to preview what would be processed
gcover qa batch /path/to/verifications --dry-run
```

##### `gcover qa stats`

Display verification statistics from the database.

```bash
# Recent statistics (last 30 days)
gcover qa stats

# Filter by verification type and timeframe
gcover qa stats --verification-type Topology --days-back 7

# Filter by RC version
gcover qa stats --rc-version 2030-12-31

# Export results to CSV
gcover qa stats --export-csv verification_report.csv
```

##### `gcover qa dashboard`

Generate an HTML dashboard with charts and statistics.

```bash
# Create dashboard with last 90 days of data
gcover qa dashboard
```

Opens `verification_dashboard.html` with interactive charts showing:
- Issue counts by type (Error/Warning)
- Top failing tests over time
- Verification trends and statistics

##### `gcover qa diagnose`

Investigate FileGDB structure and identify potential issues.

```bash
# Analyze all layers
gcover qa diagnose /path/to/issue.gdb

# Focus on specific problematic layer
gcover qa diagnose /path/to/issue.gdb --layer IssuePolygons
```

##### `gcover qa test-read`

Test different reading strategies for problematic FileGDBs.

```bash
# Test reading capabilities
gcover qa test-read /path/to/issue.gdb --layer IssuePolygons

# Limit test to fewer features
gcover qa test-read /path/to/issue.gdb --max-features 5
```

#### Configuration

Uses your existing `config/gdb_config.yaml`:

```yaml
# Standard GDB configuration
s3:
  bucket: "swisstopo-gcover-prod"
  profile: "gcover"
database:
  path: "/home/user/.config/gcover/assets.duckdb"
temp_dir: "/tmp/gcover"

# QA-specific settings (optional)
verification:
  layers: ["IssuePolygons", "IssueLines", "IssuePoints", "IssueStatistics"]
  s3_prefix: "verifications/"
  coordinate_systems:
    source_crs: "EPSG:2056"  # Swiss LV95
    target_crs: "EPSG:4326"  # WGS84 for web
```

**Environment Variables:**
```bash
export GDB_S3_BUCKET=swisstopo-gcover-prod
export GDB_S3_PROFILE=gcover
export GDB_DB_PATH=/home/user/.config/gcover/assets.duckdb
export GDB_TEMP_DIR=/tmp/gcover
```

#### File Structure

**Input:** ESRI FileGDB with verification results
```
issue.gdb/
├── IssuePolygons    # 30,743 features (complex polygons)
├── IssueLines       # 12,796 features  
├── IssuePoints      # 2,273 features
└── IssueStatistics  # 45 features (summary data)
```

**Output S3 Structure:**
```
s3://bucket/verifications/
├── Topology/2030-12-31/20250718_070012/
│   ├── IssuePolygons.parquet
│   ├── IssueLines.parquet
│   ├── IssuePoints.parquet
│   └── IssueStatistics.parquet
└── TechnicalQualityAssurance/2016-12-31/...
```

**Statistics Database:**
- `verification_stats.duckdb` (created alongside main assets DB)
- Tables: `gdb_summaries`, `layer_stats`, `test_stats`

#### Troubleshooting

##### Complex Geometry Issues

If you see warnings about complex polygons or coordinate sequences:

```bash
# Apply geometry simplification (1-meter tolerance)
gcover qa process /path/to/issue.gdb --simplify-tolerance 1.0

# For very complex geometries, use higher tolerance
gcover qa process /path/to/issue.gdb --simplify-tolerance 5.0
```

##### Reading Failures

If layers can't be read:

```bash
# Diagnose the FileGDB first
gcover qa diagnose /path/to/issue.gdb

# Test reading strategies
gcover qa test-read /path/to/issue.gdb --layer IssuePolygons

# Enable verbose logging
gcover qa process /path/to/issue.gdb --verbose
```

##### Empty Results

If all layers are skipped:
- Check that the FileGDB contains data (use `diagnose`)
- Verify file permissions and path
- Ensure the FileGDB isn't corrupted

#### Weekly Processing Workflow

Example automation script:

```bash
#!/bin/bash
# Process new weekly verification results

VERIFICATION_DIR="/media/marco/SANDISK/Verifications"

# Process new files from last week
gcover qa batch "$VERIFICATION_DIR" \
    --pattern "**/$(date -d '7 days ago' +%Y%m%d)_*/issue.gdb"

# Generate updated dashboard
gcover qa dashboard

# Export weekly report
gcover qa stats --days-back 7 \
    --export-csv "reports/weekly_$(date +%Y%m%d).csv"
```



### SDE Connection Management

The `gcover sde` command provides comprehensive tools for managing SDE (Spatial Database Engine) connections, versions, and user access.

#### Quick Start

```bash
# Find your user versions across all instances
gcover sde user-versions

# List all versions for a specific instance
gcover sde versions -i GCOVERP

# Interactive connection test
gcover sde connect -i GCOVERP --interactive

# Export version information to JSON
gcover sde versions -f json > versions.json

# Clean up active connections
gcover sde connections --cleanup
```

#### SDE Commands

##### Version Management

###### `gcover sde versions`
Lists available versions on SDE instances with filtering and export options.

```bash
# List versions for specific instance(s)
gcover sde versions -i GCOVERP
gcover sde versions -i GCOVERP -i GCOVERQ

# Show only user versions (contains your username)
gcover sde versions --user-only

# Export in different formats
gcover sde versions --format json        # JSON export
gcover sde versions --format csv         # CSV export  
gcover sde versions --format table       # Table display (default)

# Combine options
gcover sde versions -i GCOVERP --user-only --format json
```

**Options:**
- `--instance, -i`: Specify SDE instances to check (multiple allowed)
- `--user-only, -u`: Show only versions containing your username
- `--format, -f`: Output format (`table`, `json`, `csv`)

###### `gcover sde user-versions`
Automatically finds versions where you are the owner or that contain your username.

```bash
# Find user versions across all instances
gcover sde user-versions

# Check specific instances only
gcover sde user-versions -i GCOVERP -i GCOVERQ
```

##### Connection Management

###### `gcover sde connections`
Lists and manages active SDE connections.

```bash
# List active connections
gcover sde connections

# List and clean up all connections
gcover sde connections --cleanup
```

**Options:**
- `--cleanup`: Clean up all active connections after displaying them

###### `gcover sde connect`
Test connections to SDE instances with interactive version selection.

```bash
# Quick connection test (uses SDE.DEFAULT)
gcover sde connect -i GCOVERP

# Interactive version selection
gcover sde connect -i GCOVERP --interactive
```

**Options:**
- `--instance, -i`: SDE instance to connect to (required)
- `--interactive`: Enable interactive version selection menu

#### Output Examples

##### Version Listing
```
🔍 Vérification instance: GCOVERP

📊 GCOVERP
============================================================
Version                     Parent      Status
-----------------------------------------------------------------------------------
SDE.DEFAULT                 -           -
GCOVERP.RC_2016-12-31      SDE.DEFAULT  ✏️ Writable
USER.MYVERSION_20250726    SDE.DEFAULT  👤 Owner ✏️ Writable ⭐ User
```

##### User Versions
```
👤 Recherche versions pour utilisateur: MYUSER

📁 GCOVERP:
  • USER.MYVERSION_20250726 (✏️ Writable)
  • USER.BACKUP_20250720 (👁️ Read-only)
```

##### Active Connections
```
🔗 2 connexion(s) active(s):
Instance    Version                Path
--------------------------------------------------------------------------
GCOVERP     SDE.DEFAULT           /tmp/gcover_GCOVERP_DEFAULT.sde
GCOVERP     USER.MYVERSION        /tmp/gcover_GCOVERP_MYVERSION.sde
```

# Schema Management

The `gcover schema` command provides tools for extracting, comparing, and documenting database schemas from GDB files and SDE connections.

## Quick Start

```bash
# Extract schema from GDB to JSON
gcover schema extract /path/to/data.gdb -o ./schemas -n myschema

# Generate PlantUML diagram
gcover schema diagram schema.json -o schema.puml

# Compare two schemas
gcover schema diff old_schema.json new_schema.json -o changes.json

# Generate documentation
gcover schema report schema.json -o documentation.html
```

#### Schema Commands

##### `gcover schema extract`
Extracts schema information from GDB files or SDE connections.

```bash
# Basic extraction
gcover schema extract /path/to/geodatabase.gdb

# Specify output directory and name
gcover schema extract /path/to/data.gdb -o ./schemas -n production_schema

# Multiple output formats
gcover schema extract /path/to/data.gdb -f json -f html -f xml

# Filter by table prefix
gcover schema extract /path/to/data.gdb --filter-prefix GCOVER

# Remove prefix from table names in output
gcover schema extract /path/to/data.gdb --filter-prefix GCOVER --remove-prefix
```

**Arguments:**
- `SOURCE`: Path to GDB file or SDE connection file

**Options:**
- `--output, -o`: Output directory for schema files
- `--name, -n`: Name for the generated schema files
- `--format, -f`: Output formats (`json`, `html`, `xml`) - multiple allowed
- `--filter-prefix`: Include only tables with this prefix
- `--remove-prefix/--keep-prefix`: Remove prefix from table names (default: keep)

##### `gcover schema diagram`
Generates PlantUML diagrams from schema JSON files.

```bash
# Basic diagram generation
gcover schema diagram schema.json -o database_diagram.puml

# Customize diagram
gcover schema diagram schema.json -o diagram.puml --title "GeoCover Schema"

# Simplified diagram (no field details)
gcover schema diagram schema.json -o simple.puml --no-fields

# Exclude relationships
gcover schema diagram schema.json -o tables_only.puml --no-relationships

# Include only specific tables
gcover schema diagram schema.json -o filtered.puml -f GCOVER_POINTS -f GCOVER_LINES
```

**Arguments:**
- `JSON_FILE`: Path to schema JSON file

**Options:**
- `--output, -o`: Output PlantUML file (required)
- `--title`: Diagram title (default: "Database Schema")
- `--no-fields`: Exclude field details from diagram
- `--no-relationships`: Exclude relationships from diagram
- `--filter, -f`: Include only specified tables (multiple allowed)

##### `gcover schema diff`
Compares two schemas and generates difference reports.

```bash
# Basic schema comparison
gcover schema diff old_schema.json new_schema.json

# Save diff report
gcover schema diff v1.json v2.json -o changes.json

# Different output formats
gcover schema diff v1.json v2.json -o changes.html --format html
gcover schema diff v1.json v2.json -o changes.md --format markdown
```

**Arguments:**
- `OLD_SCHEMA`: Path to older schema JSON file
- `NEW_SCHEMA`: Path to newer schema JSON file

**Options:**
- `--output, -o`: Output file for diff report
- `--format`: Output format (`json`, `html`, `markdown`)

##### `gcover schema report`
Generates documentation from schema files.

```bash
# Generate HTML documentation
gcover schema report schema.json -o documentation.html

# Different report templates
gcover schema report schema.json -t datamodel -o datamodel.html
gcover schema report schema.json -t summary -o summary.html
gcover schema report schema.json -t full -o complete_doc.html

# Different output formats
gcover schema report schema.json -o doc.md --format markdown
gcover schema report schema.json -o doc.pdf --format pdf
```

**Arguments:**
- `SCHEMA_FILE`: Path to schema JSON file

**Options:**
- `--template, -t`: Report template (`datamodel`, `summary`, `full`)
- `--output, -o`: Output file (required)
- `--format`: Output format (`html`, `markdown`, `pdf`)

#### Typical Workflows

##### Schema Documentation Workflow
```bash
# 1. Extract schema from production GDB
gcover schema extract /path/to/production.gdb -o ./schemas -n prod_v2.1

# 2. Generate visual diagram
gcover schema diagram schemas/prod_v2.1.json -o docs/schema_diagram.puml --title "Production Schema v2.1"

# 3. Generate comprehensive documentation
gcover schema report schemas/prod_v2.1.json -t full -o docs/schema_doc.html
```

##### Schema Change Tracking
```bash
# 1. Extract current schema
gcover schema extract /path/to/current.gdb -n current

# 2. Compare with previous version  
gcover schema diff schemas/previous.json schemas/current.json -o changes.json

# 3. Generate change report
gcover schema report changes.json -t summary -o change_summary.html
```

##### Multi-Instance Schema Comparison
```bash
# Extract from different SDE instances
gcover schema extract gcoverp_connection.sde -n gcoverp_prod
gcover schema extract gcoverq_connection.sde -n gcoverq_prod

# Compare schemas between instances
gcover schema diff gcoverp_prod.json gcoverq_prod.json -o instance_diff.json
```

#### Requirements

The `gcover schema` commands require **arcpy** for schema extraction from GDB files and SDE connections. Ensure you have ArcGIS Pro installed and properly configured.

```bash
# Check if arcpy is available
python -c "import arcpy; print('arcpy available')"
```

If arcpy is not available, you'll see:
```
❌ This command requires arcpy
```

#### Output Files

##### Schema JSON Structure
```json
{
  "metadata": {
    "source": "/path/to/data.gdb",
    "extracted_at": "2025-07-26T10:30:00",
    "total_tables": 15
  },
  "tables": [
    {
      "name": "GCOVER_POINTS",
      "type": "FeatureClass", 
      "geometry_type": "Point",
      "fields": [...]
    }
  ]
}
```
