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


### Global Command Options

All commands support these global options:

```bash
gcover [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]

Global Options:
  --config, -c PATH    Configuration file path
  --env, -e ENV        Environment (dev/development, prod/production)
  --verbose, -v        Enable verbose output
  --help               Show help message
```

### Examples

```bash
# Use development environment (default)
gcover gdb status

# Use production environment
gcover --env prod gdb sync

# Use custom config file
gcover --config /path/to/config.yaml --env prod gdb status

# Enable verbose logging
gcover --verbose --env dev gdb scan

# Combine global options
gcover --config custom.yaml --env prod --verbose qa process file.gdb
```

### Environment Variables

Global settings can be overridden with environment variables:

```bash
# Global overrides (affect all modules)
export GCOVER_GLOBAL_LOG_LEVEL=DEBUG
export GCOVER_GLOBAL_S3_BUCKET=my-custom-bucket
export GCOVER_GLOBAL_S3_PROFILE=my-profile

# Module-specific overrides
export GCOVER_GDB_DATABASE_PATH=/custom/path/db.duckdb
export GCOVER_SDE_CONNECTION_TIMEOUT=120

# Use the overrides
gcover gdb status  # Will use custom S3 bucket and debug logging
```

### GDB Asset Management - Usage

#### Quick Start

```bash
# Initialize the system
gcover gdb --env dev init

# Scan for GDB files
gcover gdb --env dev scan

# Process all found GDBs (dry run first)
gcover gdb --env dev process-all --dry-run
gcover gdb --env dev process-all

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

**Single Asset Processing**
```bash
gcover gdb process /path/to/specific.gdb    # Process single asset
```

**Batch Processing**
```bash
# Process all discovered assets
gcover gdb process-all                      # Process all found assets
gcover gdb process-all --dry-run           # Preview what would be processed

# Filtered processing
gcover gdb process-all --filter-type backup_daily    # Only daily backups
gcover gdb process-all --filter-rc RC1              # Only RC1 assets
gcover gdb process-all --since 2025-01-01           # Assets since date

# Advanced options
gcover gdb process-all --force                       # Reprocess even if already in DB
gcover gdb process-all --continue-on-error          # Don't stop on failures
gcover gdb process-all --max-workers 2              # Parallel processing (experimental)

# Combined filters
gcover gdb process-all --filter-type verification_topology --filter-rc RC2 --since 2025-02-01
```

##### Maintenance & Utilities

**System Maintenance**
```bash
gcover gdb clean-temp                      # Clean temporary zip files
gcover gdb clean-temp --dry-run           # Preview cleanup

gcover gdb validate                        # Validate processed assets
gcover gdb validate --check-s3            # Also validate S3 uploads
gcover gdb validate --check-integrity     # Verify file integrity
```

**Advanced Statistics**
```bash
gcover gdb stats                          # Basic overview
gcover gdb stats --by-date               # Statistics by month
gcover gdb stats --by-type               # Statistics by asset type and RC
gcover gdb stats --storage               # Storage and upload statistics
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
gcover gdb clean-temp

# Database maintenance
duckdb data/dev_gdb_metadata.duckdb 'VACUUM;'

# Validate system integrity
gcover gdb validate --check-s3 --check-integrity
```

#### Troubleshooting

```bash
# Verbose output for debugging
gcover gdb --env dev --verbose scan

# Check configuration
gcover gdb --env dev --verbose status

# Debug specific processing issues
gcover gdb --env dev --verbose process-all --dry-run --filter-type backup_daily

# Verify AWS credentials
aws s3 ls s3://your-bucket/

# Check database
duckdb data/dev_gdb_metadata.duckdb 'SELECT COUNT(*) FROM gdb_assets;'
```

#### Common Workflows

##### Daily Processing Workflow
```bash
#!/bin/bash
# Daily processing script
export GDB_ENV=production

# Scan for new assets
echo "Scanning for new GDB assets..."
gcover gdb scan

# Process only new assets from today
echo "Processing today's assets..."
gcover gdb process-all --since $(date +%Y-%m-%d) --continue-on-error

# Clean up temporary files
echo "Cleaning up..."
gcover gdb clean-temp

# Generate daily report
echo "Generating report..."
gcover gdb stats --storage > daily_report_$(date +%Y%m%d).txt
```

##### Weekly Maintenance
```bash
#!/bin/bash
# Weekly maintenance script
export GDB_ENV=production

# Full system validation
echo "Validating system integrity..."
gcover gdb validate --check-s3

# Comprehensive statistics
echo "Generating weekly statistics..."
gcover gdb stats --by-date --by-type --storage > weekly_stats_$(date +%Y%m%d).txt

# Database maintenance
echo "Optimizing database..."
duckdb data/prod_gdb_metadata.duckdb 'VACUUM; ANALYZE;'
```

##### Bulk Reprocessing
```bash
# Reprocess all assets of a specific type
gcover gdb process-all --filter-type verification_topology --force --continue-on-error

# Reprocess assets from a specific time period
gcover gdb process-all --since 2025-01-01 --filter-rc RC1 --force

# Process with maximum verbosity for debugging
gcover gdb --verbose process-all --dry-run --filter-type backup_daily
```

##### Monthly Report Generation
```bash
#!/bin/bash
# Monthly comprehensive report
MONTH=$(date -d "last month" +%Y-%m)
REPORT_FILE="gcover_monthly_report_${MONTH}.md"

echo "# GeoCover GDB Assets Monthly Report - $MONTH" > $REPORT_FILE
echo "" >> $REPORT_FILE

echo "## System Overview" >> $REPORT_FILE
gcover gdb status >> $REPORT_FILE
echo "" >> $REPORT_FILE

echo "## Monthly Statistics" >> $REPORT_FILE
gcover gdb stats --by-date --storage >> $REPORT_FILE
echo "" >> $REPORT_FILE

echo "## Asset Type Breakdown" >> $REPORT_FILE
gcover gdb stats --by-type >> $REPORT_FILE

echo "Monthly report generated: $REPORT_FILE"
```

##### Emergency Recovery
```bash
# If you need to rebuild the database from S3
gcover gdb init  # Recreate database structure

# Re-scan filesystem and reprocess everything
gcover gdb scan
gcover gdb process-all --force --continue-on-error

# Validate everything was processed correctly
gcover gdb validate --check-s3
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

#### Performance Tips

##### For Large Datasets
```bash
# Use continue-on-error for resilient processing
gcover gdb process-all --continue-on-error

# Process in smaller batches by type
gcover gdb process-all --filter-type backup_daily
gcover gdb process-all --filter-type verification_tqa
gcover gdb process-all --filter-type increment

# Use date filtering for incremental processing
gcover gdb process-all --since $(date -d '1 week ago' +%Y-%m-%d)
```

##### Monitoring Processing
```bash
# Monitor with verbose output and timestamps
gcover gdb --verbose process-all 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' | tee processing.log

# Monitor S3 uploads separately
aws s3 ls s3://your-bucket/gdb-assets/ --recursive | tail -f
```

#### Data Flow

```
📁 Local GDB Files → 🔍 Scan Discovery → 📦 ZIP Creation → 🔐 Hash Verification → ☁️ S3 Upload → 💾 Database Update
```

- **GDB assets** are discovered via filesystem scanning
- **Assets** are compressed and uploaded to S3 with integrity verification
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
gcover --verbose qa process /path/to/issue.gdb
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

QA commands use the same unified configuration system:

```yaml
# config/gcover_config.yaml
global:
  s3:
    bucket: "gcover-assets-dev"
    profile: "default"

# QA-specific settings (optional)
qa:
  output_dir: "./qa_output"
  database:
    path: "data/qa_metadata.duckdb"
  default_simplify_tolerance: 1.0
```

**Environment Variables:**
```bash
export GCOVER_GLOBAL_S3_BUCKET=gcover-assets-prod
export GCOVER_GLOBAL_S3_PROFILE=production
export GCOVER_QA_DATABASE_PATH=/home/user/.config/gcover/qa.duckdb
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
gcover --verbose qa process /path/to/issue.gdb
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
gcover --env production qa batch "$VERIFICATION_DIR" \
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


##### Version Management

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



### Schema Management  ###


The `gcover schema` command provides tools for extracting, comparing, and documenting ESRI geodatabase schemas.

#### Quick Start

```bash
# Extract schema from GDB
gcover schema extract /path/to/your.gdb --output schema.json

# Compare two schemas
gcover schema diff old_schema.json new_schema.json --output report.html

# Generate all report formats
gcover schema diff-all old.json new.json --output-dir ./reports
```

#### Schema Extraction

```bash
# Extract schema to JSON
gcover schema extract /path/to/database.gdb --output schema.json

# Extract with filtering
gcover schema extract database.gdb --filter-prefix "GC_" --output filtered_schema.json

# Multiple output formats
gcover schema extract database.gdb --format json xml --output-dir ./schemas
```

#### Schema Comparison

```bash
# Basic comparison (console output)
gcover schema diff old_schema.json new_schema.json

# HTML report
gcover schema diff old.json new.json --output changes.html --format html

# Markdown documentation
gcover schema diff old.json new.json --output changes.md --format markdown

# JSON export for automation
gcover schema diff old.json new.json --output changes.json --format json
```

#### Report Templates

| Template | Description | Best For |
|----------|-------------|----------|
| `full` | Detailed technical analysis | Complete schema documentation |
| `summary` | Executive overview | Management reports |
| `minimal` | Condensed view | Dashboards and monitoring |
| `incident` | Risk assessment format | Change impact analysis |

```bash
# Use specific template
gcover schema diff old.json new.json --template summary --output executive_report.html

# Generate comprehensive reports
gcover schema diff-all old.json new.json --output-dir ./monthly_reports --filter-prefix "GC_"
```

#### Advanced Options

```bash
# Filter by object prefix
gcover schema diff old.json new.json --filter-prefix "GC_" --output geocover_changes.html

# Custom template directory
gcover schema diff old.json new.json --template-dir ./custom_templates --output report.html

# Open in browser automatically
gcover schema diff old.json new.json --output report.html --open-browser

# PDF generation (requires pandoc)
gcover schema diff old.json new.json --output report.html
pandoc report.html --pdf-engine=xelatex -o report.pdf
```

#### Schema Documentation

```bash
# Generate schema documentation
gcover schema report schema.json --output documentation.html --template datamodel

# Create PlantUML diagram
gcover schema diagram schema.json --output schema_diagram.puml --title "GeoCover Schema"

# Generate multiple documentation formats
gcover schema report schema.json --template full --format html --output docs.html
gcover schema report schema.json --template full --format markdown --output docs.md
```

#### Typical Workflows

##### Daily Schema Monitoring
```bash
#!/bin/bash
# Compare latest daily backups
LATEST=$(ls -1 /media/marco/SANDISK/GCOVER/daily/*.gdb | tail -1)
PREVIOUS=$(ls -1 /media/marco/SANDISK/GCOVER/daily/*.gdb | tail -2 | head -1)

gcover schema diff "$PREVIOUS" "$LATEST" \
    --output "daily_$(date +%Y%m%d).html" \
    --template summary \
    --filter-prefix "GC_"
```

##### Monthly Schema Reports
```bash
# Generate comprehensive monthly comparison
CURRENT_MONTH=$(ls -1 /media/marco/SANDISK/GCOVER/monthly/*$(date +%Y%m)*.gdb | tail -1)
PREVIOUS_MONTH=$(ls -1 /media/marco/SANDISK/GCOVER/monthly/*$(date -d '1 month ago' +%Y%m)*.gdb | tail -1)

gcover schema diff-all "$PREVIOUS_MONTH" "$CURRENT_MONTH" \
    --output-dir "./reports/$(date +%Y-%m)" \
    --filter-prefix "GC_"
```

##### Quality Assurance Verification
```bash
# Compare production schema with QA results
BASELINE="/media/marco/SANDISK/GCOVER/daily/baseline.gdb"
QA_RESULTS="/media/marco/SANDISK/Verifications/TechnicalQualityAssurance/RC_2030-12-31/latest/issue.gdb"

gcover schema diff "$BASELINE" "$QA_RESULTS" \
    --output qa_verification.html \
    --template incident \
    --filter-prefix "GC_"
```

#### Configuration

Schema commands can be configured via YAML:

```yaml
# config/schema_config.yaml
template:
  default_template: "full"
  default_format: "html"

filtering:
  default_prefix_filter: "GC_"
  exclude_empty_changes: true

output:
  auto_open_browser: false
  create_backup: true

notifications:
  slack_webhook: "https://hooks.slack.com/..."
  threshold: 10
```

Use with: `gcover schema diff --config config/schema_config.yaml old.json new.json`

#### Output Formats

- **HTML**: Interactive reports with styling and navigation
- **Markdown**: Documentation-friendly format for version control
- **JSON**: Structured data for programmatic processing
- **PDF**: Professional reports (requires pandoc and LaTeX)

#### Integration with Automation

```python
# Python API usage
from gcover.schema import SchemaDiff, transform_esri_json
from gcover.schema.reporter import generate_report

# Load and compare schemas
with open('old_schema.json') as f:
    old_data = json.load(f)
with open('new_schema.json') as f:
    new_data = json.load(f)

old_schema = transform_esri_json(old_data)
new_schema = transform_esri_json(new_data)

diff = SchemaDiff(old_schema, new_schema)

if diff.has_changes():
    # Generate report
    generate_report(diff, template="summary", format="html", output_file="changes.html")
    
    # Process changes programmatically
    for change in diff.domain_changes:
        if change.change_type == ChangeType.REMOVED:
            print(f"Warning: Domain {change.domain_name} was removed")
```

#### Requirements

- **arcpy**: Required for GDB schema extraction (ArcGIS Pro installation)
- **jinja2**: Template rendering for reports
- **pyyaml**: Configuration file support
- **pandoc**: Optional, for PDF generation


## Global Configuration

All gcover commands use a unified configuration system with global and module-specific settings.

### Configuration Files

**Main config**: `config/gcover_config.yaml`
```yaml
global:
  log_level: INFO
  temp_dir: /tmp/gcover
  max_workers: 4
  s3:
    bucket: "gcover-assets-dev"
    profile: "default"

gdb:
  base_paths:
    backup: "/path/to/GCOVER"
  database:
    path: "data/metadata.duckdb"

sde:
  instances:
    GCOVERP:
      host: "sde-server.com"
      port: 5151
      database: "GCOVERP"

schema:
  output_dir: "./schemas"
  default_formats: ["json"]
```

**Environment overrides**: `config/environments/{environment}.yaml`
```yaml
# config/environments/production.yaml
global:
  log_level: WARNING
  s3:
    bucket: "gcover-assets-prod"
    profile: "production"
```