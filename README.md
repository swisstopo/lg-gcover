# lg-gcover

**A Python library and CLI tool for working with Swiss GeoCover geological vector data**

lg-gcover simplifies the processing and analysis of geological vector datasets from the Swiss national Geological Survey (swisstopo). Built on modern geospatial Python tools like GeoPandas and Shapely, it provides both programmatic APIs and command-line utilities for geological data workflows.

## Key Features
- **CLI Interface**: Easy-to-use `gcover` command for batch processing
- **GeoPandas Integration**: Seamless integration with the Python geospatial ecosystem
- **ESRI Compatibility**: Full support for ArcGIS Pro workflows via arcpy
- **SDE Bridge**: High-performance data import/export with Enterprise Geodatabases
- **Rich Output**: Beautiful terminal output with progress indicators and structured logging
- **Flexible Data Handling**: Support for various geological vector formats and projections

Perfect for geologists, GIS analysts, and researchers working with Swiss geological datasets who need efficient, reproducible data processing workflows.



## Installation

### For ArcGIS Pro Users:
```bash
# Activate ArcGIS Pro's Python environment
# Usually: conda activate arcgispro-py3

# Install without GDAL (uses ESRI's version)
pip install gcover[esri]

# For development
pip install gcover[esri-dev]
```

### For Standalone Users:
```bash
# Create new conda environment
conda create -n gcover python=3.11

# Install with latest GDAL
pip install gcover[standalone]

# Or with conda for GDAL
conda install -c conda-forge gdal>=3.11
pip install gcover
```


## Development Mode Installation

### For ArcGIS Pro Development:
```bash
# Clone the repository
git clone https://github.com/swisstopo/lg-gcover.git
cd lg-gcover

# Activate ArcGIS Pro environment
conda activate arcgispro-py3

# Install in development mode (uses ESRI's GDAL)
pip install -e .[esri-dev]

# Or if you prefer separating dev tools:
pip install -e .[esri]
pip install -e .[dev]
```

### For Standalone Development:
```bash
# Clone the repository
git clone https://github.com/swisstopo/lg-gcover.git
cd lg-gcover

# Create and activate environment
conda create -n gcover-dev python=3.11
conda activate gcover-dev

# Install in development mode with latest GDAL
pip install -e .[full]

# Or step by step:
pip install -e .[standalone,dev,docs,viz]
```


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
gcover --env dev gdb init

# Scan for GDB files
gcover --env dev gdb scan

# Process all found GDBs (dry run first)
gcover --env dev gdb process-all --dry-run
gcover --env dev gdb process-all

# Check status
gcover --env dev gdb status
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
gcover gdb list-assets                    # List recent assets (default: 20)
gcover gdb list-assets --limit 50         # List more assets
gcover gdb list-assets --type backup_daily # Filter by asset type
gcover gdb list-assets --rc RC1           # Filter by release candidate
gcover gdb list-assets --since 2025-07-01 # Filter by date

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
gcover --env dev gdb scan
gcover --env development gdb scan

# Production environment  
gcover --env prod gdb sync
gcover --env production gdb sync

# With custom config
gcover --config ./my-config.yaml --env prod gdb sync

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
gcover --env dev --verbose gdb scan

# Check configuration
gcover --env dev --verbose gdb status

# Debug specific processing issues
gcover --env dev --verbose gdb process-all --dry-run --filter-type backup_daily

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
gcover --env production gdb scan

# Process only new assets from today
echo "Processing today's assets..."
gcover --env production gdb process-all --since $(date +%Y-%m-%d) --continue-on-error

# Clean up temporary files
echo "Cleaning up..."
gcover --env production gdb clean-temp

# Generate daily report
echo "Generating report..."
gcover --env production gdb stats --storage > daily_report_$(date +%Y%m%d).txt
```

##### Weekly Maintenance
```bash
#!/bin/bash
# Weekly maintenance script

# Full system validation
echo "Validating system integrity..."
gcover --env production gdb validate --check-s3

# Comprehensive statistics
echo "Generating weekly statistics..."
gcover --env production gdb stats --by-date --by-type --storage > weekly_stats_$(date +%Y%m%d).txt

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
gcover --verbose gdb process-all --dry-run --filter-type backup_daily
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
gcover --env production gdb status >> $REPORT_FILE
echo "" >> $REPORT_FILE

echo "## Monthly Statistics" >> $REPORT_FILE
gcover --env production gdb stats --by-date --storage >> $REPORT_FILE
echo "" >> $REPORT_FILE

echo "## Asset Type Breakdown" >> $REPORT_FILE
gcover --env production gdb stats --by-type >> $REPORT_FILE

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
gcover --verbose gdb process-all 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' | tee processing.log

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

### SDE Enterprise Geodatabase Management

The `gcover sde` command provides comprehensive tools for managing SDE (Spatial Database Engine) connections, versions, and user access.

#### Overview

The SDE module combines connection management with high-performance data operations:

- **Smart Connection Management**: Auto-detection of user versions and writable databases
- **Bidirectional Data Operations**: Import/export between files (GPKG, GeoJSON, Shapefile) and SDE feature classes
- **Bulk Operations**: Configurable batch imports/exports for large datasets
- **CRUD Operations**: Full Create, Read, Update, Delete support with transaction safety
- **Format Flexibility**: Support for multiple geodata formats with automatic conversion

#### Quick Start

```bash
# Test connection and find your writable version
gcover sde connect-test

# Export GeoCover bedrock data to GPKG
gcover sde export "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" bedrock.gpkg

# Import updates from file
gcover sde import updates.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation update --dryrun

# Bulk export multiple feature classes
gcover sde export-bulk config/bulk_export.yaml --output-dir exports/
```

#### Connection Management

##### Find and Test Connections

```bash
# Find your user versions automatically
gcover sde user-versions

# List all versions with filtering
gcover sde versions -i GCOVERP --user-only --writable-only

# Test connection with version auto-detection
gcover sde connect-test --instance GCOVERP

# Interactive connection with version selection
gcover sde connect-test --instance GCOVERP --interactive
```

##### Connection Status and Cleanup

```bash
# List active SDE connections
gcover sde connections

# Clean up all active connections
gcover sde connections --cleanup

# List versions in different formats
gcover sde versions --format json > versions.json
gcover sde versions --format csv > versions.csv
```

#### Data Export Operations

##### Basic Export

```bash
# Export feature class to GPKG
gcover sde export "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" bedrock.gpkg

# Export to GeoJSON
gcover sde export "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_UNCO_DESPOSIT" \
    deposits.geojson --format GeoJSON

# Export with custom layer name
gcover sde export "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    data.gpkg --layer-name "gc_bedrock"
```

##### Filtered Export

```bash
# Export with WHERE clause
gcover sde export "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" bedrock_recent.gpkg \
    --where "DATEOFCHANGE > date '2024-01-01'"

# Export with spatial filter (bounding box)
gcover sde export "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" bedrock_bern.gpkg \
    --bbox "2585000,1158000,2602500,1170000"

# Export specific fields only
gcover sde export "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" bedrock_simple.gpkg \
    --fields "UUID,ROCK_TYPE,AGE,FORMATION"

# Limit number of features
gcover sde export "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" bedrock_sample.gpkg \
    --max-features 1000
```

##### Bulk Export

Create a configuration file for bulk operations:

```yaml
# config/bulk_export.yaml
exports:
  - feature_class: "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK"
    output_file: "gc_bedrock_full.gpkg"
    layer_name: "bedrock"

  - feature_class: "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_UNCO_DESPOSIT"
    output_file: "gc_deposits_recent.gpkg"
    layer_name: "deposits"
    where_clause: "DATEOFCHANGE > date '2024-01-01'"
    fields: ["UUID", "OBJECTID", "OPERATOR", "DATEOFCHANGE"]

  - feature_class: "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK"
    output_file: "gc_bedrock_bern.gpkg"
    bbox: [2585000.0, 1158000.0, 2602500.0, 1170000.0]
```

```bash
# Execute bulk export
gcover sde export-bulk config/bulk_export.yaml --output-dir exports/

# Bulk export with format override
gcover sde export-bulk config/bulk_export.yaml --format GeoJSON --overwrite
```

#### Data Import Operations

##### Basic Import

```bash
# Update existing features from GPKG
gcover sde import data/updates.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation update

# Insert new features
gcover sde import data/new_features.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation insert

# Update or insert (upsert)
gcover sde import data/changes.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation upsert
```

##### Advanced Import Options

```bash
# Update specific fields only
gcover sde import updates.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation update \
    --update-fields "MORE_INFO,OPERATOR" \
    --operator "DataProcessor"

# Geometry-only updates
gcover sde import geometry_fixes.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation update \
    --no-attributes

# Attributes-only updates
gcover sde import attribute_updates.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation update \
    --no-geometry

# Chunked processing for large datasets
gcover sde import large_dataset.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation update \
    --chunk-size 500
```

##### Safe Import Practices

```bash
# Always test with dry run first
gcover sde import updates.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation update --dryrun

# Use confirmation for destructive operations
gcover sde import updates.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation update --confirm

# Import from specific layer in multi-layer file
gcover sde import data.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --layer "bedrock_updates" --operation update
```

#### Synchronization Operations

Handle mixed operations (insert/update/delete) using an operation field:

```bash
# Synchronize changes using operation column
gcover sde sync changes.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation-field "_operation"

# Custom operation field name
gcover sde sync changes.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --operation-field "change_type"

# Require confirmation for deletions
gcover sde sync changes.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --confirm-deletes

# Test synchronization
gcover sde sync changes.gpkg "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" \
    --dryrun --operation-field "_change"
```
🔍 Vérification instance: GCOVERP

**Operation Field Values:**
- `insert`: Add new features
- `update`: Modify existing features  
- `delete`: Remove features
- `null`/empty: Skip feature

#### Version Management

```bash
# List versions with status information
gcover sde versions -i GCOVERP
# Output shows: Version name, Parent, Status (Owner/Writable/User)

# Find only writable versions
gcover sde versions -i GCOVERP --writable-only

# Export version information
gcover sde versions -i GCOVERP --format json > gcoverp_versions.json
```

#### Configuration

SDE commands use the unified configuration system:

```yaml
# config/gcover_config.yaml
global:
  s3:
    bucket: "gcover-assets-dev"
    profile: "default"

sde:
  instances:
    GCOVERP:
      host: "sde-server.example.com"
      port: 5151
      database: "GCOVERP"
    GCOVERI:
      host: "sde-integration.example.com"
      port: 5151
      database: "GCOVERI"
  
  defaults:
    instance: "GCOVERP"
    version_type: "user_writable"  # user_writable, user_any, default
    chunk_size: 1000
    uuid_field: "UUID"
```
👤 Recherche versions pour utilisateur: MYUSER

**Environment Variables:**
```bash
# Override SDE settings
export GCOVER_SDE_DEFAULT_INSTANCE=GCOVERP
export GCOVER_SDE_CONNECTION_TIMEOUT=120
export GCOVER_SDE_CHUNK_SIZE=500

# Use with commands
gcover sde export "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" output.gpkg
```

#### Python API Usage

```python
from gcover.sde import create_bridge

# Basic usage with context manager
with create_bridge() as bridge:  # Auto-detects GCOVERP + user_writable version
    # Export data
    gdf = bridge.export_to_geodataframe(
        "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK",
        max_features=1000
    )
    
    # Save to file
    bridge.export_to_file(
        "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK",
        "output.gpkg"
    )
    
    # Import from file (if writable)
    if bridge.is_writable:
        result = bridge.import_from_file(
            "updates.gpkg",
            "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK",
            operation="update",
            dryrun=True
        )
        print(f"Would update {result['success_count']} features")

# Advanced configuration
with create_bridge(
    instance="GCOVERP",
    version="USER.MYVERSION_20250726",
    uuid_field="UUID"
) as bridge:
    print(f"Connected to {bridge.version_name}")
    print(f"RC Version: {bridge.rc_full} ({bridge.rc_short})")
    print(f"Writable: {bridge.is_writable}")
```

#### Common Workflows

##### Daily Data Synchronization

```bash
#!/bin/bash
# Daily sync workflow
INSTANCE="GCOVERP"
FEATURE_CLASS="TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK"
DATA_DIR="/data/daily_updates"

# Test connection
echo "Testing SDE connection..."
gcover sde connect-test --instance $INSTANCE

# Process daily changes
for file in $DATA_DIR/*.gpkg; do
    echo "Processing $file..."
    
    # Dry run first
    gcover sde import "$file" "$FEATURE_CLASS" --operation update --dryrun
    
    # Ask for confirmation
    echo "Proceed with import? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        gcover sde import "$file" "$FEATURE_CLASS" --operation update \
            --operator "daily_sync" --chunk-size 1000
    fi
done
```

#### Latest Asset Discovery

Find the most recent QA tests, backups, and verification runs for each Release Candidate (RC1/RC2). Essential for monitoring daily QA processes and identifying release couples.

##### Quick Discovery Commands

```bash
# Find latest topology verification tests for each RC
gcover gdb latest-topology

# Find latest assets of any type for each RC
gcover gdb latest-by-rc --type verification_topology --show-couple

# Show all latest verification runs
gcover gdb latest-verifications
```

##### `gcover gdb latest-topology`

Show the latest topology verification tests for RC1 and RC2, with release couple detection.

```bash
# Basic usage
gcover gdb latest-topology

# Example output:
#                Latest Topology Verification Tests                     
# ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
# ┃ RC     ┃ Test Date           ┃ File                                     ┃       Size ┃ Status ┃
# ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
# │ RC1    │ 2025-07-19 03:00:23 │ issue.gdb                                │     4.7 MB │ ✅     │
# │ RC2    │ 2025-07-18 07:00:12 │ issue.gdb                                │    15.7 MB │ ✅     │
# └────────┴─────────────────────┴──────────────────────────────────────────┴────────────┴────────┘
# 
# ✅ Latest Release Couple: 1 days apart
# Latest tests: 2025-07-19 and 2025-07-18
# 
# Answer: The latest topology verification tests are:
#   RC1: 2025-07-19
#   RC2: 2025-07-18
```

##### `gcover gdb latest-by-rc`

Flexible command to find latest assets for any type, with filtering and release couple detection.

**Basic Usage:**
```bash
# Latest topology verification with couple check
gcover gdb latest-by-rc --type verification_topology --show-couple

# Latest backup assets
gcover gdb latest-by-rc --type backup_daily

# Latest assets from last 60 days only
gcover gdb latest-by-rc --type verification_topology --days-back 60
```

**Available Asset Types:**
- `backup_daily` - Daily backup files
- `backup_weekly` - Weekly backup files  
- `backup_monthly` - Monthly backup files
- `verification_topology` - Topology verification tests
- `verification_tqa` - Technical Quality Assurance tests
- `increment` - Data increment files

**Options:**
- `--type`: Filter by specific asset type
- `--days-back N`: Only consider assets from last N days (default: 30)
- `--show-couple`: Check if RC1/RC2 form a release couple (within 7 days)

##### `gcover gdb latest-verifications`

Show latest verification runs for all verification types (topology, TQA, etc.).

```bash
# Show all verification types
gcover gdb latest-verifications

# Example output shows separate tables for each verification type:
# - Latest Topology Verification
# - Latest Technical Quality Assurance Verification  
# Each with RC1/RC2 entries and release couple information
```

#### Release Couple Detection

**Release Couples** are RC1/RC2 assets created close together (typically within 24-48 hours), indicating synchronized QA testing runs.

**Detection Rules:**
- Maximum 7 days apart (configurable)
- Both RC1 and RC2 must have recent data
- Useful for identifying complete QA cycles

**Example Release Couple Output:**
```bash
✅ Latest Release Couple: 1 days apart
Latest tests: 2025-07-19 and 2025-07-18

# For assets more than 7 days apart:
⚠️  RC1 and RC2 are 12 days apart (not a close couple)
```

#### Common Use Cases

##### Daily QA Monitoring

```bash
#!/bin/bash
# Check if QA tests are up to date
echo "Checking latest QA test status..."

# Get latest topology verification dates
gcover gdb latest-topology > qa_status.txt

# Check if tests are recent (last 2 days)
LATEST_RC1=$(gcover gdb latest-by-rc --type verification_topology | grep "RC1" | awk '{print $3}')
if [[ $(date -d "$LATEST_RC1" +%s) -gt $(date -d "2 days ago" +%s) ]]; then
    echo "✅ QA tests are current"
else
    echo "⚠️  QA tests may be outdated"
fi
```

##### Weekly QA Reports

```bash
# Generate weekly QA summary
echo "# Weekly QA Report - $(date +%Y-%m-%d)" > weekly_qa.md
echo "" >> weekly_qa.md

# Latest test status
echo "## Latest Test Status" >> weekly_qa.md
gcover gdb latest-topology >> weekly_qa.md

# All verification types
echo "## All Verification Types" >> weekly_qa.md  
gcover gdb latest-verifications >> weekly_qa.md

# Historical data
echo "## Recent Activity" >> weekly_qa.md
gcover gdb list-assets --type verification_topology --limit 10 >> weekly_qa.md
```

##### Automated QA Gap Detection

```bash
#!/bin/bash
# Alert if QA tests have gaps

# Check each verification type
for VERIFICATION_TYPE in "verification_topology" "verification_tqa"; do
    echo "Checking $VERIFICATION_TYPE..."
    
    # Get latest for each RC
    LATEST=$(gcover gdb latest-by-rc --type "$VERIFICATION_TYPE" --days-back 7)
    
    # Check if both RC1 and RC2 have recent data
    if echo "$LATEST" | grep -q "RC1.*Not found\|RC2.*Not found"; then
        echo "⚠️  Missing recent $VERIFICATION_TYPE data!"
        echo "$LATEST"
        
        # Send alert (example with curl/Slack)
        # curl -X POST -H 'Content-type: application/json' \
        #     --data '{"text":"QA Alert: Missing '$VERIFICATION_TYPE' data"}' \
        #     $SLACK_WEBHOOK_URL
    fi
done
```

##### Find Latest Assets for Scripts

```python
# Python API for getting latest dates
import subprocess
import json
from datetime import datetime

def get_latest_topology_dates():
    """Get latest topology verification dates as dict"""
    # Using the CLI utility function
    result = subprocess.run([
        'python', '-c', 
        'from gcover.cli.gdb_cmd import get_latest_topology_dates; '
        'import json; '
        'dates = get_latest_topology_dates("gdb_metadata.duckdb"); '
        'print(json.dumps({"rc1": dates[0], "rc2": dates[1]} if dates else {}))'
    ], capture_output=True, text=True)
    
    return json.loads(result.stdout)

# Usage
dates = get_latest_topology_dates()
if dates:
    print(f"Latest RC1: {dates['rc1']}")
    print(f"Latest RC2: {dates['rc2']}")
```

#### Integration with Automation

##### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    triggers {
        cron('0 9 * * 1')  // Weekly Monday 9 AM
    }
    stages {
        stage('QA Status Check') {
            steps {
                script {
                    // Check latest QA tests
                    def qaStatus = sh(
                        script: 'gcover --env production gdb latest-topology',
                        returnStdout: true
                    ).trim()
                    
                    // Generate report
                    writeFile file: 'qa_status.txt', text: qaStatus
                    archiveArtifacts artifacts: 'qa_status.txt'
                    
                    // Check for issues
                    if (qaStatus.contains('Not found')) {
                        currentBuild.result = 'UNSTABLE'
                        echo 'Warning: Missing QA test data'
                    }
                }
            }
        }
    }
}
```

##### Database Query Integration

For direct database access:

```sql
-- SQL query to get latest topology verification for each RC
WITH ranked_assets AS (
    SELECT *,
           CASE 
               WHEN release_candidate = '2016-12-31' THEN 'RC1'
               WHEN release_candidate = '2030-12-31' THEN 'RC2'
               ELSE 'Unknown'
           END as rc_name,
           ROW_NUMBER() OVER (
               PARTITION BY release_candidate 
               ORDER BY timestamp DESC
           ) as rn
    FROM gdb_assets 
    WHERE asset_type = 'verification_topology'
)
SELECT rc_name, timestamp::DATE as test_date, path
FROM ranked_assets 
WHERE rn = 1 AND rc_name IN ('RC1', 'RC2')
ORDER BY rc_name;
```

#### Troubleshooting

##### No Data Found

```bash
# Check if scans have been run
gcover gdb status

# Verify asset types in database
gcover gdb list-assets --limit 5

# Check date ranges
gcover gdb latest-by-rc --type verification_topology --days-back 90
```

##### Missing Release Couples

```bash
# Check larger time window
gcover gdb latest-by-rc --type verification_topology --days-back 60

# Examine historical data
gcover gdb list-assets --type verification_topology --limit 20

# Check both verification types
gcover gdb latest-verifications
```

##### Performance with Large Databases

The latest asset queries are optimized with:
- Indexed timestamp columns for fast ordering
- Partition by RC for efficient row numbering
- Limited result sets (only latest per RC)
- String interpolation to avoid parameter binding issues

For very large databases (>100k assets), consider:
```bash
# Use shorter time windows
gcover gdb latest-by-rc --days-back 30

# Regular database maintenance
duckdb data/gdb_metadata.duckdb 'VACUUM; ANALYZE;'
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
