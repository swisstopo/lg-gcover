# Development environment overrides
_environment: development

global:
log_level: DEBUG

gdb:
s3:
bucket: "gcover-assets-dev"

database:
path: "data/dev_gdb_metadata.duckdb"

# Development-specific SDE instances could go here if different