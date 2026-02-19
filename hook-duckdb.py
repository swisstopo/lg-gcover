"""
PyInstaller hook for duckdb
Ensures duckdb and its metadata are properly included
"""
from PyInstaller.utils.hooks import copy_metadata, collect_dynamic_libs

# Collect duckdb metadata (version info, etc.)
datas = copy_metadata('duckdb')

# Collect duckdb binary extensions
binaries = collect_dynamic_libs('duckdb')

# Ensure duckdb module is imported
hiddenimports = ['duckdb']

print(f"[hook-duckdb] Collected metadata: {len(datas)} entries")
print(f"[hook-duckdb] Collected binaries: {len(binaries)} entries")
