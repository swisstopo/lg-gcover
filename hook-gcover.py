"""
PyInstaller hook for gcover CLI
Ensures all Click subcommands are included even when dynamically imported
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import sys
from pathlib import Path

print("[hook-gcover] Starting gcover module collection...")

# Collect all gcover submodules to ensure CLI commands are included
hiddenimports = collect_submodules('gcover')

# Explicitly ensure all CLI command modules are included
# These are dynamically imported in main.py with try/except blocks
cli_commands = [
    'gcover.cli.schema_cmd', 
    'gcover.cli.gdb_cmd',
    'gcover.cli.qa_cmd',
    'gcover.cli.publish_cmd',
    'gcover.cli.sde_cmd',
]

# Test which commands actually exist
found_commands = []
missing_commands = []

for cmd in cli_commands:
    if cmd not in hiddenimports:
        hiddenimports.append(cmd)
    
    # Try to import to verify it exists
    try:
        __import__(cmd)
        found_commands.append(cmd)
    except ImportError as e:
        missing_commands.append((cmd, str(e)))
    except Exception as e:
        missing_commands.append((cmd, f"Error: {e}"))

# Also ensure core modules that CLI commands depend on
core_modules = [
    'gcover.schema',
    'gcover.qa',
    'gcover.gdb',
    'gcover.sde',
    'gcover.publish',
    'gcover.config',
    'gcover.utils',
    'gcover.utils.logging',
    'gcover.sde.connection_manager',
    'gcover.arcpy_compat',
]

for module in core_modules:
    if module not in hiddenimports:
        hiddenimports.append(module)

# Collect any data files from gcover package
datas = collect_data_files('gcover', include_py_files=False)

# Report findings
print(f"[hook-gcover] Found {len(found_commands)}/{len(cli_commands)} CLI commands:")
for cmd in found_commands:
    print(f"[hook-gcover]   ✓ {cmd.split('.')[-1]}")

if missing_commands:
    print(f"[hook-gcover] ⚠️  Missing {len(missing_commands)} commands:")
    for cmd, error in missing_commands:
        print(f"[hook-gcover]   ✗ {cmd.split('.')[-1]}: {error}")

print(f"[hook-gcover] Added {len(core_modules)} core modules")
print(f"[hook-gcover] Total gcover imports: {len(hiddenimports)}")
print(f"[hook-gcover] Python version: {sys.version_info.major}.{sys.version_info.minor}")


