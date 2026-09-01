# Simple test script for gcover
Write-Host "=== GCOVER QA PROCESSING  ===" -ForegroundColor Yellow

# Path to conda environment
$CondaPath = "Y:\conda\envs\ARCGIS_37"
$OutputDir = "\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly"
$InputDir = "\\v0t0020a\topgisprod\10_Production_GC\Administration\QA"
$ZonesFile ="\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\Produktableitung\R18_2026\Mapsheet\GC_MAPSHEET.gpkg"
Write-Host "Using conda env: $CondaPath"
Write-Host "Using OuputDir: $OutputDir"
Write-Host "Using InputDir: $InputDir"
Write-Host "Using ZonesFile: $ZonesFile"

# Check if conda env exists
if (Test-Path $CondaPath) {
    Write-Host "Conda environment found!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Conda environment not found!" -ForegroundColor Red
    exit 1
}

# Activate conda environment
$env:PATH = "$CondaPath;$CondaPath\Scripts;$env:PATH"
Write-Host "Conda environment activated"


$LAST_WEEK = (Get-Date).AddDays(-7).ToString("yyyy-MM-dd")
Write-Host "Last week's date: $LAST_WEEK"




# python -m gcover.cli.main qa extract      --rc1-gdb "/home/marco/DATA/QA/Verifications/Topology/RC_2016-12-31/20260718_03-00-11/issue.gdb"      --rc2-gdb "/home/marco/DATA/QA/Verifications/Topology/RC_2030-12-31/20260717_07-00-12/issue.gdb"      --zones-file /home/marco/DATA/Derivations/delivery/R18/GC_MAPSHEET.gpkg
# --mapsheets-layer mapsheet_gc      --rand-border-filter none      --output /home/marco/DATA/Derivations/output/R18/qa_topology      --format gpkg      --yes

Write-Host "--- Processing QA extract ---" -ForegroundColor Green
& gcover    --env production --verbose  qa extract  --yes --type verification_topology --zones-file $ZonesFile  --mapsheets-layer mapsheet_gc      --rand-border-filter none --no-rc-breakdown --output $OutputDir

# Clean up stray top-level RC1/RC2 dirs (siblings of Topology/, not produced
# by this run — leftover cruft, harmless to remove each time).
foreach ($name in @("RC1", "RC2")) {
    $strayDir = Join-Path $OutputDir $name
    if (Test-Path $strayDir) {
        Remove-Item -Path $strayDir -Recurse -Force
        Write-Host "Removed stray directory: $strayDir" -ForegroundColor Yellow
    }
}

$TopologyDir = Join-Path $OutputDir "Topology"
$LastLink    = Join-Path $TopologyDir "last"

# Find most recently modified run directory (named yyyymmdd, e.g. 20260821)
$LastDir = Get-ChildItem -Path $TopologyDir -Directory |
    Where-Object { $_.Name -match '^\d{8}$' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $LastDir) {
    Write-Host "ERROR: Could not find the latest run directory." -ForegroundColor Red
    exit 1
}

Write-Host "Last run: $($LastDir.FullName)" -ForegroundColor Green
Write-Host "Updating: $LastLink" -ForegroundColor Green

# Remove previous 'last' directory
if (Test-Path $LastLink) {
    Remove-Item -Path $LastLink -Recurse -Force
}

# Create new 'last' directory
New-Item -Path $LastLink -ItemType Directory -Force | Out-Null

# Recursively copy the latest run into 'last'
Copy-Item `
    -Path (Join-Path $LastDir.FullName "*") `
    -Destination $LastLink `
    -Recurse `
    -Force

Write-Host "Last run copied to: $LastLink" -ForegroundColor Green





Write-Host "--- Processing QA aggregate ---" -ForegroundColor Green
& gcover  --env production --verbose  qa aggregate --auto-discover --yes --zone-type mapsheets  --zones-file $ZonesFile  --mapsheets-layer mapsheet_gc  --rand-border-filter none   --output-format xlsx   --type  verification_topology   --base-dir $OutputDir




Write-Host "=== QA PROCESSING COMPLETE ===" -ForegroundColor Yellow