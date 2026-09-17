# Simple test script for gcover
Write-Host "=== STARTING SDE SCHEMA EXPORT  ===" -ForegroundColor Yellow

# Path to conda environment
$CondaPath = "Y:\conda\envs\ARCGIS_37"
$OutputDir = "X:\mom"


Write-Host "Using conda env: $CondaPath"
Write-Host "Using OuputDir: $OutputDir"

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



Write-Host "--- Extracting schema ---" -ForegroundColor Green
& gcover  --env production   schema extract --filter-prefix "GC_" --output X:/mom/schema/4.6/ Y:\connections\GCOVERP@osa.sde



Write-Host "--- Extracting tables ---" -ForegroundColor Green
& gcover  --env production   schema export-tables  --gc-tables-only --exclude-incremental  --format json  --output-dir  X:/mom/schema/4.6/  --workspace  Y:\connections\GCOVERP@osa.sde



Write-Host "=== EXTRACTION DONE ===" -ForegroundColor Yellow