
# Set ouput encoding
$OutputEncoding = [System.Text.Encoding]::UTF8

chcp 65001

$env:PYTHON_RICH_FORCE_ASCII = "true"



# Path to conda environment
$CondaPath = "Y:\conda\envs\ARCGIS_36"
$OutputDir = "X:\mom"
$InputDir = "\\v0t0020a\topgisprod\10_Production_GC\Administration\QA"
$TODAY = (Get-Date).ToString("yyyy-MM-dd")
$SINCE_DATE = (Get-Date).AddDays(-8).ToString("yyyy-MM-dd")
$LogFile = "$OutputDir\gcover_$TODAY.log"

Write-Host "Using conda env: $CondaPath"
Write-Host "Using OuputDir: $OutputDir"
Write-Host "Using InputDir: $InputDir"
Write-Host "Since date: $SINCE_DATE"

# Check if conda env exists
if (Test-Path $CondaPath) {
    Write-Host "Conda environment found!" -ForegroundColor Green  2>&1 | Tee-Object -FilePath $LogFile -Append
} else {
    Write-Host "ERROR: Conda environment not found!" -ForegroundColor Red  2>&1 | Tee-Object -FilePath $LogFile -Append
    exit 1
}

# Activate conda environment
$env:PATH = "$CondaPath;$CondaPath\Scripts;$env:PATH"
Write-Host "Conda environment activated"  2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Host "--- Check GCOVER Version ---" -ForegroundColor Green
& gcover   --version

Write-Host "Log to: $LogFile"


Write-Host "--- Exporting schema tables ---" -ForegroundColor Green
& gcover --env production --log-file $LogFile  schema export-tables -w  "H:\connections\GCOVERP@osa.sde" -o  $OutputDir\schema --gc-tables-only



Write-Host "=== SCHEMA TABLES EXPORTED ===" -ForegroundColor Green


