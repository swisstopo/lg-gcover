# Simple test script for gcover
Write-Host "=== GCOVER Copy to SANDISK  ===" -ForegroundColor Green

# Set environment variable for current session
$env:OGR_LAYER_CREATION_OPTIONS = "TARGET_ARCGIS_VERSION=ARCGIS_PRO_3_2_OR_LATER"

# Set ouput encoding
$OutputEncoding = [System.Text.Encoding]::UTF8

chcp 65001

$env:PYTHON_RICH_FORCE_ASCII = "true"



# Path to conda environment
$CondaPath = "Y:\conda\envs\gcover-arcgis"
$OutputDir = "X:\mom"
$InputDir = "\\v0t0020a\topgisprod\10_Production_GC\Administration\QA"
$SINCE_DATE = (Get-Date).AddDays(-31).ToString("yyyy-MM-dd")
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


Write-Host "--- Convert and upload QA test results ---" -ForegroundColor Green
& gcover --env production --verbose --log-file $LogFile  qa process-all --yes  --format all --since $SINCE_DATE --max-workers  1  $InputDir



Write-Host "=== QA PROCESSING COMPLETE ===" -ForegroundColor Green


