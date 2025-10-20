# Simple test script for gcover
Write-Host "=== GCOVER QA PROCESSING  ===" -ForegroundColor Green

# Set environment variable for current session
$env:OGR_LAYER_CREATION_OPTIONS = "TARGET_ARCGIS_VERSION=ARCGIS_PRO_3_2_OR_LATER"

# Set ouput encoding
$OutputEncoding = [System.Text.Encoding]::UTF8

chcp 65001

$env:PYTHON_RICH_FORCE_ASCII = "true"



# Path to conda environment
$CondaPath = "Y:\conda\envs\gcover-arcgis"
$OutputDir = "\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly"
$InputDir = "\\v0t0020a\topgisprod\10_Production_GC\Administration\QA"
$LAST_WEEK = (Get-Date).AddDays(-7).ToString("yyyy-MM-dd")
$TODAY = (Get-Date).ToString("yyyy-MM-dd")
$LogFile = "$OutputDir\gcover_$TODAY.log"

Write-Host "Using conda env: $CondaPath"
Write-Host "Using OuputDir: $OutputDir"
Write-Host "Using InputDir: $InputDir"
Write-Host "Last week's date: $LAST_WEEK"

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


Write-Host "Log to: $LogFile"



Write-Host "--- Processing new GDB assets ---" -ForegroundColor Green
& gcover   --env production --log-file $LogFile  gdb process-all  --yes  --continue-on-error --filter-type verification_topology  --max-workers 1  --since $LAST_WEEK



Write-Host "--- Processing new QA tests ---" -ForegroundColor Green
& gcover --env production --log-file $LogFile  qa process-all  --qa-type topology  --max-workers 1 --format flatgeobuf --since $LAST_WEEK   $InputDir

Write-Host "--- Processing QA aggregate ---" -ForegroundColor Green
& gcover  --env production qa aggregate --auto-discover --yes --zone-type mapsheets   --output-format xlsx   --type  verification_topology   --base-dir $OutputDir
& gcover   --env production qa aggregate --auto-discover --yes --zone-type lots   --output-format xlsx   --type  verification_topology   --base-dir $OutputDir
& gcover  --env production qa aggregate --auto-discover --yes --zone-type work_units   --output-format xlsx   --type  verification_topology   --base-dir $OutputDir

Write-Host "--- Processing QA extract ---" -ForegroundColor Green
& gcover    --env production qa extract  --yes --type verification_topology --format filegdb  --output $OutputDir

Write-Host "=== QA PROCESSING COMPLETE ===" -ForegroundColor Green



# gcover   --env production  gdb process-all  --yes  --continue-on-error --filter-type verification_topology  --max-workers 1  --since 2025-10-01

# gcover --env production  qa process-all  --qa-type topology  --max-workers 1 --format flatgeobuf --since "2 week2 ago"  "\\v0t0020a\topgisprod\10_Production_GC\Administration\QA"

# gcover  --env production qa aggregate --auto-discover --yes --zone-type mapsheets   --output-format xlsx   --type  verification_topology   --base-dir "\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly"

# gcover   --env production qa aggregate --auto-discover --yes --zone-type lots   --output-format xlsx   --type  verification_topology   --base-dir "\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly"
# gcover  --env production qa aggregate --auto-discover --yes --zone-type work_units   --output-format xlsx   --type  verification_topology   --base-dir "\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly"

# gcover    --env production qa extract  --yes --type verification_topology --format filegdb  --output "\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly"
