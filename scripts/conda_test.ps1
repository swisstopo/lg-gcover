# Simple test script for gcover
Write-Host "=== GCOVER TEST SIMPLE ===" -ForegroundColor Green

# Path to conda environment
$CondaPath = "Y:\conda\envs\GCOVER_PROD"
Write-Host "Using conda env: $CondaPath"

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

# Test Python
$pythonExe = "$CondaPath\python.exe"
Write-Host "Testing Python..."
& $pythonExe --version

# Test gcover import
Write-Host "Testing gcover import..."
& $pythonExe -c "import gcover; print('gcover OK')"

# Test simple gcover command (version or help)
Write-Host "Testing gcover command..."
&  gcover --help

# Test simple gcover command (version or help)
Write-Host "Testing conda command..."
&  conda list

$LAST_WEEK = (Get-Date).AddDays(-7).ToString("yyyy-MM-dd")
Write-Host "Last week's date: $LAST_WEEK"


Write-Host "=== Processing new GDB assets ===" -ForegroundColor Green
& gcover --env production  gdb process-all  --yes  --continue-on-error --filter-type verification_topology   --since $LAST_WEEK --max-workers 1

Write-Host "=== Processing new QA tests ===" -ForegroundColor Green
& gcover --env production  qa process-all  --qa-type topology --since $LAST_WEEK --max-workers 1 --format flatgeobuf  \\v0t0020a\topgisprod\10_Production_GC\Administration\QA

Write-Host "=== Processing QA aggregate ===" -ForegroundColor Green
& gcover --env production qa aggregate --auto-discover --yes --zone-type mapsheets   --output-format xlsx   --type  verification_topology   --base-dir \\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly
& gcover --env production qa aggregate --auto-discover --yes --zone-type lots   --output-format xlsx   --type  verification_topology   --base-dir \\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly
& gcover --env production qa aggregate --auto-discover --yes --zone-type work_units   --output-format xlsx   --type  verification_topology   --base-dir \\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly

Write-Host "=== Processing QA extract ===" -ForegroundColor Green
& gcover  --env production qa extract  --yes --type verification_topology --format filegdb  --output \\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly

Write-Host "=== TEST COMPLETE ===" -ForegroundColor Green