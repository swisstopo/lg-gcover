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

Write-Host "=== TEST COMPLETE ===" -ForegroundColor Green