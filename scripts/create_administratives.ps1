# Simple test script for gcover
Write-Host "=== Create Administrative zones GPKG  ===" -ForegroundColor Green

# Set ouput encoding
$OutputEncoding = [System.Text.Encoding]::UTF8

chcp 65001

# Path to conda environment
$CondaPath = "Y:\conda\envs\ARCGIS_36"
$OutputDir = "X:\mom"
$InputDir = "\\v0t0020a\topgisprod\10_Production_GC\Administration\QA"
$ProduktAbleitung = "\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\Produktableitung"
$LogFile = "$OutputDir\gcover_$TODAY.log"
$DataDir = "src\gcover\data"

Write-Host "Using conda env: $CondaPath"
Write-Host "Using OuputDir: $OutputDir"
Write-Host "Using InputDir: $InputDir"
Write-Host "ProduktAbleitung: $ProduktAbleitung"



# 1. Get R-folders and extract numeric version
$latestRFolder = Get-ChildItem -Path $ProduktAbleitung -Directory |
    Where-Object { $_.Name -match "^R(\d+)_\d{4}$" } |
    Sort-Object {
        # Extract the number after R (e.g. R16_2026 → 16)
        [int]($_.Name -replace "^R(\d+)_.*$", '$1')
    } -Descending |
    Select-Object -First 1

if (-not $latestRFolder) {
    Write-Host "No Rxx_YYYY folders found."
    exit
}

Write-Host "Newest R-folder: $($latestRFolder.FullName)"

# 2. Find GC_Sources_PA.xlsx recursively inside that folder
$latestFile = Get-ChildItem -Path $latestRFolder.FullName -Filter "GC_Sources_PA.xlsx" -Recurse |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($latestFile) {
    Write-Host "Most recent GC_Sources_PA.xlsx: $($latestFile.FullName)" -ForegroundColor Green

} else {
    Write-Host "No GC_Sources_PA.xlsx found in $($latestRFolder.FullName)"   -ForegroundColor Red
}



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


Write-Host "--- Creating administratives GPKG ---" -ForegroundColor Green
& python ./scripts/create_administrative_zones.py --lots-file $DataDir\lots.geojson --wu-file $DataDir\WU.json --mapsheets-file $DataDir\mapsheets.geojson  --sources-file $latestFile.FullName   --output "$($latestRFolder.FullName)\administrative_zones.gpkg" --overwrite

Write-Host "=== Create Administrative zones GPKG  ===" -ForegroundColor Green


