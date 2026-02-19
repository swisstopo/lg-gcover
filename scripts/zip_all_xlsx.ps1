



$root = "L:\01_PRODUKTION\GIS\TOPGIS\QA\Weekly\Topology"
$zip  = "X:\mom\all_xlsx.zip"
$temp = "X:\mom"

# Remove old zip if needed
if (Test-Path $zip) { Remove-Item $zip }

Get-ChildItem "$root\202*" -Recurse -Filter *.xlsx |
    ForEach-Object {
        $relative = $_.FullName.Substring($root.Length).TrimStart('\')
        $dest = Join-Path "$temp\_zip_temp" $relative
        New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
        Copy-Item $_.FullName $dest
    }


Write-Host "--- Zipping to  $zip ---" -ForegroundColor Green
& Compress-Archive -Path "$temp\_zip_temp\*" -DestinationPath $zip

Remove-Item "$temp\_zip_temp" -Recurse -Force





