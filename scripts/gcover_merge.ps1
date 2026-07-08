

$gcover =   'Y:\conda\envs\ARCGIS_37\Scripts\gcover.exe'
$DELIVERY = '\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\Produktableitung\R17_2026'
$SOURCES  = $DELIVERY + '\Sources'
$EXCELS   = $DELIVERY + '\Excels'
$OUTPUT   = '\\adb.intra.admin.ch\swisstopo$\Transfer\mom\R17'

# $OUTPUT   = 'D:\GeoCover\R17'

$env:HTTP_PROXY = 'http://proxy-bvcol.admin.ch:8080'
$env:HTTPS_PROXY = 'http://proxy-bvcol.admin.ch:8080'


$mergeArgs = @(
    '--env', 'production',
    '--config', 'H:\.config\gcover\environments\production.yaml',
    '--verbose',
    'publish', 'merge',
    '--rc1',                ($SOURCES + '\20260518_0341_2016-12-31.gdb'),
    '--rc2',                ($SOURCES + '\20260518_0330_2030-12-31.gdb'),
    '--custom-sources-dir', $SOURCES,
    '--sources',            ($EXCELS + '\GC_Sources_PA.xlsx'),
    '--force-2d',
    '--output',             ($OUTPUT + '\merged_master.gdb'),
    '--no-clip-to-swiss-border',
    '--enrich-mapsheet-links',
    '--exclude-metadata',
    '--schema-output',      ($OUTPUT + '\merged_final.gdb'),
    '--strati-links',       ($EXCELS + '\_Update_stratiLINK.xlsx')
)

$start = Get-Date
Write-Host ("Starting merge at {0:yyyy-MM-dd HH:mm:ss}" -f $start)

& $gcover @mergeArgs
$exitCode = $LASTEXITCODE

$end     = Get-Date
$elapsed = $end - $start
Write-Host ("Merge finished at {0:yyyy-MM-dd HH:mm:ss} - elapsed {1:hh\:mm\:ss} ({2}s), exit code {3}" -f `
    $end, $elapsed, [int]$elapsed.TotalSeconds, $exitCode)

exit $exitCode

