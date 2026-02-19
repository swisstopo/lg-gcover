$envPath = "Y:\conda"
$envName = "GCOVER_PROD"
$baseDir = "\\v0t0020a.adr.admin.ch\lg\01_PRODUKTION\GIS\TOPGIS\QA\Weekly"
$command = "gcover qa aggregate --zone-type mapsheet  --base-dir $baseDir  --type verification_topology "

# Activate Conda environment
& "$envPath\$envName\Scripts\activate.bat" $envName

# Run the command
Invoke-Expression $command
