@echo off
REM setup-arcgis-env.bat
REM Script pour crÃ©er un environnement conda propre pour gcover avec ArcGIS Pro

setlocal EnableDelayedExpansion

REM ============================================================================
REM Configuration
REM ============================================================================
set ENV_NAME=gcover-arcgis
set ENV_BASE=Y:\conda\envs
set ARCGIS_PRO=C:\Program Files\ArcGIS\Pro
set ARCGIS_PYTHON=%ARCGIS_PRO%\bin\Python\envs\arcgispro-py3

set ARCGIS_PYTHON=C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3

echo ================================================================================
echo CREATION ENVIRONNEMENT GCOVER + ARCGIS PRO
echo ================================================================================
echo.

REM ============================================================================
REM VÃ©rifications prÃ©alables
REM ============================================================================
echo Verification de ArcGIS Pro...

if not exist "%ARCGIS_PRO%" (
    echo [31m]Erreur: ArcGIS Pro non trouve a %ARCGIS_PRO%[0m
    exit /b 1
)

if not exist "%ARCGIS_PYTHON%" (
    echo Erreur: Environnement Python ArcGIS Pro introuvable[0m
    exit /b 1
)

echo [32mOK: ArcGIS Pro trouve[0m
echo.

REM ============================================================================
REM CrÃ©ation du rÃ©pertoire d'environnements
REM ============================================================================
if not exist "%ENV_BASE%" (
    echo Creation de %ENV_BASE%...
    mkdir "%ENV_BASE%"
)

set FULL_ENV_PATH=%ENV_BASE%\%ENV_NAME%

REM VÃ©rifier si l'environnement existe
if exist "%FULL_ENV_PATH%" (
    echo.
    echo Attention: L'environnement %ENV_NAME% existe deja[0m
    set /p "CONFIRM=Supprimer et recreer? (o/N): "
    
    if /i "!CONFIRM!"=="o" (
        echo Suppression de %FULL_ENV_PATH%...
        rmdir /s /q "%FULL_ENV_PATH%"
    ) else (
        echo Arret du script
        exit /b 0
    )
)

echo.
echo ================================================================================
echo ETAPE 1: Clonage de l'environnement ArcGIS Pro
echo ================================================================================
echo.

echo Clonage en cours (cela peut prendre plusieurs minutes)...
echo Source: %ARCGIS_PYTHON%
echo Destination: %FULL_ENV_PATH%
echo.

REM Utiliser conda clone
call conda create --name %ENV_NAME% --clone arcgispro-py3 --prefix "%FULL_ENV_PATH%" -y

if errorlevel 1 (
    echo Clonage conda echoue, tentative de copie manuelle...
    xcopy /E /I /H /Y "%ARCGIS_PYTHON%" "%FULL_ENV_PATH%"
    
    if errorlevel 1 (
        echo Echec de la copie manuelle
        exit /b 1
    )
)

echo [32mOK: Environnement clone
echo.

REM ============================================================================
REM Activation et nettoyage
REM ============================================================================
echo ================================================================================
echo ETAPE 2: Activation et nettoyage
echo ================================================================================
echo.

echo Activation de l'environnement...
call conda activate "%FULL_ENV_PATH%"

if errorlevel 1 (
    echo Echec de l'activation
    exit /b 1
)

REM VÃ©rifier Python
python --version
echo.

REM VÃ©rifier arcpy
echo Verification d'arcpy...
python -c "import arcpy; print('arcpy version:', arcpy.GetInstallInfo()['Version'])" 2>nul

if errorlevel 1 (
    echo Erreur: arcpy non disponible
    exit /b 1
)

echo [32mOK: arcpy disponible
echo.

REM Nettoyer les packages conflictuels
echo Nettoyage des packages conflictuels...
pip uninstall -y pyarrow fastparquet 2>nul

if errorlevel 1 (
    echo [32mOK: Aucun package conflictuel trouve
) else (
    echo [32mOK: Packages conflictuels supprimes
)

echo.

REM ============================================================================
REM Installation des dÃ©pendances
REM ============================================================================
echo ================================================================================
echo ETAPE 3: Installation des dependances gcover
echo ================================================================================
echo.

echo Installation via conda...
echo.

REM Packages conda (versions compatibles ESRI)
call conda install -y -c esri  geopandas shapely pandas pyyaml click rich

echo.
echo Installation des packages Python purs via pip...
pip install loguru structlog python-dotenv pydantic

echo.
echo [32mOK: Dependances installees

REM ============================================================================
REM Installation de lg-gcover
REM ============================================================================
echo.
echo ================================================================================
echo ETAPE 4: Installation de lg-gcover
echo ================================================================================
echo.

if exist "pyproject.toml" (
    echo Projet lg-gcover trouve dans le repertoire courant
    echo Installation en mode developpement...
    pip install -e .[dev] --no-deps
    
    if errorlevel 1 (
        echo Attention: Installation echouee
    ) else (
        echo [32mOK: lg-gcover installe
    )
) else (
    echo Projet lg-gcover non trouve
    echo Installez manuellement avec:
    echo   cd chemin\vers\lg-gcover
    echo   pip install -e . --no-deps
)

REM ============================================================================
REM VÃ©rification finale
REM ============================================================================
echo.
echo ================================================================================
echo ETAPE 5: Verification finale
echo ================================================================================
echo.

echo Verification des imports critiques...
python -c "import arcpy; print('  OK: arcpy', arcpy.GetInstallInfo()['Version'])"
python -c "import geopandas; print('  OK: geopandas', geopandas.__version__)"
python -c "import shapely; print('  OK: shapely', shapely.__version__)"
python -c "import loguru; print('  OK: loguru')"

echo.
echo ================================================================================
echo [32mSUCCES! Environnement cree
echo ================================================================================
echo.
echo Nom: %ENV_NAME%
echo Chemin: %FULL_ENV_PATH%
echo.
echo Pour activer cet environnement:
echo   conda activate "%FULL_ENV_PATH%"
echo.
echo Ou creer un raccourci activate-%ENV_NAME%.bat:
echo   @echo off
echo   call conda activate "%FULL_ENV_PATH%"
echo   cmd /k
echo.
echo Pour tester gcover:
echo   gcover --version
echo   gcover check-deps
echo.

pause