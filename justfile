# .env.local is loaded automatically if present (gitignored).
# Only two values typically change between release cycles:
#   RELEASE=R18
#   STYLES_DATE=2026-11-10
set dotenv-load
set dotenv-filename := ".env.windows"

# Overridable via .env.windows or command line: just RELEASE=R18 all
RELEASE     := env_var_or_default("RELEASE",     "R17")
STYLES_DATE := env_var_or_default("STYLES_DATE", "2026-05-26")
BRANCH      := `git rev-parse --abbrev-ref HEAD`
HOME        := env_var("HOME")

# Directory paths — overridable from .env.windows; computed from HOME+RELEASE as fallback.
# UNC example:  DELIVERY_DIR=\\server\share\delivery\R17
DELIVERY_DIR    := env_var_or_default("DELIVERY_DIR", HOME / "DATA/Derivations/delivery" / RELEASE)
OUTPUT_DIR      := env_var_or_default("OUTPUT_DIR",   HOME / "DATA/Derivations/output"   / RELEASE)
STYLES_DIR      := env_var_or_default("STYLES_DIR",   DELIVERY_DIR / "styles" / STYLES_DATE)
GCOVER_DATA_DIR := "src/gcover/data"

# File paths
TRANSLATION_CSV   := DELIVERY_DIR / "Excels/_GeolCodeText_Trad.xlsx"
STRATI_LINK_PATH  := DELIVERY_DIR / "Excels/_Update_stratiLINK.xlsx"
MASTER_GDB        := OUTPUT_DIR / "merged_master.gdb"
FINAL_GDB         := OUTPUT_DIR / "merged_final.gdb"
FULL_GDB_PATH     := DELIVERY_DIR / "RC1.gdb"
DENORMALIZED_PATH := OUTPUT_DIR / "denormalized.gpkg"
CLASSIFIED_PATH   := OUTPUT_DIR / "denormalized_classified.gpkg"
TRANSLATED_PATH   := OUTPUT_DIR / "denormalized_classified_translated.gpkg"
ADMIN_ZONES_PATH  := GCOVER_DATA_DIR / "administrative_zones.gpkg"
PA_EXCEL_PATH     := DELIVERY_DIR / "Excels/GC_Sources_PA.xlsx"
CONFIG_PATH       := "config/esri_classifier_denormalized_geocover.yaml"
MAPSERVER_OUTPUT  := "mapserver_" + BRANCH

LAYERS := "fossils exploit_polygons exploit_points linear_objects point_objects bedrock surfaces unco_deposits"
TABLES_TO_IMPORT := "GC_GEOL_MAPPING_UNIT GC_GEOL_MAPPING_UNIT_ATT GC_LITSTRAT_FORMATION_BANK GC_CHRONO GC_EX_GEO_PLG_EXP_UNIT_GC_GMU GC_EX_GEO_PNT_EXP_UNIT_GC_GMU GC_FOSS_SYSTEM_GC_SYSTEM GC_UN_DEP_CHARACT_GC_CHARCAT GC_UN_DEP_COMPOSIT_GC_COMPOS GC_UN_DEP_MAT_TYPE_GC_LITHO GC_UN_DEP_ADMIXTUR_GC_ADMIXT"

set shell         := ["bash", "-uc"]
set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

# ---

default:
    @just --list
    @just vars

# Scaffold ~/.config/gcover/ and write a .env.windows template (safe: never overwrites)
[script('python')]
setup:
    import shutil, textwrap
    from pathlib import Path

    W = 10
    def status(label, path, created):
        tag = "[created]" if created else "[exists] "
        print(f"  {tag}  {label:<{W}}  {path}")

    # --- ~/.config/gcover/ ---
    cfg = Path.home() / ".config" / "gcover"
    (cfg / "environments").mkdir(parents=True, exist_ok=True)

    src = Path("config/gcover_config.yaml")
    dst = cfg / "gcover_config.yaml"
    if not dst.exists() and src.exists():
        shutil.copy2(src, dst)
        status("config", dst, created=True)
    else:
        status("config", dst, created=False)

    env_dir = cfg / "environments"
    status("env dir", env_dir, created=False)
    print(f"  [note]    {'env files':<{W}}  add production.yaml / development.yaml manually (gitignored)")

    # --- .env.windows template ---
    env_local = Path(".env.windows")
    if not env_local.exists():
        env_local.write_text(textwrap.dedent("""\
            # Local overrides — NOT committed to git.
            # Edit RELEASE + STYLES_DATE at the start of each release cycle.
            # Uncomment and set DELIVERY_DIR / OUTPUT_DIR only if the UNC roots differ from the default.

            RELEASE=R17
            STYLES_DATE=2026-05-26

            # DELIVERY_DIR=\\\\server\\share\\delivery\\R17
            # OUTPUT_DIR=\\\\server\\share\\output\\R17
            # STYLES_DIR=\\\\server\\share\\delivery\\R17\\styles\\2026-05-26
        """))
        status(".env.windows", env_local.resolve(), created=True)
    else:
        status(".env.windows", env_local.resolve(), created=False)

    print()
    print("  Run 'just vars' to verify all paths resolve correctly.")

# Print key variables and whether their paths exist on disk
[script('python')]
vars:
    import os, sys
    if sys.platform == "win32":
        os.system("")  # enable ANSI in Windows Terminal
    NO_COLOR = not sys.stdout.isatty() or "NO_COLOR" in os.environ
    R  = "" if NO_COLOR else "\033[0m"
    B  = "" if NO_COLOR else "\033[1m"
    DIM= "" if NO_COLOR else "\033[2m"
    G  = "" if NO_COLOR else "\033[32m"
    RD = "" if NO_COLOR else "\033[31m"
    CY = "" if NO_COLOR else "\033[36m"
    YL = "" if NO_COLOR else "\033[33m"

    W = 24
    def row(label, value):
        v = str(value)
        is_path = os.path.sep in v or v.startswith("//") or v.startswith("\\\\")
        if is_path:
            status = f"  {G}[ok]{R}" if os.path.exists(v) else f"  {RD}[missing]{R}"
        else:
            status = ""
        print(f"  {DIM}{label:<{W}}{R} {CY}{v}{R}{status}")

    def header(title):
        print(f"\n  {B}{YL}{title}{R}")

    header("Release / branch")
    row("RELEASE",          "{{RELEASE}}")
    row("BRANCH",           "{{BRANCH}}")
    header("Input paths")
    row("DELIVERY_DIR",     "{{DELIVERY_DIR}}")
    row("FULL_GDB_PATH",    "{{FULL_GDB_PATH}}")
    row("STYLES_DIR",       "{{STYLES_DIR}}")
    row("TRANSLATION_CSV",  "{{TRANSLATION_CSV}}")
    row("STRATI_LINK_PATH", "{{STRATI_LINK_PATH}}")
    row("ADMIN_ZONES_PATH", "{{ADMIN_ZONES_PATH}}")
    row("CONFIG_PATH",      "{{CONFIG_PATH}}")
    header("Output paths")
    row("OUTPUT_DIR",       "{{OUTPUT_DIR}}")
    row("MASTER_GDB",       "{{MASTER_GDB}}")
    row("FINAL_GDB",        "{{FINAL_GDB}}")
    row("DENORMALIZED_PATH","{{DENORMALIZED_PATH}}")
    row("CLASSIFIED_PATH",  "{{CLASSIFIED_PATH}}")
    row("TRANSLATED_PATH",  "{{TRANSLATED_PATH}}")
    row("MAPSERVER_OUTPUT", "{{MAPSERVER_OUTPUT}}")
    print()

# Full pipeline
all: merge denormalize classify translate

# --- Download ---

# Download RC1/RC2 backup from production
download:
    gcover --env production --verbose gdb download-couple --type backup \
        --output-dir "{{DELIVERY_DIR}}" --unzip --no-keep-zip

# --- Administrative zones (run once before first merge) ---

# Create lots / WU / mapsheets zones → src/gcover/data/
administrative-zones:
    @echo "--- Creating administrative zones ---"
    python ./scripts/create_administrative_zones.py \
        --lots-file      "{{GCOVER_DATA_DIR}}/lots.geojson" \
        --wu-file        "{{GCOVER_DATA_DIR}}/WU.json" \
        --mapsheets-file "{{GCOVER_DATA_DIR}}/mapsheets.geojson" \
        --sources-file   "{{PA_EXCEL_PATH}}" \
        --output         "{{OUTPUT_DIR}}/administrative_zones.gpkg" \
        --overwrite
    python -c "import shutil; shutil.copy2(r'{{PA_EXCEL_PATH}}', r'{{GCOVER_DATA_DIR}}/GC_Sources_PA.xlsx')"
    python -c "import shutil; shutil.copy2(r'{{OUTPUT_DIR}}/administrative_zones.gpkg', r'{{ADMIN_ZONES_PATH}}')"

# --- Pipeline ---

# 1. Merge RC1 + RC2 + custom sources → merged_master.gdb
merge:
    @echo "--- Merging sources ---"
    gcover publish merge \
        --rc1                "{{DELIVERY_DIR}}/RC1.gdb" \
        --rc2                "{{DELIVERY_DIR}}/RC2.gdb" \
        --custom-sources-dir "{{DELIVERY_DIR}}" \
        --force-2d \
        --output             "{{MASTER_GDB}}" \
        --no-clip-to-swiss-border \
        --enrich-mapsheet-links \
        --exclude-metadata \
        --schema-output      "{{FINAL_GDB}}" \
        --strati-links       "{{STRATI_LINK_PATH}}"
    @echo "--- Copying GC_MAPSHEET from administrative_zones ---"
    ogr2ogr -f "OpenFileGDB" -update -overwrite -dim XY \
        "{{MASTER_GDB}}" "{{ADMIN_ZONES_PATH}}" \
        -dialect SQLite \
        -sql "SELECT geom, MSH_MAP_TITLE, MSH_MAP_NBR, MSH_TOPO_NR, MSH_REV, SOURCE_RC, Version AS VERSION, BER, ERL, ber_link AS BER_LINK, erl_link AS ERL_LINK FROM mapsheets_sources_only" \
        -nln GC_MAPSHEET

# 2. Import lookup tables + denormalize all layers → denormalized.gpkg
# [script('python')] runs the body as a Python script — no shell loops, works on Windows.
# Paths are normalised to forward slashes so ogr2ogr handles UNC correctly.
[script('python')]
denormalize:
    import subprocess, sys

    def run(*cmd):
        r = subprocess.run(list(cmd))
        if r.returncode:
            sys.exit(r.returncode)

    master = r"{{MASTER_GDB}}".replace("\\", "/")
    full   = r"{{FULL_GDB_PATH}}".replace("\\", "/")
    denorm = r"{{DENORMALIZED_PATH}}".replace("\\", "/")

    print("--- Importing lookup tables ---")
    for t in "{{TABLES_TO_IMPORT}}".split():
        run("ogr2ogr", "-f", "OpenFileGDB", "-update", "-overwrite", master, full, t)

    print("--- Denormalizing layers ---")
    for layer in "{{LAYERS}}".split():
        run("python", "scripts/denormalize_geocover.py",
            "--remove-metadata", "-o", denorm,
            "--cd-gdb-path", full, "--tables", layer, master)

# 3. Apply .lyrx classification → denormalized_classified.gpkg
classify:
    @echo "--- Applying classification ---"
    gcover --env sandisk publish apply-config \
        --styles-dir "{{STYLES_DIR}}" \
        "{{DENORMALIZED_PATH}}" "{{CONFIG_PATH}}"

# 4. Add translated labels (de/fr) → denormalized_classified_translated.gpkg
translate:
    @echo "--- Translating ---"
    python ./scripts/translate_gpkg.py \
        -t               "{{TRANSLATION_CSV}}" \
        --strati-links   "{{STRATI_LINK_PATH}}" \
        --config         "{{CONFIG_PATH}}" \
        --lowercase-columns \
        --output         "{{TRANSLATED_PATH}}" \
        --langs de,fr \
        "{{CLASSIFIED_PATH}}"

# 5. Generate MapServer mapfiles from translated GPKG
mapfiles:
    @echo "--- Generating mapfiles -> {{MAPSERVER_OUTPUT}} ---"
    gcover --env production publish mapserver \
        --use-symbol-field \
        --output-dir     "{{MAPSERVER_OUTPUT}}" \
        --generate-combined \
        --styles-dir     "{{STYLES_DIR}}/styles" \
        --pattern-file   config/patterns_catalog.yaml \
        --gml-items      label \
        "{{CONFIG_PATH}}"
