




BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
LATEST_TAG := $(shell git describe --tags --match "v*" --abbrev=0)

# --- Variables ---
RELEASE      ?= R18
DELIVERY_DIR := ${HOME}/DATA/Derivations/delivery/$(RELEASE)/
SOURCES_DIR  := $(DELIVERY_DIR)Sources/
OUTPUT_DIR   ?= ${HOME}/DATA/Derivations/output/$(RELEASE)/
STYLES_DIR   := ${HOME}/DATA/Derivations/delivery/$(RELEASE)/Styles/2026-08-31/
PREVIOUS_STYLES_DIR := ${HOME}/DATA/Derivations/delivery/R17/Styles/2026-07-02/styles/
TRANSLATION_CSV := $(DELIVERY_DIR)Excels/2026c_GeolCodeText_Trad.xlsx
STRATI_LINK_PATH := ${HOME}/DATA/Derivations/delivery/$(RELEASE)/Excels/_Update_stratiLINK.xlsx
# --strati-links is optional in `gcover publish merge` (strati_link is just
# omitted from the output if absent) — only pass it when the file is
# actually there, instead of failing the whole merge over a missing extra.
STRATI_LINK_ARG := $(if $(wildcard $(STRATI_LINK_PATH)),--strati-links $(STRATI_LINK_PATH),)
# `merge`'s "Proceed with merge?" prompt has no TTY to answer it in cron/CI —
# NONINTERACTIVE=1 (set explicitly by unattended callers, never by default)
# passes --yes to skip it. Interactive `make merge` still prompts as normal.
NONINTERACTIVE ?= 0
YES_FLAG := $(if $(filter 1,$(NONINTERACTIVE)),--yes,)
GCOVER_DATA_DIR :=  src/gcover/data/

ASPECT_LAYERS := surfaces_filtered unco_deposits_filtered
# Ensure no trailing whitespace
ASPECT_LAYERS := $(strip $(ASPECT_LAYERS))



# File Paths
MASTER_GDB        ?= $(OUTPUT_DIR)merged_master.gdb
FINAL_GDB         ?= $(OUTPUT_DIR)merged_final.gdb
DENORMALIZED_GPKG := denormalized.gpkg
DENORMALIZED_PATH := $(OUTPUT_DIR)$(DENORMALIZED_GPKG)
CLASSIFIED_GPKG	  := denormalized_classified.gpkg
CLASSIFIED_PATH   := $(OUTPUT_DIR)$(CLASSIFIED_GPKG)
#TRANSLATED_GPKG   := denormalized_classified_translated.gpkg
TRANSLATED_GPKG   := swissgeocover2d.gpkg
TRANSLATED_PATH   := $(OUTPUT_DIR)$(TRANSLATED_GPKG)
TRANSLATED_README := $(TRANSLATED_PATH:.gpkg=.README)
TRANSLATED_SCHEMA := $(TRANSLATED_PATH).schema.json
FULL_GDB_PATH     := $(SOURCES_DIR)RC1.gdb     # TODO Val Bregaglia missing GMU_ATT in RC2. RC1 OK
GEOCOVER_AUX_PATH := $(OUTPUT_DIR)geocover_aux.gpkg
ADMIN_ZONES_GPKG  := administrative_zones.gpkg
MAPSERVER_OUTPUT      ?= mapserver_$(BRANCH)
DEM_ASPECT_PATH       ?= $(DELIVERY_DIR)swissALTI3DRegio_aspect_50m.tif
PA_EXCEL_PATH         ?= $(DELIVERY_DIR)Excels/GC_Sources_PA.xlsx
PA_ZONES_PATH         := $(patsubst %.xlsx,%_zones.gpkg,$(PA_EXCEL_PATH))
# Delivered GC_MAPSHEET.gpkg, used directly (unmangled) as the --admin-zones
# input for `merge` — replaces the old lots/WU/mapsheets.geojson admin-zones
# pipeline. Source-assignment column is BKP; erl_link/ber_link (not present
# in this file) are computed on the fly by the merge and by the GC_MAPSHEET
# embed step below. GC_MAPSHEET itself is slated for removal, so this is a
# deliberately short-lived arrangement, not built via any pre-processing script.
GC_MAPSHEET_SOURCE    := $(DELIVERY_DIR)Mapsheet/GC_MAPSHEET.gpkg
QA_RAND_PATH          ?= $(DELIVERY_DIR)rand_qa_gc.geojson
CONFIG_PATH           ?= config/esri_classifier_denormalized_geocover.yaml
MERGE_LOG             := $(OUTPUT_DIR)merge.log
MERGE_CONSOLE_LOG     := $(OUTPUT_DIR)merge_console.log
MERGED_FINAL_README   := $(FINAL_GDB).README

# Pass VERBOSE=1 to keep loguru on the terminal and enable debug output.
# Default: loguru is written to MERGE_LOG only; Rich output stays in the terminal.
VERBOSE               ?= 0
_GCOVER_FLAGS         := --log-file $(MERGE_LOG) $(if $(filter 1,$(VERBOSE)),--verbose,)

# Layers for denormalization
LAYERS := fossils exploit_polygons exploit_points linear_objects point_objects bedrock surfaces unco_deposits
TABLES_TO_IMPORT := GC_GEOL_MAPPING_UNIT GC_GEOL_MAPPING_UNIT_ATT GC_LITSTRAT_FORMATION_BANK GC_CHRONO \
                    GC_EX_GEO_PLG_EXP_UNIT_GC_GMU GC_EX_GEO_PNT_EXP_UNIT_GC_GMU \
                    GC_FOSS_SYSTEM_GC_SYSTEM \
                    GC_UN_DEP_CHARACT_GC_CHARCAT GC_UN_DEP_COMPOSIT_GC_COMPOS GC_UN_DEP_MAT_TYPE_GC_LITHO GC_UN_DEP_ADMIXTUR_GC_ADMIXT


# ANSI color codes
RED    := \033[31m
GREEN  := \033[32m
YELLOW := \033[33m
BLUE   := \033[34m
BOLD   := \033[1m
RESET  := \033[0m


# The Generic Function
# $(1) = Variable Name (for display)
# $(2) = Path to check
define check_file
	@printf "   %-20s %-40s " "$(1):" "$(2)"
	@if [ -e "$(2)" ]; then \
		printf "$(GREEN)[FOUND]$(RESET)\n"; \
	else \
		printf "$(RED)[NOT FOUND]$(RESET)\n"; \
	fi
endef

# --- Targets ---

.DEFAULT_GOAL := help

## help: Show this help message

help:
	@echo "$(BOLD)Usage: make [target]"
	@echo ""
	@echo "Targets:$(RESET)"
	@awk '/^### / { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) } \
		 /^## /  { printf "  %-25s %s\n", $$2, substr($$0, index($$0, $$3)) }' \
		 $(MAKEFILE_LIST) | sed 's/://'
	@echo ""
	@echo ""
	@echo "$(YELLOW)Input$(RESET)"
	@echo "  $(BOLD)RELEASE  $(RED)$(RELEASE)$(RESET)"
	@echo "  Delivery GDBs:        $(SOURCES_DIR)"
	$(call check_file,STYLES_DIR,$(STYLES_DIR))
	$(call check_file,TRANSLATION_CSV,$(TRANSLATION_CSV))
	$(call check_file,STRATI_LINK_PATH,$(STRATI_LINK_PATH))
	$(call check_file,PA_EXCEL_PATH,$(PA_EXCEL_PATH))
	$(call check_file,GC_MAPSHEET_SOURCE,$(GC_MAPSHEET_SOURCE))
	$(call check_file,DEM_ASPECT,$(DEM_ASPECT_PATH))
	$(call check_file,QA_RAND_PATH,$(QA_RAND_PATH))
	@echo ""
	@echo "$(YELLOW)Output$(RESET)"
	@echo "  Output dir:           $(OUTPUT_DIR)"
	$(call check_file,MASTER_GDB,$(MASTER_GDB))
	$(call check_file,DENORMALIZED_PATH,$(DENORMALIZED_PATH))
	$(call check_file,CLASSIFIED_PATH,$(CLASSIFIED_PATH))
	$(call check_file,TRANSLATED_PATH,$(TRANSLATED_PATH))
	$(call check_file,GEOCOVER_AUX_PATH,$(GEOCOVER_AUX_PATH))
	$(call check_file,CONFIG_PATH,$(CONFIG_PATH))

	@echo ""
	@echo "$(YELLOW)Mapserver$(RESET)"
	@echo "  Mapserver dir:        $(MAPSERVER_OUTPUT)"
	@echo ""
	@echo "TAG:     $(LATEST_TAG)"



.PHONY: help download administrative-zones all merge merge-diagnostic \
        denormalize classify translate pipeline-check checksum \
        geometry-check line-topology-check polygon-topology-check coverage-check \
        domain-check domain-check-rc domain-check-final domain-check-custom filter-check \
        schema-snapshot-translated schema-snapshot-final \
        geocover-aux aspect aspect-simple aspect-gmm combine-aspect inject-aux-aspect \
        mapfiles \
        install-dev format lint test smoke doc check \
        clean-denormalize clean-translate clean-classify clean-merge clean-master clean-all \
        swissgeocover2d clean-swissgeocover2d

### Geocover data

## download:  Download RC1/RC2  backups
download:
	@gcover --env production --verbose  gdb download-couple --type backup --output-dir $(SOURCES_DIR)  \
	 --unzip --no-keep-zip


## all: Run the entire workflow (Merge -> Import -> Denormalize -> Symbolize)
all: merge $(CLASSIFIED_PATH) $(TRANSLATED_PATH)

## merge: Only perform the gcover merge and diagnosis
merge: $(MASTER_GDB)/timestamps

# 1. Merge sources and run diagnosis
$(MASTER_GDB)/timestamps: $(SOURCES_DIR)RC1.gdb $(SOURCES_DIR)RC2.gdb $(GC_MAPSHEET_SOURCE)
	@mkdir -p $(OUTPUT_DIR); \
	_T_START=$$(date +%s); \
	\
	echo "--- [1/2] Merging Sources ---"; \
	_T1=$$(date +%s); \
	{ gcover $(_GCOVER_FLAGS) publish merge \
		--rc1 $(SOURCES_DIR)RC1.gdb \
		--rc2 $(SOURCES_DIR)RC2.gdb \
		--custom-sources-dir $(SOURCES_DIR) \
		--admin-zones $(GC_MAPSHEET_SOURCE) \
		--force-2d --output $(MASTER_GDB) \
		--mapsheets-layer mapsheet_gc \
		--source-column BKP \
		--no-clip-to-swiss-border \
		--enrich-mapsheet-links \
		--exclude-metadata \
		--schema-output $(FINAL_GDB) \
		$(STRATI_LINK_ARG) $(YES_FLAG); \
	  echo $$? > $(OUTPUT_DIR).merge_rc; \
	} 2>&1 | tee $(MERGE_CONSOLE_LOG); \
	rc=$$(cat $(OUTPUT_DIR).merge_rc); rm -f $(OUTPUT_DIR).merge_rc; \
	_T2=$$(date +%s); \
	echo "  ↳ merge+schema: $$((_T2 - _T1))s"; \
	if [ $$rc -eq 130 ]; then \
		echo ""; \
		echo "Merge cancelled — build stopped. Run 'make merge' to retry."; \
		exit 1; \
	fi; \
	[ $$rc -ne 0 ] && exit $$rc; \
	\
	echo "--- [2/2] Copying GC_MAPSHEET from $(GC_MAPSHEET_SOURCE) ---"; \
	_T3=$$(date +%s); \
	ogr2ogr -f "OpenFileGDB" -update -overwrite -dim XY $(MASTER_GDB) \
		$(GC_MAPSHEET_SOURCE) \
		-lco TARGET_ARCGIS_VERSION=ARCGIS_PRO_3_2_OR_LATER \
		-dialect SQLite \
		-sql "SELECT geom, MSH_MAP_TITLE, MSH_MAP_NBR, MSH_TOPO_NR, BKP AS SOURCE_RC, Version AS VERSION, BER, ERL, \
			CASE WHEN BER = 'y' THEN 'https://data.geo.admin.ch/ch.swisstopo.geologie-geocover/berichte/BER_' || MSH_MAP_NBR || '.pdf' ELSE '' END AS BER_LINK, \
			CASE WHEN ERL = 'y' THEN 'https://data.geo.admin.ch/ch.swisstopo.geologie-geologischer_atlas/erlaeuterungen/GA25-ERL-' || MSH_MAP_NBR || '.pdf' ELSE '' END AS ERL_LINK \
			FROM mapsheet_gc" \
		-nln GC_MAPSHEET; \
	_T4=$$(date +%s); \
	echo "  ↳ ogr2ogr GC_MAPSHEET: $$((_T4 - _T3))s"; \
	\
	echo ""; \
	echo "  Total merge: $$((_T4 - _T_START))s"; \
	\
	RC1_SRC=$$(basename "$$(readlink -f $(SOURCES_DIR)RC1.gdb 2>/dev/null)" 2>/dev/null || echo "N/A"); \
	RC2_SRC=$$(basename "$$(readlink -f $(SOURCES_DIR)RC2.gdb 2>/dev/null)" 2>/dev/null || echo "N/A"); \
	RC1_DATE=$$(echo "$$RC1_SRC" | sed 's/\(.\{4\}\)\(.\{2\}\)\(.\{2\}\).*/\1-\2-\3/'); \
	RC2_DATE=$$(echo "$$RC2_SRC" | sed 's/\(.\{4\}\)\(.\{2\}\)\(.\{2\}\).*/\1-\2-\3/'); \
	{ printf "date_operation: %s\nrelease:        %s\nrc1_source:     %s\nrc1_date:       %s\nrc2_source:     %s\nrc2_date:       %s\ngc_mapsheet:    %s\nlg_gcover_tag:  %s\n\n--- Merge configuration (as displayed for confirmation) ---\n\n" \
		"$$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(RELEASE)" "$$RC1_SRC" "$$RC1_DATE" "$$RC2_SRC" "$$RC2_DATE" "$(GC_MAPSHEET_SOURCE)" "$(LATEST_TAG)"; \
	  cat $(MERGE_CONSOLE_LOG); \
	} > $(MERGED_FINAL_README); \
	echo "Written README to $(MERGED_FINAL_README)"

## merge-diagnostic: Merge diagnostic
merge-diagnostic:
	@echo "--- Running Diagnosis ---"
	python scripts/diagnose_merge.py $(SOURCES_DIR)RC1.gdb $(SOURCES_DIR)RC2.gdb $(GCOVER_DATA_DIR)$(ADMIN_ZONES_GPKG)



# 2. Add missing tables and Denormalize
.PHONY: add-tables
add-tables: $(MASTER_GDB)/timestamps
		@echo "--- Importing missing tables via ogr2ogr ---"
		@for table in $(TABLES_TO_IMPORT); do \
			ogr2ogr -f "OpenFileGDB" -update -overwrite $(MASTER_GDB) $(FULL_GDB_PATH) $$table; \
		done

$(DENORMALIZED_PATH): $(MASTER_GDB)/timestamps
		@echo "--- Importing missing tables via ogr2ogr ---"
		@for table in $(TABLES_TO_IMPORT); do \
			ogr2ogr -f "OpenFileGDB" -update -overwrite $(MASTER_GDB) $(FULL_GDB_PATH) $$table; \
		done

	@echo "--- Running Denormalization loop ---"
	@for layer in $(LAYERS); do \
		scripts/denormalize_geocover.py --remove-metadata  -o $(DENORMALIZED_PATH) --cd-gdb-path $(FULL_GDB_PATH)  --tables $$layer $(MASTER_GDB) ; \
	done

	@echo "--- Copying GC_MAPSHEET as 'mapsheet' (straight passthrough) ---"
	@ogr2ogr -f GPKG -update -overwrite $(DENORMALIZED_PATH) $(MASTER_GDB) GC_MAPSHEET -nln mapsheet

$(TRANSLATED_PATH): $(CLASSIFIED_PATH)
	@echo "--- Translating $(CLASSIFIED_PATH) ---"
	@echo "Saving to $(TRANSLATED_PATH)"
	python ./scripts/translate_gpkg.py -t $(TRANSLATION_CSV) \
		--strati-links $(STRATI_LINK_PATH) \
		--config $(CONFIG_PATH)  \
		--lowercase-columns --output $(TRANSLATED_PATH)  --langs de,fr  $(CLASSIFIED_PATH)
	@RC1_SRC=$$(basename "$$(readlink -f $(SOURCES_DIR)RC1.gdb 2>/dev/null)" 2>/dev/null || echo "N/A"); \
	RC2_SRC=$$(basename "$$(readlink -f $(SOURCES_DIR)RC2.gdb 2>/dev/null)" 2>/dev/null || echo "N/A"); \
	RC1_DATE=$$(echo "$$RC1_SRC" | sed 's/\(.\{4\}\)\(.\{2\}\)\(.\{2\}\).*/\1-\2-\3/'); \
	RC2_DATE=$$(echo "$$RC2_SRC" | sed 's/\(.\{4\}\)\(.\{2\}\)\(.\{2\}\).*/\1-\2-\3/'); \
	printf "date_operation: %s\nrc1_source:     %s\nrc1_date:       %s\nrc2_source:     %s\nrc2_date:       %s\nlg_gcover_tag:  %s\n" \
		"$$(date +%Y-%m-%d)" "$$RC1_SRC" "$$RC1_DATE" "$$RC2_SRC" "$$RC2_DATE" "$(LATEST_TAG)" \
		> $(TRANSLATED_README)
	@echo "Written README to $(TRANSLATED_README)"

$(CLASSIFIED_PATH): $(DENORMALIZED_PATH)
	@echo "--- Applying Style Configuration to $(DENORMALIZED_PATH)---"
	@gcover --env sandisk publish apply-config --styles-dir $(STYLES_DIR) \
		$(DENORMALIZED_PATH) $(CONFIG_PATH)

	@echo "--- Copying 'mapsheet' layer (no reclassification) ---"
	@ogr2ogr -f GPKG -update -overwrite $(CLASSIFIED_PATH) $(DENORMALIZED_PATH) mapsheet -nln mapsheet

## denormalize: Only run the table import and denormalization (requires master GDB)
denormalize: $(DENORMALIZED_PATH)

## classify: Apply classification from .lyrx to denormalized data
classify: $(CLASSIFIED_PATH)

## translate: Add human-readable values for geolcodes
translate: $(TRANSLATED_PATH) checksum schema-snapshot-translated

## swissgeocover2d: final GPKG for KOGIS
swissgeocover2d: translate

## checksum: Compute SHA256 checksum of the translated GPKG
.PHONY: checksum
checksum:
	$(call check_file,TRANSLATED_PATH,$(TRANSLATED_PATH))
	@sha256sum $(TRANSLATED_PATH) | tee $(TRANSLATED_PATH).sha256
	@echo "Written to $(TRANSLATED_PATH).sha256"

## schema-snapshot-translated: Generate a JSON schema snapshot of swissgeocover2d.gpkg and copy to config/
.PHONY: schema-snapshot-translated
schema-snapshot-translated:
	$(call check_file,TRANSLATED_PATH,$(TRANSLATED_PATH))
	gcover schema snapshot $(TRANSLATED_PATH) -o $(TRANSLATED_SCHEMA)
	@cp $(TRANSLATED_SCHEMA) config/swissgeocover2d_schema.json
	@echo "Written to $(TRANSLATED_SCHEMA) and config/swissgeocover2d_schema.json"

### Data check

## pipeline-check: Check feature counts are consistent across merge→denormalize→classify→translate
pipeline-check:
	@python scripts/check_pipeline_counts.py \
		$(DENORMALIZED_PATH) \
		$(CLASSIFIED_PATH) \
		$(TRANSLATED_PATH) \
		--gdb $(MASTER_GDB) \
		--config $(CONFIG_PATH)


## geometry-check: Check geometry validity and bedrock/unco_deposits coverage per mapsheet
geometry-check:
	@python scripts/check_geometry_coverage.py \
		$(MASTER_GDB) \
		--output-gpkg $(OUTPUT_DIR)geometry_check.gpkg \
		--report $(OUTPUT_DIR)geometry_check.txt

## line-topology-check: Check tectonic line topology against GC_BEDROCK boundaries per mapsheet. Use MAPSHEET=1145 to restrict to a single sheet during debugging.
.PHONY: line-topology-check
line-topology-check:
	@python scripts/check_line_topology.py \
		$(MASTER_GDB) \
		--output-gpkg $(OUTPUT_DIR)line_topology_check.gpkg \
		--report $(OUTPUT_DIR)line_topology_check.txt \
  		$(if $(MAPSHEET),--mapsheet $(MAPSHEET),)

## polygon-topology-check: Check polygon micro-gaps and overlaps within and across layers per mapsheet. Use MAPSHEET=1214 to restrict to a single sheet during debugging.
.PHONY: polygon-topology-check
polygon-topology-check:
	@python scripts/check_polygon_topology.py \
		$(MASTER_GDB) \
		--output-gpkg $(OUTPUT_DIR)polygon_topology_check.gpkg \
		--report $(OUTPUT_DIR)polygon_topology_check.txt \
		$(if $(MAPSHEET),--mapsheet $(MAPSHEET),)

## coverage-check: Check classification coverage and extract unclassified features
coverage-check:
	@echo "Checking on $(CLASSIFIED_PATH) with $(CONFIG_PATH)"
	python scripts/check_classification_coverage.py \
		$(CLASSIFIED_PATH) \
		$(CONFIG_PATH) \
		--report $(OUTPUT_DIR)unclassified.txt \
		--top-n 500 \
		--counts $(OUTPUT_DIR)feature_counts.xlsx  \
		--output-gpkg $(OUTPUT_DIR)unclassified.gpkg

## domain-check-rc: Check RC1 and RC2 against their own coded domains
domain-check-rc:
	@echo "--- Checking RC2.gdb (self) ---"
	@python scripts/check_domain_compliance.py $(SOURCES_DIR)RC2.gdb \
		--report $(OUTPUT_DIR)domain_check_rc2.txt
	@echo "--- Checking RC1.gdb (self) ---"
	@python scripts/check_domain_compliance.py $(SOURCES_DIR)RC1.gdb \
		--report $(OUTPUT_DIR)domain_check_rc1.txt

## domain-check-final: Check merged_final.gdb against RC2 coded domains
domain-check-final:
	$(call check_file,FINAL_GDB,$(FINAL_GDB))
	@python scripts/check_domain_compliance.py $(FINAL_GDB) \
		--reference $(SOURCES_DIR)RC2.gdb \
		--report $(OUTPUT_DIR)domain_check_final.txt

## domain-check-custom: Check each custom delivery GDB against RC2 coded domains
#  Custom sources are discovered from the mapsheets_sources_only layer (SOURCE_RC != RC1/RC2).
#  Both "Name.gdb" and bare "Name" directory conventions are handled.
domain-check-custom:
	@python -c "\
from pathlib import Path; import geopandas as gpd; \
d = Path('$(SOURCES_DIR)'); \
gdf = gpd.read_file('src/gcover/data/administrative_zones.gpkg', layer='mapsheets_sources_only'); \
srcs = sorted(set(gdf['SOURCE_RC'].dropna()) - {'RC1', 'RC2'}); \
[print(next((str(c) for c in (d/s, d/(s+'.gdb')) if (c/'timestamps').exists()), '')) for s in srcs]" \
	| grep -v '^$$' | while read gdb; do \
		echo "--- $$(basename $$gdb) vs RC2.gdb ---"; \
		python scripts/check_domain_compliance.py "$$gdb" \
			--reference $(SOURCES_DIR)RC2.gdb \
			--report $(OUTPUT_DIR)domain_check_$$(basename $$gdb).txt || true; \
	done

## domain-check: Run all domain compliance checks (RC1, RC2, merged_final, custom sources)
domain-check: domain-check-rc domain-check-final domain-check-custom

## schema-snapshot-final: Extract merged_final.gdb schema to a JSON contract file
.PHONY: schema-snapshot-final
schema-snapshot-final:
	$(call check_file,FINAL_GDB,$(FINAL_GDB))
	@gcover schema snapshot $(FINAL_GDB) \
		--output config/merged_final_schema.json

## filter-check: Check config filter: coverage against each layer's active .lyrx classification
.PHONY: filter-check
filter-check:
	@python scripts/check_filter_coverage.py $(CONFIG_PATH) --styles-dir $(STYLES_DIR)/styles


### Administratives zones


## administrative-zones: Create the adminstratives zones (lots, WU, mapsheets)
administrative-zones:
	@echo "--- Creating administrative zones to $(ADMIN_ZONES_GPKG) ---"
	@python ./scripts/create_administrative_zones.py  \
	   --lots-file $(GCOVER_DATA_DIR)lots.geojson \
       --wu-file $(GCOVER_DATA_DIR)WU.json \
       --mapsheets-file $(GCOVER_DATA_DIR)mapsheets.geojson \
       --sources-file  $(PA_EXCEL_PATH)  \
       --qa-rand-gc $(QA_RAND_PATH) \
       --output $(OUTPUT_DIR)$(ADMIN_ZONES_GPKG) \
       --format gpkg --format geojson --format filegdb --format parquet --format flatgeobuf \
       --overwrite
	@cp -f $(OUTPUT_DIR)$(ADMIN_ZONES_GPKG) $(GCOVER_DATA_DIR)$(ADMIN_ZONES_GPKG)
	@cp -f $(OUTPUT_DIR)administrative_zones.README $(GCOVER_DATA_DIR)adminstrative_zones.README
	@echo "Don't forget to copy to mapserver-geocover/data directory!"

##administrative-zones-metadata:  Show processing_metadata table of the administrative zones GPKG
administrative-zones-metadata:
	@sqlite3 -column -header $(OUTPUT_DIR)$(ADMIN_ZONES_GPKG)  \
		"SELECT role, filename, sha256, mtime, git_hash, generated_at FROM processing_metadata"


### Auxilliary data


## geocover-aux: Create auxiliary grid sur surfaces/unco deposits

$(GEOCOVER_AUX_PATH):
	python scripts/surfaces_auxilliary_points.py --copy-polygons \
		-i $(CLASSIFIED_PATH) -l surfaces \
		-s 80 -b 18 \
		--symbol-size 12 --min-symbol-size 4 \
		--output $(GEOCOVER_AUX_PATH)
	python scripts/surfaces_auxilliary_points.py --copy-polygons \
		-i $(CLASSIFIED_PATH) -l unco_deposits \
		-s 80 -b 18 \
		--symbol-size 12 --min-symbol-size 4 \
		--output $(GEOCOVER_AUX_PATH)

geocover-aux: $(GEOCOVER_AUX_PATH)

# Master target to run everything
## aspect: Add angular aspect to hexagonal grid data
aspect: aspect-gmm combine-aspect

## combine-aspect:  Combine aspect layer for surfaces and unco_deposits
combine-aspect: aspect-gmm
	# Create with first layer, stripping old FIDs
	@ogr2ogr -f GPKG $(GEOCOVER_AUX_PATH) $(GEOCOVER_AUX_PATH) \
		-nln aux_points_aspect \
		-sql "SELECT * FROM surfaces_filtered_aux_points_aspect" \
		-unsetFid -overwrite

	# Append second layer, stripping old FIDs
	@ogr2ogr -f GPKG $(GEOCOVER_AUX_PATH) $(GEOCOVER_AUX_PATH) \
		-nln aux_points_aspect \
		-sql "SELECT * FROM unco_deposits_filtered_aux_points_aspect" \
		-unsetFid -append

## inject-aux-aspect: Copy aux_points_aspect layer into the translated GPKG
inject-aux-aspect: combine-aspect | $(TRANSLATED_PATH)
	@echo "--- Injecting aux_points_aspect into $(TRANSLATED_PATH) ---"
	-ogrinfo $(TRANSLATED_PATH) -sql "DROP TABLE aux_points_aspect" -dialect OGRSQL > /dev/null 2>&1 || true
	@ogr2ogr -f GPKG -update -append $(TRANSLATED_PATH) $(GEOCOVER_AUX_PATH) aux_points_aspect
	@echo "Done."

# Group targets for easier execution
## aspect-simple: Add angular aspect using the simple model
aspect-simple: $(ASPECT_LAYERS:%=aspect-simple-%)
## aspect-gmm: Add angular aspect using the GMM model
aspect-gmm: $(ASPECT_LAYERS:%=aspect-gmm-%)

#### --- SIMPLE MODEL --- ###

aspect-simple-%: geocover-aux
	@echo "Deleting auxiliary points (simple) for $*..."
	-ogrinfo $(GEOCOVER_AUX_PATH) -sql "DROP TABLE $*_aux_points" -dialect OGRSQL > /dev/null 2>&1 || true
	@echo "Assigning aspect (simple) for $*"
	@python scripts/surfaces_assign_aspect.py \
		--polygons-layer $* \
		--output-layer $*_aux_points \
		--join-key UUID \
		$(GEOCOVER_AUX_PATH) $(DEM_ASPECT_PATH) \
		simple
	@ogrinfo $(GEOCOVER_AUX_PATH) -sql "UPDATE gpkg_contents SET description = 'model:simple' WHERE table_name = '$*_aux_points'" > /dev/null

#### --- GMM MODEL --- ###


aspect-gmm-%: geocover-aux
	@echo "Deleting auxiliary points (GMM) for $*..."
	-ogrinfo $(GEOCOVER_AUX_PATH) -sql "DROP TABLE $*_aux_points_aspect" -dialect OGRSQL > /dev/null 2>&1 || true
	@echo "Assigning aspect (GMM) for $*"
	@python scripts/surfaces_assign_aspect.py \
		--polygons-layer $* \
		--points-layer  $(patsubst %_filtered,%,$*)_aux_points \
		--output-layer $*_aux_points_aspect \
		--join-key UUID \
		$(GEOCOVER_AUX_PATH) $(DEM_ASPECT_PATH) \
		gmm --no-flip --max-components 3
	@ogrinfo $(GEOCOVER_AUX_PATH) -sql "UPDATE gpkg_contents SET description = 'model:gmm' WHERE table_name = '$*_aux_points_aspect'" > /dev/null



### Test data

data/extract_%.gpkg:
	python scripts/export_test_data.py "$*"

## test-data: Export test data GeoPackages for CI
test-data: data/extract_bulle.gpkg data/extract_sion.gpkg






## clean-test-data: Delete test data GeoPackages
clean-test-data:
	rm -f data/extract_bulle.gpkg
	rm -f data/extract_sion.gpkg

### Cleanup

## clean-denormalize: Clean denormalized artefacts
clean-denormalize: clean-classify clean-translate
	rm -rf $(DENORMALIZED_PATH)

## clean-translate: Clean translated artefacts
clean-translate:
	rm -rf $(TRANSLATED_PATH)

## clean-swissgeocover2d: Remove swissgeocover2d.gpkg and all sidecar files (.README, .sha256, …)
clean-swissgeocover2d:
	rm -f $(TRANSLATED_PATH) $(TRANSLATED_PATH).* $(TRANSLATED_README)

## clean-classify: Clean classified artefacts
clean-classify: clean-translate
	rm -rf $(CLASSIFIED_PATH)

## clean-merge: Clean merge outputs (master + schema-output GDBs)
clean-merge: clean-denormalize
	rm -rf $(MASTER_GDB)
	rm -rf $(FINAL_GDB)

## clean-master: Alias for clean-merge
clean-master: clean-merge

## clean-all: Remove generated GDB and GeoPackage files
clean-all: clean-merge clean-denormalize clean-swissgeocover2d
	rm -rf $(MASTER_GDB)
	rm -rf $(OUTPUT_DIR)surfaces_aux.gpkg


### Mapfiles
## mapfiles: Generate prod mapfiles
mapfiles:
	gcover --env production publish mapserver    \
		--use-symbol-field  \
		--output-dir $(MAPSERVER_OUTPUT)  \
		--generate-combined \
		--styles-dir $(STYLES_DIR)/styles \
		--pattern-file config/patterns_catalog.yaml  \
		--gml-items label  \
		$(CONFIG_PATH)

# Usage: make diff-styles [PREVIOUS_STYLES_DIR=.../Styles/2026-05-26/styles] [CURRENT_STYLES_DIR=.../Styles/2026-07-02/styles] [ARGS="--only symbology_changed"]
## diff-styles: Compare .lyrx classification/symbology between two style snapshots (defaults: PREVIOUS_STYLES_DIR tracked above, CURRENT_STYLES_DIR = current STYLES_DIR)
CURRENT_STYLES_DIR ?= $(STYLES_DIR)styles
.PHONY: diff-styles
diff-styles:
	$(call check_file,PREVIOUS_STYLES_DIR,$(PREVIOUS_STYLES_DIR))
	$(call check_file,CURRENT_STYLES_DIR,$(CURRENT_STYLES_DIR))
	python scripts/diff_lyrx.py $(PREVIOUS_STYLES_DIR) $(CURRENT_STYLES_DIR) $(ARGS)


### Code
## install-dev:  Install development dependencies
install-dev:
	pip install -e ".[dev]"


## format: ruff format the code
format:
	ruff format src/

## linting: Run linting
lint: format
	ruff check src/ --fix

## test: Run all tests
test:
	pytest tests/ -v

## smoke: Run smoke tests only (fastest) no coverage
smoke:
	pytest tests/test_imports.py tests/test_cli_smoke.py -v --no-cov
## doc: Generate the doc
doc:
	pdoc src/gcover  gcover.config.models gcover.publish.style_config  gcover.publish.esri_classification_extractor  --docformat google

## check: lint and smoke tests
check: lint smoke
	@echo "✅ Basic checks passed!"

## diff-pa: Compare PA_Geocover Excel between current (R17) and previous (R16) release
PREV_RELEASE     ?= R16
PREV_PA_EXCEL    := ${HOME}/DATA/Derivations/delivery/$(PREV_RELEASE)/Excels/GC_Sources_PA.xlsx
.PHONY: diff-pa
diff-pa:
	$(call check_file,PA_EXCEL_PATH,$(PA_EXCEL_PATH))
	$(call check_file,PREV_PA_EXCEL,$(PREV_PA_EXCEL))
	python scripts/diff_xlsx.py $(PA_EXCEL_PATH) $(PREV_PA_EXCEL) --cols MSH_MAP_TITLE --cols BKP
