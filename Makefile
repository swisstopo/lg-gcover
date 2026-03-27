

BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

# --- Variables ---
RELEASE      := R17
DELIVERY_DIR := ${HOME}/DATA/Derivations/delivery/$(RELEASE)/
OUTPUT_DIR   := ${HOME}/DATA/Derivations/output/$(RELEASE)/
STYLES_DIR   := ${HOME}/DATA/Derivations/delivery/$(RELEASE)/styles/2026-03-23/
TRANSLATION_CSV := ${HOME}/code/github.com/lg-geology-data-model/exports/2026-02-12/geolcodes_translated.csv
STRATI_LINK_PATH := ${HOME}/DATA/Derivations/delivery/R16/Excels/2026a_Update_stratiLINK.xlsx
GCOVER_DATA_DIR :=  src/gcover/data/

ASPECT_LAYERS := surfaces_filtered unco_deposits_filtered
# Ensure no trailing whitespace
ASPECT_LAYERS := $(strip $(ASPECT_LAYERS))



# File Paths
MASTER_GDB        := $(OUTPUT_DIR)merged_master.gdb
DENORMALIZED_GPKG := denormalized.gpkg
DENORMALIZED_PATH := $(OUTPUT_DIR)$(DENORMALIZED_GPKG)  # name auto
CLASSIFIED_GPKG	  := denormalized_classified.gpkg
CLASSIFIED_PATH   := $(OUTPUT_DIR)$(CLASSIFIED_GPKG)
TRANSLATED_GPKG   := denormalized_classified_translated.gpkg
TRANSLATED_PATH   := $(OUTPUT_DIR)$(TRANSLATED_GPKG)
FULL_GDB_PATH     := $(DELIVERY_DIR)RC1.gdb     # TODO Val Bregaglia missing GMU_ATT in RC2. RC1 OK
GEOCOVER_AUX_PATH := $(OUTPUT_DIR)geocover_aux.gpkg
ADMIN_ZONES_GPKG  := administrative_zones.gpkg
MAPSERVER_OUTPUT  := mapserver_$(BRANCH)
DEM_ASPECT_PATH   ?= $(DELIVERY_DIR)swissALTI3DRegio_aspect_50m.tif
MAPSERVER_OUTPUT  ?= mapserver_$(BRANCH)
PA_EXCEL_PATH     ?= $(DELIVERY_DIR)Excels/GC_Sources_PA.xlsx
CONFIG_PATH       ?=config/esri_classifier_denormalized_geocover.yaml

# Layers for denormalization
LAYERS := fossils exploit_polygons exploit_points linear_objects point_objects bedrock surfaces unco_deposits
TABLES_TO_IMPORT := GC_GEOL_MAPPING_UNIT GC_GEOL_MAPPING_UNIT_ATT GC_LITSTRAT_FORMATION_BANK GC_CHRONO \
                    GC_EX_GEO_PLG_EXP_UNIT_GC_GMU GC_EX_GEO_PNT_EXP_UNIT_GC_GMU \
                    GC_FOSS_SYSTEM_GC_SYSTEM \
                    GC_UN_DEP_CHARACT_GC_CHARCAT GC_UN_DEP_COMPOSIT_GC_COMPOS GC_UN_DEP_MAT_TYPE_GC_LITHO


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
	@echo "  Delivery GDBs:        $(DELIVERY_DIR)"
	$(call check_file,STYLES_DIR,$(STYLES_DIR))
	$(call check_file,TRANSLATION_CSV,$(TRANSLATION_CSV))
	$(call check_file,STRATI_LINK_PATH,$(STRATI_LINK_PATH))
	$(call check_file,PA_EXCEL_PATH,$(PA_EXCEL_PATH))
	$(call check_file,DEM_ASPECT,$(DEM_ASPECT_PATH))
	@echo ""
	@echo "$(YELLOW)Output$(RESET)"
	@echo "  Output dir:           $(OUTPUT_DIR)"
	$(call check_file,MASTER_GDB,$(MASTER_GDB))
	$(call check_file,DENORMALIZED_PATH,$(DENORMALIZED_PATH))
	$(call check_file,CLASSIFIED_PATH,$(CLASSIFIED_PATH))
	$(call check_file,CLASSIFIED_PATH,$(CLASSIFIED_PATH))
	$(call check_file,TRANSLATION_CSV,$(TRANSLATION_CSV))
	$(call check_file,TRANSLATED_PATH,$(TRANSLATED_PATH))
	$(call check_file,GEOCOVER_AUX_PATH,$(GEOCOVER_AUX_PATH))
	$(call check_file,CONFIG_PATH,$(CONFIG_PATH))

	@echo ""
	@echo "$(YELLOW)Mapserver$(RESET)"
	@echo "  Mapserver dir:        $(MAPSERVER_OUTPUT)"



.PHONY: merge-diagnostic translate classify denormalize test lint format smoke install-dev doc
### Data

## download:  Download RC1/RC2  backup
download:
	@gcover --env production --verbose  gdb download-couple --type backup --output-dir $(DELIVERY_DIR)  \
	 --unzip --no-keep-zip


## administrative-zones: Create the adminstratives zones (lots, WU, mapsheets)
administrative-zones:
	@echo "--- Creating administrative zones to $(ADMIN_ZONES_GPKG) ---"
	@python ./scripts/create_administrative_zones.py  \
	   --lots-file $(GCOVER_DATA_DIR)lots.geojson \
       --wu-file $(GCOVER_DATA_DIR)WU.json \
       --mapsheets-file $(GCOVER_DATA_DIR)mapsheets.geojson \
       --sources-file  $(PA_EXCEL_PATH)  \
       --output $(OUTPUT_DIR)$(ADMIN_ZONES_GPKG) \
       --overwrite
	@cp -i $(PA_EXCEL_PATH)   $(GCOVER_DATA_DIR)GC_Sources_PA.xlsx
	@cp -i $(OUTPUT_DIR)$(ADMIN_ZONES_GPKG) $(GCOVER_DATA_DIR)$(ADMIN_ZONES_GPKG)
	@cp -i $(OUTPUT_DIR)administrative_zones.README $(GCOVER_DATA_DIR)adminstrative_zones.README
	@echo "Don't forget to copy to mapserver-geocover/data directory!"


## all: Run the entire workflow (Merge -> Import -> Denormalize -> Symbolize)
all: merge $(CLASSIFIED_PATH) $(TRANSLATED_PATH)

## merge: Only perform the gcover merge and diagnosis
merge: $(MASTER_GDB)/timestamps

# 1. Merge sources and run diagnosis
$(MASTER_GDB)/timestamps: $(DELIVERY_DIR)RC1.gdb $(DELIVERY_DIR)RC2.gdb
	@echo "--- Merging Sources ---"
	gcover publish merge \
		--rc1 $(DELIVERY_DIR)RC1.gdb \
		--rc2 $(DELIVERY_DIR)RC2.gdb \
		--custom-sources-dir $(DELIVERY_DIR) \
		--force-2d --output $(MASTER_GDB) \
		--no-clip-to-swiss-border \
		--enrich-mapsheet-links

## merge-diagnostic: Merge diagnostic
merge-diagnostic:
	@echo "--- Running Diagnosis ---"
	python scripts/diagnose_merge.py $(DELIVERY_DIR)RC1.gdb $(DELIVERY_DIR)RC2.gdb data/administrative_zones.gpkg



# 2. Add missing tables and Denormalize
$(DENORMALIZED_PATH): $(MASTER_GDB)/timestamps
	@echo "--- Importing missing tables via ogr2ogr ---"
	@for table in $(TABLES_TO_IMPORT); do \
		ogr2ogr -f "OpenFileGDB" -update -overwrite $(MASTER_GDB) $(FULL_GDB_PATH) $$table; \
	done

	@echo "--- Running Denormalization loop ---"
	@for layer in $(LAYERS); do \
		scripts/denormalize_geocover.py --remove-metadata  -o $(DENORMALIZED_PATH) --cd-gdb-path $(FULL_GDB_PATH)  --tables $$layer $(MASTER_GDB) ; \
	done

$(TRANSLATED_PATH): $(CLASSIFIED_PATH)
	@echo "Saving to $(TRANSLATED_PATH)"
	python ./scripts/translate_gpkg.py -t $(TRANSLATION_CSV) \
		--strati-links $(STRATI_LINK_PATH) \
		--config $(CONFIG_PATH)  \
		--lowercase-columns --output $(TRANSLATED_PATH)  --langs de,fr  $(CLASSIFIED_PATH)

$(CLASSIFIED_PATH): $(DENORMALIZED_PATH)
	@echo "--- Applying Style Configuration ---"
	@gcover --env sandisk publish apply-config --styles-dir $(STYLES_DIR) \
		$(DENORMALIZED_PATH) $(CONFIG_PATH)

## denormalize: Only run the table import and denormalization (requires master GDB)
denormalize: $(DENORMALIZED_PATH)

## classify: Apply classification from .lyrx to denormalized data
classify: $(CLASSIFIED_PATH)

## translate: Add human-readable values for geolcodes
translate: $(TRANSLATED_PATH)

## geocover-aux: Create auxiliary grid sur surfaces/unco deposits

$(GEOCOVER_AUX_PATH):
	python scripts/surfaces_auxilliary_points.py --copy-polygons -i $(CLASSIFIED_PATH) -l surfaces -s 80 -b 25 --output $(GEOCOVER_AUX_PATH)
	python scripts/surfaces_auxilliary_points.py --copy-polygons -i $(CLASSIFIED_PATH) -l unco_deposits -s 80 -b 25 --output $(GEOCOVER_AUX_PATH)

.PHONY: geocover-aux combine-aspect
geocover-aux: $(GEOCOVER_AUX_PATH)
.PHONY: aspect aspect-simple aspect-gmm

# Master target to run everything
## aspect: Add angular aspect to hexagonal grid data
aspect: aspect-gmm

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

## geocover-aux: Create auxiliary grid sur surfaces/unco deposits
aspect-gmm-%: geocover-aux
	@echo "Deleting auxiliary points (GMM) for $*..."
	-ogrinfo $(GEOCOVER_AUX_PATH) -sql "DROP TABLE $*_aux_points_aspect" -dialect OGRSQL > /dev/null 2>&1 || true
	@echo "Assigning aspect (GMM) for $*"
	@$(eval LAYERNAME_NAME := $(patsubst %_filtered,%,$*))
	@python scripts/surfaces_assign_aspect.py \
		--polygons-layer $* \
		--points-layer  $(LAYERNAME_NAME)_aux_points \
		--output-layer $*_aux_points_aspect \
		--join-key UUID \
		$(GEOCOVER_AUX_PATH) $(DEM_ASPECT_PATH) \
		gmm --no-flip --max-components 3
	@ogrinfo $(GEOCOVER_AUX_PATH) -sql "UPDATE gpkg_contents SET description = 'model:gmm' WHERE table_name = '$*_aux_points_aspect'" > /dev/null






.PHONY: mapfiles


## clean-denormalize: Clean denormalized artefacts
clean-denormalize: clean-classify clean-translate
	rm -rf $(DENORMALIZED_PATH)

## clean-translate: Clean translated artefacts
clean-translate:
	rm -rf $(TRANSLATED_PATH)

## clean-classify: Clean classified artefacts
clean-classify: clean-translate
	rm -rf $(CLASSIFIED_PATH)

## clean-master:  Clean master GDB
clean-master:  clean-denormalize
	rm -rf $(MASTER_GDB)

## clean-all: Remove generated GDB and GeoPackage files
clean-all: clean-denormalize
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
