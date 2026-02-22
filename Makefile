# --- Variables ---
DELIVERY_DIR := ${HOME}/DATA/Derivations/delivery/R16/
OUTPUT_DIR   := ${HOME}/DATA/Derivations/output/test/
STYLES_DIR   := ${HOME}/DATA/Derivations/delivery/R16/styles/2026-02-19/
TRANSLATION_CSV := ${HOME}/code/github.com/lg-geology-data-model/exports/2026-02-12/geolcodes_translated.csv

# File Paths
MASTER_GDB        := $(OUTPUT_DIR)master_R16.gdb
DENORMALIZED_GPKG := R16_master_denormalized.gpkg
TRANSLATED_GPKG   := R16_master_classified_translated.gpkg
DENORMALIZED_PATH := $(OUTPUT_DIR)$(DENORMALIZED_GPKG)
CLASSIFIED_PATH   := $(OUTPUT_DIR)R16_master_denormalized_classified.gpkg
TRANSLATED_PATH   := $(OUTPUT_DIR)$(TRANSLATED_GPKG)
FULL_GDB          := $(DELIVERY_DIR)RC2.gdb
SURFACES_AUX_PATH := $(OUTPUT_DIR)surfaces_aux.gpkg

# Layers for denormalization
LAYERS := fossils exploit_polygons exploit_points linear_objects point_objects bedrock surfaces unco_deposits
TABLES_TO_IMPORT := GC_GEOL_MAPPING_UNIT GC_LITSTRAT_FORMATION_BANK GC_CHRONO \
                    GC_EX_GEO_PLG_EXP_UNIT_GC_GMU GC_EX_GEO_PNT_EXP_UNIT_GC_GMU \
                    GC_FOSS_SYSTEM_GC_SYSTEM \
                    GC_UN_DEP_CHARACT_GC_CHARCAT GC_UN_DEP_COMPOSIT_GC_COMPOS GC_UN_DEP_MAT_TYPE_GC_LITHO


# --- Targets ---

.DEFAULT_GOAL := help

## help: Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' |  sed -e 's/^/ /'
	@echo ""
	@echo "Master GDB:			$(MASTER_GDB)"
	@echo "Styles dir:			$(STYLES_DIR)"
	@echo "DENORMALIZED GPKG:	$(DENORMALIZED_PATH)"
	@echo "Classified:			$(CLASSIFIED_PATH)"
	@echo "Translation:			$(TRANSLATION_CSV)"
	@echo "Translated:			$(TRANSLATED_PATH)"
	@echo "Surfaces auxilliary:	$(SURFACES_AUX_PATH)"
	@echo "Output dir:			$(OUTPUT_DIR)"


## all: Run the entire workflow (Merge -> Import -> Denormalize -> Symbolize)
all: merge $(CLASSIFIED_PATH)

# 1. Merge sources and run diagnosis
$(MASTER_GDB)/timestamps: $(DELIVERY_DIR)RC1.gdb $(DELIVERY_DIR)RC2.gdb
	@echo "--- Merging Sources ---"
	gcover publish merge \
		--rc1 $(DELIVERY_DIR)RC1.gdb \
		--rc2 $(DELIVERY_DIR)RC2.gdb \
		--custom-sources-dir $(DELIVERY_DIR) \
		--force-2d --output $(MASTER_GDB) \
		--no-clip-to-swiss-border
.PHONY: merge-diagnostic
## merge-diagnostic: Merge diagnostic
merge-diagnostic:
	@echo "--- Running Diagnosis ---"
	python scripts/diagnose_merge.py $(DELIVERY_DIR)RC1.gdb $(DELIVERY_DIR)RC2.gdb data/administrative_zones.gpkg

## merge: Only perform the gcover merge and diagnosis
merge: $(MASTER_GDB)/timestamps

# 2. Add missing tables and Denormalize
$(DENORMALIZED_PATH): $(MASTER_GDB)/timestamps
	@echo "--- Importing missing tables via ogr2ogr ---"
	@for table in $(TABLES_TO_IMPORT); do \
		ogr2ogr -f "OpenFileGDB" -update -overwrite $(MASTER_GDB) $(FULL_GDB) $$table; \
	done

	@echo "--- Running Denormalization loop ---"
	@for layer in $(LAYERS); do \
		scripts/denormalize_geocover.py --remove-metadata  -o $(DENORMALIZED_PATH) --cd-gdb-path $(DELIVERY_DIR)RC2.gdb  --tables $$layer $(MASTER_GDB) ; \
	done

$(TRANSLATED_PATH): $(CLASSIFIED_PATH)
	@python ./scripts/translate_gpkg.py -t $(TRANSLATION_CSV) \
		--lowercase-columns \
		 -o $(TRANSLATED_PATH)  --langs de,fr  $(CLASSIFIED_PATH)

$(CLASSIFIED_PATH): $(DENORMALIZED_PATH)
	@echo "--- Applying Style Configuration ---"
	@gcover --env sandisk publish apply-config --styles-dir $(STYLES_DIR) \
		$(DENORMALIZED_PATH) config/esri_classifier_denormalized_geocover.yaml

## denormalize: Only run the table import and denormalization (requires master GDB)
denormalize: $(DENORMALIZED_PATH)

## translate: Add human-readable values for geolcodes
translate: $(TRANSLATED_PATH)

## classify: Apply classification from .lyrx to denormalized data
classify: $(CLASSIFIED_PATH)


## surfaces-aux: Create auxilliary grid sur surfaces/unco deposits
surfaces-aux:
	python scripts/surfaces_auxilliary_points.py --copy-polygons -i $(CLASSIFIED_PATH) -l surfaces -s 80 -b 25 --output $(SURFACES_AUX_PATH)
	python scripts/surfaces_auxilliary_points.py --copy-polygons -i $(CLASSIFIED_PATH) -l unco_deposits -s 80 -b 25 --output $(SURFACES_AUX_PATH)

## clean-translate: Clean denormalized artefacts
clean-denormalize: clean-translate
	rm -rf $(DENORMALIZED_PATH)

## clean-translate: Clean translated artefacts
clean-translate: clean-classify
	rm -rf $(TRANSLATED_PATH)

## clean-classify: Clean classified artefacts
clean-classify:
	rm -rf $(CLASSIFIED_PATH)

## clean-all: Remove generated GDB and GeoPackage files
clean-all: clean-denormalize
	rm -rf $(MASTER_GDB)
	rm -rf $(OUTPUT_DIR)surfaces_aux.gpkg


# Makefile for easy test execution
.PHONY: test lint format smoke install-dev doc
## install-dev:  Install development dependencies
install-dev:
	pip install -e ".[dev]"

## smoke: Run smoke tests only (fastest) no coverage
smoke:
	pytest tests/test_imports.py tests/test_cli_smoke.py -v --no-cov

## format: ruff format the code
format:
	ruff format src/

## linting: Run linting
lint: format
	ruff check src/ --fix

## test: Run all tests
test:
	pytest tests/ -v
## doc: Generate the doc
doc:
	pdoc src/gcover  gcover.config.models gcover.publish.style_config  gcover.publish.esri_classification_extractor  --docformat google

## check: lint and smoke tests
check: lint smoke
	@echo "✅ Basic checks passed!"
