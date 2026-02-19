# --- Variables ---
DELIVERY_DIR := ${HOME}/DATA/Derivations/delivery/R16/
OUTPUT_DIR   := ${HOME}/DATA/Derivations/output/test/
STYLES_DIR   := ${HOME}/DATA/Derivations/delivery/R16/styles/2026-01-28/

# File Paths
MASTER_GDB        := $(OUTPUT_DIR)master_R16.gdb
DENORMALIZED_GPKG := R16_master_denormalized.gpkg
DENORMALIZED_PATH := $(OUTPUT_DIR)$(DENORMALIZED_GPKG)
CLASSIFIED_PATH := $(OUTPUT_DIR)R16_master_denormalized.classified.gpkg
FULL_GDB          := $(DELIVERY_DIR)RC2.gdb

# Layers for denormalization
LAYERS := fossils exploit_polygons exploit_points linear_objects point_objects bedrock surfaces unco_deposits
TABLES_TO_IMPORT := GC_GEOL_MAPPING_UNIT GC_LITSTRAT_FORMATION_BANK GC_CHRONO \
                    GC_EX_GEO_PLG_EXP_UNIT_GC_GMU GC_EX_GEO_PNT_EXP_UNIT_GC_GMU \
                    GC_FOSS_SYSTEM_GC_SYSTEM

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
	@echo "Output dir:			$(OUTPUT_DIR)"


## all: Run the entire workflow (Merge -> Import -> Denormalize -> Symbolize)
all: merge $(CLASSIFIED_PATH)

# 1. Merge sources and run diagnosis
$(MASTER_GDB): $(DELIVERY_DIR)RC1.gdb $(DELIVERY_DIR)RC2.gdb
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
merge: $(MASTER_GDB)

# 2. Add missing tables and Denormalize
$(DENORMALIZED_PATH): $(MASTER_GDB)/timestamps
	@echo "--- Importing missing tables via ogr2ogr ---"
	@for table in $(TABLES_TO_IMPORT); do \
		ogr2ogr -f "OpenFileGDB" -update $(MASTER_GDB) $(FULL_GDB) $$table; \
	done

	@echo "--- Running Denormalization loop ---"
	@for layer in $(LAYERS); do \
		scripts/denormalize_geocover.py --remove-metadata -o $(DENORMALIZED_PATH) --tables $$layer $(MASTER_GDB); \
	done

$(CLASSIFIED_PATH): $(DENORMALIZED_PATH)
	@echo "--- Applying Style Configuration ---"
	gcover --env sandisk publish apply-config --styles-dir $(STYLES_DIR) \
		$(DENORMALIZED_PATH) config/esri_classifier_denormalized_geocover.yaml

## denormalize: Only run the table import and denormalization (requires master GDB)
denormalize: $(DENORMALIZED_PATH)

## classify: Apply classification from .lyrx to denormalized data
classify: $(CLASSIFIED_PATH)

## clean: Remove generated GDB and GeoPackage files
clean:
	rm -rf $(MASTER_GDB)
	rm -f $(DENORMALIZED_PATH)


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
