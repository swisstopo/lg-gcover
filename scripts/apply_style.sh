#!/bin/bash

# Path to your GPKG file
DENORMALIZED_SOURCE="/home/marco/DATA/mapfiles/sources/RC1_20250922_denormalized.gpkg"
STYLES_DIR="/home/marco/DATA/Derivations/output/R14/"
CONFIG_FILE="config/esri_classifier_denormalized_geocover.yaml"
BBOX="0,0,2590000,1270000"

# Extract layer names using ogrinfo
layers=$(ogrinfo "$DENORMALIZED_SOURCE" | grep -E '^[0-9]+:' | cut -d ':' -f2 | awk '{print $1}')

# Loop over each layer
for layer in $layers; do
  echo "🛠️ Processing layer: $layer"
  gcover publish apply-config \
    --layer "$layer" \
    --bbox "$BBOX" \
    --styles-dir "$STYLES_DIR" \
    "$DENORMALIZED_SOURCE" \
    "$CONFIG_FILE"
done
