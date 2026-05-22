#!/usr/bin/env python3
"""
inject_strati_link.py — one-shot script

Reads the strati_link column from the translated GPKG bedrock layer and
writes it back to GC_BEDROCK in merged_master.gdb, joining on UUID.
"""
import sys
from pathlib import Path

import fiona
from osgeo import ogr

TRANSLATED_GPKG = Path.home() / "DATA/Derivations/output/R17/denormalized_classified_translated.gpkg"
FINAL_GDB       = Path.home() / "DATA/Derivations/output/R17/merged_final.gdb"
LAYER           = "GC_BEDROCK"
GPKG_UUID       = "uuid"
GDB_UUID        = "UUID"
FIELD           = "strati_link"

# Build lookup: uuid → strati_link
print(f"Reading {FIELD} from {TRANSLATED_GPKG.name} …")
lookup: dict[str, str] = {}
with fiona.open(TRANSLATED_GPKG, layer="bedrock") as src:
    for feat in src:
        uid = feat["properties"].get(GPKG_UUID)
        val = feat["properties"].get(FIELD)
        if uid and val:
            lookup[uid] = val
print(f"  {len(lookup):,} non-null strati_link values")

# Open GDB layer in write mode
ds = ogr.Open(str(FINAL_GDB), 1)
lyr = ds.GetLayerByName(LAYER)
if lyr is None:
    print(f"ERROR: layer {LAYER} not found in {FINAL_GDB}", file=sys.stderr)
    sys.exit(1)

# Add field if missing
defn = lyr.GetLayerDefn()
if defn.GetFieldIndex(FIELD) == -1:
    lyr.CreateField(ogr.FieldDefn(FIELD, ogr.OFTString))
    print(f"  created field '{FIELD}'")

# Write back
updated = skipped = 0
for feat in lyr:
    uid = feat.GetField(GDB_UUID)
    val = lookup.get(uid)
    if val:
        feat.SetField(FIELD, val)
        lyr.SetFeature(feat)
        updated += 1
    else:
        skipped += 1

ds = None
print(f"  updated {updated:,} features, {skipped:,} without strati_link")
print("Done.")
