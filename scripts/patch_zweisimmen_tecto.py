"""
Prepare Zweisimmen.gdb from BKP_2016.gdb:
  1. shutil.copytree  — byte-perfect clone preserving domains, relationship
                        classes and the GC_ROCK_BODIES feature dataset.
  2. UUID join        — overwrite TECTO in GC_BEDROCK from gcover_master_de.gdb,
                        which carries the current GC_TECTO_CD vocabulary.

Spatial trimming (removing features outside the Zweisimmen mapsheet) is done
afterwards in ArcGIS Pro.

Run with:  conda run -n GEO12 python scripts/patch_zweisimmen_tecto.py
"""
import shutil
import time
from pathlib import Path

from osgeo import gdal, ogr

gdal.UseExceptions()

BKP_GDB    = Path("/home/marco/DATA/Derivations/delivery/R17/BKP_2016.gdb")
MASTER_GDB = Path("/home/marco/DATA/Derivations/delivery/R17/gcover_master_de.gdb")
OUT_GDB    = Path("/home/marco/DATA/Derivations/delivery/R17/Zweisimmen.gdb")

# ── Step 1: Clone ─────────────────────────────────────────────────────────────
if OUT_GDB.exists():
    shutil.rmtree(OUT_GDB)

print(f"Cloning {BKP_GDB.name} → {OUT_GDB.name} …")
t0 = time.time()
shutil.copytree(BKP_GDB, OUT_GDB)
print(f"  done in {time.time() - t0:.1f}s")

ds = ogr.Open(str(OUT_GDB), 0)
print(f"  layers: {ds.GetLayerCount()},  domains: {len(ds.GetFieldDomainNames() or [])}")
ds = None

# ── Step 2: Build UUID → TECTO mapping from master ────────────────────────────
print(f"\nReading TECTO mapping from {MASTER_GDB.name} …")
t0 = time.time()
uuid_to_tecto: dict[str, int] = {}

ds = ogr.Open(str(MASTER_GDB), 0)
lyr = ds.GetLayerByName("GC_BEDROCK")
for feat in lyr:
    uuid  = feat.GetField("UUID")
    tecto = feat.GetField("TECTO")
    if uuid and tecto is not None:
        uuid_to_tecto[uuid] = tecto
ds = None

print(f"  {len(uuid_to_tecto):,} records  ({time.time() - t0:.1f}s)")

# ── Step 3: Update TECTO in the cloned GDB ────────────────────────────────────
print(f"\nPatching TECTO in {OUT_GDB.name}/GC_BEDROCK …")
t0 = time.time()

ds = ogr.Open(str(OUT_GDB), 1)
lyr = ds.GetLayerByName("GC_BEDROCK")

updated  = 0
no_match = []

for feat in lyr:
    uuid = feat.GetField("UUID")
    if uuid in uuid_to_tecto:
        feat.SetField("TECTO", uuid_to_tecto[uuid])
        lyr.SetFeature(feat)
        updated += 1
    else:
        no_match.append(uuid)

ds = None  # flush

elapsed = time.time() - t0
print(f"  updated : {updated:,}  ({elapsed:.1f}s)")
print(f"  no match: {len(no_match):,}  (TECTO left as-is from BKP_2016)")

if no_match:
    print("  first 10 unmatched UUIDs:")
    for u in no_match[:10]:
        print(f"    {u}")

print(f"\nDone → {OUT_GDB}")
