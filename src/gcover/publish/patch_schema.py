"""
patch_schema.py — clone an authoritative ESRI FileGDB schema and inject
merged spatial data into it, preserving coded domains, relationship classes,
and the GC_ROCK_BODIES feature dataset.

Strategy
--------
  1. shutil.copytree  — byte-perfect clone: domains, relationship classes,
                        feature dataset and topology layers all survive.
  2. OGR update mode  — delete all features from each target layer.
  3. gdal.VectorTranslate (accessMode='append') — bulk-write from the
                        geopandas-merged GDB; much faster than a Python loop.
  4. CreateField      — add extra fields present in merged_gdb but absent
                        from the schema clone (e.g. _MERGE_SOURCE, erl_link).

Tested with GDAL 3.10.3 and GDAL 3.13.0.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

from osgeo import gdal, ogr

gdal.UseExceptions()

SPATIAL_LAYERS = [
    "GC_BEDROCK",
    "GC_UNCO_DESPOSIT",
    "GC_SURFACES",
    "GC_LINEAR_OBJECTS",
    "GC_POINT_OBJECTS",
    "GC_FOSSILS",
    "GC_EXPLOIT_GEOMAT_PLG",
    "GC_EXPLOIT_GEOMAT_PT",
]

TABLE_LAYERS = [
    "GC_GEOL_MAPPING_UNIT",
    "GC_GEOL_MAPPING_UNIT_ATT",
    "GC_SYSTEM",
    "GC_COMPOSIT",
    "GC_ADMIXTURE",
    "GC_CHARCAT",
    "GC_LITHO",
    "GC_CHRONO",
    "GC_TECTO",
    "GC_LITHO_UNCO",
    "GC_LITSTRAT_UNCO",
    "GC_CORRELATION",
    "GC_LITSTRAT_FORMATION_BANK",
    "GC_EX_GEO_PLG_EXP_UNIT_GC_GMU",
    "GC_EX_GEO_PNT_EXP_UNIT_GC_GMU",
    "GC_FOSS_SYSTEM_GC_SYSTEM",
    "GC_UN_DEP_CHARACT_GC_CHARCAT",
    "GC_UN_DEP_COMPOSIT_GC_COMPOS",
    "GC_UN_DEP_MAT_TYPE_GC_LITHO",
]

# Topology layers dropped from the publication GDB.
# GC_UN_DEP_ADMIXTUR_GC_ADMIXT is intentionally absent: kept from the clone
# because geopandas drops it entirely.
_DROP_LAYERS: frozenset[str] = frozenset({"BEDROCK_TOPOLOGY", "GC_ROCK_BODIES_TOPO"})
_DROP_PREFIXES: tuple[str, ...] = ("T_1_",)

# Fields present in merged GDB but absent from the authoritative schema GDB.
_EXTRA_FIELDS: list[tuple[str, int]] = [
    ("_MERGE_SOURCE", ogr.OFTString),
    ("erl_link",      ogr.OFTString),
    ("ber_link",      ogr.OFTString),
]


def _truncate(ds: ogr.DataSource, name: str) -> int:
    lyr = ds.GetLayerByName(name)
    if lyr is None:
        return -1
    fids = [f.GetFID() for f in lyr]
    lyr.ResetReading()
    for fid in fids:
        lyr.DeleteFeature(fid)
    return len(fids)


def _append(output_gdb: str, merged_gdb: str, name: str) -> int:
    """Bulk-append one layer from merged_gdb into output_gdb.

    explodeCollections splits MultiPoint back to Point so the geometry
    type matches the schema-clone target (geopandas promotes to multi).
    """
    gdal.VectorTranslate(
        output_gdb,
        merged_gdb,
        options=gdal.VectorTranslateOptions(
            layers=[name],
            accessMode="append",
            explodeCollections=True,
        ),
    )
    ds = ogr.Open(output_gdb, 0)
    n = ds.GetLayerByName(name).GetFeatureCount()
    ds = None
    return n


def _patch(
    output_gdb: str,
    merged_gdb: str,
    layers: list[str],
    section: str,
    log: Callable[[str], None],
) -> list[str]:
    errors: list[str] = []
    log(f"\n--- {section} ---")
    ds = ogr.Open(output_gdb, 1)

    for name in layers:
        src_ds = ogr.Open(merged_gdb, 0)
        src_lyr = src_ds.GetLayerByName(name) if src_ds else None
        if src_lyr is None:
            log(f"  SKIP  {name}  (not in merged GDB)")
            src_ds = None
            continue
        src_ds = None

        t = time.time()
        deleted = _truncate(ds, name)
        if deleted == -1:
            log(f"  WARN  {name}  (not in clone)")
            continue

        ds = None
        try:
            n = _append(output_gdb, merged_gdb, name)
            log(f"  OK    {name}: {n:,}  ({time.time()-t:.1f}s)")
        except Exception as exc:
            msg = f"{name}: {exc}"
            log(f"  ERR   {msg}")
            errors.append(msg)

        ds = ogr.Open(output_gdb, 1)

    ds = None
    return errors


def patch_schema_gdb(
    schema_gdb: Path,
    merged_gdb: Path,
    output_gdb: Path,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Clone schema_gdb, then replace its data with content from merged_gdb.

    Parameters
    ----------
    schema_gdb:
        Authoritative FileGDB whose schema (domains, relationships, feature
        dataset) should be preserved — typically RC2.gdb.
    merged_gdb:
        Flat merged FileGDB produced by the geopandas merger.
    output_gdb:
        Destination path.  Deleted and recreated if it already exists.
    log:
        Callable used for progress reporting (default: print).

    Returns
    -------
    list[str]
        Error messages for layers that failed; empty on full success.
    """
    schema_gdb = str(schema_gdb)
    merged_gdb = str(merged_gdb)
    output_gdb = str(output_gdb)

    # Step 1: byte-perfect clone
    import os
    if os.path.exists(output_gdb):
        shutil.rmtree(output_gdb)

    log(f"Cloning {schema_gdb} …")
    t0 = time.time()
    shutil.copytree(schema_gdb, output_gdb)
    log(f"  done in {time.time()-t0:.1f}s")

    # Drop topology layers not needed in publication
    ds = ogr.Open(output_gdb, 1)
    dropped: list[str] = []
    for i in range(ds.GetLayerCount() - 1, -1, -1):
        name = ds.GetLayerByIndex(i).GetName()
        if name in _DROP_LAYERS or any(name.startswith(p) for p in _DROP_PREFIXES):
            ds.DeleteLayer(i)
            dropped.append(name)
    ds = None
    if dropped:
        log(f"  dropped: {', '.join(dropped)}")

    ds = ogr.Open(output_gdb, 0)
    log(f"  layers: {ds.GetLayerCount()},  domains: {len(ds.GetFieldDomainNames() or [])}")
    ds = None

    # Step 1b: add extra fields absent from schema GDB
    ds = ogr.Open(output_gdb, 1)
    for layer_name in SPATIAL_LAYERS:
        lyr = ds.GetLayerByName(layer_name)
        if lyr is None:
            continue
        defn = lyr.GetLayerDefn()
        existing = {defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())}
        for fname, ftype in _EXTRA_FIELDS:
            if fname not in existing:
                lyr.CreateField(ogr.FieldDefn(fname, ftype))
                log(f"  added field '{fname}' to {layer_name}")
    ds = None

    # Steps 2 & 3: patch spatial layers and tables
    errors: list[str] = []
    errors += _patch(output_gdb, merged_gdb, SPATIAL_LAYERS, "Spatial layers (GC_ROCK_BODIES)", log)
    errors += _patch(output_gdb, merged_gdb, TABLE_LAYERS,   "Tables & relationship tables", log)

    ds = ogr.Open(output_gdb, 0)
    log(f"\nDone → {output_gdb}")
    log(f"  layers : {ds.GetLayerCount()}")
    log(f"  domains: {len(ds.GetFieldDomainNames() or [])}")
    ds = None

    return errors
