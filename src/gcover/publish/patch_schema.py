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
  5. Optional strati_link injection — Excel join via GC_GEOL_MAPPING_UNIT_ATT.

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
    # Skip geometry parsing — we only need FIDs, and the schema-clone source
    # (RC2.gdb) may contain unclosed rings that OGR raises as RuntimeError.
    lyr.SetIgnoredFields(["OGR_GEOMETRY"])
    fids = [f.GetFID() for f in lyr]
    lyr.SetIgnoredFields([])
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


def _inject_strati_link(
    ds: ogr.DataSource,
    xlsx_path: Path,
    log: Callable[[str], None],
) -> None:
    """Populate GC_BEDROCK.strati_link via Excel → GMU_CODE lookup chain.

    Join chain:
        GC_BEDROCK.GEOL_MAPPING_UNIT_ATT_UUID
        → GC_GEOL_MAPPING_UNIT_ATT.UUID → .GEOL_MAPPING_UNIT (GMU_CODE)
        → Excel GeolCode_GMU → stratiLINK
    """
    import pandas as pd

    att_lyr = ds.GetLayerByName("GC_GEOL_MAPPING_UNIT_ATT")
    if att_lyr is None:
        log("  WARNING: GC_GEOL_MAPPING_UNIT_ATT not found — strati_link skipped")
        return

    att_uuid_to_gmu: dict[str, int] = {}
    for feat in att_lyr:
        uid = feat.GetField("UUID")
        gmu = feat.GetField("GEOL_MAPPING_UNIT")
        if uid and gmu is not None:
            att_uuid_to_gmu[uid] = int(gmu)
    att_lyr.ResetReading()

    strati_df = pd.read_excel(
        xlsx_path,
        usecols=["GeolCode_GMU", "stratiLINK"],
        dtype={"GeolCode_GMU": "Int64"},
    ).dropna(subset=["GeolCode_GMU"])
    gmu_to_strati: dict[int, str] = {
        int(row["GeolCode_GMU"]): str(row["stratiLINK"])
        for _, row in strati_df.iterrows()
        if pd.notna(row["stratiLINK"])
    }

    att_uuid_to_strati: dict[str, str] = {
        uid: gmu_to_strati[gmu]
        for uid, gmu in att_uuid_to_gmu.items()
        if gmu in gmu_to_strati
    }
    log(f"  strati_link lookup: {len(att_uuid_to_strati):,} entries from Excel")

    bedrock_lyr = ds.GetLayerByName("GC_BEDROCK")
    if bedrock_lyr is None:
        log("  WARNING: GC_BEDROCK not found — strati_link skipped")
        return

    updated = skipped = 0
    for feat in bedrock_lyr:
        att_uuid = feat.GetField("GEOL_MAPPING_UNIT_ATT_UUID")
        val = att_uuid_to_strati.get(att_uuid) if att_uuid else None
        if val:
            feat.SetField("strati_link", val)
            bedrock_lyr.SetFeature(feat)
            updated += 1
        else:
            skipped += 1

    log(f"  strati_link: {updated:,} updated, {skipped:,} without link")


def patch_schema_gdb(
    schema_gdb: Path,
    merged_gdb: Path,
    output_gdb: Path,
    log: Callable[[str], None] = print,
    exclude_fields: set[str] | None = None,
    strati_links_path: Path | None = None,
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
    exclude_fields:
        Field names to drop from every spatial layer in the clone before
        appending data.  Pass GEOCOVER_METADATA_FIELDS to mirror the
        --exclude-metadata behaviour of the merge step.
    strati_links_path:
        Path to ``_Update_stratiLINK.xlsx``.  When provided, strati_link is
        injected into GC_BEDROCK after the append step via a two-hop join
        through GC_GEOL_MAPPING_UNIT_ATT.  When None, the field is omitted
        and a warning is logged.

    Returns
    -------
    list[str]
        Error messages for layers that failed; empty on full success.
    """
    schema_gdb = str(schema_gdb)
    merged_gdb = str(merged_gdb)
    output_gdb_str = str(output_gdb)

    # Step 1: byte-perfect clone
    import os
    if os.path.exists(output_gdb_str):
        shutil.rmtree(output_gdb_str)

    log(f"Cloning {schema_gdb} …")
    t0 = time.time()
    shutil.copytree(schema_gdb, output_gdb_str)
    log(f"  done in {time.time()-t0:.1f}s")

    # Drop topology layers not needed in publication
    ds = ogr.Open(output_gdb_str, 1)
    dropped: list[str] = []
    for i in range(ds.GetLayerCount() - 1, -1, -1):
        name = ds.GetLayerByIndex(i).GetName()
        if name in _DROP_LAYERS or any(name.startswith(p) for p in _DROP_PREFIXES):
            ds.DeleteLayer(i)
            dropped.append(name)
    ds = None
    if dropped:
        log(f"  dropped: {', '.join(dropped)}")

    ds = ogr.Open(output_gdb_str, 0)
    log(f"  layers: {ds.GetLayerCount()},  domains: {len(ds.GetFieldDomainNames() or [])}")
    ds = None

    # Step 1b: add extra fields absent from schema GDB / drop excluded fields
    if strati_links_path is None:
        log("  WARNING: --strati-links not provided, strati_link field omitted from output")

    ds = ogr.Open(output_gdb_str, 1)
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

        # strati_link only on GC_BEDROCK, and only when Excel path is given
        if layer_name == "GC_BEDROCK" and strati_links_path is not None:
            if "strati_link" not in existing:
                lyr.CreateField(ogr.FieldDefn("strati_link", ogr.OFTString))
                log(f"  added field 'strati_link' to {layer_name}")

        if exclude_fields:
            # Delete fields in reverse index order to keep indices stable
            defn = lyr.GetLayerDefn()
            to_drop = [
                i for i in range(defn.GetFieldCount())
                if defn.GetFieldDefn(i).GetName() in exclude_fields
            ]
            for i in reversed(to_drop):
                lyr.DeleteField(i)
            if to_drop:
                log(f"  dropped {len(to_drop)} metadata fields from {layer_name}")
    ds = None

    # Steps 2 & 3: patch spatial layers and tables
    errors: list[str] = []
    errors += _patch(output_gdb_str, merged_gdb, SPATIAL_LAYERS, "Spatial layers (GC_ROCK_BODIES)", log)
    errors += _patch(output_gdb_str, merged_gdb, TABLE_LAYERS,   "Tables & relationship tables", log)

    # Step 4: inject strati_link from Excel
    if strati_links_path is not None:
        log("\n--- strati_link injection ---")
        ds = ogr.Open(output_gdb_str, 1)
        _inject_strati_link(ds, strati_links_path, log)
        ds = None

    ds = ogr.Open(output_gdb_str, 0)
    log(f"\nDone → {output_gdb_str}")
    log(f"  layers : {ds.GetLayerCount()}")
    log(f"  domains: {len(ds.GetFieldDomainNames() or [])}")
    ds = None

    return errors
