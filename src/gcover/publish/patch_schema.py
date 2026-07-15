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

import gc
import shutil
import tempfile
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
    ("ERL_LINK",      ogr.OFTString),
    ("BER_LINK",      ogr.OFTString),
]


def _truncate(ds: ogr.DataSource, name: str, log: Callable[[str], None] | None = None) -> int:
    """Delete every feature in *name*.

    DeleteFeature is called one row at a time — on a large layer (RC2's
    schema-clone tables can hold ~300k features) this loop can run for a
    long time with zero other output, which looks hung. If a per-write
    slowdown is in play (e.g. antivirus real-time scanning on Windows —
    see module docstring), it's exactly this loop that eats the time.
    Report progress at least every 15s via *log* so that's visible instead
    of silent.
    """
    lyr = ds.GetLayerByName(name)
    if lyr is None:
        return -1
    # Skip geometry parsing — we only need FIDs, and the schema-clone source
    # (RC2.gdb) may contain unclosed rings that OGR raises as RuntimeError.
    lyr.SetIgnoredFields(["OGR_GEOMETRY"])
    t_scan = time.time()
    fids = [f.GetFID() for f in lyr]
    lyr.SetIgnoredFields([])
    lyr.ResetReading()
    total = len(fids)
    if log and total:
        log(f"    {name}: {total:,} features to delete (FID scan: {time.time()-t_scan:.1f}s)")

    t_start = last_report = time.time()
    for i, fid in enumerate(fids, 1):
        lyr.DeleteFeature(fid)
        if log and (time.time() - last_report >= 15 or i == total):
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else float("inf")
            log(f"    {name}: {i:,}/{total:,} deleted ({rate:.0f} feat/s, ETA {eta:.0f}s)")
            last_report = time.time()
    return total


def _append(output_gdb: str, merged_gdb: str, name: str, log: Callable[[str], None] | None = None) -> int:
    """Bulk-append one layer from merged_gdb into output_gdb.

    explodeCollections splits MultiPoint back to Point so the geometry
    type matches the schema-clone target (geopandas promotes to multi).

    A single gdal.VectorTranslate() call gives no output until it returns —
    on GC_ROCK_BODIES layers this is the organizePolygons()/topology step
    and can run tens of minutes silently, indistinguishable from a hang.
    Report progress at least every 15s via *log*, mirroring _truncate().
    """
    t_start = last_report = time.time()

    def _progress(complete: float, message: str, cb_data) -> int:
        nonlocal last_report
        now = time.time()
        if log and (now - last_report >= 15 or complete >= 1.0):
            elapsed = now - t_start
            eta = elapsed * (1 - complete) / complete if complete > 0 else float("inf")
            log(f"    {name}: {complete * 100:.0f}% ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
            last_report = now
        return 1

    gdal.VectorTranslate(
        output_gdb,
        merged_gdb,
        options=gdal.VectorTranslateOptions(
            layers=[name],
            accessMode="append",
            explodeCollections=True,
            callback=_progress,
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
        deleted = _truncate(ds, name, log=log)
        if deleted == -1:
            log(f"  WARN  {name}  (not in clone)")
            continue

        ds = None
        try:
            n = _append(output_gdb, merged_gdb, name, log=log)
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

    t0 = time.time()
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
    log(f"  strati_link lookup: {len(att_uuid_to_strati):,} entries from Excel ({time.time()-t0:.1f}s)")

    bedrock_lyr = ds.GetLayerByName("GC_BEDROCK")
    if bedrock_lyr is None:
        log("  WARNING: GC_BEDROCK not found — strati_link skipped")
        return

    # SetFeature() is a per-row rewrite; without a transaction each call is
    # its own implicit commit, which is exactly the write pattern _truncate()
    # above already flags as catastrophically slow on Windows (Defender
    # scanning every write). Wrap the whole pass in one transaction so it's a
    # single commit, and report progress every 15s like _truncate/_append do,
    # since GC_BEDROCK is large enough (~1e5 features) that silence here reads
    # as a hang.
    use_txn = ds.TestCapability(ogr.ODsCTransactions)
    if use_txn:
        ds.StartTransaction()

    total = bedrock_lyr.GetFeatureCount()
    updated = skipped = 0
    t_start = last_report = time.time()
    try:
        for i, feat in enumerate(bedrock_lyr, 1):
            att_uuid = feat.GetField("GEOL_MAPPING_UNIT_ATT_UUID")
            val = att_uuid_to_strati.get(att_uuid) if att_uuid else None
            if val:
                feat.SetField("strati_link", val)
                bedrock_lyr.SetFeature(feat)
                updated += 1
            else:
                skipped += 1
            now = time.time()
            if now - last_report >= 15 or i == total:
                elapsed = now - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else float("inf")
                log(f"    GC_BEDROCK: {i:,}/{total:,} strati_link ({rate:.0f} feat/s, ETA {eta:.0f}s)")
                last_report = now
    except Exception:
        if use_txn:
            ds.RollbackTransaction()
        raise
    else:
        if use_txn:
            ds.CommitTransaction()

    log(f"  strati_link: {updated:,} updated, {skipped:,} without link ({time.time()-t_start:.1f}s)")


def patch_schema_gdb(
    schema_gdb: Path,
    merged_gdb: Path,
    output_gdb: Path,
    log: Callable[[str], None] = print,
    exclude_fields: set[str] | None = None,
    strati_links_path: Path | None = None,
    admin_zones_path: Path | None = None,
    mapsheets_layer: str = "mapsheets_sources_only",
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
    admin_zones_path:
        Path to ``administrative_zones.gpkg``.  When provided, GC_MAPSHEET is
        replaced with the ``mapsheets_layer`` layer (fields uppercased,
        Z dimension dropped).  When None, GC_MAPSHEET is kept from the schema
        clone (RC2 data).
    mapsheets_layer:
        Layer name to read from ``admin_zones_path`` (default matches the
        legacy pre-joined ``mapsheets_sources_only`` byproduct).

    Returns
    -------
    list[str]
        Error messages for layers that failed; empty on full success.
    """
    schema_gdb = str(schema_gdb)
    merged_gdb = str(merged_gdb)
    output_gdb_str = str(output_gdb)

    # All in-place GDAL work (field edits, data append) is done on a local
    # temporary copy to avoid GDAL/Windows SMB oplock contention on UNC paths
    # (random "Permission denied" on a00000004.gdbtable during DeleteField).
    # The finished GDB is moved to the UNC destination in one shot at the end.
    tmp_dir = tempfile.mkdtemp(prefix="gcover_patch_")
    work_gdb    = str(Path(tmp_dir) / Path(output_gdb_str).name)
    local_merged = str(Path(tmp_dir) / Path(merged_gdb).name)

    try:
        import os

        # Step 0: copy merged_gdb to local temp so all GDAL reads are local.
        # On Windows, GDAL's OpenFileGDB does many small random seeks through
        # .gdbtable files; over a UNC/SMB share each seek is a network round-trip,
        # making VectorTranslate ~12× slower than on Linux with NFS.
        # One sequential copy up-front is far cheaper than millions of random reads.
        log(f"Copying merged GDB to local temp {tmp_dir}…")
        t0 = time.time()
        shutil.copytree(merged_gdb, local_merged)
        log(f"  done in {time.time()-t0:.1f}s")

        # Step 1: byte-perfect clone → local working copy
        log(f"Cloning {schema_gdb} …")
        t0 = time.time()
        shutil.copytree(schema_gdb, work_gdb)
        log(f"  done in {time.time()-t0:.1f}s")

        # Drop topology layers not needed in publication
        ds = ogr.Open(work_gdb, 1)
        dropped: list[str] = []
        for i in range(ds.GetLayerCount() - 1, -1, -1):
            name = ds.GetLayerByIndex(i).GetName()
            if name in _DROP_LAYERS or any(name.startswith(p) for p in _DROP_PREFIXES):
                ds.DeleteLayer(i)
                dropped.append(name)
        ds = None
        if dropped:
            log(f"  dropped: {', '.join(dropped)}")

        ds = ogr.Open(work_gdb, 0)
        log(f"  layers: {ds.GetLayerCount()},  domains: {len(ds.GetFieldDomainNames() or [])}")
        ds = None

        # Step 1b-pre: truncate spatial layers BEFORE any field edits.
        # DeleteField in FileGDB rewrites the entire .gdbtable for every call.
        # On layers that still hold RC2's ~300k features, that means 19 full
        # table rewrites per layer — catastrophically slow on Windows where
        # Defender scans every write.  Empty tables make field ops trivial.
        log("\nTruncating spatial layers …")
        ds = ogr.Open(work_gdb, 1)
        for layer_name in SPATIAL_LAYERS:
            t0 = time.time()
            n = _truncate(ds, layer_name, log=log)
            if n >= 0:
                log(f"  cleared {layer_name} ({n:,} features) in {time.time() - t0:.1f}s")
        ds = None
        gc.collect()

        # Step 1b: drop excluded fields, then add extra fields — now on empty layers.
        if strati_links_path is None:
            log("  WARNING: --strati-links not provided, strati_link field omitted from output")

        ds = ogr.Open(work_gdb, 1)
        for layer_name in SPATIAL_LAYERS:
            lyr = ds.GetLayerByName(layer_name)
            if lyr is None:
                continue

            if exclude_fields:
                defn = lyr.GetLayerDefn()
                to_drop = [
                    i for i in range(defn.GetFieldCount())
                    if defn.GetFieldDefn(i).GetName() in exclude_fields
                ]
                for i in reversed(to_drop):
                    lyr.DeleteField(i)
                if to_drop:
                    log(f"  dropped {len(to_drop)} metadata fields from {layer_name}")

            defn = lyr.GetLayerDefn()
            existing = {defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())}
            for fname, ftype in _EXTRA_FIELDS:
                if fname not in existing:
                    lyr.CreateField(ogr.FieldDefn(fname, ftype))
                    log(f"  added field '{fname}' to {layer_name}")

            if layer_name == "GC_BEDROCK" and strati_links_path is not None:
                if "strati_link" not in existing:
                    lyr.CreateField(ogr.FieldDefn("strati_link", ogr.OFTString))
                    log(f"  added field 'strati_link' to {layer_name}")
        ds = None
        gc.collect()

        # Steps 2 & 3: patch spatial layers and tables (both paths now local)
        errors: list[str] = []
        errors += _patch(work_gdb, local_merged, SPATIAL_LAYERS, "Spatial layers (GC_ROCK_BODIES)", log)
        errors += _patch(work_gdb, local_merged, TABLE_LAYERS,   "Tables & relationship tables", log)

        # Step 4: inject strati_link from Excel
        if strati_links_path is not None:
            log("\n--- strati_link injection ---")
            ds = ogr.Open(work_gdb, 1)
            _inject_strati_link(ds, strati_links_path, log)
            ds = None

        # Step 5: replace GC_MAPSHEET
        if admin_zones_path is not None:
            log("\n--- GC_MAPSHEET ---")
            ds = ogr.Open(work_gdb, 1)
            for i in range(ds.GetLayerCount()):
                if ds.GetLayerByIndex(i).GetName() == "GC_MAPSHEET":
                    ds.DeleteLayer(i)
                    break
            ds = None

            # admin_zones_path may be either the old pre-joined shape (SOURCE_RC,
            # erl_link, ber_link already baked in) or a raw GC_MAPSHEET.gpkg-derived
            # file (BKP, no link columns, kept unmangled on purpose) — detect which
            # and build the SELECT accordingly rather than requiring a fixed shape.
            az_ds = ogr.Open(str(admin_zones_path), 0)
            az_lyr = az_ds.GetLayerByName(mapsheets_layer)
            az_fields = {az_lyr.GetLayerDefn().GetFieldDefn(i).GetName()
                         for i in range(az_lyr.GetLayerDefn().GetFieldCount())}
            az_ds = None

            if "SOURCE_RC" in az_fields:
                source_expr = "SOURCE_RC"
            elif "BKP" in az_fields:
                source_expr = "BKP AS SOURCE_RC"
            else:
                raise ValueError(
                    f"{mapsheets_layer}: neither SOURCE_RC nor BKP column found"
                )

            if {"erl_link", "ber_link"}.issubset(az_fields):
                link_exprs = "ber_link AS BER_LINK, erl_link AS ERL_LINK"
            else:
                from gcover.publish.administrative_zones import (
                    ERLAUETERUNG_LINK, BERICHT_LINK,
                )
                link_exprs = (
                    f"CASE WHEN BER = 'y' THEN '{BERICHT_LINK}' || MSH_MAP_NBR || '.pdf' "
                    "ELSE '' END AS BER_LINK, "
                    f"CASE WHEN ERL = 'y' THEN '{ERLAUETERUNG_LINK}' || MSH_MAP_NBR || '.pdf' "
                    "ELSE '' END AS ERL_LINK"
                )

            _sql = (
                "SELECT geom, MSH_MAP_TITLE, MSH_MAP_NBR, MSH_TOPO_NR, "
                f"{source_expr}, Version AS VERSION, BER, ERL, {link_exprs} "
                f"FROM {mapsheets_layer}"
            )
            gdal.VectorTranslate(
                work_gdb,
                str(admin_zones_path),
                options=gdal.VectorTranslateOptions(
                    SQLStatement=_sql,
                    SQLDialect="SQLite",
                    layerName="GC_MAPSHEET",
                    accessMode="update",
                    geometryType="POLYGON",
                    layerCreationOptions=["TARGET_ARCGIS_VERSION=ARCGIS_PRO_3_2_OR_LATER"],
                ),
            )
            ds = ogr.Open(work_gdb, 0)
            lyr = ds.GetLayerByName("GC_MAPSHEET")
            n = lyr.GetFeatureCount() if lyr else -1
            ds = None
            log(f"  GC_MAPSHEET: {n:,} features")

        gc.collect()

        # Move finished local GDB to UNC destination (single network write)
        log(f"\nCopying result to {output_gdb_str} …")
        t0 = time.time()
        if os.path.exists(output_gdb_str):
            shutil.rmtree(output_gdb_str)
        shutil.copytree(work_gdb, output_gdb_str)
        log(f"  done in {time.time()-t0:.1f}s")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    ds = ogr.Open(output_gdb_str, 0)
    log(f"\nDone → {output_gdb_str}")
    log(f"  layers : {ds.GetLayerCount()}")
    log(f"  domains: {len(ds.GetFieldDomainNames() or [])}")
    ds = None

    return errors
