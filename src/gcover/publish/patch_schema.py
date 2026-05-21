"""
patch_schema.py — clone an ESRI FileGDB schema and populate it from a flat merged GDB.

Strategy
--------
  1. shutil.copytree  — byte-perfect clone: domains, relationship classes,
                        feature datasets and topology layers all survive.
  2. OGR update mode  — truncate every target layer (SetIgnoredFields skips
                        geometry parsing to avoid unclosed-ring RuntimeError).
  3. gdal.VectorTranslate(accessMode='append') — bulk-write from merged_gdb.
  4. Optional strati_link injection — Excel join via GC_GEOL_MAPPING_UNIT_ATT.
"""
from __future__ import annotations

import shutil
import time
import warnings
from pathlib import Path
from typing import Callable

from osgeo import gdal, ogr

gdal.UseExceptions()

SPATIAL_LAYERS: list[str] = [
    "GC_BEDROCK",
    "GC_UNCO_DESPOSIT",
    "GC_SURFACES",
    "GC_LINEAR_OBJECTS",
    "GC_POINT_OBJECTS",
    "GC_FOSSILS",
    "GC_EXPLOIT_GEOMAT_PLG",
    "GC_EXPLOIT_GEOMAT_PT",
]

# GC_UN_DEP_ADMIXTUR_GC_ADMIXT kept from the clone (geopandas dropped it).
TABLE_LAYERS: list[str] = [
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

_DROP_LAYERS: frozenset[str] = frozenset({"BEDROCK_TOPOLOGY", "GC_ROCK_BODIES_TOPO"})
_DROP_PREFIXES: tuple[str, ...] = ("T_1_",)

# Fields present in merged_master.gdb but absent from RC2 schema.
# VectorTranslate(accessMode='append') only writes fields that already exist
# in the target, so we create them here before the append step.
_EXTRA_FIELDS: list[tuple[str, int]] = [
    ("_MERGE_SOURCE", ogr.OFTString),
    ("erl_link",      ogr.OFTString),
    ("ber_link",      ogr.OFTString),
]


def _truncate(ds: ogr.DataSource, name: str) -> int:
    """Delete all features from a layer; return count deleted (or -1 if absent)."""
    lyr = ds.GetLayerByName(name)
    if lyr is None:
        return -1
    # Skip geometry parsing — RC2 clone may contain unclosed rings that raise
    # RuntimeError with gdal.UseExceptions(). FIDs are all we need here.
    lyr.SetIgnoredFields(["OGR_GEOMETRY"])
    fids = [f.GetFID() for f in lyr]
    lyr.SetIgnoredFields([])
    lyr.ResetReading()
    for fid in fids:
        lyr.DeleteFeature(fid)
    return len(fids)


def _append(output_gdb: str, merged_gdb: str, name: str) -> int:
    """Bulk-append one layer from merged_gdb into output_gdb; return feature count."""
    gdal.VectorTranslate(
        output_gdb,
        merged_gdb,
        options=gdal.VectorTranslateOptions(
            layers=[name],
            accessMode="append",
            # explodeCollections: geopandas may promote Point → MultiPoint;
            # split back to single points to match the clone's Point schema.
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
    label: str,
    log: Callable[[str], None],
) -> list[str]:
    errors: list[str] = []
    log(f"\n{label}")
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

        ds = None  # flush before VectorTranslate touches the file
        try:
            n = _append(output_gdb, merged_gdb, name)
            log(f"  OK    {name}: {n:,}  ({time.time()-t:.1f}s)")
        except Exception as exc:
            msg = f"{name}: {exc}"
            log(f"  ERR   {msg}")
            errors.append(msg)

        ds = ogr.Open(output_gdb, 1)  # reopen for next iteration

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
    """Clone schema_gdb, populate it from merged_gdb, return any error messages.

    Parameters
    ----------
    schema_gdb:
        Authoritative FileGDB to clone (e.g. RC2.gdb).  Domains, relationship
        classes, and topology are preserved.
    merged_gdb:
        Flat merged FileGDB produced by ``gcover publish merge``.
    output_gdb:
        Destination path.  Overwritten if it already exists.
    log:
        Callable for progress messages (default: print).
    exclude_fields:
        Field names to delete from the clone schema before appending.
        Prevents empty metadata columns in the output.
    strati_links_path:
        Path to ``_Update_stratiLINK.xlsx``.  When provided, strati_link is
        injected into GC_BEDROCK after the append step.  When None, the field
        is omitted and a warning is printed.
    """
    output_gdb_str = str(output_gdb)
    merged_gdb_str = str(merged_gdb)

    # ── Step 1: Clone ─────────────────────────────────────────────────────────
    if output_gdb.exists():
        shutil.rmtree(output_gdb)

    log(f"Cloning {schema_gdb.name} → {output_gdb.name} …")
    t0 = time.time()
    shutil.copytree(str(schema_gdb), output_gdb_str)
    log(f"  done in {time.time()-t0:.1f}s")

    # Drop topology-related layers not needed in the publication GDB
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

    # ── Step 1b: Add extra fields absent from schema GDB ──────────────────────
    extra_fields = list(_EXTRA_FIELDS)
    if strati_links_path is not None:
        extra_fields_bedrock = [("strati_link", ogr.OFTString)]
    else:
        warnings.warn(
            "--strati-links not provided: strati_link field will be absent from output GDB",
            stacklevel=2,
        )
        log("  WARNING: --strati-links not provided, strati_link field omitted")
        extra_fields_bedrock = []

    ds = ogr.Open(output_gdb_str, 1)
    for layer_name in SPATIAL_LAYERS:
        lyr = ds.GetLayerByName(layer_name)
        if lyr is None:
            continue
        defn = lyr.GetLayerDefn()
        existing = {defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())}

        fields_for_layer = extra_fields + (
            extra_fields_bedrock if layer_name == "GC_BEDROCK" else []
        )
        for fname, ftype in fields_for_layer:
            if fname not in existing:
                lyr.CreateField(ogr.FieldDefn(fname, ftype))

        # Remove metadata field definitions so they don't appear as empty columns
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
    ds = None

    # ── Step 2 & 3: Patch ─────────────────────────────────────────────────────
    errors: list[str] = []
    errors += _patch(output_gdb_str, merged_gdb_str, SPATIAL_LAYERS, "Spatial layers", log)
    errors += _patch(output_gdb_str, merged_gdb_str, TABLE_LAYERS, "Tables", log)

    # ── Step 4: Inject strati_link ────────────────────────────────────────────
    if strati_links_path is not None:
        log("\nInjecting strati_link …")
        ds = ogr.Open(output_gdb_str, 1)
        _inject_strati_link(ds, strati_links_path, log)
        ds = None

    ds = ogr.Open(output_gdb_str, 0)
    log(
        f"\nDone → {output_gdb.name}"
        f"  layers: {ds.GetLayerCount()}"
        f"  domains: {len(ds.GetFieldDomainNames() or [])}"
    )
    ds = None
    return errors
