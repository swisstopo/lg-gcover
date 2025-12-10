#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write back classification attributes from a GeoPackage to a FileGDB.

Joins on UUID field and updates specified attributes for all matching
layers in GC_ROCK_BODIES feature dataset.

Supports loading layer mapping from the same YAML config used for classification.
"""
from pathlib import Path
from typing import Optional
import re

import click

from loguru import logger



from gcover.arcpy_compat import HAS_ARCPY, arcpy
from gcover.publish.style_config import BatchClassificationConfig


def load_layer_mapping_from_config(config_path: Path) -> dict[str, str]:
    """
    Load layer mapping from classification YAML config.

    Extracts gpkg_layer -> gcover_layer mapping.
    If gcover_layer is not specified, derives it from gpkg_layer.

    Returns: {gpkg_layer: gcover_layer, ...}
    """

    config = BatchClassificationConfig(config_path)

    mapping = {}

    for layer_config in config.layers:
        gpkg_layer = layer_config.gpkg_layer
        if not gpkg_layer:
            continue

        # Check for explicit gcover_layer mapping
        gcover_layer = layer_config.gcover_layer

        if not gcover_layer:
            # Derive from gpkg_layer - strip any prefix like "TOPGIS_GC."
            # and use the base name
            base_name = gpkg_layer.split(".")[-1]  # Handle TOPGIS_GC.GC_BEDROCK
            gcover_layer = base_name

        mapping[gpkg_layer] = gcover_layer
        logger.debug(f"Layer mapping: {gpkg_layer} -> {gcover_layer}")

    return mapping


def extract_layer_name(full_path: str) -> str:
    """
    Extract simple layer name from full SDE-style path.

    Examples:
        "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK" -> "GC_BEDROCK"
        "GC_ROCK_BODIES/GC_BEDROCK" -> "GC_BEDROCK"
        "GC_BEDROCK" -> "GC_BEDROCK"
    """
    # Get last component after / or \
    name = re.split(r"[/\\]", full_path)[-1]
    # Strip schema prefix like "TOPGIS_GC."
    return name.split(".")[-1]


def list_gpkg_layers(gpkg_path: Path) -> list[str]:
    """List all layers in a GeoPackage."""
    import fiona

    return fiona.listlayers(str(gpkg_path))


def list_filegdb_layers(gdb_path: Path, feature_dataset: str = "GC_ROCK_BODIES") -> list[str]:
    """List feature classes in a FileGDB feature dataset."""


    ds_path = str(gdb_path / feature_dataset)
    arcpy.env.workspace = ds_path

    return arcpy.ListFeatureClasses() or []


def get_matching_layers_from_config(
        config_mapping: dict[str, str],
        gpkg_layers: list[str],
        gdb_layers: list[str]
) -> list[tuple[str, str]]:
    """
    Get layer matches using config mapping.

    Returns list of (gpkg_layer, gdb_layer) tuples.
    """
    matches = []
    gdb_lookup = {lyr.upper(): lyr for lyr in gdb_layers}

    for gpkg_layer, gcover_layer in config_mapping.items():
        # Check if GPKG layer exists
        if gpkg_layer not in gpkg_layers:
            logger.warning(f"GPKG layer not found: {gpkg_layer}")
            continue

        # Extract base name for FileGDB lookup
        base_name = extract_layer_name(gcover_layer)

        # Find in GDB
        if base_name in gdb_layers:
            matches.append((gpkg_layer, base_name))
        elif base_name.upper() in gdb_lookup:
            matches.append((gpkg_layer, gdb_lookup[base_name.upper()]))
        else:
            logger.warning(f"FileGDB layer not found: {base_name} (from {gcover_layer})")

    return matches


def get_matching_layers_auto(
        gpkg_layers: list[str],
        gdb_layers: list[str]
) -> list[tuple[str, str]]:
    """
    Auto-detect matching layers between GPKG and FileGDB by name.

    Returns list of (gpkg_layer, gdb_layer) tuples.
    """
    matches = []
    gdb_lookup = {lyr.upper(): lyr for lyr in gdb_layers}

    for gpkg_lyr in gpkg_layers:
        base_name = extract_layer_name(gpkg_lyr)

        if base_name in gdb_layers:
            matches.append((gpkg_lyr, base_name))
        elif base_name.upper() in gdb_lookup:
            matches.append((gpkg_lyr, gdb_lookup[base_name.upper()]))

    return matches


def build_uuid_lookup(gpkg_path: Path, layer: str, uuid_field: str,
                      attributes: list[str]) -> dict:
    """
    Build a UUID -> attributes lookup dict from GPKG layer.

    Returns: {uuid: {attr1: val1, attr2: val2, ...}, ...}
    """
    import geopandas as gpd

    cols = [uuid_field] + attributes
    gdf = gpd.read_file(gpkg_path, layer=layer, columns=cols)

    lookup = {}
    for _, row in gdf.iterrows():
        uuid = row[uuid_field]
        if uuid:
            lookup[uuid] = {attr: row[attr] for attr in attributes}

    return lookup


def update_filegdb_layer(
        gdb_path: Path,
        feature_dataset: str,
        layer: str,
        uuid_lookup: dict,
        uuid_field: str,
        attributes: list[str],
        dryrun: bool = False
) -> dict:
    """
    Update attributes in FileGDB layer using UUID lookup.

    Returns stats dict with counts.
    """
    arcpy = _ensure_arcpy()

    fc_path = str(gdb_path / feature_dataset / layer)

    if not arcpy.Exists(fc_path):
        raise ValueError(f"Feature class not found: {fc_path}")

    # Verify fields exist
    existing_fields = {f.name for f in arcpy.ListFields(fc_path)}
    missing_fields = set(attributes) - existing_fields

    if missing_fields:
        # Add missing fields
        for field in missing_fields:
            logger.info(f"Adding field {field} to {layer}")
            if not dryrun:
                arcpy.management.AddField(fc_path, field, "TEXT", field_length=255)

    # Fields for cursor: UUID + attributes to update
    cursor_fields = [uuid_field] + attributes

    stats = {"matched": 0, "updated": 0, "skipped": 0, "not_found": 0}

    if dryrun:
        # Read-only pass for dryrun
        with arcpy.da.SearchCursor(fc_path, [uuid_field]) as cursor:
            for row in cursor:
                uuid = row[0]
                if uuid in uuid_lookup:
                    stats["matched"] += 1
                else:
                    stats["not_found"] += 1
        return stats

    # Actual update
    with arcpy.da.UpdateCursor(fc_path, cursor_fields) as cursor:
        for row in cursor:
            uuid = row[0]

            if uuid not in uuid_lookup:
                stats["not_found"] += 1
                continue

            stats["matched"] += 1
            new_values = uuid_lookup[uuid]

            # Check if update needed
            needs_update = False
            new_row = list(row)

            for i, attr in enumerate(attributes, start=1):
                new_val = new_values.get(attr)
                if row[i] != new_val:
                    new_row[i] = new_val
                    needs_update = True

            if needs_update:
                cursor.updateRow(new_row)
                stats["updated"] += 1
            else:
                stats["skipped"] += 1

    return stats