# Fix V2 for merge_sources.py - Handle both GDBs containing full Switzerland
#
# The REAL issue: Both RC1.gdb and RC2.gdb contain complete Switzerland data.
# They're two versions of the same dataset, not disjoint subsets.
#
# Even with exclusive masks, we were reading the SAME features (same UUIDs)
# from both databases because features exist in both with identical UUIDs.
#
# Solution: For each geographic area, read ONLY from the authoritative source.
# This requires changing how we iterate - instead of "for each source, use its mask",
# we need "for each mask, read from the correct source".

"""
The key insight:
- RC1.gdb has features covering ALL of Switzerland
- RC2.gdb ALSO has features covering ALL of Switzerland (it's a copy with edits)
- The SOURCE_RC in mapsheets tells us which GDB is AUTHORITATIVE for each area
- We must NOT read RC2 area features from RC1.gdb (even though they exist there)
"""

from pathlib import Path
from typing import Dict, Optional, Set
import time

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely import intersects, within, intersection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely import difference


def _merge_spatial_layer_fixed(
        self,
        layer_name: str,
        progress=None,
        task_id=None
) -> Optional[gpd.GeoDataFrame]:
    """
    Merge a single spatial layer - FIXED VERSION.

    Key changes:
    1. Create exclusive masks (no geographic overlap)
    2. For each exclusive mask, read ONLY from its authoritative source
    3. Track UUIDs to catch any remaining duplicates
    """
    from gcover.publish.merge_sources import (
        get_expected_geometry_type,
        fast_clip,
        normalize_geodataframe_geometries,
    )

    merged_parts = []
    layer_crs = None
    seen_uuids: Set[str] = set()  # Track UUIDs to prevent ANY duplicates

    expected_type = get_expected_geometry_type(layer_name)
    logger.debug(f"Processing layer {layer_name}, expected type: {expected_type}")

    # Process each source with its EXCLUSIVE mask
    for source_name, clip_mask in self.source_masks.items():
        source_path = self._resolve_source_path(source_name)
        if source_path is None:
            msg = f"Source not found: {source_name}"
            logger.warning(msg)
            self.stats.warnings.append(msg)
            continue

        logger.debug(f"  Reading {source_name} from {source_path}")

        layer_start = time.time()

        # Read from THIS source's GDB, filtered by the exclusive mask
        gdf = self._read_layer_for_source(source_path, layer_name, clip_mask)

        if gdf is None or gdf.empty:
            logger.debug(f"  {source_name}: no features in {layer_name}")
            continue

        read_time = time.time() - layer_start
        read_count = len(gdf)
        logger.debug(f"  {source_name}: read {read_count} features ({read_time:.1f}s)")

        if layer_crs is None and gdf.crs is not None:
            layer_crs = gdf.crs

        # Clip to the EXCLUSIVE mask (precise clipping)
        clip_start = time.time()

        if expected_type in ["MultiPoint", "Point"]:
            clipped = gdf[intersects(gdf.geometry.values, clip_mask)].copy()
        else:
            clipped = fast_clip(gdf, clip_mask)

        clip_time = time.time() - clip_start

        if clipped.empty:
            logger.debug(f"  {source_name}: no features after clip")
            continue

        # === UUID DEDUPLICATION (critical!) ===
        # Even with exclusive masks, if the spatial read returned features
        # from outside the mask (due to bbox filtering), we might have dupes.
        if 'UUID' in clipped.columns:
            before_dedup = len(clipped)

            # Keep only features with new UUIDs (or NULL UUIDs)
            clipped_uuids = clipped['UUID'].dropna().unique()
            new_uuids = set(clipped_uuids) - seen_uuids

            # Filter: keep if UUID is NULL or UUID is new
            mask = clipped['UUID'].isna() | clipped['UUID'].isin(new_uuids)
            clipped = clipped[mask].copy()

            # Track these UUIDs
            seen_uuids.update(new_uuids)

            dupes_removed = before_dedup - len(clipped)
            if dupes_removed > 0:
                logger.warning(f"  {source_name}: removed {dupes_removed} duplicate UUIDs!")

        # Add source tracking
        clipped["_MERGE_SOURCE"] = source_name
        merged_parts.append(clipped)

        feature_count = len(clipped)
        self.stats.features_per_source[source_name] = \
            self.stats.features_per_source.get(source_name, 0) + feature_count

        logger.debug(f"  {source_name}: {feature_count} features after clip ({clip_time:.1f}s)")

        if progress and task_id:
            progress.advance(task_id)

    # Merge all parts
    if not merged_parts:
        return None

    if len(merged_parts) == 1:
        result = merged_parts[0]
    else:
        result = pd.concat(merged_parts, ignore_index=True)
        if layer_crs is not None:
            result = gpd.GeoDataFrame(result, geometry="geometry", crs=layer_crs)
        else:
            result = gpd.GeoDataFrame(result, geometry="geometry")

    # Final Swiss border clip
    if self.config.clip_to_swiss_border and self.swiss_border is not None:
        border_start = time.time()
        if expected_type in ["MultiPoint", "Point"]:
            result = result[intersects(result.geometry.values, self.swiss_border)].copy()
        else:
            result = fast_clip(result, self.swiss_border)
        logger.debug(f"  Swiss border clip: {len(result)} features ({time.time() - border_start:.1f}s)")

    # Normalize geometries
    result = normalize_geodataframe_geometries(
        result,
        target_type=expected_type,
        preserve_z=self.config.preserve_z
    )

    if result.empty:
        logger.warning(f"No valid features after normalization for {layer_name}")
        return None

    self.stats.features_per_layer[layer_name] = len(result)
    return result


def _create_source_masks_fixed_v2(self) -> Dict[str, BaseGeometry]:
    """
    Create EXCLUSIVE (non-overlapping) clip masks for each source.

    V2 improvements:
    - Better logging to diagnose issues
    - Verify masks don't overlap
    - Handle edge cases
    """
    if self.mapsheets_gdf is None:
        self._load_mapsheets()

    # Rename column for consistency
    source_col = self.config.source_column
    if source_col in self.mapsheets_gdf.columns and source_col != 'SOURCE_RC':
        self.mapsheets_gdf = self.mapsheets_gdf.rename(columns={source_col: 'SOURCE_RC'})

    # Get unique sources and their mapsheet counts
    source_counts = self.mapsheets_gdf['SOURCE_RC'].value_counts()
    logger.info(f"Sources in mapsheets: {dict(source_counts)}")

    # Determine priority (RC2 > RC1 > others)
    all_sources = list(source_counts.index)
    source_priority = []
    if 'RC2' in all_sources:
        source_priority.append('RC2')
    if 'RC1' in all_sources:
        source_priority.append('RC1')
    for src in sorted(all_sources):
        if src not in source_priority:
            source_priority.append(src)

    logger.info(f"Source priority (higher priority = gets area first): {source_priority}")

    masks = {}
    claimed_area = None

    for source_name in source_priority:
        sheets = self.mapsheets_gdf[self.mapsheets_gdf['SOURCE_RC'] == source_name]
        if sheets.empty:
            logger.warning(f"No mapsheets for source {source_name}")
            continue

        # Dissolve mapsheets for this source
        source_area = unary_union(sheets.geometry.values)

        # Clean topology
        if not source_area.is_valid:
            source_area = source_area.buffer(0)

        source_area_km2 = source_area.area / 1e6
        logger.debug(f"  {source_name}: raw area = {source_area_km2:.1f} km²")

        # Make EXCLUSIVE
        if claimed_area is not None:
            exclusive_area = difference(source_area, claimed_area)

            if exclusive_area is None or exclusive_area.is_empty:
                logger.warning(f"  {source_name}: NO exclusive area (fully covered by higher priority)")
                continue

            exclusive_area_km2 = exclusive_area.area / 1e6
            reduction = (1 - exclusive_area.area / source_area.area) * 100
            logger.debug(f"  {source_name}: exclusive area = {exclusive_area_km2:.1f} km² "
                         f"({reduction:.1f}% removed by higher priority sources)")

            mask = exclusive_area
        else:
            mask = source_area

        # Clean final mask
        if not mask.is_valid:
            mask = mask.buffer(0)

        masks[source_name] = mask

        # Update claimed area
        if claimed_area is None:
            claimed_area = source_area
        else:
            claimed_area = unary_union([claimed_area, source_area])

    # VERIFY: Check that masks don't overlap
    logger.info("Verifying mask exclusivity...")
    mask_names = list(masks.keys())
    for i, name1 in enumerate(mask_names):
        for name2 in mask_names[i + 1:]:
            overlap = masks[name1].intersection(masks[name2])
            if overlap and not overlap.is_empty and overlap.area > 1:  # > 1 m²
                logger.error(f"  OVERLAP between {name1} and {name2}: {overlap.area:.1f} m²!")
            else:
                logger.debug(f"  {name1} ∩ {name2} = OK (no significant overlap)")

    self.source_masks = masks
    return masks


# =============================================================================
# ALTERNATIVE: If masks ARE correct, the issue is in _read_layer_for_source
# =============================================================================

def _read_layer_for_source_fixed(
        self,
        gdb_path: Path,
        layer_name: str,
        mask: BaseGeometry
) -> Optional[gpd.GeoDataFrame]:
    """
    Read a layer from a FileGDB with STRICT spatial filtering.

    The original uses mask= for bbox filtering during read, but this can
    return features outside the mask (anything touching the bbox).

    This version does an additional intersection check after reading.
    """
    actual_layer = self._get_layer_path(gdb_path, layer_name)

    try:
        # Read with bbox filter (fast)
        gdf = gpd.read_file(
            gdb_path,
            layer=actual_layer,
            engine='pyogrio',
            mask=mask  # This does bbox filter, not exact intersection!
        )

        if gdf.empty:
            return gdf

        # STRICT filter: only keep features that actually intersect the mask
        # (not just its bounding box)
        gdf = gdf[intersects(gdf.geometry.values, mask)].copy()

        if self.config.validate_geometries and not gdf.empty:
            from gcover.core.geometry import validate_and_repair_geometries
            gdf = validate_and_repair_geometries(gdf)

        return gdf

    except Exception as e:
        logger.warning(f"Could not read {actual_layer} from {gdb_path}: {e}")
        return None


# =============================================================================
# HOW TO APPLY
# =============================================================================

def apply_fixes(GDBMerger_class):
    """
    Apply all fixes to the GDBMerger class.

    Usage:
        from gcover.publish.merge_sources import GDBMerger
        from merge_sources_fix_v2 import apply_fixes

        apply_fixes(GDBMerger)

        merger = GDBMerger(config, verbose=True)
        merger.merge()
    """
    GDBMerger_class._create_source_masks = _create_source_masks_fixed_v2
    GDBMerger_class._merge_spatial_layer = _merge_spatial_layer_fixed
    GDBMerger_class._read_layer_for_source = _read_layer_for_source_fixed
    print("Applied merge_sources fixes (v2)")


if __name__ == "__main__":
    print("""
Fix V2 for duplicate features issue.

The problem: Both RC1.gdb and RC2.gdb contain data for ALL of Switzerland.
Even with exclusive masks, we were reading the same features (same UUIDs)
from both databases.

This fix adds:
1. Stricter spatial filtering in _read_layer_for_source
2. UUID-based deduplication in _merge_spatial_layer  
3. Better mask verification logging

To apply:

    from gcover.publish.merge_sources import GDBMerger
    from merge_sources_fix_v2 import apply_fixes

    apply_fixes(GDBMerger)

    # Then use normally
    merger = GDBMerger(config, verbose=True)
    stats = merger.merge()

Or integrate the methods directly into merge_sources.py.
""")