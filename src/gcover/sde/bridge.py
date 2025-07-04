"""
GeoCover Bridge - Main interface for GeoCover geodatabase operations

This module provides a unified interface for CRUD operations on GeoCover
geodatabase feature classes, including bulk operations, version management,
and data synchronization with timestamp checking.
"""

import os
import traceback
from datetime import datetime as dt
from pathlib import Path
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Literal
import contextlib

import click
import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely import wkt
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
)
from rich.console import Console

from .connection import SDEConnectionManager, ReadOnlyError
from .config import (
    DEFAULT_OPERATOR, DEFAULT_VERSION, FEAT_CLASSES_SHORTNAMES,
    SWISS_EPSG, DEFAULT_CHUNK_SIZE, MANDATORY_FIELDS
)
from .utils.gpkg import write_gdf_to_gpkg, append_gdf_to_gpkg, estimate_gpkg_size

try:
    import arcpy
except ImportError:
    # Mock arcpy for documentation generation
    class MockArcpy:
        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            return self


    arcpy = MockArcpy()

console = Console()


class GeoCoverBridge:
    """
    Unified interface for GeoCover geodatabase operations.

    Provides comprehensive CRUD operations with support for:
    - Bulk insert, update, delete operations
    - Mixed operation execution from single GeoDataFrame
    - Timestamp-based conflict resolution
    - Version management and transaction handling
    - Efficient data export to GeoDataFrame format

    Examples:
        # Basic usage with context manager
        with GeoCoverBridge(uuid_field="UUID") as bridge:
            # Export data to GeoDataFrame
            gdf = bridge.to_geopandas("bedrock", max_features=1000)

            # Bulk update existing features
            result = bridge.bulk_update(modified_gdf, "bedrock")

            # Execute mixed operations
            gdf['_operation'] = ['insert', 'update', 'delete', ...]
            results = bridge.execute_operations(gdf, "bedrock")
    """

    def __init__(
            self,
            uuid_field: str = "UUID",
            instance: str = "GCOVERP",
            version: str = DEFAULT_VERSION
    ):
        """
        Initialize GeoCover bridge.

        Args:
            uuid_field: Name of UUID field for feature identification
            instance: Database instance (GCOVERP or GCOVERI)
            version: Database version name
        """
        self.uuid_field = uuid_field
        self.date_field = "DATEOFCHANGE"
        self.operator_field = "OPERATOR"
        self.mandatory_fields = MANDATORY_FIELDS.copy()

        # Initialize connection manager
        self.connection_manager = SDEConnectionManager(instance, version)

        logger.debug(f"Initialized GeoCoverBridge with UUID field: {uuid_field}")

    def __repr__(self):
        return (f"<GeoCoverBridge: uuid={self.uuid_field}, "
                f"instance={self.connection_manager.instance}, "
                f"version={self.connection_manager.version_name}>")

    # Connection and Version Management ==========================================

    def find_user_version(self, interactive: bool = False):
        """
        Find and configure user's database version.

        Args:
            interactive: Enable interactive version selection

        Returns:
            Version information dictionary or None
        """
        return self.connection_manager.find_user_version(interactive)

    def get_versions(self) -> List[Dict]:
        """Get list of available database versions."""
        return self.connection_manager.get_versions()

    def connect(self, version: Optional[str] = None) -> str:
        """
        Connect to database and return workspace path.

        Args:
            version: Optional version override

        Returns:
            Workspace path string
        """
        return self.connection_manager.connect(version)

    @property
    def is_writable(self) -> bool:
        """Check if current version allows write operations."""
        return self.connection_manager.is_writable

    @property
    def workspace(self) -> Optional[str]:
        """Get current workspace path."""
        return self.connection_manager.workspace

    @property
    def version_info(self) -> Dict[str, any]:
        """Get current version information."""
        return self.connection_manager.version_info

    # Feature Class and Layer Management =====================================

    def get_feature_classes(self, ignore_integration: bool = True) -> List[str]:
        """
        Get list of available feature classes.

        Args:
            ignore_integration: Skip integration tables (_I suffix)

        Returns:
            List of feature class paths
        """
        return self.connection_manager.get_feature_classes(ignore_integration)

    def resolve_feature_class(self, name_or_shortcut: str) -> str:
        """
        Resolve feature class name from shortcut or full path.

        Args:
            name_or_shortcut: Feature class name, shortcut, or full path

        Returns:
            Full feature class path

        Examples:
            >>> bridge.resolve_feature_class("bedrock")
            "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK"
        """
        if name_or_shortcut in FEAT_CLASSES_SHORTNAMES:
            return FEAT_CLASSES_SHORTNAMES[name_or_shortcut]
        return name_or_shortcut

    def create_feature_layer(self, feature_class: str, layer_name: Optional[str] = None) -> str:
        """
        Create temporary feature layer from feature class.

        Args:
            feature_class: Feature class path
            layer_name: Optional layer name (auto-generated if None)

        Returns:
            Feature layer name
        """
        logger.debug(f"Creating feature layer: {feature_class}")

        if layer_name is None:
            timestamp = dt.now().strftime("%Y%m%d%H%M%S")
            layer_name = f"temp_layer_{timestamp}"

        # Normalize path separators for ArcGIS
        feature_class = feature_class.replace("/", "\\")

        arcpy.management.MakeFeatureLayer(feature_class, layer_name)
        return layer_name

    def _get_layer_fields(self, layer_or_fc: str) -> List[str]:
        """
        Get all field names from layer or feature class.

        Args:
            layer_or_fc: Layer name or feature class path

        Returns:
            List of field names
        """
        try:
            desc = arcpy.Describe(layer_or_fc)
            if hasattr(desc, "fields"):
                return [field.name for field in desc.fields]
            logger.warning(f"No fields found for: {layer_or_fc}")
        except Exception as e:
            logger.error(f"Error getting fields for {layer_or_fc}: {e}")
        return []

    # Data Export Operations ==================================================

    def to_geopandas(
            self,
            feature_class: str,
            where_clause: Optional[str] = None,
            fields: Optional[List[str]] = None,
            spatial_filter: Optional[arcpy.Polygon] = None,
            max_features: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """
        Export feature class data to GeoDataFrame.

        Args:
            feature_class: Feature class name/path or shortcut
            where_clause: SQL WHERE clause for filtering
            fields: Specific fields to include (None = all fields)
            spatial_filter: Spatial geometry filter
            max_features: Maximum number of features to export

        Returns:
            GeoDataFrame with exported features

        Examples:
            # Export all bedrock features
            gdf = bridge.to_geopandas("bedrock")

            # Export with filters
            gdf = bridge.to_geopandas(
                "bedrock",
                where_clause="OPERATOR = 'USER1'",
                fields=["UUID", "MORE_INFO"],
                max_features=1000
            )
        """
        feature_class = self.resolve_feature_class(feature_class)
        logger.info(f"Exporting {feature_class} to GeoDataFrame")

        layer_name = self.create_feature_layer(feature_class)

        try:
            # Get and validate fields
            db_fields = [f for f in self._get_layer_fields(feature_class)
                         if not f.startswith("SHAPE")]

            if self.uuid_field not in db_fields:
                raise ValueError(f"UUID field '{self.uuid_field}' not found in layer")

            # Determine fields to export
            if fields is None:
                fields = db_fields
            elif self.uuid_field not in fields:
                fields = [self.uuid_field] + fields

            # Validate requested fields
            invalid_fields = set(fields) - set(db_fields)
            if invalid_fields:
                raise ValueError(f"Invalid fields: {invalid_fields}")

            # Prepare cursor fields (geometry + attributes)
            cursor_fields = ["SHAPE@WKT"] + [f for f in fields if f != "SHAPE"]

            # Build WHERE clause
            clauses = []
            if where_clause:
                clauses.append(f"({where_clause})")

            # Check for layer definition query
            layer_desc = arcpy.Describe(layer_name)
            if hasattr(layer_desc, 'definitionQuery') and layer_desc.definitionQuery:
                clauses.append(f"({layer_desc.definitionQuery})")

            final_where = " AND ".join(clauses) if clauses else None

            # Set up cursor parameters
            cursor_kwargs = {"where_clause": final_where}
            if spatial_filter:
                cursor_kwargs.update({
                    "spatial_filter": spatial_filter,
                    "spatial_reference": arcpy.SpatialReference(SWISS_EPSG),
                    "search_order": "SPATIALFIRST"
                })

            logger.info(f"Fields: {fields}")
            logger.info(f"WHERE clause: {final_where}")
            logger.debug(f"Cursor kwargs: {cursor_kwargs}")

            # Read data with progress tracking
            data = []
            geometries = []

            row_count = int(arcpy.management.GetCount(layer_name).getOutput(0))
            logger.info(f"Processing {row_count} features")

            # Rich progress bar for reading features
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Reading features..."),
                    BarColumn(bar_width=40),
                    TaskProgressColumn(),
                    TextColumn("features"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                    transient=False
            ) as progress:

                task = progress.add_task(
                    f"[cyan]Reading {feature_class}",
                    total=row_count
                )

                with arcpy.da.SearchCursor(layer_name, cursor_fields, **cursor_kwargs) as cursor:
                    for i, row in enumerate(cursor, 1):
                        if max_features and i > max_features:
                            logger.info(f"Reached max_features limit: {max_features}")
                            break

                        # Parse geometry and attributes
                        geom_wkt = row[0]
                        attributes = row[1:]

                        geometry = wkt.loads(geom_wkt)
                        geometries.append(geometry)
                        data.append(attributes)

                        # Update progress every 100 features or on last feature
                        if i % 100 == 0 or i == row_count:
                            progress.update(task, completed=min(i, max_features or row_count))

            # Create GeoDataFrame
            logger.info(f"Creating GeoDataFrame with {len(data)} features")
            df = pd.DataFrame(data, columns=fields)
            gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=f"EPSG:{SWISS_EPSG}")

            return gdf

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
        finally:
            if arcpy.Exists(layer_name):
                arcpy.management.Delete(layer_name)

    # Bulk CRUD Operations ===================================================

    def bulk_insert(
            self,
            gdf: gpd.GeoDataFrame,
            feature_class: str,
            ignore_duplicates: bool = True,
            dryrun: bool = False,
            operator: str = DEFAULT_OPERATOR,
    ) -> Dict[str, Union[int, List[str]]]:
        """
        Insert new features from GeoDataFrame.

        Args:
            gdf: GeoDataFrame with features to insert
            feature_class: Target feature class name/shortcut
            ignore_duplicates: Skip existing UUIDs instead of error
            dryrun: Test run without making changes
            operator: User identifier for audit trail

        Returns:
            Dictionary with success_count, errors, duplicates_skipped, sample_affected

        Examples:
            # Insert new features
            result = bridge.bulk_insert(new_features_gdf, "bedrock")
            print(f"Inserted {result['success_count']} features")
        """
        if not self.is_writable:
            raise ReadOnlyError(f"Version {self.connection_manager.version_name} is read-only")

        feature_class = self.resolve_feature_class(feature_class)

        if self.uuid_field not in gdf.columns:
            raise ValueError(f"UUID field '{self.uuid_field}' missing in GeoDataFrame")

        logger.info(f"Bulk inserting {len(gdf)} features to {feature_class}")

        # Check for duplicates
        existing_uuids = self._get_existing_uuids(feature_class)
        duplicates_mask = gdf[self.uuid_field].isin(existing_uuids)
        duplicates_count = duplicates_mask.sum()

        if duplicates_count > 0:
            if not ignore_duplicates:
                sample_dupes = gdf[duplicates_mask][self.uuid_field].head(3).tolist()
                raise ValueError(f"{duplicates_count} features already exist (e.g. {sample_dupes})")
            gdf = gdf[~duplicates_mask]

        if len(gdf) == 0:
            return {
                "success_count": 0,
                "errors": ["No new features to insert"],
                "duplicates_skipped": duplicates_count,
                "sample_affected": [],
            }

        # Prepare for insertion
        full_path = f"{self.workspace}/{feature_class}"
        db_fields = self._get_layer_fields(feature_class)

        # Determine fields to insert
        insert_fields = [f for f in gdf.columns if f in db_fields and f != self.uuid_field]
        cursor_fields = [self.uuid_field] + self.mandatory_fields + insert_fields

        if "geometry" in gdf:
            cursor_fields.append("SHAPE@")

        cursor_fields = list(set(cursor_fields))  # Remove duplicates

        errors = []
        success_count = 0

        try:
            with self.connection_manager.transaction():
                with arcpy.da.InsertCursor(full_path, cursor_fields) as cursor:
                    field_indices = {field: idx for idx, field in enumerate(cursor_fields)}

                    for _, row in gdf.iterrows():
                        try:
                            new_row = [None] * len(cursor_fields)

                            # Populate fields
                            for field in cursor_fields:
                                if field == "SHAPE@":
                                    geom = row.get("geometry")
                                    if geom is not None:
                                        new_row[field_indices[field]] = arcpy.FromWKT(
                                            geom.wkt, arcpy.SpatialReference(SWISS_EPSG)
                                        )
                                elif field == self.operator_field:
                                    new_row[field_indices[field]] = operator
                                elif field == self.date_field:
                                    new_row[field_indices[field]] = dt.now()
                                else:
                                    new_row[field_indices[field]] = row.get(field)

                            if not dryrun:
                                cursor.insertRow(new_row)
                            success_count += 1

                        except Exception as e:
                            error_msg = f"Error inserting {row[self.uuid_field]}: {str(e)}"
                            errors.append(error_msg)
                            logger.error(error_msg)

            logger.success(f"Inserted {success_count} features")

            return {
                "success_count": success_count,
                "errors": errors,
                "duplicates_skipped": duplicates_count,
                "sample_affected": gdf[self.uuid_field].head(5).tolist(),
            }

        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            return {
                "success_count": success_count,
                "errors": [f"Insert operation failed: {str(e)}"] + errors,
                "duplicates_skipped": duplicates_count,
                "sample_affected": [],
            }

    def bulk_update(
            self,
            gdf: gpd.GeoDataFrame,
            feature_class: str,
            update_fields: Optional[List[str]] = None,
            update_attributes: bool = True,
            update_geometry: bool = True,
            check_timestamp: bool = True,
            force_update: bool = False,
            dryrun: bool = False,
            operator: str = DEFAULT_OPERATOR,
    ) -> Dict[str, Union[int, List[str]]]:
        """
        Update existing features from GeoDataFrame.

        Args:
            gdf: GeoDataFrame with features to update
            feature_class: Target feature class name/shortcut
            update_fields: Specific fields to update (None = all matching)
            update_attributes: Whether to update attribute fields
            update_geometry: Whether to update geometries
            check_timestamp: Enable timestamp conflict checking
            force_update: Update regardless of timestamps
            dryrun: Test run without making changes
            operator: User identifier for audit trail

        Returns:
            Dictionary with success_count, skipped_newer, errors, sample_affected

        Examples:
            # Update all matching features
            result = bridge.bulk_update(modified_gdf, "bedrock")

            # Update only specific fields with force
            result = bridge.bulk_update(
                gdf, "bedrock",
                update_fields=["MORE_INFO"],
                force_update=True
            )
        """
        if not self.is_writable:
            raise ReadOnlyError(f"Version {self.connection_manager.version_name} is read-only")

        feature_class = self.resolve_feature_class(feature_class)

        if self.uuid_field not in gdf.columns:
            raise ValueError(f"UUID field '{self.uuid_field}' missing in GeoDataFrame")

        # Validate timestamp checking requirements
        if check_timestamp and not force_update and self.date_field not in gdf.columns:
            logger.warning(f"{self.date_field} not found in GeoDataFrame, disabling timestamp checking")
            check_timestamp = False

        logger.info(f"Bulk updating {len(gdf)} features in {feature_class}")

        # Get existing UUIDs
        existing_uuids = self._get_existing_uuids(feature_class)
        update_gdf = gdf[gdf[self.uuid_field].isin(existing_uuids)]

        if len(update_gdf) == 0:
            return {
                "success_count": 0,
                "skipped_newer": 0,
                "errors": ["No matching features to update"],
                "sample_affected": [],
            }

        # Determine fields to update
        db_fields = self._get_layer_fields(feature_class)

        if update_fields is None:
            fields_to_update = [f for f in update_gdf.columns
                                if f in db_fields and f != self.uuid_field]
        else:
            invalid_fields = [f for f in update_fields
                              if f not in db_fields or f not in update_gdf.columns]
            if invalid_fields:
                raise ValueError(f"Invalid fields: {invalid_fields}")
            fields_to_update = update_fields

        if not update_attributes:
            fields_to_update = []

        # Prepare cursor fields
        cursor_fields = [self.uuid_field] + self.mandatory_fields + fields_to_update

        if update_geometry and "geometry" in update_gdf:
            cursor_fields.extend(["SHAPE@", "SHAPE.AREA"])

        cursor_fields = list(set(cursor_fields))  # Remove duplicates

        # Process in chunks
        uuid_chunks = self._chunk_list(update_gdf[self.uuid_field].tolist())
        gdf_dict = update_gdf.set_index(self.uuid_field).to_dict("index")

        success_count = 0
        skipped_newer = 0
        errors = []
        full_path = f"{self.workspace}/{feature_class}"

        try:
            with self.connection_manager.transaction():
                for chunk in uuid_chunks:
                    where_clause = self._build_uuid_where_clause(chunk)

                    with arcpy.da.UpdateCursor(full_path, cursor_fields, where_clause) as cursor:
                        field_indices = {field: idx for idx, field in enumerate(cursor_fields)}

                        for row in cursor:
                            try:
                                uuid_val = row[field_indices[self.uuid_field]]

                                if uuid_val not in gdf_dict:
                                    continue

                                gdf_row = gdf_dict[uuid_val]

                                # Check timestamp if enabled
                                if check_timestamp and not force_update:
                                    skip_reason = self._should_skip_update(
                                        row, field_indices, gdf_row
                                    )
                                    if skip_reason:
                                        logger.debug(f"Skipping {uuid_val}: {skip_reason}")
                                        skipped_newer += 1
                                        continue

                                # Update row
                                updated_row = list(row)

                                # Update regular fields
                                for field in fields_to_update:
                                    if field in field_indices:
                                        updated_row[field_indices[field]] = gdf_row.get(field)

                                # Update geometry
                                if update_geometry and "SHAPE@" in field_indices:
                                    geom = gdf_row.get("geometry")
                                    if geom is not None:
                                        updated_row[field_indices["SHAPE@"]] = arcpy.FromWKT(
                                            geom.wkt, arcpy.SpatialReference(SWISS_EPSG)
                                        )
                                        if "SHAPE.AREA" in field_indices:
                                            updated_row[field_indices["SHAPE.AREA"]] = geom.area

                                # Update mandatory fields
                                updated_row[field_indices[self.operator_field]] = operator
                                updated_row[field_indices[self.date_field]] = dt.now()

                                if not dryrun:
                                    cursor.updateRow(updated_row)
                                success_count += 1

                            except Exception as e:
                                error_msg = f"Error updating {row[0]}: {str(e)}"
                                errors.append(error_msg)
                                logger.error(error_msg)

            logger.success(f"Updated {success_count} features, skipped {skipped_newer}")

            return {
                "success_count": success_count,
                "skipped_newer": skipped_newer,
                "errors": errors,
                "sample_affected": update_gdf[self.uuid_field].head(5).tolist(),
            }

        except Exception as e:
            logger.error(f"Bulk update failed: {e}")
            return {
                "success_count": success_count,
                "skipped_newer": skipped_newer,
                "errors": [f"Update operation failed: {str(e)}"] + errors,
                "sample_affected": [],
            }

    def bulk_delete(
            self,
            gdf: gpd.GeoDataFrame,
            feature_class: str,
            check_timestamp: bool = True,
            force_delete: bool = False,
            confirm: bool = True,
            dryrun: bool = False,
    ) -> Dict[str, Union[int, List[str]]]:
        """
        Delete features based on UUIDs in GeoDataFrame.

        Args:
            gdf: GeoDataFrame with UUIDs of features to delete
            feature_class: Target feature class name/shortcut
            check_timestamp: Enable timestamp conflict checking
            force_delete: Delete regardless of timestamps
            confirm: Require interactive confirmation
            dryrun: Test run without actual deletion

        Returns:
            Dictionary with success_count, skipped_newer, errors, sample_affected

        Examples:
            # Delete features with confirmation
            result = bridge.bulk_delete(features_to_delete, "bedrock")

            # Force delete without confirmation
            result = bridge.bulk_delete(
                gdf, "bedrock",
                force_delete=True,
                confirm=False
            )
        """
        if not self.is_writable:
            raise ReadOnlyError(f"Version {self.connection_manager.version_name} is read-only")

        feature_class = self.resolve_feature_class(feature_class)

        if self.uuid_field not in gdf.columns:
            raise ValueError(f"UUID field '{self.uuid_field}' missing in GeoDataFrame")

        # Validate timestamp checking
        if check_timestamp and not force_delete and self.date_field not in gdf.columns:
            logger.warning(f"{self.date_field} not found, disabling timestamp checking")
            check_timestamp = False

        delete_uuids = gdf[self.uuid_field].unique()
        logger.info(f"Bulk deleting {len(delete_uuids)} features from {feature_class}")

        # Confirmation dialog
        if confirm and not dryrun:
            print(f"About to delete {len(delete_uuids)} features from {feature_class}")
            print("Sample UUIDs:", list(delete_uuids)[:5])
            if force_delete:
                print("⚠️  FORCE DELETE enabled - ignoring timestamps")
            if not input("Confirm deletion? (y/n): ").lower().startswith("y"):
                return {
                    "success_count": 0,
                    "skipped_newer": 0,
                    "errors": ["Deletion canceled by user"],
                    "sample_affected": [],
                }

        # Check existence
        existing_uuids = self._get_existing_uuids(feature_class)
        valid_uuids = list(set(delete_uuids) & existing_uuids)

        if not valid_uuids:
            return {
                "success_count": 0,
                "skipped_newer": 0,
                "errors": ["No matching features to delete"],
                "sample_affected": [],
            }

        # Prepare for deletion
        gdf_dict = {}
        if check_timestamp and not force_delete:
            gdf_dict = gdf.set_index(self.uuid_field).to_dict("index")

        uuid_chunks = self._chunk_list(valid_uuids)
        success_count = 0
        skipped_newer = 0
        errors = []
        full_path = f"{self.workspace}/{feature_class}"

        try:
            with self.connection_manager.transaction():
                for chunk in uuid_chunks:
                    where_clause = self._build_uuid_where_clause(chunk)

                    # Determine fields needed for timestamp checking
                    fields = [self.uuid_field]
                    if check_timestamp and not force_delete:
                        fields.append(self.date_field)

                    with arcpy.da.UpdateCursor(full_path, fields, where_clause) as cursor:
                        for row in cursor:
                            try:
                                uuid_val = row[0]

                                # Check timestamp if enabled
                                if check_timestamp and not force_delete:
                                    sde_date = row[1] if len(row) > 1 else None
                                    skip_reason = self._should_skip_delete(
                                        uuid_val, sde_date, gdf_dict.get(uuid_val, {})
                                    )
                                    if skip_reason:
                                        logger.debug(f"Skipping delete {uuid_val}: {skip_reason}")
                                        skipped_newer += 1
                                        continue

                                if not dryrun:
                                    cursor.deleteRow()
                                success_count += 1

                            except Exception as e:
                                error_msg = f"Error deleting {row[0]}: {str(e)}"
                                errors.append(error_msg)
                                logger.error(error_msg)

            logger.success(f"Deleted {success_count} features, skipped {skipped_newer}")

            return {
                "success_count": success_count,
                "skipped_newer": skipped_newer,
                "errors": errors,
                "sample_affected": list(valid_uuids)[:5],
            }

        except Exception as e:
            logger.error(f"Bulk delete failed: {e}")
            return {
                "success_count": success_count,
                "skipped_newer": skipped_newer,
                "errors": [f"Delete operation failed: {str(e)}"] + errors,
                "sample_affected": [],
            }

    def execute_operations(
            self,
            gdf: gpd.GeoDataFrame,
            feature_class: str,
            operation_column: str = "_operation",
            default_operation: Optional[str] = None,
            update_fields: Optional[List[str]] = None,
            update_attributes: bool = True,
            update_geometry: bool = True,
            confirm_deletes: bool = True,
            dryrun: bool = False,
            operator: str = DEFAULT_OPERATOR,
    ) -> Dict[str, Dict[str, Union[int, List[str]]]]:
        """
        Execute mixed CRUD operations from a single GeoDataFrame.

        Features are processed based on the operation column which should contain
        values: 'insert', 'update', 'delete', or None/NaN (skipped).

        Args:
            gdf: GeoDataFrame with operation column
            feature_class: Target feature class name/shortcut
            operation_column: Column name specifying operations
            default_operation: Default operation for rows without specified operation
            update_fields: Fields to update (for update operations)
            update_attributes: Whether to update attributes
            update_geometry: Whether to update geometries
            confirm_deletes: Require confirmation for delete operations
            dryrun: Test run without making changes
            operator: User identifier for audit trail

        Returns:
            Dictionary with results for each operation type

        Examples:
            # Process mixed operations
            gdf['_operation'] = ['insert', 'update', 'delete', None, ...]
            results = bridge.execute_operations(gdf, "bedrock")

            for op, result in results.items():
                print(f"{op}: {result['success_count']} successful")
        """
        if not self.is_writable:
            raise ReadOnlyError(f"Version {self.connection_manager.version_name} is read-only")

        feature_class = self.resolve_feature_class(feature_class)

        # Validate operation column
        if operation_column not in gdf.columns:
            if default_operation is None:
                raise ValueError(
                    f"Operation column '{operation_column}' not found "
                    "and no default operation specified"
                )
            gdf = gdf.copy()
            gdf[operation_column] = default_operation

        # Validate operations
        valid_ops = {"insert", "update", "delete", None}
        actual_ops = set(gdf[operation_column].dropna().unique())
        invalid_ops = actual_ops - valid_ops
        if invalid_ops:
            raise ValueError(f"Invalid operations: {invalid_ops}")

        logger.info(f"Executing mixed operations on {feature_class}")
        operation_counts = gdf[operation_column].value_counts(dropna=False)
        logger.info(f"Operation breakdown: {operation_counts.to_dict()}")

        results = {}

        try:
            with self.connection_manager.transaction():
                # Process deletes first (safest order)
                if "delete" in gdf[operation_column].values:
                    delete_gdf = gdf[gdf[operation_column] == "delete"]
                    results["delete"] = self.bulk_delete(
                        delete_gdf, feature_class, confirm=confirm_deletes, dryrun=dryrun
                    )

                # Process updates
                if "update" in gdf[operation_column].values:
                    update_gdf = gdf[gdf[operation_column] == "update"]
                    results["update"] = self.bulk_update(
                        update_gdf, feature_class,
                        update_fields=update_fields,
                        update_attributes=update_attributes,
                        update_geometry=update_geometry,
                        dryrun=dryrun,
                        operator=operator,
                    )

                # Process inserts last
                if "insert" in gdf[operation_column].values:
                    insert_gdf = gdf[gdf[operation_column] == "insert"]
                    results["insert"] = self.bulk_insert(
                        insert_gdf, feature_class, dryrun=dryrun, operator=operator
                    )

                # Handle skipped rows
                skipped_mask = gdf[operation_column].isna()
                if skipped_mask.any():
                    results["skipped"] = {
                        "count": skipped_mask.sum(),
                        "sample": gdf[skipped_mask][self.uuid_field].head(5).tolist(),
                    }

            logger.success("Mixed operations completed successfully")
            return results

        except Exception as e:
            logger.error(f"Mixed operations failed: {e}")
            return {"error": str(e), "partial_results": results}

    # GPKG Export Operations ==================================================

    def save_to_gpkg(
            self,
            gdf: gpd.GeoDataFrame,
            output_path: Union[str, Path],
            layer_name: str,
            chunk_size: Optional[int] = None,
            parallel: Optional[bool] = None,
            mode: Literal["w", "a"] = "w",
            show_progress: bool = True,
            compression: bool = True,
    ) -> Path:
        """
        Save GeoDataFrame to GPKG with progress feedback.

        Args:
            gdf: GeoDataFrame to save
            output_path: Output GPKG file path
            layer_name: Layer name in GPKG
            chunk_size: Number of features per chunk (None = use default)
            parallel: Use parallel processing (None = auto-detect)
            mode: Write mode ('w' = overwrite, 'a' = append)
            show_progress: Show progress bars
            compression: Use GPKG compression

        Returns:
            Path to saved file

        Examples:
            # Save exported data
            gdf = bridge.to_geopandas("bedrock")
            bridge.save_to_gpkg(gdf, "bedrock_export.gpkg", "bedrock")

            # Large dataset with custom settings
            bridge.save_to_gpkg(
                large_gdf, "large_export.gpkg", "data",
                chunk_size=500, parallel=True
            )
        """
        if chunk_size is None:
            chunk_size = DEFAULT_CHUNK_SIZE

        # Estimate file size for user info
        if show_progress and len(gdf) > 1000:
            estimated_size, readable_size = estimate_gpkg_size(gdf, compression)
            logger.info(f"Estimated GPKG size: {readable_size}")

        return write_gdf_to_gpkg(
            gdf=gdf,
            output_path=output_path,
            layer_name=layer_name,
            chunk_size=chunk_size,
            parallel=parallel,
            mode=mode,
            show_progress=show_progress,
            compression=compression,
        )

    def export_to_gpkg(
            self,
            feature_class: str,
            output_path: Union[str, Path],
            layer_name: Optional[str] = None,
            where_clause: Optional[str] = None,
            fields: Optional[List[str]] = None,
            spatial_filter: Optional[arcpy.Polygon] = None,
            max_features: Optional[int] = None,
            chunk_size: Optional[int] = None,
            parallel: Optional[bool] = None,
            show_progress: bool = True,
            compression: bool = True,
    ) -> Path:
        """
        Export feature class directly to GPKG with progress feedback.

        Combines to_geopandas() and save_to_gpkg() in a single operation
        with optimized memory usage for large datasets.

        Args:
            feature_class: Feature class name/shortcut to export
            output_path: Output GPKG file path
            layer_name: Layer name in GPKG (None = use feature class shortcut)
            where_clause: SQL WHERE clause for filtering
            fields: Specific fields to include
            spatial_filter: Spatial geometry filter
            max_features: Maximum number of features to export
            chunk_size: Chunk size for GPKG writing
            parallel: Use parallel GPKG writing
            show_progress: Show progress bars
            compression: Use GPKG compression

        Returns:
            Path to exported GPKG file

        Examples:
            # Simple export
            bridge.export_to_gpkg("bedrock", "bedrock_export.gpkg")

            # Filtered export with custom settings
            bridge.export_to_gpkg(
                "bedrock", "filtered_bedrock.gpkg",
                where_clause="OPERATOR = 'USER1'",
                fields=["UUID", "MORE_INFO"],
                max_features=10000,
                parallel=True
            )
        """
        # Use feature class shortcut as default layer name
        if layer_name is None:
            # Try to get shortcut name, fallback to last part of path
            shortcut = None
            full_path = self.resolve_feature_class(feature_class)
            for name, path in FEAT_CLASSES_SHORTNAMES.items():
                if path == full_path:
                    shortcut = name
                    break
            layer_name = shortcut or full_path.split("/")[-1]

        logger.info(f"Exporting {feature_class} to {output_path}:{layer_name}")

        # Export to GeoDataFrame
        gdf = self.to_geopandas(
            feature_class=feature_class,
            where_clause=where_clause,
            fields=fields,
            spatial_filter=spatial_filter,
            max_features=max_features,
        )

        # Save to GPKG
        return self.save_to_gpkg(
            gdf=gdf,
            output_path=output_path,
            layer_name=layer_name,
            chunk_size=chunk_size,
            parallel=parallel,
            mode="w",
            show_progress=show_progress,
            compression=compression,
        )

    # Helper Methods ==========================================================

    def _get_existing_uuids(self, feature_class: str) -> Set[str]:
        """Get set of existing UUIDs in feature class."""
        full_path = f"{self.workspace}/{feature_class}"

        try:
            with arcpy.da.SearchCursor(full_path, [self.uuid_field]) as cursor:
                return {row[0] for row in cursor}
        except Exception as e:
            logger.warning(f"Could not get existing UUIDs: {e}")
            return set()

    def _chunk_list(self, items: List, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[List]:
        """Split list into chunks."""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _build_uuid_where_clause(self, uuids: List[str]) -> str:
        """Build WHERE clause for UUID list."""
        quoted_uuids = [f"'{uuid}'" for uuid in uuids]
        return f"{self.uuid_field} IN ({','.join(quoted_uuids)})"

    def _should_skip_update(self, row: List, field_indices: Dict, gdf_row: Dict) -> Optional[str]:
        """Check if update should be skipped based on timestamps."""
        if self.date_field not in field_indices:
            return None

        sde_date = row[field_indices[self.date_field]]
        gdf_date = gdf_row.get(self.date_field)

        if sde_date is None or gdf_date is None:
            return None

        try:
            sde_datetime = pd.to_datetime(sde_date)
            gdf_datetime = pd.to_datetime(gdf_date)

            if sde_datetime > gdf_datetime:
                return f"SDE timestamp ({sde_datetime}) newer than GDF ({gdf_datetime})"
        except Exception as e:
            logger.warning(f"Could not compare timestamps: {e}")

        return None

    def _should_skip_delete(self, uuid_val: str, sde_date, gdf_row: Dict) -> Optional[str]:
        """Check if delete should be skipped based on timestamps."""
        if not gdf_row or self.date_field not in gdf_row:
            return None

        if sde_date is None:
            return None

        gdf_date = gdf_row.get(self.date_field)
        if gdf_date is None:
            return None

        try:
            sde_datetime = pd.to_datetime(sde_date)
            gdf_datetime = pd.to_datetime(gdf_date)

            if sde_datetime > gdf_datetime:
                return f"SDE timestamp ({sde_datetime}) newer than GDF ({gdf_datetime})"
        except Exception as e:
            logger.warning(f"Could not compare timestamps: {e}")

        return None

    # Context Manager =========================================================

    def __enter__(self):
        """Context manager entry - find version and connect."""
        self.find_user_version()
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        self.connection_manager.cleanup()