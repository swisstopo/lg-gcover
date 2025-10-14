# gcover/sde/bridge.py
"""
GCover SDE Bridge - High-level interface for geodata import/export with ESRI Enterprise Geodatabase
"""

import contextlib
import os
import traceback
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd
from loguru import logger
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TaskID, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)

from gcover.utils.console import console


try:
    import arcpy
except ImportError:
    logger.warning("arcpy not available - SDE bridge functionality disabled")
    arcpy = None

from .connection_manager import SDEConnectionManager


class ReadOnlyError(Exception):
    """Raised when attempting write operations on read-only versions."""

    pass


# Feature class shortcuts mapping
FEATURE_CLASS_SHORTCUTS = {
    "bedrock": "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_BEDROCK",
    "unco": "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_UNCO_DESPOSIT",
    "points": "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_POINT_OBJECTS",
    "surfaces": "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_SURFACES",
    "lines": "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_LINEAR_OBJECTS",
    "fossils": "TOPGIS_GC.GC_ROCK_BODIES/TOPGIS_GC.GC_FOSSILS",
}


def resolve_feature_class(name: str) -> str:
    """Resolve feature class name from shortcut or return full path."""
    return FEATURE_CLASS_SHORTCUTS.get(name.lower(), name)


class GCoverSDEBridge:
    """
    High-level bridge for importing/exporting geodata to/from ESRI Enterprise Geodatabase.

    Handles CRUD operations between GeoPandas/file formats and ArcSDE feature classes.
    Uses SDEConnectionManager for connection lifecycle management.
    """

    DEFAULT_OPERATOR = "GC_Bridge"
    DEFAULT_VERSION_TYPE = "user_writable"  # user_writable, user_any, default

    def __init__(
        self,
        instance: str = "GCOVERP",
        version: Optional[str] = None,
        version_type: str = "user_writable",
        uuid_field: str = "UUID",
        connection_manager: Optional[SDEConnectionManager] = None,
        show_progress: bool = True,
    ):
        """
        Initialize SDE Bridge.

        Args:
            instance: SDE instance name (default: GCOVERP)
            version: Specific version name (auto-detected if None)
            version_type: Type of version to find if version is None
            uuid_field: Primary key field name for feature matching
            connection_manager: Optional external connection manager
            show_progress: Whether to show progress bars for operations
        """
        if arcpy is None:
            raise ImportError("arcpy is required for SDE operations")

        self.instance = instance
        self.uuid_field = uuid_field
        self.version_type = version_type
        self._requested_version = version
        self.show_progress = show_progress

        # Connection management
        self._external_conn_mgr = connection_manager is not None
        self.conn_mgr = connection_manager or SDEConnectionManager()

        # Will be set during connection
        self.workspace = None
        self.version_name = None
        self.is_writable = False
        self._version_info = None

        # Mandatory fields for write operations
        self.mandatory_fields = ["OPERATOR", "DATEOFCHANGE"]

    def __enter__(self):
        """Context manager entry - establish SDE connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup connections."""
        if not self._external_conn_mgr:
            self.conn_mgr.cleanup_all()

    def connect(self) -> None:
        """Establish connection to SDE and determine target version."""
        version = self._requested_version or self._find_target_version()

        if not version:
            raise ValueError(f"No suitable version found for {self.instance}")

        # Create SDE connection
        sde_path = self.conn_mgr.create_connection(self.instance, version)

        # Set arcpy workspace
        self.workspace = str(sde_path)
        arcpy.env.workspace = self.workspace

        # Store version info
        self.version_name = version
        self._version_info = self._get_version_info(version)
        self.is_writable = self._version_info.get("writable", False)

        logger.info(
            f"Connected to {self.instance}::{version} (writable: {self.is_writable})"
        )

    def _find_target_version(self) -> Optional[str]:
        """Find appropriate version based on version_type."""
        try:
            versions = self.conn_mgr.get_versions(self.instance)
            current_user = os.getlogin().upper()

            if self.version_type == "default":
                return "SDE.DEFAULT"

            elif self.version_type == "user_writable":
                for version in versions:
                    if current_user in version["name"].upper() and version.get(
                        "writable", False
                    ):
                        return version["name"]

            elif self.version_type == "user_any":
                for version in versions:
                    if current_user in version["name"].upper():
                        return version["name"]

            # Fallback to DEFAULT
            logger.warning(f"No {self.version_type} version found, using SDE.DEFAULT")
            return "SDE.DEFAULT"

        except Exception as e:
            logger.error(f"Error finding target version: {e}")
            return None

    def _get_version_info(self, version_name: str) -> Dict[str, Any]:
        """Get detailed version information."""
        try:
            versions = self.conn_mgr.get_versions(self.instance)
            for version in versions:
                if version["name"] == version_name:
                    return version
            return {"writable": False}
        except Exception:
            return {"writable": False}

    @property
    def rc_full(self) -> str:
        """Get full RC version from version name (e.g., '2030-12-31')."""
        if "2030-12-31" in self.version_name:
            return "2030-12-31"
        elif "2016-12-31" in self.version_name:
            return "2016-12-31"
        return "unknown"

    @property
    def rc_short(self) -> str:
        """Get short RC version (e.g., 'RC2' for 2030-12-31)."""
        rc_map = {"2030-12-31": "RC2", "2016-12-31": "RC1"}
        return rc_map.get(self.rc_full, "RC?")

    def _get_feature_count(
        self, feature_class: str, where_clause: Optional[str] = None
    ) -> int:
        """Get approximate feature count for progress tracking."""
        full_path = f"{self.workspace}/{feature_class}"
        try:
            if where_clause:
                # Use management tool for filtered count (may be slow but accurate)
                layer_name = f"temp_layer_{dt.now().strftime('%H%M%S')}"
                arcpy.management.MakeFeatureLayer(full_path, layer_name, where_clause)
                count = int(arcpy.management.GetCount(layer_name).getOutput(0))
                arcpy.management.Delete(layer_name)
                return count
            else:
                # Fast count for entire feature class
                return int(arcpy.management.GetCount(full_path).getOutput(0))
        except Exception as e:
            logger.warning(f"Could not get feature count: {e}")
            return 0

    # =============================================================================
    # GEODATA IMPORT/EXPORT METHODS
    # =============================================================================

    def export_to_geodataframe(
        self,
        feature_class: str,
        where_clause: Optional[str] = None,
        bbox: Optional[tuple] = None,
        fields: Optional[List[str]] = None,
        spatial_filter: Optional[Any] = None,
        max_features: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> gpd.GeoDataFrame:
        """
        Export SDE feature class to GeoPandas GeoDataFrame with progress tracking.

        Args:
            feature_class: Feature class path
            where_clause: SQL WHERE clause for attribute filtering
            bbox: Bounding box (minx, miny, maxx, maxy) for spatial filtering
            fields: Specific fields to export (None = all fields)
            spatial_filter: ESRI geometry object for spatial filtering
            max_features: Maximum number of features to export
            progress_callback: Optional callback function(current, total)

        Returns:
            GeoDataFrame with feature class data
        """
        full_path = f"{self.workspace}/{feature_class}"

        if not arcpy.Exists(full_path):
            raise ValueError(f"Feature class not found: {full_path}")

        # Determine fields to export
        if fields is None:
            fields = [
                f.name
                for f in arcpy.ListFields(full_path)
                if f.type not in ("OID", "Geometry")
            ]

        # Add geometry field
        cursor_fields = fields + ["SHAPE@"]

        # Build search cursor parameters
        cursor_kwargs = {"field_names": cursor_fields}
        if where_clause:
            cursor_kwargs["where_clause"] = where_clause
        if spatial_filter:
            cursor_kwargs["spatial_reference"] = spatial_filter

        # Get feature count for progress tracking
        total_features = self._get_feature_count(feature_class, where_clause)
        if max_features:
            total_features = min(total_features, max_features)

        # Read data with progress tracking
        data = []
        geometries = []

        logger.info(f"Exporting {feature_class} with fields: {fields}")

        # Setup progress tracking
        progress = None
        task = None
        if (
            self.show_progress and total_features > 100
        ):  # Only show for substantial datasets
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("â€¢"),
                TimeElapsedColumn(),
                TextColumn("â€¢"),
                TimeRemainingColumn(),
            )
            progress.start()
            task = progress.add_task(
                f"Exporting {Path(feature_class).name}", total=total_features
            )

        try:
            with arcpy.da.SearchCursor(full_path, **cursor_kwargs) as cursor:
                for i, row in enumerate(cursor):
                    if max_features and i >= max_features:
                        break

                    # Extract geometry
                    esri_geom = row[-1]  # SHAPE@ is always last
                    if esri_geom:
                        try:
                            wkt = esri_geom.WKT
                            from shapely import wkt as shapely_wkt

                            geometries.append(shapely_wkt.loads(wkt))
                        except Exception as e:
                            logger.warning(
                                f"Error converting geometry for row {i}: {e}"
                            )
                            geometries.append(None)
                    else:
                        geometries.append(None)

                    # Extract attributes
                    data.append(dict(zip(fields, row[:-1])))

                    # Update progress
                    if progress and task:
                        progress.update(task, completed=i + 1)

                    # Call custom progress callback if provided
                    if progress_callback:
                        progress_callback(i + 1, total_features)

        finally:
            if progress:
                progress.stop()

        if not data:
            logger.warning(f"No features exported from {feature_class}")
            return gpd.GeoDataFrame(columns=fields + ["geometry"])

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:2056")

        # Apply bbox filter if specified (post-processing)
        if bbox and not gdf.empty:
            minx, miny, maxx, maxy = bbox
            gdf = gdf.cx[minx:maxx, miny:maxy]

        logger.info(f"Exported {len(gdf)} features from {feature_class}")
        return gdf

    def import_from_geodataframe(
        self,
        gdf: gpd.GeoDataFrame,
        feature_class: str,
        operation: str = "update",
        update_fields: Optional[List[str]] = None,
        update_attributes: bool = True,
        update_geometry: bool = True,
        ignore_duplicates: bool = True,
        dryrun: bool = False,
        operator: Optional[str] = None,
        chunk_size: int = 1000,
    ) -> Dict[str, Union[int, List[str]]]:
        """
        Import GeoPandas GeoDataFrame to SDE feature class.

        Args:
            gdf: Source GeoDataFrame
            feature_class: Target feature class path
            operation: Operation type ('update', 'insert', 'delete', 'upsert')
            update_fields: Specific fields to update (None = smart selection)
            update_attributes: Whether to update attribute fields
            update_geometry: Whether to update geometries
            ignore_duplicates: Skip existing features for insert operations
            dryrun: Test run without making changes
            operator: User making changes (defaults to class default)
            chunk_size: Number of features per transaction chunk

        Returns:
            Dictionary with operation results
        """
        if not self.is_writable:
            raise ReadOnlyError(f"Version {self.version_name} is read-only")

        if self.uuid_field not in gdf.columns:
            raise ValueError(f"UUID field '{self.uuid_field}' missing in GeoDataFrame")

        # Delegate to specific operation method
        if operation == "update":
            return self._bulk_update(
                gdf,
                feature_class,
                update_fields,
                update_attributes,
                update_geometry,
                dryrun,
                operator,
                chunk_size,
            )
        elif operation == "insert":
            return self._bulk_insert(
                gdf, feature_class, ignore_duplicates, dryrun, operator
            )
        elif operation == "delete":
            return self._bulk_delete(gdf, feature_class, dryrun)
        elif operation == "upsert":
            return self._bulk_upsert(
                gdf,
                feature_class,
                update_fields,
                update_attributes,
                update_geometry,
                dryrun,
                operator,
                chunk_size,
            )
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def import_from_file(
        self,
        file_path: Union[str, Path],
        feature_class: str,
        layer: Optional[str] = None,
        operation: str = "update",
        **kwargs,
    ) -> Dict[str, Union[int, List[str]]]:
        """
        Import geodata from file (GPKG, GeoJSON, Shapefile, etc.) to SDE.

        Args:
            file_path: Path to source file
            feature_class: Target feature class path
            layer: Layer name for multi-layer files (e.g., GPKG)
            operation: Operation type ('update', 'insert', 'delete', 'upsert')
            **kwargs: Additional arguments passed to import_from_geodataframe

        Returns:
            Dictionary with operation results
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file using geopandas
        read_kwargs = {}
        if layer:
            read_kwargs["layer"] = layer

        try:
            gdf = gpd.read_file(file_path, **read_kwargs)
        except Exception as e:
            raise ValueError(f"Error reading {file_path}: {e}")

        logger.info(f"Read {len(gdf)} features from {file_path}")

        if layer:
            logger.info(f"Using layer: {layer}")

        # Import to SDE
        return self.import_from_geodataframe(
            gdf, feature_class, operation=operation, **kwargs
        )

    def export_to_file(
        self,
        feature_class: str,
        output_path: Union[str, Path],
        layer_name: Optional[str] = None,
        driver: str = "GPKG",
        **export_kwargs,
    ) -> Path:
        """
        Export SDE feature class to file with progress tracking.

        Args:
            feature_class: Source feature class path
            output_path: Output file path
            layer_name: Layer name for output (defaults to feature class name)
            driver: Output driver (GPKG, GeoJSON, ESRI Shapefile, etc.)
            **export_kwargs: Additional arguments for export_to_geodataframe

        Returns:
            Path to created file
        """
        output_path = Path(output_path)

        # Export to GeoDataFrame with progress
        gdf = self.export_to_geodataframe(feature_class, **export_kwargs)

        if gdf.empty:
            logger.warning(f"No data to export from {feature_class}")

        # Determine layer name
        if layer_name is None:
            layer_name = Path(feature_class).name

        # Save to file with progress for large datasets
        save_kwargs = {"driver": driver}
        if driver == "GPKG":
            save_kwargs["layer"] = layer_name
            save_kwargs["engine"] = "pyogrio"  # Better GPKG support

        if self.show_progress and len(gdf) > 1000:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Saving to {task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} features"),
            ) as progress:
                task = progress.add_task(str(output_path.name), total=len(gdf))

                # For very large datasets, save in chunks to show progress
                if len(gdf) > 10000:
                    chunk_size = 5000
                    for i in range(0, len(gdf), chunk_size):
                        chunk = gdf.iloc[i : i + chunk_size]
                        if i == 0:
                            # First chunk creates the file
                            chunk.to_file(output_path, **save_kwargs)
                        else:
                            # Subsequent chunks append (if driver supports it)
                            try:
                                chunk.to_file(output_path, mode="a", **save_kwargs)
                            except:
                                # If append not supported, just save everything at once
                                gdf.to_file(output_path, **save_kwargs)
                                break
                        progress.update(task, completed=min(i + chunk_size, len(gdf)))
                else:
                    gdf.to_file(output_path, **save_kwargs)
                    progress.update(task, completed=len(gdf))
        else:
            # Small datasets - save directly without progress
            gdf.to_file(output_path, **save_kwargs)

        logger.info(f"Exported {len(gdf)} features to {output_path}")
        return output_path

    # =============================================================================
    # INTERNAL CRUD IMPLEMENTATION
    # =============================================================================

    def _get_layer_fields(self, feature_class: str) -> List[str]:
        """Get list of field names for feature class."""
        feature_class = resolve_feature_class(feature_class)
        full_path = f"{self.workspace}/{feature_class}"
        return [f.name for f in arcpy.ListFields(full_path) if f.type not in ("OID",)]

    def _get_existing_uuids(self, feature_class: str) -> set:
        """Get set of existing UUIDs in feature class."""
        feature_class = resolve_feature_class(feature_class)
        full_path = f"{self.workspace}/{feature_class}"
        with arcpy.da.SearchCursor(full_path, [self.uuid_field]) as cursor:
            return {row[0] for row in cursor}

    def _chunk_data(self, data: list, chunk_size: int = 1000) -> list:
        """Split data into chunks for processing."""
        return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

    @contextlib.contextmanager
    def _edit_session(self):
        """Context manager for ArcSDE edit session."""
        edit = arcpy.da.Editor(self.workspace)
        edit.startEditing(False, True)  # undoable, multiuser_mode
        edit.startOperation()

        try:
            yield edit
            edit.stopOperation()
            edit.stopEditing(save_changes=True)
        except Exception:
            if edit.isEditing:
                edit.stopOperation()
                edit.stopEditing(save_changes=False)
            raise
        finally:
            arcpy.management.ClearWorkspaceCache()

    def _bulk_update(
        self,
        gdf,
        feature_class,
        update_fields,
        update_attributes,
        update_geometry,
        dryrun,
        operator,
        chunk_size,
    ):
        """Internal bulk update implementation."""
        # Implementation similar to your existing bulk_update method
        # but with improved error handling and logging
        pass  # Implement based on your existing code

    def _bulk_insert(self, gdf, feature_class, ignore_duplicates, dryrun, operator):
        """Internal bulk insert implementation."""
        # Implementation similar to your existing bulk_insert method
        pass  # Implement based on your existing code

    def _bulk_delete(self, gdf, feature_class, dryrun):
        """Internal bulk delete implementation."""
        # Implementation similar to your existing bulk_delete method
        pass  # Implement based on your existing code

    def _bulk_upsert(
        self,
        gdf,
        feature_class,
        update_fields,
        update_attributes,
        update_geometry,
        dryrun,
        operator,
        chunk_size,
    ):
        """Insert or update features (upsert operation)."""
        existing_uuids = self._get_existing_uuids(feature_class)

        # Split into insert and update operations
        insert_mask = ~gdf[self.uuid_field].isin(existing_uuids)
        update_mask = gdf[self.uuid_field].isin(existing_uuids)

        results = {"insert": {"success_count": 0}, "update": {"success_count": 0}}

        if insert_mask.any():
            insert_gdf = gdf[insert_mask]
            results["insert"] = self._bulk_insert(
                insert_gdf, feature_class, False, dryrun, operator
            )

        if update_mask.any():
            update_gdf = gdf[update_mask]
            results["update"] = self._bulk_update(
                update_gdf,
                feature_class,
                update_fields,
                update_attributes,
                update_geometry,
                dryrun,
                operator,
                chunk_size,
            )

        # Combine results
        total_success = (
            results["insert"]["success_count"] + results["update"]["success_count"]
        )
        total_errors = results["insert"].get("errors", []) + results["update"].get(
            "errors", []
        )

        return {
            "success_count": total_success,
            "errors": total_errors,
            "details": results,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_bridge(
    instance: str = "GCOVERP", version: Optional[str] = None, **kwargs
) -> GCoverSDEBridge:
    """
    Create and configure a GCoverSDEBridge instance.

    Args:
        instance: SDE instance name
        version: Specific version (auto-detected if None)
        **kwargs: Additional bridge configuration

    Returns:
        Configured bridge instance
    """
    return GCoverSDEBridge(instance=instance, version=version, **kwargs)


def quick_export(
    feature_class: str,
    output_path: Union[str, Path],
    instance: str = "GCOVERP",
    **kwargs,
) -> Path:
    """
    Quick export of feature class to file.

    Args:
        feature_class: Feature class to export
        output_path: Output file path
        instance: SDE instance name
        **kwargs: Additional export options

    Returns:
        Path to exported file
    """
    with create_bridge(instance=instance) as bridge:
        return bridge.export_to_file(feature_class, output_path, **kwargs)


def quick_import(
    input_path: Union[str, Path],
    feature_class: str,
    instance: str = "GCOVERP",
    **kwargs,
) -> Dict[str, Union[int, List[str]]]:
    """
    Quick import of file to feature class.

    Args:
        input_path: Input file path
        feature_class: Target feature class
        instance: SDE instance name
        **kwargs: Additional import options

    Returns:
        Import results dictionary
    """
    with create_bridge(instance=instance) as bridge:
        return bridge.import_from_file(input_path, feature_class, **kwargs)
