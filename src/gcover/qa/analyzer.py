# gcover/qa/analyzer.py
"""
QA Analysis module for GeoCover quality assurance data processing.

Handles aggregation and extraction of QA test results by administrative zones
(mapsheets, work units, lots) with support for multiple output formats.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import geopandas as gpd
from loguru import logger
import warnings

# Suppress shapely warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="shapely")


# Silence ogr warning
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    if category is RuntimeWarning and "organizePolygons()" in str(message):
        logger.warning(f"{filename}:{lineno} - {message}")
    else:
        # fallback to default behavior
        warnings._showwarnmsg_impl(
            warnings.WarningMessage(
                message=message,
                category=category,
                filename=filename,
                lineno=lineno,
                file=file,
                line=line,
            )
        )


warnings.showwarning = custom_warning_handler


class QAAnalyzer:
    """
    Analyzer for QA test results aggregation and extraction by administrative zones.

    Bronze → Silver data transformation for GeoCover QA workflow.
    """

    # QA layers that contain spatial data (exclude non-spatial layers)
    SPATIAL_QA_LAYERS = ["IssuePolygons", "IssueLines", "IssuePoints"]

    # Standard fields we expect in QA issue layers
    QA_FIELDS = [
        "TestName",
        "TestType",
        "IssueType",
        "StopCondition",
        "Category",
        "AffectedComponent",
        "Description",
        "Code",
    ]

    def __init__(self, zones_file: Union[str, Path]):
        """
        Initialize QA Analyzer with administrative zones.

        Args:
            zones_file: Path to GPKG file containing administrative zones
        """
        self.zones_file = Path(zones_file)
        self.zones_data = {}
        self._load_administrative_zones()

    def _load_administrative_zones(self) -> None:
        """Load administrative zones from GPKG file."""
        if not self.zones_file.exists():
            raise FileNotFoundError(f"Zones file not found: {self.zones_file}")

        try:
            # Load different zone types based on created GPKG structure
            zone_layers = {
                "mapsheets": "mapsheets_sources_only",  # Main layer with source mapping (RC1/RC2)    Nota:  mapsheets_with_sources has many dupplicates
                "work_units": "work_units",  # Work units layer
                "lots": "lots",  # Lots layer
            }

            for zone_type, layer_name in zone_layers.items():
                try:
                    gdf = gpd.read_file(self.zones_file, layer=layer_name)
                    if not gdf.empty:
                        if zone_type == "work_units":  # TODO huggly hack, to be removed
                            if "WU_NAME" not in gdf.columns and "NAME" in gdf.columns:
                                gdf["WU_NAME"] = gdf["NAME"]
                                logger.warning(f"Adding `WU_NAME` to work_units layer!")
                        if zone_type == "lots":  # TODO huggly hack, to be removed
                            if (
                                "LOT_NAME" not in gdf.columns
                                and "LOT_NR" in gdf.columns
                            ):
                                gdf["LOT_NAME"] = gdf["LOT_NR"].apply(
                                    lambda x: f"Lot{x}"
                                )
                                logger.warning(f"Adding `LOT_NAME` to lots layer!")

                        self.zones_data[zone_type] = gdf
                        logger.info(
                            f"Loaded {len(gdf)} {zone_type} from layer '{layer_name}'"
                        )
                    else:
                        logger.warning(f"Layer '{layer_name}' is empty")
                except Exception as e:
                    logger.warning(
                        f"Could not load layer '{layer_name}' for {zone_type}: {e}"
                    )

            if not self.zones_data:
                raise ValueError("No administrative zones could be loaded")

        except Exception as e:
            raise ValueError(f"Error loading administrative zones: {e}")

    def _read_qa_gdb(self, gdb_path: Path) -> Dict[str, gpd.GeoDataFrame]:
        """
        Read QA data from FileGDB, returning only spatial layers.

        Args:
            gdb_path: Path to QA FileGDB

        Returns:
            Dictionary with layer_name: GeoDataFrame
        """
        if not gdb_path.exists():
            raise FileNotFoundError(f"QA GDB not found: {gdb_path}")

        qa_data = {}

        for layer in self.SPATIAL_QA_LAYERS:
            try:
                gdf = gpd.read_file(gdb_path, layer=layer)
                if not gdf.empty:
                    qa_data[layer] = gdf
                    logger.debug(f"Read {len(gdf)} features from {layer}")
                else:
                    logger.info(f"Layer {layer} is empty in {gdb_path}")
            except Exception as e:
                logger.warning(f"Could not read layer {layer} from {gdb_path}: {e}")

        return qa_data

    def _spatial_join_with_zones(
        self, qa_gdf: gpd.GeoDataFrame, zone_gdf: gpd.GeoDataFrame, zone_type: str
    ) -> gpd.GeoDataFrame:
        """
        Perform spatial join between QA issues and administrative zones.

        Args:
            qa_gdf: QA issues GeoDataFrame
            zone_gdf: Administrative zones GeoDataFrame
            zone_type: Type of zone ('mapsheets', 'work_units', 'lots')

        Returns:
            GeoDataFrame with zone attributes joined
        """
        if qa_gdf.empty or zone_gdf.empty:
            return gpd.GeoDataFrame()

        # Ensure same CRS
        if qa_gdf.crs != zone_gdf.crs:
            qa_gdf = qa_gdf.to_crs(zone_gdf.crs)

        # Prepare zone columns for join
        zone_id_col = self._get_zone_id_column(zone_type)
        zone_name_col = self._get_zone_name_column(zone_type)

        logger.debug(f"zone_id_col: {zone_id_col}")
        logger.debug(f"zone_name_col: {zone_name_col}")

        join_cols = ["geometry", zone_id_col]
        if zone_name_col and zone_name_col in zone_gdf.columns:
            join_cols.append(zone_name_col)

        # Add source information for mapsheets
        if (
            zone_type == "mapsheets" and "SOURCE_RC" in zone_gdf.columns
        ):  # TODO BKP or SO
            join_cols.append("SOURCE_RC")  # Source (RC1/RC2)

        logger.info(f"join_cols: {join_cols}")

        # TODO: reset index
        qa_gdf = qa_gdf.reset_index()
        qa_gdf = qa_gdf.drop(columns=["level_0"], errors="ignore")
        logger.debug(f"qa_gdf columns: {qa_gdf.columns}")
        logger.debug(f"zone_gdf columns: {zone_gdf[join_cols].columns}")

        # Spatial join - issues may overlap multiple zones (important!)
        result = gpd.sjoin(
            qa_gdf,
            zone_gdf[join_cols],
            how="inner",  # Only keep issues that intersect zones
            predicate="intersects",
        )
        logger.debug(f"Spatial join: {result.head()}")
        logger.debug(f"Spatial join: {result.columns}")

        # Clean up join artifacts
        result = result.drop(columns=["index_right"], errors="ignore")

        logger.debug("After drop index")

        # Add zone type for aggregation
        result["zone_type"] = zone_type

        logger.debug("New column")

        # Log spatial join results
        original_count = len(qa_gdf)
        joined_count = len(result)
        logger.debug(f"Results: {original_count}, {joined_count}")
        dropped_count = original_count - len(
            result.drop_duplicates(subset=[qa_gdf.index.name or "index"])
        )

        logger.debug(f"Drops: {dropped_count}")

        logger.info(
            f"Spatial join with {zone_type}: {original_count} → {joined_count} "
            f"(+{joined_count - original_count} overlaps, {dropped_count} dropped)"
        )

        return result

    def _get_zone_id_column(self, zone_type: str) -> str:
        """Get the ID column name for a zone type."""
        zone_id_mapping = {
            "mapsheets": "MSH_TOPO_NR",  # Based on your script
            "work_units": "WU_ID",  # TODO  WU_ID or NAME Adjust based on actual WU structure
            "lots": "LOT_NR",
        }
        return zone_id_mapping.get(zone_type, "id")

    def _get_zone_name_column(self, zone_type: str) -> Optional[str]:
        """Get the name column for a zone type."""
        zone_name_mapping = {
            "mapsheets": "MSH_MAP_TITLE",
            "work_units": "WU_NAME",
            "lots": "LOT_NAME",
        }
        return zone_name_mapping.get(zone_type)

    def extract_relevant_issues(
        self,
        rc1_gdb: Path,
        rc2_gdb: Path,
        output_path: Path,
        output_format: str = "gpkg",
    ) -> Dict[str, int]:
        """
        Extract only relevant QA issues based on mapsheet source mapping.

        For each mapsheet, extract issues from the appropriate RC version:
        - RC1 issues for mapsheets with BKP='RC1'
        - RC2 issues for mapsheets with BKP='RC2'

        Args:
            rc1_gdb: Path to RC1 QA FileGDB
            rc2_gdb: Path to RC2 QA FileGDB
            output_path: Output path (without extension)
            output_format: 'gpkg' or 'filegdb'

        Returns:
            Dictionary with extraction statistics
        """
        SOURCE_COLUMN_NAMES = ["BKP", "SOURCE_RC"]
        if "mapsheets" not in self.zones_data:
            raise ValueError("Mapsheets data required for relevance filtering")

        mapsheets_gdf = self.zones_data["mapsheets"]

        # TODO: decide how to name the column for rc sources
        SOURCE_RC_COLUMN = next(
            (item for item in SOURCE_COLUMN_NAMES if item in mapsheets_gdf.columns),
            None,
        )

        if not SOURCE_RC_COLUMN:
            logger.error(
                f"No {','.join(SOURCE_COLUMN_NAMES)} (source RC) column found in mapsheets. Using all data."
            )
            # Fallback: combine all data without filtering
            return self._extract_all_issues(
                rc1_gdb, rc2_gdb, output_path, output_format
            )
        else:
            if SOURCE_RC_COLUMN == "SOURCE_RC":
                mapsheets_gdf = mapsheets_gdf.rename(columns={"SOURCE_RC": "source_rc"})
                SOURCE_RC_COLUMN = "source_rc"
            logger.info(f"Using '{SOURCE_RC_COLUMN}' to select source RC")

        logger.info("Extracting relevant QA issues based on mapsheet sources")

        # Split mapsheets by source
        rc1_mapsheets = mapsheets_gdf[mapsheets_gdf[SOURCE_RC_COLUMN] == "RC1"]
        rc2_mapsheets = mapsheets_gdf[mapsheets_gdf[SOURCE_RC_COLUMN] == "RC2"]

        logger.info(
            f"RC1 mapsheets: {len(rc1_mapsheets)}, RC2 mapsheets: {len(rc2_mapsheets)}"
        )

        all_filtered_data = {}
        stats = {"total_issues": 0, "rc1_issues": 0, "rc2_issues": 0}

        # Process RC1 issues with RC1 mapsheets
        if not rc1_mapsheets.empty:
            rc1_data = self._read_qa_gdb(rc1_gdb)
            for layer_name, qa_gdf in rc1_data.items():
                logger.info(f"Processing RC1 layer {layer_name}")
                if qa_gdf.empty:
                    continue

                filtered_gdf = self._spatial_join_with_zones(
                    qa_gdf, rc1_mapsheets, "mapsheets"
                )
                if not filtered_gdf.empty:
                    filtered_gdf["source_rc"] = "RC1"
                    all_filtered_data[f"{layer_name}_RC1"] = filtered_gdf
                    stats["rc1_issues"] += len(filtered_gdf)

        # Process RC2 issues with RC2 mapsheets
        if not rc2_mapsheets.empty:
            rc2_data = self._read_qa_gdb(rc2_gdb)
            for layer_name, qa_gdf in rc2_data.items():
                if qa_gdf.empty:
                    continue

                filtered_gdf = self._spatial_join_with_zones(
                    qa_gdf, rc2_mapsheets, "mapsheets"
                )
                if not filtered_gdf.empty:
                    filtered_gdf["source_rc"] = "RC2"
                    all_filtered_data[f"{layer_name}_RC2"] = filtered_gdf
                    stats["rc2_issues"] += len(filtered_gdf)

        stats["total_issues"] = stats["rc1_issues"] + stats["rc2_issues"]

        if not all_filtered_data:
            logger.warning("No relevant issues found for extraction")
            return stats

        # Combine layers by type (merge RC1 and RC2 data for each layer type)
        combined_layers = {}
        for layer_type in self.SPATIAL_QA_LAYERS:
            layer_gdfs = []
            for key, gdf in all_filtered_data.items():
                if layer_type in key:
                    layer_gdfs.append(gdf)

            if layer_gdfs:
                combined_layers[layer_type] = pd.concat(layer_gdfs, ignore_index=True)

        # Write output
        self._write_spatial_output(combined_layers, output_path, output_format)

        logger.info(
            f"Extracted {stats['total_issues']} relevant issues "
            f"({stats['rc1_issues']} RC1, {stats['rc2_issues']} RC2)"
        )

        return stats

    def _extract_all_issues(
        self, rc1_gdb: Path, rc2_gdb: Path, output_path: Path, output_format: str
    ) -> Dict[str, int]:
        """Fallback: extract all issues without filtering."""
        logger.info("Extracting all QA issues (no source filtering)")

        all_data = {}
        stats = {"total_issues": 0, "rc1_issues": 0, "rc2_issues": 0}

        # Read and combine all data
        for rc_name, gdb_path in [("RC1", rc1_gdb), ("RC2", rc2_gdb)]:
            qa_data = self._read_qa_gdb(gdb_path)
            for layer_name, qa_gdf in qa_data.items():
                if qa_gdf.empty:
                    continue

                qa_gdf["source_rc"] = rc_name

                if layer_name not in all_data:
                    all_data[layer_name] = []
                all_data[layer_name].append(qa_gdf)

                if rc_name == "RC1":
                    stats["rc1_issues"] += len(qa_gdf)
                else:
                    stats["rc2_issues"] += len(qa_gdf)

        # Combine layers
        combined_layers = {}
        for layer_name, gdf_list in all_data.items():
            if gdf_list:
                combined_layers[layer_name] = pd.concat(gdf_list, ignore_index=True)

        stats["total_issues"] = stats["rc1_issues"] + stats["rc2_issues"]

        # Write output
        self._write_spatial_output(combined_layers, output_path, output_format)

        return stats

    def aggregate_by_zone(
        self,
        input1: Path,
        input2: Optional[Path] = None,
        zone_type: str = "mapsheets",
        output_format: str = "csv",
    ) -> pd.DataFrame:
        """
        Aggregate QA statistics by administrative zones.

        Supports two modes:
        1. Two-RC mode: aggregate_by_zone(rc1_gdb, rc2_gdb, ...)
        2. Single-input mode: aggregate_by_zone(merged_data_path, None, ...)

        Args:
            input1: Primary input - either RC1 GDB or merged data path
            input2: Secondary input - RC2 GDB (None for single-input mode)
            zone_type: Type of zones to aggregate by
            output_format: Output format ('csv', 'xlsx', 'json')

        Returns:
            DataFrame with aggregated statistics
        """
        if zone_type not in self.zones_data:
            raise ValueError(
                f"Zone type '{zone_type}' not available. "
                f"Available: {list(self.zones_data.keys())}"
            )

        # Determine mode based on inputs
        if input2 is not None and input2.exists():
            # Two-RC mode (backward compatibility)
            return self._aggregate_by_zone_two_rc(
                input1, input2, zone_type, output_format
            )
        else:
            # Single-input mode (new merged data approach)
            return self._aggregate_by_zone_single_input(
                input1, zone_type, output_format
            )

    def _aggregate_by_zone_single_input(
        self,
        merged_data_path: Path,
        zone_type: str,
        output_format: str,
    ) -> pd.DataFrame:
        """
        Aggregate QA statistics using pre-merged data.

        Args:
            merged_data_path: Path to merged QA data (GPKG or FileGDB)
            zone_type: Type of zones to aggregate by
            output_format: Output format

        Returns:
            DataFrame with aggregated statistics
        """
        logger.info(f"Aggregating merged QA data by {zone_type}")
        zone_gdf = self.zones_data[zone_type]
        all_results = []

        # Read merged QA data
        merged_qa_data = self._read_merged_qa_data(merged_data_path)

        if not merged_qa_data:
            logger.warning("No merged QA data found")
            return pd.DataFrame()

        # Process each layer type in the merged data
        for layer_name, qa_gdf in merged_qa_data.items():
            if qa_gdf.empty:
                continue

            logger.info(f"Processing merged {layer_name} ({len(qa_gdf)} features)")

            # Check if data already has zone information
            zone_id_col = self._get_zone_id_column(zone_type)

            if zone_id_col in qa_gdf.columns:
                # Data already has zone information, use it directly
                joined_gdf = qa_gdf.copy()
                logger.debug(f"Using existing zone information in {layer_name}")
            else:
                # Perform spatial join with zones (fallback if needed)
                joined_gdf = self._spatial_join_with_zones(qa_gdf, zone_gdf, zone_type)

            if joined_gdf.empty:
                logger.warning(f"No {layer_name} issues intersect {zone_type} zones")
                continue

            # Aggregate by zone and test type
            zone_name_col = self._get_zone_name_column(zone_type)

            # Group by zone and test characteristics
            group_cols = [zone_id_col, "TestType", "TestName"]
            if zone_name_col and zone_name_col in joined_gdf.columns:
                group_cols.insert(1, zone_name_col)

            # Add source_rc column if available (to track RC1/RC2 origin)
            if "source_rc" in joined_gdf.columns:
                group_cols.append("source_rc")

            # Add source column for mapsheets (legacy support)
            source_column_names = ["BKP", "SOURCE_RC", "source_rc"]
            source_col = next(
                (col for col in source_column_names if col in joined_gdf.columns), None
            )
            if source_col and source_col not in group_cols:
                group_cols.append(source_col)

            # Filter valid group columns
            group_cols = [col for col in group_cols if col in joined_gdf.columns]

            logger.debug(f"Grouping by columns: {group_cols}")
            logger.debug(
                f"Before grouping: {joined_gdf.loc[:, ['MSH_TOPO_NR', 'MSH_MAP_TITLE', 'source_rc', 'TestType', 'TestName', 'IssueType', 'StopCondition']].head(20)}"
            )

            # =============================================================================
            # Count TestName occurrences by MSH_TOPO_NR
            # =============================================================================

            # Check if required columns exist
            required_cols = [zone_id_col, "TestName"]
            if all(col in joined_gdf.columns for col in required_cols):
                logger.info(f"Creating TestName count summary by {zone_id_col}")

                # Group by MSH_TOPO_NR and TestName, count occurrences and preserve other fields
                agg_stats = (
                    joined_gdf.groupby(required_cols)
                    .agg(
                        {
                            zone_name_col: "first",  # Take first occurrence
                            "source_rc": "first",  # Take first occurrence
                            "TestType": "first",  # Take first occurrence
                            "IssueType": "first",  # Take first occurrence (could be mixed, so first is representative)
                            "StopCondition": "first",  # Take first occurrence (could be mixed, so first is representative)
                        }
                    )
                    .reset_index()
                )

                # Add issue count using size() which counts the group size
                issue_counts = (
                    joined_gdf.groupby(required_cols)
                    .size()
                    .reset_index(name="issue_count")
                )

                # Merge the counts with the aggregated data
                agg_stats = agg_stats.merge(
                    issue_counts, on=[zone_id_col, "TestName"], how="left"
                )

                # Reorder columns to match desired format
                column_order = [
                    zone_id_col,
                    zone_name_col,
                    "source_rc",
                    "TestName",
                    "TestType",
                    "issue_count",
                    "IssueType",
                    "StopCondition",
                ]

                # Only include columns that exist
                available_columns = [
                    col for col in column_order if col in agg_stats.columns
                ]
                agg_stats = agg_stats[available_columns]

                # Add LayerName column with the layer_name variable value
                agg_stats["layer_name"] = layer_name

                logger.info(f"TestName summary completed. Shape: {agg_stats.shape}")
                logger.debug(f"Sample results:\n{agg_stats.head()}")

                # Optional: Display summary statistics
                logger.info("TestName Count Statistics:")
                logger.info(
                    f"- Total unique {zone_id_col}/TestName combinations: {len(agg_stats)}"
                )
                logger.info(
                    f"- Issue count range: {agg_stats['issue_count'].min()} - {agg_stats['issue_count'].max()}"
                )
                logger.info(
                    f"- Average issues per TestName/Map combination: {agg_stats['issue_count'].mean():.2f}"
                )

                # Show top combinations by issue count
                if len(agg_stats) > 0:
                    top_combinations = agg_stats.nlargest(5, "issue_count")[
                        [zone_id_col, zone_name_col, "TestName", "issue_count"]
                    ]
                    logger.info(
                        f"Top 5 TestName combinations by issue count:\n{top_combinations.to_string(index=False)}"
                    )

            else:
                logger.warning(
                    f"Cannot create TestName summary. Missing required columns: {set(required_cols) - set(joined_gdf.columns)}"
                )

                # Fall back to original aggregation approach
                agg_stats = (
                    joined_gdf.groupby(group_cols)
                    .agg(
                        {
                            "IssueType": ["count", lambda x: (x == "Error").sum()],
                            "StopCondition": lambda x: (x == "Yes").sum(),
                        }
                    )
                    .reset_index()
                )

                # Flatten column names
                agg_stats.columns = [
                    col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
                    for col in agg_stats.columns
                ]
                agg_stats = agg_stats.rename(
                    columns={
                        "IssueType_count": "total_issues",
                        "IssueType_<lambda>": "error_issues",
                        "StopCondition_<lambda>": "stop_condition_issues",
                    }
                )

            # Add metadata
            agg_stats["layer_type"] = layer_name
            agg_stats["zone_type"] = zone_type

            # Optional: Display summary statistics
            if "TestName_Count" in agg_stats.columns:
                logger.info("TestName Count Statistics:")
                logger.info(
                    f"- Total unique {zone_id_col}/TestName combinations: {len(agg_stats)}"
                )
                logger.info(
                    f"- TestName count range: {agg_stats['TestName_Count'].min()} - {agg_stats['TestName_Count'].max()}"
                )
                logger.info(
                    f"- Average TestName occurrences per combination: {agg_stats['TestName_Count'].mean():.2f}"
                )

                # Show top TestName combinations by count
                top_combinations = agg_stats.nlargest(5, "TestName_Count")[
                    [zone_id_col, zone_name_col, "TestName", "TestName_Count"]
                ]
                logger.info(
                    f"Top 5 TestName combinations by count:\n{top_combinations.to_string(index=False)}"
                )

            all_results.append(agg_stats)

        if not all_results:
            logger.warning("No QA data could be aggregated")
            return pd.DataFrame()

        # Combine all results
        final_df = pd.concat(all_results, ignore_index=True)

        logger.info(f"Aggregated {len(final_df)} rows of statistics from merged data")
        return final_df

    def _aggregate_by_zone_two_rc(
        self,
        rc1_gdb: Path,
        rc2_gdb: Path,
        zone_type: str,
        output_format: str,
    ) -> pd.DataFrame:
        """
        Original two-RC aggregation method (for backward compatibility).

        Args:
            rc1_gdb: Path to RC1 QA FileGDB
            rc2_gdb: Path to RC2 QA FileGDB
            zone_type: Type of zones to aggregate by
            output_format: Output format

        Returns:
            DataFrame with aggregated statistics
        """
        logger.info(f"Aggregating QA data by {zone_type} (two-RC mode)")
        zone_gdf = self.zones_data[zone_type]
        all_results = []

        # Process both RC1 and RC2
        for rc_name, gdb_path in [("RC1", rc1_gdb), ("RC2", rc2_gdb)]:
            if not gdb_path.exists():
                logger.warning(f"Skipping {rc_name}: {gdb_path} does not exist")
                continue

            logger.info(f"Processing {rc_name}: {gdb_path}")
            qa_data = self._read_qa_gdb(gdb_path)

            for layer_name, qa_gdf in qa_data.items():
                if qa_gdf.empty:
                    continue

                # Spatial join with zones
                joined_gdf = self._spatial_join_with_zones(qa_gdf, zone_gdf, zone_type)

                if joined_gdf.empty:
                    logger.warning(
                        f"No {layer_name} issues intersect {zone_type} zones"
                    )
                    continue

                # Aggregate by zone and test type
                zone_id_col = self._get_zone_id_column(zone_type)
                zone_name_col = self._get_zone_name_column(zone_type)

                # Group by zone and test characteristics
                group_cols = [zone_id_col, "TestType", "TestName"]
                if zone_name_col and zone_name_col in joined_gdf.columns:
                    group_cols.insert(1, zone_name_col)

                # Add source column for mapsheets
                if zone_type == "mapsheets" and "BKP" in joined_gdf.columns:
                    group_cols.append("BKP")

                # Filter valid group columns
                group_cols = [col for col in group_cols if col in joined_gdf.columns]

                # Aggregation
                agg_stats = (
                    joined_gdf.groupby(group_cols)
                    .agg(
                        {
                            "IssueType": ["count", lambda x: (x == "Error").sum()],
                            "StopCondition": lambda x: (x == "Yes").sum(),
                        }
                    )
                    .reset_index()
                )

                # Flatten column names
                agg_stats.columns = [
                    col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
                    for col in agg_stats.columns
                ]
                agg_stats = agg_stats.rename(
                    columns={
                        "IssueType_count": "total_issues",
                        "IssueType_<lambda>": "error_issues",
                        "StopCondition_<lambda>": "stop_condition_issues",
                    }
                )

                # Add metadata
                agg_stats["rc_version"] = rc_name
                agg_stats["layer_type"] = layer_name
                agg_stats["zone_type"] = zone_type

                all_results.append(agg_stats)

        if not all_results:
            logger.warning("No QA data could be aggregated")
            return pd.DataFrame()

        # Combine all results
        final_df = pd.concat(all_results, ignore_index=True)

        logger.info(f"Aggregated {len(final_df)} rows of statistics")
        return final_df

    def _read_merged_qa_data(
        self, merged_data_path: Path
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Read merged QA data from file (GPKG or FileGDB output from extract_relevant_issues).

        Args:
            merged_data_path: Path to merged QA data file

        Returns:
            Dictionary with layer_name: GeoDataFrame
        """
        if not merged_data_path.exists():
            raise FileNotFoundError(f"Merged QA data not found: {merged_data_path}")

        qa_data = {}

        # Determine file type and read accordingly
        if merged_data_path.suffix.lower() == ".gpkg":
            # Read from GPKG
            try:
                # List available layers
                import fiona

                layers = fiona.listlayers(str(merged_data_path))
                logger.debug(f"Available layers in {merged_data_path}: {layers}")

                for layer in layers:
                    if any(
                        spatial_layer in layer
                        for spatial_layer in self.SPATIAL_QA_LAYERS
                    ):
                        try:
                            gdf = gpd.read_file(merged_data_path, layer=layer)
                            if not gdf.empty:
                                # Normalize layer name (remove RC1/RC2 suffixes if present)
                                clean_layer_name = self._normalize_layer_name(layer)
                                qa_data[clean_layer_name] = gdf
                                logger.debug(
                                    f"Read {len(gdf)} features from layer {layer}"
                                )
                            else:
                                logger.info(
                                    f"Layer {layer} is empty in {merged_data_path}"
                                )
                        except Exception as e:
                            logger.warning(f"Could not read layer {layer}: {e}")

            except Exception as e:
                logger.error(f"Error reading GPKG {merged_data_path}: {e}")

        elif merged_data_path.suffix.lower() == ".gdb":
            # Read from FileGDB (same as existing _read_qa_gdb method)
            qa_data = self._read_qa_gdb(merged_data_path)

        else:
            raise ValueError(f"Unsupported file format: {merged_data_path.suffix}")

        return qa_data

    def _normalize_layer_name(self, layer_name: str) -> str:
        """
        Normalize layer name by removing RC suffixes.

        Args:
            layer_name: Original layer name (e.g., "IssuePolygons_RC1")

        Returns:
            Normalized layer name (e.g., "IssuePolygons")
        """
        for spatial_layer in self.SPATIAL_QA_LAYERS:
            if spatial_layer in layer_name:
                return spatial_layer
        return layer_name

    def _write_spatial_output(
        self,
        layer_data: Dict[str, gpd.GeoDataFrame],
        output_path: Path,
        output_format: str,
    ) -> None:
        """
        Write spatial data to specified format.

        Args:
            layer_data: Dictionary of layer_name: GeoDataFrame
            output_path: Output path without extension
            output_format: 'gpkg' or 'filegdb'
        """
        if not layer_data:
            logger.warning("No data to write")
            return

        if output_format == "gpkg":
            output_file = output_path.with_suffix(".gpkg")
            driver = "GPKG"
        elif output_format == "filegdb":
            output_file = output_path.with_suffix(".gdb")
            driver = "OpenFileGDB"
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Remove existing file
        if output_file.exists():
            logger.warning(f"Removing existing ouput file: {output_file}")
            if output_format == "filegdb":
                import shutil

                shutil.rmtree(output_file)
            else:
                output_file.unlink()

        # Write layers
        for layer_name, gdf in layer_data.items():
            if gdf.empty:
                continue

            try:
                # Clean up any problematic columns for GDAL
                gdf_clean = gdf.copy()

                # Handle datetime columns that might cause issues
                for col in gdf_clean.columns:
                    if gdf_clean[col].dtype == "object":
                        # Try to handle any problematic object columns
                        try:
                            gdf_clean[col] = gdf_clean[col].astype(str)
                        except:
                            pass

                # Write to file
                gdf_clean.to_file(
                    output_file,
                    layer=layer_name,
                    driver=driver,
                    mode="a" if output_file.exists() else "w",
                )

                logger.info(f"Wrote {len(gdf)} features to layer '{layer_name}'")

            except Exception as e:
                logger.error(f"Failed to write layer '{layer_name}': {e}")

        logger.success(f"Spatial data written to {output_file}")

    def write_aggregated_stats(
        self,
        stats_df: pd.DataFrame,
        output_path: Path,
        zone_type: str,
        output_format: str = "csv",
    ) -> None:
        """
        Write aggregated statistics to file.

        Args:
            stats_df: Aggregated statistics DataFrame
            output_path: Output file path
            output_format: 'csv', 'xlsx', or 'json'
        """
        if stats_df.empty:
            logger.warning("No statistics to write")
            return

        output_path = Path(output_path) / f"aggregated_by_{zone_type}"

        try:
            if output_format == "csv":
                output_file = output_path.with_suffix(".csv")
                stats_df.to_csv(output_file, index=False)
            elif output_format == "xlsx":
                output_file = output_path.with_suffix(".xlsx")
                stats_df.to_excel(output_file, index=False, engine="openpyxl")
            elif output_format == "json":
                output_file = output_path.with_suffix(".json")
                stats_df.to_json(output_file, orient="records", indent=2)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            logger.success(f"Statistics written to {output_file}")

        except Exception as e:
            logger.error(f"Failed to write statistics: {e}")
