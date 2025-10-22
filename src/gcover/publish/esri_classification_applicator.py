#!/usr/bin/env python

"""
Classification Symbol Applicator

Apply ESRI classification rules to GeoDataFrames/GPKG files.
Adds a SYMBOL field with generated class identifiers based on classification rules.

Author: Generated for lg-gcover project
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.prompt import Confirm
from rich.table import Table

# Import classification extractor
from gcover.publish.esri_classification_extractor import (
    ClassificationClass,
    ESRIClassificationExtractor,
    LayerClassification,
    extract_lyrx,
)
from gcover.publish.utils import translate_esri_to_pandas

console = Console()

# Configure loguru
"""logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)"""


def float_to_int_string(val):
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        else:
            return str(f)
    except (ValueError, TypeError):
        return val  # leave unchanged if not a number


def get_numeric_field_names(field_types):
    """
    Extract field names that have numerical dtypes.

    Args:
        field_types: Dict of {field_name: dtype_string}

    Returns:
        List of field names with numeric types
    """
    numeric_dtypes = {
        "int64",
        "int32",
        "int16",
        "int8",
        "Int64",
        "Int32",
        "Int16",
        "Int8",  # Nullable integers
        "uint64",
        "uint32",
        "uint16",
        "uint8",
        "float64",
        "float32",
        "float16",
        "number",
        "numeric",
    }

    numeric_fields = [
        field
        for field, dtype in field_types.items()
        if dtype.lower() in {dt.lower() for dt in numeric_dtypes}
    ]

    return numeric_fields


def get_numeric_fields_from_dataframe(gdf):
    """
    Extract numeric field names directly from GeoDataFrame.
    More reliable than string matching.
    """
    import numpy as np
    import pandas as pd

    numeric_fields = []
    for col in gdf.columns:
        if col == "geometry":  # Skip geometry column
            continue

        # Check if dtype is numeric
        if pd.api.types.is_numeric_dtype(gdf[col]):
            numeric_fields.append(col)

    return numeric_fields


def cast_geodataframe_fields(gdf, field_types, strict=False):
    """
    Cast GeoDataFrame fields to specified types with robust error handling.

    Args:
        gdf: GeoDataFrame to cast
        field_types: Dict of {field_name: dtype_string}
        strict: If True, raise errors; if False, log warnings and continue

    Returns:
        GeoDataFrame with casted fields
    """
    stats = {"success": [], "failed": [], "skipped": []}

    for field, dtype_str in field_types.items():
        if field not in gdf.columns:
            stats["skipped"].append(field)
            console.print(f"[yellow]⊘ Field '{field}' not in data, skipping[/yellow]")
            continue

        try:
            # Get original state for comparison
            original_dtype = gdf[field].dtype
            null_count_before = gdf[field].isna().sum()

            # Handle numeric types with potential NULL values
            if dtype_str.lower() in ["int64", "int32", "int16", "int8"]:
                # Use nullable Integer types (capital I)
                nullable_dtype = dtype_str.capitalize()  # int64 -> Int64

                # Convert to numeric first (coerce non-numeric to NaN)
                gdf[field] = pd.to_numeric(gdf[field], errors="coerce")

                # Cast to nullable integer
                gdf[field] = gdf[field].astype(nullable_dtype)

                null_count_after = gdf[field].isna().sum()
                new_nulls = null_count_after - null_count_before

                console.print(
                    f"[green]✓ {field}: {original_dtype} → {nullable_dtype}"
                    f"{f' (+{new_nulls} NaN)' if new_nulls > 0 else ''}[/green]"
                )

                if new_nulls > 0:
                    console.print(
                        f"  [yellow]⚠️  {new_nulls} values coerced to NaN during conversion[/yellow]"
                    )

            # Handle float types
            elif dtype_str.lower() in ["float64", "float32"]:
                gdf[field] = pd.to_numeric(gdf[field], errors="coerce")
                gdf[field] = gdf[field].astype(dtype_str)

                null_count_after = gdf[field].isna().sum()
                new_nulls = null_count_after - null_count_before

                console.print(
                    f"[green]✓ {field}: {original_dtype} → {dtype_str}"
                    f"{f' (+{new_nulls} NaN)' if new_nulls > 0 else ''}[/green]"
                )

            # Handle boolean
            elif dtype_str.lower() in ["bool", "boolean"]:
                # Use nullable boolean
                gdf[field] = gdf[field].astype("boolean")
                console.print(f"[green]✓ {field}: {original_dtype} → boolean[/green]")

            # Handle string
            elif dtype_str.lower() in ["str", "string", "object"]:
                gdf[field] = gdf[field].astype("string")
                console.print(f"[green]✓ {field}: {original_dtype} → string[/green]")

            # Handle datetime
            elif dtype_str.lower().startswith("datetime"):
                gdf[field] = pd.to_datetime(gdf[field], errors="coerce")
                console.print(
                    f"[green]✓ {field}: {original_dtype} → datetime64[/green]"
                )

            # Default: try direct casting
            else:
                gdf[field] = gdf[field].astype(dtype_str)
                console.print(
                    f"[green]✓ {field}: {original_dtype} → {dtype_str}[/green]"
                )

            stats["success"].append(field)

        except Exception as e:
            stats["failed"].append((field, str(e)))
            console.print(f"[red]✗ Failed to cast {field} to {dtype_str}: {e}[/red]")

            # Show sample values for debugging
            sample = gdf[field].head(5).tolist()
            console.print(f"  [yellow]Sample values: {sample}[/yellow]")

            if strict:
                raise

    # Summary
    console.print(f"\n[cyan]Type casting summary:[/cyan]")
    console.print(f"  • Success: {len(stats['success'])}")
    console.print(f"  • Failed: {len(stats['failed'])}")
    console.print(f"  • Skipped: {len(stats['skipped'])}")

    return gdf


def validate_field_types(gdf, field_types):
    """Check if field types can be applied without errors."""
    issues = []

    for field, dtype in field_types.items():
        if field not in gdf.columns:
            continue

        # Check for conversion issues
        if dtype.lower() in ["int64", "int32", "int16", "int8"]:
            # Try numeric conversion
            numeric_series = pd.to_numeric(gdf[field], errors="coerce")
            non_numeric = gdf[field][numeric_series.isna() & gdf[field].notna()]

            if len(non_numeric) > 0:
                issues.append(
                    {
                        "field": field,
                        "dtype": dtype,
                        "issue": "non-numeric values",
                        "count": len(non_numeric),
                        "samples": non_numeric.head(3).tolist(),
                    }
                )

    if issues:
        console.print("\n[yellow]⚠️  Type conversion issues detected:[/yellow]")
        for issue in issues:
            console.print(
                f"  • {issue['field']} ({issue['dtype']}): "
                f"{issue['count']} {issue['issue']}"
            )
            console.print(f"    Examples: {issue['samples']}")

    return issues


def apply_robust_filter(gdf, additional_filter=None, numeric_columns=None):
    """
    Apply filter with robust type handling.

    Args:
        gdf: GeoDataFrame to filter
        additional_filter: Query string like 'KIND == 14334001'
        numeric_columns: List of columns to ensure are numeric (e.g., ['KIND'])

    Returns:
        Filtered GeoDataFrame
    """
    if not additional_filter:
        return gdf.copy()

    # Create working copy
    gdf_work = gdf.copy()

    # Convert specified columns to numeric types
    if numeric_columns:
        for col in numeric_columns:
            if col in gdf_work.columns:
                original_count = len(gdf_work)

                # Convert to numeric, handling errors
                gdf_work[col] = pd.to_numeric(gdf_work[col], errors="coerce")

                # Convert to Int64 (nullable integer) to handle NaN
                # Use Int64 (capital I) instead of int64 to allow NaN
                gdf_work[col] = gdf_work[col].astype("Int64")

                # Log conversion info
                nan_count = gdf_work[col].isna().sum()
                if nan_count > 0:
                    console.print(
                        f"[yellow]⚠️  Column '{col}': {nan_count}/{original_count} values converted to NaN[/yellow]"
                    )

                logger.debug(f"Converted column '{col}' to Int64 dtype")

    # Apply filter
    try:
        gdf_filtered = gdf_work.query(additional_filter)

        # Log filter results
        filtered_count = len(gdf_filtered)
        total_count = len(gdf)
        console.print(f"[cyan]🔍 Filter applied: '{additional_filter}'[/cyan]")
        console.print(
            f"[cyan]   {filtered_count:,} / {total_count:,} features match ({filtered_count / total_count * 100:.1f}%)[/cyan]"
        )

        return gdf_filtered

    except Exception as e:
        console.print(f"[red]❌ Filter query failed: {e}[/red]")
        console.print(f"[yellow]   Query: '{additional_filter}'[/yellow]")
        logger.error(f"Filter query failed: {e}")

        # Show column dtypes for debugging
        if numeric_columns:
            console.print("[yellow]   Column types:[/yellow]")
            for col in numeric_columns:
                if col in gdf_work.columns:
                    console.print(f"     - {col}: {gdf_work[col].dtype}")

        # Return unfiltered data as fallback
        console.print("[yellow]   Returning unfiltered data[/yellow]")
        return gdf.copy()


class LayerMapping:
    """Mapping configuration between GPKG layers and classification layers."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize layer mapping from YAML config.

        Args:
            config_path: Path to YAML configuration file
        """
        self.mappings = {}

        if config_path:
            self.load_from_yaml(config_path)

    def load_from_yaml(self, config_path: Path):
        """Load mapping configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Parse layer mappings
        for mapping in config.get("layers", []):
            gpkg_layer = mapping.get("gpkg_layer")

            if not gpkg_layer:
                logger.warning("Skipping mapping entry without gpkg_layer")
                continue

            self.mappings[gpkg_layer] = {
                "classification_layer": mapping.get("classification_layer", gpkg_layer),
                "field_mapping": mapping.get("fields", {}),
                "filter": mapping.get("filter"),
                "symbol_prefix": mapping.get("symbol_prefix"),
                "field_types": mapping.get("field_types", {}),
            }

        logger.info(f"Loaded {len(self.mappings)} layer mappings from {config_path}")

    def get_classification_name(self, gpkg_layer: str) -> Optional[str]:
        """Get classification layer name for a GPKG layer."""
        if gpkg_layer in self.mappings:
            return self.mappings[gpkg_layer]["classification_layer"]
        return None

    def get_field_types(self, gpkg_layer: str) -> Dict[str, str]:
        """
        Get field types for a GPKG layer.

        Returns:
            Dictionary mapping GPKG field types
        """
        if gpkg_layer in self.mappings:
            return self.mappings[gpkg_layer].get("field_types", {})
        return {}

    def get_field_mapping(self, gpkg_layer: str) -> Dict[str, str]:
        """
        Get field mapping for a GPKG layer.

        Returns:
            Dictionary mapping GPKG field names to classification field names
        """
        if gpkg_layer in self.mappings:
            return self.mappings[gpkg_layer].get("field_mapping", {})
        return {}

    def get_filter(self, gpkg_layer: str) -> Optional[str]:
        """Get filter expression for a GPKG layer."""
        if gpkg_layer in self.mappings:
            return self.mappings[gpkg_layer].get("filter")
        return None

    def get_symbol_prefix(self, gpkg_layer: str) -> Optional[str]:
        """Get symbol prefix for a GPKG layer."""
        if gpkg_layer in self.mappings:
            return self.mappings[gpkg_layer].get("symbol_prefix")
        return None

    @staticmethod
    def create_example_yaml(output_path: Path):
        """Create an example YAML configuration file."""
        example = {
            "layers": [
                {
                    "gpkg_layer": "GC_FOSSILS",
                    "classification_layer": "Fossils",
                    "fields": {
                        "KIND": "KIND",
                        "LFOS_DIVISION": "LFOS_DIVISION",
                        "LFOS_STATUS": "LFOS_STATUS",
                    },
                    "filter": "KIND == 14601006",
                    "symbol_prefix": "fossil",
                },
                {
                    "gpkg_layer": "GC_POINTS",
                    "classification_layer": "Erratic Blocs",
                    "fields": {"TYPE": "KIND", "STATUS": "LFOS_STATUS"},
                    "filter": "TYPE == 14601008",
                    "symbol_prefix": "erratic",
                },
            ]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                example,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.success(f"Created example configuration: {output_path}")


class ClassificationMatcher:
    """Match features against ESRI classification rules."""

    def __init__(
        self,
        classification: LayerClassification,
        field_mapping: Optional[Dict[str, str]] = None,
        treat_zero_as_null: bool = False,
        debug: bool = False,
    ):
        """
        Initialize matcher with classification rules.

        Args:
            classification: Layer classification with rules
            field_mapping: Optional mapping from GPKG field names to classification field names
            treat_zero_as_null: Treat numeric 0 values as NULL for matching
            debug: Enable debug logging
        """
        self.classification = classification
        self.field_mapping = field_mapping or {}
        self.treat_zero_as_null = treat_zero_as_null
        self.debug = debug

        # Store classification field names
        self.classification_field_names = [f.name for f in classification.fields]

        # Build reverse mapping (classification_field -> gpkg_field) for easier lookup
        self.reverse_field_mapping = {v: k for k, v in self.field_mapping.items()}

        logger.debug(f"Initialized matcher for {classification.layer_name}")
        logger.debug(f"Classification fields: {self.classification_field_names}")
        logger.debug(f"Treat zero as null: {treat_zero_as_null}")
        if self.field_mapping:
            logger.debug(f"Field mapping: {self.field_mapping}")

    @property
    def required_gpkg_fields(self) -> List[str]:
        """Get list of required fields in the GPKG (after mapping)."""
        if self.field_mapping:
            # Use mapped names
            return [
                self.reverse_field_mapping.get(f, f)
                for f in self.classification_field_names
            ]
        else:
            # Use original names
            return self.classification_field_names

    def check_required_fields(self, gdf: gpd.GeoDataFrame) -> Tuple[bool, List[str]]:
        """
        Check if all required fields are present in GeoDataFrame.

        Args:
            gdf: Input GeoDataFrame

        Returns:
            Tuple of (all_present, missing_fields)
        """
        gdf_columns = set(gdf.columns)
        required_fields = set(self.required_gpkg_fields)
        missing_fields = required_fields - gdf_columns

        return len(missing_fields) == 0, list(missing_fields)

    def match_feature(
        self, feature: pd.Series, feature_index: Any = None
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Find matching classification class for a feature.

        Args:
            feature: Feature as pandas Series
            feature_index: Optional feature index for debugging

        Returns:
            Tuple of (class_label, list_of_all_matching_labels) if match found, None otherwise
        """
        if self.debug and feature_index is not None:
            logger.debug(f"--- Matching feature {feature_index} ---")
            for field in self.required_gpkg_fields:
                value = feature.get(field)
                logger.debug(f"  {field} = {value} (type: {type(value).__name__})")

        all_matches = []

        for class_idx, class_obj in enumerate(self.classification.classes):
            if not class_obj.visible:
                continue

            if self.debug and feature_index is not None:
                logger.debug(f"  Testing class {class_idx}: {class_obj.label}")

            if self._evaluate_class(feature, class_obj, feature_index):
                if self.debug and feature_index is not None:
                    logger.debug(f"  ✓ MATCHED: {class_obj.label}")
                all_matches.append(class_obj.label)

        if self.debug and feature_index is not None:
            if len(all_matches) == 0:
                logger.debug(f"  ✗ NO MATCH for feature {feature_index}")
            elif len(all_matches) > 1:
                logger.warning(
                    f"  ⚠️  MULTIPLE MATCHES for feature {feature_index}: {all_matches}"
                )
                logger.warning(f"     Using first match: {all_matches[0]}")

        # Return first match (ESRI uses first match logic) plus all matches for debugging
        if all_matches:
            return (all_matches[0], all_matches)

        return None

    def _evaluate_class(
        self,
        feature: pd.Series,
        class_obj: ClassificationClass,
        feature_index: Any = None,
    ) -> bool:
        """
        Evaluate if a feature matches a classification class.

        Args:
            feature: Feature to evaluate
            class_obj: Classification class with field values
            feature_index: Optional index for debugging

        Returns:
            True if feature matches class
        """
        # A class can have multiple value combinations (OR logic)
        for combo_idx, field_values in enumerate(class_obj.field_values):
            if self.debug and feature_index is not None:
                logger.debug(f"    Combination {combo_idx}: {field_values}")

            if self._evaluate_value_combination(feature, field_values, feature_index):
                return True

        return False

    """
    Fixed ClassificationMatcher with corrected 999997 handling and LABEL field support
    """

    def _evaluate_value_combination(
        self, feature: pd.Series, field_values: List[str], feature_index: Any = None
    ) -> bool:
        """
        Evaluate if a feature matches a specific value combination.

        Args:
            feature: Feature to evaluate
            field_values: List of expected values for each field (AND logic)
            feature_index: Optional index for debugging

        Returns:
            True if all field values match
        """
        if len(field_values) != len(self.classification_field_names):
            logger.warning(
                f"Mismatch in field count: expected {len(self.classification_field_names)}, got {len(field_values)}"
            )
            return False

        for classification_field, expected_value in zip(
            self.classification_field_names, field_values
        ):
            # Map to GPKG field name
            gpkg_field = self.reverse_field_mapping.get(
                classification_field, classification_field
            )

            # Get feature value
            feature_value = feature.get(gpkg_field)

            # Check if we should treat 0 as null
            if (
                self.treat_zero_as_null
                and not pd.isna(feature_value)
                and feature_value == 0
            ):
                feature_value = None

            # Handle NULL values in expected
            if expected_value == "<Null>":
                is_match = pd.isna(feature_value) or feature_value is None
                if self.debug and feature_index is not None:
                    logger.debug(
                        f"      {gpkg_field}: {feature_value} vs <Null> → {is_match}"
                    )
                if not is_match:
                    return False
            else:
                # FIXED: Treat 999997 as a regular value, not a wildcard
                # It represents "unknown/unspecified" in the ESRI data model
                is_match = self._values_match(feature_value, expected_value)
                if self.debug and feature_index is not None:
                    logger.debug(
                        f"      {gpkg_field}: {feature_value} vs {expected_value} → {is_match}"
                    )
                if not is_match:
                    return False

        return True

    def _values_match(self, feature_value: Any, expected_value: str) -> bool:
        """
        Compare feature value with expected value.

        Args:
            feature_value: Actual value from feature
            expected_value: Expected value from classification

        Returns:
            True if values match
        """
        # Handle null/missing values
        if pd.isna(feature_value):
            return False

        # Convert both to strings for comparison
        feature_str = str(feature_value).strip()
        expected_str = str(expected_value).strip()

        # Try numeric comparison first
        try:
            feature_num = float(feature_str)
            expected_num = float(expected_str)
            return abs(feature_num - expected_num) < 1e-9
        except (ValueError, TypeError):
            # Fall back to string comparison
            return feature_str.lower() == expected_str.lower()


class ClassificationApplicator:
    """Apply ESRI classification rules to GeoDataFrames."""

    def __init__(
        self,
        classification: LayerClassification,
        symbol_field: str = "SYMBOL",
        label_field: Optional[str] = "LABEL",
        symbol_prefix: Optional[str] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        field_types: Optional[Dict[str, str]] = {},
        treat_zero_as_null: bool = False,
        debug: bool = False,
    ):
        """
        Initialize applicator.

        Args:
            classification: Layer classification with rules
            symbol_field: Name of field to add for symbol IDs
            label_field: Name of field to add for class labels (None to disable)
            symbol_prefix: Prefix for symbol IDs (defaults to layer name)
            field_mapping: Optional field name mapping (gpkg_field -> classification_field)
            treat_zero_as_null: Treat numeric 0 values as NULL for matching
            debug: Enable debug logging
        """
        self.classification = classification
        self.symbol_field = symbol_field
        self.field_types = field_types
        self.label_field = label_field
        self.symbol_prefix = symbol_prefix or self._sanitize_prefix(
            classification.layer_name or "symbol"
        )
        self.debug = debug

        self.matcher = ClassificationMatcher(
            classification, field_mapping, treat_zero_as_null, debug
        )

        # Build symbol mapping
        self.symbol_map = self._build_symbol_map()

        logger.info(
            f"Initialized applicator for '{classification.layer_name}' with {len(self.symbol_map)} classes"
        )
        if label_field:
            label_field_msg = f"Will add both SYMBOL ({symbol_field}) and LABEL ({label_field}) fields"
            logger.info(label_field_msg)
            console.print(label_field_msg)

        if field_mapping:
            field_mapping_msg = f"  Using field mapping: {field_mapping}"
            logger.debug(field_mapping_msg)
            console.print(field_mapping_msg)
        if treat_zero_as_null:
            treat_zero_as_null_msg = "  Treating 0 values as NULL for matching"
            logger.info(treat_zero_as_null_msg)
            console.print(treat_zero_as_null_msg)

    def _sanitize_prefix(self, name: str) -> str:
        """Sanitize name for use as symbol prefix."""
        # Convert to lowercase and replace spaces/special chars with underscore
        sanitized = re.sub(r"[^\w]+", "_", name.lower())
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized

    def _build_symbol_map(self) -> Dict[str, str]:
        """Build mapping from class labels to symbol IDs."""
        symbol_map = {}

        for idx, class_obj in enumerate(self.classification.classes):
            if class_obj.visible:
                symbol_id = f"{self.symbol_prefix}_{idx}"
                symbol_map[class_obj.label] = symbol_id

        return symbol_map

    # Used by batch
    def apply_v2(
        self,
        gdf: gpd.GeoDataFrame,
        additional_filter: Optional[str] = None,
        overwrite: bool = False,
        preserve_existing: bool = True,
    ) -> gpd.GeoDataFrame:  # NEW PARAMETER
        """
        Apply classification to GeoDataFrame.

        Args:
            preserve_existing: If True and SYMBOL field exists, only update NULL/empty values
                              If False, update all matching features (overwrite mode)
        """

        # Check if fields exist
        symbol_exists = self.symbol_field in gdf.columns
        label_exists = self.label_field and self.label_field in gdf.columns

        if symbol_exists and not overwrite and not preserve_existing:
            logger.warning(
                f"Field '{self.symbol_field}' exists. Use overwrite=True or preserve_existing=True"
            )
            return gdf

        # Initialize fields if they don't exist
        if not symbol_exists or overwrite:  # TODO erase everything
            gdf[self.symbol_field] = None
            # TODO
            gdf[self.symbol_field] = np.nan  # or pd.NA
            gdf[col] = gdf[col].astype("Int64")

            logger.info(f"Created new field: {self.symbol_field}")

        if self.label_field and not label_exists:
            gdf[self.label_field] = None
            logger.info(f"Created new field: {self.label_field}")

        # TODO
        # Check required fields
        all_present, missing = self.matcher.check_required_fields(gdf)
        if not all_present:
            raise ValueError(f"Missing required fields in GPKG: {missing}")

        logger.success(
            f"All required GPKG fields present: {self.matcher.required_gpkg_fields}"
        )
        logger.info(f"Fields to cast: {self.field_types}")  # TODO check why not casting

        # Step 2: Cast fields to correct types
        if self.field_types:
            console.print(f"\n[cyan]📊 Casting field types...[/cyan]")
            for field, dtype in self.field_types.items():
                if field in gdf.columns:
                    try:
                        if dtype.lower().startswith("int"):
                            dtype = dtype.capitalize()
                            gdf[field] = pd.to_numeric(gdf[field], errors="coerce")
                        gdf[field] = gdf[field].astype(dtype)
                        console.print(f"[green]✓ {field} → {dtype}[/green]")
                    except Exception as e:
                        console.print(f"[red]✗ {field}: {e}[/red]")

        # Step 3: Extract numeric columns
        numeric_columns = get_numeric_field_names(self.field_types)
        console.print(f"[cyan]Numeric columns: {', '.join(numeric_columns)}[/cyan]")

        # Step 4: Apply filter with numeric columns
        if additional_filter:
            pandas_filter = translate_esri_to_pandas(additional_filter)
            console.print(f"[cyan]Found filter: {pandas_filter}[/cyan]")
            gdf_filtered = apply_robust_filter(
                gdf,
                additional_filter=pandas_filter,
                numeric_columns=numeric_columns,
            )
        else:
            gdf_filtered = gdf.copy()
            console.print(f"[yellow]No filter[/yellow]")

        console.print(
            f"[cyan]Using {len(gdf_filtered)} out of total {len(gdf)} features[/cyan]"
        )

        # Initialize counters
        matched_count = 0
        unmatched_count = 0
        preserved_count = 0
        symbols = []
        labels = []
        console.print(f"   Matching {len(gdf_filtered)} features...")

        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Matching features...", total=len(gdf_filtered))

            for idx, feature in gdf_filtered.iterrows():
                # Check if we should preserve existing value
                # CORRECT - explicitly check for NULL
                if preserve_existing and symbol_exists:
                    existing_symbol = gdf.loc[idx, self.symbol_field]

                    # DEBUG
                    if idx in list(gdf_filtered.index)[:20]:  # First 20 features
                        logger.debug(
                            f"Feature {idx}: existing_symbol={existing_symbol}, type={type(existing_symbol)}, notna={pd.notna(existing_symbol)}"
                        )

                    # Check for both Python None and string "None"
                    if (
                        pd.notna(existing_symbol)
                        and existing_symbol is not None
                        and existing_symbol != "None"
                        and existing_symbol != ""
                    ):  # Also check empty string
                        symbols.append(existing_symbol)
                        if self.label_field:
                            labels.append(gdf.loc[idx, self.label_field])
                        preserved_count += 1
                        progress.advance(task)
                        continue

                # Match feature
                match_result = self.matcher.match_feature(feature)

                if match_result:
                    matched_label, all_matches = match_result
                    symbol_id = self.symbol_map[matched_label]
                    symbols.append(symbol_id)
                    labels.append(matched_label)
                    matched_count += 1
                else:
                    # Leave as NULL (or keep existing if preserve_existing)
                    if preserve_existing and symbol_exists:
                        symbols.append(gdf.loc[idx, self.symbol_field])
                        if self.label_field:
                            labels.append(gdf.loc[idx, self.label_field])
                    else:
                        symbols.append(None)
                        labels.append(None)
                    unmatched_count += 1

                # Update progress with stats every 100 features
                if (matched_count + unmatched_count + preserved_count) % 100 == 0:
                    progress.update(
                        task,
                        description=f"Matching features... (✓{matched_count} ⊘{unmatched_count} ↻{preserved_count})",
                    )

                progress.advance(task)

        # Final summary after progress bar closes
        console.print(f"\n[green]✅ Feature matching complete:[/green]")
        console.print(f"  • Matched: {matched_count}")
        console.print(f"  • Unmatched: {unmatched_count}")
        console.print(f"  • Preserved: {preserved_count}")
        console.print(f"  • Total: {len(gdf_filtered)}")

        # Update the dataframe
        # gdf.loc[gdf_filtered.index, self.symbol_field] = symbols
        # cast to int TODO
        gdf.loc[gdf_filtered.index, self.symbol_field] = pd.Series(
            symbols, index=gdf_filtered.index
        ).astype("str")
        # TODO
        gdf.loc[gdf_filtered.index, self.symbol_field] = pd.Series(
            symbols, index=gdf_filtered.index
        ).apply(
            lambda val: str(int(float(val)))
            if str(val).replace(".", "", 1).isdigit() and float(val).is_integer()
            else str(val)
        )

        if self.label_field:
            gdf.loc[gdf_filtered.index, self.label_field] = labels

        logger.info(f"Classification complete:")
        logger.info(f"  • Newly matched: {matched_count} features")
        logger.info(f"  • Preserved existing: {preserved_count} features")
        logger.info(f"  • Unmatched: {unmatched_count} features")

        return gdf

    def apply(
        self,
        gdf: gpd.GeoDataFrame,
        additional_filter: Optional[str] = None,
        overwrite: bool = False,
        preserve_existing: Optional[bool] = True,  # TODO, not used
    ) -> gpd.GeoDataFrame:
        """
        Apply classification to GeoDataFrame.

        Args:
            gdf: Input GeoDataFrame
            additional_filter: Optional pandas query filter to apply first
            overwrite: Overwrite existing symbol field if present

        Returns:
            GeoDataFrame with added SYMBOL field
        """
        # Check if symbol field already exists
        if self.symbol_field in gdf.columns:
            if not overwrite:
                logger.warning(
                    f"Field '{self.symbol_field}' already exists. Use overwrite=True to replace."
                )
                return gdf
            else:
                logger.info(f"Overwriting existing '{self.symbol_field}' field")

        # Check if label field already exists
        if self.label_field and self.label_field in gdf.columns:
            if not overwrite:
                logger.warning(
                    f"Field '{self.label_field}' already exists. Use overwrite=True to replace."
                )
            else:
                logger.info(f"Overwriting existing '{self.label_field}' field")

        # Check required fields
        all_present, missing = self.matcher.check_required_fields(gdf)
        if not all_present:
            raise ValueError(f"Missing required fields in GPKG: {missing}")

        logger.success(
            f"All required GPKG fields present: {self.matcher.required_gpkg_fields}"
        )

        # Apply additional filter if provided
        if additional_filter:
            logger.info(f"Applying filter: {additional_filter}")
            try:
                gdf_filtered = gdf.query(additional_filter).copy()
                logger.info(f"Filter result: {len(gdf_filtered)}/{len(gdf)} features")
            except Exception as e:
                logger.error(f"Filter failed: {e}")
                raise
        else:
            gdf_filtered = gdf.copy()

        # Apply classification
        logger.info(f"Classifying {len(gdf_filtered)} features...")

        symbols = []
        labels = []
        matched_count = 0
        unmatched_count = 0
        multiple_matches_count = 0

        # Sample features for debugging
        if self.debug and len(gdf_filtered) > 0:
            logger.info("=== Debug: Sample features ===")
            sample_size = min(3, len(gdf_filtered))
            for i, (idx, feature) in enumerate(
                gdf_filtered.head(sample_size).iterrows()
            ):
                logger.info(f"Sample {i + 1}:")
                for field in self.matcher.required_gpkg_fields:
                    logger.info(f"  {field} = {feature[field]}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Classifying features...", total=len(gdf_filtered))

            for idx, feature in gdf_filtered.iterrows():
                # Find matching class (pass index for debug logging)
                if self.debug and idx in gdf_filtered.head(3).index:
                    match_result = self.matcher.match_feature(feature, idx)
                else:
                    match_result = self.matcher.match_feature(feature)

                if match_result:
                    matched_label, all_matches = match_result
                    symbol_id = self.symbol_map[matched_label]
                    symbols.append(symbol_id)
                    labels.append(matched_label)
                    matched_count += 1

                    # Track multiple matches
                    if len(all_matches) > 1:
                        multiple_matches_count += 1
                else:
                    symbols.append(None)
                    labels.append(None)
                    unmatched_count += 1

                progress.advance(task)

        # Add symbol field to filtered GeoDataFrame
        gdf_filtered[self.symbol_field] = symbols

        # Add label field if requested
        if self.label_field:
            gdf_filtered[self.label_field] = labels

        # If we filtered, merge back to original GeoDataFrame
        if additional_filter:
            # Create full symbol and label columns
            gdf[self.symbol_field] = None
            gdf.loc[gdf_filtered.index, self.symbol_field] = gdf_filtered[
                self.symbol_field
            ]

            if self.label_field:
                gdf[self.label_field] = None
                gdf.loc[gdf_filtered.index, self.label_field] = gdf_filtered[
                    self.label_field
                ]
        else:
            gdf = gdf_filtered

        # Log statistics
        logger.info(f"Classification complete:")
        logger.info(f"  • Matched: {matched_count} features")
        logger.info(f"  • Unmatched: {unmatched_count} features")

        if multiple_matches_count > 0:
            logger.warning(
                f"  • Multiple matches: {multiple_matches_count} features matched more than one class"
            )
            logger.warning(
                f"    (Using first match per ESRI standard, but this may indicate overlapping rules)"
            )

        if unmatched_count > 0:
            logger.warning(f"  • {unmatched_count} features did not match any class")

        return gdf

    def get_symbol_statistics(self, gdf: gpd.GeoDataFrame) -> Dict[str, int]:
        """
        Get statistics on symbol assignments.

        Args:
            gdf: GeoDataFrame with SYMBOL field

        Returns:
            Dictionary mapping symbol IDs to counts
        """
        if self.symbol_field not in gdf.columns:
            return {}

        return gdf[self.symbol_field].value_counts().to_dict()


def apply_classification_to_gdf(
    gdf: gpd.GeoDataFrame,
    classification: LayerClassification,
    symbol_field: str = "SYMBOL",
    symbol_prefix: Optional[str] = None,
    additional_filter: Optional[str] = None,
    field_mapping: Optional[Dict[str, str]] = None,
    field_types: Optional[Dict[str, str]] = None,
    treat_zero_as_null: bool = False,
    debug: bool = False,
    overwrite: bool = False,
) -> gpd.GeoDataFrame:
    """
    Convenience function to apply classification to GeoDataFrame.

    Args:
        gdf: Input GeoDataFrame
        classification: Layer classification with rules
        symbol_field: Name of field to add for symbols
        symbol_prefix: Prefix for symbol IDs
        additional_filter: Optional pandas query filter
        field_mapping: Optional field name mapping (gpkg_field -> classification_field)
        treat_zero_as_null: Treat numeric 0 values as NULL for matching
        debug: Enable debug logging
        overwrite: Overwrite existing symbol field

    Returns:
        GeoDataFrame with added SYMBOL field
    """
    applicator = ClassificationApplicator(
        classification=classification,
        symbol_field=symbol_field,
        symbol_prefix=symbol_prefix,
        field_mapping=field_mapping,
        field_types=field_types,
        treat_zero_as_null=treat_zero_as_null,
        debug=debug,
    )

    return applicator.apply(
        gdf, additional_filter=additional_filter, overwrite=overwrite
    )


def apply_classification_to_gpkg(
    gpkg_path: Union[str, Path],
    classification: LayerClassification,
    layer_name: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    symbol_field: str = "SYMBOL",
    symbol_prefix: Optional[str] = None,
    additional_filter: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """
    Apply classification to GPKG file.

    Args:
        gpkg_path: Path to input GPKG
        classification: Layer classification with rules
        layer_name: Specific layer to process (required if GPKG has multiple layers)
        output_path: Output GPKG path (defaults to input_classified.gpkg)
        symbol_field: Name of field to add for symbols
        symbol_prefix: Prefix for symbol IDs
        additional_filter: Optional pandas query filter
        overwrite: Overwrite existing symbol field

    Returns:
        Path to output GPKG
    """
    gpkg_path = Path(gpkg_path)

    if not gpkg_path.exists():
        raise FileNotFoundError(f"GPKG not found: {gpkg_path}")

    logger.info(f"Loading GPKG: {gpkg_path}")

    # List available layers
    import fiona

    layers = fiona.listlayers(str(gpkg_path))
    logger.info(f"Available layers: {layers}")

    # Determine which layer to use
    if layer_name:
        if layer_name not in layers:
            raise ValueError(f"Layer '{layer_name}' not found. Available: {layers}")
        target_layer = layer_name
    elif len(layers) == 1:
        target_layer = layers[0]
        logger.info(f"Using single layer: {target_layer}")
    else:
        raise ValueError(
            f"Multiple layers found. Please specify --layer. Available: {layers}"
        )

    # Read GeoDataFrame
    logger.info(f"Reading layer: {target_layer}")
    gdf = gpd.read_file(gpkg_path, layer=target_layer)
    logger.info(f"Loaded {len(gdf)} features with {len(gdf.columns)} columns")

    # Apply classification
    gdf_classified = apply_classification_to_gdf(
        gdf=gdf,
        classification=classification,
        symbol_field=symbol_field,
        symbol_prefix=symbol_prefix,
        additional_filter=additional_filter,
        overwrite=overwrite,
    )

    # Determine output path
    if output_path is None:
        output_path = gpkg_path.parent / f"{gpkg_path.stem}_classified.gpkg"
    else:
        output_path = Path(output_path)

    # Save to GPKG
    logger.info(f"Saving to: {output_path}")
    gdf_classified.to_file(output_path, layer=target_layer, driver="GPKG")

    logger.success(f"Saved classified data to {output_path}")

    return output_path


def display_classification_summary(
    gdf: gpd.GeoDataFrame,
    classification: LayerClassification,
    symbol_field: str = "SYMBOL",
    symbol_prefix: Optional[str] = None,
):
    """Display summary of classification results."""
    if symbol_field not in gdf.columns:
        console.print(
            f"[red]Symbol field '{symbol_field}' not found in GeoDataFrame[/red]"
        )
        return

    # Build reverse mapping (symbol_id -> label) using the SAME prefix as during application
    applicator = ClassificationApplicator(
        classification,
        symbol_field=symbol_field,
        symbol_prefix=symbol_prefix,  # CRITICAL: Use the same prefix!
    )
    reverse_map = {v: k for k, v in applicator.symbol_map.items()}

    # Get statistics
    symbol_counts = gdf[symbol_field].value_counts()

    # Create results table
    table = Table(title="🎨 Classification Results", show_header=True)
    table.add_column("Symbol ID", style="cyan", width=20)
    table.add_column("Class Label", style="yellow", width=30)
    table.add_column("Count", style="green", justify="right", width=10)
    table.add_column("Percentage", style="blue", justify="right", width=12)

    total_features = len(gdf)

    for symbol_id, count in symbol_counts.items():
        if pd.isna(symbol_id) or symbol_id is None:
            label = "[dim]<unmatched>[/dim]"
            symbol_display = "[dim]None[/dim]"
        else:
            # Convert to string for lookup
            symbol_str = str(symbol_id)
            label = reverse_map.get(symbol_str, "[yellow]<unknown>[/yellow]")
            symbol_display = symbol_str

        percentage = (count / total_features) * 100
        table.add_row(symbol_display, label, str(count), f"{percentage:.1f}%")

    console.print(table)

    # Summary
    # Count only recognized symbols (those in reverse_map)
    matched = 0
    for symbol_id, count in symbol_counts.items():
        if symbol_id and str(symbol_id) in reverse_map:
            matched += count

    unmatched = total_features - matched

    console.print(f"\n[bold yellow]Summary:[/bold yellow]")
    console.print(f"  • Total features: {total_features}")
    console.print(
        f"  • Matched: [green]{matched}[/green] ({matched / total_features * 100:.1f}%)"
    )
    if unmatched > 0:
        console.print(
            f"  • Unmatched/Unknown: [red]{unmatched}[/red] ({unmatched / total_features * 100:.1f}%)"
        )
        console.print(
            f"  [dim]Hint: Use --overwrite to replace existing SYMBOL values[/dim]"
        )

    # Debug info if all showing as unknown
    if matched == 0 and total_features > 0:
        console.print(
            f"\n[red]⚠️  DEBUG: No symbols matched the expected mapping![/red]"
        )
        console.print(
            f"Symbol prefix used for display: [cyan]{applicator.symbol_prefix}[/cyan]"
        )
        console.print(f"Expected symbols: [dim]{list(reverse_map.keys())[:5]}...[/dim]")
        actual_symbols = [str(s) for s in symbol_counts.index[:5] if s is not None]
        console.print(f"Actual symbols in data: [dim]{actual_symbols}...[/dim]")


# =============================================================================
# CLICK CLI INTERFACE
# =============================================================================


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
def cli(verbose: bool, quiet: bool):
    """🎨 Classification Symbol Applicator

    Apply ESRI classification rules to GeoDataFrames/GPKG files.
    Adds a SYMBOL field with generated class identifiers based on classification rules.
    """
    if quiet:
        logger.remove()
        logger.add(sys.stdout, level="ERROR", format="<red>{level}</red>: {message}")
    elif verbose:
        logger.remove()
        logger.add(
            sys.stdout,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        )


@cli.command()
@click.argument(
    "gpkg_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="GPKG_FILE",
)
@click.argument(
    "style_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="STYLE_FILE",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output GPKG path (default: input_classified.gpkg)",
)
@click.option(
    "--layer", "-l", help="GPKG layer name to process (required for multi-layer GPKG)"
)
@click.option(
    "--classification-name",
    "-c",
    help='Classification name from style file (e.g., "Fossils" when GPKG layer is "GC_FOSSILS")',
)
@click.option(
    "--use-style-name/--no-use-style-name",
    default=True,
    help="Use style name as output layer name",
)
@click.option(
    "--mapping-config",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    help="YAML configuration file for layer and field mappings",
)
@click.option(
    "--filter",
    "-f",
    "filter_expr",
    help='Additional filter expression (pandas query syntax, e.g., "KIND == 14601006")',
)
@click.option(
    "--symbol-field",
    default="SYMBOL",
    show_default=True,
    help="Name of symbol field to add/update",
)
@click.option(
    "--symbol-prefix",
    "-p",
    help="Prefix for symbol IDs (default: derived from classification name)",
)
@click.option(
    "--treat-zero-as-null",
    is_flag=True,
    help="🔧 Treat numeric 0 values as NULL when matching (useful when GPKG has 0 instead of NULL)",
)
@click.option(
    "--debug-matching",
    is_flag=True,
    help="🐛 Show detailed debug info for first 3 features (helps diagnose matching issues)",
)
@click.option(
    "--overwrite", is_flag=True, help="⚠️  Overwrite existing symbol field if present"
)
@click.option(
    "--no-arcpy", is_flag=True, help="Force JSON parsing for style file (disable arcpy)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Check compatibility without applying classification",
)
def apply(
    gpkg_file: Path,
    style_file: Path,
    output: Optional[Path],
    layer: Optional[str],
    classification_name: Optional[str],
    mapping_config: Optional[Path],
    filter_expr: Optional[str],
    symbol_field: str,
    symbol_prefix: Optional[str],
    treat_zero_as_null: bool,
    debug_matching: bool,
    overwrite: bool,
    no_arcpy: bool,
    dry_run: bool,
    use_style_name: bool,
):
    """Apply classification rules from ESRI style to GPKG file.

    \b
    Arguments:
      GPKG_FILE    Input GeoPackage file (.gpkg)
      STYLE_FILE   ESRI layer style file (.lyrx)

    \b
    Basic workflow:
      1. Extracts classification rules from .lyrx style file
      2. Matches features in GPKG against classification rules
      3. Adds SYMBOL field with generated IDs (e.g., fossils_0, fossils_1)

    \b
    Examples:
      # Simple case - single layer GPKG, single classification
      classifier apply fossils.gpkg fossils_style.lyrx

      # Multi-layer GPKG with different classification name
      classifier apply geocover.gpkg fossils_style.lyrx -l GC_FOSSILS -c "Fossils"

      # Debug matching issues (shows first 3 features in detail)
      classifier apply fossils.gpkg style.lyrx --debug-matching

      # Treat 0 values as NULL (when GPKG has 0 instead of NULL)
      classifier apply fossils.gpkg style.lyrx --treat-zero-as-null

      # Combined: debug + treat zero as null
      classifier apply fossils.gpkg style.lyrx --debug-matching --treat-zero-as-null --overwrite

      # With filter to select specific features
      classifier apply data.gpkg fossils.lyrx -l points -f "KIND == 14601006"

      # Overwrite existing SYMBOL field
      classifier apply fossils.gpkg new_style.lyrx --overwrite

      # Use YAML mapping configuration
      classifier apply geocover.gpkg styles.lyrx --mapping-config mapping.yaml

      # Dry run to check compatibility
      classifier apply geocover.gpkg fossils.lyrx -l GC_FOSSILS -c "Fossils" --dry-run
    """
    overwritten = False
    target_layer = None
    classification_layer_name = None
    try:
        console.print(f"\n[bold blue]🎨 Applying classification[/bold blue]")
        console.print(f"[dim]GPKG: {gpkg_file.name}[/dim]")
        console.print(f"[dim]Style: {style_file.name}[/dim]\n")

        # Load mapping configuration if provided
        layer_mapping = None
        if mapping_config:
            logger.info(f"Loading mapping configuration from {mapping_config}")
            layer_mapping = LayerMapping(mapping_config)

        # Extract classifications from style file
        with console.status("[cyan]Loading classification rules...", spinner="dots"):
            all_classifications = extract_lyrx(
                style_file, use_arcpy=not no_arcpy, display=False
            )

        if not all_classifications:
            console.print("[red]❌ No classifications found in style file![/red]")
            return

        # Select appropriate classification
        classification = None

        if classification_name:
            # Find classification by name
            for c in all_classifications:
                if c.layer_name == classification_name:
                    classification = c
                    logger.success(f"Found classification: {classification_name}")
                    classification_layer_name = classification_name
                    break

            if not classification:
                console.print(
                    f"[red]❌ Classification '{classification_name}' not found![/red]"
                )
                console.print(
                    f"Available: {', '.join([c.layer_name for c in all_classifications])}"
                )
                return
        elif len(all_classifications) == 1:
            classification = all_classifications[0]
            classification_layer_name = classification_name
            logger.info(f"Using single classification: {classification.layer_name}")
        else:
            console.print(
                f"[yellow]⚠️  Multiple classifications found. Please specify --classification-name[/yellow]"
            )
            console.print(
                f"Available: {', '.join([c.layer_name for c in all_classifications])}"
            )
            return

        logger.success(
            f"Using classification: {classification.layer_name} with {len(classification.classes)} classes"
        )

        # Load GPKG
        with console.status("[cyan]Loading GPKG...", spinner="dots"):
            import fiona

            layers_available = fiona.listlayers(str(gpkg_file))

        logger.info(f"Available layers in GPKG: {layers_available}")

        # Determine target layer
        if layer:
            if layer not in layers_available:
                console.print(f"[red]❌ Layer '{layer}' not found![/red]")
                console.print(f"Available layers: {', '.join(layers_available)}")
                return
            target_layer = layer
        elif len(layers_available) == 1:
            target_layer = layers_available[0]
            logger.info(f"Using single layer: {target_layer}")
        else:
            console.print(
                f"[red]❌ Multiple layers found. Please specify --layer[/red]"
            )
            console.print(f"Available layers: {', '.join(layers_available)}")
            return

        # Get mapping configuration for this layer
        field_mapping = None
        if layer_mapping:
            field_mapping = layer_mapping.get_field_mapping(target_layer)

            # Override other options from mapping config if not specified
            if not filter_expr:
                filter_expr = layer_mapping.get_filter(target_layer)
            if not symbol_prefix:
                symbol_prefix = layer_mapping.get_symbol_prefix(target_layer)

            # Also check if classification name should be overridden
            if not classification_name:
                mapped_class_name = layer_mapping.get_classification_name(target_layer)
                if mapped_class_name and mapped_class_name != classification.layer_name:
                    logger.info(
                        f"Mapping config specifies classification: {mapped_class_name}"
                    )
                    # Find the right classification
                    for c in all_classifications:
                        if c.layer_name == mapped_class_name:
                            classification = c
                            logger.success(
                                f"Switched to classification: {mapped_class_name}"
                            )
                            break

        # Read GeoDataFrame
        with console.status(f"[cyan]Reading layer '{target_layer}'...", spinner="dots"):
            gdf = gpd.read_file(gpkg_file, layer=target_layer)

        logger.info(f"Loaded {len(gdf)} features")

        # Check field compatibility
        matcher = ClassificationMatcher(
            classification, field_mapping, treat_zero_as_null, debug_matching
        )
        all_present, missing = matcher.check_required_fields(gdf)

        if not all_present:
            console.print(f"[red]❌ Missing required fields in GPKG: {missing}[/red]")
            console.print(f"Required GPKG fields: {matcher.required_gpkg_fields}")
            console.print(
                f"Classification expects: {matcher.classification_field_names}"
            )
            console.print(f"Available in GPKG: {list(gdf.columns)}")
            if field_mapping:
                console.print(f"Field mapping used: {field_mapping}")
            else:
                console.print(
                    "[yellow]Hint: Use --mapping-config to specify field name mappings[/yellow]"
                )
            return

        console.print(
            f"[green]✅ All required fields present: {matcher.required_gpkg_fields}[/green]"
        )
        if layer_mapping:
            field_types = layer_mapping.get_field_types(target_layer)
        for field, dtype in field_types.items():
            if field in gdf.columns:
                try:
                    gdf[field] = gdf[field].astype(dtype)
                    console.print(f"[green]Casted {field} to {dtype}: {e}[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to cast {field} to {dtype}: {e}[/red]")

        # TODO
        if layer_mapping:
            field_types = layer_mapping.get_field_types(target_layer)

            if field_types:
                issues = validate_field_types(gdf, field_types)
                # if not issues:
                #     gdf = cast_geodataframe_fields(gdf, field_types)
                console.print(
                    f"\n[cyan]📊 Casting field types for '{target_layer}'...[/cyan]"
                )
                gdf = cast_geodataframe_fields(gdf, field_types, strict=False)
            else:
                console.print(
                    f"[yellow]No field type mappings for '{target_layer}'[/yellow]"
                )

        if field_mapping:
            console.print(f"[dim]Field mapping: {field_mapping}[/dim]")
        if treat_zero_as_null:
            console.print(f"[yellow]⚠️  Treating 0 values as NULL for matching[/yellow]")

        # Show sample data if debug is enabled
        if debug_matching:
            console.print("\n[bold yellow]🐛 Debug: Sample Data[/bold yellow]")
            sample_df = gdf[matcher.required_gpkg_fields].head(5)

            table = Table(title="First 5 features", show_header=True)
            table.add_column("Index", style="dim")
            for field in matcher.required_gpkg_fields:
                table.add_column(field, style="cyan")

            for idx, row in sample_df.iterrows():
                values = [str(idx)] + [
                    str(row[field]) for field in matcher.required_gpkg_fields
                ]
                table.add_row(*values)

            console.print(table)

            # Show dtypes
            console.print("\n[bold yellow]Data types:[/bold yellow]")
            for field in matcher.required_gpkg_fields:
                console.print(f"  {field}: {gdf[field].dtype}")

            # Show value counts for each field
            console.print("\n[bold yellow]Unique values (sample):[/bold yellow]")
            for field in matcher.required_gpkg_fields:
                unique_vals = gdf[field].value_counts().head(5)
                console.print(f"  {field}:")
                for val, count in unique_vals.items():
                    console.print(f"    {val}: {count} occurrences")

            # Show expected classification values
            console.print(
                "\n[bold yellow]Expected classification values:[/bold yellow]"
            )
            for class_idx, class_obj in enumerate(classification.classes):
                console.print(f"  Class {class_idx}: {class_obj.label}")
                for combo_idx, field_values in enumerate(class_obj.field_values):
                    console.print(
                        f"    Combination {combo_idx}: {dict(zip(matcher.classification_field_names, field_values))}"
                    )
            console.print()

        if dry_run:
            console.print(
                "\n[yellow]🔍 Dry run mode - no changes will be made[/yellow]\n"
            )

            # Show what would happen
            console.print(f"Would classify {len(gdf)} features")
            if filter_expr:
                console.print(f"With filter: {filter_expr}")
            console.print(f"Symbol field: {symbol_field}")
            console.print(
                f"Symbol prefix: {symbol_prefix or classification.layer_name}"
            )

            return

        # Apply classification
        gdf_classified = apply_classification_to_gdf(
            gdf=gdf,
            classification=classification,
            symbol_field=symbol_field,
            symbol_prefix=symbol_prefix,
            field_mapping=field_mapping,
            treat_zero_as_null=treat_zero_as_null,
            debug=debug_matching,
            additional_filter=filter_expr,
            overwrite=overwrite,
        )

        # Display results
        display_classification_summary(
            gdf_classified, classification, symbol_field, symbol_prefix
        )

        # Save output
        if output is None:
            output = gpkg_file.parent / f"{gpkg_file.stem}_classified.gpkg"

        if use_style_name:
            target_layer = str(classification.layer_name)
            original_nb = len(gdf_classified)
            # Separate rows with null SYMBOL
            gdf_missing = gdf_classified[gdf_classified[symbol_field].isna()]

            missing_output = output.parent / f"{output.stem}.missing.gpkg"

            # Save them to a file (e.g., GeoPackage or Shapefile)
            gdf_missing.to_file(missing_output, layer=target_layer)

            # Keep only rows with valid SYMBOL
            gdf_classified = gdf_classified.dropna(subset=[symbol_field])
            logger.warning(
                f"Dropped {len(gdf_missing)} features with missing symbol from result "
            )

        kwargs = {
            "driver": "GPKG",
            "layer": target_layer,
            "engine": "fiona",
        }

        if output.exists():
            target_layers = fiona.listlayers(str(output))
            if target_layer in target_layers:
                overwritten = True
                kwargs["OVERWRITE"] = "YES"
                mode = "w"
                if not overwrite:
                    if not Confirm.ask(
                        f"Output file '{output}' and layer '{target_layer}' exist. Overwrite?"
                    ):
                        raise click.Abort()
        else:
            mode = "w"

        kwargs["mode"] = mode

        # Save the file
        with console.status(f"[cyan]Saving to {output}...", spinner="dots"):
            gdf_classified.to_file(output, **kwargs)

        console.print(
            Panel.fit(
                f"[green]✅ Classification applied successfully![/green]\n"
                f"[dim]Input: {gpkg_file.name}[/dim]\n"
                f"[dim]Output: {output.name}[/dim]\n"
                f"[dim]Layer: {target_layer}[/dim]\n"
                f"[dim]Classification: {classification.layer_name}[/dim]\n"
                f"[dim]Symbol field: {symbol_field}[/dim]\n"
                f"[dim]Overwritten: {'yes' if overwritten else 'no'}[/dim]",
                title="🎉 Success",
                border_style="green",
            )
        )

    except Exception as e:
        logger.error(f"Style application failed: {e}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        console.print(kwargs)

        logger.debug(traceback.format_exc())
        return


@cli.command()
@click.argument(
    "output_path", type=click.Path(path_type=Path), default="mapping_example.yaml"
)
def create_mapping(output_path: Path):
    """Create an example YAML mapping configuration file.

    This creates a template configuration file showing how to map
    GPKG layer names to classification names and remap field names.

    Example:

      classifier create-mapping my_mapping.yaml
    """
    try:
        LayerMapping.create_example_yaml(output_path)

        console.print(
            Panel.fit(
                f"[green]✅ Example configuration created![/green]\n\n"
                f"[dim]File: {output_path}[/dim]\n\n"
                f"Edit this file to match your GPKG layers and classifications.\n"
                f"Then use with: [cyan]classifier apply --mapping-config {output_path}[/cyan]",
                title="📝 Mapping Configuration",
                border_style="green",
            )
        )

        # Display the example content
        with open(output_path, "r") as f:
            content = f.read()

        console.print("\n[bold yellow]Example content:[/bold yellow]")
        console.print(f"[dim]{content}[/dim]")

    except Exception as e:
        logger.error(f"Failed to create mapping: {e}")
        return


@cli.command()
@click.argument(
    "gpkg_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="GPKG_FILE",
)
@click.argument(
    "style_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="STYLE_FILE",
)
@click.option("--layer", "-l", help="GPKG layer name to check")
@click.option("--classification-name", "-c", help="Classification name from style file")
@click.option(
    "--mapping-config",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    help="YAML configuration file for field mappings",
)
@click.option("--no-arcpy", is_flag=True, help="Force JSON parsing for style file")
def check(
    gpkg_file: Path,
    style_file: Path,
    layer: Optional[str],
    classification_name: Optional[str],
    mapping_config: Optional[Path],
    no_arcpy: bool,
):
    """Check field compatibility between GPKG and style file.

    \b
    Arguments:
      GPKG_FILE    Input GeoPackage file (.gpkg)
      STYLE_FILE   ESRI layer style file (.lyrx)

    Analyzes the GPKG and style file to verify that all required fields
    are present before applying classification.

    \b
    Examples:
      # Basic check
      classifier check fossils.gpkg fossils_style.lyrx

      # Check specific layer and classification
      classifier check geocover.gpkg style.lyrx -l GC_FOSSILS -c "Fossils"

      # Check with YAML mapping
      classifier check geocover.gpkg styles.lyrx --mapping-config mapping.yaml
    """
    try:
        console.print(f"\n[bold blue]🔍 Checking compatibility[/bold blue]")
        console.print(f"[dim]GPKG: {gpkg_file.name}[/dim]")
        console.print(f"[dim]Style: {style_file.name}[/dim]\n")

        # Load mapping configuration if provided
        layer_mapping = None
        if mapping_config:
            logger.info(f"Loading mapping configuration from {mapping_config}")
            layer_mapping = LayerMapping(mapping_config)

        # Load classification
        with console.status("[cyan]Loading classification rules...", spinner="dots"):
            all_classifications = extract_lyrx(
                style_file, use_arcpy=not no_arcpy, display=False
            )

        if not all_classifications:
            console.print("[red]❌ No classifications found![/red]")
            return

        # Select classification
        classification = None
        if classification_name:
            for c in all_classifications:
                if c.layer_name == classification_name:
                    classification = c
                    break
            if not classification:
                console.print(
                    f"[red]Classification '{classification_name}' not found![/red]"
                )
                console.print(
                    f"Available: {', '.join([c.layer_name for c in all_classifications])}"
                )
                return
        elif len(all_classifications) == 1:
            classification = all_classifications[0]
        else:
            console.print(
                "[yellow]Multiple classifications found. Specify --classification-name[/yellow]"
            )
            console.print(
                f"Available: {', '.join([c.layer_name for c in all_classifications])}"
            )
            return

        # Load GPKG
        import fiona

        layers_available = fiona.listlayers(str(gpkg_file))

        target_layer = layer or (
            layers_available[0] if len(layers_available) == 1 else None
        )

        if not target_layer:
            console.print(f"[red]Multiple layers found. Specify --layer[/red]")
            console.print(f"Available: {', '.join(layers_available)}")
            return

        gdf = gpd.read_file(gpkg_file, layer=target_layer)

        # Get field mapping for this layer
        field_mapping = None
        if layer_mapping:
            field_mapping = layer_mapping.get_field_mapping(target_layer)

        # Check compatibility
        matcher = ClassificationMatcher(classification, field_mapping)
        all_present, missing = matcher.check_required_fields(gdf)

        # Display results
        table = Table(title="📋 Field Compatibility Check", show_header=True)
        table.add_column("Classification Field", style="yellow", width=25)
        table.add_column("GPKG Field", style="cyan", width=25)
        table.add_column("Present", style="green", width=10)
        table.add_column("Type", style="blue", width=15)

        for classification_field in matcher.classification_field_names:
            # Get GPKG field name (after mapping)
            gpkg_field = matcher.reverse_field_mapping.get(
                classification_field, classification_field
            )

            is_present = gpkg_field in gdf.columns
            field_type = str(gdf[gpkg_field].dtype) if is_present else "—"

            table.add_row(
                classification_field,
                gpkg_field
                if gpkg_field != classification_field
                else f"[dim]{gpkg_field}[/dim]",
                "✓" if is_present else "✗",
                field_type,
            )

        console.print(table)

        if field_mapping:
            console.print(f"\n[dim]Using field mapping: {field_mapping}[/dim]")

        if all_present:
            console.print(f"\n[green]✅ All required fields present![/green]")
            console.print(
                f"Ready to apply classification '{classification.layer_name}' with {len(classification.classes)} classes."
            )
        else:
            console.print(f"\n[red]❌ Missing GPKG fields: {missing}[/red]")
            console.print(
                "Cannot apply classification until missing fields are added or mapped."
            )
            if not field_mapping:
                console.print(
                    "[yellow]Hint: Use --mapping-config to specify field name mappings[/yellow]"
                )

    except Exception as e:
        logger.error(f"Check failed: {e}")
        return


if __name__ == "__main__":
    cli()
