#!/usr/bin/env python
"""
ArcPy-based in-place SYMBOL field updater for FileGDB
Uses native arcpy cursors to avoid potential FileGDB corruption from geopandas/pyogrio

This module provides a drop-in replacement for the geopandas-based classification
logic, but operates directly on FileGDB using arcpy.da cursors.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import arcpy
import yaml
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()


@dataclass
class ClassificationConfig:
    """Configuration for a single classification application."""
    style_file: Path
    mapfile_name: Optional[str] = None
    classification_name: Optional[str] = None
    fields: Optional[Dict[str, str]] = None
    filter: Optional[str] = None
    symbol_prefix: Optional[str] = None


@dataclass
class LayerConfig:
    """Configuration for a feature class with multiple classifications."""
    feature_class: str  # Changed from gpkg_layer
    classifications: List[ClassificationConfig]
    field_types: Optional[Dict[str, str]] = None


class ArcPyClassificationConfig:
    """Parse and manage batch classification configuration for FileGDB."""

    def __init__(self, config_path: Path, styles_base_path: Optional[Path] = None):
        """
        Load batch configuration from YAML.

        Args:
            config_path: Path to YAML config file
            styles_base_path: Base directory for resolving relative style paths
        """
        self.config_path = config_path
        self.styles_base_path = styles_base_path or config_path.parent

        with open(config_path, "r", encoding="utf-8") as f:
            self.raw_config = yaml.safe_load(f)

        # Parse global settings
        self.global_settings = self.raw_config.get("global", {})
        self.treat_zero_as_null = self.global_settings.get("treat_zero_as_null", False)
        self.symbol_field = self.global_settings.get("symbol_field", "SYMBOL")
        self.label_field = self.global_settings.get("label_field", "LABEL")
        self.overwrite = self.global_settings.get("overwrite", False)

        # Parse layer configurations
        self.layers: List[LayerConfig] = []
        for layer_config in self.raw_config.get("layers", []):
            self.layers.append(self._parse_layer_config(layer_config))

        logger.info(f"Loaded config with {len(self.layers)} feature classes")

    def _parse_layer_config(self, layer_dict: dict) -> LayerConfig:
        """Parse a single layer configuration."""
        # Support both 'feature_class' and 'gpkg_layer' keys for compatibility
        feature_class = layer_dict.get("feature_class") or layer_dict.get("gpkg_layer")
        classifications = []
        field_types = layer_dict.get("field_types", {})

        for class_dict in layer_dict.get("classifications", []):
            # Resolve style file path
            style_file = Path(class_dict["style_file"])
            if not style_file.is_absolute():
                style_file = self.styles_base_path / style_file

            classifications.append(
                ClassificationConfig(
                    style_file=style_file,
                    classification_name=class_dict.get("classification_name"),
                    fields=class_dict.get("fields"),
                    filter=class_dict.get("filter"),
                    symbol_prefix=class_dict.get("symbol_prefix"),
                )
            )

        return LayerConfig(
            feature_class=feature_class,
            classifications=classifications,
            field_types=field_types,
        )

    def get_layer_config(self, feature_class: str) -> Optional[LayerConfig]:
        """Get configuration for a specific feature class."""
        for layer in self.layers:
            if layer.feature_class == feature_class:
                return layer
        return None


class ArcPySymbolUpdater:
    """
    Apply classification rules to FileGDB feature classes using native arcpy cursors.

    This class operates in-place on FileGDB without loading data into memory,
    avoiding potential corruption from geopandas/pyogrio.
    """

    def __init__(
            self,
            gdb_path: Path,
            symbol_field: str = "SYMBOL",
            label_field: str = "LABEL",
            treat_zero_as_null: bool = False,
            overwrite: bool = False,
    ):
        """
        Initialize the ArcPy symbol updater.

        Args:
            gdb_path: Path to FileGDB
            symbol_field: Name of symbol field to update
            label_field: Name of label field to update
            treat_zero_as_null: Treat 0 values as NULL in comparisons
            overwrite: Overwrite existing symbol values
        """
        self.gdb_path = Path(gdb_path)
        self.symbol_field = symbol_field
        self.label_field = label_field
        self.treat_zero_as_null = treat_zero_as_null
        self.overwrite = overwrite

        # Set workspace
        arcpy.env.workspace = str(self.gdb_path)
        arcpy.env.overwriteOutput = True

        # Verify GDB exists
        if not arcpy.Exists(str(self.gdb_path)):
            raise FileNotFoundError(f"FileGDB not found: {self.gdb_path}")

        logger.info(f"Initialized ArcPy updater for: {self.gdb_path}")

    def ensure_symbol_fields(self, feature_class: str) -> None:
        """
        Ensure SYMBOL and LABEL fields exist in feature class.

        Args:
            feature_class: Path to feature class (can include dataset)
        """
        fc_path = str(self.gdb_path / feature_class)

        # Get existing fields
        existing_fields = {f.name.upper() for f in arcpy.ListFields(fc_path)}

        # Add SYMBOL field if missing
        if self.symbol_field.upper() not in existing_fields:
            arcpy.management.AddField(
                fc_path,
                self.symbol_field,
                "TEXT",
                field_length=100,
                field_alias="Classification Symbol"
            )
            logger.info(f"Added {self.symbol_field} field to {feature_class}")

        # Add LABEL field if missing
        if self.label_field and self.label_field.upper() not in existing_fields:
            arcpy.management.AddField(
                fc_path,
                self.label_field,
                "TEXT",
                field_length=255,
                field_alias="Classification Label"
            )
            logger.info(f"Added {self.label_field} field to {feature_class}")

    def load_classification_rules(
            self,
            style_file: Path,
            classification_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load classification rules from ESRI .lyrx file.

        Args:
            style_file: Path to .lyrx style file
            classification_name: Specific classification to extract

        Returns:
            List of classification rules with conditions and symbols
        """
        # This is a simplified version - you'll need to use your existing
        # extract_lyrx() function from esri_classification_extractor
        # For now, returning placeholder structure

        # Import your existing extractor
        try:
            from gcover.publish.esri_classification_extractor import extract_lyrx
            classifications = extract_lyrx(style_file, display=False)

            # Find the right classification
            if classification_name:
                for c in classifications:
                    if c.layer_name == classification_name:
                        return self._convert_classification_to_rules(c)
                logger.warning(f"Classification '{classification_name}' not found")
                return []
            elif len(classifications) == 1:
                return self._convert_classification_to_rules(classifications[0])
            else:
                logger.error(f"Multiple classifications found, specify classification_name")
                return []

        except ImportError:
            logger.error("Cannot import esri_classification_extractor")
            return []

    def _convert_classification_to_rules(self, classification) -> List[Dict[str, Any]]:
        """
        Convert classification object to list of rules.

        Args:
            classification: Classification object from extract_lyrx

        Returns:
            List of rule dictionaries with 'condition', 'symbol', 'label'
        """
        rules = []

        # Extract rules from classification renderer
        # This structure depends on your ClassificationRenderer implementation
        if hasattr(classification, 'renderer') and hasattr(classification.renderer, 'groups'):
            for group in classification.renderer.groups:
                for class_obj in group.classes:
                    rule = {
                        'condition': class_obj.where_clause if hasattr(class_obj, 'where_clause') else None,
                        'symbol': class_obj.symbol if hasattr(class_obj, 'symbol') else None,
                        'label': class_obj.label if hasattr(class_obj, 'label') else None,
                        'values': class_obj.values if hasattr(class_obj, 'values') else None,
                    }
                    rules.append(rule)

        return rules

    def build_sql_where_clause(
            self,
            rule: Dict[str, Any],
            field_mapping: Optional[Dict[str, str]] = None,
            additional_filter: Optional[str] = None
    ) -> str:
        """
        Build SQL WHERE clause from classification rule.

        Args:
            rule: Rule dictionary with condition/values
            field_mapping: Map config field names to actual field names
            additional_filter: Additional SQL filter to combine

        Returns:
            SQL WHERE clause string
        """
        clauses = []

        # Add rule condition
        if rule.get('condition'):
            condition = rule['condition']

            # Apply field mapping
            if field_mapping:
                for config_field, actual_field in field_mapping.items():
                    condition = condition.replace(f"${config_field}$", actual_field)
                    condition = condition.replace(config_field, actual_field)

            clauses.append(f"({condition})")

        # Add additional filter
        if additional_filter:
            clauses.append(f"({additional_filter})")

        # Combine with AND
        where_clause = " AND ".join(clauses) if clauses else "1=1"

        return where_clause

    def apply_classification_to_feature_class(
            self,
            feature_class: str,
            class_config: ClassificationConfig,
            progress: Optional[Progress] = None,
            task_id: Optional[Any] = None,
    ) -> Dict[str, int]:
        """
        Apply a single classification to a feature class using arcpy cursors.

        Args:
            feature_class: Feature class path (can include dataset)
            class_config: Classification configuration
            progress: Optional Rich progress bar
            task_id: Optional task ID for progress updates

        Returns:
            Dictionary with statistics (updated_count, skipped_count, etc.)
        """
        fc_path = str(self.gdb_path / feature_class)

        # Load classification rules
        rules = self.load_classification_rules(
            class_config.style_file,
            class_config.classification_name
        )

        if not rules:
            logger.warning(f"No rules loaded from {class_config.style_file.name}")
            return {"updated": 0, "skipped": 0, "errors": 0}

        stats = {"updated": 0, "skipped": 0, "errors": 0}

        # Get all fields we need to read
        read_fields = [self.symbol_field]
        if self.label_field:
            read_fields.append(self.label_field)

        # Add fields from field mapping
        if class_config.fields:
            read_fields.extend(class_config.fields.values())

        # Make sure OID is included for debugging
        read_fields.insert(0, "OID@")

        # Remove duplicates while preserving order
        read_fields = list(dict.fromkeys(read_fields))

        logger.info(f"Processing {feature_class} with {len(rules)} rules")
        logger.debug(f"Reading fields: {read_fields}")

        # Start edit session for File Geodatabase
        edit = arcpy.da.Editor(str(self.gdb_path))
        edit.startEditing(False, True)
        edit.startOperation()

        try:
            # Process each rule
            for rule_idx, rule in enumerate(rules):
                # Build WHERE clause for this rule
                where_clause = self.build_sql_where_clause(
                    rule,
                    class_config.fields,
                    class_config.filter
                )

                # Determine symbol value
                symbol_value = rule.get('symbol')
                if class_config.symbol_prefix and symbol_value:
                    symbol_value = f"{class_config.symbol_prefix}_{symbol_value}"

                label_value = rule.get('label')

                # Use UpdateCursor to update matching features
                with arcpy.da.UpdateCursor(
                        fc_path,
                        read_fields,
                        where_clause=where_clause
                ) as cursor:
                    for row in cursor:
                        oid = row[0]
                        current_symbol = row[read_fields.index(self.symbol_field)]

                        # Check if we should update
                        should_update = (
                                self.overwrite or
                                current_symbol is None or
                                current_symbol == "" or
                                current_symbol == "None"
                        )

                        if should_update and symbol_value:
                            # Update symbol
                            row[read_fields.index(self.symbol_field)] = symbol_value

                            # Update label if configured
                            if self.label_field and label_value:
                                row[read_fields.index(self.label_field)] = label_value

                            cursor.updateRow(row)
                            stats["updated"] += 1
                        else:
                            stats["skipped"] += 1

                        # Update progress
                        if progress and task_id:
                            progress.update(task_id, advance=1)

                logger.debug(f"Rule {rule_idx + 1}: {where_clause} -> {symbol_value} ({stats['updated']} updated)")

            # Commit changes
            edit.stopOperation()
            edit.stopEditing(True)

            logger.info(f"Updated {stats['updated']} features in {feature_class}")

        except Exception as e:
            logger.error(f"Error processing {feature_class}: {e}")
            stats["errors"] += 1

            # Rollback on error
            if edit.isEditing:
                edit.stopOperation()
                edit.stopEditing(False)

            raise

        return stats

    def apply_batch_from_config(
            self,
            config: ArcPyClassificationConfig,
            feature_classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Apply all classifications from config to FileGDB.

        Args:
            config: Batch classification configuration
            feature_classes: Specific feature classes to process (None = all)

        Returns:
            Dictionary with processing statistics
        """
        # Get list of all feature classes in GDB
        arcpy.env.workspace = str(self.gdb_path)
        available_fcs = []

        # List feature classes in root
        available_fcs.extend(arcpy.ListFeatureClasses())

        # List feature classes in datasets
        for dataset in arcpy.ListDatasets(feature_type='feature'):
            arcpy.env.workspace = str(self.gdb_path / dataset)
            for fc in arcpy.ListFeatureClasses():
                available_fcs.append(f"{dataset}/{fc}")

        # Reset workspace
        arcpy.env.workspace = str(self.gdb_path)

        # Determine which feature classes to process
        if feature_classes:
            fcs_to_process = [fc for fc in feature_classes if fc in available_fcs]
        else:
            # Process all feature classes that have config
            fcs_to_process = [
                layer.feature_class
                for layer in config.layers
                if layer.feature_class in available_fcs
            ]

        if not fcs_to_process:
            logger.warning("No feature classes to process!")
            return {}

        logger.info(f"Processing {len(fcs_to_process)} feature classes: {fcs_to_process}")

        # Statistics
        stats = {
            "feature_classes_processed": 0,
            "classifications_applied": 0,
            "features_updated": 0,
            "features_skipped": 0,
            "errors": 0,
        }

        # Process each feature class
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
        ) as progress:

            for fc in fcs_to_process:
                layer_config = config.get_layer_config(fc)
                if not layer_config:
                    logger.warning(f"No configuration for {fc}, skipping")
                    continue

                console.print(f"\n[bold blue]Processing: {fc}[/bold blue]")
                console.print(f"Applying {len(layer_config.classifications)} classifications")

                # Ensure symbol fields exist
                self.ensure_symbol_fields(fc)

                # Get total feature count for progress
                fc_path = str(self.gdb_path / fc)
                total_features = int(arcpy.management.GetCount(fc_path)[0])

                task = progress.add_task(
                    f"[cyan]{fc}",
                    total=total_features * len(layer_config.classifications)
                )

                # Apply each classification
                for i, class_config in enumerate(layer_config.classifications, 1):
                    console.print(
                        f"  [{i}/{len(layer_config.classifications)}] {class_config.style_file.name}"
                    )

                    try:
                        result = self.apply_classification_to_feature_class(
                            fc,
                            class_config,
                            progress=progress,
                            task_id=task
                        )

                        stats["features_updated"] += result["updated"]
                        stats["features_skipped"] += result["skipped"]
                        stats["errors"] += result["errors"]
                        stats["classifications_applied"] += 1

                        console.print(
                            f"    [green]✓ Updated {result['updated']} features[/green]"
                        )

                    except Exception as e:
                        logger.error(f"Failed to apply classification: {e}")
                        stats["errors"] += 1
                        continue

                stats["feature_classes_processed"] += 1
                progress.remove_task(task)

                # Display summary for this feature class
                self._display_fc_summary(fc)

        return stats

    def _display_fc_summary(self, feature_class: str) -> None:
        """Display summary statistics for a processed feature class."""
        fc_path = str(self.gdb_path / feature_class)

        # Count total and classified features
        total = int(arcpy.management.GetCount(fc_path)[0])

        # Count features with symbols using SearchCursor
        classified = 0
        with arcpy.da.SearchCursor(fc_path, [self.symbol_field]) as cursor:
            for row in cursor:
                if row[0] and row[0] != "" and row[0] != "None":
                    classified += 1

        unclassified = total - classified

        # Display table
        table = Table(title=f"Feature Class: {feature_class}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_column("Percentage", style="blue", justify="right")

        table.add_row("Total features", str(total), "100%")
        if total > 0:
            table.add_row("Classified", str(classified), f"{classified / total * 100:.1f}%")
            table.add_row("Unclassified", str(unclassified), f"{unclassified / total * 100:.1f}%")

        console.print(table)


def apply_config_to_filegdb(
        gdb_path: Path,
        config_path: Path,
        feature_classes: Optional[List[str]] = None,
        styles_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convenience function to apply classification config to FileGDB.

    Args:
        gdb_path: Path to FileGDB
        config_path: Path to YAML configuration file
        feature_classes: Specific feature classes to process (None = all)
        styles_dir: Base directory for style files

    Returns:
        Processing statistics dictionary
    """
    # Load configuration
    config = ArcPyClassificationConfig(config_path, styles_dir)

    # Create updater
    updater = ArcPySymbolUpdater(
        gdb_path,
        symbol_field=config.symbol_field,
        label_field=config.label_field,
        treat_zero_as_null=config.treat_zero_as_null,
        overwrite=config.overwrite,
    )

    # Apply classifications
    return updater.apply_batch_from_config(config, feature_classes)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python arcpy_symbol_updater.py <gdb_path> <config_path> [feature_class]")
        sys.exit(1)

    gdb_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])
    feature_classes = [sys.argv[3]] if len(sys.argv) > 3 else None

    stats = apply_config_to_filegdb(gdb_path, config_path, feature_classes)
    console.print(stats)

    console.print("\n[bold green]Processing Complete![/bold green]")
    console.print(f"Feature classes: {stats['feature_classes_processed']}")
    console.print(f"Features updated: {stats['features_updated']}")
    console.print(f"Errors: {stats['errors']}")