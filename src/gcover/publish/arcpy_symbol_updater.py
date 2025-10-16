#!/usr/bin/env python
"""
ArcPy-based in-place SYMBOL field updater for FileGDB
Uses native arcpy cursors to avoid potential FileGDB corruption from geopandas/pyogrio

This module provides a drop-in replacement for the geopandas-based classification
logic, but operates directly on FileGDB using arcpy.da cursors.
"""

import json
import re
from dataclasses import dataclass, field
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
    """Configuration for a layer with multiple classifications."""
    gpkg_layer: Optional[str] = None  # Used for GPKG files
    feature_class: Optional[str] = None  # Used for FileGDB
    classifications: List[ClassificationConfig] = field(default_factory=list)
    field_types: Optional[Dict[str, str]] = None
    layer_type: Optional[str] = None

    def get_target_name(self, format_type: str = "filegdb") -> str:
        """
        Get the appropriate target name based on format.

        Args:
            format_type: 'gpkg' or 'filegdb'

        Returns:
            Layer/feature class name
        """
        if format_type.lower() == "filegdb":
            return self.feature_class or self.gpkg_layer
        else:
            return self.gpkg_layer or self.feature_class


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
        gpkg_layer = layer_dict.get("gpkg_layer")
        feature_class = layer_dict.get("feature_class")
        layer_type = layer_dict.get("layer_type")

        # Validate that at least one is provided
        if not gpkg_layer and not feature_class:
            raise ValueError("Layer config must have either 'gpkg_layer' or 'feature_class'")

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
            gpkg_layer=gpkg_layer,
            feature_class=feature_class,
            classifications=classifications,
            field_types=field_types,
            layer_type=layer_type,
        )

    def get_layer_config(self, target_name: str, format_type: str = "filegdb") -> Optional[LayerConfig]:
        """
        Get configuration for a specific layer.

        Args:
            target_name: Layer/feature class name to find
            format_type: 'gpkg' or 'filegdb' - determines which field to check

        Returns:
            LayerConfig if found, None otherwise
        """
        for layer in self.layers:
            layer_target = layer.get_target_name(format_type)
            if layer_target == target_name:
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
            classification_name: Optional[str] = None,
            field_types: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[List[str]]]:
        """
        Load classification rules from ESRI .lyrx file.

        Args:
            style_file: Path to .lyrx style file
            classification_name: Specific classification to extract
            field_types: Optional field type mapping from config (e.g., {'KIND': 'Int64'})

        Returns:
            Tuple of (rules list, field_names list)
        """
        # Import your existing extractor
        try:
            from gcover.publish.esri_classification_extractor import extract_lyrx

            classifications = extract_lyrx(style_file, display=False)

            if not classifications:
                logger.warning(f"No classifications found in {style_file.name}")
                return [], None

            # Find the right classification
            classification = None
            if classification_name:
                for c in classifications:
                    if c.layer_name == classification_name:
                        classification = c
                        break
                if not classification:
                    logger.warning(f"Classification '{classification_name}' not found in {style_file.name}")
                    logger.info(f"Available classifications: {[c.layer_name for c in classifications]}")
                    return [], None
            elif len(classifications) == 1:
                classification = classifications[0]
            else:
                logger.error(
                    f"Multiple classifications found in {style_file.name}, "
                    f"specify classification_name. Available: {[c.layer_name for c in classifications]}"
                )
                return [], None

            # Extract field names
            field_names = [f.name for f in classification.fields] if classification.fields else None

            # Convert to rules (passing field_types)
            rules = self._convert_classification_to_rules(classification, field_types)

            logger.info(f"Loaded {len(rules)} rules from {classification.layer_name}")
            if field_names:
                logger.debug(f"  Fields: {field_names}")
            if field_types:
                logger.debug(f"  Field types: {field_types}")

            return rules, field_names

        except ImportError:
            logger.error("Cannot import esri_classification_extractor")
            return [], None
        except Exception as e:
            logger.error(f"Error loading classification: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return [], None

    def _convert_classification_to_rules(self, classification, field_types: Optional[Dict[str, str]] = None) -> List[
        Dict[str, Any]]:
        """
        Convert LayerClassification object to list of rules.

        Args:
            classification: LayerClassification object from extract_lyrx
            field_types: Optional field type mapping from config (e.g., {'KIND': 'Int64'})

        Returns:
            List of rule dictionaries with 'condition', 'symbol', 'label', 'field_values'
        """
        rules = []

        if not hasattr(classification, 'classes'):
            logger.warning("Classification has no classes attribute")
            return rules

        logger.debug(f"Converting {len(classification.classes)} classes to rules")

        # Get field names from classification
        field_names = [f.name for f in classification.fields] if classification.fields else []
        logger.debug(f"Classification fields: {field_names}")

        # Process each ClassificationClass
        for class_obj in classification.classes:
            rule = self._extract_rule_from_classification_class(class_obj, field_names, field_types)
            if rule:
                rules.append(rule)

        logger.info(f"Extracted {len(rules)} rules from classification")
        return rules

    def _extract_rule_from_classification_class(
            self,
            class_obj,
            field_names: List[str],
            field_types: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract rule dictionary from a ClassificationClass object.

        Args:
            class_obj: ClassificationClass with label, field_values, symbol_info
            field_names: List of field names used in the classification
            field_types: Optional field type mapping (e.g., {'KIND': 'Int64'})

        Returns:
            Rule dictionary or None
        """
        rule = {
            'label': class_obj.label,
            'visible': class_obj.visible,
            'field_values': class_obj.field_values,
        }

        # Extract symbol identifier
        if class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Build symbol identifier from various attributes
            symbol_parts = []

            # For character markers (font-based symbols)
            if symbol_info.font_family and symbol_info.character_index is not None:
                symbol_parts.append(f"{symbol_info.font_family}_{symbol_info.character_index}")

            # Add color if available
            if symbol_info.color:
                symbol_parts.append(symbol_info.color.to_hex())

            # Add size/width
            if symbol_info.size:
                symbol_parts.append(f"size{symbol_info.size}")
            elif symbol_info.width:
                symbol_parts.append(f"width{symbol_info.width}")

            # Combine or use a simple identifier
            if symbol_parts:
                rule['symbol'] = "_".join(str(p) for p in symbol_parts)
            else:
                # Fallback: use label as symbol
                rule['symbol'] = class_obj.label.replace(" ", "_").replace("/", "_")
        else:
            # No symbol info - use label as identifier
            rule['symbol'] = class_obj.label.replace(" ", "_").replace("/", "_")

        # Build SQL condition from field_values
        # field_values is List[List[str]] - each inner list is a combination of field values
        if class_obj.field_values and field_names:
            conditions = []

            for value_combination in class_obj.field_values:
                if len(value_combination) != len(field_names):
                    logger.warning(
                        f"Value combination length ({len(value_combination)}) doesn't match "
                        f"field count ({len(field_names)})"
                    )
                    continue

                # Build condition for this combination
                combo_conditions = []
                for field_name, value in zip(field_names, value_combination):
                    # Determine if field is numeric based on field_types
                    is_numeric = False
                    if field_types and field_name in field_types:
                        field_type = field_types[field_name].lower()
                        is_numeric = any(t in field_type for t in ['int', 'float', 'double', 'numeric', 'number'])

                    # Format value for SQL
                    if value is None:
                        combo_conditions.append(f"{field_name} IS NULL")
                    elif isinstance(value, str):
                        # Check if it's a NULL representation
                        # ESRI uses various representations: <Null>, NULL, None, empty string
                        if value.upper() in ('NULL', 'NONE', '') or value in ('<Null>', '<null>', '<NULL>'):
                            combo_conditions.append(f"{field_name} IS NULL")
                        elif is_numeric:
                            # Numeric field - don't quote the value
                            try:
                                # Try to convert to number to validate
                                if '.' in value:
                                    float(value)
                                else:
                                    int(value)
                                combo_conditions.append(f"{field_name} = {value}")
                            except ValueError:
                                logger.warning(
                                    f"Expected numeric value for {field_name} but got '{value}', treating as string")
                                escaped_value = value.replace("'", "''")
                                combo_conditions.append(f"{field_name} = '{escaped_value}'")
                        else:
                            # String field - quote the value
                            escaped_value = value.replace("'", "''")
                            combo_conditions.append(f"{field_name} = '{escaped_value}'")
                    else:
                        # Already numeric type (int, float)
                        combo_conditions.append(f"{field_name} = {value}")

                # Combine field conditions with AND
                if combo_conditions:
                    conditions.append("(" + " AND ".join(combo_conditions) + ")")

            # Combine all value combinations with OR
            if conditions:
                if len(conditions) == 1:
                    rule['condition'] = conditions[0]
                else:
                    rule['condition'] = "(" + " OR ".join(conditions) + ")"

        return rule if (rule.get('symbol') or rule.get('label')) else None

    def build_sql_where_clause(
            self,
            rule: Dict[str, Any],
            field_mapping: Optional[Dict[str, str]] = None,
            additional_filter: Optional[str] = None,
            classification_fields: Optional[List[str]] = None,
    ) -> str:
        """
        Build SQL WHERE clause from classification rule.

        Args:
            rule: Rule dictionary with condition/field_values
            field_mapping: Map config field names to actual field names
            additional_filter: Additional SQL filter to combine
            classification_fields: Fields used by the classification (from LayerClassification)

        Returns:
            SQL WHERE clause string
        """
        clauses = []

        # Add rule condition if it exists (already built from field_values)
        if rule.get('condition'):
            condition = rule['condition']

            # Apply field mapping
            if field_mapping:
                for config_field, actual_field in field_mapping.items():
                    # Replace both $field$ syntax and plain field names
                    condition = condition.replace(f"${config_field}$", actual_field)
                    # Use word boundaries to avoid partial replacements
                    import re
                    condition = re.sub(rf'\b{re.escape(config_field)}\b', actual_field, condition)

            clauses.append(condition)

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
            field_types: Optional[Dict[str, str]] = None,
            progress: Optional[Progress] = None,
            task_id: Optional[Any] = None,
    ) -> Dict[str, int]:
        """
        Apply a single classification to a feature class using arcpy cursors.

        Args:
            feature_class: Feature class path (can include dataset)
            class_config: Classification configuration
            field_types: Optional field type mapping from layer config
            progress: Optional Rich progress bar
            task_id: Optional task ID for progress updates

        Returns:
            Dictionary with statistics (updated_count, skipped_count, etc.)
        """
        fc_path = str(self.gdb_path / feature_class)

        # Load classification rules (pass field_types for proper SQL generation)
        rules, field_names = self.load_classification_rules(
            class_config.style_file,
            class_config.classification_name,
            field_types
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

        # Add classification fields if not already included
        if field_names:
            for field in field_names:
                if field not in read_fields:
                    read_fields.append(field)

        # Make sure OID is included for debugging
        read_fields.insert(0, "OID@")

        # Remove duplicates while preserving order
        read_fields = list(dict.fromkeys(read_fields))

        logger.info(f"Processing {feature_class} with {len(rules)} rules")
        logger.debug(f"Reading fields: {read_fields}")
        logger.debug(f"Classification fields: {field_names}")

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
                    class_config.filter,
                    field_names
                )

                # Determine symbol value
                symbol_value = rule.get('symbol')
                if class_config.symbol_prefix and symbol_value:
                    symbol_value = f"{class_config.symbol_prefix}_{symbol_value}"

                label_value = rule.get('label')

                logger.debug(f"Rule {rule_idx + 1}/{len(rules)}: {label_value}")
                logger.debug(f"  WHERE: {where_clause}")
                logger.debug(f"  Symbol: {symbol_value}")

                # Use UpdateCursor to update matching features
                try:
                    with arcpy.da.UpdateCursor(
                            fc_path,
                            read_fields,
                            where_clause=where_clause
                    ) as cursor:
                        rule_updated = 0
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
                                rule_updated += 1
                            else:
                                stats["skipped"] += 1

                            # Update progress
                            if progress and task_id:
                                progress.update(task_id, advance=1)

                    if rule_updated > 0:
                        logger.debug(f"  -> Updated {rule_updated} features")

                except Exception as e:
                    logger.warning(f"Error processing rule {rule_idx + 1} ({label_value}): {e}")
                    logger.debug(f"  WHERE clause: {where_clause}")
                    stats["errors"] += 1
                    continue

            # Commit changes
            edit.stopOperation()
            edit.stopEditing(True)

            logger.info(
                f"✓ {feature_class}: Updated {stats['updated']} features, skipped {stats['skipped']}, errors {stats['errors']}")

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
            fcs_to_process = []
            for layer in config.layers:
                fc_name = layer.get_target_name("filegdb")
                if fc_name and fc_name in available_fcs:
                    fcs_to_process.append(fc_name)

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
                layer_config = config.get_layer_config(fc, format_type="filegdb")
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
                            field_types=layer_config.field_types,  # Pass field types
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

    console.print("\n[bold green]Processing Complete![/bold green]")
    console.print(f"Feature classes: {stats['feature_classes_processed']}")
    console.print(f"Features updated: {stats['features_updated']}")
    console.print(f"Errors: {stats['errors']}")