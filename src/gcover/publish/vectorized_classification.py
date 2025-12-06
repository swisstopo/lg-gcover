#!/usr/bin/env python
"""
Vectorized Classification Applicator for GeoCover

Replaces O(n×m) row-by-row matching with O(n) pandas merge operations.
Provides two-level QA tracking:
  - Per-classification: filter vs rule coverage (performance info)
  - Final layer: unclassified features after all rules (data quality issue)

Author: Refactored for lg-gcover project
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table

# Assuming these exist in your codebase
from gcover.publish.esri_classification_extractor import (
    ClassificationClass,
    LayerClassification,
)
from gcover.publish.utils import translate_esri_to_pandas

console = Console()

# Sentinel value for NULL handling in joins
NULL_SENTINEL = '__NULL__'


@dataclass
class ClassificationStats:
    """Statistics for a single classification application."""
    classification_name: str
    filter_expression: Optional[str]
    total_in_layer: int
    filtered_count: int
    matched_count: int
    preserved_count: int  # Already had a value
    newly_classified: int
    unmatched_in_filter: int  # Filtered but no rule matched
    
    @property
    def filter_coverage_pct(self) -> float:
        """Percentage of filtered features that matched a rule."""
        eligible = self.filtered_count - self.preserved_count
        if eligible <= 0:
            return 100.0
        return (self.newly_classified / eligible) * 100
    
    def to_dict(self) -> Dict:
        return {
            'classification': self.classification_name,
            'filter': self.filter_expression,
            'total': self.total_in_layer,
            'filtered': self.filtered_count,
            'matched': self.matched_count,
            'preserved': self.preserved_count,
            'newly_classified': self.newly_classified,
            'unmatched_in_filter': self.unmatched_in_filter,
            'filter_coverage_pct': self.filter_coverage_pct,
        }


@dataclass 
class LayerClassificationReport:
    """Final QA report after all classifications applied to a layer."""
    layer_name: str
    total_features: int
    classified_count: int
    unclassified_count: int
    classification_stats: List[ClassificationStats] = field(default_factory=list)
    unclassified_patterns: Dict[Tuple, int] = field(default_factory=dict)
    unclassified_sample_indices: List[Any] = field(default_factory=list)
    
    @property
    def coverage_pct(self) -> float:
        if self.total_features == 0:
            return 100.0
        return (self.classified_count / self.total_features) * 100
    
    @property
    def is_complete(self) -> bool:
        return self.unclassified_count == 0
    
    def display(self):
        """Display the report using Rich."""
        # Summary
        status = "[green]✓ COMPLETE[/green]" if self.is_complete else f"[red]✗ INCOMPLETE ({self.unclassified_count} missing)[/red]"
        console.print(f"\n[bold]Layer: {self.layer_name}[/bold] - {status}")
        console.print(f"  Total: {self.total_features:,} | Classified: {self.classified_count:,} | Coverage: {self.coverage_pct:.1f}%")
        
        # Per-classification breakdown
        if self.classification_stats:
            table = Table(title="Classification Breakdown", show_header=True)
            table.add_column("Classification", style="cyan")
            table.add_column("Filter", style="dim", max_width=30)
            table.add_column("Filtered", justify="right")
            table.add_column("Matched", justify="right")
            table.add_column("New", justify="right", style="green")
            table.add_column("Coverage", justify="right")
            
            for stat in self.classification_stats:
                coverage_style = "green" if stat.filter_coverage_pct >= 95 else "yellow" if stat.filter_coverage_pct >= 80 else "red"
                table.add_row(
                    stat.classification_name,
                    (stat.filter_expression or "—")[:30],
                    str(stat.filtered_count),
                    str(stat.matched_count),
                    str(stat.newly_classified),
                    f"[{coverage_style}]{stat.filter_coverage_pct:.1f}%[/{coverage_style}]",
                )
            console.print(table)
        
        # Unclassified patterns (the important QA info)
        if self.unclassified_patterns:
            console.print(f"\n[red]⚠️  Unclassified feature patterns (top 15):[/red]")
            sorted_patterns = sorted(self.unclassified_patterns.items(), key=lambda x: -x[1])
            for pattern, count in sorted_patterns[:15]:
                console.print(f"  {pattern}: {count} features")
            
            if len(sorted_patterns) > 15:
                remaining = sum(c for _, c in sorted_patterns[15:])
                console.print(f"  ... and {len(sorted_patterns) - 15} more patterns ({remaining} features)")


def build_lookup_table(
    classification: LayerClassification,
    symbol_prefix: str,
    field_names: List[str],
) -> pd.DataFrame:
    """
    Build lookup DataFrame from classification rules.
    
    Args:
        classification: The LayerClassification with rules
        symbol_prefix: Prefix for symbol IDs
        field_names: List of classification field names
        
    Returns:
        DataFrame with columns: [field1, field2, ..., _SYMBOL, _LABEL]
    """
    rows = []
    
    for idx, class_obj in enumerate(classification.classes):
        if not class_obj.visible:
            continue
        
        # Build symbol_id (using identifier if available, else index)
        identifier_value = idx
        if hasattr(class_obj, 'identifier') and class_obj.identifier:
            try:
                identifier_key = class_obj.identifier.to_key()
                identifier_value = identifier_key.split('::')[-1]
            except Exception:
                pass
        
        symbol_id = f"{symbol_prefix}_{identifier_value}"
        
        # Each class can have multiple value combinations (OR logic)
        for field_values in class_obj.field_values:
            if len(field_values) != len(field_names):
                logger.warning(f"Field count mismatch in class '{class_obj.label}': "
                             f"expected {len(field_names)}, got {len(field_values)}")
                continue
            
            row = {
                '_SYMBOL': symbol_id,
                '_LABEL': class_obj.label,
            }
            
            for fname, fval in zip(field_names, field_values):
                # Normalize: <Null> → sentinel, everything else → string
                if fval == '<Null>' or fval is None:
                    row[fname] = NULL_SENTINEL
                else:
                    row[fname] = str(fval).strip()
            
            rows.append(row)
    
    if not rows:
        logger.warning(f"No visible classes found in classification")
        return pd.DataFrame(columns=field_names + ['_SYMBOL', '_LABEL'])
    
    return pd.DataFrame(rows)


def normalize_join_column(
    series: pd.Series,
    treat_zero_as_null: bool = False,
) -> pd.Series:
    """
    Normalize a column for join operations.
    
    - NaN/None → NULL_SENTINEL
    - Optionally 0 → NULL_SENTINEL  
    - Everything else → string
    """
    result = series.copy()
    
    # Convert to string first
    result = result.astype(str)
    
    # Handle NaN (which becomes 'nan' after astype)
    result = result.replace(['nan', 'None', '', '<NA>'], NULL_SENTINEL)
    
    # Handle original NaN values
    result = result.fillna(NULL_SENTINEL)
    
    # Optionally treat 0 as null
    if treat_zero_as_null:
        result = result.replace(['0', '0.0'], NULL_SENTINEL)
    
    # Strip whitespace
    result = result.str.strip()
    
    return result


class VectorizedClassificationApplicator:
    """
    Apply ESRI UniqueValue classification rules using vectorized pandas operations.
    
    This is O(n) instead of O(n×m), providing ~100-1000x speedup for large datasets.
    """
    
    def __init__(
        self,
        classification: LayerClassification,
        symbol_field: str = 'SYMBOL',
        label_field: Optional[str] = 'LABEL',
        symbol_prefix: Optional[str] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        treat_zero_as_null: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the applicator.
        
        Args:
            classification: LayerClassification with rules
            symbol_field: Output field for symbol IDs
            label_field: Output field for labels (None to disable)
            symbol_prefix: Prefix for symbol IDs
            field_mapping: Map GPKG fields → classification fields
            treat_zero_as_null: Treat 0 as NULL for matching
            debug: Enable debug logging
        """
        self.classification = classification
        self.symbol_field = symbol_field
        self.label_field = label_field
        self.symbol_prefix = symbol_prefix or self._sanitize_prefix(
            classification.layer_name or 'symbol'
        )
        self.field_mapping = field_mapping or {}
        self.treat_zero_as_null = treat_zero_as_null
        self.debug = debug
        
        # Extract classification field names
        self.classification_field_names = [f.name for f in classification.fields]
        
        # Build reverse mapping (classification_field → gpkg_field)
        self.reverse_field_mapping = {v: k for k, v in self.field_mapping.items()}
        
        # Build lookup table once
        self.lookup_df = build_lookup_table(
            classification,
            self.symbol_prefix,
            self.classification_field_names,
        )
        
        # Check for duplicate rules
        self._check_duplicate_rules()
        
        logger.info(f"Initialized vectorized applicator for '{classification.layer_name}'")
        logger.info(f"  Fields: {self.classification_field_names}")
        logger.info(f"  Rules: {len(self.lookup_df)} value combinations")
        logger.info(f"  Symbol prefix: {self.symbol_prefix}")
    
    def _sanitize_prefix(self, name: str) -> str:
        """Sanitize name for use as symbol prefix."""
        sanitized = re.sub(r'[^\w]+', '_', name.lower())
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        return sanitized
    
    def _check_duplicate_rules(self):
        """Check for and warn about duplicate/overlapping rules."""
        if self.lookup_df.empty:
            return
        
        duplicates = self.lookup_df[
            self.lookup_df.duplicated(subset=self.classification_field_names, keep=False)
        ]
        
        if len(duplicates) > 0:
            logger.warning(f"Found {len(duplicates)} overlapping rules (same key → different symbols):")
            
            # Group by key to show conflicts
            for key, group in duplicates.groupby(self.classification_field_names):
                if len(group) > 1:
                    labels = group['_LABEL'].tolist()
                    logger.warning(f"  Key {key} matches: {labels}")
            
            # Keep first match (ESRI behavior)
            self.lookup_df = self.lookup_df.drop_duplicates(
                subset=self.classification_field_names, 
                keep='first'
            )
            logger.warning(f"  Keeping first match for each key (ESRI behavior)")
    
    @property
    def gpkg_field_names(self) -> List[str]:
        """Get the field names as they appear in the GPKG (after reverse mapping)."""
        return [
            self.reverse_field_mapping.get(f, f) 
            for f in self.classification_field_names
        ]
    
    def check_required_fields(self, gdf: gpd.GeoDataFrame) -> Tuple[bool, List[str]]:
        """Check if all required fields are present."""
        missing = [f for f in self.gpkg_field_names if f not in gdf.columns]
        return len(missing) == 0, missing
    
    def apply(
        self,
        gdf: gpd.GeoDataFrame,
        additional_filter: Optional[str] = None,
        preserve_existing: bool = True,
    ) -> Tuple[gpd.GeoDataFrame, ClassificationStats]:
        """
        Apply classification using vectorized merge.
        
        Args:
            gdf: Input GeoDataFrame
            additional_filter: Optional filter expression (ESRI or pandas syntax)
            preserve_existing: If True, don't overwrite existing non-NULL values
            
        Returns:
            Tuple of (modified GeoDataFrame, ClassificationStats)
        """
        classification_name = self.classification.layer_name or 'unknown'
        total_features = len(gdf)
        
        # Check required fields
        all_present, missing = self.check_required_fields(gdf)
        if not all_present:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Initialize output fields if needed
        if self.symbol_field not in gdf.columns:
            gdf[self.symbol_field] = pd.Series(dtype='string', index=gdf.index)
        else:
            gdf[self.symbol_field] = gdf[self.symbol_field].astype('string')
        
        if self.label_field:
            if self.label_field not in gdf.columns:
                gdf[self.label_field] = pd.Series(dtype='string', index=gdf.index)
            else:
                gdf[self.label_field] = gdf[self.label_field].astype('string')
        
        # Apply filter if provided
        if additional_filter:
            pandas_filter = translate_esri_to_pandas(additional_filter)
            try:
                mask = gdf.eval(pandas_filter)
                target_indices = gdf.index[mask]
                logger.info(f"Filter '{additional_filter}' → {len(target_indices)}/{total_features} features")
            except Exception as e:
                logger.error(f"Filter evaluation failed: {e}")
                logger.warning(f"Falling back to all features")
                target_indices = gdf.index
        else:
            target_indices = gdf.index
        
        filtered_count = len(target_indices)
        
        # Determine which features to actually classify (respect preserve_existing)
        if preserve_existing:
            existing_null_mask = gdf.loc[target_indices, self.symbol_field].isna() | \
                                 (gdf.loc[target_indices, self.symbol_field] == '') | \
                                 (gdf.loc[target_indices, self.symbol_field] == 'None')
            classify_indices = target_indices[existing_null_mask]
            preserved_count = filtered_count - len(classify_indices)
        else:
            classify_indices = target_indices
            preserved_count = 0
        
        if len(classify_indices) == 0:
            logger.info(f"No features to classify (all preserved)")
            return gdf, ClassificationStats(
                classification_name=classification_name,
                filter_expression=additional_filter,
                total_in_layer=total_features,
                filtered_count=filtered_count,
                matched_count=0,
                preserved_count=preserved_count,
                newly_classified=0,
                unmatched_in_filter=0,
            )
        
        # Prepare join columns with normalization
        join_col_map = {}  # gpkg_field → temp_join_col
        for gpkg_field, class_field in zip(self.gpkg_field_names, self.classification_field_names):
            join_col = f'_join_{class_field}'
            gdf[join_col] = normalize_join_column(
                gdf[gpkg_field], 
                self.treat_zero_as_null
            )
            join_col_map[gpkg_field] = join_col
        
        join_cols = list(join_col_map.values())
        
        # Extract subset for classification
        gdf_subset = gdf.loc[classify_indices, join_cols].copy()
        gdf_subset['_orig_idx'] = gdf_subset.index
        
        # THE CLASSIFICATION - single merge operation!
        merged = gdf_subset.merge(
            self.lookup_df,
            left_on=join_cols,
            right_on=self.classification_field_names,
            how='left',
            indicator='_match_status',
        )
        
        # Calculate statistics
        matched_mask = merged['_match_status'] == 'both'
        matched_count = matched_mask.sum()
        newly_classified = matched_count  # All matches are new (we only process NULLs)
        unmatched_in_filter = len(classify_indices) - matched_count
        
        if self.debug and unmatched_in_filter > 0:
            # Log unmatched patterns within this filter
            unmatched_df = merged.loc[~matched_mask, join_cols]
            patterns = unmatched_df.groupby(join_cols).size().sort_values(ascending=False)
            logger.debug(f"Unmatched patterns in filter (top 5):")
            for pattern, count in list(patterns.head(5).items()):
                logger.debug(f"  {pattern}: {count}")
        
        # Assign results back
        merged_indexed = merged.set_index('_orig_idx')
        
        # Only assign where we got a match
        match_indices = merged_indexed.index[matched_mask.values]
        gdf.loc[match_indices, self.symbol_field] = merged_indexed.loc[match_indices, '_SYMBOL'].values
        if self.label_field:
            gdf.loc[match_indices, self.label_field] = merged_indexed.loc[match_indices, '_LABEL'].values
        
        # Cleanup temporary columns
        gdf.drop(columns=join_cols, inplace=True)
        
        # Build statistics
        stats = ClassificationStats(
            classification_name=classification_name,
            filter_expression=additional_filter,
            total_in_layer=total_features,
            filtered_count=filtered_count,
            matched_count=matched_count,
            preserved_count=preserved_count,
            newly_classified=newly_classified,
            unmatched_in_filter=unmatched_in_filter,
        )
        
        # Log summary
        logger.info(f"Classification '{classification_name}': "
                   f"filtered={filtered_count}, matched={matched_count}, "
                   f"new={newly_classified}, preserved={preserved_count}")
        
        if unmatched_in_filter > 0 and stats.filter_coverage_pct < 95:
            logger.warning(f"  ⚠️  Filter coverage only {stats.filter_coverage_pct:.1f}% "
                          f"({unmatched_in_filter} filtered features have no matching rule)")
        
        return gdf, stats


def analyze_unclassified_features(
    gdf: gpd.GeoDataFrame,
    symbol_field: str,
    analysis_fields: List[str],
    max_patterns: int = 50,
) -> Tuple[Dict[Tuple, int], List[Any]]:
    """
    Analyze unclassified features to identify patterns.
    
    Args:
        gdf: GeoDataFrame with symbol field
        symbol_field: Name of symbol field
        analysis_fields: Fields to group by for pattern analysis
        max_patterns: Maximum patterns to return
        
    Returns:
        Tuple of (pattern_counts dict, sample_indices list)
    """
    # Find unclassified features
    unclassified_mask = gdf[symbol_field].isna() | \
                        (gdf[symbol_field] == '') | \
                        (gdf[symbol_field] == 'None')
    
    unclassified_df = gdf.loc[unclassified_mask]
    
    if len(unclassified_df) == 0:
        return {}, []
    
    # Filter to fields that exist
    existing_fields = [f for f in analysis_fields if f in gdf.columns]
    
    if not existing_fields:
        # No analysis fields available, just return count
        return {('unknown',): len(unclassified_df)}, unclassified_df.index[:10].tolist()
    
    # Group by value combinations
    # Normalize values for grouping
    group_df = unclassified_df[existing_fields].copy()
    for col in existing_fields:
        # Convert to string FIRST (handles Int64, float, etc.), then replace NaN
        group_df[col] = group_df[col].astype(str).replace({'<NA>': '<NULL>', 'nan': '<NULL>', 'None': '<NULL>'})
    
    patterns = group_df.groupby(existing_fields).size().sort_values(ascending=False)
    
    # Convert to dict with tuple keys
    pattern_dict = {}
    for idx, count in patterns.head(max_patterns).items():
        if isinstance(idx, tuple):
            key = idx
        else:
            key = (idx,)
        pattern_dict[key] = count
    
    # Sample indices
    sample_indices = unclassified_df.index[:20].tolist()
    
    return pattern_dict, sample_indices


def apply_batch_classifications_vectorized(
    gdf: gpd.GeoDataFrame,
    classifications: List[Tuple[LayerClassification, Optional[str], Optional[str]]],
    symbol_field: str = 'SYMBOL',
    label_field: Optional[str] = 'LABEL',
    treat_zero_as_null: bool = False,
    analysis_fields: Optional[List[str]] = None,
    layer_name: str = 'unknown',
    debug: bool = False,
) -> Tuple[gpd.GeoDataFrame, LayerClassificationReport]:
    """
    Apply multiple classifications to a layer and generate final QA report.
    
    This is the main entry point for batch classification.
    
    Args:
        gdf: Input GeoDataFrame
        classifications: List of (LayerClassification, filter_expr, symbol_prefix) tuples
        symbol_field: Output symbol field name
        label_field: Output label field name (None to disable)
        treat_zero_as_null: Treat 0 as NULL for matching
        analysis_fields: Fields to analyze for unclassified patterns
        layer_name: Name for reporting
        debug: Enable debug output
        
    Returns:
        Tuple of (classified GeoDataFrame, LayerClassificationReport)
    """
    total_features = len(gdf)
    all_stats: List[ClassificationStats] = []
    
    console.print(f"\n[bold blue]🎨 Classifying layer: {layer_name}[/bold blue]")
    console.print(f"  Features: {total_features:,}")
    console.print(f"  Classifications to apply: {len(classifications)}")
    
    # Apply each classification in sequence
    for i, (classification, filter_expr, prefix) in enumerate(classifications, 1):
        class_name = classification.layer_name or f'classification_{i}'
        console.print(f"\n  [{i}/{len(classifications)}] {class_name}")
        
        applicator = VectorizedClassificationApplicator(
            classification=classification,
            symbol_field=symbol_field,
            label_field=label_field,
            symbol_prefix=prefix,
            treat_zero_as_null=treat_zero_as_null,
            debug=debug,
        )
        
        gdf, stats = applicator.apply(
            gdf,
            additional_filter=filter_expr,
            preserve_existing=True,  # Don't overwrite previous classifications
        )
        
        all_stats.append(stats)
        
        # Progress indicator
        current_classified = gdf[symbol_field].notna().sum()
        current_pct = (current_classified / total_features) * 100
        console.print(f"      → Cumulative: {current_classified:,}/{total_features:,} ({current_pct:.1f}%)")
    
    # Final analysis
    final_classified = gdf[symbol_field].notna().sum()
    final_unclassified = total_features - final_classified
    
    # Analyze unclassified patterns for QA
    if analysis_fields is None:
        # Collect all fields used in classifications
        analysis_fields = set()
        for classification, _, _ in classifications:
            analysis_fields.update(f.name for f in classification.fields)
        analysis_fields = list(analysis_fields)
    
    unclassified_patterns, sample_indices = analyze_unclassified_features(
        gdf, symbol_field, analysis_fields
    )
    
    # Build report
    report = LayerClassificationReport(
        layer_name=layer_name,
        total_features=total_features,
        classified_count=final_classified,
        unclassified_count=final_unclassified,
        classification_stats=all_stats,
        unclassified_patterns=unclassified_patterns,
        unclassified_sample_indices=sample_indices,
    )
    
    # Display report
    report.display()
    
    # Log final status
    if report.is_complete:
        logger.success(f"Layer '{layer_name}': 100% classified ({total_features:,} features)")
    else:
        logger.error(f"Layer '{layer_name}': {final_unclassified:,} features UNCLASSIFIED "
                    f"({report.coverage_pct:.1f}% coverage)")
    
    return gdf, report


# =============================================================================
# INTEGRATION WITH EXISTING BATCH SYSTEM
# =============================================================================

def apply_batch_from_config_vectorized(
    gpkg_path: Path,
    config,  # BatchClassificationConfig
    layer_name: Optional[str] = None,
    output_path: Optional[Path] = None,
    debug: bool = False,
    bbox: Optional[tuple] = None,
    continue_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Drop-in replacement for apply_batch_from_config using vectorized classification.
    
    This integrates with your existing BatchClassificationConfig structure.
    """
    import fiona
    from gcover.publish.esri_classification_extractor import extract_lyrx_complete
    
    # Determine output path
    if output_path is None:
        output_path = gpkg_path.parent / f"{gpkg_path.stem}_classified.gpkg"
    
    # Get available layers
    available_layers = fiona.listlayers(str(gpkg_path))
    
    # Determine which layers to process
    if layer_name:
        if layer_name not in available_layers:
            console.print(f"[red]Layer '{layer_name}' not found in GPKG[/red]")
            return {}
        layers_to_process = [layer_name]
    else:
        layers_to_process = [
            layer.gpkg_layer
            for layer in config.layers
            if layer.gpkg_layer in available_layers
        ]
    
    if not layers_to_process:
        logger.warning("No layers to process!")
        return {}
    
    # Global statistics
    global_stats = {
        "layers_processed": 0,
        "layers_complete": 0,
        "classifications_applied": 0,
        "features_classified": 0,
        "features_total": 0,
        "features_unclassified": 0,
        "layer_reports": {},
    }
    
    # Process each layer
    for layer in layers_to_process:
        layer_config = config.get_layer_config(layer)
        if not layer_config:
            logger.warning(f"No configuration for layer '{layer}', skipping")
            continue
        
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold cyan]Processing layer: {layer}[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        
        # Load layer
        kwargs = {"layer": layer}
        if bbox:
            kwargs["bbox"] = bbox
        
        gdf = gpd.read_file(gpkg_path, **kwargs)
        global_stats["features_total"] += len(gdf)
        
        # Cast field types if specified
        if layer_config.field_types:
            console.print(f"\n[cyan]Casting field types...[/cyan]")
            for field, dtype in layer_config.field_types.items():
                if field in gdf.columns:
                    try:
                        if dtype.lower().startswith('int'):
                            gdf[field] = pd.to_numeric(gdf[field], errors='coerce')
                            gdf[field] = gdf[field].astype(dtype.capitalize())
                        else:
                            gdf[field] = gdf[field].astype(dtype)
                        console.print(f"  [green]✓ {field} → {dtype}[/green]")
                    except Exception as e:
                        console.print(f"  [red]✗ {field}: {e}[/red]")
        
        # Build list of (classification, filter, prefix) tuples
        classifications_to_apply = []
        
        for class_config in layer_config.classifications:
            try:
                # Load classification from style file
                classifications = extract_lyrx_complete(class_config.style_file, display=False)
                
                # Find the right classification
                classification = None
                if class_config.classification_name:
                    for c in classifications:
                        if c.layer_name == class_config.classification_name:
                            classification = c
                            break
                    if not classification:
                        logger.error(f"Classification '{class_config.classification_name}' "
                                   f"not found in {class_config.style_file.name}")
                        if not continue_on_error:
                            raise ValueError(f"Classification not found")
                        continue
                elif len(classifications) == 1:
                    classification = classifications[0]
                else:
                    logger.error(f"Multiple classifications in {class_config.style_file.name}, "
                               f"specify classification_name")
                    if not continue_on_error:
                        raise ValueError("Ambiguous classification")
                    continue
                
                classifications_to_apply.append((
                    classification,
                    class_config.filter,
                    class_config.symbol_prefix,
                ))
                
            except FileNotFoundError:
                logger.error(f"Style file not found: {class_config.style_file}")
                if not continue_on_error:
                    raise
            except Exception as e:
                logger.error(f"Error loading classification: {e}")
                if not continue_on_error:
                    raise
        
        if not classifications_to_apply:
            logger.warning(f"No valid classifications for layer '{layer}'")
            continue
        
        # Collect analysis fields from all classifications
        analysis_fields = set()
        for classification, _, _ in classifications_to_apply:
            analysis_fields.update(f.name for f in classification.fields)
        
        # Apply all classifications
        gdf, report = apply_batch_classifications_vectorized(
            gdf=gdf,
            classifications=classifications_to_apply,
            symbol_field=config.symbol_field,
            label_field=config.label_field,
            treat_zero_as_null=config.treat_zero_as_null,
            analysis_fields=list(analysis_fields),
            layer_name=layer,
            debug=debug,
        )
        
        # Update global stats
        global_stats["layers_processed"] += 1
        global_stats["classifications_applied"] += len(classifications_to_apply)
        global_stats["features_classified"] += report.classified_count
        global_stats["features_unclassified"] += report.unclassified_count
        global_stats["layer_reports"][layer] = report
        
        if report.is_complete:
            global_stats["layers_complete"] += 1
        
        # Save layer to output
        console.print(f"\n[cyan]Saving layer to {output_path}...[/cyan]")
        
        if output_path.exists() and layer != layers_to_process[0]:
            gdf.to_file(output_path, layer=layer, driver="GPKG", mode="a")
        else:
            gdf.to_file(output_path, layer=layer, driver="GPKG")
    
    # Final summary
    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]BATCH CLASSIFICATION COMPLETE[/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]")
    
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Layers processed", str(global_stats["layers_processed"]))
    summary_table.add_row("Layers 100% complete", str(global_stats["layers_complete"]))
    summary_table.add_row("Classifications applied", str(global_stats["classifications_applied"]))
    summary_table.add_row("Total features", f"{global_stats['features_total']:,}")
    summary_table.add_row("Features classified", f"{global_stats['features_classified']:,}")
    summary_table.add_row("Features unclassified", f"{global_stats['features_unclassified']:,}")
    
    if global_stats["features_total"] > 0:
        coverage = (global_stats["features_classified"] / global_stats["features_total"]) * 100
        status = "[green]✓[/green]" if coverage == 100 else "[red]✗[/red]"
        summary_table.add_row("Overall coverage", f"{status} {coverage:.1f}%")
    
    console.print(summary_table)
    console.print(f"\n[dim]Output: {output_path}[/dim]")
    
    # Log any incomplete layers
    incomplete_layers = [
        name for name, report in global_stats["layer_reports"].items()
        if not report.is_complete
    ]
    if incomplete_layers:
        console.print(f"\n[red]⚠️  Incomplete layers: {', '.join(incomplete_layers)}[/red]")
        console.print("[red]   Review the unclassified patterns above to identify missing rules.[/red]")
    
    return global_stats
