# src/gcover/publish/merge_sources_arcpy.py
"""
ArcPy-based merger for FileGDB sources.

Uses arcpy for optimal FileGDB handling when available, including:
- Proper handling of relationship classes
- Feature dataset groups
- Topology preservation
- Better performance for large datasets

Falls back to geopandas implementation when arcpy is not available.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import warnings

from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from gcover.arcpy_compat import HAS_ARCPY

if HAS_ARCPY:
    import arcpy
    
from .merge_sources import (
    MergeConfig,
    MergeStats,
    SourceConfig,
    SourceType,
    GDBMerger as GDBMergerGeopandas,
)

console = Console()


class GDBMergerArcPy:
    """
    ArcPy-based merger for FileGDB sources.
    
    Provides better performance and compatibility with ESRI FileGDB features
    compared to the geopandas-based merger.
    """
    
    def __init__(self, config: MergeConfig, verbose: bool = False):
        if not HAS_ARCPY:
            raise ImportError("arcpy is required for GDBMergerArcPy")
            
        self.config = config
        self.verbose = verbose
        self.sources: Dict[str, SourceConfig] = {}
        self.stats = MergeStats()
        self._clip_fc_cache: Dict[str, str] = {}  # Cache for clip geometries
        
        self._log_debug("Initializing GDBMergerArcPy")
        self._setup_sources()
        self._setup_workspace()
        
    def _log_debug(self, message: str) -> None:
        """Log debug message if verbose mode is enabled."""
        if self.verbose:
            console.print(f"  [dim]DEBUG: {message}[/dim]")
        logger.debug(message)
        
    def _log_info(self, message: str) -> None:
        """Log info message."""
        if self.verbose:
            console.print(f"  [cyan]INFO: {message}[/cyan]")
        logger.info(message)
        
    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        console.print(f"  [yellow]⚠ {message}[/yellow]")
        logger.warning(message)
        
    def _log_error(self, message: str) -> None:
        """Log error message."""
        console.print(f"  [red]✗ {message}[/red]")
        logger.error(message)
        
    def _setup_sources(self) -> None:
        """Configure available source databases."""
        
        self._log_debug("Setting up sources...")
        
        if self.config.rc1_path and self.config.rc1_path.exists():
            self.sources["RC1"] = SourceConfig(
                name="RC1",
                path=self.config.rc1_path,
                source_type=SourceType.RC1
            )
            self._log_debug(f"  Registered RC1: {self.config.rc1_path}")
            
        if self.config.rc2_path and self.config.rc2_path.exists():
            self.sources["RC2"] = SourceConfig(
                name="RC2",
                path=self.config.rc2_path,
                source_type=SourceType.RC2
            )
            self._log_debug(f"  Registered RC2: {self.config.rc2_path}")
            
        if self.config.custom_sources_dir and self.config.custom_sources_dir.exists():
            for gdb_path in self.config.custom_sources_dir.glob("*.gdb"):
                source_name = gdb_path.name
                self.sources[source_name] = SourceConfig(
                    name=source_name,
                    path=gdb_path,
                    source_type=SourceType.CUSTOM
                )
                self._log_debug(f"  Registered custom source: {gdb_path}")
                
        self._log_info(f"Configured {len(self.sources)} source(s)")
                
    def _setup_workspace(self) -> None:
        """Set up arcpy workspace and environment."""
        
        self._log_debug("Setting up arcpy environment...")
        
        arcpy.env.overwriteOutput = True
        arcpy.env.preserveGlobalIds = True
        
        # Ensure scratch workspace exists
        if not arcpy.env.scratchGDB:
            arcpy.env.scratchWorkspace = str(Path.home() / "arcpy_scratch")
            
        self._log_debug(f"  scratchGDB: {arcpy.env.scratchGDB}")
        
        # Use first available source as template for workspace
        template_path = None
        for source in self.sources.values():
            if source.path.exists():
                template_path = source.path
                break
                
        if template_path:
            arcpy.env.workspace = str(template_path)
            self._log_debug(f"  workspace: {template_path}")
            
    def _resolve_source_path(self, source_name: str) -> Optional[Path]:
        """Resolve source name to actual GDB path."""
        
        if source_name in self.sources:
            return self.sources[source_name].path
            
        source_with_ext = f"{source_name}.gdb" if not source_name.endswith(".gdb") else source_name
        if source_with_ext in self.sources:
            return self.sources[source_with_ext].path
            
        if self.config.custom_sources_dir:
            potential_path = self.config.custom_sources_dir / source_with_ext
            if potential_path.exists():
                return potential_path
                
        return None
    
    def _create_output_gdb(self) -> None:
        """Create the output FileGDB."""
        
        output_path = self.config.output_path
        
        self._log_info(f"Creating output GDB: {output_path}")
        
        if output_path.exists():
            self._log_warning(f"Deleting existing output: {output_path}")
            arcpy.management.Delete(str(output_path))
            
        # Create new FileGDB
        start_time = time.time()
        arcpy.management.CreateFileGDB(
            out_folder_path=str(output_path.parent),
            out_name=output_path.name
        )
        elapsed = time.time() - start_time
        
        self._log_debug(f"  Created in {elapsed:.2f}s")
        
    def _create_feature_dataset(self, name: str = "GC_ROCK_BODIES") -> str:
        """Create feature dataset in output GDB."""
        
        import os
        
        output_gdb = str(self.config.output_path)
        feature_dataset_path = os.path.join(output_gdb, name)
        
        self._log_info(f"Creating feature dataset: {name}")
        self._log_debug(f"  Full path: {feature_dataset_path}")
        
        if not arcpy.Exists(feature_dataset_path):
            # Get spatial reference from first source
            sr = None
            for source in self.sources.values():
                for layer in self.config.spatial_layers:
                    layer_name = layer.split("/")[-1]
                    source_fc = os.path.join(str(source.path), "GC_ROCK_BODIES", layer_name)
                    if arcpy.Exists(source_fc):
                        sr = arcpy.Describe(source_fc).spatialReference
                        self._log_debug(f"  Using spatial reference from: {source_fc}")
                        self._log_debug(f"  SR: {sr.name} (WKID: {sr.factoryCode})")
                        break
                if sr:
                    break
                    
            if sr:
                arcpy.management.CreateFeatureDataset(
                    out_dataset_path=output_gdb,
                    out_name=name,
                    spatial_reference=sr
                )
                self._log_debug(f"  Feature dataset created")
                
                # Verify it was created
                if arcpy.Exists(feature_dataset_path):
                    self._log_debug(f"  Verified: feature dataset exists")
                else:
                    self._log_error(f"  Feature dataset creation failed!")
            else:
                self._log_warning("Could not determine spatial reference for feature dataset")
        else:
            self._log_debug(f"  Feature dataset already exists")
                
        return feature_dataset_path
    
    def _load_mapsheets_arcpy(self) -> Dict[str, List[int]]:
        """Load mapsheets grouped by source using arcpy."""
        
        self._log_info(f"Loading mapsheets from: {self.config.admin_zones_path}")
        self._log_debug(f"  Layer: {self.config.mapsheets_layer}")
        self._log_debug(f"  Source column: {self.config.source_column}")
        
        # Build the full path to the layer in the GPKG
        gpkg_layer = f"{self.config.admin_zones_path}/{self.config.mapsheets_layer}"
        
        if not arcpy.Exists(gpkg_layer):
            self._log_error(f"Layer not found: {gpkg_layer}")
            raise ValueError(f"Mapsheets layer not found: {gpkg_layer}")
        
        # Build where clause for mapsheet filter
        where_clause = None
        if self.config.mapsheet_numbers:
            numbers_str = ",".join(str(n) for n in self.config.mapsheet_numbers)
            where_clause = f"{self.config.mapsheet_nbr_column} IN ({numbers_str})"
            self._log_debug(f"  Filter: {where_clause}")
        
        # Read mapsheets and group by source
        grouped = {}
        total_count = 0
        
        fields = [self.config.mapsheet_nbr_column, self.config.source_column]
        
        with arcpy.da.SearchCursor(gpkg_layer, fields, where_clause) as cursor:
            for row in cursor:
                mapsheet_nbr = row[0]
                source_name = row[1]
                
                if source_name not in grouped:
                    grouped[source_name] = []
                grouped[source_name].append(mapsheet_nbr)
                total_count += 1
        
        self._log_info(f"Loaded {total_count} mapsheets across {len(grouped)} sources")
        
        for source_name, mapsheets in grouped.items():
            self._log_debug(f"  {source_name}: {len(mapsheets)} mapsheets")
            
        self.stats.mapsheets_processed = total_count
        return grouped
    
    def _create_clip_geometry_arcpy(self, mapsheet_numbers: List[int], cache_key: str) -> str:
        """
        Create clip geometry for mapsheets using native arcpy functions.
        
        Args:
            mapsheet_numbers: List of mapsheet numbers to include
            cache_key: Key for caching the result
            
        Returns:
            Path to the dissolved clip feature class
        """
        # Sanitize cache key for use in feature class names
        # Remove/replace invalid characters: . / \ spaces etc.
        safe_key = cache_key.replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
        
        # Check cache first (use original key for lookup)
        if cache_key in self._clip_fc_cache:
            cached_path = self._clip_fc_cache[cache_key]
            if arcpy.Exists(cached_path):
                self._log_debug(f"  Using cached clip geometry: {cache_key}")
                return cached_path
        
        self._log_debug(f"  Creating clip geometry for {len(mapsheet_numbers)} mapsheets (safe_key: {safe_key})")
        start_time = time.time()
        
        # Build the full path to the layer in the GPKG
        gpkg_layer = f"{self.config.admin_zones_path}/{self.config.mapsheets_layer}"
        
        # Create where clause for selection
        numbers_str = ",".join(str(n) for n in mapsheet_numbers)
        where_clause = f"{self.config.mapsheet_nbr_column} IN ({numbers_str})"
        
        # Make a feature layer with selection (use safe_key in layer name)
        layer_name = f"mapsheets_sel_{safe_key}"
        if arcpy.Exists(layer_name):
            arcpy.management.Delete(layer_name)
            
        arcpy.management.MakeFeatureLayer(gpkg_layer, layer_name, where_clause)
        
        # Get count of selected features
        selected_count = int(arcpy.management.GetCount(layer_name)[0])
        self._log_debug(f"  Selected {selected_count} mapsheet polygons")
        
        if selected_count == 0:
            self._log_warning(f"No mapsheets found for: {where_clause}")
            arcpy.management.Delete(layer_name)
            return None
        
        # Copy to scratch and dissolve (use safe_key in FC names)
        temp_fc = arcpy.CreateScratchName(
            prefix=f"clip_{safe_key}_", 
            suffix="", 
            data_type="FeatureClass", 
            workspace=arcpy.env.scratchGDB
        )
        
        self._log_debug(f"  Temp FC: {temp_fc}")
        
        # Copy selected features to scratch
        arcpy.management.CopyFeatures(layer_name, temp_fc)
        self._log_debug(f"  Copied features to temp FC")
        
        # Verify the temp_fc was created
        if not arcpy.Exists(temp_fc):
            self._log_error(f"  Failed to create temp feature class: {temp_fc}")
            arcpy.management.Delete(layer_name)
            return None
        
        # Dissolve to single polygon (use safe_key in FC name)
        dissolved_fc = arcpy.CreateScratchName(
            prefix=f"diss_{safe_key}_", 
            suffix="", 
            data_type="FeatureClass", 
            workspace=arcpy.env.scratchGDB
        )
        
        self._log_debug(f"  Dissolving to: {dissolved_fc}")
        arcpy.management.Dissolve(temp_fc, dissolved_fc)
        
        # Cleanup temp feature class
        arcpy.management.Delete(temp_fc)
        arcpy.management.Delete(layer_name)
        
        elapsed = time.time() - start_time
        self._log_debug(f"  Clip geometry created in {elapsed:.2f}s")
        
        # Cache the result
        self._clip_fc_cache[cache_key] = dissolved_fc
        
        return dissolved_fc
    
    def _clip_and_append_layer(
        self,
        layer_name: str,
        source_path: Path,
        mapsheet_numbers: List[int],
        output_fc: str,
        source_name: str
    ) -> int:
        """Clip layer from source and append to output."""
        
        import os
        
        actual_layer = layer_name.split("/")[-1]
        
        # Source feature class path - use os.path.join for proper Windows paths
        source_gdb = str(source_path)
        if "/" in layer_name:
            feature_dataset = layer_name.split("/")[0]
            source_fc = os.path.join(source_gdb, feature_dataset, actual_layer)
        else:
            source_fc = os.path.join(source_gdb, actual_layer)
        
        self._log_debug(f"  Processing {source_name}:{actual_layer}")
        self._log_debug(f"    Source FC: {source_fc}")
        self._log_debug(f"    Output FC: {output_fc}")
            
        if not arcpy.Exists(source_fc):
            self._log_debug(f"    Layer not found, skipping")
            return 0
        
        # Get source feature count
        source_count = int(arcpy.management.GetCount(source_fc)[0])
        self._log_debug(f"    Source features: {source_count}")
            
        # Get clip geometry (use source_name as cache key)
        clip_fc = self._create_clip_geometry_arcpy(mapsheet_numbers, source_name)
        
        if clip_fc is None:
            self._log_warning(f"    No clip geometry for {source_name}")
            return 0
        
        try:
            start_time = time.time()
            
            # Sanitize source_name for use in FC names
            safe_source = source_name.replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
            
            # Clip features to scratch
            clipped_fc = arcpy.CreateScratchName(
                prefix=f"clp_{actual_layer[:10]}_{safe_source[:10]}_", 
                suffix="", 
                data_type="FeatureClass", 
                workspace=arcpy.env.scratchGDB
            )
            
            self._log_debug(f"    Clipping to scratch: {clipped_fc}")
            
            arcpy.analysis.Clip(
                in_features=source_fc,
                clip_features=clip_fc,
                out_feature_class=clipped_fc
            )
            
            # Count clipped features
            count = int(arcpy.management.GetCount(clipped_fc)[0])
            
            clip_time = time.time() - start_time
            self._log_debug(f"    Clipped {count} features in {clip_time:.2f}s")
            
            if count > 0:
                # Add source tracking field if not exists
                existing_fields = [f.name for f in arcpy.ListFields(clipped_fc)]
                if "_MERGE_SOURCE" not in existing_fields:
                    arcpy.management.AddField(clipped_fc, "_MERGE_SOURCE", "TEXT", field_length=50)
                    self._log_debug(f"    Added _MERGE_SOURCE field")
                    
                # Update source field
                arcpy.management.CalculateField(
                    clipped_fc, "_MERGE_SOURCE", f"'{source_name}'", "PYTHON3"
                )
                
                # Check if output FC already exists - normalize path for comparison
                output_fc_normalized = os.path.normpath(output_fc)
                output_exists = arcpy.Exists(output_fc_normalized)
                self._log_debug(f"    Output exists check ({output_fc_normalized}): {output_exists}")
                
                if output_exists:
                    # Append to existing FC
                    self._log_debug(f"    Appending {count} features to existing output")
                    try:
                        arcpy.management.Append(
                            inputs=clipped_fc, 
                            target=output_fc_normalized, 
                            schema_type="NO_TEST"
                        )
                        self._log_debug(f"    Append successful")
                    except arcpy.ExecuteError as append_err:
                        self._log_error(f"    Append failed: {append_err}")
                        self._log_debug(f"    ArcPy messages: {arcpy.GetMessages()}")
                        count = 0
                else:
                    # Create new FC by copying - ensure parent feature dataset exists
                    self._log_debug(f"    Creating new output FC: {output_fc_normalized}")
                    arcpy.management.CopyFeatures(clipped_fc, output_fc_normalized)
                    
                    # Verify it was created at the expected path
                    if arcpy.Exists(output_fc_normalized):
                        self._log_debug(f"    Created successfully")
                    else:
                        self._log_error(f"    Failed to create output FC at expected path!")
                        count = 0
                    
            # Cleanup clipped features
            arcpy.management.Delete(clipped_fc)
            
            return count
            
        except arcpy.ExecuteError as e:
            self._log_error(f"ArcPy error clipping {layer_name}: {e}")
            self._log_debug(f"    Full error: {arcpy.GetMessages(2)}")
            return 0
    
    def _copy_table(self, table_name: str, source_path: Path) -> bool:
        """Copy a non-spatial table from source to output."""
        
        source_table = str(source_path / table_name)
        output_table = str(self.config.output_path / table_name)
        
        self._log_debug(f"  Copying table: {table_name}")
        self._log_debug(f"    From: {source_table}")
        self._log_debug(f"    To: {output_table}")
        
        if not arcpy.Exists(source_table):
            self._log_warning(f"Table not found: {source_table}")
            return False
            
        try:
            start_time = time.time()
            arcpy.management.Copy(source_table, output_table)
            
            # Get row count
            row_count = int(arcpy.management.GetCount(output_table)[0])
            elapsed = time.time() - start_time
            
            self._log_debug(f"    Copied {row_count} rows in {elapsed:.2f}s")
            return True
            
        except arcpy.ExecuteError as e:
            self._log_error(f"Error copying table {table_name}: {e}")
            return False
    
    def _cleanup_scratch(self) -> None:
        """Clean up scratch workspace."""
        
        self._log_debug("Cleaning up scratch workspace...")
        
        # Delete cached clip geometries
        for cache_key, fc_path in self._clip_fc_cache.items():
            try:
                if arcpy.Exists(fc_path):
                    arcpy.management.Delete(fc_path)
                    self._log_debug(f"  Deleted: {fc_path}")
            except:
                pass
                
        self._clip_fc_cache.clear()
    
    def merge(self) -> MergeStats:
        """Execute the merge operation using arcpy."""
        
        import os
        
        console.print("\n[bold blue]🔀 Starting ArcPy GDB Merge[/bold blue]\n")
        
        total_start_time = time.time()
        
        try:
            # Create output GDB
            self._create_output_gdb()
            
            # Create feature dataset
            feature_dataset = self._create_feature_dataset()
            self._log_debug(f"Feature dataset path: {feature_dataset}")
            
            # Load mapsheet assignments
            mapsheets_by_source = self._load_mapsheets_arcpy()
            
            # Display summary
            self._display_source_summary(mapsheets_by_source)
            
            # Process spatial layers
            console.print("\n[cyan]Processing spatial layers...[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=not self.verbose  # Keep progress visible in verbose mode
            ) as progress:
                
                layers_task = progress.add_task(
                    "[cyan]Processing spatial layers...",
                    total=len(self.config.spatial_layers)
                )
                
                for layer_name in self.config.spatial_layers:
                    actual_layer = layer_name.split("/")[-1]
                    progress.update(layers_task, description=f"[cyan]Processing {actual_layer}...")
                    
                    layer_start_time = time.time()
                    
                    # Determine output location using os.path.join for proper path format
                    output_gdb = str(self.config.output_path)
                    if "/" in layer_name:
                        # Layer is in a feature dataset
                        output_fc = os.path.join(output_gdb, "GC_ROCK_BODIES", actual_layer)
                    else:
                        output_fc = os.path.join(output_gdb, actual_layer)
                    
                    self._log_debug(f"Layer: {layer_name}")
                    self._log_debug(f"  Output FC: {output_fc}")
                    
                    layer_total = 0
                    
                    for source_name, mapsheet_numbers in mapsheets_by_source.items():
                        source_path = self._resolve_source_path(source_name)
                        if source_path is None:
                            self._log_warning(f"Source not found: {source_name}")
                            continue
                            
                        count = self._clip_and_append_layer(
                            layer_name, source_path, mapsheet_numbers, output_fc, source_name
                        )
                        
                        layer_total += count
                        self.stats.features_per_source[source_name] = \
                            self.stats.features_per_source.get(source_name, 0) + count
                    
                    layer_elapsed = time.time() - layer_start_time
                    
                    if layer_total > 0:
                        self.stats.features_per_layer[layer_name] = layer_total
                        self.stats.layers_processed += 1
                        console.print(f"  [green]✓[/green] {actual_layer}: {layer_total:,} features ({layer_elapsed:.1f}s)")
                    else:
                        console.print(f"  [yellow]○[/yellow] {actual_layer}: no features")
                        
                    progress.advance(layers_task)
                            self._log_warning(f"Source not found: {source_name}")
                            continue
                            
                        count = self._clip_and_append_layer(
                            layer_name, source_path, mapsheet_numbers, output_fc, source_name
                        )
                        
                        layer_total += count
                        self.stats.features_per_source[source_name] = \
                            self.stats.features_per_source.get(source_name, 0) + count
                    
                    layer_elapsed = time.time() - layer_start_time
                    
                    if layer_total > 0:
                        self.stats.features_per_layer[layer_name] = layer_total
                        self.stats.layers_processed += 1
                        console.print(f"  [green]✓[/green] {actual_layer}: {layer_total:,} features ({layer_elapsed:.1f}s)")
                    else:
                        console.print(f"  [yellow]○[/yellow] {actual_layer}: no features")
                        
                    progress.advance(layers_task)
            
            # Copy non-spatial tables
            if self.config.non_spatial_tables:
                console.print("\n[cyan]Copying reference tables...[/cyan]")
                
                ref_path = self._resolve_source_path(self.config.reference_source)
                if ref_path:
                    self._log_info(f"Reference source: {ref_path}")
                    
                    for table_name in self.config.non_spatial_tables:
                        if self._copy_table(table_name, ref_path):
                            self.stats.tables_copied += 1
                            console.print(f"  [green]✓[/green] {table_name}")
                        else:
                            console.print(f"  [yellow]○[/yellow] {table_name}: not found")
                else:
                    self._log_warning(f"Reference source not found: {self.config.reference_source}")
            
            # Cleanup
            self._cleanup_scratch()
            
            # Display results
            total_elapsed = time.time() - total_start_time
            self._display_results(total_elapsed)
            
        except Exception as e:
            self._log_error(f"Merge failed: {e}")
            self._cleanup_scratch()
            raise
        
        return self.stats
    
    def _display_source_summary(self, mapsheets_by_source: Dict[str, List[int]]) -> None:
        """Display summary of sources and mapsheet assignments."""
        
        table = Table(title="Source Assignments", show_header=True)
        table.add_column("Source", style="cyan")
        table.add_column("Mapsheets", justify="right", style="yellow")
        table.add_column("Sample", style="dim")
        table.add_column("Status", style="green")
        
        for source_name, mapsheets in mapsheets_by_source.items():
            source_path = self._resolve_source_path(source_name)
            status = "✓ Found" if source_path else "✗ Missing"
            
            # Show sample of mapsheet numbers
            sample = ", ".join(str(n) for n in sorted(mapsheets)[:5])
            if len(mapsheets) > 5:
                sample += f" ... (+{len(mapsheets) - 5})"
                
            table.add_row(source_name, str(len(mapsheets)), sample, status)
            
        console.print(table)
    
    def _display_results(self, elapsed_time: float = 0) -> None:
        """Display merge results."""
        
        console.print("\n[bold green]✅ Merge Complete![/bold green]\n")
        
        table = Table(title="Merge Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("Mapsheets processed", str(self.stats.mapsheets_processed))
        table.add_row("Spatial layers merged", str(self.stats.layers_processed))
        table.add_row("Reference tables copied", str(self.stats.tables_copied))
        
        total_features = sum(self.stats.features_per_layer.values())
        table.add_row("Total features", f"{total_features:,}")
        
        if elapsed_time > 0:
            mins, secs = divmod(elapsed_time, 60)
            table.add_row("Processing time", f"{int(mins)}m {secs:.1f}s")
        
        console.print(table)
        
        # Features by source
        if self.stats.features_per_source:
            console.print("\n[bold]Features by Source:[/bold]")
            for source, count in sorted(self.stats.features_per_source.items(), key=lambda x: -x[1]):
                console.print(f"  • {source}: {count:,}")
        
        # Features by layer (if verbose)
        if self.verbose and self.stats.features_per_layer:
            console.print("\n[bold]Features by Layer:[/bold]")
            for layer, count in sorted(self.stats.features_per_layer.items()):
                layer_short = layer.split("/")[-1]
                console.print(f"  • {layer_short}: {count:,}")
        
        console.print(f"\n[dim]Output: {self.config.output_path}[/dim]")


def create_merger(config: MergeConfig, prefer_arcpy: bool = True, verbose: bool = False):
    """
    Create appropriate merger based on available libraries.
    
    Args:
        config: Merge configuration
        prefer_arcpy: If True, use arcpy when available
        verbose: Enable verbose logging
        
    Returns:
        GDBMergerArcPy or GDBMergerGeopandas instance
    """
    
    if prefer_arcpy and HAS_ARCPY:
        logger.info("Using arcpy-based merger for optimal FileGDB handling")
        return GDBMergerArcPy(config, verbose=verbose)
    else:
        logger.info("Using geopandas-based merger")
        return GDBMergerGeopandas(config)
