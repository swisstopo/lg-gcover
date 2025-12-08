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
    
    def __init__(self, config: MergeConfig):
        if not HAS_ARCPY:
            raise ImportError("arcpy is required for GDBMergerArcPy")
            
        self.config = config
        self.sources: Dict[str, SourceConfig] = {}
        self.stats = MergeStats()
        
        self._setup_sources()
        self._setup_workspace()
        
    def _setup_sources(self) -> None:
        """Configure available source databases."""
        
        if self.config.rc1_path and self.config.rc1_path.exists():
            self.sources["RC1"] = SourceConfig(
                name="RC1",
                path=self.config.rc1_path,
                source_type=SourceType.RC1
            )
            
        if self.config.rc2_path and self.config.rc2_path.exists():
            self.sources["RC2"] = SourceConfig(
                name="RC2",
                path=self.config.rc2_path,
                source_type=SourceType.RC2
            )
            
        if self.config.custom_sources_dir and self.config.custom_sources_dir.exists():
            for gdb_path in self.config.custom_sources_dir.glob("*.gdb"):
                source_name = gdb_path.name
                self.sources[source_name] = SourceConfig(
                    name=source_name,
                    path=gdb_path,
                    source_type=SourceType.CUSTOM
                )
                
    def _setup_workspace(self) -> None:
        """Set up arcpy workspace and environment."""
        
        arcpy.env.overwriteOutput = True
        arcpy.env.preserveGlobalIds = True
        
        # Use first available source as template
        template_path = None
        for source in self.sources.values():
            if source.path.exists():
                template_path = source.path
                break
                
        if template_path:
            arcpy.env.workspace = str(template_path)
            
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
        
        if output_path.exists():
            logger.warning(f"Deleting existing output: {output_path}")
            arcpy.management.Delete(str(output_path))
            
        # Create new FileGDB
        arcpy.management.CreateFileGDB(
            out_folder_path=str(output_path.parent),
            out_name=output_path.name
        )
        
        logger.info(f"Created output GDB: {output_path}")
        
    def _create_feature_dataset(self, name: str = "GC_ROCK_BODIES") -> str:
        """Create feature dataset in output GDB."""
        
        output_path = self.config.output_path
        feature_dataset_path = str(output_path / name)
        
        if not arcpy.Exists(feature_dataset_path):
            # Get spatial reference from first source
            sr = None
            for source in self.sources.values():
                for layer in self.config.spatial_layers:
                    layer_name = layer.split("/")[-1]
                    source_fc = str(source.path / "GC_ROCK_BODIES" / layer_name)
                    if arcpy.Exists(source_fc):
                        sr = arcpy.Describe(source_fc).spatialReference
                        break
                if sr:
                    break
                    
            if sr:
                arcpy.management.CreateFeatureDataset(
                    out_dataset_path=str(output_path),
                    out_name=name,
                    spatial_reference=sr
                )
                
        return feature_dataset_path
    
    def _load_mapsheets(self) -> Dict[str, List[int]]:
        """Load mapsheets grouped by source."""
        
        import geopandas as gpd
        
        gdf = gpd.read_file(
            self.config.admin_zones_path,
            layer=self.config.mapsheets_layer
        )
        
        if self.config.mapsheet_numbers:
            gdf = gdf[gdf[self.config.mapsheet_nbr_column].isin(self.config.mapsheet_numbers)]
            
        # Group by source
        grouped = {}
        for source_name in gdf[self.config.source_column].unique():
            mask = gdf[self.config.source_column] == source_name
            grouped[source_name] = gdf[mask][self.config.mapsheet_nbr_column].tolist()
            
        self.stats.mapsheets_processed = len(gdf)
        return grouped
    
    def _get_mapsheet_geometry(self, mapsheet_numbers: List[int]) -> str:
        """Get union geometry for mapsheets as a temporary feature class."""
        
        import geopandas as gpd
        import tempfile
        
        # Read mapsheets
        gdf = gpd.read_file(
            self.config.admin_zones_path,
            layer=self.config.mapsheets_layer
        )
        
        # Filter to specified mapsheets
        gdf = gdf[gdf[self.config.mapsheet_nbr_column].isin(mapsheet_numbers)]
        
        # Create temporary feature class
        temp_fc = arcpy.CreateScratchName("clip_", "", "FeatureClass", arcpy.env.scratchGDB)
        
        # Convert geopandas to feature class
        gdf.to_file(temp_fc, driver="OpenFileGDB")
        
        # Dissolve to single polygon
        dissolved_fc = arcpy.CreateScratchName("dissolved_", "", "FeatureClass", arcpy.env.scratchGDB)
        arcpy.management.Dissolve(temp_fc, dissolved_fc)
        
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
        
        actual_layer = layer_name.split("/")[-1]
        
        # Source feature class path
        if "/" in layer_name:
            feature_dataset = layer_name.split("/")[0]
            source_fc = str(source_path / feature_dataset / actual_layer)
        else:
            source_fc = str(source_path / actual_layer)
            
        if not arcpy.Exists(source_fc):
            logger.debug(f"Layer not found: {source_fc}")
            return 0
            
        # Get clip geometry
        clip_fc = self._get_mapsheet_geometry(mapsheet_numbers)
        
        try:
            # Clip features
            clipped_fc = arcpy.CreateScratchName("clipped_", "", "FeatureClass", arcpy.env.scratchGDB)
            
            arcpy.analysis.Clip(
                in_features=source_fc,
                clip_features=clip_fc,
                out_feature_class=clipped_fc
            )
            
            # Count clipped features
            count = int(arcpy.management.GetCount(clipped_fc)[0])
            
            if count > 0:
                # Add source tracking field if not exists
                if "_MERGE_SOURCE" not in [f.name for f in arcpy.ListFields(clipped_fc)]:
                    arcpy.management.AddField(clipped_fc, "_MERGE_SOURCE", "TEXT", field_length=50)
                    
                # Update source field
                arcpy.management.CalculateField(
                    clipped_fc, "_MERGE_SOURCE", f"'{source_name}'", "PYTHON3"
                )
                
                # Append to output
                if arcpy.Exists(output_fc):
                    arcpy.management.Append(clipped_fc, output_fc, "NO_TEST")
                else:
                    arcpy.management.CopyFeatures(clipped_fc, output_fc)
                    
            # Cleanup
            arcpy.management.Delete(clipped_fc)
            arcpy.management.Delete(clip_fc)
            
            return count
            
        except arcpy.ExecuteError as e:
            logger.error(f"ArcPy error clipping {layer_name}: {e}")
            return 0
    
    def _copy_table(self, table_name: str, source_path: Path) -> bool:
        """Copy a non-spatial table from source to output."""
        
        source_table = str(source_path / table_name)
        output_table = str(self.config.output_path / table_name)
        
        if not arcpy.Exists(source_table):
            logger.warning(f"Table not found: {source_table}")
            return False
            
        try:
            arcpy.management.Copy(source_table, output_table)
            return True
        except arcpy.ExecuteError as e:
            logger.error(f"Error copying table {table_name}: {e}")
            return False
    
    def merge(self) -> MergeStats:
        """Execute the merge operation using arcpy."""
        
        console.print("\n[bold blue]🔀 Starting ArcPy GDB Merge[/bold blue]\n")
        
        # Create output GDB
        self._create_output_gdb()
        
        # Create feature dataset
        feature_dataset = self._create_feature_dataset()
        
        # Load mapsheet assignments
        mapsheets_by_source = self._load_mapsheets()
        
        # Display summary
        self._display_source_summary(mapsheets_by_source)
        
        # Process spatial layers
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            layers_task = progress.add_task(
                "[cyan]Processing spatial layers...",
                total=len(self.config.spatial_layers)
            )
            
            for layer_name in self.config.spatial_layers:
                actual_layer = layer_name.split("/")[-1]
                progress.update(layers_task, description=f"[cyan]Processing {actual_layer}...")
                
                # Determine output location
                if "/" in layer_name:
                    output_fc = str(self.config.output_path / "GC_ROCK_BODIES" / actual_layer)
                else:
                    output_fc = str(self.config.output_path / actual_layer)
                
                layer_total = 0
                
                for source_name, mapsheet_numbers in mapsheets_by_source.items():
                    source_path = self._resolve_source_path(source_name)
                    if source_path is None:
                        continue
                        
                    count = self._clip_and_append_layer(
                        layer_name, source_path, mapsheet_numbers, output_fc, source_name
                    )
                    
                    layer_total += count
                    self.stats.features_per_source[source_name] = \
                        self.stats.features_per_source.get(source_name, 0) + count
                
                if layer_total > 0:
                    self.stats.features_per_layer[layer_name] = layer_total
                    self.stats.layers_processed += 1
                    
                progress.advance(layers_task)
        
        # Copy non-spatial tables
        if self.config.non_spatial_tables:
            console.print("\n[cyan]Copying reference tables...[/cyan]")
            
            ref_path = self._resolve_source_path(self.config.reference_source)
            if ref_path:
                for table_name in self.config.non_spatial_tables:
                    if self._copy_table(table_name, ref_path):
                        self.stats.tables_copied += 1
                        console.print(f"  [green]✓[/green] {table_name}")
        
        # Display results
        self._display_results()
        
        return self.stats
    
    def _display_source_summary(self, mapsheets_by_source: Dict[str, List[int]]) -> None:
        """Display summary of sources and mapsheet assignments."""
        
        table = Table(title="Source Assignments", show_header=True)
        table.add_column("Source", style="cyan")
        table.add_column("Mapsheets", justify="right", style="yellow")
        table.add_column("Status", style="green")
        
        for source_name, mapsheets in mapsheets_by_source.items():
            source_path = self._resolve_source_path(source_name)
            status = "✓ Found" if source_path else "✗ Missing"
            table.add_row(source_name, str(len(mapsheets)), status)
            
        console.print(table)
    
    def _display_results(self) -> None:
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
        
        console.print(table)
        
        console.print(f"\n[dim]Output: {self.config.output_path}[/dim]")


def create_merger(config: MergeConfig, prefer_arcpy: bool = True):
    """
    Create appropriate merger based on available libraries.
    
    Args:
        config: Merge configuration
        prefer_arcpy: If True, use arcpy when available
        
    Returns:
        GDBMergerArcPy or GDBMergerGeopandas instance
    """
    
    if prefer_arcpy and HAS_ARCPY:
        logger.info("Using arcpy-based merger for optimal FileGDB handling")
        return GDBMergerArcPy(config)
    else:
        logger.info("Using geopandas-based merger")
        return GDBMergerGeopandas(config)
