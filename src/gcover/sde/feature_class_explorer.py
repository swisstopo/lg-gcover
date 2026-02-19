#!/usr/bin/env python
"""
List feature classes in GCOVERP database with Rich display.
Can be used standalone or integrated into gcover CLI.
"""
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text
from rich import box

from gcover.arcpy_compat import HAS_ARCPY, arcpy

console = Console()


def list_all_feature_classes(workspace):
    """
    List all feature classes including those in feature datasets.
    
    Returns:
        dict: Organized structure of feature datasets and feature classes
    """
    arcpy.env.workspace = workspace
    
    results = {
        "standalone": [],  # Feature classes not in a dataset
        "datasets": {}     # Feature datasets with their feature classes
    }
    
    # List standalone feature classes (not in any dataset)
    standalone_fcs = arcpy.ListFeatureClasses()
    if standalone_fcs:
        results["standalone"] = standalone_fcs
    
    # List feature datasets
    feature_datasets = arcpy.ListDatasets(feature_type="Feature")
    
    if feature_datasets:
        for dataset in feature_datasets:
            # List feature classes within this dataset
            arcpy.env.workspace = f"{workspace}\\{dataset}"
            fcs_in_dataset = arcpy.ListFeatureClasses()
            
            results["datasets"][dataset] = fcs_in_dataset if fcs_in_dataset else []
            
            # Reset workspace
            arcpy.env.workspace = workspace
    
    return results


def display_feature_class_tree(results, workspace, show_paths=True):
    """Display feature class hierarchy using Rich Tree."""
    
    # Create root tree
    tree = Tree(
        f"[bold cyan]📦 {workspace}[/bold cyan]",
        guide_style="bright_blue"
    )
    
    # Add standalone feature classes
    if results["standalone"]:
        standalone_branch = tree.add("[yellow]📁 Standalone Feature Classes[/yellow]")
        for fc in results["standalone"]:
            fc_node = standalone_branch.add(f"[green]{fc}[/green]")
            if show_paths:
                fc_node.add(f"[dim]Path: {fc}[/dim]")
    
    # Add feature datasets and their contents
    if results["datasets"]:
        datasets_branch = tree.add("[yellow]📁 Feature Datasets[/yellow]")
        
        for dataset, fcs in sorted(results["datasets"].items()):
            dataset_branch = datasets_branch.add(f"[blue]{dataset}[/blue]")
            
            if fcs:
                for fc in sorted(fcs):
                    fc_node = dataset_branch.add(f"[green]{fc}[/green]")
                    if show_paths:
                        full_path = f"{dataset}/{fc}"
                        fc_node.add(f"[dim]Path: {full_path}[/dim]")
            else:
                dataset_branch.add("[dim italic](empty)[/dim italic]")
    
    console.print(tree)
    
    # Summary panel
    total_standalone = len(results["standalone"])
    total_in_datasets = sum(len(fcs) for fcs in results["datasets"].values())
    total = total_standalone + total_in_datasets
    
    summary = f"""
[bold]Summary:[/bold]
  • Total feature classes: [cyan]{total}[/cyan]
  • Standalone: [cyan]{total_standalone}[/cyan]
  • In datasets: [cyan]{total_in_datasets}[/cyan] across [cyan]{len(results['datasets'])}[/cyan] dataset(s)
    """
    
    console.print(Panel(summary.strip(), title="[bold]Statistics[/bold]", border_style="green"))


def display_feature_class_table(results, workspace):
    """Display feature classes as a table."""
    
    table = Table(
        title=f"Feature Classes in {workspace}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Dataset", style="blue")
    table.add_column("Feature Class", style="green")
    table.add_column("Full Path", style="dim")
    
    # Add standalone feature classes
    for fc in sorted(results["standalone"]):
        table.add_row("Standalone", "-", fc, fc)
    
    # Add feature classes in datasets
    for dataset, fcs in sorted(results["datasets"].items()):
        if fcs:
            for fc in sorted(fcs):
                full_path = f"{dataset}/{fc}"
                table.add_row("In Dataset", dataset, fc, full_path)
        else:
            table.add_row("In Dataset", dataset, "[dim](empty)[/dim]", "-")
    
    console.print(table)


def get_feature_class_info(workspace, feature_class_path):
    """
    Get detailed info about a specific feature class.
    
    Args:
        workspace: SDE workspace path
        feature_class_path: Either "FeatureClass" or "Dataset/FeatureClass"
    """
    arcpy.env.workspace = workspace
    full_path = str(Path(workspace) / Path(feature_class_path))
    
    try:
        with console.status(f"[bold green]Loading info for {feature_class_path}..."):
            desc = arcpy.Describe(full_path)
            count = arcpy.management.GetCount(full_path)
            fields = arcpy.ListFields(full_path)
        
        # Create info panel
        info_text = f"""[bold cyan]Path:[/bold cyan] {full_path}
[bold cyan]Type:[/bold cyan] {desc.dataType}
[bold cyan]Shape:[/bold cyan] {desc.shapeType}
[bold cyan]Dimensions:[/bold cyan] {'3D' if desc.hasZ else '2D'}, {'M-enabled' if desc.hasM else 'No M values'}
[bold cyan]Spatial Reference:[/bold cyan] {desc.spatialReference.name}
[bold cyan]Feature Count:[/bold cyan] {count}
"""
        
        console.print(Panel(
            info_text.strip(),
            title=f"[bold]{feature_class_path}[/bold]",
            border_style="blue"
        ))
        
        # Create fields table
        fields_table = Table(
            title="Fields",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold yellow"
        )
        
        fields_table.add_column("Name", style="cyan", width=30)
        fields_table.add_column("Type", style="green", width=15)
        fields_table.add_column("Length", justify="right", width=8)
        fields_table.add_column("Nullable", justify="center", width=10)
        
        for field in fields:
            nullable = "✓" if field.isNullable else "✗"
            length = str(field.length) if hasattr(field, 'length') and field.length else "-"
            fields_table.add_row(
                field.name,
                field.type,
                length,
                nullable
            )
        
        console.print(fields_table)
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return False


def search_feature_classes(workspace, search_term):
    """Search for feature classes by name."""
    results = list_all_feature_classes(workspace)
    matches = []
    
    search_term = search_term.lower()
    
    # Search standalone
    for fc in results["standalone"]:
        if search_term in fc.lower():
            matches.append({"type": "standalone", "path": fc, "name": fc, "dataset": None})
    
    # Search in datasets
    for dataset, fcs in results["datasets"].items():
        for fc in fcs:
            if search_term in fc.lower() or search_term in dataset.lower():
                matches.append({
                    "type": "in_dataset",
                    "path": f"{dataset}/{fc}",
                    "name": fc,
                    "dataset": dataset
                })
    
    return matches


def display_search_results(matches, search_term):
    """Display search results in a table."""
    if not matches:
        console.print(f"[yellow]No feature classes found matching '[bold]{search_term}[/bold]'[/yellow]")
        return
    
    table = Table(
        title=f"Search Results: '{search_term}' ({len(matches)} matches)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("#", justify="right", style="cyan", width=4)
    table.add_column("Type", style="yellow", width=12)
    table.add_column("Dataset", style="blue")
    table.add_column("Feature Class", style="green")
    table.add_column("Full Path", style="dim")
    
    for i, match in enumerate(matches, 1):
        table.add_row(
            str(i),
            match["type"].replace("_", " ").title(),
            match["dataset"] or "-",
            match["name"],
            match["path"]
        )
    
    console.print(table)


# =============================================================================
# Standalone usage
# =============================================================================

if __name__ == "__main__":
    from gcover.sde.connection_manager import SDEConnectionManager
    
    console.rule("[bold blue]GCOVERP Feature Class Explorer[/bold blue]")
    
    with SDEConnectionManager() as conn_mgr:
        # Connect to GCOVERP
        with console.status("[bold green]Connecting to GCOVERP..."):
            conn = conn_mgr.create_connection("GCOVERP")
            workspace = str(conn)
        
        console.print(f"[green]✓[/green] Connected to: [cyan]{workspace}[/cyan]\n")
        
        # List all feature classes
        with console.status("[bold green]Scanning database structure..."):
            results = list_all_feature_classes(workspace)
        
        console.print()
        display_feature_class_tree(results, workspace, show_paths=True)
        
        # Example: Search for BEDROCK
        console.print("\n")
        console.rule("[bold blue]Example: Searching for 'bedrock'[/bold blue]")
        console.print()
        
        bedrock_matches = search_feature_classes(workspace, "bedrock")
        display_search_results(bedrock_matches, "bedrock")
        
        if bedrock_matches:
            console.print("\n")
            console.rule("[bold blue]Details for first match[/bold blue]")
            console.print()
            get_feature_class_info(workspace, bedrock_matches[0]["path"])
    
    console.print("\n[green]✓ Done![/green]")
