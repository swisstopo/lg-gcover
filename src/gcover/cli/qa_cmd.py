"""
Create this as gcover/cli/qa_cmd.py
Then import it in your main.py with: from .qa_cmd import qa; cli.add_command(qa)
"""

import click
from pathlib import Path
from typing import Optional
import sys
from tabulate import tabulate

# Your existing imports - adjust path as needed
from ..qa.lines_in_unco import (
    load_geodatabase_layers,
    filter_tectonic_lines,
    process_intersecting_lines,
    save_results,
    create_test_bbox_alps,
    create_custom_bbox,
)
from loguru import logger


@click.group()
def qa():
    """Quality assurance management commands."""
    pass


@qa.command("tectonic-lines")
@click.argument("gdb_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path (default: <input>_corrected.gpkg)",
)
@click.option(
    "--bbox", "-b", help='Bounding box "minx,miny,maxx,maxy" in LV95 coordinates'
)
@click.option(
    "--bbox-center",
    help='BBOX center "x,y,size_km" in LV95 coordinates (e.g., "2621000,1099000,10")',
)
@click.option(
    "--bbox-preset",
    type=click.Choice(["alps-3km", "alps-10km", "alps-20km"]),
    help="Predefined BBOX for Alps testing",
)
@click.option(
    "--linear-layer",
    default="GC_LINEAR_OBJECTS",
    help="Name of the linear objects layer",
)
@click.option(
    "--unco-layer", default="GC_UNCO_DESPOSIT", help="Name of the UNCO deposit layer"
)
@click.option(
    "--save-deleted/--no-save-deleted",
    default=False,
    help='Include original lines marked as "delete" in output',
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["gpkg", "shp", "gdb"]),
    default="gpkg",
    help="Output format",
)
@click.option("--verbose", "-v", is_flag=True, help="Detailed logging")
@click.option("--dry-run", is_flag=True, help="Simulation without modifications")
@click.option("--interactive", is_flag=True, help="Interactive mode for confirmations")
def tectonic_lines(
    gdb_path,
    output,
    bbox,
    bbox_center,
    bbox_preset,
    linear_layer,
    unco_layer,
    save_deleted,
    format,
    verbose,
    dry_run,
    interactive,
):
    """
    Correct tectonic lines intersecting UNCO deposit polygons.

    Tectonic lines (KIND 14901001-14901009) intersecting UNCO polygons
    are split at polygon boundaries. Parts inside UNCO polygons
    get TTEC_STATUS=14906003 (not certain).

    Examples:

      # Process complete dataset
      gcover qa tectonic-lines data.gdb -o corrected.gpkg

      # Alps test area (10km x 10km)
      gcover qa tectonic-lines data.gdb --bbox-preset alps-10km

      # Custom area around specific coordinates
      gcover qa tectonic-lines data.gdb --bbox-center "2621000,1099000,5"

      # Simulation to see what would be processed
      gcover qa tectonic-lines data.gdb --dry-run --verbose
    """

    # Configure logging
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        sink=lambda msg: click.echo(msg.rstrip(), err=True),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=log_level,
    )

    # Set default output path
    if output is None:
        suffix_map = {"gpkg": ".gpkg", "shp": ".shp", "gdb": ".gdb"}
        output = gdb_path.parent / f"{gdb_path.stem}_corrected{suffix_map[format]}"

    # Parse BBOX options
    test_bbox = None

    if bbox:
        try:
            coords = [float(x.strip()) for x in bbox.split(",")]
            if len(coords) != 4:
                raise ValueError("BBOX must have 4 coordinates")
            test_bbox = tuple(coords)
            click.echo(f"🗺️  Explicit BBOX: {test_bbox}")
        except ValueError as e:
            click.echo(f"❌ Error parsing BBOX: {e}", err=True)
            sys.exit(1)

    elif bbox_center:
        try:
            parts = [x.strip() for x in bbox_center.split(",")]
            if len(parts) != 3:
                raise ValueError("BBOX center must be 'x,y,size_km'")
            center_x, center_y, size_km = (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
            )
            test_bbox = create_custom_bbox(center_x, center_y, size_km)
        except ValueError as e:
            click.echo(f"❌ Error parsing BBOX center: {e}", err=True)
            sys.exit(1)

    elif bbox_preset:
        preset_map = {
            "alps-3km": lambda: create_test_bbox_alps(size_km=3),
            "alps-10km": lambda: create_test_bbox_alps(size_km=10),
            "alps-20km": lambda: create_test_bbox_alps(size_km=20),
        }
        test_bbox = preset_map[bbox_preset]()

    # Display start info
    click.echo(f"🔧 Tectonic lines correction")
    click.echo(f"📂 Input: {gdb_path}")
    click.echo(f"💾 Output: {output}")

    if dry_run:
        click.echo(f"🧪 SIMULATION MODE - No modifications")

    # Load data
    try:
        with click.progressbar(label="📁 Loading data") as bar:
            tectonic_lines, unco_polygons = load_geodatabase_layers(
                gdb_path, linear_layer, unco_layer, bbox=test_bbox
            )
            bar.update(1)
    except Exception as e:
        click.echo(f"❌ Loading error: {e}", err=True)
        sys.exit(1)

    if tectonic_lines is None or unco_polygons is None:
        click.echo("❌ Failed to load required layers", err=True)
        sys.exit(1)

    # Filter tectonic lines
    filtered_lines = filter_tectonic_lines(tectonic_lines)

    if len(filtered_lines) == 0:
        click.echo("ℹ️  No tectonic lines found (KIND 14901001-14901009)")
        return

    # Display initial statistics
    stats_data = [
        ["Tectonic lines", len(filtered_lines)],
        ["UNCO polygons", len(unco_polygons)],
        ["Processing area", f"{test_bbox}" if test_bbox else "Complete dataset"],
    ]
    click.echo("\n📊 Statistics:")
    click.echo(tabulate(stats_data, headers=["Type", "Count"], tablefmt="grid"))

    # Interactive mode
    if interactive and not dry_run:
        if not click.confirm("🚀 Continue processing?"):
            click.echo("⏹️  Processing cancelled")
            return

    # Process intersections
    try:
        with click.progressbar(label="🔄 Processing intersections") as bar:
            new_features, updated_features, intersecting_indices = (
                process_intersecting_lines(filtered_lines, unco_polygons)
            )
            bar.update(1)
    except Exception as e:
        click.echo(f"❌ Error processing intersections: {e}", err=True)
        sys.exit(1)

    if not new_features and not updated_features:
        click.echo("ℹ️  No intersections found. No modifications needed.")
        return

    # Display summary
    result_data = [
        ["New segments (outside)", len(new_features)],
        ["Modified segments (inside)", len(updated_features)],
        ["Original lines processed", len(intersecting_indices)],
    ]
    click.echo("\n📈 Results:")
    click.echo(tabulate(result_data, headers=["Type", "Count"], tablefmt="grid"))

    if dry_run:
        click.echo(f"🧪 SIMULATION: Would save to {output}")
        click.echo("✅ Simulation completed successfully!")
        return

    # Save results
    try:
        with click.progressbar(label="💾 Saving results") as bar:
            save_results(
                filtered_lines,
                new_features,
                updated_features,
                intersecting_indices,
                output,
                save_deleted=save_deleted,
            )
            bar.update(1)

        click.echo("✅ Tectonic lines correction completed successfully!")
        click.echo(f"📁 Results saved: {output}")

    except Exception as e:
        click.echo(f"❌ Save error: {e}", err=True)
        sys.exit(1)


# Add more QA commands here as needed
@qa.command("info")
def qa_info():
    """Display QA module information and available commands."""
    click.echo("🔍 Quality Assurance Module")
    click.echo("\nAvailable commands:")
    commands_data = [
        ["tectonic-lines", "Correct tectonic lines intersecting UNCO polygons"],
        # Add more commands here as you develop them
    ]
    click.echo(
        tabulate(commands_data, headers=["Command", "Description"], tablefmt="grid")
    )


# If you want to add more QA subcommands later:
"""
@qa.command("validate-topology")
@click.argument('gdb_path', type=click.Path(exists=True, path_type=Path))
def validate_topology(gdb_path):
    \"\"\"Validate topology rules in geodatabase.\"\"\"
    # Your topology validation logic here
    pass

@qa.command("check-attributes")
@click.argument('gdb_path', type=click.Path(exists=True, path_type=Path))
def check_attributes(gdb_path):
    \"\"\"Check attribute consistency and completeness.\"\"\"
    # Your attribute checking logic here
    pass
"""
