#!/usr/bin/env python3
"""
GPKG Field Manager - A CLI tool to list and rename fields in GeoPackage files.
"""

import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import click
import fiona
import pyogrio
import geopandas as gpd
import numpy as np
import pandas as pd
import pandas.api.types as ptypes
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

warnings.filterwarnings("ignore", category=RuntimeWarning)


SAMPLE_VALUE_COLUMN_WIDTH = 40


console = Console()


def get_supported_layers(filename: str, spatial_only: bool = False) -> List[str]:
    """Get list of available layers in the file. Optionally filter to spatial layers only."""
    try:
        layers = gpd.list_layers(filename)

        # Normalize layer list
        if isinstance(layers, pd.DataFrame):
            layer_names = layers.iloc[:, 0].tolist()
        elif isinstance(layers, list):
            layer_names = layers
        else:
            layer_names = list(layers)

        # Filter to spatial layers if requested
        if spatial_only:
            spatial_layers = []
            for layer in layer_names:
                try:
                    gdf = gpd.read_file(
                        filename, layer=layer, engine="pyogrio", use_arrow=True
                    )
                    if gdf.geometry.name and not gdf.empty:
                        spatial_layers.append(layer)
                except Exception:
                    continue  # skip unreadable or non-spatial layers
            return spatial_layers

        return layer_names

    except Exception as e:
        console.print(f"[red]Error reading layers: {e}[/red]")
        return []


def safe_bool_check(value) -> bool:
    """Safely check if a value is truthy, handling DataFrames and Series."""
    if hasattr(value, "empty"):
        return not value.empty
    elif hasattr(value, "__len__"):
        return len(value) > 0
    else:
        return bool(value)


def get_field_info(gdf: gpd.GeoDataFrame) -> Table:
    """Create a rich table with field information."""
    table = Table(
        title="Field Information", show_header=True, header_style="bold magenta"
    )

    table.add_column("Index", style="dim", width=6)
    table.add_column("Field Name", style="cyan", width=20)
    table.add_column("Data Type", style="green", width=15)
    table.add_column("Nullable", style="yellow", width=10)
    table.add_column("Sample Value", style="white", width=SAMPLE_VALUE_COLUMN_WIDTH)

    for i, (col_name, col_type) in enumerate(gdf.dtypes.items()):
        # Check if column has any null values safely
        has_nulls = "Yes" if gdf[col_name].isnull().any() else "No"

        # Get sample value (first non-null if possible)
        sample_val = ""
        non_null_vals = gdf[col_name].dropna()
        if not non_null_vals.empty:
            sample_val = str(non_null_vals.iloc[0])
            if len(sample_val) > SAMPLE_VALUE_COLUMN_WIDTH:
                sample_val = sample_val[: SAMPLE_VALUE_COLUMN_WIDTH - 3] + "..."

        table.add_row(str(i), col_name, str(col_type), has_nulls, sample_val)

    return table


def change_field_type(
    gdf: gpd.GeoDataFrame, field_name: str, new_type: str
) -> gpd.GeoDataFrame:
    """Change field type with proper error handling."""
    try:
        if new_type in ("Int64", "Int32", "Int16", "Int8", "int"):  # Nullable type
            # Handle conversion to integer, dealing with NaN values
            new_type = "Int64" if new_type == "int" else new_type
            gdf[field_name] = pd.to_numeric(gdf[field_name], errors="coerce").astype(
                new_type
            )
        elif new_type in (
            "int64",
            "int32",
            "int16",
            "int8",
            "uint64",
            "uint32",
            "uint16",
            "uint8",
        ):
            gdf[field_name] = (
                pd.to_numeric(gdf[field_name], errors="coerce")
                .fillna(0)
                .astype(new_type)
            )

        elif new_type == "float":
            gdf[field_name] = pd.to_numeric(gdf[field_name], errors="coerce")
        elif new_type == "str":
            gdf[field_name] = gdf[field_name].astype(str)
        else:
            console.print(
                f"[yellow]Unsupported type '{new_type}'. Keeping original type.[/yellow]"
            )

        console.print(
            f"[green]✓ Successfully converted {field_name} to {new_type}[/green]"
        )
    except Exception as e:
        console.print(f"[red]Error converting {field_name} to {new_type}: {e}[/red]")

    return gdf


def process_layer(
    filename: str, layer: str, gdf: Optional[gpd.GeoDataFrame] = None
) -> tuple[Optional[gpd.GeoDataFrame], bool]:
    """
    Process a single layer with field management.

    Returns:
        - Modified GeoDataFrame (or None if no changes)
        - Boolean indicating whether to go back to layer selection (True) or quit/save (False)
    """

    if gdf is None:
        try:
            console.print(f"\n[bold]Reading layer:[/bold] {layer}")
            schema = pyogrio.read_info(filename, layer=layer)
            int_columns = [
                name
                for name, dtype in zip(schema["fields"], schema["dtypes"])
                if dtype in ["int32", "int64"]
            ]
            gdf = gpd.read_file(filename, layer=layer, engine="pyogrio", use_arrow=True)
            for col in int_columns:
                gdf[col] = gdf[col].astype("Int64")

            if gdf.empty:
                console.print("[yellow]Warning: The layer is empty.[/yellow]")
                return None, True
        except Exception as e:
            import traceback

            console.print(f"[red]Error reading layer: {e}[/red]")
            console.print(traceback.format_exc())
            return None, True

    # Display field information
    console.print(get_field_info(gdf))

    # Field management loop
    while True:
        console.print("\n[bold]Options:[/bold]")
        console.print("  [cyan]a[/cyan] - Add a new field")
        console.print("  [cyan]r[/cyan] - Rename a field")
        console.print("  [cyan]c[/cyan] - Change field type")
        console.print("  [cyan]d[/cyan] - Delete field")
        console.print("  [cyan]n[/cyan] - Nullify field")
        console.print("  [cyan]b[/cyan] - Back to layer selection (keep changes)")
        console.print("  [cyan]s[/cyan] - Save all changes and exit")
        console.print("  [cyan]q[/cyan] - Quit without saving")

        choice = Prompt.ask(
            "What would you like to do?",
            choices=["a", "r", "c", "d", "n", "b", "s", "q"],
            default="q",
        )

        if choice == "a":
            # New field
            new_name = Prompt.ask("Enter new field name")
            new_type = Prompt.ask(
                "Enter new type",
                choices=[
                    "object",
                    "string",
                    "float64",
                    "int64",
                    "int32",
                    "int16",
                    "int8",
                    "uint64",
                    "uint32",
                    "uint16",
                    "uint8",
                    "Int64",
                    "Int32",
                    "Int16",
                    "Int8",
                ],
                default="Int64",
            )

            gdf[new_name] = pd.Series([pd.NA] * len(gdf), dtype=new_type)

            console.print(f"[green]✓ Added new field '{new_name}' [{new_type}][/green]")

            # Show updated field info
            console.print(get_field_info(gdf))

        if choice == "r":
            # Rename field
            current_name = Prompt.ask("Enter field name to rename")

            if current_name not in gdf.columns:
                console.print(f"[red]Field '{current_name}' not found.[/red]")
                continue

            new_name = Prompt.ask("Enter new field name")

            if new_name in gdf.columns and new_name != current_name:
                if not Confirm.ask(f"Field '{new_name}' already exists. Overwrite?"):
                    continue

            gdf = gdf.rename(columns={current_name: new_name})
            console.print(f"[green]✓ Renamed '{current_name}' to '{new_name}'[/green]")

            # Show updated field info
            console.print(get_field_info(gdf))

        elif choice == "c":
            # Change field type
            field_name = Prompt.ask("Enter field name to change type")

            if field_name not in gdf.columns:
                console.print(f"[red]Field '{field_name}' not found.[/red]")
                continue

            current_type = gdf[field_name].dtype
            console.print(f"Current type: {current_type}")

            new_type = Prompt.ask(
                "Enter new type",
                choices=[
                    "str",
                    "int",
                    "float",
                    "int64",
                    "int32",
                    "int16",
                    "int8",
                    "uint64",
                    "uint32",
                    "uint16",
                    "uint8",
                    "Int64",
                    "Int32",
                    "Int16",
                    "Int8",
                ],
                default="Int64",
            )

            if str(current_type) == new_type:
                console.print("[yellow]Field already has the specified type.[/yellow]")
                continue

            gdf = change_field_type(gdf, field_name, new_type)

            # Show updated field info
            console.print(get_field_info(gdf))

        elif choice == "d":
            # Delete field
            current_name = Prompt.ask("Enter field name to delete")

            if current_name not in gdf.columns:
                console.print(f"[red]Field '{current_name}' not found.[/red]")
                continue

            if current_name in gdf.columns:
                if not Confirm.ask(f"Delete field '{current_name}'?"):
                    continue
            gdf = gdf.drop(current_name, axis=1)
            console.print(f"[green]✓ Dropped '{current_name}'[/green]")

            # Show updated field info
            console.print(get_field_info(gdf))

        elif choice == "n":
            # Nullify field
            current_name = Prompt.ask(
                "Enter field name to nullify (set all values to null)"
            )

            if current_name not in gdf.columns:
                console.print(f"[red]Field '{current_name}' not found.[/red]")
                continue

            if current_name in gdf.columns:
                if not Confirm.ask(f"Erase all values of field '{current_name}'?"):
                    continue
            if ptypes.is_numeric_dtype(gdf[current_name]):
                gdf[current_name] = np.nan
            elif ptypes.is_string_dtype(gdf[current_name]):
                gdf[current_name] = ""
            else:
                gdf[current_name] = (
                    None  # fallback for other types (e.g., datetime, object)
                )
            console.print(f"[green]✓ Set '{current_name}' all values to <None>[/green]")

            # Show updated field info
            console.print(get_field_info(gdf))

        elif choice == "b":
            # Return to layer selection with changes
            console.print(
                "[yellow]Returning to layer selection (changes kept in memory)[/yellow]"
            )
            return gdf, True

        elif choice == "s":
            # Save changes - return gdf and indicate we want to save and exit
            return gdf, False

        elif choice == "q":
            if Confirm.ask("Quit without saving?"):
                console.print("[yellow]Changes discarded.[/yellow]")
                return None, False


def save_modified_layers(
    modified_layers: Dict[str, gpd.GeoDataFrame], filename: str, new_filename: str
) -> int:
    """
    Save all modified layers to a file.
    For GPKG with existing file, recreates the entire file to avoid schema conflicts.
    """
    success_count = 0

    # Special handling for GPKG when overwriting
    if (
        new_filename.endswith(".gpkg")
        and Path(new_filename).exists()
        and new_filename == filename
    ):
        # Get list of all layers in original file
        all_layers = fiona.listlayers(new_filename)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Copy unmodified layers to temp file
            for layer_name in all_layers:
                if layer_name not in modified_layers:
                    gdf_original = gpd.read_file(
                        new_filename, layer=layer_name, engine="pyogrio", use_arrow=True
                    )
                    gdf_original.to_file(
                        tmp_path,
                        layer=layer_name,
                        driver="GPKG",
                        engine="pyogrio",
                        mode="a" if Path(tmp_path).stat().st_size > 0 else "w",
                    )

            # Add modified layers to temp file
            for layer_name, gdf in modified_layers.items():
                gdf.to_file(
                    tmp_path,
                    layer=layer_name,
                    driver="GPKG",
                    engine="pyogrio",
                    mode="a",
                )
                success_count += 1
                console.print(
                    f"[green]✓ Successfully saved layer '{layer_name}'[/green]"
                )

            # Replace original file with temp file
            shutil.move(tmp_path, new_filename)

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
            raise

    else:
        # Standard save for new files or non-GPKG
        for layer_name, gdf in modified_layers.items():
            try:
                if new_filename.endswith(".gpkg"):
                    if Path(new_filename).exists():
                        gdf.to_file(
                            filename=new_filename,
                            layer=layer_name,
                            driver="GPKG",
                            engine="pyogrio",
                            mode="a",
                        )
                    else:
                        gdf.to_file(
                            new_filename,
                            layer=layer_name,
                            driver="GPKG",
                            engine="pyogrio",
                        )
                else:
                    gdf.to_file(new_filename, engine="pyogrio")

                console.print(
                    f"[green]✓ Successfully saved layer '{layer_name}'[/green]"
                )
                success_count += 1

            except Exception as e:
                console.print(f"[red]✗ Error saving layer '{layer_name}': {e}[/red]")

    return success_count


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--layer", "-l", help="Layer name (if not specified, will list available layers)"
)
@click.option("--in-place", "-i", is_flag=True, help="Modify the file in place")
@click.option("--spatial-only", is_flag=True, help="List only spatial tables")
def info(filename: str, layer: Optional[str], in_place: bool, spatial_only: bool):
    """
    List and rename fields in a GeoPackage or other GIS file format.

    FILENAME: Path to the GeoPackage or other supported GIS file
    """

    # Dictionary to store modified layers
    modified_layers: Dict[str, gpd.GeoDataFrame] = {}

    # List available layers
    available_layers = get_supported_layers(filename, spatial_only=spatial_only)

    if not safe_bool_check(available_layers):
        console.print(
            "[red]No layers found in the file or unable to read the file.[/red]"
        )
        return

    # Main loop for layer selection
    while True:
        # Show modified layers status
        if modified_layers:
            console.print(f"\n[green]Modified layers ({len(modified_layers)}):[/green]")
            for layer_name in modified_layers.keys():
                console.print(f"  • {layer_name}")

        console.print("\n[bold]Available layers:[/bold]")
        for i, layer_name in enumerate(available_layers):
            status = (
                " [green](modified)[/green]" if layer_name in modified_layers else ""
            )
            console.print(f"  {i}: {layer_name}{status}")

        # Add options for saving or quitting
        console.print("\n[bold]Options:[/bold]")
        console.print("  [cyan]<number>[/cyan] - Select a layer to edit")
        if modified_layers:
            console.print("  [cyan]s[/cyan] - Save all changes and exit")
        console.print("  [cyan]q[/cyan] - Quit without saving")

        # Get user choice
        valid_choices = [str(i) for i in range(len(available_layers))]
        if modified_layers:
            valid_choices.append("s")
        valid_choices.append("q")

        layer_choice = Prompt.ask(
            "\nSelect a layer or option",
            choices=valid_choices,
            default="0" if not layer else "s" if modified_layers else "q",
        )

        # Handle save option
        if layer_choice == "s" and modified_layers:
            if not in_place:
                new_filename = Prompt.ask("Enter output filename", default=filename)
                if new_filename == filename:
                    if not Confirm.ask(
                        "This will overwrite the original file. Continue?"
                    ):
                        continue
            else:
                new_filename = filename
                if not Confirm.ask(
                    f"This will modify the original file '{filename}'. Continue?"
                ):
                    continue

            # Save all modified layers
            success_count = 0
            try:
                success_count = save_modified_layers(
                    modified_layers, filename, new_filename
                )

            except Exception as e:
                console.print(f"[red]Error saving layer '{layer_name}': {e}[/red]")

            console.print(
                f"\n[green]✓ Saved {success_count}/{len(modified_layers)} layers to {new_filename}[/green]"
            )
            break

        # Handle quit option
        elif layer_choice == "q":
            if modified_layers:
                if not Confirm.ask(
                    f"You have {len(modified_layers)} unsaved layer(s). Quit without saving?"
                ):
                    continue
            console.print("[yellow]Exiting without saving.[/yellow]")
            break

        # Handle layer selection
        else:
            selected_layer = available_layers[int(layer_choice)]

            # Check if we already have modifications for this layer
            gdf = modified_layers.get(selected_layer)

            # Process the layer
            try:
                modified_gdf, go_back = process_layer(filename, selected_layer, gdf)

                if modified_gdf is not None:
                    modified_layers[selected_layer] = modified_gdf

                if not go_back:
                    # User chose to save or quit from within the layer
                    if modified_gdf is not None:
                        # User chose to save
                        if not in_place:
                            new_filename = Prompt.ask(
                                "Enter output filename", default=filename
                            )
                            if new_filename == filename:
                                if not Confirm.ask(
                                    "This will overwrite the original file. Continue?"
                                ):
                                    continue
                        else:
                            new_filename = filename
                            if not Confirm.ask(
                                f"This will modify the original file '{filename}'. Continue?"
                            ):
                                continue

                        # Save all modified layers
                        success_count = 0

                        try:
                            success_count = save_modified_layers(
                                modified_layers, filename, new_filename
                            )

                        except Exception as e:
                            console.print(
                                f"[red]Error saving layer '{layer_name}': {e}[/red]"
                            )

                        console.print(
                            f"\n[green]✓ Saved {success_count}/{len(modified_layers)} layers to {new_filename}[/green]"
                        )
                    break

            except Exception as e:
                console.print(f"[red]Error processing layer: {e}[/red]")
                import traceback

                console.print(traceback.format_exc())


if __name__ == "__main__":
    info()
