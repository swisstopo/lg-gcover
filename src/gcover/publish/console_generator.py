#!/usr/bin/env python3
"""
Console Style Inspector - CLI command to display style information

Prints extracted style information to console in readable format.
Useful for inspecting what was extracted from ESRI style files.
"""

from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text

from gcover.publish.esri_classification_extractor import (
    extract_lyrx_complete,
    IdentifierMode,
)
from gcover.publish.symbol_utils import extract_polygon_symbol_layers

console = Console()


class ConsoleStyleGenerator:
    """
    Generate formatted console output for style information.

    Shows what was extracted from ESRI classification files including:
    - Classification metadata
    - Class definitions with expressions
    - Symbol information (colors, fonts, patterns)
    - Multi-layer symbol details for polygons
    - Generated map symbol IDs
    """

    def __init__(self, symbol_prefix: Optional[str] = None):
        """
        Initialize console generator.
        
        Args:
            symbol_prefix: Optional prefix for generated symbol IDs
        """
        self.symbol_prefix = symbol_prefix
        self.style_summary = {
            "total_classes": 0,
            "visible_classes": 0,
            "fonts_used": set(),
            "colors_used": set(),
            "identifiers": [],
        }

    def display_classification(
        self,
        classification,
        detailed: bool = False,
        show_identifiers: bool = True,
    ):
        """
        Display a single classification to console.

        Args:
            classification: LayerClassification object
            detailed: Show detailed symbol information
            show_identifiers: Show generated map symbol IDs
        """
        # Header
        title = f"[bold cyan]{classification.layer_name or 'Unnamed Layer'}[/bold cyan]"
        console.print(Panel(title, expand=False))

        # Metadata table
        meta_table = Table(show_header=False, box=None, padding=(0, 2))
        meta_table.add_column("Property", style="yellow")
        meta_table.add_column("Value", style="white")

        meta_table.add_row("Renderer Type", classification.renderer_type)
        meta_table.add_row("Total Classes", str(len(classification.classes)))
        meta_table.add_row(
            "Visible Classes", str(sum(1 for c in classification.classes if c.visible))
        )

        if classification.fields:
            field_names = ", ".join(f.name for f in classification.fields)
            meta_table.add_row("Classification Fields", field_names)

        if self.symbol_prefix:
            meta_table.add_row("Symbol Prefix", self.symbol_prefix)

        console.print(meta_table)
        console.print()

        # Classes table (compact view with identifiers)
        if show_identifiers:
            self._display_classes_table(classification)
        else:
            # Original detailed view
            console.print("[bold]Classes:[/bold]")
            console.print()

            for idx, class_obj in enumerate(classification.classes, 1):
                self._display_class(class_obj, idx, classification.fields, detailed)
                console.print()

        # Summary
        self._display_summary()

    def _display_classes_table(self, classification):
        """Display classes in a compact table format with identifiers."""
        table = Table(
            title="Classification Classes",
            show_header=True,
            header_style="bold yellow",
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Label", style="cyan", width=45, no_wrap=False)
        table.add_column("Map Symbol ID", style="green", width=35)
        table.add_column("Field Values", style="white", width=25)
        table.add_column("Vis", style="yellow", width=3, justify="center")

        for idx, class_obj in enumerate(classification.classes, 1):
            # Get identifier
            identifier_str = ""
            full_symbol_id = ""
            
            if hasattr(class_obj, "identifier") and class_obj.identifier:
                try:
                    key = class_obj.identifier.to_key()
                    # Extract just the identifier value (last part after ::)
                    identifier_str = key.split("::")[-1] if "::" in key else key
                    
                    # Build full symbol ID with prefix
                    if self.symbol_prefix:
                        full_symbol_id = f"{self.symbol_prefix}_{identifier_str}"
                    else:
                        full_symbol_id = identifier_str
                        
                    self.style_summary["identifiers"].append(full_symbol_id)
                except Exception:
                    identifier_str = "?"
                    full_symbol_id = "?"

            # Field values
            values_str = ""
            if class_obj.field_values:
                for fv in class_obj.field_values[:2]:  # Show first 2 value sets
                    values_str += " | ".join(str(v)[:10] for v in fv) + "\n"
                values_str = values_str.strip()
                if len(class_obj.field_values) > 2:
                    values_str += f"\n(+{len(class_obj.field_values) - 2} more)"

            # Visibility
            visibility = "✓" if class_obj.visible else "✗"

            # Update stats
            if class_obj.visible:
                self.style_summary["total_classes"] += 1
                self.style_summary["visible_classes"] += 1

            table.add_row(
                str(idx),
                class_obj.label[:45] if class_obj.label else "",
                full_symbol_id,
                values_str[:25] if values_str else "",
                visibility,
            )

        console.print(table)

    def _display_class(self, class_obj, index: int, fields: List, detailed: bool):
        """Display a single classification class (detailed view)."""
        # Class header
        visibility = "✓" if class_obj.visible else "✗"
        header = f"[bold]{visibility} Class {index}:[/bold] {class_obj.label}"
        console.print(header)

        if not class_obj.visible:
            console.print("  [dim]Not visible[/dim]")
            return

        self.style_summary["total_classes"] += 1
        self.style_summary["visible_classes"] += 1

        # Show identifier / map symbol ID
        if hasattr(class_obj, "identifier") and class_obj.identifier:
            try:
                key = class_obj.identifier.to_key()
                identifier_value = key.split("::")[-1] if "::" in key else key
                
                if self.symbol_prefix:
                    full_id = f"{self.symbol_prefix}_{identifier_value}"
                else:
                    full_id = identifier_value
                    
                console.print(f"  [green]Map Symbol ID:[/green] [bold]{full_id}[/bold]")
                self.style_summary["identifiers"].append(full_id)
            except Exception as e:
                console.print(f"  [red]Identifier error: {e}[/red]")

        # Field values (expressions)
        if class_obj.field_values and fields:
            console.print("  [yellow]Expression:[/yellow]")
            field_names = [f.name for f in fields]

            for fv_idx, field_values in enumerate(class_obj.field_values, 1):
                if len(class_obj.field_values) > 1:
                    console.print(f"    [dim]Condition {fv_idx}:[/dim]")

                for field_name, value in zip(field_names, field_values):
                    if value == "<Null>":
                        console.print(f"      {field_name} = [red]NULL[/red]")
                    else:
                        console.print(f"      {field_name} = [green]{value}[/green]")

        # Symbol information
        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            if detailed:
                self._display_symbol_detailed(class_obj.symbol_info)
            else:
                self._display_symbol_summary(class_obj.symbol_info)

    def _display_symbol_summary(self, symbol_info):
        """Display brief symbol information."""
        console.print("  [yellow]Symbol:[/yellow]")

        # Color
        if hasattr(symbol_info, "color") and symbol_info.color:
            color = symbol_info.color
            if hasattr(color, "r"):
                rgb = f"RGB({color.r}, {color.g}, {color.b})"
                self.style_summary["colors_used"].add((color.r, color.g, color.b))
                console.print(f"    Color: [bold]{rgb}[/bold]")

        # Font marker
        if hasattr(symbol_info, "font_family") and symbol_info.font_family:
            font = symbol_info.font_family
            char = (
                symbol_info.character_index
                if hasattr(symbol_info, "character_index")
                else "?"
            )
            self.style_summary["fonts_used"].add(font)
            console.print(f"    Font: [bold]{font}[/bold] (char: {char})")

        # Size
        if hasattr(symbol_info, "size") and symbol_info.size:
            console.print(f"    Size: {symbol_info.size} pt")

        # Width (for lines)
        if hasattr(symbol_info, "width") and symbol_info.width:
            console.print(f"    Width: {symbol_info.width} pt")

    def _display_symbol_detailed(self, symbol_info):
        """Display detailed symbol information including layers."""
        console.print("  [yellow]Symbol Details:[/yellow]")

        # Check for complex polygon symbol
        if hasattr(symbol_info, "raw_symbol") and symbol_info.raw_symbol:
            layers_info = extract_polygon_symbol_layers(symbol_info.raw_symbol)

            if layers_info:
                # Create tree structure
                tree = Tree("[bold]Symbol Layers[/bold]")

                # Fill layers
                if layers_info.fills:
                    fills_branch = tree.add("[cyan]Fill Layers[/cyan]")
                    for fill in layers_info.fills:
                        if fill["type"] == "solid":
                            r, g, b, a = fill["color"]
                            fills_branch.add(
                                f"Solid Fill: RGB({r}, {g}, {b}) Alpha={a}"
                            )
                        else:
                            fills_branch.add(f"Fill: {fill['type']}")

                # Character markers (patterns)
                if layers_info.character_markers:
                    markers_branch = tree.add("[magenta]Pattern Fills[/magenta]")
                    for marker in layers_info.character_markers:
                        r, g, b, a = marker.color
                        self.style_summary["fonts_used"].add(marker.font_family)

                        marker_text = (
                            f"{marker.font_family} char #{marker.character_index} "
                            f"RGB({r}, {g}, {b}) "
                            f"size={marker.size:.1f}pt "
                            f"spacing=({marker.step_x:.1f}, {marker.step_y:.1f})"
                        )
                        markers_branch.add(marker_text)

                # Outline
                if layers_info.outline:
                    outline_branch = tree.add("[yellow]Outline[/yellow]")
                    outline = layers_info.outline
                    r, g, b, a = outline["color"]
                    outline_text = (
                        f"Color: RGB({r}, {g}, {b}) "
                        f"Width: {outline['width']:.2f}pt "
                        f"Style: {outline['line_style']['type']}"
                    )
                    outline_branch.add(outline_text)

                console.print(tree)
                return

        # Fallback to simple display
        self._display_symbol_summary(symbol_info)

    def _display_summary(self):
        """Display overall summary."""
        console.print()
        console.print("[bold cyan]═" * 50 + "[/bold cyan]")
        console.print("[bold]Summary:[/bold]")
        console.print(f"  Total classes: {self.style_summary['total_classes']}")
        console.print(f"  Visible: {self.style_summary['visible_classes']}")

        if self.style_summary["fonts_used"]:
            console.print(f"  Fonts used: {len(self.style_summary['fonts_used'])}")
            for font in sorted(self.style_summary["fonts_used"]):
                console.print(f"    • {font}")

        if self.style_summary["colors_used"]:
            console.print(f"  Unique colors: {len(self.style_summary['colors_used'])}")

        if self.style_summary["identifiers"]:
            console.print(f"  Generated identifiers: {len(self.style_summary['identifiers'])}")
            # Show sample identifiers
            sample_ids = self.style_summary["identifiers"][:5]
            for id_str in sample_ids:
                console.print(f"    • [green]{id_str}[/green]")
            if len(self.style_summary["identifiers"]) > 5:
                console.print(f"    [dim]... and {len(self.style_summary['identifiers']) - 5} more[/dim]")


def inspect_styles_main(
    style_files: tuple,
    detailed: bool,
    config_file: Optional[Path],
    symbol_prefix: Optional[str] = None,
    identifier_mode: str = "label",
    identifier_field: Optional[str] = None,
    head: Optional[int] = None,
    verbose: Optional[bool]= False,
):
    """
    Core logic for inspecting ESRI style files.

    This is the main function that can be called directly or via CLI.

    Args:
        style_files: Tuple of Path objects to style files
        detailed: Whether to show detailed symbol information
        config_file: Optional path to YAML configuration file
        symbol_prefix: Optional prefix for generated symbol IDs
        identifier_mode: How to generate identifiers ('label', 'index', 'field')
        identifier_field: Field name for 'field' mode
        head: Limit number of classes displayed
    """

    # Load style files
    if config_file:
        console.print(f"[cyan]Loading style files from {config_file}[/cyan]")
        # TODO: Load from config (similar to other commands)
        console.print("[yellow]Config loading not implemented yet[/yellow]")
        return

    if not style_files:
        console.print("[red]Error: No style files specified[/red]")
        console.print("Use: gcover publish inspect <style_files> or --config-file")
        raise click.Abort()

    # Convert string mode to enum for default
    mode_enum = IdentifierMode(identifier_mode.lower())
    
    console.print(f"[dim]Identifier mode: {mode_enum.value}[/dim]")
    if symbol_prefix:
        console.print(f"[dim]Symbol prefix: {symbol_prefix}[/dim]")
    if identifier_field:
        console.print(f"[dim]Identifier field: {identifier_field}[/dim]")

    # Process each style file
    generator = ConsoleStyleGenerator(symbol_prefix=symbol_prefix)

    for style_file in style_files:
        console.print()
        console.print(f"[bold blue]{'═' * 60}[/bold blue]")
        console.print(f"[bold blue]File:[/bold blue] {style_file}")
        console.print(f"[bold blue]{'═' * 60}[/bold blue]")
        console.print()

        # Extract classifications
        try:
            classifications = extract_lyrx_complete(
                style_file,
                display=False,
                default_identifier_mode=mode_enum,
                default_identifier_field=identifier_field,
            )

            if not classifications:
                console.print("[yellow]⚠ No classifications found[/yellow]")
                continue

            # Limit classes if head is specified
            if head:
                for classification in classifications:
                    classification.classes = classification.classes[:head]

            # Display each classification
            for classification in classifications:
                generator.display_classification(
                    classification,
                    detailed=detailed,
                    show_identifiers=True,
                )
                console.print()

        except Exception as e:
            console.print(f"[red]✗ Error processing {style_file.name}: {e}[/red]")
            if detailed:
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")

    console.print(f"[bold green]✓ Processed {len(style_files)} file(s)[/bold green]")


@click.command()
@click.argument("style_files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    help="Show detailed symbol information including layers",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Load style files from YAML configuration",
)
@click.option(
    "--symbol-prefix",
    "-p",
    type=str,
    default=None,
    help="Prefix for generated symbol IDs (e.g., 'unco_litho')",
)
@click.option(
    "--identifier-mode",
    "-m",
    type=click.Choice(["label", "index", "field"]),
    default="label",
    help="How to generate identifiers: label (default), index, or field",
)
@click.option(
    "--identifier-field",
    "-f",
    type=str,
    default=None,
    help="Field name for 'field' identifier mode",
)
@click.option(
    "--head",
    "-n",
    type=int,
    default=None,
    help="Limit number of classes to display",
)
def inspect_styles_cli(
    style_files: tuple,
    detailed: bool,
    config_file: Optional[Path],
    symbol_prefix: Optional[str],
    identifier_mode: str,
    identifier_field: Optional[str],
    head: Optional[int],
):
    """
    Inspect and display ESRI style file contents.

    Shows extracted classification information in a readable console format.
    Useful for debugging and understanding what was extracted from .lyrx files.

    \b
    Identifier modes:
      - label: Use slugified label text (default, most stable)
      - index: Use sequential index (legacy, unstable)
      - field: Use value from specified field

    Examples:

    \b
    # Inspect single style file with default label-based identifiers
    gcover publish inspect styles/Bedrock.lyrx

    \b
    # Inspect with symbol prefix to see full map_symbol values
    gcover publish inspect styles/Unco_Litho.lyrx -p unco_litho

    \b
    # Compare label vs index identifiers
    gcover publish inspect styles/Surfaces.lyrx -p surfaces -m label
    gcover publish inspect styles/Surfaces.lyrx -p surfaces -m index

    \b
    # Inspect multiple files with details
    gcover publish inspect styles/*.lyrx --detailed

    \b
    # Show only first 10 classes
    gcover publish inspect styles/Bedrock.lyrx --head 10

    \b
    # Load from configuration
    gcover publish inspect --config-file config/styles.yaml --detailed
    """
    inspect_styles_main(
        style_files,
        detailed,
        config_file,
        symbol_prefix=symbol_prefix,
        identifier_mode=identifier_mode,
        identifier_field=identifier_field,
        head=head,
    )


if __name__ == "__main__":
    inspect_styles_cli()


# ============================================================================
# Integration with publish_cmd.py
# ============================================================================
# To add this command to your publish CLI group, add this to publish_cmd.py:
#
# from gcover.publish.console_generator import inspect_styles_main
#
# @publish_commands.command(name="inspect")
# @click.pass_context
# @click.argument("style_files", nargs=-1, type=click.Path(exists=True, path_type=Path))
# @click.option("--detailed", "-d", is_flag=True,
#               help="Show detailed symbol information including layers")
# @click.option("--config-file", "-c", type=click.Path(exists=True, path_type=Path),
#               help="Load style files from YAML configuration")
# @click.option("--symbol-prefix", "-p", type=str, default=None,
#               help="Prefix for generated symbol IDs")
# @click.option("--identifier-mode", "-m", type=click.Choice(["label", "index", "field"]),
#               default="label", help="How to generate identifiers")
# @click.option("--head", "-n", type=int, default=None,
#               help="Limit number of classes to display")
# def inspect_styles_cmd(ctx, style_files: tuple, detailed: bool, config_file: Optional[Path],
#                        symbol_prefix: Optional[str], identifier_mode: str, head: Optional[int]):
#     """Inspect and display ESRI style file contents."""
#     inspect_styles_main(style_files, detailed, config_file,
#                         symbol_prefix=symbol_prefix, identifier_mode=identifier_mode, head=head)
