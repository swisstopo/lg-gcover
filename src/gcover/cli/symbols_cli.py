#!/usr/bin/env python3
"""
CLI commands for symbol pattern management.

Integrates with gcover publish commands.

Usage:
    gcover publish symbols inventory -c config.yaml
    gcover publish symbols generate -c patterns_catalog.yaml -o mapserver/patterns/
    gcover publish symbols symbolset -c patterns_catalog.yaml -o mapserver/symbols.sym
    gcover publish symbols preview -c patterns_catalog.yaml
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console()


# =============================================================================
# SYMBOLS COMMAND GROUP
# =============================================================================


@click.group(name="symbols")
def symbols_commands():
    """Commands for managing symbol patterns (inventory, generate, catalog)."""
    pass


# =============================================================================
# INVENTORY COMMAND
# =============================================================================


@symbols_commands.command()
@click.pass_context
@click.option(
    "--config-file", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="YAML configuration file with layer classifications",
)
@click.option(
    "--styles-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Base directory for resolving relative style paths",
)
@click.option(
    "--color-threshold",
    type=float,
    default=5.0,
    help="ΔE CIE2000 threshold for color similarity (default: 5.0)",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output YAML catalog path (default: patterns_catalog.yaml)",
)
def inventory(
    ctx,
    config_file: Path,
    styles_dir: Optional[Path],
    color_threshold: float,
    output: Optional[Path],
):
    """
    Inventory all unique patterns from ESRI style files.
    
    Scans all .lyrx files referenced in the config and extracts:
    - Character marker patterns (polka dots, geological symbols)
    - Hatch fill patterns
    - Color information with ΔE CIE2000 deduplication
    
    Example:
        gcover publish symbols inventory -c config/esri_classifier.yaml
    """
    from gcover.publish.style_config import BatchClassificationConfig
    from gcover.publish.esri_classification_extractor import extract_lyrx_complete
    from gcover.publish.symbol_catalog import PatternCatalog, inventory_patterns
    
    console.print(f"\n[bold blue]🔍 Pattern Inventory[/bold blue]\n")
    console.print(f"Config: {config_file}")
    console.print(f"Color threshold (ΔE): {color_threshold}")
    
    env = ctx.obj.get('environment', 'development')
    
    # Load configuration
    config = BatchClassificationConfig(config_file, styles_dir, env=env)
    
    if not styles_dir:
        styles_dir = config_file.parent
    
    # Collect all style files
    style_files = set()
    for layer_config in config.layers:
        for class_config in layer_config.classifications:
            style_files.add(class_config.style_file)
    
    console.print(f"Found {len(style_files)} style files\n")
    
    # Extract classifications from all style files
    all_classifications = []
    
    for style_file in style_files:
        lyrx_path = styles_dir / style_file.name if not style_file.is_absolute() else style_file
        
        if not lyrx_path.exists():
            console.print(f"[yellow]⚠ Style file not found: {lyrx_path}[/yellow]")
            continue
        
        console.print(f"Processing {lyrx_path.name}...")
        
        classifications = extract_lyrx_complete(lyrx_path, display=False)
        all_classifications.extend(classifications)
    
    console.print(f"\nExtracted {len(all_classifications)} classifications")
    
    # Build catalog
    catalog = inventory_patterns(all_classifications, color_threshold=color_threshold)
    
    # Display inventory
    catalog.display_inventory()
    
    # Export catalog
    if output:
        catalog.export_to_yaml(output)
    else:
        default_output = config_file.parent / "patterns_catalog.yaml"
        catalog.export_to_yaml(default_output)


# =============================================================================
# GENERATE COMMAND
# =============================================================================


@symbols_commands.command()
@click.pass_context
@click.option(
    "--catalog", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Pattern catalog YAML file",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    default="mapserver/patterns",
    help="Output directory for PNG tiles (default: mapserver/patterns)",
)
@click.option(
    "--font-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing font files",
)
@click.option(
    "--tile-size",
    type=int,
    default=24,
    help="Base tile size in pixels (default: 24, recommended: 16-32)",
)
@click.option(
    "--scale",
    type=float,
    default=1.33,
    help="Scale factor for points to pixels (default: 1.33)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Regenerate existing PNG files",
)
def generate(
    ctx,
    catalog: Path,
    output_dir: Path,
    font_dir: Optional[Path],
    tile_size: int,
    scale: float,
    force: bool,
):
    """
    Generate PNG tiles for pattern fills.
    
    Reads a pattern catalog and generates PNG tiles that MapServer
    can use as PIXMAP symbols for polygon fills.
    
    Tile sizes of 16-32 pixels offer the best balance of quality
    and performance for MapServer.
    
    Example:
        gcover publish symbols generate -c patterns_catalog.yaml -o mapserver/patterns/
    """
    import yaml
    from gcover.publish.symbol_catalog import PatternTileGenerator
    
    console.print(f"\n[bold blue]🎨 Generate Pattern Tiles[/bold blue]\n")
    console.print(f"Catalog: {catalog}")
    console.print(f"Output: {output_dir}")
    console.print(f"Tile size: {tile_size}px")
    console.print(f"Scale: {scale}")
    
    # Load catalog
    with open(catalog, 'r') as f:
        catalog_data = yaml.safe_load(f)
    
    patterns = catalog_data.get('patterns', {})
    console.print(f"Found {len(patterns)} patterns in catalog\n")
    
    # Setup generator
    generator = PatternTileGenerator()
    
    if font_dir:
        # Update font paths
        for font_file in font_dir.glob("*.ttf"):
            font_name = font_file.stem.lower()
            generator.font_paths[font_name] = str(font_file)
            # Also add common aliases
            if "geofont" in font_name:
                generator.font_paths["GeoFonts 1"] = str(font_file)
                generator.font_paths["geofonts1"] = str(font_file)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_count = 0
    skipped_count = 0
    error_count = 0
    
    for key, pattern_data in patterns.items():
        pattern_type = pattern_data.get('type', '')
        name = pattern_data.get('name', key)
        
        # Skip native hatch patterns
        mapserver_info = pattern_data.get('mapserver', {})
        if mapserver_info.get('native', False):
            console.print(f"[dim]Skipping {name} (native MapServer symbol)[/dim]")
            skipped_count += 1
            continue
        
        # Check if PNG already exists
        png_path = output_dir / f"{name}.png"
        if png_path.exists() and not force:
            console.print(f"[dim]Skipping {name} (already exists)[/dim]")
            skipped_count += 1
            continue
        
        # Generate tile
        try:
            char_info = pattern_data.get('character', {})
            color_info = pattern_data.get('color', {})
            
            if not char_info:
                console.print(f"[yellow]⚠ No character info for {name}[/yellow]")
                continue
            
            # Get step size from catalog or use tile_size
            step = char_info.get('step', [tile_size / scale, tile_size / scale])
            
            result = _generate_single_tile(
                generator=generator,
                name=name,
                font_family=char_info.get('font_family', 'GeoFonts 1'),
                char_index=char_info.get('char_index', 51),
                size=char_info.get('size', 3),
                step_x=step[0] if step else tile_size / scale,
                step_y=step[1] if step else tile_size / scale,
                offset=char_info.get('offset', [0, 0]),
                color=color_info.get('rgba', [0, 0, 0, 255]),
                output_dir=output_dir,
                scale=scale,
                min_tile_size=tile_size,
            )
            
            if result:
                console.print(f"[green]✓ Generated {result.name}[/green]")
                generated_count += 1
            
        except Exception as e:
            console.print(f"[red]✗ Failed to generate {name}: {e}[/red]")
            error_count += 1
    
    console.print(f"\n[bold green]✅ Generated {generated_count} tiles[/bold green]")
    if skipped_count:
        console.print(f"[dim]Skipped {skipped_count} patterns[/dim]")
    if error_count:
        console.print(f"[yellow]Errors: {error_count}[/yellow]")


def _generate_single_tile(
    generator,
    name: str,
    font_family: str,
    char_index: int,
    size: float,
    step_x: float,
    step_y: float,
    offset: list,
    color: list,
    output_dir: Path,
    scale: float,
    min_tile_size: int = 16,
) -> Optional[Path]:
    """Generate a single PNG tile."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("PIL/Pillow not installed. Run: pip install Pillow")
    
    # Calculate tile size from step, with minimum
    tile_width = max(int(step_x * scale), min_tile_size)
    tile_height = max(int(step_y * scale), min_tile_size)
    
    # Create transparent image
    img = Image.new('RGBA', (tile_width, tile_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Load font
    font_path = generator.font_paths.get(font_family)
    
    if not font_path:
        # Try common variations
        for key in generator.font_paths:
            if font_family.lower().replace(' ', '') in key.lower().replace(' ', ''):
                font_path = generator.font_paths[key]
                break
    
    if not font_path or not Path(font_path).exists():
        raise FileNotFoundError(f"Font not found: {font_family}")
    
    # Calculate font size
    font_size = max(int(size * scale), 4)
    
    font = ImageFont.truetype(font_path, font_size)
    
    # Get character
    char = chr(char_index) if char_index else '•'
    
    # Get color
    if len(color) >= 4:
        r, g, b, a = color[:4]
    else:
        r, g, b = color[:3]
        a = 255
    
    # Calculate position (center of tile, with offset)
    offset_x = int((offset[0] or 0) * scale) if offset else 0
    offset_y = int((offset[1] or 0) * scale) if offset else 0
    
    x = tile_width // 2 + offset_x
    y = tile_height // 2 + offset_y
    
    # Draw character
    draw.text((x, y), char, font=font, fill=(r, g, b, a), anchor='mm')
    
    # Save
    output_path = output_dir / f"{name}.png"
    img.save(output_path, 'PNG')
    
    return output_path


# =============================================================================
# SYMBOLSET COMMAND
# =============================================================================


@symbols_commands.command()
@click.pass_context
@click.option(
    "--catalog", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Pattern catalog YAML file",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output MapServer symbol file (.sym)",
)
@click.option(
    "--patterns-dir",
    type=click.Path(path_type=Path),
    default="patterns",
    help="Relative path to patterns directory (default: patterns)",
)
def symbolset(
    ctx,
    catalog: Path,
    output: Optional[Path],
    patterns_dir: Path,
):
    """
    Generate MapServer SYMBOLSET file from catalog.
    
    Creates symbol definitions for both:
    - Native hatch patterns (TYPE HATCH)
    - PNG pattern tiles (TYPE PIXMAP)
    
    Example:
        gcover publish symbols symbolset -c patterns_catalog.yaml -o mapserver/symbols.sym
    """
    import yaml
    
    console.print(f"\n[bold blue]📝 Generate MapServer SYMBOLSET[/bold blue]\n")
    
    # Load catalog
    with open(catalog, 'r') as f:
        catalog_data = yaml.safe_load(f)
    
    patterns = catalog_data.get('patterns', {})
    console.print(f"Found {len(patterns)} patterns\n")
    
    lines = [
        "SYMBOLSET",
        "",
        "  # Generated from pattern catalog",
        "  # Do not edit manually - regenerate with: gcover publish symbols symbolset",
        "",
        "  # ============================================================",
        "  # Native Hatch Symbol (shared by all hatch patterns)",
        "  # ============================================================",
        "",
        "  SYMBOL",
        '    NAME "hatchsymbol"',
        "    TYPE HATCH",
        "  END",
        "",
        "  # ============================================================",
        "  # Pattern Fill Symbols (PNG tiles)",
        "  # ============================================================",
        "",
    ]
    
    pixmap_count = 0
    
    for key, pattern_data in sorted(patterns.items()):
        mapserver_info = pattern_data.get('mapserver', {})
        
        # Skip native symbols (already have hatchsymbol)
        if mapserver_info.get('native', False):
            continue
        
        name = pattern_data.get('name', key)
        png_file = mapserver_info.get('png_file', f"{patterns_dir}/{name}.png")
        
        lines.extend([
            "  SYMBOL",
            f'    NAME "{name}"',
            "    TYPE PIXMAP",
            f'    IMAGE "{png_file}"',
            "  END",
            "",
        ])
        pixmap_count += 1
    
    lines.append("END # SYMBOLSET")
    
    content = "\n".join(lines)
    
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
        console.print(f"[green]✓ Generated {output}[/green]")
    else:
        console.print(content)
    
    console.print(f"\n[dim]Symbols: 1 hatch + {pixmap_count} pixmap[/dim]")


# =============================================================================
# PREVIEW COMMAND
# =============================================================================


@symbols_commands.command()
@click.pass_context
@click.option(
    "--catalog", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Pattern catalog YAML file",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output HTML file for preview",
)
def preview(
    ctx,
    catalog: Path,
    output: Optional[Path],
):
    """
    Generate HTML preview of pattern catalog.
    
    Creates a visual reference showing all patterns with their
    properties and usage information.
    
    Example:
        gcover publish symbols preview -c patterns_catalog.yaml -o patterns_preview.html
    """
    import yaml
    
    console.print(f"\n[bold blue]👁️ Generate Pattern Preview[/bold blue]\n")
    
    # Load catalog
    with open(catalog, 'r') as f:
        catalog_data = yaml.safe_load(f)
    
    metadata = catalog_data.get('metadata', {})
    patterns = catalog_data.get('patterns', {})
    
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>GeoCover Pattern Catalog</title>",
        "  <style>",
        "    body { font-family: -apple-system, sans-serif; margin: 2em; background: #f5f5f5; }",
        "    h1 { color: #333; }",
        "    .stats { background: #fff; padding: 1em; border-radius: 8px; margin-bottom: 2em; }",
        "    .patterns { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 1em; }",
        "    .pattern { background: #fff; border-radius: 8px; padding: 1em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "    .pattern h3 { margin-top: 0; color: #2563eb; }",
        "    .pattern-type { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }",
        "    .type-polka_dot { background: #dbeafe; color: #1e40af; }",
        "    .type-character { background: #fef3c7; color: #92400e; }",
        "    .type-hatch { background: #d1fae5; color: #065f46; }",
        "    .color-swatch { display: inline-block; width: 24px; height: 24px; border-radius: 4px; border: 1px solid #ccc; vertical-align: middle; margin-right: 8px; }",
        "    .props { font-size: 0.9em; color: #666; }",
        "    .props dt { font-weight: bold; }",
        "    .used-by { font-size: 0.8em; color: #888; max-height: 100px; overflow-y: auto; }",
        "    .used-by li { margin: 2px 0; }",
        "    .preview-img { max-width: 48px; max-height: 48px; border: 1px solid #ddd; background: repeating-linear-gradient(45deg, #f0f0f0, #f0f0f0 5px, #fff 5px, #fff 10px); }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>🎨 GeoCover Pattern Catalog</h1>",
        "",
        "  <div class='stats'>",
        f"    <strong>Total Patterns:</strong> {metadata.get('unique_patterns', len(patterns))} unique",
        f"    (from {metadata.get('total_instances', '?')} instances) |",
        f"    <strong>Color Threshold:</strong> ΔE = {metadata.get('color_threshold_delta_e', 5.0)}",
        "  </div>",
        "",
        "  <div class='patterns'>",
    ]
    
    for key, pattern_data in sorted(patterns.items()):
        name = pattern_data.get('name', key)
        ptype = pattern_data.get('type', 'unknown')
        color_info = pattern_data.get('color', {})
        char_info = pattern_data.get('character', {})
        hatch_info = pattern_data.get('hatch', {})
        used_by = pattern_data.get('used_by', [])
        mapserver_info = pattern_data.get('mapserver', {})
        
        color_hex = color_info.get('hex', '#808080')
        color_name = color_info.get('name', 'gray')
        
        html_parts.append(f"    <div class='pattern'>")
        html_parts.append(f"      <h3>{name}</h3>")
        html_parts.append(f"      <span class='pattern-type type-{ptype}'>{ptype}</span>")
        html_parts.append(f"      <span class='color-swatch' style='background: {color_hex};'></span>")
        html_parts.append(f"      <span>{color_name} ({color_hex})</span>")
        
        # Show PNG preview if available
        png_file = mapserver_info.get('png_file')
        if png_file and not mapserver_info.get('native'):
            html_parts.append(f"      <img class='preview-img' src='{png_file}' alt='{name}' onerror=\"this.style.display='none'\">")
        
        html_parts.append(f"      <dl class='props'>")
        
        if char_info:
            html_parts.append(f"        <dt>Character</dt>")
            html_parts.append(f"        <dd>{char_info.get('font_family', '?')} #{char_info.get('char_index', '?')}</dd>")
            html_parts.append(f"        <dt>Size / Step</dt>")
            html_parts.append(f"        <dd>{char_info.get('size', '?')} / {char_info.get('step', [])}</dd>")
        
        if hatch_info:
            html_parts.append(f"        <dt>Rotation</dt>")
            html_parts.append(f"        <dd>{hatch_info.get('rotation', 0)}°</dd>")
            html_parts.append(f"        <dt>Separation</dt>")
            html_parts.append(f"        <dd>{hatch_info.get('separation', 0)}</dd>")
        
        html_parts.append(f"        <dt>MapServer</dt>")
        if mapserver_info.get('native'):
            html_parts.append(f"        <dd>Native ({mapserver_info.get('symbol', 'hatchsymbol')})</dd>")
        else:
            html_parts.append(f"        <dd>{mapserver_info.get('png_file', 'N/A')}</dd>")
        
        html_parts.append(f"      </dl>")
        
        if used_by:
            html_parts.append(f"      <div class='used-by'>")
            html_parts.append(f"        <strong>Used by ({len(used_by)}):</strong>")
            html_parts.append(f"        <ul>")
            for use in used_by[:10]:  # Limit display
                html_parts.append(f"          <li>{use}</li>")
            if len(used_by) > 10:
                html_parts.append(f"          <li>... and {len(used_by) - 10} more</li>")
            html_parts.append(f"        </ul>")
            html_parts.append(f"      </div>")
        
        html_parts.append(f"    </div>")
    
    html_parts.extend([
        "  </div>",
        "</body>",
        "</html>",
    ])
    
    content = "\n".join(html_parts)
    
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
        console.print(f"[green]✓ Generated {output}[/green]")
    else:
        # Default output
        default_output = catalog.with_suffix('.html')
        default_output.write_text(content)
        console.print(f"[green]✓ Generated {default_output}[/green]")
    
    console.print(f"[dim]Patterns documented: {len(patterns)}[/dim]")


# =============================================================================
# INTEGRATION SNIPPET
# =============================================================================

"""
To integrate with publish_cmd.py, add this to the imports:

    from gcover.publish.symbols_cli import symbols_commands

And add this after the publish_commands group definition:

    # Add symbols subcommands
    publish_commands.add_command(symbols_commands)

This will make the following commands available:
    gcover publish symbols inventory
    gcover publish symbols generate
    gcover publish symbols symbolset  
    gcover publish symbols preview
"""


# For testing standalone
if __name__ == "__main__":
    symbols_commands()
