import re

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
import tempfile
import shutil
import geopandas as gpd
from pathlib import Path
import fiona

console = Console()


def save_layer_preserving_types(
    gdf: gpd.GeoDataFrame, output_path: Path, layer_name: str, is_first_layer: bool
) -> None:
    """
    Save a layer to GPKG while preserving field types.
    Recreates the entire GPKG to avoid schema conflicts.

    Args:
        gdf: GeoDataFrame to save
        output_path: Output GPKG path
        layer_name: Name of the layer
        is_first_layer: True if this is the first layer being written
    """
    if output_path.exists() and not is_first_layer:
        # Get list of existing layers
        existing_layers = fiona.listlayers(str(output_path))

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Track if we've written the first layer to temp file
            first_write = True

            # Copy existing layers to temp file (except the one we're updating)
            for existing_layer in existing_layers:
                if existing_layer != layer_name:
                    gdf_existing = gpd.read_file(
                        output_path, layer=existing_layer, engine="pyogrio"
                    )
                    gdf_existing.to_file(
                        tmp_path,
                        layer=existing_layer,
                        driver="GPKG",
                        engine="pyogrio",
                        mode="w"
                        if first_write
                        else "a",  # ← Fix: first write must be 'w'
                    )
                    first_write = False

            # Add the new/updated layer
            gdf.to_file(
                tmp_path,
                layer=layer_name,
                driver="GPKG",
                engine="pyogrio",
                mode="w"
                if first_write
                else "a",  # ← Fix: handle case where this is the only layer
            )

            # Replace original file
            shutil.move(str(tmp_path), str(output_path))

        except Exception as e:
            console.print(f"[red]✗ Error during save: {e}[/red]")
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    else:
        # First layer or new file - simple save
        gdf.to_file(output_path, layer=layer_name, driver="GPKG", engine="pyogrio")


def generate_font_image(font_symbol_name, font_name, char_index):
    font_size = 100
    img_size = (200, 200)
    image = None

    font_paths = {
        "geofonts1": "/home/marco/.fonts/g/GeoFonts1.ttf",
        "geofonts2": "/home/marco/.fonts/g/GeoFonts2.ttf",
        "default": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    }

    char = chr(char_index)  # chr(index)

    font_path = font_paths.get(font_name)
    if not font_path:
        console.print(f"[red]Font {font_name} not found on system")
        return image

    image = Image.new("RGB", img_size, color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    # Draw character
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = (
        (img_size[0] - text_width) // 2,
        (img_size[1] - text_height) // 2,
    )
    draw.text(position, char, font=font, fill="black")

    # Add index label
    draw.text(
        (10, 10),
        f"{font_symbol_name}: '{char}' ({char_index})",
        font=ImageFont.truetype(font_paths["default"], 12),
        fill="gray",
    )

    return image


# ============================================================================
# ESRI vs PANDAS SYNTAX
# ============================================================================

# ❌ ESRI/SQL syntax (won't work in pandas)
esri_filter = "KIND IN (14401001,14401002) AND (PRINTED = 1 OR PRINTED IS NULL)"

# ✅ Pandas syntax (correct)
pandas_filter = "KIND in (14401001, 14401002) & ((PRINTED == 1) | (PRINTED.isna()))"


# ============================================================================
# AUTOMATIC TRANSLATION FUNCTION
# ============================================================================


def translate_esri_to_pandas(esri_expression):
    """
    Translate ESRI definitionExpression to pandas query syntax.
    Handles:
    - AND/OR → &/|
    - = → ==
    - IS NULL → .isna()
    - IS NOT NULL → .notna()
    - IN clause (with or without parentheses)
    - NOT IN clause

    Args:
        esri_expression: ESRI-style filter string
    Returns:
        Pandas-compatible query string
    """
    expr = esri_expression

    # Replace NOT IN first (before replacing IN alone)
    expr = re.sub(r"\bNOT\s+IN\b", "not in", expr, flags=re.IGNORECASE)

    # Replace IN (case insensitive)
    expr = re.sub(r"\bIN\b", "in", expr, flags=re.IGNORECASE)

    # Handle IN clause without parentheses
    # Pattern: FIELD in value(s) → FIELD in (value(s))
    # Matches values (numbers, strings, commas) until AND/OR or end
    # The (?!\() ensures we don't match if parentheses already exist
    expr = re.sub(
        r"(\w+)\s+in\s+(?!\()([\w\s,\'\"]+?)(?=\s+(?:AND|OR)\b|$)",
        r"\1 in (\2)",
        expr,
        flags=re.IGNORECASE,
    )

    # Replace logical operators
    expr = re.sub(r"\bAND\b", "&", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bOR\b", "|", expr, flags=re.IGNORECASE)

    # Replace IS NOT NULL with .notna()
    expr = re.sub(r"(\w+)\s+IS\s+NOT\s+NULL", r"\1.notna()", expr, flags=re.IGNORECASE)

    # Replace IS NULL with .isna()
    expr = re.sub(r"(\w+)\s+IS\s+NULL", r"\1.isna()", expr, flags=re.IGNORECASE)

    # Replace single = with == (but not in != or ==)
    expr = re.sub(r"(?<![=!<>])=(?![=])", "==", expr)

    # Replace <> with !=
    expr = re.sub(r"<>", "!=", expr)

    return expr


# ============================================================================
# APPLYING TRANSLATED FILTERS
# ============================================================================


def apply_esri_filter(gdf, esri_expression, numeric_columns=None, auto_translate=True):
    """
    Apply ESRI-style filter to GeoDataFrame with automatic translation.

    Args:
        gdf: GeoDataFrame
        esri_expression: ESRI definitionExpression string
        numeric_columns: Columns to ensure are numeric
        auto_translate: If True, translate ESRI syntax to pandas

    Returns:
        Filtered GeoDataFrame
    """
    gdf_work = gdf.copy()

    # Ensure numeric columns are properly typed
    if numeric_columns:
        for col in numeric_columns:
            if col in gdf_work.columns:
                gdf_work[col] = pd.to_numeric(gdf_work[col], errors="coerce")
                gdf_work[col] = gdf_work[col].astype("Int64")

    # Translate ESRI syntax to pandas if needed
    if auto_translate:
        pandas_expression = translate_esri_to_pandas(esri_expression)
        console.print(f"[yellow]Original:[/yellow] {esri_expression}")
        console.print(f"[green]Translated:[/green] {pandas_expression}")
    else:
        pandas_expression = esri_expression

    try:
        gdf_filtered = gdf_work.query(pandas_expression)

        console.print(
            f"[cyan]🔍 Filter result: {len(gdf_filtered):,} / {len(gdf):,} features "
            f"({len(gdf_filtered) / len(gdf) * 100:.1f}%)[/cyan]"
        )

        return gdf_filtered

    except Exception as e:
        console.print(f"[red]❌ Filter failed: {e}[/red]")
        console.print(f"[yellow]   Query: '{pandas_expression}'[/yellow]")

        # Debug info
        if numeric_columns:
            for col in numeric_columns:
                if col in gdf_work.columns:
                    console.print(
                        f"[yellow]   {col}: dtype={gdf_work[col].dtype}, "
                        f"nulls={gdf_work[col].isna().sum()}, "
                        f"sample={gdf_work[col].head(3).tolist()}[/yellow]"
                    )

        return gdf.copy()


if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    # Test the translator
    esri_expr = "KIND IN (14401001,14401002) AND (PRINTED = 1 OR PRINTED IS NULL)"
    pandas_expr = translate_esri_to_pandas(esri_expr)

    console.print(f"[yellow]ESRI:[/yellow] {esri_expr}")
    console.print(f"[green]Pandas:[/green] {pandas_expr}")
