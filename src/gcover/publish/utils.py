import re

import pandas as pd

from gcover.utils.console import console

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
    - IN clause

    Args:
        esri_expression: ESRI-style filter string

    Returns:
        Pandas-compatible query string
    """
    expr = esri_expression

    # Replace logical operators (case insensitive)
    expr = re.sub(r"\bAND\b", "&", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bOR\b", "|", expr, flags=re.IGNORECASE)

    # Replace IN (case insensitive)
    expr = re.sub(r"\bIN\b", "in", expr, flags=re.IGNORECASE)

    # Replace IS NOT NULL with .notna()
    # Pattern: FIELD IS NOT NULL → FIELD.notna()
    expr = re.sub(r"(\w+)\s+IS\s+NOT\s+NULL", r"\1.notna()", expr, flags=re.IGNORECASE)

    # Replace IS NULL with .isna()
    # Pattern: FIELD IS NULL → FIELD.isna()
    expr = re.sub(r"(\w+)\s+IS\s+NULL", r"\1.isna()", expr, flags=re.IGNORECASE)

    # Replace single = with == (but not in != or ==)
    # Look for = that's not preceded or followed by =, !, <, >
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

   
    # Test the translator
    esri_expr = "KIND IN (14401001,14401002) AND (PRINTED = 1 OR PRINTED IS NULL)"
    pandas_expr = translate_esri_to_pandas(esri_expr)

    console.print(f"[yellow]ESRI:[/yellow] {esri_expr}")
    console.print(f"[green]Pandas:[/green] {pandas_expr}")
