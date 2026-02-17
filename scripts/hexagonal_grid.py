import geopandas as gpd
import numpy as np
from shapely.geometry import Point, MultiPoint

from rich.console import Console
from shapely.geometry import Point
from rich.progress import Progress

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from rich.progress import Progress
from rich import print as rprint
from rich.panel import Panel


import geopandas as gpd
import numpy as np
from shapely.geometry import Point, MultiPoint
from rich.progress import Progress
from rich import print as rprint
from rich.table import Table


def create_hex_multipoints(input_gpkg, layer_name, spacing=80, buffer_dist=30):
    rprint(f"[bold blue]→ Loading layer:[/bold blue] {layer_name}")
    gdf = gpd.read_file(input_gpkg, layer=layer_name)

    target_symbols = [
        "surfaces_gins_sackungsgebiet",
        "surfaces_gins_gebiet_mit_hakenwurf",
        "surfaces_gins_rutschgebiet",
        "surfaces_gins_gebiet_mit_solifluktion"
    ]

    # Filter and keep only necessary columns to keep it light
    filtered_gdf = gdf[gdf['map_symbol'].isin(target_symbols)].copy()

    if filtered_gdf.empty:
        rprint("[bold red]Error:[/bold red] No matching features found.")
        return None

    results = []

    with Progress() as progress:
        task = progress.add_task("[yellow]Generating MultiPoint grids...", total=len(filtered_gdf))

        for index, row in filtered_gdf.iterrows():
            poly_geom = row.geometry
            buffered_poly = poly_geom.buffer(-abs(buffer_dist))

            if buffered_poly.is_empty:
                progress.update(task, advance=1)
                continue

            # Grid logic
            xmin, ymin, xmax, ymax = buffered_poly.bounds
            dx = spacing
            dy = spacing * np.sqrt(3) / 2

            rows = np.arange(ymin, ymax + dy, dy)
            cols = np.arange(xmin, xmax + dx, dx)

            current_feature_points = []

            for i, y in enumerate(rows):
                x_offset = (dx / 2) if i % 2 == 1 else 0
                for x in cols:
                    p = Point(x + x_offset, y)
                    if buffered_poly.contains(p):
                        current_feature_points.append(p)

            if current_feature_points:
                # Convert list of Points to a single MultiPoint geometry
                new_row = row.copy()
                new_row.geometry = MultiPoint(current_feature_points)
                new_row['point_count'] = len(current_feature_points)
                results.append(new_row)

            progress.update(task, advance=1)

    if not results:
        return None

    # Create GeoDataFrame from the list of Series
    output_gdf = gpd.GeoDataFrame(results, crs=gdf.crs)

    # Print Summary Table
    table = Table(title="MultiPoint Generation Results")
    table.add_column("Map Symbol", style="cyan")
    table.add_column("Features", style="green")
    table.add_column("Total Points", style="magenta")

    for symbol in target_symbols:
        sub = output_gdf[output_gdf['map_symbol'] == symbol]
        table.add_row(symbol, str(len(sub)), str(sub['point_count'].sum()))

    rprint(table)
    return output_gdf

# --- Configuration ---
FILE_PATH = "/home/marco/DATA/Derivations/output/test/R16_master_denormalized_classified.gpkg"
LAYER = "surfaces"

if __name__ == "__main__":
    # Update these paths to your actual local files
    final_gdf = create_hex_multipoints(FILE_PATH, LAYER)

    if final_gdf is not None:
        final_gdf.to_file("hex_multipoints_output.gpkg", layer="hex_representation", driver="GPKG")
        rprint("[bold green]File exported successfully.[/bold green]")




