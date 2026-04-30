import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import geopandas as gpd
from sqlalchemy import create_engine
from loguru import logger
from rich.console import Console

from gcover.publish.style_config import BatchClassificationConfig

# --- Setup ---
console = Console()
ROOT_DIR = Path(__file__).resolve().parent.parent  # Assumes script is in /scripts/
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CONFIG_FILE = ROOT_DIR / "config" / "esri_classifier_denormalized_geocover.yaml"

# Configuration
EXTRACTS = {
    "bulle": "2561560.25,1147552.25,2588327.5,1174937.75",
    "sion": "2582000,1108000,2599000,1130000",
}

LAYERS = [
    "bedrock", "unco_deposits", "fossils", "linear_objects",
    "point_objects", "exploit_polygons", "exploit_points",
    "surfaces", "glacier_hydrology", "surfaces_aux_points"
]


def get_db_engine():
    """Loads env and returns (engine, schema)."""
    load_dotenv(dotenv_path=ROOT_DIR / ".env.local")

    user = os.getenv("GCOVER_POSTGIS_USER")
    pw = os.getenv("GCOVER_POSTGIS_PASSWORD")
    db = os.getenv("GCOVER_POSTGIS_DBNAME")
    schema = os.getenv("GCOVER_POSTGIS_SCHEMA", "geol")

    if not all([user, pw, db]):
        logger.error("Missing DB environment variables in .env.local")
        sys.exit(1)

    url = f"postgresql://{user}:{pw}@localhost:54321/{db}"
    return create_engine(url), schema


def run_extraction(target, extent, engine, schema):
    """Handles the heavy lifting of spatial queries and file saving."""
    main_gpkg = DATA_DIR / f"extract_{target}.gpkg"

    # Cleanup existing file for this target
    if main_gpkg.exists():
        main_gpkg.unlink()

    for layer in LAYERS:
        # Determine schema/table name
        prefix = "" if layer in {"glacier_hydrology", "surfaces_aux_points"} else "geocover_"
        table_name = f"{schema}.{prefix}{layer}"

        # Decide output path (Special case for aux points)
        current_out = DATA_DIR / "surfaces_aux.gpkg" if layer == 'surfaces_aux_points' else main_gpkg

        if current_out.exists() and layer == 'surfaces_aux_points':
            console.print(f"[yellow]Skipping {layer}: {current_out.name} already exists.[/yellow]")
            continue

        sql = f"""
            SELECT *, ST_Intersection(geom, ST_MakeEnvelope({extent}, 2056)) as geom_clipped
            FROM {table_name}
            WHERE ST_Intersects(geom, ST_MakeEnvelope({extent}, 2056))
        """

        try:
            # 1. Load the data
            gdf = gpd.read_postgis(sql, engine, geom_col="geom_clipped")

            if gdf.empty:
                console.print(f"[yellow]⚠ Layer {layer} is empty for this extent.[/yellow]")
                continue

            # 2. Cleanup: Remove the old 'geom' and rename 'geom_clipped' to 'geom'
            # GeoPandas is picky; it's safest to drop the old one first.
            if "geom" in gdf.columns:
                gdf = gdf.drop(columns=["geom"])

            gdf = gdf.rename(columns={"geom_clipped": "geom"})

            # 3. Explicitly set the active geometry column
            gdf = gdf.set_geometry("geom")

            # 4. Set CRS (2056 is LV95)
            gdf.set_crs(2056, allow_override=True, inplace=True)

            gdf.to_file(current_out, layer=layer, driver="GPKG")
            console.print(f"[green]✔[/green] Saved [bold]{layer}[/bold] ({len(gdf)} features)")

        except Exception as e:
            logger.error(f"Failed to extract {layer}: {e}")


N_SAMPLES = 5  # candidate points per WMS layer; test code picks randomly


def build_wms_test_points(gpkg_path: Path) -> None:
    """
    Append a wms_test_points layer to the GPKG: up to N_SAMPLES evenly-spaced
    representative points per active WMS layer (mapfile_name).
    """
    if not CONFIG_FILE.exists():
        logger.warning(f"Config not found, skipping wms_test_points: {CONFIG_FILE}")
        return

    config = BatchClassificationConfig(CONFIG_FILE)

    # Build gpkg_layer → [mapfile_name, ...] for active classifications
    layer_map: dict[str, list[str]] = {}
    for layer_cfg in config.layers:
        names = [
            c.mapfile_name.strip()
            for c in layer_cfg.classifications
            if c.active and c.mapfile_name and c.mapfile_name.strip()
        ]
        if names:
            layer_map.setdefault(layer_cfg.gpkg_layer, []).extend(names)

    rows = []
    for gpkg_layer, mapfile_names in layer_map.items():
        if not gpkg_path.exists():
            continue
        try:
            gdf = gpd.read_file(gpkg_path, layer=gpkg_layer)
        except Exception:
            continue
        if gdf.empty:
            continue

        # Evenly-spaced sample for spatial variety across the extract
        step = max(1, len(gdf) // N_SAMPLES)
        sample_geoms = [
            geom.representative_point()
            for geom in gdf.geometry.iloc[::step].iloc[:N_SAMPLES]
        ]
        for name in mapfile_names:
            for geom in sample_geoms:
                rows.append({"wms_layer": name, "geometry": geom})

    if not rows:
        console.print("[yellow]⚠ No wms_test_points rows generated.[/yellow]")
        return

    result = gpd.GeoDataFrame(rows, crs=2056)
    result.to_file(gpkg_path, layer="wms_test_points", driver="GPKG")
    console.print(f"[green]✔[/green] Saved [bold]wms_test_points[/bold] ({len(result)} rows)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/bold red] python scripts/export_test_data.py <target_name>")
        sys.exit(1)

    target = sys.argv[1].lower()
    if target not in EXTRACTS:
        console.print(f"[red]Error:[/red] Target must be one of: {list(EXTRACTS.keys())}")
        sys.exit(1)

    bbox = EXTRACTS[target]
    engine, schema = get_db_engine()

    console.print(f"=== [magenta]Extracting {target.upper()}[/magenta] ===", style="bold")
    run_extraction(target, bbox, engine, schema)
    build_wms_test_points(DATA_DIR / f"extract_{target}.gpkg")

    # Info for next steps
    profile = os.getenv("MAPSERVER_S3_PROFILE", "default")
    bucket = os.getenv("MAPSERVER_S3_BUCKET", "my-bucket")

    console.print("\n[bold]Next Step (Upload):[/bold]")
    console.print(f"  [dim]aws s3 --profile {profile} cp {DATA_DIR}/extract_{target}.gpkg "
                  f"s3://{bucket}/GEODATA/mapserver-geocover/[/dim]\n")