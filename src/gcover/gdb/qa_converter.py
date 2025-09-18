#!/usr/bin/env python3
"""
FileGDB conversion and statistics module for lg-gcover.

Converts ESRI FileGDB verification results to web formats and generates statistics.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import warnings


import duckdb
import fiona
import geopandas as gpd
import pandas as pd
from botocore.exceptions import ClientError
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from gcover.config import load_config
from gcover.gdb.storage import S3Uploader

console = Console()


@dataclass
class LayerStats:
    """Statistics for a single layer."""

    layer_name: str
    feature_count: int
    test_names: Dict[str, int]  # TestName -> count
    issue_types: Dict[str, int]  # IssueType -> count
    test_issue_matrix: Dict[Tuple[str, str], int]  # (TestName, IssueType) -> count


@dataclass
class GDBSummary:
    """Summary statistics for an entire GDB."""

    gdb_path: str
    timestamp: datetime
    rc_version: str  # e.g., "2030-12-31"
    verification_type: str  # e.g., "Topology", "TechnicalQualityAssurance"
    total_features: int
    layers: Dict[str, LayerStats]


class FileGDBConverter:
    """Converts FileGDB verification results to web formats and generates statistics."""

    # Layers to process (excluding multipatches)
    SPATIAL_LAYERS = [
        "IssuePolygons",
        "IssueLines",
        "IssuePoints",
        "IssueRows",  # Non-spatial but contains metadata
    ]

    def __init__(
        self,
        db_path: str | Path,
        temp_dir: str | Path,
        s3_bucket: str,
        s3_profile: str,
        s3_config: Optional[Dict[str, Any]] = None,
        s3_prefix: str = "verifications/",
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the converter using existing GDBConfig.

        Args:
            config: GDBConfig instance (loads from file if None)
            s3_prefix: S3 prefix for verification files
        """
        # TODO from .config import load_config
        # TODO no used from gcover.config import load_config, AppConfig

        self.s3_prefix = s3_prefix.rstrip("/") + "/"
        self.db_path = Path(db_path)
        self.temp_dir = Path(temp_dir)
        self.s3_bucket = s3_bucket
        self.s3_profile = s3_profile
        self.max_workers = max_workers or 4
        self.s3_config = s3_config

        if s3_bucket or s3_profile:
            warnings.warn(
                "Passing s3_bucket and s3_profile directly is deprecated. Use s3_config instead.",
                DeprecationWarning,
            )

        # Use verification-specific database path
        # verification_db = self.config.db_path.parent / "verification_stats.duckdb"
        self.duckdb_path = db_path

        # Initialize S3 client with profile support
        # session = boto3.Session(profile_name=self.s3_profile)
        # self.s3_client = session.client("s3")

        self.s3_uploader = S3Uploader(
            bucket_name=s3_config.bucket,
            aws_profile=s3_config.profile,
            lambda_endpoint=s3_config.lambda_endpoint,
            totp_secret=s3_config.lambda_endpoint,
            proxy_config=s3_config.proxy,
        )

        # Initialize DuckDB connection
        console.print(f"DuckDB: {self.duckdb_path}")
        try:
            self.conn = duckdb.connect(str(self.duckdb_path))
            self._init_stats_tables()
        except duckdb.duckdb.IOException as e:
            raise IOError(
                f"Could not open/connect to DuckDB: {self.duckdb_path}: {str(e)}"
            )

        # Setup logging
        # TODO configure logging
        """import logging

        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)

        self.logger.info(self.config)"""

    def _init_stats_tables(self):
        """Initialize DuckDB tables for statistics storage."""
        logger.debug("Initializing table `gdb_summaries`")
        # Main summary table
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS  gdb_summaries_id_seq START 1;
            CREATE TABLE IF NOT EXISTS gdb_summaries (
                id INTEGER DEFAULT nextval('gdb_summaries_id_seq') PRIMARY KEY,
                gdb_path VARCHAR,
                timestamp TIMESTAMP,
                rc_version VARCHAR,
                verification_type VARCHAR,
                total_features INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (timestamp, rc_version, verification_type, total_features)
            );

        """)

        # Layer statistics table
        logger.debug("Initializing table `layer_stats`")
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS  layer_stats_id_seq START 1;
            CREATE TABLE IF NOT EXISTS layer_stats (
                id INTEGER DEFAULT nextval('layer_stats_id_seq') PRIMARY KEY,
                gdb_summary_id INTEGER,
                layer_name VARCHAR,
                feature_count INTEGER,
                FOREIGN KEY (gdb_summary_id) REFERENCES gdb_summaries(id)
            )
        """)

        # Test statistics table
        logger.debug("Initializing table `test_stats`")
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS  test_stats_id_seq START 1;
            CREATE TABLE IF NOT EXISTS test_stats (
                id INTEGER DEFAULT nextval('test_stats_id_seq') PRIMARY KEY,
                gdb_summary_id INTEGER,
                layer_name VARCHAR,
                test_name VARCHAR,
                issue_type VARCHAR,
                feature_count INTEGER,
                FOREIGN KEY (gdb_summary_id) REFERENCES gdb_summaries(id)
            )
        """)

    @staticmethod
    def parse_gdb_path(gdb_path: Path) -> Tuple[str, str, datetime]:
        """
        Parse GDB path to extract metadata.

        Expected format: /path/to/Verifications/Topology/RC_2030-12-31/20250718_07-00-12/issue.gdb

        Returns:
            verification_type, rc_version, timestamp
        """
        parts = gdb_path.parts

        # Find verification type and RC version
        verification_type = None
        rc_version = None
        timestamp_str = None

        for i, part in enumerate(parts):
            if part in ["Topology", "TechnicalQualityAssurance"]:
                verification_type = part
                if i + 1 < len(parts) and parts[i + 1].startswith("RC_"):
                    rc_version = parts[i + 1].replace("RC_", "")
                if i + 2 < len(parts):
                    timestamp_str = parts[i + 2]
                break

        if not all([verification_type, rc_version, timestamp_str]):
            raise ValueError(f"Could not parse GDB path: {gdb_path}")

        # Parse timestamp (format: 20250718_07-00-12)
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H-%M-%S")
        except ValueError:
            raise ValueError(f"Could not parse timestamp: {timestamp_str}")

        return verification_type, rc_version, timestamp

    def _read_layer(self, gdb_path: Path, layer_name: str) -> Optional[pd.DataFrame]:
        """Simplified layer reader focusing on getting data out."""
        try:
            records = []
            geometries = []
            has_geometry = False

            with fiona.open(str(gdb_path), layer=layer_name) as src:
                if len(src) == 0:
                    self.logger.warning(f"Layer {layer_name} is empty")
                    return None

                for record in src:
                    try:
                        # Get properties
                        props = record.get("properties", {})
                        if not props:
                            continue

                        records.append(props)

                        # Handle geometry
                        geom_data = record.get("geometry")
                        if geom_data:
                            try:
                                from shapely.geometry import shape

                                geom = shape(geom_data)
                                geometries.append(geom)
                                has_geometry = True
                            except Exception:
                                geometries.append(None)
                        else:
                            geometries.append(None)

                    except Exception:
                        continue

            if not records:
                logger.warning(f"No valid records found in {layer_name}")
                return None

            # Create DataFrame
            df = pd.DataFrame(records)

            # Add geometry if available
            if has_geometry and any(g is not None for g in geometries):
                df["geometry"] = geometries
                gdf = gpd.GeoDataFrame(df, geometry="geometry")
                if gdf.crs is None:
                    gdf.set_crs("EPSG:2056", inplace=True)
                return gdf
            else:
                return df

        except Exception as e:
            logger.warning(f"Could not read layer {layer_name}: {e}")
            return None

    def _read_layer_complexe(
        self, gdb_path: Path, layer_name: str
    ) -> Optional[pd.DataFrame]:
        """Read a layer from FileGDB using GeoPandas/GDAL."""
        try:
            # Check if layer exists first
            import fiona

            with fiona.Env():
                try:
                    # List available layers
                    layers = fiona.listlayers(str(gdb_path))
                    if layer_name not in layers:
                        logger.warning(f"Layer {layer_name} not found in {gdb_path}")
                        return None
                except Exception:
                    logger.error(f"Could not list layers in {gdb_path}")
                    return None

            # Read the layer
            if layer_name == "IssueRows":
                # Non-spatial layer - read as regular DataFrame
                import fiona

                with fiona.open(str(gdb_path), layer=layer_name) as src:
                    records = [record["properties"] for record in src]
                    if records:
                        df = pd.DataFrame(records)
                        return df
                    return None
            else:
                # Spatial layer - read as GeoDataFrame
                gdf = gpd.read_file(str(gdb_path), layer=layer_name)

                # Handle 3D geometries - convert to 2D for web compatibility
                if hasattr(gdf, "geometry") and not gdf.empty:
                    from shapely.geometry import LineString, Point, Polygon
                    from shapely.ops import transform

                    def force_2d(geom):
                        """Force geometry to 2D."""
                        if geom is None:
                            return None
                        if hasattr(geom, "coords"):
                            if isinstance(geom, Point):
                                return Point(geom.coords[0][:2])
                            elif isinstance(geom, LineString):
                                return LineString([coord[:2] for coord in geom.coords])
                        elif hasattr(geom, "exterior"):
                            if isinstance(geom, Polygon):
                                exterior = [coord[:2] for coord in geom.exterior.coords]
                                holes = [
                                    [coord[:2] for coord in hole.coords]
                                    for hole in geom.interiors
                                ]
                                return Polygon(exterior, holes)
                        # For other geometry types, try generic approach
                        try:
                            return transform(lambda x, y, z=None: (x, y), geom)
                        except:
                            return geom

                    gdf["geometry"] = gdf["geometry"].apply(force_2d)

                    return gdf

        except Exception as e:
            logger.warning(f"Could not read layer {layer_name} from {gdb_path}: {e}")
        return None

    def _analyze_layer(self, df: pd.DataFrame, layer_name: str) -> LayerStats:
        """Analyze a layer and generate statistics."""
        feature_count = len(df)

        # Count by TestName
        test_names = {}
        if "TestName" in df.columns:
            test_names = df["TestName"].value_counts().to_dict()

        # Count by IssueType
        issue_types = {}
        if "IssueType" in df.columns:
            issue_types = df["IssueType"].value_counts().to_dict()

        # Create test/issue matrix
        test_issue_matrix = {}
        if "TestName" in df.columns and "IssueType" in df.columns:
            matrix = df.groupby(["TestName", "IssueType"]).size()
            test_issue_matrix = matrix.to_dict()

        return LayerStats(
            layer_name=layer_name,
            feature_count=feature_count,
            test_names=test_names,
            issue_types=issue_types,
            test_issue_matrix=test_issue_matrix,
        )

    def _convert_to_web_format(
        self, df: pd.DataFrame, output_path: Path, format: str = "geoparquet"
    ) -> None:
        """Convert DataFrame to web format."""
        try:
            # Handle spatial vs non-spatial data
            is_spatial = hasattr(df, "geometry") and "geometry" in df.columns

            if format == "geoparquet" and is_spatial:
                # Ensure we have a valid CRS
                if df.crs is None:
                    logger.warning(
                        "No CRS found, assuming EPSG:2056 (Swiss coordinates)"
                    )
                    df.set_crs("EPSG:2056", inplace=True)

                # Convert to WGS84 for web compatibility
                df_web = df.to_crs("EPSG:4326")
                df_web.to_parquet(output_path, compression="snappy")
                logger.info(f"Converted to GeoParquet (EPSG:4326): {output_path}")

            elif format.lower() == "flatgeobuf" and is_spatial:
                # Ensure we have a valid CRS
                if df.crs is None:
                    logger.warning(
                        "No CRS found, assuming EPSG:2056 (Swiss coordinates)"
                    )
                    df.set_crs("EPSG:2056", inplace=True)

                # Convert to WGS84 for web compatibility
                df_web = df.to_crs("EPSG:4326")
                df_web.to_file(output_path, driver="FlatGeobuf")
                logger.info(f"Converted to FlatGeoBuffer (EPSG:4326): {output_path}")

            elif format == "geojson" and is_spatial:
                # Ensure WGS84 for GeoJSON
                if df.crs is None:
                    df.set_crs("EPSG:2056", inplace=True)
                df_web = df.to_crs("EPSG:4326")
                df_web.to_file(output_path, driver="GeoJSON")
                logger.info(f"Converted to GeoJSON: {output_path}")

            else:
                # Non-spatial data or fallback - save as JSON/Parquet
                if output_path.suffix.lower() == ".json":
                    df.to_json(output_path, orient="records", date_format="iso")
                else:
                    # Change extension to .parquet for non-spatial data
                    parquet_path = output_path.with_suffix(".parquet")
                    df.to_parquet(parquet_path, compression="snappy")
                    logger.info(f"Converted to Parquet: {parquet_path}")

        except Exception as e:
            logger.error(f"Failed to convert to {format}: {e}")
            raise

    def _upload_to_s3(self, local_path: Path, s3_key: str) -> bool:
        """Upload a file to S3."""
        # Upload to S3

        try:
            if not self.s3_uploader.file_exists(s3_key):
                uploaded = self.s3_uploader.upload_file(local_path, s3_key)
            else:
                logger.info(f"File already exists in S3: {s3_key}")
            logger.info(f"Uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False

    def _normalize_gdb_path(self, gdb_path: str) -> str:
        """
        Normalize GDB path to extract only the relevant part for duplicate detection.

        Examples:
            /media/marco/SANDISK/Verifications/Topology/RC_2030-12-31/20250905_07-00-12/issue.gdb
            -> Topology/RC_2030-12-31/20250905_07-00-12/issue.gdb

            /some/other/path/Verifications/TechnicalQualityAssurance/RC_2016-12-31/20231203_22-00-09/issue.gdb
            -> TechnicalQualityAssurance/RC_2016-12-31/20231203_22-00-09/issue.gdb
        """
        path_parts = Path(gdb_path).parts

        # Find the index where "Verifications" appears, or specific verification types
        start_idx = None
        for i, part in enumerate(path_parts):
            if part in ["Verifications", "Topology", "TechnicalQualityAssurance"]:
                # If we find "Verifications", start from the next part
                if part == "Verifications":
                    start_idx = i + 1
                else:
                    # If we find verification type directly, start from there
                    start_idx = i
                break

        if start_idx is not None and start_idx < len(path_parts):
            # Join the relevant parts
            relevant_parts = path_parts[start_idx:]
            return "/".join(relevant_parts)
        else:
            # Fallback: return the last 4 parts (should cover most cases)
            return "/".join(path_parts[-4:]) if len(path_parts) >= 4 else gdb_path

    def _check_existing_summary(self, summary: GDBSummary) -> Optional[int]:
        """
        Check if a summary with the same key characteristics already exists.

        Args:
            summary: The GDBSummary to check

        Returns:
            The existing summary ID if found, None otherwise
        """
        normalized_path = self._normalize_gdb_path(summary.gdb_path)

        # Check for exact match on key fields
        result = self.conn.execute(
            """
            SELECT id, gdb_path
            FROM gdb_summaries
            WHERE timestamp = ?
              AND rc_version = ?
              AND verification_type = ?
              AND total_features = ?
            ORDER BY created_at DESC
            LIMIT 1
        """,
            [
                summary.timestamp,
                summary.rc_version,
                summary.verification_type,
                summary.total_features,
            ],
        ).fetchone()

        if result:
            existing_id, existing_path = result
            existing_normalized = self._normalize_gdb_path(existing_path)

            # Additional check: compare normalized paths
            if existing_normalized == normalized_path:
                logger.info(
                    f"Found exact duplicate for {normalized_path} "
                    f"(timestamp: {summary.timestamp}, rc: {summary.rc_version}, "
                    f"type: {summary.verification_type}, features: {summary.total_features}) "
                    f"- using existing ID {existing_id}"
                )
                return existing_id
            else:
                # Same metadata but different paths - log warning but allow insert
                logger.warning(
                    f"Found summary with same metadata but different path: "
                    f"existing='{existing_normalized}' vs new='{normalized_path}' "
                    f"- will insert as new record"
                )

        return None

    def _store_statistics(self, summary: GDBSummary) -> int:
        """
        Store summary statistics in DuckDB, avoiding duplicates.

        Args:
            summary: The GDBSummary to store

        Returns:
            The summary ID (either existing or newly created)
        """
        # Check if this summary already exists
        existing_id = self._check_existing_summary(summary)
        if existing_id is not None:
            logger.info(
                f"Skipping duplicate summary, returning existing ID: {existing_id}"
            )
            return existing_id

        # Insert new summary record
        logger.info(
            f"Inserting new summary for {self._normalize_gdb_path(summary.gdb_path)}"
        )

        summary_id = self.conn.execute(
            """
            INSERT INTO gdb_summaries
            (gdb_path, timestamp, rc_version, verification_type, total_features)
            VALUES (?, ?, ?, ?, ?)
            RETURNING id
            """,
            [
                summary.gdb_path,
                summary.timestamp,
                summary.rc_version,
                summary.verification_type,
                summary.total_features,
            ],
        ).fetchone()[0]

        # Insert layer statistics
        for layer_stats in summary.layers.values():
            self.conn.execute(
                """
                INSERT INTO layer_stats
                (gdb_summary_id, layer_name, feature_count)
                VALUES (?, ?, ?)
                """,
                [summary_id, layer_stats.layer_name, layer_stats.feature_count],
            )

            # Insert test statistics
            for (test_name, issue_type), count in layer_stats.test_issue_matrix.items():
                self.conn.execute(
                    """
                    INSERT INTO test_stats
                    (gdb_summary_id, layer_name, test_name, issue_type, feature_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [summary_id, layer_stats.layer_name, test_name, issue_type, count],
                )

        self.conn.commit()
        logger.info(f"Successfully stored new summary with ID: {summary_id}")
        return summary_id

    def _read_with_fiona_fallback(
        self, gdb_path: Path, layer_name: str
    ) -> Optional[gpd.GeoDataFrame]:
        """Fallback: read with Fiona, handling geometry errors gracefully."""
        try:
            import fiona
            from shapely.geometry import shape

            records = []
            geometry_errors = 0

            with fiona.open(str(gdb_path), layer=layer_name) as src:
                logger.info(f"Reading {len(src)} features from {layer_name}...")

                for i, record in enumerate(src):
                    try:
                        if record["geometry"]:
                            geom = shape(record["geometry"])
                            if geom and geom.is_valid:
                                record_data = record["properties"].copy()
                                record_data["geometry"] = geom
                                records.append(record_data)
                            else:
                                geometry_errors += 1
                        else:
                            records.append(record["properties"])
                    except Exception:
                        geometry_errors += 1
                        continue

            if geometry_errors > 0:
                logger.warning(
                    f"Skipped {geometry_errors} features with geometry errors in {layer_name}"
                )

            if records:
                df = pd.DataFrame(records)
                if "geometry" in df.columns:
                    gdf = gpd.GeoDataFrame(df, geometry="geometry")
                    if gdf.crs is None:
                        gdf.set_crs("EPSG:2056", inplace=True)
                    return gdf
                else:
                    return df
            return None

        except Exception as e:
            logger.error(f"Fiona fallback failed for {layer_name}: {e}")
            return None

    def _simplify_geometries(
        self, gdf: gpd.GeoDataFrame, tolerance: float, layer_name: str
    ) -> gpd.GeoDataFrame:
        """Simplify complex geometries to reduce processing overhead."""
        try:
            if "geometry" not in gdf.columns or gdf.empty:
                return gdf

            logger.info(
                f"Simplifying geometries in {layer_name} with tolerance {tolerance}"
            )
            gdf["geometry"] = gdf["geometry"].simplify(
                tolerance, preserve_topology=True
            )
            gdf = gdf[gdf.geometry.is_valid]
            gdf = gdf.dropna(subset=["geometry"])
            return gdf

        except Exception as e:
            logger.warning(f"Error simplifying geometries for {layer_name}: {e}")
            return gdf

    def process_gdb(
        self,
        gdb_path: Path,
        output_dir: Optional[Path] = None,
        upload_to_s3: bool = True,
        output_format: str = "geoparquet",
        convert_to_web: bool = True,
        simplify_tolerance: Optional[float] = None,  # Add this line
    ) -> GDBSummary:
        """
        Process a single FileGDB: convert layers and generate statistics.

        Args:
            gdb_path: Path to the issue.gdb file
            output_dir: Local output directory for converted files
            upload_to_s3: Whether to upload converted files to S3

        Returns:
            GDBSummary object with statistics
        """
        console.print(f"[bold blue]Processing GDB:[/bold blue] {gdb_path}")

        if upload_to_s3 and not convert_to_web:
            raise ValueError("Must convert to Web format prior to S3 Upload")

        # Parse metadata from path
        verification_type, rc_version, timestamp = self.parse_gdb_path(gdb_path)

        # Setup output directory
        if output_dir is None:
            output_dir = (
                self.temp_dir / f"converted_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        layers_stats = {}
        total_features = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for layer_name in self.SPATIAL_LAYERS:
                task = progress.add_task(f"Processing {layer_name}...", total=None)

                # Read layer
                gdf = self._read_layer(gdb_path, layer_name)
                if gdf is None:
                    progress.update(task, description=f"⚠️  Skipped {layer_name}")
                    continue

                # Apply geometry simplification if requested
                if (
                    simplify_tolerance
                    and hasattr(gdf, "geometry")
                    and "geometry" in gdf.columns
                ):
                    gdf = self._simplify_geometries(gdf, simplify_tolerance, layer_name)

                # Analyze layer
                layer_stats = self._analyze_layer(gdf, layer_name)
                layers_stats[layer_name] = layer_stats
                total_features += layer_stats.feature_count

                # Convert to web format
                if layer_name != "IssueRows" and hasattr(gdf, "geometry"):
                    if convert_to_web:
                        # Determine file extension based on format
                        if output_format == "geoparquet":
                            file_path = output_dir / f"{layer_name}.parquet"
                        elif output_format == "geojson":
                            file_path = output_dir / f"{layer_name}.geojson"
                        elif output_format == "flatgeobuf":
                            file_path = output_dir / f"{layer_name}.fgb"
                        else:  # all
                            # Create all formats
                            parquet_path = output_dir / f"{layer_name}.parquet"
                            geojson_path = output_dir / f"{layer_name}.geojson"
                            flatgeobuf_path = output_dir / f"{layer_name}.fgb"

                            self._convert_to_web_format(gdf, parquet_path, "geoparquet")
                            self._convert_to_web_format(gdf, geojson_path, "geojson")
                            self._convert_to_web_format(
                                gdf, flatgeobuf_path, "flatgeobuf"
                            )

                            if upload_to_s3:
                                s3_key_parquet = f"{self.s3_prefix}{verification_type}/{rc_version}/{timestamp.strftime('%Y%m%d_%H%M%S')}/{layer_name}.parquet"
                                s3_key_geojson = f"{self.s3_prefix}{verification_type}/{rc_version}/{timestamp.strftime('%Y%m%d_%H%M%S')}/{layer_name}.geojson"
                                s3_key_fgb = f"{self.s3_prefix}{verification_type}/{rc_version}/{timestamp.strftime('%Y%m%d_%H%M%S')}/{layer_name}.fgb"
                                self._upload_to_s3(parquet_path, s3_key_parquet)
                                self._upload_to_s3(geojson_path, s3_key_geojson)
                                self._upload_to_s3(flatgeobuf_path, s3_key_fgb)

                            progress.update(
                                task,
                                description=f"✅ Processed {layer_name} ({layer_stats.feature_count:,} features) - all formats",
                            )
                            continue

                        self._convert_to_web_format(gdf, file_path, output_format)

                        # Upload to S3
                        if upload_to_s3:
                            s3_key = f"{self.s3_prefix}{verification_type}/{rc_version}/{timestamp.strftime('%Y%m%d_%H%M%S')}/{file_path.name}"
                            self._upload_to_s3(file_path, s3_key)

                # Save non-spatial data as JSON/Parquet
                else:
                    if output_format in ["geoparquet", "all"]:
                        file_path = output_dir / f"{layer_name}.parquet"
                    else:
                        file_path = output_dir / f"{layer_name}.json"

                    self._convert_to_web_format(gdf, file_path)

                    if upload_to_s3:
                        s3_key = f"{self.s3_prefix}{verification_type}/{rc_version}/{timestamp.strftime('%Y%m%d_%H%M%S')}/{file_path.name}"
                        self._upload_to_s3(file_path, s3_key)

                progress.update(
                    task,
                    description=f"✅ Processed {layer_name} ({layer_stats.feature_count:,} features)",
                )

        # Create summary
        summary = GDBSummary(
            gdb_path=str(gdb_path),
            timestamp=timestamp,
            rc_version=rc_version,
            verification_type=verification_type,
            total_features=total_features,
            layers=layers_stats,
        )

        # Store statistics
        summary_id = self._store_statistics(summary)
        console.print(f"[green]✅ Stored statistics with ID: {summary_id}[/green]")

        return summary

    def get_statistics_summary(
        self,
        verification_type: Optional[str] = None,
        rc_version: Optional[str] = None,
        days_back: int = 30,
    ) -> pd.DataFrame:
        """
        Get aggregated statistics for dashboard display.

        Args:
            verification_type: Filter by verification type
            rc_version: Filter by RC version
            days_back: Number of days to look back

        Returns:
            DataFrame with aggregated statistics
        """
        # Build the base query with proper date handling
        query = f"""
            SELECT
                s.verification_type,
                s.rc_version,
                ts.test_name,
                ts.issue_type,
                CAST(SUM(ts.feature_count) AS INTEGER) as total_count,
                COUNT(DISTINCT s.id) as num_runs,
                MAX(s.timestamp) as latest_run
            FROM test_stats ts
            JOIN gdb_summaries s ON ts.gdb_summary_id = s.id
            WHERE s.timestamp >= CURRENT_DATE - INTERVAL '{days_back}' DAY
        """

        params = []
        if verification_type:
            query += " AND s.verification_type = ?"
            params.append(verification_type)

        if rc_version:
            query += " AND s.rc_version = ?"
            params.append(rc_version)

        query += """
            GROUP BY s.verification_type, s.rc_version, ts.test_name, ts.issue_type
            ORDER BY total_count DESC
        """

        return self.conn.execute(query, params).df()

    def close(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()


def main():
    """Example usage of the FileGDBConverter."""
    # from .config import load_config TODO

    # Load configuration
    config = load_config()

    # Initialize converter with config
    converter = FileGDBConverter(config=config)

    try:
        # Process a single GDB (using configured base paths if available)
        if "verifications" in config.base_paths:
            base_path = config.base_paths["verifications"]
            gdb_path = base_path / "Topology/RC_2030-12-31/20250718_07-00-12/issue.gdb"
        else:
            gdb_path = Path(
                "/media/marco/SANDISK/Verifications/Topology/RC_2030-12-31/20250718_07-00-12/issue.gdb"
            )

        if gdb_path.exists():
            summary = converter.process_gdb(gdb_path)
            console.print(
                f"[green]Processed {summary.total_features:,} total features[/green]"
            )

            # Get recent statistics
            stats_df = converter.get_statistics_summary(days_back=7)
            console.print(f"[blue]Found {len(stats_df)} recent test results[/blue]")

        else:
            console.print(f"[red]GDB not found: {gdb_path}[/red]")

    finally:
        converter.close()
