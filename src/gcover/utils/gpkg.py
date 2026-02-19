"""
GPKG utilities with progress feedback for large datasets

Provides efficient methods for writing large GeoDataFrames to GPKG files
with rich progress bars and chunked processing for better user experience.
"""

import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional, Union

import geopandas as gpd
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)

from gcover.config import DEFAULT_CHUNK_SIZE, DEFAULT_NUM_WORKERS

console = Console()


class GPKGWriter:
    """
    Efficient GPKG writer with progress feedback and chunked processing.

    Supports both single-threaded (with detailed progress) and multi-threaded
    writing for optimal performance with large datasets.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        num_workers: Optional[int] = None,
        show_progress: bool = True,
    ):
        """
        Initialize GPKG writer.

        Args:
            chunk_size: Number of features per chunk
            num_workers: Number of parallel workers (None = auto-detect)
            show_progress: Show rich progress bars
        """
        self.chunk_size = chunk_size
        self.num_workers = num_workers or min(4, mp.cpu_count())
        self.show_progress = show_progress

        logger.debug(
            f"GPKGWriter initialized: chunk_size={chunk_size}, workers={self.num_workers}"
        )

    def write_gdf(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: Union[str, Path],
        layer_name: str,
        mode: Literal["w", "a"] = "w",
        parallel: bool = False,
        compression: bool = True,
    ) -> Path:
        """
        Write GeoDataFrame to GPKG with progress feedback.

        Args:
            gdf: GeoDataFrame to write
            output_path: Output GPKG file path
            layer_name: Layer name in GPKG
            mode: Write mode ('w' = overwrite, 'a' = append)
            parallel: Use parallel processing (faster but no detailed progress)
            compression: Use GPKG compression

        Returns:
            Path to written file

        Examples:
            writer = GPKGWriter(chunk_size=500)

            # Simple write with progress
            writer.write_gdf(gdf, "output.gpkg", "my_layer")

            # Parallel write for large datasets
            writer.write_gdf(large_gdf, "output.gpkg", "big_layer", parallel=True)
        """
        output_path = Path(output_path)

        if len(gdf) == 0:
            logger.warning("Empty GeoDataFrame, creating empty GPKG")
            gdf.to_file(output_path, layer=layer_name, driver="GPKG")
            return output_path

        logger.info(f"Writing {len(gdf)} features to {output_path}:{layer_name}")

        # Choose writing strategy based on size and parallel flag
        if parallel and len(gdf) > self.chunk_size * 2:
            return self._write_parallel(gdf, output_path, layer_name, mode, compression)
        else:
            return self._write_chunked(gdf, output_path, layer_name, mode, compression)

    def _write_chunked(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: Path,
        layer_name: str,
        mode: str,
        compression: bool,
    ) -> Path:
        """Write GeoDataFrame in chunks with detailed progress."""

        # Prepare GPKG options
        options = []
        if compression:
            options.extend(["SPATIAL_INDEX=YES", "COMPRESS_GEOM=YES"])

        # Split into chunks
        chunks = self._create_chunks(gdf)

        if not self.show_progress:
            # Simple chunk writing without progress
            for i, chunk in enumerate(chunks):
                write_mode = mode if i == 0 else "a"
                chunk.to_file(
                    output_path,
                    layer=layer_name,
                    driver="GPKG",
                    mode=write_mode,
                    engine="pyogrio" if compression else "fiona",
                )
            return output_path

        # Rich progress bar setup
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Writing GPKG..."),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("chunks"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(f"[cyan]Writing {layer_name}", total=len(chunks))

            for i, chunk in enumerate(chunks):
                # Determine write mode
                write_mode = mode if i == 0 else "a"

                try:
                    # Write chunk
                    chunk.to_file(
                        output_path,
                        layer=layer_name,
                        driver="GPKG",
                        mode=write_mode,
                        engine="pyogrio" if compression else "fiona",
                    )

                    progress.update(
                        task,
                        advance=1,
                        description=f"[cyan]Writing {layer_name} (chunk {i + 1}/{len(chunks)})",
                    )

                except Exception as e:
                    logger.error(f"Failed to write chunk {i + 1}: {e}")
                    raise

        logger.success(f"Successfully wrote {len(gdf)} features to {output_path}")
        return output_path

    def _write_parallel(
        self,
        gdf: gpd.GeoDataFrame,
        output_path: Path,
        layer_name: str,
        mode: str,
        compression: bool,
    ) -> Path:
        """Write GeoDataFrame using parallel processing."""

        logger.info(f"Using parallel writing with {self.num_workers} workers")

        # Split into chunks
        chunks = self._create_chunks(gdf)

        # Create temporary directory for chunk files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            if not self.show_progress:
                # Simple parallel execution
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = []
                    for i, chunk in enumerate(chunks):
                        temp_file = temp_path / f"chunk_{i:06d}.gpkg"
                        future = executor.submit(
                            self._write_chunk_worker,
                            chunk,
                            temp_file,
                            layer_name,
                            compression,
                        )
                        futures.append((future, temp_file))

                    # Wait for completion
                    for future, _ in futures:
                        future.result()
            else:
                # Parallel execution with progress
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Writing chunks in parallel..."),
                    BarColumn(bar_width=40),
                    TaskProgressColumn(),
                    TextColumn("chunks"),
                    TimeElapsedColumn(),
                    console=console,
                    transient=False,
                ) as progress:
                    task = progress.add_task(
                        f"[cyan]Processing {layer_name}", total=len(chunks)
                    )

                    with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                        # Submit all chunks
                        futures = []
                        for i, chunk in enumerate(chunks):
                            temp_file = temp_path / f"chunk_{i:06d}.gpkg"
                            future = executor.submit(
                                self._write_chunk_worker,
                                chunk,
                                temp_file,
                                layer_name,
                                compression,
                            )
                            futures.append((future, temp_file))

                        # Monitor completion
                        for future in as_completed([f[0] for f in futures]):
                            try:
                                future.result()  # Raise any exceptions
                                progress.update(task, advance=1)
                            except Exception as e:
                                logger.error(f"Chunk writing failed: {e}")
                                raise

            # Merge all chunk files into final GPKG
            self._merge_chunks(
                [temp_file for _, temp_file in futures], output_path, layer_name, mode
            )

        logger.success(
            f"Successfully wrote {len(gdf)} features using parallel processing"
        )
        return output_path

    def _create_chunks(self, gdf: gpd.GeoDataFrame) -> list[gpd.GeoDataFrame]:
        """Split GeoDataFrame into chunks."""
        chunks = []
        for i in range(0, len(gdf), self.chunk_size):
            chunk = gdf.iloc[i : i + self.chunk_size].copy()
            chunks.append(chunk)

        logger.debug(f"Created {len(chunks)} chunks of max size {self.chunk_size}")
        return chunks

    @staticmethod
    def _write_chunk_worker(
        chunk: gpd.GeoDataFrame, temp_file: Path, layer_name: str, compression: bool
    ) -> Path:
        """Worker function for parallel chunk writing."""
        try:
            chunk.to_file(
                temp_file,
                layer=layer_name,
                driver="GPKG",
                engine="pyogrio" if compression else "fiona",
            )
            return temp_file
        except Exception as e:
            logger.error(f"Worker failed to write chunk: {e}")
            raise

    def _merge_chunks(
        self, chunk_files: list[Path], output_path: Path, layer_name: str, mode: str
    ) -> None:
        """Merge chunk files into final GPKG."""

        if not self.show_progress:
            # Simple merge
            for i, chunk_file in enumerate(chunk_files):
                if chunk_file.exists():
                    chunk_gdf = gpd.read_file(chunk_file, layer=layer_name)
                    write_mode = mode if i == 0 else "a"
                    chunk_gdf.to_file(
                        output_path, layer=layer_name, driver="GPKG", mode=write_mode
                    )
            return

        # Merge with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Merging chunks..."),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("files"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("[green]Merging", total=len(chunk_files))

            for i, chunk_file in enumerate(chunk_files):
                if chunk_file.exists():
                    try:
                        chunk_gdf = gpd.read_file(chunk_file, layer=layer_name)
                        write_mode = mode if i == 0 else "a"
                        chunk_gdf.to_file(
                            output_path,
                            layer=layer_name,
                            driver="GPKG",
                            mode=write_mode,
                        )
                        progress.update(task, advance=1)
                    except Exception as e:
                        logger.error(f"Failed to merge chunk {chunk_file}: {e}")
                        raise
                else:
                    logger.warning(f"Chunk file not found: {chunk_file}")
                    progress.update(task, advance=1)


def write_gdf_to_gpkg(
    gdf: gpd.GeoDataFrame,
    output_path: Union[str, Path],
    layer_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    mode: Literal["w", "a"] = "w",
    parallel: bool = None,
    show_progress: bool = True,
    compression: bool = True,
) -> Path:
    """
    Convenience function to write GeoDataFrame to GPKG with progress.

    Args:
        gdf: GeoDataFrame to write
        output_path: Output GPKG file path
        layer_name: Layer name in GPKG
        chunk_size: Number of features per chunk
        num_workers: Number of parallel workers
        mode: Write mode ('w' = overwrite, 'a' = append)
        parallel: Use parallel processing (auto-detect if None)
        show_progress: Show progress bars
        compression: Use GPKG compression

    Returns:
        Path to written file

    Examples:
        # Simple usage
        write_gdf_to_gpkg(gdf, "output.gpkg", "my_layer")

        # Large dataset with parallel processing
        write_gdf_to_gpkg(
            large_gdf, "big_data.gpkg", "big_layer",
            parallel=True, chunk_size=1000
        )
    """
    # Auto-detect parallel processing for large datasets
    if parallel is None:
        parallel = len(gdf) > chunk_size * 4

    writer = GPKGWriter(
        chunk_size=chunk_size, num_workers=num_workers, show_progress=show_progress
    )

    return writer.write_gdf(gdf, output_path, layer_name, mode, parallel, compression)


def append_gdf_to_gpkg(
    gdf: gpd.GeoDataFrame, output_path: Union[str, Path], layer_name: str, **kwargs
) -> Path:
    """
    Convenience function to append GeoDataFrame to existing GPKG.

    Args:
        gdf: GeoDataFrame to append
        output_path: Existing GPKG file path
        layer_name: Layer name in GPKG
        **kwargs: Additional arguments for write_gdf_to_gpkg

    Returns:
        Path to GPKG file
    """
    return write_gdf_to_gpkg(gdf, output_path, layer_name, mode="a", **kwargs)


def estimate_gpkg_size(
    gdf: gpd.GeoDataFrame, compression: bool = True
) -> tuple[int, str]:
    """
    Estimate GPKG file size for planning.

    Args:
        gdf: GeoDataFrame to estimate
        compression: Whether compression will be used

    Returns:
        Tuple of (size_bytes, human_readable_size)
    """
    # Rough estimation based on data types and geometry complexity
    base_size = gdf.memory_usage(deep=True).sum()

    # Geometry typically adds significant overhead
    if hasattr(gdf, "geometry") and not gdf.geometry.empty:
        geom_complexity = gdf.geometry.apply(lambda g: len(str(g)) if g else 0).mean()
        base_size += len(gdf) * geom_complexity * 0.5

    # GPKG overhead and compression factor
    overhead_factor = 1.3  # GPKG indexing and metadata
    compression_factor = 0.7 if compression else 1.0

    estimated_size = int(base_size * overhead_factor * compression_factor)

    # Human readable format
    for unit in ["B", "KB", "MB", "GB"]:
        if estimated_size < 1024:
            readable_size = f"{estimated_size:.1f} {unit}"
            break
        estimated_size /= 1024
    else:
        readable_size = f"{estimated_size:.1f} TB"

    return int(base_size * overhead_factor * compression_factor), readable_size
