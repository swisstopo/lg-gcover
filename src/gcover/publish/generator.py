#!/usr/bin/env python3
"""
MapServer Mapfile and QGIS QML Generator

Generate map server configuration from ESRI classification rules.
Supports complex multi-layer symbols including pattern fills, font markers,
and sophisticated polygon styling.

ENHANCED: Support for pattern catalog with PIXMAP symbols for polygon fills.
"""

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from importlib.resources import files

import click
from loguru import logger
from rich.console import Console

from gcover.publish.esri_classification_extractor import (
    ClassificationClass,
    ESRIClassificationExtractor,
    LayerClassification,
    SymbolType,
)
from gcover.publish.style_config import MapfileGenerationConfig
from gcover.publish.symbol_models import (
    CharacterMarkerInfo,
    FontSymbol,
    SymbolLayersInfo,
)
from gcover.publish.symbol_utils import (
    extract_color_from_cim,
    extract_line_style_from_effects,
    extract_polygon_symbol_layers,
    sanitize_font_name,
)
from gcover.publish.tooltips_enricher import LayerType
from gcover.publish.utils import generate_font_image, translate_esri_to_sql
from gcover.config.models import MapserverConnection

from gcover.publish.label_extractor_extension import LabelInfo
from gcover.publish.rotation_extractor_extension import (
    RotationInfo,
    format_rotation_for_mapserver,
)


DEFAULT_IMAGES_DIR = "patterns"  # or img or "etc/img"  for KOGIS

# GDB uppercase field names → PostGIS lowercase, for filter injection
_FILTER_FIELD_MAP = {
    "KIND": "kind",
    "RUNC_LITHO": "runc_litho",
    "ABOR_DEPTH_BEDR": "abor_depth_bedr",
}

console = Console()


# =============================================================================
# PATTERN CATALOG SUPPORT
# =============================================================================


class PatternCatalogReader:
    """
    Read and query pattern catalog for MapServer symbol generation.

    The pattern catalog maps ESRI CIMCharacterMarker patterns to
    pre-rendered PNG tiles for use as MapServer PIXMAP symbols.
    """

    def __init__(self, catalog_path: Optional[Path] = None):
        """
        Initialize catalog reader.

        Args:
            catalog_path: Path to patterns_catalog.yaml (None = no catalog)
        """
        self.catalog_path = catalog_path
        self.patterns: Dict[str, dict] = {}
        self.loaded = False

        if catalog_path and catalog_path.exists():
            self._load_catalog()

    def _load_catalog(self) -> None:
        """Load catalog from YAML file."""
        try:
            import yaml
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            self.patterns = data.get('patterns', {})
            self.loaded = True
            logger.info(f"Loaded pattern catalog with {len(self.patterns)} patterns")

        except Exception as e:
            logger.warning(f"Could not load pattern catalog: {e}")
            self.loaded = False

    def find_pattern(
            self,
            font_family: str,
            char_index: int,
            size: float,
            step_x: float,
            step_y: float,
            color: Tuple[int, int, int, int],
    ) -> Optional[dict]:
        """
        Find matching pattern in catalog.

        Uses fuzzy matching on size/step/color to find the best match.

        Returns:
            Pattern dict with mapserver info, or None if no match
        """
        if not self.loaded:
            return None

        best_match = None
        best_score = float('inf')

        for key, pattern in self.patterns.items():
            char_info = pattern.get('character', {})

            # Must match font and character exactly
            pattern_font = char_info.get('font_family', '')
            pattern_char = char_info.get('char_index')

            if not self._fonts_match(font_family, pattern_font):
                continue
            if pattern_char != char_index:
                continue

            # Score based on size/step similarity
            pattern_size = char_info.get('size', 0)
            pattern_step = char_info.get('step', [0, 0])

            size_diff = abs(size - pattern_size) / max(size, pattern_size, 0.1)
            step_diff = (
                                abs(step_x - pattern_step[0]) / max(step_x, pattern_step[0], 0.1) +
                                abs(step_y - pattern_step[1]) / max(step_y, pattern_step[1], 0.1)
                        ) / 2

            # Color difference (simple RGB distance)
            pattern_color = pattern.get('color', {}).get('rgba', [0, 0, 0, 255])
            color_diff = (
                                 abs(color[0] - pattern_color[0]) +
                                 abs(color[1] - pattern_color[1]) +
                                 abs(color[2] - pattern_color[2])
                         ) / 765  # Normalize to 0-1

            # Combined score (lower is better)
            score = size_diff * 0.3 + step_diff * 0.3 + color_diff * 0.4

            if score < best_score:
                best_score = score
                best_match = pattern

        # Only return if reasonably good match (threshold: 0.5)
        if best_match and best_score < 0.5:
            return best_match

        return None

    def _fonts_match(self, font1: str, font2: str) -> bool:
        """Check if two font names refer to the same font."""

        # Normalize: lowercase, remove spaces
        def normalize(f):
            return f.lower().replace(' ', '').replace('-', '').replace('_', '')

        return normalize(font1) == normalize(font2)

    def get_pixmap_symbol_name(self, pattern: dict) -> Optional[str]:
        """Get MapServer PIXMAP symbol name from pattern."""
        mapserver_info = pattern.get('mapserver', {})
        if mapserver_info.get('native', False):
            return None  # Native hatch, no pixmap
        return mapserver_info.get('symbol')

    def get_png_path(self, pattern: dict) -> Optional[str]:
        """Get relative PNG file path from pattern."""
        mapserver_info = pattern.get('mapserver', {})
        return mapserver_info.get('png_file')


class MapServerGenerator:
    """
    Generate MapServer mapfile CLASS sections from classifications.

    Features:
    - Complete pattern symbol definitions in symbols.txt
    - Font symbol tracking and deduplication
    - Complex polygon symbols with multiple layers
    - Point and line font markers
    - PDF generation of all used symbols
    - ENHANCED: Pattern catalog support for PIXMAP polygon fills
    """

    def __init__(
            self,
            layer_type: str = "Polygon",
            use_symbol_field: bool = False,
            symbol_field: str = None,
            font_name_prefix: str = "esri",
            no_scale: bool = False,
            default_language: str = "de",
            pattern_catalog: Optional[Path] = None,
            gml_items: Optional[str] = 'default',
            output_dir: Optional[Path] = None,
            languages: Optional[List[str]] = None,

    ):
        """
        Initialize generator.

        Args:
            layer_type: POLYGON, LINE, or POINT
            use_symbol_field: If True, use CLASSITEM with simple expressions
            symbol_field: Name of symbol field in data (default: SYMBOL)
            font_name_prefix: Prefix for font names in FONTSET (default: "esri")
            pattern_catalog: Path to patterns_catalog.yaml for PIXMAP symbols
        """
        self.layer_type = layer_type.upper()
        self.use_symbol_field = use_symbol_field
        self.symbol_field = symbol_field
        self.font_name_prefix = font_name_prefix
        self.no_scale = no_scale
        self.gml_items = gml_items
        self.output_dir = output_dir

        # Track fonts and symbols used (for symbol file generation)
        self.fonts_used: Set[str] = set()
        self.pattern_symbols: List[Dict] = []
        self.symbol_registry: Dict[FontSymbol, str] = {}

        # NEW: Pattern catalog for PIXMAP symbols
        self.pattern_catalog = PatternCatalogReader(pattern_catalog)
        self.pixmap_symbols_used: Set[str] = set()
        # Languages
        self.default_language = default_language
        self.languages = languages


    def render_maxscale(self, layer_scale: Optional[bool | int],
                        style_scale: Optional[int]) -> Optional[str]:
        """
        Retourne la ligne MAXSCALEDENOM ou None

        Cas 1: yaml=None,   style=None   → None (pas de scale)
        Cas 2: yaml=None,   style=20000  → "MAXSCALEDENOM 20000" (utilise style)
        Cas 3: yaml=False,  style=None   → None (désactivé explicitement)
        Cas 4: yaml=False,  style=20000  → None (override: désactivé)
        Cas 5: yaml=True,   style=None   → None (veut le style, mais n'existe pas)
        Cas 6: yaml=True,   style=20000  → "MAXSCALEDENOM 20000" (utilise style)
        Cas 7: yaml=50000,  style=None   → "MAXSCALEDENOM 50000" (override)
        Cas 8: yaml=50000,  style=20000  → "MAXSCALEDENOM 50000" (override)
        """

        # YAML désactive explicitement
        if layer_scale is False:
            return None

        # YAML définit une valeur spécifique
        if isinstance(layer_scale, int):
            return f"  MAXSCALEDENOM {layer_scale}"

        # YAML = None ou True → utilise le style si disponible
        if style_scale is not None:
            return f"  MAXSCALEDENOM {style_scale}"

        return None

    def _get_classes_file_path(
            self,
            layer_name: str,
            symbol_prefix: str,
            mapfile_config,
    ) -> Path:
        """Determine path for classes include file."""

        # Check if explicit path in config
        if hasattr(mapfile_config, "classes_file") and mapfile_config.classes_file:
            return Path(mapfile_config.classes_file)

        # Default: output_dir/classes/<symbol_prefix>_classes.inc
        classes_dir = self.output_dir / "classes"
        return classes_dir / f"{symbol_prefix}_classes.inc"

    def _add_classes_header(
            self, classes_content: str, classification, symbol_prefix: str
    ) -> str:
        """Add header comment to classes file."""

        from datetime import datetime

        header_lines = [
            f"# Auto-generated from {getattr(classification, 'style_file', 'unknown')}",
            f"# Symbol prefix: {symbol_prefix}",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Mode: {getattr(getattr(classification, 'mapfile_config', None), 'classes_mode', 'auto')}",
            "",
            classes_content,
        ]

        return "\n".join(header_lines)

    def generate_to_staging(self, classes_file: Path, new_content: str) -> Path:
        """
        Generate classes to staging area for comparison.

        Args:
            classes_file: Original classes file path
            new_content: Newly generated classes content

        Returns:
            Path to staging file (.new)
        """
        # Create staging directory
        staging_dir = classes_file.parent / ".staging"
        staging_dir.mkdir(exist_ok=True, parents=True)

        # Write to .new file
        staging_file = staging_dir / f"{classes_file.name}.new"
        staging_file.write_text(new_content)

        logger.info(f"✓ Generated staging file: {staging_file}")

        return staging_file

    # ========================================================================
    # MULTI-LANGUAGE DATA HELPER
    # ========================================================================

    def _build_lang_data(
            self,
            data: str,
            lang_fields: Dict[str, str],
            include_items: Optional[str] = None,
            symbol_field: Optional[str] = None,
            geom_col: Optional[str] = None,
            label_col: Optional[str] = None,
    ) -> str:
        """
        Inject language-aliased columns into a DATA string, keeping %language%
        as a runtime placeholder (resolved by MapServer VALIDATION block).

        Replaces SELECT * with an explicit column list built from:
          - mandatory columns: geom, gid, label
          - symbol_field (map_symbol) if provided
          - all fields listed in include_items
          - lang_fields alias expressions  (col_%language% AS alias)

        Case A — plain table ref:
            "geom FROM schema.table USING UNIQUE gid USING SRID=2056"
            → wraps in subquery with explicit column list

        Case B — existing subquery:
            "geom FROM (SELECT col1, col2, ... FROM ...) AS sub USING ..."
            → appends alias expressions to the existing SELECT list,
              and ensures mandatory columns are present
        """
        if not lang_fields:
            return data

            # 1. Define Priority Groups
        head_names = ["gid", geom_col or "geom"]
        tail_names = [n for n in [symbol_field, label_col or "label"] if n]
        include_list = [c.strip() for c in (include_items or "").split(",") if c.strip()]
        lang_aliases = set(lang_fields.keys())

        # 2. Parsing the MapServer DATA string
        # Flag (?i) moved to the absolute start
        pattern = r"(?i)^\s*(?P<geom_ref>\S+)\s+FROM\s+(?P<source>.*?)\s+(?P<clauses>USING.*)$"
        match = re.match(pattern, data.strip(), re.DOTALL)

        if not match:
            return f"{geom_col or 'geom'} FROM ({data}) AS lang_sub USING UNIQUE gid"

        geom_ref = match.group("geom_ref")
        source_content = match.group("source").strip()
        using_clauses = match.group("clauses").strip()

        # FIXED REGEX: (?i) is now at the start, before the literal escaped parenthesis \(
        subquery_pattern = r"(?i)\(\s*SELECT\s+(?P<cols>.*?)\s+FROM\s+(?P<table>.*?)\)\s+AS\s+(?P<alias>\w+)"
        subquery_match = re.match(subquery_pattern, source_content, re.DOTALL)

        existing_cols = []
        if subquery_match:
            existing_cols = [c.strip() for c in subquery_match.group("cols").split(",") if c.strip()]
            from_table = subquery_match.group("table").strip()
            alias_name = subquery_match.group("alias")
        else:
            from_table = source_content
            alias_name = "lang_sub"

        # 3. Assemble the ordered column dictionary
        final_cols = {}

        def add_col(name: str, expr: str, force_tail: bool = False):
            if name in tail_names and not force_tail:
                return
            if name not in final_cols:
                final_cols[name] = expr

        # TIER 1: Technical Head
        for col in head_names:
            add_col(col, col)

        # TIER 2: Main Body (Include items + Translations)
        for col in include_list:
            if col in lang_aliases:
                pattern = lang_fields[col]
                if pattern:  # skip None/empty lang_fields entries
                    add_col(col, f"{pattern} AS {col}")
                else:
                    add_col(col, col)
            else:
                add_col(col, col)

        # TIER 3: Residual Translations
        for alias, pattern_str in lang_fields.items():
            if pattern_str:  # skip None/empty lang_fields entries
                add_col(alias, f"{pattern_str} AS {alias}")

        # TIER 4: Residual Subquery columns
        for col_expr in existing_cols:
            clean_name = col_expr.split()[-1] if " AS " in col_expr.upper() else col_expr
            add_col(clean_name, col_expr)

        # TIER 5: Technical Tail
        for col in tail_names:
            add_col(col, col, force_tail=True)

        # 4. Final Reconstruction — multi-line for readability
        indent = "\n      "
        cols_formatted = indent.join(
            f"{expr}," for expr in list(final_cols.values())[:-1]
        ) + indent + list(final_cols.values())[-1]
        return (
            f"{geom_ref} FROM (\n"
            f"    SELECT\n"
            f"      {cols_formatted}\n"
            f"    FROM {from_table}\n"
            f"  ) AS {alias_name} {using_clauses}"
        )

    def _build_validation_block(
            self,
    ) -> List[str]:
        """
        Generate MapServer VALIDATION block for %lang% runtime substitution.

        The regex anchors exact matches to prevent SQL injection, e.g.:
            "lang" "^(eng|ger|fra|ita)$"
        """
        languages = self.languages or [self.default_language]
        default_language = self.default_language

        lang_pattern = "|".join(re.escape(l) for l in languages)
        return [
            "  VALIDATION",
            f'    "lang"         "^({lang_pattern})$"',
            f'    "default_lang" "{default_language}"',
            "  END",
        ]

    def _inject_data_filter(self, data: str, filter_expr: str) -> str:
        """
        Append a YAML filter expression to the WHERE clause of a MapServer DATA string.

        Field names are normalized from GDB uppercase (KIND, RUNC_LITHO, ABOR_DEPTH_BEDR)
        to their PostGIS lowercase equivalents.  Layers with no filter are left untouched.
        """
        if not filter_expr:
            return data

        pg_filter = translate_esri_to_sql(filter_expr)
        for gdb_name, pg_name in _FILTER_FIELD_MAP.items():
            pg_filter = re.sub(rf"\b{gdb_name}\b", pg_name, pg_filter)

        outer = re.match(
            r"(?i)^\s*(?P<geom_ref>\S+)\s+FROM\s+(?P<source>.*?)\s+(?P<clauses>USING.*)$",
            data.strip(),
            re.DOTALL,
        )
        if not outer:
            return data

        geom_ref = outer.group("geom_ref")
        source = outer.group("source").strip()
        clauses = outer.group("clauses").strip()

        subq = re.match(
            r"(?i)\(\s*SELECT\s+(?P<cols>.*?)\s+FROM\s+(?P<from_clause>.*?)\s*\)\s+AS\s+(?P<alias>\w+)",
            source,
            re.DOTALL,
        )
        if subq:
            cols = subq.group("cols")
            from_clause = subq.group("from_clause").strip()
            alias = subq.group("alias")
            if re.search(r"\bWHERE\b", from_clause, re.IGNORECASE):
                new_from = f"{from_clause}\n      AND ({pg_filter})"
            else:
                new_from = f"{from_clause}\n    WHERE ({pg_filter})"
            return (
                f"{geom_ref} FROM (\n"
                f"    SELECT\n"
                f"      {cols}\n"
                f"    FROM {new_from}\n"
                f"  ) AS {alias} {clauses}"
            )
        else:
            # Plain table ref: wrap in a subquery
            return (
                f"{geom_ref} FROM (\n"
                f"    SELECT * FROM {source}\n"
                f"    WHERE ({pg_filter})\n"
                f"  ) AS filter_sub {clauses}"
            )

    # ========================================================================
    # LAYER AND CLASS GENERATION
    # ========================================================================

    def generate_layer(
            self,
            classification: LayerClassification,
            layer_name: str,
            layer_type: str,
            symbol_field: str,
            symbol_prefix: str = "class",
            connection: Optional[MapserverConnection] = None,
            data: Optional[str] = None,
            template: Optional[str] = "empty",
            layer_group: Optional[str] = None,
            map_label: Optional[Union[None, bool, str]] = None,
            layer_max_scale: Optional[bool | int] = None,
            layer_min_scale: Optional[bool | int] = None,
            include_items: Optional[str] = 'all',
            mapfile_config: Optional[MapfileGenerationConfig] = None,
            staging_mode: bool = False,
            lang_fields: Optional[Dict[str, str]] = None,
            translations: Optional[Dict[str, Dict[str, str]]] = None,
            data_filter: Optional[str] = None,

    ) -> str:
        """
        Generate complete MapServer LAYER block.

        Args:
          classification: `gcover.publish.esri_classification_extractor.LayerClassification` with styling
          layer_name: Name for the layer
          layer_type: MapServer layer type (e.g., POLYGON)
          symbol_field: Attribute used for symbol selection
          symbol_prefix: Prefix for symbol IDs (default: "class")
          connection: Optional MapServer connection object
          data: Optional data source path
          template: Template name (default: "empty")
          layer_group: Optional layer group name
          map_label: Optional label override
          layer_max_scale: Optional max scale
          layer_min_scale: Optional min scale
          include_items: Items to include in the LAYER block
          mapfile_config: Mapfile generation configuration

        Returns:
            Complete LAYER block as a string.
        """

        label_info = None
        rotation_field = None
        label_item = None

        if layer_type:
            if isinstance(layer_type, LayerType):
                layer_type = layer_type.name

        else:
            layer_type = self.layer_type  # e.g. POLYGON

        if not symbol_field:
            symbol_field = self.symbol_field

        if not template:
            template = "empty"

        logger.info(f"=== {layer_name} ===")

        def _get_label_field(classification, map_label):


          try:
            if classification.label_classes:
                label_info = classification.label_classes[0]
                label_item = label_info.get_simple_field_name()
          except Exception:
            logger.error("Error while retrieving labels info")

          if label_item and map_label is None:
            return label_item.lower()
          elif isinstance(map_label, str):
            return map_label.lower()
          return None

        # --- Rotation extraction ---

        try:
            rotation_info = classification.rotation_info
            if rotation_info:
                console.print(f"=== Rotation field ===")
                if mapfile_config and mapfile_config.rotation_field is not None:
                    # TODO: ugly
                    rotation_info.field_name = mapfile_config.rotation_field
                    console.print(f"[yellow]Using  {mapfile_config.rotation_field} from config YAML[/yellow]")

                else:
                    rotation_field = rotation_info.field_name
                    console.print(f"Using {rotation_field} from ESRI .lyrx")
        except Exception:
            logger.error("Error while retrieving rotation info")

        # Resolve language-specific column aliases in DATA string
        if data and (lang_fields or "%lang%" in (data or "")):
            label_col = _get_label_field(classification, map_label)
            data = self._build_lang_data(
                data=data,
                lang_fields=lang_fields or {},
                include_items=include_items,
                symbol_field=self.symbol_field,  # e.g. "map_symbol"
                geom_col="geom",  # or pull from connection config
                label_col=label_col,
            )

        # Append YAML filter to the DATA subquery WHERE clause
        if data and data_filter:
            data = self._inject_data_filter(data, data_filter)

        def build_layer_block(
                classification,
                layer_name: str,
                layer_group: str,
                layer_type: str,
                symbol_field: str,
                include_items: str = "all",
                template: str = "empty",
                connection=None,
                data=None,
                map_label=None,
                layer_max_scale=None,
                layer_min_scale=None,
                mapfile_config: Optional[MapfileGenerationConfig] = None,
        ):
            """Generate a MapServer LAYER block as a list of lines."""

            lines = []

            # --- Label extraction ---
            label_item = None
            try:
                if classification.label_classes:
                    label_info = classification.label_classes[0]
                    label_item = label_info.get_simple_field_name()
            except Exception:
                logger.error("Error while retrieving labels info")


            # --- Base LAYER header ---
            lines.extend(
                [
                    "",
                    f'  NAME "{layer_name}"'
                ]
            )
            if layer_group:
                lines.append( f'  GROUP "{layer_group}"')
            lines.extend(
                [

                    f"  TYPE {layer_type}",
                    "  STATUS ON",
                    "",
                ]
            )


            # --- Metadata ---
            lines.extend(
                [
                    "",
                    "  METADATA",
                    f'    "wms_title"    "{layer_name.capitalize()}"']
            )
            if layer_group:
                lines.append(f'    "wms_enable_request" "*"')
                # lines.append(f'    "wms_layer_group"      "/{layer_group}"')
            else:
                lines.append('    "wms_enable_request" "*"')
            if mapfile_config:
                metadata = getattr(mapfile_config, "metadata", None)
                if metadata:
                    for metadata_key in ["wms_group_title"]:
                        metadata_value = metadata.get(metadata_key)
                        if metadata_value:
                            lines.append(f'    "{metadata_key}" "{metadata_value}"')




            lines.extend(
                    [
                    f'    "wms_abstract" "{layer_name.capitalize()}"',
                    '    "ows_srs"      "EPSG:2056 EPSG:21781 EPSG:4326 EPSG:3857 EPSG:3034 EPSG:3035 EPSG:4258 EPSG:25832 EPSG:25833 EPSG:31467 EPSG:32632 EPSG:32633 EPSG:900913"',
                    '    "wms_extent" "2300000 900000 3100000 1450000"',
                    f'    "wms_include_items" "{include_items}"',
                    f'    "gml_include_items" "{include_items}"',
                    '    "gml_types" "auto"',
                    "  END",
                ]
            )

            # VALIDATION block for %lang% runtime substitution
            # Required whenever lang_fields or %lang% appears in the DATA string
            if lang_fields or (data and "%lang%" in data):
                lines.extend(
                    self._build_validation_block(
                    )
                )

            # Special case
            if "bedrock" in layer_name and layer_group and "geocover" in layer_group:
                lines.insert(-1, '    "wms_group_title"  "GeoCover 2D"')

            # --- Data source ---
            if connection:
                lines.extend(
                    [
                        "",
                        f"  CONNECTIONTYPE {connection.connection_type.name}",
                        f'  CONNECTION "{connection.connection}"',
                        f'  DATA "{data}"',
                    ]
                )

            # --- Scale handling ---
            # Names are swapped between Mapserver and ESRI
            # minScale → MAXSCALEDENOM
            # maxScale → MINSCALEDENOM

            if not self.no_scale:
                console.print(f"=== Scales hell ===")
                console.print(f"    layer_max_scale (MAXSCALEDOM): {layer_max_scale}")
                console.print(f"    layer_min_scale (MINSCALEDENOM): {layer_max_scale}")
                console.print(
                    f"    ESRI lyrx classification.max_scale: {classification.max_scale}"
                )
                console.print(
                    f"    ESRI lyrx  classification.min_scale: {classification.min_scale}"
                )
                console.print(f"    maxscale: {layer_max_scale}")

                max_scale_line = self.render_maxscale(
                    layer_max_scale, classification.min_scale
                )
                if max_scale_line:
                    lines.extend(["", max_scale_line])

                if classification.max_scale:
                    lines.extend(
                        [
                            "",
                            f"MINSCALEDENOM   {classification.max_scale}",
                        ]
                    )

            # Tolerance
            if mapfile_config:
                tolerance = getattr(mapfile_config, "tolerance", None)
                if tolerance:
                    lines.append(f'  TOLERANCE      {tolerance}')
                    lines.append(f'  TOLERANCEUNITS pixels')




            # --- Projection ---
            lines.extend(
                [
                    "",
                    "  PROJECTION",
                    '    "init=epsg:2056"',
                    "  END",
                    "",
                    "  EXTENT 2300000 900000 3100000 1450000",
                ]
            )

            # --- Template ---
            lines.extend(["", f'  TEMPLATE "{template}"', ""])

            # --- CLASSITEM ---
            if self.use_symbol_field:
                lines.append("  # Using CLASSITEM for simplified expressions")
                lines.append(f'  CLASSITEM "{symbol_field}"')
            else:
                lines.append("  # Styled using classification field values")

            # --- LABELITEM ---
            if label_item and map_label is None:
                lines.append(f'  LABELITEM "{label_item.lower()}"')
            elif isinstance(map_label, str):
                lines.append(f'  LABELITEM "{map_label.lower()}"')

            lines.append("")

            return lines

        # Generate layer block

        layer_block = build_layer_block(
                classification,
                layer_name,
                layer_group,
                layer_type,
                symbol_field,
                include_items,
                template,
                connection,
                data,
                map_label,
                layer_max_scale,
                layer_min_scale,
                mapfile_config,
        )

        # Generate CLASS blocks
        field_names = [f.name for f in classification.fields]


        # TODO moved to _generate_all_classes()
        all_classes_blocks = self._generate_all_classes(
            classification,
            field_names,
            symbol_prefix,
            rotation_info,
            label_info,
            map_label,
            mapfile_config,
        )

        # === DECISION: Inline vs Include mode ===

        # Check if we should use .inc file
        use_inc_file = False
        classes_mode = None

        lines = []

        if mapfile_config:
            classes_mode = getattr(mapfile_config, "classes_mode", None)
            use_inc_file = classes_mode in ("regenerate", "frozen")


        # use_inc_file = True  # TODO: remove

        console.print(f"Using include: {use_inc_file} (classes mode: {classes_mode})", style="magenta")

        # ========================================
        # PART 4A: INLINE MODE (default)
        # ========================================

        if not use_inc_file:
            # Mode inline - comme avant

            lines.extend(
                [
                    "LAYER",
                    *layer_block,
                    all_classes_blocks,
                    "",
                    "END # LAYER",
                ]
            )

            return "\n".join(lines)
        # ========================================
        # PART 4B: INCLUDE MODE
        # ========================================

        else:
            # Determine output directory
            # Tu dois avoir un output_dir quelque part - soit passé en paramètre
            # soit accessible via self
            # Pour l'instant, je suppose que tu as self.output_dir ou il faut l'ajouter

            # Si tu n'as pas output_dir, il faut l'ajouter comme paramètre
            # Pour l'instant, supposons qu'on peut le déduire

            console.print(f"=== INCLUDE MODE ===")

            # Determine classes file path
            classes_file = self._get_classes_file_path(
                layer_name,
                symbol_prefix,
                mapfile_config,
            )

            # Add header to classes
            classes_with_header = self._add_classes_header(
                all_classes_blocks, classification, symbol_prefix
            )

            # === STAGING MODE ===
            if staging_mode:
                logger.info(f"Generating staging file for {layer_name}")
                staging_file = self.generate_to_staging(
                    classes_file, classes_with_header
                )
                return str(staging_file)  # Return Path as string

            # === DECIDE: Regenerate or preserve ===
            '''should_regen = self.should_regenerate_classes(
                classes_file, classes_mode, force=force_regenerate
            )'''

            should_regen = True  # TODO

            if should_regen:
                logger.info(f"Writing classes to {classes_file}")
                classes_file.parent.mkdir(parents=True, exist_ok=True)
                classes_file.write_text(classes_with_header)
            else:
                logger.info(f"Preserving {classes_file} (mode: frozen)")

            # Build LAYER with INCLUDE
            # Determine relative path (relatif à où le .map sera écrit)
            # Pour simplifier, assume que classes/ est relatif à output_dir
            relative_path = f"layers/classes/{classes_file.name}"  # relative to geocover.map

            lines.extend(
                [
                    "LAYER",
                    *layer_block,
                    f'  INCLUDE "{relative_path}"',
                    "",
                    "END # LAYER",
                ]
            )

            return "\n".join(lines)






        #  TODO  ---

        '''for idx, class_obj in enumerate(classification.classes):
            if not class_obj.visible:
                continue
            # NOUVEAU: Extraire la valeur depuis l'identifier si disponible
            identifier_value = idx  # Par défaut: utiliser l'index

            if hasattr(class_obj, "identifier") and class_obj.identifier:
                try:
                    # Extraire la valeur depuis le ClassIdentifier
                    identifier_key = class_obj.identifier.to_key()
                    # La dernière partie contient la valeur (après le "::")
                    identifier_value = identifier_key.split("::")[-1]

                    logger.debug(
                        f"Using identifier value '{identifier_value}' "
                        f"for class '{class_obj.label}' (instead of index {idx})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not extract identifier value for '{class_obj.label}': {e}, "
                        f"using index {idx}"
                    )
                    identifier_value = idx

            # Construire le symbol_id avec la valeur extraite
            symbol_id = f"{symbol_prefix}_{identifier_value}"

            logger.debug(f"class idx: {symbol_id}")

            class_block = self.generate_class(
                class_obj,
                field_names,
                identifier_value,
                symbol_prefix,
                rotation_info,
                label_info,
                map_label,
                mapfile_config,
            )
            lines.append(class_block)

        lines.append("END # LAYER")

        return "\n".join(lines)'''

    def _generate_all_classes(
            self,
            classification,
            field_names: List[str],
            symbol_prefix: str,
            rotation_info: Optional[RotationInfo] = None,
            label_info: Optional[LabelInfo] = None,
            map_label: Optional[Union[None, bool, str]] = None,
            mapfile_config=None,
    ) -> str:
        """
        Generate ALL CLASS blocks for a classification.

        This method contains the loop that calls generate_class() multiple times.
        Returns all CLASS blocks as a single string.
        """
        all_classes = []

        if mapfile_config and mapfile_config.symbol_scale:
            all_classes.extend([
                "# Fixing symbol size",
                f" SYMBOLSCALEDENOM {mapfile_config.symbol_scale}",
                ""

            ])

        for idx, class_obj in enumerate(classification.classes):
            if not class_obj.visible:
                continue
            # NOUVEAU: Extraire la valeur depuis l'identifier si disponible
            identifier_value = idx  # Par défaut: utiliser l'index

            if hasattr(class_obj, "identifier") and class_obj.identifier:
                try:
                    # Extraire la valeur depuis le ClassIdentifier
                    identifier_key = class_obj.identifier.to_key()
                    # La dernière partie contient la valeur (après le "::")
                    identifier_value = identifier_key.split("::")[-1]

                    logger.debug(
                        f"Using identifier value '{identifier_value}' "
                        f"for class '{class_obj.label}' (instead of index {idx})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not extract identifier value for '{class_obj.label}': {e}, "
                        f"using index {idx}"
                    )
                    identifier_value = idx

            # Construire le symbol_id avec la valeur extraite
            symbol_id = f"{symbol_prefix}_{identifier_value}"

            logger.debug(f"class idx: {symbol_id}")

            class_block = self.generate_class(
                class_obj,
                field_names,
                identifier_value,
                symbol_prefix,
                rotation_info,
                label_info,
                map_label,
                mapfile_config,
            )

            all_classes.append(class_block)

        # Retourne TOUS les CLASS blocks comme une seule string
        return "\n".join(all_classes)

    def generate_class(
            self,
            class_obj,
            field_names: List[str],
            class_index: int,
            symbol_prefix: str = "class",
            rotation_info: Optional[RotationInfo] = None,
            label_info: Optional[LabelInfo] = None,
            map_label: Optional[Union[None, bool, str]] = None,
            mapfile_config=None,
    ) -> str:
        """Generate a single MapServer CLASS block."""
        if not class_obj.visible:
            return ""



        # Get symbol adjustments if available
        symbol_adjustments = None
        if mapfile_config and hasattr(mapfile_config, 'symbol_adjustments'):
            symbol_adjustments = mapfile_config.symbol_adjustments


        # Generate expression
        if self.use_symbol_field:
            symbol_id = f"{symbol_prefix}_{class_index}"
            expression = self._generate_expression_from_symbol(symbol_id)
        else:
            expression = self._generate_expression_from_fields(class_obj, field_names)

        # Sanitize class name
        class_name = class_obj.label.replace('"', '\\"')
        logger.debug(f"  {class_name}")

        # Build CLASS block
        lines = [
            "  CLASS",
            f'    NAME "{class_name}"',
            f"    EXPRESSION {expression}",
        ]

        # Add LABEL
        if label_info and map_label is not False:
            field = label_info.get_simple_field_name()
            color = label_info.font_color.to_rgb_tuple()

            # NEW: Apply size adjustment to labels if specified
            label_size = 8
            if symbol_adjustments and symbol_adjustments.point_size_multiplier != 1.0:
                label_size = int(label_size * symbol_adjustments.point_size_multiplier)

            lines.extend(
                [
                    "    LABEL",
                    f"        COLOR {' '.join(list(map(str, color)))}",
                    '         FONT "sans"',
                    "         TYPE truetype",
                    f"        SIZE {label_size}",
                    "         POSITION AUTO",
                    "         PARTIALS FALSE",
                    "    END",
                ]
            )


        # Add STYLE blocks based on layer type
        if self.layer_type == "POLYGON":
                self._add_polygon_styles(
                    lines,
                    class_obj,
                    class_index,
                    symbol_prefix,
                    symbol_adjustments=symbol_adjustments  # NEW: Pass adjustments
                )
        elif self.layer_type == "LINE":
                lines.append("    STYLE")
                self._add_line_style(
                    lines,
                    class_obj,
                    class_index,
                    symbol_prefix,
                    symbol_adjustments=symbol_adjustments  # NEW: Pass adjustments
                )
                lines.append("    END # STYLE")
        elif self.layer_type == "POINT":
                lines.append("    STYLE")
                self._add_point_style(
                    lines,
                    class_obj,
                    class_index,
                    symbol_prefix,
                    rotation_info,
                    symbol_adjustments=symbol_adjustments  # NEW: Pass adjustments
                )
                lines.append("    END # STYLE")

        lines.append("  END # CLASS")
        lines.append("")

        return "\n".join(lines)

    # ========================================================================
    # EXPRESSION GENERATION
    # ========================================================================

    def _generate_expression_from_fields(
            self, class_obj, field_names: List[str]
    ) -> str:
        """Generate MapServer EXPRESSION from classification field values."""
        expressions = []

        for field_values in class_obj.field_values:
            conditions = []

            for field_name, expected_value in zip(field_names, field_values):
                if expected_value == "<Null>":
                    conditions.append(
                        f'([{field_name}] eq "" OR NOT [DEFINED_{field_name}])'
                    )
                elif expected_value == "999997":
                    conditions.append(f'([{field_name}] eq "999997")')
                elif expected_value == "999999":
                    conditions.append(f'([{field_name}] eq "999999")')
                else:
                    conditions.append(f'([{field_name}] eq "{expected_value}")')

            if len(conditions) == 1:
                expressions.append(conditions[0])
            else:
                expressions.append(f"({' AND '.join(conditions)})")

        if len(expressions) == 1:
            return expressions[0]
        else:
            return f"({' OR '.join(expressions)})"

    def _generate_expression_from_symbol(self, symbol_id: str) -> str:
        """
        Generate simple MapServer EXPRESSION using SYMBOL field.

        When CLASSITEM is used, the expression is just the value string.
        """
        if self.use_symbol_field:
            return f'"{symbol_id}"'
        else:
            return f'([{self.symbol_field}] eq "{symbol_id}")'

    # ========================================================================
    # POLYGON STYLING (COMPLEX MULTI-LAYER SUPPORT)
    # ========================================================================

    def _add_polygon_styles(
            self,
            lines: List[str],
            class_obj,
            class_index: int,
            symbol_prefix: str,
            symbol_adjustments=None,  # NEW
    ) -> None:
        """
        Add MapServer STYLE blocks for polygon symbols.

        NEW: Applies symbol_adjustments to all style properties.
        """
        # Try full_symbol_layers first
        if hasattr(class_obj, "full_symbol_layers") and class_obj.full_symbol_layers:
            self._add_polygon_styles_from_full_layers(
                lines,
                class_obj.full_symbol_layers,
                class_index,
                symbol_prefix,
                symbol_adjustments=symbol_adjustments  # NEW
            )
            return

        # Fallback to legacy extraction
        if not hasattr(class_obj, "symbol_info") or not class_obj.symbol_info:
            self._add_simple_polygon_style(lines)
            return

        symbol_info = class_obj.symbol_info

        if not hasattr(symbol_info, "raw_symbol") or not symbol_info.raw_symbol:
            self._add_simple_polygon_style(lines)
            return

        # Extract all symbol layers
        layers_info = extract_polygon_symbol_layers(symbol_info.raw_symbol)

        has_fill = False

        # Add character marker pattern fills first
        for i, marker_info in enumerate(layers_info.character_markers):
            self._add_pattern_fill_style(
                lines,
                marker_info,
                class_index,
                i,
                symbol_prefix,
                symbol_adjustments=symbol_adjustments  # NEW
            )
            has_fill = True

        # Add solid fills (with transparency adjustment)
        for fill_info in layers_info.fills:
            if fill_info["type"] == "solid":
                self._add_solid_fill_style(
                    lines,
                    fill_info["color"],
                    symbol_adjustments=symbol_adjustments  # NEW
                )
                has_fill = True

        # Add outline
        if layers_info.outline:
            self._add_outline_style(
                lines,
                layers_info.outline,
                symbol_adjustments=symbol_adjustments  # NEW
            )

        if not has_fill:
            if layers_info.outline:
                self._add_default_fill_style(lines)
            else:
                self._add_simple_polygon_style(lines)

    def _add_polygon_styles_from_full_layers(
            self,
            lines: List[str],
            full_layers,  # SymbolLayersInfo object
            class_index: int,
            symbol_prefix: str,
            symbol_adjustments=None,  # NEW: Added parameter
    ) -> None:
        """
        NEW: Add polygon styles using complete SymbolLayersInfo.

        Renders all fill types in proper order.

        Args:
            lines: List of output lines
            full_layers: SymbolLayersInfo object with complete symbol data
            class_index: Index of this class
            symbol_prefix: Prefix for symbol names
            symbol_adjustments: NEW - SymbolAdjustments to apply
        """
        has_fill = False

        # Process fills in order
        for i, fill_info in enumerate(full_layers.fills):
            fill_type = fill_info.get("type", "solid")

            if fill_type == "solid":
                self._add_solid_fill_style(
                    lines,
                    fill_info["color"],
                    symbol_adjustments=symbol_adjustments  # NEW: Pass adjustments
                )
                has_fill = True

            elif fill_type == "hatch":
                logger.debug("Found hatch")
                self._add_hatch_fill_style(
                    lines,
                    fill_info,
                    class_index,
                    i,
                    symbol_prefix,
                    symbol_adjustments=symbol_adjustments  # NEW: Pass adjustments
                )
                has_fill = True

            elif fill_type == "character":
                marker = fill_info.get("marker_info")
                if marker:
                    self._add_pattern_fill_style(
                        lines,
                        marker,
                        class_index,
                        i,
                        symbol_prefix,
                        symbol_adjustments=symbol_adjustments  # NEW: Pass adjustments
                    )
                    has_fill = True

            elif fill_type == "picture":
                logger.warning(f"Picture fill not yet supported in MapServer")

            elif fill_type == "gradient":
                logger.warning(f"Gradient fill not yet supported in MapServer")

        # Add outline (top layer)
        if full_layers.outline:
            self._add_outline_style(
                lines,
                full_layers.outline,
                symbol_adjustments=symbol_adjustments  # NEW: Pass adjustments
            )

        # Fallback if no fills
        if not has_fill:
            if full_layers.outline:
                self._add_default_fill_style(lines)
            else:
                self._add_simple_polygon_style(lines)

    def _add_hatch_fill_style(
            self,
            lines: List[str],
            hatch_info: Dict,
            class_index: int,
            fill_index: int,
            symbol_prefix: str,
            symbol_adjustments=None,  # NEW: Added parameter
    ) -> None:
        """
        Add MapServer STYLE for hatch fill pattern.

        Uses the generic "hatchsymbol" with customization in STYLE.

        NEW: Applies line_width_multiplier and dash_pattern_override to hatch lines.
        """
        rotation = hatch_info["rotation"]
        separation = hatch_info["separation"]
        line_sym = hatch_info["line_symbol"]

        # Extract line properties
        line_color = line_sym["color"]
        line_width = line_sym["width"]
        line_style = line_sym.get("line_style", {})
        r, g, b, a = line_color

        # Register that we need hatchsymbol (just once)
        if not any(s.get("type") == "hatch" for s in self.pattern_symbols):
            self.pattern_symbols.append({"type": "hatch"})

        # Convert to pixels
        separation_px = separation * 1.33
        width_px = line_width * 1.33

        # NEW: Apply width adjustment to hatch lines
        if symbol_adjustments and symbol_adjustments.line_width_multiplier != 1.0:
            width_px *= symbol_adjustments.line_width_multiplier

        # Add STYLE block
        lines.append("    STYLE")
        lines.append('      SYMBOL "hatchsymbol"')
        lines.append(f"      COLOR {r} {g} {b}")
        lines.append(f"      ANGLE {rotation}")
        lines.append(f"      SIZE {separation_px:.2f}")
        lines.append(f"      WIDTH {width_px:.2f}")

        # Add PATTERN if the line itself is dashed/dotted
        # NEW: Apply dash pattern override if specified
        if symbol_adjustments and symbol_adjustments.dash_pattern_override:
            pattern = ' '.join(str(int(p)) for p in symbol_adjustments.dash_pattern_override)
            lines.append(f"      PATTERN {pattern} END")
        elif line_style.get("type") in ["dash", "dot"] and line_style.get("pattern"):
            pattern = line_style["pattern"]
            pattern_str = " ".join(str(int(p)) for p in pattern)
            lines.append(f"      PATTERN {pattern_str} END")

        # NEW: Apply transparency override
        final_alpha = a
        if symbol_adjustments and symbol_adjustments.transparency_override is not None:
            final_alpha = int(symbol_adjustments.transparency_override * 2.55)

        if final_alpha < 255:
            opacity = int((final_alpha / 255) * 100)
            lines.append(f"      OPACITY {opacity}")

        lines.append("    END # STYLE")

    def _generate_hatch_pattern_symbols(self) -> List[str]:
        """
        Generate single generic HATCH symbol.

        MapServer HATCH symbols are customized entirely in the STYLE block,
        so we only need one generic hatch symbol definition.
        """
        # Only generate if we actually have hatch fills
        has_hatch = any(s.get("type") == "hatch" for s in self.pattern_symbols)

        if not has_hatch:
            return []

        return [
            "  SYMBOL",
            '    NAME "hatchsymbol"',
            "    TYPE HATCH",
            "  END",
            "",
        ]

    def _add_pattern_fill_style(
            self,
            lines: List[str],
            marker_info: CharacterMarkerInfo,
            class_index: int,
            marker_index: int,
            symbol_prefix: str,
            symbol_adjustments=None,  # NEW
    ) -> None:
        """
        Add MapServer STYLE for character marker pattern fill.

        ENHANCED: Uses PIXMAP symbol from catalog if available,
        falls back to TRUETYPE (which doesn't work for polygons but
        maintains backward compatibility).
        """
        r, g, b, a = marker_info.color

        # Try to find pattern in catalog first
        if self.pattern_catalog.loaded:
            pattern = self.pattern_catalog.find_pattern(
                font_family=marker_info.font_family,
                char_index=marker_info.character_index,
                size=marker_info.size,
                step_x=marker_info.step_x,
                step_y=marker_info.step_y,
                color=marker_info.color,
            )

            if pattern:
                symbol_name = self.pattern_catalog.get_pixmap_symbol_name(pattern)
                if symbol_name:
                    # Use PIXMAP symbol from catalog
                    self.pixmap_symbols_used.add(symbol_name)

                    lines.append("    STYLE")
                    lines.append(f'      SYMBOL "{symbol_name}"')
                    # Note: Color is baked into PNG, no COLOR needed
                    if a < 255:
                        lines.append(f"      OPACITY {a}")
                    lines.append("    END # STYLE")

                    logger.debug(f"Using PIXMAP symbol '{symbol_name}' for pattern fill")
                    return

        # FALLBACK: Use TRUETYPE symbol (legacy behavior)
        # Note: This doesn't actually work for polygon fills in MapServer,
        # but we keep it for backward compatibility and to show intent
        font_name = sanitize_font_name(marker_info.font_family)
        char_index = marker_info.character_index
        symbol_name = f"{font_name}_{char_index}"

        # Track font usage
        self.fonts_used.add(marker_info.font_family)

        # Convert points to pixels
        size_px = marker_info.size * 1.33


        # NEW: Apply size adjustment
        if symbol_adjustments and symbol_adjustments.point_size_multiplier != 1.0:
            size_px *= symbol_adjustments.point_size_multiplier

        lines.append("    STYLE")
        lines.append(f'      SYMBOL "{symbol_name}"')
        lines.append(f"      COLOR {r} {g} {b}")

        # NEW: Apply transparency override if specified
        final_alpha = a
        if symbol_adjustments and symbol_adjustments.transparency_override is not None:
            final_alpha = int(symbol_adjustments.transparency_override * 2.55)

        if final_alpha < 255:
            lines.append(f"      OPACITY {final_alpha}")

        lines.append(f"      SIZE {size_px:.1f}")
        lines.append("    END # STYLE")

        # Warn if no catalog available
        if not self.pattern_catalog.loaded:
            logger.warning(
                f"No pattern catalog - using TRUETYPE for polygon fill "
                f"(font={font_name}, char={char_index}). "
                f"This may not render correctly. "
                f"Generate a pattern catalog with: gcover publish symbols inventory"
            )

    def _add_solid_fill_style(
            self,
            lines: List[str],
            color: Tuple[int, int, int, int],
            symbol_adjustments=None,  # NEW
    ) -> None:
        """Add MapServer STYLE for solid fill."""
        r, g, b, a = color

        lines.append("    STYLE")
        lines.append(f"      COLOR {r} {g} {b}")

        # NEW: Apply transparency override if specified
        final_alpha = a
        if symbol_adjustments and symbol_adjustments.transparency_override is not None:
            final_alpha = int(symbol_adjustments.transparency_override * 2.55)

        if final_alpha < 255:
            lines.append(f"      OPACITY {final_alpha}")

        lines.append("    END # STYLE")

    def _add_outline_style(self, lines: List[str], outline_info: Dict, symbol_adjustments = None,) -> None:
        """Add MapServer STYLE for polygon outline."""
        r, g, b, a = outline_info["color"]
        width_px = outline_info["width"] * 1.33

        lines.append("    STYLE")
        lines.append(f"      OUTLINECOLOR {r} {g} {b}")
        lines.append(f"      WIDTH {width_px:.2f}")

        # Handle line style
        line_style_info = outline_info["line_style"]
        if line_style_info["type"] == "dash":
            lines.append('       PATTERN 5 3 END')
        elif line_style_info["type"] == "dot":
            lines.append('       PATTERN 1 3 END')

        lines.append("    END # STYLE")

    def _add_simple_polygon_style(self, lines: List[str]) -> None:
        """Add simple fallback polygon style with fill and outline."""
        lines.append("    STYLE")
        lines.append("      COLOR 128 128 128")
        lines.append("      OUTLINECOLOR 64 64 64")
        lines.append("    END # STYLE")

    def _add_default_fill_style(self, lines: List[str]) -> None:
        """Add a default fill (used when only outline is present)."""
        lines.append("    STYLE")
        lines.append("      COLOR 255 255 255")  # White fill
        lines.append("    END # STYLE")

    # ========================================================================
    # LINE STYLING
    # ========================================================================

    def _add_line_style(
            self,
            lines: List[str],
            class_obj,
            class_index: int,
            symbol_prefix: str,
            symbol_adjustments=None,  # NEW
    ):
        """Add line styling from ESRI symbol_info."""
        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Check if font marker on line
            if (
                    hasattr(symbol_info, "font_family")
                    and symbol_info.font_family
                    and hasattr(symbol_info, "character_index")
                    and symbol_info.character_index is not None
            ):
                # Get or create font symbol
                font_family = sanitize_font_name(symbol_info.font_family)
                char_index = symbol_info.character_index
                spec = FontSymbol(font_family, char_index)

                if spec not in self.symbol_registry:
                    font_symbol_name = f"{spec.font_family}_{spec.char_index}"
                    self.symbol_registry[spec] = font_symbol_name
                else:
                    font_symbol_name = self.symbol_registry[spec]

                self._add_truetype_line_marker(
                    lines,
                    symbol_info,
                    font_symbol_name,
                    symbol_adjustments=symbol_adjustments  # NEW
                )
                self.fonts_used.add(symbol_info.font_family)
            else:
                # Regular line
                self._add_regular_line_style(
                    lines,
                    symbol_info,
                    symbol_adjustments=symbol_adjustments  # NEW
                )
        else:
            lines.append("      COLOR 128 128 128")
            lines.append("      WIDTH 1.0")

    def _add_regular_line_style(
            self,
            lines: List[str],
            symbol_info,
            symbol_adjustments=None,  # NEW
    ) -> None:
        """Add regular line style (no font markers)."""
        if hasattr(symbol_info, "color") and symbol_info.color:
            color_info = symbol_info.color
            if hasattr(color_info, "r"):
                lines.append(
                    f"      COLOR {color_info.r} {color_info.g} {color_info.b}"
                )
            else:
                lines.append("      COLOR 128 128 128")
        else:
            lines.append("      COLOR 128 128 128")

        if hasattr(symbol_info, "width") and symbol_info.width:
            width = symbol_info.width * 1.33

            # NEW: Apply width adjustment

            if symbol_adjustments and symbol_adjustments.line_width_multiplier != 1.0:
                width *= symbol_adjustments.line_width_multiplier
                logger.debug(f"Line width adjustment: {width}")

            lines.append(f"      WIDTH {width:.2f}")
        else:
            lines.append("      WIDTH 1.0")

        # Dash pattern
        if (
                hasattr(symbol_info, "line_style")
                and symbol_info.line_style
                and hasattr(symbol_info, "dash_pattern")
                and symbol_info.dash_pattern
        ):
            line_style = symbol_info.line_style

            # NEW: Apply dash pattern override if specified
            if symbol_adjustments and symbol_adjustments.dash_pattern_override:
                pattern = ' '.join(str(int(p)) for p in symbol_adjustments.dash_pattern_override)
                lines.append(f'       PATTERN {pattern} END')
            elif line_style == "dash":
                lines.append('       PATTERN 5 3 END')
            elif line_style == "dot":
                lines.append('       PATTERN 1 3 END')

    def _add_truetype_line_marker(
            self,
            lines: List[str],
            symbol_info,
            font_symbol_name: str,
            symbol_adjustments=None,  # NEW
    ):
        """Add TrueType font marker on line."""
        lines.append(f'      SYMBOL "{font_symbol_name}"')

        if hasattr(symbol_info, "color") and symbol_info.color:
            color_info = symbol_info.color
            if hasattr(color_info, "r"):
                lines.append(
                    f"      COLOR {color_info.r} {color_info.g} {color_info.b}"
                )
            else:
                lines.append("      COLOR 128 128 128")
        else:
            lines.append("      COLOR 128 128 128")

        if hasattr(symbol_info, "size") and symbol_info.size:
            size = symbol_info.size * 1.33

            # NEW: Apply size adjustment
            if symbol_adjustments and symbol_adjustments.point_size_multiplier != 1.0:
                size *= symbol_adjustments.point_size_multiplier

            lines.append(f"      SIZE {size:.1f}")
        else:
            lines.append("      SIZE 10")

        lines.append("      GAP -30")

    # ========================================================================
    # POINT STYLING
    # ========================================================================

    def _add_point_style(
            self,
            lines: List[str],
            class_obj,
            class_index: int,
            symbol_prefix: str,
            rotation_info: Optional[RotationInfo] = None,
            symbol_adjustments=None,  # NEW
    ):
        """Add point styling from ESRI symbol_info."""
        if hasattr(class_obj, "symbol_info") and class_obj.symbol_info:
            symbol_info = class_obj.symbol_info

            # Check if font marker
            if (
                    hasattr(symbol_info, "font_family")
                    and symbol_info.font_family
                    and hasattr(symbol_info, "character_index")
                    and symbol_info.character_index is not None
            ):
                # Get or create font symbol
                font_family = sanitize_font_name(symbol_info.font_family)
                char_index = symbol_info.character_index
                spec = FontSymbol(font_family, char_index)

                if spec not in self.symbol_registry:
                    font_symbol_name = f"{spec.font_family}_{spec.char_index}"
                    self.symbol_registry[spec] = font_symbol_name
                else:
                    font_symbol_name = self.symbol_registry[spec]

                lines.append(f'      SYMBOL "{font_symbol_name}"')
                self.fonts_used.add(symbol_info.font_family)
            else:
                # Geometric marker
                marker_type = "circle"
                if hasattr(symbol_info, "symbol_type"):
                    type_str = str(symbol_info.symbol_type)
                    if "Square" in type_str:
                        marker_type = "square"
                    elif "Triangle" in type_str:
                        marker_type = "triangle"
                    elif "Star" in type_str:
                        marker_type = "star"

                lines.append(f'      SYMBOL "{marker_type}"')

            if rotation_info and rotation_info.field_name:
                lines.append(
                    f"      ANGLE [{rotation_info.field_name.lower()}]"
                )

            if hasattr(symbol_info, "size") and symbol_info.size:
                size = symbol_info.size * 1.33

                # NEW: Apply size adjustment
                if symbol_adjustments and symbol_adjustments.point_size_multiplier != 1.0:
                    size *= symbol_adjustments.point_size_multiplier

                lines.append(f"      SIZE {size:.1f}")
            else:
                lines.append("      SIZE 8")

            if hasattr(symbol_info, "color") and symbol_info.color:
                color_info = symbol_info.color
                if hasattr(color_info, "r"):
                    lines.append(
                        f"      COLOR {color_info.r} {color_info.g} {color_info.b}"
                    )
                else:
                    lines.append("      COLOR 128 128 128")
            else:
                lines.append("      COLOR 128 128 128")
        else:
            lines.append('      SYMBOL "circle"')
            lines.append("      SIZE 8")
            lines.append("      COLOR 128 128 128")

    def _scan_required_symbols(self, classification_list: List) -> None:
        """
        Scan classifications to determine which symbol types are needed.
        """
        for classification in classification_list:
            for class_obj in classification.classes:
                if not class_obj.visible:
                    continue

                # Check for full_symbol_layers
                if (
                        hasattr(class_obj, "full_symbol_layers")
                        and class_obj.full_symbol_layers
                ):
                    layers = class_obj.full_symbol_layers

                    # If it's a single SymbolLayersInfo
                    if isinstance(layers, SymbolLayersInfo):
                        fills = layers.fills
                    elif isinstance(layers, list):
                        # If it's a list of SymbolLayersInfo
                        fills = []
                        for l in layers:
                            if isinstance(l, SymbolLayersInfo):
                                fills.extend(l.fills)
                    else:
                        fills = []

                    for fill_info in fills:
                        if (
                                isinstance(fill_info, dict)
                                and fill_info.get("type") == "hatch"
                        ):
                            if not any(
                                    s.get("type") == "hatch" for s in self.pattern_symbols
                            ):
                                self.pattern_symbols.append({"type": "hatch"})
                            break

    # ========================================================================
    # SYMBOL FILE GENERATION
    # ========================================================================

    def generate_symbol_file(
            self, classification_list: List = None, prefixes: Dict = {}
    ) -> str:
        """
        Generate MapServer symbol file with all symbol types.

        Args:
            classification_list: List of LayerClassification objects
            prefixes: Layer name to prefix mapping

        Returns:
            Complete SYMBOLSET file content
        """
        # Pre-scan to see what symbol types we need
        if classification_list:
            self._scan_required_symbols(classification_list)

        lines = [
            "SYMBOLSET",
            "",
            "  # Basic geometric symbols for GeoCover (always available)",
            "",
        ]

        # Add basic geometric symbols
        lines.extend(self._generate_basic_symbols())

        # Add line pattern symbols (dashed, dotted)
        lines.append("")
        lines.append("  # Line pattern symbols")
        lines.append("")
        lines.extend(self._generate_line_pattern_symbols())

        # Add generic hatch symbol (if needed)
        hatch_symbols = self._generate_hatch_pattern_symbols()
        if hatch_symbols:
            lines.append("")
            lines.append(
                "  # Hatch pattern symbol (customize with ANGLE, SIZE, WIDTH in STYLE)"
            )
            lines.append("")
            lines.extend(hatch_symbols)

        # NEW: Add PIXMAP symbols from catalog
        pixmap_symbols = self._generate_pixmap_symbols()
        if pixmap_symbols:
            lines.append("")
            lines.append("  # PIXMAP pattern fill symbols (from pattern catalog)")
            lines.append("")
            lines.extend(pixmap_symbols)

        # Generate font symbols if classifications provided
        if classification_list:
            # Point and line font markers
            font_symbols = self._generate_font_symbols_from_classifications(
                classification_list, prefixes
            )
            if font_symbols:
                lines.append("")
                lines.append("  # TrueType font marker symbols (points and lines)")
                lines.append("")
                lines.extend(font_symbols)

            # Polygon pattern fills (legacy TRUETYPE - only if no catalog)
            if not self.pattern_catalog.loaded:
                pattern_symbols = self._generate_polygon_pattern_symbols(
                    classification_list
                )
                if pattern_symbols:
                    lines.append("")
                    lines.append("  # TrueType pattern fill symbols (polygons) - LEGACY")
                    lines.append("  # NOTE: TrueType symbols don't work for polygon fills!")
                    lines.append("  # Generate pattern catalog: gcover publish symbols inventory")
                    lines.append("")
                    lines.extend(pattern_symbols)

        lines.append("")
        lines.append("END # SYMBOLSET")

        return "\n".join(lines)

    def _generate_pixmap_symbols(self) -> List[str]:
        """
        Generate PIXMAP symbol definitions from pattern catalog.

        Only includes symbols that were actually used during generation.
        """
        if not self.pattern_catalog.loaded:
            return []

        if not self.pixmap_symbols_used:
            return []

        symbols = []

        for symbol_name in sorted(self.pixmap_symbols_used):
            # Find pattern in catalog to get PNG path
            png_path = None
            for key, pattern in self.pattern_catalog.patterns.items():
                mapserver_info = pattern.get('mapserver', {})
                if mapserver_info.get('symbol') == symbol_name:
                    png_path = mapserver_info.get('png_file')
                    break

            if not png_path:
                png_path = f"{DEFAULT_IMAGES_DIR}/{symbol_name}.png"

            symbols.extend([
                "  SYMBOL",
                f'    NAME "{symbol_name}"',
                "    TYPE PIXMAP",
                f'    IMAGE "{png_path}"',
                "  END",
                "",
            ])

        return symbols

    def _generate_basic_symbols(self) -> List[str]:
            """Load basic geometric symbols from an external file."""

            symbol_file = files("gcover.data").joinpath("basic_symbols.map")

            with symbol_file.open("r", encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f]

            return lines

    def _generate_line_pattern_symbols(self) -> List[str]:
        """Generate line pattern symbols (dashed, dotted)."""
        return ["# No dash SYMBOL"]

    def _generate_font_symbols_from_classifications(
            self, classification_list: List, prefixes: Dict
    ) -> List[str]:
        """Generate TrueType symbols for point and line markers."""
        symbols = []

        def escape_mapserver_string(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")

        # Generate symbols from registry
        images = []
        for spec in sorted(
                self.symbol_registry.keys(), key=lambda s: (s.font_family, s.char_index)
        ):
            font_symbol_name = self.symbol_registry[spec]
            font_name = spec.font_family
            character = escape_mapserver_string(chr(spec.char_index))

            # Generate font image for PDF
            image = generate_font_image(
                font_symbol_name, spec.font_family, spec.char_index
            )
            if image:
                images.append(image)

            symbols.extend(
                [
                    "  SYMBOL",
                    f'    NAME "{font_symbol_name}"',
                    "    TYPE TRUETYPE",
                    f'    FONT "{font_name}"',
                    f'    CHARACTER "{character}"',
                    "    FILLED TRUE",
                    "    ANTIALIAS TRUE",
                    "  END",
                    "",
                ]
            )

        # Save PDF of all used font symbols
        if images:
            images[0].save(
                "mapserver/font_characters.pdf",
                save_all=True,
                append_images=images[1:],
            )

        return symbols

    def _generate_polygon_pattern_symbols(self, classification_list: List) -> List[str]:
        """Generate TrueType symbols for polygon pattern fills (LEGACY)."""
        symbols = []
        symbols_generated = set()

        for classification in classification_list:
            for class_index, class_obj in enumerate(classification.classes):
                if not class_obj.visible:
                    continue

                if not hasattr(class_obj, "symbol_info") or not class_obj.symbol_info:
                    continue

                symbol_info = class_obj.symbol_info

                if not hasattr(symbol_info, "raw_symbol") or not symbol_info.raw_symbol:
                    continue

                # Extract symbol layers
                layers_info = extract_polygon_symbol_layers(symbol_info.raw_symbol)

                # Generate symbol for each character marker
                for marker_info in layers_info.character_markers:
                    font_name = sanitize_font_name(marker_info.font_family)
                    char_index = marker_info.character_index
                    symbol_name = f"{font_name}_{char_index}"

                    # Avoid duplicates
                    if symbol_name in symbols_generated:
                        continue

                    symbols_generated.add(symbol_name)
                    self.fonts_used.add(marker_info.font_family)

                    # Convert spacing from points to pixels
                    step_x_px = marker_info.step_x * 1.33
                    step_y_px = marker_info.step_y * 1.33
                    r, g, b, a = marker_info.color

                    html_code = f"&#{char_index};"
                    character = chr(char_index)

                    symbols.extend(
                        [
                            "  SYMBOL",
                            f'    NAME "{symbol_name}"',
                            "    TYPE TRUETYPE",
                            f'    FONT "{font_name}"',
                            f"    CHARACTER \"{html_code}\"   # → '{character}'",
                            "    FILLED TRUE",
                            "    ANTIALIAS TRUE",
                            f"    # Color: RGB({r}, {g}, {b})",
                            f"    # Grid spacing: {step_x_px:.1f}x{step_y_px:.1f} px",
                            "  END",
                            "",
                        ]
                    )

        return symbols

    # ========================================================================
    # FONTSET GENERATION
    # ========================================================================

    def generate_fontset(self, font_paths: Dict[str, str] = None) -> str:
        """Generate MapServer FONTSET file."""
        lines = [
            "# MapServer FONTSET file",
            "# Font definitions for TrueType fonts used in GeoCover",
            "#",
            "# Format: <alias> <font_file_path>",
            "",
        ]

        # Hardcoded GeoCover fonts
        GEOCOVER_FONTS = {
            "geofonts": "fonts/geofontsregular.ttf",
            "geofonts1": "fonts/GeoFonts1.ttf",
            "geofonts2": "fonts/GeoFonts2.ttf",
        }

        # Override with provided paths if given
        if font_paths:
            GEOCOVER_FONTS.update(font_paths)

        # Add fonts that were actually used
        fonts_added = set()
        for font_family in sorted(self.fonts_used):
            sanitized_name = sanitize_font_name(font_family)

            # Avoid duplicates
            if sanitized_name in fonts_added:
                continue

            fonts_added.add(sanitized_name)

            # Get font path
            font_path = GEOCOVER_FONTS.get(font_family, f"fonts/{sanitized_name}.ttf")

            lines.append(f'{sanitized_name} "{font_path}"')
            if font_family != sanitized_name:
                lines.append(f"# Original name: {font_family}")
            lines.append("")

        # Add common system fonts
        lines.append("# Common system fonts")
        lines.append(
            'arial "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"'
        )
        lines.append("")

        return "\n".join(lines)
# ============================================================================
# QGIS GENERATOR (continued in next section due to length)
# ============================================================================


class QGISGenerator:
    """Generate QGIS QML style files from classifications."""

    def __init__(
        self,
        geometry_type: str = "Polygon",
        use_symbol_field: bool = False,
        symbol_field: str = "SYMBOL",
    ):
        """
        Initialize generator.

        Args:
            geometry_type: Polygon, Line, or Point
            use_symbol_field: If True, use SYMBOL field for rules
            symbol_field: Name of symbol field
        """
        self.geometry_type = geometry_type
        self.use_symbol_field = use_symbol_field
        self.symbol_field = symbol_field
        self.symbol_counter = 0

    # Expression building methods...
    def _build_expression_from_fields(
        self, class_obj: ClassificationClass, field_names: List[str]
    ) -> str:
        """Build QGIS expression from classification field values."""
        expressions = []

        for field_values in class_obj.field_values:
            conditions = []

            for field_name, expected_value in zip(field_names, field_values):
                if expected_value == "<Null>":
                    conditions.append(f'"{field_name}" IS NULL')
                elif expected_value in ("999997", "999999"):
                    conditions.append(f'"{field_name}" = {expected_value}')
                else:
                    # Try to determine if numeric or string
                    try:
                        int(expected_value)
                        conditions.append(f'"{field_name}" = {expected_value}')
                    except ValueError:
                        conditions.append(f"\"{field_name}\" = '{expected_value}'")

            # Combine with AND
            if len(conditions) == 1:
                expressions.append(conditions[0])
            else:
                expressions.append(f"({' AND '.join(conditions)})")

        # Combine with OR
        if len(expressions) == 1:
            return expressions[0]
        else:
            return f"({' OR '.join(expressions)})"

    def _build_expression_from_symbol(self, symbol_id: str) -> str:
        """Build simple QGIS expression using SYMBOL field."""
        return f"\"{self.symbol_field}\" = '{symbol_id}'"

    def generate_qml(
        self, classification: LayerClassification, symbol_prefix: str = "class"
    ) -> str:
        """
        Generate QGIS QML style file.

        Args:
            classification: Layer classification with styling
            symbol_prefix: Prefix for symbol IDs

        Returns:
            QML XML as string
        """
        # Implementation continues...
        # (Due to length, implement similar to existing code but with integrated
        # complex polygon support from symbol_utils)
        pass
