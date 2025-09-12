"""
Direct FileGDB parser that converts File Geodatabase to ESRISchema without arcpy.

This module uses GDAL/OGR to directly read FileGDB system tables and metadata,
parsing the XML definitions to extract all schema information.
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    from osgeo import ogr, osr
except ImportError:
    import ogr
    import osr

from loguru import logger

# Import your dataclasses
from .models import (
    ESRISchema,
    CodedDomain,
    CodedValue,
    RangeDomain,
    FeatureClass,
    Table,
    Field,
    Index,
    RelationshipClass,
    Subtype,
    SubtypeValue,
)


class FileGDBParser:
    """Direct FileGDB parser that extracts schema information without arcpy."""

    def __init__(self, gdb_path: str):
        """
        Initialize the parser with a FileGDB path.

        Args:
            gdb_path: Path to the .gdb file/directory
        """
        self.gdb_path = Path(gdb_path)
        self.ds = None
        self.spatial_ref_cache = {}

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def _spatial_ref_to_dict(self, spatial_ref):
        if isinstance(spatial_ref, osr.SpatialReference):
            return spatial_ref.ExportToPROJJSON()
        return {}

    def open(self):
        """Open the FileGDB connection."""
        logger.info(f"Opening FileGDB: {self.gdb_path}")
        self.ds = ogr.Open(str(self.gdb_path))
        if not self.ds:
            raise RuntimeError(f"Could not open FileGDB: {self.gdb_path}")

    def close(self):
        """Close the FileGDB connection."""
        if self.ds:
            self.ds = None

    def parse_to_esri_schema(
        self, target_prefix: str = "GC_", excluded_tables: set = None
    ) -> ESRISchema:
        """
        Parse the entire FileGDB into an ESRISchema object.

        Args:
            target_prefix: Only include items with this prefix
            excluded_tables: Set of table names to exclude

        Returns:
            ESRISchema instance with all parsed data
        """
        if not self.ds:
            raise RuntimeError(
                "FileGDB not opened. Use context manager or call open() first."
            )

        logger.info("Starting FileGDB parsing...")

        schema = ESRISchema()

        # Set default exclusions
        if excluded_tables is None:
            excluded_tables = {
                "GC_CONFLICT_POLYGON",
                "GC_ERRORS_LINE",
                "GC_ERRORS_ROW",
                "GC_CONFLICT_ROW",
                "GC_VERSION",
                "GC_ERRORS_MULTIPOINT",
                "GC_ERRORS_POLYGON",
                "GC_REVISIONSEBENE",
            }

        # Extract all metadata from system tables
        gdb_items = self._extract_gdb_items()

        # Parse domains
        logger.info("Parsing domains...")
        coded_domains, range_domains = self._parse_domains(gdb_items)
        schema.coded_domains = coded_domains
        schema.range_domains = range_domains

        # Parse feature classes and tables
        logger.info("Parsing feature classes and tables...")
        feature_classes, tables = self._parse_datasets(
            gdb_items, target_prefix, excluded_tables
        )
        schema.feature_classes = feature_classes
        schema.tables = tables

        # Parse relationships (note: GDAL has limited FileGDB relationship support)
        logger.info("Parsing relationships...")
        logger.warning(
            "Note: GDAL/OGR has limited support for FileGDB relationships. "
            "Trying multiple methods to extract relationship information."
        )
        relationships = self._parse_relationships(
            gdb_items, target_prefix, excluded_tables
        )

        # Fallback: try ogrinfo subprocess approach if no relationships found
        if not relationships:
            logger.info("Trying ogrinfo subprocess fallback for relationships...")
            try:
                ogrinfo_relationships = extract_relationships_via_ogrinfo(self.gdb_path)
                for rel_info in ogrinfo_relationships:
                    if self._should_import_dataset(
                        rel_info["name"], target_prefix, excluded_tables
                    ):
                        rel = RelationshipClass(
                            name=rel_info["name"],
                            origin_table=rel_info["origin_table"],
                            destination_table=rel_info["destination_table"],
                            relationship_type=rel_info["type"],
                            cardinality=rel_info["type"],
                        )
                        relationships[rel.name] = rel
                logger.info(f"Extracted {len(relationships)} relationships via ogrinfo")
            except Exception as e:
                logger.debug(f"ogrinfo fallback failed: {e}")

        schema.relationships = relationships

        # Parse subtypes (note: GDAL has very limited FileGDB subtype support)
        logger.info("Parsing subtypes...")
        subtypes = self._parse_subtypes(gdb_items)
        schema.subtypes = subtypes

        # Add metadata
        schema.metadata = {
            "source": "FileGDB",
            "gdb_path": str(self.gdb_path),
            "extraction_method": "direct_gdal_parsing",
        }

        # Post-processing
        schema.infer_keys_from_relationships()
        schema.detect_primary_keys()

        logger.info(
            f"FileGDB parsing completed! Summary: {schema.get_schema_summary()}"
        )

        return schema

    def _extract_gdb_items(self) -> List[Dict[str, Any]]:
        """Extract all items from the GDB_Items system table."""
        logger.debug("Extracting GDB_Items...")

        items = []
        try:
            # Query the GDB_Items table which contains all metadata
            sql = "SELECT * FROM GDB_Items"
            result_set = self.ds.ExecuteSQL(sql)

            if not result_set:
                logger.warning("Could not query GDB_Items table")
                return items

            for feature in result_set:
                # Convert to dict for easier handling
                item_data = {}
                for i in range(feature.GetFieldCount()):
                    field_def = feature.GetFieldDefnRef(i)
                    field_name = field_def.GetName()
                    field_value = feature.GetField(i)
                    item_data[field_name] = field_value

                items.append(item_data)

            self.ds.ReleaseResultSet(result_set)

        except Exception as e:
            logger.error(f"Error extracting GDB_Items: {e}")

        logger.debug(f"Found {len(items)} items in GDB_Items")
        return items

    def _parse_domains(
        self, gdb_items: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, CodedDomain], Dict[str, RangeDomain]]:
        """Parse domain definitions from GDB_Items."""
        coded_domains = {}
        range_domains = {}

        for item in gdb_items:
            try:
                # Look for domain definitions
                definition = item.get("Definition")
                if not definition:
                    continue

                # Parse XML definition
                xml_root = ET.fromstring(definition)

                # Handle coded value domains
                if xml_root.tag == "GPCodedValueDomain2":
                    domain = self._parse_coded_domain_xml(xml_root)
                    if domain:
                        coded_domains[domain.name] = domain

                # Handle range domains
                elif xml_root.tag == "GPRangeDomain2":
                    domain = self._parse_range_domain_xml(xml_root)
                    if domain:
                        range_domains[domain.name] = domain

            except (ET.ParseError, AttributeError) as e:
                logger.debug(f"Could not parse domain definition: {e}")
                continue

        logger.info(
            f"Found {len(coded_domains)} coded domains, {len(range_domains)} range domains"
        )
        return coded_domains, range_domains

    def _parse_coded_domain_xml(self, xml_root: ET.Element) -> Optional[CodedDomain]:
        """Parse a coded domain from XML definition."""
        try:
            domain_name_elem = xml_root.find("DomainName")
            if domain_name_elem is None:
                return None

            domain_name = domain_name_elem.text
            description_elem = xml_root.find("Description")
            description = (
                description_elem.text if description_elem is not None else None
            )

            field_type_elem = xml_root.find("FieldType")
            field_type = field_type_elem.text if field_type_elem is not None else None

            # Create domain
            domain = CodedDomain(
                name=domain_name, description=description, field_type=field_type
            )

            # Extract coded values
            for coded_values_elem in xml_root.iter("CodedValues"):
                for code_elem in coded_values_elem:
                    code_val_elem = code_elem.find("Code")
                    name_elem = code_elem.find("Name")

                    if code_val_elem is not None and name_elem is not None:
                        try:
                            # Try to convert code to appropriate type
                            code_text = code_val_elem.text
                            try:
                                code = int(code_text)
                            except ValueError:
                                try:
                                    code = float(code_text)
                                except ValueError:
                                    code = code_text

                            coded_value = CodedValue(code=code, name=name_elem.text)
                            domain.coded_values.append(coded_value)
                        except Exception as e:
                            logger.debug(f"Error parsing coded value: {e}")

            return domain

        except Exception as e:
            logger.error(f"Error parsing coded domain XML: {e}")
            return None

    def _parse_range_domain_xml(self, xml_root: ET.Element) -> Optional[RangeDomain]:
        """Parse a range domain from XML definition."""
        try:
            domain_name_elem = xml_root.find("DomainName")
            if domain_name_elem is None:
                return None

            domain_name = domain_name_elem.text
            description_elem = xml_root.find("Description")
            description = (
                description_elem.text if description_elem is not None else None
            )

            field_type_elem = xml_root.find("FieldType")
            field_type = field_type_elem.text if field_type_elem is not None else None

            # Extract range values
            min_value = None
            max_value = None

            min_elem = xml_root.find("MinValue")
            if min_elem is not None:
                try:
                    min_value = float(min_elem.text)
                except (ValueError, TypeError):
                    pass

            max_elem = xml_root.find("MaxValue")
            if max_elem is not None:
                try:
                    max_value = float(max_elem.text)
                except (ValueError, TypeError):
                    pass

            return RangeDomain(
                name=domain_name,
                description=description,
                field_type=field_type,
                min_value=min_value,
                max_value=max_value,
            )

        except Exception as e:
            logger.error(f"Error parsing range domain XML: {e}")
            return None

    def _parse_datasets(
        self, gdb_items: List[Dict[str, Any]], target_prefix: str, excluded_tables: set
    ) -> Tuple[Dict[str, FeatureClass], Dict[str, Table]]:
        """Parse feature classes and tables from GDB metadata."""
        feature_classes = {}
        tables = {}

        # Get all layers (feature classes and tables) from the datasource
        layer_count = self.ds.GetLayerCount()
        logger.info(f"Found {layer_count} layers in FileGDB")

        for i in range(layer_count):
            layer = self.ds.GetLayerByIndex(i)
            layer_name = layer.GetName()

            logger.debug(f"Processing layer {i}: {layer_name}")

            # Apply filtering
            if not self._should_import_dataset(
                layer_name, target_prefix, excluded_tables
            ):
                logger.debug(f"  Skipping {layer_name} (filtered out)")
                continue

            try:
                # Check if it's a feature class (has geometry) or table
                layer_defn = layer.GetLayerDefn()
                geom_type = layer_defn.GetGeomType()

                logger.debug(
                    f"  Layer {layer_name}: geometry type = {geom_type} ({self._get_geom_type_name(geom_type)})"
                )

                if geom_type != ogr.wkbNone:
                    # It's a feature class
                    logger.debug(f"  Processing as feature class: {layer_name}")
                    fc = self._parse_feature_class(layer, gdb_items)
                    if fc:
                        feature_classes[layer_name] = fc
                        logger.debug(f"  ✓ Added feature class: {layer_name}")
                    else:
                        logger.warning(
                            f"  ✗ Failed to parse feature class: {layer_name}"
                        )
                else:
                    # It's a table
                    logger.debug(f"  Processing as table: {layer_name}")
                    table = self._parse_table(layer, gdb_items)
                    if table:
                        tables[layer_name] = table
                        logger.debug(f"  ✓ Added table: {layer_name}")
                    else:
                        logger.warning(f"  ✗ Failed to parse table: {layer_name}")

            except Exception as e:
                logger.error(f"Error parsing dataset {layer_name}: {e}")
                continue

        logger.info(
            f"Parsing complete: {len(feature_classes)} feature classes, {len(tables)} tables"
        )

        # Log what we found
        if feature_classes:
            logger.info(
                f"Feature classes: {list(feature_classes.keys())[:5]}{'...' if len(feature_classes) > 5 else ''}"
            )
        if tables:
            logger.info(
                f"Tables: {list(tables.keys())[:5]}{'...' if len(tables) > 5 else ''}"
            )

        return feature_classes, tables

    def _get_geom_type_name(self, geom_type: int) -> str:
        """Get human-readable geometry type name for debugging."""
        type_names = {
            ogr.wkbNone: "None",
            ogr.wkbPoint: "Point",
            ogr.wkbLineString: "LineString",
            ogr.wkbPolygon: "Polygon",
            ogr.wkbMultiPoint: "MultiPoint",
            ogr.wkbMultiLineString: "MultiLineString",
            ogr.wkbMultiPolygon: "MultiPolygon",
            ogr.wkbPoint25D: "Point25D",
            ogr.wkbLineString25D: "LineString25D",
            ogr.wkbPolygon25D: "Polygon25D",
            ogr.wkbMultiPoint25D: "MultiPoint25D",
            ogr.wkbMultiLineString25D: "MultiLineString25D",
            ogr.wkbMultiPolygon25D: "MultiPolygon25D",
        }

        # Handle newer 3D geometry types if available
        if hasattr(ogr, "wkbPointZ"):
            type_names.update(
                {
                    ogr.wkbPointZ: "PointZ",
                    ogr.wkbLineStringZ: "LineStringZ",
                    ogr.wkbPolygonZ: "PolygonZ",
                    ogr.wkbMultiPointZ: "MultiPointZ",
                    ogr.wkbMultiLineStringZ: "MultiLineStringZ",
                    ogr.wkbMultiPolygonZ: "MultiPolygonZ",
                }
            )

        # Handle measured geometries
        if hasattr(ogr, "wkbPointM"):
            type_names.update(
                {
                    ogr.wkbPointM: "PointM",
                    ogr.wkbLineStringM: "LineStringM",
                    ogr.wkbPolygonM: "PolygonM",
                    ogr.wkbMultiPointM: "MultiPointM",
                    ogr.wkbMultiLineStringM: "MultiLineStringM",
                    ogr.wkbMultiPolygonM: "MultiPolygonM",
                }
            )

        # Handle ZM geometries
        if hasattr(ogr, "wkbPointZM"):
            type_names.update(
                {
                    ogr.wkbPointZM: "PointZM",
                    ogr.wkbLineStringZM: "LineStringZM",
                    ogr.wkbPolygonZM: "PolygonZM",
                    ogr.wkbMultiPointZM: "MultiPointZM",
                    ogr.wkbMultiLineStringZM: "MultiLineStringZM",
                    ogr.wkbMultiPolygonZM: "MultiPolygonZM",
                }
            )

        geom_name = type_names.get(geom_type, f"Unknown({geom_type})")

        # If still unknown, try to decode the type manually
        if geom_name.startswith("Unknown"):
            # Check if it's a 3D variant by checking bits
            if geom_type & 0x80000000:  # Check for 3D flag
                base_type = geom_type & 0x7FFFFFFF
                base_name = type_names.get(base_type, f"Unknown({base_type})")
                geom_name = f"3D_{base_name}"

        return geom_name

    def _parse_feature_class(
        self, layer: ogr.Layer, gdb_items: List[Dict[str, Any]]
    ) -> Optional[FeatureClass]:
        """Parse a feature class from an OGR layer."""
        try:
            layer_name = layer.GetName()
            layer_defn = layer.GetLayerDefn()

            # Get geometry information
            geom_type = layer_defn.GetGeomType()
            geometry_type = self._ogr_geom_type_to_esri(geom_type)

            # Get spatial reference
            spatial_ref = layer.GetSpatialRef()
            spatial_ref_dict = None
            if spatial_ref:
                spatial_ref_dict = self._spatial_ref_to_dict(spatial_ref)

            # Detect Z and M dimensions more accurately
            has_z = self._geometry_has_z(geom_type)
            has_m = self._geometry_has_m(geom_type)

            # Create feature class
            fc = FeatureClass(
                name=layer_name,
                geometry_type=geometry_type,
                spatial_reference=spatial_ref_dict,
                has_z=has_z,
                has_m=has_m,
            )

            # Parse fields
            fc.fields = self._parse_fields(layer_defn)

            # Try to find additional metadata from GDB_Items
            fc_metadata = self._find_dataset_metadata(layer_name, gdb_items)
            if fc_metadata:
                self._apply_metadata_to_feature_class(fc, fc_metadata)

            logger.debug(
                f"  ✓ Created feature class: {layer_name} ({geometry_type}, Z={has_z}, M={has_m})"
            )

            return fc

        except Exception as e:
            logger.error(f"Error parsing feature class {layer.GetName()}: {e}")
            return None

    def _geometry_has_z(self, geom_type: int) -> bool:
        """Check if geometry type has Z dimension."""
        # Check for 25D types (older format)
        if geom_type in [
            ogr.wkbPoint25D,
            ogr.wkbLineString25D,
            ogr.wkbPolygon25D,
            ogr.wkbMultiPoint25D,
            ogr.wkbMultiLineString25D,
            ogr.wkbMultiPolygon25D,
        ]:
            return True

        # Check for newer Z types if available
        if hasattr(ogr, "wkbPointZ"):
            z_types = [
                ogr.wkbPointZ,
                ogr.wkbLineStringZ,
                ogr.wkbPolygonZ,
                ogr.wkbMultiPointZ,
                ogr.wkbMultiLineStringZ,
                ogr.wkbMultiPolygonZ,
            ]
            if geom_type in z_types:
                return True

        # Check for ZM types
        if hasattr(ogr, "wkbPointZM"):
            zm_types = [
                ogr.wkbPointZM,
                ogr.wkbLineStringZM,
                ogr.wkbPolygonZM,
                ogr.wkbMultiPointZM,
                ogr.wkbMultiLineStringZM,
                ogr.wkbMultiPolygonZM,
            ]
            if geom_type in zm_types:
                return True

        # Try using OGR utility function if available
        if hasattr(ogr, "GT_HasZ"):
            return ogr.GT_HasZ(geom_type)

        return False

    def _geometry_has_m(self, geom_type: int) -> bool:
        """Check if geometry type has M dimension."""
        # Check for M types if available
        if hasattr(ogr, "wkbPointM"):
            m_types = [
                ogr.wkbPointM,
                ogr.wkbLineStringM,
                ogr.wkbPolygonM,
                ogr.wkbMultiPointM,
                ogr.wkbMultiLineStringM,
                ogr.wkbMultiPolygonM,
            ]
            if geom_type in m_types:
                return True

        # Check for ZM types
        if hasattr(ogr, "wkbPointZM"):
            zm_types = [
                ogr.wkbPointZM,
                ogr.wkbLineStringZM,
                ogr.wkbPolygonZM,
                ogr.wkbMultiPointZM,
                ogr.wkbMultiLineStringZM,
                ogr.wkbMultiPolygonZM,
            ]
            if geom_type in zm_types:
                return True

        # Try using OGR utility function if available
        if hasattr(ogr, "GT_HasM"):
            return ogr.GT_HasM(geom_type)

        return False

    def _parse_table(
        self, layer: ogr.Layer, gdb_items: List[Dict[str, Any]]
    ) -> Optional[Table]:
        """Parse a table from an OGR layer."""
        try:
            layer_name = layer.GetName()
            layer_defn = layer.GetLayerDefn()

            # Create table
            table = Table(name=layer_name)

            # Parse fields
            table.fields = self._parse_fields(layer_defn)

            # Try to find additional metadata from GDB_Items
            table_metadata = self._find_dataset_metadata(layer_name, gdb_items)
            if table_metadata:
                self._apply_metadata_to_table(table, table_metadata)

            return table

        except Exception as e:
            logger.error(f"Error parsing table {layer.GetName()}: {e}")
            return None

    def _parse_fields(self, layer_defn: ogr.FeatureDefn) -> List[Field]:
        """Parse fields from an OGR layer definition."""
        fields = []

        try:
            field_count = layer_defn.GetFieldCount()
            logger.debug(f"Parsing {field_count} fields")

            for i in range(field_count):
                try:
                    field_defn = layer_defn.GetFieldDefn(
                        i
                    )  # Fixed: removed 'Ref' suffix

                    # Get basic field properties
                    field_name = field_defn.GetName()
                    field_type = field_defn.GetType()

                    # Get field dimensions
                    width = field_defn.GetWidth()
                    precision = field_defn.GetPrecision()

                    # Check nullability (may not be available in all OGR versions)
                    nullable = True  # Default to nullable
                    if hasattr(field_defn, "IsNullable"):
                        try:
                            nullable = field_defn.IsNullable()
                        except:
                            pass  # Use default if method fails

                    field = Field(
                        name=field_name,
                        type=self._ogr_field_type_to_esri(field_type),
                        length=width if width > 0 else None,
                        precision=precision if precision > 0 else None,
                        nullable=nullable,
                    )

                    fields.append(field)
                    logger.debug(
                        f"  Field {i}: {field_name} ({self._ogr_field_type_to_esri(field_type)})"
                    )

                except Exception as e:
                    logger.warning(f"Error parsing field {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error parsing fields: {e}")

        return fields

    def _parse_relationships(
        self, gdb_items: List[Dict[str, Any]], target_prefix: str, excluded_tables: set
    ) -> Dict[str, RelationshipClass]:
        """Parse relationship classes from GDB metadata."""
        relationships = {}

        # Method 1: Try to extract from GDB_Items (may not work fully for relationships)
        relationships.update(
            self._parse_relationships_from_gdb_items(
                gdb_items, target_prefix, excluded_tables
            )
        )

        # Method 2: Try to extract from datasource relationship info (GDAL limitation workaround)
        relationships.update(
            self._parse_relationships_from_datasource(target_prefix, excluded_tables)
        )

        logger.info(f"Found {len(relationships)} relationships")
        return relationships

    def _parse_relationships_from_gdb_items(
        self, gdb_items: List[Dict[str, Any]], target_prefix: str, excluded_tables: set
    ) -> Dict[str, RelationshipClass]:
        """Try to parse relationships from GDB_Items (may be limited)."""
        relationships = {}

        for item in gdb_items:
            try:
                item_type = item.get("Type")
                if (
                    item_type != "{B606A7E1-FA5B-439C-849C-6E9C2481537B}"
                ):  # Relationship class GUID
                    continue

                definition = item.get("Definition")
                if not definition:
                    continue

                # Parse XML definition
                xml_root = ET.fromstring(definition)

                if xml_root.tag == "DERelationshipClass":
                    rel = self._parse_relationship_xml(xml_root)
                    if rel and self._should_import_dataset(
                        rel.name, target_prefix, excluded_tables
                    ):
                        relationships[rel.name] = rel

            except (ET.ParseError, AttributeError) as e:
                logger.debug(
                    f"Could not parse relationship definition from GDB_Items: {e}"
                )
                continue

        return relationships

    def _parse_relationships_from_datasource(
        self, target_prefix: str, excluded_tables: set
    ) -> Dict[str, RelationshipClass]:
        """
        Try to extract relationship info from datasource capabilities.
        This is a workaround for GDAL's limited FileGDB relationship support.
        """
        relationships = {}

        try:
            # GDAL doesn't provide direct access to relationship details
            # but we can try to query system tables directly
            relationships.update(
                self._query_relationship_system_tables(target_prefix, excluded_tables)
            )

        except Exception as e:
            logger.debug(f"Could not extract relationships from datasource: {e}")

        return relationships

    def _query_relationship_system_tables(
        self, target_prefix: str, excluded_tables: set
    ) -> Dict[str, RelationshipClass]:
        """
        Try to query FileGDB relationship system tables directly.
        This may work with some GDAL versions but is not guaranteed.
        """
        relationships = {}

        try:
            # Try to query GDB_RelClasses table (if accessible)
            sql_queries = [
                "SELECT * FROM GDB_RelClasses",
                "SELECT * FROM GDB_Relationships",
                "SELECT Name, Type, Definition FROM GDB_Items WHERE Type LIKE '%Relationship%'",
            ]

            for sql in sql_queries:
                try:
                    logger.debug(f"Trying SQL: {sql}")
                    result_set = self.ds.ExecuteSQL(sql)

                    if result_set:
                        relationships.update(
                            self._process_relationship_query_results(
                                result_set, target_prefix, excluded_tables
                            )
                        )
                        self.ds.ReleaseResultSet(result_set)
                        break  # Stop on first successful query

                except Exception as e:
                    logger.debug(f"SQL query failed: {sql} - {e}")
                    continue

        except Exception as e:
            logger.debug(f"Could not query relationship system tables: {e}")

        return relationships

    def _process_relationship_query_results(
        self, result_set, target_prefix: str, excluded_tables: set
    ) -> Dict[str, RelationshipClass]:
        """Process results from relationship system table queries."""
        relationships = {}

        try:
            for feature in result_set:
                try:
                    # Extract field values
                    rel_data = {}
                    for i in range(feature.GetFieldCount()):
                        field_def = feature.GetFieldDefnRef(i)
                        field_name = field_def.GetName()
                        field_value = feature.GetField(i)
                        rel_data[field_name] = field_value

                    # Try to create relationship from available data
                    rel = self._create_relationship_from_system_data(rel_data)
                    if rel and self._should_import_dataset(
                        rel.name, target_prefix, excluded_tables
                    ):
                        relationships[rel.name] = rel

                except Exception as e:
                    logger.debug(f"Error processing relationship result: {e}")
                    continue

        except Exception as e:
            logger.debug(f"Error processing relationship query results: {e}")

        return relationships

    def _create_relationship_from_system_data(
        self, rel_data: Dict[str, Any]
    ) -> Optional[RelationshipClass]:
        """Create a RelationshipClass from system table data."""
        try:
            name = rel_data.get("Name")
            if not name:
                return None

            # Try to parse definition if available
            definition = rel_data.get("Definition")
            if definition:
                try:
                    xml_root = ET.fromstring(definition)
                    return self._parse_relationship_xml(xml_root)
                except ET.ParseError:
                    pass

            # Create minimal relationship with available info
            # (This is a fallback when full definition isn't available)
            return RelationshipClass(
                name=name,
                origin_table="",  # Would need more parsing to get these
                destination_table="",
                relationship_type="Unknown",
            )

        except Exception as e:
            logger.debug(f"Error creating relationship from system data: {e}")
            return None

    def _parse_relationship_xml(
        self, xml_root: ET.Element
    ) -> Optional[RelationshipClass]:
        """Parse a relationship class from XML definition."""
        try:
            # Extract basic information
            name_elem = xml_root.find("Name")
            if name_elem is None:
                return None

            name = name_elem.text

            # Extract origin and destination
            origin_elem = xml_root.find("OriginClassNames/Name")
            dest_elem = xml_root.find("DestinationClassNames/Name")

            origin_table = origin_elem.text if origin_elem is not None else ""
            destination_table = dest_elem.text if dest_elem is not None else ""

            # Extract cardinality
            cardinality_elem = xml_root.find("Cardinality")
            cardinality = (
                cardinality_elem.text if cardinality_elem is not None else "OneToMany"
            )

            # Map ESRI cardinality values
            cardinality_map = {
                "esriRelCardinalityOneToOne": "OneToOne",
                "esriRelCardinalityOneToMany": "OneToMany",
                "esriRelCardinalityManyToMany": "ManyToMany",
            }
            cardinality = cardinality_map.get(cardinality, cardinality)

            return RelationshipClass(
                name=name,
                origin_table=origin_table,
                destination_table=destination_table,
                relationship_type=cardinality,
                cardinality=cardinality,
            )

        except Exception as e:
            logger.error(f"Error parsing relationship XML: {e}")
            return None

    def _parse_subtypes(self, gdb_items: List[Dict[str, Any]]) -> Dict[str, Subtype]:
        """
        Parse subtype definitions from GDB metadata.
        Note: GDAL has very limited support for FileGDB subtypes.
        """
        subtypes = {}

        logger.warning(
            "Note: GDAL/OGR has very limited support for FileGDB subtypes. "
            "Most subtype information may not be accessible."
        )

        # Method 1: Try to find subtypes in GDB_Items XML definitions
        subtypes.update(self._parse_subtypes_from_gdb_items(gdb_items))

        # Method 2: Try to infer subtypes from field domains (common pattern)
        # This is a fallback that looks for fields with coded domains that might represent subtypes
        if not subtypes:
            logger.info(
                "No subtypes found in metadata, trying to infer from field domains..."
            )
            subtypes.update(self._infer_subtypes_from_domains())

        if subtypes:
            logger.info(f"Found {len(subtypes)} subtype definitions")
        else:
            logger.warning(
                "No subtypes could be extracted. This is a known GDAL limitation."
            )
            logger.info(
                "For complete subtype information, use the JSON export method with arcpy."
            )

        return subtypes

    def _parse_subtypes_from_gdb_items(
        self, gdb_items: List[Dict[str, Any]]
    ) -> Dict[str, Subtype]:
        """Try to parse subtypes from GDB_Items XML definitions."""
        subtypes = {}

        for item in gdb_items:
            try:
                definition = item.get("Definition")
                if not definition:
                    continue

                xml_root = ET.fromstring(definition)

                # Look for subtype definitions in feature classes
                if xml_root.tag in ["DEFeatureClass", "DETable"]:
                    subtype_field_elem = xml_root.find("SubtypeFieldName")
                    if subtype_field_elem is not None:
                        # Found a dataset with subtypes
                        name_elem = xml_root.find("Name")
                        if name_elem is not None:
                            dataset_name = name_elem.text
                            subtype = self._parse_subtype_xml(xml_root, dataset_name)
                            if subtype:
                                subtypes[subtype.name] = subtype

            except (ET.ParseError, AttributeError) as e:
                logger.debug(f"Could not parse subtype definition: {e}")
                continue

        return subtypes

    def _infer_subtypes_from_domains(self) -> Dict[str, Subtype]:
        """
        Fallback method: Try to infer subtypes from coded domains.
        This looks for domains that might represent subtype values.
        """
        subtypes = {}

        # This is a heuristic approach - not guaranteed to work
        # but might catch some subtype-like patterns
        logger.debug("Attempting to infer subtypes from domain patterns...")

        # Look for fields named SUBTYPE, TYPE, etc. with coded domains
        try:
            layer_count = self.ds.GetLayerCount()

            for i in range(layer_count):
                layer = self.ds.GetLayerByIndex(i)
                layer_name = layer.GetName()
                layer_defn = layer.GetLayerDefn()

                # Look for potential subtype fields
                for j in range(layer_defn.GetFieldCount()):
                    field_defn = layer_defn.GetFieldDefn(j)
                    field_name = field_defn.GetName()

                    # Common subtype field names
                    if field_name.upper() in ["SUBTYPE", "TYPE", "CATEGORY", "CLASS"]:
                        logger.debug(
                            f"Found potential subtype field: {layer_name}.{field_name}"
                        )
                        # This would require additional logic to extract values
                        # For now, just log the discovery

        except Exception as e:
            logger.debug(f"Error inferring subtypes from domains: {e}")

        return subtypes

    def _parse_subtype_xml(
        self, xml_root: ET.Element, dataset_name: str
    ) -> Optional[Subtype]:
        """Parse subtype definition from XML."""
        try:
            subtype_field_elem = xml_root.find("SubtypeFieldName")
            if subtype_field_elem is None:
                return None

            subtype_field = subtype_field_elem.text

            subtype = Subtype(
                name=f"{dataset_name}_Subtypes", subtype_field=subtype_field
            )

            # Extract subtype values
            for subtype_elem in xml_root.iter("Subtype"):
                code_elem = subtype_elem.find("SubtypeCode")
                name_elem = subtype_elem.find("SubtypeName")

                if code_elem is not None and name_elem is not None:
                    try:
                        code = int(code_elem.text)
                    except ValueError:
                        code = code_elem.text

                    subtype_value = SubtypeValue(code=code, name=name_elem.text)
                    subtype.subtypes.append(subtype_value)

            return subtype if subtype.subtypes else None

        except Exception as e:
            logger.error(f"Error parsing subtype XML: {e}")
            return None

    def _should_import_dataset(
        self, name: str, target_prefix: str, excluded_tables: set
    ) -> bool:
        """Check if a dataset should be imported."""
        if not name:
            logger.debug("    Filtering: Empty name - SKIP")
            return False

        # Get base name
        base_name = name.split(".")[-1] if "." in name else name
        logger.debug(f"    Filtering: {name} (base: {base_name})")

        # Check exclusions
        if name in excluded_tables or base_name in excluded_tables:
            logger.debug(f"    Filtering: {name} in excluded tables - SKIP")
            return False

        # Skip items ending with _I
        if base_name.endswith("_I"):
            logger.debug(f"    Filtering: {name} ends with _I - SKIP")
            return False

        # Check prefix
        if target_prefix:
            if base_name.startswith(target_prefix):
                logger.debug(
                    f"    Filtering: {name} starts with {target_prefix} - INCLUDE"
                )
                return True
            else:
                logger.debug(
                    f"    Filtering: {name} does not start with {target_prefix} - SKIP"
                )
                return False

        logger.debug(f"    Filtering: {name} no prefix required - INCLUDE")
        return True

    def _find_dataset_metadata(
        self, layer_name: str, gdb_items: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find metadata for a specific dataset in GDB_Items."""
        for item in gdb_items:
            name = item.get("Name")
            if name == layer_name:
                return item
        return None

    def _apply_metadata_to_feature_class(
        self, fc: FeatureClass, metadata: Dict[str, Any]
    ):
        """Apply additional metadata to a feature class."""
        # This could be extended to parse more metadata from the Definition XML
        pass

    def _apply_metadata_to_table(self, table: Table, metadata: Dict[str, Any]):
        """Apply additional metadata to a table."""
        # This could be extended to parse more metadata from the Definition XML
        pass

    def _ogr_geom_type_to_esri(self, ogr_type: int) -> str:
        """Convert OGR geometry type to ESRI geometry type string."""
        # Handle basic 2D types
        type_map = {
            ogr.wkbPoint: "esriGeometryPoint",
            ogr.wkbLineString: "esriGeometryPolyline",
            ogr.wkbPolygon: "esriGeometryPolygon",
            ogr.wkbMultiPoint: "esriGeometryMultipoint",
            ogr.wkbMultiLineString: "esriGeometryPolyline",
            ogr.wkbMultiPolygon: "esriGeometryPolygon",
        }

        # Handle 25D types (older 3D format)
        type_map.update(
            {
                ogr.wkbPoint25D: "esriGeometryPoint",
                ogr.wkbLineString25D: "esriGeometryPolyline",
                ogr.wkbPolygon25D: "esriGeometryPolygon",
                ogr.wkbMultiPoint25D: "esriGeometryMultipoint",
                ogr.wkbMultiLineString25D: "esriGeometryPolyline",
                ogr.wkbMultiPolygon25D: "esriGeometryPolygon",
            }
        )

        # Handle newer 3D types if available
        if hasattr(ogr, "wkbPointZ"):
            type_map.update(
                {
                    ogr.wkbPointZ: "esriGeometryPoint",
                    ogr.wkbLineStringZ: "esriGeometryPolyline",
                    ogr.wkbPolygonZ: "esriGeometryPolygon",
                    ogr.wkbMultiPointZ: "esriGeometryMultipoint",
                    ogr.wkbMultiLineStringZ: "esriGeometryPolyline",
                    ogr.wkbMultiPolygonZ: "esriGeometryPolygon",
                }
            )

        # Handle measured geometries
        if hasattr(ogr, "wkbPointM"):
            type_map.update(
                {
                    ogr.wkbPointM: "esriGeometryPoint",
                    ogr.wkbLineStringM: "esriGeometryPolyline",
                    ogr.wkbPolygonM: "esriGeometryPolygon",
                    ogr.wkbMultiPointM: "esriGeometryMultipoint",
                    ogr.wkbMultiLineStringM: "esriGeometryPolyline",
                    ogr.wkbMultiPolygonM: "esriGeometryPolygon",
                }
            )

        # Handle ZM geometries
        if hasattr(ogr, "wkbPointZM"):
            type_map.update(
                {
                    ogr.wkbPointZM: "esriGeometryPoint",
                    ogr.wkbLineStringZM: "esriGeometryPolyline",
                    ogr.wkbPolygonZM: "esriGeometryPolygon",
                    ogr.wkbMultiPointZM: "esriGeometryMultipoint",
                    ogr.wkbMultiLineStringZM: "esriGeometryPolyline",
                    ogr.wkbMultiPolygonZM: "esriGeometryPolygon",
                }
            )

        esri_type = type_map.get(ogr_type)

        # If not found, try to decode manually for unknown 3D types
        if not esri_type:
            # Remove potential 3D/M flags and check base type
            base_type = ogr_type
            if ogr_type & 0x80000000:  # Has 3D flag
                base_type = ogr_type & 0x7FFFFFFF

            esri_type = type_map.get(base_type)

        # Final fallback
        if not esri_type:
            logger.debug(f"Unknown OGR geometry type {ogr_type}, defaulting to null")
            esri_type = "esriGeometryNull"

        logger.debug(
            f"OGR geometry type {ogr_type} ({self._get_geom_type_name(ogr_type)}) mapped to {esri_type}"
        )

        return esri_type

    def _ogr_field_type_to_esri(self, ogr_type: int) -> str:
        """Convert OGR field type to ESRI field type string."""
        type_map = {
            ogr.OFTInteger: "esriFieldTypeInteger",
            ogr.OFTIntegerList: "esriFieldTypeInteger",
            ogr.OFTReal: "esriFieldTypeDouble",
            ogr.OFTRealList: "esriFieldTypeDouble",
            ogr.OFTString: "esriFieldTypeString",
            ogr.OFTStringList: "esriFieldTypeString",
            ogr.OFTBinary: "esriFieldTypeBlob",
            ogr.OFTDate: "esriFieldTypeDate",
            ogr.OFTTime: "esriFieldTypeDate",
            ogr.OFTDateTime: "esriFieldTypeDate",
        }

        # Handle newer OGR field types if available
        if hasattr(ogr, "OFTWideString"):
            type_map[ogr.OFTWideString] = "esriFieldTypeString"
        if hasattr(ogr, "OFTWideStringList"):
            type_map[ogr.OFTWideStringList] = "esriFieldTypeString"
        if hasattr(ogr, "OFTInteger64"):
            type_map[ogr.OFTInteger64] = "esriFieldTypeInteger"

        esri_type = type_map.get(ogr_type, "esriFieldTypeString")
        logger.debug(f"OGR type {ogr_type} mapped to {esri_type}")

        return esri_type

    def list_visible_relationships(self) -> List[str]:
        """
        List relationships that are visible in the FileGDB but may not be accessible.
        This uses the same approach as 'ogrinfo' to see what relationships exist.
        """
        relationships = []

        try:
            # Unfortunately, GDAL doesn't provide a direct API to list relationships
            # This would require calling ogrinfo externally or using undocumented APIs
            logger.info(
                "GDAL limitation: Cannot directly list relationships programmatically"
            )
            logger.info("Use 'ogrinfo <gdb_path>' to see available relationships")

        except Exception as e:
            logger.debug(f"Error listing relationships: {e}")

        return relationships
        """Convert spatial reference to dictionary."""
        try:
            # Get WKID (Well-Known ID) if available
            wkid = None
            auth_name = spatial_ref.GetAuthorityName(None)
            auth_code = spatial_ref.GetAuthorityCode(None)

            if auth_name == "EPSG" and auth_code:
                wkid = int(auth_code)

            return {
                "wkid": wkid,
                "wkt": spatial_ref.ExportToWkt(),
                "authority": auth_name,
                "code": auth_code,
            }
        except Exception:
            return {}


def extract_relationships_via_ogrinfo(gdb_path: str) -> List[Dict[str, str]]:
    """
    Workaround: Extract relationship info using ogrinfo subprocess call.
    This parses the output of 'ogrinfo <gdb>' to get relationship details.
    """
    import subprocess
    import re

    relationships = []

    try:
        # Run ogrinfo to get database info
        result = subprocess.run(
            ["ogrinfo", str(gdb_path)], capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            # Parse relationship lines from output
            # Format: "Relationship: GC_BEDR_GEOL_MAPPING_UNIT_ATT (Association, GC_GEOL_MAPPING_UNIT_ATT, GC_BEDROCK)"
            relationship_pattern = (
                r"Relationship:\s+(\S+)\s+\((\w+),\s+(\S+),\s+(\S+)\)"
            )

            for line in result.stdout.split("\n"):
                match = re.match(relationship_pattern, line.strip())
                if match:
                    rel_name, rel_type, origin_table, dest_table = match.groups()
                    relationships.append(
                        {
                            "name": rel_name,
                            "type": rel_type,
                            "origin_table": origin_table,
                            "destination_table": dest_table,
                        }
                    )
        else:
            logger.warning(f"ogrinfo failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.warning("ogrinfo command timed out")
    except subprocess.CalledProcessError as e:
        logger.warning(f"ogrinfo command failed: {e}")
    except Exception as e:
        logger.debug(f"Could not run ogrinfo: {e}")

    return relationships


def parse_filegdb_to_esri_schema(
    gdb_path: str, target_prefix: str = "GC_", excluded_tables: set = None
) -> ESRISchema:
    """
    Main function to parse a FileGDB directly to ESRISchema.

    Args:
        gdb_path: Path to the .gdb file/directory
        target_prefix: Only include items with this prefix
        excluded_tables: Set of table names to exclude

    Returns:
        ESRISchema instance with all parsed data
    """
    with FileGDBParser(gdb_path) as parser:
        return parser.parse_to_esri_schema(target_prefix, excluded_tables)


def diagnose_filegdb_parsing(gdb_path: str) -> Dict[str, Any]:
    """
    Diagnostic function to check what can be extracted from a FileGDB.
    Shows current limitations and workarounds.
    """
    diagnosis = {
        "gdb_path": gdb_path,
        "gdal_version": None,
        "accessible_layers": [],
        "relationships_via_ogrinfo": [],
        "parsing_limitations": [],
    }

    try:
        from osgeo import gdal

        diagnosis["gdal_version"] = gdal.VersionInfo()
    except:
        pass

    # Check layer accessibility
    try:
        with FileGDBParser(gdb_path) as parser:
            layer_count = parser.ds.GetLayerCount()
            for i in range(layer_count):
                layer = parser.ds.GetLayerByIndex(i)
                layer_name = layer.GetName()
                layer_defn = layer.GetLayerDefn()
                geom_type = layer_defn.GetGeomType()
                field_count = layer_defn.GetFieldCount()

                diagnosis["accessible_layers"].append(
                    {
                        "name": layer_name,
                        "geometry_type": parser._get_geom_type_name(geom_type),
                        "field_count": field_count,
                        "is_feature_class": geom_type != ogr.wkbNone,
                    }
                )
    except Exception as e:
        diagnosis["parsing_limitations"].append(f"Layer enumeration failed: {e}")

    # Check relationships
    try:
        diagnosis["relationships_via_ogrinfo"] = extract_relationships_via_ogrinfo(
            gdb_path
        )
    except Exception as e:
        diagnosis["parsing_limitations"].append(f"Relationship extraction failed: {e}")

    # Note known limitations
    diagnosis["parsing_limitations"].extend(
        [
            "GDAL has limited support for FileGDB subtypes",
            "Relationship details may be incomplete",
            "Some metadata may not be accessible",
            "For complete schema, consider using arcpy JSON export",
        ]
    )

    return diagnosis


def main():
    """Example usage of the FileGDB parser."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python filegdb_parser.py <path_to_gdb>")
        print("   or: python filegdb_parser.py <path_to_gdb> --diagnose")
        return

    gdb_path = sys.argv[1]

    # Diagnostic mode
    if len(sys.argv) > 2 and sys.argv[2] == "--diagnose":
        print("=== FileGDB Diagnostic Mode ===")
        diagnosis = diagnose_filegdb_parsing(gdb_path)

        print(f"GDB Path: {diagnosis['gdb_path']}")
        print(f"GDAL Version: {diagnosis.get('gdal_version', 'Unknown')}")
        print(f"Accessible layers: {len(diagnosis['accessible_layers'])}")

        for layer in diagnosis["accessible_layers"][:10]:  # Show first 10
            print(
                f"  - {layer['name']} ({layer['geometry_type']}, {layer['field_count']} fields)"
            )

        print(f"Relationships found: {len(diagnosis['relationships_via_ogrinfo'])}")
        for rel in diagnosis["relationships_via_ogrinfo"][:5]:  # Show first 5
            print(
                f"  - {rel['name']}: {rel['origin_table']} → {rel['destination_table']}"
            )

        if diagnosis["parsing_limitations"]:
            print("Known limitations:")
            for limitation in diagnosis["parsing_limitations"]:
                print(f"  ! {limitation}")

        return

    # Normal parsing mode
    try:
        logger.info(f"Parsing FileGDB: {gdb_path}")
        schema = parse_filegdb_to_esri_schema(gdb_path)

        print("Successfully parsed FileGDB!")
        print(f"Summary: {schema.get_schema_summary()}")

        # Show some examples
        if schema.feature_classes:
            print("\nFeature classes found:")
            for name in list(schema.feature_classes.keys())[:5]:
                fc = schema.feature_classes[name]
                print(f"  - {name} ({fc.geometry_type}, {len(fc.fields)} fields)")

        if schema.coded_domains:
            print("\nCoded domains found:")
            for name in list(schema.coded_domains.keys())[:5]:
                domain = schema.coded_domains[name]
                print(f"  - {name} ({len(domain.coded_values)} values)")

        if schema.relationships:
            print("\nRelationships found:")
            for name in list(schema.relationships.keys())[:5]:
                rel = schema.relationships[name]
                print(f"  - {name}: {rel.origin_table} → {rel.destination_table}")

    except Exception as e:
        logger.error(f"Error parsing FileGDB: {e}")
        print("Try running with --diagnose flag to see what's accessible")
        raise


if __name__ == "__main__":
    main()
