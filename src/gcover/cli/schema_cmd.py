import json
from pathlib import Path
import traceback
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table


import click

from gcover.schema import SchemaDiff, extract_schema, transform_esri_json
from gcover.schema.exporters.plantuml import generate_plantuml_from_schema
from gcover.config import GlobalConfig, SchemaConfig
from gcover.schema.serializer import (
    serialize_esri_schema_to_dict,
    save_esri_schema_to_file,
)
from gcover.schema.simple_transformer import transform_esri_flat_json


# TODO
from gcover.config import load_config, AppConfig, SchemaConfig

from loguru import logger

console = Console()


def get_schema_configs(ctx) -> tuple[SchemaConfig, GlobalConfig]:
    """Get schema and global configs from context"""
    app_config: AppConfig = ctx.obj["app_config"]

    if app_config.schema_config:  # 🔧 Updated field name
        schema_config = app_config.schema_config
    else:
        rprint("[yellow]No schema config found[/yellow]")
        # You could create a default or raise an error
        from ..config.models import SchemaConfig

        schema_config = SchemaConfig()  # Use defaults

    return schema_config, app_config.global_


@click.group()
def schema():
    """Schema management commands."""
    pass



@schema.command(name="export-tables")
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    help="Output directory for exports (default: current directory)"
)
@click.option(
    "--workspace", "-w",
    type=str,
    required=True,
    help="SDE connection string or GDB path"
)
@click.option(
    "--all-tables/--gc-tables-only",
    default=False,
    help="Export all tables or only GC_ tables (default: GC_ only)"
)
@click.option(
    "--include-incremental/--exclude-incremental",
    default=False,
    help="Include/exclude tables with _I suffix (default: exclude)"
)
@click.option(
    "--format", "-f",
    multiple=True,
    type=click.Choice(["excel", "csv", "json"]),
    default=["excel", "json"],
    help="Output formats (can specify multiple)"
)
@click.option(
    "--table-filter",
    multiple=True,
    help="Specific tables to export (can specify multiple)"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be exported without actually doing it"
)
def export_tables(output_dir, workspace, all_tables, include_incremental, format, table_filter, dry_run):
    """Export tables from SDE/GDB to various formats.

    Exports GeoCover tables from an SDE connection or geodatabase to Excel,
    CSV, and JSON formats. Handles hierarchical table structures and applies
    appropriate data transformations.

    Examples:

        # Export GC_ tables from SDE to current directory
        gcover schema export-tables -w "path/to/connection.sde"

        # Export all tables to specific directory with multiple formats
        gcover schema export-tables -w "database.gdb" -o exports/ --all-tables -f excel -f csv -f json

        # Export specific tables only
        gcover schema export-tables -w "connection.sde" --table-filter GC_LITHO --table-filter GC_CHRONO

        # Dry run to see what would be exported
        gcover schema export-tables -w "connection.sde" --dry-run
    """
    from ..utils.imports import HAS_ARCPY

    if not HAS_ARCPY:
        console.print("❌ [bold red]This command requires ArcPy[/bold red]")
        console.print("ArcPy is available with ArcGIS Pro installation")
        raise click.Abort()

    import arcpy

    # Define table configurations
    TREE_TABLES = [
        "GC_LITHO", "GC_LITSTRAT", "GC_CHRONO", "GC_TECTO",
        "GC_LITSTRAT_FORMATION_BANK"
    ]

    STANDARD_TABLES = [
        "GC_CHARCAT", "GC_ADMIXTURE", "GC_COMPOSIT",
        "GC_GEOL_MAPPING_UNIT", "GC_GEOL_MAPPING_UNIT_ATT", "GC_LITSTRAT_UNCO", "GC_CORRELATION"
    ]

    ALL_GC_TABLES = TREE_TABLES + STANDARD_TABLES

    COLUMN_TYPE_MAPPING = {
        "LITSTRAT_FORMATION_BANK": int,
        "GEOL_MAPPING_UNIT": int,
        "CHRONO_TOP": int,
        "CHRONO_BASE": int,
        "LITHO_MAIN": int,
        "LITHO_SEC": int,
        "LITHO_TER": int,
        "CORRELATION": int,
    }

    try:
        # Setup workspace
        console.print(f"🔗 Connecting to workspace: [bold blue]{workspace}[/bold blue]")
        arcpy.env.workspace = workspace

        # Determine output directory
        if output_dir is None:
            output_dir = Path.cwd() / f"table_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get available tables
        console.print("📋 Scanning for tables...")
        available_tables = arcpy.ListTables()

        if not available_tables:
            console.print("❌ [bold red]No tables found in workspace[/bold red]")
            raise click.Abort()

        # Filter tables based on criteria
        tables_to_export = []

        for table in available_tables:
            # Skip incremental tables if not requested
            if not include_incremental and table.endswith("_I"):
                continue

            # Parse table name (handle schema prefixes)
            if "." in table:
                schema_prefix, short_name = table.split(".", 1)
            else:
                short_name = table
                schema_prefix = ""

            # Apply filters
            if table_filter:
                # If specific tables requested, only include those
                if not any(filter_name in table for filter_name in table_filter):
                    continue
            elif not all_tables:
                # Default: only GC_ tables
                if short_name not in ALL_GC_TABLES:
                    continue

            tables_to_export.append((table, short_name.replace("GC_", "")))

        if not tables_to_export:
            console.print("❌ [bold red]No tables match the specified criteria[/bold red]")
            raise click.Abort()

        console.print(f"📊 Found [bold cyan]{len(tables_to_export)}[/bold cyan] tables to export")

        # Show what will be exported
        for table_name, short_name in tables_to_export:
            console.print(f"  • {table_name} → {short_name}")

        if dry_run:
            console.print(f"\n📁 Would export to: [bold green]{output_dir}[/bold green]")
            console.print(f"📝 Formats: {', '.join(format)}")
            console.print("🏃 [bold yellow]Dry run complete - no files were created[/bold yellow]")
            return

        # Export tables
        console.print(f"\n🚀 Starting export to: [bold green]{output_dir}[/bold green]")

        # Prepare Excel writer if needed
        excel_writer = None
        if "excel" in format:
            excel_path = output_dir / f"exported_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            excel_writer = pd.ExcelWriter(excel_path, engine='openpyxl')

        exported_count = 0

        try:
            for table_name, short_name in tables_to_export:
                console.print(f"📤 Exporting [bold]{table_name}[/bold]...")

                try:
                    # Convert ArcGIS table to DataFrame
                    df = _arcgis_table_to_df(table_name)

                    if df.empty:
                        console.print(f"  ⚠️ Table {table_name} is empty, skipping...")
                        continue

                    # Apply data transformations
                    df = _transform_table_data(df, table_name, COLUMN_TYPE_MAPPING)

                    # Generate output filename
                    clean_name = "_".join([w.capitalize() for w in short_name.split("_")])

                    # Export to requested formats
                    if "csv" in format:
                        csv_path = output_dir / f"{clean_name}.csv"
                        df.to_csv(csv_path, index=True, encoding='utf-8')
                        console.print(f"  ✅ CSV: {csv_path}")

                    if "json" in format:
                        json_path = output_dir / f"{clean_name}.json"
                        _export_table_to_json(df, json_path)
                        console.print(f"  ✅ JSON: {json_path}")

                    if "excel" in format and excel_writer is not None:
                        # Excel sheet names have 31 char limit
                        sheet_name = short_name.upper()[:31]
                        df.to_excel(excel_writer, sheet_name=sheet_name, index=True)
                        console.print(f"  ✅ Excel sheet: {sheet_name}")

                    exported_count += 1

                except Exception as e:
                    console.print(f"  ❌ [bold red]Failed to export {table_name}:[/bold red] {e}")
                    logger.error(f"Export error for {table_name}: {traceback.format_exc()}")
                    continue

        finally:
            if excel_writer is not None:
                excel_writer.close()
                console.print(f"📊 Excel file saved: [bold green]{excel_path}[/bold green]")

        # Create README
        readme_path = output_dir / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("GeoCover Tables Export\n")
            f.write("=====================\n\n")
            f.write(f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source workspace: {workspace}\n")
            f.write(f"Tables exported: {exported_count}\n")
            f.write(f"Formats: {', '.join(format)}\n\n")
            f.write("Tables included:\n")
            for table_name, short_name in tables_to_export:
                f.write(f"  - {table_name}\n")

        console.print("\n🎉 [bold green]Export completed![/bold green]")
        console.print(f"📊 Exported {exported_count} tables")
        console.print(f"📁 Output directory: [bold green]{output_dir}[/bold green]")

    except Exception as e:
        console.print(f"❌ [bold red]Export failed:[/bold red] {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        raise click.Abort()


def _arcgis_table_to_df(table_name, input_fields=None, query=""):
    """Convert ArcGIS table to pandas DataFrame with proper error handling."""
    import arcpy

    try:
        # Get available fields
        available_fields = [field.name for field in arcpy.ListFields(table_name)]
        oid_fields = [f.name for f in arcpy.ListFields(table_name) if f.type == "OID"]

        # Determine fields to include
        if input_fields:
            final_fields = list(set(oid_fields + input_fields) & set(available_fields))
        else:
            final_fields = available_fields

        # Extract data using SearchCursor
        data = []
        with arcpy.da.SearchCursor(table_name, final_fields, where_clause=query) as cursor:
            for row in cursor:
                data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data, columns=final_fields)

        # Set index to OID field if available
        if oid_fields and len(oid_fields) > 0:
            df = df.set_index(oid_fields[0], drop=True)

        return df

    except Exception as e:
        logger.error(f"Error reading table {table_name}: {e}")
        raise


def _transform_table_data(df, table_name, column_type_mapping):
    """Apply data transformations specific to table types."""
    from gcover.config import EXCLUDED_FIELDS

    # Remove excluded fields
    df = df.drop(columns=EXCLUDED_FIELDS, errors="ignore")

    # Sort hierarchical tables
    sort_keys = ["GEOLCODE", "PARENT_REF"]
    common_columns = set(df.columns).intersection(sort_keys)

    if common_columns:
        # Fill NaN values in PARENT_REF for sorting
        if "PARENT_REF" in df.columns:
            df["PARENT_REF"] = df["PARENT_REF"].fillna(0)

        df = df.sort_values(by=list(common_columns))

    # Apply column type mappings
    required_columns = set(column_type_mapping.keys())
    existing_columns = set(df.columns)
    mappable_columns = required_columns & existing_columns

    if mappable_columns:
        # Create mapping for only existing columns
        applicable_mapping = {col: column_type_mapping[col] for col in mappable_columns}
        df = df.fillna(0).astype(applicable_mapping)

    return df


def _export_table_to_json(df, json_path):
    """Export DataFrame to JSON with appropriate format based on content."""

    # Check if this is a simple lookup table (GEOLCODE -> DESCRIPTION)
    if {"DESCRIPTION", "GEOLCODE"}.issubset(df.columns):
        # Create simple key-value mapping
        lookup_df = df[["GEOLCODE", "DESCRIPTION"]].copy()
        simple_dict = dict(lookup_df.values)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(simple_dict, f, ensure_ascii=False, indent=2)
    else:
        # Export as records for complex tables
        df.to_json(json_path, indent=2, orient="records", force_ascii=False)

        
        
@schema.command(name="transform-simple")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output JSON file (default: adds '_simple' suffix to input)",
)
@click.option(
    "--pretty/--compact",
    default=True,
    help="Pretty-print JSON output (default: pretty)",
)
@click.option(
    "--validate", is_flag=True, help="Validate input as ESRI DEWorkspace export"
)
def transform_simple_format(input_file, output, pretty, validate):
    """Transform ESRI schema JSON to simple flat format.

    Converts an ESRI schema export to a simplified flat JSON structure
    with version 2 format. This is a lighter alternative to the full
    dataclass-based transformation.

    Examples:

        # Basic transformation
        gcover schema transform-simple esri_export.json

        # Custom output file
        gcover schema transform-simple esri_export.json -o simple_schema.json

        # Compact output without validation
        gcover schema transform-simple esri_export.json --compact
    """
    console.print(
        f"Converting ESRI schema to simple format: [bold blue]{input_file}[/bold blue]"
    )

    try:
        # Determine output path
        if output is None:
            output = input_file.parent / f"{input_file.stem}_simple.json"

        # Load and validate input JSON
        console.print("Loading input file...")
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # Optional validation
        if validate:
            if (
                input_data.get("datasetType") != "DEWorkspace"
                or input_data.get("majorVersion", 0) < 3
            ):
                console.print(
                    "[bold red]Error:[/bold red] Not a valid ESRI 'DEWorkspace' export JSON"
                )
                raise click.Abort()
            console.print("✅ Input validation passed")

        # Transform using simple format
        console.print("Transforming schema...")
        with console.status("[bold green]Processing datasets..."):
            result = transform_esri_flat_json(input_data)

        console.print("✅ Schema transformation completed")

        # Save output
        console.print(f"Saving to: [bold green]{output}[/bold green]")
        with open(output, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(result, f, indent=4, ensure_ascii=False)
            else:
                json.dump(result, f, ensure_ascii=False)

        console.print("✅ Output saved successfully")

        # Show summary
        console.print("\n📊 [bold]Transformation Summary[/bold]")
        console.print("─" * 40)

        stats = {}
        if "coded_domain" in result:
            stats["Coded Domains"] = len(result["coded_domain"])
        if "tables" in result:
            stats["Tables"] = len(result["tables"])
        if "featclasses" in result:
            stats["Feature Classes"] = len(result["featclasses"])
        if "relationships" in result:
            stats["Relationships"] = len(result["relationships"])
        if "subtypes" in result:
            stats["Subtypes"] = len(result["subtypes"])

        for key, value in stats.items():
            console.print(f"  {key}: [bold cyan]{value}[/bold cyan]")

        console.print(f"\n📁 Output file: [bold green]{output}[/bold green]")
        console.print(
            f"📏 File size: [bold cyan]{output.stat().st_size:,}[/bold cyan] bytes"
        )
        console.print(
            f"🔖 Format version: [bold cyan]{result.get('version', 'unknown')}[/bold cyan]"
        )

        # Show sample content
        if "tables" in result and result["tables"]:
            console.print("\n📋 [bold]Sample Tables:[/bold]")
            for i, table_name in enumerate(list(result["tables"].keys())[:3]):
                field_count = len(result["tables"][table_name].get("fields", []))
                console.print(f"  {i + 1}. {table_name} ({field_count} fields)")

        if "featclasses" in result and result["featclasses"]:
            console.print("\n🗺️  [bold]Sample Feature Classes:[/bold]")
            for i, fc_name in enumerate(list(result["featclasses"].keys())[:3]):
                field_count = len(result["featclasses"][fc_name].get("fields", []))
                console.print(f"  {i + 1}. {fc_name} ({field_count} fields)")

        console.print(
            "\n🎉 [bold green]Simple format transformation completed![/bold green]"
        )

    except FileNotFoundError:
        console.print(f"❌ [bold red]Input file not found:[/bold red] {input_file}")
        raise click.Abort()

    except json.JSONDecodeError as e:
        console.print(f"❌ [bold red]Invalid JSON in input file:[/bold red] {e}")
        raise click.Abort()

    except Exception as e:
        console.print(f"❌ [bold red]Transformation failed:[/bold red] {e}")
        logger.error(f"Full error details: {traceback.format_exc()}")
        raise click.Abort()


@schema.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output JSON file (default: adds '_transformed' suffix to input)",
)
@click.option(
    "--target-prefix",
    default="GC_",
    help="Only import items with this prefix (default: GC_)",
)
@click.option(
    "--exclude-tables",
    multiple=True,
    help="Table names to exclude (can be specified multiple times)",
)
@click.option(
    "--include-metadata-fields/--exclude-metadata-fields",
    default=False,
    help="Include/exclude metadata fields (default: exclude)",
)
@click.option(
    "--schema-prefix",
    default="TOPGIS_GC.",
    help="Default schema prefix for non-prefixed names",
)
@click.option(
    "--pretty/--compact",
    default=True,
    help="Pretty-print JSON output (default: pretty)",
)
@click.option(
    "--show-summary", is_flag=True, default=True, help="Show transformation summary"
)
def transform(
    input_file,
    output,
    target_prefix,
    exclude_tables,
    include_metadata_fields,
    schema_prefix,
    pretty,
    show_summary,
):
    """Transform ESRI schema JSON to simplified ESRISchema format.

    Takes an ESRI schema export (JSON file) and transforms it into a simplified
    ESRISchema format suitable for further processing.
    """
    console.print(f"🔄 Transforming ESRI schema: [bold blue]{input_file}[/bold blue]")

    try:
        # Determine output path
        if output is None:
            output = input_file.parent / f"{input_file.stem}_transformed.json"

        # Load input JSON
        console.print("📖 Loading input file...")
        with open(input_file, "r", encoding="utf-8") as f:
            esri_json_data = json.load(f)

        console.print("✅ Input JSON loaded successfully")

        # Prepare excluded tables set
        excluded_tables_set = set(exclude_tables) if exclude_tables else None

        # Transform the schema
        console.print("⚙️ Transforming schema...")

        with console.status("[bold green]Processing datasets...") as status:
            schema = transform_esri_json(
                input_data=esri_json_data,
                target_prefix=target_prefix,
                excluded_tables=excluded_tables_set,
                exclude_metadata_fields=not include_metadata_fields,
                default_schema_prefix=schema_prefix,
            )

        console.print("✅ Schema transformation completed")

        # Save output using your existing serializer
        console.print(f"💾 Saving to: [bold green]{output}[/bold green]")

        save_esri_schema_to_file(
            schema=schema,
            filepath=str(output),
            indent=2 if pretty else None,
            ensure_ascii=False,
            add_timestamp=True,
        )

        console.print("✅ Output saved successfully")

        # Show summary if requested
        if show_summary:
            summary = schema.get_schema_summary()

            console.print("\n📊 [bold]Transformation Summary[/bold]")
            console.print("─" * 40)

            # Parse the summary and display nicely
            if isinstance(summary, str):
                console.print(summary)
            else:
                for key, value in summary.items():
                    console.print(
                        f"  {key.replace('_', ' ').title()}: [bold cyan]{value}[/bold cyan]"
                    )

            console.print(f"\n📁 Output file: [bold green]{output}[/bold green]")
            console.print(
                f"📏 File size: [bold cyan]{output.stat().st_size:,}[/bold cyan] bytes"
            )

            # Show some examples of what was found
            if schema.tables:
                console.print("\n📋 [bold]Sample Tables[/bold] (showing first 5):")
                for i, table_name in enumerate(list(schema.tables.keys())[:5]):
                    console.print(f"  {i + 1}. {table_name}")
                if len(schema.tables) > 5:
                    console.print(f"  ... and {len(schema.tables) - 5} more")

            if schema.feature_classes:
                console.print(
                    "\n🗺️  [bold]Sample Feature Classes[/bold] (showing first 5):"
                )
                for i, fc_name in enumerate(list(schema.feature_classes.keys())[:5]):
                    console.print(f"  {i + 1}. {fc_name}")
                if len(schema.feature_classes) > 5:
                    console.print(f"  ... and {len(schema.feature_classes) - 5} more")

        console.print(
            "\n🎉 [bold green]Transformation completed successfully![/bold green]"
        )

    except FileNotFoundError:
        console.print(f"❌ [bold red]Input file not found:[/bold red] {input_file}")
        raise click.Abort()

    except json.JSONDecodeError as e:
        console.print(f"❌ [bold red]Invalid JSON in input file:[/bold red] {e}")
        raise click.Abort()

    except Exception as e:
        console.print(f"❌ [bold red]Transformation failed:[/bold red] {e}")
        logger.error(f"Full error details: {traceback.format_exc()}")
        raise click.Abort()



@schema.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--name", "-n", help="Report name")
@click.option(
    "--format",
    "-f",
    multiple=True,
    default=["json"],
    type=click.Choice(["json", "html", "xml"]),
)
@click.option("--filter-prefix", help="Filter tables by prefix")
@click.option("--remove-prefix/--keep-prefix", default=False)
def extract(source, output, name, format, filter_prefix, remove_prefix):
    """Extract schema from GDB or SDE connection."""
    from ..utils.imports import HAS_ARCPY

    if not HAS_ARCPY:
        click.secho("❌ This command requires arcpy", fg="red")
        raise click.Abort()

    click.echo(f"Extracting schema from {source}...")

    try:
        schema_config, global_config = get_schema_configs(ctx)

        # Use config defaults if not specified
        output_dir = Path(output) if output else schema_config.output_dir
        formats = list(format) if format else schema_config.default_formats

        rprint(f"Extracting schema from {source}...")

        schema = extract_schema(
            source=source,
            output_dir=output_dir,
            name=name,
            formats=list(format),
        )

        # Appliquer les filtres si spécifiés
        if filter_prefix:
            # Filtrer le schéma...
            pass

        click.secho("✅ Schema extracted successfully", fg="green")

        # Your extraction logic here
        rprint("✅ Schema extracted successfully")
        rprint(f"Output directory: {output_dir}")

    except Exception as e:
        rprint(f"❌ Error: {e}")
        raise click.Abort()


@schema.command()
@click.argument("json_file", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output PlantUML file")
@click.option("--title", default="Database Schema")
@click.option("--no-fields", is_flag=True, help="Exclude field details")
@click.option("--no-relationships", is_flag=True, help="Exclude relationships")
@click.option("--filter", "-f", multiple=True, help="Include only these tables")
def diagram(json_file, output, title, no_fields, no_relationships, filter):
    """Generate PlantUML diagram from schema JSON."""
    click.echo(f"Generating diagram from {json_file}...")

    # Charger le JSON
    with open(json_file) as f:
        data = json.load(f)

    # Transformer en ESRISchema
    schema = transform_esri_json(data)

    # Générer le PlantUML
    puml_content = generate_plantuml_from_schema(
        schema=schema,
        title=title,
        include_fields=not no_fields,
        include_relationships=not no_relationships,
        filter_tables=list(filter) if filter else None,
    )

    # Sauvegarder
    Path(output).write_text(puml_content)
    click.secho(f"✅ Diagram saved to {output}", fg="green")


@schema.command()
@click.argument("old_schema", type=click.Path(exists=True))
@click.argument("new_schema", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output file for diff report")
@click.option(
    "--format",
    type=click.Choice(["json", "html", "markdown"]),
    default="json",
    help="Output format for the report",
)
@click.option(
    "--template",
    type=click.Choice(["summary", "full", "minimal"]),
    default="full",
    help="Template type to use for report generation",
)
@click.option(
    "--template-dir", type=click.Path(exists=True), help="Custom template directory"
)
@click.option("--filter-prefix", help="Filter objects by prefix (e.g., 'GC_')")
@click.option("--open-browser", is_flag=True, help="Open HTML report in browser")
def diff(
    old_schema,
    new_schema,
    output,
    format,
    template,
    template_dir,
    filter_prefix,
    open_browser,
):
    """Compare two schemas and generate diff report."""

    from ..schema.reporter import generate_report

    click.echo("Loading schemas...")

    # Charger les schémas
    with open(old_schema) as f:
        old_data = json.load(f)
    with open(new_schema) as f:
        new_data = json.load(f)

    # Transformer avec filtrage optionnel
    old = transform_esri_json(old_data, target_prefix=filter_prefix)
    new = transform_esri_json(
        new_data, target_prefix=filter_prefix
    )  # was filter_prefix TODO

    click.echo("Comparing schemas...")

    # Comparer
    diff = SchemaDiff(old, new)

    # Afficher le résumé en console
    summary = diff.get_summary()
    total_changes = sum(sum(counts.values()) for counts in summary.values())

    if total_changes == 0:
        click.secho("✅ No changes detected", fg="green")
    else:
        click.echo(f"\n📊 Summary of changes (Total: {total_changes}):")
        for category, counts in summary.items():
            category_total = sum(counts.values())
            if category_total > 0:
                click.echo(
                    f"  {category.replace('_', ' ').title()}: "
                    f"{counts['added']} added, "
                    f"{counts['removed']} removed, "
                    f"{counts['modified']} modified"
                )

    # Générer le rapport si demandé
    if output:
        click.echo(f"\nGenerating {format} report...")

        try:
            report_content = generate_report(
                diff=diff,
                template=template,
                format=format,
                template_dir=Path(template_dir) if template_dir else None,
                output_file=Path(output),
            )

            click.secho(f"✅ Report saved to {output}", fg="green")

            # Ouvrir dans le navigateur si demandé
            if open_browser and format == "html":
                import webbrowser

                webbrowser.open(f"file://{Path(output).absolute()}")
                click.echo("🌐 Report opened in browser")

        except Exception as e:
            click.secho(f"❌ Error generating report: {e}", fg="red")
            error_details = traceback.format_exc()
            click.secho(error_details, fg="red")

            raise click.Abort()
    else:
        # Afficher un rapport simple en console
        if total_changes > 0:
            click.echo("\nDetailed changes:")
            for change in diff.domain_changes[:5]:  # Limite à 5 pour éviter le spam
                click.echo(f"  Domain {change.domain_name}: {change.change_type.value}")
            for change in diff.table_changes[:5]:
                click.echo(f"  Table {change.table_name}: {change.change_type.value}")
            for change in diff.feature_class_changes[:5]:
                click.echo(
                    f"  Feature Class {change.table_name}: {change.change_type.value}"
                )

            if (
                len(diff.domain_changes)
                + len(diff.table_changes)
                + len(diff.feature_class_changes)
                > 15
            ):
                click.echo("  ... (use --output to see full report)")


@schema.command()
@click.argument("schema_file", type=click.Path(exists=True))
@click.option(
    "--template",
    "-t",
    type=click.Choice(["datamodel", "summary", "full"]),
    default="datamodel",
    help="Template type for documentation",
)
@click.option("--output", "-o", required=True, help="Output file")
@click.option(
    "--format",
    type=click.Choice(["html", "markdown", "pdf"]),
    default="html",
    help="Output format",
)
@click.option(
    "--template-dir", type=click.Path(exists=True), help="Custom template directory"
)
def report(schema_file, template, output, format, template_dir):
    """Generate documentation from schema."""
    click.echo(f"Generating {template} report...")
    from ..schema.reporter import generate_report

    # Charger le schéma
    with open(schema_file) as f:
        data = json.load(f)

    schema = transform_esri_json(data)

    # Générer le rapport selon le template
    try:
        report_content = generate_report(
            schema=schema,
            template=template,
            format=format,
            template_dir=Path(template_dir) if template_dir else None,
            output_file=Path(output),
        )

        click.secho(f"✅ Report saved to {output}", fg="green")

    except Exception as e:
        click.secho(f"❌ Error generating report: {e}", fg="red")
        raise click.Abort()


@schema.command()
@click.argument("old_schema", type=click.Path(exists=True))
@click.argument("new_schema", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o", required=True, help="Output directory for all reports"
)
@click.option("--filter-prefix", help="Filter objects by prefix")
@click.option("--open-browser", is_flag=True, help="Open HTML reports in browser")
def diff_all(old_schema, new_schema, output_dir, filter_prefix, open_browser):
    """Generate comprehensive diff reports in all formats."""
    from ..schema.reporter import generate_report

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo("Loading schemas...")

    # Charger les schémas
    with open(old_schema) as f:
        old_data = json.load(f)
    with open(new_schema) as f:
        new_data = json.load(f)

    # Transformer
    old = transform_esri_json(old_data, filter_prefix=filter_prefix)
    new = transform_esri_json(new_data, filter_prefix=filter_prefix)

    click.echo("Comparing schemas...")
    diff = SchemaDiff(old, new)

    # Générer tous les formats
    formats_templates = [
        ("json", "full"),
        ("html", "full"),
        ("html", "summary"),
        ("markdown", "full"),
        ("markdown", "summary"),
    ]

    generated_files = []

    for fmt, tpl in formats_templates:
        filename = f"schema_diff_{tpl}.{fmt}"
        output_file = output_path / filename

        try:
            click.echo(f"Generating {fmt} {tpl} report...")
            generate_report(
                diff=diff, template=tpl, format=fmt, output_file=output_file
            )
            generated_files.append(output_file)
            click.secho(f"✅ {filename}", fg="green")

        except Exception as e:
            click.secho(f"❌ Failed to generate {filename}: {e}", fg="red")

    # Résumé
    summary = diff.get_summary()
    total_changes = sum(sum(counts.values()) for counts in summary.values())

    click.echo(f"\n📊 Total changes detected: {total_changes}")
    click.echo(f"📁 Reports saved to: {output_path}")

    # Ouvrir dans le navigateur
    if open_browser and generated_files:
        html_files = [f for f in generated_files if f.suffix == ".html"]
        if html_files:
            import webbrowser

            webbrowser.open(f"file://{html_files[0].absolute()}")
            click.echo("🌐 HTML report opened in browser")
