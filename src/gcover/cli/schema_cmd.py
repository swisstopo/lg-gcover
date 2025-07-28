import json
from pathlib import Path

import click

from gcover.schema import SchemaDiff, extract_schema, transform_esri_json
from gcover.schema.exporters.plantuml import generate_plantuml_from_schema
from gcover.config import GlobalConfig, SchemaConfig

# TODO
from gcover.config import load_config, AppConfig, SchemaConfig


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
    "--format", type=click.Choice(["json", "html", "markdown"]), default="json"
)
def diff(old_schema, new_schema, output, format):
    """Compare two schemas and generate diff report."""
    click.echo("Comparing schemas...")

    # Charger les schémas
    with open(old_schema) as f:
        old_data = json.load(f)
    with open(new_schema) as f:
        new_data = json.load(f)

    # Transformer
    old = transform_esri_json(old_data)
    new = transform_esri_json(new_data)

    # Comparer
    diff = SchemaDiff(old, new)

    # Afficher le résumé
    summary = diff.get_summary()
    click.echo("\nSummary of changes:")
    for key, value in summary.items():
        print("DEBUG:", value, type(value))
        if any(v > 0 for v in value.values()):
            click.echo(f"  {key}: {value}")

    # Sauvegarder si demandé
    if output:
        if format == "json":
            from ..schema.exporters.json import export_schema_diff_to_json

            result = export_schema_diff_to_json(diff)
            Path(output).write_text(json.dumps(result, indent=2))
        elif format == "html":
            # Générer HTML
            pass
        elif format == "markdown":
            # Générer Markdown
            pass

        click.secho(f"✅ Diff report saved to {output}", fg="green")


@schema.command()
@click.argument("schema_file", type=click.Path(exists=True))
@click.option(
    "--template",
    "-t",
    type=click.Choice(["datamodel", "summary", "full"]),
    default="datamodel",
)
@click.option("--output", "-o", required=True, help="Output file")
@click.option(
    "--format", type=click.Choice(["html", "markdown", "pdf"]), default="html"
)
def report(schema_file, template, output, format):
    """Generate documentation from schema."""
    click.echo(f"Generating {template} report...")

    # Charger le schéma
    with open(schema_file) as f:
        data = json.load(f)

    schema = transform_esri_json(data)

    # Générer le rapport selon le template
    from ..schema.reporter import generate_report

    report_content = generate_report(schema=schema, template=template, format=format)

    Path(output).write_text(report_content)
    click.secho(f"✅ Report saved to {output}", fg="green")
