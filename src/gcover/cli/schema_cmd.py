import json
from pathlib import Path
import traceback

import click

from gcover.schema import SchemaDiff, extract_schema, transform_esri_json
from gcover.schema.exporters.plantuml import generate_plantuml_from_schema
from gcover.config import GlobalConfig, SchemaConfig

# TODO
from gcover.config import load_config, AppConfig, SchemaConfig

from loguru import logger


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
