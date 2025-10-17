



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
        with arcpy.da.SearchCursor(
            table_name, final_fields, where_clause=query
        ) as cursor:
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

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(simple_dict, f, ensure_ascii=False, indent=2)
    else:
        # Export as records for complex tables
        df.to_json(json_path, indent=2, orient="records", force_ascii=False)

