
import arcpy
import uuid
import pandas as pd
from pathlib import Path

# Configuration
DEFAULT_EXCLUDED_FIELDS = {
    "REVISION_MONTH", "CREATION_DAY", "REVISION_DAY",
    "CREATION_MONTH", "REVISION_YEAR", "CREATION_YEAR",
    "REVISION_DATE", "CREATION_DATE", "LAST_UPDATE",
    "OPERATOR", "DATEOFCREATION", "DATEOFCHANGE", "REASONFORCHANGE", "OBJECTORIGIN",
    "OBJECTORIGIN_YEAR", "OBJECTORIGIN_MONTH", "RC_ID", "WU_ID", "RC_ID_CREATION", "WU_ID_CREATION",
    "REVISION_QUALITY", "ORIGINAL_ORIGIN", "INTEGRATION_OBJECT_UUID",
    "CREATED_USER", "LAST_USER", "OBJECTID", "OID",
    "SHAPE", "SHAPE_Length", "SHAPE_Area", "Shape_Length", "Shape_Area",
    "GlobalID", "GLOBALID", "SHAPE.AREA", "SHAPE.LEN",
}
def generate_new_uuid(use_braces=True):
    """
    Generate a new UUID in ESRI format

    Parameters:
    -----------
    use_braces : bool
        If True, format as {UUID}, otherwise just UUID

    Returns:
    --------
    str : New UUID in uppercase
    """
    new_uuid = str(uuid.uuid4()).upper()

    if use_braces:
        return f"{{{new_uuid}}}"
    else:
        return new_uuid

def get_comparable_fields(feature_class, excluded_fields=None):
    """Get list of fields to compare, excluding metadata and system fields"""
    if excluded_fields is None:
        excluded_fields = DEFAULT_EXCLUDED_FIELDS

    all_fields = [f.name for f in arcpy.ListFields(feature_class)]
    comparable = [f for f in all_fields if f.upper() not in {e.upper() for e in excluded_fields}]

    return comparable

def compare_geometries(geom1, geom2, tolerance=0.001):
    """Compare two geometries for similarity"""
    if geom1 is None or geom2 is None:
        return geom1 is None and geom2 is None

    if geom1.type != geom2.type:
        return False

    return geom1.equals(geom2) or geom1.buffer(tolerance).contains(geom2.centroid)

def compare_attributes(row1, row2, fields, tolerance=1e-6):
    """Compare attribute values between two rows"""
    differences = []

    for i, field in enumerate(fields):
        val1, val2 = row1[i], row2[i]

        if val1 is None and val2 is None:
            continue
        if (val1 is None) != (val2 is None):
            differences.append(f"{field}: {val1} != {val2}")
            continue

        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > tolerance:
                differences.append(f"{field}: {val1} != {val2}")
        elif isinstance(val1, str) and isinstance(val2, str):
            if val1.strip().upper() != val2.strip().upper():
                differences.append(f"{field}: '{val1}' != '{val2}'")
        elif val1 != val2:
            differences.append(f"{field}: {val1} != {val2}")

    are_similar = len(differences) == 0
    return are_similar, differences



def get_current_version(sde_connection):
    """Get the current version for the SDE connection"""
    try:
        # Check version info
        desc = arcpy.Describe(sde_connection)
        if hasattr(desc, 'connectionProperties'):
            version = desc.connectionProperties.version
            return version
    except:
        pass
    return None

def list_available_versions(workspace):
    """List all available versions in the workspace"""
    try:
        versions = arcpy.da.ListVersions(workspace)
        return [v.name for v in versions]
    except Exception as e:
        print(f"Could not list versions: {e}")
        return []

def find_and_remove_duplicate_uuids(sde_connection, feature_class_name,
                                    uuid_field="UUID", dry_run=True,
                                    geometry_tolerance=0.001,
                                    version_name=None):
    """
    Find features with duplicate UUIDs and remove the one with smaller OID if similar

    Parameters:
    -----------
    sde_connection : str
        Path to .sde connection file
    feature_class_name : str
        Name of feature class (can be schema-qualified like 'TOPGIS_GC.GC_UNCO_DESPOSIT')
    uuid_field : str
        Name of UUID field
    dry_run : bool
        If True, only report what would be deleted without actually deleting
    geometry_tolerance : float
        Tolerance for geometry comparison
    version_name : str, optional
        Version name to use (e.g., 'GCOVERP.GCOVERP_Chatel'). If None, uses current version.

    Returns:
    --------
    pandas.DataFrame : Summary of actions taken
    """
    # Build full feature class path
    feature_class = f"{sde_connection}\\{feature_class_name}"

    # Check if feature class exists
    if not arcpy.Exists(feature_class):
        print(f"ERROR: Feature class does not exist: {feature_class}")
        return pd.DataFrame()

    print(f"Processing: {feature_class}")
    print("= " *80)

    # Get workspace
    workspace = arcpy.Describe(feature_class).path

    # Check current version
    current_version = get_current_version(sde_connection)
    print(f"Current version: {current_version}")

    # List available versions
    available_versions = list_available_versions(workspace)
    if available_versions:
        print(f"Available versions: {', '.join(available_versions)}")

    # Change to specified version if provided
    if version_name:
        print(f"Changing to version: {version_name}")
        try:
            arcpy.management.ChangeVersion(feature_class, 'TRANSACTIONAL', version_name)
            print(f"✓ Successfully changed to version {version_name}")
        except Exception as e:
            print(f"ERROR: Could not change to version {version_name}: {e}")
            return pd.DataFrame()

    # Check if data is versioned
    desc = arcpy.Describe(feature_class)
    is_versioned = desc.isVersioned if hasattr(desc, 'isVersioned') else False
    print(f"Versioned: {is_versioned}")

    # Get comparable fields
    comparable_fields = get_comparable_fields(feature_class)

    if uuid_field not in comparable_fields:
        print(f"ERROR: UUID field '{uuid_field}' not found in feature class")
        print(f"Available fields: {', '.join(comparable_fields)}")
        return pd.DataFrame()

    print(f"Comparing fields: {', '.join(comparable_fields)}")
    print()

    # Find all UUIDs and their OIDs
    uuid_dict = {}
    fields_to_read = ["OID@", "SHAPE@"] + comparable_fields

    print("Reading features...")
    with arcpy.da.SearchCursor(feature_class, fields_to_read) as cursor:
        for row in cursor:
            oid = row[0]
            uuid_idx = comparable_fields.index(uuid_field) + 2  # +2 for OID@ and SHAPE@
            uuid = row[uuid_idx]

            if uuid:  # Only process non-null UUIDs
                if uuid not in uuid_dict:
                    uuid_dict[uuid] = []
                uuid_dict[uuid].append(row)

    # Find duplicates
    duplicates = {uuid: rows for uuid, rows in uuid_dict.items() if len(rows) > 1}

    if not duplicates:
        print("No duplicate UUIDs found!")
        return pd.DataFrame()

    print(f"Found {len(duplicates)} duplicate UUIDs")
    print()

    # Process duplicates
    actions = []

    for uuid, rows in duplicates.items():
        print(f"\nUUID: {uuid}")
        print(f"  Found {len(rows)} features")

        # Sort by OID
        rows_sorted = sorted(rows, key=lambda x: x[0])

        # Compare all pairs
        for i in range(len(rows_sorted) - 1):
            row1 = rows_sorted[i]
            row2 = rows_sorted[i + 1]

            oid1, geom1 = row1[0], row1[1]
            oid2, geom2 = row2[0], row2[1]
            attrs1 = row1[2:]
            attrs2 = row2[2:]

            print(f"\n  Comparing OID {oid1} vs OID {oid2}:")

            # Compare geometries
            geom_similar = compare_geometries(geom1, geom2, geometry_tolerance)
            print(f"    Geometries similar: {geom_similar}")

            # Compare attributes
            attr_similar, differences = compare_attributes(attrs1, attrs2, comparable_fields)
            print(f"    Attributes similar: {attr_similar}")

            if differences:
                print(f"    Differences ({len(differences)}):")
                for diff in differences[:5]:  # Show first 5
                    print(f"      - {diff}")
                if len(differences) > 5:
                    print(f"      ... and {len(differences) - 5} more")

            # Decide whether to delete
            if geom_similar and attr_similar:
                oid_to_delete = min(oid1, oid2)
                oid_to_keep = max(oid1, oid2)

                print(f"    ✓ Features are similar")
                print(f"    → Will DELETE OID {oid_to_delete}, KEEP OID {oid_to_keep}")

                actions.append({
                    'uuid': uuid,
                    'deleted_oid': oid_to_delete,
                    'kept_oid': oid_to_keep,
                    'reason': 'Similar features',
                    'geom_similar': geom_similar,
                    'attr_similar': attr_similar
                })
            else:
                print(f"    ✗ Features are NOT similar - manual review needed")
                actions.append({
                    'uuid': uuid,
                    'deleted_oid': None,
                    'kept_oid': None,
                    'reason': 'Features differ - needs manual review',
                    'geom_similar': geom_similar,
                    'attr_similar': attr_similar
                })

    # Create summary DataFrame
    summary_df = pd.DataFrame(actions)

    # Perform deletions if not dry_run
    if not dry_run:
        oids_to_delete = [a['deleted_oid'] for a in actions if a['deleted_oid'] is not None]

        if oids_to_delete:
            print("\n" + "= " *80)
            print(f"DELETING {len(oids_to_delete)} FEATURES")
            print("= " *80)

            # Start edit session
            edit = arcpy.da.Editor(workspace)

            try:
                edit.startEditing(False, True)  # with_undo=False, multiuser_mode=True
                edit.startOperation()

                # Delete features
                oid_field = arcpy.Describe(feature_class).OIDFieldName
                where_clause = f"{oid_field} IN ({','.join(map(str, oids_to_delete))})"

                with arcpy.da.UpdateCursor(feature_class, [oid_field], where_clause) as cursor:
                    count = 0
                    for row in cursor:
                        cursor.deleteRow()
                        count += 1

                print(f"✓ Successfully deleted {count} duplicate features")

                edit.stopOperation()
                edit.stopEditing(True)  # save_changes=True

            except Exception as e:
                print(f"ERROR during deletion: {e}")
                try:
                    edit.stopOperation()
                    edit.stopEditing(False)  # save_changes=False
                except:
                    pass
                raise
    else:
        print("\n" + "= " *80)
        print("DRY RUN - No features were deleted")
        print("Set dry_run=False to perform actual deletions")
        print("= " *80)

    return summary_df

def find_and_fix_duplicate_uuids(sde_connection, feature_class_name,
                                 uuid_field="UUID", dry_run=True,
                                 geometry_tolerance=0.001,
                                 version_name=None,
                                 use_braces=True):
    """
    Find features with duplicate UUIDs and handle them:
    - If similar: delete the one with smaller OID
    - If different: assign new UUID to the one with smaller OID

    Parameters:
    -----------
    sde_connection : str
        Path to .sde connection file
    feature_class_name : str
        Name of feature class (can be schema-qualified like 'TOPGIS_GC.GC_UNCO_DESPOSIT')
    uuid_field : str
        Name of UUID field
    dry_run : bool
        If True, only report what would be changed without actually changing
    geometry_tolerance : float
        Tolerance for geometry comparison
    version_name : str, optional
        Version name to use (e.g., 'GCOVERP.GCOVERP_Chatel')
    use_braces : bool
        If True, format UUIDs as {UUID}, otherwise as UUID

    Returns:
    --------
    pandas.DataFrame : Summary of actions taken
    """
    # Build full feature class path
    feature_class = f"{sde_connection}\\{feature_class_name}"

    # Check if feature class exists
    if not arcpy.Exists(feature_class):
        print(f"ERROR: Feature class does not exist: {feature_class}")
        return pd.DataFrame()

    print(f"Processing: {feature_class}")
    print("= " *80)

    # Get workspace
    workspace = arcpy.Describe(feature_class).path

    # Check current version
    current_version = get_current_version(sde_connection)
    print(f"Current version: {current_version}")

    # Change to specified version if provided
    if version_name:
        print(f"Changing to version: {version_name}")
        try:
            arcpy.management.ChangeVersion(feature_class, 'TRANSACTIONAL', version_name)
            print(f"✓ Successfully changed to version {version_name}")
        except Exception as e:
            print(f"ERROR: Could not change to version {version_name}: {e}")
            return pd.DataFrame()

    # Check if data is versioned
    desc = arcpy.Describe(feature_class)
    is_versioned = desc.isVersioned if hasattr(desc, 'isVersioned') else False
    print(f"Versioned: {is_versioned}")

    # Get comparable fields
    comparable_fields = get_comparable_fields(feature_class)

    if uuid_field not in comparable_fields:
        print(f"ERROR: UUID field '{uuid_field}' not found in feature class")
        print(f"Available fields: {', '.join(comparable_fields)}")
        return pd.DataFrame()

    print(f"Comparing fields: {', '.join(comparable_fields)}")
    print()

    # Find all UUIDs and their OIDs
    uuid_dict = {}
    fields_to_read = ["OID@", "SHAPE@"] + comparable_fields

    print("Reading features...")
    with arcpy.da.SearchCursor(feature_class, fields_to_read) as cursor:
        for row in cursor:
            oid = row[0]
            uuid_idx = comparable_fields.index(uuid_field) + 2  # +2 for OID@ and SHAPE@
            uuid_value = row[uuid_idx]

            if uuid_value:  # Only process non-null UUIDs
                if uuid_value not in uuid_dict:
                    uuid_dict[uuid_value] = []
                uuid_dict[uuid_value].append(row)

    # Find duplicates
    duplicates = {uuid_val: rows for uuid_val, rows in uuid_dict.items() if len(rows) > 1}

    if not duplicates:
        print("No duplicate UUIDs found!")
        return pd.DataFrame()

    print(f"Found {len(duplicates)} duplicate UUIDs")
    print()

    # Process duplicates
    actions = []

    for uuid_value, rows in duplicates.items():
        print(f"\nUUID: {uuid_value}")
        print(f"  Found {len(rows)} features")

        # Sort by OID
        rows_sorted = sorted(rows, key=lambda x: x[0])

        # Compare all pairs
        for i in range(len(rows_sorted) - 1):
            row1 = rows_sorted[i]
            row2 = rows_sorted[i + 1]

            oid1, geom1 = row1[0], row1[1]
            oid2, geom2 = row2[0], row2[1]
            attrs1 = row1[2:]
            attrs2 = row2[2:]

            print(f"\n  Comparing OID {oid1} vs OID {oid2}:")

            # Compare geometries
            geom_similar = compare_geometries(geom1, geom2, geometry_tolerance)
            print(f"    Geometries similar: {geom_similar}")

            # Compare attributes
            attr_similar, differences = compare_attributes(attrs1, attrs2, comparable_fields)
            print(f"    Attributes similar: {attr_similar}")

            if differences:
                print(f"    Differences ({len(differences)}):")
                for diff in differences[:5]:  # Show first 5
                    print(f"      - {diff}")
                if len(differences) > 5:
                    print(f"      ... and {len(differences) - 5} more")

            # Decide action based on similarity
            if geom_similar and attr_similar:
                # Both are similar - delete the smaller OID
                oid_to_delete = min(oid1, oid2)
                oid_to_keep = max(oid1, oid2)

                print(f"    ✓ Features are SIMILAR")
                print(f"    → Will DELETE OID {oid_to_delete}, KEEP OID {oid_to_keep}")

                actions.append({
                    'uuid': uuid_value,
                    'action': 'DELETE',
                    'target_oid': oid_to_delete,
                    'kept_oid': oid_to_keep,
                    'new_uuid': None,
                    'reason': 'Similar features - deleted duplicate',
                    'geom_similar': geom_similar,
                    'attr_similar': attr_similar
                })
            else:
                # Features are different - assign new UUID to smaller OID
                oid_to_update = min(oid1, oid2)
                oid_to_keep = max(oid1, oid2)
                new_uuid = generate_new_uuid(use_braces)

                print(f"    ✗ Features are DIFFERENT")
                print(f"    → Will ASSIGN NEW UUID to OID {oid_to_update}")
                print(f"    → New UUID: {new_uuid}")
                print(f"    → Keep original UUID for OID {oid_to_keep}")

                actions.append({
                    'uuid': uuid_value,
                    'action': 'UPDATE_UUID',
                    'target_oid': oid_to_update,
                    'kept_oid': oid_to_keep,
                    'new_uuid': new_uuid,
                    'reason': 'Different features - assigned new UUID',
                    'geom_similar': geom_similar,
                    'attr_similar': attr_similar
                })

    # Create summary DataFrame
    summary_df = pd.DataFrame(actions)

    # Perform changes if not dry_run
    if not dry_run:
        print("\n" + "= " *80)
        print(f"APPLYING CHANGES")
        print("= " *80)

        # Start edit session
        edit = arcpy.da.Editor(workspace)

        try:
            edit.startEditing(False, True)  # with_undo=False, multiuser_mode=True
            edit.startOperation()

            # Get OID field name
            oid_field = arcpy.Describe(feature_class).OIDFieldName

            # Process deletions
            oids_to_delete = [a['target_oid'] for a in actions if a['action'] == 'DELETE']
            if oids_to_delete:
                print(f"\nDeleting {len(oids_to_delete)} duplicate features...")
                where_clause = f"{oid_field} IN ({','.join(map(str, oids_to_delete))})"

                with arcpy.da.UpdateCursor(feature_class, [oid_field], where_clause) as cursor:
                    count = 0
                    for row in cursor:
                        cursor.deleteRow()
                        count += 1
                print(f"✓ Deleted {count} features")

            # Process UUID updates
            uuid_updates = [(a['target_oid'], a['new_uuid']) for a in actions if a['action'] == 'UPDATE_UUID']
            if uuid_updates:
                print(f"\nUpdating {len(uuid_updates)} features with new UUIDs...")

                for oid, new_uuid_val in uuid_updates:
                    where_clause = f"{oid_field} = {oid}"

                    with arcpy.da.UpdateCursor(feature_class, [oid_field, uuid_field], where_clause) as cursor:
                        for row in cursor:
                            row[1] = new_uuid_val
                            cursor.updateRow(row)
                            print(f"  OID {oid}: {new_uuid_val}")

                print(f"✓ Updated {len(uuid_updates)} UUIDs")

            edit.stopOperation()
            edit.stopEditing(True)  # save_changes=True

            print("\n✓ All changes applied successfully")

        except Exception as e:
            print(f"\nERROR during changes: {e}")
            try:
                edit.stopOperation()
                edit.stopEditing(False)  # save_changes=False
            except:
                pass
            raise
    else:
        print("\n" + "= " *80)
        print("DRY RUN - No changes were made")
        print("Set dry_run=False to apply changes")
        print("= " *80)

    return summary_df





# Path to your SDE connection file
sde_connection = r"Y:\ArcGis\0_2_0\connections\GCOVERP_Chatel@osa.sde"

# Feature class name (schema-qualified)


feature_class_name = "TOPGIS_GC.GC_BEDROCK"
feature_class_name = "TOPGIS_GC.GC_LINEAR_OBJECTS"
feature_class_name = "TOPGIS_GC.GC_UNCO_DESPOSIT"

# IMPORTANT: Specify your version name
# Format is usually: SCHEMA.VERSION_NAME or just VERSION_NAME
# Examples: 'GCOVERP.GCOVERP_Chatel' or 'GCOVERP_Chatel'
version_name = "U80795753.DV_GC_2016-12-31_Comologno"  # Change this to your version!
use_braces = True

try:
    with arcpy.da.SearchCursor(feature_class_name, ["UUID"]) as cursor:
        for row in cursor:
            if row[0]:
                print(row[0])
                use_braces = row[0].startswith('{')
                print(f"Detected UUID format: {'with braces' if use_braces else 'without braces'}")
                print(f"Example: {row[0]}")
                break
except:
    print("Could not detect UUID format, using braces by default")

# Run in dry-run mode first
print("RUNNING IN DRY-RUN MODE")
print("= " *80)
'''summary = find_and_remove_duplicate_uuids(
        sde_connection=sde_connection,
        feature_class_name=feature_class_name,
        uuid_field="UUID",
        dry_run=False,
        geometry_tolerance=0.001,
        version_name=None  # Specify your version
    )'''

summary = find_and_fix_duplicate_uuids(
    sde_connection=sde_connection,
    feature_class_name=feature_class_name,
    uuid_field="UUID",
    dry_run=False,
    geometry_tolerance=0.001,
    version_name=None,
    use_braces=use_braces
)

# Display summary
print("\n" + "= " *80)
print("SUMMARY")
print("= " *80)
print(summary)

# Save summary
summary.to_csv(rf"Y:\ArcGis\0_2_0\RC1_{feature_class_name}_duplicate_uuid_summary.csv", index=False)
print(f"\nSummary saved to {feature_class_name}_duplicate_uuid_summary.csv")



