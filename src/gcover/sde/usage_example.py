"""
Example usage of the GeoCoverBridge class

This file demonstrates common workflows with the merged GeoCoverBridge class
for various GeoCover data operations.
"""

import geopandas as gpd
from shapely.geometry import box
from loguru import logger

from gcover.bridge import GeoCoverBridge
from gcover.config import FEAT_CLASSES_SHORTNAMES


def example_export_data():
    """Example: Export data to GeoDataFrame with spatial filter"""
    logger.info("=== Export Example ===")
    
    # Define a bounding box (Boltigen area)
    extent = box(2585000.0, 1158000.0, 2602500.0, 1170000.0)
    
    with GeoCoverBridge(uuid_field="UUID") as bridge:
        logger.info(f"Connected to: {bridge.version_info}")
        
        # Export bedrock data with spatial filter
        gdf = bridge.to_geopandas(
            "bedrock",  # Using shortcut name
            spatial_filter=None,  # Would use extent in real scenario
            fields=["UUID", "MORE_INFO", "OPERATOR"],
            max_features=100,
        )
        
        logger.info(f"Exported {len(gdf)} features")
        
        # Save to file using new GPKG utilities with progress
        output_file = "exported_bedrock.gpkg"
        bridge.save_to_gpkg(
            gdf, 
            output_file, 
            "bedrock",
            chunk_size=50,  # Small chunks for demo
            show_progress=True,
            compression=True
        )
        logger.success(f"Saved to {output_file}")
        
        return gdf


def example_direct_export():
    """Example: Direct export to GPKG without intermediate GeoDataFrame"""
    logger.info("=== Direct Export Example ===")
    
    with GeoCoverBridge(uuid_field="UUID") as bridge:
        # Export directly to GPKG - more memory efficient for large datasets
        output_file = bridge.export_to_gpkg(
            "bedrock",
            "direct_export.gpkg",
            layer_name="bedrock_data",
            where_clause="OPERATOR LIKE 'USER%'",
            fields=["UUID", "MORE_INFO", "OPERATOR", "DATEOFCHANGE"],
            max_features=500,
            chunk_size=100,
            parallel=False,  # Use False for smaller datasets to see detailed progress
            compression=True
        )
        
        logger.success(f"Direct export completed: {output_file}")


def example_large_dataset_export():
    """Example: Export large dataset with parallel processing"""
    logger.info("=== Large Dataset Export Example ===")
    
    with GeoCoverBridge(uuid_field="UUID") as bridge:
        # For large datasets, use parallel processing
        gdf = bridge.to_geopandas(
            "unco",  # Unconsolidated deposits
            max_features=5000,  # Simulate large dataset
        )
        
        logger.info(f"Exporting {len(gdf)} features with parallel processing")
        
        # Use parallel writing for better performance
        output_file = bridge.save_to_gpkg(
            gdf,
            "large_dataset.gpkg",
            "unco_deposits",
            chunk_size=500,
            parallel=True,  # Enable parallel processing
            compression=True
        )
        
        logger.success(f"Large dataset export completed: {output_file}")


def example_bulk_insert():
    """Example: Bulk insert new features"""
    logger.info("=== Bulk Insert Example ===")
    
    # This would typically be real data from external source
    # For demo, we'll create a simple test feature
    test_gdf = gpd.read_file("new_features.gpkg", layer="new_bedrock")
    
    with GeoCoverBridge(uuid_field="UUID") as bridge:
        if not bridge.is_writable:
            logger.warning("Database version is read-only, cannot insert")
            return
        
        # Insert new features
        result = bridge.bulk_insert(
            test_gdf,
            "bedrock",
            ignore_duplicates=True,
            dryrun=True,  # Set to False for actual insert
            operator="EXAMPLE_USER"
        )
        
        logger.info(f"Insert result: {result}")
        
        if result["success_count"] > 0:
            logger.success(f"Successfully inserted {result['success_count']} features")
        
        if result["duplicates_skipped"] > 0:
            logger.info(f"Skipped {result['duplicates_skipped']} duplicate features")
        
        if result["errors"]:
            logger.error(f"Errors encountered: {result['errors']}")


def example_bulk_update():
    """Example: Bulk update existing features"""
    logger.info("=== Bulk Update Example ===")
    
    # Load modified data
    modified_gdf = gpd.read_file("modified_features.gpkg", layer="modified_bedrock")
    
    with GeoCoverBridge(uuid_field="UUID") as bridge:
        if not bridge.is_writable:
            logger.warning("Database version is read-only, cannot update")
            return
        
        # Update specific fields only
        result = bridge.bulk_update(
            modified_gdf,
            "bedrock",
            update_fields=["MORE_INFO", "OPERATOR"],
            update_geometry=False,
            check_timestamp=True,
            dryrun=True,  # Set to False for actual update
            operator="EXAMPLE_USER"
        )
        
        logger.info(f"Update result: {result}")
        
        if result["success_count"] > 0:
            logger.success(f"Successfully updated {result['success_count']} features")
        
        if result["skipped_newer"] > 0:
            logger.info(f"Skipped {result['skipped_newer']} features (newer in DB)")
        
        if result["errors"]:
            logger.error(f"Errors encountered: {result['errors']}")


def example_mixed_operations():
    """Example: Execute mixed CRUD operations from single GeoDataFrame"""
    logger.info("=== Mixed Operations Example ===")
    
    # Load data with operation column
    mixed_gdf = gpd.read_file("mixed_operations.gpkg", layer="changes")
    
    # The GeoDataFrame should have an '_operation' column with values:
    # 'insert', 'update', 'delete', or None/NaN
    
    with GeoCoverBridge(uuid_field="UUID") as bridge:
        if not bridge.is_writable:
            logger.warning("Database version is read-only, cannot perform operations")
            return
        
        # Execute all operations in a single transaction
        results = bridge.execute_operations(
            mixed_gdf,
            "bedrock",
            operation_column="_operation",
            update_fields=["MORE_INFO"],
            confirm_deletes=False,  # Skip confirmation for demo
            dryrun=True,  # Set to False for actual operations
            operator="EXAMPLE_USER"
        )
        
        logger.info("Mixed operations results:")
        for operation, result in results.items():
            if isinstance(result, dict) and "success_count" in result:
                logger.info(f"  {operation}: {result['success_count']} successful")
                if result.get("errors"):
                    logger.warning(f"    Errors: {len(result['errors'])}")
            else:
                logger.info(f"  {operation}: {result}")


def example_version_selection():
    """Example: Interactive version selection"""
    logger.info("=== Version Selection Example ===")
    
    bridge = GeoCoverBridge(uuid_field="UUID")
    
    # List available versions
    versions = bridge.get_versions()
    logger.info(f"Available versions: {len(versions)}")
    for version in versions[:3]:  # Show first 3
        logger.info(f"  - {version['name']} (writable: {version['writable']})")
    
    # Find user version automatically
    user_version = bridge.find_user_version(interactive=False)
    if user_version:
        logger.info(f"Auto-selected version: {user_version['name']}")
    else:
        logger.info("No user version found, using default")
    
    # For interactive selection, use:
    # user_version = bridge.find_user_version(interactive=True)


def example_feature_class_shortcuts():
    """Example: Using feature class shortcuts"""
    logger.info("=== Feature Class Shortcuts Example ===")
    
    with GeoCoverBridge(uuid_field="UUID") as bridge:
        # Show available shortcuts
        logger.info("Available feature class shortcuts:")
        for shortcut, full_path in FEAT_CLASSES_SHORTNAMES.items():
            logger.info(f"  {shortcut} -> {full_path}")
        
        # Use shortcuts in operations
        bedrock_gdf = bridge.to_geopandas("bedrock", max_features=10)
        unco_gdf = bridge.to_geopandas("unco", max_features=10)
        
        logger.info(f"Bedrock features: {len(bedrock_gdf)}")
        logger.info(f"Unconsolidated deposit features: {len(unco_gdf)}")


def main():
    """Run all examples"""
    logger.add("gcover_example.log", rotation="1 MB")
    
    try:
        # Run examples
        example_version_selection()
        example_feature_class_shortcuts()
        
        # Export examples with new GPKG utilities
        gdf = example_export_data()
        example_direct_export()
        example_large_dataset_export()
        
        # Only run write operations if we have a writable version
        # example_bulk_insert()
        # example_bulk_update()
        # example_mixed_operations()
        
        logger.success("All examples completed successfully")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()