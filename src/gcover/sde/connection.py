"""
SDE Connection Manager for GeoCover tool
Handles database connections, versions, and workspace management
"""

import os
import tempfile
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import contextlib

from loguru import logger

from .config import DEFAULT_VERSION, DB_INSTANCES

try:
    import arcpy
except ImportError:
    # Mock arcpy for documentation generation or non-ArcGIS environments
    class MockArcpy:
        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            return self


    arcpy = MockArcpy()


class ReadOnlyError(Exception):
    """Raised when attempting to write to a read-only version"""
    pass


class SDEConnectionManager:
    """
    Manages SDE database connections for GeoCover operations.

    Handles temporary connection files, version selection, and workspace management.
    """

    def __init__(self, instance: str = "GCOVERP", version: str = DEFAULT_VERSION):
        """
        Initialize SDE connection manager.

        Args:
            instance: Database instance name (GCOVERP or GCOVERI)
            version: Database version name
        """
        self.instance = instance
        self.version_name = version
        self.is_writable = False

        # Create unique temporary directory for this instance
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sde_path = None
        self.workspace = None

        logger.debug(f"Initialized connection manager for {instance}/{version}")

    def __repr__(self):
        return (f"<SDEConnectionManager: instance={self.instance}, "
                f"version={self.version_name}, writable={self.is_writable}>")

    def _check_arcpy(self):
        """Verify arcpy is available"""
        if not hasattr(arcpy, 'management'):
            raise ModuleNotFoundError(
                "The 'arcpy' module is required for this operation. "
                "Please ensure ArcGIS Pro is installed and available."
            )

    def connect(self, version: Optional[str] = None) -> str:
        """
        Create a temporary SDE connection file and set as workspace.

        Args:
            version: Optional version to connect to (overrides instance version)

        Returns:
            str: Path to the workspace
        """
        self._check_arcpy()

        # Generate unique name for sde file
        timestamp = dt.now().strftime("%Y%m%d%H%M%S")
        sde_name = f"temp_{self.instance}_{timestamp}.sde"
        self.sde_path = self.temp_dir / sde_name

        # Use provided version or instance version
        version_to_use = version or self.version_name

        logger.info(f"Creating SDE connection to {self.instance}/{version_to_use}")

        try:
            # Create the connection
            arcpy.management.CreateDatabaseConnection(
                out_folder_path=str(self.temp_dir),
                out_name=sde_name,
                database_platform="ORACLE",
                instance=self.instance,
                account_authentication="OPERATING_SYSTEM_AUTH",
                username="",
                password="",
                version=version_to_use,
                save_user_pass="DO_NOT_SAVE_USERNAME",
            )

            # Set as workspace
            arcpy.env.workspace = str(self.sde_path)
            self.workspace = str(self.sde_path)

            logger.success(f"Connected to workspace: {self.workspace}")
            return self.workspace

        except Exception as e:
            logger.error(f"Failed to create SDE connection: {e}")
            raise

    def get_versions(self) -> List[Dict[str, any]]:
        """
        List all versions in the geodatabase with metadata.

        Returns:
            List of version dictionaries with name, parent, isOwner, writable
        """
        temp_conn = None
        try:
            # Create temporary connection to default version
            temp_conn = self.connect(version=DEFAULT_VERSION)
            versions = []

            ver_list = arcpy.da.ListVersions(temp_conn)
            for version in ver_list:
                versions.append({
                    "name": version.name,
                    "parent": version.parentVersionName,
                    "isOwner": version.isOwner,
                    "writable": version.isOwner,  # Owners can write to their versions
                    "description": getattr(version, 'description', ''),
                })

            logger.debug(f"Found {len(versions)} versions")
            return versions

        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []
        finally:
            if temp_conn and self.sde_path and self.sde_path.exists():
                try:
                    os.remove(self.sde_path)
                except Exception:
                    pass

    def find_user_version(self, interactive: bool = False) -> Optional[Dict[str, any]]:
        """
        Find and set the user's version.

        Args:
            interactive: If True, allows interactive version selection

        Returns:
            Dictionary with version information or None
        """
        current_user = os.getlogin().upper()
        versions = self.get_versions()

        if not versions:
            logger.warning("No versions available")
            self.version_name = DEFAULT_VERSION
            self.is_writable = False
            return None

        if interactive:
            return self._interactive_version_selection(versions, current_user)
        else:
            return self._auto_version_selection(versions, current_user)

    def _auto_version_selection(self, versions: List[Dict], current_user: str) -> Optional[Dict]:
        """Automatically select user's version"""
        user_version = None
        for version in versions:
            if current_user in version["name"].upper():
                user_version = version
                break

        if user_version:
            self.version_name = user_version["name"]
            self.is_writable = user_version["writable"]
            logger.info(f"Using user version: {user_version['name']}")
            return user_version
        else:
            self.version_name = DEFAULT_VERSION
            self.is_writable = False
            logger.warning(f"Using default version: {DEFAULT_VERSION}")
            return None

    def _interactive_version_selection(self, versions: List[Dict], current_user: str) -> Optional[Dict]:
        """Interactive version selection with user input"""
        print(f"\nCurrent user: {current_user}")
        print("Available database connections:")
        print("-" * 50)

        # Display available versions
        for i, version in enumerate(versions, 1):
            status_indicators = []
            if version["isOwner"]:
                status_indicators.append("Owner")
            if version["writable"]:
                status_indicators.append("Writable")
            if current_user in version["name"].upper():
                status_indicators.append("⭐ Suggested")

            status = f" ({', '.join(status_indicators)})" if status_indicators else ""
            parent_info = f" [Parent: {version['parent']}]" if version["parent"] else ""

            print(f"{i:2d}. {version['name']}{status}{parent_info}")

        # Add default option
        print(f"{len(versions) + 1:2d}. Use default version ({DEFAULT_VERSION})")

        while True:
            try:
                choice = input(f"\nSelect a connection (1-{len(versions) + 1}): ").strip()

                if not choice:
                    continue

                choice_num = int(choice)

                if 1 <= choice_num <= len(versions):
                    selected_version = versions[choice_num - 1]
                    self.version_name = selected_version["name"]
                    self.is_writable = selected_version["writable"]

                    print(f"✓ Selected: {selected_version['name']}")
                    if not selected_version["writable"]:
                        print("⚠️  Note: This connection is read-only")

                    logger.info(f"User selected version: {selected_version}")
                    return selected_version

                elif choice_num == len(versions) + 1:
                    # Default version selected
                    self.version_name = DEFAULT_VERSION
                    self.is_writable = False
                    print(f"✓ Using default version: {DEFAULT_VERSION}")
                    logger.info(f"User selected default version")
                    return None
                else:
                    print(f"Please enter a number between 1 and {len(versions) + 1}")

            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                logger.info("Version selection cancelled by user")
                self.version_name = DEFAULT_VERSION
                self.is_writable = False
                return None

    def get_feature_classes(self, ignore_integration: bool = True, as_fc_name: bool = True) -> List[str]:
        """
        Get list of feature classes in the geodatabase.

        Args:
            ignore_integration: Skip feature classes ending with '_I'
            as_fc_name: Return as 'dataset/fc' format vs tuple

        Returns:
            List of feature class names or tuples
        """
        if not self.workspace:
            raise RuntimeError("No active workspace connection")

        arcpy.env.workspace = self.workspace
        feat_classes = []

        try:
            datasets = arcpy.ListDatasets()
            for dataset in datasets:
                feature_classes = arcpy.ListFeatureClasses(feature_dataset=dataset)
                for fc in feature_classes:
                    if ignore_integration and fc.endswith("_I"):
                        continue

                    if as_fc_name:
                        name = f"{dataset}/{fc}"
                    else:
                        name = (dataset, fc)

                    feat_classes.append(name)

        except Exception as e:
            logger.error(f"Error retrieving feature classes: {e}")

        logger.debug(f"Found {len(feat_classes)} feature classes")
        return feat_classes

    @property
    def version_info(self) -> Dict[str, any]:
        """Return current version information"""
        return {
            "name": self.version_name,
            "writable": self.is_writable,
            "instance": self.instance,
            "workspace": self.workspace,
        }

    @property
    def rc_full(self) -> str:
        """Return full RC version based on version name"""
        if "2030" in self.version_name or "SDE.DEFAULT" in self.version_name:
            return "2030-12-31"
        else:
            return "2016-12-31"

    @property
    def rc_short(self) -> str:
        """Return short RC version"""
        return "RC2" if "2030-12-31" in self.rc_full else "RC1"

    @contextlib.contextmanager
    def transaction(self):
        """Context manager for edit transactions"""
        if not self.is_writable:
            raise ReadOnlyError(f"Version {self.version_name} is read-only")

        edit = arcpy.da.Editor(self.workspace)
        edit.startEditing(False, True)
        edit.startOperation()

        try:
            yield edit
            edit.stopOperation()
            edit.stopEditing(save_changes=True)
            logger.debug("Transaction committed successfully")
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            if edit.isEditing:
                edit.stopOperation()
                edit.stopEditing(save_changes=False)
            raise
        finally:
            arcpy.management.ClearWorkspaceCache()

    def cleanup(self):
        """Clean up temporary files and connections"""
        try:
            if self.sde_path and self.sde_path.exists():
                os.remove(self.sde_path)
                logger.debug(f"Removed SDE file: {self.sde_path}")

            if self.temp_dir.exists():
                self.temp_dir.rmdir()
                logger.debug(f"Removed temp directory: {self.temp_dir}")

        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def __enter__(self):
        """Context manager entry"""
        self.find_user_version()
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.cleanup()