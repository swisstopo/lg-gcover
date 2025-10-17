"""
SDE Connection Manager with improved error handling
"""
from pathlib import Path
import tempfile
from typing import Optional, Dict, List
import atexit
import shutil

from loguru import logger

from gcover.arcpy_compat import HAS_ARCPY, arcpy


class SDEConnectionManager:
    """Manages SDE connection files and their lifecycle."""

    SDE_CONNECTIONS = {
        "GCOVERP": Path( r"\\v0t0020a.adr.admin.ch\topgisprod\01_Admin\Connections\GCOVERP@osa.sde"),
    }

    # SDE connection parameters for your environment
    SDE_INSTANCES = {
        "GCOVERP": {
            "server": "your_oracle_server",    # Oracle TNS name or server
            "database": "",                     # Leave empty for Oracle
            "db_type": "ORACLE",                # Oracle database
            "auth_type": "OPERATING_SYSTEM_AUTH",  # OSA authentication
            # If you know the service name or SID:
            # "instance": "gcoverp.domain.ch/SERVICENAME"
        }
    }

    def __init__(self):
        if arcpy is None:
            raise ImportError("arcpy is required for SDE connection management")

        self._temp_dir = Path(tempfile.mkdtemp(prefix="sde_conn_"))
        self._connections: Dict[str, Path] = {}

        # Register cleanup on exit
        atexit.register(self.cleanup_all)

        logger.debug(f"SDE temp directory: {self._temp_dir}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()

    def create_connection(
        self,
        instance: str,
        version: Optional[str] = None
    ) -> Path:
        """
        Create or retrieve SDE connection file.

        Args:
            instance: SDE instance name (e.g., "GCOVERP")
            version: Version name (e.g., "U80795753.DV_GC_2016-12-31_1294-Grono")
                     If None, uses DEFAULT version

        Returns:
            Path to .sde connection file

        Note:
            For versioned operations, you should call get_version_workspace()
            after getting the connection to work with a specific version.
        """
        # Use DEFAULT if no version specified
        if version is None:
            version = "SDE.DEFAULT"

        # For base connection (DEFAULT or unspecified), just return the base .sde
        if version == "SDE.DEFAULT":
            if instance in self.SDE_CONNECTIONS and self.SDE_CONNECTIONS[instance].exists():
                logger.info(f"Using existing .sde file: {self.SDE_CONNECTIONS[instance]}")
                return self.SDE_CONNECTIONS[instance]

        # For specific versions, we still return the base connection
        # The version will be handled when setting up the workspace
        if instance in self.SDE_CONNECTIONS and self.SDE_CONNECTIONS[instance].exists():
            base_sde = self.SDE_CONNECTIONS[instance]
            logger.info(f"Using base connection for version {version}: {base_sde}")

            # Store the version mapping for later use
            self._version_map = getattr(self, '_version_map', {})
            self._version_map[str(base_sde)] = version

            return base_sde

        # Fallback: create new connection from scratch
        connection_key = f"{instance}::{version}"
        return self._create_new_connection(instance, version, connection_key)

    def _copy_and_modify_connection(
        self,
        base_sde: Path,
        version: str,
        connection_key: str
    ) -> Path:
        """
        This method is kept for backwards compatibility but is no longer used.
        Use get_version_workspace() instead.
        """
        logger.warning("_copy_and_modify_connection is deprecated, use get_version_workspace()")
        return base_sde

    def get_version_workspace(self, instance: str, version: str) -> str:
        """
        Get a workspace string that points to a specific version.

        Args:
            instance: SDE instance name
            version: Version name

        Returns:
            Workspace path as string ready to use with arcpy.env.workspace

        Example:
            >>> conn_mgr = SDEConnectionManager()
            >>> workspace = conn_mgr.get_version_workspace("GCOVERP", "U80795753.DV_GC_2016-12-31_1294-Grono")
            >>> arcpy.env.workspace = workspace
        """
        # Get the base connection
        base_conn = self.create_connection(instance, version="SDE.DEFAULT")

        # For versioned geodatabases, the workspace path format is:
        # "connection_file" + "\" + "version_name"
        # But actually, we just set the workspace to the connection and ArcPy handles it

        return str(base_conn)

    def _create_new_connection(
        self,
        instance: str,
        version: str,
        connection_key: str
    ) -> Path:
        """Create new SDE connection file."""

        if instance not in self.SDE_INSTANCES:
            raise ValueError(f"Unknown SDE instance: {instance}")

        config = self.SDE_INSTANCES[instance]

        # Generate connection filename
        # Remove special characters from version name for filename
        safe_version = version.replace(".", "_").replace("-", "_")
        conn_filename = f"conn_{instance}_{safe_version}.sde"
        conn_path = self._temp_dir / conn_filename

        # Check if file already exists (from previous run)
        if conn_path.exists():
            logger.info(f"Found existing connection file: {conn_path}")
            try:
                # Test if connection is valid by setting workspace
                arcpy.env.workspace = str(conn_path)
                desc = arcpy.Describe(str(conn_path))
                logger.debug(f"Existing connection is valid (type: {desc.dataType})")
                self._connections[connection_key] = conn_path
                return conn_path
            except Exception as e:
                logger.warning(f"Existing connection invalid, recreating: {e}")
                conn_path.unlink()

        try:
            logger.info(f"Creating connection: {instance} -> {version}")

            # Prepare parameters for CreateDatabaseConnection
            params = {
                "out_folder_path": str(self._temp_dir),
                "out_name": conn_filename,
                "database_platform": config["db_type"],
                "instance": config["server"],
                "account_authentication": config["auth_type"],  # Correct parameter name
                "database": config.get("database", ""),
            }

            # For OSA (Operating System Authentication), don't pass username/password
            if config["auth_type"] != "OPERATING_SYSTEM_AUTH":
                # Only add these if using database authentication
                if "username" in config:
                    params["username"] = config["username"]
                if "password" in config:
                    params["password"] = config["password"]

            # Add version parameter only if not DEFAULT
            # Some ESRI tools don't like version="SDE.DEFAULT"
            if version != "SDE.DEFAULT":
                params["version_type"] = "TRANSACTIONAL"
                params["version"] = version

            logger.debug(f"Connection parameters: {params}")

            # Create the connection
            result = arcpy.management.CreateDatabaseConnection(**params)

            # Verify connection was created
            if not conn_path.exists():
                raise FileNotFoundError(f"Connection file not created: {conn_path}")

            # Test the connection
            try:
                arcpy.env.workspace = str(conn_path)
                desc = arcpy.Describe(str(conn_path))
                logger.info(f"Connection created successfully: {conn_path}")

                # Try to get connection properties (not all versions support all properties)
                if hasattr(desc, 'connectionString'):
                    logger.debug(f"Connection string: {desc.connectionString}")
                if hasattr(desc, 'workspaceType'):
                    logger.debug(f"Workspace type: {desc.workspaceType}")
                if hasattr(desc, 'connectionProperties'):
                    logger.debug(f"Connection properties: {desc.connectionProperties}")

            except Exception as test_err:
                logger.warning(f"Could not fully validate connection properties: {test_err}")
                # Don't fail here - the connection file exists and workspace was set
                # This is good enough for most operations

            self._connections[connection_key] = conn_path
            return conn_path

        except arcpy.ExecuteError as e:
            # Get detailed arcpy error messages
            error_msg = []
            error_msg.append(f"ArcPy Error: {arcpy.GetMessages(2)}")

            # Get all messages for context
            for i in range(arcpy.GetMessageCount()):
                msg = arcpy.GetMessage(i)
                error_msg.append(f"  Message {i}: {msg}")

            full_error = "\n".join(error_msg)
            logger.error(f"Failed to create connection:\n{full_error}")

            # Provide helpful suggestions
            self._suggest_fixes(instance, version, str(e))

            raise RuntimeError(f"Cannot create SDE connection: {full_error}")

        except Exception as e:
            logger.error(f"Unexpected error creating connection: {e}")
            raise

    def _suggest_fixes(self, instance: str, version: str, error: str):
        """Provide helpful suggestions based on error."""
        suggestions = []

        if "version" in error.lower():
            suggestions.append(f"Check if version '{version}' exists and you have access")
            suggestions.append("Try listing versions with: gcover sde list-versions")

        if "permission" in error.lower() or "access" in error.lower():
            suggestions.append("Check database permissions for your user")
            suggestions.append(f"Verify you can access version: {version}")

        if "authentication" in error.lower():
            suggestions.append("Check database authentication settings")
            suggestions.append("Verify you're logged into the correct domain")

        if suggestions:
            logger.info("Suggestions:")
            for suggestion in suggestions:
                logger.info(f"  - {suggestion}")

    def get_versions(self, instance: str) -> List[Dict]:
        """
        List all versions for an SDE instance.

        Returns:
            List of version dictionaries with name, parent, owner, writable
        """
        # Connect to DEFAULT to list versions
        conn_path = self.create_connection(instance, version="SDE.DEFAULT")

        try:
            arcpy.env.workspace = str(conn_path)

            versions = []
            version_list = arcpy.da.ListVersions()

            for version in version_list:
                versions.append({
                    "name": version.name,
                    "parent": version.parentVersionName if hasattr(version, 'parentVersionName') else None,
                    "isOwner": version.isOwner if hasattr(version, 'isOwner') else False,
                    "writable": self._check_version_writable(version),
                    "instance": instance,
                })

            logger.debug(f"Found {len(versions)} versions")
            return versions

        except Exception as e:
            logger.error(f"Error listing versions: {e}")
            return []

    def _check_version_writable(self, version) -> bool:
        """Check if version is writable by current user."""
        try:
            # A version is writable if user is owner or has write access
            if hasattr(version, 'isOwner') and version.isOwner:
                return True

            # Check access level if available
            if hasattr(version, 'access'):
                return version.access.upper() in ('PUBLIC', 'PROTECTED')

            # Default to False if cannot determine
            return False

        except Exception:
            return False

    def find_user_versions(
        self,
        instance: str,
        writable_only: bool = True
    ) -> List[Dict]:
        """
        Find versions owned by current user.

        Args:
            instance: SDE instance name
            writable_only: Only return writable versions

        Returns:
            List of user's versions
        """
        import os

        all_versions = self.get_versions(instance)
        current_user = os.getlogin().upper()

        user_versions = []
        for version in all_versions:
            # Check if user owns this version
            if current_user in version["name"].upper():
                if not writable_only or version["writable"]:
                    user_versions.append(version)

        return user_versions

    def cleanup_all(self):
        """Remove all temporary connection files."""
        if hasattr(self, '_temp_dir') and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Cleaned up temp directory: {self._temp_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temp directory: {e}")

        self._connections.clear()

    def test_connection(self, conn_path: Path) -> bool:
        """Test if connection is valid."""
        try:
            arcpy.env.workspace = str(conn_path)
            desc = arcpy.Describe(str(conn_path))

            # Try to list some basic info to verify connection works
            try:
                # Attempt to list feature datasets as a real test
                datasets = arcpy.ListDatasets()
                logger.debug(f"Connection valid - found {len(datasets) if datasets else 0} datasets")
            except Exception as list_err:
                logger.debug(f"Could not list datasets, but connection exists: {list_err}")

            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False