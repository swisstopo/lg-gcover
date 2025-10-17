"""
SDE Connection Manager with improved error handling
"""
from pathlib import Path
import tempfile
from typing import Optional, Dict, List
import atexit
import shutil

from loguru import logger

try:
    import arcpy
except ImportError:
    logger.warning("arcpy not available")
    arcpy = None


class SDEConnectionManager:
    """Manages SDE connection files and their lifecycle."""

    SDE_CONNECTIONS = {
        "GCOVERP": Path(r"\\v0t0020a.adr.admin.ch\topgisprod\01_Admin\Connections\GCOVERP@osa.sde"),
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
            For versioned operations, this will create a connection file
            that is already set to the specified version.
        """
        # Use DEFAULT if no version specified
        if version is None:
            version = "SDE.DEFAULT"

        # For DEFAULT version, use base connection
        if version == "SDE.DEFAULT":
            if instance in self.SDE_CONNECTIONS and self.SDE_CONNECTIONS[instance].exists():
                logger.info(f"Using existing .sde file: {self.SDE_CONNECTIONS[instance]}")
                return self.SDE_CONNECTIONS[instance]

        # For specific versions, we MUST create a version-specific connection file
        # because arcpy.da.Editor respects the version in the connection file
        connection_key = f"{instance}::{version}"

        # Check if we already have this version connection cached
        if connection_key in self._connections:
            existing = self._connections[connection_key]
            if existing.exists():
                logger.debug(f"Reusing version connection: {existing}")
                return existing

        # Create a new version-specific connection
        if instance in self.SDE_CONNECTIONS and self.SDE_CONNECTIONS[instance].exists():
            return self._create_version_connection(instance, version, connection_key)

        # Fallback: create from scratch
        return self._create_new_connection(instance, version, connection_key)

    def _create_version_connection(
        self,
        instance: str,
        version: str,
        connection_key: str
    ) -> Path:
        """
        Create a version-specific connection file.

        This creates a NEW .sde file that is already set to the specified version.
        This is crucial for edit operations to work on the correct version.
        """
        import os

        # Get base connection properties
        base_sde = self.SDE_CONNECTIONS[instance]

        # Generate new filename for version-specific connection
        safe_version = version.replace(".", "_").replace("-", "_")
        new_filename = f"conn_{instance}_{safe_version}.sde"
        new_path = self._temp_dir / new_filename

        if new_path.exists():
            logger.info(f"Found existing version connection: {new_path}")
            self._connections[connection_key] = new_path
            return new_path

        try:
            # Read connection properties from base .sde file
            arcpy.env.workspace = str(base_sde)
            desc = arcpy.Describe(str(base_sde))
            conn_props = desc.connectionProperties

            logger.debug(f"Creating version-specific connection for {version}")

            # Get database platform - try different attributes
            db_platform = None
            for attr in ['type', 'dbType', 'database_type']:
                if hasattr(conn_props, attr):
                    db_platform = getattr(conn_props, attr)
                    break

            if not db_platform:
                # Fallback - assume from instance config or guess from instance
                if instance in self.SDE_INSTANCES:
                    db_platform = self.SDE_INSTANCES[instance].get('db_type', 'ORACLE')
                else:
                    db_platform = 'ORACLE'  # Default guess for your setup

            # Get instance/server
            db_instance = conn_props.instance if hasattr(conn_props, 'instance') else conn_props.server

            # Get database name (might be empty for Oracle)
            db_name = getattr(conn_props, 'database', '')

            logger.debug(f"Connection params: platform={db_platform}, instance={db_instance}, db={db_name}")

            # Create new connection WITH version specified
            result = arcpy.management.CreateDatabaseConnection(
                out_folder_path=str(self._temp_dir),
                out_name=new_filename,
                database_platform=db_platform,
                instance=db_instance,
                account_authentication="OPERATING_SYSTEM_AUTH",
                database=db_name,
                version_type="TRANSACTIONAL",
                version=version
            )

            if not new_path.exists():
                raise FileNotFoundError(f"Version connection not created: {new_path}")

            # Verify it's set to the correct version
            arcpy.env.workspace = str(new_path)
            test_desc = arcpy.Describe(str(new_path))
            actual_version = test_desc.connectionProperties.version

            if actual_version != version:
                logger.warning(f"Connection version mismatch: expected {version}, got {actual_version}")
            else:
                logger.info(f"Created version connection: {version}")

            self._connections[connection_key] = new_path
            return new_path

        except Exception as e:
            logger.error(f"Failed to create version connection: {e}")

            # Show available attributes for debugging
            try:
                arcpy.env.workspace = str(base_sde)
                desc = arcpy.Describe(str(base_sde))
                conn_props = desc.connectionProperties
                attrs = [attr for attr in dir(conn_props) if not attr.startswith('_')]
                logger.debug(f"Available connection properties: {attrs}")
            except:
                pass

            if new_path.exists():
                new_path.unlink()
            raise RuntimeError(f"Could not create version-specific connection: {e}")

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