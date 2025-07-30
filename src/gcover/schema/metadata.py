# Dans votre models.py, ajoutez en haut du fichier:
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Optional, Union


# Ajoutez cette classe complète après vos imports:
@dataclass
class SchemaMetadata:
    """Container for schema metadata extracted from ESRI JSON exports."""

    # From ESRI JSON root
    date_exported: Optional[datetime] = None
    dataset_type: Optional[str] = None
    catalog_path: Optional[str] = None
    gdb_name: Optional[str] = None
    workspace_type: Optional[str] = None
    workspace_factory_prog_id: Optional[str] = None
    connection_string: Optional[str] = None

    # Version information
    major_version: Optional[int] = None
    minor_version: Optional[int] = None
    bugfix_version: Optional[int] = None

    # From GDB filename parsing (e.g., "20250707_0330_2030-12-31.gdb")
    backup_date: Optional[datetime] = None
    backup_time: Optional[str] = None  # HHMM format
    reference_date: Optional[str] = None  # YYYY-MM-DD format
    backup_type: Optional[str] = None  # daily, weekly, monthly (inferred from path)

    # Additional computed metadata
    file_size: Optional[int] = None
    schema_complexity_score: Optional[int] = None
    geological_datamodel_version: Optional[str] = None

    def __post_init__(self):
        """Parse GDB name if available."""
        if self.gdb_name and not self.backup_date:
            self._parse_gdb_name()

    def _parse_gdb_name(self):
        """Parse GeoCover GDB filename to extract metadata."""
        # Pattern: YYYYMMDD_HHMM_YYYY-MM-DD.gdb
        pattern = r"(\d{8})_(\d{4})_(\d{4}-\d{2}-\d{2})\.gdb$"
        match = re.search(pattern, self.gdb_name)

        if match:
            date_str, time_str, ref_date = match.groups()

            try:
                # Parse backup date and time
                self.backup_date = datetime.strptime(date_str, "%Y%m%d")
                self.backup_time = time_str
                self.reference_date = ref_date

                # Infer backup type from catalog path
                if self.catalog_path:
                    path_lower = self.catalog_path.lower()
                    if "daily" in path_lower:
                        self.backup_type = "daily"
                    elif "weekly" in path_lower:
                        self.backup_type = "weekly"
                    elif "monthly" in path_lower:
                        self.backup_type = "monthly"
                    else:
                        self.backup_type = "unknown"

            except ValueError as e:
                # If date parsing fails, keep the strings for manual inspection
                pass

    @property
    def backup_datetime(self) -> Optional[datetime]:
        """Get combined backup date and time."""
        if self.backup_date and self.backup_time:
            try:
                time_dt = datetime.strptime(self.backup_time, "%H%M").time()
                return datetime.combine(self.backup_date.date(), time_dt)
            except ValueError:
                return self.backup_date
        return self.backup_date

    @property
    def is_file_gdb(self) -> bool:
        """Check if this is a File Geodatabase."""
        return self.workspace_type == "esriLocalDatabaseWorkspace"

    @property
    def is_sde(self) -> bool:
        """Check if this is an SDE/Enterprise Geodatabase."""
        return self.workspace_type == "esriRemoteDatabaseWorkspace"

    @property
    def version_string(self) -> str:
        """Get formatted version string."""
        if all(
            v is not None
            for v in [self.major_version, self.minor_version, self.bugfix_version]
        ):
            return f"{self.major_version}.{self.minor_version}.{self.bugfix_version}"
        return "Unknown"

    @property
    def age_days(self) -> Optional[int]:
        """Get age of backup in days from today."""
        if self.backup_date:
            return (datetime.now().date() - self.backup_date.date()).days
        return None

    @property
    def formatted_backup_info(self) -> str:
        """Get human-readable backup information."""
        if self.backup_date and self.backup_time:
            date_str = self.backup_date.strftime("%Y-%m-%d")
            time_str = f"{self.backup_time[:2]}:{self.backup_time[2:]}"
            type_str = f" ({self.backup_type})" if self.backup_type else ""
            return f"{date_str} {time_str}{type_str}"
        elif self.backup_date:
            return self.backup_date.strftime("%Y-%m-%d")
        return "Unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value

        # Add computed properties
        result["backup_datetime"] = (
            self.backup_datetime.isoformat() if self.backup_datetime else None
        )
        result["is_file_gdb"] = self.is_file_gdb
        result["is_sde"] = self.is_sde
        result["version_string"] = self.version_string
        result["age_days"] = self.age_days
        result["formatted_backup_info"] = self.formatted_backup_info

        return result
