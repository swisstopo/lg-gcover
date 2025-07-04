# gcover/config.py (exemple)
"""Configuration pour lg-gcover"""

# Instances SDE disponibles
SDE_INSTANCES = {
    "prod": "GCOVERP",
    "integration": "GCOVERI",
}

# Versions par défaut
DEFAULT_VERSIONS = {"GCOVERP": "SDE.DEFAULT", "GCOVERI": "SDE.DEFAULT"}

DEFAULT_CHUNK_SIZE = 1024

DEFAULT_NUM_WORKERS = 4
