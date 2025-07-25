# lg-gcover

**A Python library and CLI tool for working with Swiss GeoCover geological vector data**

lg-gcover simplifies the processing and analysis of geological vector datasets from the Swiss national Geological Survey (swisstopo). Built on modern geospatial Python tools like GeoPandas and Shapely, it provides both programmatic APIs and command-line utilities for geological data workflows.

## Key Features
- **CLI Interface**: Easy-to-use `gcover` command for batch processing
- **GeoPandas Integration**: Seamless integration with the Python geospatial ecosystem
- **ESRI Compatibility**: Optional support for ArcGIS Pro workflows via arcpy
- **Rich Output**: Beautiful terminal output with progress indicators and structured logging
- **Flexible Data Handling**: Support for various geological vector formats and projections

Perfect for geologists, GIS analysts, and researchers working with Swiss geological datasets who need efficient, reproducible data processing workflows.


## Usage


### SDE connection

    # Voir vos versions utilisateur
    gcover sde user-versions

    # Lister toutes les versions de GCOVERP
    gcover sde versions -i GCOVERP

    # Test de connexion interactif
    gcover sde connect -i GCOVERP --interactive

    # Export JSON des versions
    gcover sde versions -f json > versions.json

    # Nettoyer les connexions
    gcover sde connections --cleanup