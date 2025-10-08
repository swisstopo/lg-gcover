class SDEConnectionError(Exception):
    """Raised when there is a problem connecting to the SDE database."""

    pass


class SDEVersionError(Exception):
    """Raised when an SDE version is invalid or cannot be resolved."""

    pass
