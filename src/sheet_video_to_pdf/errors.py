class SheetVideoToPdfError(Exception):
    """Base exception for user-facing tool errors."""


class ConfigError(SheetVideoToPdfError):
    """Raised when configuration or CLI input is invalid."""


class VideoReadError(SheetVideoToPdfError):
    """Raised when an MP4 cannot be opened or decoded."""


class NoNotationError(SheetVideoToPdfError):
    """Raised when no reconstructable notation pages are produced."""


class UnsupportedLayoutError(SheetVideoToPdfError):
    """Raised when detected notation cannot fit the configured PDF layout."""
