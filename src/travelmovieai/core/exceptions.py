"""Project-specific exceptions."""


class TravelMovieError(Exception):
    """Base exception for expected application failures."""


class DependencyUnavailableError(TravelMovieError):
    """Raised when an external binary or optional package is unavailable."""


class MediaProbeError(TravelMovieError):
    """Raised when FFprobe cannot read a media file."""


class PipelineStageError(TravelMovieError):
    """Raised when a pipeline stage cannot complete."""
