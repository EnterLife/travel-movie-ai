"""Project-specific exceptions."""


class TravelMovieError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(TravelMovieError):
    """Raised when the local TOML configuration is invalid."""


class DependencyUnavailableError(TravelMovieError):
    """Raised when an external binary or optional package is unavailable."""


class MediaProbeError(TravelMovieError):
    """Raised when FFprobe cannot read a media file."""


class PipelineStageError(TravelMovieError):
    """Raised when a pipeline stage cannot complete."""


class ProjectArchiveError(PipelineStageError):
    """Raised when a project backup cannot be created or restored safely."""


class InvalidProjectPathError(TravelMovieError):
    """Raised when input and workspace paths cannot form a safe project."""


class WorkspaceIdentityError(InvalidProjectPathError):
    """Raised when a workspace does not belong to the requested source folder."""


class WorkspaceBusyError(TravelMovieError):
    """Raised when another active job already owns a workspace."""


class MontageError(TravelMovieError):
    """Raised when a quick montage cannot be planned or rendered."""


class VisionAnalysisError(TravelMovieError):
    """Raised when local semantic scene analysis cannot complete."""


class StoryGenerationError(TravelMovieError):
    """Raised when a local story model cannot produce a valid storyboard."""


class MusicGenerationError(TravelMovieError):
    """Raised when a local music model cannot generate the soundtrack."""
