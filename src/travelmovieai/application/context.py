"""Per-project execution context."""

from dataclasses import dataclass
from pathlib import Path

from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import StoryStyle


@dataclass(frozen=True, slots=True)
class ProjectContext:
    input_path: Path
    workspace: Path
    settings: Settings
    output_path: Path | None = None
    style: StoryStyle = StoryStyle.CINEMATIC
    cloud: bool = False

    @property
    def frames_dir(self) -> Path:
        return self.workspace / "frames"

    @property
    def cache_dir(self) -> Path:
        return self.workspace / "cache"

    @property
    def artifacts_dir(self) -> Path:
        return self.workspace / "artifacts"

    @property
    def database_path(self) -> Path:
        return self.workspace / self.settings.database_filename

    def prepare(self) -> None:
        for path in (self.workspace, self.frames_dir, self.cache_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)
