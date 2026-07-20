"""Per-project execution context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from travelmovieai.application.workspace_identity import ensure_workspace_identity
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import StoryStyle
from travelmovieai.domain.models import QuickMontageSettings
from travelmovieai.infrastructure.system import ResourceProfile

ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class ProjectContext:
    input_path: Path
    workspace: Path
    settings: Settings
    output_path: Path | None = None
    style: StoryStyle = StoryStyle.CINEMATIC
    montage_settings: QuickMontageSettings | None = None
    variant_name: str = "Default"
    variant_slug: str = "default"
    progress: ProgressCallback | None = None
    resources: ResourceProfile | None = None

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
        ensure_workspace_identity(self.input_path, self.workspace)
        for path in (self.frames_dir, self.cache_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)
