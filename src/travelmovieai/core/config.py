"""Typed application settings loaded from a local TOML file."""

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from travelmovieai.core.exceptions import ConfigurationError

DEFAULT_CONFIG_PATH = Path("configs/settings.toml")


class Settings(BaseModel):
    """Runtime settings shared by CLI commands and pipeline stages."""

    model_config = ConfigDict(extra="forbid")

    workspace: Path = Path("workspace")
    database_filename: str = Field(
        default="project.db",
        min_length=1,
        pattern=r"^[^/\\]+$",
    )
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    frame_extraction_timeout_seconds: float = Field(default=120, gt=0)
    render_timeout_seconds: float = Field(default=7200, gt=0)
    vision_model: str = "auto"
    model_cache: Path = Path("models")
    allow_model_download: bool = True
    vision_provider: Literal["local", "qwen", "florence"] = "local"
    music_library: Path = Path("assets/music")
    music_model: str = "auto"
    generated_music_filename: str = Field(
        default="generated_soundtrack.wav",
        min_length=1,
        pattern=r"^[^/\\]+\.wav$",
    )
    whisper_model: Literal["medium", "large-v3"] = "medium"
    device: Literal["auto", "cuda", "directml", "cpu"] = "auto"
    resource_mode: Literal["safe", "balanced", "performance"] = "balanced"
    gpu_memory_reserve_mb: int = Field(default=1536, ge=512, le=16384)
    max_gpu_processes: int = Field(default=2, ge=1, le=8)
    batch_size: int = Field(default=0, ge=0)
    workers: int = Field(default=0, ge=0)
    web_host: str = "127.0.0.1"
    web_port: int = Field(default=8000, ge=1, le=65535)
    web_history_limit: int = Field(default=100, ge=1, le=1000)


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load the checked-in local configuration or use typed defaults."""

    resolved = path.expanduser()
    if not resolved.is_file():
        return Settings()
    try:
        with resolved.open("rb") as config_file:
            payload: dict[str, Any] = tomllib.load(config_file)
        return Settings.model_validate(payload)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        raise ConfigurationError(
            f"Could not read configuration {resolved.resolve()}: {error}"
        ) from error
