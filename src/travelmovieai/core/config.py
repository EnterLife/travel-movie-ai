"""Typed application settings loaded from a local TOML file."""

import sys
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from travelmovieai.core.exceptions import ConfigurationError

_BUNDLED_ROOT = getattr(sys, "_MEIPASS", None)
DEFAULT_CONFIG_PATH = (
    Path(_BUNDLED_ROOT) / "configs" / "settings.toml"
    if isinstance(_BUNDLED_ROOT, str)
    else Path("configs/settings.toml")
)


def validate_loopback_web_host(value: str) -> str:
    """Normalize and validate a host for the unauthenticated local web API."""

    normalized = value.strip()
    if normalized.casefold() not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "web_host must be a loopback address because the local API has no remote auth"
        )
    return normalized


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
    analysis_proxy_mode: Literal["auto", "disabled", "always"] = "auto"
    analysis_proxy_max_dimension: int = Field(default=1920, ge=640, le=3840)
    analysis_proxy_video_bitrate_mbps: float = Field(default=6.0, ge=1.0, le=50.0)
    analysis_proxy_timeout_seconds: float = Field(default=3600, gt=0)
    render_timeout_seconds: float = Field(default=7200, gt=0)
    render_disk_reserve_mb: int = Field(default=1024, ge=0, le=1_048_576)
    render_disk_safety_factor: float = Field(default=3.0, ge=1.0, le=10.0)
    vision_model: str = "auto"
    model_cache: Path = Path("models")
    allow_model_download: bool = True
    vision_provider: Literal["local", "qwen", "florence"] = "local"
    vision_model_pool_size: int = Field(default=1, ge=0, le=4)
    embedding_backend: Literal["feature-hash", "sentence-transformers"] = "feature-hash"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_index: Literal["auto", "faiss", "disabled"] = "auto"
    embedding_batch_size: int = Field(default=32, ge=1, le=1024)
    story_provider: Literal["deterministic", "local"] = "deterministic"
    story_model: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct", min_length=1, max_length=300)
    story_max_new_tokens: int = Field(default=768, ge=128, le=4096)
    music_library: Path = Path("assets/music")
    music_model: str = "auto"
    generated_music_filename: str = Field(
        default="generated_soundtrack.wav",
        min_length=1,
        pattern=r"^[^/\\]+\.wav$",
    )
    whisper_model: Literal["medium", "large-v3"] = "medium"
    voice_provider: Literal["disabled", "piper"] = "disabled"
    piper_binary: str = "piper"
    piper_model: Path | None = None
    voice_synthesis_timeout_seconds: float = Field(default=600, gt=0)
    project_cache_limit_mb: int = Field(default=20_480, ge=0, le=10_485_760)
    project_cache_target_ratio: float = Field(default=0.85, ge=0.1, le=1)
    device: Literal["auto", "cuda", "directml", "cpu"] = "auto"
    resource_mode: Literal["auto", "safe", "balanced", "performance"] = "auto"
    gpu_memory_reserve_mb: int = Field(default=1536, ge=512, le=16384)
    max_gpu_processes: int = Field(default=2, ge=1, le=8)
    batch_size: int = Field(default=0, ge=0)
    workers: int = Field(default=0, ge=0)
    web_host: str = "127.0.0.1"
    web_port: int = Field(default=8000, ge=1, le=65535)
    web_history_limit: int = Field(default=100, ge=1, le=1000)

    @field_validator("web_host")
    @classmethod
    def require_loopback_web_host(cls, value: str) -> str:
        return validate_loopback_web_host(value)


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
