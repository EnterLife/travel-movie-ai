"""Application settings loaded from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by CLI commands and pipeline stages."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRAVELMOVIEAI_",
        extra="ignore",
    )

    workspace: Path = Path("workspace")
    database_filename: str = Field(
        default="project.db",
        min_length=1,
        pattern=r"^[^/\\]+$",
    )
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    lm_studio_url: str = "http://localhost:1234/v1"
    vision_provider: Literal["qwen", "florence"] = "qwen"
    whisper_model: Literal["medium", "large-v3"] = "medium"
    device: Literal["auto", "cuda", "directml", "cpu"] = "auto"
    cloud_enabled: bool = False
    batch_size: int = Field(default=8, ge=1)
    workers: int = Field(default=4, ge=1)
    web_host: str = "127.0.0.1"
    web_port: int = Field(default=8000, ge=1, le=65535)
