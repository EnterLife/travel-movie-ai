import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from travelmovieai.analysis.audio import SAMPLE_RATE, analyze_audio, classify_audio_samples
from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import MediaType, StageStatus
from travelmovieai.domain.models import MediaAsset, Scene
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.stages.audio_analysis import AudioAnalysisStage
from travelmovieai.story.ranking import rank_scenes


def test_audio_classifier_detects_silence_and_speech_from_transcript() -> None:
    silence = np.zeros(SAMPLE_RATE, dtype=np.float64)
    speech_like = 0.08 * np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)

    silent = classify_audio_samples(uuid4(), silence)
    speech = classify_audio_samples(uuid4(), speech_like, transcript="hello from the beach")

    assert silent.primary_label == "silence"
    assert silent.noise_score == 0
    assert speech.primary_label == "speech"
    assert speech.speech_likelihood > 0.9
    assert speech.candidate_windows


def test_audio_features_affect_scene_ranking() -> None:
    quiet_asset = uuid4()
    noisy_asset = uuid4()
    speech_scene = Scene(
        asset_id=quiet_asset,
        start_seconds=0,
        end_seconds=3,
        quality_score=70,
        importance_score=65,
        metadata={
            "audio_features": {
                "primary_label": "speech",
                "speech_likelihood": 0.9,
                "noise_score": 20,
                "ambience_score": 70,
            }
        },
    )
    wind_scene = Scene(
        asset_id=noisy_asset,
        start_seconds=0,
        end_seconds=3,
        quality_score=70,
        importance_score=65,
        metadata={
            "audio_features": {
                "primary_label": "wind",
                "speech_likelihood": 0.0,
                "noise_score": 92,
                "ambience_score": 20,
            }
        },
    )

    ranked = rank_scenes([wind_scene, speech_scene])
    by_id = {scene.id: scene for scene in ranked}

    assert ranked[0].id == speech_scene.id
    assert by_id[speech_scene.id].metadata["ranking_factors"]["audio_bonus"] > 0
    assert by_id[wind_scene.id].metadata["ranking_factors"]["audio_penalty"] > 0


def test_audio_analysis_reports_ffmpeg_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = MediaAsset(
        path=tmp_path / "clip.mp4",
        relative_path=Path("clip.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=10,
        modified_at=datetime.now(UTC),
        modified_ns=123,
        duration_seconds=4,
        probe_metadata={"streams": [{"codec_type": "audio"}]},
    )
    scene = Scene(asset_id=asset.id, start_seconds=0, end_seconds=3)
    timeouts: list[float] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, int | float)
        timeouts.append(float(timeout))
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    monkeypatch.setattr("travelmovieai.analysis.audio.subprocess.run", fake_run)

    with pytest.raises(PipelineStageError, match="timed out after 0.25s"):
        analyze_audio([scene], [asset], "ffmpeg", timeout_seconds=0.25)

    assert timeouts == [0.25]


def test_audio_analysis_reports_sanitized_ffmpeg_decode_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset, scene = _audio_scene(tmp_path)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=b"",
            stderr=f"{asset.path} Authorization=private-token".encode(),
        )

    monkeypatch.setattr("travelmovieai.analysis.audio.subprocess.run", fake_run)

    with pytest.raises(PipelineStageError) as captured:
        analyze_audio([scene], [asset], "ffmpeg")

    message = str(captured.value)
    assert str(asset.path) not in message
    assert "private-token" not in message
    assert "<local-path>" in message
    assert "<redacted>" in message


def test_audio_stage_treats_empty_successful_decode_as_scene_without_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset, scene = _audio_scene(tmp_path)
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()
    with MediaAssetRepository(context.database_path) as repository:
        repository.initialize()
        repository.synchronize([asset], datetime.now(UTC))
        repository.synchronize_scenes([scene])

    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("travelmovieai.analysis.audio.subprocess.run", fake_run)
    stage = AudioAnalysisStage()

    completed = stage.run(context)
    cached = stage.run(context)

    assert completed.status is StageStatus.COMPLETED
    assert cached.status is StageStatus.CACHED
    assert calls == 1
    with MediaAssetRepository(context.database_path) as repository:
        stored = repository.list_scenes()[0]
    analysis = stored.metadata["audio_analysis"]
    assert analysis["has_audio"] is False
    assert analysis["primary_label"] == "silence"


def test_audio_analysis_rejects_invalid_pcm_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset, scene = _audio_scene(tmp_path)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        return subprocess.CompletedProcess(command, 0, stdout=b"\x00", stderr=b"")

    monkeypatch.setattr("travelmovieai.analysis.audio.subprocess.run", fake_run)

    with pytest.raises(PipelineStageError, match="invalid PCM payload"):
        analyze_audio([scene], [asset], "ffmpeg")


def _audio_scene(tmp_path: Path) -> tuple[MediaAsset, Scene]:
    asset = MediaAsset(
        path=tmp_path / "секретный клип.mp4",
        relative_path=Path("секретный клип.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=10,
        modified_at=datetime.now(UTC),
        modified_ns=123,
        duration_seconds=4,
        probe_metadata={"streams": [{"codec_type": "audio"}]},
    )
    return asset, Scene(asset_id=asset.id, start_seconds=0, end_seconds=3)
