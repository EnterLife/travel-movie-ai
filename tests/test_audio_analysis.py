import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from travelmovieai.analysis.audio import SAMPLE_RATE, analyze_audio, classify_audio_samples
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, Scene
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
