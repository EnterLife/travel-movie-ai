import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from travelmovieai.analysis import speech
from travelmovieai.analysis.speech import analyze_speech, speech_cache_key
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, Scene, SpeechSegment, SpeechTranscript
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.whisper import FasterWhisperProvider
from travelmovieai.pipeline.stages.speech_analysis import (
    _load_speech_checkpoints,
    _SpeechSceneCheckpoint,
)


class FakeSpeechProvider:
    name = "fake-whisper"
    model = "tiny-test"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio_path: Path) -> SpeechTranscript:
        assert audio_path.is_file()
        self.calls += 1
        return SpeechTranscript(
            text="Welcome to the city.",
            language="en",
            confidence=0.91,
            segments=[
                SpeechSegment(
                    start_seconds=0.4,
                    end_seconds=1.6,
                    text="Welcome to the city.",
                    confidence=0.91,
                )
            ],
        )


def test_whisper_provider_releases_loaded_model() -> None:
    released: list[bool] = []

    class LoadedModel:
        def unload_model(self) -> None:
            released.append(True)

    provider = FasterWhisperProvider("medium")
    provider._loaded_model = LoadedModel()

    provider.release()

    assert released == [True]
    assert provider._loaded_model is None


def test_speech_analysis_transcribes_and_reuses_cache(
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
    provider = FakeSpeechProvider()

    def fake_extract(
        ffmpeg_binary: str,
        source_path: Path,
        source_scene: Scene,
        output_path: Path,
        timeout_seconds: float = 120,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"wav")

    monkeypatch.setattr(speech, "_extract_scene_audio", fake_extract)
    first = analyze_speech(
        [scene],
        [asset],
        provider,
        "ffmpeg",
        tmp_path / "speech",
    )
    second = analyze_speech(
        first.scenes,
        [asset],
        provider,
        "ffmpeg",
        tmp_path / "speech",
    )

    assert first.scenes[0].transcript == "Welcome to the city."
    assert first.scenes[0].metadata["speech_language"] == "en"
    assert first.scenes[0].metadata["speech_segments"][0]["start_seconds"] == 0.4
    assert provider.calls == 1
    assert second.cached_count == 1


def test_speech_analysis_checkpoints_before_later_scene_failure(
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
    scenes = [
        Scene(asset_id=asset.id, start_seconds=0, end_seconds=2),
        Scene(asset_id=asset.id, start_seconds=2, end_seconds=4),
    ]

    def fake_extract(
        ffmpeg_binary: str,
        source_path: Path,
        source_scene: Scene,
        output_path: Path,
        timeout_seconds: float = 120,
    ) -> None:
        del ffmpeg_binary, source_path, source_scene, timeout_seconds
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"wav")

    class FailSecondProvider(FakeSpeechProvider):
        def transcribe(self, audio_path: Path) -> SpeechTranscript:
            if self.calls == 1:
                raise RuntimeError("synthetic whisper failure")
            return super().transcribe(audio_path)

    monkeypatch.setattr(speech, "_extract_scene_audio", fake_extract)
    checkpoints: list[Scene] = []
    with pytest.raises(RuntimeError, match="synthetic whisper failure"):
        analyze_speech(
            scenes,
            [asset],
            FailSecondProvider(),
            "ffmpeg",
            tmp_path / "speech",
            checkpoint=checkpoints.append,
        )

    assert len(checkpoints) == 1
    resumed_provider = FakeSpeechProvider()
    resumed = analyze_speech(
        [checkpoints[0], scenes[1]],
        [asset],
        resumed_provider,
        "ffmpeg",
        tmp_path / "speech",
    )
    assert resumed.cached_count == 1
    assert resumed.transcribed_count == 1
    assert resumed_provider.calls == 1


def test_speech_checkpoint_invalidates_with_source_identity(tmp_path: Path) -> None:
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
    checkpoint = scene.model_copy(
        update={
            "transcript": "Saved before interruption.",
            "metadata": {
                "speech_cache_key": speech_cache_key(scene, asset, "tiny-test"),
            },
        }
    )
    checkpoint_dir = tmp_path / "speech-shards"
    checkpoint_path = checkpoint_dir / f"{scene.id}.json"
    write_json_atomic(
        checkpoint_path,
        _SpeechSceneCheckpoint(
            config_fingerprint="a" * 64,
            scene=checkpoint,
        ),
    )

    restored = _load_speech_checkpoints(
        [scene],
        [asset],
        "tiny-test",
        "a" * 64,
        checkpoint_dir,
    )
    invalid_config = _load_speech_checkpoints(
        [scene],
        [asset],
        "tiny-test",
        "b" * 64,
        checkpoint_dir,
    )
    write_json_atomic(
        checkpoint_path,
        _SpeechSceneCheckpoint(
            config_fingerprint="a" * 64,
            scene=checkpoint,
        ),
    )
    changed_asset = asset.model_copy(update={"modified_ns": 456})
    invalidated = _load_speech_checkpoints(
        [scene],
        [changed_asset],
        "tiny-test",
        "a" * 64,
        checkpoint_dir,
    )

    assert restored[0].transcript == "Saved before interruption."
    assert invalid_config == [scene]
    assert invalidated == [scene]
    assert not checkpoint_path.exists()


def test_speech_analysis_reports_ffmpeg_timeout(
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

    monkeypatch.setattr("travelmovieai.analysis.speech.subprocess.run", fake_run)

    with pytest.raises(PipelineStageError, match="timed out after 0.25s"):
        analyze_speech(
            [scene],
            [asset],
            FakeSpeechProvider(),
            "ffmpeg",
            tmp_path / "speech",
            timeout_seconds=0.25,
        )

    assert timeouts == [0.25]
