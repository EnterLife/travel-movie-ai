import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from travelmovieai.analysis import speech
from travelmovieai.analysis.speech import analyze_speech
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, Scene, SpeechSegment, SpeechTranscript


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
