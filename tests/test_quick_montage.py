import json
import math
import shutil
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest

from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import MediaProbeError, MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MediaAsset,
    MediaScanReport,
    MontageClip,
    MontageQualityReport,
    QuickMontagePlan,
    QuickMontageSettings,
    SceneUnderstanding,
)
from travelmovieai.editing.renderer import (
    QuickMontageRenderer,
    _build_filter_graph,
    _build_hard_cut_audio_graph,
    _is_nvenc_unavailable_error,
    _NvencUnavailableError,
    _transition_duration,
)
from travelmovieai.editing.timeline import build_quick_montage_plan
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.system import CudaStatus


class FakeVisionProvider:
    name = "fake-vision"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, image_path: Path, style: object) -> SceneUnderstanding:
        self.calls += 1
        return SceneUnderstanding(
            caption=f"Travel scene {self.calls}",
            detailed_description=f"Detailed travel scene {self.calls}.",
            location_type="city" if self.calls == 1 else "beach",
            activity="walking",
            emotion="joyful",
            people_count=2,
            people_groups=["group"],
            landmarks=[],
            vision_score=90 if self.calls == 1 else 70,
            score_factors={
                "uniqueness": 80,
                "people": 70,
                "emotion": 80,
                "visual_quality": 50,
                "landmark": 20,
                "unusual_event": 30,
            },
            story_relevance="Useful travel moment.",
            tags=["travel", f"scene-{self.calls}"],
        )


def test_quick_montage_plan_orders_assets_and_respects_duration(tmp_path: Path) -> None:
    early = _asset(
        tmp_path / "early.mp4",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        duration=20,
    )
    photo = _asset(
        tmp_path / "photo.jpg",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        media_type=MediaType.PHOTO,
    )
    late = _asset(
        tmp_path / "late.mp4",
        created_at=datetime(2026, 1, 3, tzinfo=UTC),
        duration=20,
    )

    plan = build_quick_montage_plan(
        [late, photo, early],
        QuickMontageSettings(
            target_duration_seconds=10,
            max_video_clip_seconds=5,
            photo_duration_seconds=3,
        ),
    )

    assert [clip.relative_path.name for clip in plan.clips] == [
        "early.mp4",
        "photo.jpg",
        "late.mp4",
    ]
    assert plan.total_duration_seconds == 10
    assert plan.clips[0].source_start_seconds == 7.5
    assert plan.clips[-1].duration_seconds == 2


def test_quick_montage_caps_target_with_long_transition_and_short_clips(
    tmp_path: Path,
) -> None:
    assets = [
        _asset(
            tmp_path / f"photo-{index}.jpg",
            created_at=datetime(2026, 1, index + 1, tzinfo=UTC),
            media_type=MediaType.PHOTO,
        )
        for index in range(10)
    ]
    settings = QuickMontageSettings(
        target_duration_seconds=5,
        photo_duration_seconds=3,
        transition="fade",
        transition_duration_seconds=3,
    )

    plan = build_quick_montage_plan(assets, settings)

    assert len(plan.clips) == 2
    assert plan.total_duration_seconds == pytest.approx(4.65)
    assert plan.total_duration_seconds <= settings.target_duration_seconds + 0.05


def test_renderer_builds_explicitly_requested_transition_graph(tmp_path: Path) -> None:
    settings = QuickMontageSettings(
        transition="slideright",
        transition_duration_seconds=0.4,
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "first.mp4",
                relative_path=Path("first.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
            ),
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "second.mp4",
                relative_path=Path("second.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
                transition="fade",
            ),
        ],
        total_duration_seconds=5.6,
    )

    transition_duration = _transition_duration(plan)
    graph = _build_filter_graph(plan, transition_duration=transition_duration)

    assert transition_duration == 0.4
    assert "xfade=transition=slideright:duration=0.400:offset=2.600" in graph
    assert "acrossfade=d=0.400" in graph
    assert "concat=n=2:v=1:a=0" not in graph


def test_renderer_builds_cut_and_fade_graph_for_cinematic_policy(tmp_path: Path) -> None:
    settings = QuickMontageSettings(
        transition="cinematic",
        transition_duration_seconds=0.4,
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "first.mp4",
                relative_path=Path("first.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
            ),
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "second.mp4",
                relative_path=Path("second.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
                transition="cut",
            ),
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "third.mp4",
                relative_path=Path("third.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
                transition="fade",
            ),
        ],
        total_duration_seconds=8.6,
    )

    graph = _build_filter_graph(plan, transition_duration=_transition_duration(plan))

    assert "concat=n=2:v=1:a=0" in graph
    assert "concat=n=2:v=0:a=1" in graph
    assert "xfade=transition=fadeblack:duration=0.400:offset=5.600" in graph
    assert "acrossfade=d=0.400" in graph
    assert "transition=dissolve" not in graph


def test_renderer_requires_explicit_opt_in_for_stylized_transition(tmp_path: Path) -> None:
    settings = QuickMontageSettings(transition="cinematic")
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "first.mp4",
                relative_path=Path("first.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
            ),
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "second.mp4",
                relative_path=Path("second.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
                transition="wipeleft",
            ),
        ],
        total_duration_seconds=6,
    )

    transition_duration = _transition_duration(plan)
    graph = _build_filter_graph(plan, transition_duration=transition_duration)

    assert transition_duration == 0
    assert "xfade=" not in graph
    assert "concat=n=2:v=1:a=0" in graph


def test_renderer_never_emits_unvalidated_pixel_dissolve(tmp_path: Path) -> None:
    settings = QuickMontageSettings.model_construct(transition="dissolve")
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "first.mp4",
                relative_path=Path("first.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
            ),
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "second.mp4",
                relative_path=Path("second.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
            ),
        ],
        total_duration_seconds=6,
    )

    transition_duration = _transition_duration(plan)
    graph = _build_filter_graph(plan, transition_duration=transition_duration)

    assert transition_duration == 0
    assert "transition=dissolve" not in graph
    assert "concat=n=2:v=1:a=0" in graph


def test_renderer_uses_preroll_and_trim_for_video_segments(tmp_path: Path) -> None:
    captured: list[list[str]] = []

    class CapturingRenderer(QuickMontageRenderer):
        def _run(self, command: list[str], message: str) -> None:
            captured.append(command)

    settings = QuickMontageSettings(transition="none", music_enabled=False)
    clip = MontageClip(
        asset_id=uuid4(),
        source_path=tmp_path / "source.mp4",
        relative_path=Path("source.mp4"),
        media_type=MediaType.VIDEO,
        source_start_seconds=5.2,
        duration_seconds=2,
        has_audio=True,
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[clip],
        total_duration_seconds=2,
    )

    CapturingRenderer()._render_segment(clip, plan, tmp_path / "segment.mp4")

    command = captured[0]
    filter_graph = command[command.index("-filter_complex") + 1]
    assert command[command.index("-ss") + 1] == "4.200"
    assert command[command.index("-t") + 1] == "3.250"
    assert command[command.index("-c:a") + 1] == "alac"
    assert "trim=start=1.000:duration=2.000" in filter_graph
    assert "atrim=start=1.000:duration=2.000" in filter_graph


def test_transition_segments_use_lossless_cpu_mezzanine_before_final_encode() -> None:
    renderer = QuickMontageRenderer(ffmpeg_threads=3)
    renderer._encoder = "h264_nvenc"
    renderer._mezzanine_segments = True

    arguments = renderer._segment_video_encoder_args()

    assert arguments[arguments.index("-c:v") + 1] == "libx264"
    assert arguments[arguments.index("-qp") + 1] == "0"
    assert arguments[arguments.index("-threads") + 1] == "3"
    assert "-cq" not in arguments


def test_renderer_resumes_from_atomic_segment_checkpoints(tmp_path: Path) -> None:
    calls: list[str] = []
    fail_second_once = True

    class CheckpointRenderer(QuickMontageRenderer):
        def _valid_cached_segment(
            self,
            path: Path,
            clip: MontageClip | None = None,
            plan: QuickMontagePlan | None = None,
        ) -> bool:
            del clip, plan
            return path.is_file() and path.stat().st_size > 0

        def _render_segment(
            self,
            clip: MontageClip,
            plan: QuickMontagePlan,
            output_path: Path,
            *,
            show_event_title: bool = False,
            show_credits: bool = False,
        ) -> None:
            nonlocal fail_second_once
            del plan, show_event_title, show_credits
            calls.append(clip.relative_path.as_posix())
            output_path.write_bytes(b"partial-or-complete")
            if clip.relative_path.name == "second.mp4" and fail_second_once:
                fail_second_once = False
                raise MontageError("simulated interruption")

    clips = []
    for name in ("first.mp4", "second.mp4", "third.mp4"):
        source = tmp_path / name
        source.write_bytes(name.encode())
        clips.append(
            MontageClip(
                asset_id=uuid4(),
                source_path=source,
                relative_path=Path(name),
                media_type=MediaType.VIDEO,
                duration_seconds=1,
            )
        )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(music_enabled=False, render_device="cpu"),
        clips=clips,
        total_duration_seconds=3,
    )
    renderer = CheckpointRenderer(workers=1)
    segments_dir = tmp_path / "cache" / "quick_montage_segments"
    segments_dir.mkdir(parents=True)

    with pytest.raises(MontageError, match="simulated interruption"):
        renderer._render_segments(plan, segments_dir, None, 4)
    resumed = renderer._render_segments(plan, segments_dir, None, 4)

    assert calls == ["first.mp4", "second.mp4", "second.mp4", "third.mp4"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in resumed)
    assert not list(segments_dir.glob(".*.tmp.mp4"))


def test_parallel_renderer_does_not_start_queued_clips_after_cancel(tmp_path: Path) -> None:
    barrier = Barrier(2)
    calls: list[Path] = []

    class BlockingRenderer(QuickMontageRenderer):
        def _valid_cached_segment(
            self,
            path: Path,
            clip: MontageClip | None = None,
            plan: QuickMontagePlan | None = None,
        ) -> bool:
            del clip, plan
            return path.is_file() and path.stat().st_size > 0

        def _render_segment(
            self,
            clip: MontageClip,
            plan: QuickMontagePlan,
            output_path: Path,
            *,
            show_event_title: bool = False,
            show_credits: bool = False,
        ) -> None:
            del plan, show_event_title, show_credits
            calls.append(clip.relative_path)
            barrier.wait(timeout=5)
            output_path.write_bytes(b"complete")

    clips = []
    for index in range(5):
        source = tmp_path / f"clip-{index}.mp4"
        source.write_bytes(b"source")
        clips.append(
            MontageClip(
                asset_id=uuid4(),
                source_path=source,
                relative_path=Path(source.name),
                media_type=MediaType.VIDEO,
                duration_seconds=1,
            )
        )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(music_enabled=False, render_device="cpu"),
        clips=clips,
        total_duration_seconds=5,
    )

    def cancel_after_first(current: int, total: int, message: str) -> None:
        del total
        if current >= 1 and "Clips complete" in message:
            raise MontageError("cancel render queue")

    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()
    with pytest.raises(MontageError, match="cancel render queue"):
        BlockingRenderer(workers=2)._render_segments(
            plan,
            segments_dir,
            cancel_after_first,
            6,
        )

    assert len(calls) == 2


def test_cached_segment_requires_valid_probed_video_and_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = tmp_path / "segment.mp4"
    segment.write_bytes(b"nonempty but corrupt")

    class FailingProbe:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def probe(self, path: Path) -> object:
            del path
            raise MediaProbeError("corrupt segment")

    monkeypatch.setattr("travelmovieai.editing.renderer.FFprobeClient", FailingProbe)
    renderer = QuickMontageRenderer()
    assert renderer._valid_cached_segment(segment) is False

    class ValidProbe:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def probe(self, path: Path) -> object:
            del path
            return type(
                "Probe",
                (),
                {
                    "duration_seconds": 1.0,
                    "metadata": {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]},
                },
            )()

    monkeypatch.setattr("travelmovieai.editing.renderer.FFprobeClient", ValidProbe)
    assert renderer._valid_cached_segment(segment) is True


def test_cached_segment_must_match_planned_delivery_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = tmp_path / "segment.mp4"
    segment.write_bytes(b"prepared")
    video_duration_seconds = 2.0

    class DetailedProbe:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def probe(self, path: Path) -> object:
            del path
            return type(
                "Probe",
                (),
                {
                    "duration_seconds": 2.0,
                    "video_duration_seconds": video_duration_seconds,
                    "width": 320,
                    "height": 240,
                    "fps": 24.0,
                    "metadata": {
                        "streams": [
                            {
                                "codec_type": "video",
                                "codec_name": "h264",
                                "profile": "High",
                                "pix_fmt": "yuv420p",
                            },
                            {"codec_type": "audio", "codec_name": "alac"},
                        ]
                    },
                },
            )()

    monkeypatch.setattr("travelmovieai.editing.renderer.FFprobeClient", DetailedProbe)
    clip = MontageClip(
        asset_id=uuid4(),
        source_path=tmp_path / "source.mp4",
        relative_path=Path("source.mp4"),
        media_type=MediaType.VIDEO,
        duration_seconds=2,
    )
    valid_plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(width=320, height=240, fps=24),
        clips=[clip],
        total_duration_seconds=2,
    )

    renderer = QuickMontageRenderer()
    assert renderer._valid_cached_segment(segment, clip, valid_plan) is True
    video_duration_seconds = 1.7
    assert renderer._valid_cached_segment(segment, clip, valid_plan) is False
    video_duration_seconds = 2.0
    wrong_size = valid_plan.model_copy(
        update={"settings": valid_plan.settings.model_copy(update={"width": 640})}
    )
    assert renderer._valid_cached_segment(segment, clip, wrong_size) is False


def test_transition_cache_rejects_lossy_h264_and_accepts_qp0_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = tmp_path / "transition-segment.mp4"
    segment.write_bytes(b"prepared")
    profile = "High"
    audio_codec = "aac"

    class DetailedProbe:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def probe(self, path: Path) -> object:
            del path
            return type(
                "Probe",
                (),
                {
                    "duration_seconds": 2.0,
                    "width": 320,
                    "height": 240,
                    "fps": 24.0,
                    "metadata": {
                        "streams": [
                            {
                                "codec_type": "video",
                                "codec_name": "h264",
                                "profile": profile,
                                "pix_fmt": "yuv420p",
                            },
                            {"codec_type": "audio", "codec_name": audio_codec},
                        ]
                    },
                },
            )()

    first = MontageClip(
        asset_id=uuid4(),
        source_path=tmp_path / "first.mp4",
        relative_path=Path("first.mp4"),
        media_type=MediaType.VIDEO,
        duration_seconds=2,
    )
    second = first.model_copy(
        update={
            "asset_id": uuid4(),
            "source_path": tmp_path / "second.mp4",
            "relative_path": Path("second.mp4"),
            "transition": "fade",
        }
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(
            width=320,
            height=240,
            fps=24,
            transition="fade",
            transition_duration_seconds=0.25,
        ),
        clips=[first, second],
        total_duration_seconds=3.75,
    )
    monkeypatch.setattr("travelmovieai.editing.renderer.FFprobeClient", DetailedProbe)
    renderer = QuickMontageRenderer()

    assert renderer._valid_cached_segment(segment, first, plan) is False
    profile = "High 4:4:4 Predictive"
    assert renderer._valid_cached_segment(segment, first, plan) is False
    audio_codec = "alac"
    assert renderer._valid_cached_segment(segment, first, plan) is True


def test_full_decode_validation_uses_both_required_streams(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    class CapturingRenderer(QuickMontageRenderer):
        def _run(self, command: list[str], message: str) -> None:
            del message
            commands.append(command)

    movie = tmp_path / "movie.mp4"
    CapturingRenderer()._validate_full_decode(movie)

    assert commands[0][commands[0].index("-map") + 1] == "0:v:0"
    second_map = commands[0].index("-map", commands[0].index("-map") + 1)
    assert commands[0][second_map + 1] == "0:a:0"


def test_renderer_reports_ffmpeg_timeout_and_stops_process_tree() -> None:
    with pytest.raises(MontageError, match="timed out after 0.25s"):
        QuickMontageRenderer(timeout_seconds=0.25)._run(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            "Could not render",
        )


def test_renderer_honors_process_cancellation() -> None:
    with pytest.raises(MontageError, match="rendering was cancelled"):
        QuickMontageRenderer(cancel_requested=lambda: True)._run(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            "Could not render",
        )


def test_renderer_progress_heartbeat_can_cancel_active_ffmpeg() -> None:
    renderer = QuickMontageRenderer(timeout_seconds=10)

    def cancel_from_progress() -> None:
        raise MontageError("cancelled by progress callback")

    renderer._heartbeat_callback = cancel_from_progress

    with pytest.raises(MontageError, match="cancelled by progress callback"):
        renderer._run(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            "Could not render",
        )


def test_renderer_classifies_only_confirmed_nvenc_runtime_failures() -> None:
    command = ["ffmpeg", "-c:v", "h264_nvenc", "out.mp4"]

    assert _is_nvenc_unavailable_error(command, "Cannot load nvcuda.dll") is True
    assert (
        _is_nvenc_unavailable_error(
            command,
            "Error initializing complex filters: No such file or directory",
        )
        is False
    )


def test_renderer_auto_falls_back_only_for_typed_nvenc_failure(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    clip = MontageClip(
        asset_id=uuid4(),
        source_path=source,
        relative_path=Path("clip.mp4"),
        media_type=MediaType.VIDEO,
        duration_seconds=1,
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(render_device="auto", music_enabled=False),
        clips=[clip],
        total_duration_seconds=1,
    )

    class StubRenderer(QuickMontageRenderer):
        def __init__(self, failure: MontageError) -> None:
            super().__init__()
            self.failure = failure
            self.calls = 0

        def _select_encoder(self, render_device: str) -> str:
            del render_device
            return "h264_nvenc"

        def _render_segments(
            self,
            montage_plan: QuickMontagePlan,
            segments_dir: Path,
            progress: object,
            total_steps: int,
        ) -> list[Path]:
            del montage_plan, segments_dir, progress, total_steps
            self.calls += 1
            if self.calls == 1:
                raise self.failure
            return []

        def _compose_segments(self, *args: object, **kwargs: object) -> None:
            pass

        def _validate_output(self, *args: object, **kwargs: object) -> None:
            pass

    unavailable = StubRenderer(_NvencUnavailableError("NVENC unavailable"))
    assert unavailable.render(plan, tmp_path / "fallback.mp4", tmp_path / "work") == "libx264"
    assert unavailable.calls == 2

    source_error = StubRenderer(MontageError("invalid source filter"))
    with pytest.raises(MontageError, match="invalid source filter"):
        source_error.render(plan, tmp_path / "failed.mp4", tmp_path / "other-work")
    assert source_error.calls == 1


def test_hard_cut_audio_mix_stream_copies_prepared_video(tmp_path: Path) -> None:
    segments = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    for segment in segments:
        segment.write_bytes(b"prepared")
    music = tmp_path / "music.wav"
    music.write_bytes(b"music")
    clips = [
        MontageClip(
            asset_id=uuid4(),
            source_path=tmp_path / f"source-{index}.mp4",
            relative_path=Path(f"source-{index}.mp4"),
            media_type=MediaType.VIDEO,
            duration_seconds=1,
        )
        for index in range(2)
    ]
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(transition="none"),
        clips=clips,
        total_duration_seconds=2,
        music_path=music,
    )
    commands: list[list[str]] = []
    graphs: list[str] = []

    class CapturingRenderer(QuickMontageRenderer):
        def _run(self, command: list[str], message: str) -> None:
            del message
            commands.append(command)
            graph_path = Path(command[command.index("-filter_complex_script") + 1])
            graphs.append(graph_path.read_text(encoding="utf-8"))
            Path(command[-1]).write_bytes(b"movie")

    output = tmp_path / "movie.mp4"
    CapturingRenderer()._compose_segments(segments, plan, output, tmp_path)

    assert output.read_bytes() == b"movie"
    assert commands[0][commands[0].index("-c:v") + 1] == "copy"
    assert "h264_nvenc" not in commands[0]
    assert "libx264" not in commands[0]
    assert "loudnorm=I=-16.0:TP=-1.5" in graphs[0]
    assert _build_hard_cut_audio_graph(plan) == graphs[0]

    commands.clear()
    graphs.clear()
    plan_without_music = plan.model_copy(update={"music_path": None})
    dry_output = tmp_path / "movie-with-source-audio-only.mp4"
    CapturingRenderer()._compose_segments(
        segments,
        plan_without_music,
        dry_output,
        tmp_path,
    )

    assert dry_output.read_bytes() == b"movie"
    assert commands[0][commands[0].index("-c:v") + 1] == "copy"
    assert "volume=0.550" in graphs[0]
    assert "loudnorm=I=-16.0:TP=-1.5" in graphs[0]


def test_renderer_auto_uses_nvenc_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "travelmovieai.editing.renderer.check_cuda",
        lambda ffmpeg_binary: CudaStatus(available=True, ffmpeg_nvenc=True),
    )

    assert QuickMontageRenderer()._select_encoder("auto") == "h264_nvenc"
    assert QuickMontageRenderer()._select_encoder("cpu") == "libx264"


def test_renderer_cuda_requires_explicit_available_nvenc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "travelmovieai.editing.renderer.check_cuda",
        lambda ffmpeg_binary: CudaStatus(available=True, ffmpeg_nvenc=True),
    )

    assert QuickMontageRenderer()._select_encoder("cuda") == "h264_nvenc"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="FFprobe is not installed")
def test_renderer_strips_source_metadata_from_outputs(tmp_path: Path) -> None:
    source = tmp_path / "tagged-source.mp4"
    output = tmp_path / "clean-output.mp4"
    _generate_tagged_video(source)
    clip = MontageClip(
        asset_id=uuid4(),
        source_path=source,
        relative_path=Path("tagged-source.mp4"),
        media_type=MediaType.VIDEO,
        duration_seconds=1,
        has_audio=True,
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(
            width=320,
            height=240,
            fps=24,
            music_enabled=False,
            render_device="cpu",
        ),
        clips=[clip],
        total_duration_seconds=1,
    )

    QuickMontageRenderer().render(plan, output, tmp_path / "render")

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(output),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    format_tags = payload.get("format", {}).get("tags", {})
    stream_tags = [
        stream.get("tags", {}) for stream in payload.get("streams", []) if isinstance(stream, dict)
    ]

    assert "location" not in format_tags
    assert "comment" not in format_tags
    assert all("location" not in tags and "comment" not in tags for tags in stream_tags)


def test_renderer_rejects_missing_soundtrack_before_ffmpeg(tmp_path: Path) -> None:
    missing_music = tmp_path / "missing.wav"
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(),
        clips=[
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "clip.mp4",
                relative_path=Path("clip.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=2,
            )
        ],
        total_duration_seconds=2,
        music_path=missing_music,
    )

    with pytest.raises(MontageError, match="Soundtrack file does not exist"):
        QuickMontageRenderer().render(plan, tmp_path / "out.mp4", tmp_path)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (QuickMontageSettings(speech_analysis=True), "Speech analysis requires semantic"),
        (QuickMontageSettings(narration_enabled=True), "Narration requires semantic"),
        (QuickMontageSettings(framing_mode="smart"), "smart crop.*require semantic"),
        (QuickMontageSettings(color_normalization=True), "color normalization.*require semantic"),
        (
            QuickMontageSettings(text_overlays_enabled=True, event_titles_enabled=True),
            "event titles.*require semantic",
        ),
        (
            QuickMontageSettings(text_overlays_enabled=True, scene_subtitles_enabled=True),
            "scene subtitles.*require semantic",
        ),
    ],
)
def test_service_rejects_ai_audio_features_without_semantic_pipeline(
    tmp_path: Path,
    settings: QuickMontageSettings,
    message: str,
) -> None:
    with pytest.raises(MontageError, match=message):
        TravelMovieService(Settings()).create_quick_montage(
            input_path=tmp_path / "media",
            workspace=tmp_path / "workspace",
            settings=settings,
        )


def test_service_rejects_requested_narration_when_piper_is_disabled(tmp_path: Path) -> None:
    with pytest.raises(MontageError, match="voice_provider is disabled"):
        TravelMovieService(Settings(voice_provider="disabled")).create_quick_montage(
            input_path=tmp_path / "media",
            workspace=tmp_path / "workspace",
            settings=QuickMontageSettings(
                semantic_analysis=True,
                narration_enabled=True,
            ),
        )


@pytest.mark.parametrize("reject_quality", [False, True], ids=["publish", "reject"])
def test_nonsemantic_service_publishes_only_after_quality_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reject_quality: bool,
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    source = media / "clip.mp4"
    source.write_bytes(b"source")
    workspace = tmp_path / "workspace"
    delivery = tmp_path / "delivery.mp4"
    delivery.write_bytes(b"previous movie")
    service = TravelMovieService(Settings())
    context = service._context(input_path=media, workspace=workspace)
    context.prepare()
    asset = _asset(
        source,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        duration=6,
    )
    analysis = MediaScanReport(
        input_path=media.resolve(),
        scanned_at=datetime.now(UTC),
        assets=[asset],
        discovered_count=1,
        probed_count=1,
    )
    (context.artifacts_dir / "analysis.json").write_text(
        analysis.model_dump_json(),
        encoding="utf-8",
    )
    timeline_path = context.artifacts_dir / "quick_timeline.json"
    quality_path = context.artifacts_dir / "montage_quality_report.json"
    timeline_path.write_bytes(b"previous timeline")
    quality_path.write_bytes(b"previous quality report")
    render_targets: list[Path] = []

    class FakeRenderer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def render(
            self,
            plan: QuickMontagePlan,
            candidate_path: Path,
            work_dir: Path,
            progress: object,
        ) -> str:
            del plan, work_dir, progress
            assert candidate_path.parent == delivery.parent
            assert candidate_path != delivery
            assert delivery.read_bytes() == b"previous movie"
            render_targets.append(candidate_path)
            candidate_path.write_bytes(b"validated movie")
            return "fake-encoder"

    def fake_enrich(
        report: MontageQualityReport,
        candidate_path: Path,
        **kwargs: object,
    ) -> MontageQualityReport:
        return report.model_copy(
            update={
                "rendered_path": candidate_path,
                "rendered_duration_seconds": report.planned_duration_seconds,
                "rendered_has_video": True,
                "rendered_has_audio": True,
            }
        )

    def quality_gate(report: MontageQualityReport) -> None:
        assert report.rendered_path == render_targets[0]
        assert delivery.read_bytes() == b"previous movie"
        assert timeline_path.read_bytes() == b"previous timeline"
        assert quality_path.read_bytes() == b"previous quality report"
        if reject_quality:
            raise MontageError("simulated quality gate failure")

    profile = type(
        "Profile",
        (),
        {
            "summary": "test resources",
            "render_workers": 1,
            "ffmpeg_threads": 1,
        },
    )()
    monkeypatch.setattr(service, "get_resource_profile", lambda *, refresh=False: profile)
    monkeypatch.setattr(service, "analyze", lambda **kwargs: None)
    monkeypatch.setattr(
        "travelmovieai.application.service.ensure_render_disk_space",
        lambda **kwargs: None,
    )
    monkeypatch.setattr("travelmovieai.application.service.QuickMontageRenderer", FakeRenderer)
    monkeypatch.setattr(
        "travelmovieai.application.service.enrich_montage_quality_report_with_render",
        fake_enrich,
    )
    monkeypatch.setattr(
        "travelmovieai.application.service.enforce_montage_quality",
        quality_gate,
    )
    monkeypatch.setattr(
        "travelmovieai.application.service._record_timeline_version",
        lambda *args, **kwargs: None,
    )
    settings = QuickMontageSettings(
        target_duration_seconds=5,
        transition="none",
        music_enabled=False,
        music_mode="none",
    )

    if reject_quality:
        with pytest.raises(MontageError, match="quality gate failure"):
            service.create_quick_montage(
                input_path=media,
                workspace=workspace,
                settings=settings,
                output_path=delivery,
            )
        assert delivery.read_bytes() == b"previous movie"
        assert timeline_path.read_bytes() == b"previous timeline"
        assert quality_path.read_bytes() == b"previous quality report"
    else:
        result = service.create_quick_montage(
            input_path=media,
            workspace=workspace,
            settings=settings,
            output_path=delivery,
        )
        quality = MontageQualityReport.model_validate_json(quality_path.read_text(encoding="utf-8"))
        assert result.output_path == delivery
        assert delivery.read_bytes() == b"validated movie"
        assert QuickMontagePlan.model_validate_json(timeline_path.read_text(encoding="utf-8")).clips
        assert quality.rendered_path == delivery
        assert quality.render_encoder == "fake-encoder"

    assert len(render_targets) == 1
    assert not render_targets[0].exists()


def test_renderer_rejects_output_that_overwrites_source_media(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(),
        clips=[
            MontageClip(
                asset_id=uuid4(),
                source_path=source,
                relative_path=Path("clip.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=2,
            )
        ],
        total_duration_seconds=2,
    )

    with pytest.raises(MontageError, match="must not overwrite source media"):
        QuickMontageRenderer().render(plan, source, tmp_path / "render")

    soundtrack = tmp_path / "soundtrack.wav"
    plan_with_music = plan.model_copy(update={"music_path": soundtrack})
    with pytest.raises(MontageError, match="or the soundtrack"):
        QuickMontageRenderer().render(plan_with_music, soundtrack, tmp_path / "render")

    work_dir = tmp_path / "render"
    work_output = work_dir / "quick_montage_segments" / "final.mp4"
    with pytest.raises(MontageError, match="renderer working files"):
        QuickMontageRenderer().render(plan, work_output, work_dir)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_service_creates_playable_quick_montage(tmp_path: Path) -> None:
    media = tmp_path / "Моя поездка"
    media.mkdir()
    video = media / "clip with audio.mp4"
    photo = media / "photo.jpg"
    _generate_video(video)
    _generate_photo(photo)
    workspace = tmp_path / "workspace"

    result = TravelMovieService(Settings()).create_quick_montage(
        input_path=media,
        workspace=workspace,
        settings=QuickMontageSettings(
            target_duration_seconds=5,
            max_video_clip_seconds=1,
            photo_duration_seconds=1,
            width=320,
            height=240,
            fps=24,
            music_engine="procedural",
        ),
        variant_name="Quick history",
    )

    assert result.output_path.is_file()
    assert result.output_path.stat().st_size > 0
    assert result.clip_count == 2
    assert result.timeline_path.is_file()
    versions = MediaAssetRepository(workspace / "project.db").list_timeline_versions()
    assert [version.phase for version in versions[:2]] == ["rendered", "built"]
    assert all(version.variant_name == "Quick history" for version in versions[:2])
    quality_report = json.loads(
        (workspace / "artifacts" / "montage_quality_report.json").read_text(encoding="utf-8")
    )
    assert quality_report["clip_count"] == result.clip_count
    assert quality_report["planned_duration_seconds"] == pytest.approx(result.duration_seconds)
    assert quality_report["rendered_path"] == str(result.output_path)
    assert quality_report["rendered_has_video"] is True
    assert quality_report["rendered_has_audio"] is True
    assert quality_report["rendered_duration_seconds"] == pytest.approx(
        result.duration_seconds,
        abs=0.2,
    )
    assert set(quality_report["rendered_audio_rms"]) == {"start", "middle", "end"}
    assert quality_report["rendered_audio_rms"]["middle"] > 10
    assert set(quality_report["rendered_video_luma"]) == {"start", "middle", "end"}
    timeline = json.loads(result.timeline_path.read_text(encoding="utf-8"))
    music_plan = timeline["music_plan"]
    assert music_plan["generated"] is True
    assert music_plan["duration_seconds"] == pytest.approx(result.duration_seconds)
    assert music_plan["arrangement_version"] == "story-music-v9-tail-audit"
    assert music_plan["accents"][0]["kind"] == "intro"
    assert music_plan["accents"][-1]["kind"] == "finale"

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(result.output_path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    stream_types = {stream["codec_type"] for stream in payload["streams"]}
    assert stream_types == {"video", "audio"}
    assert float(payload["format"]["duration"]) == pytest.approx(
        result.duration_seconds,
        abs=0.15,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_service_creates_cached_semantic_montage_with_music(tmp_path: Path) -> None:
    media = tmp_path / "AI поездка"
    media.mkdir()
    _generate_video(media / "city.mp4")
    _generate_photo(media / "beach.jpg")
    _generate_music(media / "cinematic theme.wav")
    workspace = tmp_path / "workspace"
    provider = FakeVisionProvider()
    service = TravelMovieService(
        Settings(),
        vision_provider_factory=lambda _: provider,
    )
    settings = QuickMontageSettings(
        target_duration_seconds=5,
        max_video_clip_seconds=1,
        photo_duration_seconds=1,
        width=320,
        height=240,
        fps=24,
        semantic_analysis=True,
        music_mode="library",
        transition="cinematic",
        transition_duration_seconds=0.25,
    )

    first = service.create_quick_montage(
        input_path=media,
        workspace=workspace,
        settings=settings,
    )
    calls_after_first_run = provider.calls
    second = service.create_quick_montage(
        input_path=media,
        workspace=workspace,
        settings=settings,
    )

    assert first.selection_mode == "semantic"
    assert first.output_path.is_file()
    assert calls_after_first_run >= 2
    assert provider.calls == calls_after_first_run
    assert second.output_path.is_file()
    vision = json.loads(
        (workspace / "artifacts" / "vision_analysis.json").read_text(encoding="utf-8")
    )
    timeline = json.loads(first.timeline_path.read_text(encoding="utf-8"))
    assert len(vision["scenes"]) >= 2
    assert (workspace / "artifacts" / "events.json").is_file()
    assert (workspace / "artifacts" / "embeddings.json").is_file()
    assert (workspace / "artifacts" / "scene_descriptions.json").is_file()
    assert (workspace / "artifacts" / "storyboard.json").is_file()
    quality_report = json.loads(
        (workspace / "artifacts" / "montage_quality_report.json").read_text(encoding="utf-8")
    )
    assert quality_report["selected_scene_count"] == first.clip_count
    assert quality_report["music_mode"] == "library"
    assert quality_report["rendered_has_video"] is True
    assert quality_report["rendered_has_audio"] is True
    assert quality_report["render_encoder"] == first.render_encoder
    assert timeline["selection_mode"] == "semantic"
    assert timeline["music_path"].endswith("cinematic theme.wav")
    assert all(clip["semantic_score"] is not None for clip in timeline["clips"])
    assert {clip["transition"] for clip in timeline["clips"]} <= {None, "cut", "fade"}


def test_semantic_montage_rejects_empty_media_directory(tmp_path: Path) -> None:
    media = tmp_path / "empty trip"
    media.mkdir()

    with pytest.raises(MontageError, match="timeline|usable media"):
        TravelMovieService(Settings()).create_quick_montage(
            input_path=media,
            workspace=tmp_path / "workspace",
            settings=QuickMontageSettings(
                semantic_analysis=True,
                music_enabled=False,
            ),
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="FFprobe is not installed")
def test_renderer_keeps_short_music_audible_until_the_end(tmp_path: Path) -> None:
    photo = tmp_path / "quiet-photo.jpg"
    music = tmp_path / "short-theme.wav"
    output = tmp_path / "movie.mp4"
    _generate_photo(photo)
    _generate_music(music, duration=0.25)
    settings = QuickMontageSettings(
        width=320,
        height=240,
        fps=24,
        transition="none",
        music_volume=0.5,
    )
    clip = MontageClip(
        asset_id=uuid4(),
        source_path=photo,
        relative_path=Path("quiet-photo.jpg"),
        media_type=MediaType.PHOTO,
        duration_seconds=2,
        has_audio=False,
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[clip],
        total_duration_seconds=2,
        music_path=music,
    )

    QuickMontageRenderer().render(plan, output, tmp_path / "render")

    assert _audio_rms(output, start_seconds=1.45, duration_seconds=0.35) > 100


def _asset(
    path: Path,
    *,
    created_at: datetime,
    duration: float | None = None,
    media_type: MediaType = MediaType.VIDEO,
) -> MediaAsset:
    return MediaAsset(
        path=path,
        relative_path=Path(path.name),
        media_type=media_type,
        extension=path.suffix,
        size_bytes=1,
        modified_at=created_at,
        modified_ns=1,
        created_at=created_at,
        duration_seconds=duration,
    )


def _generate_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=24:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def _generate_tagged_video(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=24:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-metadata",
            "location=+43.6411+040.2682/",
            "-metadata",
            "comment=private drone metadata",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def _generate_photo(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=orange:s=320x240",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(path),
        ],
        check=True,
    )


def _generate_music(path: Path, duration: float = 4) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:sample_rate=48000:duration={duration}",
            str(path),
        ],
        check=True,
    )


def _audio_rms(path: Path, *, start_seconds: float, duration_seconds: float) -> float:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(path),
            "-vn",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "8000",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    sample_count = len(completed.stdout) // 2
    if sample_count == 0:
        return 0.0
    samples = struct.unpack(f"<{sample_count}h", completed.stdout[: sample_count * 2])
    return math.sqrt(sum(sample * sample for sample in samples) / sample_count)
