import hashlib
import os
from pathlib import Path
from uuid import uuid4

import pytest

from travelmovieai.analysis.vision import analyze_scenes, scene_vision_input_identity
from travelmovieai.core.exceptions import DependencyUnavailableError
from travelmovieai.domain.enums import StoryStyle
from travelmovieai.domain.models import Scene, SceneUnderstanding, VisionAnalysisReport
from travelmovieai.editing.timeline import _scene_pacing


class _VisionProvider:
    name = "fake"
    model = "fake-model"
    batch_size = 1

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def analyze(self, image_path: Path, style: StoryStyle) -> SceneUnderstanding:
        del image_path, style
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("interrupted")
        return SceneUnderstanding(
            caption=f"Scene {self.calls}",
            detailed_description=f"Detailed scene {self.calls}",
            shot_scale="wide",
            camera_motion="tracking",
            focus_x=0.35,
            focus_y=0.65,
            focus_source="subject",
            score_factors={
                "uniqueness": 70,
                "people": 50,
                "emotion": 60,
                "visual_quality": 50,
                "landmark": 40,
                "unusual_event": 30,
            },
        )


def test_vision_analysis_checkpoints_and_resumes_after_interruption(tmp_path: Path) -> None:
    scenes = []
    for index in range(2):
        keyframe = tmp_path / f"frame-{index}.jpg"
        keyframe.touch()
        scenes.append(
            Scene(
                asset_id=uuid4(),
                start_seconds=float(index),
                end_seconds=float(index + 1),
                keyframe_path=keyframe,
                quality_score=70,
                metadata={"cache_key": f"scene-{index}"},
            )
        )
    checkpoints: list[VisionAnalysisReport] = []

    with pytest.raises(RuntimeError, match="interrupted"):
        analyze_scenes(
            scenes,
            _VisionProvider(fail_on_call=2),
            StoryStyle.CINEMATIC,
            checkpoint=checkpoints.append,
        )

    assert len(checkpoints) == 1
    assert checkpoints[0].analyzed_count == 1
    resumed_provider = _VisionProvider()
    resumed = analyze_scenes(
        scenes,
        resumed_provider,
        StoryStyle.CINEMATIC,
        cached_report=checkpoints[0],
    )

    assert resumed_provider.calls == 1
    assert resumed.analyzed_count == 1
    assert resumed.cached_count == 1
    assert len(resumed.scenes) == 2
    assert resumed.scenes[0].metadata["focus_point"] == {"x": 0.35, "y": 0.65}
    assert resumed.scenes[0].metadata["focus_source"] == "subject"


def test_vision_cache_invalidates_when_measured_quality_changes(tmp_path: Path) -> None:
    keyframe = tmp_path / "frame.jpg"
    keyframe.touch()
    scene = Scene(
        asset_id=uuid4(),
        start_seconds=0,
        end_seconds=1,
        keyframe_path=keyframe,
        quality_score=70,
        metadata={"cache_key": "stable-scene"},
    )
    first = analyze_scenes([scene], _VisionProvider(), StoryStyle.CINEMATIC)
    changed = scene.model_copy(update={"quality_score": 25})
    provider = _VisionProvider()

    second = analyze_scenes(
        [changed],
        provider,
        StoryStyle.CINEMATIC,
        cached_report=first,
    )

    assert provider.calls == 1
    assert second.cached_count == 0
    assert second.scenes[0].metadata["vision_score_factors"]["visual_quality"] == 25


def test_vision_analysis_preserves_scene_without_keyframe() -> None:
    scene = Scene(
        asset_id=uuid4(),
        start_seconds=0,
        end_seconds=1,
        metadata={"cache_key": "missing-frame"},
    )
    provider = _VisionProvider()

    report = analyze_scenes([scene], provider, StoryStyle.CINEMATIC)

    assert report.scenes == [scene]
    assert report.analyzed_count == 0
    assert provider.calls == 0


def test_vision_shot_language_is_persisted_and_used_for_timeline_pacing(
    tmp_path: Path,
) -> None:
    keyframe = tmp_path / "frame.jpg"
    keyframe.touch()
    scene = Scene(
        asset_id=uuid4(),
        start_seconds=0,
        end_seconds=5,
        keyframe_path=keyframe,
        metadata={"cache_key": "shot-language"},
    )

    report = analyze_scenes([scene], _VisionProvider(), StoryStyle.CINEMATIC)
    analyzed = report.scenes[0]
    pacing_factor, pacing_reason = _scene_pacing(analyzed)

    assert analyzed.metadata["shot_scale"] == "wide"
    assert analyzed.metadata["camera_motion"] == "tracking"
    assert pacing_factor < 1
    assert pacing_reason == "pacing: high energy"


def test_vision_cache_invalidates_when_contact_sheet_mode_changes(tmp_path: Path) -> None:
    deep_sheet = tmp_path / "scene-contact-v4-9.png"
    fast_sheet = tmp_path / "scene-contact-v4-3.png"
    deep_sheet.write_bytes(b"nine chronological frames")
    fast_sheet.write_bytes(b"three chronological frames")
    scene = Scene(
        asset_id=uuid4(),
        start_seconds=0,
        end_seconds=5,
        keyframe_path=deep_sheet,
        quality_score=70,
        metadata={
            "cache_key": "stable-scene",
            "contact_sheet": {
                "schema_version": "contact-sheet-v1-temporal",
                "sample_count": 9,
                "sample_positions": [0.06, 0.17, 0.29, 0.4, 0.5, 0.6, 0.71, 0.83, 0.94],
            },
        },
    )
    first = analyze_scenes([scene], _VisionProvider(), StoryStyle.CINEMATIC)
    changed = scene.model_copy(
        update={
            "keyframe_path": fast_sheet,
            "metadata": {
                **scene.metadata,
                "contact_sheet": {
                    "schema_version": "contact-sheet-v1-temporal",
                    "sample_count": 3,
                    "sample_positions": [0.12, 0.5, 0.88],
                },
            },
        }
    )
    provider = _VisionProvider()

    second = analyze_scenes(
        [changed],
        provider,
        StoryStyle.CINEMATIC,
        cached_report=first,
    )

    assert provider.calls == 1
    assert second.cached_count == 0


def test_vision_cache_uses_contact_sheet_content_not_only_stat_metadata(
    tmp_path: Path,
) -> None:
    keyframe = tmp_path / "scene-contact-v4-3.png"
    keyframe.write_bytes(b"AAAA")
    original_stat = keyframe.stat()
    scene = Scene(
        asset_id=uuid4(),
        start_seconds=0,
        end_seconds=3,
        keyframe_path=keyframe,
        metadata={"cache_key": "content-sensitive"},
    )
    first = analyze_scenes([scene], _VisionProvider(), StoryStyle.CINEMATIC)
    keyframe.write_bytes(b"BBBB")
    os.utime(
        keyframe,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    provider = _VisionProvider()

    second = analyze_scenes(
        [scene],
        provider,
        StoryStyle.CINEMATIC,
        cached_report=first,
    )

    assert provider.calls == 1
    assert second.cached_count == 0


def test_vision_input_identity_ignores_mtime_when_content_is_unchanged(
    tmp_path: Path,
) -> None:
    keyframe = tmp_path / "scene-contact-v4-3.png"
    keyframe.write_bytes(b"stable contact sheet")
    scene = Scene(
        asset_id=uuid4(),
        start_seconds=0,
        end_seconds=3,
        keyframe_path=keyframe,
    )
    before = scene_vision_input_identity(scene)
    original = keyframe.stat()
    os.utime(
        keyframe,
        ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000),
    )

    assert scene_vision_input_identity(scene) == before


def test_vision_cache_reuses_identical_content_after_legacy_identity_change(
    tmp_path: Path,
) -> None:
    keyframe = tmp_path / "scene-contact-v4-3.png"
    keyframe.write_bytes(b"stable contact sheet")
    scene = Scene(
        asset_id=uuid4(),
        start_seconds=0,
        end_seconds=3,
        keyframe_path=keyframe,
        quality_score=72,
        metadata={
            "cache_key": "stable-scene",
            "quality_metrics": {"quality_score": 72},
            "contact_sheet": {
                "schema_version": "contact-sheet-v1-temporal",
                "sample_count": 3,
                "sample_positions": [0.12, 0.5, 0.88],
                "content_sha256": hashlib.sha256(b"stable contact sheet").hexdigest(),
            },
        },
    )
    first = analyze_scenes([scene], _VisionProvider(), StoryStyle.CINEMATIC)
    legacy_scene = first.scenes[0].model_copy(
        update={
            "metadata": {
                **first.scenes[0].metadata,
                "vision_cache_key": "legacy-stat-based-key",
                "vision_style": None,
            }
        }
    )
    cached_report = first.model_copy(update={"scenes": [legacy_scene]})
    blocked_provider = _VisionProvider()
    refreshed = analyze_scenes(
        [scene],
        blocked_provider,
        StoryStyle.CINEMATIC,
        cached_report=cached_report,
    )
    provider = _VisionProvider()

    reused = analyze_scenes(
        [scene],
        provider,
        StoryStyle.CINEMATIC,
        cached_report=cached_report,
        allow_content_identity_migration=True,
    )

    assert blocked_provider.calls == 1
    assert refreshed.cached_count == 0
    assert provider.calls == 0
    assert reused.analyzed_count == 0
    assert reused.cached_count == 1
    assert reused.scenes[0].metadata["vision_cache_key"] != "legacy-stat-based-key"
    assert reused.scenes[0].metadata["vision_style"] == StoryStyle.CINEMATIC.value


def test_vision_batch_isolates_poison_scene_and_retries_degraded_result(
    tmp_path: Path,
) -> None:
    scenes: list[Scene] = []
    for name in ("good-first.png", "poison.png", "good-last.png"):
        frame = tmp_path / name
        frame.write_bytes(name.encode())
        scenes.append(
            Scene(
                asset_id=uuid4(),
                start_seconds=0,
                end_seconds=3,
                keyframe_path=frame,
                quality_score=70,
                metadata={"cache_key": name},
            )
        )

    class BatchProvider(_VisionProvider):
        batch_size = 3

        def __init__(self, *, poison_fails: bool) -> None:
            super().__init__()
            self.poison_fails = poison_fails
            self.batch_calls: list[list[str]] = []

        def analyze_batch(
            self,
            image_paths: list[Path],
            style: StoryStyle,
        ) -> list[SceneUnderstanding]:
            self.batch_calls.append([path.name for path in image_paths])
            if self.poison_fails and any(path.name == "poison.png" for path in image_paths):
                raise RuntimeError("malformed model output")
            return [self.analyze(path, style) for path in image_paths]

    failing = BatchProvider(poison_fails=True)
    first = analyze_scenes(
        scenes,
        failing,
        StoryStyle.CINEMATIC,
        max_scene_retries=1,
        allow_degraded_fallback=True,
    )

    assert first.analyzed_count == 2
    assert first.degraded_count == 1
    assert first.retry_count == 6
    assert first.scenes[1].metadata["vision_status"] == "degraded"
    assert sum(call == ["poison.png"] for call in failing.batch_calls) == 2

    recovered_provider = BatchProvider(poison_fails=False)
    recovered = analyze_scenes(
        scenes,
        recovered_provider,
        StoryStyle.CINEMATIC,
        cached_report=first,
        max_scene_retries=1,
        allow_degraded_fallback=True,
    )

    assert recovered.cached_count == 2
    assert recovered.analyzed_count == 1
    assert recovered.degraded_count == 0
    assert recovered.retry_count == 0
    assert recovered_provider.batch_calls == [["poison.png"]]
    assert recovered.scenes[1].metadata["vision_status"] == "analyzed"


def test_vision_degraded_fallback_does_not_hide_missing_runtime(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"frame")
    scene = Scene(
        asset_id=uuid4(),
        start_seconds=0,
        end_seconds=3,
        keyframe_path=frame,
    )

    class MissingRuntimeProvider(_VisionProvider):
        def analyze(self, image_path: Path, style: StoryStyle) -> SceneUnderstanding:
            del image_path, style
            raise DependencyUnavailableError("local model runtime missing")

    with pytest.raises(DependencyUnavailableError, match="runtime missing"):
        analyze_scenes(
            [scene],
            MissingRuntimeProvider(),
            StoryStyle.CINEMATIC,
            max_scene_retries=2,
            allow_degraded_fallback=True,
        )
