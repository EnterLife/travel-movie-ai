from pathlib import Path
from uuid import uuid4

import pytest

from travelmovieai.analysis.vision import analyze_scenes
from travelmovieai.domain.enums import StoryStyle
from travelmovieai.domain.models import Scene, SceneUnderstanding, VisionAnalysisReport


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
