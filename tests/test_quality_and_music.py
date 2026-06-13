import wave
from pathlib import Path

from PIL import Image

from travelmovieai.analysis.quality import VisualQualityAnalyzer, analyze_scene_quality
from travelmovieai.domain.enums import StoryStyle
from travelmovieai.domain.models import QuickMontageSettings, Scene
from travelmovieai.story.music import build_music_plan, choose_music_profile


def test_quality_analysis_persists_explainable_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "bright.jpg"
    Image.new("RGB", (160, 90), (230, 130, 60)).save(image_path)
    scene = Scene(
        asset_id="b04173c0-6122-40e8-b12c-1101942f64d7",
        start_seconds=0,
        end_seconds=2,
        keyframe_path=image_path,
    )

    report = analyze_scene_quality([scene], VisualQualityAnalyzer())

    analyzed = report.scenes[0]
    assert analyzed.quality_score is not None
    metrics = analyzed.metadata["quality_metrics"]
    assert metrics["brightness"] > 40
    assert metrics["backend"] in {"opencv", "pillow"}


def test_auto_music_profile_uses_visual_metrics_and_generates_wav(
    tmp_path: Path,
) -> None:
    scene = Scene(
        asset_id="b04173c0-6122-40e8-b12c-1101942f64d7",
        start_seconds=0,
        end_seconds=2,
        metadata={
            "emotion": "exciting",
            "quality_metrics": {
                "brightness": 65,
                "saturation": 70,
                "sharpness": 75,
            },
        },
    )
    settings = QuickMontageSettings(
        target_duration_seconds=8,
        story_style=StoryStyle.ADVENTURE,
        music_mode="auto",
    )
    output = tmp_path / "generated.wav"

    profile, _ = choose_music_profile([scene], settings)
    plan = build_music_plan(
        [],
        [scene],
        settings,
        tmp_path / "music",
        output,
        8,
    )

    assert profile == "energetic"
    assert plan.mode == "generated"
    assert plan.source_path == output
    assert plan.generated is True
    with wave.open(str(output), "rb") as soundtrack:
        assert soundtrack.getnchannels() == 2
        assert soundtrack.getnframes() > 0
