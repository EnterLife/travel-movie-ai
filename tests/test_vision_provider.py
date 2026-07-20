from pathlib import Path

import pytest

from travelmovieai.domain.enums import StoryStyle
from travelmovieai.infrastructure.vision import (
    LocalQwenVisionProvider,
    _parse_local_qwen_understanding,
    build_vision_provider,
    resolve_local_vision_model,
)


def test_local_model_auto_selection_matches_available_memory() -> None:
    assert (
        resolve_local_vision_model(
            "auto",
            gpu_memory_mb=6144,
            system_memory_mb=32768,
        )
        == "Qwen/Qwen2.5-VL-3B-Instruct"
    )
    assert (
        resolve_local_vision_model(
            "auto",
            gpu_memory_mb=12288,
            system_memory_mb=32768,
        )
        == "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    assert (
        resolve_local_vision_model(
            "custom/model",
            gpu_memory_mb=6144,
            system_memory_mb=32768,
        )
        == "custom/model"
    )


def test_local_provider_factory_is_lazy(tmp_path: Path) -> None:
    provider = build_vision_provider(
        provider="local",
        model="auto",
        device="auto",
        cache_dir=tmp_path / "models",
        allow_download=True,
        gpu_memory_mb=6144,
        system_memory_mb=32768,
        model_batch_size=2,
    )

    assert isinstance(provider, LocalQwenVisionProvider)
    assert provider.model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert provider.quantize_4bit is True
    assert provider.batch_size == 2
    assert provider._loaded_model is None
    assert not provider.cache_dir.exists()


def test_local_qwen_empty_batch_does_not_load_model(tmp_path: Path) -> None:
    provider = LocalQwenVisionProvider(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        cache_dir=tmp_path / "models",
    )

    assert provider.analyze_batch([], StoryStyle.CINEMATIC) == []
    assert provider._loaded_model is None


def test_large_local_model_uses_gpu_and_ram_offload_on_six_gb_gpu(
    tmp_path: Path,
) -> None:
    provider = build_vision_provider(
        provider="local",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        device="auto",
        cache_dir=tmp_path / "models",
        allow_download=True,
        gpu_memory_mb=6144,
        system_memory_mb=32768,
        model_batch_size=8,
    )

    assert isinstance(provider, LocalQwenVisionProvider)
    assert provider.quantize_4bit is True
    assert provider.use_cpu_offload is True
    assert provider.batch_size == 1
    assert provider._max_memory() == {0: "5376MiB", "cpu": "28672MiB"}


def test_local_qwen_response_normalizes_common_schema_drift() -> None:
    result = _parse_local_qwen_understanding(
        """
        ```json
        {
          "caption": "Beach scene",
          "detailed_description": "Three objects stand on a beach.",
          "location_type": "beach",
          "activity": "walking",
          "emotion": "neutral",
          "shot_scale": "establishing shot",
          "camera_motion": "gimbal",
          "focus_x": 0.42,
          "focus_y": 0.31,
          "focus_source": "person",
          "people_count": "2-4",
          "people_groups": "none",
          "landmarks": [],
          "vision_score": {"all": 60},
          "score_factors": {
            "uniqueness": 75,
            "people": 25,
            "emotion": 30,
            "visual_quality": 80,
            "landmark": 50,
            "unusual_event": 40,
            "all": 60
          },
          "story_relevance": 70,
          "tags": "beach"
        }
        ```
        """
    )

    assert result.people_count == 4
    assert result.people_groups[0].value == "none"
    assert result.vision_score == 60
    assert result.shot_scale == "extreme_wide"
    assert result.camera_motion == "tracking"
    assert result.focus_x == pytest.approx(0.42)
    assert result.focus_y == pytest.approx(0.31)
    assert result.focus_source == "face"
    assert result.story_relevance == "Model relevance score: 70/100."
    assert result.tags == ["beach"]


def test_local_qwen_response_normalizes_textual_people_count() -> None:
    result = _parse_local_qwen_understanding(
        """
        {
          "caption": "Empty coastline",
          "detailed_description": "A coastline with no people visible.",
          "location_type": "coastline",
          "activity": "drone shot",
          "emotion": "calm",
          "people_count": "no people",
          "people_groups": "none",
          "landmarks": [],
          "vision_score": 55,
          "score_factors": {},
          "story_relevance": "Useful establishing shot.",
          "tags": ["sea"]
        }
        """
    )

    assert result.people_count == 0
    assert result.location_type.value == "sea"
    assert result.activity.value == "sightseeing"
    assert result.emotion.value == "relaxing"
    assert result.shot_scale == "unknown"
    assert result.camera_motion == "unknown"
    assert result.focus_x is None
    assert result.focus_y is None
    assert result.focus_source is None


def test_local_qwen_discards_incomplete_or_out_of_range_focus() -> None:
    result = _parse_local_qwen_understanding(
        """
        {
          "caption": "Wide landscape",
          "detailed_description": "A mountain landscape.",
          "focus_x": 1.4,
          "focus_y": 0.2,
          "focus_source": "object",
          "score_factors": {}
        }
        """
    )

    assert result.focus_x is None
    assert result.focus_y is None
    assert result.focus_source is None


def test_local_qwen_normalizes_and_validates_temporal_highlights() -> None:
    result = _parse_local_qwen_understanding(
        """
        {
          "caption": "Changing coastal view",
          "detailed_description": "The view opens toward the sea.",
          "score_factors": {},
          "highlight_windows": [
            {
              "relative_start": 0.68,
              "relative_end": 0.9,
              "relative_position": 0.82,
              "confidence": 92,
              "label": "coast revealed"
            },
            {
              "relative_start": 0.7,
              "relative_end": 0.4,
              "relative_position": 0.5,
              "confidence": 0.9,
              "label": "invalid reversed interval"
            },
            {
              "relative_position": 0.02,
              "confidence": 0.7,
              "label": "opening"
            }
          ]
        }
        """
    )

    assert len(result.highlight_windows) == 2
    opening, reveal = result.highlight_windows
    assert opening.relative_start == 0
    assert opening.relative_position == pytest.approx(0.02)
    assert reveal.relative_end == pytest.approx(0.9)
    assert reveal.confidence == pytest.approx(0.92)
    assert reveal.source == "vision"
