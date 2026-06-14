import json
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from travelmovieai.core.exceptions import VisionAnalysisError
from travelmovieai.domain.enums import StoryStyle
from travelmovieai.infrastructure import vision
from travelmovieai.infrastructure.vision import (
    LMStudioVisionProvider,
    LocalQwenVisionProvider,
    build_vision_provider,
    resolve_local_vision_model,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


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
        lm_studio_url="http://localhost:1234/v1",
        lm_studio_api_key=None,
        timeout_seconds=120,
    )

    assert isinstance(provider, LocalQwenVisionProvider)
    assert provider.model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert provider._loaded_model is None
    assert not provider.cache_dir.exists()


def test_lm_studio_provider_validates_structured_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake jpeg")
    content = {
        "caption": "Family walking near the sea",
        "detailed_description": "A family walks together along a sunny beach.",
        "location_type": "beach",
        "activity": "walking",
        "emotion": "joyful",
        "people_count": 3,
        "people_groups": ["family", "adults", "children"],
        "landmarks": [],
        "vision_score": 88,
        "score_factors": {
            "uniqueness": 70,
            "people": 80,
            "emotion": 85,
            "visual_quality": 50,
            "landmark": 0,
            "unusual_event": 30,
        },
        "story_relevance": "A warm family travel moment.",
        "tags": ["family", "sunset"],
    }
    monkeypatch.setattr(
        vision,
        "urlopen",
        lambda request, timeout: FakeResponse(
            {"choices": [{"message": {"content": json.dumps(content)}}]}
        ),
    )

    result = LMStudioVisionProvider(
        "http://localhost:1234/v1",
        "qwen-test",
        10,
    ).analyze(image, StoryStyle.FAMILY)

    assert result.caption == content["caption"]
    assert result.vision_score == 88


def test_lm_studio_provider_reports_unavailable_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake jpeg")

    def unavailable(request: object, timeout: float) -> FakeResponse:
        raise URLError("connection refused")

    monkeypatch.setattr(vision, "urlopen", unavailable)

    with pytest.raises(VisionAnalysisError, match="LM Studio недоступен"):
        LMStudioVisionProvider(
            "http://localhost:1234/v1",
            "qwen-test",
            10,
        ).analyze(image, StoryStyle.CINEMATIC)


def test_lm_studio_provider_reports_inference_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake jpeg")

    def timeout(request: object, timeout: float) -> FakeResponse:
        raise TimeoutError

    monkeypatch.setattr(vision, "urlopen", timeout)

    with pytest.raises(VisionAnalysisError, match="не завершила анализ"):
        LMStudioVisionProvider(
            "http://localhost:1234/v1",
            "slow-vision",
            10,
        ).analyze(image, StoryStyle.CINEMATIC)


def test_lm_studio_provider_retries_without_json_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake jpeg")
    responses = iter(
        [
            FakeResponse({"choices": [{"message": {"content": ""}}]}),
            FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Result:\n"
                                    '{"caption":"Beach","location_type":"beach",'
                                    '"detailed_description":"A person walks on a beach.",'
                                    '"activity":"walking","emotion":"relaxing",'
                                    '"people_count":1,"people_groups":["solo"],'
                                    '"landmarks":[],"vision_score":75,'
                                    '"score_factors":{"uniqueness":50,"people":30,'
                                    '"emotion":60,"visual_quality":50,"landmark":0,'
                                    '"unusual_event":20},"story_relevance":"Travel walk.",'
                                    '"tags":[]}'
                                )
                            }
                        }
                    ]
                }
            ),
        ]
    )
    calls: list[dict[str, Any]] = []

    def respond(request: Any, timeout: float) -> FakeResponse:
        calls.append(json.loads(request.data.decode("utf-8")))
        return next(responses)

    monkeypatch.setattr(vision, "urlopen", respond)
    result = LMStudioVisionProvider(
        "http://localhost:1234/v1",
        "reasoning-vision",
        10,
    ).analyze(image, StoryStyle.CINEMATIC)

    assert result.caption == "Beach"
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]
