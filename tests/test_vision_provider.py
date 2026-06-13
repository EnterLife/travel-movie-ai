import json
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from travelmovieai.core.exceptions import VisionAnalysisError
from travelmovieai.domain.enums import StoryStyle
from travelmovieai.infrastructure import vision
from travelmovieai.infrastructure.vision import LMStudioVisionProvider


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_lm_studio_provider_validates_structured_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake jpeg")
    content = {
        "caption": "Family walking near the sea",
        "location_type": "beach",
        "activity": "walking",
        "emotion": "joyful",
        "people_count": 3,
        "importance_score": 88,
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
    assert result.importance_score == 88


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
