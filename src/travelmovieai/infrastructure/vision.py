"""Local vision provider adapters."""

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from travelmovieai.core.exceptions import VisionAnalysisError
from travelmovieai.domain.enums import StoryStyle
from travelmovieai.domain.models import SceneUnderstanding

PROMPT_VERSION = "scene-understanding-v2-contact-sheet"


class LMStudioVisionProvider:
    """Analyze representative frames through LM Studio's OpenAI-compatible API."""

    name = "lm-studio"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self._structured_output_supported = True

    def analyze(self, image_path: Path, style: StoryStyle) -> SceneUnderstanding:
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as error:
            raise VisionAnalysisError(
                f"Не удалось прочитать representative frame: {image_path.name}"
            ) from error
        schema = SceneUnderstanding.model_json_schema()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a travel film editor. Analyze only visible evidence. "
                    "Return concise JSON matching the supplied schema. "
                    "Do not identify unknown people or invent locations."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Evaluate this scene for a {style.value} travel movie. "
                            "For video, the image may contain three chronological "
                            "frames from the same scene (start, middle, end). "
                            "importance_score measures story value, visual interest, "
                            "emotion, and uniqueness from 0 to 100. "
                            f"Return only JSON matching this schema: {json.dumps(schema)}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{encoded}",
                        },
                    },
                ],
            },
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 1200,
            "messages": messages,
        }
        if self._structured_output_supported:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "scene_understanding",
                    "strict": True,
                    "schema": schema,
                },
            }
        response = self._post(payload)
        try:
            return _parse_understanding(response)
        except (KeyError, IndexError, TypeError, ValidationError, json.JSONDecodeError):
            if not self._structured_output_supported:
                raise VisionAnalysisError(
                    "Локальная vision-модель вернула ответ, не соответствующий схеме."
                ) from None
            self._structured_output_supported = False
            payload.pop("response_format", None)
            retry = self._post(payload)
            try:
                return _parse_understanding(retry)
            except (
                KeyError,
                IndexError,
                TypeError,
                ValidationError,
                json.JSONDecodeError,
            ) as error:
                raise VisionAnalysisError(
                    "Локальная vision-модель не смогла вернуть структурированный "
                    "анализ сцены. Выберите другую vision-модель в интерфейсе."
                ) from error

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                parsed: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return parsed
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise VisionAnalysisError(
                f"LM Studio отклонил vision-запрос (HTTP {error.code}): {detail}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise VisionAnalysisError(
                "LM Studio недоступен. Запустите локальный сервер, загрузите "
                f"vision-модель '{self.model}' и проверьте {self.base_url}."
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise VisionAnalysisError("Не удалось прочитать ответ LM Studio.") from error


def _parse_understanding(response: dict[str, Any]) -> SceneUnderstanding:
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    normalized = _extract_json(_strip_json_fence(str(content)))
    return SceneUnderstanding.model_validate_json(normalized)


def _extract_json(content: str) -> str:
    stripped = content.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1 :] if first_newline >= 0 else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()
