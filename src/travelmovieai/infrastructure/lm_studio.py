"""OpenAI-compatible LM Studio discovery client."""

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class LMStudioModels:
    available: bool
    models: tuple[str, ...] = ()
    error: str | None = None


def list_lm_studio_models(
    base_url: str,
    api_key: str | None = None,
    timeout_seconds: float = 5,
) -> LMStudioModels:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"{base_url.rstrip('/')}/models", headers=headers)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return LMStudioModels(
            available=False,
            error=f"LM Studio HTTP {error.code}",
        )
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        return LMStudioModels(available=False, error=str(error))

    models = tuple(
        str(item["id"])
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    )
    return LMStudioModels(available=True, models=models)


def resolve_vision_model(
    discovered: LMStudioModels,
    configured_model: str,
) -> str:
    if configured_model in discovered.models:
        return configured_model
    markers = ("vision", "-vl", "/vl", "omni", "gemma-3", "gemma-4")
    return next(
        (
            model
            for model in discovered.models
            if any(marker in model.casefold() for marker in markers)
        ),
        discovered.models[0] if discovered.models else configured_model,
    )
