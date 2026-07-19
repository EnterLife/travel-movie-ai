"""Lazy local text-model adapter for structured storyboards."""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from travelmovieai.core.exceptions import DependencyUnavailableError, StoryGenerationError
from travelmovieai.domain.enums import StoryStyle
from travelmovieai.domain.models import (
    Event,
    Scene,
    Storyboard,
    StoryModelOutput,
    StorySection,
)

STORY_PROMPT_VERSION = "local-story-v1"


class StoryProvider(Protocol):
    name: str
    model: str

    def build(
        self,
        events: list[Event],
        scenes: list[Scene],
        style: StoryStyle,
    ) -> Storyboard: ...

    def release(self) -> None: ...


class LocalTransformersStoryProvider:
    """Generate a storyboard locally with deterministic decoding."""

    name = "local-transformers"

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        cache_dir: Path = Path("models/story"),
        allow_download: bool = True,
        max_new_tokens: int = 768,
    ) -> None:
        self.model = model
        self.device = device
        self.cache_dir = cache_dir
        self.allow_download = allow_download
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any = None
        self._loaded_model: Any = None
        self._torch: Any = None

    def build(
        self,
        events: list[Event],
        scenes: list[Scene],
        style: StoryStyle,
    ) -> Storyboard:
        if not events:
            return Storyboard(
                title="Travel Movie",
                style=style,
                provider=self.name,
                model=self.model,
                prompt_version=STORY_PROMPT_VERSION,
            )
        self._ensure_loaded()
        prompt = _story_prompt(events, scenes, style)
        try:
            resolved_device = self._resolved_device()
            apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
            if callable(apply_chat_template):
                prompt = apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(resolved_device)
            with self._torch.inference_mode():
                generated = self._loaded_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            input_length = int(inputs["input_ids"].shape[-1])
            content = self._tokenizer.decode(
                generated[0][input_length:],
                skip_special_tokens=True,
            )
            return parse_story_model_output(
                content,
                events,
                style,
                provider=self.name,
                model=self.model,
            )
        except StoryGenerationError:
            raise
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise StoryGenerationError(
                "The local story model could not generate a storyboard. "
                "Check available RAM/VRAM and the selected model."
            ) from error

    def release(self) -> None:
        self._loaded_model = None
        self._tokenizer = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _ensure_loaded(self) -> None:
        if self._loaded_model is not None:
            return
        try:
            self._torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            importlib.import_module("accelerate")
        except ImportError as error:
            raise DependencyUnavailableError(
                "Install the story optional dependency group for the local story model: "
                'python -m pip install -e ".[story]".'
            ) from error

        resolved_device = self._resolved_device()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        load_options: dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": not self.allow_download,
        }
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model,
                **load_options,
            )
            self._loaded_model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model,
                low_cpu_mem_usage=True,
                torch_dtype="auto",
                **load_options,
            ).to(resolved_device)
            self._loaded_model.eval()
        except (AttributeError, OSError, RuntimeError, ValueError) as error:
            mode = "download-enabled mode" if self.allow_download else "cache-only mode"
            raise StoryGenerationError(
                f"Could not load local story model {self.model!r} in {mode}."
            ) from error

    def _resolved_device(self) -> str:
        if self.device == "cuda":
            if self._torch is not None and not self._torch.cuda.is_available():
                raise DependencyUnavailableError(
                    "CUDA was selected for the story model, but PyTorch cannot see the GPU."
                )
            return "cuda"
        if self.device == "auto" and self._torch is not None and self._torch.cuda.is_available():
            return "cuda"
        return "cpu"


def build_story_provider(
    *,
    model: str,
    device: str,
    cache_dir: Path,
    allow_download: bool,
    max_new_tokens: int,
) -> StoryProvider:
    """Build the selected local adapter without importing heavy dependencies."""

    return LocalTransformersStoryProvider(
        model,
        device=device,
        cache_dir=cache_dir,
        allow_download=allow_download,
        max_new_tokens=max_new_tokens,
    )


def parse_story_model_output(
    content: str,
    events: list[Event],
    style: StoryStyle,
    *,
    provider: str,
    model: str,
) -> Storyboard:
    """Validate model JSON and derive scene membership from trusted events."""

    try:
        payload = StoryModelOutput.model_validate_json(_extract_json(content))
    except (ValidationError, ValueError) as error:
        raise StoryGenerationError(
            "The local story model returned invalid structured JSON."
        ) from error

    expected_ids = [event.id for event in events]
    received_ids = [event_id for section in payload.sections for event_id in section.event_ids]
    if len(received_ids) != len(set(received_ids)) or set(received_ids) != set(expected_ids):
        raise StoryGenerationError(
            "The local story model must include every known event exactly once."
        )
    if payload.sections[0].role != "opening":
        raise StoryGenerationError("The local story must begin with an opening section.")
    roles = [section.role for section in payload.sections]
    if roles.count("opening") != 1:
        raise StoryGenerationError("The local story must contain exactly one opening section.")
    if len(events) > 1:
        if payload.sections[-1].role != "finale" or roles.count("finale") != 1:
            raise StoryGenerationError("The local story must end with exactly one finale section.")
    elif "finale" in roles:
        raise StoryGenerationError("A one-event local story must use one opening section.")
    if roles.count("highlight") > 1:
        raise StoryGenerationError("The local story may contain at most one highlight section.")

    events_by_id = {event.id: event for event in events}
    sections = [
        StorySection(
            role=section.role,
            title=section.title,
            event_ids=section.event_ids,
            scene_ids=[
                scene_id
                for event_id in section.event_ids
                for scene_id in events_by_id[event_id].scene_ids
            ],
        )
        for section in payload.sections
    ]
    return Storyboard(
        title=payload.title,
        style=style,
        event_ids=received_ids,
        sections=sections,
        provider=provider,
        model=model,
        prompt_version=STORY_PROMPT_VERSION,
    )


def _story_prompt(events: list[Event], scenes: list[Scene], style: StoryStyle) -> str:
    scenes_by_id = {scene.id: scene for scene in scenes}
    event_payload = [
        {
            "id": str(event.id),
            "title": event.title,
            "summary": event.summary[:500],
            "importance_score": event.importance_score,
            "location_type": event.location_type.value,
            "activity": event.activity.value,
            "landmarks": event.landmarks,
            "scene_context": [
                {
                    "caption": (scene.caption or "")[:160],
                    "description": str(scene.metadata.get("detailed_description", ""))[:240],
                    "transcript": (scene.transcript or "")[:160],
                    "emotion": str(scene.metadata.get("emotion", ""))[:40],
                    "shot_scale": str(scene.metadata.get("shot_scale", ""))[:40],
                    "camera_motion": str(scene.metadata.get("camera_motion", ""))[:40],
                }
                for scene_id in event.scene_ids[:2]
                if (scene := scenes_by_id.get(scene_id)) is not None
            ],
        }
        for event in events
    ]
    return (
        "You are a local travel-film story editor. Build a concise "
        f"{style.value} narrative using only the supplied events. Return only one JSON "
        "object with title and sections. Every section must contain role "
        "(opening|journey|highlight|finale), title, and event_ids. Include every supplied "
        "event ID exactly once; do not invent IDs. The first role must be opening and, when "
        "there is more than one event, the last role must be finale. Events: "
        + json.dumps(event_payload, ensure_ascii=False, separators=(",", ":"))
    )


def _extract_json(content: str) -> str:
    stripped = content.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped
