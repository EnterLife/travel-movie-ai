"""Local vision provider adapters."""

from __future__ import annotations

import gc
import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from PIL import Image
from pydantic import ValidationError

from travelmovieai.core.exceptions import DependencyUnavailableError, VisionAnalysisError
from travelmovieai.domain.enums import (
    ActivityType,
    EmotionType,
    LocationType,
    PersonGroup,
    StoryStyle,
)
from travelmovieai.domain.models import SceneUnderstanding, VisionScoreFactors
from travelmovieai.infrastructure.model_pool import BoundedModelPool, ModelLease, ModelPoolStats

PROMPT_VERSION = "scene-understanding-v5-focus-point"
LOCAL_QWEN_MODELS = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
)
COMPACT_OUTPUT_CONTRACT = """
Return one JSON object with these fields:
caption, detailed_description,
location_type (unknown|beach|sea|mountains|forest|city|airport|hotel|restaurant|
museum|park|landmark|transport|indoor|other),
activity (unknown|walking|sightseeing|swimming|hiking|cycling|traveling|relaxing|
sports|dining|boating|arriving|departing|other),
emotion (neutral|joyful|exciting|relaxing|romantic|emotional|adventurous|cinematic),
shot_scale (unknown|extreme_wide|wide|full|medium|close_up|extreme_close_up),
camera_motion (unknown|static|pan|tilt|tracking|handheld|zoom|drone|orbit),
focus_x and focus_y (0-1 normalized center of the primary visible subject),
focus_source (face|object|subject; use null for all three focus fields if no clear subject),
people_count, people_groups (none|adults|children|family|group|solo|mixed),
landmarks [{name, confidence 0-1, evidence}],
vision_score 0-100,
score_factors {uniqueness, people, emotion, visual_quality, landmark,
unusual_event}, story_relevance, tags.
""".strip()


@dataclass(slots=True)
class _VisionRuntime:
    torch: Any
    processor: Any
    model: Any
    inference_lock: RLock = field(default_factory=RLock)


_VISION_RUNTIME_POOL: BoundedModelPool[_VisionRuntime] = BoundedModelPool(1)


class LocalQwenVisionProvider:
    """Run Qwen2.5-VL directly with Transformers and cached local weights."""

    name = "local-qwen"

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        cache_dir: Path = Path("models"),
        allow_download: bool = True,
        quantize_4bit: bool = False,
        use_cpu_offload: bool = False,
        gpu_memory_mb: int | None = None,
        system_memory_mb: int | None = None,
        batch_size: int = 1,
        model_pool_size: int = 1,
    ) -> None:
        self.model = model
        self.device = device
        self.cache_dir = cache_dir
        self.allow_download = allow_download
        self.quantize_4bit = quantize_4bit
        self.use_cpu_offload = use_cpu_offload
        self.gpu_memory_mb = gpu_memory_mb
        self.system_memory_mb = system_memory_mb
        self.batch_size = max(1, batch_size)
        self.model_pool_size = model_pool_size
        self._processor: Any = None
        self._loaded_model: Any = None
        self._torch: Any = None
        self._runtime_lease: ModelLease[_VisionRuntime] | None = None

    @property
    def runtime_description(self) -> str:
        if self._loaded_model is None:
            return "not loaded"
        device = str(self._loaded_model.device)
        precision = "4-bit NF4" if self.quantize_4bit else "native precision"
        placement = ", GPU + RAM offload" if self.use_cpu_offload else ""
        return f"{device}, {precision}{placement}"

    def prepare(self) -> None:
        self._ensure_loaded()

    def release(self) -> None:
        lease = self._runtime_lease
        self._runtime_lease = None
        self._loaded_model = None
        self._processor = None
        self._torch = None
        if lease is not None:
            lease.release()

    def analyze(self, image_path: Path, style: StoryStyle) -> SceneUnderstanding:
        return self.analyze_batch([image_path], style)[0]

    def analyze_batch(
        self,
        image_paths: list[Path],
        style: StoryStyle,
    ) -> list[SceneUnderstanding]:
        if not image_paths:
            return []
        self._ensure_loaded()
        prompt = (
            "You are the Vision AI module of a travel film editor. Analyze only "
            "visible evidence. Do not identify unknown people or invent landmarks. "
            "A landmark requires visible architectural or textual evidence. "
            f"Evaluate this scene for a {style.value} travel movie. The contact "
            "sheet contains chronological frames from one scene (start, middle, "
            "end), so describe meaningful changes across them. Return only compact "
            "valid JSON without markdown. people_groups and landmarks must be arrays. "
            "vision_score must be one number. story_relevance must be text. "
            "Use 0-100 for vision_score and every score factor. Set visual_quality "
            "to 50 because measured OpenCV quality replaces it later. "
            f"{COMPACT_OUTPUT_CONTRACT}"
        )
        try:
            images = []
            for image_path in image_paths:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                images.append(image)
            contents = self._generate_contents(images, prompt, max_new_tokens=320)
            results = []
            for image, content in zip(images, contents, strict=True):
                try:
                    results.append(_parse_local_qwen_understanding(content))
                except (ValidationError, json.JSONDecodeError):
                    retry_prompt = (
                        f"{prompt} Keep caption and descriptions concise. "
                        "The complete JSON must fit within the response limit."
                    )
                    retry = self._generate_contents(
                        [image],
                        retry_prompt,
                        max_new_tokens=480,
                    )[0]
                    results.append(_parse_local_qwen_understanding(retry))
            return results
        except (ValidationError, json.JSONDecodeError) as error:
            raise VisionAnalysisError(
                "The local Qwen model returned a response that did not match the "
                "scene analysis schema. Retry the scene or choose another model."
            ) from error
        except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
            raise VisionAnalysisError(
                f"The local Qwen model could not analyze the scene image batch "
                f"({len(image_paths)} image(s)). "
                "Check available RAM/VRAM and choose the 3B model if needed."
            ) from error

    def _generate_contents(
        self,
        images: list[Image.Image],
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> list[str]:
        lease = self._runtime_lease
        if lease is None:
            raise RuntimeError("Vision runtime is not acquired.")
        with lease.value.inference_lock:
            return self._generate_contents_locked(
                images,
                prompt,
                max_new_tokens=max_new_tokens,
            )

    def _generate_contents_locked(
        self,
        images: list[Image.Image],
        prompt: str,
        *,
        max_new_tokens: int,
    ) -> list[str]:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for image in images
        ]
        texts = [
            self._processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        inputs = self._processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self._loaded_model.device)
        with self._torch.inference_mode():
            generated_ids = self._loaded_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                inputs.input_ids,
                generated_ids,
                strict=True,
            )
        ]
        decoded: list[str] = self._processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded

    def _ensure_loaded(self) -> None:
        if self._loaded_model is not None:
            return
        _VISION_RUNTIME_POOL.configure(self.model_pool_size)
        lease = _VISION_RUNTIME_POOL.acquire(
            self._runtime_key(),
            self._load_runtime,
            _dispose_runtime,
        )
        self._runtime_lease = lease
        self._torch = lease.value.torch
        self._processor = lease.value.processor
        self._loaded_model = lease.value.model

    def _load_runtime(self) -> _VisionRuntime:
        try:
            torch_module = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            importlib.import_module("accelerate")
            if self.quantize_4bit:
                importlib.import_module("bitsandbytes")
        except ImportError as error:
            raise DependencyUnavailableError(
                "Install the vision optional dependency group for local Qwen Vision: "
                'python -m pip install -e ".[vision]".'
            ) from error

        if self.device == "cuda" and not torch_module.cuda.is_available():
            raise DependencyUnavailableError(
                "CUDA was selected, but the installed PyTorch build cannot see the GPU."
            )
        resolved_device = (
            "cuda"
            if self.device in {"auto", "cuda"} and torch_module.cuda.is_available()
            else "cpu"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            if resolved_device == "cuda":
                torch_module.backends.cuda.matmul.allow_tf32 = True
            processor_type = transformers.AutoProcessor
            model_type = transformers.Qwen2_5_VLForConditionalGeneration
            processor = processor_type.from_pretrained(
                self.model,
                cache_dir=self.cache_dir,
                local_files_only=not self.allow_download,
                min_pixels=256 * 28 * 28,
                max_pixels=(768 if _model_size_billions(self.model) >= 7 else 640) * 28 * 28,
            )
            model_options: dict[str, Any] = {
                "cache_dir": self.cache_dir,
                "local_files_only": not self.allow_download,
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",
            }
            if resolved_device == "cuda" and self.quantize_4bit:
                device_map: dict[str, int] | str = {"": 0}
                model_options.update(
                    {
                        "dtype": torch_module.float16,
                        "device_map": device_map,
                        "quantization_config": transformers.BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch_module.float16,
                            bnb_4bit_use_double_quant=True,
                            llm_int8_enable_fp32_cpu_offload=self.use_cpu_offload,
                        ),
                    }
                )
                if self.use_cpu_offload:
                    offload_dir = self.cache_dir / "offload" / self.model.replace("/", "--")
                    offload_dir.mkdir(parents=True, exist_ok=True)
                    model_options.update(
                        {
                            "device_map": "auto",
                            "max_memory": self._max_memory(),
                            "offload_folder": offload_dir,
                            "offload_state_dict": True,
                        }
                    )
            else:
                model_options.update(
                    {
                        "dtype": "auto",
                        "device_map": "auto" if resolved_device == "cuda" else "cpu",
                    }
                )
            loaded_model = model_type.from_pretrained(
                self.model,
                **model_options,
            )
            loaded_model.eval()
            return _VisionRuntime(
                torch=torch_module,
                processor=processor,
                model=loaded_model,
            )
        except (AttributeError, OSError, RuntimeError, ValueError) as error:
            download_hint = (
                "Check internet access and free space in the model cache."
                if self.allow_download
                else "Auto-download is disabled and the model is missing from the local cache."
            )
            raise VisionAnalysisError(
                f"Could not load Qwen Vision '{self.model}'. {download_hint}"
            ) from error

    def _runtime_key(self) -> tuple[object, ...]:
        return (
            "qwen",
            self.model,
            self.device,
            str(self.cache_dir.expanduser().resolve()),
            self.allow_download,
            self.quantize_4bit,
            self.use_cpu_offload,
            self.gpu_memory_mb,
            self.system_memory_mb,
        )

    def _max_memory(self) -> dict[int | str, str]:
        gpu_memory_mb = max(1024, (self.gpu_memory_mb or 6144) - 768)
        system_memory_mb = max(4096, (self.system_memory_mb or 16384) - 4096)
        return {
            0: f"{gpu_memory_mb}MiB",
            "cpu": f"{system_memory_mb}MiB",
        }


class Florence2VisionProvider:
    """Run Florence-2 locally and normalize its caption into the scene schema."""

    name = "florence-2"

    def __init__(
        self,
        model: str,
        device: str = "auto",
        *,
        cache_dir: Path = Path("models"),
        allow_download: bool = True,
        model_pool_size: int = 1,
    ) -> None:
        self.model = model
        self.device = device
        self.cache_dir = cache_dir
        self.allow_download = allow_download
        self.model_pool_size = model_pool_size
        self._processor: Any = None
        self._loaded_model: Any = None
        self._torch: Any = None
        self._runtime_lease: ModelLease[_VisionRuntime] | None = None

    def analyze(self, image_path: Path, style: StoryStyle) -> SceneUnderstanding:
        self._ensure_loaded()
        lease = self._runtime_lease
        if lease is None:
            raise VisionAnalysisError("Florence-2 runtime is not acquired.")
        try:
            with lease.value.inference_lock, Image.open(image_path) as source:
                image = source.convert("RGB")
                image_size = image.size
                task = "<MORE_DETAILED_CAPTION>"
                inputs = self._processor(text=task, images=image, return_tensors="pt")
                inputs = {key: value.to(self._resolved_device()) for key, value in inputs.items()}
                with self._torch.inference_mode():
                    generated_ids = self._loaded_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        num_beams=3,
                        do_sample=False,
                    )
                generated = self._processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=False,
                )[0]
                parsed = self._processor.post_process_generation(
                    generated,
                    task=task,
                    image_size=image_size,
                )
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            raise VisionAnalysisError(f"Florence-2 could not analyze {image_path.name}.") from error
        caption = str(parsed.get(task, generated)).strip()
        return _understanding_from_caption(caption, style)

    def release(self) -> None:
        lease = self._runtime_lease
        self._runtime_lease = None
        self._loaded_model = None
        self._processor = None
        self._torch = None
        if lease is not None:
            lease.release()

    def _ensure_loaded(self) -> None:
        if self._loaded_model is not None:
            return
        _VISION_RUNTIME_POOL.configure(self.model_pool_size)
        lease = _VISION_RUNTIME_POOL.acquire(
            self._runtime_key(),
            self._load_runtime,
            _dispose_runtime,
        )
        self._runtime_lease = lease
        self._torch = lease.value.torch
        self._processor = lease.value.processor
        self._loaded_model = lease.value.model

    def _load_runtime(self) -> _VisionRuntime:
        try:
            torch_module = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise DependencyUnavailableError(
                "Install the vision optional dependency group for Florence-2 and a "
                "PyTorch build compatible with your system."
            ) from error
        device = self._resolved_device(torch_module)
        dtype = torch_module.float16 if device == "cuda" else torch_module.float32
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            processor_type = transformers.AutoProcessor
            model_type = transformers.AutoModelForCausalLM
            processor = processor_type.from_pretrained(
                self.model,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                local_files_only=not self.allow_download,
            )
            loaded_model = model_type.from_pretrained(
                self.model,
                trust_remote_code=True,
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
                local_files_only=not self.allow_download,
            ).to(device)
            loaded_model.eval()
            return _VisionRuntime(
                torch=torch_module,
                processor=processor,
                model=loaded_model,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise VisionAnalysisError(
                f"Could not load local Florence-2 model '{self.model}'. "
                "Check internet access, free space, and auto-download settings."
            ) from error

    def _runtime_key(self) -> tuple[object, ...]:
        return (
            "florence",
            self.model,
            self.device,
            str(self.cache_dir.expanduser().resolve()),
            self.allow_download,
        )

    def _resolved_device(self, torch_module: Any | None = None) -> str:
        if self.device == "cuda":
            return "cuda"
        if self.device == "cpu":
            return "cpu"
        runtime_torch = torch_module if torch_module is not None else self._torch
        if runtime_torch is not None and runtime_torch.cuda.is_available():
            return "cuda"
        return "cpu"


def resolve_local_vision_model(
    configured_model: str | None,
    *,
    gpu_memory_mb: int | None,
    system_memory_mb: int | None,
) -> str:
    """Choose a practical Qwen size while preserving explicit model choices."""

    if configured_model and configured_model != "auto":
        return configured_model
    if (gpu_memory_mb or 0) >= 10 * 1024 or (
        not gpu_memory_mb and (system_memory_mb or 0) >= 48 * 1024
    ):
        return LOCAL_QWEN_MODELS[1]
    return LOCAL_QWEN_MODELS[0]


def clear_idle_vision_models() -> None:
    """Explicitly unload reusable Vision runtimes that have no active stage."""

    _VISION_RUNTIME_POOL.clear_idle()


def vision_model_pool_stats() -> ModelPoolStats:
    return _VISION_RUNTIME_POOL.stats()


def _dispose_runtime(runtime: _VisionRuntime) -> None:
    torch_module = runtime.torch
    runtime.model = None
    runtime.processor = None
    gc.collect()
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    runtime.torch = None


def _model_size_billions(model: str) -> int:
    normalized = model.casefold()
    if "32b" in normalized:
        return 32
    if "7b" in normalized:
        return 7
    if "3b" in normalized:
        return 3
    return 7


def build_vision_provider(
    *,
    provider: str,
    model: str | None,
    device: str,
    cache_dir: Path,
    allow_download: bool,
    gpu_memory_mb: int | None,
    system_memory_mb: int | None,
    model_batch_size: int = 1,
    model_pool_size: int = 1,
) -> LocalQwenVisionProvider | Florence2VisionProvider:
    """Build the selected backend without importing model-heavy packages."""

    configured_model = model or "auto"
    if provider == "florence":
        return Florence2VisionProvider(
            model=(
                configured_model if configured_model != "auto" else "microsoft/Florence-2-large"
            ),
            device=device,
            cache_dir=cache_dir,
            allow_download=allow_download,
            model_pool_size=model_pool_size,
        )
    resolved_model = resolve_local_vision_model(
        configured_model,
        gpu_memory_mb=gpu_memory_mb,
        system_memory_mb=system_memory_mb,
    )
    model_size = _model_size_billions(resolved_model)
    gpu_memory = gpu_memory_mb or 0
    quantize_4bit = (
        device in {"auto", "cuda"} and gpu_memory > 0 and gpu_memory < model_size * 2048 + 1536
    )
    use_cpu_offload = quantize_4bit and model_size >= 7 and gpu_memory < model_size * 768 + 1536
    return LocalQwenVisionProvider(
        resolved_model,
        device=device,
        cache_dir=cache_dir,
        allow_download=allow_download,
        quantize_4bit=quantize_4bit,
        use_cpu_offload=use_cpu_offload,
        gpu_memory_mb=gpu_memory_mb,
        system_memory_mb=system_memory_mb,
        model_pool_size=model_pool_size,
        batch_size=(
            1
            if model_size >= 7
            else min(2, model_batch_size)
            if gpu_memory < 10 * 1024
            else model_batch_size
        ),
    )


def _parse_local_qwen_understanding(content: str) -> SceneUnderstanding:
    normalized_content = _extract_json(_strip_json_fence(content))
    try:
        payload = json.loads(normalized_content)
    except json.JSONDecodeError:
        json_repair: Any = importlib.import_module("json_repair")
        payload = json_repair.repair_json(normalized_content, return_objects=True)
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("Expected a JSON object", content, 0)

    factors = payload.get("score_factors")
    if not isinstance(factors, dict):
        factors = {}
    normalized_factors = {
        key: _score(factors.get(key), default)
        for key, default in {
            "uniqueness": 50,
            "people": 30,
            "emotion": 50,
            "visual_quality": 50,
            "landmark": 0,
            "unusual_event": 30,
        }.items()
    }

    groups = payload.get("people_groups", [])
    if isinstance(groups, str):
        groups = [groups]
    if not isinstance(groups, list):
        groups = []
    allowed_groups = {group.value for group in PersonGroup}
    groups = [str(group) for group in groups if str(group) in allowed_groups][:6]
    if not groups:
        groups = [PersonGroup.NONE.value]

    vision_score = payload.get("vision_score", 50)
    if isinstance(vision_score, dict):
        vision_score = vision_score.get("all", vision_score.get("overall"))
    if not isinstance(vision_score, (int, float)):
        vision_score = sum(normalized_factors.values()) / len(normalized_factors)

    relevance = payload.get("story_relevance", "")
    if isinstance(relevance, (int, float)):
        relevance = f"Model relevance score: {_score(relevance, 50):.0f}/100."

    landmarks = payload.get("landmarks", [])
    if isinstance(landmarks, dict):
        landmarks = [landmarks]
    if not isinstance(landmarks, list):
        landmarks = []

    tags = payload.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []
    focus_x, focus_y, focus_source = _normalized_focus(payload)

    payload.update(
        {
            "caption": str(payload.get("caption") or "Travel scene")[:500],
            "detailed_description": str(
                payload.get("detailed_description") or payload.get("caption") or "Travel scene."
            )[:1500],
            "location_type": _enum_value(
                payload.get("location_type"),
                {item.value for item in LocationType},
                LocationType.UNKNOWN.value,
                {
                    "coast": LocationType.SEA.value,
                    "coastline": LocationType.SEA.value,
                    "shore": LocationType.BEACH.value,
                    "urban": LocationType.CITY.value,
                    "drone": LocationType.OTHER.value,
                },
            ),
            "activity": _enum_value(
                payload.get("activity"),
                {item.value for item in ActivityType},
                ActivityType.UNKNOWN.value,
                {
                    "flying": ActivityType.TRAVELING.value,
                    "driving": ActivityType.TRAVELING.value,
                    "drone_shot": ActivityType.SIGHTSEEING.value,
                    "aerial": ActivityType.SIGHTSEEING.value,
                },
            ),
            "emotion": _enum_value(
                payload.get("emotion"),
                {item.value for item in EmotionType},
                EmotionType.NEUTRAL.value,
                {
                    "calm": EmotionType.RELAXING.value,
                    "peaceful": EmotionType.RELAXING.value,
                    "serene": EmotionType.RELAXING.value,
                    "dramatic": EmotionType.CINEMATIC.value,
                },
            ),
            "shot_scale": _enum_value(
                payload.get("shot_scale"),
                {
                    "unknown",
                    "extreme_wide",
                    "wide",
                    "full",
                    "medium",
                    "close_up",
                    "extreme_close_up",
                },
                "unknown",
                {
                    "establishing": "extreme_wide",
                    "establishing_shot": "extreme_wide",
                    "long": "wide",
                    "long_shot": "wide",
                    "full_shot": "full",
                    "medium_shot": "medium",
                    "closeup": "close_up",
                    "close_up_shot": "close_up",
                    "extreme_closeup": "extreme_close_up",
                },
            ),
            "camera_motion": _enum_value(
                payload.get("camera_motion"),
                {
                    "unknown",
                    "static",
                    "pan",
                    "tilt",
                    "tracking",
                    "handheld",
                    "zoom",
                    "drone",
                    "orbit",
                },
                "unknown",
                {
                    "locked_off": "static",
                    "locked": "static",
                    "panning": "pan",
                    "tilting": "tilt",
                    "dolly": "tracking",
                    "gimbal": "tracking",
                    "aerial": "drone",
                    "drone_shot": "drone",
                    "orbiting": "orbit",
                },
            ),
            "focus_x": focus_x,
            "focus_y": focus_y,
            "focus_source": focus_source,
            "people_count": _people_count(payload.get("people_count")),
            "people_groups": groups,
            "landmarks": landmarks,
            "vision_score": _score(vision_score, 50),
            "score_factors": normalized_factors,
            "story_relevance": str(relevance)[:500],
            "tags": [str(tag)[:100] for tag in tags[:20]],
        }
    )
    return SceneUnderstanding.model_validate(payload)


def _normalized_focus(payload: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    point = payload.get("focus_point")
    point_payload = point if isinstance(point, dict) else {}
    x = _unit_coordinate(payload.get("focus_x", point_payload.get("x")))
    y = _unit_coordinate(payload.get("focus_y", point_payload.get("y")))
    source = _enum_value(
        payload.get("focus_source", point_payload.get("source")),
        {"face", "object", "subject"},
        "",
        {"person": "face", "primary_subject": "subject", "main_subject": "subject"},
    )
    if x is None or y is None or not source:
        return None, None, None
    return x, y, source


def _unit_coordinate(value: Any) -> float | None:
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return None
    return coordinate if 0 <= coordinate <= 1 else None


def _enum_value(
    value: Any,
    allowed: set[str],
    default: str,
    aliases: dict[str, str] | None = None,
) -> str:
    aliases = aliases or {}
    normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in allowed else default


def _people_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, min(50, int(round(value))))
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"", "none", "no", "no people", "n/a", "unknown"}:
            return 0
        numbers = [int(item) for item in re.findall(r"\d+", normalized)]
        if numbers:
            return max(0, min(50, max(numbers)))
    return 0


def _score(value: Any, default: float) -> float:
    try:
        return min(100.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


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


def _understanding_from_caption(
    caption: str,
    style: StoryStyle,
) -> SceneUnderstanding:
    normalized = caption.casefold()
    location = _keyword_enum(
        normalized,
        {
            LocationType.BEACH: ("beach", "shore", "sand"),
            LocationType.SEA: ("sea", "ocean", "coast"),
            LocationType.MOUNTAINS: ("mountain", "peak", "alpine"),
            LocationType.FOREST: ("forest", "woodland", "trees"),
            LocationType.AIRPORT: ("airport", "terminal", "airplane"),
            LocationType.HOTEL: ("hotel", "resort", "lobby"),
            LocationType.RESTAURANT: ("restaurant", "cafe", "dining"),
            LocationType.MUSEUM: ("museum", "gallery", "exhibition"),
            LocationType.PARK: ("park", "garden"),
            LocationType.CITY: ("city", "street", "building", "urban"),
        },
        LocationType.UNKNOWN,
    )
    activity = _keyword_enum(
        normalized,
        {
            ActivityType.WALKING: ("walking", "strolling"),
            ActivityType.SWIMMING: ("swimming", "diving"),
            ActivityType.HIKING: ("hiking", "trail"),
            ActivityType.CYCLING: ("cycling", "bicycle", "bike"),
            ActivityType.DINING: ("eating", "dining", "meal"),
            ActivityType.BOATING: ("boat", "sailing", "cruise"),
            ActivityType.SIGHTSEEING: ("sightseeing", "touring", "visiting"),
            ActivityType.TRAVELING: ("driving", "train", "flight", "traveling"),
            ActivityType.RELAXING: ("relaxing", "resting", "sunbathing"),
        },
        ActivityType.UNKNOWN,
    )
    emotion = _keyword_enum(
        normalized,
        {
            EmotionType.JOYFUL: ("smiling", "happy", "joyful", "laughing"),
            EmotionType.EXCITING: ("exciting", "energetic", "crowded"),
            EmotionType.RELAXING: ("calm", "peaceful", "relaxing", "quiet"),
            EmotionType.ROMANTIC: ("romantic", "couple", "sunset"),
            EmotionType.ADVENTUROUS: ("adventure", "hiking", "climbing"),
            EmotionType.CINEMATIC: ("dramatic", "panoramic", "scenic"),
        },
        EmotionType.NEUTRAL,
    )
    people_groups, people_count = _people_from_caption(normalized)
    uniqueness = 65.0 if location != LocationType.UNKNOWN else 45.0
    emotion_score = 70.0 if emotion != EmotionType.NEUTRAL else 45.0
    people_score = min(90.0, 30.0 + people_count * 12)
    factors = VisionScoreFactors(
        uniqueness=uniqueness,
        people=people_score,
        emotion=emotion_score,
        visual_quality=50,
        landmark=0,
        unusual_event=35,
    )
    return SceneUnderstanding(
        caption=caption[:500] or "Travel scene",
        detailed_description=caption[:1500] or "Travel scene.",
        location_type=location,
        activity=activity,
        emotion=emotion,
        people_count=people_count,
        people_groups=people_groups,
        landmarks=[],
        vision_score=(
            uniqueness * 0.3 + people_score * 0.15 + emotion_score * 0.25 + 50 * 0.2 + 35 * 0.1
        ),
        score_factors=factors,
        story_relevance=f"Potential {style.value} travel-story scene.",
        tags=[
            value
            for value in (location.value, activity.value, emotion.value)
            if value not in {"unknown", "neutral"}
        ],
    )


def _keyword_enum(
    text: str,
    candidates: dict[Any, tuple[str, ...]],
    default: Any,
) -> Any:
    return next(
        (
            value
            for value, keywords in candidates.items()
            if any(keyword in text for keyword in keywords)
        ),
        default,
    )


def _people_from_caption(text: str) -> tuple[list[PersonGroup], int]:
    if "family" in text:
        return [PersonGroup.FAMILY, PersonGroup.MIXED], 3
    if any(word in text for word in ("children", "kids", "child")):
        return [PersonGroup.CHILDREN], 2
    if any(word in text for word in ("group", "crowd", "people")):
        return [PersonGroup.GROUP], 3
    if any(word in text for word in ("person", "man", "woman", "traveler")):
        return [PersonGroup.SOLO], 1
    return [PersonGroup.NONE], 0
