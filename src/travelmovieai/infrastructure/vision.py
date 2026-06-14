"""Local vision provider adapters."""

from __future__ import annotations

import base64
import importlib
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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

PROMPT_VERSION = "scene-understanding-v3-stage-4.5"
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
people_count, people_groups (none|adults|children|family|group|solo|mixed),
landmarks [{name, confidence 0-1, evidence}],
vision_score 0-100,
score_factors {uniqueness, people, emotion, visual_quality, landmark,
unusual_event}, story_relevance, tags.
""".strip()


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
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.device = device
        self.cache_dir = cache_dir
        self.allow_download = allow_download
        self.quantize_4bit = quantize_4bit
        self.batch_size = max(1, batch_size)
        self._processor: Any = None
        self._loaded_model: Any = None
        self._torch: Any = None

    @property
    def runtime_description(self) -> str:
        if self._loaded_model is None:
            return "не загружена"
        device = str(self._loaded_model.device)
        precision = "4-bit NF4" if self.quantize_4bit else "native precision"
        return f"{device}, {precision}"

    def prepare(self) -> None:
        self._ensure_loaded()

    def analyze(self, image_path: Path, style: StoryStyle) -> SceneUnderstanding:
        return self.analyze_batch([image_path], style)[0]

    def analyze_batch(
        self,
        image_paths: list[Path],
        style: StoryStyle,
    ) -> list[SceneUnderstanding]:
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
                "Локальная Qwen-модель вернула ответ, который не удалось привести "
                "к схеме анализа сцены. Повторите сцену или выберите другую модель."
            ) from error
        except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
            raise VisionAnalysisError(
                f"Локальная Qwen-модель не смогла проанализировать {image_path.name}. "
                "Проверьте свободную RAM/VRAM и при необходимости выберите модель 3B."
            ) from error

    def _generate_contents(
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
        try:
            self._torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            importlib.import_module("accelerate")
            if self.quantize_4bit:
                importlib.import_module("bitsandbytes")
        except ImportError as error:
            raise DependencyUnavailableError(
                "Для локальной Qwen Vision установите optional-группу vision: "
                'python -m pip install -e ".[vision]".'
            ) from error

        if self.device == "cuda" and not self._torch.cuda.is_available():
            raise DependencyUnavailableError(
                "Выбран CUDA, но установленная сборка PyTorch не видит видеокарту."
            )
        resolved_device = (
            "cuda" if self.device in {"auto", "cuda"} and self._torch.cuda.is_available() else "cpu"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            if resolved_device == "cuda":
                self._torch.backends.cuda.matmul.allow_tf32 = True
            processor_type = transformers.AutoProcessor
            model_type = transformers.Qwen2_5_VLForConditionalGeneration
            self._processor = processor_type.from_pretrained(
                self.model,
                cache_dir=self.cache_dir,
                local_files_only=not self.allow_download,
                min_pixels=256 * 28 * 28,
                max_pixels=640 * 28 * 28,
            )
            model_options: dict[str, Any] = {
                "cache_dir": self.cache_dir,
                "local_files_only": not self.allow_download,
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",
            }
            if resolved_device == "cuda" and self.quantize_4bit:
                model_options.update(
                    {
                        "dtype": self._torch.float16,
                        "device_map": {"": 0},
                        "quantization_config": transformers.BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=self._torch.float16,
                            bnb_4bit_use_double_quant=True,
                        ),
                    }
                )
            else:
                model_options.update(
                    {
                        "dtype": "auto",
                        "device_map": "auto" if resolved_device == "cuda" else "cpu",
                    }
                )
            self._loaded_model = model_type.from_pretrained(
                self.model,
                **model_options,
            )
            self._loaded_model.eval()
        except (AttributeError, OSError, RuntimeError, ValueError) as error:
            download_hint = (
                "Проверьте интернет и свободное место в кэше моделей."
                if self.allow_download
                else "Автозагрузка отключена, а модель отсутствует в локальном кэше."
            )
            raise VisionAnalysisError(
                f"Не удалось загрузить Qwen Vision '{self.model}'. {download_hint}"
            ) from error


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
                    "You are the Vision AI module of a travel film editor. Analyze "
                    "only visible evidence across the chronological frames. Return "
                    "JSON matching the schema exactly. Use only enum values from the "
                    "schema. Do not identify unknown people or invent landmarks. A "
                    "landmark requires visible architectural or textual evidence; "
                    "otherwise return an empty landmarks list."
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
                            "Describe what changes across them. vision_score and each "
                            "score factor use 0-100. Evaluate uniqueness, people, "
                            "emotion, landmark value, and unusual_event. Set "
                            "visual_quality to 50 because the application replaces it "
                            "with measured OpenCV quality. "
                            f"{COMPACT_OUTPUT_CONTRACT}"
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
            "max_tokens": 900,
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
        except TimeoutError as error:
            raise VisionAnalysisError(
                "Vision-модель в LM Studio не завершила анализ за "
                f"{self.timeout_seconds:.0f} с. Увеличьте "
                "TRAVELMOVIEAI_VISION_TIMEOUT_SECONDS, уменьшите модель или "
                "проверьте GPU offload в LM Studio."
            ) from error
        except URLError as error:
            raise VisionAnalysisError(
                "LM Studio недоступен. Запустите локальный сервер, загрузите "
                f"vision-модель '{self.model}' и проверьте {self.base_url}."
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise VisionAnalysisError("Не удалось прочитать ответ LM Studio.") from error


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
    ) -> None:
        self.model = model
        self.device = device
        self.cache_dir = cache_dir
        self.allow_download = allow_download
        self._processor: Any = None
        self._loaded_model: Any = None
        self._torch: Any = None

    def analyze(self, image_path: Path, style: StoryStyle) -> SceneUnderstanding:
        self._ensure_loaded()
        try:
            with Image.open(image_path) as source:
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
            raise VisionAnalysisError(
                f"Florence-2 не смогла проанализировать {image_path.name}."
            ) from error
        caption = str(parsed.get(task, generated)).strip()
        return _understanding_from_caption(caption, style)

    def _ensure_loaded(self) -> None:
        if self._loaded_model is not None:
            return
        try:
            self._torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise DependencyUnavailableError(
                "Для Florence-2 установите optional-группу vision и совместимую "
                "с вашей системой сборку PyTorch."
            ) from error
        device = self._resolved_device()
        dtype = self._torch.float16 if device == "cuda" else self._torch.float32
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            processor_type = transformers.AutoProcessor
            model_type = transformers.AutoModelForCausalLM
            self._processor = processor_type.from_pretrained(
                self.model,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                local_files_only=not self.allow_download,
            )
            self._loaded_model = model_type.from_pretrained(
                self.model,
                trust_remote_code=True,
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
                local_files_only=not self.allow_download,
            ).to(device)
            self._loaded_model.eval()
        except (OSError, RuntimeError, ValueError) as error:
            raise VisionAnalysisError(
                f"Не удалось загрузить локальную Florence-2 модель '{self.model}'. "
                "Проверьте интернет, свободное место и настройки автозагрузки."
            ) from error

    def _resolved_device(self) -> str:
        if self.device == "cuda":
            return "cuda"
        if self.device == "cpu":
            return "cpu"
        if self._torch is not None and self._torch.cuda.is_available():
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


def build_vision_provider(
    *,
    provider: str,
    model: str | None,
    device: str,
    cache_dir: Path,
    allow_download: bool,
    gpu_memory_mb: int | None,
    system_memory_mb: int | None,
    lm_studio_url: str,
    lm_studio_api_key: str | None,
    timeout_seconds: float,
    model_batch_size: int = 1,
) -> LocalQwenVisionProvider | LMStudioVisionProvider | Florence2VisionProvider:
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
        )
    if provider == "lm-studio":
        from travelmovieai.infrastructure.lm_studio import (
            list_lm_studio_models,
            resolve_vision_model,
        )

        discovered = list_lm_studio_models(
            lm_studio_url,
            lm_studio_api_key,
            5,
        )
        resolved_model = resolve_vision_model(discovered, configured_model)
        return LMStudioVisionProvider(
            base_url=lm_studio_url,
            model=resolved_model,
            timeout_seconds=timeout_seconds,
            api_key=lm_studio_api_key,
        )
    return LocalQwenVisionProvider(
        resolve_local_vision_model(
            configured_model,
            gpu_memory_mb=gpu_memory_mb,
            system_memory_mb=system_memory_mb,
        ),
        device=device,
        cache_dir=cache_dir,
        allow_download=allow_download,
        quantize_4bit=(
            device in {"auto", "cuda"}
            and gpu_memory_mb is not None
            and 0 < gpu_memory_mb < 10 * 1024
        ),
        batch_size=(
            min(2, model_batch_size)
            if (gpu_memory_mb or 0) < 10 * 1024
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

    payload.update(
        {
            "caption": str(payload.get("caption") or "Travel scene")[:500],
            "detailed_description": str(
                payload.get("detailed_description")
                or payload.get("caption")
                or "Travel scene."
            )[:1500],
            "people_groups": groups,
            "landmarks": landmarks,
            "vision_score": _score(vision_score, 50),
            "score_factors": normalized_factors,
            "story_relevance": str(relevance)[:500],
            "tags": [str(tag)[:100] for tag in tags[:20]],
        }
    )
    return SceneUnderstanding.model_validate(payload)


def _score(value: Any, default: float) -> float:
    try:
        return min(100.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


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
