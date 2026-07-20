"""OpenCV-based visual quality metrics with a Pillow fallback."""

import importlib
import math
import re
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from PIL import Image, ImageChops, ImageFilter, ImageStat

from travelmovieai.analysis.scenes import sample_positions_for_count
from travelmovieai.domain.models import (
    QualityAnalysisReport,
    Scene,
    TemporalHighlightWindow,
    VisualQualityMetrics,
)

QUALITY_ALGORITHM_VERSION = "visual-quality-v3-temporal-grid"


class QualityAnalyzer(Protocol):
    def analyze(self, image_path: Path) -> VisualQualityMetrics: ...


@dataclass(frozen=True, slots=True)
class QualityBackend:
    name: Literal["torch-cuda", "opencv", "pillow"]
    device: Literal["cuda", "cpu"]
    library_version: str

    def fingerprint_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "device": self.device,
            "library_version": self.library_version,
        }


class VisualQualityAnalyzer:
    def analyze(self, image_path: Path) -> VisualQualityMetrics:
        return self.analyze_contact_sheet(image_path)

    def analyze_contact_sheet(
        self,
        image_path: Path,
        sample_positions: Sequence[float] | None = None,
    ) -> VisualQualityMetrics:
        try:
            cv2: Any = importlib.import_module("cv2")
        except ImportError:
            return self._analyze_pillow(image_path, sample_positions)

        image = cv2.imread(str(image_path))
        if image is None:
            return self._analyze_pillow(image_path, sample_positions)
        positions = _resolve_sample_positions(
            image_path,
            image.shape[1],
            image.shape[0],
            sample_positions,
        )
        panels = _split_contact_sheet_cv(image, len(positions))
        panel_metrics = [_opencv_panel_metrics(cv2, panel) for panel in panels]
        panel_scores = [_panel_quality_score(*item) for item in panel_metrics]
        panel_details = _panel_details(panel_metrics, panel_scores, positions)
        brightness = _average(item[0] for item in panel_metrics)
        contrast = _average(item[1] for item in panel_metrics)
        sharpness = _average(item[2] for item in panel_metrics)
        saturation = _average(item[3] for item in panel_metrics)
        colorfulness = _average(item[4] for item in panel_metrics)
        noise_score = _average(item[5] for item in panel_metrics)
        motion_score, camera_shake_score = _opencv_temporal_metrics(cv2, panels)
        return _metrics(
            brightness,
            contrast,
            sharpness,
            saturation,
            colorfulness,
            noise_score,
            motion_score,
            camera_shake_score,
            "opencv",
            panel_scores,
            panel_details,
            positions,
        )

    def _analyze_pillow(
        self,
        image_path: Path,
        sample_positions: Sequence[float] | None = None,
    ) -> VisualQualityMetrics:
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            positions = _resolve_sample_positions(
                image_path,
                rgb.width,
                rgb.height,
                sample_positions,
            )
            panels = _split_contact_sheet_pillow(rgb, len(positions))
            panel_metrics = [_pillow_panel_metrics(panel) for panel in panels]
            panel_scores = [_panel_quality_score(*item) for item in panel_metrics]
            panel_details = _panel_details(panel_metrics, panel_scores, positions)
            brightness = _average(item[0] for item in panel_metrics)
            contrast = _average(item[1] for item in panel_metrics)
            sharpness = _average(item[2] for item in panel_metrics)
            saturation = _average(item[3] for item in panel_metrics)
            colorfulness = _average(item[4] for item in panel_metrics)
            noise_score = _average(item[5] for item in panel_metrics)
            motion_score, camera_shake_score = _pillow_temporal_metrics(panels)
        return _metrics(
            brightness,
            contrast,
            sharpness,
            saturation,
            colorfulness,
            noise_score,
            motion_score,
            camera_shake_score,
            "pillow",
            panel_scores,
            panel_details,
            positions,
        )


class TorchCudaQualityAnalyzer:
    """Compute dense frame metrics on CUDA when OpenCV lacks CUDA support."""

    def __init__(self) -> None:
        self._torch: Any = importlib.import_module("torch")
        self._functional: Any = importlib.import_module("torch.nn.functional")

    def analyze(self, image_path: Path) -> VisualQualityMetrics:
        return self.analyze_contact_sheet(image_path)

    def analyze_contact_sheet(
        self,
        image_path: Path,
        sample_positions: Sequence[float] | None = None,
    ) -> VisualQualityMetrics:
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            positions = _resolve_sample_positions(
                image_path,
                rgb.width,
                rgb.height,
                sample_positions,
            )
            np: Any = importlib.import_module("numpy")
            array = np.asarray(rgb, dtype="float32") / 255.0
        tensor = (
            self._torch.from_numpy(array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to("cuda", non_blocking=True)
        )
        panels = _split_contact_sheet_tensor(tensor, len(positions))
        static = [self._panel_metrics(panel) for panel in panels]
        panel_scores = [_panel_quality_score(*item) for item in static]
        panel_details = _panel_details(static, panel_scores, positions)
        brightness = _average(item[0] for item in static)
        contrast = _average(item[1] for item in static)
        sharpness = _average(item[2] for item in static)
        saturation = _average(item[3] for item in static)
        colorfulness = _average(item[4] for item in static)
        noise_score = _average(item[5] for item in static)
        motion_score, camera_shake_score = self._temporal_metrics(panels)
        return _metrics(
            brightness,
            contrast,
            sharpness,
            saturation,
            colorfulness,
            noise_score,
            motion_score,
            camera_shake_score,
            "torch-cuda",
            panel_scores,
            panel_details,
            positions,
        )

    def _panel_metrics(
        self,
        panel: Any,
    ) -> tuple[float, float, float, float, float, float]:
        red, green, blue = panel[:, 0:1], panel[:, 1:2], panel[:, 2:3]
        gray = red * 0.299 + green * 0.587 + blue * 0.114
        brightness = _clamp(float(gray.mean().item()) * 100)
        contrast = _clamp(float(gray.std().item()) / (96 / 255) * 100)
        kernel = self._torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            device=gray.device,
            dtype=gray.dtype,
        ).reshape(1, 1, 3, 3)
        laplacian = self._functional.conv2d(gray, kernel, padding=1)
        sharpness = _log_score(float(laplacian.var().item()) * 255**2, 1800)
        maximum = panel.max(dim=1, keepdim=True).values
        minimum = panel.min(dim=1, keepdim=True).values
        saturation = _clamp(
            float(((maximum - minimum) / maximum.clamp_min(1e-4)).mean().item()) * 100
        )
        red_green = red - green
        yellow_blue = 0.5 * (red + green) - blue
        colorfulness_raw = math.sqrt(
            float(red_green.std().item() * 255) ** 2 + float(yellow_blue.std().item() * 255) ** 2
        ) + 0.3 * math.sqrt(
            float(red_green.mean().item() * 255) ** 2 + float(yellow_blue.mean().item() * 255) ** 2
        )
        colorfulness = _clamp(colorfulness_raw / 90 * 100)
        blurred = self._functional.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        noise_score = _clamp(float((gray - blurred).std().item() * 255) / 22 * 100)
        return brightness, contrast, sharpness, saturation, colorfulness, noise_score

    def _temporal_metrics(self, panels: list[Any]) -> tuple[float, float]:
        if len(panels) < 2:
            return 0, 0
        motion_values = [
            _clamp(float((second - first).abs().mean().item()) * 180)
            for first, second in zip(panels, panels[1:], strict=False)
        ]
        edge_values: list[float] = []
        for first, second in zip(panels, panels[1:], strict=False):
            first_gray = first.mean(dim=1, keepdim=True)
            second_gray = second.mean(dim=1, keepdim=True)
            first_dy = first_gray[:, :, 1:, :] - first_gray[:, :, :-1, :]
            second_dy = second_gray[:, :, 1:, :] - second_gray[:, :, :-1, :]
            first_dx = first_gray[:, :, :, 1:] - first_gray[:, :, :, :-1]
            second_dx = second_gray[:, :, :, 1:] - second_gray[:, :, :, :-1]
            edge_delta = (
                (second_dx - first_dx).abs().mean() + (second_dy - first_dy).abs().mean()
            ) / 2
            edge_values.append(_clamp(float(edge_delta.item()) * 500))
        return _average(motion_values), _average(edge_values)


def analyze_scene_quality(
    scenes: list[Scene],
    analyzer: QualityAnalyzer | None = None,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> QualityAnalysisReport:
    resolved_analyzer = analyzer or _default_quality_analyzer()
    backend_label = (
        "CUDA quality" if isinstance(resolved_analyzer, TorchCudaQualityAnalyzer) else "OpenCV"
    )
    if isinstance(resolved_analyzer, TorchCudaQualityAnalyzer):
        workers = 1
    if workers <= 1 or len(scenes) <= 1:
        analyzed = []
        for index, scene in enumerate(scenes, start=1):
            analyzed.append(_analyze_scene_quality(scene, resolved_analyzer))
            if progress:
                progress(index, len(scenes), f"{backend_label}: scene {index}/{len(scenes)}")
    else:
        analyzed_by_index: dict[int, Scene] = {}
        worker_count = min(workers, len(scenes))
        if progress:
            progress(0, len(scenes), f"{backend_label}: scene 0/{len(scenes)}")
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="travelmovieai-quality",
        )
        futures: dict[Future[Scene], int] = {}
        scene_iterator = iter(enumerate(scenes))

        def submit_next() -> bool:
            try:
                index, scene = next(scene_iterator)
            except StopIteration:
                return False
            futures[executor.submit(_analyze_scene_quality, scene, resolved_analyzer)] = index
            return True

        completed = 0
        try:
            for _ in range(worker_count):
                if not submit_next():
                    break
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    analyzed_by_index[futures.pop(future)] = future.result()
                    completed += 1
                    if progress:
                        progress(
                            completed,
                            len(scenes),
                            f"{backend_label}: scene {completed}/{len(scenes)}, workers={workers}",
                        )
                    submit_next()
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        analyzed = [analyzed_by_index[index] for index in range(len(scenes))]
    return QualityAnalysisReport(
        created_at=datetime.now(UTC),
        scenes=analyzed,
    )


def _default_quality_analyzer() -> QualityAnalyzer:
    return create_quality_analyzer(resolve_quality_backend("auto"))


def resolve_quality_backend(device: str) -> QualityBackend:
    if device not in {"auto", "cuda", "directml", "cpu"}:
        raise ValueError(f"Unsupported quality analysis device: {device}")
    if device in {"auto", "cuda"}:
        try:
            torch: Any = importlib.import_module("torch")
            if torch.cuda.is_available():
                return QualityBackend(
                    name="torch-cuda",
                    device="cuda",
                    library_version=str(getattr(torch, "__version__", "unknown")),
                )
        except (ImportError, RuntimeError):
            pass
    try:
        cv2: Any = importlib.import_module("cv2")
    except ImportError:
        return QualityBackend(
            name="pillow",
            device="cpu",
            library_version=str(getattr(Image, "__version__", "unknown")),
        )
    return QualityBackend(
        name="opencv",
        device="cpu",
        library_version=str(getattr(cv2, "__version__", "unknown")),
    )


def create_quality_analyzer(backend: QualityBackend) -> QualityAnalyzer:
    if backend.name == "torch-cuda":
        return TorchCudaQualityAnalyzer()
    return VisualQualityAnalyzer()


def _analyze_scene_quality(
    scene: Scene,
    analyzer: QualityAnalyzer,
) -> Scene:
    if scene.keyframe_path is None:
        return scene
    sample_positions = _scene_sample_positions(scene)
    analyze_contact_sheet = getattr(analyzer, "analyze_contact_sheet", None)
    metrics = (
        analyze_contact_sheet(scene.keyframe_path, sample_positions)
        if callable(analyze_contact_sheet)
        else analyzer.analyze(scene.keyframe_path)
    )
    return scene.model_copy(
        update={
            "quality_score": metrics.quality_score,
            "metadata": {
                **scene.metadata,
                "quality_metrics": metrics.model_dump(),
                "technical_rejection_reasons": metrics.rejection_reasons,
            },
        }
    )


def _metrics(
    brightness: float,
    contrast: float,
    sharpness: float,
    saturation: float,
    colorfulness: float,
    noise_score: float,
    motion_score: float,
    camera_shake_score: float,
    backend: str,
    panel_quality_scores: list[float] | None = None,
    panel_details: list[dict[str, float | int]] | None = None,
    sample_positions: Sequence[float] | None = None,
) -> VisualQualityMetrics:
    exposure = _clamp(100 - abs(brightness - 52) * 2.3)
    saturation_quality = _clamp(100 - abs(saturation - 45) * 1.2)
    stability = 100 - camera_shake_score
    noise_quality = 100 - noise_score
    score = _quality_score_from_components(
        sharpness=sharpness,
        contrast=contrast,
        exposure=exposure,
        saturation_quality=saturation_quality,
        colorfulness=colorfulness,
        stability=stability,
        noise_quality=noise_quality,
    )
    rejection_reasons = _rejection_reasons(
        brightness,
        contrast,
        sharpness,
        noise_score,
        camera_shake_score,
        score,
    )
    panel_scores = panel_quality_scores or []
    positions = list(sample_positions or sample_positions_for_count(len(panel_scores) or 1))
    best_index = _best_panel_index(panel_scores)
    panel_position = _panel_position(best_index, positions)
    return VisualQualityMetrics(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        saturation=saturation,
        colorfulness=colorfulness,
        exposure_score=exposure,
        noise_score=noise_score,
        motion_score=motion_score,
        camera_shake_score=camera_shake_score,
        quality_score=score,
        panel_quality_scores=panel_scores,
        best_panel_index=best_index,
        best_panel_position=panel_position,
        sample_count=len(positions),
        sample_positions=positions,
        panel_details=panel_details or [],
        candidate_windows=_quality_candidate_windows(panel_scores, positions),
        rejection_reasons=rejection_reasons,
        backend=backend,
    )


def _panel_details(
    panel_metrics: list[tuple[float, float, float, float, float, float]],
    panel_scores: list[float],
    sample_positions: Sequence[float],
) -> list[dict[str, float | int]]:
    return [
        {
            "index": index,
            "position": _panel_position(index, sample_positions) or 0.5,
            "score": score,
            "brightness": metrics[0],
            "contrast": metrics[1],
            "sharpness": metrics[2],
            "saturation": metrics[3],
            "colorfulness": metrics[4],
            "noise_score": metrics[5],
        }
        for index, (metrics, score) in enumerate(zip(panel_metrics, panel_scores, strict=True))
    ]


def _quality_candidate_windows(
    panel_scores: list[float],
    sample_positions: Sequence[float],
) -> list[TemporalHighlightWindow]:
    return [
        TemporalHighlightWindow(
            relative_start=_sample_window_bounds(sample_positions, index)[0],
            relative_end=_sample_window_bounds(sample_positions, index)[1],
            relative_position=_panel_position(index, sample_positions) or 0.5,
            confidence=min(1.0, max(0.0, score / 100)),
            score=score,
            source="visual_quality",
            label=f"visual panel {index + 1}/{len(panel_scores)}",
        )
        for index, score in enumerate(panel_scores)
    ]


def _panel_quality_score(
    brightness: float,
    contrast: float,
    sharpness: float,
    saturation: float,
    colorfulness: float,
    noise_score: float,
) -> float:
    exposure = _clamp(100 - abs(brightness - 52) * 2.3)
    saturation_quality = _clamp(100 - abs(saturation - 45) * 1.2)
    return _quality_score_from_components(
        sharpness=sharpness,
        contrast=contrast,
        exposure=exposure,
        saturation_quality=saturation_quality,
        colorfulness=colorfulness,
        stability=85,
        noise_quality=100 - noise_score,
    )


def _quality_score_from_components(
    *,
    sharpness: float,
    contrast: float,
    exposure: float,
    saturation_quality: float,
    colorfulness: float,
    stability: float,
    noise_quality: float,
) -> float:
    return _clamp(
        sharpness * 0.28
        + contrast * 0.15
        + exposure * 0.2
        + saturation_quality * 0.08
        + colorfulness * 0.07
        + stability * 0.14
        + noise_quality * 0.08
    )


def _best_panel_index(scores: list[float]) -> int | None:
    if not scores:
        return None
    return max(range(len(scores)), key=lambda index: scores[index])


def _sample_window_bounds(
    positions: Sequence[float],
    index: int,
) -> tuple[float, float]:
    position = positions[index]
    start = 0.0 if index == 0 else (positions[index - 1] + position) / 2
    end = 1.0 if index == len(positions) - 1 else (position + positions[index + 1]) / 2
    if end - start < 1e-6:
        start = max(0.0, position - 0.05)
        end = min(1.0, position + 0.05)
        if end - start < 1e-6:
            start, end = (0.0, 0.1) if position <= 0 else (0.9, 1.0)
    return max(0.0, start), min(1.0, end)


def _panel_position(index: int | None, positions: Sequence[float]) -> float | None:
    if index is None or not positions or index >= len(positions):
        return None
    return positions[index]


def _opencv_panel_metrics(
    cv2: Any,
    image: Any,
) -> tuple[float, float, float, float, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = _clamp(float(gray.mean()) / 255 * 100)
    contrast = _clamp(float(gray.std()) / 96 * 100)
    sharpness = _log_score(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1800)
    saturation = _clamp(float(hsv[:, :, 1].mean()) / 255 * 100)
    blue, green, red = cv2.split(image.astype("float32"))
    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue
    colorfulness_raw = math.sqrt(
        float(red_green.std()) ** 2 + float(yellow_blue.std()) ** 2
    ) + 0.3 * math.sqrt(float(red_green.mean()) ** 2 + float(yellow_blue.mean()) ** 2)
    colorfulness = _clamp(colorfulness_raw / 90 * 100)
    residual = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))
    noise_score = _clamp(float(residual.std()) / 22 * 100)
    return brightness, contrast, sharpness, saturation, colorfulness, noise_score


def _opencv_temporal_metrics(cv2: Any, panels: list[Any]) -> tuple[float, float]:
    if len(panels) < 2:
        return 0, 0
    try:
        np: Any = importlib.import_module("numpy")
    except ImportError:
        return 0, 0
    motion_values: list[float] = []
    shake_values: list[float] = []
    directions: list[Any] = []
    grays = [cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY) for panel in panels]
    for first, second in zip(grays, grays[1:], strict=False):
        frame_delta = float(cv2.absdiff(first, second).mean())
        first_edges = cv2.Laplacian(first, cv2.CV_32F)
        second_edges = cv2.Laplacian(second, cv2.CV_32F)
        edge_delta = float(np.abs(second_edges - first_edges).mean())
        motion_value = _clamp(frame_delta / 255 * 180)
        shake_value = _clamp(edge_delta / 64 * 100)
        points = cv2.goodFeaturesToTrack(
            first,
            maxCorners=180,
            qualityLevel=0.01,
            minDistance=8,
        )
        if points is None or len(points) < 8:
            motion_values.append(motion_value)
            shake_values.append(shake_value)
            continue
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None)
        if tracked is None or status is None:
            motion_values.append(motion_value)
            shake_values.append(shake_value)
            continue
        valid = status.reshape(-1) == 1
        flow = tracked.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
        if len(flow) < 8:
            motion_values.append(motion_value)
            shake_values.append(shake_value)
            continue
        median = np.median(flow, axis=0)
        directions.append(median)
        residual = np.linalg.norm(flow - median, axis=1)
        diagonal = math.hypot(first.shape[1], first.shape[0])
        motion_values.append(
            max(motion_value, _clamp(float(np.linalg.norm(median)) / diagonal * 900))
        )
        shake_values.append(max(shake_value, _clamp(float(np.percentile(residual, 75)) / 10 * 100)))
    for first_direction, second_direction in zip(
        directions,
        directions[1:],
        strict=False,
    ):
        direction_change = float(np.linalg.norm(second_direction - first_direction))
        shake_values.append(_clamp(direction_change / 12 * 100))
    return _average(motion_values), _average(shake_values)


def _pillow_panel_metrics(
    rgb: Image.Image,
) -> tuple[float, float, float, float, float, float]:
    grayscale = rgb.convert("L")
    gray_stats = ImageStat.Stat(grayscale)
    brightness = _clamp(gray_stats.mean[0] / 255 * 100)
    contrast = _clamp(gray_stats.stddev[0] / 96 * 100)
    edge_stats = ImageStat.Stat(grayscale.filter(ImageFilter.FIND_EDGES))
    sharpness = _log_score(edge_stats.var[0], 4500)
    rgb_stats = ImageStat.Stat(rgb)
    colorfulness = _clamp(sum(rgb_stats.stddev) / 3 / 80 * 100)
    extrema = rgb_stats.extrema
    saturation = _clamp(sum(high - low for low, high in extrema) / (3 * 255) * 100)
    residual = ImageChops.difference(grayscale, grayscale.filter(ImageFilter.GaussianBlur(2)))
    noise_score = _clamp(ImageStat.Stat(residual).stddev[0] / 22 * 100)
    return brightness, contrast, sharpness, saturation, colorfulness, noise_score


def _pillow_temporal_metrics(panels: list[Image.Image]) -> tuple[float, float]:
    if len(panels) < 2:
        return 0, 0
    motion_values = []
    shake_values = []
    for first, second in zip(panels, panels[1:], strict=False):
        first_gray = first.convert("L")
        second_gray = second.convert("L")
        difference = ImageChops.difference(first_gray, second_gray)
        motion_values.append(_clamp(ImageStat.Stat(difference).mean[0] / 255 * 180))
        first_edges = first_gray.filter(ImageFilter.FIND_EDGES)
        second_edges = second_gray.filter(ImageFilter.FIND_EDGES)
        edge_difference = ImageChops.difference(first_edges, second_edges)
        shake_values.append(_clamp(ImageStat.Stat(edge_difference).mean[0] / 255 * 240))
    return _average(motion_values), _average(shake_values)


def _split_contact_sheet_cv(image: Any, sample_count: int | None = None) -> list[Any]:
    height, width = image.shape[:2]
    count = _resolved_grid_count(width, height, sample_count)
    if count == 1:
        return [image]
    columns = min(3, count)
    rows = math.ceil(count / columns)
    panel_width = width // columns
    panel_height = height // rows
    return [
        image[
            (index // columns) * panel_height : (index // columns + 1) * panel_height,
            (index % columns) * panel_width : (index % columns + 1) * panel_width,
        ]
        for index in range(count)
    ]


def _split_contact_sheet_pillow(
    image: Image.Image,
    sample_count: int | None = None,
) -> list[Image.Image]:
    width, height = image.size
    count = _resolved_grid_count(width, height, sample_count)
    if count == 1:
        return [image]
    columns = min(3, count)
    rows = math.ceil(count / columns)
    panel_width = width // columns
    panel_height = height // rows
    return [
        image.crop(
            (
                (index % columns) * panel_width,
                (index // columns) * panel_height,
                (index % columns + 1) * panel_width,
                (index // columns + 1) * panel_height,
            )
        )
        for index in range(count)
    ]


def _split_contact_sheet_tensor(tensor: Any, sample_count: int) -> list[Any]:
    count = _resolved_grid_count(int(tensor.shape[3]), int(tensor.shape[2]), sample_count)
    if count == 1:
        return [tensor]
    columns = min(3, count)
    rows = math.ceil(count / columns)
    panel_width = int(tensor.shape[3]) // columns
    panel_height = int(tensor.shape[2]) // rows
    return [
        tensor[
            :,
            :,
            (index // columns) * panel_height : (index // columns + 1) * panel_height,
            (index % columns) * panel_width : (index % columns + 1) * panel_width,
        ]
        for index in range(count)
    ]


def _resolved_grid_count(width: int, height: int, sample_count: int | None) -> int:
    if sample_count in {1, 3, 5, 9}:
        return sample_count
    return 3 if width >= height * 2.2 else 1


def _resolve_sample_positions(
    image_path: Path,
    width: int,
    height: int,
    supplied: Sequence[float] | None,
) -> tuple[float, ...]:
    if supplied:
        normalized = tuple(float(position) for position in supplied)
        if (
            len(normalized) in {1, 3, 5, 9}
            and all(0 <= item <= 1 for item in normalized)
            and all(
                second >= first for first, second in zip(normalized, normalized[1:], strict=False)
            )
        ):
            return normalized
    match = re.search(r"-contact-v\d+-(3|5|9)(?:-|\.)", image_path.name)
    count = int(match.group(1)) if match else _resolved_grid_count(width, height, None)
    return sample_positions_for_count(count)


def _scene_sample_positions(scene: Scene) -> tuple[float, ...] | None:
    contact_sheet = scene.metadata.get("contact_sheet")
    if not isinstance(contact_sheet, dict):
        return None
    raw = contact_sheet.get("sample_positions")
    if not isinstance(raw, list):
        return None
    try:
        positions = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if len(positions) not in {1, 3, 5, 9}:
        return None
    if any(position < 0 or position > 1 for position in positions):
        return None
    if any(second < first for first, second in zip(positions, positions[1:], strict=False)):
        return None
    return positions


def _rejection_reasons(
    brightness: float,
    contrast: float,
    sharpness: float,
    noise_score: float,
    camera_shake_score: float,
    quality_score: float,
) -> list[str]:
    reasons: list[str] = []
    if sharpness < 24:
        reasons.append("blurred")
    if brightness < 12:
        reasons.append("too_dark")
    elif brightness > 90:
        reasons.append("overexposed")
    if contrast < 12:
        reasons.append("low_contrast")
    if noise_score > 72:
        reasons.append("noisy")
    if camera_shake_score > 72:
        reasons.append("camera_shake")
    if quality_score < 22 and not reasons:
        reasons.append("low_technical_quality")
    return reasons


def _log_score(value: float, reference: float) -> float:
    return _clamp(math.log1p(max(0, value)) / math.log1p(reference) * 100)


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _average(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
