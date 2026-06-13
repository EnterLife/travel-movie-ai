"""OpenCV-based visual quality metrics with a Pillow fallback."""

import importlib
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageStat

from travelmovieai.domain.models import (
    QualityAnalysisReport,
    Scene,
    VisualQualityMetrics,
)


class VisualQualityAnalyzer:
    def analyze(self, image_path: Path) -> VisualQualityMetrics:
        try:
            cv2: Any = importlib.import_module("cv2")
        except ImportError:
            return self._analyze_pillow(image_path)

        image = cv2.imread(str(image_path))
        if image is None:
            return self._analyze_pillow(image_path)
        panels = _split_contact_sheet_cv(image)
        panel_metrics = [_opencv_panel_metrics(cv2, panel) for panel in panels]
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
        )

    def _analyze_pillow(self, image_path: Path) -> VisualQualityMetrics:
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            panels = _split_contact_sheet_pillow(rgb)
            panel_metrics = [_pillow_panel_metrics(panel) for panel in panels]
            brightness = _average(item[0] for item in panel_metrics)
            contrast = _average(item[1] for item in panel_metrics)
            sharpness = _average(item[2] for item in panel_metrics)
            saturation = _average(item[3] for item in panel_metrics)
            colorfulness = _average(item[4] for item in panel_metrics)
            noise_score = _average(item[5] for item in panel_metrics)
            motion_score = _pillow_motion_score(panels)
        return _metrics(
            brightness,
            contrast,
            sharpness,
            saturation,
            colorfulness,
            noise_score,
            motion_score,
            0,
            "pillow",
        )


def analyze_scene_quality(
    scenes: list[Scene],
    analyzer: VisualQualityAnalyzer | None = None,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> QualityAnalysisReport:
    resolved_analyzer = analyzer or VisualQualityAnalyzer()
    if workers <= 1 or len(scenes) <= 1:
        analyzed = []
        for index, scene in enumerate(scenes, start=1):
            analyzed.append(_analyze_scene_quality(scene, resolved_analyzer))
            if progress:
                progress(index, len(scenes), f"OpenCV: сцена {index}/{len(scenes)}")
    else:
        analyzed_by_index: dict[int, Scene] = {}
        with ThreadPoolExecutor(
            max_workers=min(workers, len(scenes)),
            thread_name_prefix="travelmovieai-quality",
        ) as executor:
            futures = {
                executor.submit(_analyze_scene_quality, scene, resolved_analyzer): index
                for index, scene in enumerate(scenes)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                analyzed_by_index[futures[future]] = future.result()
                if progress:
                    progress(
                        completed,
                        len(scenes),
                        f"OpenCV: сцена {completed}/{len(scenes)}, workers={workers}",
                    )
        analyzed = [analyzed_by_index[index] for index in range(len(scenes))]
    return QualityAnalysisReport(
        created_at=datetime.now(UTC),
        scenes=analyzed,
    )


def _analyze_scene_quality(
    scene: Scene,
    analyzer: VisualQualityAnalyzer,
) -> Scene:
    if scene.keyframe_path is None:
        return scene
    metrics = analyzer.analyze(scene.keyframe_path)
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
) -> VisualQualityMetrics:
    exposure = _clamp(100 - abs(brightness - 52) * 2.3)
    saturation_quality = _clamp(100 - abs(saturation - 45) * 1.2)
    stability = 100 - camera_shake_score
    noise_quality = 100 - noise_score
    score = _clamp(
        sharpness * 0.28
        + contrast * 0.15
        + exposure * 0.2
        + saturation_quality * 0.08
        + colorfulness * 0.07
        + stability * 0.14
        + noise_quality * 0.08
    )
    rejection_reasons = _rejection_reasons(
        brightness,
        contrast,
        sharpness,
        noise_score,
        camera_shake_score,
        score,
    )
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
        rejection_reasons=rejection_reasons,
        backend=backend,
    )


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
        points = cv2.goodFeaturesToTrack(
            first,
            maxCorners=180,
            qualityLevel=0.01,
            minDistance=8,
        )
        if points is None or len(points) < 8:
            continue
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None)
        if tracked is None or status is None:
            continue
        valid = status.reshape(-1) == 1
        flow = tracked.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
        if len(flow) < 8:
            continue
        median = np.median(flow, axis=0)
        directions.append(median)
        residual = np.linalg.norm(flow - median, axis=1)
        diagonal = math.hypot(first.shape[1], first.shape[0])
        motion_values.append(_clamp(float(np.linalg.norm(median)) / diagonal * 900))
        shake_values.append(_clamp(float(np.percentile(residual, 75)) / 10 * 100))
    if len(directions) > 1:
        direction_change = float(np.linalg.norm(directions[1] - directions[0]))
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


def _pillow_motion_score(panels: list[Image.Image]) -> float:
    if len(panels) < 2:
        return 0
    values = []
    for first, second in zip(panels, panels[1:], strict=False):
        difference = ImageChops.difference(first.convert("L"), second.convert("L"))
        values.append(_clamp(ImageStat.Stat(difference).mean[0] / 255 * 180))
    return _average(values)


def _split_contact_sheet_cv(image: Any) -> list[Any]:
    height, width = image.shape[:2]
    if width < height * 2.2:
        return [image]
    panel_width = width // 3
    return [image[:, index * panel_width : (index + 1) * panel_width] for index in range(3)]


def _split_contact_sheet_pillow(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    if width < height * 2.2:
        return [image]
    panel_width = width // 3
    return [
        image.crop((index * panel_width, 0, (index + 1) * panel_width, height))
        for index in range(3)
    ]


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
