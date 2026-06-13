"""OpenCV-based visual quality metrics with a Pillow fallback."""

import importlib
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = _clamp(float(gray.mean()) / 255 * 100)
        contrast = _clamp(float(gray.std()) / 96 * 100)
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = _log_score(laplacian_variance, 1800)
        saturation = _clamp(float(hsv[:, :, 1].mean()) / 255 * 100)

        blue, green, red = cv2.split(image.astype("float32"))
        red_green = red - green
        yellow_blue = 0.5 * (red + green) - blue
        colorfulness_raw = math.sqrt(
            float(red_green.std()) ** 2 + float(yellow_blue.std()) ** 2
        ) + 0.3 * math.sqrt(float(red_green.mean()) ** 2 + float(yellow_blue.mean()) ** 2)
        colorfulness = _clamp(colorfulness_raw / 90 * 100)
        return _metrics(
            brightness,
            contrast,
            sharpness,
            saturation,
            colorfulness,
            "opencv",
        )

    def _analyze_pillow(self, image_path: Path) -> VisualQualityMetrics:
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            grayscale = rgb.convert("L")
            gray_stats = ImageStat.Stat(grayscale)
            brightness = _clamp(gray_stats.mean[0] / 255 * 100)
            contrast = _clamp(gray_stats.stddev[0] / 96 * 100)
            edge_stats = ImageStat.Stat(grayscale.filter(ImageFilter.FIND_EDGES))
            sharpness = _log_score(edge_stats.var[0], 4500)
            rgb_stats = ImageStat.Stat(rgb)
            channel_spread = sum(rgb_stats.stddev) / 3
            colorfulness = _clamp(channel_spread / 80 * 100)
            maximum = ImageStat.Stat(rgb).extrema
            saturation = _clamp(sum(high - low for low, high in maximum) / (3 * 255) * 100)
        return _metrics(
            brightness,
            contrast,
            sharpness,
            saturation,
            colorfulness,
            "pillow",
        )


def analyze_scene_quality(
    scenes: list[Scene],
    analyzer: VisualQualityAnalyzer | None = None,
) -> QualityAnalysisReport:
    resolved_analyzer = analyzer or VisualQualityAnalyzer()
    analyzed: list[Scene] = []
    for scene in scenes:
        if scene.keyframe_path is None:
            analyzed.append(scene)
            continue
        metrics = resolved_analyzer.analyze(scene.keyframe_path)
        analyzed.append(
            scene.model_copy(
                update={
                    "quality_score": metrics.quality_score,
                    "metadata": {
                        **scene.metadata,
                        "quality_metrics": metrics.model_dump(),
                    },
                }
            )
        )
    return QualityAnalysisReport(
        created_at=datetime.now(UTC),
        scenes=analyzed,
    )


def _metrics(
    brightness: float,
    contrast: float,
    sharpness: float,
    saturation: float,
    colorfulness: float,
    backend: str,
) -> VisualQualityMetrics:
    exposure = _clamp(100 - abs(brightness - 52) * 2.3)
    saturation_quality = _clamp(100 - abs(saturation - 45) * 1.2)
    score = _clamp(
        sharpness * 0.35
        + contrast * 0.2
        + exposure * 0.25
        + saturation_quality * 0.1
        + colorfulness * 0.1
    )
    return VisualQualityMetrics(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        saturation=saturation,
        colorfulness=colorfulness,
        quality_score=score,
        backend=backend,
    )


def _log_score(value: float, reference: float) -> float:
    return _clamp(math.log1p(max(0, value)) / math.log1p(reference) * 100)


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))
