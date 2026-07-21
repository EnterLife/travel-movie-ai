from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from travelmovieai.domain.models import (
    AudioAnalysisReport,
    DuplicateDetectionReport,
    EmbeddingAnalysisReport,
    FrameSamplingReport,
    MediaScanReport,
    MusicCandidate,
    MusicPlan,
    Scene,
    SceneDetectionReport,
    SpeechAnalysisReport,
    SpeechSegment,
    SynthesizedNarrationLine,
    VisionAnalysisReport,
    VisualQualityMetrics,
    VoiceSynthesisReport,
)


def test_scene_accepts_valid_window_and_rejects_non_increasing_boundaries() -> None:
    scene = Scene(
        asset_id=UUID(int=1),
        start_seconds=1.25,
        end_seconds=2.5,
    )

    assert scene.end_seconds == 2.5
    for end_seconds in (1.25, 1.0):
        with pytest.raises(ValidationError, match="end_seconds must be greater"):
            Scene(
                asset_id=UUID(int=1),
                start_seconds=1.25,
                end_seconds=end_seconds,
            )
    with pytest.raises(ValidationError):
        Scene(asset_id=UUID(int=1), start_seconds=1.25, end_seconds=float("nan"))
    for field in ("start_seconds", "end_seconds"):
        values = {"start_seconds": 0.0, "end_seconds": 1.0, field: -0.1}
        with pytest.raises(ValidationError):
            Scene(asset_id=UUID(int=1), **values)


def test_scene_detection_report_validates_counts() -> None:
    scene = Scene(asset_id=UUID(int=1), start_seconds=0, end_seconds=1)

    report = SceneDetectionReport(
        created_at=datetime.now(UTC),
        scenes=[scene],
        detected_count=0,
        cached_count=1,
        fallback_count=1,
    )

    assert report.cached_count == 1
    with pytest.raises(ValidationError):
        SceneDetectionReport(created_at=datetime.now(UTC), detected_count=-1)
    with pytest.raises(ValidationError, match="must match the scene count"):
        SceneDetectionReport(
            created_at=datetime.now(UTC),
            scenes=[scene],
            detected_count=0,
            cached_count=0,
        )
    with pytest.raises(ValidationError, match="cannot exceed the scene count"):
        SceneDetectionReport(
            created_at=datetime.now(UTC),
            scenes=[scene],
            detected_count=1,
            fallback_count=2,
        )


def test_speech_segment_validates_time_window() -> None:
    segment = SpeechSegment(start_seconds=0.5, end_seconds=1.5, text="Arrival")

    assert segment.end_seconds == 1.5
    for end_seconds in (0.5, 0.25):
        with pytest.raises(ValidationError, match="end_seconds must be greater"):
            SpeechSegment(start_seconds=0.5, end_seconds=end_seconds)
    with pytest.raises(ValidationError):
        SpeechSegment(start_seconds=0.5, end_seconds=float("nan"))


def test_vision_report_validates_identity_and_processed_counts() -> None:
    scene = Scene(asset_id=UUID(int=1), start_seconds=0, end_seconds=1)
    values = {
        "created_at": datetime.now(UTC),
        "provider": "qwen",
        "model": "qwen-local",
        "prompt_version": "vision-v1",
        "scenes": [scene],
    }

    skipped = VisionAnalysisReport.model_validate(values)
    analyzed = VisionAnalysisReport.model_validate({**values, "analyzed_count": 1})

    assert skipped.analyzed_count == 0
    assert analyzed.analyzed_count == 1
    for field in ("provider", "model", "prompt_version"):
        with pytest.raises(ValidationError):
            VisionAnalysisReport.model_validate({**values, field: ""})
    for field in ("analyzed_count", "cached_count", "degraded_count"):
        with pytest.raises(ValidationError):
            VisionAnalysisReport.model_validate({**values, field: -1})
    with pytest.raises(ValidationError, match="cannot exceed the scene count"):
        VisionAnalysisReport.model_validate(
            {
                **values,
                "analyzed_count": 1,
                "cached_count": 1,
            }
        )


def test_music_plan_preserves_legacy_generated_shape_and_validates_known_generator() -> None:
    legacy = MusicPlan(mode="generated")
    neural = MusicPlan(
        mode="generated",
        source_content_sha256="a" * 64,
        generator="ace-step",
        model="ace-step-local",
    )
    procedural = MusicPlan(mode="generated", generator="procedural")
    complete = MusicPlan(
        mode="generated",
        source_path=Path("generated.wav"),
        source_content_sha256="b" * 64,
        duration_seconds=2.5,
        generator="ace-step",
        model="ace-step-local",
        cache_key="c" * 64,
        generated=True,
    )

    assert legacy.generator is None
    assert legacy.source_content_sha256 is None
    assert neural.source_content_sha256 == "a" * 64
    assert neural.model == "ace-step-local"
    assert procedural.model is None
    assert complete.generated is True
    for values in (
        {"generator": "ace-step"},
        {"generator": "musicgen", "model": ""},
        {"generator": "procedural", "model": "unexpected-model"},
    ):
        with pytest.raises(ValidationError):
            MusicPlan.model_validate({"mode": "generated", **values})
    with pytest.raises(ValidationError):
        MusicPlan(mode="generated", source_content_sha256="a" * 63)
    with pytest.raises(ValidationError):
        MusicPlan(mode="generated", source_content_sha256="G" * 64)
    complete_values = complete.model_dump()
    for field, invalid in (
        ("mode", "none"),
        ("source_path", None),
        ("generator", None),
        ("duration_seconds", 0),
        ("cache_key", "short"),
        ("source_content_sha256", None),
    ):
        with pytest.raises(ValidationError):
            MusicPlan.model_validate({**complete_values, field: invalid})


def test_music_plan_validates_selected_candidate_consistency() -> None:
    candidate = MusicCandidate(
        index=0,
        source_path=Path("candidate.wav"),
        source_content_sha256="a" * 64,
        seed=42,
        total_score=90,
        technical_score=92,
        structure_score=88,
        style_score=89,
        duration_seconds=30,
        sample_rate=48000,
        channels=2,
        selected=True,
    )

    plan = MusicPlan(
        mode="generated",
        generator="ace-step",
        model="ACE-Step/acestep-v15-turbo",
        candidates=[candidate],
        selected_candidate_index=0,
    )

    assert plan.candidates[0].selected is True
    with pytest.raises(ValidationError, match="exactly one"):
        MusicPlan(
            mode="generated",
            generator="ace-step",
            model="ACE-Step/acestep-v15-turbo",
            candidates=[candidate.model_copy(update={"selected": False})],
            selected_candidate_index=0,
        )
    with pytest.raises(ValidationError, match="more than one"):
        MusicPlan(
            mode="generated",
            generator="ace-step",
            model="ACE-Step/acestep-v15-turbo",
            candidates=[candidate, candidate.model_copy(update={"index": 1})],
            selected_candidate_index=0,
        )
    with pytest.raises(ValidationError, match="must be unique"):
        MusicPlan(
            mode="generated",
            generator="ace-step",
            model="ACE-Step/acestep-v15-turbo",
            candidates=[candidate, candidate.model_copy(update={"selected": False})],
            selected_candidate_index=0,
        )
    with pytest.raises(ValidationError, match="requires a selected candidate index"):
        MusicPlan(
            mode="generated",
            generator="ace-step",
            model="ACE-Step/acestep-v15-turbo",
            candidates=[candidate],
        )


def test_persisted_report_counts_and_identities_reject_invalid_values() -> None:
    now = datetime.now(UTC)
    valid_models = (
        MediaScanReport(input_path=Path("input"), scanned_at=now),
        FrameSamplingReport(created_at=now),
        SpeechAnalysisReport(created_at=now, provider="whisper", model="medium"),
        AudioAnalysisReport(created_at=now),
        DuplicateDetectionReport(created_at=now),
        EmbeddingAnalysisReport(created_at=now, backend="faiss", dimensions=3),
        VisualQualityMetrics(
            brightness=50,
            contrast=50,
            sharpness=50,
            saturation=50,
            colorfulness=50,
            quality_score=50,
            backend="opencv",
        ),
    )

    for model in valid_models:
        assert model is not None
    invalid_counts = (
        (MediaScanReport, {"input_path": Path("input"), "scanned_at": now, "error_count": -1}),
        (FrameSamplingReport, {"created_at": now, "cached_count": -1}),
        (
            SpeechAnalysisReport,
            {"created_at": now, "provider": "whisper", "model": "medium", "cached_count": -1},
        ),
        (AudioAnalysisReport, {"created_at": now, "skipped_count": -1}),
        (DuplicateDetectionReport, {"created_at": now, "duplicate_count": -1}),
    )
    for model_type, values in invalid_counts:
        with pytest.raises(ValidationError):
            model_type.model_validate(values)

    with pytest.raises(ValidationError):
        SpeechAnalysisReport(created_at=now, provider="", model="medium")
    with pytest.raises(ValidationError):
        EmbeddingAnalysisReport(created_at=now, backend="", dimensions=3)
    quality_values = valid_models[-1].model_dump()
    with pytest.raises(ValidationError):
        VisualQualityMetrics.model_validate({**quality_values, "backend": ""})


def test_voice_report_requires_nonempty_provider_and_model() -> None:
    line = SynthesizedNarrationLine(
        line_index=0,
        section_role="opening",
        audio_path=Path("line.wav"),
        duration_seconds=1,
        sample_rate=16_000,
        channels=1,
    )
    values = {
        "created_at": datetime.now(UTC),
        "provider": "piper",
        "model": "voice.onnx",
        "line_count": 1,
        "lines": [line],
    }

    report = VoiceSynthesisReport.model_validate(values)

    assert report.provider == "piper"
    for field in ("provider", "model"):
        with pytest.raises(ValidationError):
            VoiceSynthesisReport.model_validate({**values, field: ""})
