import wave
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image, ImageDraw

from travelmovieai.analysis.quality import (
    TorchCudaQualityAnalyzer,
    VisualQualityAnalyzer,
    analyze_scene_quality,
)
from travelmovieai.domain.enums import MediaType, StoryStyle
from travelmovieai.domain.models import (
    MontageClip,
    MusicAccent,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
    TemporalHighlightWindow,
    VisualQualityMetrics,
)
from travelmovieai.story.music import (
    apply_music_accents,
    build_music_accents,
    build_music_beat_grid,
    build_music_cue_sections,
    build_music_plan,
    choose_music_profile,
    generate_ambient_soundtrack,
)


def test_quality_analysis_persists_explainable_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "bright.jpg"
    Image.new("RGB", (160, 90), (230, 130, 60)).save(image_path)
    scene = Scene(
        asset_id="b04173c0-6122-40e8-b12c-1101942f64d7",
        start_seconds=0,
        end_seconds=2,
        keyframe_path=image_path,
    )

    report = analyze_scene_quality([scene], VisualQualityAnalyzer())

    analyzed = report.scenes[0]
    assert analyzed.quality_score is not None
    metrics = analyzed.metadata["quality_metrics"]
    assert metrics["brightness"] > 40
    assert metrics["backend"] in {"opencv", "pillow"}
    assert 0 <= metrics["noise_score"] <= 100
    assert 0 <= metrics["motion_score"] <= 100
    assert 0 <= metrics["camera_shake_score"] <= 100
    assert metrics["panel_quality_scores"]
    assert metrics["panel_details"]
    assert metrics["best_panel_position"] is not None
    assert metrics["candidate_windows"]
    assert metrics["candidate_windows"][0]["source"] == "visual_quality"
    assert isinstance(metrics["rejection_reasons"], list)


@pytest.mark.parametrize(
    ("sample_count", "height"),
    [(5, 540), (9, 810)],
)
def test_quality_analysis_splits_temporal_contact_sheet_row_major(
    tmp_path: Path,
    sample_count: int,
    height: int,
) -> None:
    image_path = tmp_path / f"scene-contact-v4-{sample_count}.png"
    sheet = Image.new("RGB", (1440, height), "black")
    for index in range(sample_count):
        panel = Image.new("RGB", (480, 270), (20 + index * 12, 35, 50))
        draw = ImageDraw.Draw(panel)
        offset = 30 + (index % 3) * 45
        draw.rectangle((offset, 35, offset + 90, 230), fill="white")
        draw.line((0, 20 + index * 9, 479, 245 - index * 7), fill="yellow", width=6)
        sheet.paste(panel, ((index % 3) * 480, (index // 3) * 270))
    sheet.save(image_path)

    metrics = VisualQualityAnalyzer().analyze(image_path)

    assert metrics.sample_count == sample_count
    assert len(metrics.panel_details) == sample_count
    assert len(metrics.panel_quality_scores) == sample_count
    assert len(metrics.candidate_windows) == sample_count
    assert metrics.sample_positions == list(
        (0.08, 0.3, 0.5, 0.7, 0.92)
        if sample_count == 5
        else (0.06, 0.17, 0.29, 0.4, 0.5, 0.6, 0.71, 0.83, 0.94)
    )
    assert metrics.motion_score > 0
    assert metrics.camera_shake_score > 0


def test_quality_analysis_does_not_misread_plain_sixteen_by_nine_photo_as_grid(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "ordinary-photo.png"
    Image.new("RGB", (1440, 810), (80, 120, 160)).save(image_path)

    metrics = VisualQualityAnalyzer().analyze(image_path)

    assert metrics.sample_count == 1
    assert len(metrics.panel_details) == 1


def test_quality_analysis_uses_persisted_actual_sample_positions(tmp_path: Path) -> None:
    image_path = tmp_path / "scene-contact-v4-3.png"
    Image.new("RGB", (1440, 270), (80, 120, 160)).save(image_path)
    positions = [0.1, 0.48, 0.84]
    scene = Scene(
        asset_id=uuid4(),
        start_seconds=0,
        end_seconds=2,
        keyframe_path=image_path,
        metadata={
            "contact_sheet": {
                "sample_count": 3,
                "sample_positions": positions,
            }
        },
    )

    report = analyze_scene_quality([scene], VisualQualityAnalyzer())

    metrics = report.scenes[0].metadata["quality_metrics"]
    assert metrics["sample_positions"] == positions
    assert [item["relative_position"] for item in metrics["candidate_windows"]] == positions


def test_temporal_highlight_window_rejects_position_outside_interval() -> None:
    with pytest.raises(ValueError, match="inside the highlight interval"):
        TemporalHighlightWindow(
            relative_start=0.2,
            relative_end=0.4,
            relative_position=0.8,
            confidence=0.9,
            source="vision",
        )


def test_parallel_quality_analysis_does_not_start_queued_work_after_cancel(
    tmp_path: Path,
) -> None:
    barrier = Barrier(2)
    calls: list[Path] = []
    scenes = []
    for index in range(5):
        image = tmp_path / f"frame-{index}.png"
        image.write_bytes(b"frame")
        scenes.append(
            Scene(
                asset_id=uuid4(),
                start_seconds=0,
                end_seconds=2,
                keyframe_path=image,
            )
        )

    class BlockingAnalyzer:
        def analyze(self, image_path: Path) -> VisualQualityMetrics:
            calls.append(image_path)
            barrier.wait(timeout=5)
            return VisualQualityMetrics(
                brightness=50,
                contrast=50,
                sharpness=50,
                saturation=50,
                colorfulness=50,
                quality_score=50,
                backend="test",
            )

    def cancel_after_first(current: int, total: int, message: str) -> None:
        del total, message
        if current >= 1:
            raise RuntimeError("cancel quality queue")

    with pytest.raises(RuntimeError, match="cancel quality queue"):
        analyze_scene_quality(
            scenes,
            analyzer=BlockingAnalyzer(),
            workers=2,
            progress=cancel_after_first,
        )

    assert len(calls) == 2


def test_cuda_quality_analyzer_uses_gpu_when_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch", reason="Torch is an optional Vision dependency")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    image_path = tmp_path / "contact.png"
    Image.new("RGB", (480, 90), (120, 180, 220)).save(image_path)

    metrics = TorchCudaQualityAnalyzer().analyze(image_path)

    assert metrics.backend == "torch-cuda"
    assert 0 <= metrics.quality_score <= 100


def test_auto_music_profile_uses_visual_metrics_and_generates_wav(
    tmp_path: Path,
) -> None:
    scene = Scene(
        asset_id="b04173c0-6122-40e8-b12c-1101942f64d7",
        start_seconds=0,
        end_seconds=2,
        metadata={
            "emotion": "exciting",
            "quality_metrics": {
                "brightness": 65,
                "saturation": 70,
                "sharpness": 75,
            },
        },
    )
    settings = QuickMontageSettings(
        target_duration_seconds=8,
        story_style=StoryStyle.ADVENTURE,
        music_mode="auto",
    )
    output = tmp_path / "generated.wav"
    montage_plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        total_duration_seconds=8,
    )

    profile, _ = choose_music_profile([scene], settings)
    plan = build_music_plan(
        [],
        [scene],
        settings,
        tmp_path / "music",
        output,
        montage_plan,
    )

    assert profile == "calm"
    assert plan.mode == "generated"
    assert plan.source_path == output
    assert plan.generated is True
    assert plan.duration_seconds == 8
    assert plan.arrangement_version == "adaptive-lounge-v7-content-revision"
    assert plan.cue_sections
    assert plan.cue_sections[0].bpm == 60
    assert plan.beat_grid
    assert plan.beat_grid[0].bar_index == 0
    with wave.open(str(output), "rb") as soundtrack:
        assert soundtrack.getnchannels() == 2
        assert soundtrack.getnframes() / soundtrack.getframerate() == 8


def test_music_beat_grid_adds_markers_for_off_grid_scene_accents() -> None:
    accents = [
        MusicAccent(
            time_seconds=5.274,
            kind="scene_change",
            strength=0.16,
            label="Scene change",
        )
    ]

    beat_grid = build_music_beat_grid(12, 60, accents)

    marker = min(beat_grid, key=lambda beat: abs(beat.time_seconds - 5.274))
    assert marker.time_seconds == 5.274
    assert marker.nearest_accent_kind == "scene_change"
    assert marker.strength >= 0.68


def test_lounge_music_is_melodic_stereo_and_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "lounge-first.wav"
    second = tmp_path / "lounge-second.wav"

    generate_ambient_soundtrack(
        first,
        duration_seconds=2,
        profile="lounge",
        bpm=84,
    )
    generate_ambient_soundtrack(
        second,
        duration_seconds=2,
        profile="lounge",
        bpm=84,
    )

    assert first.read_bytes() == second.read_bytes()
    with wave.open(str(first), "rb") as soundtrack:
        assert soundtrack.getframerate() == 44100
        assert soundtrack.getnchannels() == 2
        frames = soundtrack.readframes(soundtrack.getnframes())
    left = frames[0::4] + frames[1::4]
    right = frames[2::4] + frames[3::4]
    assert left != right


def test_generated_music_keeps_master_headroom(tmp_path: Path) -> None:
    output = tmp_path / "mastered.wav"

    generate_ambient_soundtrack(
        output,
        duration_seconds=4,
        profile="lounge",
        bpm=76,
        accents=[
            MusicAccent(
                time_seconds=1.8,
                kind="highlight",
                strength=0.9,
                label="Peak visual moment",
            )
        ],
    )

    with wave.open(str(output), "rb") as soundtrack:
        samples = np.frombuffer(
            soundtrack.readframes(soundtrack.getnframes()),
            dtype="<i2",
        )

    peak = int(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    clipped = int(np.count_nonzero(np.abs(samples) >= 32760))

    assert peak <= 30000
    assert rms > 100
    assert clipped == 0


def test_default_generated_music_keeps_high_frequencies_restrained(tmp_path: Path) -> None:
    output = tmp_path / "low-register.wav"

    generate_ambient_soundtrack(
        output,
        duration_seconds=4,
        profile="calm",
        bpm=60,
        accents=[
            MusicAccent(
                time_seconds=1.8,
                kind="highlight",
                strength=0.9,
                label="Peak visual moment",
            )
        ],
    )

    with wave.open(str(output), "rb") as soundtrack:
        sample_rate = soundtrack.getframerate()
        samples = np.frombuffer(
            soundtrack.readframes(soundtrack.getnframes()),
            dtype="<i2",
        ).reshape(-1, 2)

    mono = samples.astype(np.float64).mean(axis=1)
    spectrum = np.abs(np.fft.rfft(mono))
    frequencies = np.fft.rfftfreq(len(mono), d=1 / sample_rate)
    low_mid_energy = float(np.sum(spectrum[(frequencies >= 45) & (frequencies <= 700)]))
    high_energy = float(np.sum(spectrum[frequencies > 2500]))

    assert low_mid_energy > 0
    assert high_energy / low_mid_energy < 0.08


def test_short_model_music_is_repeated_across_full_timeline(tmp_path: Path) -> None:
    output = tmp_path / "short-model.wav"
    sample_rate = 8000
    time = np.arange(sample_rate // 4, dtype=np.float64) / sample_rate
    samples = (np.sin(2 * np.pi * 220 * time) * 5000).astype("<i2")
    stereo = np.column_stack((samples, samples))
    with wave.open(str(output), "wb") as soundtrack:
        soundtrack.setnchannels(2)
        soundtrack.setsampwidth(2)
        soundtrack.setframerate(sample_rate)
        soundtrack.writeframes(stereo.tobytes())

    apply_music_accents(output, duration_seconds=1, accents=[])

    with wave.open(str(output), "rb") as soundtrack:
        extended = np.frombuffer(
            soundtrack.readframes(soundtrack.getnframes()),
            dtype="<i2",
        ).reshape(-1, 2)
    ending = extended[int(0.55 * sample_rate) : int(0.7 * sample_rate)]

    assert len(extended) == sample_rate
    assert float(np.sqrt(np.mean(ending.astype(np.float64) ** 2))) > 100


def test_auto_music_selects_calm_default_for_relaxing_travel_scene() -> None:
    scene = Scene(
        asset_id="b04173c0-6122-40e8-b12c-1101942f64d7",
        start_seconds=0,
        end_seconds=2,
        metadata={
            "emotion": "relaxing",
            "location_type": "beach",
            "activity": "walking",
        },
    )
    settings = QuickMontageSettings(story_style=StoryStyle.DOCUMENTARY)

    profile, _ = choose_music_profile([scene], settings)

    assert profile == "calm"


def test_music_cue_sheet_follows_timeline_and_scene_importance(
    tmp_path: Path,
) -> None:
    event_a = uuid4()
    event_b = uuid4()
    settings = QuickMontageSettings(
        target_duration_seconds=12,
        transition="fade",
        transition_duration_seconds=0.5,
    )
    clips = [
        MontageClip(
            asset_id=uuid4(),
            scene_id=uuid4(),
            source_path=tmp_path / "one.mp4",
            relative_path=Path("one.mp4"),
            media_type=MediaType.VIDEO,
            duration_seconds=4,
            semantic_score=92,
            event_id=event_a,
            caption="Main viewpoint",
        ),
        MontageClip(
            asset_id=uuid4(),
            scene_id=uuid4(),
            source_path=tmp_path / "two.mp4",
            relative_path=Path("two.mp4"),
            media_type=MediaType.VIDEO,
            duration_seconds=4,
            semantic_score=55,
            event_id=event_b,
        ),
        MontageClip(
            asset_id=uuid4(),
            scene_id=uuid4(),
            source_path=tmp_path / "three.mp4",
            relative_path=Path("three.mp4"),
            media_type=MediaType.VIDEO,
            duration_seconds=4,
            semantic_score=75,
            event_id=event_b,
        ),
    ]
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=clips,
        total_duration_seconds=11,
        selection_mode="semantic",
    )

    accents = build_music_accents(plan)
    sections = build_music_cue_sections(plan, accents, bpm=76)
    beat_grid = build_music_beat_grid(plan.total_duration_seconds, 76, accents)

    assert accents[0].kind == "intro"
    assert any(accent.kind == "event_change" and accent.time_seconds == 3.5 for accent in accents)
    assert any(
        accent.kind == "highlight"
        and accent.scene_id == clips[0].scene_id
        and 1.5 < accent.time_seconds < 2
        for accent in accents
    )
    assert accents[-1].kind == "finale"
    assert accents[-1].time_seconds < plan.total_duration_seconds
    assert sections[0].role == "intro"
    assert sections[-1].role == "finale"
    assert all(section.bpm == 76 for section in sections)
    assert any(section.role == "highlight" for section in sections)
    assert beat_grid
    assert beat_grid[0].strength > beat_grid[1].strength
    assert any(beat.nearest_accent_kind == "intro" for beat in beat_grid[:2])


def test_highlight_cue_creates_audible_accent(tmp_path: Path) -> None:
    output = tmp_path / "accent.wav"
    generate_ambient_soundtrack(
        output,
        duration_seconds=5,
        profile="lounge",
        bpm=84,
        accents=[
            MusicAccent(
                time_seconds=2.17,
                kind="highlight",
                strength=1,
                label="Main moment",
            )
        ],
    )

    with wave.open(str(output), "rb") as soundtrack:
        sample_rate = soundtrack.getframerate()
        samples = np.frombuffer(
            soundtrack.readframes(soundtrack.getnframes()),
            dtype="<i2",
        ).reshape(-1, 2)
    accent = samples[int(2.10 * sample_rate) : int(2.24 * sample_rate)]
    baseline = samples[int(1.50 * sample_rate) : int(1.64 * sample_rate)]
    accent_rms = float(np.sqrt(np.mean(accent.astype(np.float64) ** 2)))
    baseline_rms = float(np.sqrt(np.mean(baseline.astype(np.float64) ** 2)))

    assert accent_rms > baseline_rms * 1.05
