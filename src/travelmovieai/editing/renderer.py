"""FFmpeg rendering for declarative montage plans."""

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from travelmovieai.core.exceptions import DependencyUnavailableError, MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MontageClip, QuickMontagePlan

ProgressCallback = Callable[[int, int, str], None]


class QuickMontageRenderer:
    def __init__(self, ffmpeg_binary: str = "ffmpeg") -> None:
        self.ffmpeg_binary = ffmpeg_binary

    def render(
        self,
        plan: QuickMontagePlan,
        output_path: Path,
        work_dir: Path,
        progress: ProgressCallback | None = None,
    ) -> None:
        segments_dir = work_dir / "quick_montage_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        segment_paths: list[Path] = []
        total_steps = len(plan.clips) + 1

        for index, clip in enumerate(plan.clips, start=1):
            segment_path = segments_dir / f"{index:05d}.mp4"
            if progress:
                progress(index - 1, total_steps, f"Подготовка клипа {index}/{len(plan.clips)}")
            self._render_segment(clip, plan, segment_path)
            segment_paths.append(segment_path)

        if progress:
            progress(
                len(plan.clips),
                total_steps,
                "Переходы, музыка и финальная сборка",
            )
        self._compose_segments(segment_paths, plan, output_path, work_dir)
        if progress:
            progress(total_steps, total_steps, "Фильм готов")

    def _render_segment(
        self,
        clip: MontageClip,
        plan: QuickMontagePlan,
        output_path: Path,
    ) -> None:
        settings = plan.settings
        duration = _decimal(clip.duration_seconds)
        video_filter = (
            f"scale={settings.width}:{settings.height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={settings.width}:{settings.height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={settings.fps},setsar=1,format=yuv420p"
        )

        if clip.media_type is MediaType.PHOTO:
            command = [
                self.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-loop",
                "1",
                "-t",
                duration,
                "-i",
                str(clip.source_path),
                "-f",
                "lavfi",
                "-t",
                duration,
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-filter_complex",
                f"[0:v:0]{video_filter}[v];[1:a:0]atrim=0:{duration},asetpts=PTS-STARTPTS[a]",
                "-map",
                "[v]",
                "-map",
                "[a]",
            ]
        else:
            audio_input = "0:a:0" if clip.has_audio else "1:a:0"
            command = [
                self.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                _decimal(clip.source_start_seconds),
                "-t",
                duration,
                "-i",
                str(clip.source_path),
            ]
            if not clip.has_audio:
                command.extend(
                    [
                        "-f",
                        "lavfi",
                        "-t",
                        duration,
                        "-i",
                        "anullsrc=r=48000:cl=stereo",
                    ]
                )
            command.extend(
                [
                    "-filter_complex",
                    f"[0:v:0]{video_filter}[v];"
                    f"[{audio_input}]aresample=48000,"
                    f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                    f"apad,atrim=0:{duration},asetpts=PTS-STARTPTS[a]",
                    "-map",
                    "[v]",
                    "-map",
                    "[a]",
                ]
            )

        command.extend(
            [
                "-t",
                duration,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "21",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        self._run(command, f"Не удалось подготовить {clip.relative_path}")

    def _compose_segments(
        self,
        segments: list[Path],
        plan: QuickMontagePlan,
        output_path: Path,
        work_dir: Path,
    ) -> None:
        transition_duration = _transition_duration(plan)
        if transition_duration <= 0 and plan.music_path is None:
            self._concat_segments(segments, output_path, work_dir)
            return

        filter_path = work_dir / "quick_montage_filters.txt"
        filter_path.write_text(
            _build_filter_graph(plan, transition_duration),
            encoding="utf-8",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_name(f".{output_path.stem}.{uuid4().hex}.tmp.mp4")
        command = [self.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-y"]
        for segment in segments:
            command.extend(["-i", str(segment)])
        if plan.music_path is not None:
            command.extend(["-stream_loop", "-1", "-i", str(plan.music_path)])
        command.extend(
            [
                "-filter_complex_script",
                str(filter_path),
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-t",
                _decimal(plan.total_duration_seconds),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "21",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ]
        )
        try:
            self._run(command, "Не удалось применить переходы и музыку")
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)
            shutil.rmtree(work_dir / "quick_montage_segments", ignore_errors=True)
            filter_path.unlink(missing_ok=True)

    def _concat_segments(
        self,
        segments: list[Path],
        output_path: Path,
        work_dir: Path,
    ) -> None:
        concat_path = work_dir / "quick_montage_concat.txt"
        concat_path.write_text(
            "".join(f"file '{_concat_path(path)}'\n" for path in segments),
            encoding="utf-8",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_name(f".{output_path.stem}.{uuid4().hex}.tmp.mp4")
        try:
            self._run(
                [
                    self.ffmpeg_binary,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_path),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(temporary_output),
                ],
                "Не удалось объединить подготовленные клипы",
            )
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)
            shutil.rmtree(work_dir / "quick_montage_segments", ignore_errors=True)
            concat_path.unlink(missing_ok=True)

    def _run(self, command: list[str], message: str) -> None:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as error:
            raise DependencyUnavailableError(
                f"FFmpeg executable was not found: {self.ffmpeg_binary}"
            ) from error

        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown FFmpeg error"
            raise MontageError(f"{message}: {detail}")


def _decimal(value: float) -> str:
    return f"{value:.3f}"


def _concat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def _transition_duration(plan: QuickMontagePlan) -> float:
    if plan.settings.transition == "none" or len(plan.clips) < 2:
        return 0.0
    shortest = min(clip.duration_seconds for clip in plan.clips)
    return min(plan.settings.transition_duration_seconds, shortest * 0.45)


def _build_filter_graph(
    plan: QuickMontagePlan,
    transition_duration: float,
) -> str:
    lines: list[str] = []
    clip_count = len(plan.clips)
    for index in range(clip_count):
        lines.append(f"[{index}:v]settb=AVTB,setpts=PTS-STARTPTS[v{index}base]")
        lines.append(f"[{index}:a]aresample=48000,asetpts=PTS-STARTPTS[a{index}base]")

    video_label = "v0base"
    audio_label = "a0base"
    elapsed = plan.clips[0].duration_seconds
    for index in range(1, clip_count):
        next_video = f"v{index}mix"
        next_audio = f"a{index}mix"
        if transition_duration > 0:
            offset = max(0.0, elapsed - transition_duration)
            lines.append(
                f"[{video_label}][v{index}base]"
                f"xfade=transition={plan.settings.transition}:"
                f"duration={transition_duration:.3f}:offset={offset:.3f}"
                f"[{next_video}]"
            )
            lines.append(
                f"[{audio_label}][a{index}base]"
                f"acrossfade=d={transition_duration:.3f}:c1=tri:c2=tri"
                f"[{next_audio}]"
            )
            elapsed += plan.clips[index].duration_seconds - transition_duration
        else:
            lines.append(f"[{video_label}][v{index}base]concat=n=2:v=1:a=0[{next_video}]")
            lines.append(f"[{audio_label}][a{index}base]concat=n=2:v=0:a=1[{next_audio}]")
            elapsed += plan.clips[index].duration_seconds
        video_label = next_video
        audio_label = next_audio

    lines.append(f"[{video_label}]format=yuv420p[vout]")
    if plan.music_path is None:
        lines.append(f"[{audio_label}]anull[aout]")
    else:
        music_index = clip_count
        fade_out_start = max(0.0, plan.total_duration_seconds - 1.5)
        lines.append(
            f"[{music_index}:a]aresample=48000,"
            f"atrim=0:{plan.total_duration_seconds:.3f},"
            "asetpts=PTS-STARTPTS,"
            f"volume={plan.settings.music_volume:.3f},"
            "afade=t=in:st=0:d=1.5,"
            f"afade=t=out:st={fade_out_start:.3f}:d=1.5[music]"
        )
        lines.append(
            f"[music][{audio_label}]"
            "sidechaincompress=threshold=0.04:ratio=8:attack=20:release=500[ducked]"
        )
        lines.append(
            f"[{audio_label}][ducked]"
            "amix=inputs=2:duration=first:dropout_transition=2,"
            "alimiter=limit=0.95[aout]"
        )
    return ";\n".join(lines)
