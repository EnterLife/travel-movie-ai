"""FFmpeg rendering for declarative montage plans."""

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

from travelmovieai.core.exceptions import DependencyUnavailableError, MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MontageClip, QuickMontagePlan
from travelmovieai.infrastructure.system import check_cuda

ProgressCallback = Callable[[int, int, str], None]


class QuickMontageRenderer:
    def __init__(
        self,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        workers: int = 1,
        ffmpeg_threads: int = 1,
        timeout_seconds: float = 7200,
    ) -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self.workers = max(1, workers)
        self.ffmpeg_threads = max(1, ffmpeg_threads)
        self.timeout_seconds = timeout_seconds if timeout_seconds > 0 else 7200
        self._render_device = "cpu"
        self._encoder = "libx264"

    def render(
        self,
        plan: QuickMontagePlan,
        output_path: Path,
        work_dir: Path,
        progress: ProgressCallback | None = None,
    ) -> str:
        if plan.music_path is not None and not plan.music_path.is_file():
            raise MontageError(f"Soundtrack file does not exist: {plan.music_path}")
        self._render_device = plan.settings.render_device
        self._encoder = self._select_encoder(plan.settings.render_device)
        segments_dir = work_dir / "quick_montage_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        total_steps = len(plan.clips) + 1
        try:
            segment_paths = self._render_segments(plan, segments_dir, progress, total_steps)
        except MontageError:
            if self._render_device != "auto" or self._encoder != "h264_nvenc":
                raise
            self._encoder = "libx264"
            shutil.rmtree(segments_dir, ignore_errors=True)
            segments_dir.mkdir(parents=True, exist_ok=True)
            segment_paths = self._render_segments(plan, segments_dir, progress, total_steps)

        if progress:
            progress(
                len(plan.clips),
                total_steps,
                "Music and final assembly",
            )
        self._compose_segments(segment_paths, plan, output_path, work_dir)
        self._validate_output(output_path)
        if progress:
            progress(total_steps, total_steps, "Film ready")
        return self._encoder

    def _render_segments(
        self,
        plan: QuickMontagePlan,
        segments_dir: Path,
        progress: ProgressCallback | None,
        total_steps: int,
    ) -> list[Path]:
        segment_paths = [
            segments_dir / f"{index:05d}.mp4" for index in range(1, len(plan.clips) + 1)
        ]
        worker_count = min(self.workers, len(plan.clips))
        if worker_count <= 1:
            for index, (clip, segment_path) in enumerate(
                zip(plan.clips, segment_paths, strict=True),
                start=1,
            ):
                if progress:
                    progress(
                        index - 1,
                        total_steps,
                        f"Rendering clip {index}/{len(plan.clips)}, "
                        f"encoder={self._encoder}, threads={self.ffmpeg_threads}",
                    )
                self._render_segment(clip, plan, segment_path)
            return segment_paths

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="travelmovieai-render",
        ) as executor:
            futures = {
                executor.submit(self._render_segment, clip, plan, segment_path): index
                for index, (clip, segment_path) in enumerate(
                    zip(plan.clips, segment_paths, strict=True),
                    start=1,
                )
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                future.result()
                if progress:
                    progress(
                        completed,
                        total_steps,
                        f"Clips complete {completed}/{len(plan.clips)}, "
                        f"render workers={worker_count}, encoder={self._encoder}",
                    )
        return segment_paths

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
                "-filter_threads",
                str(self.ffmpeg_threads),
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
            seek_start = _seek_start_seconds(clip.source_start_seconds)
            trim_start = clip.source_start_seconds - seek_start
            decode_duration = _decimal(clip.duration_seconds + trim_start + 0.25)
            command = [
                self.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-filter_threads",
                str(self.ffmpeg_threads),
                "-ss",
                _decimal(seek_start),
                "-t",
                decode_duration,
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
                    f"[0:v:0]trim=start={_decimal(trim_start)}:duration={duration},"
                    f"setpts=PTS-STARTPTS,{video_filter}[v];"
                    f"[{audio_input}]"
                    f"{_audio_trim_filter(clip.has_audio, trim_start, duration)}"
                    "aresample=48000,"
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
                *self._video_encoder_args(),
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-map_metadata",
                "-1",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        self._run(command, f"Could not prepare {clip.relative_path}")

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
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-filter_threads",
            str(self.ffmpeg_threads),
        ]
        for segment in segments:
            command.extend(["-i", str(segment)])
        if plan.music_path is not None:
            command.extend(["-stream_loop", "-1"])
            command.extend(["-i", str(plan.music_path)])
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
                *self._video_encoder_args(),
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-map_metadata",
                "-1",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ]
        )
        try:
            try:
                self._run(command, "Could not assemble clips and music")
            except MontageError:
                if self._render_device != "auto" or self._encoder != "h264_nvenc":
                    raise
                self._encoder = "libx264"
                command = _replace_video_encoder(command, self._video_encoder_args())
                self._run(command, "Could not assemble clips and music")
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
                    "-map_metadata",
                    "-1",
                    "-movflags",
                    "+faststart",
                    str(temporary_output),
                ],
                "Could not join prepared clips",
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
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as error:
            raise DependencyUnavailableError(
                f"FFmpeg executable was not found: {self.ffmpeg_binary}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise MontageError(
                f"{message}: FFmpeg timed out after {self.timeout_seconds:g}s."
            ) from error

        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown FFmpeg error"
            raise MontageError(f"{message}: {detail}")

    def _select_encoder(self, render_device: str) -> str:
        cuda = check_cuda(self.ffmpeg_binary)
        if render_device == "cuda":
            if not cuda.available or not cuda.ffmpeg_nvenc:
                raise DependencyUnavailableError(
                    "CUDA rendering was selected, but NVIDIA GPU or h264_nvenc is unavailable."
                )
            return "h264_nvenc"
        if render_device == "auto" and cuda.available and cuda.ffmpeg_nvenc:
            return "h264_nvenc"
        return "libx264"

    def _video_encoder_args(self) -> list[str]:
        if self._encoder == "h264_nvenc":
            return [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p5",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "21",
                "-b:v",
                "0",
            ]
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-threads",
            str(self.ffmpeg_threads),
        ]

    def _validate_output(self, output_path: Path) -> None:
        command = [
            self.ffprobe_binary,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as error:
            raise DependencyUnavailableError(
                f"FFprobe executable was not found: {self.ffprobe_binary}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise MontageError(
                f"The final movie failed FFprobe validation: FFprobe timed out after "
                f"{self.timeout_seconds:g}s."
            ) from error
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown FFprobe error"
            raise MontageError(f"The final movie failed FFprobe validation: {detail}")
        try:
            payload = json.loads(completed.stdout)
            stream_types = {stream.get("codec_type") for stream in payload.get("streams", [])}
            duration = float(payload.get("format", {}).get("duration", 0))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise MontageError("FFprobe returned invalid final movie data.") from error
        if "video" not in stream_types or "audio" not in stream_types or duration <= 0:
            raise MontageError(
                "The final file does not contain the expected video, audio, or duration."
            )


def _decimal(value: float) -> str:
    return f"{value:.3f}"


def _seek_start_seconds(source_start_seconds: float) -> float:
    return max(0.0, source_start_seconds - 1.0)


def _audio_trim_filter(has_audio: bool, trim_start: float, duration: str) -> str:
    if has_audio:
        return f"atrim=start={_decimal(trim_start)}:duration={duration},asetpts=PTS-STARTPTS,"
    return f"atrim=0:{duration},asetpts=PTS-STARTPTS,"


def _concat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def _transition_duration(plan: QuickMontagePlan) -> float:
    return 0.0


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
    for index in range(1, clip_count):
        next_video = f"v{index}mix"
        next_audio = f"a{index}mix"
        lines.append(f"[{video_label}][v{index}base]concat=n=2:v=1:a=0[{next_video}]")
        lines.append(f"[{audio_label}][a{index}base]concat=n=2:v=0:a=1[{next_audio}]")
        video_label = next_video
        audio_label = next_audio

    lines.append(f"[{video_label}]format=yuv420p[vout]")
    if plan.music_path is None:
        lines.append(f"[{audio_label}]anull[aout]")
    else:
        music_index = clip_count
        fade_out_start = max(0.0, plan.total_duration_seconds - 1.5)
        duration = plan.total_duration_seconds
        lines.append(
            f"[{music_index}:a]aresample=48000,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            "apad,"
            f"atrim=0:{duration:.3f},"
            "asetpts=PTS-STARTPTS,"
            f"volume={plan.settings.music_volume:.3f},"
            "afade=t=in:st=0:d=1.5,"
            f"afade=t=out:st={fade_out_start:.3f}:d=1.5[music]"
        )
        lines.append(
            f"[{audio_label}]aresample=48000,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            "apad,"
            f"atrim=0:{duration:.3f},"
            "asetpts=PTS-STARTPTS,"
            "volume=0.55[sourceaudio]"
        )
        lines.append(
            "[music][sourceaudio]"
            "sidechaincompress=threshold=0.08:ratio=2.5:attack=35:release=650[duckedmusic]"
        )
        lines.append(
            "[sourceaudio][duckedmusic]"
            "amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
            f"atrim=0:{duration:.3f},"
            "alimiter=limit=0.95[aout]"
        )
    return ";\n".join(lines)


def _replace_video_encoder(command: list[str], encoder_args: list[str]) -> list[str]:
    replaced = list(command)
    index = replaced.index("-c:v")
    end = index + 2
    while end < len(replaced) and replaced[end].startswith("-"):
        if replaced[end] in {"-c:a", "-movflags"}:
            break
        end += 2
    return [*replaced[:index], *encoder_args, *replaced[end:]]
