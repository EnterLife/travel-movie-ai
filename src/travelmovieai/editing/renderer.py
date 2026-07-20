"""FFmpeg rendering for declarative montage plans."""

import hashlib
import json
import os
import subprocess
import textwrap
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from uuid import uuid4

from travelmovieai.core.exceptions import DependencyUnavailableError, MontageError, TravelMovieError
from travelmovieai.core.security import absolute_command_paths, sanitize_process_error
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MontageClip, QuickMontagePlan, QuickMontageSettings
from travelmovieai.infrastructure.artifacts import artifact_fingerprint
from travelmovieai.infrastructure.ffmpeg import FFprobeClient
from travelmovieai.infrastructure.processes import (
    release_process_resources,
    start_process,
    terminate_process_tree,
)
from travelmovieai.infrastructure.system import check_cuda
from travelmovieai.story.editorial import clean_caption, clean_title

ProgressCallback = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]


class _NvencUnavailableError(MontageError):
    """An FFmpeg failure that specifically proves NVENC cannot be initialized."""


class QuickMontageRenderer:
    def __init__(
        self,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        workers: int = 1,
        ffmpeg_threads: int = 1,
        timeout_seconds: float = 7200,
        cancel_requested: CancelCheck | None = None,
    ) -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self.workers = max(1, workers)
        self.ffmpeg_threads = max(1, ffmpeg_threads)
        self.timeout_seconds = timeout_seconds if timeout_seconds > 0 else 7200
        self.cancel_requested = cancel_requested
        self._render_device = "cpu"
        self._encoder = "libx264"
        self._mezzanine_segments = False
        self._heartbeat_callback: Callable[[], None] | None = None

    def render(
        self,
        plan: QuickMontagePlan,
        output_path: Path,
        work_dir: Path,
        progress: ProgressCallback | None = None,
        cancel_requested: CancelCheck | None = None,
    ) -> str:
        previous_cancel = self.cancel_requested
        previous_heartbeat = self._heartbeat_callback
        if cancel_requested is not None:
            self.cancel_requested = cancel_requested
        try:
            return self._render_impl(plan, output_path, work_dir, progress)
        finally:
            self.cancel_requested = previous_cancel
            self._heartbeat_callback = previous_heartbeat

    def _render_impl(
        self,
        plan: QuickMontagePlan,
        output_path: Path,
        work_dir: Path,
        progress: ProgressCallback | None,
    ) -> str:
        _validate_output_target(plan, output_path, work_dir)
        if plan.music_path is not None and not plan.music_path.is_file():
            raise MontageError(f"Soundtrack file does not exist: {plan.music_path}")
        if plan.settings.narration_enabled and plan.narration_path is None:
            raise MontageError("Narration is enabled, but the timeline has no narration audio.")
        if (
            plan.settings.narration_enabled
            and plan.narration_path is not None
            and not plan.narration_path.is_file()
        ):
            raise MontageError(f"Narration file does not exist: {plan.narration_path}")
        _validate_narration_cues(plan)
        self._render_device = plan.settings.render_device
        self._encoder = self._select_encoder(plan.settings.render_device)
        segments_dir = work_dir / "quick_montage_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        total_steps = len(plan.clips) + 1
        tracked_progress = progress
        if progress is not None:
            progress_lock = threading.Lock()
            progress_state: list[int | str] = [0, total_steps, "Preparing FFmpeg render"]

            def report_progress(current: int, total: int, message: str) -> None:
                with progress_lock:
                    progress_state[:] = [current, total, message]
                progress(current, total, message)

            def heartbeat() -> None:
                with progress_lock:
                    current, total, message = progress_state
                assert isinstance(current, int)
                assert isinstance(total, int)
                assert isinstance(message, str)
                progress(current, total, message)

            tracked_progress = report_progress
            self._heartbeat_callback = heartbeat
        else:
            self._heartbeat_callback = None
        try:
            segment_paths = self._render_segments(
                plan,
                segments_dir,
                tracked_progress,
                total_steps,
            )
        except _NvencUnavailableError:
            if self._render_device != "auto" or self._encoder != "h264_nvenc":
                raise
            self._encoder = "libx264"
            segment_paths = self._render_segments(
                plan,
                segments_dir,
                tracked_progress,
                total_steps,
            )

        if tracked_progress:
            tracked_progress(
                len(plan.clips),
                total_steps,
                "Music and final assembly",
            )
        self._compose_segments(segment_paths, plan, output_path, work_dir)
        self._validate_output(output_path, plan)
        if tracked_progress:
            tracked_progress(total_steps, total_steps, "Film ready")
        return self._encoder

    def _render_segments(
        self,
        plan: QuickMontagePlan,
        segments_dir: Path,
        progress: ProgressCallback | None,
        total_steps: int,
    ) -> list[Path]:
        self._mezzanine_segments = _transition_duration(plan) > 0
        segment_encoder = "libx264-lossless" if self._mezzanine_segments else self._encoder
        render_items: list[tuple[MontageClip, Path, bool, bool]] = []
        for index, clip in enumerate(plan.clips):
            show_event_title = _show_event_title(plan, index)
            show_credits = index == len(plan.clips) - 1
            fingerprint = _segment_fingerprint(
                clip,
                plan,
                encoder=segment_encoder,
                show_event_title=show_event_title,
                show_credits=show_credits,
            )
            render_items.append(
                (
                    clip,
                    segments_dir / f"{index + 1:05d}-{fingerprint[:20]}.mp4",
                    show_event_title,
                    show_credits,
                )
            )
        segment_paths = [item[1] for item in render_items]
        cache_validity = {
            item[1]: self._valid_cached_segment(item[1], item[0], plan) for item in render_items
        }
        cached_items = [item for item in render_items if cache_validity[item[1]]]
        pending_items = [
            (index, item)
            for index, item in enumerate(render_items, start=1)
            if not cache_validity[item[1]]
        ]
        for _, path, _, _ in cached_items:
            path.touch()
        worker_count = min(self.workers, len(pending_items))
        if worker_count <= 1:
            for index, (clip, segment_path, show_event_title, show_credits) in enumerate(
                render_items,
                start=1,
            ):
                if cache_validity[segment_path]:
                    if progress:
                        progress(
                            index,
                            total_steps,
                            f"Render cache: clip {index}/{len(plan.clips)}",
                        )
                    continue
                if progress:
                    progress(
                        index - 1,
                        total_steps,
                        f"Rendering clip {index}/{len(plan.clips)}, "
                        f"encoder={segment_encoder}, threads={self.ffmpeg_threads}",
                    )
                self._render_segment_atomic(
                    clip,
                    plan,
                    segment_path,
                    show_event_title=show_event_title,
                    show_credits=show_credits,
                )
            return segment_paths

        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="travelmovieai-render",
        )
        futures: dict[Future[None], int] = {}
        pending_iterator = iter(pending_items)

        def submit_next() -> bool:
            try:
                index, (clip, segment_path, show_event_title, show_credits) = next(pending_iterator)
            except StopIteration:
                return False
            if progress:
                progress(
                    completed_count,
                    total_steps,
                    f"Rendering clip {index}/{len(plan.clips)}, "
                    f"encoder={segment_encoder}, threads={self.ffmpeg_threads}",
                )
            future = executor.submit(
                self._render_segment_atomic,
                clip,
                plan,
                segment_path,
                show_event_title=show_event_title,
                show_credits=show_credits,
            )
            futures[future] = index
            return True

        completed_count = len(cached_items)
        try:
            completed_count = len(cached_items)
            if progress and completed_count:
                progress(
                    completed_count,
                    total_steps,
                    f"Render cache: {completed_count}/{len(plan.clips)} clips",
                )
            for _ in range(worker_count):
                if not submit_next():
                    break
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    future.result()
                    completed_count += 1
                    if progress:
                        progress(
                            completed_count,
                            total_steps,
                            f"Clips complete {completed_count}/{len(plan.clips)}, "
                            f"render workers={worker_count}, encoder={segment_encoder}",
                        )
                    submit_next()
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        return segment_paths

    def _valid_cached_segment(
        self,
        path: Path,
        clip: MontageClip | None = None,
        plan: QuickMontagePlan | None = None,
    ) -> bool:
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
            probe = FFprobeClient(
                self.ffprobe_binary,
                timeout_seconds=min(60, self.timeout_seconds),
            ).probe(path)
        except (OSError, TravelMovieError):
            return False
        stream_types = {
            stream.get("codec_type")
            for stream in probe.metadata.get("streams", [])
            if isinstance(stream, dict)
        }
        basic_valid = (
            "video" in stream_types
            and "audio" in stream_types
            and probe.duration_seconds is not None
            and probe.duration_seconds > 0
        )
        if not basic_valid or clip is None or plan is None:
            return basic_valid
        video_stream = next(
            (
                stream
                for stream in probe.metadata.get("streams", [])
                if isinstance(stream, dict) and stream.get("codec_type") == "video"
            ),
            None,
        )
        audio_stream = next(
            (
                stream
                for stream in probe.metadata.get("streams", [])
                if isinstance(stream, dict) and stream.get("codec_type") == "audio"
            ),
            None,
        )
        transition_mezzanine = _transition_duration(plan) > 0
        return bool(
            abs((probe.duration_seconds or 0) - clip.duration_seconds)
            <= max(0.2, 2 / plan.settings.fps)
            and probe.width == plan.settings.width
            and probe.height == plan.settings.height
            and probe.fps is not None
            and abs(probe.fps - plan.settings.fps) <= 0.05
            and isinstance(video_stream, dict)
            and video_stream.get("pix_fmt") == "yuv420p"
            and _segment_codec_contract_matches(
                video_stream,
                audio_stream,
                transition_mezzanine=transition_mezzanine,
            )
        )

    def _render_segment_atomic(
        self,
        clip: MontageClip,
        plan: QuickMontagePlan,
        output_path: Path,
        *,
        show_event_title: bool = False,
        show_credits: bool = False,
    ) -> None:
        temporary = output_path.with_name(
            f".{output_path.stem}.{uuid4().hex}.tmp{output_path.suffix}"
        )
        try:
            self._render_segment(
                clip,
                plan,
                temporary,
                show_event_title=show_event_title,
                show_credits=show_credits,
            )
            if not self._valid_cached_segment(temporary, clip, plan):
                raise MontageError(
                    f"FFmpeg did not create a valid delivery segment for {clip.relative_path}."
                )
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _render_segment(
        self,
        clip: MontageClip,
        plan: QuickMontagePlan,
        output_path: Path,
        *,
        show_event_title: bool = False,
        show_credits: bool = False,
    ) -> None:
        duration = _decimal(clip.duration_seconds)
        source_audio_fades = _audio_fade_filters(
            clip.duration_seconds,
            plan.settings.source_audio_fade_seconds,
        )

        if clip.media_type is MediaType.PHOTO:
            video_graph = _build_segment_video_graph(
                clip,
                plan,
                trim_start=0,
                show_event_title=show_event_title,
                show_credits=show_credits,
            )
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
                "-noautorotate",
                "-i",
                str(clip.source_path),
                "-f",
                "lavfi",
                "-t",
                duration,
                "-i",
                "anullsrc=r=48000:cl=stereo",
                "-filter_complex",
                f"{video_graph};[1:a:0]atrim=0:{duration},"
                f"{source_audio_fades}"
                "asetpts=PTS-STARTPTS[a]",
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
            video_graph = _build_segment_video_graph(
                clip,
                plan,
                trim_start=trim_start,
                show_event_title=show_event_title,
                show_credits=show_credits,
            )
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
                "-noautorotate",
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
                    f"{video_graph};"
                    f"[{audio_input}]"
                    f"{_audio_trim_filter(clip.has_audio, trim_start, duration)}"
                    "aresample=48000,"
                    f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                    f"apad,atrim=0:{duration},"
                    f"{source_audio_fades}"
                    "asetpts=PTS-STARTPTS[a]",
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
                *self._segment_video_encoder_args(),
                "-c:a",
                "alac",
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
        if transition_duration <= 0:
            self._compose_hard_cut_segments(segments, plan, output_path, work_dir)
            return

        narration_path = _active_narration_path(plan)

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
        if narration_path is not None:
            command.extend(["-i", str(narration_path)])
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
                "-ar",
                "48000",
                "-ac",
                "2",
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
            except _NvencUnavailableError:
                if self._render_device != "auto" or self._encoder != "h264_nvenc":
                    raise
                self._encoder = "libx264"
                command = _replace_video_encoder(command, self._video_encoder_args())
                self._run(command, "Could not assemble clips and music")
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)
            filter_path.unlink(missing_ok=True)

    def _compose_hard_cut_segments(
        self,
        segments: list[Path],
        plan: QuickMontagePlan,
        output_path: Path,
        work_dir: Path,
    ) -> None:
        """Mix delivery audio while stream-copying already prepared hard-cut video."""

        concat_path = work_dir / "quick_montage_concat.txt"
        filter_path = work_dir / "quick_montage_audio_filters.txt"
        concat_path.write_text(
            "".join(f"file '{_concat_path(path)}'\n" for path in segments),
            encoding="utf-8",
        )
        filter_path.write_text(_build_hard_cut_audio_graph(plan), encoding="utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_name(f".{output_path.stem}.{uuid4().hex}.tmp.mp4")
        command = [
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
        ]
        if plan.music_path is not None:
            command.extend(["-stream_loop", "-1", "-i", str(plan.music_path)])
        narration_path = _active_narration_path(plan)
        if narration_path is not None:
            command.extend(["-i", str(narration_path)])
        command.extend(
            [
                "-filter_complex_script",
                str(filter_path),
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-t",
                _decimal(plan.total_duration_seconds),
                "-c:v",
                "copy",
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
                str(temporary_output),
            ]
        )
        try:
            self._run(command, "Could not mix hard-cut montage audio")
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)
            concat_path.unlink(missing_ok=True)
            filter_path.unlink(missing_ok=True)

    def _run(self, command: list[str], message: str) -> None:
        try:
            process = start_process(
                command,
                popen_factory=subprocess.Popen,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                text=True,
            )
        except FileNotFoundError as error:
            raise DependencyUnavailableError(
                f"FFmpeg executable was not found: {self.ffmpeg_binary}"
            ) from error
        except OSError as error:
            raise MontageError(f"{message}: FFmpeg could not start.") from error

        try:
            stderr_lines: deque[str] = deque(maxlen=400)
            stderr_stream = process.stderr
            assert stderr_stream is not None

            def read_stderr() -> None:
                for line in stderr_stream:
                    stderr_lines.append(line)

            reader = threading.Thread(
                target=read_stderr,
                name="travelmovieai-ffmpeg-stderr",
                daemon=True,
            )
            reader.start()
            deadline = time.monotonic() + self.timeout_seconds
            next_heartbeat = time.monotonic() + 0.5
            while process.poll() is None:
                if self.cancel_requested is not None and self.cancel_requested():
                    terminate_process_tree(process)
                    reader.join(timeout=1)
                    raise MontageError(f"{message}: FFmpeg rendering was cancelled.")
                if time.monotonic() >= deadline:
                    terminate_process_tree(process)
                    reader.join(timeout=1)
                    raise MontageError(
                        f"{message}: FFmpeg timed out after {self.timeout_seconds:g}s."
                    )
                if self._heartbeat_callback is not None and time.monotonic() >= next_heartbeat:
                    try:
                        self._heartbeat_callback()
                    except BaseException:
                        terminate_process_tree(process)
                        reader.join(timeout=1)
                        raise
                    next_heartbeat = time.monotonic() + 0.5
                time.sleep(0.1)
            reader.join(timeout=2)
            stderr = "".join(stderr_lines)
            if process.returncode != 0:
                encoder_unavailable = _is_nvenc_unavailable_error(command, stderr)
                detail = sanitize_process_error(
                    stderr,
                    private_paths=absolute_command_paths(command),
                    fallback="unknown FFmpeg error",
                )
                error_type = _NvencUnavailableError if encoder_unavailable else MontageError
                raise error_type(f"{message}: {detail}")
        finally:
            release_process_resources(process)

    def _select_encoder(self, render_device: str) -> str:
        cuda = check_cuda(self.ffmpeg_binary)
        if not cuda.available or not cuda.ffmpeg_nvenc:
            if render_device == "cuda":
                raise DependencyUnavailableError(
                    "CUDA rendering was selected, but NVIDIA GPU or h264_nvenc is unavailable."
                )
            return "libx264"
        if render_device == "cuda":
            return "h264_nvenc"
        if render_device == "auto":
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
                "-pix_fmt",
                "yuv420p",
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
            "-pix_fmt",
            "yuv420p",
        ]

    def _segment_video_encoder_args(self) -> list[str]:
        if not self._mezzanine_segments:
            return self._video_encoder_args()
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-qp",
            "0",
            "-threads",
            str(self.ffmpeg_threads),
            "-pix_fmt",
            "yuv420p",
        ]

    def _validate_output(self, output_path: Path, plan: QuickMontagePlan) -> None:
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
            detail = sanitize_process_error(
                completed.stderr,
                private_paths=[output_path],
                fallback="unknown FFprobe error",
            )
            raise MontageError(f"The final movie failed FFprobe validation: {detail}")
        try:
            payload = json.loads(completed.stdout)
            streams = payload.get("streams", [])
            if not isinstance(streams, list) or not all(
                isinstance(stream, dict) for stream in streams
            ):
                raise TypeError("FFprobe streams must be a list")
            stream_types = {stream.get("codec_type") for stream in streams}
            duration = float(payload.get("format", {}).get("duration", 0))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise MontageError("FFprobe returned invalid final movie data.") from error
        if "video" not in stream_types or "audio" not in stream_types or duration <= 0:
            raise MontageError(
                "The final file does not contain the expected video, audio, or duration."
            )
        if not _probe_payload_matches_plan(payload, plan):
            raise MontageError(
                "The final movie does not match the planned duration, resolution, frame rate, "
                "or yuv420p delivery pixel format."
            )
        if plan.settings.validate_full_render_decode:
            self._validate_full_decode(output_path)

    def _validate_full_decode(self, output_path: Path) -> None:
        self._run(
            [
                self.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(output_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            "The final movie failed full decode validation",
        )


def _segment_fingerprint(
    clip: MontageClip,
    plan: QuickMontagePlan,
    *,
    encoder: str,
    show_event_title: bool,
    show_credits: bool,
) -> str:
    try:
        stat = clip.source_path.stat()
        source_state: dict[str, object] = {
            "size": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }
    except OSError:
        source_state = {"missing": True}
    return artifact_fingerprint(
        "render-segment-v12-native-overlay-linebreaks",
        clip,
        plan.settings,
        overlay_font_revision(plan.settings),
        {
            "encoder": encoder,
            "show_event_title": show_event_title,
            "show_credits": show_credits,
            "source_state": source_state,
        },
    )


def _segment_codec_contract_matches(
    video_stream: dict[str, object],
    audio_stream: object,
    *,
    transition_mezzanine: bool,
) -> bool:
    if not isinstance(audio_stream, dict):
        return False
    if video_stream.get("codec_name") != "h264" or audio_stream.get("codec_name") != "alac":
        return False
    if not transition_mezzanine:
        return True
    profile = video_stream.get("profile")
    return isinstance(profile, str) and profile.casefold() == "high 4:4:4 predictive"


def _decimal(value: float) -> str:
    return f"{value:.3f}"


def _is_nvenc_unavailable_error(command: list[str], stderr: str | None) -> bool:
    if "h264_nvenc" not in command:
        return False
    detail = (stderr or "").casefold()
    markers = (
        "no nvenc capable devices found",
        "cannot load nvcuda",
        "cannot load libcuda",
        "cannot load nvencodeapi",
        "cannot load libnvidia-encode",
        "driver does not support the required nvenc api version",
        "minimum required nvidia driver",
        "openencodesessionex failed",
        "failed to initialize nvenc",
        "failed to open nvenc codec",
        "cannot init cuda",
        "provided device doesn't support required nvenc features",
    )
    return any(marker in detail for marker in markers)


def _probe_payload_matches_plan(payload: dict[str, object], plan: QuickMontagePlan) -> bool:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return False
    video = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        None,
    )
    if not isinstance(video, dict):
        return False
    format_payload = payload.get("format")
    if not isinstance(format_payload, dict):
        return False
    try:
        duration = float(format_payload.get("duration", 0))
        width = int(video.get("width", 0))
        height = int(video.get("height", 0))
    except (TypeError, ValueError):
        return False
    fps = _parse_ffmpeg_rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    tolerance = max(0.25, 3 / plan.settings.fps)
    return (
        abs(duration - plan.total_duration_seconds) <= tolerance
        and width == plan.settings.width
        and height == plan.settings.height
        and fps is not None
        and abs(fps - plan.settings.fps) <= 0.05
        and video.get("pix_fmt") == "yuv420p"
    )


def _parse_ffmpeg_rate(value: object) -> float | None:
    if not isinstance(value, str) or not value or value == "0/0":
        return None
    try:
        if "/" not in value:
            return float(value)
        numerator, denominator = value.split("/", maxsplit=1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _validate_narration_cues(plan: QuickMontagePlan) -> None:
    if not plan.narration_cues:
        return
    previous_end = 0.0
    for index, cue in enumerate(plan.narration_cues):
        if cue.line_index != index or cue.cue_start_seconds < previous_end - 0.01:
            raise MontageError("Timed narration cues are out of order or overlap.")
        if cue.cue_end_seconds > plan.total_duration_seconds + 0.05:
            raise MontageError("Timed narration exceeds the planned movie duration.")
        if not cue.audio_path.is_file() or cue.audio_path.stat().st_size <= 0:
            raise MontageError("Timed narration references a missing line audio file.")
        previous_end = cue.cue_end_seconds


def _validate_output_target(
    plan: QuickMontagePlan,
    output_path: Path,
    work_dir: Path,
) -> None:
    output_key = _path_key(output_path)
    input_paths = [clip.source_path for clip in plan.clips]
    if plan.music_path is not None:
        input_paths.append(plan.music_path)
    narration_path = _active_narration_path(plan)
    if narration_path is not None:
        input_paths.append(narration_path)
    if any(_path_key(path) == output_key for path in input_paths):
        raise MontageError("Movie output must not overwrite source media or the soundtrack.")
    reserved_paths = (
        work_dir / "quick_montage_filters.txt",
        work_dir / "quick_montage_concat.txt",
    )
    if any(_path_key(path) == output_key for path in reserved_paths) or _is_within(
        output_path,
        work_dir / "quick_montage_segments",
    ):
        raise MontageError("Movie output conflicts with the renderer working files.")


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve()))


def _is_within(path: Path, root: Path) -> bool:
    normalized_path = Path(_path_key(path))
    normalized_root = Path(_path_key(root))
    return normalized_path.is_relative_to(normalized_root)


def _seek_start_seconds(source_start_seconds: float) -> float:
    return max(0.0, source_start_seconds - 1.0)


def _audio_trim_filter(has_audio: bool, trim_start: float, duration: str) -> str:
    if has_audio:
        return f"atrim=start={_decimal(trim_start)}:duration={duration},asetpts=PTS-STARTPTS,"
    return f"atrim=0:{duration},asetpts=PTS-STARTPTS,"


def _audio_fade_filters(duration_seconds: float, requested_seconds: float) -> str:
    fade = min(requested_seconds, duration_seconds * 0.45)
    if fade <= 0:
        return ""
    fade_out_start = max(0.0, duration_seconds - fade)
    return f"afade=t=in:st=0:d={fade:.3f},afade=t=out:st={fade_out_start:.3f}:d={fade:.3f},"


def _build_segment_video_graph(
    clip: MontageClip,
    plan: QuickMontagePlan,
    *,
    trim_start: float,
    show_event_title: bool,
    show_credits: bool,
) -> str:
    settings = plan.settings
    duration = clip.duration_seconds
    lines: list[str] = []
    if clip.media_type is MediaType.PHOTO:
        lines.append("[0:v:0]setpts=PTS-STARTPTS[vsource]")
    else:
        lines.append(
            f"[0:v:0]trim=start={_decimal(trim_start)}:duration={_decimal(duration)},"
            "setpts=PTS-STARTPTS[vsource]"
        )

    label = "vsource"
    preparation = [*_rotation_filters(clip)]
    if settings.hdr_to_sdr and _is_hdr(clip.color_transfer):
        preparation.extend(
            [
                "zscale=transfer=linear:npl=100",
                "tonemap=mobius:desat=0",
                "zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=tv",
            ]
        )
    if settings.color_normalization:
        preparation.append(
            "eq="
            f"contrast={clip.contrast_multiplier:.4f}:"
            f"brightness={clip.brightness_adjustment:.4f}:"
            f"saturation={clip.saturation_multiplier:.4f}"
        )
    if preparation:
        lines.append(f"[{label}]{','.join(preparation)}[vprepared]")
        label = "vprepared"

    framing = _resolved_framing(clip, settings)
    if framing == "blur":
        lines.extend(
            [
                f"[{label}]split=2[vbackgroundsource][vforegroundsource]",
                f"[vbackgroundsource]scale={settings.width}:{settings.height}:"
                "force_original_aspect_ratio=increase,"
                f"crop={settings.width}:{settings.height},boxblur=20:1[vbackground]",
                f"[vforegroundsource]scale={settings.width}:{settings.height}:"
                "force_original_aspect_ratio=decrease[vforeground]",
                "[vbackground][vforeground]overlay=(W-w)/2:(H-h)/2[vframed]",
            ]
        )
    elif framing == "crop":
        crop_x, crop_y = _crop_expressions(clip)
        lines.append(
            f"[{label}]scale={settings.width}:{settings.height}:"
            "force_original_aspect_ratio=increase,"
            f"crop={settings.width}:{settings.height}:x='{crop_x}':y='{crop_y}'[vframed]"
        )
    else:
        lines.append(
            f"[{label}]scale={settings.width}:{settings.height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={settings.width}:{settings.height}:(ow-iw)/2:(oh-ih)/2:black[vframed]"
        )
    label = "vframed"

    if clip.media_type is MediaType.PHOTO and settings.photo_motion == "ken_burns":
        frame_count = max(1, int(round(duration * settings.fps)))
        zoom_step = (settings.photo_zoom_ratio - 1.0) / max(1, frame_count - 1)
        focus_x = clip.focus_x if clip.focus_x is not None else 0.5
        focus_y = clip.focus_y if clip.focus_y is not None else 0.5
        lines.append(
            f"[{label}]zoompan=z='min(zoom+{zoom_step:.8f},{settings.photo_zoom_ratio:.4f})':"
            f"x='max(0,min(iw-iw/zoom,iw*{focus_x:.4f}-(iw/zoom)/2))':"
            f"y='max(0,min(ih-ih/zoom,ih*{focus_y:.4f}-(ih/zoom)/2))':"
            f"d=1:s={settings.width}x{settings.height}:fps={settings.fps}[vmotion]"
        )
        label = "vmotion"

    lines.append(f"[{label}]fps={settings.fps},setsar=1[vtimed]")
    label = "vtimed"
    overlays = _overlay_filters(
        clip,
        settings,
        duration=duration,
        show_event_title=show_event_title,
        show_credits=show_credits,
    )
    if overlays:
        lines.append(f"[{label}]{','.join(overlays)}[voverlay]")
        label = "voverlay"
    lines.append(f"[{label}]setparams=range=tv,format=yuv420p[v]")
    return ";".join(lines)


def _rotation_filters(clip: MontageClip) -> list[str]:
    if clip.rotation_degrees == 90:
        return ["transpose=cclock"]
    if clip.rotation_degrees == 180:
        return ["hflip", "vflip"]
    if clip.rotation_degrees == 270:
        return ["transpose=clock"]
    return []


def _resolved_framing(clip: MontageClip, settings: QuickMontageSettings) -> str:
    width, height = _display_dimensions(clip)
    if width is not None and height is not None and height > width:
        if settings.vertical_video_layout == "blur":
            return "blur"
        if settings.vertical_video_layout == "crop":
            return "crop"
        return "fit"
    if settings.framing_mode == "fill":
        return "crop"
    if settings.framing_mode == "smart":
        return "crop" if clip.focus_x is not None and clip.focus_y is not None else "fit"
    return "fit"


def _display_dimensions(clip: MontageClip) -> tuple[int | None, int | None]:
    width, height = clip.source_width, clip.source_height
    if clip.rotation_degrees in {90, 270}:
        return height, width
    return width, height


def _crop_expressions(clip: MontageClip) -> tuple[str, str]:
    focus_x = clip.focus_x if clip.focus_x is not None else 0.5
    focus_y = clip.focus_y if clip.focus_y is not None else 0.5
    return (
        f"max(0,min(iw-ow,iw*{focus_x:.4f}-ow/2))",
        f"max(0,min(ih-oh,ih*{focus_y:.4f}-oh/2))",
    )


def _is_hdr(color_transfer: str | None) -> bool:
    return (color_transfer or "").strip().casefold() in {
        "smpte2084",
        "arib-std-b67",
        "pq",
        "hlg",
    }


def _overlay_filters(
    clip: MontageClip,
    settings: QuickMontageSettings,
    *,
    duration: float,
    show_event_title: bool,
    show_credits: bool,
) -> list[str]:
    filters: list[str] = []
    margin = settings.overlay_safe_margin
    font = _drawtext_font_option(settings)
    credits_start = (
        max(0.0, duration - settings.credits_duration_seconds)
        if settings.credits_text and show_credits
        else duration
    )
    event_title = clean_title(clip.event_title)
    if settings.event_titles_enabled and show_event_title and event_title:
        title = _escape_drawtext(
            _truncate_overlay_text(event_title, settings, font_height_divisor=18)
        )
        title_end = min(duration, 2.8, credits_start)
        if title_end > 0.05:
            filters.append(
                f"drawtext=text='{title}':{font}fontcolor=white:fontsize=h/18:"
                "box=1:boxcolor=black@0.55:boxborderw=12:fix_bounds=1:"
                f"x='max(w*{margin:.3f},min(w-tw-w*{margin:.3f},w*{margin:.3f}))':"
                f"y='h*{margin:.3f}':enable='between(t,0,{title_end:.3f})'"
            )
    caption_text = _caption_overlay_text(clip.caption, settings, duration=duration)
    if settings.scene_subtitles_enabled and caption_text and credits_start > 0.05:
        caption = _escape_drawtext(caption_text)
        caption_enable = (
            "" if credits_start >= duration else f":enable='between(t,0,{credits_start:.3f})'"
        )
        filters.append(
            f"drawtext=text='{caption}':{font}fontcolor=white:fontsize=h/24:"
            "box=1:boxcolor=black@0.60:boxborderw=10:fix_bounds=1:"
            f"x='max(w*{margin:.3f},min(w-tw-w*{margin:.3f},(w-tw)/2))':"
            f"y='h-th-h*{margin:.3f}'{caption_enable}"
        )
    if settings.credits_text and show_credits:
        credits = _escape_drawtext(
            _truncate_overlay_text(settings.credits_text, settings, font_height_divisor=20)
        )
        credits_start = max(0.0, duration - settings.credits_duration_seconds)
        filters.append(
            f"drawtext=text='{credits}':{font}fontcolor=white:fontsize=h/20:"
            "box=1:boxcolor=black@0.65:boxborderw=14:fix_bounds=1:"
            f"x='max(w*{margin:.3f},min(w-tw-w*{margin:.3f},(w-tw)/2))':"
            f"y='max(h*{margin:.3f},min(h-th-h*{margin:.3f},(h-th)/2))':"
            f"enable='between(t,{credits_start:.3f},{duration:.3f})'"
        )
    return filters


def _truncate_overlay_text(
    text: str,
    settings: QuickMontageSettings,
    *,
    font_height_divisor: int,
) -> str:
    normalized = " ".join(_portable_overlay_text(text).split())
    safe_width = settings.width * (1 - 2 * settings.overlay_safe_margin)
    estimated_character_width = (settings.height / font_height_divisor) * 0.62
    safe_character_limit = max(1, int(safe_width / estimated_character_width))
    character_limit = min(settings.overlay_max_characters, safe_character_limit)
    if len(normalized) <= character_limit:
        return normalized
    suffix = "..."
    if character_limit <= len(suffix):
        return suffix[:character_limit]

    body_limit = character_limit - len(suffix)
    body = normalized[:body_limit].rstrip()
    word_boundary = body.rfind(" ")
    if word_boundary >= body_limit // 2:
        body = body[:word_boundary].rstrip()
    return body + suffix


def _caption_overlay_text(
    text: object,
    settings: QuickMontageSettings,
    *,
    duration: float,
) -> str | None:
    read_limit = max(20, int(duration * settings.caption_characters_per_second))
    editorial = clean_caption(
        text,
        max_characters=min(settings.overlay_max_characters, read_limit),
    )
    if editorial is None:
        return None
    normalized = " ".join(_portable_overlay_text(editorial).split())
    safe_width = settings.width * (1 - 2 * settings.overlay_safe_margin)
    character_width = (settings.height / 24) * 0.62
    per_line = max(8, int(safe_width / character_width))
    total_limit = min(settings.overlay_max_characters, read_limit, per_line * 2)
    normalized = _truncate_text_to_limit(normalized, total_limit)
    lines = textwrap.wrap(
        normalized,
        width=per_line,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(lines) > 2:
        lines = [lines[0], _truncate_text_to_limit(" ".join(lines[1:]), per_line)]
    return "\n".join(lines)


def _truncate_text_to_limit(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "." * limit
    body = text[: limit - 3].rstrip()
    boundary = body.rfind(" ")
    if boundary >= (limit - 3) // 2:
        body = body[:boundary].rstrip()
    return f"{body}..."


def _drawtext_font_option(settings: QuickMontageSettings) -> str:
    font_path = _resolve_overlay_font(settings.overlay_font_path)
    if font_path is None:
        return ""
    escaped = (
        font_path.resolve().as_posix().replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")
    )
    return f"fontfile='{escaped}':"


def _resolve_overlay_font(configured: Path | None) -> Path | None:
    if configured is not None:
        resolved = configured.expanduser().resolve()
        if not resolved.is_file():
            raise MontageError(
                f"Configured overlay font does not exist: {configured.name or '<unnamed-font>'}"
            )
        return resolved
    project_root = Path(__file__).resolve().parents[3]
    candidates = (
        project_root / "assets" / "fonts" / "DejaVuSans.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    return next((path for path in candidates if path.is_file()), None)


def overlay_font_revision(settings: QuickMontageSettings) -> dict[str, object] | None:
    """Return the selected overlay font identity for render cache keys."""

    if not (
        settings.event_titles_enabled or settings.scene_subtitles_enabled or settings.credits_text
    ):
        return None
    font_path = _resolve_overlay_font(settings.overlay_font_path)
    if font_path is None:
        return {"provider": "ffmpeg-default"}
    try:
        stat = font_path.stat()
        digest = hashlib.sha256()
        with font_path.open("rb") as font_file:
            for chunk in iter(lambda: font_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise MontageError("Could not inspect the selected overlay font.") from error
    return {
        "path": font_path,
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "content_sha256": digest.hexdigest(),
    }


def _portable_overlay_text(text: str) -> str:
    translations: dict[str, str | int | None] = {
        "\u00a0": " ",
        "\u00b7": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2026": "...",
        "\u2027": "-",
        "\u2212": "-",
    }
    return text.translate(str.maketrans(translations))


def _escape_drawtext(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return (
        normalized.replace("\\", "\\\\")
        .replace("'", "'\\''")
        .replace(":", "\\:")
        .replace("%", "\\%")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
        .replace(";", "\\;")
    )


def _show_event_title(plan: QuickMontagePlan, clip_index: int) -> bool:
    clip = plan.clips[clip_index]
    title = clean_title(clip.event_title)
    if title is None:
        return False
    if clip_index == 0:
        return True
    normalized_title = title.casefold()
    if any(
        (previous_title := clean_title(previous.event_title)) is not None
        and previous_title.casefold() == normalized_title
        for previous in plan.clips[:clip_index]
    ):
        return False
    previous = plan.clips[clip_index - 1]
    return previous.event_id != clip.event_id or clean_title(previous.event_title) != title


def _active_narration_path(plan: QuickMontagePlan) -> Path | None:
    if not plan.settings.narration_enabled:
        return None
    return plan.narration_path


def _concat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def _transition_duration(plan: QuickMontagePlan) -> float:
    if (
        plan.settings.transition == "none"
        or len(plan.clips) < 2
        or not any(
            _selected_transition(plan, index) is not None for index in range(1, len(plan.clips))
        )
    ):
        return 0.0
    return min(
        plan.settings.transition_duration_seconds,
        min(clip.duration_seconds for clip in plan.clips) * 0.45,
    )


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
        transition = _selected_transition(plan, index)
        if transition_duration > 0 and transition is not None:
            offset = max(0.0, elapsed - transition_duration)
            lines.append(
                f"[{video_label}][v{index}base]"
                f"xfade=transition={transition}:duration={transition_duration:.3f}:"
                f"offset={offset:.3f}[{next_video}]"
            )
            lines.append(
                f"[{audio_label}][a{index}base]"
                f"acrossfade=d={transition_duration:.3f}:c1=tri:c2=tri[{next_audio}]"
            )
            elapsed += plan.clips[index].duration_seconds - transition_duration
        else:
            lines.append(f"[{video_label}][v{index}base]concat=n=2:v=1:a=0[{next_video}]")
            lines.append(f"[{audio_label}][a{index}base]concat=n=2:v=0:a=1[{next_audio}]")
            elapsed += plan.clips[index].duration_seconds
        video_label = next_video
        audio_label = next_audio

    lines.append(f"[{video_label}]format=yuv420p[vout]")
    _append_audio_mix(
        lines,
        plan,
        audio_label=audio_label,
        music_index=clip_count if plan.music_path is not None else None,
        narration_index=(
            clip_count + (1 if plan.music_path is not None else 0)
            if _active_narration_path(plan) is not None
            else None
        ),
    )
    return ";\n".join(lines)


def _build_hard_cut_audio_graph(plan: QuickMontagePlan) -> str:
    lines: list[str] = []
    music_index = 1 if plan.music_path is not None else None
    narration_index = (
        1 + (1 if plan.music_path is not None else 0)
        if _active_narration_path(plan) is not None
        else None
    )
    _append_audio_mix(
        lines,
        plan,
        audio_label="0:a:0",
        music_index=music_index,
        narration_index=narration_index,
    )
    return ";\n".join(lines)


def _append_audio_mix(
    lines: list[str],
    plan: QuickMontagePlan,
    *,
    audio_label: str,
    music_index: int | None,
    narration_index: int | None,
) -> None:
    duration = plan.total_duration_seconds
    narration_path = _active_narration_path(plan)
    lines.append(
        f"[{audio_label}]aresample=48000,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "apad,"
        f"atrim=0:{duration:.3f},"
        "asetpts=PTS-STARTPTS,"
        f"volume={plan.settings.source_audio_volume:.3f}[sourceaudio]"
    )
    background_label = "sourceaudio"
    if plan.music_path is not None:
        assert music_index is not None
        music_fade = min(plan.settings.music_fade_seconds, duration * 0.45)
        volume = _music_volume_filter(plan)
        fades = _audio_fade_filters(duration, music_fade)
        lines.append(
            f"[{music_index}:a]aresample=48000,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            "apad,"
            f"atrim=0:{duration:.3f},"
            "asetpts=PTS-STARTPTS,"
            f"{volume},"
            f"{fades}anull[music]"
        )
        lines.append(
            "[music][sourceaudio]"
            "sidechaincompress=threshold=0.08:ratio=2.5:attack=35:release=650[duckedmusic]"
        )
        lines.append(
            "[sourceaudio][duckedmusic]"
            "amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
            f"atrim=0:{duration:.3f}[background]"
        )
        background_label = "background"

    if narration_path is None:
        _append_delivery_audio(lines, plan, background_label)
        return

    assert narration_index is not None
    lines.append(
        f"[{narration_index}:a]aresample=48000,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "apad,"
        f"atrim=0:{duration:.3f},"
        "asetpts=PTS-STARTPTS,"
        f"{_narration_volume_filter(plan)},"
        "asplit=2[narrationsc][narrationmix]"
    )
    duck_ratio = _narration_duck_ratio(plan.settings.background_volume_during_narration)
    lines.append(
        f"[{background_label}][narrationsc]"
        f"sidechaincompress=threshold=0.015:ratio={duck_ratio:.2f}:"
        "attack=15:release=350[duckedbackground]"
    )
    lines.append(
        "[duckedbackground][narrationmix]"
        "amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
        f"atrim=0:{duration:.3f}[premaster]"
    )
    _append_delivery_audio(lines, plan, "premaster")


def _append_delivery_audio(
    lines: list[str],
    plan: QuickMontagePlan,
    input_label: str,
) -> None:
    settings = plan.settings
    duration = plan.total_duration_seconds
    fade = min(settings.final_audio_fade_seconds, duration * 0.45)
    filters = [
        (
            "loudnorm="
            f"I={settings.delivery_loudness_lufs:.1f}:"
            f"TP={settings.delivery_true_peak_dbfs:.1f}:LRA=11:linear=true"
        ),
        "aresample=48000",
        "aformat=sample_fmts=fltp:channel_layouts=stereo",
        (
            f"alimiter=limit={10 ** (settings.delivery_true_peak_dbfs / 20):.6f}:"
            "level=false:latency=true"
        ),
    ]
    if fade > 0:
        filters.extend(
            [
                f"afade=t=in:st=0:d={fade:.3f}",
                f"afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade:.3f}",
            ]
        )
    lines.append(f"[{input_label}]{','.join(filters)}[aout]")


def _narration_volume_filter(plan: QuickMontagePlan) -> str:
    base_volume = plan.settings.narration_volume
    if not plan.narration_cues:
        return f"volume={base_volume:.3f}"
    expressions: list[str] = []
    for cue in plan.narration_cues:
        fade = min(plan.settings.narration_fade_seconds, cue.duration_seconds * 0.45)
        start = cue.cue_start_seconds
        end = cue.cue_end_seconds
        if fade <= 0:
            expressions.append(f"between(t,{start:.3f},{end:.3f})")
            continue
        expressions.append(
            f"if(between(t,{start:.3f},{start + fade:.3f}),"
            f"(t-{start:.3f})/{fade:.3f},"
            f"if(between(t,{end - fade:.3f},{end:.3f}),"
            f"({end:.3f}-t)/{fade:.3f},between(t,{start:.3f},{end:.3f})))"
        )
    envelope = expressions[0]
    for expression in expressions[1:]:
        envelope = f"max({envelope},{expression})"
    return f"volume='{base_volume:.3f}*({envelope})':eval=frame"


def _music_volume_filter(plan: QuickMontagePlan) -> str:
    base_volume = plan.settings.music_volume
    music_plan = plan.music_plan
    if not plan.settings.music_volume_envelope or music_plan is None or not music_plan.cue_sections:
        return f"volume={base_volume:.3f}"
    expression = "0.700"
    for section in reversed(music_plan.cue_sections):
        section_gain = 0.65 + 0.35 * section.intensity
        expression = (
            f"if(between(t,{section.start_seconds:.3f},{section.end_seconds:.3f}),"
            f"{section_gain:.3f},{expression})"
        )
    return f"volume='{base_volume:.3f}*({expression})':eval=frame"


def _narration_duck_ratio(background_volume: float) -> float:
    if background_volume >= 0.999:
        return 1.0
    return min(20.0, max(1.0, 1.0 + (1.0 - background_volume) * 19.0))


def _selected_transition(plan: QuickMontagePlan, clip_index: int) -> str | None:
    settings = plan.settings
    if settings.transition == "none" or settings.transition_duration_seconds <= 0:
        return None
    if settings.transition == "cinematic":
        return "fadeblack" if plan.clips[clip_index].transition == "fade" else None
    if settings.transition == "fade":
        return "fadeblack"
    if settings.transition in {"wipeleft", "slideright"}:
        return settings.transition
    return None


def _replace_video_encoder(command: list[str], encoder_args: list[str]) -> list[str]:
    replaced = list(command)
    index = replaced.index("-c:v")
    end = index + 2
    while end < len(replaced) and replaced[end].startswith("-"):
        if replaced[end] in {"-c:a", "-movflags"}:
            break
        end += 2
    return [*replaced[:index], *encoder_args, *replaced[end:]]
