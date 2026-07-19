"""FFmpeg rendering for declarative montage plans."""

import json
import os
import subprocess
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
        _validate_output_target(plan, output_path, work_dir)
        if plan.music_path is not None and not plan.music_path.is_file():
            raise MontageError(f"Soundtrack file does not exist: {plan.music_path}")
        if (
            plan.settings.narration_enabled
            and plan.narration_path is not None
            and not plan.narration_path.is_file()
        ):
            raise MontageError(f"Narration file does not exist: {plan.narration_path}")
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
        render_items: list[tuple[MontageClip, Path, bool, bool]] = []
        for index, clip in enumerate(plan.clips):
            show_event_title = _show_event_title(plan, index)
            show_credits = index == len(plan.clips) - 1
            fingerprint = _segment_fingerprint(
                clip,
                plan,
                encoder=self._encoder,
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
        cache_validity = {item[1]: self._valid_cached_segment(item[1]) for item in render_items}
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
                        f"encoder={self._encoder}, threads={self.ffmpeg_threads}",
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
                    f"encoder={self._encoder}, threads={self.ffmpeg_threads}",
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
                            f"render workers={worker_count}, encoder={self._encoder}",
                        )
                    submit_next()
        finally:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        return segment_paths

    def _valid_cached_segment(self, path: Path) -> bool:
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
        return (
            "video" in stream_types
            and "audio" in stream_types
            and probe.duration_seconds is not None
            and probe.duration_seconds > 0
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
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise MontageError(
                    f"FFmpeg did not create a complete segment for {clip.relative_path}."
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
                f"{video_graph};[1:a:0]atrim=0:{duration},asetpts=PTS-STARTPTS[a]",
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
        narration_path = _active_narration_path(plan)
        if transition_duration <= 0 and plan.music_path is None and narration_path is None:
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
            detail = sanitize_process_error(
                completed.stderr,
                private_paths=absolute_command_paths(command),
                fallback="unknown FFmpeg error",
            )
            raise MontageError(f"{message}: {detail}")

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
            detail = sanitize_process_error(
                completed.stderr,
                private_paths=[output_path],
                fallback="unknown FFprobe error",
            )
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
        "render-segment-v4-portable-overlays",
        clip,
        plan.settings,
        {
            "encoder": encoder,
            "show_event_title": show_event_title,
            "show_credits": show_credits,
            "source_state": source_state,
        },
    )


def _decimal(value: float) -> str:
    return f"{value:.3f}"


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
    lines.append(f"[{label}]format=yuv420p[v]")
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
    if settings.event_titles_enabled and show_event_title and clip.event_title:
        title = _escape_drawtext(
            _truncate_overlay_text(clip.event_title, settings, font_height_divisor=18)
        )
        title_end = min(duration, 2.8)
        filters.append(
            f"drawtext=text='{title}':fontcolor=white:fontsize=h/18:"
            "box=1:boxcolor=black@0.55:boxborderw=12:fix_bounds=1:"
            f"x='max(w*{margin:.3f},min(w-tw-w*{margin:.3f},w*{margin:.3f}))':"
            f"y='h*{margin:.3f}':enable='between(t,0,{title_end:.3f})'"
        )
    if settings.scene_subtitles_enabled and clip.caption:
        caption = _escape_drawtext(
            _truncate_overlay_text(clip.caption, settings, font_height_divisor=24)
        )
        filters.append(
            f"drawtext=text='{caption}':fontcolor=white:fontsize=h/24:"
            "box=1:boxcolor=black@0.60:boxborderw=10:fix_bounds=1:"
            f"x='max(w*{margin:.3f},min(w-tw-w*{margin:.3f},(w-tw)/2))':"
            f"y='h-th-h*{margin:.3f}'"
        )
    if settings.credits_text and show_credits:
        credits = _escape_drawtext(
            _truncate_overlay_text(settings.credits_text, settings, font_height_divisor=20)
        )
        credits_start = max(0.0, duration - settings.credits_duration_seconds)
        filters.append(
            f"drawtext=text='{credits}':fontcolor=white:fontsize=h/20:"
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
    return (
        text.replace("\\", "\\\\")
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
    if not clip.event_title:
        return False
    if clip_index == 0:
        return True
    previous = plan.clips[clip_index - 1]
    return previous.event_id != clip.event_id or previous.event_title != clip.event_title


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
    _append_audio_mix(lines, plan, audio_label=audio_label, clip_count=clip_count)
    return ";\n".join(lines)


def _append_audio_mix(
    lines: list[str],
    plan: QuickMontagePlan,
    *,
    audio_label: str,
    clip_count: int,
) -> None:
    duration = plan.total_duration_seconds
    narration_path = _active_narration_path(plan)
    if plan.music_path is None and narration_path is None:
        lines.append(f"[{audio_label}]anull[aout]")
        return

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
        music_index = clip_count
        fade_out_start = max(0.0, duration - 1.5)
        volume = _music_volume_filter(plan)
        lines.append(
            f"[{music_index}:a]aresample=48000,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            "apad,"
            f"atrim=0:{duration:.3f},"
            "asetpts=PTS-STARTPTS,"
            f"{volume},"
            "afade=t=in:st=0:d=1.5,"
            f"afade=t=out:st={fade_out_start:.3f}:d=1.5[music]"
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
        lines.append(f"[{background_label}]alimiter=limit=0.95[aout]")
        return

    narration_index = clip_count + (1 if plan.music_path is not None else 0)
    lines.append(
        f"[{narration_index}:a]aresample=48000,"
        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
        "apad,"
        f"atrim=0:{duration:.3f},"
        "asetpts=PTS-STARTPTS,"
        f"volume={plan.settings.narration_volume:.3f},"
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
        f"atrim=0:{duration:.3f},alimiter=limit=0.95[aout]"
    )


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
