"""FastAPI application for the local TravelMovieAI web interface."""

import json
import logging
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from travelmovieai import __version__
from travelmovieai.application.diagnostics import (
    model_snapshot_present,
    run_system_diagnostics,
)
from travelmovieai.application.service import TravelMovieService
from travelmovieai.application.validation import ProjectPaths
from travelmovieai.application.workspace_lease import WorkspaceLease
from travelmovieai.core.config import Settings, load_settings
from travelmovieai.core.exceptions import (
    InvalidProjectPathError,
    JobPersistenceError,
    WorkspaceBusyError,
)
from travelmovieai.core.logging import configured_log_path, correlation_context
from travelmovieai.core.security import redact_sensitive_text
from travelmovieai.domain.manual_editing import (
    compare_timeline_versions,
    summarize_timeline_version,
)
from travelmovieai.domain.models import MediaScanReport, QuickMontageSettings
from travelmovieai.infrastructure.database import (
    EditConflictError,
    EditValidationError,
    MediaAssetRepository,
)
from travelmovieai.infrastructure.directory_dialog import select_directory
from travelmovieai.infrastructure.music_generation import (
    LOCAL_MUSIC_MODELS,
    resolve_local_music_model,
)
from travelmovieai.infrastructure.system import (
    CudaStatus,
    ExecutableStatus,
    check_cuda,
    check_executable,
    detect_resource_profile,
)
from travelmovieai.infrastructure.vision import (
    LOCAL_QWEN_MODELS,
    resolve_local_vision_model,
)
from travelmovieai.web.jobs import ScanJobManager
from travelmovieai.web.movie_jobs import MovieJobManager
from travelmovieai.web.schemas import (
    CapabilitiesResponse,
    CudaStatusResponse,
    DependencyStatus,
    DirectoryDialogRequest,
    DirectoryDialogResponse,
    EventListResponse,
    EventPatchRequest,
    FeatureCapability,
    HealthResponse,
    JobStatus,
    LocalAIStatus,
    ModelOption,
    MovieJobHistory,
    MovieJobResponse,
    MovieRequest,
    MusicAIStatus,
    ReorderRequest,
    ResourceProfileResponse,
    ScanJobHistory,
    ScanJobResponse,
    ScanRequest,
    SceneListResponse,
    SceneOverrideRequest,
    TimelineVersionCompareResponse,
    TimelineVersionListResponse,
    TimelineVersionResponse,
)

STATIC_DIR = Path(__file__).with_name("static")
LOGGER = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    job_manager: ScanJobManager | None = None,
    movie_job_manager: MovieJobManager | None = None,
    executable_checker: Callable[[str], ExecutableStatus] = check_executable,
    cuda_checker: Callable[[str], CudaStatus] = check_cuda,
    package_checker: Callable[[str], bool] | None = None,
    directory_selector: Callable[[Path | None, str, bool], Path | None] = (select_directory),
) -> FastAPI:
    resolved_settings = settings or load_settings()
    has_package = package_checker or _package_available
    service = TravelMovieService(resolved_settings)
    manager = job_manager or ScanJobManager(
        service,
        state_path=resolved_settings.workspace.resolve() / ".web" / "jobs.json",
        history_limit=resolved_settings.web_history_limit,
    )
    movie_manager = movie_job_manager or MovieJobManager(
        service,
        state_path=resolved_settings.workspace.resolve() / ".web" / "movie_jobs.json",
        history_limit=resolved_settings.web_history_limit,
        render_disk_reserve_mb=resolved_settings.render_disk_reserve_mb,
        render_disk_safety_factor=resolved_settings.render_disk_safety_factor,
    )
    workspace_mutation_lock = RLock()

    @contextmanager
    def workspace_edit(request: Request, workspace: Path) -> Iterator[None]:
        with workspace_mutation_lock:
            _ensure_workspace_editable(request, workspace)
            try:
                with WorkspaceLease(workspace, operation="manual_edit"):
                    yield
            except WorkspaceBusyError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown()
        movie_manager.shutdown()

    app = FastAPI(
        title="TravelMovieAI",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )

    @app.middleware("http")
    async def enforce_local_request_boundaries(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid4())
        request.state.request_id = request_id
        response: Response
        with correlation_context(request_id):
            if not _is_allowed_host(request.headers.get("host", "")):
                response = JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "detail": "Requests require a loopback Host header.",
                        "request_id": request_id,
                    },
                )
                return _add_browser_security_headers(response, request_id)
            if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
                fetch_site = request.headers.get("sec-fetch-site", "").casefold()
                if fetch_site and fetch_site not in {"same-origin", "none"}:
                    response = JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "detail": "Cross-site mutating requests are not allowed.",
                            "request_id": request_id,
                        },
                    )
                    return _add_browser_security_headers(response, request_id)
                origin = request.headers.get("origin")
                if origin and not _is_same_origin(
                    origin,
                    scheme=request.url.scheme,
                    host_header=request.headers.get("host", ""),
                ):
                    response = JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "detail": "Mutating requests require the exact local browser origin.",
                            "request_id": request_id,
                        },
                    )
                    return _add_browser_security_headers(response, request_id)
            try:
                response = await call_next(request)
            except Exception:
                LOGGER.exception(
                    "Unhandled local HTTP request failure: %s %s",
                    request.method,
                    request.url.path,
                    extra={"request_id": request_id},
                )
                response = JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "detail": "Internal server error. Details were written to the local log.",
                        "request_id": request_id,
                    },
                )
            return _add_browser_security_headers(response, request_id)

    app.state.job_manager = manager
    app.state.movie_job_manager = movie_manager
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        ffmpeg = executable_checker(resolved_settings.ffmpeg_binary)
        ffprobe = executable_checker(resolved_settings.ffprobe_binary)
        return HealthResponse(
            status="ok" if ffmpeg.available and ffprobe.available else "degraded",
            ready=ffmpeg.available and ffprobe.available,
            ffmpeg=DependencyStatus.model_validate(asdict(ffmpeg)),
            ffprobe=DependencyStatus.model_validate(asdict(ffprobe)),
        )

    @app.get("/api/diagnostics/bundle", response_class=Response)
    def diagnostics_bundle() -> Response:
        report = run_system_diagnostics(resolved_settings)
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "application": "TravelMovieAI",
            "version": __version__,
            "ready": report.ready,
            "checks": [asdict(check) for check in report.checks],
            "configuration": {
                "downloads_enabled": resolved_settings.allow_model_download,
                "device": resolved_settings.device,
                "resource_mode": resolved_settings.resource_mode,
                "vision_model": resolved_settings.vision_model,
                "embedding_backend": resolved_settings.embedding_backend,
                "story_provider": resolved_settings.story_provider,
                "voice_provider": resolved_settings.voice_provider,
            },
        }
        serialized = redact_sensitive_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            private_paths=(
                Path.home(),
                resolved_settings.workspace,
                resolved_settings.model_cache,
                resolved_settings.music_library,
            ),
            max_characters=100_000,
        )
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("diagnostics.json", serialized)
            log_tail = _diagnostic_log_tail(resolved_settings)
            if log_tail:
                archive.writestr("application.log", log_tail)
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": ('attachment; filename="travelmovieai-diagnostics.zip"')
            },
        )

    @app.get("/api/capabilities", response_model=CapabilitiesResponse)
    def capabilities() -> CapabilitiesResponse:
        cuda_status = cuda_checker(resolved_settings.ffmpeg_binary)
        resources = detect_resource_profile(
            resolved_settings.ffmpeg_binary,
            cuda=cuda_status,
            worker_override=resolved_settings.workers,
            batch_override=resolved_settings.batch_size,
            resource_mode=resolved_settings.resource_mode,
            gpu_memory_reserve_mb=resolved_settings.gpu_memory_reserve_mb,
            max_gpu_processes=resolved_settings.max_gpu_processes,
        )
        local_model = _resolve_requested_vision_model(
            resolved_settings.vision_provider,
            resolved_settings.vision_model,
            gpu_memory_mb=resources.gpu_memory_mb,
            system_memory_mb=resources.memory_mb,
        )
        music_model = resolve_local_music_model(
            resolved_settings.music_model,
            gpu_memory_mb=resources.gpu_memory_mb,
        )
        music_runtime = Path(".cache/ace-step/.venv/Scripts/python.exe").resolve()
        semantic_capability = _semantic_capability(
            has_package,
            resolved_settings,
            local_model,
        )
        speech_capability = _speech_capability(has_package, resolved_settings)
        narration_capability = _narration_capability(
            resolved_settings,
            executable_checker,
        )
        return CapabilitiesResponse(
            default_workspace_root=str(resolved_settings.workspace.expanduser().resolve()),
            local_ai=LocalAIStatus(
                available=semantic_capability.available,
                configured_model=resolved_settings.vision_model,
                resolved_model=local_model,
                cache_dir=str(resolved_settings.model_cache.expanduser().resolve()),
                downloads_enabled=resolved_settings.allow_model_download,
                models=[
                    ModelOption(
                        id=model,
                        likely_vision=True,
                        recommended=model == local_model,
                    )
                    for model in LOCAL_QWEN_MODELS
                ],
                reason=semantic_capability.reason,
                action=semantic_capability.action,
            ),
            music_ai=MusicAIStatus(
                available=music_runtime.is_file(),
                configured_model=resolved_settings.music_model,
                resolved_model=music_model,
                cache_dir=str((resolved_settings.model_cache / "ace-step").expanduser().resolve()),
                downloads_enabled=resolved_settings.allow_model_download,
                runtime_installed=music_runtime.is_file(),
                models=[
                    ModelOption(
                        id=model,
                        recommended=model == music_model,
                    )
                    for model in LOCAL_MUSIC_MODELS
                ],
                reason=(
                    None
                    if music_runtime.is_file()
                    else "The optional ACE-Step runtime is not installed."
                ),
                action=(
                    None
                    if music_runtime.is_file()
                    else "Use procedural or library music, or install ACE-Step explicitly."
                ),
            ),
            speech=speech_capability,
            narration=narration_capability,
            cuda=CudaStatusResponse.model_validate(asdict(cuda_status)),
            resources=ResourceProfileResponse.model_validate(asdict(resources)),
            opencv_available=find_spec("cv2") is not None,
            scenedetect_available=find_spec("scenedetect") is not None,
            music_modes=["auto", "generated", "library", "manual", "none"],
            render_devices=["auto", "cuda", "cpu"],
            recommended_render_device=("cuda" if cuda_status.ffmpeg_nvenc else "cpu"),
            recommended_resource_mode=resources.resource_mode,
        )

    @app.post("/api/dialogs/directory", response_model=DirectoryDialogResponse)
    def directory_dialog(payload: DirectoryDialogRequest) -> DirectoryDialogResponse:
        initial_path = (
            Path(payload.initial_path)
            if payload.initial_path and payload.initial_path.strip()
            else None
        )
        is_input = payload.purpose == "input"
        selected = directory_selector(
            initial_path,
            ("Choose a folder with videos and photos" if is_input else "Choose a workspace folder"),
            is_input,
        )
        return DirectoryDialogResponse(selected_path=selected)

    @app.post(
        "/api/scans",
        response_model=ScanJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_scan(payload: ScanRequest, request: Request) -> ScanJobResponse:
        workspace = (
            Path(payload.workspace) if payload.workspace and payload.workspace.strip() else None
        )
        manager_from_app = _manager(request)
        try:
            paths = manager_from_app.resolve_project_paths(Path(payload.input_path), workspace)
            with workspace_mutation_lock:
                if _movie_manager(request).is_workspace_active(paths.workspace):
                    raise WorkspaceBusyError("A movie edit is already running for this workspace.")
                return manager_from_app.submit(paths.input_path, paths.workspace)
        except InvalidProjectPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except WorkspaceBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except JobPersistenceError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/scans", response_model=ScanJobHistory)
    def list_scans(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> ScanJobHistory:
        return ScanJobHistory(jobs=_manager(request).list(limit))

    @app.get("/api/scans/{job_id}", response_model=ScanJobResponse)
    def get_scan(job_id: UUID, request: Request) -> ScanJobResponse:
        job = _manager(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    @app.get("/api/scans/{job_id}/result", response_model=MediaScanReport)
    def get_scan_result(job_id: UUID, request: Request) -> MediaScanReport:
        manager_from_app = _manager(request)
        job = manager_from_app.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.status == "failed":
            raise HTTPException(status_code=409, detail=job.error or job.message)
        report = manager_from_app.get_report(job_id)
        if report is None:
            raise HTTPException(status_code=409, detail="Result is not ready yet.")
        return report

    @app.post(
        "/api/movies",
        response_model=MovieJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_movie(payload: MovieRequest, request: Request) -> MovieJobResponse:
        workspace = (
            Path(payload.workspace) if payload.workspace and payload.workspace.strip() else None
        )
        movie_manager_from_app = _movie_manager(request)
        try:
            _validate_requested_capabilities(
                payload.settings,
                semantic=_semantic_capability(
                    has_package,
                    resolved_settings,
                    _requested_vision_model(payload.settings, resolved_settings, service),
                ),
                speech=_speech_capability(has_package, resolved_settings),
                narration=_narration_capability(
                    resolved_settings,
                    executable_checker,
                ),
                render_cuda=_render_cuda_capability(
                    payload.settings,
                    resolved_settings,
                    cuda_checker,
                ),
            )
            paths = movie_manager_from_app.resolve_project_paths(
                Path(payload.input_path), workspace
            )
            with workspace_mutation_lock:
                if _manager(request).is_workspace_active(paths.workspace):
                    raise WorkspaceBusyError("A media scan is already running for this workspace.")
                return movie_manager_from_app.submit(
                    paths.input_path,
                    paths.workspace,
                    payload.settings,
                    payload.variant_name,
                )
        except InvalidProjectPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except WorkspaceBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except JobPersistenceError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/api/movies/{job_id}", response_model=MovieJobResponse)
    def get_movie(job_id: UUID, request: Request) -> MovieJobResponse:
        job = _movie_manager(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Movie job not found.")
        return job

    @app.get("/api/movies", response_model=MovieJobHistory)
    def list_movies(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> MovieJobHistory:
        return MovieJobHistory(jobs=_movie_manager(request).list(limit))

    @app.post("/api/movies/{job_id}/pause", response_model=MovieJobResponse)
    def pause_movie(job_id: UUID, request: Request) -> MovieJobResponse:
        job = _movie_manager(request).pause(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Movie job not found.")
        return job

    @app.post("/api/movies/{job_id}/resume", response_model=MovieJobResponse)
    def resume_movie(job_id: UUID, request: Request) -> MovieJobResponse:
        job = _movie_manager(request).resume(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Movie job not found.")
        return job

    @app.post("/api/movies/{job_id}/cancel", response_model=MovieJobResponse)
    def cancel_movie(job_id: UUID, request: Request) -> MovieJobResponse:
        job = _movie_manager(request).cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Movie job not found.")
        return job

    @app.get("/api/movies/{job_id}/download", response_class=FileResponse)
    def download_movie(job_id: UUID, request: Request) -> FileResponse:
        movie_manager_from_app = _movie_manager(request)
        job = movie_manager_from_app.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Movie job not found.")
        if job.status == JobStatus.FAILED:
            raise HTTPException(status_code=409, detail=job.error or job.message)
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(status_code=409, detail="The movie is not ready yet.")
        output_path = movie_manager_from_app.output_path(job_id)
        if output_path is None:
            raise HTTPException(status_code=409, detail="The movie is not ready yet.")
        resolved_output = output_path.expanduser().resolve()
        if not resolved_output.is_relative_to(job.workspace.expanduser().resolve()):
            raise HTTPException(status_code=403, detail="Invalid movie output path.")
        if not resolved_output.is_file():
            raise HTTPException(status_code=404, detail="Rendered movie file not found.")
        return FileResponse(
            resolved_output,
            media_type="video/mp4",
            filename=resolved_output.name,
        )

    @app.get("/api/scenes", response_model=SceneListResponse)
    def list_scenes(
        input_path: str = Query(min_length=1),
        workspace: str | None = Query(default=None),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=120, ge=1, le=500),
        event_id: UUID | None = None,
    ) -> SceneListResponse:
        paths = _validated_paths(service, input_path, workspace)
        repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
        repository.initialize()
        page = repository.list_editable_scenes_page(
            offset=offset,
            limit=limit,
            event_id=event_id,
        )
        if page is None:
            raise HTTPException(status_code=404, detail="Event not found.")
        scenes, total = page
        return SceneListResponse(
            scenes=scenes,
            total=total,
            offset=offset,
            limit=limit,
        )

    @app.patch("/api/scenes/{scene_id}", response_model=SceneListResponse)
    def update_scene_override(
        scene_id: UUID,
        payload: SceneOverrideRequest,
        request: Request,
    ) -> SceneListResponse:
        paths = _validated_paths(service, payload.input_path, payload.workspace)
        with workspace_edit(request, paths.workspace):
            repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
            repository.initialize()
            try:
                if payload.decision is not None:
                    scene = repository.set_scene_selection_override(scene_id, payload.decision)
                    if scene is None:
                        raise HTTPException(status_code=404, detail="Scene not found.")
                edit_fields = {
                    "caption",
                    "transcript",
                    "landmarks",
                } & payload.model_fields_set
                if edit_fields:
                    edited = repository.update_scene(
                        scene_id,
                        expected_version=payload.expected_version or 1,
                        caption=payload.caption,
                        transcript=payload.transcript,
                        landmarks=payload.landmarks,
                        update_caption="caption" in edit_fields,
                        update_transcript="transcript" in edit_fields,
                        update_landmarks="landmarks" in edit_fields,
                    )
                    if edited is None:
                        raise HTTPException(status_code=404, detail="Scene not found.")
                    return SceneListResponse(scenes=[edited], total=1, limit=1)
            except EditConflictError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            current = repository.get_editable_scene(scene_id)
            if current is None:
                raise HTTPException(status_code=404, detail="Scene not found.")
            return SceneListResponse(scenes=[current], total=1, limit=1)

    @app.get("/api/events", response_model=EventListResponse)
    def list_events(
        input_path: str = Query(min_length=1),
        workspace: str | None = Query(default=None),
    ) -> EventListResponse:
        paths = _validated_paths(service, input_path, workspace)
        repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
        repository.initialize()
        return EventListResponse(events=repository.list_editable_events())

    @app.put("/api/events/order", response_model=EventListResponse)
    def reorder_events(payload: ReorderRequest, request: Request) -> EventListResponse:
        paths = _validated_paths(service, payload.input_path, payload.workspace)
        with workspace_edit(request, paths.workspace):
            repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
            repository.initialize()
            try:
                events = repository.reorder_events(
                    payload.ordered_ids,
                    payload.expected_versions,
                )
            except EditConflictError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            except EditValidationError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            return EventListResponse(events=events)

    @app.patch("/api/events/{event_id}", response_model=EventListResponse)
    def update_event(
        event_id: UUID,
        payload: EventPatchRequest,
        request: Request,
    ) -> EventListResponse:
        paths = _validated_paths(service, payload.input_path, payload.workspace)
        with workspace_edit(request, paths.workspace):
            repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
            repository.initialize()
            fields = {"title", "summary", "landmarks"} & payload.model_fields_set
            try:
                event = repository.update_event(
                    event_id,
                    expected_version=payload.expected_version,
                    title=payload.title,
                    summary=payload.summary,
                    landmarks=payload.landmarks,
                    update_title="title" in fields,
                    update_summary="summary" in fields,
                    update_landmarks="landmarks" in fields,
                )
            except EditConflictError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            if event is None:
                raise HTTPException(status_code=404, detail="Event not found.")
            return EventListResponse(events=[event])

    @app.put("/api/events/{event_id}/scenes/order", response_model=SceneListResponse)
    def reorder_event_scenes(
        event_id: UUID,
        payload: ReorderRequest,
        request: Request,
    ) -> SceneListResponse:
        paths = _validated_paths(service, payload.input_path, payload.workspace)
        with workspace_edit(request, paths.workspace):
            repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
            repository.initialize()
            try:
                scenes = repository.reorder_scenes(
                    event_id,
                    payload.ordered_ids,
                    payload.expected_versions,
                )
            except EditConflictError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            except EditValidationError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            if scenes is None:
                raise HTTPException(status_code=404, detail="Event not found.")
            return SceneListResponse(
                scenes=scenes,
                total=len(scenes),
                limit=max(1, len(scenes)),
            )

    @app.get("/api/timeline-versions", response_model=TimelineVersionListResponse)
    def list_timeline_versions(
        input_path: str = Query(min_length=1),
        workspace: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> TimelineVersionListResponse:
        paths = _validated_paths(service, input_path, workspace)
        repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
        repository.initialize()
        return TimelineVersionListResponse(
            versions=[
                summarize_timeline_version(version)
                for version in repository.list_timeline_versions(limit)
            ]
        )

    @app.get(
        "/api/timeline-versions/compare",
        response_model=TimelineVersionCompareResponse,
    )
    def compare_versions(
        before_id: UUID,
        after_id: UUID,
        input_path: str = Query(min_length=1),
        workspace: str | None = Query(default=None),
    ) -> TimelineVersionCompareResponse:
        paths = _validated_paths(service, input_path, workspace)
        repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
        repository.initialize()
        before = repository.get_timeline_version(before_id)
        after = repository.get_timeline_version(after_id)
        if before is None or after is None:
            raise HTTPException(status_code=404, detail="Timeline version not found.")
        return TimelineVersionCompareResponse(comparison=compare_timeline_versions(before, after))

    @app.get(
        "/api/timeline-versions/{version_id}",
        response_model=TimelineVersionResponse,
    )
    def get_timeline_version(
        version_id: UUID,
        input_path: str = Query(min_length=1),
        workspace: str | None = Query(default=None),
    ) -> TimelineVersionResponse:
        paths = _validated_paths(service, input_path, workspace)
        repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
        repository.initialize()
        version = repository.get_timeline_version(version_id)
        if version is None:
            raise HTTPException(status_code=404, detail="Timeline version not found.")
        return TimelineVersionResponse(version=version)

    @app.get("/api/scenes/{scene_id}/thumbnail", response_class=FileResponse)
    def scene_thumbnail(
        scene_id: UUID,
        input_path: str = Query(min_length=1),
        workspace: str | None = Query(default=None),
    ) -> FileResponse:
        paths = _validated_paths(service, input_path, workspace)
        repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
        repository.initialize()
        scene = repository.get_scene(scene_id)
        if scene is None or scene.keyframe_path is None or not scene.keyframe_path.is_file():
            raise HTTPException(status_code=404, detail="Scene thumbnail not found.")
        frame_path = scene.keyframe_path.resolve()
        if not (
            frame_path.is_relative_to(paths.workspace)
            or frame_path.is_relative_to(paths.input_path)
        ):
            raise HTTPException(status_code=403, detail="Invalid thumbnail path.")
        return FileResponse(frame_path)

    return app


def _manager(request: Request) -> ScanJobManager:
    manager: ScanJobManager = request.app.state.job_manager
    return manager


def _movie_manager(request: Request) -> MovieJobManager:
    manager: MovieJobManager = request.app.state.movie_job_manager
    return manager


def _ensure_workspace_editable(request: Request, workspace: Path) -> None:
    if _manager(request).is_workspace_active(workspace) or _movie_manager(
        request
    ).is_workspace_active(workspace):
        raise HTTPException(
            status_code=409,
            detail="Manual edits are locked while this project has an active job.",
        )


def _validated_paths(
    service: TravelMovieService,
    input_path: str,
    workspace: str | None,
) -> ProjectPaths:
    resolved_workspace = Path(workspace) if workspace and workspace.strip() else None
    try:
        return service.resolve_project_paths(Path(input_path), resolved_workspace)
    except InvalidProjectPathError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _package_available(package: str) -> bool:
    return find_spec(package) is not None


def _semantic_capability(
    package_checker: Callable[[str], bool],
    settings: Settings,
    resolved_model: str,
) -> FeatureCapability:
    required = ["accelerate", "torch", "transformers"]
    if settings.embedding_backend == "sentence-transformers":
        required.append("sentence_transformers")
    if settings.embedding_index == "faiss":
        required.append("faiss")
    missing = [package for package in required if not package_checker(package)]
    if missing:
        return FeatureCapability(
            available=False,
            reason=f"Missing local semantic dependencies: {', '.join(missing)}.",
            action='Install the required local groups with python -m pip install -e ".[all]".',
        )
    if not settings.allow_model_download:
        required_models = [resolved_model]
        if settings.embedding_backend == "sentence-transformers":
            required_models.append(settings.embedding_model)
        if settings.story_provider == "local":
            required_models.append(settings.story_model)
        missing_models = [
            model
            for model in required_models
            if not model_snapshot_present(settings.model_cache, model)
        ]
        if missing_models:
            return FeatureCapability(
                available=False,
                reason="Offline model cache is incomplete for: " + ", ".join(missing_models),
                action="Download the selected local models or enable explicit model downloads.",
            )
    return FeatureCapability(available=True)


def _speech_capability(
    package_checker: Callable[[str], bool],
    settings: Settings,
) -> FeatureCapability:
    if not package_checker("faster_whisper"):
        return FeatureCapability(
            available=False,
            reason="Faster Whisper is not installed.",
            action='Install it with python -m pip install -e ".[speech]".',
        )
    if not settings.allow_model_download:
        whisper_model = f"Systran/faster-whisper-{settings.whisper_model}"
        if not model_snapshot_present(settings.model_cache, whisper_model):
            return FeatureCapability(
                available=False,
                reason=f"Offline Whisper snapshot is missing: {whisper_model}.",
                action="Download the configured speech model or enable explicit downloads.",
            )
    return FeatureCapability(available=True)


def _requested_vision_model(
    montage: QuickMontageSettings,
    settings: Settings,
    service: TravelMovieService,
) -> str:
    resources = service.get_resource_profile()
    return _resolve_requested_vision_model(
        montage.vision_provider,
        montage.vision_model or settings.vision_model,
        gpu_memory_mb=resources.gpu_memory_mb,
        system_memory_mb=resources.memory_mb,
    )


def _resolve_requested_vision_model(
    provider: str,
    model: str,
    *,
    gpu_memory_mb: int | None,
    system_memory_mb: int | None,
) -> str:
    if provider == "florence":
        return model if model != "auto" else "microsoft/Florence-2-large"
    return resolve_local_vision_model(
        model,
        gpu_memory_mb=gpu_memory_mb,
        system_memory_mb=system_memory_mb,
    )


def _diagnostic_log_tail(settings: Settings) -> str | None:
    path = configured_log_path()
    if path is None or not path.is_file():
        return None
    try:
        with path.open("rb") as log_file:
            size = path.stat().st_size
            log_file.seek(max(0, size - 128 * 1024))
            content = log_file.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    return redact_sensitive_text(
        content,
        private_paths=(
            Path.home(),
            settings.workspace,
            settings.model_cache,
            settings.music_library,
        ),
        max_characters=100_000,
    )


def _narration_capability(
    settings: Settings,
    executable_checker: Callable[[str], ExecutableStatus],
) -> FeatureCapability:
    if settings.voice_provider != "piper":
        return FeatureCapability(
            available=False,
            reason="Local narration is disabled in settings.toml.",
            action="Set voice_provider to piper and configure piper_model.",
        )
    if settings.piper_model is None:
        return FeatureCapability(
            available=False,
            reason="No local Piper voice model is configured.",
            action="Set piper_model to a local .onnx voice file.",
        )
    model_path = settings.piper_model.expanduser().resolve()
    if not model_path.is_file():
        return FeatureCapability(
            available=False,
            reason="The configured local Piper voice model was not found.",
            action="Check piper_model in settings.toml.",
        )
    piper = executable_checker(settings.piper_binary)
    if not piper.available:
        return FeatureCapability(
            available=False,
            reason=piper.error or "The Piper executable was not found.",
            action="Install Piper or configure piper_binary in settings.toml.",
        )
    return FeatureCapability(available=True)


def _validate_requested_capabilities(
    settings: QuickMontageSettings,
    *,
    semantic: FeatureCapability,
    speech: FeatureCapability,
    narration: FeatureCapability,
    render_cuda: FeatureCapability,
) -> None:
    semantic_render_features = []
    if settings.framing_mode == "smart":
        semantic_render_features.append("smart crop")
    if settings.color_normalization:
        semantic_render_features.append("color normalization")
    if settings.event_titles_enabled:
        semantic_render_features.append("event titles")
    if settings.scene_subtitles_enabled:
        semantic_render_features.append("scene subtitles")
    if semantic_render_features and not settings.semantic_analysis:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{', '.join(semantic_render_features)} require semantic scene analysis "
                "for movie creation."
            ),
        )
    if settings.speech_analysis and not settings.semantic_analysis:
        raise HTTPException(
            status_code=422,
            detail="Speech recognition requires semantic scene selection for movie creation.",
        )
    if settings.narration_enabled and not settings.semantic_analysis:
        raise HTTPException(
            status_code=422,
            detail="Local narration requires semantic scene selection for movie creation.",
        )
    requested = (
        (settings.semantic_analysis, "Semantic scene selection", semantic),
        (settings.speech_analysis, "Speech recognition", speech),
        (settings.narration_enabled, "Local narration", narration),
        (settings.render_device == "cuda", "CUDA rendering", render_cuda),
    )
    for enabled, label, capability in requested:
        if enabled and not capability.available:
            detail = capability.reason or "The required local runtime is unavailable."
            if capability.action:
                detail = f"{detail} {capability.action}"
            raise HTTPException(status_code=422, detail=f"{label} is unavailable: {detail}")


def _render_cuda_capability(
    settings: QuickMontageSettings,
    application_settings: Settings,
    cuda_checker: Callable[[str], CudaStatus],
) -> FeatureCapability:
    if settings.render_device != "cuda":
        return FeatureCapability(available=True)
    cuda = cuda_checker(application_settings.ffmpeg_binary)
    if cuda.available and cuda.ffmpeg_nvenc:
        return FeatureCapability(available=True)
    return FeatureCapability(
        available=False,
        reason="NVIDIA h264_nvenc encoding is unavailable.",
        action="Choose Auto or CPU rendering, or configure an FFmpeg build with NVENC.",
    )


def _is_same_origin(origin: str, *, scheme: str, host_header: str) -> bool:
    try:
        parsed = urlsplit(origin)
        expected = urlsplit(f"//{host_header}")
        origin_port = parsed.port
        expected_port = expected.port
    except ValueError:
        return False
    normalized_scheme = scheme.casefold()
    if origin_port is None:
        origin_port = 443 if parsed.scheme.casefold() == "https" else 80
    if expected_port is None:
        expected_port = 443 if normalized_scheme == "https" else 80
    return (
        parsed.scheme.casefold() == normalized_scheme
        and parsed.scheme.casefold() in {"http", "https"}
        and parsed.hostname is not None
        and expected.hostname is not None
        and parsed.hostname.casefold() == expected.hostname.casefold()
        and parsed.hostname.casefold() in {"127.0.0.1", "localhost", "::1", "testserver"}
        and origin_port == expected_port
        and parsed.username is None
        and parsed.password is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def _add_browser_security_headers(response: Response, request_id: str) -> Response:
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; font-src 'self' data:; media-src 'self' blob:; "
        "connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Request-ID"] = request_id
    return response


def _is_allowed_host(host_header: str) -> bool:
    try:
        parsed = urlsplit(f"//{host_header}")
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.hostname is not None
        and parsed.hostname.casefold() in {"127.0.0.1", "localhost", "::1", "testserver"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
    )
