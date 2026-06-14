"""FastAPI application for the local TravelMovieAI web interface."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from travelmovieai.application.service import TravelMovieService
from travelmovieai.application.validation import ProjectPaths
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import InvalidProjectPathError, WorkspaceBusyError
from travelmovieai.domain.models import MediaScanReport
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.directory_dialog import select_directory
from travelmovieai.infrastructure.lm_studio import (
    LMStudioModels,
    list_lm_studio_models,
)
from travelmovieai.infrastructure.system import (
    CudaStatus,
    ExecutableStatus,
    check_cuda,
    check_executable,
    detect_resource_profile,
)
from travelmovieai.web.jobs import ScanJobManager
from travelmovieai.web.movie_jobs import MovieJobManager
from travelmovieai.web.schemas import (
    AIProviderStatus,
    CapabilitiesResponse,
    CudaStatusResponse,
    DependencyStatus,
    DirectoryDialogRequest,
    DirectoryDialogResponse,
    HealthResponse,
    ModelOption,
    MovieJobResponse,
    MovieRequest,
    ResourceProfileResponse,
    ScanJobHistory,
    ScanJobResponse,
    ScanRequest,
    SceneListResponse,
    SceneOverrideRequest,
)

STATIC_DIR = Path(__file__).with_name("static")


def create_app(
    settings: Settings | None = None,
    job_manager: ScanJobManager | None = None,
    movie_job_manager: MovieJobManager | None = None,
    executable_checker: Callable[[str], ExecutableStatus] = check_executable,
    model_lister: Callable[[str, str | None, float], LMStudioModels] = (list_lm_studio_models),
    cuda_checker: Callable[[str], CudaStatus] = check_cuda,
    directory_selector: Callable[[Path | None, str, bool], Path | None] = (
        select_directory
    ),
) -> FastAPI:
    resolved_settings = settings or Settings()
    service = TravelMovieService(resolved_settings)
    manager = job_manager or ScanJobManager(
        service,
        state_path=resolved_settings.workspace.resolve() / ".web" / "jobs.json",
        history_limit=resolved_settings.web_history_limit,
    )
    movie_manager = movie_job_manager or MovieJobManager(service)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown()
        movie_manager.shutdown()

    app = FastAPI(
        title="TravelMovieAI",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )
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
            ready=ffprobe.available,
            ffmpeg=DependencyStatus.model_validate(asdict(ffmpeg)),
            ffprobe=DependencyStatus.model_validate(asdict(ffprobe)),
        )

    @app.get("/api/capabilities", response_model=CapabilitiesResponse)
    def capabilities() -> CapabilitiesResponse:
        discovered = model_lister(
            resolved_settings.lm_studio_url,
            resolved_settings.lm_studio_api_key,
            5,
        )
        model_options = _model_options(
            discovered.models,
            resolved_settings.vision_model,
        )
        cuda_status = cuda_checker(resolved_settings.ffmpeg_binary)
        resources = detect_resource_profile(
            resolved_settings.ffmpeg_binary,
            cuda=cuda_status,
            worker_override=resolved_settings.workers,
            batch_override=resolved_settings.batch_size,
        )
        return CapabilitiesResponse(
            ai=AIProviderStatus(
                available=discovered.available,
                base_url=resolved_settings.lm_studio_url,
                configured_model=resolved_settings.vision_model,
                models=model_options,
                error=discovered.error,
            ),
            cuda=CudaStatusResponse.model_validate(asdict(cuda_status)),
            resources=ResourceProfileResponse.model_validate(asdict(resources)),
            opencv_available=find_spec("cv2") is not None,
            scenedetect_available=find_spec("scenedetect") is not None,
            music_modes=["auto", "generated", "library", "manual", "none"],
            render_devices=["auto", "cuda", "cpu"],
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
            (
                "Выберите папку с видео и фотографиями"
                if is_input
                else "Выберите папку workspace"
            ),
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
            if _movie_manager(request).is_workspace_active(paths.workspace):
                raise WorkspaceBusyError("Для этого workspace уже выполняется монтаж фильма.")
            return manager_from_app.submit(paths.input_path, paths.workspace)
        except InvalidProjectPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except WorkspaceBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

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
            raise HTTPException(status_code=404, detail="Задание не найдено.")
        return job

    @app.get("/api/scans/{job_id}/result", response_model=MediaScanReport)
    def get_scan_result(job_id: UUID, request: Request) -> MediaScanReport:
        manager_from_app = _manager(request)
        job = manager_from_app.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Задание не найдено.")
        if job.status == "failed":
            raise HTTPException(status_code=409, detail=job.error or job.message)
        report = manager_from_app.get_report(job_id)
        if report is None:
            raise HTTPException(status_code=409, detail="Результат ещё не готов.")
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
            paths = movie_manager_from_app.resolve_project_paths(
                Path(payload.input_path), workspace
            )
            if _manager(request).is_workspace_active(paths.workspace):
                raise WorkspaceBusyError("Для этого workspace уже выполняется анализ медиатеки.")
            return movie_manager_from_app.submit(
                paths.input_path,
                paths.workspace,
                payload.settings,
            )
        except InvalidProjectPathError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except WorkspaceBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/movies/{job_id}", response_model=MovieJobResponse)
    def get_movie(job_id: UUID, request: Request) -> MovieJobResponse:
        job = _movie_manager(request).get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Задание монтажа не найдено.")
        return job

    @app.get("/api/movies/{job_id}/download", response_class=FileResponse)
    def download_movie(job_id: UUID, request: Request) -> FileResponse:
        output_path = _movie_manager(request).output_path(job_id)
        if output_path is None or not output_path.is_file():
            raise HTTPException(status_code=409, detail="Фильм ещё не готов.")
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=output_path.name,
        )

    @app.get("/api/scenes", response_model=SceneListResponse)
    def list_scenes(
        input_path: str = Query(min_length=1),
        workspace: str | None = Query(default=None),
    ) -> SceneListResponse:
        paths = _validated_paths(service, input_path, workspace)
        repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
        repository.initialize()
        return SceneListResponse(scenes=repository.list_scenes())

    @app.patch("/api/scenes/{scene_id}", response_model=SceneListResponse)
    def update_scene_override(
        scene_id: UUID,
        payload: SceneOverrideRequest,
    ) -> SceneListResponse:
        paths = _validated_paths(service, payload.input_path, payload.workspace)
        repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
        repository.initialize()
        scene = repository.set_scene_selection_override(scene_id, payload.decision)
        if scene is None:
            raise HTTPException(status_code=404, detail="Сцена не найдена.")
        return SceneListResponse(scenes=[scene])

    @app.get("/api/scenes/{scene_id}/thumbnail", response_class=FileResponse)
    def scene_thumbnail(
        scene_id: UUID,
        input_path: str = Query(min_length=1),
        workspace: str | None = Query(default=None),
    ) -> FileResponse:
        paths = _validated_paths(service, input_path, workspace)
        repository = MediaAssetRepository(paths.workspace / resolved_settings.database_filename)
        repository.initialize()
        scene = next(
            (item for item in repository.list_scenes() if item.id == scene_id),
            None,
        )
        if scene is None or scene.keyframe_path is None or not scene.keyframe_path.is_file():
            raise HTTPException(status_code=404, detail="Кадр сцены не найден.")
        frame_path = scene.keyframe_path.resolve()
        if not (
            frame_path.is_relative_to(paths.workspace)
            or frame_path.is_relative_to(paths.input_path)
        ):
            raise HTTPException(status_code=403, detail="Недопустимый путь кадра.")
        return FileResponse(frame_path)

    return app


def _manager(request: Request) -> ScanJobManager:
    manager: ScanJobManager = request.app.state.job_manager
    return manager


def _movie_manager(request: Request) -> MovieJobManager:
    manager: MovieJobManager = request.app.state.movie_job_manager
    return manager


def _model_options(models: tuple[str, ...], configured_model: str) -> list[ModelOption]:
    vision_markers = ("vision", "-vl", "/vl", "omni", "gemma-3", "gemma-4")
    likely = [
        model for model in models if any(marker in model.casefold() for marker in vision_markers)
    ]
    recommended = (
        configured_model
        if configured_model in models
        else (likely[0] if likely else (models[0] if models else configured_model))
    )
    return [
        ModelOption(
            id=model,
            likely_vision=model in likely,
            recommended=model == recommended,
        )
        for model in models
    ]


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
