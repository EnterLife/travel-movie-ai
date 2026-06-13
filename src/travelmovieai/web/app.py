"""FastAPI application for the local TravelMovieAI web interface."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import InvalidProjectPathError, WorkspaceBusyError
from travelmovieai.domain.models import MediaScanReport
from travelmovieai.infrastructure.system import ExecutableStatus, check_executable
from travelmovieai.web.jobs import ScanJobManager
from travelmovieai.web.movie_jobs import MovieJobManager
from travelmovieai.web.schemas import (
    DependencyStatus,
    HealthResponse,
    MovieJobResponse,
    MovieRequest,
    ScanJobHistory,
    ScanJobResponse,
    ScanRequest,
)

STATIC_DIR = Path(__file__).with_name("static")


def create_app(
    settings: Settings | None = None,
    job_manager: ScanJobManager | None = None,
    movie_job_manager: MovieJobManager | None = None,
    executable_checker: Callable[[str], ExecutableStatus] = check_executable,
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

    return app


def _manager(request: Request) -> ScanJobManager:
    manager: ScanJobManager = request.app.state.job_manager
    return manager


def _movie_manager(request: Request) -> MovieJobManager:
    manager: MovieJobManager = request.app.state.movie_job_manager
    return manager
