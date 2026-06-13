"""FastAPI application for the local TravelMovieAI web interface."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.domain.models import MediaScanReport
from travelmovieai.web.jobs import ScanJobManager
from travelmovieai.web.schemas import HealthResponse, ScanJobResponse, ScanRequest

STATIC_DIR = Path(__file__).with_name("static")


def create_app(
    settings: Settings | None = None,
    job_manager: ScanJobManager | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    manager = job_manager or ScanJobManager(TravelMovieService(resolved_settings))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown()

    app = FastAPI(
        title="TravelMovieAI",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.job_manager = manager
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post(
        "/api/scans",
        response_model=ScanJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_scan(payload: ScanRequest, request: Request) -> ScanJobResponse:
        input_path = Path(payload.input_path).expanduser().resolve()
        if not input_path.exists():
            raise HTTPException(status_code=400, detail="Исходная папка не существует.")
        if not input_path.is_dir():
            raise HTTPException(status_code=400, detail="Исходный путь должен быть папкой.")

        workspace = (
            Path(payload.workspace).expanduser().resolve()
            if payload.workspace and payload.workspace.strip()
            else None
        )
        manager_from_app = _manager(request)
        resolved_workspace = manager_from_app.resolve_workspace(input_path, workspace)
        if input_path.is_relative_to(resolved_workspace):
            raise HTTPException(
                status_code=400,
                detail="Workspace не может совпадать с исходной папкой или быть её родителем.",
            )
        return manager_from_app.submit(input_path, workspace)

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

    return app


def _manager(request: Request) -> ScanJobManager:
    manager: ScanJobManager = request.app.state.job_manager
    return manager
