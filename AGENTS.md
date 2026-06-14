# AGENTS.md

## Project

TravelMovieAI is a local-first Python application that turns raw travel videos,
photos, and audio into a story-driven movie.

The system scans media, detects scenes, samples frames, analyzes vision, speech,
audio, and quality, groups scenes into events, builds a storyboard and timeline,
and renders the final movie with FFmpeg.

The MVP is CLI-first. A PySide6 desktop UI may be added later, but the core
pipeline must remain usable without a GUI.

## Tech Stack

- Runtime: Python 3.12+.
- CLI: Typer.
- Local web UI: FastAPI, Uvicorn, and package-local HTML/CSS/JavaScript.
- Configuration and schemas: Pydantic and Pydantic Settings.
- Storage: SQLite and SQLAlchemy.
- Video processing: FFmpeg, FFprobe, PySceneDetect, and OpenCV.
- Speech recognition: Faster Whisper.
- Vision: Qwen2.5-VL, with Florence-2 as an alternative.
- Local LLM: LM Studio through an OpenAI-compatible API.
- Optional cloud LLM: Yandex GPT OSS 120B.
- Embeddings: sentence-transformers and FAISS.
- Tests and quality: pytest, Ruff, and mypy.

Heavy AI and media dependencies are optional dependency groups in
`pyproject.toml`. Keep the base package and CLI importable without installing or
initializing model-heavy dependencies.

## Repository Structure

- `src/travelmovieai/cli.py` - Typer commands and CLI option definitions.
- `main.py` - repository entry point for the local web server.
- `scripts/setup_windows.bat` - complete Windows system and virtual-environment setup.
- `scripts/run_web.bat` - one-click Windows environment bootstrap and web launch.
- `src/travelmovieai/web/` - HTTP API, background jobs, and static web interface.
- `src/travelmovieai/core/` - settings and shared exceptions.
- `src/travelmovieai/domain/` - stable enums and Pydantic data contracts.
- `src/travelmovieai/application/` - use cases and per-project execution context.
- `src/travelmovieai/pipeline/` - stage contracts, registry, and orchestration.
- `src/travelmovieai/media/` - media discovery and metadata extraction.
- `src/travelmovieai/analysis/` - scene, frame, vision, quality, speech, audio,
  and embedding analysis.
- `src/travelmovieai/story/` - event clustering, story generation, ranking,
  music selection, and narration.
- `src/travelmovieai/editing/` - timeline construction and FFmpeg rendering.
- `src/travelmovieai/infrastructure/` - database and external provider adapters.
- `tests/` - fast unit and contract tests.
- `configs/` - checked-in example or default configuration.
- `assets/music/` - local soundtrack assets that are safe to distribute.
- `assets/fonts/` - local fonts used by titles, subtitles, and reports.
- `README.md` - installation, usage, architecture, product requirements, and roadmap.
- `workspace/` - generated per-project data; never commit it.

See `README.md` for product requirements, package boundaries, runtime
artifacts, setup instructions, and the prioritized roadmap.

## Product Direction

- Keep normal operation fully local and offline.
- Cloud integrations must remain optional and explicitly enabled.
- Treat vision AI as the primary source of scene understanding.
- Use OpenCV for quality metrics, not semantic scene interpretation.
- Build the story before making final editing decisions.
- Preserve CLI workflows as the stable application interface.
- Keep pipeline output deterministic where practical and record enough metadata
  to reproduce decisions.
- Support incremental processing. A rerun must reuse valid cached artifacts
  instead of repeating expensive model work.
- Design for hundreds of videos and more than 100 GB of source media. Avoid
  loading an entire project or full videos into memory.

## Pipeline Contract

The canonical stage order is defined by `PipelineStage` and registered in
`build_default_pipeline()`:

1. Media scan.
2. Scene detection.
3. Frame sampling.
4. Visual quality analysis.
5. Vision analysis.
6. Speech analysis.
7. Audio analysis.
8. Embeddings.
9. Duplicate detection.
10. Scene captioning.
11. Event detection.
12. Story builder.
13. Scene ranking.
14. Music selection.
15. Narration.
16. Voice synthesis.
17. Timeline builder.
18. Rendering.

Do not reorder stages or change artifact contracts casually. When a stage
contract changes, update its domain model, serialization, downstream consumers,
  tests, and the README architecture documentation together.

Each stage should:

- implement the `Stage` contract;
- receive a `ProjectContext`;
- return a `StageResult`;
- write generated data inside the project workspace;
- be restartable after interruption;
- skip work only when its cached inputs and configuration still match;
- report expected failures through project-specific exceptions;
- avoid importing or initializing optional models until the stage needs them.

The scaffold initially contains placeholder stages. Replace placeholders
incrementally without changing CLI command behavior or stage order.

## Architecture Rules

- Keep CLI commands thin: validate options, call `TravelMovieService`, and print
  a concise result.
- Keep orchestration in `application` and `pipeline`, not in provider adapters.
- Keep stable business data in `domain`; do not import infrastructure modules
  from domain code.
- Keep FFmpeg, FFprobe, LM Studio, Whisper, vision model, and database details
  behind `infrastructure` adapters.
- Keep analysis modules focused on interpreting media and producing structured
  metadata.
- Keep story modules independent from rendering details.
- Keep timeline models declarative. The renderer should consume a timeline
  rather than decide the story.
- Prefer typed Pydantic models over unstructured dictionaries for persisted
  artifacts and cross-stage contracts.
- Prefer SQLAlchemy and structured JSON APIs over ad hoc SQL or string-built
  JSON.
- Preserve CPU fallback. CUDA and DirectML acceleration must improve execution,
  not become mandatory for ordinary imports or basic commands.

## Coding Rules

- Keep edits focused on the requested behavior.
- Preserve user changes already present in the working tree.
- Follow existing package boundaries before adding a new abstraction.
- Use Python 3.12 typing and explicit return types.
- Keep public contracts small and typed.
- Use `pathlib.Path` for filesystem paths.
- Use timezone-aware dates when creating new timestamps.
- Do not use mutable default arguments.
- Do not hide broad failures with `except Exception` unless re-raising with
  useful context at an application boundary.
- Keep comments short and limited to non-obvious decisions.
- Remove unused imports, helpers, parameters, and dead code.
- Do not edit `.env` unless explicitly requested.
- Never hardcode API keys, tokens, user media paths, or machine-specific model
  paths.
- Do not commit databases, model weights, generated frames, media, rendered
  movies, caches, reports, or virtual environments.

Do not modify generated/runtime folders unless the task explicitly requires it:

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `.cache/`
- `models/`
- `workspace/`
- `output/`

## Media And Process Rules

- Probe metadata with FFprobe rather than parsing FFmpeg console output.
- Invoke external processes with argument lists, never shell-built command
  strings containing user paths.
- Check process return codes and include actionable stderr in raised errors.
- Handle Windows paths, spaces, Unicode filenames, and long media filenames.
- Write artifacts atomically where interruption could otherwise leave a valid
  looking partial file.
- Use bounded worker pools and batches. Do not create one process or model
  instance per scene.
- Reuse loaded models within a stage execution.
- Keep frame extraction and temporary files inside the project workspace.
- Preserve original source media. The pipeline must never edit or delete input
  files.
- Validate output paths before rendering or replacing an existing movie.

## Local AI Rules

- Do not download models during module import or test collection.
- Make model names, devices, precision, and batch sizes configurable.
- Keep provider-specific prompts and response parsing close to the adapter that
  owns them.
- Require structured output from vision and LLM providers where supported.
- Validate all model output before persisting or passing it downstream.
- Record provider, model, prompt/schema version, and relevant settings in cache
  metadata.
- Set finite timeouts for local and cloud HTTP calls.
- Do not silently fall back to cloud processing.
- Cloud mode must send only the minimum required context and must not upload raw
  media unless a future feature explicitly documents and requests that behavior.

## Data And Privacy

- Treat source media, faces, voices, transcripts, GPS coordinates, and project
  databases as private user data.
- Keep raw media and derived frames local by default.
- Do not include real user paths, transcripts, coordinates, or frame contents in
  logs, fixtures, documentation, or final responses.
- Redact API keys and authorization headers from logs and errors.
- Avoid telemetry and analytics in the local-first pipeline.
- Tests must use synthetic fixtures or generated tiny media samples.

## Development Setup

Create a virtual environment and install the base package with development
tools:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Install only the heavy groups needed for the current task:

```powershell
python -m pip install -e ".[video]"
python -m pip install -e ".[speech]"
python -m pip install -e ".[vision]"
python -m pip install -e ".[embeddings]"
```

Install all processing dependencies:

```powershell
python -m pip install -e ".[all,dev]"
```

FFmpeg and FFprobe must be installed separately and available on `PATH`, or
configured through the corresponding `TRAVELMOVIEAI_*` settings.

Copy `.env.example` to `.env` only for local development configuration. Never
commit `.env`.

## Useful Commands

Run the CLI:

```powershell
travelmovieai --help
python -m travelmovieai --help
travelmovieai analyze --input C:\Media\Trip
travelmovieai create --input C:\Media\Trip --output C:\Movies\Trip.mp4
```

Run checks:

```powershell
python -m pytest
python -m pytest --cov=travelmovieai
python -m compileall -q src tests
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

When the package is not installed yet:

```powershell
$env:PYTHONPATH="src"
python -m pytest
python -m travelmovieai --help
```

Check external binaries:

```powershell
ffmpeg -version
ffprobe -version
```

## Test Strategy

- Add unit tests for domain validation, scoring, cache decisions, and pure
  transformations.
- Add contract tests for stage order, artifact schemas, and provider adapters.
- Use fake providers for Whisper, vision, embeddings, LLM, and voice synthesis.
- Mock process boundaries in ordinary unit tests.
- Use tiny generated media for FFmpeg integration tests and mark model-heavy or
  slow tests explicitly.
- Do not require internet access, GPU hardware, LM Studio, or model downloads for
  the default test suite.
- Test interrupted and repeated execution when adding caching or persistence.
- Test paths containing spaces and Unicode when changing media discovery or
  external process invocation.
- Verify both success and actionable failure messages for missing binaries,
  unavailable providers, corrupt media, and invalid model responses.

Run the smallest relevant checks during development, then run the full fast
suite before finishing. If model-heavy or end-to-end rendering checks were not
run, state that explicitly.

## Documentation Rules

- Keep all project documentation in `README.md`.
- Update `README.md` when installation, user-facing behavior, architecture,
  pipeline contracts, product requirements, or roadmap status changes.
- Update `.env.example` when adding or renaming settings.
- Do not create separate setup, architecture, specification, or roadmap
  documents unless explicitly requested.
- Keep command examples valid for Windows PowerShell, the primary development
  environment.

## Commit Message Suggestions

After each completed work chunk, include a suggested commit message in the final
response.

Use a lowercase prefix and a short lowercase summary in one line without a
period:

- `add:` for a new feature, stage, adapter, test coverage, or user flow.
- `fix:` for a bug or broken behavior.
- `upd:` for changes to existing behavior, configuration, or documentation.
- `refactor:` for internal restructuring without behavior changes.
- `docs:` for documentation-only changes.
- `test:` for test-only maintenance.
- `chore:` for tooling, dependencies, or repository maintenance.

Examples:

- `add: media scan pipeline stage`
- `fix: handle unicode paths in ffprobe adapter`
- `upd: document local vision model settings`
- `refactor: isolate ffmpeg process adapter`

## Before Finishing Work

- Review all changed files and the working tree.
- Run the smallest relevant checks that cover the change.
- Run the fast test suite when feasible.
- Mention checks that passed.
- Mention checks that could not run and why.
- Mention remaining risk when FFmpeg, GPU, model-heavy, or full rendering flows
  were not exercised end to end.
- Include one suggested commit message.
