# TravelMovieAI

TravelMovieAI is a local-first Python application that turns raw travel videos,
photos, and audio into a story-driven movie. It scans a media archive, detects
scenes, samples representative frames, evaluates visual quality, understands
scene content with Vision AI, groups scenes into events, builds an editing plan,
and renders a validated MP4 with FFmpeg.

The application provides a local web interface and a CLI. Source media is opened
read-only and is never modified or deleted.

## Current Status

Implemented:

- recursive media discovery with FFprobe metadata and SQLite caching;
- scene detection with PySceneDetect and a deterministic fallback;
- RGB PNG contact sheets sampled from the start, middle, and end of scenes;
- OpenCV analysis for sharpness, exposure, contrast, motion, shake, and noise;
- direct local Qwen2.5-VL and Florence-2 analysis with automatic model download;
- optional LM Studio compatibility mode;
- optional speech recognition with Faster Whisper;
- perceptual duplicate detection;
- event grouping, multimodal captions, storyboard generation, and scene ranking;
- explainable scene selection with `Auto`, `Include`, and `Exclude` overrides;
- generated, library, manual, or disabled music modes with ducking;
- transitions, quick preview, and final H.264/AAC rendering;
- NVIDIA NVENC acceleration with CPU fallback;
- automatic CPU, RAM, GPU, and worker-profile detection;
- global progress, per-stage progress bars, ETA, and live processing logs;
- incremental reruns that reuse compatible cached artifacts.

Not yet complete:

- semantic embeddings and FAISS archive search;
- full audio-event classification and beat-aware editing;
- manual event and timeline editor;
- subtitles, titles, credits, narration, and voice synthesis;
- HDR tone mapping, face-aware crop, and Ken Burns effects;
- persistent movie-job recovery and a distributable Windows installer.

## Requirements

- Windows 10 or Windows 11 x64;
- Python 3.12 or newer;
- FFmpeg and FFprobe available on `PATH`, or configured explicitly;
- free SSD space for frames, cache files, and intermediate video segments.

A GPU is optional: media scanning, quick montage, and local Vision AI all have
CPU fallbacks. An NVIDIA GPU with CUDA is strongly recommended for semantic
analysis.

Check the required tools:

```powershell
python --version
ffmpeg -version
ffprobe -version
```

## Quick Start

From the repository root:

```powershell
.\scripts\setup_windows.bat
.\scripts\run_web.bat
```

`setup_windows.bat`:

1. finds or installs Python 3.12 through `winget`;
2. finds or installs Gyan FFmpeg through `winget`;
3. creates `.venv`;
4. upgrades pip, setuptools, and wheel;
5. installs CUDA-enabled PyTorch when an NVIDIA GPU is detected;
6. installs all media, speech, Vision, embeddings, and development dependencies;
7. creates `.env` from `.env.example` without overwriting an existing file;
8. verifies Python imports, FFmpeg, and FFprobe.

If the environment already contains CPU-only PyTorch, setup removes that wheel
before installing the CUDA build. This is necessary because pip otherwise treats
CPU and CUDA wheels with the same public version as already satisfied.

Use a smaller runtime-only environment when development tools are unnecessary:

```powershell
.\scripts\setup_windows.bat --runtime-only
```

`run_web.bat` starts `main.py` and opens `http://127.0.0.1:8000`. If `.venv`
does not exist or lacks the basic web/video dependencies, it automatically runs
the runtime-only setup first.

Optional launcher arguments:

```powershell
.\scripts\run_web.bat --port 8080
.\scripts\run_web.bat --no-browser
```

Stop the server with `Ctrl+C`.

## Manual Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[video]"
python main.py
```

If PowerShell prevents activation, call the virtual-environment interpreter
directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[video]"
.\.venv\Scripts\python.exe main.py
```

Install optional dependency groups only when required:

```powershell
python -m pip install -e ".[speech]"
python -m pip install -e ".[vision]"
python -m pip install -e ".[embeddings]"
python -m pip install -e ".[all,dev]"
```

Models are not downloaded during import, startup, or test collection. The
selected model is downloaded only when semantic analysis first needs it.

## Generated Development Files

`.coverage` is a local SQLite data file created by `pytest-cov` when coverage is
measured:

```powershell
python -m pytest --cov=travelmovieai
```

It is not required by TravelMovieAI, is ignored by Git, and can be deleted at
any time. HTML coverage reports under `htmlcov/` are also disposable.

## FFmpeg Configuration

If FFmpeg is not on `PATH`, copy `.env.example` to `.env` and specify full
executable paths:

```dotenv
TRAVELMOVIEAI_FFMPEG_BINARY=C:\Tools\ffmpeg\bin\ffmpeg.exe
TRAVELMOVIEAI_FFPROBE_BINARY=C:\Tools\ffmpeg\bin\ffprobe.exe
```

Frame extraction uses RGB PNG instead of MJPEG. Sampling is clamped to the
actual video-stream duration, which handles DJI and similar files whose
container duration is longer than their video stream.

## AI Setup

### Local Auto: Qwen2.5-VL

No separate model server is required. Select `Local Auto` in the web interface
and start semantic analysis. TravelMovieAI downloads the model from Hugging Face
into `models/`, loads it directly through Transformers, and reuses the cached
weights on later runs.

Automatic selection uses Qwen2.5-VL-3B on systems with less than 10 GB of VRAM
and Qwen2.5-VL-7B on larger GPUs. The 32B model is available as an explicit
choice and is not selected automatically because it requires substantially more
RAM and VRAM.

On NVIDIA GPUs with less than 10 GB VRAM, Qwen automatically uses 4-bit NF4
quantization. This keeps all model layers on CUDA instead of offloading roughly
half of the model to system RAM. Larger GPUs use the model's native precision.

```dotenv
TRAVELMOVIEAI_VISION_PROVIDER=local
TRAVELMOVIEAI_VISION_MODEL=auto
TRAVELMOVIEAI_MODEL_CACHE=./models
TRAVELMOVIEAI_ALLOW_MODEL_DOWNLOAD=true
TRAVELMOVIEAI_DEVICE=auto
```

The first run requires internet access and several gigabytes of free disk space.
Set `TRAVELMOVIEAI_ALLOW_MODEL_DOWNLOAD=false` for cache-only offline operation.
Once downloaded, normal inference stays local and does not upload media.

### Florence-2

Florence-2 runs directly through Transformers and PyTorch:

```powershell
python -m pip install -e ".[vision]"
```

Model weights use the same application cache and are downloaded on first use.

```dotenv
TRAVELMOVIEAI_VISION_PROVIDER=florence
TRAVELMOVIEAI_VISION_MODEL=microsoft/Florence-2-large
TRAVELMOVIEAI_DEVICE=auto
```

### Optional LM Studio compatibility

LM Studio is no longer required. To use an already configured LM Studio server,
select `LM Studio` in the interface or configure:

```dotenv
TRAVELMOVIEAI_VISION_PROVIDER=lm-studio
TRAVELMOVIEAI_LM_STUDIO_URL=http://localhost:1234/v1
TRAVELMOVIEAI_VISION_MODEL=auto
```

The web application does not contact LM Studio unless this backend is selected.

### Faster Whisper

```powershell
python -m pip install -e ".[speech]"
```

Speech recognition is enabled separately in the web interface. It adds
transcripts, language, and confidence data, but increases processing time.

## Web Interface Workflow

1. Click the directory picker next to the source field, or enter a path manually.
2. Review the automatically selected workspace under
   `<repository>\workspace\<source-folder>`, or choose another directory.
3. Start media analysis.
4. Choose a Vision backend, model, story style, and render device.
5. Keep semantic and OpenCV analysis enabled for AI-directed editing.
6. Enable Faster Whisper only when speech matters.
7. Enable quick preview for the first iteration.
8. Configure duration, transitions, and music.
9. Start AI montage.
10. Monitor global progress, individual stage bars, ETA, and logs.
11. Review scenes and set `Auto`, `Include`, or `Exclude`.
12. Rerun the montage. Unchanged expensive analysis is reused from cache.
13. Disable preview and render the final movie.

Quick mode selects short clips chronologically without Vision AI. Semantic mode
adds scene detection, frame sampling, quality and Vision analysis, optional
speech recognition, duplicate detection, event grouping, story building, and
ranked selection.

## CLI

Show available commands:

```powershell
travelmovieai --help
python -m travelmovieai --help
```

Scan a media folder:

```powershell
travelmovieai analyze `
  --input "D:\Media\Trip" `
  --workspace "D:\TravelMovieAI\Trip"
```

Create a chronological montage:

```powershell
travelmovieai create `
  --input "D:\Media\Trip" `
  --workspace "D:\TravelMovieAI\Trip" `
  --output "D:\Movies\Trip.mp4" `
  --quick
```

Create a semantic montage:

```powershell
travelmovieai create `
  --input "D:\Media\Trip" `
  --workspace "D:\TravelMovieAI\Trip" `
  --output "D:\Movies\Trip.mp4" `
  --semantic `
  --style cinematic
```

Story styles: `cinematic`, `documentary`, `family`, `vlog`, `adventure`, and
`romantic`.

The `storyboard`, `render`, and `report` commands expose pipeline entry points.
Some later pipeline stages still contain placeholder behavior.

## Configuration

Create a local configuration file:

```powershell
Copy-Item .env.example .env
```

Do not commit `.env`.

| Variable | Purpose | Default |
| --- | --- | --- |
| `TRAVELMOVIEAI_WORKSPACE` | Default project workspace root | `./workspace` |
| `TRAVELMOVIEAI_DATABASE_FILENAME` | SQLite database filename | `project.db` |
| `TRAVELMOVIEAI_FFMPEG_BINARY` | FFmpeg command or full path | `ffmpeg` |
| `TRAVELMOVIEAI_FFPROBE_BINARY` | FFprobe command or full path | `ffprobe` |
| `TRAVELMOVIEAI_LM_STUDIO_URL` | OpenAI-compatible LM Studio API | `http://localhost:1234/v1` |
| `TRAVELMOVIEAI_LM_STUDIO_API_KEY` | Optional local API key | unset |
| `TRAVELMOVIEAI_VISION_PROVIDER` | `local`, `florence`, or `lm-studio` | `local` |
| `TRAVELMOVIEAI_VISION_MODEL` | Model identifier or `auto` | `auto` |
| `TRAVELMOVIEAI_MODEL_CACHE` | Downloaded local model cache | `./models` |
| `TRAVELMOVIEAI_ALLOW_MODEL_DOWNLOAD` | Download missing models on first use | `true` |
| `TRAVELMOVIEAI_VISION_TIMEOUT_SECONDS` | Vision request timeout | `120` |
| `TRAVELMOVIEAI_WHISPER_MODEL` | `medium` or `large-v3` | `medium` |
| `TRAVELMOVIEAI_DEVICE` | `auto`, `cuda`, `directml`, or `cpu` | `auto` |
| `TRAVELMOVIEAI_MUSIC_LIBRARY` | Local soundtrack directory | `./assets/music` |
| `TRAVELMOVIEAI_GENERATED_MUSIC_FILENAME` | Generated soundtrack filename | `generated_soundtrack.wav` |
| `TRAVELMOVIEAI_WORKERS` | Parallel worker override; `0` means auto | `0` |
| `TRAVELMOVIEAI_BATCH_SIZE` | Model batch override; `0` means auto | `0` |
| `TRAVELMOVIEAI_CLOUD_ENABLED` | Reserved explicit cloud switch | `false` |
| `TRAVELMOVIEAI_WEB_HOST` | Web server bind address | `127.0.0.1` |
| `TRAVELMOVIEAI_WEB_PORT` | Web server port | `8000` |
| `TRAVELMOVIEAI_WEB_HISTORY_LIMIT` | Saved scan-job history limit | `100` |

## Automatic Hardware Utilization

At the first montage, TravelMovieAI detects:

- logical CPU count;
- installed RAM;
- NVIDIA GPU and VRAM;
- CUDA availability in PyTorch and OpenCV;
- FFmpeg NVENC support.

The resulting profile separately selects concurrency for frame extraction,
OpenCV analysis, and segment rendering. CPU rendering divides FFmpeg threads
between concurrent jobs. NVENC is selected automatically when available and
falls back to `libx264` if initialization fails.

On the tested 16-thread CPU with 32 GB RAM and an RTX 3060, the automatic
profile uses up to 14 concurrent frame jobs, 16 quality-analysis workers, a
two-scene Vision batch, and four parallel render workers. These stages run
sequentially, so CPU, CUDA, NVDEC, and NVENC graphs are not expected to peak at
the same time.

GPU usage by stage:

- frame sampling: FFmpeg NVDEC with automatic CPU fallback per source;
- quality metrics: PyTorch CUDA for dense pixel metrics, with OpenCV/Pillow fallback;
- Vision AI: Qwen CUDA with 4-bit NF4 and hardware-sized batches;
- rendering: NVENC encoding; software transitions and audio filters can still use CPU.

Keep `TRAVELMOVIEAI_WORKERS=0` for automatic operation. Set a manual limit only
to reserve resources for other applications or reduce heat and power use.

## Supported Media

| Type | Extensions |
| --- | --- |
| Video | `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v` |
| Photo | `.jpg`, `.jpeg`, `.png`, `.heic` |
| Audio | `.mp3`, `.wav`, `.flac`, `.m4a` |

Windows paths with spaces, Unicode characters, and long filenames are
supported.

## Processing Pipeline

The canonical pipeline order is:

```text
Media Scan
-> Scene Detection
-> Frame Sampling
-> Visual Quality Analysis
-> Vision AI Analysis
-> Speech Analysis
-> Audio Analysis
-> Embeddings
-> Duplicate Detection
-> Scene Captioning
-> Event Detection
-> Story Builder
-> Scene Ranking
-> Music Selection
-> Narration
-> Voice Synthesis
-> Timeline Builder
-> Rendering
```

| Stage | Purpose | Status |
| --- | --- | --- |
| Media Scan | Discover media and cache FFprobe/EXIF metadata | Implemented |
| Scene Detection | Create bounded scenes with PySceneDetect or fallback | Implemented |
| Frame Sampling | Generate cached RGB contact sheets | Implemented |
| Visual Quality | Measure technical quality with OpenCV/Pillow | Implemented |
| Vision AI | Generate structured semantic scene understanding | Implemented |
| Speech | Transcribe scene audio with Faster Whisper | Implemented, optional |
| Audio Analysis | Classify speech, music, silence, crowds, and ambience | Planned |
| Embeddings | Semantic similarity and archive search | Planned |
| Duplicate Detection | Group visually similar scenes | Implemented |
| Scene Captioning | Merge Vision, quality, speech, and event context | Implemented |
| Event Detection | Group scenes into trip events | Implemented |
| Story Builder | Build opening, journey, highlights, and finale sections | Basic implementation |
| Scene Ranking | Explain selection and rejection decisions | Implemented |
| Music Selection | Generate or select a local soundtrack | Basic implementation |
| Narration and Voice | Generate and synthesize optional voice-over | Planned |
| Timeline Builder | Produce a declarative edit plan | Implemented |
| Rendering | Render, atomically replace, and validate the MP4 | Implemented |

Stage contract changes must update domain models, serialization, downstream
consumers, tests, and this README together.

## Vision AI Contract

Vision AI is the primary source of semantic understanding. OpenCV provides
measurable technical features only.

The preferred model is Qwen2.5-VL 3B, 7B, or 32B. Florence-2 base or large is the
local alternative. A validated scene response contains fields such as:

```json
{
  "caption": "A family walking along the beach during sunset.",
  "detailed_description": "The family continues along the shoreline.",
  "location_type": "beach",
  "activity": "walking",
  "emotion": "relaxing",
  "people_count": 4,
  "people_groups": ["family", "adults", "children"],
  "landmarks": [],
  "vision_score": 82,
  "score_factors": {
    "uniqueness": 70,
    "people": 85,
    "emotion": 80,
    "visual_quality": 76,
    "landmark": 0,
    "unusual_event": 35
  },
  "story_relevance": "Warm family moment.",
  "tags": ["family", "beach", "sunset"]
}
```

Landmarks must not be invented without visual or textual evidence. Provider,
model, prompt/schema version, and relevant settings are stored in cache
metadata.

## Selection and Story Requirements

The final scene score considers:

- Vision importance;
- technical quality;
- emotional and landmark value;
- uniqueness and event diversity;
- speech and future audio importance;
- duplicate and technical penalties;
- manual include/exclude decisions.

Every selected or rejected scene should retain an explainable reason. The story
is built before final editing decisions. Story Builder consumes structured
metadata and transcripts, not raw media.

## Architecture

| Package | Responsibility |
| --- | --- |
| `domain` | Stable enums and Pydantic contracts |
| `application` | Use cases and `TravelMovieService` |
| `pipeline` | Stage registry, contracts, and orchestration |
| `media` | Media discovery and metadata normalization |
| `analysis` | Scene, frame, quality, Vision, speech, and duplicate analysis |
| `story` | Events, storyboard, ranking, music, and future narration |
| `editing` | Declarative timeline construction and FFmpeg rendering |
| `infrastructure` | SQLite and external provider/process adapters |
| `web` | FastAPI jobs, API contracts, and package-local UI |

Dependency direction points inward: domain code does not import infrastructure
adapters. CLI commands remain thin, orchestration belongs to application and
pipeline modules, and renderers consume timelines rather than deciding the
story.

Main runtime flow:

```text
Browser or CLI
      |
      v
TravelMovieService
      |
      +--> Media Scan --> FFprobe/Pillow --> SQLite
      +--> Scene Detection --> PySceneDetect/fallback
      +--> Frame Sampling --> RGB PNG contact sheets
      +--> OpenCV Quality --> Vision AI --> optional Whisper
      +--> Duplicates --> Events --> Storyboard --> Ranking
      +--> Timeline + Music
      |
      v
QuickMontageRenderer --> FFmpeg --> FFprobe validation
```

Web entry point:

```text
scripts/run_web.bat -> main.py -> travelmovieai.web.server -> Uvicorn/FastAPI
```

## HTTP API

Important local endpoints:

```text
GET   /api/health
GET   /api/capabilities
POST  /api/dialogs/directory
POST  /api/scans
GET   /api/scans
GET   /api/scans/{id}
GET   /api/scans/{id}/result
POST  /api/movies
GET   /api/movies/{id}
GET   /api/movies/{id}/download
GET   /api/scenes
PATCH /api/scenes/{id}
GET   /api/scenes/{id}/thumbnail
```

Scan and movie jobs use bounded worker pools. Concurrent jobs targeting the
same workspace are rejected. The server binds to `127.0.0.1` by default and
does not currently implement authentication.

Movie-job responses include global progress, the active phase, elapsed time,
ETA, hardware profile, individual stage status, and up to 250 recent log
messages.

## Workspace

Generated data is stored under a project workspace:

```text
workspace/<project>/
|-- project.db
|-- frames/
|-- cache/
`-- artifacts/
    |-- analysis.json
    |-- scenes.json
    |-- frame_sampling.json
    |-- quality_analysis.json
    |-- vision_analysis.json
    |-- speech_analysis.json
    |-- duplicates.json
    |-- scene_descriptions.json
    |-- events.json
    |-- storyboard.json
    |-- selection_decisions.json
    |-- music_plan.json
    |-- quick_timeline.json
    |-- preview.mp4
    `-- final.mp4
```

The web interface fills this path automatically after a source folder is
selected. Editing the field or using the directory picker disables automatic
replacement for the current page session.

`project.db` stores media assets, scenes, events, scores, transcripts, and
manual overrides. SQLite uses foreign keys and WAL mode.

Critical JSON and media outputs are written atomically. Source media remains
read-only.

## Cache and Reproducibility

Media metadata is reused when path, size, and `modified_ns` match. Scene,
Vision, and speech cache keys include the relevant source metadata, time
boundaries, model, style, and prompt/schema version.

Manual scene decisions do not invalidate Vision analysis. A full project reset
can be performed by deleting only the workspace after carefully verifying its
path:

```powershell
Remove-Item -LiteralPath "D:\TravelMovieAI\Trip" -Recurse
```

Never delete the source-media directory.

## Rendering

The renderer:

- normalizes resolution, FPS, pixel format, and audio format;
- creates silent audio for sources without audio;
- prepares independent segments in parallel;
- applies `xfade` and `acrossfade` transitions;
- adds generated, library, or manual music;
- ducks music around source audio;
- uses `h264_nvenc` or `libx264`;
- writes the final movie atomically;
- validates video, audio, and duration with FFprobe.

During a movie job, `Pause` stops before the next scene or subtask and
`Continue` resumes it. `Full stop` cancels the remaining work while preserving
valid cache artifacts. An already running FFmpeg process or AI batch is allowed
to finish before the worker fully releases the workspace.

Preview mode is limited to 854x480 and 24 FPS. The standard output defaults to
1280x720 at 30 FPS.

## Troubleshooting

### A local Vision model cannot be loaded

- run `.\scripts\setup_windows.bat` to install the Vision dependencies;
- check internet access and free space in `models/`;
- use Qwen2.5-VL-3B on GPUs with 6-8 GB VRAM;
- verify CUDA with
  `.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"`;
- delete only an incomplete model snapshot from `models/` and retry the download.

### FFmpeg or FFprobe is unavailable

```powershell
Get-Command ffmpeg
Get-Command ffprobe
```

Add the FFmpeg `bin` directory to `PATH` or configure full paths in `.env`.

### No media files are found

Verify the selected directory, nested folders, and supported extensions. The
workspace should not contain the source-media directory.

### SQLite is busy

Close database viewers and other TravelMovieAI processes. A workspace cannot be
processed by multiple jobs simultaneously.

### The port is already in use

```powershell
.\scripts\run_web.bat --port 8080
```

### A model-heavy stage is too slow

- use preview mode;
- load a smaller Vision model;
- disable speech analysis when unnecessary;
- keep automatic workers enabled;
- verify that PyTorch CUDA and FFmpeg NVENC are using the expected GPU.

Windows Task Manager often opens the GPU page on the `3D` graph, which does not
represent AI inference. Change one graph to `CUDA` or `Compute_0`, or verify with:

```powershell
nvidia-smi --loop=2
```

The Vision AI log reports the actual runtime placement, for example
`cuda:0, 4-bit NF4`. During model loading and between generated tokens GPU usage
can fluctuate rather than remain at 100%.

## Product Requirements

TravelMovieAI follows these principles:

- **Local first:** normal operation remains local and offline.
- **Story before editing:** build narrative structure before final cuts.
- **Vision first:** use Vision AI for meaning and OpenCV for measurements.
- **Non-destructive:** never modify or delete source media.
- **Incremental:** reuse valid cached artifacts.
- **Reproducible:** record decisions, models, and relevant settings.
- **Optional acceleration:** CUDA, DirectML, and NVENC improve speed but do not
  become import-time requirements.
- **Explicit cloud:** cloud providers are never enabled silently.

The target scale is hundreds of videos and more than 100 GB of source media.
Processing must use bounded concurrency, batches, proxy media where necessary,
model reuse, disk-space checks, and CPU fallback.

Private data includes raw media, frames, faces, voices, transcripts, GPS
coordinates, and project databases. Telemetry is not required. Future cloud
mode must send only the minimum context and must not upload raw media without
separate explicit permission.

MVP acceptance criteria:

1. select a large local media directory;
2. produce repeatable local analysis;
3. create a reasonable preview;
4. inspect and override scene selection;
5. rerender without repeating unchanged expensive analysis;
6. produce a valid H.264/AAC MP4;
7. explain why scenes were selected or rejected.

## Roadmap

### P0: Long-running job reliability

- pause and cancel movie jobs;
- resume after process interruption;
- persist movie-job history;
- enforce disk-cache limits and cleanup;
- check free disk space before rendering;
- add managed SQLite migrations.

### P1: Editing quality

- semantic duplicates with embeddings and FAISS;
- GPS and embeddings in event detection;
- full Whisper segment boundaries;
- protection against cutting important speech;
- audio classification for speech, music, silence, crowds, laughter, applause,
  and ambient sound;
- beat-aware cuts and preservation of meaningful ambience;
- continuity rules for movement, light, location, and shot scale.

### P1: Story and manual editing

- direct local LLM story adapter with optional LM Studio compatibility;
- structured narrative and section duration budgets;
- multiple movie variants from one analysis;
- event and scene reordering;
- editable event titles, summaries, captions, transcripts, and landmarks;
- timeline versioning and comparison.

### P2: Visual processing, music, and narration

- Ken Burns effects for photos;
- face/object-aware crop;
- rotation metadata and vertical-video layouts;
- color and exposure normalization;
- HDR-to-SDR tone mapping;
- event titles, subtitles, credits, and safe-area validation;
- BPM analysis, beat grids, and storyboard-aware music;
- Piper or XTTS narration synthesis.

### P2: Performance

- batched Vision inference;
- persistent loaded-model reuse;
- proxy media for 4K and 8K;
- benchmarks for 500+ videos and 100+ GB;
- improved runtime and disk-space estimates.

### P3: Product delivery

- Windows installer;
- automatic FFmpeg and model diagnostics;
- project backup and export;
- HTML report;
- optional PySide6 desktop shell;
- documented provider/plugin interface.

## Development

Install development dependencies:

```powershell
python -m pip install -e ".[video,dev]"
```

Run checks:

```powershell
python -m pytest
python -m pytest --cov=travelmovieai
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m compileall -q src tests
```

The default test suite does not require internet access, GPU hardware, LM
Studio, or model downloads. FFmpeg integration tests use small synthetic media,
including limited-range YUV and Unicode paths.

Repository layout:

```text
main.py                         Local web entry point
scripts/setup_windows.bat       Complete Windows environment setup
scripts/run_web.bat             Windows bootstrap and launcher
src/travelmovieai/cli.py        Typer commands
src/travelmovieai/web/          API, jobs, and static interface
src/travelmovieai/core/         Settings and shared exceptions
src/travelmovieai/domain/       Stable data contracts
src/travelmovieai/application/  Use cases and project context
src/travelmovieai/pipeline/     Stage registry and orchestration
src/travelmovieai/media/        Discovery and metadata extraction
src/travelmovieai/analysis/     Media analysis
src/travelmovieai/story/        Story, events, ranking, and music
src/travelmovieai/editing/      Timeline and rendering
src/travelmovieai/infrastructure/ External adapters
tests/                          Fast unit and contract tests
assets/music/                   Distributable local soundtracks
assets/fonts/                   Fonts for future titles and reports
workspace/                      Generated project data; never commit
```

## Privacy and Security

- the web server listens on loopback by default;
- raw media and derived frames remain local;
- external processes receive argument lists rather than shell-built commands;
- API keys and authorization headers must not be logged;
- cloud mode is disabled by default;
- workspace data, `.env`, models, databases, frames, and rendered movies must
  not be committed.

## License

See [LICENSE](LICENSE).
