# TravelMovieAI

TravelMovieAI is a local-first Python pipeline for turning travel videos, photos,
and audio into a story-driven movie.

The target product will detect scenes, analyze image and sound, build a story,
create an editing timeline, and render a finished video. Development is
incremental: the media catalog and SQLite project storage are currently
implemented. The first local semantic montage flow is also available.

## Current Status

Implemented:

- recursive media discovery;
- video, photo, and audio format classification;
- FFprobe metadata extraction;
- photo dimensions and EXIF GPS extraction with Pillow;
- incremental SQLite indexing;
- atomic `analysis.json` generation;
- CLI command `travelmovieai analyze`;
- local web interface with background scan jobs;
- persistent web job history and workspace conflict protection;
- one-click quick montage from videos and photos;
- PySceneDetect scene boundaries with a deterministic uniform fallback;
- representative frame extraction and cached scene metadata;
- start/middle/end contact sheets for stronger scene understanding;
- local Qwen-compatible vision analysis through LM Studio;
- selectable loaded LM Studio models in the web interface;
- OpenCV visual quality scoring used by scene ranking;
- explainable semantic scene ranking;
- local procedural soundtrack generation guided by scene metrics;
- local soundtrack selection, audio ducking, and FFmpeg transitions;
- NVIDIA CUDA/NVENC rendering with automatic CPU fallback;
- H.264/AAC MP4 preview and download from the web interface;
- one-click Windows launcher in `scripts\run_web.bat`;
- Windows paths containing spaces and Unicode characters.

Not implemented yet:

- full start/middle/end frame sampling and thumbnail gallery;
- speech, audio classification, and visual quality analysis;
- embeddings and duplicate detection;
- event grouping and storyboard generation;
- narration, subtitles, titles, and manual storyboard editing;
- HTML report generation.

The web interface can create either a chronological quick montage or a locally
AI-ranked montage. The latter requires LM Studio with a loaded vision model.
The `storyboard`, advanced `render`, and `report` commands remain reserved for
later pipeline stages.

## Requirements

- Windows 10 or Windows 11;
- Python 3.12 or newer;
- FFmpeg and FFprobe available on `PATH`;
- Git, if the repository is being cloned rather than downloaded.

GPU hardware and local AI models are not required for the currently implemented
Media Scan stage.

## Quick Start

Install Python 3.12+ and FFmpeg, then run:

```powershell
Set-Location C:\Users\bdo\travel-movie-ai
.\scripts\run_web.bat
```

You can also double-click `scripts\run_web.bat` in Explorer.

On the first launch, the script:

- creates `.venv`;
- installs the base project dependencies;
- starts the server at `http://127.0.0.1:8000`;
- opens the interface in the default browser.

Paste the full path to the media folder, optionally set a workspace, and click
`Запустить анализ`.

After the scan:

1. Adjust the target movie duration and clip limits.
2. Select one of the vision models currently loaded in LM Studio.
3. Keep semantic and OpenCV analysis enabled for the best automatic selection.
4. Select `Auto`, CUDA, or CPU rendering and configure the music director.
5. Click `Запустить AI-монтаж`.
6. Preview or download the generated MP4.

Stop the server with `Ctrl+C` or close its console window.

## Manual Server Launch

After installing the project, start the interface with:

```powershell
python main.py
```

Options:

```powershell
python main.py --port 8080
python main.py --no-browser
travelmovieai-web --host 127.0.0.1 --port 8000
```

The server binds to `127.0.0.1` by default. API documentation is available at
`http://127.0.0.1:8000/api/docs`.

## CLI Usage

The original CLI remains available:

```powershell
travelmovieai analyze `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026"
```

Create a quick montage directly from CLI:

```powershell
travelmovieai create `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026" `
  --output "D:\Movies\Japan2026.mp4"
```

Create a semantic montage through local LM Studio:

```powershell
travelmovieai create `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026" `
  --output "D:\Movies\Japan2026.mp4" `
  --semantic `
  --style cinematic
```

The web interface and CLI create:

```text
D:\TravelMovieAI\Japan2026\
├── project.db
├── frames\
├── cache\
└── artifacts\
    ├── analysis.json
    ├── scenes.json
    ├── vision_analysis.json
    ├── quick_timeline.json
    └── final.mp4
```

Run the same command again to reuse cached metadata for unchanged files.

## Movie Builder Behavior

Quick mode:

- orders usable videos and photos chronologically;
- takes a centered excerpt from long videos;
- displays photos for a configurable duration;
- skips files with scan errors;
- normalizes all clips to a shared H.264/AAC profile;
- adds silent audio when a source has no audio track;
- works without an AI server.

Semantic mode additionally:

- detects cuts with PySceneDetect when the optional video group is installed;
- falls back to bounded uniform scenes when PySceneDetect is unavailable;
- extracts one representative frame per scene;
- asks the configured local vision model for a validated JSON description;
- ranks scenes by importance and semantic diversity;
- preserves chronological order after selecting the strongest scenes;
- reuses valid scene and vision cache data on repeated runs.

Both modes can apply FFmpeg video/audio transitions. Music modes include:

- `AI Auto`: derive a calm/cinematic/warm/energetic profile from OpenCV metrics
  and scene emotion, then generate a deterministic local ambient WAV;
- generated music with a manually selected profile;
- a track from source media or `assets/music`;
- an explicit local file;
- no music.

Music is faded and ducked under source audio. Rendering uses NVIDIA
`h264_nvenc` when `Auto` or CUDA is selected and available, with CPU fallback
for `Auto`. The builder writes `quality_analysis.json`, `music_plan.json`,
`quick_timeline.json`, and `final.mp4`.

This is not yet the complete Story Builder: event clustering, duplicate
removal, quality scoring, speech context, titles, and narration are pending.

## Supported Media

Video:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.m4v`

Photos:

- `.jpg`
- `.jpeg`
- `.png`
- `.heic`

Audio:

- `.mp3`
- `.wav`
- `.flac`
- `.m4a`

HEIC support depends on the codecs available to the installed FFmpeg and Pillow
environment. A HEIC file can be discovered even when some metadata cannot be
decoded.

## Workspace Behavior

If `--workspace` is omitted, the default directory is:

```text
<current-directory>\workspace\<input-folder-name>
```

For example:

```powershell
travelmovieai analyze --input "D:\Vacation\Japan2026"
```

when launched from the repository root writes to:

```text
C:\Users\bdo\travel-movie-ai\workspace\Japan2026
```

Source media is never edited or deleted. The workspace can be removed to reset
the local index and force a complete rescan.

## Configuration

Copy the example configuration:

```powershell
Copy-Item .env.example .env
```

The default Media Scan setup usually needs no changes. Important settings:

- `TRAVELMOVIEAI_WORKSPACE`: default workspace parent directory;
- `TRAVELMOVIEAI_DATABASE_FILENAME`: SQLite filename;
- `TRAVELMOVIEAI_FFMPEG_BINARY`: FFmpeg executable name or full path;
- `TRAVELMOVIEAI_FFPROBE_BINARY`: FFprobe executable name or full path;
- `TRAVELMOVIEAI_LM_STUDIO_URL`: local OpenAI-compatible endpoint;
- `TRAVELMOVIEAI_VISION_MODEL`: `auto` or a loaded vision model identifier;
- `TRAVELMOVIEAI_VISION_TIMEOUT_SECONDS`: finite request timeout;
- `TRAVELMOVIEAI_MUSIC_LIBRARY`: local soundtrack directory;
- `TRAVELMOVIEAI_GENERATED_MUSIC_FILENAME`: generated WAV filename;
- `TRAVELMOVIEAI_WORKERS`: worker limit reserved for processing stages;
- `TRAVELMOVIEAI_BATCH_SIZE`: batch size reserved for processing stages;
- `TRAVELMOVIEAI_WEB_HOST`: local web server host;
- `TRAVELMOVIEAI_WEB_PORT`: local web server port;
- `TRAVELMOVIEAI_WEB_HISTORY_LIMIT`: retained web job records.

Do not commit `.env`, model files, source media, databases, caches, or rendered
movies.

## Developer Setup

Install the project with test and quality tools:

```powershell
python -m pip install -e ".[dev]"
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

Optional dependency groups:

```powershell
python -m pip install -e ".[video]"
python -m pip install -e ".[speech]"
python -m pip install -e ".[vision]"
python -m pip install -e ".[embeddings]"
python -m pip install -e ".[all,dev]"
```

These groups prepare dependencies for future stages. Installing them does not
make unimplemented pipeline stages functional.

## Documentation

- [Detailed installation and usage guide](docs/installation-and-usage.md)
- [Architecture](docs/architecture.md)
- [Development roadmap](docs/roadmap.md)
- [Technical specification](docs/TECHNICAL_SPECIFICATION.md)
- [Agent development rules](AGENTS.md)

## License

See [LICENSE](LICENSE).
