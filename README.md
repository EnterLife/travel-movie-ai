# TravelMovieAI

TravelMovieAI is a local-first Python pipeline for turning travel videos, photos,
and audio into a story-driven movie.

The target product will detect scenes, analyze image and sound, build a story,
create an editing timeline, and render a finished video. Development is
incremental: the media catalog and SQLite project storage are currently
implemented, while the later AI and rendering stages are still scaffolds.

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
- one-click Windows launcher in `scripts\run_web.bat`;
- Windows paths containing spaces and Unicode characters.

Not implemented yet:

- scene detection and frame sampling;
- vision, speech, audio, and quality analysis;
- embeddings and duplicate detection;
- event grouping and storyboard generation;
- music, narration, timeline creation, and rendering;
- HTML report generation.

The commands `create`, `storyboard`, `render`, and `report` are reserved in the
CLI, but they do not yet create their final artifacts.

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

Both the web interface and CLI create:

```text
D:\TravelMovieAI\Japan2026\
├── project.db
├── frames\
├── cache\
└── artifacts\
    └── analysis.json
```

Run the same command again to reuse cached metadata for unchanged files.

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
