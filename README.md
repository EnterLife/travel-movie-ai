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

Open PowerShell in the repository directory:

```powershell
Set-Location C:\Users\bdo\travel-movie-ai
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install TravelMovieAI:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify Python, FFmpeg, and the CLI:

```powershell
python --version
ffmpeg -version
ffprobe -version
travelmovieai --help
```

Scan a media folder:

```powershell
travelmovieai analyze `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026"
```

The command creates:

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
- `TRAVELMOVIEAI_BATCH_SIZE`: batch size reserved for processing stages.

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
- [Technical specification](docs/TECHNICAL_SPECIFICATION.md)
- [Agent development rules](AGENTS.md)

## License

See [LICENSE](LICENSE).
