# TravelMovieAI

TravelMovieAI is an AI-powered video editing pipeline that automatically transforms raw travel footage into a complete movie.

Simply point the tool to a folder containing videos and photos from your trip, and TravelMovieAI will:

* Analyze videos and detect scenes
* Transcribe speech with Whisper
* Detect people, landmarks, activities, and key moments
* Remove duplicates and low-quality clips
* Build a coherent story using local or cloud LLMs
* Generate titles and optional voice-over narration
* Select music and create a final timeline
* Render a polished movie with FFmpeg

Designed to run locally on Windows, with optional support for LM Studio and Yandex Cloud models for advanced story generation.

## Features

* Fully automated travel video editing
* Scene detection and quality scoring
* Speech-to-text with Faster-Whisper
* Computer vision and event detection
* AI-generated storytelling
* Optional voice-over narration
* FFmpeg-based rendering
* Local-first architecture
* GPU acceleration support
* CLI-first, GUI planned

## Example

```bash
travelmovieai create \
  --input D:\Vacation\Japan2026 \
  --output D:\Movies\Japan.mp4
```

From hundreds of clips to a finished travel movie — automatically.

## Development

Requires Python 3.12 or newer and FFmpeg available on `PATH`.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
travelmovieai --help
pytest
```

Heavy processing dependencies are optional:

```bash
python -m pip install -e ".[video,speech,vision,embeddings]"
```

Copy `.env.example` to `.env` to override local model, device, workspace, and
external binary settings. See [docs/architecture.md](docs/architecture.md) for the
package boundaries and runtime artifact layout.

## Media Scan

The first pipeline stage is implemented. It recursively discovers supported
videos, photos, and audio, extracts metadata with FFprobe and photo EXIF data
with Pillow, and writes:

```text
workspace/<project>/project.db
workspace/<project>/artifacts/analysis.json
```

The SQLite index is incremental. Files whose path, size, and modification time
have not changed reuse cached metadata. Changed files are probed again, and
files removed from the source folder are removed from the project index.

Run the current analysis pipeline:

```powershell
travelmovieai analyze --input C:\Media\Trip --workspace C:\TravelMovieAI\Trip
```
