# Architecture

TravelMovieAI uses a layered, local-first architecture.

## Layers

- `domain`: stable media, scene, event, storyboard, and timeline contracts.
- `application`: CLI use cases and per-project execution context.
- `pipeline`: ordered processing stages and orchestration.
- `media`: file discovery and metadata extraction.
- `analysis`: scene, frame, vision, quality, speech, audio, and embedding analysis.
- `story`: event clustering, story generation, ranking, music, and narration.
- `editing`: timeline assembly and FFmpeg rendering.
- `infrastructure`: database and external tool/model adapters.

## Runtime Data

Each source folder gets an isolated workspace:

```text
workspace/<project>/
├── project.db
├── frames/
├── cache/
└── artifacts/
    ├── analysis.json
    ├── events.json
    ├── storyboard.json
    ├── timeline.json
    ├── render_config.json
    ├── report.html
    └── final.mp4
```

Pipeline stages are intentionally registered in the same order as the technical
specification. Unimplemented stages use placeholders so each implementation can land
independently without changing the CLI or orchestration contracts.

## Media Scan

`MediaScanStage` is the first concrete pipeline stage. It:

- recursively discovers supported video, photo, and audio extensions;
- excludes the project workspace when it is located under the source folder;
- reads duration, dimensions, FPS, creation tags, and location tags with FFprobe;
- supplements photo dimensions and GPS coordinates from EXIF with Pillow;
- stores per-file probe failures without stopping the whole project;
- synchronizes the `media_assets` table in `project.db`;
- writes an atomic `artifacts/analysis.json` snapshot.

Incremental reuse is based on normalized absolute path, file size, and nanosecond
modification time. Deleted source files are removed from the project database during
the next successful scan. Source files are never modified.
