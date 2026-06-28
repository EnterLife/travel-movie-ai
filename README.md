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
- optional speech recognition with Faster Whisper;
- perceptual duplicate detection;
- event grouping, multimodal captions, storyboard generation, and scene ranking;
- explainable scene selection with `Auto`, `Include`, and `Exclude` overrides;
- energy-aware semantic clip pacing with speech and people protection;
- generated, library, manual, or disabled music modes with ducking;
- cut-only editing, quick preview, and final H.264/AAC rendering;
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
7. installs Git when needed and prepares ACE-Step in an isolated environment;
8. validates the checked-in `configs/settings.toml`;
9. verifies Python imports, FFmpeg, and FFprobe.

If the environment already contains CPU-only PyTorch, setup removes that wheel
before installing the CUDA build. This is necessary because pip otherwise treats
CPU and CUDA wheels with the same public version as already satisfied.

Use a smaller runtime-only environment when development tools are unnecessary:

```powershell
.\scripts\setup_windows.bat --runtime-only
```

This still prepares the ACE-Step runtime. To install the application without
neural music generation:

```powershell
.\scripts\setup_windows.bat --runtime-only --skip-music-ai
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

If FFmpeg is not on `PATH`, specify full executable paths in
`configs/settings.toml`:

```toml
ffmpeg_binary = 'C:\Tools\ffmpeg\bin\ffmpeg.exe'
ffprobe_binary = 'C:\Tools\ffmpeg\bin\ffprobe.exe'
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

For higher scene-understanding quality, select `Qwen2.5-VL-7B-Instruct` in the
web UI. On a 6 GB NVIDIA GPU the application loads it in 4-bit NF4 mode and
automatically places part of the model in system RAM. This keeps CUDA busy while
using available memory, but it is slower than 3B. The explicit 32B option is
experimental on consumer hardware and may be impractically slow even with
offload. Use 3B for previews and 7B for the final semantic pass.

When a selected Qwen model does not fit in native precision, TravelMovieAI uses
4-bit NF4 quantization. Models that still exceed the available VRAM use
Accelerate placement across CUDA and system RAM. Smaller models remain entirely
on CUDA whenever possible.

```toml
vision_provider = "local"
vision_model = "auto"
model_cache = "models"
allow_model_download = true
device = "auto"
```

The first run requires internet access and several gigabytes of free disk space.
Set `allow_model_download = false` for cache-only offline operation.
Once downloaded, normal inference stays local and does not upload media.

### Florence-2

Florence-2 runs directly through Transformers and PyTorch:

```powershell
python -m pip install -e ".[vision]"
```

Model weights use the same application cache and are downloaded on first use.

```toml
vision_provider = "florence"
vision_model = "microsoft/Florence-2-large"
device = "auto"
```

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
   Keep the workspace separate from the source folder; neither directory may be
   nested inside the other.
3. Start media analysis.
4. Choose a Vision backend, model, story style, and render device.
5. Keep semantic and OpenCV analysis enabled for AI-directed editing.
6. Enable Faster Whisper only when speech matters.
7. Enable quick preview for the first iteration.
8. Configure duration and music.
9. Start AI montage.
10. Monitor global progress, individual stage bars, ETA, and logs.
11. Review scenes and set `Auto`, `Include`, or `Exclude`.
12. Rerun the montage. Unchanged expensive analysis is reused from cache.
13. Disable preview and render the final movie.

Quick mode selects short clips chronologically without Vision AI. Semantic mode
adds scene detection, frame sampling, quality and Vision analysis, optional
speech recognition, duplicate detection, event grouping, story building, and
ranked selection.

Semantic mode is intentionally selective. It does not try to use every video in
the folder and it does not fill the target duration with weak material.
`min_semantic_score` is a base quality target, but the actual threshold is
computed from the score distribution of the current project: it rises for strong
archives and relaxes for consistently modest material. The `max_scenes_per_source`
setting is a strict diversity guard by default when more than one source video is
available, so one strong roll cannot dominate the movie. Set
`strict_source_diversity=false` only when filling the requested duration is more
important than source variety. A single long source video can still contribute
multiple scenes because there is no alternate source to use. Use scene overrides
when a specific fragment must be included or excluded.

Semantic mode preserves capture chronology by default. Vision AI scores scenes
and describes their story value, but the final timeline uses deterministic
constraints for chronology, source diversity, event diversity, duplicate
rejection, and technical quality. Set `preserve_chronology=false` for a more
storyboard-driven order, or increase `chronology_tolerance_seconds` to allow
small story-based reorderings inside a time window.

Frame sampling depth is controlled by `analysis_quality_mode`. `fast` samples 3
frames per scene, `balanced` samples 5, and `deep` samples 9. The web interface
defaults AI edits to `deep` so the first semantic pass sees more of each scene.
Use `fast` for rough previews and `balanced` when runtime matters more than
maximum scene-understanding quality.

For long scenes, semantic montage does not blindly cut the middle of the scene.
It builds candidate windows inside the scene and prefers explicit highlight
windows, then the best visual panel from the sampled contact sheet, then a
neutral center cut. This keeps the final movie focused on the strongest moment
inside each selected scene.

Visual quality analysis stores per-panel scores and ready-to-use
`candidate_windows` in scene metadata. Future audio, face, speech, and object
analysis can add their own candidate windows to the same contract, allowing the
timeline builder to choose the best moment inside a long source scene without
changing the renderer.

Audio Analysis stores scene-level labels such as `speech`, `silence`, `wind`,
`music`, `crowd`, `water`, and `transport`. It adds audio candidate windows,
boosts scenes with speech or useful ambience, and penalizes strong wind or
transport noise during ranking.

Speech Analysis stores Whisper segment boundaries in scene metadata when the
provider returns them. Semantic timeline planning uses those boundaries as
speech-safe candidate windows and penalizes source windows whose start or end
would cut through a spoken phrase.

When music sync is enabled and the selected music plan contains a beat grid, the
final timeline softly nudges neighboring clip durations so scene changes can land
on strong beats or music accents. The adjustment keeps the same selected scenes,
stays inside the available source scene window, and preserves the planned movie
duration where possible.

Story Timeline Optimizer follows source chronology by default and uses
storyboard sections as tie-breakers inside identical or configured-near capture
times. When chronology preservation is disabled, selected clips are arranged as
opening, journey, highlight, and finale before falling back to source chronology.
It also applies section duration budgets and story-aware pacing for longer
movies. High-energy, shaky, or noisy moments are cut tighter, while speech and
people moments resist overly aggressive shortening. The pacing decision uses
Vision emotion/activity, OpenCV motion and shake metrics, speech boundaries, and
audio context, and the reason is written into the selection explanation. The
optimizer avoids adjacent repeats across location, activity, shot type, shot
scale, camera motion, movement direction, lighting, tags, and large brightness
jumps. `semantic_diversity_weight` controls how strongly these repeat penalties
affect selection. Semantic
timeline clips use direct cuts with no visual transition, because most travel
footage looks cleaner and avoids dissolve artifacts that can resemble
pixelization. One strong but repetitive location or activity should not fill the
whole movie when varied alternatives are available.

### Generated Lounge Music

`AI Auto` and `Generate locally` create a soundtrack entirely on the local
machine. The default AI engine is
[ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5), a specialized
open-source music generation model. It generates an instrumental composition
from the story style, BPM, duration budget, and music cue sheet.

The unified Windows setup:

1. installs ACE-Step into the isolated `.cache/ace-step` environment;
2. keeps its dependencies separate from the main `.venv`.

The first generation then:

1. downloads model weights into `models/ace-step`;
2. detects the GPU tier and enables CPU offload on low-VRAM systems;
3. generates a bounded base WAV and normalizes it for the exact movie duration.

This does not replace packages in the main `.venv`. On a 6 GB NVIDIA GPU,
ACE-Step uses its 2B Turbo model with low-VRAM offload. The initial installation
and model download require internet access, several gigabytes of disk space,
and significantly more time than subsequent runs.

The music engine options are:

- `AI Auto`: use ACE-Step and fall back to deterministic procedural music if
  model installation or generation fails;
- `ACE-Step only`: require neural generation and show an actionable error
  instead of falling back;
- `Procedural synthesis`: use the fast built-in lounge arranger without model
  downloads.

ACE-Step is prepared together with the application:

```powershell
.\scripts\setup_windows.bat
```

With `Very calm default`, TravelMovieAI now favors very quiet, low-register,
melodic background music by default. Lounge, warm, energetic, and cinematic
profiles remain available as explicit choices, but automatic music avoids high
notes, bright bells, sharp synths, cymbal shimmer, loud hits, aggressive
percussion, and dramatic build-ups.

`Synchronize with editing` is enabled by default. The application first builds
the final clip timeline and then requests one composition for the movie. For
longer edits, ACE-Step generation is capped at 90 seconds and then normalized
to the full timeline so 120-second and longer renders do not depend on a single
very long model pass. If the model returns a shorter WAV, TravelMovieAI extends
it to the full timeline instead of filling the remainder with silence. A cue sheet is a
first-class contract with arrangement sections, BPM, intensity, and restrained
accent points. It places musical structure at:

- cut points between clips;
- changes between detected trip events;
- the center of high-scoring Vision AI scenes;
- the opening and final moments.

The cue sections, beat grid, timestamps, strengths, BPM, intensity, arrangement
version, generator, model identifier, and fallback status are stored in
`artifacts/music_plan.json`. Local music models receive both the prompt and the
cue sheet. The prompt asks for a clean low-register instrumental travel
underscore with a recurring motif, mellow midrange melody, no vocals or lyrics,
no high-pitched sounds, polished production, and mastering headroom so the
rendered movie can duck music under source audio without clipped peaks. The
procedural fallback also follows the sections, varying melody energy, stereo
width, electric-piano tones, muted-guitar pulses, and restrained low accent
layers across intro, journey, highlight, and finale parts instead of producing
a flat loop. Rebuilding the same timeline uses a
deterministic seed, while changing clip order, duration, or selected highlights
reshapes the composition to match the new movie.

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

The `create --semantic` command runs the canonical pipeline through rendering.
The `storyboard`, `render`, and `report` commands expose additional pipeline
entry points. Some optional AI stages still contain placeholder behavior until
narration, voice synthesis, and embeddings are promoted into the canonical
pipeline.

## Configuration

Runtime settings live in the checked-in `configs/settings.toml`. It contains no
secrets or remote service credentials. CLI and web entry points validate this
file at startup; unknown keys and invalid values fail with an actionable error.

| Key | Purpose | Default |
| --- | --- | --- |
| `workspace` | Default project workspace root | `workspace` |
| `database_filename` | SQLite database filename | `project.db` |
| `ffmpeg_binary` | FFmpeg command or full path | `ffmpeg` |
| `ffprobe_binary` | FFprobe command or full path | `ffprobe` |
| `frame_extraction_timeout_seconds` | Per-scene FFmpeg frame extraction timeout | `120` |
| `render_timeout_seconds` | Per-FFmpeg render, validation, or music-normalization timeout | `7200` |
| `vision_provider` | `local`, `qwen`, or `florence` | `local` |
| `vision_model` | Vision model identifier or `auto` | `auto` |
| `model_cache` | Downloaded local model cache | `models` |
| `allow_model_download` | Download missing models on first use | `true` |
| `whisper_model` | `medium` or `large-v3` | `medium` |
| `device` | `auto`, `cuda`, `directml`, or `cpu` | `auto` |
| `music_library` | Local soundtrack directory | `assets/music` |
| `music_model` | Local music model identifier or `auto` | `auto` |
| `generated_music_filename` | Generated soundtrack filename | `generated_soundtrack.wav` |
| `workers` | Parallel worker override; `0` means auto | `0` |
| `batch_size` | Model batch override; `0` means auto | `0` |
| `web_host` | Web server bind address | `127.0.0.1` |
| `web_port` | Web server port | `8000` |
| `web_history_limit` | Saved scan-job history limit | `100` |

## Automatic Hardware Utilization

At the first montage, TravelMovieAI detects:

- logical CPU count;
- installed RAM;
- NVIDIA GPU and VRAM;
- CUDA availability in PyTorch and OpenCV;
- FFmpeg NVENC support.

The resulting profile separately selects concurrency for frame extraction,
OpenCV analysis, Vision AI batching, and segment rendering. CPU rendering
divides FFmpeg threads between concurrent jobs. NVENC is selected automatically
when available and falls back to `libx264` if initialization fails.

On the tested 16-thread CPU with 32 GB RAM and an RTX 3060, the automatic
profile uses up to 14 concurrent frame jobs, 16 quality-analysis workers, a
two-scene Vision batch, and four parallel render workers. On high-memory
workstations, the frame-extraction and OpenCV-analysis caps rise further while
remaining bounded. These stages run sequentially, so CPU, CUDA, NVDEC, and
NVENC graphs are not expected to peak at the same time.

GPU usage by stage:

- frame sampling: FFmpeg NVDEC with automatic CPU fallback per source and a
  per-scene timeout;
- quality metrics: PyTorch CUDA for dense pixel metrics, with OpenCV/Pillow fallback;
- Vision AI: Qwen CUDA with 4-bit NF4 and hardware-sized batches;
- rendering: NVENC encoding; audio filters can still use CPU.

Keep `workers = 0` for automatic operation. Set a manual limit only
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
| Audio Analysis | Classify speech, silence, wind, music, crowds, water, transport, and ambience | Implemented |
| Embeddings | Semantic similarity and archive search | Planned |
| Duplicate Detection | Group visually similar scenes | Implemented |
| Scene Captioning | Merge Vision, quality, speech, and event context | Implemented |
| Event Detection | Group scenes into trip events | Implemented |
| Story Builder | Build opening, journey, highlights, finale, story budgets, and diverse clip ordering | Implemented |
| Scene Ranking | Explain selection and rejection decisions | Implemented |
| Music Selection | Generate melodic lounge music or select a local soundtrack | Implemented |
| Narration and Voice | Generate and synthesize optional voice-over | Planned |
| Timeline Builder | Produce a declarative edit plan with chronological and diversity constraints | Implemented |
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
- audio context, speech likelihood, ambience, and noise penalties;
- uniqueness and event diversity;
- speech importance;
- duplicate and technical penalties;
- manual include/exclude decisions.

Every selected or rejected scene should retain an explainable reason. The story
is built before final editing decisions. Story Builder consumes structured
metadata and transcripts, not raw media. The timeline optimizer should preserve
capture chronology unless explicitly configured otherwise, preserve the story
shape with approximate section budgets, and avoid adjacent repeats by location,
activity, shot type, source asset, and semantic tags when alternatives exist.

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
    |-- frame_sampling.cache.json
    |-- quality_analysis.json
    |-- quality_analysis.cache.json
    |-- vision_analysis.json
    |-- vision_analysis.cache.json
    |-- speech_analysis.json
    |-- speech_analysis.cache.json
    |-- audio_analysis.json
    |-- audio_analysis.cache.json
    |-- duplicates.json
    |-- scene_descriptions.json
    |-- events.json
    |-- storyboard.json
    |-- selection_decisions.json
    |-- quick_timeline.cache.json
    |-- music_plan.json
    |-- music_plan.cache.json
    |-- montage_quality_report.json
    |-- rendering.cache.json
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

`montage_quality_report.json` is a pre-render quality gate for the planned
movie. It records duration coverage, event and source diversity, average
semantic and visual quality, selected window types, music coverage, and
music diagnostics such as cue section count, beat grid size, WAV loudness,
peak level, and clipping ratio. It reports actionable issues such as a short
timeline, repeated source dominance, disabled music, missing music cue metadata,
unsynced music cuts, speech boundary cuts, excessive center cuts, quiet/clipped
source music, or selected dark/blurred scenes.
After rendering, the same report is enriched with FFprobe/FFmpeg checks for the
actual MP4: rendered duration, video/audio stream presence, plan-vs-render
duration delta, sampled audio RMS, and sampled video luma near the beginning,
middle, and end of the movie.

## Cache and Reproducibility

Media metadata is reused when path, size, and `modified_ns` match. Scene,
Vision, and speech cache keys include the relevant source metadata, time
boundaries, model, style, and prompt/schema version.

Frame Sampling, Quality Analysis, Vision Analysis, Speech Analysis, Audio
Analysis, Timeline Builder, Music Selection, and Rendering write typed sidecar
cache manifests with input fingerprints, configuration fingerprints, artifact
schema versions, and output paths. A rerun skips these stages only when the
manifest matches current inputs and all required artifacts still exist and
validate. Frame fingerprints include source media, scene boundaries, and
`analysis_quality_mode`, while ignoring later semantic metadata. Quality
fingerprints include source media and scene boundaries. Vision, speech, and
audio fingerprints include only the inputs those stages consume. Timeline
fingerprints include ranked scenes and media assets. Music fingerprints include
the timeline without embedded music, scene metadata, media assets, and local
soundtrack file metadata. Rendering fingerprints include the final timeline,
output path, FFmpeg/FFprobe settings, and worker configuration.

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
- joins prepared segments with direct cuts and no `xfade`/`acrossfade` visual
  transition;
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

Add the FFmpeg `bin` directory to `PATH` or configure full paths in
`configs/settings.toml`.

### No media files are found

Verify the selected directory, nested folders, and supported extensions. Keep
the workspace and source-media directory separate; neither should contain the
other.

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
- **Local inference:** media analysis, story decisions, and generation use
  models running on the user's computer.

The target scale is hundreds of videos and more than 100 GB of source media.
Processing must use bounded concurrency, batches, proxy media where necessary,
model reuse, disk-space checks, and CPU fallback.

Private data includes raw media, frames, faces, voices, transcripts, GPS
coordinates, and project databases. The application does not require telemetry
or upload these artifacts to a remote inference service.

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
- replace remaining placeholder stages with explicit Embeddings, Narration,
  and Voice Synthesis implementations;
- extend per-stage input/config/model fingerprints to the remaining lightweight
  and optional AI stages;
- enforce disk-cache limits and cleanup;
- check free disk space before rendering;
- add managed SQLite migrations.

### P1: Editing quality

- semantic duplicates with embeddings and FAISS;
- GPS and embeddings in event detection;
- richer shot-scale and camera-motion extraction from Vision AI;
- timeline version comparison in the web UI.

### P1: Story and manual editing

- direct local story-model adapter;
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
- BPM analysis for library/manual tracks and automatic music volume envelopes;
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
- no remote inference provider or cloud credential is configured;
- workspace data, models, databases, frames, and rendered movies must not be
  committed.

## License

See [LICENSE](LICENSE).
