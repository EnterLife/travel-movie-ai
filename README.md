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
- deterministic row-major RGB PNG contact sheets with 3, 5, or 9 temporal
  samples per scene and content-hash metadata;
- OpenCV analysis for sharpness, exposure, contrast, motion, shake, and noise;
- direct local Qwen2.5-VL and Florence-2 analysis with automatic model download;
- optional speech recognition with Faster Whisper;
- perceptual duplicate detection;
- deterministic and sentence-transformer embeddings, optional FAISS search,
  GPS/semantic event grouping, multimodal captions, storyboard generation, and
  scene ranking;
- explainable scene selection with `Auto`, `Include`, and `Exclude` overrides;
- editable events/scenes, manual ordering, named movie variants, and immutable
  timeline-version comparison;
- energy-aware semantic clip pacing with speech and people protection;
- deterministic or optional local-transformer story building;
- generated, library, manual, or disabled music modes with story-aware ACE-Step
  generation, Draft/Balanced/Studio quality, Best-of-N audition, BPM analysis,
  envelopes, narration ducking, and Piper voice synthesis;
- smart crop, vertical layouts, Ken Burns, color/HDR processing, overlays,
  quick preview, and final H.264/AAC rendering;
- NVIDIA NVENC acceleration with CPU fallback;
- automatic CPU, RAM, GPU, model, FFmpeg, worker-profile, runtime, and disk
  diagnostics;
- bounded 4K/8K analysis proxies, process-local Vision model reuse, validated
  temporal highlight windows, isolated retries, and explicit degraded results;
- persistent pause/cancel/recovery-aware movie jobs with history;
- cross-process workspace leases, restart-safe per-asset/per-scene checkpoints,
  weighted progress, and privacy-safe run manifests;
- managed SQLite migrations, bounded cache cleanup, backup/export/restore, and
  self-contained HTML reports;
- optional PySide6 desktop shell, Windows installer recipe, CI, and an explicit
  local-provider plugin contract;
- incremental reruns that reuse only validated, configuration-compatible
  artifacts.

Current limitations:

- heavyweight Vision, speech, embedding, story, and desktop runtimes remain
  optional installs;
- Piper requires a separately installed local executable and voice model;
- the Windows installer must be built on Windows with PyInstaller and Inno
  Setup 6; it provides the base quick-edit UI, while FFmpeg, optional AI
  runtimes, Piper, and model weights remain separate local installations;
- the loopback web interface has no authentication and must not be exposed on a
  public network;
- transition renders use lossless H.264 mezzanine segments before the final
  delivery encode and therefore require more temporary disk/time than hard cuts;
  delivery loudness uses one-pass normalization, and distributable overlay fonts
  are not bundled yet.

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
the non-interactive base web/video setup first. This launcher does not install
PyTorch, local AI models, or ACE-Step; install the optional groups explicitly
when those features are needed.

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
python -m pip install -e ".[story]"
python -m pip install -e ".[desktop]"
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

### Embeddings, FAISS, and local story model

The base install uses deterministic feature-hash embeddings and remains
model-free. Install the optional group to use multilingual
sentence-transformers and FAISS archive search:

```powershell
python -m pip install -e ".[embeddings]"
travelmovieai search --input "D:\Media\Trip" "sunset near the sea"
```

Set `embedding_backend = "sentence-transformers"` after installation.
`embedding_index = "auto"` builds a FAISS index when FAISS is present and
continues without an index otherwise; `faiss` makes the dependency mandatory.
No query or source frame leaves the computer.

The deterministic story builder is the default. An optional local text model
can produce a validated structured storyboard:

```powershell
python -m pip install -e ".[story]"
```

Then set `story_provider = "local"`. Invalid or unavailable model output falls
back to the deterministic builder and is deliberately not cached, so a later
run retries the configured model.

For longer trips, the deterministic builder assigns several chronological edge
events to the opening and finale and distributes the strongest middle events
across a highlight section. This prevents one large catch-all journey section
from dominating the automatic edit while keeping every event in exactly one
story section.

### Piper narration

Voice Synthesis is disabled by default. To enable it, install Piper locally,
download a compatible `.onnx` voice, set `voice_provider = "piper"`, and set
`piper_model` to that file in `configs/settings.toml`. The provider receives
narration through standard input, validates the generated WAV, and never sends
text to a remote service. Enabling `narration_enabled` adds the WAV to the
timeline; rendering mixes it with configurable source/music ducking and a final
limiter.

## Web Interface Workflow

1. Click the directory picker next to the source field, or enter a path manually.
2. Leave the workspace field blank to use the source-bound default under
   `<repository>\workspace\<source-folder>-<source-fingerprint>`, or choose an
   explicit directory. Keep the workspace separate from the source folder;
   neither directory may be nested inside the other.
3. Start media analysis.
4. Choose a Vision backend, model, story style, and render device.
5. Keep semantic and OpenCV analysis enabled for AI-directed editing.
6. Enable Faster Whisper only when speech matters.
7. Enable quick preview for the first iteration.
8. Configure duration and music.
9. Start AI montage.
10. Give the render a variant name when keeping several edits from the same
    analysis.
11. Monitor global progress, individual stage bars, ETA, and logs. Jobs can be
    paused, continued, or cancelled, and interrupted history is recovered after
    a server restart.
12. Review events and scenes: edit titles, summaries, captions, transcripts, or
    landmarks; reorder events/scenes; and set `Auto`, `Include`, or `Exclude`.
13. Rerun the montage. Unchanged expensive analysis is reused from cache.
14. Compare built/rendered timeline versions, then disable preview and render
    the final named variant.

Quick mode selects short clips chronologically without Vision AI. Semantic mode
adds scene detection, frame sampling, quality and Vision analysis, optional
speech recognition, embeddings, duplicate detection, event grouping, story
building, ranked selection, narration, timeline construction, and rendering.
The CLI and web semantic flows execute this same canonical 18-stage pipeline.

Semantic mode is intentionally selective. It does not try to use every video in
the folder and it does not fill the target duration with scenes that fail the
semantic or technical gates. After story pacing shortens energetic clips, the
selector backfills from the remaining eligible scenes until it reaches the
requested duration or exhausts the safe candidate pool.
`min_semantic_score` is a base quality target, but the actual threshold is
computed from the score distribution of the current project: it rises for strong
archives and relaxes for consistently modest material. The quality report records
that effective threshold and evaluates the selected lower-score tail against the
same value, rather than flagging the intentional adaptive relaxation as a defect.
The `max_scenes_per_source`
setting is a hard automatic-selection guard by default, so one strong roll cannot
dominate the movie. Set `strict_source_diversity=false` only when source variety
is less important. `max_scenes_per_event` is also a hard automatic-selection cap,
including during target-duration backfill. Explicit `Include` overrides may
exceed either cap because a manual editorial decision takes precedence. The
automatic timeline can therefore be shorter than the requested target when the
remaining candidates would violate diversity or technical-quality constraints.
Use scene overrides when a specific fragment must be included or excluded.

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
maximum scene-understanding quality. Samples are stored in chronological,
row-major contact sheets with exact source timestamps, normalized positions,
grid geometry, and a SHA-256 content identity used by downstream caches.

For long scenes, semantic montage does not blindly cut the middle of the scene.
It builds typed candidate windows inside the scene and prefers validated Vision
highlight windows, then the best visual panel from the sampled contact sheet,
then a neutral center cut. Every chosen window records its source and is clipped
to the scene bounds. This keeps the final movie focused on the strongest moment
inside each selected scene.

Visual quality analysis stores per-panel scores and ready-to-use
`candidate_windows` in scene metadata. Future audio, face, speech, and object
analysis can add their own candidate windows to the same contract. Vision AI
also returns a validated normalized focus point and `face`, `object`, or
`subject` source for smart crop, allowing the timeline builder to choose the
best moment inside a long source scene without changing the renderer. Vision
cache identity includes the contact-sheet content, provider/model, effective
device, prompt/parser versions, scoring settings, and analysis depth. Failed
batches are isolated and retried; a per-scene deterministic fallback is reported
as `degraded` instead of being cached as a successful model result.

Audio Analysis stores scene-level labels such as `speech`, `silence`, `wind`,
`music`, `crowd`, `water`, and `transport`. It adds audio candidate windows,
boosts scenes with speech or useful ambience, and penalizes strong wind or
transport noise during ranking.

Speech Analysis stores Whisper segment boundaries in scene metadata when the
provider returns them. Semantic timeline planning uses those boundaries as
speech-safe candidate windows and penalizes source windows whose start or end
would cut through a spoken phrase.

Narration text is bounded by `narration_characters_per_second` before Piper is
called. Voice Synthesis checkpoints one typed WAV per line without assigning
premature absolute timestamps. Timeline Builder then uses the actual selected
movie duration, places only lines that fit, creates declarative audio cues, and
atomically materializes the combined narration track. A short diversity-limited
timeline therefore degrades by omitting excess lines instead of failing after
expensive synthesis or creating out-of-range cues.

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
affect selection. The safe `cinematic` default uses hard cuts within an event
and a fade through black between events. Pixel dissolve is prohibited. Wipes,
slides, and other non-default transitions are used only when explicitly selected.
When a transition is selected, the timeline uses real video `xfade` and audio
`acrossfade` overlaps and accounts for those overlaps in its duration and
beat-sync calculations. One strong but repetitive location or activity should
not fill the whole movie when varied alternatives are available.

### Generated AI Music

`AI Auto` and `Generate locally` create a soundtrack entirely on the local
machine. The default AI engine is
[ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5), a specialized
open-source music generation model. It generates a complete instrumental
composition from the story style, Vision captions, BPM, key, duration budget,
and a macro arrangement derived from the final timeline.

The unified Windows setup:

1. installs ACE-Step into the isolated `.cache/ace-step` environment;
2. keeps its dependencies separate from the main `.venv`.

The first generation then:

1. downloads model weights into `models/ace-step`;
2. validates required model configuration files and repairs incomplete metadata
   when downloads are allowed;
3. detects the GPU tier, chooses an appropriate Turbo/SFT/XL model, and enables
   CPU offload on low-VRAM systems;
4. generates the requested duration natively and normalizes every candidate to
   stereo 48 kHz, 24-bit PCM without looping model output.

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

Quality presets control the time/VRAM/quality tradeoff:

- `Draft`: one 2B Turbo candidate with 8 diffusion steps;
- `Balanced` (default): four candidates, automatic technical/structure/style
  scoring, and selection of the strongest candidate; on a 20+ GB GPU it can use
  XL Turbo;
- `Studio`: six candidates, slower SFT/XL SFT sampling where VRAM permits,
  classifier-free guidance, adaptive guidance, and an ACE-Step language model
  on GPUs with at least 8 GB VRAM. A 6 GB GPU safely retains the Turbo model and
  spends the extra budget on candidates and sampling steps.

Candidate count can be overridden from `1` through `8`. The Web UI exposes every
generated candidate with its technical, structure, and style score and an audio
player. The winner is copied atomically to the final soundtrack; all candidate
metadata and hashes remain in `artifacts/music_plan.json` for reproducibility.

ACE-Step is prepared together with the application:

```powershell
.\scripts\setup_windows.bat
```

Automatic style selection uses scene captions and the selected story profile to
choose modern cinematic, organic electronic, indie travel, melodic ambient,
ambient house, or documentary language. Prompts request premium 2020s
production, a coherent motif, natural phrasing, stereo detail, mastering
headroom, and a resolved ending while leaving space for location audio and
dialogue. They reject vocals, speech, clipping, abrupt genre changes, stock-music
cliches, and mechanical looping.

`Synchronize with editing` is enabled by default. The application first builds
the final clip timeline and then requests a native full-duration composition.
If a model returns a slightly short WAV, normalization pads the tail with silence;
it never repeats a generated passage. A cue sheet is a first-class contract with
arrangement sections, BPM, intensity, and restrained accent points. It places
musical structure at:

- cut points between clips;
- changes between detected trip events;
- the center of high-scoring Vision AI scenes;
- the opening and final moments.

ACE-Step's native composition limit is 600 seconds. For a longer movie, `AI
Auto` can use the procedural fallback; explicit ACE-Step mode reports an
actionable error so it never silently creates a mostly empty soundtrack. A
manual or library track is another option for long-form edits.

The cue sections, beat grid, timestamps, strengths, BPM, key, generation prompt,
quality preset, candidates, scores, seeds, arrangement version, generator, model
identifier, source-content SHA-256, reference/LoRA usage, and fallback status are
stored in `artifacts/music_plan.json`. Cached generated music is reused only while
the soundtrack and every generation input still match. Changing a reference
track or LoRA file invalidates the relevant music-stage cache. The procedural
fallback remains available, follows the cue sections, and is reported as degraded
instead of being promoted to a successful neural-generation cache entry.

Optional reference audio can steer the style of ACE-Step while keeping all
inference local. Use only audio you own or are licensed to use. A local ACE-Step
LoRA file or adapter directory can provide a reusable house style. Reference and
LoRA paths are validated before generation, are rejected with the procedural
engine, and are never uploaded. Their strengths are independently configurable.
Rebuilding the same timeline uses a
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

The same command exposes advanced local render controls. For example:

```powershell
travelmovieai create `
  --input "D:\Media\Trip" `
  --output "D:\Movies\Trip-smart.mp4" `
  --semantic --variant "smart vertical cut" `
  --analysis-quality deep --width 1920 --height 1080 --fps 30 `
  --framing smart --vertical-layout blur --photo-motion ken_burns `
  --color-normalization --hdr-to-sdr --event-titles --subtitles `
  --music-quality studio --music-candidates 6 `
  --music-style modern_cinematic --bpm-analysis --music-envelope `
  --validate-full-render-decode
```

Optional local style conditioning:

```powershell
travelmovieai create `
  --input "D:\Media\Trip" --output "D:\Movies\Trip-styled.mp4" `
  --music-quality studio `
  --music-reference "D:\Music\Owned reference.wav" `
  --music-reference-strength 0.25 `
  --music-lora "D:\Models\ace-step-travel-lora" `
  --music-lora-strength 0.7
```

Estimate a cold-run runtime range and peak workspace size from probed metadata:

```powershell
travelmovieai estimate --input "D:\Media\Trip" --semantic
travelmovieai estimate --input "D:\Media\Trip" --semantic --json
```

Search the analyzed archive and create a self-contained HTML report:

```powershell
travelmovieai search --input "D:\Media\Trip" "mountain sunrise" --limit 8
travelmovieai report --input "D:\Media\Trip"
```

Run local runtime diagnostics:

```powershell
travelmovieai doctor
```

Export a checksummed project archive and restore it into an empty workspace:

```powershell
travelmovieai export `
  --input "D:\Media\Trip" `
  --output "D:\Backups\Trip.travelmovie.zip"
travelmovieai restore `
  --archive "D:\Backups\Trip.travelmovie.zip" `
  --workspace "D:\TravelMovieAI\Trip-restored"
```

Rendered media is excluded from export unless `--include-rendered-media` is
specified. Restore validates the typed manifest, entry paths, sizes, and SHA-256
checksums before atomically installing the workspace. The source-bound
`.travelmovieai-project.json` identity is included and revalidated; duplicate,
encrypted, link, special, malformed, or path-traversing ZIP entries are rejected.

Story styles: `cinematic`, `documentary`, `family`, `vlog`, `adventure`, and
`romantic`.

The `.mp4` output movie must be outside the source media folder and the
workspace `cache` and `frames` folders. Rendering rejects an output path that
would overwrite a source clip, soundtrack, project database, or renderer working
file. Custom render width and height values must be even for H.264 compatibility.

The `create --semantic` command and web AI Edit use the same canonical stage
pipeline, so they share Vision lifecycle, embeddings, story, music, narration,
timeline, rendering, caching, and quality-gate behavior. `run_until` and the
`storyboard` and `render` commands expose the same incremental pipeline for
diagnostic workflows.

## Configuration

Runtime settings live in the checked-in `configs/settings.toml`. It contains no
secrets or remote service credentials. CLI and web entry points validate this
file at startup; unknown keys and invalid values fail with an actionable error.

| Key | Purpose | Default |
| --- | --- | --- |
| `workspace` | Root for source-bound, uniquely fingerprinted project workspaces | `workspace` |
| `database_filename` | SQLite database filename | `project.db` |
| `ffmpeg_binary` | FFmpeg command or full path | `ffmpeg` |
| `ffprobe_binary` | FFprobe command or full path | `ffprobe` |
| `frame_extraction_timeout_seconds` | Per-scene FFmpeg frame extraction timeout | `120` |
| `analysis_proxy_mode` | 4K/8K proxy policy: `auto`, `disabled`, or `always` | `auto` |
| `analysis_proxy_max_dimension` | Maximum long edge for analysis proxies | `1920` |
| `analysis_proxy_video_bitrate_mbps` | Proxy target bitrate | `6.0` |
| `analysis_proxy_timeout_seconds` | Proxy FFmpeg timeout | `3600` |
| `render_timeout_seconds` | Per-FFmpeg render, validation, or music-normalization timeout | `7200` |
| `render_disk_reserve_mb` | Free disk reserve retained after render | `1024` |
| `render_disk_safety_factor` | Temporary/output render-space multiplier | `3.0` |
| `vision_provider` | `local`, `qwen`, or `florence` | `local` |
| `vision_model` | Vision model identifier or `auto` | `auto` |
| `vision_model_pool_size` | Maximum idle reusable Vision runtimes | `1` |
| `model_cache` | Downloaded local model cache | `models` |
| `allow_model_download` | Download missing models on first use | `true` |
| `embedding_backend` | `feature-hash` or `sentence-transformers` | `feature-hash` |
| `embedding_model` | Local sentence-transformer model identifier | multilingual MiniLM |
| `embedding_index` | `auto`, `faiss`, or `disabled` | `auto` |
| `embedding_batch_size` | Sentence embedding batch size | `32` |
| `story_provider` | `deterministic` or local `local` adapter | `deterministic` |
| `story_model` | Local structured-story model identifier | Qwen2.5 1.5B Instruct |
| `story_max_new_tokens` | Maximum local story response tokens | `768` |
| `whisper_model` | `medium` or `large-v3` | `medium` |
| `voice_provider` | `disabled` or local `piper` | `disabled` |
| `piper_binary` | Piper command or full executable path | `piper` |
| `piper_model` | Local Piper `.onnx` voice path | unset |
| `voice_synthesis_timeout_seconds` | Piper process timeout | `600` |
| `device` | `auto`, `cuda`, `directml` compatibility fallback, or `cpu` | `auto` |
| `resource_mode` | Load profile: `auto`, `safe`, `balanced`, or `performance` | `auto` |
| `gpu_memory_reserve_mb` | VRAM kept free for Windows, the driver, and stage handoff | `1536` |
| `max_gpu_processes` | Maximum simultaneous NVDEC/NVENC FFmpeg processes | `2` |
| `music_library` | Optional local soundtrack directory; empty by default | `assets/music` |
| `music_model` | Local music model identifier or `auto` | `auto` |
| `generated_music_filename` | Generated soundtrack filename | `generated_soundtrack.wav` |
| `project_cache_limit_mb` | Combined project cache/frames limit; `0` disables cleanup | `20480` |
| `project_cache_target_ratio` | Cleanup target after the limit is exceeded | `0.85` |
| `workers` | Parallel worker limit; `0` means automatic hardware-based selection | `0` |
| `batch_size` | Model batch limit; `0` means automatic hardware-based selection | `0` |
| `web_host` | Loopback-only bind (`127.0.0.1`, `localhost`, or `::1`) | `127.0.0.1` |
| `web_port` | Web server port | `8000` |
| `web_history_limit` | Saved scan- and movie-job history limit | `100` |

`directml` is currently accepted as a compatibility value but falls back to
CPU/OpenCV in the bundled Qwen, Florence, embedding, and quality adapters; it
does not currently promise DirectML acceleration.

Music quality, candidate count, modern style, reference audio, and LoRA are
per-movie settings exposed by Web AI Edit and `travelmovieai create`; they are
intentionally not machine-global keys in `configs/settings.toml`.

## Automatic Hardware Utilization

At each montage start, TravelMovieAI detects:

- logical CPU count;
- installed RAM;
- NVIDIA GPU plus total and currently free VRAM;
- CUDA availability in PyTorch and OpenCV;
- FFmpeg NVENC support.

The resulting profile separately selects concurrency for frame extraction,
OpenCV analysis, Vision AI batching, and segment rendering. The defaults
`device = "auto"`, `resource_mode = "auto"`, `workers = 0`, and
`batch_size = 0` select CUDA and NVENC when they are usable. With at least 16 GB
RAM and sufficient free VRAM, `auto` resolves to `performance` and uses the
maximum bounded CPU concurrency. Low RAM or heavily occupied VRAM resolves to a
more conservative profile. The web form selects NVIDIA NVENC by default when it
is available, otherwise CPU/libx264.
Vision batching is based on free VRAM at job start rather than total installed
VRAM. Explicit `workers` and `batch_size` overrides remain available, but the
NVENC process count still respects `max_gpu_processes`.
One profile is captured in the project execution context and shared by all
pipeline stages, avoiding repeated CUDA, NVENC, RAM, and CPU probes during the
same run.

When a source exceeds the configured analysis dimension, Scene Detection creates
one atomically written project-local H.264 proxy before decoding the 4K/8K
original; Frame Sampling reuses the same proxy.
Scene Detection processes assets with a bounded deterministic worker pool
(`safe` 1, `balanced` up to 4, `performance` up to 8) and checkpoints each
completed asset before submitting more work. A cancelled run stops feeding the
queue, and a retry resumes valid per-asset checkpoints.
The proxy cache key includes source metadata and proxy settings. Vision and
quality analysis consume the resulting contact sheet, while the declarative
timeline and final renderer always retain the original source path. A bounded
process-local LRU can keep a released Qwen or Florence runtime ready for the
next project without creating one model per scene.

`scripts/benchmark_metadata_scale.py` exercises typed metadata construction,
scene generation, SQLite migration/write/read round trips, serialization,
fingerprinting, throughput, and traced peak memory for 512 assets and
a virtual source set above 100 GiB without allocating those media bytes. It is
not a codec/GPU throughput substitute for a real archive. The shared
resource estimator reports a cold-run runtime range plus proxy, frame, database,
artifact, output, and peak-workspace byte estimates; render disk preflight uses
the configured reserve and safety factor before FFmpeg starts.

On a 16-thread CPU with 32 GB RAM and a 6 GB RTX GPU, the automatic profile uses
many CPU workers for CPU-bound stages, up to two concurrent NVDEC/NVENC jobs,
and a Vision batch sized from the remaining safe VRAM. Higher-VRAM GPUs receive
larger Vision batches only when that memory is actually free.

GPU usage by stage:

- frame sampling: `auto` uses bounded parallel FFmpeg CUDA decode when NVENC is
  available; the default limit is two NVDEC jobs, while `safe` mode or
  `max_gpu_processes = 1` provides serialized GPU decode;
- quality metrics: OpenCV/Pillow by default, with CUDA analysis serialized if a
  CUDA quality analyzer is active;
- Vision AI: `auto` uses CUDA through the local provider when PyTorch and the
  selected model can use it, with CPU/offload fallback where needed;
- Speech AI: Faster Whisper releases its CTranslate2 model immediately after
  transcription so its VRAM is available to later stages;
- rendering: `auto` uses NVENC when available and falls back to `libx264` if
  the automatic NVENC render fails.

If Windows records `VIDEO_SCHEDULER_INTERNAL_ERROR` (`0x119`) or repeated
`nvlddmkm` events, first set `resource_mode = "safe"`,
`max_gpu_processes = 1`, and increase `gpu_memory_reserve_mb`. If resets still
occur with `device = "cpu"`, the likely cause is outside the application (driver,
overclock/undervolt, temperature, PSU, or hardware stability); software cannot
guarantee protection from a machine-level reset.

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
| Scene Detection | Create/reuse 4K/8K proxies, then detect bounded scenes in a restartable worker pool | Implemented |
| Frame Sampling | Generate cached RGB contact sheets | Implemented |
| Visual Quality | Measure technical quality with OpenCV/Pillow | Implemented |
| Vision AI | Generate structured semantic understanding, shot scale, and camera motion | Implemented |
| Speech | Transcribe scene audio with Faster Whisper | Implemented, optional |
| Audio Analysis | Classify speech, silence, wind, music, crowds, water, transport, and ambience | Implemented |
| Embeddings | Feature-hash or sentence-transformer vectors plus optional FAISS index | Implemented |
| Duplicate Detection | Group visually similar scenes | Implemented |
| Scene Captioning | Merge Vision, quality, speech, and event context | Implemented |
| Event Detection | Group scenes using time, GPS distance, and embedding similarity | Implemented |
| Story Builder | Build validated deterministic or local-model story sections and budgets | Implemented |
| Scene Ranking | Explain selection and rejection decisions | Implemented |
| Music Selection | Generate melodic lounge music or select a local soundtrack | Implemented |
| Narration | Generate deterministic local story text | Implemented |
| Voice Synthesis | Synthesize optional local Piper voice-over | Implemented, opt-in |
| Timeline Builder | Produce a declarative edit plan with chronological and diversity constraints | Implemented |
| Rendering | Render, atomically replace, and validate the MP4 | Implemented |

Stage contract changes must update domain models, serialization, downstream
consumers, tests, and this README together.

Provider leases are released at the end of their stages. Whisper unloads its
runtime immediately; Vision may retain an idle runtime in the bounded
process-local LRU when `vision_model_pool_size > 0`, including its allocated
VRAM. Set the pool size to `0` to unload Vision after the stage. AI Auto music
uses the same ACE-Step adapter in the web use case and canonical Music Selection
stage. Editing defaults to safe,
event-aware hard cuts inside events and fades through black between events; the
speech remains an explicit opt-in. Pixel dissolve is not supported;
additional transition styles must be selected explicitly.

The final render is validated with FFprobe and a typed montage quality report.
Critical quality issues fail the job instead of returning `Film ready`; the web
result shows the quality score and issue count for non-critical diagnostics.

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
  "shot_scale": "wide",
  "camera_motion": "tracking",
  "focus_x": 0.46,
  "focus_y": 0.38,
  "focus_source": "face",
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

### Local provider/plugin contract

`travelmovieai.infrastructure.providers.ProviderRegistry` is the explicit,
lazy extension boundary for local `vision`, `speech`, `embeddings`, `story`,
`music`, and `voice` adapters. A plugin declares the entry-point group
`travelmovieai.providers` and exposes `register(registry)`:

```toml
[project.entry-points."travelmovieai.providers"]
my-local-provider = "my_package.travelmovie_plugin"
```

```python
from travelmovieai.infrastructure.providers import ProviderDescriptor


def register(registry):
    registry.register(
        ProviderDescriptor(
            name="my-local-provider",
            kind="vision",
            version="1",
            local_only=True,
            optional_dependency="my-package[vision]",
            model_heavy=True,
        ),
        lambda settings: MyLazyVisionProvider(settings),
    )
```

Importing TravelMovieAI never discovers or initializes plugins. A trusted host
must construct a registry, call `load_entry_points()` explicitly, inspect the
typed descriptors, and pass the selected factory into its service integration.
Remote descriptors, duplicate registrations, malformed names, and plugins
without `register(registry)` are rejected. Factories must stay lazy, local-only,
and obey the same typed artifact, privacy, and optional-dependency contracts as
built-in providers.

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
GET   /api/movies
GET   /api/movies/{id}
POST  /api/movies/{id}/pause
POST  /api/movies/{id}/resume
POST  /api/movies/{id}/cancel
GET   /api/movies/{id}/download
GET   /api/scenes?offset=0&limit=60&event_id={optional-event-id}
PATCH /api/scenes/{id}
GET   /api/scenes/{id}/thumbnail
GET   /api/events
PATCH /api/events/{id}
PUT   /api/events/order
PUT   /api/events/{id}/scenes/order
GET   /api/timeline-versions
GET   /api/timeline-versions/{id}
GET   /api/timeline-versions/compare
```

Scan and movie jobs use bounded worker pools. A shared atomic workspace
reservation serializes scan, movie, and manual-edit mutations, so simultaneous
requests targeting the same workspace cannot both pass the active-job check.
Configuration rejects non-loopback bind addresses because the local API does
not currently implement remote authentication.

Movie-job responses include global progress, the active phase, elapsed time,
ETA, hardware profile, individual stage status, and up to 250 recent log
messages. Job state is atomically persisted below the workspace-root `.web`
directory. A process restart requeues interrupted scans and movie jobs with the
same job ID; paused jobs remain paused. Resume reuses every valid stage artifact
and each atomically completed render segment. Manual edits are locked while the
same workspace has an active scan or movie job. Every persisted message is
redacted against the job's source, workspace, output, model, music, font, and
voice paths, and background log records carry the job UUID as their correlation
ID.

Canonical runs emit typed `ProgressEvent` records with the exact stage ID,
stage-local current/total/unit, and weighted monotonic overall progress. The
legacy `(current, total, message)` callback remains supported. CLI progress is
written to stderr, leaving stdout as the concise command result; web jobs use
the typed stage ID instead of inferring phases from provider-specific text.

## Workspace

Representative generated data is stored under a project workspace; additional
stage-specific `*.cache.json` manifests are expected:

```text
workspace/<source-name>-<source-fingerprint>/
|-- .travelmovieai-project.json
|-- .travelmovieai.lock
|-- .travelmovieai.lock.json
|-- project.db
|-- frames/
|-- cache/
|   |-- proxies/
|   `-- quick_montage_segments/
`-- artifacts/
    |-- pipeline_run.json
    |-- analysis.json
    |-- scenes.json
    |-- scene_detection_shards/
    |-- frame_sampling.json
    |-- frame_sampling.cache.json
    |-- quality_analysis.json
    |-- quality_analysis.cache.json
    |-- vision_analysis.json
    |-- vision_analysis.cache.json
    |-- speech_analysis.json
    |-- speech_analysis.cache.json
    |-- speech_analysis_shards/
    |-- audio_analysis.json
    |-- audio_analysis.cache.json
    |-- embeddings.json
    |-- embeddings.cache.json
    |-- embeddings.faiss
    |-- embeddings.index.json
    |-- duplicates.json
    |-- scene_descriptions.json
    |-- events.json
    |-- storyboard.json
    |-- storyboard.cache.json
    |-- narration.json
    |-- voice_synthesis.json
    |-- narration_lines/
    |-- narration.wav
    |-- selection_decisions.json
    |-- quick_timeline.cache.json
    |-- music_plan.json
    |-- music_plan.cache.json
    |-- montage_quality_report.json
    |-- rendering.cache.json
    |-- report.html
    |-- variants/
    |-- quick_timeline.json
    |-- preview.mp4
    `-- final.mp4
```

When the workspace field is blank, the backend derives a Windows-safe name from
the canonical source path plus a SHA-256 prefix. The identity manifest binds the
workspace to that source, so equal basenames in different directories cannot
mix databases or timelines. A pre-identity basename-only workspace is reused
only when its existing analysis artifact proves the same source. Choosing an
explicit workspace remains supported, but reusing a non-empty workspace for a
different source is rejected before pipeline files are created.

`project.db` stores media assets, scenes, events, scores, transcripts, manual
overrides, optimistic edit revisions, and immutable built/rendered timeline
versions. SQLite uses foreign keys, WAL mode, `PRAGMA user_version`, and ordered
idempotent migrations backed by frozen version-specific DDL. A failed migration
does not advance `user_version`, so it can be retried safely. Databases created
by a newer unsupported application version are rejected instead of being
silently modified.

Critical JSON and media outputs are written atomically. Source media remains
read-only. An operating-system workspace lease serializes mutating CLI, service,
pipeline, and manual-edit operations across processes. Lease metadata identifies
the process, operation, and start time; the OS releases ownership after a crash,
so a stale metadata file cannot permanently lock a project. Restore and first-use
operations additionally take a target-keyed sidecar lease outside the workspace;
concurrent restores are serialized without making an otherwise empty restore
target non-empty or copying lock metadata into an archive.

`pipeline_run.json` is a privacy-safe run manifest containing a fresh run ID,
target, start/end time, weighted stage durations, status/cache outcomes, and
allow-listed retry counts, primary provider/model, and explicit fallback-provider
metadata. Cache reuse is recorded independently from `completed`/`degraded`, so
a cached fallback cannot look like a fresh successful inference. It never stores
source, workspace, output, or artifact paths, and persisted failures redact both
relative and resolved local paths plus secrets.

`montage_quality_report.json` is a pre-render quality gate for the planned
movie. It records duration coverage, event and source diversity, average
semantic and visual quality, the effective adaptive semantic threshold, selected
window types, music coverage, and
music diagnostics such as cue section count, beat grid size, WAV loudness,
peak level, and clipping ratio. It reports actionable issues such as a short
timeline, repeated source dominance, disabled music, missing music cue metadata,
unsynced music cuts, speech boundary cuts, excessive center cuts, quiet/clipped
source music, or selected dark/blurred scenes.
After rendering, the same report is enriched with FFprobe/FFmpeg checks for the
actual MP4: rendered duration, video/audio stream presence, plan-vs-render
duration delta, sampled audio RMS and video-luma distributions, plus a
full-duration scan for black video, freezes, silence, integrated loudness,
loudness range, and true peak. The typed `gate_status` is `passed`, `degraded`,
or `failed`; an unavailable full scan is a warning rather than a false pass, and
`--validate-full-render-decode` makes that scan mandatory. Critical delivery
faults fail the job. Tail checks sample several windows before
the intentional final fade so a short musical pause does not become a false
warning.

## Cache and Reproducibility

Media metadata is reused when path, size, and `modified_ns` match. Scene,
Vision, and speech cache keys include the relevant source metadata, time
boundaries, measured quality, model, style, and prompt/schema version. Vision AI
uses the validated contact-sheet SHA-256 rather than its filesystem timestamp,
so atomically regenerating identical pixels does not repeat model inference.
Legacy stat-based Vision entries migrate only when the provider configuration,
scene identity, measured quality, contact-sheet metadata, and actual content hash
all match. Integral numeric scene settings are canonicalized before hashing, so
equivalent TOML/JSON values such as `27` and `27.0` share one cache identity.
Vision AI also atomically checkpoints every completed inference batch. If a long run is
interrupted, the next run validates and reuses the completed scene records from
the partial artifact instead of restarting the whole model pass.

Scene Detection keeps atomic per-asset shards, while Speech Analysis keeps
config- and source-validated per-scene transcript shards. These shards preserve
completed work across a later asset/model failure and merge only speech-owned
fields back into the current scene, so newer quality or Vision metadata is not
rolled back. Embedding vectors live once in `embeddings.json` and the optional
FAISS index; Event Detection loads them transiently and does not duplicate the
vectors in SQLite scene metadata.

Frame Sampling, Quality Analysis, Vision Analysis, Speech Analysis, Audio
Analysis, Embeddings, Story Builder, Timeline Builder, Music Selection, Voice
Synthesis, and Rendering write typed sidecar cache manifests with input
fingerprints, configuration fingerprints, artifact schema versions, and output
paths. A stage reports `completed`, `cached`, `degraded`, `disabled`, or
`no_input`; it
reuses work only when the manifest matches current inputs and every required
artifact still exists and validates. Frame fingerprints include source media, scene boundaries, and
`analysis_quality_mode`, while ignoring later semantic metadata. Quality
fingerprints include source media and scene boundaries. Vision, speech, and
audio fingerprints include only the inputs those stages consume. Timeline
fingerprints include ranked scenes and media assets. Music fingerprints include
the timeline without embedded music, scene metadata, media assets, and local
soundtrack file metadata. Rendering fingerprints include the final timeline,
output path, FFmpeg/FFprobe settings, and worker configuration.
Generated soundtrack content is also matched by SHA-256; replacing a WAV while
preserving its name or timestamps invalidates both the stage and internal music
cache. Voice cache hits decode every WAV and verify duration, sample rate, and
channel count rather than accepting a merely non-empty file.

Frame Sampling does not trust metadata alone: every reused contact sheet is
fully decoded, checked against its expected grid geometry and sample positions,
and matched to its stored SHA-256. A truncated, substituted, or untracked PNG is
atomically regenerated before quality or Vision analysis can reuse downstream
artifacts.

Rendering additionally publishes each prepared clip segment atomically under a
fingerprinted cache key. A cancelled, paused, or interrupted job therefore
reuses valid completed segments on the same timeline instead of transcoding
them again; changed source metadata, clip settings, or encoder settings produce
a different key. Custom overlay-font content and timestamps also participate in
the cache identity. FFprobe verifies cached segment video, audio, and duration
before reuse. Transition checkpoints must additionally be lossless H.264
`High 4:4:4 Predictive` with ALAC audio; a geometrically valid lossy checkpoint
is rebuilt. On cache hits, Vision restores only Vision-owned fields and Story
Builder reapplies only story-role fields, preserving later speech, audio, and
manual-edit state.

Before a pipeline run, the project `cache` and `frames` trees are measured. If
their combined size exceeds `project_cache_limit_mb`, oldest safe regular files
are removed down to `project_cache_target_ratio`; symlinks and paths outside the
two owned roots are never followed. Missing/disabled upstream inputs also remove
stage-owned stale artifacts so a previous successful run cannot masquerade as
current output.

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
- honors camera rotation metadata and supports `fit`, blurred-background, or
  crop layouts for vertical video;
- optionally applies face/object/subject-aware smart crop with a safe fit
  fallback and configurable Ken Burns motion for photos;
- optionally normalizes exposure/color and performs HDR-to-SDR tone mapping;
- creates silent audio for sources without audio;
- prepares independent segments with bounded parallelism and lossless ALAC
  intermediate audio;
- joins prepared segments with direct cuts by default, or real `xfade` and
  `acrossfade` overlaps when a transition is requested;
- stream-copies prepared H.264 video for hard cuts and uses lossless H.264
  mezzanine video before the one delivery encode when transitions are active;
- adds generated, library, or manual music;
- analyzes BPM for local tracks, applies timeline-aware volume envelopes, and
  ducks music/source ambience around optional Piper narration;
- applies source/music/narration/final fades, delivery loudness normalization,
  and a true-peak limiter before AAC encoding;
- optionally draws event titles, scene captions, and credits inside validated
  safe areas when the master text-overlay switch is enabled;
- uses `h264_nvenc` automatically when available and falls back to `libx264`;
- strips source container metadata such as camera comments and GPS tags from
  rendered movies;
- renders to a hidden sibling candidate and validates video, audio, shape,
  duration, and optional full decode before running the full-duration delivery
  quality scan;
- atomically publishes the candidate only after the quality gate accepts it, so
  a failed render cannot replace an earlier valid movie.

Prepared clip segments are independent, bounded-parallel tasks with atomic,
fingerprinted checkpoints. Recovery can resume at the first missing segment;
the final concat/transition pass is rebuilt from the validated segment set.
The render preflight uses the actual timeline to reserve the complete peak
working set: QP0 H.264 and ALAC mezzanines, the delivery movie, its atomic
temporary output, and the configured safety reserve. Workspace and output
volumes are evaluated independently when they are on different drives; hard-cut
plans retain the smaller delivery-oriented estimate.

All new visual treatments are opt-in in `QuickMontageSettings`, preserving
legacy output by default. `framing_mode = "smart"`,
`photo_motion = "ken_burns"`, `vertical_video_layout`,
`color_normalization`, `hdr_to_sdr`, `text_overlays_enabled`,
`event_titles_enabled`, `scene_subtitles_enabled`, `credits_text`,
`music_bpm_analysis`, and
`music_volume_envelope` control them. The timeline remains declarative: the
renderer consumes focus coordinates, rotation/color metadata, overlays, music,
and narration paths rather than choosing story content.
`text_overlays_enabled` is the master switch and defaults to `false`; event
titles, scene captions, and credits are never burned into the movie unless it is
explicitly enabled (CLI: `--text-overlays`). The individual title, subtitle, and
credits settings remain available as sub-controls.
Smart crop, metric-based color normalization, event titles, and scene subtitles
require semantic analysis because quick chronological mode does not produce the
focus, quality, event, or caption metadata they consume. CLI/API requests reject
those combinations early, and the web controls disable them in quick mode.

During a movie job, `Pause` stops before the next scene or subtask and
`Continue` resumes it. `Full stop` cancels the remaining work while preserving
valid cache artifacts. Active FFmpeg, ACE-Step, and Piper subprocess trees poll
the cancellation heartbeat and are terminated promptly; on Windows each process
is created suspended, assigned to a kill-on-close Job Object, and only then
resumed, so descendants cannot escape before cancellation ownership is in place.
Completed atomic stage, scene, line, and segment checkpoints remain reusable.

Preview mode is limited to 854x480 and 24 FPS. The standard output defaults to
1280x720 at 30 FPS.

## Desktop shell and Windows installer

The CLI and loopback web application remain the stable core. An optional thin
PySide6 shell starts the same FastAPI application on `127.0.0.1`, opens it in
the default browser, and stops Uvicorn when the window closes:

```powershell
python -m pip install -e ".[desktop]"
travelmovieai-desktop
```

Build a per-user Windows installer from a clean project-local build environment:

```powershell
.\scripts\build_windows_installer.ps1
```

The build requires Python 3.12 and Inno Setup 6. It uses PyInstaller, writes
generated files only below ignored `.cache`, `build`, and `dist` directories,
reads the version from `pyproject.toml`, and emits a SHA-256 sidecar. Pass
`-SignCertificateThumbprint` (or set `TRAVELMOVIEAI_SIGN_CERTIFICATE`) to sign
the exact current-version installer with `signtool.exe`; stale executables in
the output directory are never signed or reported as the new build. The desktop
holds the same named mutex used by Inno Setup for its full lifetime, protecting
upgrade and uninstall operations while it is running. It does not bundle model
weights or make administrator-level changes. The
installed shell includes package-local web assets and default configuration,
stores mutable workspace/model state below `%LOCALAPPDATA%\TravelMovieAI`, checks
for a port conflict, keeps an editable per-user `settings.toml`, writes bounded
privacy-redacted logs, and opens the browser only after Uvicorn reports ready.
Unavailable semantic, speech, narration, CUDA, or FFmpeg-dependent controls are
disabled by the web capability response. From a Python installation,
`travelmovieai doctor` reports missing FFmpeg filters, configured offline model
snapshots, and optional AI runtimes; the base installer does not install that
shell command. The web UI's `Diagnostics` download contains a sanitized runtime
report and, when available, the tail of the rotating application log.

`travelmovieai report` writes one CSP-restricted, self-contained offline HTML
file with project metrics, events, scene selection explanations, diagnostics,
and escaped user/model text. It references no remote scripts, fonts, images, or
analytics.

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
- keep `device = "auto"`, `workers = 0`, and `batch_size = 0` so the hardware
  profile can use the available CPU and GPU resources;
- choose `resource_mode = "performance"` for more CPU pressure, or `safe` when
  diagnosing instability;
- use `device = "cpu"` to isolate a GPU-driver issue; use
  `max_gpu_processes = 1` to retain CUDA with serialized FFmpeg GPU work.

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

The following planned capabilities are implemented in the current development
tree. Checkboxes describe delivery status; optional model/tool installation is
still required where noted above.

### P0: Long-running job reliability

- [x] pause and cancel movie jobs;
- [x] terminate active FFmpeg, ACE-Step, and Piper process trees on cancellation;
- [x] serialize pipeline, manual edits, reports, exports, and restores with
  cross-process workspace/target leases;
- [x] fail fast when restart-safe job state cannot be written and expose
  degraded persistence after later write failures;
- [x] add correlation IDs, privacy-redacted rotating logs, safe background-error
  boundaries, and a downloadable diagnostic bundle;
- [x] persist truthful cache, retry, primary/fallback provider, and model
  execution metadata without resolved local paths;
- [x] requeue interrupted scan/movie jobs with the same ID and reuse valid
  stage/render-segment checkpoints;
- [x] persist movie-job history;
- [x] add a local Piper provider to the explicit Voice Synthesis stage;
- [x] enforce disk-cache limits and cleanup;
- [x] check free disk space before rendering;
- [x] size transition preflight from the full lossless mezzanine working set and
  account for split workspace/output volumes;
- [x] validate persisted interval/count/provider contracts and self-repair
  Vision, story, voice, and generated-music caches without overwriting manual
  fields;
- [x] add frozen, ordered, retry-safe SQLite migrations.
- [x] distinguish missing audio windows from decode failures and retry previously
  failed media probes instead of caching errors as success;

### P1: Editing quality

- [x] FAISS indexing and archive search over local semantic embeddings;
- [x] GPS and embeddings in event detection;
- [x] richer shot-scale and camera-motion extraction from Vision AI;
- [x] 3/5/9-frame contact-sheet integrity checks and typed temporal highlight
  windows with provider-aware Vision retry/degraded metadata;
- [x] hard automatic source/event diversity caps, micro-event limits, editorial
  title filtering, and role-aware deterministic backfill;
- [x] full-duration black/freeze/silence/loudness/true-peak quality gates;
- [x] lossless transition mezzanine video, ALAC segment audio, delivery fades,
  loudness normalization, and custom-font cache invalidation;
- [x] verify lossless transition checkpoint codecs and publish rendered movies
  only after the delivery quality gate passes;
- [x] timeline version comparison in the web UI.

### P1: Story and manual editing

- [x] direct local story-model adapter;
- [x] multiple movie variants from one analysis;
- [x] event and scene reordering;
- [x] editable event titles, summaries, captions, transcripts, and landmarks;
- [x] timeline versioning and comparison.

### P2: Visual processing, music, and narration

- [x] Ken Burns effects for photos;
- [x] Vision focus-point and face/object-aware crop;
- [x] rotation metadata and vertical-video layouts;
- [x] per-scene quality-metric color and exposure normalization;
- [x] HDR-to-SDR tone mapping;
- [x] event titles, subtitles, credits, and safe-area validation;
- [x] BPM analysis for library/manual tracks and automatic music volume envelopes;
- [x] native full-duration 48 kHz ACE-Step generation without soundtrack loops
  or synthetic post-generation accents;
- [x] story-aware modern prompts, key/macro arrangement, Draft/Balanced/Studio
  quality presets, and hardware-aware Turbo/SFT/XL selection;
- [x] Best-of-N candidate generation, deterministic automated scoring, Web
  audition, reference audio, LoRA support, and input-aware music caching;
- [x] Piper narration synthesis.
- [x] speech-budgeted narration lines timed against the actual selected timeline;

### P2: Performance

- [x] batched Vision inference;
- [x] persistent loaded-model reuse;
- [x] hardware-aware shared resource profiles, offline model diagnostics, and a
  configurable bounded Vision model LRU;
- [x] weighted typed pipeline progress and privacy-safe per-run timing manifest;
- [x] bounded parallel scene detection with per-asset restart checkpoints;
- [x] per-scene Whisper restart checkpoints;
- [x] artifact-only embedding vectors without SQLite duplication;
- [x] non-retaining SQLite connection pools for short-lived CLI/web repositories;
- [x] source-keyed proxy media before 4K/8K scene and frame decoding;
- [x] metadata/SQLite benchmark for 512 assets and a virtual 100+ GB source set;
- [x] CLI runtime, output, and peak-workspace estimates.

### P3: Product delivery

- [x] per-user Windows installer recipe with `%LOCALAPPDATA%` runtime state;
- [x] frozen per-user configuration bootstrap, installer hashing/signing hooks,
  and a base-only non-interactive launcher;
- [x] application-lifetime installer mutex, exact-version installer artifact
  selection, and kill-on-close Windows Job Objects for long-running tools;
- [x] exact same-origin mutation checks and a restrictive CSP for the local web UI;
- [x] FFmpeg/filter, configured model-snapshot, and optional-runtime diagnostics;
- [x] project backup and export;
- [x] HTML report;
- [x] optional PySide6 desktop shell;
- [x] documented provider/plugin interface.

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
scripts/build_windows_installer.ps1 Windows installer build
scripts/benchmark_metadata_scale.py Large-project metadata benchmark
installer/                      PyInstaller and Inno Setup definitions
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
assets/music/                   Optional local soundtrack library; empty by default
assets/fonts/                   Reserved for distributable fonts; empty by default
workspace/                      Generated project data; never commit
```

## Privacy and Security

- the web server enforces a loopback bind, loopback Host validation, and a
  loopback Origin for browser mutations;
- raw media and derived frames remain local;
- external processes receive argument lists rather than shell-built commands;
- no remote inference provider or cloud credential is configured;
- workspace data, models, databases, frames, and rendered movies must not be
  committed.

## License

See [LICENSE](LICENSE).
