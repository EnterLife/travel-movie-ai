# TravelMovieAI

## Technical Specification (MVP + Architecture)

Version: 1.0

Repository: travel-movie-ai

> Implementation note: this document describes the target MVP and architecture.
> It is not a statement that every stage is already available. Current
> implementation status and installation instructions are maintained in
> [README.md](../README.md) and
> [installation-and-usage.md](installation-and-usage.md).

Tagline:

> Your personal AI travel filmmaker.

---

# 1. Project Overview

TravelMovieAI is a local AI-powered video editing system that automatically transforms raw travel footage into a story-driven movie.

The user provides a folder containing videos and photos from a trip.

The system automatically:

* analyzes media content;
* understands scenes and events;
* detects people, landmarks, activities, and emotions;
* removes duplicates and low-quality footage;
* creates a coherent story;
* generates titles and optional narration;
* selects music;
* builds a timeline;
* renders a complete movie.

The primary goal is to create a fully automated travel movie with minimal user interaction.

---

# 2. Core Principles

## Local First

The system must work completely offline.

Supported local AI:

* Whisper
* Qwen2.5-VL
* Florence-2
* LM Studio

Cloud AI is optional.

---

## AI Director Approach

The system should behave as:

* video editor;
* director;
* storyteller;
* archivist.

The objective is not to create a clip compilation.

The objective is to create a meaningful movie.

---

## Story Before Editing

Editing decisions must be based on story structure.

The pipeline should first understand:

* what happened;
* where it happened;
* who participated;
* why it is important.

Only then should editing begin.

---

# 3. Supported Media

## Video

Supported formats:

* mp4
* mov
* avi
* mkv
* m4v

---

## Photos

Supported formats:

* jpg
* jpeg
* png
* heic

Photos may be inserted into the final movie.

---

## Audio

Supported formats:

* mp3
* wav
* flac
* m4a

Used for:

* music
* narration
* soundtrack replacement

---

# 4. Technology Stack

## Core

Python 3.12+

---

## Video Processing

* FFmpeg
* FFprobe
* PySceneDetect
* OpenCV

---

## AI Speech

* Faster Whisper

Models:

* medium
* large-v3

---

## Vision AI

Primary:

* Qwen2.5-VL

Supported:

* 7B
* 32B

Alternative:

* Florence-2

Supported:

* base
* large

---

## LLM

Local:

* LM Studio

Cloud:

* Yandex GPT OSS 120B

---

## Embeddings

* sentence-transformers
* FAISS

---

## Database

* SQLite
* SQLAlchemy

---

## UI

MVP:

* CLI
* Local web interface

Future:

PySide6 Desktop UI

---

# 5. High-Level Architecture

Media Folder

↓

Media Scan

↓

Scene Detection

↓

Frame Sampling

↓

Vision AI Analysis

↓

Speech Analysis

↓

Audio Analysis

↓

Embeddings

↓

Event Detection

↓

Story Builder

↓

Scene Ranking

↓

Timeline Builder

↓

Music Selection

↓

Narration

↓

FFmpeg Renderer

↓

Movie

---

# 6. Pipeline Stages

## Stage 1. Media Scan

Scan all media files.

Extract:

* path
* file type
* duration
* resolution
* FPS
* size
* creation date
* GPS metadata

Store in SQLite.

Output:

project.db

---

## Stage 2. Scene Detection

Split videos into scenes.

For each scene:

* start time
* end time
* duration
* keyframe

Output:

scenes.json

---

## Stage 3. Frame Sampling

Extract representative frames.

Required frames:

* scene start
* scene middle
* scene end
* keyframes

Output:

frames/

---

## Stage 4. Vision AI Analysis

IMPORTANT:

Vision AI is the primary source of scene understanding.

OpenCV acts only as a supplementary quality-analysis tool.

---

### Vision Models

Primary:

Qwen2.5-VL

Fallback:

Florence-2

---

### Objectives

Understand:

* people
* places
* activities
* landmarks
* emotions
* story relevance

---

### Scene Understanding

Generate structured metadata.

Example:

```json
{
  "caption": "Family walking on a beach during sunset",
  "location_type": "beach",
  "activity": "walking",
  "emotion": "relaxing",
  "people_count": 4
}
```

---

### Location Detection

Categories:

* beach
* sea
* city
* mountains
* airport
* hotel
* museum
* restaurant
* forest
* park
* landmark

---

### Activity Detection

Examples:

* walking
* swimming
* sightseeing
* dining
* hiking
* cycling
* traveling

---

### Emotion Detection

Categories:

* joyful
* exciting
* relaxing
* romantic
* emotional
* adventurous
* cinematic

---

### Landmark Detection

Examples:

* Eiffel Tower
* Colosseum
* Louvre
* Tokyo Tower

---

### Importance Detection

Vision AI must estimate scene importance.

Output:

0–100 score

---

### Event Clustering

Scenes should be grouped into events.

Example:

Airport

↓

Taxi

↓

Hotel

↓

Event: Arrival Day

---

## Stage 5. Visual Quality Analysis

OpenCV-based metrics.

Calculate:

* blur score
* brightness
* contrast
* noise
* motion
* camera shake

Purpose:

quality ranking only.

Not scene understanding.

---

## Stage 6. Speech Analysis

Use Faster Whisper.

Extract:

* transcript
* language
* confidence

Store per scene.

---

## Stage 7. Audio Analysis

Detect:

* speech
* music
* applause
* laughter
* silence
* crowd noise

Generate audio importance score.

---

## Stage 8. Embeddings

Generate embeddings for:

* captions
* transcripts
* events

Use:

* sentence-transformers
* FAISS

Tasks:

* duplicate detection
* similarity search
* clustering

---

## Stage 9. Event Detection

Combine:

* Vision AI
* Whisper
* Audio

Into logical events.

Examples:

* Arrival
* City Walk
* Museum Visit
* Beach Day
* Dinner
* Sunset

Output:

events.json

---

## Stage 10. Story Builder

Most important AI component.

Input:

* scenes
* events
* transcripts
* scores

Output:

storyboard.json

---

### Story Structure

Example:

Introduction

↓

Arrival

↓

Exploration

↓

Highlights

↓

Best Moments

↓

Finale

↓

Credits

---

### Story Styles

Supported:

* cinematic
* documentary
* family
* vlog
* adventure
* romantic

---

## Stage 11. Scene Ranking

Generate final score.

Factors:

* vision score
* emotion score
* uniqueness
* video quality
* transcript importance
* event importance

Output:

0–100

---

## Stage 12. Music Selection

Music categories:

* cinematic
* emotional
* calm
* energetic
* travel

Music stored locally.

Output:

music_plan.json

---

## Stage 13. Narration Generation

Optional.

Generate movie narration.

Providers:

Local:

* LM Studio

Cloud:

* Yandex GPT OSS 120B

Example:

"On the third day of the trip, the family visited Kyoto..."

---

## Stage 14. Voice Synthesis

Optional.

Supported:

* XTTS
* Piper
* Edge TTS

Output:

voiceover.wav

---

## Stage 15. Timeline Builder

Generate final editing plan.

Output:

timeline.json

Contains:

* scene order
* transitions
* music
* narration
* subtitles

---

## Stage 16. Rendering

Use FFmpeg.

Features:

* crossfade
* fade in
* fade out
* subtitles
* music
* narration
* titles

Output:

final.mp4

---

# 7. Cloud Integration

Optional mode.

Flag:

```bash
--cloud
```

Provider:

Yandex GPT OSS 120B

Usage:

* advanced story generation
* narration writing
* title generation
* alternate movie versions

Cloud must never be required.

---

# 8. CLI

Create movie:

```bash
travelmovieai create
```

Analyze media:

```bash
travelmovieai analyze
```

Generate storyboard:

```bash
travelmovieai storyboard
```

Render movie:

```bash
travelmovieai render
```

Generate report:

```bash
travelmovieai report
```

---

# 9. Output Files

project.db

analysis.json

events.json

storyboard.json

timeline.json

render_config.json

final.mp4

report.html

Current implementation note:

* `scenes.json`, representative frames, and `vision_analysis.json` are available;
* local Qwen-compatible semantic analysis runs through LM Studio;
* scene ranking, local music, ducking, transitions, `quick_timeline.json`, and
  `final.mp4` are available;
* event clustering, multimodal Story Builder, narration, subtitles, and full
  quality-aware ranking are not yet implemented.

---

# 10. Performance Requirements

Must support:

* 500+ videos
* 100+ GB footage

Support:

* CUDA
* DirectML
* CPU fallback

Batch processing required.

Multi-threading required.

Incremental caching required.

---

# 11. Future Versions

## v2

Desktop GUI

PySide6

---

## v3

Natural language editing

Example:

"Create a movie focused on beaches and sunsets."

---

## v4

Personal media archive search

Example:

"Show all beach videos from the last 5 years."

---

## v5

AI Director Mode

Generate multiple movie versions automatically:

* cinematic
* documentary
* family
* adventure

and rank them by predicted viewer engagement.
