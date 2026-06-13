# Архитектура TravelMovieAI

TravelMovieAI — локальный многостадийный конвейер. Сначала система индексирует
и понимает материалы, затем строит сюжет и только после этого принимает
монтажные решения.

## Слои

| Пакет | Ответственность |
| --- | --- |
| `domain` | Pydantic-контракты, enum и стабильные модели данных |
| `application` | пользовательские сценарии и `TravelMovieService` |
| `pipeline` | порядок и запуск стадий |
| `media` | поиск файлов и нормализация metadata |
| `analysis` | сцены, кадры, качество, Vision, Speech, дубли |
| `story` | события, storyboard, ranking, музыка |
| `editing` | timeline и FFmpeg renderer |
| `infrastructure` | SQLite, FFmpeg, LM Studio, Whisper и системные адаптеры |
| `web` | FastAPI, background jobs и статический интерфейс |

Зависимости направлены внутрь: `domain` не импортирует инфраструктуру, а
внешние провайдеры скрыты за адаптерами.

## Pipeline

Канонический порядок задаётся `PipelineStage`:

```text
Media Scan
→ Scene Detection
→ Frame Sampling
→ Quality Analysis
→ Vision AI Analysis
→ Speech Analysis
→ Audio Analysis
→ Embeddings
→ Duplicate Detection
→ Scene Captioning
→ Event Detection
→ Story Builder
→ Scene Ranking
→ Music Selection
→ Narration / Voice
→ Timeline Builder
→ Rendering
```

Реализованы Media Scan, Scene Detection, Frame Sampling, Quality/Vision/Speech,
perceptual Duplicate Detection, Scene Captioning, Event Detection и базовый
Story Builder. Рабочий `create`-маршрут также выполняет ranking, music plan,
timeline и rendering.

Audio Analysis, embeddings, narration, voice synthesis и расширенный report
пока остаются незавершёнными.

Каждая стадия:

- получает `ProjectContext`;
- возвращает `StageResult`;
- пишет данные только в workspace;
- не изменяет исходные медиа;
- лениво загружает optional dependencies;
- должна поддерживать повторный запуск и проверяемый кэш.

## Основной поток

```text
Browser / CLI
      |
      v
TravelMovieService
      |
      +--> Media Scan --> FFprobe/Pillow --> SQLite
      |
      +--> Scene Detection --> PySceneDetect/fallback
      |
      +--> Frame Sampling --> RGB PNG contact sheets
      |
      +--> OpenCV Quality --> Vision AI --> optional Whisper
      |
      +--> Duplicates --> Events --> Storyboard --> Ranking
      |
      +--> Timeline + Music
      |
      v
QuickMontageRenderer --> FFmpeg --> FFprobe validation
```

Vision AI понимает смысл сцены. OpenCV используется только для измеримых
технических признаков. Renderer исполняет готовый timeline и не должен заново
решать, какие сцены важны.

## Web

Entrypoints:

```text
scripts/run_web.bat → main.py → travelmovieai.web.server → Uvicorn/FastAPI
```

Основные API:

```text
GET   /api/health
GET   /api/capabilities
POST  /api/scans
GET   /api/scans/{id}
GET   /api/scans/{id}/result
POST  /api/movies
GET   /api/movies/{id}
GET   /api/movies/{id}/download
GET   /api/scenes
PATCH /api/scenes/{id}
GET   /api/scenes/{id}/thumbnail
```

Scan и Movie jobs выполняются в ограниченных worker pools. Параллельные
операции с одним workspace отклоняются. Сервер по умолчанию слушает
`127.0.0.1`; авторизация не реализована.

Movie job хранит общий прогресс `0–100%`, текущую фазу, elapsed/ETA, профиль
ресурсов, состояние отдельных подзадач и до 250 последних сообщений. Web UI
получает это состояние коротким polling-запросом и отображает отдельные бары
стадий и журнал без записи приватных кадров или транскриптов в него.

## Ресурсы и параллелизм

`infrastructure.system.detect_resource_profile()` определяет CPU, RAM, CUDA и
NVENC. При `WORKERS=0` профиль отдельно выбирает число задач для Frame Sampling,
OpenCV и FFmpeg rendering. CPU-рендер распределяет потоки FFmpeg между
одновременными сегментами, а NVENC используется автоматически при доступности.
Vision-запросы к LM Studio остаются последовательными: загрузкой GPU и слоёв
модели управляет сам LM Studio.

## Workspace

```text
workspace/<project>/
├── project.db
├── frames/
├── cache/
└── artifacts/
```

`project.db` содержит:

- `media_assets` — пути и FFprobe/EXIF metadata;
- `scenes` — границы, кадры, transcript, scores и ручные overrides;
- `events` — группы сцен, время, location/activity, landmarks и importance.

SQLite работает с `foreign_keys=ON` и `journal_mode=WAL`.

Основные артефакты:

| Файл | Назначение |
| --- | --- |
| `analysis.json` | снимок Media Scan |
| `scenes.json` | границы сцен |
| `quality_analysis.json` | технические метрики |
| `vision_analysis.json` | структурированное понимание сцен |
| `speech_analysis.json` | transcript и language |
| `duplicates.json` | группы визуальных дублей |
| `events.json` | события |
| `storyboard.json` | секции истории |
| `selection_decisions.json` | объяснения выбора |
| `quick_timeline.json` | монтажный план |
| `preview.mp4`, `final.mp4` | результаты рендера |

JSON записывается атомарно. Contact sheets имеют версию в имени файла и
сохраняются как RGB PNG для совместимости с limited-range YUV.

## Кэш

Media metadata переиспользуется при совпадении пути, размера и `modified_ns`.
Scene/Vision/Speech stages дополняют ключ параметрами detector, модели,
prompt/schema version и временными границами.

Ручные решения хранятся в scene metadata и не требуют повторного Vision AI
анализа.

## Rendering

Renderer:

- нормализует resolution, FPS, pixel format и аудиоформат;
- параллельно готовит независимые сегменты в рамках профиля ресурсов;
- создаёт silent audio для немых источников;
- применяет `xfade`/`acrossfade`;
- добавляет generated/library/manual music и ducking;
- использует `h264_nvenc` или `libx264`;
- записывает файл атомарно;
- проверяет video/audio streams и duration через FFprobe.

Preview ограничивается 854×480 и 24 FPS.

## Приватность и безопасность

- исходные файлы открываются только для чтения;
- процессы запускаются списками аргументов без shell interpolation;
- cloud mode не включается автоматически;
- модели не скачиваются при импорте;
- raw media, кадры, GPS, лица, голоса и transcripts считаются приватными;
- workspace, `.env`, модели и rendered media не коммитятся.

## Тестирование

Default suite не требует интернета, GPU или моделей. Провайдеры заменяются
fake-адаптерами, а FFmpeg integration tests используют маленькие синтетические
медиа, включая limited-range YUV и Unicode-пути.
