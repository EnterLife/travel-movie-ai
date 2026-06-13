# Архитектура TravelMovieAI

TravelMovieAI строится как локальный многостадийный Python-конвейер. Основной
принцип: сначала понять содержимое и историю поездки, затем принимать монтажные
решения.

## Статус реализации

| Стадия | Статус |
| --- | --- |
| 1. Media Scan | Реализована |
| 2. Scene Detection | Реализована |
| 3. Frame Sampling | Реализована, start/middle/end contact sheet |
| 4. Quality Analysis | Реализована, OpenCV/Pillow metrics |
| 4.5. Vision AI Analysis | Qwen/LM Studio и Florence-2 adapters реализованы |
| 6. Speech Analysis | Заглушка |
| 7. Audio Analysis | Заглушка |
| 8. Embeddings | Заглушка |
| 8.5. Scene Captioning | Базовая multimodal-композиция реализована |
| 9. Event Detection | Базовая semantic/temporal clustering реализована |
| 10. Story Builder | Полный storyboard пока заглушка |
| 11. Scene Ranking | Базовое semantic ranking реализовано |
| 12. Music Selection | Локальный выбор реализован |
| 13. Narration | Заглушка |
| 14. Voice Synthesis | Заглушка |
| 15. Timeline Builder | Quick/semantic montage plan реализован |
| 16. Rendering | Переходы, музыка, ducking и NVENC реализованы |

Порядок стадий определён перечислением `PipelineStage`. Реестр
`build_default_pipeline()` обязан сохранять тот же порядок.

Локальный веб-интерфейс и фоновые Media Scan/Movie jobs реализованы.
Movie builder поддерживает быстрый хронологический и локальный semantic режимы.

## Слои

### `domain`

Стабильные Pydantic-контракты:

- `MediaAsset`;
- `MediaScanReport`;
- `Scene`;
- `SceneDetectionReport`;
- `FrameSamplingReport`;
- `SceneUnderstanding`;
- `VisionAnalysisReport`;
- `Event`;
- `EventDetectionReport`;
- `MultimodalSceneDescription`;
- `Storyboard`;
- `Timeline`;
- `StageResult`.

Domain не зависит от SQLAlchemy, FFmpeg, моделей или файловой системы.

### `application`

Пользовательские сценарии и контекст проекта:

- `TravelMovieService`;
- `ProjectContext`;
- выбор input, workspace, output, style и cloud mode.

### `web`

Локальный HTTP-слой:

- FastAPI application factory;
- Uvicorn launcher;
- очередь фоновых сканирований с одним worker;
- Pydantic API schemas;
- автономные HTML/CSS/JavaScript assets;
- health, scan status и scan result endpoints.
- movie status, progress и MP4 download endpoints.

### `pipeline`

Оркестрация стадий:

- интерфейс `Stage`;
- `PipelineRunner`;
- реестр стадий;
- конкретные реализации в `pipeline/stages`.

### `media`

Поиск файлов и преобразование внешних метаданных в доменную модель.

### `analysis`

Scene detection, frame sampling, OpenCV quality и Vision AI реализованы.
Speech, audio и embeddings остаются следующими модулями.

### `story`

Реализованы базовые event clustering, multimodal scene descriptions, ranking и
музыкальный план. Полный storyboard и narration остаются следующими этапами.

### `editing`

Будущие timeline builder и FFmpeg renderer.

### `infrastructure`

Интеграции:

- SQLite/SQLAlchemy;
- FFmpeg и FFprobe;
- атомарная запись артефактов;
- LM Studio;
- vision providers;
- Faster Whisper.

## Поток текущей стадии

```text
Browser / CLI
        |
        +--> FastAPI --> ScanJobManager
        |                    |
        +--------------------+
        |
        v
TravelMovieService
        |
        v
PipelineRunner
        |
        v
MediaScanStage
        |
        +--> MediaAssetRepository --> project.db
        |
        +--> MediaScanner
               |
               +--> FFprobeClient
               |
               +--> Pillow / EXIF
        |
        +--> write_json_atomic --> artifacts/analysis.json
```

## Веб-сервер

Основные entrypoints:

```text
scripts/run_web.bat
        |
        v
main.py
        |
        v
travelmovieai.web.server
        |
        v
Uvicorn / FastAPI
```

BAT-скрипт создаёт `.venv`, если окружение отсутствует, устанавливает базовый
пакет и запускает сервер. По умолчанию браузер открывает:

```text
http://127.0.0.1:8000
```

HTTP API:

```text
GET  /api/health
POST /api/scans
GET  /api/scans
GET  /api/scans/{job_id}
GET  /api/scans/{job_id}/result
POST /api/movies
GET  /api/movies/{job_id}
GET  /api/movies/{job_id}/download
GET  /api/docs
```

Media Scan выполняется в `ThreadPoolExecutor` с одним worker. Это не блокирует
HTTP-запрос, но предотвращает параллельную обработку нескольких проектов внутри
одного процесса. Повторное активное задание в тот же workspace получает HTTP
409.

История сохраняется атомарно:

```text
workspace/.web/jobs.json
```

После перезапуска completed jobs и их `analysis.json` снова доступны через API.
Jobs, которые были `queued` или `running` во время остановки, переводятся в
`failed` с причиной о прерывании.

`/api/health` проверяет доступность и версии FFmpeg/FFprobe. Сервер считается
готовым к Media Scan, когда доступен FFprobe.

Сервер слушает только loopback-интерфейс по умолчанию. Авторизация пока не
реализована, поэтому публикация через `0.0.0.0` допустима только в доверенной
изолированной сети.

## Workspace

Текущая структура:

```text
workspace/<project>/
├── project.db
├── project.db-wal
├── project.db-shm
├── frames/
├── cache/
└── artifacts/
    └── analysis.json
```

`project.db-wal` и `project.db-shm` являются временными файлами SQLite и могут
отсутствовать после закрытия соединений.

Будущая целевая структура:

```text
workspace/<project>/
├── project.db
├── frames/
├── cache/
└── artifacts/
    ├── analysis.json
    ├── scenes.json
    ├── frame_sampling.json
    ├── quality_analysis.json
    ├── vision_analysis.json
    ├── scene_descriptions.json
    ├── events.json
    ├── storyboard.json
    ├── music_plan.json
    ├── timeline.json
    ├── render_config.json
    ├── voiceover.wav
    ├── report.html
    └── final.mp4
```

Наличие файла в целевой схеме не означает, что его генерация уже реализована.

## Media Scan

`MediaScanStage` выполняет:

- рекурсивный поиск поддерживаемых расширений;
- регистронезависимое сравнение расширений;
- исключение workspace, если он расположен внутри input;
- чтение FFprobe JSON через список аргументов процесса;
- чтение размеров и EXIF GPS фотографий;
- создание доменных `MediaAsset`;
- синхронизацию SQLite;
- атомарную запись `analysis.json`.

### Поддерживаемые расширения

```text
Видео:       .mp4 .mov .avi .mkv .m4v
Фотографии:  .jpg .jpeg .png .heic
Аудио:       .mp3 .wav .flac .m4a
```

### Инкрементальный кэш

Запись переиспользуется, если совпадают:

- нормализованный абсолютный путь;
- `size_bytes`;
- `modified_ns`.

Новый или изменённый файл проходит FFprobe заново. Исчезнувший файл удаляется
из `media_assets` во время успешной синхронизации.

Ограничение: содержимое, заменённое при сохранении размера и timestamp, не
обнаруживается. В будущем можно добавить опциональный быстрый fingerprint.

### Ошибки отдельных файлов

`MediaProbeError` сохраняется в `MediaAsset.scan_error`. Ошибка одного файла не
останавливает весь проект.

Отсутствующий `ffprobe` является ошибкой зависимости и останавливает стадию,
поскольку без него нельзя гарантировать единообразное сканирование.

### Атомарность JSON

`analysis.json` сначала записывается во временный файл в том же каталоге, затем
заменяется через `os.replace`. При сбое предыдущий корректный JSON не должен
превратиться в частично записанный документ.

## Movie Builder

Рабочий маршрут поддерживает два режима:

```text
Media Scan
    |
    v
Scene Detection (semantic mode)
    |
    +--> PySceneDetect or uniform fallback
    +--> start/middle/end contact sheet
    +--> OpenCV quality metrics
    +--> Qwen/LM Studio or Florence-2 structured vision analysis
    +--> multimodal scene descriptions
    +--> temporal and semantic event clustering
    +--> importance, landmark, event, quality and diversity ranking
    |
    v
quick_timeline.json
    |
    v
QuickMontageRenderer
    |
    +--> normalized H.264/AAC segments
    +--> silent audio for sources without audio
    +--> xfade/acrossfade transitions
    +--> generated/library/manual music plan and ducking
    +--> h264_nvenc or libx264
    |
    v
final.mp4
```

Все сегменты приводятся к одинаковым width, height, FPS, pixel format, sample
rate и channel layout. Без эффектов используется concat demuxer; переходы и
музыка собираются через FFmpeg filter graph.

Текущие ограничения:

- нет duplicate removal;
- нет полноценного storyboard, ручного event editor, titles и subtitles;
- фотографии показываются статично;
- rotation metadata и сложные HDR/color pipelines пока не обрабатываются;
- movie jobs пока не сохраняются после перезапуска сервера.

## SQLite

Хранилище использует SQLAlchemy 2 и SQLite.

При подключении включаются:

```sql
PRAGMA foreign_keys=ON;
PRAGMA journal_mode=WAL;
```

Таблица `media_assets` хранит:

- UUID;
- абсолютный `source_path`;
- `relative_path`;
- media type и extension;
- размер;
- время изменения;
- время создания;
- duration, width, height и FPS;
- latitude и longitude;
- сокращённые probe metadata;
- scan error;
- время последнего сканирования.

Синхронизация выполняется в транзакции. Сначала обновляются или добавляются
актуальные записи, затем удаляются отсутствующие пути.

Таблица `scenes` хранит временные границы, representative frame, validated
vision metadata и scores. Внешний ключ на `media_assets` использует
`ON DELETE CASCADE`.

Таблица `events` хранит состав события, временной диапазон, location/activity,
landmarks, summary, importance и confidence.

## Конфигурация

Настройки загружаются через Pydantic Settings:

1. значения, явно переданные приложением;
2. переменные окружения `TRAVELMOVIEAI_*`;
3. `.env`;
4. значения по умолчанию.

Основные параметры:

```text
workspace
database_filename
ffmpeg_binary
ffprobe_binary
lm_studio_url
vision_provider
whisper_model
device
cloud_enabled
batch_size
workers
web_host
web_port
web_history_limit
```

`database_filename` валидируется как имя файла без каталогов.

## Безопасность и приватность

- Исходные медиа открываются только для чтения.
- Внешние процессы запускаются списком аргументов без shell interpolation.
- Cloud mode выключен по умолчанию.
- Пути, GPS, кадры, транскрипты и SQLite считаются приватными данными.
- Workspace, `.env`, модели и итоговые медиа исключаются из Git.

## Границы будущих стадий

Каждая стадия должна:

- реализовывать `Stage`;
- получать `ProjectContext`;
- возвращать `StageResult`;
- писать данные только в workspace;
- поддерживать повторный запуск;
- проверять валидность кэша;
- лениво загружать optional dependencies и модели;
- выдавать типизированные артефакты;
- не менять исходные медиа.

Timeline должен оставаться декларативным. Renderer исполняет готовый timeline, а
не принимает решения о сюжете.

## Тестирование

Текущий набор тестов проверяет:

- порядок стадий;
- подготовку workspace;
- FFprobe parsing;
- Unicode и пробелы в путях;
- фильтрацию расширений;
- исключение workspace;
- кэширование;
- обработку повреждённого медиа;
- SQLite update/delete;
- запись `analysis.json`;
- пользовательский вывод CLI;
- выдачу web page и static assets;
- health endpoint;
- валидацию web paths;
- полный цикл фонового scan job через HTTP API.
- HTTP 409 для занятого workspace;
- восстановление job history после перезапуска;
- перевод interrupted jobs в failed;
- degraded health без FFprobe.

Обычные тесты используют fake probe и не требуют моделей, GPU или интернета.
Реальные FFmpeg integration tests должны использовать маленькие синтетические
медиа.
