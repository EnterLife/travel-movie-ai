# Техническое задание TravelMovieAI

TravelMovieAI — local-first система автоматического монтажа, превращающая
видео, фотографии и аудио из поездки в связный фильм.

Статус реализации ведётся в [README.md](../README.md), план развития — в
[roadmap.md](roadmap.md).

## 1. Цель продукта

Пользователь выбирает папку с материалами. Система должна:

1. проиндексировать медиа;
2. разделить видео на сцены;
3. понять содержание, качество, речь и звук;
4. найти дубли и объединить сцены в события;
5. построить историю;
6. создать timeline;
7. отрендерить готовый фильм.

Результат должен быть осмысленным travel movie, а не случайной нарезкой.

## 2. Принципы

- **Local first:** обычная работа полностью локальна.
- **Story before editing:** сначала история, затем монтаж.
- **Vision first:** смысл сцены определяет Vision AI, не OpenCV.
- **Non-destructive:** исходные файлы не изменяются и не удаляются.
- **Deterministic where practical:** решения и параметры сохраняются.
- **Incremental:** повторный запуск переиспользует валидный кэш.
- **Optional acceleration:** CUDA/DirectML ускоряют работу, но CPU остаётся
  доступным.
- **Explicit cloud:** облачные провайдеры никогда не включаются автоматически.

## 3. Поддерживаемые медиа

| Тип | Форматы |
| --- | --- |
| Видео | `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v` |
| Фото | `.jpg`, `.jpeg`, `.png`, `.heic` |
| Аудио | `.mp3`, `.wav`, `.flac`, `.m4a` |

Пути могут содержать пробелы, Unicode и длинные имена.

## 4. Технологии

- Python 3.12+;
- Typer CLI;
- FastAPI/Uvicorn web UI;
- Pydantic/Pydantic Settings;
- SQLite/SQLAlchemy;
- FFmpeg/FFprobe;
- PySceneDetect/OpenCV/Pillow;
- Faster Whisper;
- Qwen2.5-VL через LM Studio;
- Florence-2 как локальная альтернатива;
- sentence-transformers/FAISS;
- pytest, Ruff, mypy.

Тяжёлые AI-зависимости должны оставаться optional dependency groups.

## 5. Pipeline

```text
Media Scan
→ Scene Detection
→ Frame Sampling
→ Visual Quality Analysis
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

Изменение порядка или контракта стадии требует синхронного обновления domain
models, serialization, consumers, tests и architecture docs.

## 6. Контракты стадий

### 6.1 Media Scan

Извлекает:

- путь и тип;
- размер и timestamps;
- duration, resolution, FPS;
- GPS/EXIF при наличии;
- streams и сокращённые probe metadata.

Результат сохраняется в SQLite и `analysis.json`.

### 6.2 Scene Detection

Для каждого видео создаёт сцены с:

- `start_seconds`;
- `end_seconds`;
- detector/cache metadata.

Используется PySceneDetect, при невозможности — ограниченные по длине
равномерные сегменты.

### 6.3 Frame Sampling

Для сцены извлекаются начало, середина и конец. Представление должно:

- быть компактным;
- подходить Vision AI и web preview;
- кэшироваться;
- корректно обрабатывать limited/full-range YUV.

Текущий целевой формат contact sheet — RGB PNG.

### 6.4 Visual Quality Analysis

OpenCV измеряет только технические признаки:

- sharpness/blur;
- brightness/exposure;
- contrast;
- saturation/colorfulness;
- noise;
- motion;
- camera shake.

Выход: score 0–100 и объяснимые причины брака. Пользователь может вручную
сохранить сцену.

### 6.5 Vision AI Analysis

Основная модель: Qwen2.5-VL 7B/32B. Альтернатива: Florence-2 base/large.

Модель получает representative frames и возвращает валидированный JSON:

```json
{
  "caption": "A family walking along the beach during sunset.",
  "detailed_description": "The family continues walking along the shoreline.",
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

Landmark нельзя выдумывать без визуального или текстового подтверждения.
Provider, model, prompt/schema version и параметры входят в cache metadata.

### 6.6 Speech Analysis

Faster Whisper извлекает:

- transcript;
- language;
- confidence;
- в целевой версии — timestamps реплик.

Модель загружается лениво и поддерживает CPU/CUDA.

### 6.7 Audio Analysis

Определяет:

- speech;
- music;
- silence;
- crowd;
- laughter;
- applause;
- значимые ambient sounds.

Выход должен использоваться при ranking, music ducking и сохранении атмосферы.

### 6.8 Embeddings и дубли

Perceptual hash находит визуально почти одинаковые сцены. Embeddings/FAISS
должны расширить поиск на семантически похожие материалы.

Дубли группируются, но не удаляются. Keeper выбирается по manual override,
importance и quality.

### 6.9 Scene Captioning

Единое описание сцены строится из:

- Vision AI;
- Whisper transcript;
- OpenCV quality;
- Audio Analysis.

Все model outputs валидируются до сохранения.

### 6.10 Event Detection

Сцены объединяются по:

- времени;
- GPS;
- location/activity;
- landmarks;
- embeddings;
- transcript/audio context.

Пример: аэропорт → такси → отель = `Arrival Day`.

### 6.11 Story Builder

Строит секции:

- opening;
- journey;
- highlights;
- finale;
- optional credits.

Поддерживаемые стили: `cinematic`, `documentary`, `family`, `vlog`,
`adventure`, `romantic`.

Story Builder получает metadata, но не raw media.

### 6.12 Scene Ranking

Итоговая оценка учитывает:

- vision importance;
- quality;
- emotion;
- uniqueness/diversity;
- landmark value;
- transcript/audio importance;
- event importance;
- duplicate/technical penalties;
- manual include/exclude.

Для каждой сцены сохраняется причина выбора или отклонения.

### 6.13 Music и narration

Музыка может быть:

- сгенерированной локально;
- выбрана из локальной библиотеки;
- указана вручную;
- отключена.

Целевая версия учитывает storyboard, BPM и beat grid. Narration и voice
synthesis опциональны.

### 6.14 Timeline и Rendering

Timeline декларативно хранит:

- порядок и границы сцен;
- transitions;
- titles/subtitles;
- music;
- narration.

Renderer:

- нормализует media streams;
- создаёт silent audio при необходимости;
- применяет transitions и ducking;
- использует NVENC или CPU;
- пишет результат атомарно;
- проверяет итоговый MP4 через FFprobe.

## 7. Пользовательский интерфейс

MVP web UI должен поддерживать:

- выбор input и workspace;
- запуск и прогресс анализа;
- выбор backend/model/device;
- preview и final render;
- просмотр кадров и scores;
- `Авто / Обязательно / Исключить`;
- скачивание MP4;
- понятные ошибки зависимостей и моделей.

Сервер по умолчанию доступен только на loopback-интерфейсе.

CLI остаётся стабильным интерфейсом для `analyze` и `create`.

## 8. Workspace и данные

Все generated data записываются внутрь workspace:

```text
project.db
frames/
cache/
artifacts/
```

Обязательные свойства:

- атомарная запись критичных JSON и media outputs;
- schema/version metadata;
- bounded worker pools;
- отсутствие raw media в логах и fixtures;
- возможность полного сброса удалением workspace;
- исходная папка только для чтения.

## 9. Производительность

Целевая нагрузка:

- 500+ видео;
- 100+ GB исходных материалов.

Необходимо:

- batch processing;
- bounded concurrency;
- proxy media для тяжёлых форматов;
- повторное использование моделей;
- инкрементальный кэш;
- CPU fallback;
- оценка времени и дискового пространства.

## 10. Приватность

Приватными считаются:

- исходные медиа;
- кадры;
- лица и голоса;
- transcripts;
- GPS;
- project database.

Телеметрия не требуется. Cloud mode должен отправлять только минимально
необходимый контекст и не загружать raw media без отдельного явно
документированного разрешения.

## 11. Критерии MVP

MVP считается готовым, когда пользователь может:

1. выбрать большую папку с медиа;
2. получить повторяемый локальный анализ;
3. создать разумный preview;
4. исправить выбор сцен;
5. повторно собрать фильм без повторного дорогого анализа;
6. получить валидный H.264/AAC MP4;
7. понять причины выбора и отклонения сцен.
