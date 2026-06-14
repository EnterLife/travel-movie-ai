# TravelMovieAI

Локальное Python-приложение, которое анализирует видео и фотографии из поездки,
отбирает сцены, строит сюжетный план и рендерит готовый MP4 через FFmpeg.

## Возможности

- рекурсивный Media Scan с FFprobe и SQLite-кэшем;
- Scene Detection через PySceneDetect с равномерным fallback;
- RGB PNG contact sheets из начала, середины и конца сцены;
- OpenCV-анализ качества, движения, тряски, шума и экспозиции;
- Vision AI через LM Studio или локальную Florence-2;
- опциональное распознавание речи через Faster Whisper;
- поиск визуальных дублей;
- группировка сцен в события и базовый Story Builder;
- объяснимый AI-отбор и ручные решения `Авто / Обязательно / Исключить`;
- быстрый preview и финальный H.264/AAC-рендер;
- переходы, музыка, ducking и NVIDIA NVENC с CPU fallback;
- автоматический профиль CPU/RAM/GPU и параллельная обработка кадров;
- общий прогресс, отдельные бары подзадач, ETA и журнал AI-монтажа;
- полностью локальная обработка без обязательных облачных сервисов.

Пока не реализованы: audio classification, embedding-поиск, ручной редактор
событий, субтитры, титры, narration и HTML-отчёт.

## Требования

- Windows 10/11;
- Python 3.12+;
- FFmpeg и FFprobe в `PATH`;
- LM Studio с мультимодальной моделью для Qwen/VLM-режима.

GPU необязателен. Media Scan и быстрый монтаж работают без AI-моделей.

## Быстрый запуск

```powershell
Set-Location C:\Users\bdo\travel-movie-ai
.\scripts\run_web.bat
```

Скрипт создаёт `.venv`, устанавливает базовые video-зависимости, запускает
сервер и открывает:

```text
http://127.0.0.1:8000
```

Рабочий сценарий:

1. Выберите папку с медиа и workspace через системный диалог или введите путь.
2. Запустите анализ.
3. Выберите Vision backend и модель.
4. Для первого прохода включите `Быстрый preview`.
5. Запустите AI-монтаж.
6. Следите за этапом, загрузкой ресурсов, ETA и журналом обработки.
7. Исправьте выбор сцен в галерее.
8. Отключите preview и соберите финальный фильм.

По умолчанию количество обработчиков определяется автоматически по числу
логических CPU, объёму RAM и наличию NVIDIA NVENC. Значения можно ограничить
через `TRAVELMOVIEAI_WORKERS`.

## CLI

```powershell
travelmovieai analyze `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026"

travelmovieai create `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026" `
  --output "D:\Movies\Japan2026.mp4" `
  --semantic `
  --style cinematic
```

Без `--semantic` используется хронологический quick montage.

## Установка зависимостей

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[video,dev]"
```

Дополнительные группы:

```powershell
python -m pip install -e ".[speech]"
python -m pip install -e ".[vision]"
python -m pip install -e ".[embeddings]"
python -m pip install -e ".[all,dev]"
```

Модели не загружаются при импорте или запуске тестов.

## Workspace

```text
workspace/<project>/
├── project.db
├── frames/
├── cache/
└── artifacts/
    ├── analysis.json
    ├── scenes.json
    ├── quality_analysis.json
    ├── vision_analysis.json
    ├── speech_analysis.json
    ├── duplicates.json
    ├── events.json
    ├── storyboard.json
    ├── selection_decisions.json
    ├── quick_timeline.json
    └── final.mp4
```

Исходные файлы никогда не изменяются и не удаляются.

## Проверки

```powershell
python -m pytest
python -m ruff check .
python -m mypy
python -m compileall -q src tests
```

## Документация

- [Установка и использование](docs/installation-and-usage.md)
- [Архитектура](docs/architecture.md)
- [Техническое задание](docs/TECHNICAL_SPECIFICATION.md)
- [Roadmap](docs/roadmap.md)
- [Правила разработки](AGENTS.md)

## License

See [LICENSE](LICENSE).
