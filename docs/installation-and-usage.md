# Установка и использование

Практическое руководство для Windows. Архитектура описана отдельно в
[architecture.md](architecture.md), продуктовые требования — в
[TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md).

## 1. Требования

- Windows 10/11 x64;
- Python 3.12+;
- FFmpeg и FFprobe;
- свободное место на SSD для кадров, кэша и промежуточных видео.

Проверка:

```powershell
python --version
ffmpeg -version
ffprobe -version
```

Если FFmpeg не находится, добавьте его `bin` в `PATH` или задайте полные пути:

```dotenv
TRAVELMOVIEAI_FFMPEG_BINARY=C:\Tools\ffmpeg\bin\ffmpeg.exe
TRAVELMOVIEAI_FFPROBE_BINARY=C:\Tools\ffmpeg\bin\ffprobe.exe
```

## 2. Быстрый запуск

Из корня проекта:

```powershell
.\scripts\run_web.bat
```

Скрипт:

1. создаёт `.venv`, если она отсутствует;
2. устанавливает приложение, OpenCV и PySceneDetect;
3. запускает `main.py`;
4. открывает `http://127.0.0.1:8000`.

Дополнительные аргументы:

```powershell
.\scripts\run_web.bat --port 8080
.\scripts\run_web.bat --no-browser
```

Остановка сервера: `Ctrl+C`.

## 3. Ручная установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[video]"
python main.py
```

Если PowerShell запрещает `Activate.ps1`, вызывайте интерпретатор напрямую:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[video]"
.\.venv\Scripts\python.exe main.py
```

Для разработки:

```powershell
python -m pip install -e ".[video,dev]"
```

Опциональные группы:

```powershell
python -m pip install -e ".[speech]"
python -m pip install -e ".[vision]"
python -m pip install -e ".[embeddings]"
python -m pip install -e ".[all,dev]"
```

## 4. Настройка AI

### Qwen/VLM через LM Studio

1. Установите LM Studio.
2. Загрузите мультимодальную модель, например Qwen2.5-VL.
3. Запустите Local Server.
4. Проверьте endpoint:

```powershell
Invoke-RestMethod http://localhost:1234/v1/models
```

Минимальная конфигурация:

```dotenv
TRAVELMOVIEAI_LM_STUDIO_URL=http://localhost:1234/v1
TRAVELMOVIEAI_VISION_PROVIDER=qwen
TRAVELMOVIEAI_VISION_MODEL=auto
TRAVELMOVIEAI_VISION_TIMEOUT_SECONDS=120
```

LM Studio самостоятельно управляет GPU offload. Если image inference не
успевает завершиться, увеличьте timeout или выберите меньшую модель.

### Florence-2

Florence-2 запускается напрямую через Transformers/PyTorch:

```powershell
python -m pip install -e ".[vision]"
```

Веса должны заранее находиться в локальном Hugging Face cache или локальном
каталоге. Приложение использует `local_files_only` и не начинает скрытую
загрузку во время анализа.

```dotenv
TRAVELMOVIEAI_VISION_PROVIDER=florence
TRAVELMOVIEAI_VISION_MODEL=microsoft/Florence-2-large
TRAVELMOVIEAI_DEVICE=auto
```

### Faster Whisper

```powershell
python -m pip install -e ".[speech]"
```

Распознавание речи включается отдельно в веб-интерфейсе. Оно увеличивает время
анализа, но добавляет transcript, language и confidence для каждой сцены.

## 5. Рабочий сценарий

1. Введите полный путь к папке с видео и фотографиями.
2. Укажите отдельный workspace.
3. Нажмите `Запустить анализ`.
4. Выберите Vision backend и модель.
5. Оставьте semantic/OpenCV анализ включёнными.
6. При необходимости включите Faster Whisper.
7. Включите `Быстрый preview`.
8. Настройте длительность, переходы, музыку и устройство рендера.
9. Запустите AI-монтаж.
10. Следите за текущим этапом, процентом, ETA и журналом обработки.
11. В галерее задайте сценам `Авто`, `Обязательно` или `Исключить`.
12. Повторите монтаж: неизменённый AI-анализ возьмётся из кэша.
13. Отключите preview и создайте финальный MP4.

Quick mode работает без AI: он располагает материалы по времени и берёт
короткие фрагменты видео. Semantic mode дополнительно выполняет Scene
Detection, Quality/Vision/Speech Analysis, поиск дублей, Event Detection и
Story Builder.

## 6. CLI

Media Scan:

```powershell
travelmovieai analyze `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026"
```

Quick montage:

```powershell
travelmovieai create `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026" `
  --output "D:\Movies\Japan2026.mp4"
```

Semantic montage:

```powershell
travelmovieai create `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026" `
  --output "D:\Movies\Japan2026.mp4" `
  --semantic `
  --style cinematic
```

Стили: `cinematic`, `documentary`, `family`, `vlog`, `adventure`, `romantic`.

## 7. Конфигурация

Создайте локальный `.env`:

```powershell
Copy-Item .env.example .env
```

Основные переменные:

| Переменная | Назначение |
| --- | --- |
| `TRAVELMOVIEAI_WORKSPACE` | workspace по умолчанию |
| `TRAVELMOVIEAI_FFMPEG_BINARY` | путь к FFmpeg |
| `TRAVELMOVIEAI_FFPROBE_BINARY` | путь к FFprobe |
| `TRAVELMOVIEAI_LM_STUDIO_URL` | API LM Studio |
| `TRAVELMOVIEAI_VISION_PROVIDER` | `qwen` или `florence` |
| `TRAVELMOVIEAI_VISION_MODEL` | модель или `auto` |
| `TRAVELMOVIEAI_VISION_TIMEOUT_SECONDS` | timeout Vision-запроса |
| `TRAVELMOVIEAI_WHISPER_MODEL` | `medium` или `large-v3` |
| `TRAVELMOVIEAI_DEVICE` | `auto`, `cuda`, `directml`, `cpu` |
| `TRAVELMOVIEAI_WORKERS` | число параллельных задач; `0` — авто |
| `TRAVELMOVIEAI_MUSIC_LIBRARY` | локальная библиотека музыки |
| `TRAVELMOVIEAI_WEB_HOST` | адрес сервера, обычно `127.0.0.1` |
| `TRAVELMOVIEAI_WEB_PORT` | порт, обычно `8000` |

Не коммитьте `.env`.

### Автоматическое использование ресурсов

При первом монтаже приложение определяет логические CPU, объём RAM, NVIDIA GPU
и поддержку NVENC. На основе профиля оно параллельно извлекает кадры, выполняет
OpenCV-анализ и готовит сегменты FFmpeg. Итоговый профиль показан в блоке
прогресса и возвращается из `/api/capabilities`.

Оставляйте `WORKERS=0` для обычной работы. Ручное значение полезно, если нужно
сохранить ресурсы для других программ или ограничить нагрев компьютера.

## 8. Результаты

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
    ├── speech_analysis.json
    ├── duplicates.json
    ├── scene_descriptions.json
    ├── events.json
    ├── storyboard.json
    ├── selection_decisions.json
    ├── music_plan.json
    ├── quick_timeline.json
    ├── preview.mp4
    └── final.mp4
```

Ключевые файлы:

- `project.db` — индекс медиа, сцен, событий и ручных решений;
- `quality_analysis.json` — качество, движение, тряска, шум и причины брака;
- `vision_analysis.json` — структурированное понимание сцен;
- `duplicates.json` — группы похожих сцен и выбранный keeper;
- `selection_decisions.json` — причины выбора или отклонения каждой сцены;
- `quick_timeline.json` — декларативный план монтажа;
- `preview.mp4` / `final.mp4` — готовые H.264/AAC-файлы.

RGB PNG contact sheets используются вместо MJPEG, чтобы корректно обрабатывать
limited-range YUV из DJI и других камер в новых версиях FFmpeg.

## 9. Кэш и сброс

Повторный запуск переиспользует:

- FFprobe metadata неизменённых файлов;
- границы сцен;
- contact sheets;
- Vision/Speech результаты при совпадении модели и cache key.

Для полной перестройки удалите только workspace. Перед удалением внимательно
проверьте путь:

```powershell
Remove-Item -LiteralPath "D:\TravelMovieAI\Japan2026" -Recurse
```

Исходная медиатека приложением не изменяется.

## 10. Диагностика

### LM Studio недоступен или модель зависает

- запустите Local Server;
- проверьте `/v1/models`;
- убедитесь, что модель поддерживает изображения;
- проверьте GPU offload;
- увеличьте `TRAVELMOVIEAI_VISION_TIMEOUT_SECONDS`.

### FFmpeg/FFprobe не найден

```powershell
Get-Command ffmpeg
Get-Command ffprobe
```

Добавьте FFmpeg `bin` в `PATH` или задайте полные пути в `.env`.

### Найдено 0 файлов

Проверьте путь, вложенные каталоги и расширения:

```text
Видео: .mp4 .mov .avi .mkv .m4v
Фото:  .jpg .jpeg .png .heic
Аудио: .mp3 .wav .flac .m4a
```

### SQLite занят

Закройте редакторы базы и другие процессы TravelMovieAI. Один workspace не
должен одновременно обрабатываться несколькими заданиями.

### Порт занят

```powershell
.\scripts\run_web.bat --port 8080
```

### Проверка проекта

```powershell
python -m pytest
python -m ruff check .
python -m mypy
python -m compileall -q src tests
```

## 11. Приватность

- сервер по умолчанию слушает только `127.0.0.1`;
- raw media и derived frames остаются локальными;
- cloud mode выключен;
- пути, GPS, лица, голоса и transcripts считаются приватными данными;
- не публикуйте workspace, `.env`, базы, кадры и итоговые фильмы.
