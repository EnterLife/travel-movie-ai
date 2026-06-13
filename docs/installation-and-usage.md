# Установка и использование TravelMovieAI

Это руководство описывает установку текущей версии TravelMovieAI на Windows,
настройку окружения, сканирование медиатеки и диагностику типовых проблем.

## 1. Что работает сейчас

Текущая версия включает Media Scan, Scene Detection, Quick Montage и первый
локальный Semantic Montage. Media Scan:

- рекурсивно находит поддерживаемые видео, фотографии и аудиофайлы;
- определяет тип файла по расширению;
- получает длительность, разрешение, FPS, дату создания и GPS через FFprobe;
- дополняет данные фотографий размером изображения и EXIF GPS через Pillow;
- сохраняет индекс проекта в SQLite;
- создаёт JSON-снимок результатов;
- повторно использует метаданные неизменённых файлов;
- не изменяет исходные медиафайлы.

Команда `travelmovieai analyze` и локальный веб-интерфейс полностью работают.

После анализа кнопка `Собрать фильм` в быстром режиме:

- располагает пригодные материалы по времени съёмки;
- берёт короткие фрагменты длинных видео;
- добавляет фотографии;
- нормализует видео и звук;
- создаёт `quick_timeline.json`;
- рендерит и предлагает скачать `final.mp4`.

Семантический режим дополнительно:

- делит видео на сцены через PySceneDetect или равномерный fallback;
- извлекает contact sheet из начала, середины и конца сцены;
- оценивает резкость, яркость, контраст, насыщенность и цветность через OpenCV;
- анализирует кадр Qwen/VLM через LM Studio либо локальной Florence-2;
- позволяет выбрать backend и модель в интерфейсе;
- валидирует caption, detailed description, people, activity, location,
  emotion, landmarks, tags и score factors;
- объединяет связанные сцены в события и сохраняет их в SQLite;
- создаёт мультимодальные описания для следующего Story Builder;
- выбирает сцены с учётом качества, событий, landmarks и разнообразия;
- создаёт локальную музыку по визуальным метрикам или использует
  библиотеку/ручной файл;
- добавляет переходы и ducking;
- использует NVIDIA NVENC с автоматическим CPU fallback;
- кэширует результаты повторного AI-анализа.

Команда `create` создаёт готовый MP4. Команды `storyboard`, advanced `render` и
`report` пока остаются контрактами будущих стадий.

## 2. Системные требования

Обязательно:

- Windows 10 или Windows 11;
- 64-разрядный Python 3.12 или новее;
- `pip`;
- FFmpeg, включающий `ffmpeg.exe` и `ffprobe.exe`;
- свободное место для базы, кадров и будущих промежуточных файлов.

Рекомендуется:

- хранить workspace на быстром SSD;
- использовать отдельный workspace для каждой поездки;
- не размещать workspace внутри папки, синхронизируемой нестабильным облачным
  клиентом.

Для Media Scan и быстрого монтажа не требуются CUDA, GPU или LM Studio.
Qwen-режим требует запущенный локальный LM Studio с vision-моделью.
Florence-2 требует заранее загруженные в локальный Hugging Face cache веса.
CUDA не обязательна: AI и rendering сохраняют CPU fallback.

При первом запуске BAT-скрипту нужен интернет для установки Python-зависимостей.

## 3. Установка Python

Установите Python 3.12 или более новую совместимую версию с официального
дистрибутива Python. При установке включите добавление Python в `PATH`.

После установки откройте новое окно PowerShell и проверьте:

```powershell
python --version
python -m pip --version
```

Ожидается версия Python не ниже `3.12`.

Если команда `python` открывает Microsoft Store или не находится, проверьте
параметр `App execution aliases` Windows и порядок каталогов Python в `PATH`.

## 4. Установка FFmpeg

TravelMovieAI вызывает FFmpeg как внешний процесс. Нужны оба файла:

```text
ffmpeg.exe
ffprobe.exe
```

Общий порядок:

1. Скачайте Windows-сборку FFmpeg.
2. Распакуйте архив, например в `C:\Tools\ffmpeg`.
3. Убедитесь, что существует каталог `C:\Tools\ffmpeg\bin`.
4. Добавьте `C:\Tools\ffmpeg\bin` в пользовательскую переменную `PATH`.
5. Закройте и заново откройте PowerShell.
6. Проверьте установку.

```powershell
ffmpeg -version
ffprobe -version
```

Если изменять `PATH` нельзя, укажите полные пути в `.env`:

```dotenv
TRAVELMOVIEAI_FFMPEG_BINARY=C:\Tools\ffmpeg\bin\ffmpeg.exe
TRAVELMOVIEAI_FFPROBE_BINARY=C:\Tools\ffmpeg\bin\ffprobe.exe
```

Пути в `.env` не нужно заключать в кавычки.

## 5. Получение проекта

Если репозиторий уже находится на компьютере:

```powershell
Set-Location C:\Users\bdo\travel-movie-ai
```

Если проект клонируется из Git:

```powershell
git clone https://github.com/EnterLife/travel-movie-ai.git
Set-Location .\travel-movie-ai
```

Все следующие команды выполняются из корня репозитория, где расположен
`pyproject.toml`.

## 6. Виртуальное окружение

Создайте и активируйте окружение:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

После активации в начале строки PowerShell обычно появляется `(.venv)`.

Если PowerShell запрещает запуск `Activate.ps1`, можно не менять execution
policy, а запускать Python окружения напрямую:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m travelmovieai --help
```

## 7. Быстрый запуск веб-интерфейса

Рекомендуемый способ:

```powershell
.\scripts\run_web.bat
```

Скрипт также можно запустить двойным кликом из Проводника.

При первом запуске `run_web.bat`:

1. Переходит в корень репозитория.
2. Создаёт `.venv`, если окружение отсутствует.
3. Устанавливает базовые зависимости, OpenCV и PySceneDetect.
4. Запускает `main.py`.
5. Открывает `http://127.0.0.1:8000` в браузере.

Последующие запуски не переустанавливают зависимости и стартуют быстрее.

В интерфейсе:

1. Вставьте полный путь к папке с материалами.
2. При необходимости укажите workspace.
3. Нажмите `Запустить анализ`.
4. Дождитесь статуса `Готово`.
5. Просмотрите статистику и таблицу файлов.
6. Выберите backend `Qwen / LM Studio` или `Florence-2`.
7. Выберите vision-модель.
8. Настройте стиль, длительность, переходы и `Auto/CUDA/CPU`.
9. Оставьте semantic и OpenCV analysis включёнными для лучшего отбора.
10. Выберите музыкальный режим: `AI Auto`, генерация, библиотека, ручной файл
   или без музыки.
11. Нажмите `Запустить AI-монтаж`.
12. Просмотрите или скачайте MP4.

Остановить сервер можно сочетанием `Ctrl+C` в его консоли или закрытием окна
консоли.

Аргументы передаются из BAT в `main.py`:

```powershell
.\scripts\run_web.bat --port 8080
.\scripts\run_web.bat --no-browser
```

Если браузер не открылся автоматически, перейдите вручную:

```text
http://127.0.0.1:8000
```

Браузер не может безопасно передать серверу произвольную системную папку через
обычный HTML file picker, поэтому путь к медиатеке вводится или вставляется
полностью.

## 8. Ручная установка TravelMovieAI

Для текущего Media Scan достаточно базовой установки:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

Флаг `-e` устанавливает проект в editable-режиме: изменения в `src/` сразу
используются без переустановки пакета.

Проверьте CLI:

```powershell
travelmovieai --help
```

Альтернативный запуск:

```powershell
python -m travelmovieai --help
```

Если `travelmovieai` не найден, но модуль запускается через
`python -m travelmovieai`, проверьте активацию виртуального окружения.

Ручной запуск веб-сервера:

```powershell
python main.py
```

Дополнительные варианты:

```powershell
python main.py --host 127.0.0.1 --port 8000
python main.py --no-browser
travelmovieai-web --port 8000
```

По умолчанию сервер доступен только на текущем компьютере. Не запускайте его с
`--host 0.0.0.0` в недоверенной сети: авторизация пока отсутствует.

Swagger UI для HTTP API:

```text
http://127.0.0.1:8000/api/docs
```

## 9. Установка для разработки

Установите тесты, линтер, форматтер и mypy:

```powershell
python -m pip install -e ".[dev]"
```

Для PySceneDetect и OpenCV установите video group:

```powershell
python -m pip install -e ".[video]"
```

Без этой группы semantic montage продолжит работать, но будет использовать
равномерные сцены. Зависимости следующих стадий устанавливаются отдельно:

```powershell
python -m pip install -e ".[speech]"
python -m pip install -e ".[vision]"
python -m pip install -e ".[embeddings]"
```

Все группы:

```powershell
python -m pip install -e ".[all,dev]"
```

AI-группы могут занимать много места. Для Media Scan они не нужны. Модели не
входят в пакет и не скачиваются при базовой установке.

### Настройка LM Studio

1. Установите LM Studio.
2. Загрузите vision-language модель, например Qwen2.5-VL.
3. Откройте модель и запустите Local Server.
4. Проверьте OpenAI-compatible endpoint `http://localhost:1234/v1`.
5. Укажите точный identifier загруженной модели в `.env`.

```dotenv
TRAVELMOVIEAI_LM_STUDIO_URL=http://localhost:1234/v1
TRAVELMOVIEAI_VISION_MODEL=auto
TRAVELMOVIEAI_VISION_TIMEOUT_SECONDS=120
```

Значение `auto` выбирает первую вероятно мультимодальную модель из `/v1/models`.
Можно указать точный identifier модели LM Studio. Если локальный сервер или
модель недоступны, semantic job завершится явной ошибкой. Автоматического
перехода в облако нет.

Интерфейс запрашивает `/v1/models` и показывает фактически доступные модели.
Маркеры `omni`, `vision`, `VL` и `Gemma` используются только как подсказка
мультимодальности. Окончательная совместимость проверяется image-запросом.

### Настройка Florence-2

Florence-2 запускается напрямую через `transformers` и PyTorch. Установите
optional-группу и совместимую сборку PyTorch:

```powershell
python -m pip install -e ".[vision]"
```

Заранее поместите `microsoft/Florence-2-base` или
`microsoft/Florence-2-large` в локальный Hugging Face cache либо укажите путь
к локальному каталогу модели через `TRAVELMOVIEAI_VISION_MODEL`. Приложение
использует `local_files_only` и не начинает скрытую загрузку модели во время
анализа. `TRAVELMOVIEAI_DEVICE=auto` выбирает CUDA при доступности PyTorch,
иначе CPU.

### CUDA и NVIDIA

Приложение проверяет `nvidia-smi`, `h264_nvenc`, CUDA devices OpenCV и
CUDA-доступность PyTorch. Режим `Auto` использует `h264_nvenc` и повторяет
рендер через `libx264` при ошибке. Режим `CUDA` требует рабочий NVENC, а
`CPU` всегда использует `libx264`.

LM Studio самостоятельно управляет CUDA offload модели. Выбор CUDA в
TravelMovieAI относится к FFmpeg-рендерингу и не меняет настройки модели в
LM Studio.

## 10. Настройка `.env`

Создайте локальный файл конфигурации:

```powershell
Copy-Item .env.example .env
```

Файл `.env` читается из текущего рабочего каталога. При обычной разработке
запускайте CLI из корня репозитория. При запуске из другого каталога задайте
нужные `TRAVELMOVIEAI_*` переменные в окружении или разместите там подходящий
`.env`.

Полный пример:

```dotenv
TRAVELMOVIEAI_WORKSPACE=./workspace
TRAVELMOVIEAI_DATABASE_FILENAME=project.db
TRAVELMOVIEAI_FFMPEG_BINARY=ffmpeg
TRAVELMOVIEAI_FFPROBE_BINARY=ffprobe
TRAVELMOVIEAI_LM_STUDIO_URL=http://localhost:1234/v1
TRAVELMOVIEAI_VISION_MODEL=auto
TRAVELMOVIEAI_VISION_TIMEOUT_SECONDS=120
TRAVELMOVIEAI_VISION_PROVIDER=qwen
TRAVELMOVIEAI_MUSIC_LIBRARY=./assets/music
TRAVELMOVIEAI_GENERATED_MUSIC_FILENAME=generated_soundtrack.wav
TRAVELMOVIEAI_WHISPER_MODEL=medium
TRAVELMOVIEAI_DEVICE=auto
TRAVELMOVIEAI_CLOUD_ENABLED=false
TRAVELMOVIEAI_BATCH_SIZE=8
TRAVELMOVIEAI_WORKERS=4
TRAVELMOVIEAI_WEB_HOST=127.0.0.1
TRAVELMOVIEAI_WEB_PORT=8000
TRAVELMOVIEAI_WEB_HISTORY_LIMIT=100
```

### Параметры Media Scan

`TRAVELMOVIEAI_WORKSPACE`

: Родительский каталог workspace по умолчанию. Относительный путь считается от
  текущего рабочего каталога процесса.

`TRAVELMOVIEAI_DATABASE_FILENAME`

: Имя SQLite-файла внутри workspace без каталогов.

`TRAVELMOVIEAI_FFMPEG_BINARY`

: Имя или полный путь к `ffmpeg.exe`. Настройка понадобится следующим
  видеоэтапам.

`TRAVELMOVIEAI_FFPROBE_BINARY`

: Имя или полный путь к `ffprobe.exe`.

### Параметры semantic montage и будущих стадий

- `TRAVELMOVIEAI_LM_STUDIO_URL`: OpenAI-совместимый API LM Studio.
- `TRAVELMOVIEAI_VISION_MODEL`: `auto` или identifier vision-модели.
- `TRAVELMOVIEAI_VISION_TIMEOUT_SECONDS`: timeout одного AI-запроса.
- `TRAVELMOVIEAI_VISION_PROVIDER`: `qwen` или `florence`.
- `TRAVELMOVIEAI_MUSIC_LIBRARY`: папка локальной музыкальной библиотеки.
- `TRAVELMOVIEAI_GENERATED_MUSIC_FILENAME`: имя создаваемого WAV-файла.
- `TRAVELMOVIEAI_WHISPER_MODEL`: `medium` или `large-v3`.
- `TRAVELMOVIEAI_DEVICE`: `auto`, `cuda`, `directml` или `cpu`.
- `TRAVELMOVIEAI_CLOUD_ENABLED`: разрешение будущих облачных интеграций.
- `TRAVELMOVIEAI_BATCH_SIZE`: размер пакета, целое число не меньше 1.
- `TRAVELMOVIEAI_WORKERS`: число обработчиков, целое число не меньше 1.
- `TRAVELMOVIEAI_WEB_HOST`: адрес локального сервера.
- `TRAVELMOVIEAI_WEB_PORT`: порт от 1 до 65535.
- `TRAVELMOVIEAI_WEB_HISTORY_LIMIT`: число записей истории от 1 до 1000.

`.env` не должен попадать в Git.

## 11. Подготовка исходной папки

Пример:

```text
D:\Vacation\Japan2026\
├── Day 01\
│   ├── airport.mp4
│   └── hotel.jpg
├── Day 02\
│   ├── city walk.mov
│   └── voice note.m4a
└── notes.txt
```

Сканирование рекурсивное. Вложенные каталоги сохраняются в `relative_path`.
Неподдерживаемые файлы, например `notes.txt`, игнорируются.

Поддерживаемые расширения:

```text
Видео:       .mp4 .mov .avi .mkv .m4v
Фотографии:  .jpg .jpeg .png .heic
Аудио:       .mp3 .wav .flac .m4a
```

Расширения регистронезависимы. Пути могут содержать пробелы и кириллицу.
Полнота HEIC-метаданных зависит от кодеков FFmpeg и Pillow.

## 12. Первый запуск через CLI

Рекомендуемый вариант с явным workspace:

```powershell
travelmovieai analyze `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026"
```

Короткая форма:

```powershell
travelmovieai analyze `
  -i "D:\Vacation\Japan2026" `
  -w "D:\TravelMovieAI\Japan2026"
```

PowerShell использует обратный апостроф в конце строки для продолжения команды.
Однострочный вариант:

```powershell
travelmovieai analyze --input "D:\Vacation\Japan2026" --workspace "D:\TravelMovieAI\Japan2026"
```

Пример результата:

```text
Media scan found 245 file(s): 245 inspected, 0 cached, 2 with errors.
```

- `found`: обнаруженные поддерживаемые файлы;
- `inspected`: файлы, для которых заново запущен FFprobe;
- `cached`: файлы с переиспользованными метаданными;
- `with errors`: файлы с неполными метаданными.

Ошибка отдельного файла не останавливает проект. Подробность сохраняется в
`scan_error` соответствующего элемента `analysis.json`.

## 13. Workspace по умолчанию

Если `--workspace` не задан, используется:

```text
<current-directory>\workspace\<input-folder-name>
```

Например, запуск из:

```text
C:\Users\bdo\travel-movie-ai
```

для `D:\Vacation\Japan2026` создаст:

```text
C:\Users\bdo\travel-movie-ai\workspace\Japan2026
```

Для постоянных проектов лучше задавать `--workspace` явно.

Workspace можно разместить внутри input: сканер исключит его из поиска. Но
отдельный каталог обычно удобнее для обслуживания и резервного копирования.

## 14. Результаты

После запуска создаётся:

```text
<workspace>\
├── project.db
├── frames\
├── cache\
└── artifacts\
    ├── analysis.json
    ├── scenes.json
    ├── vision_analysis.json
    ├── quality_analysis.json
    ├── music_plan.json
    ├── generated_soundtrack.wav
    ├── quick_timeline.json
    └── final.mp4
```

Для web-интерфейса дополнительно создаётся общая история:

```text
<default-workspace>\.web\jobs.json
```

Она позволяет показывать последние задания после перезапуска сервера.

### `project.db`

SQLite-база проекта. Таблица `media_assets` содержит:

- UUID;
- абсолютный и относительный пути;
- тип и расширение;
- размер и время изменения;
- дату создания;
- длительность;
- ширину, высоту и FPS;
- GPS;
- краткие FFprobe-метаданные;
- ошибку сканирования;
- время последней синхронизации.

SQLite работает в режиме WAL. Во время работы могут появляться
`project.db-wal` и `project.db-shm`.

Таблица `scenes` хранит временные границы, contact sheet, vision
metadata и оценки. При удалении asset связанные сцены удаляются каскадно.

Таблица `events` хранит группы сцен, временные диапазоны, location/activity,
landmarks, importance и confidence.

### `artifacts/analysis.json`

Читаемый JSON-снимок индекса:

```json
{
  "input_path": "D:\\Vacation\\Japan2026",
  "scanned_at": "2026-06-13T12:00:00Z",
  "assets": [],
  "discovered_count": 245,
  "probed_count": 245,
  "cached_count": 0,
  "error_count": 2
}
```

JSON записывается атомарно. Каталоги `frames` и `cache` создаются заранее для
следующих стадий.

### `artifacts/quick_timeline.json`

План текущего монтажа:

- порядок материалов;
- source path и media type;
- начало и длительность видеофрагмента;
- наличие аудиодорожки;
- параметры разрешения и FPS.
- режим отбора и semantic score;
- переходы и выбранный локальный soundtrack.

### `artifacts/scenes.json`

Границы сцен, detector/cache metadata и пути contact sheets. Если
PySceneDetect не установлен или не смог обработать файл, используются
ограниченные по длительности равномерные сцены.

### `artifacts/vision_analysis.json`

Структурированный результат локальной vision-модели: caption, detailed
description, location, activity, emotion, people groups/count, landmarks,
score factors, vision score, story relevance, tags, provider, model и prompt
version.

### `artifacts/quality_analysis.json`

Содержит brightness, contrast, sharpness, saturation, colorfulness, итоговый
quality score и backend (`opencv` или `pillow`). Метрики участвуют в ranking.

### `artifacts/scene_descriptions.json`

Объединяет доступные Vision AI, Whisper, OpenCV и audio данные в проверенное
описание сцены. На текущем этапе speech/audio поля добавляются только если уже
присутствуют в metadata.

### `artifacts/events.json`

Содержит temporal/semantic группы сцен, заголовок, summary, location, activity,
landmarks, importance и confidence. Базовый clustering использует время съёмки,
совпадение location/activity, landmarks и принадлежность одному asset.

### `artifacts/music_plan.json`

Фиксирует музыкальный режим, профиль, BPM, источник и объяснение решения.
`AI Auto` выбирает характер по стилю, эмоциям и OpenCV-метрикам, затем создаёт
детерминированный `generated_soundtrack.wav`.

### `artifacts/final.mp4`

Готовый H.264/AAC фильм. Веб-интерфейс позволяет воспроизвести его и скачать.

Текущий semantic montage ещё не удаляет дубли и не строит полноценный
storyboard. Transcript и audio context начнут влиять на описания после
реализации соответствующих стадий.

## 15. Повторный запуск и кэш

Повторите ту же команду с тем же workspace. Для неизменённых файлов:

```text
Media scan found 245 file(s): 0 inspected, 245 cached, 2 with errors.
```

Кэш действителен при совпадении:

- нормализованного абсолютного пути;
- размера;
- времени изменения.

Изменённый файл анализируется заново. Новый файл добавляется. Удалённый исходный
файл удаляется из SQLite-индекса при следующем успешном сканировании.

Если содержимое заменено с сохранением размера и timestamp, кэш может считать
его прежним. Для полного пересканирования удалите только workspace.

## 16. Сброс проекта

Перед удалением убедитесь, что выбран workspace, а не исходная медиатека:

```powershell
Remove-Item -LiteralPath "D:\TravelMovieAI\Japan2026" -Recurse
```

После этого повторите `travelmovieai analyze`. Исходные файлы приложение не
изменяет.

## 17. Команды CLI

Список:

```powershell
travelmovieai --help
```

Рабочая команда:

```powershell
travelmovieai analyze --input <directory> [--workspace <directory>]
```

Пока зарезервированные команды:

```powershell
travelmovieai storyboard --input <directory>
travelmovieai render --input <directory> --output <file>
travelmovieai report --input <directory>
```

Они пока доходят до заглушек. Для создания готового фильма используйте
`travelmovieai create`.

Рабочая команда создания quick montage:

```powershell
travelmovieai create `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026" `
  --output "D:\Movies\Japan2026.mp4"
```

Semantic montage через LM Studio:

```powershell
travelmovieai create `
  --input "D:\Vacation\Japan2026" `
  --workspace "D:\TravelMovieAI\Japan2026" `
  --output "D:\Movies\Japan2026.mp4" `
  --semantic `
  --style cinematic
```

Без флага `--semantic` используется быстрый режим.

Доступные story styles:

```text
cinematic documentary family vlog adventure romantic
```

## 18. Проверка установки

```powershell
python --version
ffmpeg -version
ffprobe -version
travelmovieai --help
```

Для разработчика:

```powershell
python -m pytest
python -m pytest --cov=travelmovieai
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m compileall -q src tests
```

## 19. Типовые проблемы

### `python` не найден

Переустановите Python с добавлением в `PATH` или используйте полный путь к
`python.exe`. После изменения `PATH` откройте новое окно PowerShell.

### `travelmovieai` не найден

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m travelmovieai --help
```

### `FFprobe executable was not found`

```powershell
Get-Command ffprobe
ffprobe -version
```

Добавьте FFmpeg `bin` в `PATH` или задайте
`TRAVELMOVIEAI_FFPROBE_BINARY`.

### `LM Studio недоступен`

Проверьте, что в LM Studio:

- загружена именно vision-language модель;
- запущен Local Server;
- endpoint совпадает с `TRAVELMOVIEAI_LM_STUDIO_URL`;
- identifier модели совпадает с выбранным в интерфейсе или `.env`.

Быстрая проверка:

```powershell
Invoke-RestMethod http://localhost:1234/v1/models
```

Чтобы собрать фильм без AI, отключите `Семантический AI-отбор` в web UI или
используйте CLI без `--semantic`.

### Найдено `0 file(s)`

Проверьте:

- путь `--input`;
- наличие файлов во вложенных папках;
- поддерживаемые расширения;
- что передан каталог, а не файл;
- что медиа не находятся только внутри исключённого workspace.

### Есть `with errors`

Откройте `<workspace>\artifacts\analysis.json` и найдите `scan_error`.

Частые причины:

- повреждённый файл;
- неподдерживаемый контейнер или кодек;
- файл заблокирован;
- недостаточно прав на чтение;
- HEIC-кодек недоступен.

### Данные не обновились

Убедитесь, что используется тот же workspace. Для полной перестройки удалите
workspace и повторите запуск.

### SQLite занят

Закройте редакторы базы и другой процесс TravelMovieAI. Не запускайте два
сканирования одновременно в один workspace.

Web API дополнительно отклоняет второй активный запрос к тому же workspace с
HTTP 409.

### Пути с пробелами

Всегда заключайте их в двойные кавычки:

```powershell
travelmovieai analyze --input "D:\My Trips\Japan 2026"
```

### `Activate.ps1` запрещён

Запускайте интерпретатор окружения напрямую:

```powershell
.\.venv\Scripts\python.exe -m travelmovieai --help
```

### Порт `8000` уже занят

Запустите сервер на другом порту:

```powershell
.\scripts\run_web.bat --port 8080
```

### Браузер не открылся

Откройте `http://127.0.0.1:8000` вручную. Проверить сервер можно командой:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

### BAT не может установить зависимости

Проверьте интернет, доступность PyPI и Python:

```powershell
python --version
.\.venv\Scripts\python.exe -m pip install -e .
```

## 20. Конфиденциальность

Media Scan работает локально:

- медиа не отправляются в облако;
- телеметрия отсутствует;
- SQLite и JSON остаются в workspace;
- GPS и локальные пути являются приватными данными;
- `.env`, workspace и исходные медиа не следует публиковать.

Cloud mode предназначен для будущих интеграций и по умолчанию выключен.

Веб-сервер по умолчанию слушает `127.0.0.1`. Он не использует облако, но API
позволяет передавать локальные пути, поэтому не публикуйте сервер в интернет.

## 21. Обновление проекта

```powershell
git pull
python -m pip install -e .
python -m pytest
```

При изменении optional dependencies:

```powershell
python -m pip install -e ".[all,dev]"
```

Не удаляйте workspace без необходимости: совместимые версии смогут
переиспользовать собранные метаданные.
