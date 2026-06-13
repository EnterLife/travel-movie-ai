# Установка и использование TravelMovieAI

Это руководство описывает установку текущей версии TravelMovieAI на Windows,
настройку окружения, сканирование медиатеки и диагностику типовых проблем.

## 1. Что работает сейчас

Текущая рабочая стадия называется Media Scan. Она:

- рекурсивно находит поддерживаемые видео, фотографии и аудиофайлы;
- определяет тип файла по расширению;
- получает длительность, разрешение, FPS, дату создания и GPS через FFprobe;
- дополняет данные фотографий размером изображения и EXIF GPS через Pillow;
- сохраняет индекс проекта в SQLite;
- создаёт JSON-снимок результатов;
- повторно использует метаданные неизменённых файлов;
- не изменяет исходные медиафайлы.

Команда `travelmovieai analyze` полностью работает.

Команды `create`, `storyboard`, `render` и `report` уже присутствуют в CLI как
контракты будущих стадий, но пока не создают фильм, storyboard, HTML-отчёт или
готовый MP4.

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

Для Media Scan не требуются CUDA, GPU, Whisper, Qwen, Florence, LM Studio или
доступ в интернет после установки зависимостей.

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

## 7. Установка TravelMovieAI

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

## 8. Установка для разработки

Установите тесты, линтер, форматтер и mypy:

```powershell
python -m pip install -e ".[dev]"
```

Зависимости будущих стадий устанавливаются отдельно:

```powershell
python -m pip install -e ".[video]"
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

## 9. Настройка `.env`

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
TRAVELMOVIEAI_VISION_PROVIDER=qwen
TRAVELMOVIEAI_WHISPER_MODEL=medium
TRAVELMOVIEAI_DEVICE=auto
TRAVELMOVIEAI_CLOUD_ENABLED=false
TRAVELMOVIEAI_BATCH_SIZE=8
TRAVELMOVIEAI_WORKERS=4
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

### Параметры будущих стадий

- `TRAVELMOVIEAI_LM_STUDIO_URL`: OpenAI-совместимый API LM Studio.
- `TRAVELMOVIEAI_VISION_PROVIDER`: `qwen` или `florence`.
- `TRAVELMOVIEAI_WHISPER_MODEL`: `medium` или `large-v3`.
- `TRAVELMOVIEAI_DEVICE`: `auto`, `cuda`, `directml` или `cpu`.
- `TRAVELMOVIEAI_CLOUD_ENABLED`: разрешение будущих облачных интеграций.
- `TRAVELMOVIEAI_BATCH_SIZE`: размер пакета, целое число не меньше 1.
- `TRAVELMOVIEAI_WORKERS`: число обработчиков, целое число не меньше 1.

`.env` не должен попадать в Git.

## 10. Подготовка исходной папки

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

## 11. Первый запуск

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

## 12. Workspace по умолчанию

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

## 13. Результаты

После запуска создаётся:

```text
<workspace>\
├── project.db
├── frames\
├── cache\
└── artifacts\
    └── analysis.json
```

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

## 14. Повторный запуск и кэш

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

## 15. Сброс проекта

Перед удалением убедитесь, что выбран workspace, а не исходная медиатека:

```powershell
Remove-Item -LiteralPath "D:\TravelMovieAI\Japan2026" -Recurse
```

После этого повторите `travelmovieai analyze`. Исходные файлы приложение не
изменяет.

## 16. Команды CLI

Список:

```powershell
travelmovieai --help
```

Рабочая команда:

```powershell
travelmovieai analyze --input <directory> [--workspace <directory>]
```

Зарезервированные команды:

```powershell
travelmovieai create --input <directory> --output <file>
travelmovieai storyboard --input <directory>
travelmovieai render --input <directory> --output <file>
travelmovieai report --input <directory>
```

Они пока доходят до заглушек. Не используйте их для создания готового фильма.

Будущие story styles:

```text
cinematic documentary family vlog adventure romantic
```

## 17. Проверка установки

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

## 18. Типовые проблемы

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

## 19. Конфиденциальность

Media Scan работает локально:

- медиа не отправляются в облако;
- телеметрия отсутствует;
- SQLite и JSON остаются в workspace;
- GPS и локальные пути являются приватными данными;
- `.env`, workspace и исходные медиа не следует публиковать.

Cloud mode предназначен для будущих интеграций и по умолчанию выключен.

## 20. Обновление проекта

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
