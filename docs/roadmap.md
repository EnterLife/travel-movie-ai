# План дальнейшей разработки TravelMovieAI

Roadmap построен от надёжного локального проекта к полноценному AI-монтажу.
Каждый этап должен завершаться рабочим пользовательским сценарием, устойчивыми
артефактами и автоматическими тестами.

## Принципы приоритизации

1. Сначала защищаем исходные данные и воспроизводимость проекта.
2. Затем строим измеримый video pipeline без AI-зависимостей.
3. После этого подключаем локальные модели по одной задаче.
4. Story Builder получает только проверенные структурированные данные.
5. Renderer реализуется после стабилизации timeline-контракта.

## Этап 1. Надёжность проекта и web jobs

Приоритет: критический.

Статус: реализован в текущей версии.

Задачи:

- единая валидация input и workspace для CLI, API и application service;
- запрет конфликтующих активных заданий в одном workspace;
- сохранение истории web jobs на диск;
- восстановление завершённых результатов после перезапуска;
- перевод прерванных jobs в понятный failed status;
- список последних jobs через API;
- health-check Python, FFmpeg и FFprobe;
- контролируемые пользовательские ошибки без публикации traceback в API;
- тесты перезапуска, конфликтов workspace и отсутствующих бинарников.

Готовность:

- повторный запрос к занятому workspace получает HTTP 409;
- завершённый job доступен после создания нового `ScanJobManager`;
- прерванный job после перезапуска имеет статус `failed`;
- CLI и web используют одинаковые правила путей;
- `/api/health` показывает готовность FFprobe.

Реализованные артефакты:

```text
workspace/.web/jobs.json
```

Следующая активная задача: расширение Frame Sampling и Quality Analysis.

## Этап 2. Scene Detection

Приоритет: следующий после этапа 1.

Статус: реализован базовый рабочий вариант.

Задачи:

- таблица `scenes` в SQLite;
- адаптер PySceneDetect;
- fallback на равномерные сегменты для файлов без уверенных cuts;
- настройка минимальной и максимальной длительности сцены;
- инкрементальный cache key по asset fingerprint и параметрам detector;
- `artifacts/scenes.json`;
- прогресс на уровне файлов;
- отображение количества сцен в web UI.

Готовность:

- видео делится на сцены с валидными временными границами;
- повторный запуск не анализирует неизменённые видео;
- удаление asset удаляет связанные scenes;
- synthetic video integration tests проходят без GPU.

Реализовано:

- таблица `scenes`;
- PySceneDetect adapter с lazy import;
- равномерный fallback;
- min/max duration и cache key;
- `artifacts/scenes.json`;
- каскадное удаление сцен вместе с asset.

Отображение галереи и расширенного прогресса остаётся в этапе 3.

## Этап 3. Frame Sampling

Приоритет: высокий.

Статус: реализована базовая стадия с start/middle/end contact sheet.

Задачи:

- извлечение start/middle/end и keyframe кадров;
- единый FFmpeg frame extraction adapter;
- JPEG/WebP thumbnails для web UI;
- таблица sampled frames;
- атомарная запись и очистка устаревших кадров;
- галерея сцен в web UI.

Готовность:

- каждая валидная сцена имеет representative frames;
- кадры переиспользуются при повторном запуске;
- UI не читает исходные видео целиком.

## Этап 4. Visual Quality Analysis

Приоритет: высокий, до vision LLM.

Статус: реализован базовый анализ representative contact sheets.

Задачи:

- blur, brightness, contrast и noise;
- оценка motion/camera shake на ограниченной выборке кадров;
- нормализация метрик в score 0-100;
- причины низкой оценки;
- фильтры и сортировка сцен в UI.

Готовность:

- метрики воспроизводимы на synthetic fixtures;
- quality analysis не интерпретирует содержание сцены;
- плохие сцены видны пользователю до AI-анализа.

Реализованы brightness, contrast, sharpness, saturation, colorfulness, noise,
motion, camera shake, exposure, нормализованный quality score, технические
причины отбраковки и участие quality score в scene ranking.
Фильтры и просмотр метрик в галерее остаются следующей UI-задачей.

## Этап 5. Speech Analysis

Приоритет: высокий.

Статус: базовый опциональный Faster Whisper pipeline реализован.

Задачи:

- lazy Faster Whisper provider;
- извлечение аудиодорожки через FFmpeg;
- language detection, transcript, timestamps и confidence;
- CPU/CUDA конфигурация;
- model cache и понятный первый запуск;
- просмотр транскрипта в UI.

Реализованы lazy provider, CPU/CUDA с auto fallback, scene-level WAV extraction,
language/confidence metadata, cache key и `speech_analysis.json`. Отображение и
редактирование transcript в UI остаются следующими улучшениями.

Готовность:

- default tests не скачивают модель;
- provider покрыт fake-contract tests;
- отсутствующая модель или CUDA не ломает базовый Media Scan.

## Этап 4.5. Vision AI Analysis

Приоритет: высокий.

Статус: реализован базовый Stage 4.5.

Задачи:

- provider contracts для Qwen2.5-VL и Florence-2;
- строгая JSON schema scene understanding;
- caption, location, activity, emotion, people и landmark;
- versioned prompts;
- batch processing и cache metadata;
- ручной просмотр/исправление результата в UI.

Готовность:

- невалидный ответ модели не попадает в domain без validation;
- provider/model/prompt version входят в cache key;
- cloud fallback никогда не включается автоматически.

Реализованы Qwen-compatible provider через LM Studio, ленивый Florence-2
adapter, строгая JSON schema, versioned prompt, cache metadata, landmarks,
people groups, score factors и пересчёт vision score с измеренным OpenCV
quality. В web UI выбираются backend и модель. Ручная коррекция metadata,
batch inference и расширенная gallery остаются следующими улучшениями.

## Этап 7. Audio, embeddings и duplicate detection

Приоритет: средний.

Статус: perceptual duplicate detection реализован; audio и embeddings ожидают.

Задачи:

- speech/music/silence/crowd/laughter classification;
- sentence-transformers embeddings;
- FAISS index;
- near-duplicate scenes и photos;
- similarity groups в UI.

Готовность:

- индекс можно перестроить независимо;
- duplicate decisions объяснимы и не удаляют исходные файлы.

Текущий duplicate detector использует локальный perceptual hash representative
frame, выбирает keeper по manual override, vision importance и quality и пишет
`duplicates.json`. Embeddings нужны для семантически похожих, но визуально
различающихся дублей.

## Этап 8. Event Detection и Scene Ranking

Приоритет: средний.

Статус: базовые Event Detection и event-aware Scene Ranking реализованы.

Задачи:

- multimodal event clustering;
- temporal и GPS признаки;
- event titles и summaries;
- итоговый scene score с объяснимыми факторами;
- редактор event groups в UI.

Готовность:

- `events.json` стабилен и версионирован;
- пользователь может исправить clustering до Story Builder.

Реализованы temporal/semantic clustering, `events.json`, таблица `events` в
SQLite, event title/summary/confidence и участие event importance и landmarks
в итоговом ranking. GPS clustering и ручной event editor пока не реализованы.

## Этап 9. Story Builder

Приоритет: после стабилизации events.

Статус: базовый детерминированный event-based storyboard реализован.

Задачи:

- local LM Studio adapter;
- story styles;
- structured storyboard schema;
- duration budget;
- narration draft;
- ручное изменение порядка событий и сцен.

Текущая версия строит opening/journey/highlight/finale, обеспечивает покрытие
событий, сохраняет `storyboard.json` и `selection_decisions.json`. LLM narrative
и ручной event editor ещё не реализованы.

Готовность:

- storyboard создаётся полностью локально;
- LLM не получает raw media;
- результат проходит schema validation.

## Этап 10. Timeline, music и rendering

Приоритет: финальный MVP.

Статус: частично реализован renderer с semantic scene selection, OpenCV
quality ranking, локально генерируемой музыкой, ducking, переходами и
CUDA/NVENC с CPU fallback. Storyboard, титры и субтитры не реализованы.

Задачи:

- declarative timeline contract;
- music plan и ducking;
- subtitles, titles и transitions;
- FFmpeg filter graph builder;
- render validation и resumable intermediates;
- preview render и final render;
- HTML report.

Готовность:

- один и тот же timeline даёт воспроизводимый render;
- renderer не меняет story decisions;
- итоговый MP4 проверяется FFprobe после записи.

## Сквозные улучшения

- миграции базы вместо неуправляемого `create_all`;
- структурированное локальное логирование с ротацией;
- schema/version metadata для каждого артефакта;
- cancellation и реальный progress для jobs;
- ограничение дискового cache и команды очистки;
- benchmark на 500+ видео;
- Windows installer после стабилизации MVP;
- PySide6 desktop shell только поверх стабильного application API.

## Ближайшая последовательность

1. Добавить редактирование event groups и transcript в gallery.
2. Реализовать audio classification и сохранение важных ambient sounds.
3. Добавить embeddings/GPS признаки в duplicate и Event Detection.
4. Добавить beat-aware timeline и сохранение границ реплик.
5. Расширить Story Builder локальным LLM narrative.

Не следует начинать Story Builder или renderer до появления проверенных scenes,
frames и quality metadata.
