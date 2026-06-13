# Roadmap TravelMovieAI

Roadmap содержит только незавершённые улучшения. Текущий функционал перечислен
в [README.md](../README.md), архитектура — в [architecture.md](architecture.md).

## P0. Надёжность длительных задач

- отмена и пауза Movie jobs;
- возобновление после остановки процесса;
- реальный progress по файлам и сценам;
- лимит и очистка дискового кэша;
- проверка свободного места до рендера;
- сохранение истории Movie jobs после перезапуска;
- миграции SQLite вместо неуправляемого `create_all`.

Критерий готовности: обработка большого проекта безопасно продолжается после
сбоя и не оставляет валидно выглядящих частичных артефактов.

## P1. Качество автоматического монтажа

- embeddings и FAISS для семантических дублей;
- GPS и embeddings в Event Detection;
- сохранение полных границ реплик Whisper;
- запрет обрезки важных реплик;
- audio classification: speech, music, silence, crowd, laughter, applause;
- сохранение важных ambient sounds;
- beat-aware смена сцен;
- continuity rules для направления движения, света и локации;
- автоматическое чередование общих, средних и крупных планов.

Критерий готовности: фильм не повторяется, не обрывает речь и сохраняет
характерные звуки поездки.

## P1. Story Builder

- локальный LLM adapter через LM Studio;
- структурированный narrative поверх событий;
- duration budget по секциям истории;
- несколько вариантов фильма из одного анализа;
- ручная перестановка событий и сцен;
- редактирование event title/summary;
- генерация названия, вступления и финала.

LLM получает только проверенные metadata и transcripts, но не raw media.

## P1. Редактор сцен

- фильтры по quality, event, location, activity и людям;
- редактирование caption, transcript и landmark;
- объединение и разделение событий;
- preview отдельной сцены;
- drag-and-drop порядка;
- сохранение пользовательских версий timeline;
- сравнение двух вариантов монтажа.

## P2. Визуальная обработка

- Ken Burns для фотографий;
- face/object-aware crop;
- корректная обработка rotation metadata;
- вертикальное видео и configurable background;
- базовое выравнивание цвета и экспозиции;
- HDR/SDR tone mapping;
- титры событий, субтитры и credits;
- локальные шрифты и safe-area validation.

## P2. Музыка и narration

- анализ BPM и структуры локальных треков;
- beat grid и музыкальные акценты;
- выбор трека по storyboard, а не только средним метрикам;
- narration draft;
- Piper/XTTS voice synthesis;
- отдельный ducking для речи и ambient sounds.

## P2. Производительность

- batch Vision inference;
- повторное использование загруженных моделей;
- proxy media для 4K/8K;
- bounded parallel frame extraction;
- benchmark на 500+ видео и 100+ GB;
- оценка времени выполнения;
- CUDA/CPU профили качества и скорости.

## P3. Доставка продукта

- Windows installer;
- автоматическая проверка FFmpeg и моделей;
- backup/export проекта;
- HTML report;
- PySide6 shell поверх существующего application API;
- документированный plugin/provider interface.

## Ближайшая последовательность

1. Audio classification и границы реплик Whisper.
2. Embeddings/FAISS для дублей и Event Detection.
3. Редактор событий и transcript.
4. Beat-aware timeline.
5. LLM Story Builder.
6. Crop, Ken Burns, subtitles и titles.
