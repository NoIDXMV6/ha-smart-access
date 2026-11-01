# DOCS.md — Smart Access Core (справка аддона)

Краткая справка отображается в карточке аддона.

## Что делает

* Раздаёт камеры HA (включая облачные А.М.Видео/ЕГДС) локальным MJPEG-потоком
  для Frigate: `http://127.0.0.1:8100/mjpeg/<name>`.
* Слушает события Frigate (MQTT), распознаёт машины (CLIP, CPU) и лица (SFace).
* Открывает исполнителей через HA: `executor_service` + `executor_target`
  (например `amvideo.open_barrier` + `button.1_shlagbaum_..._otkryt`,
  `lock.unlock` + `lock.gate`).
* Ведёт базу «своих»: фото в `/media/smart_access/{cars,faces}/<имя>/`,
  эмбеддинги в `/data/db/`.

## Обучение

* Кнопка «Обучить» на карточке → сервис `smart_access.enroll`.
* Или панель Smart Access (боковая панель): загрузка фото файлом, удаление, тест.

## VLM (опционально)

OpenAI-совместимый endpoint: `base_url` (например `http://192.168.1.50:11434/v1`),
`api_key`, `model` (например `moondream`). Режимы: `gray_zone` (рекомендуется),
`verify_all`, `off`. При недоступности решение остаётся за локальными моделями.

## Журнал и события HA

* Сенсоры: `sensor.smart_access_status`, `sensor.smart_access_last_event`.
* События: `smart_access_event` (допуск/отказ/обучение), `smart_access_ack`.

## Диагностика

* Лог аддона: очередь событий, score матчей, ошибки MQTT/Frigate.
* Панель: счётчики сбоев камер моста, состояние моделей, статус Frigate.
