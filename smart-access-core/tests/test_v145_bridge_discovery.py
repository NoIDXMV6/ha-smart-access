"""Тесты v1.4.5 — фикс «Этот объект недоступен» (карточки без видео).

Регресс на реальную жалобу пользователя: карточки examples/cards показывают
«Этот объект недоступен», видео с камер не идёт. Причина: сущности моста
camera.smart_access_bridge_* создавались ТОЛЬКО по MQTT-статусу аддона —
без статуса (брокер не настроен/не готов, retain потерян, аддон остановлен)
сущностей не было вовсе, карточки молча пусты.

Что проверяем:

1. camera.py: у платформы камер ВТОРОЙ источник списка камер — HTTP-опрос
   {bridge_url}/api/status (объект bridge), отложенный старт + периодический
   повтор; цикл переживает ошибки (не умирает после первого сбоя).
2. camera.py: если ни MQTT, ни HTTP не дали ни одной камеры — ОДНО
   уведомление-чеклист (notification_id smart_access_cameras), а не тишина.
3. Создание сущностей идемпотентно по имени камеры, entity_id
   детерминирован (camera.smart_access_bridge_<имя>) — карточки примеров
   остаются корректными.
4. Версии аддона/интеграции синхронны и зафиксированы: 1.4.5.
5. CHANGELOG начинается с секции версии из manifest (release.yml извлекает
   заметки по заголовку «# X.Y.Z»), README содержит разбор «Этот объект
   недоступен».
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "smart-access-core"
INTEGRATION = REPO / "custom_components" / "smart_access"

CAMERA_PY = (INTEGRATION / "camera.py").read_text(encoding="utf-8")


# --- 1. HTTP-резерв списка камер -----------------------------------------------

def test_camera_platform_has_http_fallback():
    assert "/api/status" in CAMERA_PY, (
        "camera.py не опрашивает сайт аддона /api/status — без MQTT-статуса "
        "сущности моста не создаются («Этот объект недоступен» в карточках)"
    )
    assert '(data or {}).get("bridge")' in CAMERA_PY, (
        "список камер должен разбираться из объекта bridge ответа /api/status"
    )


def test_http_fallback_is_periodic_with_delayed_start():
    # отложенный первый запуск (дать retain-MQTT прийти первым)...
    assert re.search(r"_FIRST_PROBE_DELAY\s*=\s*8\.0", CAMERA_PY)
    assert re.search(r"await asyncio\.sleep\(_FIRST_PROBE_DELAY\)", CAMERA_PY)
    # ...и периодический повтор
    assert re.search(r"_PROBE_INTERVAL\s*=\s*60\.0", CAMERA_PY)
    assert re.search(r"await asyncio\.sleep\(_PROBE_INTERVAL\)", CAMERA_PY)


def test_resync_loop_survives_errors():
    assert "LOG.exception" in CAMERA_PY, (
        "цикл HTTP-резерва должен переживать любые ошибки (LOG.exception), "
        "а не умирать после первого сбоя — иначе фикс не работает"
    )
    assert re.search(
        r"except Exception:.*\n\s*LOG\.exception", CAMERA_PY, re.MULTILINE
    )


def test_resync_task_cancelled_on_reload_and_unload():
    # reload платформы: старый цикл останавливается
    assert 'data.get("camera_resync_task")' in CAMERA_PY
    assert "old.cancel()" in CAMERA_PY
    # unload entry: задача отменяется
    assert "entry.async_on_unload(lambda: task.cancel())" in CAMERA_PY


# --- 2. Уведомление-чеклист вместо молчаливой пустоты ---------------------------

def test_no_cameras_notification_exists_and_is_once_per_boot():
    assert "persistent_notification" in CAMERA_PY
    assert '"smart_access_cameras"' in CAMERA_PY, (
        "уведомление «камеры моста не найдены» должно иметь стабильный "
        "notification_id smart_access_cameras"
    )
    # одноразовость: флаг _notified + порог после двух неудачных проб
    assert re.search(r"_NOTIFY_AFTER_FAILS\s*=\s*2", CAMERA_PY)
    assert re.search(r"self\._notified\s*=\s*True", CAMERA_PY)
    # чеклист в тексте уведомления: аддон, брокер MQTT, перезапуск HA
    for needle in ("Smart Access Core запущен", "MQTT", "ПЕРЕЗАПУЩЕН"):
        assert needle in CAMERA_PY, f"в уведомлении-чеклисте нет «{needle}»"


def test_notification_fires_only_when_nothing_discovered():
    # уведомление гейтится отсутствием и известных камер, и статуса
    assert re.search(
        r"if not self\.known and not self\.got_cameras", CAMERA_PY
    ), "уведомление не должно срабатывать, если камеры уже найдены"


# --- 3. Идемпотентность и детерминированный entity_id ---------------------------

def test_entities_are_idempotent_by_camera_name():
    assert re.search(r"if name in self\.known:\n\s*continue", CAMERA_PY)
    assert "self.known.add(name)" in CAMERA_PY


def test_entity_id_is_deterministic():
    assert 'self.entity_id = f"camera.{DOMAIN}_bridge_{cam}"' in CAMERA_PY, (
        "entity_id моста обязан быть детерминированным — на него ссылаются "
        "карточки examples/cards"
    )


# --- 4-5. Версии, CHANGELOG, README ---------------------------------------------

def test_versions_are_in_sync_v145():
    addon = json.loads((CORE / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((INTEGRATION / "manifest.json").read_text(encoding="utf-8"))
    assert addon["version"] == manifest["version"] == "1.4.5"


def test_changelog_starts_with_manifest_version_section():
    manifest = json.loads((INTEGRATION / "manifest.json").read_text(encoding="utf-8"))
    text = (CORE / "CHANGELOG.md").read_text(encoding="utf-8")
    # release.yml извлекает секцию по заголовку «# <версия>» — секция обязана
    # быть ПЕРВОЙ строкой файла, иначе в релиз уедут чужие заметки
    assert text.startswith(f"# {manifest['version']}")
    first_section = text.split("\n# ", 1)[0]
    assert "Этот объект недоступен" in first_section


def test_readme_has_unavailable_checklist():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Этот объект недоступен" in readme, (
        "в README нет разбора «Этот объект недоступен»"
    )
    assert "camera.smart_access_bridge_" in readme
