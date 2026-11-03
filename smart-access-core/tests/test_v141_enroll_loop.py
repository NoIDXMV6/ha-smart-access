"""v1.4.1: сквозной тест петли обратной связи «Обучить» (интеграция + ядро).

Проверяет ВЕСЬ путь кнопки «Обучить» из карточки HA:
  svc_enroll (интеграция) -> MQTT smart_access/enroll/<kind> -> ядро
  (Brain._handle_enroll) -> ack smart_access/ack -> svc_enroll просыпается
  -> уведомление persistent_notification + события smart_access_event.

HA-рантайм в песочнице недоступен — модули homeassistant.* заменяются
минимальными заглушками, но логика интеграции и ядра — НАСТОЯЩАЯ.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

CORE = Path(__file__).resolve().parents[1]
REPO = CORE.parent
sys.path.insert(0, str(CORE))

from app import brain as brain_mod      # noqa: E402
from app import config as cfg           # noqa: E402
from app.store import Store             # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32 + b"\xff\xd9"


def _real_jpeg(w=640, h=480) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (80, 100, 120)).save(buf, "JPEG", quality=75)
    return buf.getvalue()


# --------------------------------------------------------------------- #
# Заглушки homeassistant.* (только то, что импортирует интеграция)

def _install_ha_stubs(monkeypatch) -> dict:
    """Возвращает словарь «полигон»: опубликованное MQTT, подписки, события."""
    published: list[tuple[str, str, int]] = []
    subscribed: dict[str, object] = {}
    notifications: list[dict] = []
    service_calls: list[tuple] = []

    ha = types.ModuleType("homeassistant")
    components = types.ModuleType("homeassistant.components")
    mqtt_mod = types.ModuleType("homeassistant.components.mqtt")

    async def async_publish(hass, topic, payload, qos=0, retain=False):
        published.append((topic, payload, qos))

    async def async_subscribe(hass, topic, handler, qos=0):
        subscribed[topic] = handler
        return lambda: subscribed.pop(topic, None)

    mqtt_mod.async_publish = async_publish
    mqtt_mod.async_subscribe = async_subscribe
    mqtt_mod.ReceiveMessage = object

    ce_mod = types.ModuleType("homeassistant.config_entries")
    ce_mod.ConfigEntry = object

    core_mod = types.ModuleType("homeassistant.core")

    def callback(fn):
        return fn

    class HomeAssistant:  # noqa: D101 — заглушка для аннотаций
        pass

    class ServiceCall:  # noqa: D101
        def __init__(self, data=None):
            self.data = data or {}

    core_mod.HomeAssistant = HomeAssistant
    core_mod.ServiceCall = ServiceCall
    core_mod.callback = callback

    exc_mod = types.ModuleType("homeassistant.exceptions")

    class HomeAssistantError(Exception):
        pass

    exc_mod.HomeAssistantError = HomeAssistantError

    disp_mod = types.ModuleType("homeassistant.helpers.dispatcher")
    disp_mod.async_dispatcher_send = lambda *a, **k: None

    helpers = types.ModuleType("homeassistant.helpers")
    entity_mod = types.ModuleType("homeassistant.helpers.entity")

    class Entity:  # noqa: D101
        pass

    entity_mod.Entity = Entity

    camera_pkg = types.ModuleType("homeassistant.components.camera")

    async def async_get_image(hass, entity_id, timeout=None):
        return SimpleNamespace(content=_real_jpeg())

    camera_pkg.async_get_image = async_get_image

    pn_mod = types.ModuleType("homeassistant.components.persistent_notification")

    async def pn_create(call):  # сервис вызывается через async_call
        pass

    pn_mod.async_create = pn_create

    ha.components = components
    components.mqtt = mqtt_mod
    components.camera = camera_pkg
    components.persistent_notification = pn_mod
    ha.config_entries = ce_mod
    ha.core = core_mod
    ha.exceptions = exc_mod
    ha.helpers = helpers
    helpers.dispatcher = disp_mod
    helpers.entity = entity_mod

    for name, mod in [("homeassistant", ha),
                      ("homeassistant.components", components),
                      ("homeassistant.components.mqtt", mqtt_mod),
                      ("homeassistant.components.camera", camera_pkg),
                      ("homeassistant.components.persistent_notification", pn_mod),
                      ("homeassistant.config_entries", ce_mod),
                      ("homeassistant.core", core_mod),
                      ("homeassistant.exceptions", exc_mod),
                      ("homeassistant.helpers", helpers),
                      ("homeassistant.helpers.dispatcher", disp_mod),
                      ("homeassistant.helpers.entity", entity_mod)]:
        monkeypatch.setitem(sys.modules, name, mod)

    return {"published": published, "subscribed": subscribed,
            "notifications": notifications, "service_calls": service_calls}


class _FakeBus:
    def __init__(self, box):
        self.box = box

    def async_fire(self, event, data=None):
        self.box.setdefault("events", []).append((event, data or {}))


class _FakeServices:
    def __init__(self, box):
        self.box = box
        self.registered: dict[str, object] = {}

    def async_register(self, domain, name, fn, schema=None):
        self.registered[f"{domain}.{name}"] = fn

    def has_service(self, domain, name):
        return f"{domain}.{name}" in self.registered

    async def async_call(self, domain, service, data=None, blocking=False):
        self.box["service_calls"].append((domain, service, data or {}))


class _FakeHass:
    def __init__(self, box):
        self.data = {}
        self.bus = _FakeBus(box)
        self.services = _FakeServices(box)
        self.loop = asyncio.get_running_loop()

    def async_create_task(self, coro):
        return self.loop.create_task(coro)


class _FakeAddonMQTT:
    """MQTT-клиент ядра: отдаёт ack наружу в «полигон»."""

    def __init__(self, box):
        self.box = box

    def publish(self, topic, payload, qos=0, retain=False):
        self.box.setdefault("addon_published", []).append((topic, payload))


def _make_addon_brain(tmp: str, box: dict, face_found=True):
    from app.store import Store as _Store

    cfg.DATA_DIR = f"{tmp}/addon-data"
    cfg.MEDIA_DIR = f"{tmp}/addon-media"
    st = _Store()

    class _Models:
        ok = True
        faces_ok = True
        gestures_ok = False
        status = {"clip": "ok", "face": "ok"}

        @staticmethod
        def car_embedding(jpg, with_views=False):
            return [[0.1] * 512] * (8 if with_views else 1)

        @staticmethod
        def detect_face_raw(jpg):
            if not face_found:
                return None, 0, 0
            import numpy as np

            return np.zeros((112, 112, 3), dtype=np.uint8), 640, 480

        @staticmethod
        def face_embedding_of_aligned(jpg):
            return [0.2] * 128

    b = brain_mod.Brain.__new__(brain_mod.Brain)
    b.store = st
    b.models = _Models()
    b.client = _FakeAddonMQTT(box)
    b.bridge = None
    b.stats = {"events_seen": 0, "grants": 0, "denied": 0, "alarms": 0,
               "vlm_calls": 0, "last_grant": None, "started": time.time(),
               "mqtt": "ok"}
    b.retrain_status = {"running": False}
    b.publish_status = lambda: None
    return b


# --------------------------------------------------------------------- #
# Тесты петли

def _run_svc(box, call_data: dict):
    """Регистрирует сервисы и запускает svc_enroll как задачу HA.

    Модуль интеграции выкидываем из кэша: он при первом импорте связывается
    с заглушками homeassistant.* ЭТОГО теста (monkeypatch восстанавливает
    sys.modules после теста — без переимпорта публикации ушли бы в полигон
    первого теста).
    """
    for m in [k for k in sys.modules
              if k == "custom_components" or k.startswith("custom_components.")]:
        del sys.modules[m]
    from custom_components.smart_access import _ensure_services

    hass = _FakeHass(box)
    _ensure_services(hass)
    fn = hass.services.registered["smart_access.enroll"]
    return fn, hass


def _deliver_ack(box, brain) -> None:
    """Ядро опубликовало ack — брокер доставляет его подписке интеграции."""
    acks = [(t, p) for t, p in box["addon_published"] if t == "smart_access/ack"]
    assert acks, "ядро не опубликовало ack"
    handler = box["subscribed"].get("smart_access/ack")
    assert handler, "интеграция не подписана на smart_access/ack"
    handler(SimpleNamespace(payload=acks[-1][1]))


def test_full_loop_success_face(monkeypatch):
    """Лицо распознано: фото в базе, услуга завершилась, уведомление «Готово»."""
    box = _install_ha_stubs(monkeypatch)
    with tempfile.TemporaryDirectory() as tmp:
        brain = _make_addon_brain(tmp, box, face_found=True)

        async def run():
            fn, hass = _run_svc(box, {})
            task = asyncio.ensure_future(fn(
                SimpleNamespace(data={"entity_id": "camera.kalitka",
                                      "subject": "Я", "kind": "face"})))
            await asyncio.sleep(0.2)  # услуга публикует кадр и ждёт ack
            # кадр дошёл до MQTT — прогоняем через НАСТОЯЩЕЕ ядро
            enroll_msgs = [(t, p) for t, p, *_ in box["published"]
                           if t == "smart_access/enroll/face"]
            assert enroll_msgs, "интеграция не опубликовала кадр в smart_access/enroll/face"
            brain._handle_enroll("face", enroll_msgs[-1][1].encode())
            _deliver_ack(box, brain)
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(run())

        assert "Я" in brain.store.subjects("face"), "фото должно попасть в базу ядра"
        evs = [e for e in box.get("events", []) if e[0] == "smart_access_event"]
        assert evs and evs[-1][1]["action"] == "enroll_done", evs
        notes = [c for c in box["service_calls"]
                 if c[:2] == ("persistent_notification", "create")]
        assert notes, "пользователь должен получить уведомление о результате"
        assert "Готово" in notes[-1][2]["message"], notes[-1][2]["message"]
        assert "Лица" in notes[-1][2]["message"], "уведомление ведёт во вкладку «Лица»"


def test_full_loop_face_not_found(monkeypatch):
    """Лица на кадре нет: услуга падает с вразумительной ошибкой,
    пользователь получает уведомление с подсказкой, журнал панели знает."""
    box = _install_ha_stubs(monkeypatch)
    with tempfile.TemporaryDirectory() as tmp:
        brain = _make_addon_brain(tmp, box, face_found=False)

        async def run():
            fn, hass = _run_svc(box, {})
            task = asyncio.ensure_future(fn(
                SimpleNamespace(data={"entity_id": "camera.kalitka",
                                      "subject": "Жена", "kind": "face"})))
            await asyncio.sleep(0.2)
            enroll_msgs = [(t, p) for t, p, *_ in box["published"]
                           if t == "smart_access/enroll/face"]
            assert enroll_msgs, "кадр должен уйти в MQTT до ошибки ядра"
            brain._handle_enroll("face", enroll_msgs[-1][1].encode())
            _deliver_ack(box, brain)
            with pytest.raises(Exception) as exc:
                await asyncio.wait_for(task, timeout=5)
            return exc

        exc = asyncio.run(run())
        assert "лицо на фото не найдено" in str(exc.value), str(exc.value)
        evs = [e for e in box.get("events", []) if e[0] == "smart_access_event"]
        assert evs and evs[-1][1]["action"] == "enroll_failed", evs
        notes = [c for c in box["service_calls"]
                 if c[:2] == ("persistent_notification", "create")]
        assert notes and "подойдите вплотную" in notes[-1][2]["message"], \
            "уведомление должно содержать подсказку «как правильно»"


def test_full_loop_timeout(monkeypatch):
    """Аддон молчит (не запущен): услуга падает по таймауту с уведомлением."""
    box = _install_ha_stubs(monkeypatch)
    import custom_components.smart_access as sa

    with tempfile.TemporaryDirectory():
        pass
    monkeypatch.setattr(sa, "ENROLL_ACK_TIMEOUT", 0.3)

    async def run():
        fn, hass = _run_svc(box, {})
        # _run_svc переимпортирует модуль — патчим константу в ЕГО globals
        fn.__globals__["ENROLL_ACK_TIMEOUT"] = 0.3
        task = asyncio.ensure_future(fn(
            SimpleNamespace(data={"entity_id": "camera.prud",
                                  "subject": "Моя машина", "kind": "car"})))
        with pytest.raises(Exception) as exc:
            await asyncio.wait_for(task, timeout=5)
        return exc

    exc = asyncio.run(run())
    assert "не подтвердил" in str(exc.value), str(exc.value)
    notes = [c for c in box["service_calls"]
             if c[:2] == ("persistent_notification", "create")]
    assert notes and "не подтвердил" in notes[-1][2]["message"]


def test_full_loop_success_car(monkeypatch):
    box = _install_ha_stubs(monkeypatch)
    with tempfile.TemporaryDirectory() as tmp:
        brain = _make_addon_brain(tmp, box)

        async def run():
            fn, hass = _run_svc(box, {})
            task = asyncio.ensure_future(fn(
                SimpleNamespace(data={"entity_id": "camera.prud",
                                      "subject": "Моя машина", "kind": "car"})))
            await asyncio.sleep(0.2)
            enroll_msgs = [(t, p) for t, p, *_ in box["published"]
                           if t == "smart_access/enroll/car"]
            assert enroll_msgs, "нет публикации в smart_access/enroll/car"
            brain._handle_enroll("car", enroll_msgs[-1][1].encode())
            _deliver_ack(box, brain)
            await asyncio.wait_for(task, timeout=5)

        asyncio.run(run())
        assert "Моя машина" in brain.store.subjects("car")
        evs = [e for e in box.get("events", []) if e[0] == "smart_access_event"]
        assert evs[-1][1]["action"] == "enroll_done"
