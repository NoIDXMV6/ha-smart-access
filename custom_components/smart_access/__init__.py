"""Интеграция Smart Access: мост между HA и аддоном smart-access-core по MQTT.

Зона ответственности интеграции:
  * сервисы (кнопка «Обучить» на карточках, тест-распознавание);
  * сенсоры статуса/последнего события;
  * события HA (smart_access_ack) для автоматизаций;
  * UI-настройки (имена «своих» и привязка камер) — OptionsFlow.

Всё общение с ядром — через локальный Mosquitto, никаких внешних зависимостей.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import voluptuous as vol

from homeassistant.components import mqtt
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.entity import Entity

from .const import (
    DOMAIN,
    EVENT_ACK,
    MQTT_PREFIX,
    SIGNAL_EVENT,
    SIGNAL_STATUS,
    TOPIC_ACK,
    TOPIC_CMD_IDENTIFY,
    TOPIC_ENROLL_CAR,
    TOPIC_ENROLL_FACE,
    TOPIC_EVENT,
    TOPIC_STATUS,
    CONF_CAR_SUBJECTS,
    CONF_FACE_SUBJECTS,
    CONF_ENROLL_MAP,
)

LOG = logging.getLogger(__name__)

PLATFORMS = ["sensor"]

SERVICE_ENROLL = "enroll"
SERVICE_IDENTIFY = "identify"

ENROLL_SCHEMA = vol.Schema(
    {
        vol.Required("entity_id"): str,
        vol.Required("subject"): str,
        vol.Optional("kind", default="auto"): vol.In(["auto", "car", "face"]),
    }
)

IDENTIFY_SCHEMA = vol.Schema(
    {
        vol.Required("entity_id"): str,
        vol.Optional("kind", default="car"): vol.In(["car", "face"]),
    }
)


def _parse_map(raw: str) -> dict[str, str]:
    """'camera.x=car:Моя машина' построчно -> {'camera.x': {'kind','subject'}}"""
    result: dict[str, str] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        entity, rest = line.split("=", 1)
        result[entity.strip()] = rest.strip()
    return result


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    hass.data.setdefault(DOMAIN, {})
    data = hass.data[DOMAIN].setdefault("state", {"status": {}, "last_event": {}, "ack": {}})
    data.setdefault("options", {})
    data["options"] = {
        CONF_CAR_SUBJECTS: entry.options.get(CONF_CAR_SUBJECTS, "Моя машина, Машина жены"),
        CONF_FACE_SUBJECTS: entry.options.get(CONF_FACE_SUBJECTS, "Я, Жена, Сын, Дочь"),
        CONF_ENROLL_MAP: entry.options.get(CONF_ENROLL_MAP, ""),
    }

    @callback
    def on_status(msg: mqtt.ReceiveMessage) -> None:
        try:
            payload = json.loads(msg.payload)
        except ValueError:
            return
        data["status"] = payload
        async_dispatcher_send(hass, SIGNAL_STATUS, payload)

    @callback
    def on_event(msg: mqtt.ReceiveMessage) -> None:
        try:
            payload = json.loads(msg.payload)
        except ValueError:
            return
        data["last_event"] = payload
        hass.bus.async_fire(f"{DOMAIN}_event", payload)
        async_dispatcher_send(hass, SIGNAL_EVENT, payload)

    @callback
    def on_ack(msg: mqtt.ReceiveMessage) -> None:
        try:
            payload = json.loads(msg.payload)
        except ValueError:
            return
        hass.bus.async_fire(EVENT_ACK, payload)

    unsub = [
        await mqtt.async_subscribe(hass, TOPIC_STATUS, on_status, qos=1),
        await mqtt.async_subscribe(hass, TOPIC_EVENT, on_event, qos=1),
        await mqtt.async_subscribe(hass, TOPIC_ACK, on_ack, qos=1),
    ]
    entry.async_on_unload(*unsub)

    async def svc_enroll(call: ServiceCall) -> None:
        entity_id = call.data["entity_id"].strip()
        subject = call.data["subject"].strip()
        kind = call.data.get("kind", "auto")

        if kind == "auto":
            mapping = _parse_map(data["options"].get(CONF_ENROLL_MAP, ""))
            spec = mapping.get(entity_id, "")
            kind = "face" if spec.startswith("face") else "car"

        image_bytes = await _get_camera_jpeg(hass, entity_id)
        payload = json.dumps(
            {"subject": subject, "image_b64": base64.b64encode(image_bytes).decode("ascii"),
             "entity_id": entity_id},
            ensure_ascii=False,
        )
        topic = TOPIC_ENROLL_FACE if kind == "face" else TOPIC_ENROLL_CAR
        await mqtt.async_publish(hass, topic, payload, qos=1)
        LOG.info("Обучение отправлено: %s «%s» (%s байт)", kind, subject, len(image_bytes))
        hass.bus.async_fire(f"{DOMAIN}_event", {
            "kind": kind, "camera": entity_id, "action": "enroll_sent",
            "detail": subject, "score": 0, "subject": subject,
        })

    async def svc_identify(call: ServiceCall) -> None:
        entity_id = call.data["entity_id"].strip()
        kind = call.data.get("kind", "car")
        await mqtt.async_publish(
            hass, TOPIC_CMD_IDENTIFY,
            json.dumps({"entity_id": entity_id, "kind": kind}, ensure_ascii=False),
            qos=1,
        )

    if not hass.services.has_service(DOMAIN, SERVICE_ENROLL):
        hass.services.async_register(DOMAIN, SERVICE_ENROLL, svc_enroll, schema=ENROLL_SCHEMA)
        hass.services.async_register(DOMAIN, SERVICE_IDENTIFY, svc_identify, schema=IDENTIFY_SCHEMA)

    # Бренд-иконка интеграции (без авторизации): entity_picture и карточки
    try:
        from .brand import BrandIconView

        hass.http.register_view(BrandIconView())
    except Exception as exc:  # noqa: BLE001
        LOG.debug("Brand view не зарегистрирован: %s", exc)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    return unload_ok


async def _get_camera_jpeg(hass: HomeAssistant, entity_id: str) -> bytes:
    from homeassistant.components import camera

    try:
        image = await camera.async_get_image(hass, entity_id, timeout=15)
        jpg = image.content
    except (HomeAssistantError, ValueError) as exc:
        raise HomeAssistantError(f"Не удалось получить кадр с {entity_id}: {exc}") from exc

    return _downscale(jpg)


def _downscale(jpg: bytes, max_side: int = 1280, quality: int = 85) -> bytes:
    """Масштабирует кадр для MQTT (на выходе всё равно ресайз моделей)."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(jpg))
        img = img.convert("RGB")
        w, h = img.size
        scale = max(1.0, max(w, h) / max_side)
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return jpg
