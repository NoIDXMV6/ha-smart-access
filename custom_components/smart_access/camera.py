"""Камеры Smart Access — карточки HA играют поток МОСТА (v1.4.0).

ЗАЧЕМ: у каждой физической камеры должен быть ОДИН получатель — мост
аддона Smart Access (см. DOCS «Архитектура камер: один получатель»).
Frigate читает мост по http://172.30.32.1:8100/mjpeg/<name>; карточки HA
должны читать ТОТ ЖЕ мост, а не открывать камере вторую сессию.

Раньше карточку предлагалось переключать на мост через «Универсальную
камеру» с URL /mjpeg/... в поле «Поток» — это НЕ работает: Generic Camera
открывает stream_source как RTSP/HTTP-видео средствами stream-компоненты
и не понимает multipart/x-mixed-replace, карточка остаётся серой.

Теперь интеграция сама создаёт по списку камер из MQTT-статуса аддона
сущности camera.smart_access_bridge_<name>:
  * live-вид в карточках — байтовый прокси multipart-потока моста
    (handle_async_mjpeg_stream), без перекодирования;
  * снимок (карточка-плитка, «Обучить», уведомления) — /snapshot/<name>.

Адрес моста — опция интеграции bridge_url (по умолчанию
http://172.30.32.1:8100 — шлюз hassio-сети, по которому сайт аддона
доступен и из HA Core, и из аддонов).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiohttp import ClientError, ClientTimeout, web
from homeassistant.components.camera import Camera
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_BRIDGE_URL,
    DEFAULT_BRIDGE_URL,
    DOMAIN,
    SIGNAL_STATUS,
)

LOG = logging.getLogger(__name__)

# /snapshot/<name>: кадр есть быстро, но для ночных облачных камер HA может
# собирать его долго — даём щедрый таймаут
_SNAP_TIMEOUT = ClientTimeout(total=20)
# /mjpeg/<name>: бесконечный поток, рвём только по молчанию сети
_STREAM_TIMEOUT = ClientTimeout(total=None, sock_connect=10, sock_read=60)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Создаёт сущности камер по списку из MQTT-статуса аддона (динамически).

    Статус приходит с retain-флагом: и сразу после перезапуска HA (до того,
    как аддон что-то опубликует), и при изменении списка камер в опциях
    аддона. Новая камера в статусе -> новая сущность без перезагрузки.
    """
    known: set[str] = set()

    @callback
    def _sync_from_status(payload: dict[str, Any]) -> None:
        cams = (payload or {}).get("cameras") or []
        bridge = _bridge_url(hass)
        new: list[SmartAccessBridgeCamera] = []
        for c in cams:
            name = str((c or {}).get("name") or "").strip()
            if not name or name in known:
                continue
            known.add(name)
            display = str((c or {}).get("display") or "") or name
            new.append(SmartAccessBridgeCamera(bridge, name, display))
        if new:
            LOG.info("Сущности камер моста Smart Access: %s",
                     ", ".join(f"camera.smart_access_bridge_{c._cam}" for c in new))
            async_add_entities(new)

    # статус мог прийти раньше платформы (retain-MQTT) — не теряем его
    state = hass.data.get(DOMAIN, {}).get("state") or {}
    _sync_from_status({"cameras": state.get("cameras") or []})

    async_dispatcher_connect(hass, SIGNAL_STATUS, _sync_from_status)


def _bridge_url(hass: HomeAssistant) -> str:
    state = hass.data.get(DOMAIN, {}).get("state") or {}
    return str(state.get("bridge_url") or DEFAULT_BRIDGE_URL).rstrip("/")


class SmartAccessBridgeCamera(Camera):
    """Камера-прокси моста: live = MJPEG моста, снимок = /snapshot/<name>."""

    _attr_should_poll = False
    _attr_has_entity_name = False
    _attr_icon = "mdi:cctv"
    _attr_brand = "Smart Access"

    def __init__(self, bridge_url: str, cam: str, display: str) -> None:
        super().__init__()
        self._bridge = bridge_url.rstrip("/")
        self._cam = cam
        self._attr_name = f"Smart Access {display}"
        self._attr_unique_id = f"{DOMAIN}_camera_{cam}"
        # детерминированный entity_id: имя камеры уже в формате Frigate
        # ([a-z0-9_]) — карточки в документации/примерах ссылаются на него
        self.entity_id = f"camera.{DOMAIN}_bridge_{cam}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, "addon")},
            name="Smart Access",
            manufacturer="Smart Access",
            model="smart-access-core",
        )

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "bridge_url": self._bridge,
            "bridge_mjpeg": f"{self._bridge}/mjpeg/{self._cam}",
            "bridge_snapshot": f"{self._bridge}/snapshot/{self._cam}",
        }

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Один кадр моста (для плиток, уведомлений, «Обучить»)."""
        try:
            resp = await async_get_clientsession(self.hass).get(
                f"{self._bridge}/snapshot/{self._cam}", timeout=_SNAP_TIMEOUT
            )
            if resp.ok:
                data = await resp.read()
                return data or None
            LOG.debug("Мост: снимок %s: HTTP %s", self._cam, resp.status)
        except (ClientError, asyncio.TimeoutError) as exc:
            LOG.debug("Мост: снимок %s не удался: %s", self._cam, exc)
        return None

    async def handle_async_mjpeg_stream(self, request: web.Request) -> web.StreamResponse:
        """Live-вид: байтовый прокси multipart-потока моста (без перекодирования).

        Переопределяем базовый метод (тот, что обслуживает
        /api/camera_proxy_stream/<entity>): отдаём браузеру ровно то, что
        мост отдаёт Frigate, с тем же content-type и границей.
        """
        url = f"{self._bridge}/mjpeg/{self._cam}"
        session = async_get_clientsession(self.hass)
        try:
            async with session.get(url, timeout=_STREAM_TIMEOUT) as upstream:
                if not upstream.ok:
                    return web.Response(status=502, text=f"bridge HTTP {upstream.status}")
                resp = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": upstream.headers.get(
                            "Content-Type", "multipart/x-mixed-replace"
                        ),
                        "Cache-Control": "no-cache, no-store",
                    },
                )
                await resp.prepare(request)
                try:
                    async for chunk in upstream.content.iter_chunked(65536):
                        await resp.write(chunk)
                except (ClientError, ConnectionResetError, asyncio.TimeoutError):
                    pass
                with suppress_break():
                    await resp.write_eof()
                return resp
        except (ClientError, asyncio.TimeoutError) as exc:
            LOG.debug("Мост: поток %s недоступен: %s", self._cam, exc)
            return web.Response(status=502, text=f"bridge unavailable: {exc}")


class suppress_break:
    """Гасит разрыв клиентского соединения при write_eof (норма для MJPEG)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, (ConnectionResetError, asyncio.CancelledError))
