"""Камеры Smart Access — карточки HA играют поток МОСТА (v1.4.0).

ЗАЧЕМ: у каждой физической камеры должен быть ОДИН получатель — мост
аддона Smart Access (см. DOCS «Архитектура камер: один получатель»).
Frigate читает мост по http://172.30.32.1:8100/mjpeg/<name>; карточки HA
должны читать ТОТ ЖЕ мост, а не открывать камере вторую сессию.

Раньше карточку предлагалось переключать на мост через «Универсальную
камеру» с URL /mjpeg/... в поле «Поток» — это НЕ работает: Generic Camera
открывает stream_source как RTSP/HTTP-видео средствами stream-компоненты
и не понимает multipart/x-mixed-replace, карточка остаётся серой.

Интеграция сама создаёт сущности camera.smart_access_bridge_<name>:
  * live-вид в карточках — байтовый прокси multipart-потока моста
    (handle_async_mjpeg_stream), без перекодирования;
  * снимок (карточка-плитка, «Обучить», уведомления) — /snapshot/<name>.

Адрес моста — опция интеграции bridge_url (по умолчанию
http://172.30.32.1:8100 — шлюз hassio-сети, по которому сайт аддона
доступен и из HA Core, и из аддонов).

v1.4.5 — ФИКС «Этот объект недоступен» в карточках (видео с камер не идёт):
сущности моста создавались ТОЛЬКО по MQTT-статусу аддона, и если статус
не приходил (брокер MQTT не настроен/не готов в момент старта HA, retain
потерян, аддон остановлен), сущностей не было вовсе — карточки показывали
«Этот объект недоступен», а в журнале было тихо. Теперь у платформы два
НЕЗАВИСИМЫХ источника списка камер:
  1) MQTT-статус (retain + dispatcher) — как в v1.4.0;
  2) HTTP-резерв: через 8 с после загрузки HA и далее раз в 60 с платформа
     опрашивает сайт аддона {bridge_url}/api/status (объект «bridge» со
     списком камер) и создаёт недостающие сущности сама.
Если ни MQTT, ни HTTP не дали ни одной камеры — приходит ОДНО уведомление
с чеклистом («Smart Access: камеры моста не найдены») вместо молчаливой
пустоты; после устранения причины сущности появляются сами, карточки
трогать не нужно.
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

# v1.4.5: HTTP-резерв списка камер ({bridge}/api/status)
_PROBE_TIMEOUT = ClientTimeout(total=10)
_FIRST_PROBE_DELAY = 8.0   # даём retain-MQTT прийти первым — не дублируем работу
_PROBE_INTERVAL = 60.0     # повтор опроса; камеры в аддоне меняются редко
_NOTIFY_AFTER_FAILS = 2    # две неудачные пробы подряд -> уведомление-чеклист
_NOTIFY_ID = "smart_access_cameras"


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Создаёт сущности камер по списку камер аддона (MQTT-статус + HTTP-резерв).

    MQTT-статус приходит с retain-флагом: и сразу после перезапуска HA (до
    того, как аддон что-то опубликует), и при изменении списка камер в
    опциях аддона. Новая камера в списке -> новая сущность без перезагрузки.
    v1.4.5: если MQTT-статуса нет, список берётся по HTTP с сайта аддона —
    сущности моста больше не зависят только от MQTT.
    """
    sync = _BridgeEntitySync(hass, async_add_entities)

    # статус мог прийти раньше платформы (retain-MQTT) — не теряем его
    state = hass.data.get(DOMAIN, {}).get("state") or {}
    sync.feed({"cameras": state.get("cameras") or []})

    sync.listen()
    sync.start_http_fallback(entry)

    if not sync.known:
        LOG.info(
            "Платформа камер моста Smart Access запущена: сущности появятся по "
            "MQTT-статусу аддона или по HTTP-опросу %s/api/status",
            _bridge_url(hass),
        )


def _bridge_url(hass: HomeAssistant) -> str:
    state = hass.data.get(DOMAIN, {}).get("state") or {}
    return str(state.get("bridge_url") or DEFAULT_BRIDGE_URL).rstrip("/")


class _BridgeEntitySync:
    """Создаёт/дополняет сущности камер моста; идемпотентно по имени камеры.

    Источники списка (v1.4.5): MQTT-статус аддона (feed, dispatcher
    SIGNAL_STATUS) и HTTP-опрос {bridge}/api/status (_resync_loop).
    Сущности не удаляются и не переименовываются — entity_id детерминирован
    именем камеры, карточки в примерах ссылаются на него.
    """

    def __init__(
        self, hass: HomeAssistant, async_add_entities: AddEntitiesCallback
    ) -> None:
        self.hass = hass
        self._add = async_add_entities
        self.known: set[str] = set()
        self.got_cameras = False   # хоть один источник отдал непустой список
        self._probe_fails = 0
        self._notified = False     # уведомление-чеклист: одно на запуск HA

    @callback
    def feed(self, payload: dict[str, Any]) -> None:
        """Список камер из MQTT-статуса (или стартового state интеграции)."""
        self._create((payload or {}).get("cameras") or [], "mqtt")

    def listen(self) -> None:
        """Подписка на MQTT-статус: новая камера -> новая сущность на лету."""
        async_dispatcher_connect(self.hass, SIGNAL_STATUS, self.feed)

    def start_http_fallback(self, entry: ConfigEntry) -> None:
        """v1.4.5: периодический HTTP-опрос /api/status — независимый от MQTT."""
        data = self.hass.data.setdefault(DOMAIN, {})
        old: asyncio.Task | None = data.get("camera_resync_task")
        if old is not None and not old.done():
            old.cancel()  # reload платформы — старый цикл останавливаем
        task = data["camera_resync_task"] = self.hass.async_create_task(
            self._resync_loop()
        )
        entry.async_on_unload(lambda: task.cancel())

    def _create(self, cams: list, source: str) -> None:
        cams = [c for c in (cams or []) if str((c or {}).get("name") or "").strip()]
        if cams:
            # непустой список из любого источника — значит «канал доставки жив»
            self.got_cameras = True
            self._probe_fails = 0
        bridge = _bridge_url(self.hass)
        new: list[SmartAccessBridgeCamera] = []
        for c in cams:
            name = str(c.get("name")).strip()
            if name in self.known:
                continue
            self.known.add(name)
            display = str(c.get("display") or "") or name
            new.append(SmartAccessBridgeCamera(bridge, name, display))
        if new:
            LOG.info(
                "Сущности камер моста Smart Access (%s): %s", source,
                ", ".join(f"camera.smart_access_bridge_{c._cam}" for c in new),
            )
            self._add(new)

    async def _resync_loop(self) -> None:
        await asyncio.sleep(_FIRST_PROBE_DELAY)
        while True:
            try:
                await self._probe_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — цикл переживает любые ошибки
                LOG.exception("HTTP-резерв списка камер моста: неожиданная ошибка")
            await asyncio.sleep(_PROBE_INTERVAL)

    async def _probe_once(self) -> None:
        bridge = _bridge_url(self.hass)
        ok, cams = await _fetch_bridge_cameras(self.hass, bridge)
        if cams:
            self._create(cams, "http")
            return
        if ok:
            # сайт аддона отвечает, но камер не вернул — в аддоне не настроено
            # ни одной камеры; уведомление тут только собьёт с толку
            if not self.known and not self.got_cameras:
                LOG.warning(
                    "Сайт аддона Smart Access отвечает, но /api/status не содержит "
                    "камер (bridge: %s) — камеры не настроены в аддоне", bridge
                )
            return
        self._probe_fails += 1
        if not self.known and not self.got_cameras:
            if self._probe_fails == 1:
                LOG.warning(
                    "Список камер моста пуст: MQTT-статус аддона не приходил, сайт "
                    "аддона %s не отвечает на /api/status (аддон остановлен? брокер "
                    "MQTT не настроен?) — повторяю каждые %d с",
                    bridge, int(_PROBE_INTERVAL),
                )
            if self._probe_fails >= _NOTIFY_AFTER_FAILS and not self._notified:
                self._notified = True
                _notify_no_cameras(self.hass, bridge)


async def _fetch_bridge_cameras(
    hass: HomeAssistant, bridge: str
) -> tuple[bool, list[dict[str, str]]]:
    """Список камер с сайта аддона: GET {bridge}/api/status -> (ok, [{name, display}]).

    ok=False — сайт недоступен (HTTP-ошибка/таймаут/не JSON); ok=True с
    пустым списком — аддон жив, но камер в нём не настроено.
    """
    try:
        resp = await async_get_clientsession(hass).get(
            f"{bridge}/api/status", timeout=_PROBE_TIMEOUT
        )
        if not resp.ok:
            LOG.debug("Мост: /api/status: HTTP %s", resp.status)
            return False, []
        data = await resp.json(content_type=None)
    except (ClientError, asyncio.TimeoutError, ValueError) as exc:
        LOG.debug("Мост: /api/status недоступен: %s", exc)
        return False, []
    items = (data or {}).get("bridge")
    if not isinstance(items, dict):
        return True, []
    out: list[dict[str, str]] = []
    for name, info in items.items():
        name = str(name or "").strip()
        if not name:
            continue
        info = info if isinstance(info, dict) else {}
        out.append({"name": name, "display": str(info.get("display") or "")})
    return True, out


def _notify_no_cameras(hass: HomeAssistant, bridge: str) -> None:
    """v1.4.5: вместо молчаливой пустоты — уведомление-чеклист (одно на запуск).

    ФИКС «Этот объект недоступен»: без списка камер сущности моста не
    создаются, и карточки с camera.smart_access_bridge_* не работают.
    """
    hass.async_create_task(hass.services.async_call(
        "persistent_notification", "create",
        {
            "title": "Smart Access: камеры моста не найдены",
            "message": (
                "Сущности камер моста (camera.smart_access_bridge_…) не созданы: "
                "MQTT-статус аддона не приходил, сайт аддона не отвечает на "
                f"{bridge}/api/status.\n\n"
                "Проверьте по порядку:\n"
                "1. Аддон Smart Access Core запущен: Настройки → Дополнения → "
                "Smart Access Core → вкладка «Статус» (строка MQTT — «подключён»).\n"
                "2. Брокер MQTT настроен в HA: Настройки → Устройства и службы → "
                "есть интеграция Mosquitto broker / MQTT.\n"
                "3. Интеграция Smart Access обновлена в HACS и HA ПЕРЕЗАПУЩЕН "
                "(Настройки → Система → Перезапустить).\n\n"
                "После устранения причины сущности появятся сами (до 2 минут) — "
                "карточки дашборда трогать не нужно."
            ),
            "notification_id": _NOTIFY_ID,
        },
        blocking=False,
    ))


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
        /core/api/camera_proxy_stream/<entity>): отдаём браузеру ровно то,
        что мост отдаёт Frigate, с тем же content-type и границей.
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
