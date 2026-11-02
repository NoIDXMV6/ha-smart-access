"""Мост: снимки камер HA (включая облачные А.М.Видео/ЕГДС) -> локальный MJPEG.

Frigate не умеет читать облачные потоки А.М.Видео (WSS MPEG1-TS) и ЕГДС
(авторизованные снимки). Этот мост тянет кадры через camera_proxy самого HA
(авторизация — SUPERVISOR_TOKEN аддона) и раздаёт их как обычный
MJPEG-поток, который Frigate берёт как:
    inputs:
      - path: http://172.30.32.1:8100/mjpeg/govorovo
        input_args: -f mjpeg

АДРЕС ДЛЯ FRIGATE — http://172.30.32.1:8100 (не 127.0.0.1!): Frigate —
мостовой аддон (сеть hassio), для него 127.0.0.1 — его собственный
контейнер; наш аддон слушает на хосте, а хост из мостовой сети — шлюз
172.30.32.1. Сайт на :8100 поднимается в main.start_bridge_site
(до v1.1.11 порт вообще не слушался — «Connection refused» у ffmpeg).

Универсально: работает с ЛЮБОЙ камерой HA (облачные шлагбаумы, домофония
Hikvision, локальные RTSP-камеры — если у тех есть camera entity).

Отказоустойчивость: при недоступности камеры поток отдаёт «заглушку» и
продолжает попытки; Frigate переживает паузы в потоке.
"""
from __future__ import annotations

import asyncio
import logging
import time

import cv2
import numpy as np
from aiohttp import web

from . import ha_api

LOG = logging.getLogger("smart_access.bridge")

STUB = None  # lazy placeholder jpeg


def _placeholder_jpeg() -> bytes:
    global STUB
    if STUB is None:
        img = np.full((360, 640, 3), 24, np.uint8)
        cv2.putText(img, "camera offline", (170, 190), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (90, 90, 90), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        STUB = buf.tobytes() if ok else b""
    return STUB


class Bridge:
    """Опрос камер HA и раздача MJPEG-потоков."""

    def __init__(self, fps: float = 1.5) -> None:
        self.interval = 1.0 / max(0.2, min(5.0, fps))
        self.latest: dict[str, bytes] = {}       # name -> jpeg
        self.clients: dict[str, int] = {}        # name -> количество подписчиков
        self.entities: dict[str, str] = {}       # name -> ha entity
        self._task: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()      # фоновые задачи (не дать сборщику их съесть)
        self.failures: dict[str, int] = {}
        # телеметрия реального потока (видна в UI: «fps» рядом со «сбоев»)
        self.fps: dict[str, float] = {}          # name -> EMA кадров/с
        self.last_frame_ts: dict[str, float] = {}  # name -> время последнего кадра
        self.frames: dict[str, int] = {}         # name -> всего получено кадров
        # v1.2.1: КАК получен кадр и ПОЧЕМУ его нет — видны в панели
        self.source: dict[str, str] = {}         # 'proxy' | 'stream' | ''
        self.last_error: dict[str, str] = {}     # последняя причина «сбоев»
        # v1.2.1: опрос без наложений и с бэкоффом при стойких сбоях.
        # Раньше камера с вечно таймаутящим снимком (калитка: camera_proxy
        # универсальной камеры не работает в принципе) тянулась каждые
        # 0.67 c по 12 c, заставляя HA раз за разом запускать извлечение
        # кадра из потока, а тик цикла жил столько же, сколько самый
        # медленный опрос — соседние камеры недополучали кадры.
        self._busy: set[str] = set()             # опрос которых сейчас идёт
        self._next_try: dict[str, float] = {}    # не раньше какого времени опрашивать

    def set_cameras(self, cams: list[dict]) -> None:
        """Список из конфига: [{'name', 'ha_entity'}, ...] (перезаписывает карту)."""
        self.entities = {c["name"]: c["ha_entity"] for c in cams}
        for name in self.entities:
            self.latest.setdefault(name, _placeholder_jpeg())
            self.failures.setdefault(name, 0)
            self.fps.setdefault(name, 0.0)
            self.frames.setdefault(name, 0)
            self.source.setdefault(name, "")
            self.last_error.setdefault(name, "")
        for gone in list(self.fps):
            if gone not in self.entities:
                self.fps.pop(gone, None)
                self.last_frame_ts.pop(gone, None)
                self.frames.pop(gone, None)
                self.failures.pop(gone, None)
                self.latest.pop(gone, None)
                self.source.pop(gone, None)
                self.last_error.pop(gone, None)
                self._next_try.pop(gone, None)

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:8100/mjpeg/{name}"

    def frigate_url(self, name: str) -> str:
        """Адрес потока С ТОЧКИ ЗРЕНИЯ Frigate.

        Аддон Frigate работает в мостовой Docker-сети hassio (172.30.32.0/23),
        поэтому 127.0.0.1 для него — его собственный контейнер. Аддон Smart
        Access слушает на хосте (host_network), а хост из мостовой сети — это
        шлюз 172.30.32.1 (константа Supervisor: DOCKER_IPV4_NETWORK_MASK[1]).
        """
        return f"http://172.30.32.1:8100/mjpeg/{name}"

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                await self._tick(loop)
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001
                LOG.exception("Мост: сбой цикла: %s", exc)
            await asyncio.sleep(self.interval)

    async def _fetch_one(self, loop: asyncio.AbstractEventLoop,
                         name: str, entity: str) -> None:
        """Один опрос камеры + телеметрия. Не бросает исключений.

        v1.2.1: get_camera_image_ex — с запасным путём «первый кадр из
        потока HA» для камер без рабочих одиночных снимков (универсальная
        камера с RTSP без still_image_url) и с внятной причиной сбоя в журнале.
        """
        try:
            jpeg, source, err = await loop.run_in_executor(
                None, ha_api.get_camera_image_ex, entity)
        except Exception:  # noqa: BLE001
            jpeg, source, err = None, "", "опрос упал"
        if jpeg:
            self.latest[name] = jpeg
            self.failures[name] = 0
            self.source[name] = source
            self.last_error[name] = ""
            self._next_try[name] = 0.0
            now = time.time()
            prev = self.last_frame_ts.get(name)
            self.last_frame_ts[name] = now
            self.frames[name] = self.frames.get(name, 0) + 1
            if prev:
                dt = max(0.05, now - prev)
                inst = 1.0 / dt
                # EMA: сглаживает скачки, но быстро видит «0 fps» при сбое
                self.fps[name] = round(0.7 * self.fps.get(name, inst) + 0.3 * inst, 2)
        else:
            self.failures[name] = self.failures.get(name, 0) + 1
            self.source[name] = ""
            self.last_error[name] = err or "нет кадра"
            # Бэкофф: 0.67 c → 1.3 → 2.7 → 5 c (кап), чтобы не долбить HA
            # извлечением кадра из потока каждые 0.67 c при стойком сбое
            backoff = min(5.0, self.interval * (2 ** min(max(self.failures[name] - 1, 0), 4)))
            self._next_try[name] = time.time() + backoff
            if self.failures[name] in (1, 30, 300):
                LOG.warning("Мост: камера %s (%s) не отвечает — %s (%s подряд)",
                            name, entity, self.last_error[name], self.failures[name])

    def _kick(self, name: str, entity: str) -> None:
        """Фоновый внеочередной опрос одной камеры (не ждёт ближайшего тика)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if name in self._busy:
            return  # опрос уже идёт — дубликат не нужен
        self._busy.add(name)
        task = loop.create_task(self._guarded_fetch(loop, name, entity))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _guarded_fetch(self, loop: asyncio.AbstractEventLoop,
                             name: str, entity: str) -> None:
        """Опрос с освобождением слота _busy в любом исходе."""
        try:
            await self._fetch_one(loop, name, entity)
        finally:
            self._busy.discard(name)

    async def _tick(self, loop: asyncio.AbstractEventLoop) -> None:
        # v1.1.13: камеры с подписчиками опрашиваются ПАРАЛЛЕЛЬНО.
        # v1.2.1: опросы уходят в фон без ожидания самого медленного —
        # застрявшая камера (12-секундный таймаут снимка) больше не тормозит
        # тики соседних; наложений нет (_busy), стойкие сбои — с бэкоффом.
        now = time.time()
        for name, entity in self.entities.items():
            if self.clients.get(name, 0) <= 0:
                continue
            if name in self._busy or now < self._next_try.get(name, 0.0):
                continue
            self._busy.add(name)
            task = loop.create_task(self._guarded_fetch(loop, name, entity))
            self._bg.add(task)
            task.add_done_callback(self._bg.discard)

    def hint(self, name: str) -> str:
        """Подсказка для панели, когда мост не может нормально получить кадры."""
        failures = self.failures.get(name, 0)
        if self.source.get(name) == "stream" and failures == 0:
            return ("Кадры идут запасным путём — из потока HA (медленно): одиночные "
                    "снимки camera_proxy для этой камеры не работают. Обычно так себя "
                    "ведёт универсальная камера (Generic Camera) с RTSP без «URL кадра». "
                    "Для полной скорости задайте камере rtsp_url (Frigate читает её "
                    "напрямую) или добавьте камере в HA still_image_url.")
        if failures >= 30:
            return (f"camera_proxy не отвечает постоянно ({self.last_error.get(name, '')}). "
                    "Локальная RTSP-камера → впишите rtsp_url в настройках этой камеры "
                    "(лучший путь, мост не нужен). Универсальная камера без still_image_url "
                    "→ добавьте ей «URL кадра» (например http://admin:пароль@IP/"
                    "ISAPI/Streaming/channels/101/picture у Hikvision).")
        return ""

    # ---------- aiohttp ----------

    async def handle_mjpeg(self, request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        if name not in self.entities:
            # частая причина «404» — имя в конфиге Frigate не совпадает с name
            # камеры в Smart Access (например, поток govorovo при камере a_m_video)
            LOG.warning("Мост: запрос НЕИЗВЕСТНОЙ камеры «%s» (известны: %s)",
                        name, ", ".join(sorted(self.entities)) or "нет")
            return web.json_response({"error": "unknown camera"}, status=404)
        entity = self.entities[name]
        first_client = self.clients.get(name, 0) == 0
        self.clients[name] = self.clients.get(name, 0) + 1
        if first_client:
            # первый подписчик: кадр тянем сразу, а не с ближайшего тика —
            # Frigate получает живую картинку на секунды раньше
            self._kick(name, entity)
        started = time.time()
        LOG.info("Мост: клиент %s подключился к «%s» (%s), подписчиков: %d",
                 request.remote or "?", name, entity, self.clients[name])
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=smartaccess",
                "Cache-Control": "no-cache, no-store",
                "Connection": "close",
            },
        )
        await resp.prepare(request)
        try:
            while True:
                jpeg = self.latest.get(name) or _placeholder_jpeg()
                await resp.write(
                    b"--smartaccess\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                )
                await asyncio.sleep(self.interval)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.clients[name] = max(0, self.clients.get(name, 1) - 1)
            LOG.info("Мост: клиент %s отключился от «%s» (%.1f с, подписчиков: %d)",
                     request.remote or "?", name, time.time() - started,
                     self.clients[name])
        return resp

    async def handle_latest(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        jpeg = self.latest.get(name) or _placeholder_jpeg()
        return web.Response(body=jpeg, content_type="image/jpeg")

    def register(self, app: web.Application) -> None:
        app.router.add_get("/mjpeg/{name}", self.handle_mjpeg)
        app.router.add_get("/latest/{name}", self.handle_latest)
