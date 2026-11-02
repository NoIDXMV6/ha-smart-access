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

    def set_cameras(self, cams: list[dict]) -> None:
        """Список из конфига: [{'name', 'ha_entity'}, ...] (перезаписывает карту)."""
        self.entities = {c["name"]: c["ha_entity"] for c in cams}
        for name in self.entities:
            self.latest.setdefault(name, _placeholder_jpeg())
            self.failures.setdefault(name, 0)
            self.fps.setdefault(name, 0.0)
            self.frames.setdefault(name, 0)
        for gone in list(self.fps):
            if gone not in self.entities:
                self.fps.pop(gone, None)
                self.last_frame_ts.pop(gone, None)
                self.frames.pop(gone, None)
                self.failures.pop(gone, None)
                self.latest.pop(gone, None)

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
        """Один опрос камеры + телеметрия. Не бросает исключений."""
        try:
            jpeg = await loop.run_in_executor(None, ha_api.get_camera_image, entity)
        except Exception:  # noqa: BLE001
            jpeg = None
        if jpeg:
            self.latest[name] = jpeg
            self.failures[name] = 0
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
            # после 5 неудач подряд — отдаём заглушку и логируем редко
            if self.failures[name] in (1, 30, 300):
                LOG.warning("Мост: камера %s (%s) не отвечает (%s подряд)",
                            name, entity, self.failures[name])

    def _kick(self, name: str, entity: str) -> None:
        """Фоновый внеочередной опрос одной камеры (не ждёт ближайшего тика)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._fetch_one(loop, name, entity))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _tick(self, loop: asyncio.AbstractEventLoop) -> None:
        # v1.1.13: камеры с подписчиками опрашиваются ПАРАЛЛЕЛЬНО.
        # Раньше опрос шёл последовательно: пока тянется медленный облачный
        # снимок одной камеры (до 12 с), остальные ждут, и с ростом числа
        # камер реальный fps каждой падал до долей кадра в секунду
        # (замер на стенде: 3 камеры с задержками 1.0/0.5/0.9 с — реальный
        # fps prud 0.3 при отдаче клиенту 1.5 fps).
        pending = [
            self._fetch_one(loop, name, entity)
            for name, entity in self.entities.items()
            if self.clients.get(name, 0) > 0
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

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
