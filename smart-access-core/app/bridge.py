"""Мост: снимки камер HA (включая облачные А.М.Видео/ЕГДС) -> локальный MJPEG.

Frigate не умеет читать облачные потоки А.М.Видео (WSS MPEG1-TS) и ЕГДС
(авторизованные снимки). Этот мост тянет кадры через camera_proxy самого HA
(авторизация — SUPERVISOR_TOKEN аддона) и раздаёт их как обычный
MJPEG-поток http://127.0.0.1:8100/mjpeg/<name>, который Frigate берёт как:
    inputs:
      - path: http://127.0.0.1:8100/mjpeg/govorovo
        input_args: -f mjpeg

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
        self.failures: dict[str, int] = {}

    def set_cameras(self, cams: list[dict]) -> None:
        """Список из конфига: [{'name', 'ha_entity'}, ...] (перезаписывает карту)."""
        self.entities = {c["name"]: c["ha_entity"] for c in cams}
        for name in self.entities:
            self.latest.setdefault(name, _placeholder_jpeg())
            self.failures.setdefault(name, 0)

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

    async def _tick(self, loop: asyncio.get_running_loop) -> None:
        # Кадр нужен только для камер с активными подписчиками (Frigate)
        for name, entity in self.entities.items():
            if self.clients.get(name, 0) <= 0:
                continue
            jpeg = await loop.run_in_executor(None, ha_api.get_camera_image, entity)
            if jpeg:
                self.latest[name] = jpeg
                self.failures[name] = 0
            else:
                self.failures[name] = self.failures.get(name, 0) + 1
                # после 5 неудач подряд — отдаём заглушку и логируем редко
                if self.failures[name] in (1, 30, 300):
                    LOG.warning("Мост: камера %s (%s) не отвечает (%s подряд)",
                                name, entity, self.failures[name])

    # ---------- aiohttp ----------

    async def handle_mjpeg(self, request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        if name not in self.entities:
            return web.json_response({"error": "unknown camera"}, status=404)
        self.clients[name] = self.clients.get(name, 0) + 1
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
        return resp

    async def handle_latest(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        jpeg = self.latest.get(name) or _placeholder_jpeg()
        return web.Response(body=jpeg, content_type="image/jpeg")

    def register(self, app: web.Application) -> None:
        app.router.add_get("/mjpeg/{name}", self.handle_mjpeg)
        app.router.add_get("/latest/{name}", self.handle_latest)
