"""v1.2.1 (актуальная часть в v1.3.0): камеры без рабочих одиночных снимков.

Баг (реальный случай, kalitka): camera.kalitka — Generic Camera (RTSP Hikvision
без still_image_url). Одиночный camera_proxy в HA ради одного кадра запускает
извлечение кадра из потока (ffmpeg + HLS) и не укладывается в таймаут → у
моста «сбоев: 440, fps: 0», Frigate получает серую заглушку, карточка при
этом играет (она читает ПОТОК, а не снимки).

В v1.2.1 добавлены: парсер MJPEG-потока HA, запасной путь
get_camera_image_ex («первый кадр из потока») для ручных действий и причины
сбоев в журнале. В v1.3.0 мост переведён на ПОСТОЯННОЕ соединение к потоку
(одна сессия вместо опроса) — поведение моста проверяет
test_v130_single_consumer.py; здесь остаются контракты ha_api.

Запуск:  python3 -m pytest smart-access-core/tests/ -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from aiohttp import web

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

from app import config as cfg            # noqa: E402
from app import ha_api                   # noqa: E402

# Минимальные валидные JPEG (SOI…EOI), различимые по телу
JPEG_A = b"\xff\xd8\xff\xe0" + b"A" * 64 + b"\xff\xd9"
JPEG_B = b"\xff\xd8\xff\xe0" + b"B" * 64 + b"\xff\xd9"

BOUNDARY = "v121boundary"


def _multipart(frame: bytes) -> bytes:
    return (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame)}\r\n\r\n".encode() + frame + b"\r\n")


class _FakeHA:
    """HA, у которого одиночный снимок сломан, а поток жив (универсальная камера)."""

    def __init__(self) -> None:
        self.stream_calls = 0

    async def proxy(self, request: web.Request) -> web.Response:
        return web.json_response({"error": "stream not ready"}, status=500)

    async def stream(self, request: web.Request) -> web.StreamResponse:
        self.stream_calls += 1
        resp = web.StreamResponse(headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        })
        await resp.prepare(request)
        for f in (JPEG_A, JPEG_B, JPEG_A):
            await resp.write(_multipart(f))
            await asyncio.sleep(0.02)
        return resp


class _OkHA:
    """HA, у которого одиночный снимок работает (камера tatyanin_park)."""

    def __init__(self) -> None:
        self.stream_calls = 0

    async def proxy(self, request: web.Request) -> web.Response:
        return web.Response(body=JPEG_B, content_type="image/jpeg")

    async def stream(self, request: web.Request) -> web.Response:
        self.stream_calls += 1
        return web.Response(body=JPEG_A, content_type="image/jpeg")


def _start_ha(ha) -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/core/api/camera_proxy/{e}", ha.proxy)
    app.router.add_get("/core/api/camera_proxy_stream/{e}", ha.stream)
    runner = web.AppRunner(app)
    return runner


# --------------------------------------------------------------------- #
# 1. Парсер MJPEG и запасной путь (контракты ha_api)

def test_stream_frame_parses_multipart_and_returns_first_frame():
    async def run():
        ha = _FakeHA()
        runner = _start_ha(ha)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18231).start()
        old_base, old_token = cfg.supervisor_base, cfg.SUPERVISOR_TOKEN
        cfg.supervisor_base = lambda: "http://127.0.0.1:18231"  # type: ignore[assignment]
        cfg.SUPERVISOR_TOKEN = "t"
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, ha_api.get_camera_stream_frame, "camera.kalitka")
        finally:
            cfg.supervisor_base, cfg.SUPERVISOR_TOKEN = old_base, old_token
            await runner.cleanup()

    jpg, err = asyncio.run(run())
    assert jpg == JPEG_A, f"первый кадр не распознан: {err}"
    assert err == ""


def test_image_ex_falls_back_to_stream_when_proxy_broken():
    async def run():
        ha = _FakeHA()
        runner = _start_ha(ha)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18232).start()
        old_base, old_token = cfg.supervisor_base, cfg.SUPERVISOR_TOKEN
        cfg.supervisor_base = lambda: "http://127.0.0.1:18232"  # type: ignore[assignment]
        cfg.SUPERVISOR_TOKEN = "t"
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, ha_api.get_camera_image_ex, "camera.kalitka")
        finally:
            cfg.supervisor_base, cfg.SUPERVISOR_TOKEN = old_base, old_token
            await runner.cleanup()

    jpg, source, err = asyncio.run(run())
    assert jpg == JPEG_A and source == "stream" and err == ""


def test_image_ex_prefers_proxy_when_it_works():
    async def run():
        ha = _OkHA()
        runner = _start_ha(ha)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18233).start()
        old_base, old_token = cfg.supervisor_base, cfg.SUPERVISOR_TOKEN
        cfg.supervisor_base = lambda: "http://127.0.0.1:18233"  # type: ignore[assignment]
        cfg.SUPERVISOR_TOKEN = "t"
        try:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None, ha_api.get_camera_image_ex, "camera.tat")
            return res, ha.stream_calls
        finally:
            cfg.supervisor_base, cfg.SUPERVISOR_TOKEN = old_base, old_token
            await runner.cleanup()

    (jpg, source, err), stream_calls = asyncio.run(run())
    assert jpg == JPEG_B and source == "proxy" and err == ""
    assert stream_calls == 0, "запасной путь не должен запускаться при рабочем proxy"


def test_snapshot_ex_is_proxy_only():
    """v1.3.0: get_camera_snapshot_ex НЕ открывает поток (иначе мост снова
    плодит сессии «ради одного кадра»)."""

    async def run():
        ha = _FakeHA()
        runner = _start_ha(ha)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18235).start()
        old_base, old_token = cfg.supervisor_base, cfg.SUPERVISOR_TOKEN
        cfg.supervisor_base = lambda: "http://127.0.0.1:18235"  # type: ignore[assignment]
        cfg.SUPERVISOR_TOKEN = "t"
        try:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None, ha_api.get_camera_snapshot_ex, "camera.kalitka")
            return res, ha.stream_calls
        finally:
            cfg.supervisor_base, cfg.SUPERVISOR_TOKEN = old_base, old_token
            await runner.cleanup()

    (jpg, err), stream_calls = asyncio.run(run())
    assert jpg is None and err, "сломанный снимок должен вернуть ошибку"
    assert stream_calls == 0, "snapshot_ex не должен открывать поток"
