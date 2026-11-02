"""v1.2.1: мост × камеры без рабочих одиночных снимков (универсальная RTSP-камера).

Баг (реальный случай, kalitka): camera.kalitka — Generic Camera (RTSP Hikvision
без still_image_url). Одиночный camera_proxy в HA ради одного кадра запускает
извлечение кадра из потока (ffmpeg + HLS) и не укладывается в таймаут → у
моста «сбоев: 440, fps: 0», Frigate получает серую заглушку (~5.4 КБ/кадр,
latest.webp ~1 КБ) и не находит объектов, при этом карточка в HA играет
(она читает ПОТОК, а не снимки). Причины сбоя писались на DEBUG — журнал
молчал.

Фикс: запасной путь get_camera_image_ex — первый кадр из потока HA
(/core/api/camera_proxy_stream), бэкофф опроса при стойких сбоях (без
наложения опросов одной камеры и без торможения соседних), телеметрия
источника кадра + подсказка в панели (rtsp_url / still_image_url).

Запуск:  python3 -m pytest smart-access-core/tests/ -v
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest
from aiohttp import web

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

from app import config as cfg            # noqa: E402
from app import ha_api                   # noqa: E402
from app.bridge import Bridge            # noqa: E402

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
# 1. Парсер MJPEG и запасной путь

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


# --------------------------------------------------------------------- #
# 2. Мост: кадр через запасной путь + подсказки

def test_bridge_gets_frames_via_fallback_and_hints():
    async def run():
        ha = _FakeHA()
        runner = _start_ha(ha)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 18234).start()
        old_base, old_token = cfg.supervisor_base, cfg.SUPERVISOR_TOKEN
        cfg.supervisor_base = lambda: "http://127.0.0.1:18234"  # type: ignore[assignment]
        cfg.SUPERVISOR_TOKEN = "t"
        try:
            b = Bridge(fps=1.5)
            b.set_cameras([{"name": "kalitka", "ha_entity": "camera.kalitka"}])
            b.clients["kalitka"] = 1
            loop = asyncio.get_running_loop()
            await b._fetch_one(loop, "kalitka", "camera.kalitka")
            return (b.latest.get("kalitka"), b.source.get("kalitka"),
                    b.failures.get("kalitka"))
        finally:
            cfg.supervisor_base, cfg.SUPERVISOR_TOKEN = old_base, old_token
            await runner.cleanup()

    latest, source, failures = asyncio.run(run())
    assert latest == JPEG_A, "мост не получил кадр из запасного пути"
    assert source == "stream"
    assert failures == 0


def test_hint_degraded_stream_mode():
    b = Bridge(fps=1.5)
    b.set_cameras([{"name": "kalitka", "ha_entity": "camera.kalitka"}])
    b.source["kalitka"] = "stream"
    b.failures["kalitka"] = 0
    h = b.hint("kalitka")
    assert "потока HA" in h and "rtsp_url" in h


def test_hint_persistent_failures_names_both_fixes():
    b = Bridge(fps=1.5)
    b.set_cameras([{"name": "kalitka", "ha_entity": "camera.kalitka"}])
    b.failures["kalitka"] = 440
    b.last_error["kalitka"] = "camera_proxy HTTP 500"
    h = b.hint("kalitka")
    assert "rtsp_url" in h and "still_image_url" in h
    # до 30 сбоев подряд подсказки ещё нет (это мог быть разовый сбой)
    b.failures["kalitka"] = 5
    assert b.hint("kalitka") == ""


# --------------------------------------------------------------------- #
# 3. Бэкофф и отсутствие наложений

def test_backoff_bounds_retry_rate(monkeypatch):
    """Стойкий сбой: повторный опрос отложен бэкоффом, а не каждые 0.67 c."""
    b = Bridge(fps=1.5)
    b.set_cameras([{"name": "kalitka", "ha_entity": "camera.kalitka"}])
    b.clients["kalitka"] = 1
    calls = {"n": 0}

    async def fake_fetch(self, loop, name, entity):
        calls["n"] += 1
        b.failures[name] = 5
        b._next_try[name] = time.time() + 5.0

    monkeypatch.setattr(Bridge, "_fetch_one", fake_fetch)

    async def run():
        loop = asyncio.get_running_loop()
        t0 = time.time()
        while time.time() - t0 < 1.0:
            await b._tick(loop)
            await asyncio.sleep(0.02)

    asyncio.run(run())
    assert calls["n"] == 1, \
        f"за 1 c при бэкоффе 5 c должен быть 1 опрос, было {calls['n']}"


def test_stuck_camera_does_not_slow_neighbors(monkeypatch):
    """kalitka «висит» 1.2 c на опросе — prud продолжает получать кадры."""
    b = Bridge(fps=1.5)
    b.set_cameras([
        {"name": "prud", "ha_entity": "camera.prud"},
        {"name": "kalitka", "ha_entity": "camera.kalitka"},
    ])
    b.clients["prud"] = 1
    b.clients["kalitka"] = 1
    calls = {"prud": 0, "kalitka": 0}
    kalitka_busy = {"active": 0, "max": 0}

    async def fake_fetch(self, loop, name, entity):
        calls[name] += 1
        if name == "kalitka":
            kalitka_busy["active"] += 1
            kalitka_busy["max"] = max(kalitka_busy["max"], kalitka_busy["active"])
            await asyncio.sleep(1.2)
            kalitka_busy["active"] -= 1
            b.failures[name] = 1
            b._next_try[name] = time.time() + 5.0

    monkeypatch.setattr(Bridge, "_fetch_one", fake_fetch)

    async def run():
        loop = asyncio.get_running_loop()
        t0 = time.time()
        while time.time() - t0 < 1.5:
            await b._tick(loop)
            await asyncio.sleep(0.05)

    asyncio.run(run())
    assert calls["prud"] >= 3, \
        f"здоровая камера должна опрашиваться каждый тик, было {calls['prud']}"
    assert kalitka_busy["max"] == 1, "опросы застрявшей камеры наложились"
