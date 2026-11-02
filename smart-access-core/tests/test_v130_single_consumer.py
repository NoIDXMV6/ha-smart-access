"""v1.3.0: АРХИТЕКТУРА «ОДИН ПОЛУЧАТЕЛЬ НА КАМЕРУ».

Первоопричина нестабильности (prud: сбоев 139, fps 0; kalitka: сбоев 440):
мост опрашивал камеры одиночными снимками, а «снимок» для этих камер —
полноценная сессия потока (А.М.Видео: новая WSS-сессия на каждый кадр;
универсальная RTSP-камера: новый ffmpeg/RTSP на каждый кадр). 1.5 опроса/с =
полторы новые сессии в секунду — камеры и облачные прокси захлёбывались.

Теперь у камеры ОДИН получатель — мост:
  * rtsp   — rtsp_url задан: мост сам читает RTSP (TCP);
  * proxy  — быстрые одиночные снимки (не порождают сессий);
  * stream — ОДНО постоянное соединение к потоку HA (как открытая карточка).
Плодить сессии больше некому: Frigate и HA читают мост.

Запуск:  python3 -m pytest smart-access-core/tests/ -v
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
import types
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web, ClientSession

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

from app import config as cfg            # noqa: E402
from app import ha_api                   # noqa: E402
from app.bridge import Bridge            # noqa: E402
from app.frigate_config import build_frigate_config  # noqa: E402

JPEG_A = b"\xff\xd8\xff\xe0" + b"A" * 64 + b"\xff\xd9"
JPEG_B = b"\xff\xd8\xff\xe0" + b"B" * 64 + b"\xff\xd9"
BOUNDARY = "v130boundary"


def _multipart(frame: bytes) -> bytes:
    return (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame)}\r\n\r\n".encode() + frame + b"\r\n")


class _StreamHA:
    """HA: одиночный снимок сломан, поток живой и бесконечный (kalitka/prud)."""

    def __init__(self) -> None:
        self.stream_opens = 0

    async def proxy(self, request: web.Request) -> web.Response:
        return web.json_response({"error": "stream not ready"}, status=500)

    async def stream(self, request: web.Request) -> web.StreamResponse:
        self.stream_opens += 1
        resp = web.StreamResponse(headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        })
        await resp.prepare(request)
        try:
            i = 0
            while True:
                await resp.write(_multipart(JPEG_A if i % 2 == 0 else JPEG_B))
                i += 1
                await asyncio.sleep(0.05)
        except (ConnectionResetError, ConnectionError):
            pass
        return resp


class _SnapshotHA:
    """HA: одиночный снимок быстрый и рабочий (tatyanin_park / ЕГДС)."""

    def __init__(self) -> None:
        self.calls = 0

    async def proxy(self, request: web.Request) -> web.Response:
        self.calls += 1
        return web.Response(body=JPEG_B, content_type="image/jpeg")

    async def stream(self, request: web.Request) -> web.Response:
        return web.Response(body=JPEG_A, content_type="image/jpeg")


class _SwitchingHA:
    """HA: снимок работает N раз, потом ломается НАВСЕГДА; поток живой."""

    def __init__(self, ok_times: int = 1) -> None:
        self.ok_times = ok_times
        self.calls = 0
        self.stream_opens = 0

    async def proxy(self, request: web.Request) -> web.Response:
        self.calls += 1
        if self.calls <= self.ok_times:
            return web.Response(body=JPEG_B, content_type="image/jpeg")
        return web.json_response({"error": "boom"}, status=500)

    async def stream(self, request: web.Request) -> web.StreamResponse:
        self.stream_opens += 1
        resp = web.StreamResponse(headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        })
        await resp.prepare(request)
        try:
            while True:
                await resp.write(_multipart(JPEG_A))
                await asyncio.sleep(0.05)
        except (ConnectionResetError, ConnectionError):
            pass
        return resp


@asynccontextmanager
async def _ha_server(ha, port: int):
    """Поднимает фейковый HA и подменяет supervisor_base на него."""
    old_base, old_token = cfg.supervisor_base, cfg.SUPERVISOR_TOKEN
    cfg.SUPERVISOR_TOKEN = "t"
    cfg.supervisor_base = lambda: f"http://127.0.0.1:{port}"  # type: ignore[assignment]
    app = web.Application()
    app.router.add_get("/core/api/camera_proxy/{e}", ha.proxy)
    app.router.add_get("/core/api/camera_proxy_stream/{e}", ha.stream)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        yield runner
    finally:
        cfg.supervisor_base, cfg.SUPERVISOR_TOKEN = old_base, old_token
        await runner.cleanup()


async def _wait(predicate, timeout: float = 8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


# --------------------------------------------------------------------- #
# 1. Выбор режима получателя

def test_mode_stream_single_persistent_session():
    """Камера без снимков: режим stream, поток открыт ОДИН раз, кадры идут."""

    async def run():
        ha = _StreamHA()
        async with _ha_server(ha, 18301):
            b = Bridge(fps=5.0)
            b.set_cameras([{"name": "prud", "ha_entity": "camera.prud"}])
            b.clients["prud"] = 1
            b._kick("prud")
            ok = await _wait(lambda: b.mode.get("prud") == "stream")
            assert ok, f"режим не стал stream: {b.mode} ({b.last_error})"
            got = await _wait(lambda: b.frames.get("prud", 0) >= 3)
            assert got, "мост не получает кадры из постоянного потока"
            opens = ha.stream_opens
            await asyncio.sleep(0.6)
            assert b.frames.get("prud", 0) >= 5, "кадры должны приходить непрерывно"
            assert ha.stream_opens == opens, \
                "мост переоткрывает поток — сессии должны жить, а не наслаиваться"
            assert b.source.get("prud") == "stream"
            await b.stop()

    asyncio.run(run())


def test_mode_proxy_when_snapshots_cheap():
    """Быстрые снимки: режим proxy, поток HA не открывается вовсе."""

    async def run():
        ha = _SnapshotHA()
        async with _ha_server(ha, 18302):
            b = Bridge(fps=5.0)
            b.set_cameras([{"name": "tat", "ha_entity": "camera.tat"}])
            b.clients["tat"] = 1
            b._kick("tat")
            ok = await _wait(lambda: b.mode.get("tat") == "proxy")
            assert ok, f"режим не стал proxy: {b.mode}"
            got = await _wait(lambda: b.frames.get("tat", 0) >= 2)
            assert got, "снимки не опрашиваются"
            await b.stop()

    asyncio.run(run())


def test_sticky_switch_proxy_to_stream_after_3_failures():
    """Снимок работал, потом сломался: после 3 неудач мост переводит камеру
    на постоянный поток и НЕ возвращается к долбёжке снимками."""

    async def run():
        ha = _SwitchingHA(ok_times=1)
        async with _ha_server(ha, 18303):
            b = Bridge(fps=5.0)
            b.set_cameras([{"name": "kalitka", "ha_entity": "camera.kalitka"}])
            b.clients["kalitka"] = 1
            b._kick("kalitka")
            ok = await _wait(lambda: b.mode.get("kalitka") == "stream", timeout=10)
            assert ok, f"камера не переведена на stream: {b.mode}"
            got = await _wait(lambda: b.frames.get("kalitka", 0) >= 2)
            assert got, "после переключения кадры не идут"
            # снимки больше не дёргаются
            calls_after = ha.calls
            await asyncio.sleep(0.5)
            assert ha.calls == calls_after, "мост продолжает ломить снимками"
            await b.stop()

    asyncio.run(run())


def test_mode_rtsp_without_ha_probe():
    """rtsp_url задан: мост сам читает RTSP (режим rtsp), HA не опрашивается.

    Соединение к фейковому адресу не удастся — но режим должен выставиться
    сразу, а попытки — идти с бэкоффом, без обращения к camera_proxy.
    """

    async def run():
        ha = _SnapshotHA()
        async with _ha_server(ha, 18304):
            b = Bridge(fps=5.0)
            b.set_cameras([{"name": "kalitka",
                            "ha_entity": "camera.kalitka",
                            "rtsp_url": "rtsp://127.0.0.1:9/bad"}])
            b.clients["kalitka"] = 1
            b._kick("kalitka")
            ok = await _wait(lambda: b.mode.get("kalitka") == "rtsp")
            assert ok, f"режим не стал rtsp: {b.mode}"
            ok = await _wait(lambda: b.failures.get("kalitka", 0) >= 1, timeout=10)
            assert ok, "неудача RTSP должна быть видна"
            await asyncio.sleep(0.3)
            assert ha.calls == 0, "при rtsp-режиме HA не должен опрашиваться"
            await b.stop()

    asyncio.run(run())


# --------------------------------------------------------------------- #
# 2. Экономика клиентов: гашение получателя и живой кадр для identify

def test_feeder_stops_after_idle_grace():
    """Подписчиков нет дольше GRACE_STOP — получатель гасится (0 сессий)."""

    async def run():
        ha = _StreamHA()
        async with _ha_server(ha, 18305):
            b = Bridge(fps=5.0)
            b.WATCH_INTERVAL = 0.2
            b.GRACE_STOP = 0.5
            b.set_cameras([{"name": "prud", "ha_entity": "camera.prud"}])
            await b.start()  # запускает _idle_watch, гасящий получателя
            # подключился и сразу отключился
            b.clients["prud"] = 1
            b._kick("prud")
            await _wait(lambda: b.frames.get("prud", 0) >= 1)
            b.clients["prud"] = 0
            b._last_client_ts["prud"] = time.time() - 10.0
            ok = await _wait(lambda: "prud" not in b._feeder, timeout=5)
            assert ok, "получатель не погашен после простоя"
            opens = ha.stream_opens
            await asyncio.sleep(0.4)
            assert ha.stream_opens == opens, "после гашения сессии продолжают открываться"
            await b.stop()

    asyncio.run(run())


def test_snapshot_for_entity_uses_bridge_frame():
    """identify берёт живой кадр моста — камере не нужна ещё одна сессия."""
    b = Bridge(fps=2.0)
    b.set_cameras([{"name": "prud", "ha_entity": "camera.prud"}])
    assert b.snapshot_for_entity("camera.prud") is None, \
        "до первого кадра должен быть None (не заглушка!)"
    b.latest["prud"] = JPEG_A
    b.frames["prud"] = 5
    assert b.snapshot_for_entity("camera.prud") == JPEG_A
    assert b.snapshot_for_entity("camera.other") is None


# --------------------------------------------------------------------- #
# 3. Генератор конфига Frigate: ВСЕ камеры через мост

def _conf(cameras: list[dict], bridge_fps: float = 1.5,
          stream_fps: float = 5.0) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        cameras=cameras,
        mqtt={"host": "", "port": 1883, "user": "mqtt", "password": "secret"},
        recognition={"bridge_fps": bridge_fps, "stream_fps": stream_fps},
    )


def _bridge() -> Bridge:
    b = Bridge(fps=1.5, stream_fps=5.0)
    b.set_cameras([
        {"name": "prud", "ha_entity": "camera.prud"},
        {"name": "kalitka", "ha_entity": "camera.kalitka"},
    ])
    return b


def test_generator_routes_rtsp_camera_through_bridge():
    """rtsp-камера тоже через мост (один клиент камеры), без preset-rtsp-tcp."""
    text = build_frigate_config(
        _conf([{"name": "kalitka", "mode": "face",
                "rtsp_url": "rtsp://admin:pw@192.168.1.7:554/sub",
                "display_name": "Калитка"}]),
        _bridge(),
    )
    assert "http://172.30.32.1:8100/mjpeg/kalitka" in text
    assert "preset-rtsp-tcp" not in text, "rtsp_url — источник моста, не прямой вход Frigate"
    assert "  kalitka:  # «Калитка»" in text


def test_generator_fps_per_source():
    m = re.search(r"^  kalitka:.*?fps: (\d+)", build_frigate_config(
        _conf([{"name": "kalitka", "mode": "face",
                "rtsp_url": "rtsp://cam/x"}]), _bridge()), re.S | re.M)
    assert m and m.group(1) == "5", "rtsp-камера: fps = stream_fps (полный темп моста)"
    m = re.search(r"^  prud:.*?fps: (\d+)", build_frigate_config(
        _conf([{"name": "prud", "mode": "car", "rtsp_url": ""}],
              bridge_fps=1.5), _bridge()), re.S | re.M)
    assert m and m.group(1) == "2", "мостовая камера на снимках/потоке: fps = bridge_fps"


def test_generator_never_writes_f_mjpeg():
    text = build_frigate_config(
        _conf([
            {"name": "prud", "mode": "car", "rtsp_url": ""},
            {"name": "kalitka", "mode": "face", "rtsp_url": "rtsp://cam/x"},
        ]),
        _bridge(),
    )
    assert "-f mjpeg" not in text
    assert "input_args: >-" in text
    assert "-use_wallclock_as_timestamps 1" in text


# --------------------------------------------------------------------- #
# 4. Подсказки панели под новые режимы

def test_hint_stream_ok_explains_single_session():
    b = Bridge(fps=2.0)
    b.set_cameras([{"name": "kalitka", "ha_entity": "camera.kalitka"}])
    b.mode["kalitka"] = b.source["kalitka"] = "stream"
    b.frames["kalitka"] = 10
    b.fps["kalitka"] = 4.0
    h = b.hint("kalitka")
    assert "ОДНО" in h and "rtsp_url" in h


def test_hint_rtsp_failure():
    b = Bridge(fps=2.0)
    b.set_cameras([{"name": "kalitka", "ha_entity": "camera.kalitka",
                    "rtsp_url": "rtsp://x"}])
    b.mode["kalitka"] = "rtsp"
    b.failures["kalitka"] = 3
    b.last_error["kalitka"] = "RTSP: соединение не открылось"
    h = b.hint("kalitka")
    assert "RTSP" in h and "rtsp_url" in h


def test_hint_persistent_stream_failures():
    b = Bridge(fps=2.0)
    b.set_cameras([{"name": "kalitka", "ha_entity": "camera.kalitka"}])
    b.mode["kalitka"] = "stream"
    b.failures["kalitka"] = 30
    b.last_error["kalitka"] = "поток HA недоступен"
    h = b.hint("kalitka")
    assert "rtsp_url" in h and "still_image_url" in h


# --------------------------------------------------------------------- #
# 5. UI: пане­ли «Машины»/«Лица» (регресс «pane is null»)

def test_ui_subjects_pane_ids_match_lookup():
    """Баг развёртки: renderSubjectsPane искала 'pane-car', а элемент —
    'pane-cars' → вкладки пустые, в консоли «TypeError: pane is null»."""
    html = (CORE / "app" / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="pane-cars"' in html and 'id="pane-faces"' in html
    assert "const pane = $('pane-' + kind + 's');" in html, \
        "renderSubjectsPane должен искать панель по МНОЖЕСТВЕННОМУ id"
    assert "const pane = $('pane-' + kind);" not in html, \
        "старый (сломанный) поиск панели остался в коде"


# --------------------------------------------------------------------- #
# 6. Сквозной сценарий: Frigate (ffmpeg) читает мост через режим stream

def test_mjpeg_stream_serves_stream_mode_frames():
    """Клиент моста (как Frigate) получает кадры, пока получатель в stream."""

    async def run():
        ha = _StreamHA()
        async with _ha_server(ha, 18306):
            b = Bridge(fps=5.0)
            b.set_cameras([{"name": "prud", "ha_entity": "camera.prud"}])
            app = web.Application()
            b.register(app)
            r2 = web.AppRunner(app, shutdown_timeout=1)
            await r2.setup()
            await web.TCPSite(r2, "127.0.0.1", 18316).start()
            try:
                async with ClientSession() as s:
                    async with s.get("http://127.0.0.1:18316/mjpeg/prud") as r:
                        assert r.status == 200
                        assert "multipart/x-mixed-replace" in r.headers.get("Content-Type", "")
                        buf = b""
                        t0 = time.time()
                        got_frames = 0
                        async for chunk in r.content.iter_any():
                            buf += chunk
                            while ha_api.first_jpeg(buf):
                                eoi = buf.find(b"\xff\xd9", buf.find(b"\xff\xd8\xff") + 3)
                                buf = buf[eoi + 2:]
                                got_frames += 1
                            if got_frames >= 3 or time.time() - t0 > 10:
                                break
                        assert got_frames >= 3, \
                            f"клиент моста получил только {got_frames} кадров"
            finally:
                await r2.cleanup()
            await b.stop()

    asyncio.run(run())
