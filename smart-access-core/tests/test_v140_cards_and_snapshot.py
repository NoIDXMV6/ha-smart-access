"""v1.4.0: карточки HA (сущности интеграции) и надёжность моста.

Проблемы из эксплуатации:
  * карточки: «Я подключил generic camera к потокам /mjpeg/… — но изображения
    нет» — HA не открывает multipart-поток как RTSP; карточки должны играть
    мост через сущности интеграции (camera.smart_access_bridge_<имя>) или
    «URL кадра» (still_image_url) на /snapshot/<name>. Мост обязан отдавать
    одиночный JPEG, а запрос снимка — оживлять получателя камеры;
  * «поток HA недоступен» (prud: сбоев 11 за 10 минут): когда endpoint
    camera_proxy_stream лежит, камера оставалась без кадров на время
    бэкоффа. Теперь после 3 неудач мост пробует одиночный снимок (режим
    stream сохраняется) — кадры не пропадают;
  * сущности интеграции создаются по списку камер из MQTT-статуса аддона —
    статус обязан содержать cameras: [{name, display, entity}].

Запуск:  python3 -m pytest smart-access-core/tests/ -v
"""
from __future__ import annotations

import asyncio
import io
import json
import py_compile
import sys
import time
import types
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

CORE = Path(__file__).resolve().parents[1]
REPO = CORE.parent
sys.path.insert(0, str(CORE))

from app import brain as brain_mod      # noqa: E402
from app import config as cfg           # noqa: E402
from app import ha_api                  # noqa: E402
from app.bridge import Bridge           # noqa: E402

JPEG_A = b"\xff\xd8\xff\xe0" + b"A" * 64 + b"\xff\xd9"
JPEG_B = b"\xff\xd8\xff\xe0" + b"B" * 64 + b"\xff\xd9"
BOUNDARY = "v140boundary"


def _multipart(frame: bytes) -> bytes:
    return (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame)}\r\n\r\n".encode() + frame + b"\r\n")


def _real_jpeg() -> bytes:
    """Настоящий JPEG (маркеры SOI…EOI), а не синтетика: card проверяет поля."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (40, 90, 60)).save(buf, "JPEG", quality=70)
    return buf.getvalue()


# --------------------------------------------------------------------- #
# Инфраструктура: фейковый Supervisor/HA и мост под aiohttp-тестом

class _StreamBrokenProxyOK:
    """HA: endpoint потока лежит (как «поток HA недоступен»), одиночные
    снимки начинают работать со 2-го вызова (первый — прогрев камеры)."""

    def __init__(self) -> None:
        self.proxy_calls = 0
        self.stream_opens = 0

    async def proxy(self, request: web.Request) -> web.Response:
        self.proxy_calls += 1
        if self.proxy_calls <= 1:
            return web.json_response({"error": "not ready"}, status=500)
        return web.Response(body=JPEG_B, content_type="image/jpeg")

    async def stream(self, request: web.Request) -> web.Response:
        self.stream_opens += 1
        return web.json_response({"error": "stream unavailable"}, status=503)


@asynccontextmanager
async def _ha_server(ha, port: int):
    """Поднимает фейковый Supervisor/HA и подменяет supervisor_base на него."""
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


@asynccontextmanager
async def _bridge_site(bridge: Bridge, port: int):
    """Сайт моста на 127.0.0.1 (как start_bridge_site в аддоне)."""
    app = web.Application()
    bridge.register(app)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    try:
        yield runner
    finally:
        await runner.cleanup()


async def _wait(predicate, timeout: float = 10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


# --------------------------------------------------------------------- #
# 1. /snapshot/<name> — одиночный JPEG для карточек

def test_snapshot_returns_jpeg_for_known_camera():
    """Снимок известной камеры: 200 + JPEG, запрос запускает получателя."""

    async def run():
        from aiohttp import ClientSession

        b = Bridge(fps=5.0)
        b.set_cameras([{"name": "prud", "ha_entity": "camera.prud"}])
        async with _bridge_site(b, 18401):
            async with ClientSession() as s:
                resp = await s.get("http://127.0.0.1:18401/snapshot/prud")
                assert resp.status == 200, "снимок известной камеры должен отдаваться"
                body = await resp.read()
                assert body[:3] == b"\xff\xd8\xff", "тело должно быть JPEG"
                assert resp.headers["Content-Type"].startswith("image/jpeg")
                # даже без кадра запрос снимка оживляет получателя камеры
                assert "prud" in b._feeder and not b._feeder["prud"].done(), \
                    "запрос /snapshot должен запускать получателя (иначе камера "\
                    "только со снимком-карточкой никогда не оживёт)"
        await b.stop()

    asyncio.run(run())


def test_snapshot_unknown_camera_404():
    async def run():
        from aiohttp import ClientSession

        b = Bridge(fps=5.0)
        b.set_cameras([{"name": "prud", "ha_entity": "camera.prud"}])
        async with _bridge_site(b, 18402):
            async with ClientSession() as s:
                resp = await s.get("http://127.0.0.1:18402/snapshot/govorovo")
                assert resp.status == 404
        await b.stop()

    asyncio.run(run())


def test_snapshot_serves_latest_frame():
    """После получения кадра мост отдаёт ЕГО (не заглушку)."""

    async def run():
        b = Bridge(fps=5.0)
        b.set_cameras([{"name": "prud", "ha_entity": "camera.prud"}])
        b._accept("prud", _real_jpeg(), "proxy")
        async with _bridge_site(b, 18403):
            from aiohttp import ClientSession

            async with ClientSession() as s:
                resp = await s.get("http://127.0.0.1:18403/snapshot/prud")
                body = await resp.read()
                assert body == _real_jpeg(), "должен отдаваться последний кадр моста"
        await b.stop()

    asyncio.run(run())


# --------------------------------------------------------------------- #
# 2. Гибрид: поток HA лежит — кадры берутся одиночным снимком

def test_stream_outage_falls_back_to_proxy_frames():
    """«Поток HA недоступен» (503): после 3 неудач мост берёт кадр снимком,
    режим ОСТАЁТСЯ stream (подсказка панели не врёт), счётчик сбоев гаснет."""

    async def run():
        ha = _StreamBrokenProxyOK()
        async with _ha_server(ha, 18404):
            b = Bridge(fps=5.0)
            b.set_cameras([{"name": "prud", "ha_entity": "camera.prud"}])
            b.clients["prud"] = 1
            b._kick("prud")
            ok = await _wait(lambda: b.mode.get("prud") == "stream")
            assert ok, f"режим должен стать stream: {b.mode} ({b.last_error})"
            got = await _wait(lambda: b.frames.get("prud", 0) >= 1, timeout=12)
            assert got, (f"при лежащем потоке HA камера осталась без кадров "
                         f"(сбоев: {b.failures.get('prud')}, ошибка: {b.last_error})")
            assert b.source.get("prud") == "proxy", "кадр должен прийти запасным снимком"
            assert b.mode.get("prud") == "stream", "режим stream не должен переключаться"
            assert b.failures.get("prud", 0) == 0, "после кадра счётчик сбоев гаснет"
            assert ha.stream_opens >= 3, "поток должен был реально открыться и отпасть"
            await b.stop()

    asyncio.run(run())


# --------------------------------------------------------------------- #
# 3. MQTT-статус аддона несёт список камер (для сущностей интеграции)

def test_status_payload_includes_cameras():
    """publish_status обязан добавить cameras: [{name, display, entity}] —
    по этому списку интеграция создаёт camera.smart_access_bridge_<имя>."""

    class _FakeMQTT:
        def __init__(self) -> None:
            self.published: dict[str, str] = {}

        def publish(self, topic: str, payload: str, retain: bool = False) -> None:
            self.published[topic] = payload

    async def run():
        conf = types.SimpleNamespace(
            master_enable=True,
            cameras=[
                {"name": "prud", "display_name": "Пруд", "ha_entity": "camera.prud"},
                {"name": "kalitka", "ha_entity": "camera.kalitka"},
            ],
            gestures={"enabled": False, "threshold": 0.8},
            alarm={"enabled": False, "numbers": ""},
        )
        b = brain_mod.Brain.__new__(brain_mod.Brain)  # без тяжёлого __init__
        b.client = _FakeMQTT()
        b.models = SimpleNamespace(ok=True, faces_ok=True, gestures_ok=False,
                                   status={"clip": "ok"})
        b.store = SimpleNamespace(cars={}, faces={},
                                  gestures_configured_subjects=lambda: [])
        b.stats = {"events_seen": 0, "grants": 0, "denied": 0, "vlm_calls": 0,
                   "started": time.time(), "last_grant": None}
        old_conf = brain_mod.cfg.CONF
        brain_mod.cfg.CONF = conf  # type: ignore[assignment]
        try:
            brain_mod.Brain.publish_status(b)
        finally:
            brain_mod.cfg.CONF = old_conf
        payload = json.loads(b.client.published["smart_access/status"])
        cams = payload.get("cameras")
        assert isinstance(cams, list) and len(cams) == 2, payload
        assert cams[0] == {"name": "prud", "display": "Пруд",
                           "entity": "camera.prud"}
        # у камеры без display_name подставляется slug-имя
        assert cams[1]["display"] == "kalitka"

    asyncio.run(run())


# --------------------------------------------------------------------- #
# 4. Интеграция: синтаксис модулей (включая новую платформу camera.py)

def test_integration_modules_compile():
    """Все модули интеграции компилируются (python-синтаксис без HA-рантайма):
    camera.py — новая платформа сущностей моста."""
    for f in sorted((REPO / "custom_components" / "smart_access").glob("*.py")):
        py_compile.compile(str(f), doraise=True)


# --------------------------------------------------------------------- #
# 5. Совместность версий аддона и интеграции (регресс v1.3.1)

def test_versions_in_sync():
    addon = json.load(open(CORE / "config.json", encoding="utf-8"))
    manifest = json.load(open(REPO / "custom_components" / "smart_access" / "manifest.json",
                              encoding="utf-8"))
    assert addon["version"] == manifest["version"]
