"""v1.1.13: мост × Frigate — регресс главного бага выпуска.

Баг: генератор писал мостовой камере `input_args: -f mjpeg`. Сырой демуксер
mjpeg не завершает пробу формата на multipart-потоке моста (считает поток
25 fps, ждёт ~125 кадров «виртуального» времени) — ffmpeg не выдаёт ни кадра
за 20 с, watchdog Frigate убивает его по кругу («No frames received …»).

Запуск:  python3 -m pytest smart-access-core/tests/ -v
"""
from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest
from aiohttp import web, ClientSession

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

from app import config as cfg            # noqa: E402
from app import ha_api                   # noqa: E402
from app.bridge import Bridge            # noqa: E402
from app.frigate_config import build_frigate_config  # noqa: E402


# --------------------------------------------------------------------- #
# Фикстуры

class _FakeHA:
    """Минимальный /core/api/camera_proxy с настраиваемой задержкой."""

    def __init__(self, latencies: dict[str, float]) -> None:
        self.latencies = latencies
        self.counters: dict[str, int] = {k: 0 for k in latencies}

    async def camera_proxy(self, request: web.Request) -> web.Response:
        entity = request.match_info["entity"]
        await asyncio.sleep(self.latencies.get(entity, 0.1))
        self.counters[entity] = self.counters.get(entity, 0) + 1
        img = _jpeg(self.counters[entity])
        return web.Response(body=img, content_type="image/jpeg")


def _jpeg(n: int) -> bytes:
    import cv2
    import numpy as np
    img = np.full((720, 1280, 3), 30, np.uint8)
    cv2.putText(img, f"frame {n}", (60, 360), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (255, 255, 255), 3, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else b"\xff\xd8\xff\xff\xd9"


def _conf(cameras: list[dict], bridge_fps: float = 1.5) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        cameras=cameras,
        mqtt={"host": "", "port": 1883, "user": "mqtt", "password": "secret"},
        recognition={"bridge_fps": bridge_fps},
    )


def _bridge() -> Bridge:
    b = Bridge(fps=1.5)
    b.set_cameras([
        {"name": "prud", "ha_entity": "camera.prud"},
        {"name": "kalitka", "ha_entity": "camera.kalitka"},
    ])
    return b


# --------------------------------------------------------------------- #
# 1. Генератор конфига

def test_generator_no_f_mjpeg_for_bridge_cameras():
    """Регресс главного бага: '-f mjpeg' не должен появляться никогда."""
    text = build_frigate_config(
        _conf([{"name": "prud", "mode": "car", "rtsp_url": ""}]), _bridge()
    )
    assert "-f mjpeg" not in text
    assert "input_args: >-" in text
    assert "-use_wallclock_as_timestamps 1" in text
    assert "-fflags +genpts+discardcorrupt" in text
    assert "http://172.30.32.1:8100/mjpeg/prud" in text


def test_generator_fps_follows_bridge_fps_for_bridge_and_5_for_rtsp():
    text = build_frigate_config(
        _conf([
            {"name": "prud", "mode": "car", "rtsp_url": ""},
            {"name": "kalitka", "mode": "face", "rtsp_url": "rtsp://cam/x"},
        ], bridge_fps=1.5),
        _bridge(),
    )
    m = re.search(r"^  prud:.*?fps: (\d+)", text, re.S | re.M)
    assert m and m.group(1) == "2", "мостовая камера: fps = округлённый bridge_fps"
    m2 = re.search(r"^  kalitka:.*?fps: (\d+)", text, re.S | re.M)
    assert m2 and m2.group(1) == "5", "RTSP-камера: fps остаётся 5"


def test_generator_keeps_rtsp_preset_and_comments():
    text = build_frigate_config(
        _conf([{"name": "kalitka", "mode": "face",
                "rtsp_url": "rtsp://admin:pw@192.168.1.7:554/sub",
                "display_name": "Калитка"}]),
        _bridge(),
    )
    assert "input_args: preset-rtsp-tcp" in text
    assert "  kalitka:  # «Калитка»" in text  # исходное имя рядом со slug


# --------------------------------------------------------------------- #
# 2. Мост: параллельный опрос

def test_bridge_polls_cameras_in_parallel(monkeypatch):
    """3 медленные камеры: у КАЖДОЙ свой fps, а не 1/(сумма задержек)."""

    async def run() -> dict[str, int]:
        def fake_get(entity: str) -> bytes:
            # как настоящий get_camera_image: БЛОКИРУЮЩИЙ вызов в executor-потоке
            time.sleep(0.8)
            return b"\xff\xd8\xff\x00"

        monkeypatch.setattr(ha_api, "get_camera_image", fake_get)
        b = Bridge(fps=1.5)
        b.set_cameras([
            {"name": "prud", "ha_entity": "camera.prud"},
            {"name": "kalitka", "ha_entity": "camera.kalitka"},
            {"name": "tat", "ha_entity": "camera.tat"},
        ])
        for c in b.entities:
            b.clients[c] = 1  # имитируем подключённых клиентов (без HTTP)
        await b.start()
        await asyncio.sleep(3.0)
        await b.stop()
        return dict(b.frames)

    got = asyncio.run(run())
    # при старом (последовательном) опросе тик = 3*0.8 = 2.4 c → за 3 c
    # последняя камера получила бы ≤1 кадр; при параллельном у всех ≥2
    for cam, n in got.items():
        assert n >= 2, f"{cam}: {n} кадров за 3 c — опрос не параллельный?"


# --------------------------------------------------------------------- #
# 3. Мост: HTTP-поведение

def test_bridge_404_unknown_camera_is_27_bytes():
    """«govorovo: 27 байт» из диагностики = 404 неизвестного имени."""

    async def run() -> int:
        b = Bridge(fps=1.5)
        b.set_cameras([{"name": "a_m_video", "ha_entity": "camera.x"}])
        app = web.Application()
        b.register(app)
        runner = web.AppRunner(app, shutdown_timeout=1)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 18100)
        await site.start()
        try:
            async with ClientSession() as s:
                async with s.get("http://127.0.0.1:18100/mjpeg/govorovo") as r:
                    body = await r.read()
                    assert r.status == 404
                    return len(body)
        finally:
            await runner.cleanup()

    assert asyncio.run(run()) == 27


# --------------------------------------------------------------------- #
# 4. Интеграция с НАСТОЯЩИМ ffmpeg (как клиент Frigate)

FFMPEG = shutil.which("ffmpeg")


def _input_args_from_generator() -> list[str]:
    """Ровно те аргументы, что генератор пишет мостовой камере."""
    text = build_frigate_config(
        _conf([{"name": "prud", "mode": "car", "rtsp_url": ""}]), _bridge()
    )
    block = re.search(r"input_args: >-\n((?:[ \t]+-[^\n]+\n)+)", text)
    assert block, "генератор не выдал input_args"
    args: list[str] = []
    for line in block.group(1).strip().splitlines():
        args.extend(line.split())
    return args


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg не найден")
def test_real_ffmpeg_decodes_bridge_stream_cold_start():
    """Холодный мост (первый кадр — заглушка) + аргументы генератора:
    ffmpeg обязан открыть вход и декодировать кадры за 15 с."""

    async def run() -> tuple[bool, int]:
        fake = _FakeHA({"camera.prud": 0.3})
        ha_app = web.Application()
        ha_app.router.add_get("/core/api/camera_proxy/{entity}", fake.camera_proxy)
        r1 = web.AppRunner(ha_app)
        await r1.setup()
        await web.TCPSite(r1, "127.0.0.1", 18223).start()

        cfg.supervisor_base = lambda: "http://127.0.0.1:18223"  # type: ignore[assignment]
        cfg.SUPERVISOR_TOKEN = "t"
        ha_api.cfg.SUPERVISOR_TOKEN = "t"

        b = Bridge(fps=1.5)
        b.set_cameras([{"name": "prud", "ha_entity": "camera.prud"}])
        app = web.Application()
        b.register(app)
        r2 = web.AppRunner(app, shutdown_timeout=1)
        await r2.setup()
        await web.TCPSite(r2, "127.0.0.1", 18100).start()
        await b.start()
        await asyncio.sleep(0.2)

        url = "http://127.0.0.1:18100/mjpeg/prud"
        with tempfile.NamedTemporaryFile("w+b", suffix=".log", delete=False) as fh:
            logpath = fh.name
        args = _input_args_from_generator()
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "info", *args,
               "-i", url, "-an", "-f", "null", "-"]
        with open(logpath, "wb") as fh:
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=fh)
        t0 = time.time()
        while time.time() - t0 < 15 and p.poll() is None:
            await asyncio.sleep(0.5)
        p.kill()
        p.wait()
        text = Path(logpath).read_text(errors="replace")
        Path(logpath).unlink(missing_ok=True)
        await r1.cleanup()
        await r2.cleanup()
        opened = "Input #0" in text
        m = re.findall(r"frame=\s*(\d+)", text)
        return opened, (int(m[-1]) if m else 0)

    opened, frames = asyncio.run(run())
    assert opened, "ffmpeg не смог открыть поток моста за 15 с"
    assert frames > 0, "ffmpeg открыл поток, но не декодировал ни кадра"
