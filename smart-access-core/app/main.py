"""Smart Access Core — точка входа.

Состав (всё на одной VM с HAOS, без GPU):
  * bridge  — камерные снимки HA -> MJPEG для Frigate (облачные А.М.Видео/ЕГДС,
              Hikvision, любые camera-сущности);
  * brain   — события Frigate -> распознавание машин (CLIP CPU) и лиц (SFace)
              -> открытие исполнителей через HA API;
  * MQTT    — Frigate events, enrollment от интеграции, статус;
  * HTTP    — UI-панель (статус, база «своих», генератор конфига Frigate, тест).

Устойчивость: каждый компонент изолирован сбоями, при деградации LLM/камер/
HA система продолжает работать (fail-safe: никаких лишних открытий).
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config as cfg        # noqa: E402
from app import ha_api               # noqa: E402
from app.brain import Brain          # noqa: E402
from app.frigate_config import build_frigate_config  # noqa: E402
from app.bridge import Bridge        # noqa: E402
from app.models import Models        # noqa: E402
from app.store import Store          # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOG = logging.getLogger("smart_access.main")

UI_DIR = Path(__file__).parent / "ui"
MODEL_NAMES = ["clip_vision_q.onnx", "sface.onnx", "yunet.onnx"]


async def main() -> None:
    cfg.CONF  # initial load
    store = Store()
    models_ = Models()

    bridge = Bridge(fps=float(cfg.CONF.recognition.get("bridge_fps", 1.5)))
    brain = Brain(store, models_, on_status=lambda: None)

    # --------- начальная настройка камер ---------
    def apply_cameras() -> None:
        # камеры с rtsp_url идут во Frigate напрямую — мост для них не нужен
        cams = [
            {"name": c["name"], "ha_entity": c.get("ha_entity", "")}
            for c in cfg.CONF.cameras
            if not (c.get("rtsp_url") or "").strip()
        ]
        bridge.set_cameras(cams)

    apply_cameras()

    brain.start_mqtt()
    await bridge.start()

    # фоновые задачи
    loop = asyncio.get_running_loop()
    loop.create_task(status_loop(brain))
    loop.create_task(cleanup_loop(store))

    # workers для тяжёлых задач (распознавание) — не блокируем event loop
    app = web.Application(middlewares=[])
    bridge.register(app)

    # ---------------- HTTP API ----------------

    async def handle_index(_req: web.Request) -> web.Response:
        return web.FileResponse(UI_DIR / "index.html")

    # кэш проверки Frigate: не блокируем event loop синхронным requests
    # и не долбим Frigate на каждый опрос UI (обновление кэша 10 с)
    probe_cache: dict = {"ts": 0.0, "res": {"ok": False, "error": "ещё не проверялся"}}
    # хвост журнала аддона Frigate (только когда связи нет, кэш 60 с)
    addon_log_cache: dict = {"ts": 0.0, "res": None}

    async def handle_status(_req: web.Request) -> web.Response:
        conf = cfg.CONF
        if time.time() - probe_cache["ts"] > 10:
            probe_cache["res"] = await loop.run_in_executor(
                None, ha_api.frigate_probe, conf.frigate_url
            )
            probe_cache["ts"] = time.time()
        probe = probe_cache["res"]
        addon_log = None
        if not probe["ok"]:
            # журнал аддона Frigate прямо в панели — без ручного копирования логов
            if time.time() - addon_log_cache["ts"] > 60:
                addon_log_cache["res"] = await loop.run_in_executor(
                    None, ha_api.frigate_addon_log_tail
                )
                addon_log_cache["ts"] = time.time()
            addon_log = addon_log_cache["res"]
        return web.json_response({
            "state": "ok" if (conf.master_enable and (models_.ok or models_.faces_ok)) else "degraded",
            "master_enable": conf.master_enable,
            "models": models_.status,
            "frigate": probe["ok"],
            "frigate_url": conf.frigate_url,
            "frigate_version": probe.get("version"),
            "frigate_detected_url": probe.get("detected_url"),
            "frigate_error": probe["error"],
            "frigate_addon_log": addon_log,
            "mqtt_host": conf.mqtt.get("host") or "core-mosquitto",
            "bridge": {
                c["name"]: {
                    "entity": c.get("ha_entity"),
                    "display": c.get("display_name") or "",
                    "direct": bool((c.get("rtsp_url") or "").strip()),
                    "url": (c.get("rtsp_url") or "").strip() or bridge.frigate_url(c["name"]),
                    "failures": bridge.failures.get(c["name"], 0),
                    "clients": bridge.clients.get(c["name"], 0),
                    "fps": bridge.fps.get(c["name"], 0.0),
                    "frames": bridge.frames.get(c["name"], 0),
                }
                for c in conf.cameras
            },
            "stats": brain.stats,
        })

    async def handle_config(_req: web.Request) -> web.Response:
        return web.json_response(cfg.CONF.as_safe_dict())

    async def handle_reload(_req: web.Request) -> web.Response:
        cfg.reload_conf()
        apply_cameras()
        brain.publish_status()
        probe_cache["ts"] = 0.0  # конфиг мог сменить frigate_url — проверить заново
        return web.json_response({"ok": True})

    async def handle_master(req: web.Request) -> web.Response:
        body = await req.json()
        # Kill-switch: /data/master.json переживает перенастройку аддона
        enabled = bool(body.get("enabled", True))
        with open(os.path.join(cfg.DATA_DIR, "master.json"), "w") as fh:
            import json as _json
            _json.dump({"enabled": enabled}, fh)
        cfg.reload_conf()
        apply_cameras()
        brain.publish_status()
        LOG.warning("MASTER ENABLE = %s", enabled)
        return web.json_response({"ok": True, "master_enable": enabled})

    async def handle_subjects(_req: web.Request) -> web.Response:
        return web.json_response({
            "cars": store.subjects("car"),
            "faces": store.subjects("face"),
        })

    async def handle_subject_delete(req: web.Request) -> web.Response:
        body = await req.json()
        kind = str(body.get("kind") or "car")
        subject = str(body.get("subject") or "")
        ok = store.remove_subject(kind, subject)
        brain.publish_status()
        return web.json_response({"ok": ok})

    async def handle_enroll(req: web.Request) -> web.Response:
        """Прямой enrollment (файл/фото из UI): multipart form."""
        reader = await req.multipart()
        kind, subject, jpg = "car", "", None
        async for part in reader:
            if part.name == "kind":
                kind = (await part.text()).strip()
            elif part.name == "subject":
                subject = (await part.text()).strip()
            elif part.name == "photo":
                jpg = await part.read(decode=False)
        if jpg is None or not subject:
            return web.json_response({"ok": False, "error": "нужны subject и photo"}, status=400)
        result = await loop.run_in_executor(None, brain.enroll, kind, subject, jpg)
        return web.json_response(result)

    async def handle_events(_req: web.Request) -> web.Response:
        return web.json_response({"events": store.events[:80]})

    async def handle_frigate_config(_req: web.Request) -> web.Response:
        return web.Response(
            text=build_frigate_config(cfg.CONF, bridge),
            content_type="text/plain",
            charset="utf-8",
        )

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/config", handle_config)
    app.router.add_post("/api/reload", handle_reload)
    app.router.add_post("/api/master", handle_master)
    app.router.add_get("/api/subjects", handle_subjects)
    app.router.add_post("/api/subjects/delete", handle_subject_delete)
    app.router.add_post("/api/enroll", handle_enroll)
    app.router.add_get("/api/events", handle_events)
    app.router.add_get("/api/frigate-config", handle_frigate_config)
    if UI_DIR.exists():
        app.router.add_static("/ui/", UI_DIR)

    # ---------------- сигналы ----------------

    def request_reload(_sig, _frm):  # noqa: ANN001
        cfg.reload_conf()
        apply_cameras()
        brain.publish_status()

    loop.add_signal_handler(signal.SIGHUP, request_reload)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8099)
    await site.start()

    # ---------------- MJPEG-порт 8100 ----------------
    # v1.1.10 и раньше сайт на 8100 НЕ поднимался вовсе: маршруты /mjpeg/*
    # были зарегистрированы, а слушали только :8099 — Frigate получал
    # «Connection to tcp://172.30.32.1:8100 refused» и ffmpeg-цикл падал.
    # ВАЖНО: только ПОСЛЕ runner.setup() — в 1.1.11 вызов стоял до создания
    # runner (UnboundLocalError, аддон не стартовал).
    await start_bridge_site(runner)
    LOG.info("Smart Access Core запущен: UI на :8099, MJPEG-мост на :8100")
    brain.publish_status()

    while True:
        await asyncio.sleep(3600)


async def start_bridge_site(runner: web.AppRunner, port: int = 8100) -> None:
    """Слушаем MJPEG-порт на адресах, доступных Frigate.

    Frigate — мостовой аддон (сеть hassio): хост для него — шлюз 172.30.32.1;
    127.0.0.1 оставляем для локальных проверок с самого хоста.
    Список переопределяется env SMART_ACCESS_BRIDGE_BIND (через запятую);
    если ни одна привязка не удалась — фолбэк 0.0.0.0.
    """
    env_binds = os.environ.get("SMART_ACCESS_BRIDGE_BIND", "")
    binds = [a.strip() for a in env_binds.split(",") if a.strip()] or \
        ["172.30.32.1", "127.0.0.1"]
    started: list[str] = []
    for addr in binds:
        try:
            await web.TCPSite(runner, addr, port).start()
            LOG.info("MJPEG-мост слушает %s:%s", addr, port)
            started.append(addr)
        except OSError as exc:
            LOG.warning("MJPEG-мост: привязка %s:%s не удалась (%s)", addr, port, exc)
    if not started:
        await web.TCPSite(runner, "0.0.0.0", port).start()
        LOG.warning("MJPEG-мост слушает 0.0.0.0:%s (фолбэк: ни одна привязка не удалась)", port)


async def status_loop(brain: Brain) -> None:
    while True:
        try:
            brain.publish_status()
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(30)


async def cleanup_loop(store: Store) -> None:
    while True:
        try:
            store.cleanup_unknown(int(cfg.CONF.recognition.get("unknown_keep_days", 7)))
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(6 * 3600)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
