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
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent))

from app import config as cfg        # noqa: E402
from app import ha_api               # noqa: E402
from app.brain import Brain          # noqa: E402
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
        cams = [
            {"name": c["name"], "ha_entity": c.get("ha_entity", "")}
            for c in cfg.CONF.cameras
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

    async def handle_status(_req: web.Request) -> web.Response:
        conf = cfg.CONF
        return web.json_response({
            "state": "ok" if (conf.master_enable and (models_.ok or models_.faces_ok)) else "degraded",
            "master_enable": conf.master_enable,
            "models": models_.status,
            "frigate": ha_api.frigate_ping(conf.frigate_url),
            "mqtt_host": conf.mqtt.get("host") or "core-mosquitto",
            "bridge": {
                c["name"]: {
                    "entity": c.get("ha_entity"),
                    "url": bridge.url(c["name"]),
                    "failures": bridge.failures.get(c["name"], 0),
                    "clients": bridge.clients.get(c["name"], 0),
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
        conf = cfg.CONF
        cams = []
        for c in conf.cameras:
            if c.get("mode", "off") == "off":
                continue
            cams.append(c)
        rec = conf.recognition
        mqtt_host = conf.mqtt.get("host") or "core-mosquitto"
        lines: list[str] = []
        lines.append("mqtt:")
        lines.append(f"  host: {mqtt_host}")
        lines.append(f"  port: {int(conf.mqtt.get('port') or 1883)}")
        if conf.mqtt.get("user"):
            lines.append(f"  user: {conf.mqtt['user']}")
            lines.append("  password: ВАШ_ПАРОЛЬ_БРОКЕРА")
        lines.append("")
        lines.append("detectors:")
        lines.append("  cpu:")
        lines.append("    type: openvino")
        lines.append("    device: CPU")
        lines.append("    model:")
        lines.append("      path: /openvino-model/ssdlite_mobilenet_v2.xml")
        lines.append("model:")
        lines.append("  width: 300")
        lines.append("  height: 300")
        lines.append("  input_tensor: nhwc")
        lines.append("  input_pixel_format: bgr")
        lines.append("  labelmap_path: /openvino-model/coco_91cl_bkgr.txt")
        lines.append("")
        lines.append("cameras:")
        for c in cams:
            name = c["name"]
            lines.append(f"  {name}:")
            lines.append("    ffmpeg:")
            lines.append("      inputs:")
            lines.append(f"        - path: {bridge.url(name)}")
            lines.append("          input_args: -f mjpeg")
            lines.append("          roles:")
            lines.append("            - detect")
            lines.append("    detect:")
            lines.append("      enabled: true")
            lines.append("      fps: 5")
            lines.append("      width: 1280")
            lines.append("      height: 720")
            lines.append("    objects:")
            lines.append("      track:")
            if "car" in c.get("mode", ""):
                lines.append("        - car")
            if "face" in c.get("mode", ""):
                lines.append("        - person")
            lines.append("    snapshots:")
            lines.append("      enabled: true")
            lines.append("      retain:")
            lines.append("        default: 2")
            lines.append("    record:")
            lines.append("      enabled: false")
            lines.append("")
        text = "\n".join(lines)
        return web.Response(text=text, content_type="text/plain", charset="utf-8")

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
    LOG.info("Smart Access Core запущен: UI на :8099, MJPEG на :8100")
    brain.publish_status()

    while True:
        await asyncio.sleep(3600)


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
