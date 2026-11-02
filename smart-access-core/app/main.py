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
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config as cfg        # noqa: E402
from app import gestures as gest_lib  # noqa: E402
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

    # v1.3.0: мост — ЕДИНСТВЕННЫЙ получатель камер (включая камеры с rtsp_url:
    # мост сам читает RTSP и раздаёт кадры Frigate и HA — клиентов камеры: 1)
    bridge = Bridge(
        fps=float(cfg.CONF.recognition.get("bridge_fps", 1.5)),
        stream_fps=float(cfg.CONF.recognition.get("stream_fps", 5.0)),
    )
    brain = Brain(store, models_, on_status=lambda: None, bridge=bridge)

    # --------- начальная настройка камер ---------
    def apply_cameras() -> None:
        # v1.3.0: в мост входят ВСЕ камеры. rtsp_url — теперь источник кадра
        # для моста (мост сам читает камеру), а не «обход моста» для Frigate:
        # у камеры остаётся ОДИН клиент.
        cams = [
            {"name": c["name"],
             "ha_entity": c.get("ha_entity", ""),
             "rtsp_url": (c.get("rtsp_url") or "").strip()}
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
                    # v1.3.0: rtsp_url — источник моста, Frigate всегда читает мост
                    "direct": False,
                    "url": bridge.frigate_url(c["name"]),
                    "failures": bridge.failures.get(c["name"], 0),
                    "clients": bridge.clients.get(c["name"], 0),
                    "fps": bridge.fps.get(c["name"], 0.0),
                    "frames": bridge.frames.get(c["name"], 0),
                    # v1.3.0: РЕЖИМ получателя камеры: rtsp|proxy|stream|''
                    "mode": bridge.mode.get(c["name"], ""),
                    "source": bridge.source.get(c["name"], ""),
                    "error": bridge.last_error.get(c["name"], ""),
                    "hint": bridge.hint(c["name"]),
                }
                for c in conf.cameras
            },
            "stats": brain.stats,
            "gestures": {
                "enabled": bool((conf.gestures or {}).get("enabled")),
                "model": models_.status.get("pose", "?"),
                "available": models_.gestures_ok,
                "threshold": float((conf.gestures or {}).get("threshold", 0.80)),
                "configured": len(store.gestures_configured_subjects()),
            },
            "alarm": {
                "enabled": bool((conf.alarm or {}).get("enabled")),
                "addon": (conf.alarm or {}).get("addon") or "",
                "numbers": [n.strip() for n in str((conf.alarm or {}).get("numbers") or "").splitlines() if n.strip()],
                "ring_timeout": (conf.alarm or {}).get("ring_timeout"),
                "message_template": (conf.alarm or {}).get("message_template") or "",
                "tts_language": (conf.alarm or {}).get("language") or "ru",
            },
            "retrain": brain.retrain_status,
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

    # ---------------- настраиваемые жесты (v1.3.0) ----------------

    async def handle_gestures(_req: web.Request) -> web.Response:
        """Каталог жестов + привязка к лицам + статус обучения."""
        conf = cfg.CONF
        faces = store.subjects("face")
        subjects: dict[str, dict] = {}
        for name in faces:
            gcfg = store.get_subject_gestures(name)
            subjects[name] = {
                **gcfg,
                "enrolled": store.gesture_info(name),
                "configured": bool(gcfg.get("confirm") or gcfg.get("alarm")),
            }
        return web.json_response({
            "available": models_.gestures_ok,
            "model": models_.status.get("pose", "?"),
            "enabled": bool((conf.gestures or {}).get("enabled")),
            "threshold": float((conf.gestures or {}).get("threshold", 0.80)),
            "defaults": {
                "confirm": gest_lib.gesture_list_safe((conf.gestures or {}).get("default_confirm"), ["one_hand_up"]),
                "alarm": gest_lib.gesture_list_safe((conf.gestures or {}).get("default_alarm"), ["both_hands_up"]),
            },
            "catalog": [{"id": gid, **meta} for gid, meta in gest_lib.GESTURES.items()],
            "subjects": subjects,
        })

    async def handle_gestures_config(req: web.Request) -> web.Response:
        """Привязка жестов к лицу: {subject, confirm: [gid], alarm: [gid]}."""
        body = await req.json()
        subject = str(body.get("subject") or "").strip()
        if not subject:
            return web.json_response({"ok": False, "error": "нужен subject"}, status=400)
        if subject not in store.faces:
            return web.json_response({"ok": False, "error": f"лицо «{subject}» не найдено — сначала обучите его во вкладке «Лица»"}, status=400)
        def _ids(v) -> list[str]:
            if isinstance(v, str):
                v = [x.strip() for x in v.split(",") if x.strip()]
            return [str(x).strip() for x in (v or []) if gest_lib.is_valid_gesture(str(x).strip())]
        confirm, alarm = _ids(body.get("confirm")), _ids(body.get("alarm"))
        dup = set(confirm) & set(alarm)
        if dup:
            return web.json_response(
                {"ok": False, "error": f"жест не может быть и подтверждением, и тревогой: {', '.join(sorted(dup))}"},
                status=400)
        store.set_subject_gestures(subject, confirm, alarm)
        brain.publish_status()
        return web.json_response({"ok": True, "confirm": confirm, "alarm": alarm})

    async def handle_gestures_enroll(req: web.Request) -> web.Response:
        """Обучение hand-жеста фото: multipart (subject, gesture, photo[])."""
        reader = await req.multipart()
        subject, gesture, jpgs = "", "", []
        try:
            async for part in reader:
                if part.name == "subject":
                    subject = (await part.text()).strip()
                elif part.name == "gesture":
                    gesture = (await part.text()).strip()
                elif part.name == "photo":
                    jpgs.append(await part.read(decode=False))
        except ValueError:
            pass
        if not subject or not gesture or not jpgs:
            return web.json_response({"ok": False, "error": "нужны subject, gesture и photo"}, status=400)
        if subject not in store.faces:
            return web.json_response({"ok": False, "error": f"лицо «{subject}» не найдено — сначала обучите его во вкладке «Лица»"}, status=400)
        added, crops, errors = 0, 0, []
        for jpg in jpgs:
            r = await loop.run_in_executor(None, brain.enroll_gesture, subject, gesture, jpg)
            if r.get("ok"):
                added += int(r.get("added") or 0)
                crops += int(r.get("crops") or 0)
            else:
                errors.append(str(r.get("error") or "ошибка"))
        return web.json_response({
            "ok": added > 0,
            "added": added, "crops": crops, "files": len(jpgs), "errors": errors,
        })

    async def handle_gestures_photos(req: web.Request) -> web.Response:
        q = req.rel_url.query
        subject, gesture = (q.get("subject") or "").strip(), (q.get("gesture") or "").strip()
        if not subject or not gesture:
            return web.json_response({"ok": False, "error": "нужны subject и gesture"}, status=400)
        try:
            return web.json_response({"ok": True,
                                      "photos": store.list_gesture_photos(subject, gesture)})
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    async def handle_gestures_photo(req: web.Request) -> web.Response:
        p = store.gesture_photo_path(req.match_info.get("subject", ""),
                                     req.match_info.get("gesture", ""),
                                     req.match_info.get("file", ""))
        if p is None:
            return web.json_response({"ok": False, "error": "фото не найдено"}, status=404)
        return web.FileResponse(p)

    async def handle_gestures_photos_delete(req: web.Request) -> web.Response:
        """Удаление кропов кисти + пересчёт прототипов по оставшимся."""
        body = await req.json()
        subject = str(body.get("subject") or "")
        gesture = str(body.get("gesture") or "")
        files = body.get("files") or []
        if not subject or not gesture or not files:
            return web.json_response({"ok": False, "error": "нужны subject, gesture, files[]"}, status=400)
        try:
            deleted = store.delete_gesture_photos(subject, gesture, files)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        r = await loop.run_in_executor(None, brain.retrain_gesture, subject, gesture)
        return web.json_response({"ok": True, "deleted": deleted,
                                  "protos": int(r.get("added") or 0)})

    async def handle_gestures_test(req: web.Request) -> web.Response:
        """Проверка: какие жесты лица видны на кадре (multipart: subject, photo)."""
        reader = await req.multipart()
        subject, jpg = "", None
        try:
            async for part in reader:
                if part.name == "subject":
                    subject = (await part.text()).strip()
                elif part.name == "photo":
                    jpg = await part.read(decode=False)
        except ValueError:
            pass
        if jpg is None:
            return web.json_response({"ok": False, "error": "нужно фото"}, status=400)
        r = await loop.run_in_executor(None, brain.test_gesture, subject, jpg)
        return web.json_response(r)

    async def handle_camera_snapshot(req: web.Request) -> web.Response:
        """Кадр камеры HA: /api/camera_snapshot?entity=camera.x — для обучения жестов."""
        entity = (req.rel_url.query.get("entity") or "").strip()
        if not entity:
            return web.json_response({"ok": False, "error": "нужен entity"}, status=400)
        jpg = await loop.run_in_executor(None, lambda: ha_api.get_camera_image_ex(entity)[0])
        if not jpg:
            return web.json_response({"ok": False, "error": "кадр недоступен"}, status=502)
        return web.Response(body=jpg, content_type="image/jpeg")

    async def handle_subject_delete(req: web.Request) -> web.Response:
        body = await req.json()
        kind = str(body.get("kind") or "car")
        subject = str(body.get("subject") or "")
        ok = store.remove_subject(kind, subject)
        brain.publish_status()
        return web.json_response({"ok": ok})

    async def handle_enroll(req: web.Request) -> web.Response:
        """Прямой enrollment (фото из UI): multipart form, photo может повторяться."""
        reader = await req.multipart()
        kind, subject, jpgs = "car", "", []
        try:
            async for part in reader:
                if part.name == "kind":
                    kind = (await part.text()).strip()
                elif part.name == "subject":
                    subject = (await part.text()).strip()
                elif part.name == "photo":
                    jpgs.append(await part.read(decode=False))
        except ValueError:
            pass
        if not jpgs or not subject:
            return web.json_response({"ok": False, "error": "нужны subject и photo"}, status=400)
        if kind not in ("car", "face"):
            return web.json_response({"ok": False, "error": "kind must be car|face"}, status=400)
        added, errors = 0, []
        for jpg in jpgs:
            result = await loop.run_in_executor(None, brain.enroll, kind, subject, jpg)
            if result.get("ok"):
                added += int(result.get("added") or 0)
            else:
                errors.append(str(result.get("error") or "ошибка"))
        return web.json_response({
            "ok": added > 0 or not errors,
            "added": added, "files": len(jpgs),
            "errors": errors,
        })

    async def handle_events(req: web.Request) -> web.Response:
        q = req.rel_url.query
        def _fnum(key):
            v = q.get(key)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None
        filters = {
            "kind": (q.get("kind") or "").strip() or None,
            "action": (q.get("action") or "").strip() or None,
            "camera": (q.get("camera") or "").strip() or None,
            "subject": (q.get("subject") or "").strip() or None,
            "score_min": _fnum("score_min"),
            "score_max": _fnum("score_max"),
            "from": _fnum("from"),
            "to": _fnum("to"),
        }
        try:
            limit = min(1000, max(1, int(q.get("limit") or 200)))
        except ValueError:
            limit = 200
        return web.json_response({"events": store.query_events(filters, limit)})

    async def handle_photos(req: web.Request) -> web.Response:
        """Список фото subject: /api/photos?kind=face&subject=Я"""
        q = req.rel_url.query
        kind = (q.get("kind") or "").strip()
        subject = (q.get("subject") or "").strip()
        if kind not in ("car", "face") or not subject:
            return web.json_response({"ok": False, "error": "нужны kind=car|face и subject"}, status=400)
        try:
            return web.json_response({"ok": True, "photos": store.list_photos(kind, subject)})
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    async def handle_photo(req: web.Request) -> web.Response:
        """Раздача фото: /api/photo/{kind}/{subject}/{file} (для <img src>)."""
        kind = req.match_info.get("kind", "")
        subject = req.match_info.get("subject", "")
        fname = req.match_info.get("file", "")
        kind = {"cars": "car", "faces": "face"}.get(kind, kind)
        if kind not in ("car", "face"):
            return web.json_response({"ok": False, "error": "bad kind"}, status=400)
        p = store.photo_path(kind, subject, fname)
        if p is None:
            return web.json_response({"ok": False, "error": "фото не найдено"}, status=404)
        return web.FileResponse(p)

    async def handle_photos_delete(req: web.Request) -> web.Response:
        """Удаление ошибочных фото + авто-переобучение subject."""
        body = await req.json()
        kind = str(body.get("kind") or "")
        subject = str(body.get("subject") or "")
        files = body.get("files") or []
        if kind not in ("car", "face") or not subject or not files:
            return web.json_response({"ok": False, "error": "нужны kind, subject, files[]"}, status=400)
        try:
            deleted = store.delete_photos(kind, subject, files)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        # после удаления прототипы должны соответствовать факту: переобучаем
        started = brain.retrain_async(kind, subject)
        return web.json_response({"ok": True, "deleted": deleted, "retrain_started": started})

    async def handle_retrain(req: web.Request) -> web.Response:
        """Переобучение: /api/retrain {kind, subject?} — фон, статус в /api/status.retrain."""
        body = await req.json()
        kind = str(body.get("kind") or "")
        subject = str(body.get("subject") or "").strip() or None
        if kind not in ("car", "face"):
            return web.json_response({"ok": False, "error": "kind must be car|face"}, status=400)
        ok = brain.retrain_async(kind, subject)
        if not ok:
            return web.json_response({"ok": False, "error": "переобучение уже идёт"}, status=409)
        return web.json_response({"ok": True, "started": True})

    async def handle_settings_get(_req: web.Request) -> web.Response:
        return web.json_response({
            "gestures_enabled": bool((cfg.CONF.gestures or {}).get("enabled")),
            "alarm_enabled": bool((cfg.CONF.alarm or {}).get("enabled")),
        })

    async def handle_settings_post(req: web.Request) -> web.Response:
        """Переключатели панели: /data/settings.json (переживают перенастройку)."""
        body = await req.json()
        sp = os.path.join(cfg.DATA_DIR, "settings.json")
        cur: dict = {}
        try:
            with open(sp, "r", encoding="utf-8") as fh:
                cur = json.load(fh)
        except Exception:  # noqa: BLE001
            cur = {}
        for key in ("gestures_enabled", "alarm_enabled"):
            if key in body and isinstance(body[key], bool):
                cur[key] = body[key]
        import json as _json
        tmp = sp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(cur, fh)
        os.replace(tmp, sp)
        cfg.reload_conf()
        brain.publish_status()
        return web.json_response({"ok": True, **{
            "gestures_enabled": bool((cfg.CONF.gestures or {}).get("enabled")),
            "alarm_enabled": bool((cfg.CONF.alarm or {}).get("enabled")),
        }})

    async def handle_alarm_test(_req: web.Request) -> web.Response:
        return web.json_response(brain.alarm_test())

    async def handle_events_archive(_req: web.Request) -> web.Response:
        return web.json_response(store.archive_events())

    async def handle_events_clear(_req: web.Request) -> web.Response:
        count = store.clear_events()
        LOG.warning("Журнал очищен (%s событий) из панели", count)
        return web.json_response({"ok": True, "cleared": count})

    async def handle_events_archives(_req: web.Request) -> web.Response:
        return web.json_response({"archives": store.list_archives()})

    async def handle_events_archive_file(req: web.Request) -> web.Response:
        name = req.rel_url.query.get("name") or ""
        p = store.archive_path(name)
        if p is None or not os.path.isfile(p):
            return web.json_response({"ok": False, "error": "архив не найден"}, status=404)
        return web.FileResponse(p)

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
    app.router.add_get("/api/gestures", handle_gestures)
    app.router.add_post("/api/gestures/config", handle_gestures_config)
    app.router.add_post("/api/gestures/enroll", handle_gestures_enroll)
    app.router.add_get("/api/gestures/photos", handle_gestures_photos)
    app.router.add_get("/api/gestures/photo/{subject}/{gesture}/{file}", handle_gestures_photo)
    app.router.add_post("/api/gestures/photos/delete", handle_gestures_photos_delete)
    app.router.add_post("/api/gestures/test", handle_gestures_test)
    app.router.add_get("/api/camera_snapshot", handle_camera_snapshot)
    app.router.add_get("/api/events", handle_events)
    app.router.add_post("/api/events/archive", handle_events_archive)
    app.router.add_post("/api/events/clear", handle_events_clear)
    app.router.add_get("/api/events/archives", handle_events_archives)
    app.router.add_get("/api/events/archive-file", handle_events_archive_file)
    app.router.add_get("/api/photos", handle_photos)
    app.router.add_get("/api/photo/{kind}/{subject}/{file}", handle_photo)
    app.router.add_post("/api/photos/delete", handle_photos_delete)
    app.router.add_post("/api/retrain", handle_retrain)
    app.router.add_get("/api/settings", handle_settings_get)
    app.router.add_post("/api/settings", handle_settings_post)
    app.router.add_post("/api/alarm/test", handle_alarm_test)
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
