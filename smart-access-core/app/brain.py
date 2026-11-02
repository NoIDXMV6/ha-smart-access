"""Мозг системы: Frigate-события -> локальное распознавание -> решение -> открытие.

Логика устойчивости (система безопасности):
  * Решение принимает ТОЛЬКО локальный пайплайн (CLIP/SFace, CPU).
  * VLM — факультативный «второй голос»: подтверждает совпадение в серой зоне
    или (режим verify_all) каждое срабатывание. Недоступен -> локальное решение.
  * master_enable=false или failure моделей -> НИКАКИХ авто-открытий (fail-safe),
    но мост/статус/UI продолжают работать.
  * Подтверждение N кадрами подряд (min_confirm_frames) против ложных срабатываний.
  * Cooldown на пару (камера, subject) + учёт игнора повторных команд шлагбаумом (~20 c).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

import paho.mqtt.client as mqtt

from . import config as cfg
from . import gestures, ha_api, models, store, vlm
from .store import Store

LOG = logging.getLogger("smart_access.brain")


def _in_schedule(spec: str) -> bool:
    """Формат '07:00-23:00' или '' = всегда. Локальное время."""
    if not spec or "-" not in spec:
        return True
    try:
        start, end = [x.strip() for x in spec.split("-", 1)]
        now = time.localtime()
        cur = now.tm_hour * 60 + now.tm_min
        sh, sm = map(int, start.split(":"))
        eh, em = map(int, end.split(":"))
        return sh * 60 + sm <= cur <= eh * 60 + em
    except Exception:  # noqa: BLE001
        return True


class Brain:
    def __init__(self, store: Store, models_: models.Models,
                 on_status: Callable[[], None], bridge=None) -> None:
        self.store = store
        self.models = models_
        self.on_status = on_status
        # v1.3.0: мост (если передан) — источник живых кадров для identify
        # без новой сессии к камере (у камеры и так один клиент — мост)
        self.bridge = bridge
        self.client: mqtt.Client | None = None
        self.stats: dict[str, Any] = {
            "events_seen": 0, "grants": 0, "denied": 0, "alarms": 0,
            "vlm_calls": 0, "last_grant": None, "started": time.time(),
            "mqtt": "подключение…",
        }
        self.retrain_status: dict[str, Any] = {"running": False}
        self._seen_events: dict[str, float] = {}
        self._confirm: dict[str, dict[str, float]] = {}  # event_id -> {subject, count, score}
        self._cooldowns: dict[str, float] = {}           # cam|subject -> ts
        self._processed: set[str] = set()
        self._unknown_logged: set[str] = set()
        self._lock = threading.RLock()

    # ---------------- MQTT ----------------

    def start_mqtt(self) -> None:
        conf = cfg.CONF
        host = conf.mqtt.get("host") or "core-mosquitto"
        port = int(conf.mqtt.get("port") or 1883)
        client = mqtt.Client(client_id="smart-access-core", clean_session=True)
        if conf.mqtt.get("user"):
            client.username_pw_set(conf.mqtt["user"], conf.mqtt.get("password") or None)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        client.reconnect_delay_set(min_delay=2, max_delay=30)
        self.client = client
        try:
            client.connect_async(host, port, keepalive=30)
            client.loop_start()
            LOG.info("MQTT: подключаюсь к %s:%s", host, port)
        except Exception as exc:  # noqa: BLE001
            LOG.error("MQTT: не удалось подключиться: %s", exc)

    def _on_connect(self, client, _u, _f, rc, _p=None):  # noqa: ANN001
        LOG.info("MQTT: подключено (rc=%s)", rc)
        self.stats["mqtt"] = "ok" if rc == 0 else f"ошибка брокера (rc={rc})"
        client.subscribe("frigate/events", qos=1)
        client.subscribe("smart_access/enroll/#", qos=1)
        client.subscribe("smart_access/cmd/#", qos=1)
        self.publish_status()

    def _on_disconnect(self, *_a, **_k):  # noqa: ANN001
        self.stats["mqtt"] = "нет связи"
        LOG.warning("MQTT: отключение — авто-реконнект")

    def _on_message(self, _c, _u, msg: mqtt.MQTTMessage):  # noqa: ANN001
        try:
            if msg.topic == "frigate/events":
                self._handle_frigate_event(msg.payload)
            elif msg.topic.startswith("smart_access/enroll/"):
                self._handle_enroll(msg.topic.rsplit("/", 1)[-1], msg.payload)
            elif msg.topic.startswith("smart_access/cmd/"):
                self._handle_cmd(msg.topic.rsplit("/", 1)[-1], msg.payload)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Обработка MQTT-сообщения (%s) упала: %s", msg.topic, exc)

    # ---------------- Frigate ----------------

    def _handle_frigate_event(self, payload: bytes) -> None:
        conf = cfg.CONF
        try:
            data = json.loads(payload)
        except ValueError:
            return
        etype = data.get("type")
        after = data.get("after") or {}
        event_id = after.get("id")
        label = after.get("label")
        camera = after.get("camera")
        if not event_id or not label or not camera:
            return
        with self._lock:
            if etype == "end":
                # событие завершено: снимаем подтверждения
                self._confirm.pop(event_id, None)
                self._seen_events.pop(event_id, None)
                return
            if etype not in ("new", "update"):
                return
            # троттлинг по событию: кадр улучшается по ходу трека, обрабатываем
            # не чаще раза в 2 с (подтверждение N кадрами копится между раундами)
            now = time.time()
            if now - self._seen_events.get(event_id, 0) < 2.0:
                return
            self._seen_events[event_id] = now
            if len(self._seen_events) > 2000:
                cutoff = now - 3600
                self._seen_events = {k: v for k, v in self._seen_events.items() if v > cutoff}
            if len(self._processed) > 5000:
                self._processed = set()
            if len(self._unknown_logged) > 5000:
                self._unknown_logged = set()
        self.stats["events_seen"] += 1

        cam = conf.camera_by_name(camera)
        if cam is None:
            return  # камера не заведена в Smart Access
        mode = cam.get("mode", "off")
        if label == "car" and "car" in mode:
            self._process_car(cam, event_id)
        elif label == "person" and "face" in mode:
            self._process_face(cam, event_id)

    # ---------------- машины ----------------

    def _process_car(self, cam: dict[str, Any], event_id: str) -> None:
        conf = cfg.CONF
        rec = conf.recognition
        name = cam["name"]
        threshold = float(rec.get("car_threshold", 0.76))
        gray = float(rec.get("car_gray_zone", 0.06))
        need_confirm = int(rec.get("min_confirm_frames", 2))

        if not conf.master_enable:
            self._log_event("car", name, "denied", "выключено (master_enable)", 0)
            return
        if not self.models.ok:
            self._log_event("car", name, "denied", "модели не загружены (fail-safe)", 0)
            return
        if not _in_schedule(cam.get("schedule", "")):
            return
        if event_id in self._processed:
            return

        jpg = ha_api.frigate_event_snapshot(conf.frigate_url, event_id, crop=True)
        if not jpg:
            jpg = ha_api.frigate_event_snapshot(conf.frigate_url, event_id, crop=False)
        if not jpg:
            return

        embeds = self.models.car_embedding(jpg, with_views=True)
        if embeds is None:
            return
        best_subject, best_score = store.Store.match(self.store.cars, embeds[0])
        # уточняем по всем аугментациям (медианный подход: max по view'ам)
        for vec in embeds[1:]:
            s2, sc2 = store.Store.match(self.store.cars, vec)
            if sc2 > best_score:
                best_subject, best_score = s2, sc2

        LOG.info("car %s: best=%s score=%.3f (threshold=%.2f)", name, best_subject, best_score, threshold)

        vlm_mode = conf.vlm.get("mode", "gray_zone") if conf.vlm.get("enabled") else "off"
        vlm_opinion: bool | None = None
        if best_subject is not None and threshold - gray <= best_score < threshold and vlm_mode in ("gray_zone", "verify_all"):
            self.stats["vlm_calls"] += 1
            vlm_opinion = vlm.verify_same_car(self._reference_photo("cars", best_subject), jpg)
            LOG.info("VLM gray-zone: %s (%.3f) -> %s", best_subject, best_score, vlm_opinion)

        decided = best_score >= threshold or (
            best_subject is not None and vlm_opinion is True and best_score >= threshold - gray
        )

        if not decided:
            if (best_subject is None or best_score < threshold - gray) and event_id not in self._unknown_logged:
                self._confirm.pop(event_id, None)
                self._unknown_logged.add(event_id)
                self.store.save_unknown("car", jpg)
                self._log_event("car", name, "unknown", f"score={best_score:.3f}", best_score)
                self.stats["denied"] += 1
            return

        # подтверждение N-м кадром (разные раунды обработки одного трека)
        pending = self._confirm.setdefault(event_id, {"subject": best_subject, "count": 0, "score": 0.0})
        if pending.get("subject") != best_subject:
            pending = self._confirm[event_id] = {"subject": best_subject, "count": 0, "score": 0.0}
        pending["count"] += 1
        pending["score"] = max(float(pending.get("score") or 0.0), best_score)
        if need_confirm > 1 and pending["count"] < need_confirm:
            return
        self._confirm.pop(event_id, None)
        self._processed.add(event_id)

        self._grant("car", cam, best_subject, pending["score"], jpg)

    # ---------------- лица ----------------

    def _process_face(self, cam: dict[str, Any], event_id: str) -> None:
        conf = cfg.CONF
        rec = conf.recognition
        name = cam["name"]
        threshold = float(rec.get("face_threshold", 0.363))
        if not conf.master_enable:
            self._log_event("face", name, "denied", "выключено (master_enable)", 0)
            return
        if not self.models.faces_ok:
            self._log_event("face", name, "denied", "модели лиц не загружены (fail-safe)", 0)
            return
        if not _in_schedule(cam.get("schedule", "")):
            return
        if not self.store.faces:
            return
        if event_id in self._processed:
            return

        jpg_crop = ha_api.frigate_event_snapshot(conf.frigate_url, event_id, crop=True)
        jpg_full = ha_api.frigate_event_snapshot(conf.frigate_url, event_id, crop=False)
        if not jpg_full:
            return

        import numpy as np

        best_subject, best_score = None, -1.0
        best_face_jpg: bytes | None = None
        for emb in self.models.face_embeddings(jpg_full, jpg_crop):
            vec = np.asarray(emb, dtype=np.float32)
            s2, sc2 = store.Store.match(self.store.faces, vec)
            if sc2 > best_score:
                best_subject, best_score = s2, sc2
                best_face_jpg = jpg_crop or jpg_full

        if best_subject is None:
            if event_id not in self._unknown_logged:
                self._unknown_logged.add(event_id)
                self.store.save_unknown("face", jpg_full)
                self._log_event("face", name, "unknown", "лицо не распознано", 0)
                self.stats["denied"] += 1
            return

        LOG.info("face %s: best=%s score=%.3f (threshold=%.3f)", name, best_subject, best_score, threshold)

        vlm_mode = conf.vlm.get("mode", "gray_zone") if conf.vlm.get("enabled") else "off"
        decided = best_score >= threshold
        if not decided and vlm_mode == "verify_all":
            self.stats["vlm_calls"] += 1
            if vlm.verify_same_face(self._reference_photo("faces", best_subject), best_face_jpg or jpg_full):
                decided = True  # VLM подтвердил

        if not decided:
            if event_id not in self._unknown_logged:
                self._unknown_logged.add(event_id)
                self.store.save_unknown("face", jpg_full)
                self._log_event("face", name, "unknown", f"score={best_score:.3f}", best_score)
                self.stats["denied"] += 1
            return

        # --- жесты (v1.3.0): настраиваемые, отдельно для каждого лица ---
        gest = conf.gestures or {}
        gestures_on = bool(gest.get("enabled")) and self.models.gestures_ok
        if gestures_on:
            if self._evaluate_gestures(cam, name, event_id, best_subject,
                                       best_score, best_face_jpg or jpg_full):
                return

        self._processed.add(event_id)
        self._grant("face", cam, best_subject, best_score, best_face_jpg or jpg_full)

    # ---------------- настраиваемые жесты (v1.3.0) ----------------

    @staticmethod
    def _gesture_list(value: Any, fallback: list[str]) -> list[str]:
        """Список id жестов из конфига: список или 'a,b' строка."""
        return gestures.gesture_list_safe(value, fallback)

    def _gesture_sets_for(self, subject: str) -> tuple[list[str], list[str], bool]:
        """Жесты лица: (confirm, alarm, настроено_ли_персонально).

        Лицо без своей настройки — совместимость v1.2.0: default_confirm /
        default_alarm из опций (по умолчанию «одна рука вверх» / «обе руки»).
        """
        g = self.store.get_subject_gestures(subject)
        if g.get("confirm") or g.get("alarm"):
            return g["confirm"], g["alarm"], True
        gest = cfg.CONF.gestures or {}
        confirm = self._gesture_list(gest.get("default_confirm"), ["one_hand_up"])
        alarm = self._gesture_list(gest.get("default_alarm"), ["both_hands_up"])
        return confirm, alarm, False

    def _evaluate_gestures(self, cam: dict[str, Any], name: str, event_id: str,
                           subject: str, score: float, jpg: bytes) -> bool:
        """Решение по жестам для распознанного лица. True — решение принято.

        Тревожный жест лица -> открытие + инцидент; жест-подтверждение ->
        открытие; иначе — gesture_wait (трек продолжается, следующие
        update-кадры повторят проверку).
        """
        gest = cfg.CONF.gestures or {}
        threshold = float(gest.get("threshold", 0.80))
        confirm_ids, alarm_ids, configured = self._gesture_sets_for(subject)

        # прототипы hand-жестов этого лица (CLIP)
        enrolled: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for gid in dict.fromkeys(confirm_ids + alarm_ids):
            if gestures.GESTURES.get(gid, {}).get("type") != "hand":
                continue
            protos = self.store.gesture_protos(subject, gid)
            if protos is None or len(protos) == 0:
                missing.append(gid)
            else:
                enrolled[gid] = protos

        res = self.models.analyze_gestures(jpg, enrolled=enrolled or None,
                                           threshold=threshold)
        detected = set(res.get("pose") or []) | set(res.get("hands") or [])
        alarm_hits = [g for g in alarm_ids if g in detected]
        confirm_hits = [g for g in confirm_ids if g in detected]
        lab = gestures.gesture_label

        if alarm_hits:
            self._processed.add(event_id)
            note = ", ".join(map(lab, alarm_hits))
            LOG.warning("ALARM: %s на %s — тревожный жест (%s), score=%.3f",
                        subject, name, note, score)
            self._grant("face", cam, subject, score, jpg, alarm=True, gesture_note=note)
            return True
        if confirm_hits:
            self._processed.add(event_id)
            self._grant("face", cam, subject, score, jpg,
                        gesture_note=", ".join(map(lab, confirm_hits)))
            return True

        # жеста нет — ждём: трек жив, следующие кадры повторят проверку
        if event_id not in self._unknown_logged:
            self._unknown_logged.add(event_id)
            want = " или ".join(map(lab, dict.fromkeys(confirm_ids + alarm_ids))) or "—"
            seen = ", ".join(map(lab, sorted(detected))) or "жестов не видно"
            miss = f"; НЕ ОБУЧЕН: {', '.join(map(lab, missing))}" if missing else ""
            suffix = "" if configured else " (жесты лица не настроены — работает старое правило)"
            self._log_event(
                "face", name, "gesture_wait",
                f"{subject} score={score:.3f} — ждём жест: {want}{suffix} "
                f"(в кадре: {seen}{miss})", score, subject)
        return True

    # ---------------- открытие ----------------

    def _grant(self, kind: str, cam: dict[str, Any], subject: str, score: float,
               jpg: bytes, alarm: bool = False, gesture_note: str = "") -> None:
        conf = cfg.CONF
        name = cam["name"]
        cooldown = int(cam.get("cooldown", 45))
        key = f"{name}|{subject}"
        now = time.time()
        if now - self._cooldowns.get(key, 0) < cooldown:
            LOG.info("Допуск %s/%s в cooldown — пропускаю", name, subject)
            return
        self._cooldowns[key] = now

        self.stats["grants"] += 1
        self.stats["last_grant"] = {"camera": name, "subject": subject,
                                    "score": round(score, 3), "ts": now, "alarm": alarm}

        if alarm:
            action = "alarm"
            detail = f"{subject} score={score:.3f} — ТРЕВОЖНЫЙ ЖЕСТ ({gesture_note or 'не указан'})"
        else:
            action = "opened" if (cam.get("executor_service") or "").strip() else "detected"
            detail = f"{subject} score={score:.3f}"
            if gesture_note:
                detail += f" — жест: {gesture_note}"
        self._log_event(kind, name, action, detail, score, subject)

        payload_event = {
            "kind": kind, "camera": name, "camera_entity": cam.get("ha_entity"),
            "subject": subject, "score": round(score, 3), "action": action,
            "alarm": alarm, "gesture": gesture_note or None, "ts": now,
        }
        if self.client:
            self.client.publish("smart_access/event", json.dumps(payload_event, ensure_ascii=False), qos=1)

        if alarm:
            self._alarm(kind, cam, subject, score)

        service = (cam.get("executor_service") or "").strip()
        target = (cam.get("executor_target") or "").strip()
        if service and "." in service:
            domain, svc = service.split(".", 1)
            data: dict[str, Any] = {}
            if target:
                data["entity_id"] = target
            ok = ha_api.call_ha_service(domain, svc, data)
            if not ok:
                LOG.error("Не удалось открыть (%s.%s) — событие уже отправлено в MQTT", domain, svc)

    # ---------------- тревога (SIP-звонки через ha-sip + TTS Piper) ----------------

    def _alarm(self, kind: str, cam: dict[str, Any], subject: str, score: float) -> None:
        """Инцидент безопасности: звонок на тревожные номера с сообщением.

        Звонок — hassio.addon_stdin в аддон ha-sip: command=dial, number,
        menu.message = ТЕКСТ (ha-sip озвучивает его TTS-движком HA —
        у пользователя установлен Piper). post_action=hangup — трубка
        ложится после проигрывания сообщения.
        """
        alarm = cfg.CONF.alarm or {}
        self.stats["alarms"] = self.stats.get("alarms", 0) + 1

        numbers = [n.strip() for n in str(alarm.get("numbers") or "").splitlines() if n.strip()]
        addon = str(alarm.get("addon") or "").strip() or "c7744bff_ha-sip"
        template = str(alarm.get("message_template") or
                       "Тревога! {time}, {name} открыл калитку с тревожным жестом.")
        language = str(alarm.get("language") or "ru")
        ring_timeout = int(alarm.get("ring_timeout") or 15)
        sip_account = int(alarm.get("sip_account") or 1)
        display = cam.get("display_name") or cam["name"]
        message = template.format(
            name=subject, camera=display, kind=kind,
            time=time.strftime("%H:%M"), date=time.strftime("%d.%m.%Y"),
        )
        if not alarm.get("enabled"):
            LOG.warning("ALARM: инцидент %s/%s — звонки отключены (alarm.enabled=false)",
                        cam["name"], subject)
            return
        if not numbers:
            LOG.warning("ALARM: инцидент %s/%s — список тревожных номеров пуст",
                        cam["name"], subject)
            return
        # звонки — в отдельном потоке: не блокируем MQTT-обработку
        threading.Thread(
            target=self._dial_numbers,
            args=(addon, numbers, message, language, ring_timeout, sip_account),
            daemon=True, name="sa-alarm-dial",
        ).start()

    @staticmethod
    def _dial_numbers(addon: str, numbers: list[str], message: str,
                      language: str, ring_timeout: int, sip_account: int) -> None:
        for num in numbers:
            data = {
                "addon": addon,
                "input": {
                    "command": "dial",
                    "number": num,
                    "ring_timeout": ring_timeout,
                    "sip_account": sip_account,
                    "menu": {"message": message, "language": language,
                             "post_action": "hangup"},
                },
            }
            ok = ha_api.call_ha_service("hassio", "addon_stdin", data)
            LOG.warning("ALARM: звонок %s через %s -> %s",
                        num, addon, "отправлен" if ok else "НЕ удался")

    def alarm_test(self) -> dict[str, Any]:
        """Тест тревоги из панели: контрольный звонок на все номера."""
        alarm = cfg.CONF.alarm or {}
        numbers = [n.strip() for n in str(alarm.get("numbers") or "").splitlines() if n.strip()]
        if not numbers:
            return {"ok": False,
                    "error": "список тревожных номеров пуст (Конфигурация аддона → alarm.numbers)"}
        addon = str(alarm.get("addon") or "").strip() or "c7744bff_ha-sip"
        language = str(alarm.get("language") or "ru")
        message = f"Тест тревоги Smart Access. Время {time.strftime('%H:%M')}. Связь с системой работает."
        threading.Thread(
            target=self._dial_numbers,
            args=(addon, numbers, message, language,
                  int(alarm.get("ring_timeout") or 15), int(alarm.get("sip_account") or 1)),
            daemon=True,
        ).start()
        return {"ok": True, "numbers": len(numbers), "addon": addon}

    # ---------------- переобучение ----------------

    def retrain_async(self, kind: str, subject: str | None = None) -> bool:
        """Запуск переобучения в фоне (поток). False — уже идёт."""
        if self.retrain_status.get("running"):
            return False
        self.retrain_status = {"running": True, "kind": kind,
                               "subject": subject or "*", "started": time.time()}
        threading.Thread(target=self._retrain_thread, args=(kind, subject),
                         daemon=True, name="sa-retrain").start()
        return True

    def _retrain_thread(self, kind: str, subject: str | None) -> None:
        try:
            res = self.retrain(kind, subject)
            self.retrain_status.update({"running": False, "finished": time.time(), "last": res})
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Переобучение упало: %s", exc)
            self.retrain_status.update({"running": False, "finished": time.time(),
                                        "last": {"ok": False, "error": str(exc)}})

    def retrain(self, kind: str, subject: str | None = None) -> dict[str, Any]:
        """Пересчёт эмбеддингов по сохранённым фото (/media/smart_access/...).

        После удаления ошибочных фото и/или при обновлении моделей база
        прототипов пересобирается с нуля по фактическим файлам.
        """
        if kind not in ("car", "face"):
            return {"ok": False, "error": "kind must be car|face"}
        photos_map = self.store.all_photos(kind)
        if subject:
            photos_map = {subject: photos_map.get(subject, [])}
        results: dict[str, Any] = {}
        for subj, files in photos_map.items():
            embeds: list[list[float]] = []
            for path in files:
                try:
                    with open(path, "rb") as fh:
                        jpg = fh.read()
                except OSError:
                    continue
                if kind == "car":
                    embs = self.models.car_embedding(jpg, with_views=True)
                    if embs is not None:
                        embeds.extend([list(map(float, v)) for v in embs])
                else:
                    aligned, _w, _h = self.models.detect_face_raw(jpg)
                    if aligned is None:
                        continue
                    import cv2 as _cv2
                    ok, buf = _cv2.imencode(".jpg", aligned)
                    if not ok:
                        continue
                    emb = self.models.face_embedding_of_aligned(buf.tobytes())
                    if emb is not None:
                        embeds.append([float(x) for x in emb])
            self.store.replace_subject(kind, subj, embeds)
            results[subj] = len(embeds)
        LOG.info("Переобучение %s завершено: %s", kind, results)
        self._log_event(kind, "retrain", "retrained",
                        ", ".join(f"{k}: +{v}" for k, v in results.items()), 0)
        self.publish_status()
        return {"ok": True, "results": results}

    # ---------------- обучение hand-жестов (v1.3.0, «как лица») ----------------

    def enroll_gesture(self, subject: str, gesture: str, jpg: bytes) -> dict[str, Any]:
        """Обучение hand-жеста по кадру: поза -> кропы кистей -> CLIP-прототипы.

        С каждого валидного запястья берётся кроп-квадрат, из него 8
        аугментаций -> прототипы. Кропы сохраняются в
        /media/smart_access/gestures/<subject>/<gid>/ — ошибочные можно
        удалить в панели (прототипы пересчитаются).
        """
        gid = str(gesture or "").strip()
        if not gestures.is_valid_gesture(gid):
            return {"ok": False, "error": f"неизвестный жест: {gesture}"}
        if gestures.GESTURES[gid]["type"] != "hand":
            return {"ok": False,
                    "error": "этот жест распознаётся по правилам — обучение не нужно"}
        if not self.models.gestures_ok:
            return {"ok": False, "error": "модель поз недоступна"}
        if not self.models.ok:
            return {"ok": False, "error": "CLIP недоступен"}
        crops = self.models.hand_crop_jpegs(jpg)
        if not crops:
            return {"ok": False,
                    "error": "рука на кадре не найдена (запястье не видно — подойдите ближе)"}
        added = 0
        for _arm, crop_jpg in crops:
            embeds = self.models.gesture_embedding(crop_jpg, n_views=8)
            if embeds:
                self.store.add_gesture_proto(subject, gid, embeds, photo=crop_jpg)
                added += len(embeds)
        if not added:
            return {"ok": False, "error": "CLIP недоступен"}
        LOG.info("Enroll жеста «%s» для «%s»: +%s прототипов", gid, subject, added)
        self._log_event("face", "gesture", "enrolled",
                        f"жест {gestures.gesture_label(gid)}: {subject}", 0, subject)
        self.publish_status()
        return {"ok": True, "added": added, "crops": len(crops)}

    def retrain_gesture(self, subject: str, gesture: str) -> dict[str, Any]:
        """Пересчёт прототипов (subject, gesture) по сохранённым кропам."""
        gid = str(gesture or "").strip()
        if not gestures.is_valid_gesture(gid) or gestures.GESTURES[gid]["type"] != "hand":
            return {"ok": False, "error": "жест не относится к обучаемым"}
        embeds: list[list[float]] = []
        for path in self.store.all_gesture_photos(subject, gid):
            try:
                with open(path, "rb") as fh:
                    jpg = fh.read()
            except OSError:
                continue
            embs = self.models.gesture_embedding(jpg, n_views=8)
            if embs:
                embeds.extend(embs)
        self.store.replace_gesture_protos(subject, gid, embeds)
        LOG.info("Переобучение жеста «%s» для «%s»: %s прототипов", gid, subject, len(embeds))
        self.publish_status()
        return {"ok": True, "added": len(embeds)}

    def test_gesture(self, subject: str, jpg: bytes) -> dict[str, Any]:
        """Проверка кадра: какие настроенные жесты лица видны (для панели)."""
        threshold = float((cfg.CONF.gestures or {}).get("threshold", 0.80))
        confirm_ids, alarm_ids, configured = self._gesture_sets_for(subject)
        enrolled: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for gid in dict.fromkeys(confirm_ids + alarm_ids):
            if gestures.GESTURES.get(gid, {}).get("type") != "hand":
                continue
            protos = self.store.gesture_protos(subject, gid)
            if protos is None or len(protos) == 0:
                missing.append(gid)
            else:
                enrolled[gid] = protos
        res = self.models.analyze_gestures(jpg, enrolled=enrolled or None,
                                           threshold=threshold)
        detected = set(res.get("pose") or []) | set(res.get("hands") or [])
        return {
            "ok": res.get("available", False),
            "persons": res.get("persons", 0),
            "pose": res.get("pose", []),
            "hand_scores": {k: round(float(v), 3)
                            for k, v in (res.get("hand_scores") or {}).items()},
            "threshold": threshold,
            "hits_confirm": [g for g in confirm_ids if g in detected],
            "hits_alarm": [g for g in alarm_ids if g in detected],
            "missing": missing,
            "configured": configured,
            "labels": {g: gestures.gesture_label(g)
                       for g in set(confirm_ids) | set(alarm_ids) | set(detected)},
        }

    # ---------------- enrollment ----------------

    def _handle_enroll(self, kind: str, payload: bytes) -> None:
        try:
            data = json.loads(payload)
        except ValueError:
            return
        subject = str(data.get("subject") or "").strip()
        image_b64 = str(data.get("image_b64") or "")
        if not subject or not image_b64:
            return
        import base64

        try:
            jpg = base64.b64decode(image_b64.split(",")[-1])
        except Exception:  # noqa: BLE001
            return
        self.enroll(kind, subject, jpg)

    def enroll(self, kind: str, subject: str, jpg: bytes) -> dict[str, Any]:
        if kind not in ("car", "face"):
            return {"ok": False, "error": "kind must be car|face"}
        if kind == "car":
            embeds = self.models.car_embedding(jpg, with_views=True)
            if embeds is None:
                return {"ok": False, "error": "CLIP недоступен"}
            self.store.add_car(subject, [list(map(float, v)) for v in embeds], jpg)
            added = len(embeds)
        else:
            if not self.models.faces_ok:
                return {"ok": False, "error": "модели лиц недоступны"}
            aligned, _w, _h = self.models.detect_face_raw(jpg)
            if aligned is None:
                return {"ok": False, "error": "лицо на фото не найдено"}
            ok, buf = None, None
            import cv2 as _cv2
            ok, buf = _cv2.imencode(".jpg", aligned)
            emb = self.models.face_embedding_of_aligned(buf.tobytes()) if ok else None
            if emb is None:
                return {"ok": False, "error": "SFace недоступен"}
            self.store.add_face(subject, [emb], jpg)
            added = 1
        LOG.info("Enroll %s «%s»: +%s эмбеддингов", kind, subject, added)
        self._log_event(kind, "enroll", "enrolled", subject, 0, subject)
        self.publish_status()
        return {"ok": True, "added": added}

    def _handle_cmd(self, cmd: str, payload: bytes) -> None:
        try:
            data = json.loads(payload) if payload else {}
        except ValueError:
            data = {}
        if cmd == "reload":
            cfg.reload_conf()
            self.publish_status()
        elif cmd == "retrain":
            kind = str(data.get("kind") or "face")
            subj = str(data.get("subject") or "").strip() or None
            if kind not in ("car", "face"):
                self._ack({"cmd": "retrain", "ok": False, "error": "kind must be car|face"})
                return
            if not self.retrain_async(kind, subj):
                self._ack({"cmd": "retrain", "ok": False, "error": "переобучение уже идёт"})
                return
            self._ack({"cmd": "retrain", "ok": True, "started": True,
                       "kind": kind, "subject": subj or "*"})
        elif cmd == "identify":
            entity = str(data.get("entity_id") or "")
            kind = str(data.get("kind") or "car")
            # v1.3.0: сначала живой кадр моста (камера не получает ещё одну
            # сессию); нет — прежний путь с запасным потоком (ручное действие,
            # допустимо); для камеры-«потока» это одна сессия, не шторм
            jpg = self.bridge.snapshot_for_entity(entity) if self.bridge else None
            if not jpg:
                jpg = ha_api.get_camera_image_ex(entity)[0]
            if not jpg:
                self._ack({"cmd": "identify", "ok": False, "error": "кадр недоступен"})
                return
            if kind == "car":
                emb = self.models.car_embedding(jpg)
                if emb is None:
                    self._ack({"cmd": "identify", "ok": False, "error": "CLIP недоступен"})
                    return
                import numpy as np
                subject, score = store.Store.match(self.store.cars, np.asarray(emb, dtype=np.float32))
                threshold = float(cfg.CONF.recognition.get("car_threshold", 0.76))
                self._ack({"cmd": "identify", "ok": True, "kind": kind, "subject": subject,
                           "score": round(float(score), 3), "threshold": threshold,
                           "match": bool(subject and score >= threshold)})
            else:
                aligned, _w, _h = self.models.detect_face_raw(jpg)
                if aligned is None:
                    self._ack({"cmd": "identify", "ok": True, "kind": kind, "subject": None, "score": 0.0, "match": False})
                    return
                import cv2 as _cv2, numpy as np
                ok, buf = _cv2.imencode(".jpg", aligned)
                emb = self.models.face_embedding_of_aligned(buf.tobytes()) if ok else None
                if emb is None:
                    self._ack({"cmd": "identify", "ok": False, "error": "SFace недоступен"})
                    return
                subject, score = store.Store.match(self.store.faces, np.asarray(emb, dtype=np.float32))
                threshold = float(cfg.CONF.recognition.get("face_threshold", 0.363))
                self._ack({"cmd": "identify", "ok": True, "kind": kind, "subject": subject,
                           "score": round(float(score), 3), "threshold": threshold,
                           "match": bool(subject and score >= threshold)})

    def _ack(self, payload: dict[str, Any]) -> None:
        if self.client:
            self.client.publish("smart_access/ack", json.dumps(payload, ensure_ascii=False), qos=1)

    # ---------------- служебное ----------------

    def _reference_photo(self, media_kind: str, subject: str) -> bytes | None:
        """Свежее фото subject из /media — для VLM-сравнения."""
        import os

        directory = os.path.join(cfg.MEDIA_DIR, media_kind, subject)
        try:
            files = sorted(f for f in os.listdir(directory) if f.endswith(".jpg"))
            if files:
                with open(os.path.join(directory, files[-1]), "rb") as fh:
                    return fh.read()
        except OSError:
            pass
        return None

    def _log_event(self, kind: str, camera: str, action: str, detail: str, score: float,
                   subject: str | None = None) -> None:
        payload = {"kind": kind, "camera": camera, "action": action,
                   "detail": detail, "score": round(score, 3), "subject": subject}
        self.store.log_event(payload)
        if self.client:
            self.client.publish("smart_access/event", json.dumps(payload, ensure_ascii=False))

    def publish_status(self) -> None:
        if not self.client:
            return
        healthy = self.models.ok or self.models.faces_ok
        payload = {
            "state": "ok" if (cfg.CONF.master_enable and healthy) else "degraded",
            "master_enable": cfg.CONF.master_enable,
            "models": self.models.status,
            "stats": {
                "events_seen": self.stats["events_seen"],
                "grants": self.stats["grants"],
                "denied": self.stats["denied"],
                "vlm_calls": self.stats["vlm_calls"],
                "uptime_sec": int(time.time() - self.stats["started"]),
            },
            "cars": list(self.store.cars.keys()),
            "faces": list(self.store.faces.keys()),
            # v1.4.0: список камер моста для интеграции — она создаёт по ним
            # сущности camera.smart_access_<name> (карточки HA играют мост)
            "cameras": [
                {"name": c.get("name") or "",
                 "display": c.get("display_name") or c.get("name") or "",
                 "entity": c.get("ha_entity") or ""}
                for c in (cfg.CONF.cameras or [])
            ],
            "last_grant": self.stats["last_grant"],
            "gestures": {
                "enabled": bool((cfg.CONF.gestures or {}).get("enabled")),
                "model_ok": self.models.gestures_ok,
                "threshold": float((cfg.CONF.gestures or {}).get("threshold", 0.80)),
                "configured": len(self.store.gestures_configured_subjects()),
            },
            "alarm": {
                "enabled": bool((cfg.CONF.alarm or {}).get("enabled")),
                "numbers": len([n for n in str((cfg.CONF.alarm or {}).get("numbers") or "").splitlines() if n.strip()]),
            },
        }
        self.client.publish("smart_access/status", json.dumps(payload, ensure_ascii=False), retain=True)
