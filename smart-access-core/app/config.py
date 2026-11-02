"""Конфигурация Smart Access Core (опции аддона + сервисные параметры)."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

LOG = logging.getLogger("smart_access.config")

MODELS_DIR = os.environ.get("SMART_ACCESS_MODELS_DIR", "/models")
DATA_DIR = os.environ.get("SMART_ACCESS_DATA_DIR", "/data")
MEDIA_DIR = os.environ.get("SMART_ACCESS_MEDIA_DIR", "/media/smart_access")

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")

# База Supervisor API определяется один раз и кэшируется
_SV_BASE: dict[str, Any] = {"url": None, "ts": 0.0, "fail_ts": 0.0}


def supervisor_base() -> str:
    """База Supervisor API: http://supervisor или IP-алиас 172.30.32.2.

    Имя 'supervisor' резолвится docker-DNS внутри моста hassio; из
    host-network окружения (наш аддон) оно может быть недоступно — тогда
    используем известный IP Supervisor в hassio-сети.
    """
    now = time.time()
    if _SV_BASE["url"] and now - _SV_BASE["ts"] < 600:
        return _SV_BASE["url"]
    if _SV_BASE["fail_ts"] and now - _SV_BASE["fail_ts"] < 30:
        return "http://supervisor"  # недавно оба варианта не ответили — не долбим
    import requests
    headers = {"Authorization": f"Bearer {SUPERVISOR_TOKEN}"} if SUPERVISOR_TOKEN else {}
    for base in ("http://supervisor", "http://172.30.32.2"):
        try:
            # любой ответ (даже 401) доказывает достижимость
            requests.get(f"{base}/addons", headers=headers, timeout=3)
            _SV_BASE.update({"url": base, "ts": now})
            if base != "http://supervisor":
                LOG.info("Supervisor API доступен по IP-алиасу %s (имя 'supervisor' не резолвится)", base)
            return base
        except Exception:  # noqa: BLE001
            continue
    _SV_BASE["fail_ts"] = now
    return "http://supervisor"


def _read_options() -> dict[str, Any]:
    """Читает /data/options.json (аддон-UI) с безопасными дефолтами."""
    defaults: dict[str, Any] = {
        "master_enable": True,
        "frigate_url": "http://127.0.0.1:8971",
        "mqtt": {"host": "", "port": 1883, "user": "", "password": ""},
        "cameras": [],
        "recognition": {
            "car_threshold": 0.76,
            "car_gray_zone": 0.06,
            "face_threshold": 0.363,
            "min_confirm_frames": 2,
            "bridge_fps": 1.5,
            "unknown_keep_days": 7,
        },
        "vlm": {
            "enabled": False,
            "base_url": "",
            "api_key": "",
            "model": "",
            "mode": "gray_zone",
            "timeout_sec": 6,
        },
        # Жесты: подтверждение открытия (одна рука вверх) и тревога (обе руки).
        # Требуют модель /models/yolov8n-pose.onnx (вшита в образ).
        "gestures": {
            "enabled": False,
            "min_frames": 1,
        },
        # Тревога (насильственное открытие под угрозой): звонок через ha-sip
        # с текстом (Piper озвучивает TTS-движком HA). numbers — по одному
        # номеру в строке, например 'sip:89256758881@pbx.kozhevatov.ru'.
        "alarm": {
            "enabled": False,
            "addon": "c7744bff_ha-sip",
            "numbers": "",
            "sip_account": 1,
            "ring_timeout": 15,
            "language": "ru",
            "message_template": "Тревога! {time}, {name} открыл калитку с тревожным жестом.",
            "notify": True,
        },
    }
    try:
        with open("/data/options.json", "r", encoding="utf-8") as fh:
            user = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Не удалось прочитать /data/options.json: %s — работаю на дефолтах", exc)
        return defaults

    def merge(dst: dict, src: dict) -> dict:
        for key, val in src.items():
            if isinstance(val, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], val)
            else:
                dst[key] = val
        return dst

    return merge(defaults, user or {})


def slugify_camera_name(name: str) -> str:
    """Имя камеры в формате Frigate: только [a-z0-9_].

    Frigate отвергает имена с кириллицей и точками («а.м.видео», «егдс»,
    «калитка» — реальные имена камер пользователя). Нормализуем: транслит
    кириллицы, прочие символы → «_», схлопывание и обрезка подчёркиваний.
    """
    translit = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
    out: list[str] = []
    for ch in str(name or "").strip().lower():
        if ch in translit:
            out.append(translit[ch])
        elif (ch.isascii() and ch.isalnum()) or ch in "-_":
            out.append(ch)
        else:
            out.append("_")
    import re as _re
    slug = _re.sub(r"_+", "_", "".join(out)).strip("_-")
    return slug or "camera"


class Config:
    """Иммутабельный снимок конфигурации (перечитывается по SIGHUP/на кнопке в UI)."""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        opts = _read_options()
        self.master_enable: bool = bool(opts["master_enable"])
        # Kill-switch из UI аддона: /data/master.json имеет приоритет над опциями
        # (Supervisor может перезаписывать options.json при перенастройке).
        try:
            import json as _json
            import os as _os
            mp = _os.path.join(DATA_DIR, "master.json")
            if _os.path.exists(mp):
                with open(mp, "r", encoding="utf-8") as fh:
                    self.master_enable = bool(_json.load(fh).get("enabled", self.master_enable))
        except Exception:  # noqa: BLE001
            pass
        self.frigate_url: str = str(opts["frigate_url"]).rstrip("/")
        self.mqtt: dict[str, Any] = opts["mqtt"]

        # Имена камер приводим к формату Frigate ([a-z0-9_]): пользователь
        # вписывает «а.м.видео», «егдс», «калитка» — Frigate такие ключи
        # отвергает, а MQTT-события приходят с slug-именами.
        cams: list[dict[str, Any]] = []
        used: set[str] = set()
        for c in list(opts.get("cameras") or []):
            if not isinstance(c, dict):
                continue
            orig = str(c.get("name") or "").strip()
            slug = slugify_camera_name(orig)
            base, n = slug, 2
            while slug in used:
                slug = f"{base}_{n}"
                n += 1
            used.add(slug)
            cc = dict(c)
            cc["name"] = slug
            if orig and slug != orig:
                cc["display_name"] = orig  # исходное имя для подписи в конфиге/UI
                LOG.warning("Имя камеры «%s» нормализовано в «%s» (формат Frigate: [a-z0-9_])",
                            orig, slug)
            cams.append(cc)
        self.cameras: list[dict[str, Any]] = cams
        self.recognition: dict[str, Any] = opts["recognition"]
        self.vlm: dict[str, Any] = opts["vlm"]
        self.gestures: dict[str, Any] = opts.get("gestures") or {"enabled": False, "min_frames": 1}
        self.alarm: dict[str, Any] = opts.get("alarm") or {}

        # /data/settings.json — переключатели из панели (переживают
        # перенастройку аддона, как master.json). Не None -> переопределяем.
        try:
            sp = os.path.join(DATA_DIR, "settings.json")
            if os.path.exists(sp):
                with open(sp, "r", encoding="utf-8") as fh:
                    s = json.load(fh)
                if isinstance(s.get("gestures_enabled"), bool):
                    self.gestures["enabled"] = s["gestures_enabled"]
                if isinstance(s.get("alarm_enabled"), bool):
                    self.alarm["enabled"] = s["alarm_enabled"]
        except Exception as exc:  # noqa: BLE001
            LOG.debug("settings.json не прочитан: %s", exc)

        # MQTT: если в опциях пусто или указан localhost (а наш аддон работает
        # в host_network), берём адрес/креды брокера у Supervisor (сервис mqtt:want).
        # Имя core-mosquitto из host_network не резолвится — слушаем
        # опубликованный на хосте порт по 127.0.0.1.
        host = str(self.mqtt.get("host") or "").strip()
        user_opt = str(self.mqtt.get("user") or "").strip()
        pass_opt = str(self.mqtt.get("password") or "")
        if host in ("", "localhost", "127.0.0.1", "::1"):
            if user_opt and pass_opt:
                # полная пара пользователя — используем как есть
                self.mqtt = {**self.mqtt, "host": "127.0.0.1"}
                LOG.info("MQTT: креды из опций (user=%s)", user_opt)
            else:
                svc = discover_mqtt_service()
                if svc:
                    # НЕ смешиваем: неполная пара из опций (например user «mqtt»
                    # без пароля) + пароль Supervisor — несовпадающая пара,
                    # у Frigate с ней «MQTT Not authorized». Пара берётся
                    # ЦЕЛИКОМ из одного источника.
                    self.mqtt = {
                        **self.mqtt,
                        "host": "127.0.0.1",
                        "port": svc.get("port") or self.mqtt.get("port") or 1883,
                        "user": svc.get("user") or "",
                        "password": svc.get("password") or "",
                    }
                    LOG.info("MQTT: брокер обнаружен через Supervisor: 127.0.0.1:%s (user=%s)",
                             self.mqtt["port"], self.mqtt["user"] or "-")
                elif host in ("", "localhost"):
                    self.mqtt = {**self.mqtt, "host": "127.0.0.1"}

    def camera_by_name(self, name: str) -> dict[str, Any] | None:
        for cam in self.cameras:
            if cam.get("name") == name:
                return cam
        return None

    def camera_by_entity(self, entity_id: str) -> dict[str, Any] | None:
        for cam in self.cameras:
            if cam.get("ha_entity") == entity_id:
                return cam
        return None

    def as_safe_dict(self) -> dict[str, Any]:
        """Конфиг для UI без секретов."""
        vlm = dict(self.vlm)
        if vlm.get("api_key"):
            vlm["api_key"] = "***"
        mqtt = dict(self.mqtt)
        if mqtt.get("password"):
            mqtt["password"] = "***"
        return {
            "master_enable": self.master_enable,
            "frigate_url": self.frigate_url,
            "mqtt": mqtt,
            "cameras": self.cameras,
            "recognition": self.recognition,
            "vlm": vlm,
            "gestures": self.gestures,
            "alarm": {**self.alarm, "numbers": self.alarm.get("numbers") or ""},
        }


def discover_mqtt_service() -> dict[str, Any] | None:
    """Спрашивает у Supervisor настройки MQTT-брокера (mosquitto-аддон)."""
    if not SUPERVISOR_TOKEN:
        return None
    import requests

    try:
        resp = requests.get(
            f"{supervisor_base()}/services/mqtt",
            headers={"Authorization": f"Bearer {SUPERVISOR_TOKEN}"},
            timeout=5,
        )
        if resp.ok:
            data = resp.json()["data"]
            return {
                "host": data.get("host") or "core-mosquitto",
                "port": int(data.get("port") or 1883),
                "user": data.get("username") or "",
                "password": data.get("password") or "",
            }
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Supervisor /services/mqtt недоступен: %s", exc)
    return None


# Глобальный снимок конфигурации; перечитывается reload_conf().
CONF: Config = Config()


def reload_conf() -> Config:
    """Перечитать /data/options.json в общий снимок CONF."""
    CONF.reload()
    return CONF
