"""Конфигурация Smart Access Core (опции аддона + сервисные параметры)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

LOG = logging.getLogger("smart_access.config")

MODELS_DIR = os.environ.get("SMART_ACCESS_MODELS_DIR", "/models")
DATA_DIR = os.environ.get("SMART_ACCESS_DATA_DIR", "/data")
MEDIA_DIR = os.environ.get("SMART_ACCESS_MEDIA_DIR", "/media/smart_access")

SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN", "")


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
        self.cameras: list[dict[str, Any]] = list(opts.get("cameras") or [])
        self.recognition: dict[str, Any] = opts["recognition"]
        self.vlm: dict[str, Any] = opts["vlm"]

        # MQTT: если в опциях пусто — берём креды брокера у Supervisor (сервис mqtt:want),
        # это же работает и когда аддон подключают к внешнему брокеру через опции.
        if not self.mqtt.get("host"):
            svc = discover_mqtt_service()
            if svc:
                self.mqtt = {**self.mqtt, **svc}
                LOG.info("MQTT получен из Supervisor: %s:%s (user=%s)",
                         svc.get("host"), svc.get("port"), svc.get("user") or "-")

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
        }


def discover_mqtt_service() -> dict[str, Any] | None:
    """Спрашивает у Supervisor настройки MQTT-брокера (mosquitto-аддон)."""
    if not SUPERVISOR_TOKEN:
        return None
    import requests

    try:
        resp = requests.get(
            "http://supervisor/services/mqtt",
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
