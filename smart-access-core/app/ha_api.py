"""Клиенты Home Assistant (через Supervisor-прокси) и Frigate.

Все вызовы устойчивы: таймауты + ретраи, любое падение не роняет систему —
функции возвращают None/False, вызывающий код решает сам.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from . import config as cfg

LOG = logging.getLogger("smart_access.ha")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {cfg.SUPERVISOR_TOKEN}"}


def call_ha_service(domain: str, service: str, data: dict[str, Any] | None = None,
                    retries: int = 3) -> bool:
    """Вызов сервиса HA из аддона: POST http://supervisor/core/api/services/..."""
    if not cfg.SUPERVISOR_TOKEN:
        LOG.error("SUPERVISOR_TOKEN отсутствует — вызов сервиса %s.%s невозможен", domain, service)
        return False
    url = f"http://supervisor/core/api/services/{domain}/{service}"
    payload = data or {}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=_headers(), json=payload, timeout=10)
            if resp.ok:
                LOG.info("HA-сервис вызван: %s.%s %s", domain, service, payload)
                return True
            LOG.warning("HA-сервис %s.%s -> HTTP %s (попытка %s)", domain, service, resp.status_code, attempt)
        except requests.RequestException as exc:
            LOG.warning("HA-сервис %s.%s недоступен: %s (попытка %s)", domain, service, exc, attempt)
        time.sleep(2 * attempt)
    return False


def get_camera_image(entity_id: str) -> bytes | None:
    """JPEG-кадр с камеры HA (camera_proxy — работает и для облачных камер)."""
    if not cfg.SUPERVISOR_TOKEN:
        return None
    try:
        resp = requests.get(
            f"http://supervisor/core/api/camera_proxy/{entity_id}",
            headers=_headers(),
            timeout=12,
        )
        if resp.ok and resp.content[:3] == b"\xff\xd8\xff":
            return resp.content
        LOG.debug("camera_proxy %s -> %s", entity_id, resp.status_code)
    except requests.RequestException as exc:
        LOG.debug("camera_proxy %s недоступна: %s", entity_id, exc)
    return None


def get_ha_state(entity_id: str) -> dict[str, Any] | None:
    if not cfg.SUPERVISOR_TOKEN:
        return None
    try:
        resp = requests.get(
            f"http://supervisor/core/api/states/{entity_id}",
            headers=_headers(), timeout=8,
        )
        if resp.ok:
            return resp.json()
    except requests.RequestException:
        pass
    return None


# ---------------- Frigate ----------------

def frigate_event_snapshot(frigate_url: str, event_id: str, crop: bool = True) -> bytes | None:
    """Лучший кадр события Frigate (?crop=1 — кроп по bbox объекта)."""
    url = f"{frigate_url}/api/events/{event_id}/snapshot.jpg?quality=90" + ("&crop=1" if crop else "")
    try:
        resp = requests.get(url, timeout=10)
        if resp.ok and resp.content[:3] == b"\xff\xd8\xff":
            return resp.content
    except requests.RequestException as exc:
        LOG.debug("Frigate snapshot %s недоступен: %s", event_id, exc)
    return None


def frigate_probe(frigate_url: str) -> dict[str, Any]:
    """Проверка связи с Frigate API + ПОНЯТНАЯ причина ошибки для UI.

    Никогда не бросает исключений: возвращает {'ok': bool, 'error': str | None}.
    """
    url = f"{frigate_url}/api/version"
    try:
        resp = requests.get(url, timeout=4)
    except requests.RequestException as exc:
        hint = ""
        s = str(exc).lower()
        if "refused" in s:
            hint = (" Соединение отклонено: приложение Frigate не запущено, либо порт 5000 "
                    "не опубликован. Откройте Frigate → Конфигурация → Сеть и впишите 5000 "
                    "в поле «Unauthenticated Web interface», затем Сохранить.")
        elif "timed out" in s or "timeout" in s:
            hint = " Frigate не ответил за 4 с — проверьте, что приложение запущено."
        return {"ok": False, "error": f"Нет связи с {url}: {exc}.{hint}"}
    if resp.ok:
        return {"ok": True, "error": None}
    ctype = resp.headers.get("content-type", "")
    if "text/html" in ctype:
        return {"ok": False, "error": (
            f"По адресу {url} отвечает не Frigate API, а HTML-страница (HTTP {resp.status_code}) — "
            "почти наверняка указан неверный порт в frigate_url. В аддоне Frigate порт 8556 — "
            "это WebRTC (go2rtc), а HTTP API — порт 5000. Поставьте frigate_url = "
            "http://127.0.0.1:5000 и опубликуйте порт 5000 в настройках сети приложения "
            "Frigate (Сеть → «Unauthenticated Web interface» → 5000).")}
    return {"ok": False, "error": f"{url} -> HTTP {resp.status_code} ({ctype or 'без content-type'})"}


def frigate_ping(frigate_url: str) -> bool:
    """Совместимость: просто bool из frigate_probe."""
    return frigate_probe(frigate_url)["ok"]
