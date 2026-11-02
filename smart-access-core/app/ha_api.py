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

# Типовые порты, где Frigate может отвечать HTTP API, если настроенный
# frigate_url не работает: 8971 — ВНЕШНИЙ порт аддона Frigate 0.14+ (панель
# «Конфигурация прослушиваемых портов» → «Внешний порт»), 5000 — внутренний
# API-порт (в оригинальном аддоне — поле «Unauthenticated Web interface»).
_FRIGATE_PORT_CANDIDATES = (8971, 5000)
_DETECTED_TTL = 600.0  # сколько секунд доверяем автоматически найденному адресу

_detected: dict[str, Any] = {"url": None, "ts": 0.0}


def _frigate_version(url: str, timeout: float = 4.0) -> dict[str, Any]:
    """GET {url}/api/version. Не бросает: {'ok','version','kind','error'}.

    kind: ok | refused | timeout | html | http | unreachable.
    HTML-страница вместо JSON — это НЕ Frigate API (чужая служба на порту
    или включённая аутентификация Frigate).
    """
    try:
        resp = requests.get(f"{url}/api/version", timeout=timeout)
    except requests.RequestException as exc:
        s = str(exc).lower()
        if "refused" in s:
            kind = "refused"
        elif "timed out" in s or "timeout" in s:
            kind = "timeout"
        else:
            kind = "unreachable"
        return {"ok": False, "version": None, "kind": kind, "error": str(exc)}
    ctype = resp.headers.get("content-type", "")
    if resp.ok:
        try:
            version = str(resp.json()).strip().strip('"')
        except ValueError:
            return {"ok": False, "version": None, "kind": "html",
                    "error": f"HTTP 200, но тело не JSON ({ctype or 'без content-type'})"}
        return {"ok": True, "version": version, "kind": "ok", "error": None}
    if "text/html" in ctype:
        return {"ok": False, "version": None, "kind": "html",
                "error": f"HTML-страница вместо API (HTTP {resp.status_code})"}
    return {"ok": False, "version": None, "kind": "http",
            "error": f"HTTP {resp.status_code} ({ctype or 'без content-type'})"}


def frigate_probe(frigate_url: str) -> dict[str, Any]:
    """Проверка связи с Frigate: понятная причина ошибки + авто-поиск порта.

    Никогда не бросает исключений:
      {'ok': bool, 'error': str|None, 'version': str|None, 'detected_url': str|None}

    Если настроенный frigate_url не отвечает, пробуем типовые порты (8971,
    5000) на том же хосте. Найденный рабочий адрес возвращается в
    detected_url (его стоит вписать в frigate_url) и используется системой,
    пока в настройках остался неверный адрес.
    """
    first = _frigate_version(frigate_url)
    if first["ok"]:
        _detected["url"] = frigate_url
        _detected["ts"] = time.time()
        return {"ok": True, "error": None, "version": first["version"], "detected_url": None}

    # настроенный адрес не работает — ищем Frigate на типовых портах того же хоста
    from urllib.parse import urlsplit
    parts = urlsplit(frigate_url)
    host = parts.hostname or ""
    try:
        port = parts.port
    except ValueError:
        port = None
    candidates = [f"http://{host}:{p}" for p in _FRIGATE_PORT_CANDIDATES] if host else []
    cfg_norm = f"http://{host}:{port}" if (host and port) else frigate_url.rstrip("/")
    candidates = [c for c in candidates if c != cfg_norm]

    detected_url, version, html_cand = None, None, None
    for cand in candidates:
        res = _frigate_version(cand, timeout=3)
        if res["ok"]:
            detected_url, version = cand, res["version"]
            break
        if res["kind"] == "html":
            html_cand = f"{cand} ({res['error']})"

    if detected_url:
        _detected["url"] = detected_url
        _detected["ts"] = time.time()
        return {
            "ok": True,
            "error": (f"Frigate найден автоматически на {detected_url} (версия {version}); "
                      f"настроенный {frigate_url} не работает: {first['error']}. "
                      f"Впишите {detected_url} в поле Frigate URL в настройках аддона, "
                      "чтобы закрепить адрес."),
            "version": version,
            "detected_url": detected_url,
        }

    kind = first["kind"]
    if kind == "refused":
        hint = (" Соединение отклонено: по этому адресу на хосте никто не слушает. "
                "frigate_url должен указывать на ВНЕШНИЙ порт приложения Frigate — тот, "
                "что опубликован на самом хосте. В панели Frigate «Конфигурация "
                "прослушиваемых портов» это «Внешний порт» (по умолчанию 8971): тогда "
                "frigate_url = http://127.0.0.1:8971. «Внутренний порт» (5000) наружу не "
                "публикуется — 127.0.0.1:5000 из соседнего аддона недостижим. В "
                "оригинальном аддоне Frigate то же поле называется «Unauthenticated Web "
                "interface»: впишите туда 5000 и используйте http://127.0.0.1:5000.")
    elif kind == "timeout":
        hint = " Frigate не ответил за 4 с — проверьте, что приложение запущено."
    elif kind == "html":
        hint = (" По адресу отвечает HTML, а не Frigate API: либо это чужая служба "
                "(8556 у аддона Frigate — WebRTC/go2rtc, там нет API), либо включена "
                "аутентификация — для доступа Smart Access отключите её (в конфиге "
                "Frigate auth: enabled: false, затем Save & Restart).")
    else:
        hint = ""
    extra = ""
    if html_cand:
        extra += (f" Дополнительно: на {html_cand} похоже живёт Frigate с включённой "
                  "аутентификацией — отключите её (auth: enabled: false).")
    elif candidates:
        extra += f" Дополнительно проверен(ы) {', '.join(candidates)} — Frigate API там нет."
    return {"ok": False,
            "error": f"Нет связи с {frigate_url}/api/version: {first['error']}.{hint}{extra}",
            "version": None, "detected_url": None}


def frigate_effective_url(configured: str) -> str:
    """Адрес, по которому система реально ходит во Frigate.

    Настроенный, а если он не работает — автоматически найденный
    (действует _DETECTED_TTL секунд с момента последней успешной проверки).
    """
    det = _detected["url"]
    if det and det != configured and time.time() - _detected["ts"] < _DETECTED_TTL:
        return det
    return configured


def _snapshot_get(frigate_url: str, event_id: str, crop: bool) -> bytes | None:
    url = f"{frigate_url}/api/events/{event_id}/snapshot.jpg?quality=90" + ("&crop=1" if crop else "")
    try:
        resp = requests.get(url, timeout=10)
        if resp.ok and resp.content[:3] == b"\xff\xd8\xff":
            return resp.content
        LOG.debug("Frigate snapshot %s -> HTTP %s", event_id, resp.status_code)
    except requests.RequestException as exc:
        LOG.debug("Frigate snapshot %s недоступен: %s", event_id, exc)
    return None


def frigate_event_snapshot(frigate_url: str, event_id: str, crop: bool = True) -> bytes | None:
    """Лучший кадр события Frigate (?crop=1 — кроп по bbox объекта).

    Если настроенный frigate_url не отвечает, повторяем по автоматически
    найденному адресу — распознавание не ломается, пока в настройках
    остался неверный порт.
    """
    jpg = _snapshot_get(frigate_url, event_id, crop)
    if jpg is None:
        alt = frigate_effective_url(frigate_url)
        if alt != frigate_url:
            jpg = _snapshot_get(alt, event_id, crop)
    return jpg


def frigate_ping(frigate_url: str) -> bool:
    """Совместимость: просто bool из frigate_probe."""
    return frigate_probe(frigate_url)["ok"]
