"""Клиенты Home Assistant (через Supervisor-прокси) и Frigate.

Все вызовы устойчивы: таймауты + ретраи, любое падение не роняет систему —
функции возвращают None/False, вызывающий код решает сам.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from . import config as cfg

try:  # Frigate 0.16+ по умолчанию включает TLS на 8971 с self-signed сертификатом
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # noqa: BLE001
    pass

LOG = logging.getLogger("smart_access.ha")


def _sv_get(path: str, timeout: int = 8) -> requests.Response:
    """GET к Supervisor API (база определяется автоматически)."""
    return requests.get(f"{cfg.supervisor_base()}{path}", headers=_headers(), timeout=timeout)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {cfg.SUPERVISOR_TOKEN}"}


def call_ha_service(domain: str, service: str, data: dict[str, Any] | None = None,
                    retries: int = 3) -> bool:
    """Вызов сервиса HA из аддона: POST http://supervisor/core/api/services/..."""
    if not cfg.SUPERVISOR_TOKEN:
        LOG.error("SUPERVISOR_TOKEN отсутствует — вызов сервиса %s.%s невозможен", domain, service)
        return False
    url = f"{cfg.supervisor_base()}/core/api/services/{domain}/{service}"
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


def _short_exc(exc: BaseException, limit: int = 120) -> str:
    """Короткое человекочитаемое описание исключения для журнала."""
    s = str(exc).strip()
    return (s[:limit] + "…") if len(s) > limit else (s or exc.__class__.__name__)


def first_jpeg(buf: bytes) -> bytes | None:
    """Первый ПОЛНЫЙ JPEG (маркеры SOI…EOI) в буфере; None, если он ещё не целиком.

    В MJPEG-потоках энтропийные данные не содержат 0xFFD9 (байтовое
   Stuffing), поэтому поиск EOI после SOI даёт ровно границы кадра.
    """
    soi = buf.find(b"\xff\xd8\xff")
    if soi < 0:
        return None
    eoi = buf.find(b"\xff\xd9", soi + 3)
    if eoi < 0:
        return None
    return buf[soi:eoi + 2]


# Совместимость со старым внутренним именем
_first_jpeg = first_jpeg


# Троттлинг WARNING о сбоях камер. Раньше причины писались на DEBUG и в
# журнал не попадали: панель показывала «сбоев: 440», а ПОЧЕМУ — не говорил
# никто (реальный случай: camera.kalitka — универсальная камера с RTSP без
# still_image_url, у которой одиночные снимки не работают в принципе).
_WARN_TS: dict[str, float] = {}


def _warn_throttled(entity_id: str, msg: str, every: float = 60.0,
                    once: bool = False) -> None:
    key = f"{entity_id}|{'once' if once else 'each'}|{msg[:60]}"
    now = time.time()
    if once:
        if _WARN_TS.get(key):
            return
        _WARN_TS[key] = now
    else:
        if now - _WARN_TS.get(key, 0.0) < every:
            return
        _WARN_TS[key] = now
    LOG.warning("Камера %s: %s", entity_id, msg)


def get_camera_stream_frame(entity_id: str, timeout: float = 12.0,
                            max_bytes: int = 4_000_000) -> tuple[bytes | None, str]:
    """Один кадр из MJPEG-потока HA (/core/api/camera_proxy_stream/<entity>).

    Запасной путь для камер-«потоков». Универсальная камера (Generic Camera)
    с RTSP и без «URL кадра» (still_image_url) часто НЕ отдаёт одиночный
    снимок: HA ради одного кадра запускает извлечение кадра из потока
    (ffmpeg + HLS), это медленно и не укладывается в таймаут, а НЕПРЕРЫВНЫЙ
    MJPEG при этом работает — именно по нему играет карточка в HA. Читаем
    ПЕРВЫЙ полный JPEG (SOI…EOI) и закрываем соединение.
    Возвращает (jpeg|None, причина_отказа). Никогда не бросает.
    """
    if not cfg.SUPERVISOR_TOKEN:
        return None, "нет SUPERVISOR_TOKEN"
    deadline = time.monotonic() + timeout
    try:
        resp = requests.get(
            f"{cfg.supervisor_base()}/core/api/camera_proxy_stream/{entity_id}",
            headers=_headers(), timeout=(5, 5), stream=True,
        )
    except requests.RequestException as exc:
        return None, f"поток HA недоступен: {_short_exc(exc)}"
    if not resp.ok:
        ctype = resp.headers.get("content-type", "?")
        resp.close()
        return None, f"поток HA: HTTP {resp.status_code} ({ctype})"
    buf = b""
    try:
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                buf += chunk
                jpg = _first_jpeg(buf)
                if jpg is not None:
                    return jpg, ""
                if len(buf) > max_bytes:
                    return None, "поток HA не отдал JPEG (нет маркеров SOI/EOI)"
            if time.monotonic() > deadline:
                break
    except requests.RequestException as exc:
        return None, f"поток HA прервался: {_short_exc(exc)}"
    finally:
        resp.close()
    return None, "таймаут ожидания кадра из потока HA"


def get_camera_snapshot_ex(entity_id: str, timeout: int = 8) -> tuple[bytes | None, str]:
    """ТОЛЬКО одиночный снимок (camera_proxy): (jpeg|None, причина_отказа).

    v1.3.0: мост решает по нему, умеет ли камера дешёвые снимки (режим
    «proxy» — nginx still_image_url, ЕГДС, облачные snapshot-API), или ей
    нужен постоянный поток (режим «stream»). В отличие от
    get_camera_image_ex НЕ тянет поток запасным путём: для камеры-«потока»
    каждый такой «снимок» — целая сессия, и именно они топили камеры.
    Никогда не бросает.
    """
    if not cfg.SUPERVISOR_TOKEN:
        return None, "нет SUPERVISOR_TOKEN"
    try:
        resp = requests.get(
            f"{cfg.supervisor_base()}/core/api/camera_proxy/{entity_id}",
            headers=_headers(),
            timeout=timeout,
        )
        if resp.ok and resp.content[:3] == b"\xff\xd8\xff":
            return resp.content, ""
        err = (f"camera_proxy HTTP {resp.status_code} "
               f"({resp.headers.get('content-type', '?')})")
        if not resp.ok and resp.content:
            err += f": {resp.content[:120]!r}"
        elif resp.ok:
            err += ": тело не JPEG"
        return None, err
    except requests.RequestException as exc:
        return None, f"camera_proxy недоступна: {_short_exc(exc)}"


def open_camera_stream(entity_id: str):
    """Открыть ПОСТОЯННОЕ соединение к потоку HA (для моста, v1.3.0).

    Возвращает открытый streaming-ответ (multipart-MJPEG) или None.
    Один вызов = ОДНА сессия HA↔камера (как открытая карточка): кадры
    разбирает мост, соединение не рвётся между кадрами — вместо старого
    «открыл поток ради одного кадра и закрыл» на каждый опрос.
    Ошибки — в журнал с троттлингом 60 с.
    """
    if not cfg.SUPERVISOR_TOKEN:
        _warn_throttled(entity_id, "нет SUPERVISOR_TOKEN — поток HA недоступен")
        return None
    try:
        resp = requests.get(
            f"{cfg.supervisor_base()}/core/api/camera_proxy_stream/{entity_id}",
            headers=_headers(), stream=True, timeout=(5, 30),
        )
    except requests.RequestException as exc:
        _warn_throttled(entity_id, f"поток HA недоступен: {_short_exc(exc)}")
        return None
    if not resp.ok:
        ctype = resp.headers.get("content-type", "?")
        _warn_throttled(entity_id,
                        f"поток HA: HTTP {resp.status_code} ({ctype})")
        resp.close()
        return None
    return resp


def get_camera_image_ex(entity_id: str, timeout: int = 12) -> tuple[bytes | None, str, str]:
    """Кадр камеры HA с запасным путём: (jpeg, источник, причина_отказа).

    источник: 'proxy' — одиночный снимок camera_proxy (обычный путь);
              'stream' — первый кадр из MJPEG-потока HA (запасной путь);
              '' — кадра нет.
    Никогда не бросает исключений. Причины отказов пишутся в журнал
    (WARNING с троттлингом 60 с на камеру/причину).
    """
    if not cfg.SUPERVISOR_TOKEN:
        return None, "", "нет SUPERVISOR_TOKEN"
    try:
        resp = requests.get(
            f"{cfg.supervisor_base()}/core/api/camera_proxy/{entity_id}",
            headers=_headers(),
            timeout=timeout,
        )
        if resp.ok and resp.content[:3] == b"\xff\xd8\xff":
            return resp.content, "proxy", ""
        err = (f"camera_proxy HTTP {resp.status_code} "
               f"({resp.headers.get('content-type', '?')})")
        if not resp.ok and resp.content:
            err += f": {resp.content[:120]!r}"
        elif resp.ok:
            err += ": тело не JPEG"
    except requests.RequestException as exc:
        err = f"camera_proxy недоступна: {_short_exc(exc)}"
    _warn_throttled(entity_id, err)
    jpg, err2 = get_camera_stream_frame(entity_id)
    if jpg is not None:
        _warn_throttled(entity_id,
                        f"одиночный снимок не работает ({err}); кадр взят из потока HA",
                        once=True)
        return jpg, "stream", ""
    return None, "", f"{err}; {err2}"


def get_camera_image(entity_id: str) -> bytes | None:
    """JPEG-кадр с камеры HA (совместимость: только jpeg|None)."""
    return get_camera_image_ex(entity_id)[0]


def get_ha_state(entity_id: str) -> dict[str, Any] | None:
    if not cfg.SUPERVISOR_TOKEN:
        return None
    try:
        resp = requests.get(
            f"{cfg.supervisor_base()}/core/api/states/{entity_id}",
            headers=_headers(), timeout=8,
        )
        if resp.ok:
            return resp.json()
    except requests.RequestException:
        pass
    return None


# ---------------- Frigate ----------------

# Типовые порты Frigate API (док.: docs.frigate.video/configuration/advanced/system):
#   5000 — ВНУТРЕННИЙ порт, БЕЗ аутентификации (запросы с правами
#          администратора) — главный порт для аддон-интеграции;
#   8971 — ВНЕШНИЙ порт, ТРЕБУЕТ аутентификации (а у 0.16+ ещё и TLS).
# 5000 приоритетнее: из аддона в аддон он ходит напрямую по внутренней сети.
_FRIGATE_PORT_CANDIDATES = (5000, 8971)
_DETECTED_TTL = 600.0  # сколько секунд доверяем автоматически найденному адресу

_detected: dict[str, Any] = {"url": None, "ts": 0.0, "version": None}
_last_scan: dict[str, float] = {"ts": 0.0}
_SCAN_MIN_INTERVAL = 60.0  # между полными сканами внутренней сети

# /api/version отвечает версией в начале тела: '0.18.2', '0.18.2-1e2f3a4',
# '"0.16.0"' (JSON-строка) или '{"version": "..."}'. Якорь на НАЧАЛО строки
# защищает от ложных срабатываний на HTML, где внутри страницы любые числа.
_VERSION_RE = re.compile(r"^v?(\d+\.\d+(?:\.\d+)?(?:[-+.][0-9A-Za-z.\-]+)?)")


def _version_from_body(data: bytes) -> str | None:
    """Достаёт версию Frigate из тела /api/version: plain text ИЛИ JSON.

    Frigate 0.16+ отвечает PlainTextResponse (например '0.18.2') —
    док.: docs.frigate.video/integrations/api. Старые версии отдавали
    JSON-строку '"0.15.0"', некоторые прокси — '{"version": "..."}'.
    Всё остальное (HTML и т.п.) — не Frigate API.
    """
    text = data[:256].decode("utf-8", "replace").strip()
    if not text or len(text) > 64:
        return None  # версию такой длины не пишут — это страница/мусор
    if text[0] in '"{':
        try:
            parsed = json.loads(text)
        except ValueError:
            return None
        if isinstance(parsed, str):
            text = parsed.strip()
        elif isinstance(parsed, dict):
            text = str(parsed.get("version") or parsed.get("appVersion") or "").strip()
        else:
            return None
    m = _VERSION_RE.match(text)
    if not m:
        return None
    return m.group(1).rstrip(".,") or None


def _frigate_version(url: str, timeout: float = 4.0) -> dict[str, Any]:
    """GET {url}/api/version. Не бросает: {'ok','version','kind','error'}.

    kind: ok | refused | timeout | ssl | auth | html | http | unreachable.
    Тело /api/version — plain text ИЛИ JSON (см. _version_from_body):
    требовать JSON нельзя — на этом ломалась вся связь с Frigate 0.16+
    («HTTP 200, но тело не JSON (text/plain)» при живом API).
    HTTP 400/401/403 тоже информативны:
      * 400 — у внешнего порта 8971 (аутентификация/TLS) и у TLS-порта на
        plain-HTTP запрос; пробуем https (Frigate 0.16+ включает TLS).
      * 401/403 — порт требует аутентификацию (kind=auth), значит живой
        Frigate найден, но не на том порту.
    HTML-страница вместо версии — чужая служба или страница входа.
    """
    attempts = [("http", url)]
    if url.startswith("http://"):
        attempts.append(("https", "https://" + url[len("http://"):]))
    last: dict[str, Any] | None = None
    res400: dict[str, Any] | None = None
    for scheme, base in attempts:
        try:
            resp = requests.get(f"{base}/api/version", timeout=timeout, verify=False)
        except requests.exceptions.SSLError as exc:
            last = {"ok": False, "version": None, "kind": "ssl", "error": str(exc)}
            continue  # пробуем следующую схему
        except requests.RequestException as exc:
            s = str(exc).lower()
            if "refused" in s:
                kind = "refused"
            elif "timed out" in s or "timeout" in s:
                kind = "timeout"
            else:
                kind = "unreachable"
            last = {"ok": False, "version": None, "kind": kind, "error": str(exc)}
            # сброс соединения на HTTP-запрос — типичный признак TLS-порта:
            # пробуем следующую схему (https), иначе — соединения нет
            if ("aborted" in s or "reset" in s) and scheme == "http":
                continue
            break  # refused/timeout — https не спасёт
        ctype = resp.headers.get("content-type", "")
        if resp.status_code == 400:
            low = resp.content[:300].lower()
            if b"https" in low:  # «HTTPS server», «sent to HTTPS port» и т.п.
                last = {"ok": False, "version": None, "kind": "ssl",
                        "error": "порт отвечает только по HTTPS"}
            else:
                # обычный 400 (например внешний порт 8971 без аутентификации):
                # запоминаем и пробуем https — TLS-порт тоже отвечает 400
                res400 = {"ok": False, "version": None, "kind": "http",
                          "error": f"HTTP 400 ({ctype or 'без content-type'})"}
                last = res400
            continue  # 400 бывает и у TLS-порта — пробуем https
        if resp.ok:
            version = _version_from_body(resp.content)
            if version:
                return {"ok": True, "version": version, "kind": "ok", "error": None}
            last = {"ok": False, "version": None, "kind": "html",
                    "error": (f"HTTP 200, но по адресу не Frigate API "
                              f"({ctype or 'без content-type'})")}
            break
        if resp.status_code in (401, 403):
            last = {"ok": False, "version": None, "kind": "auth",
                    "error": f"HTTP {resp.status_code} — порт требует аутентификации"}
            break
        if "text/html" in ctype:
            last = {"ok": False, "version": None, "kind": "html",
                    "error": f"HTML-страница вместо API (HTTP {resp.status_code})"}
            break
        last = {"ok": False, "version": None, "kind": "http",
                "error": f"HTTP {resp.status_code} ({ctype or 'без content-type'})"}
        break
    if res400 is not None and (last is None or last.get("kind") == "ssl"):
        last = res400  # честнее показать HTTP 400, чем вторичную SSL-ошибку
    return last or {"ok": False, "version": None, "kind": "unreachable", "error": "no attempt"}


def _supervisor_frigate_addons() -> list[dict[str, str]] | None:
    """Аддоны, похожие на Frigate, через Supervisor API.

    None — Supervisor недоступен (нет токена/сети); [] — аддон не найден.
    """
    if not cfg.SUPERVISOR_TOKEN:
        return None
    try:
        resp = _sv_get("/addons", timeout=6)
        if not resp.ok:
            return None
        addons = resp.json().get("data", {}).get("addons", []) or []
    except Exception:  # noqa: BLE001
        return None
    out: list[dict[str, str]] = []
    for a in addons:
        hay = f"{a.get('slug', '')} {a.get('name', '')}".lower()
        if "frigate" not in hay:
            continue
        item = {"name": a.get("name") or a.get("slug") or "Frigate",
                "state": a.get("state") or "?",
                "version": a.get("version") or "",
                "slug": a.get("slug") or "",
                "hostname": ""}
        # hostname нужен для подсказки внутреннего адреса вида
        # http://ccab4aaf-frigate-fa:5000 — он резолвится из аддонов
        try:
            det = _sv_get(f"/addons/{item['slug']}", timeout=6)
            if det.ok:
                item["hostname"] = str(det.json().get("data", {}).get("hostname") or "")
        except Exception:  # noqa: BLE001
            pass
        out.append(item)
    return out


def frigate_addon_log_tail(limit: int = 1200) -> str | None:
    """Хвост журнала аддона Frigate через Supervisor API — для панели.

    Пользователю больше не нужно копировать логи вручную: если связи нет,
    /api/status отдаёт последние строки журнала аддона (там видна причина,
    например ошибка в config.yml). None — нет токена/аддона или Supervisor
    ответил отказом. Никогда не бросает.
    """
    if not cfg.SUPERVISOR_TOKEN:
        return None
    try:
        addons = _supervisor_frigate_addons()
    except Exception:  # noqa: BLE001
        return None
    if not addons:
        return None
    slug = addons[0].get("slug") or ""
    if not slug:
        return None
    try:
        resp = _sv_get(f"/addons/{slug}/logs", timeout=8)
        if not resp.ok:
            LOG.debug("Журнал аддона %s -> HTTP %s", slug, resp.status_code)
            return None
        return resp.text[-limit:]
    except Exception as exc:  # noqa: BLE001
        LOG.debug("Журнал аддона %s недоступен: %s", slug, exc)
        return None


def _bridge_subnets() -> list[Any]:
    """Подсети docker-мостов (в них живут аддоны: Frigate и др.).

    Источники: env SMART_ACCESS_BRIDGE_SUBNETS (например
    '172.30.32.0/23,10.55.0.0/24'), затем /proc/net/route (интерфейсы br-*).
    Дефолт — стандартная hassio-сеть 172.30.32.0/23.
    """
    nets: list[Any] = []
    env = os.environ.get("SMART_ACCESS_BRIDGE_SUBNETS", "").strip()
    if env:
        for part in env.split(","):
            try:
                n = ipaddress.ip_network(part.strip())
                if n.num_addresses <= 1024:
                    nets.append(n)
            except ValueError:
                pass
        if nets:
            return nets[:2]
    try:
        with open("/proc/net/route", encoding="ascii") as fh:
            for line in fh.readlines()[1:]:
                f = line.split()
                if not f or not f[0].startswith("br-"):
                    continue
                try:
                    dst = socket.inet_ntoa(struct.pack("<I", int(f[1], 16)))
                    mask = socket.inet_ntoa(struct.pack("<I", int(f[7], 16)))
                    net = ipaddress.ip_network(f"{dst}/{mask}", strict=False)
                    if net.num_addresses <= 1024 and net not in nets:
                        nets.append(net)
                except (ValueError, OSError, ipaddress.AddressValueError):
                    continue
    except OSError:
        pass
    default = ipaddress.ip_network("172.30.32.0/23")
    if not any(n.overlaps(default) for n in nets):
        nets.append(default)
    return nets[:2]


def _scan_bridge(subnets: list[Any] | None = None) -> str | None:
    """Ищет Frigate по портам 5000/8971 в подсетях docker-мостов.

    Из host-network аддона контейнерные IP моста доступны напрямую —
    публикация портов на хост для этого не нужна. Возвращает
    'http://IP:PORT' или None.
    """
    nets = subnets if subnets is not None else _bridge_subnets()
    targets: list[tuple[str, int]] = []
    for net in nets:
        try:
            for h in net.hosts():
                for p in _FRIGATE_PORT_CANDIDATES:
                    targets.append((str(h), p))
        except ValueError:
            continue
    if not targets or len(targets) > 2048:
        return None

    def _tcp_probe(t: tuple[str, int]) -> tuple[str, int] | None:
        ip, p = t
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.35)
        try:
            return t if s.connect_ex((ip, p)) == 0 else None
        finally:
            s.close()

    opens: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=64) as ex:
        for r in ex.map(_tcp_probe, targets):
            if r:
                opens.append(r)
    # порт 5000 (API без аутентификации) приоритетнее 8971 (UI/TLS)
    for ip, p in sorted(opens, key=lambda t: 0 if t[1] == 5000 else 1):
        res = _frigate_version(f"http://{ip}:{p}", timeout=2.5)
        if res["ok"]:
            LOG.info("Frigate найден во внутренней сети аддонов: http://%s:%s (v%s)",
                     ip, p, res["version"])
            return f"http://{ip}:{p}"
    return None


def _find_frigate() -> tuple[str | None, str | None]:
    """Рабочий адрес Frigate: кэш, иначе скан внутренней сети (с троттлингом)."""
    det = _detected["url"]
    if det and time.time() - _detected["ts"] < _DETECTED_TTL:
        return det, _detected.get("version")
    if time.time() - _last_scan["ts"] < _SCAN_MIN_INTERVAL:
        return None, None
    _last_scan["ts"] = time.time()
    url = _scan_bridge()
    if url:
        res = _frigate_version(url, timeout=2.5)
        _detected.update({"url": url, "ts": time.time(), "version": res.get("version")})
        return url, res.get("version")
    return None, None


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
        _detected.update({"url": frigate_url, "ts": time.time(), "version": first["version"]})
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
        _detected.update({"url": detected_url, "ts": time.time(), "version": version})
        return {
            "ok": True,
            "error": (f"Frigate найден автоматически на {detected_url} (версия {version}); "
                      f"настроенный {frigate_url} не работает: {first['error']}. "
                      f"Впишите {detected_url} в поле Frigate URL в настройках аддона, "
                      "чтобы закрепить адрес."),
            "version": version,
            "detected_url": detected_url,
        }

    # внешние порты не помогли — ищем Frigate по внутренней сети аддонов
    # (IP контейнера в docker-мосту, публикация портов не нужна)
    bridge_url, bridge_ver = _find_frigate()
    if bridge_url:
        return {
            "ok": True,
            "error": (f"Frigate найден по внутренней сети аддонов: {bridge_url} "
                      f"(версия {bridge_ver}); настроенный {frigate_url} не работает: "
                      f"{first['error']}. Система уже использует найденный адрес "
                      "(IP контейнера может смениться при перезапуске — Smart Access "
                      "найдёт его заново); впишите его в Frigate URL, чтобы закрепить."),
            "version": bridge_ver,
            "detected_url": bridge_url,
        }

    kind = first["kind"]
    if kind == "refused":
        hint = (" Соединение отклонено: по этому адресу никто не слушает. Для аддона "
                "правильный адрес — внутренний: http://<имя-аддона-Frigate>:5000 "
                "(порт 5000 работает без аутентификации и с правами администратора — "
                "docs.frigate.video/configuration/advanced/system). «Внешний порт» (8971) "
                "требует аутентификации и публикуется на хост не всегда.")
    elif kind == "timeout":
        hint = " Frigate не ответил за 4 с — проверьте, что приложение запущено."
    elif kind == "auth":
        hint = (" Frigate отвечает, но порт требует аутентификацию. Внутренний порт 5000 "
                "того же адреса работает без неё — используйте его (либо отключите "
                "аутентификацию: auth: enabled: false).")
    elif kind == "html":
        hint = (" По адресу отвечает HTML, а не Frigate API: либо это чужая служба "
                "(8556 у аддона Frigate — WebRTC/go2rtc, там нет API), либо страница "
                "входа. Правильный адрес для интеграции — порт 5000 (/api/version "
                "отвечает там текстом версии), а не веб-UI.")
    else:
        hint = ""
    if kind == "http" and str(first.get("error", "")).startswith("HTTP 400"):
        hint = (" HTTP 400 на этом порту — ожидаемо для ВНЕШНЕГО порта Frigate (8971): "
                "он требует аутентификации и/или TLS (Frigate 0.16+). Для интеграции "
                "используйте внутренний порт 5000 того же адреса — он без аутентификации "
                "(docs.frigate.video/configuration/advanced/system).")
    extra = ""
    if html_cand:
        extra += (f" Дополнительно: на {html_cand} — тоже не API Frigate (см. выше про "
                  "внешний порт и аутентификацию).")
    elif candidates:
        extra += f" Дополнительно проверен(ы) {', '.join(candidates)} — Frigate API там нет."

    # диагностика через Supervisor API: установлен ли аддон Frigate и запущен ли
    addons = _supervisor_frigate_addons()
    if addons:
        started = [a for a in addons if a["state"] == "started"]
        if started:
            names = "; ".join(f"«{a['name']}» v{a['version']}" for a in started)
            extra += (f" Аддон Frigate запущен ({names}), но его API не найден ни на "
                      "настроенном адресе, ни на внешних портах хоста, ни во внутренней "
                      "сети аддонов. Откройте вкладку «Журнал» аддона Frigate — вероятно, "
                      "он падает при старте (частая причина: ошибка в config.yml); также "
                      "проверьте, что настройки портов в разделе сети сохранены.")
            hosts = [a.get("hostname") for a in started if a.get("hostname")]
            if hosts:
                extra += (f" Попробуйте внутренний адрес аддона: http://{hosts[0]}:5000 "
                          "(имя контейнера во внутренней сети HA).")
        else:
            states = "; ".join(f"«{a['name']}» — {a['state']}" for a in addons)
            extra += (f" Через Supervisor API найден аддон Frigate: {states}. "
                      "Запустите его (Настройки → Аддоны → Frigate → Запустить), "
                      "затем обновите статус.")
    elif addons == []:
        extra += (" Через Supervisor API установленный аддон Frigate не найден. "
                  "Если Frigate работает на отдельном сервере/Docker — укажите его "
                  "адрес: http://<IP-сервера>:5000.")

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
        if alt == frigate_url:
            found, _ver = _find_frigate()  # троттлинг внутри
            alt = found or ""
        if alt and alt != frigate_url:
            jpg = _snapshot_get(alt, event_id, crop)
    return jpg


def frigate_ping(frigate_url: str) -> bool:
    """Совместимость: просто bool из frigate_probe."""
    return frigate_probe(frigate_url)["ok"]
