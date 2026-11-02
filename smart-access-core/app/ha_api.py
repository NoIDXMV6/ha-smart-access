"""Клиенты Home Assistant (через Supervisor-прокси) и Frigate.

Все вызовы устойчивы: таймауты + ретраи, любое падение не роняет систему —
функции возвращают None/False, вызывающий код решает сам.
"""
from __future__ import annotations

import ipaddress
import logging
import os
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


def get_camera_image(entity_id: str) -> bytes | None:
    """JPEG-кадр с камеры HA (camera_proxy — работает и для облачных камер)."""
    if not cfg.SUPERVISOR_TOKEN:
        return None
    try:
        resp = requests.get(
            f"{cfg.supervisor_base()}/core/api/camera_proxy/{entity_id}",
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
            f"{cfg.supervisor_base()}/core/api/states/{entity_id}",
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

_detected: dict[str, Any] = {"url": None, "ts": 0.0, "version": None}
_last_scan: dict[str, float] = {"ts": 0.0}
_SCAN_MIN_INTERVAL = 60.0  # между полными сканами внутренней сети


def _frigate_version(url: str, timeout: float = 4.0) -> dict[str, Any]:
    """GET {url}/api/version. Не бросает: {'ok','version','kind','error'}.

    kind: ok | refused | timeout | html | http | unreachable.
    HTML-страница вместо JSON — это НЕ Frigate API (чужая служба на порту
    или включённая аутентификация Frigate). Если по http получена SSL-ошибка
    или ответ «это HTTPS-сервер» — повторяем по https (Frigate 0.16+ включает
    TLS на 8971 по умолчанию, сертификат self-signed).
    """
    attempts = [("http", url)]
    if url.startswith("http://"):
        attempts.append(("https", "https://" + url[len("http://"):]))
    last: dict[str, Any] | None = None
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
        if resp.status_code == 400 and b"HTTPS server" in resp.content[:200]:
            last = {"ok": False, "version": None, "kind": "ssl",
                    "error": "порт отвечает только по HTTPS"}
            continue
        ctype = resp.headers.get("content-type", "")
        if resp.ok:
            try:
                version = str(resp.json()).strip().strip('"')
            except ValueError:
                last = {"ok": False, "version": None, "kind": "html",
                        "error": f"HTTP 200, но тело не JSON ({ctype or 'без content-type'})"}
                break
            return {"ok": True, "version": version, "kind": "ok", "error": None}
        if "text/html" in ctype:
            last = {"ok": False, "version": None, "kind": "html",
                    "error": f"HTML-страница вместо API (HTTP {resp.status_code})"}
            break
        last = {"ok": False, "version": None, "kind": "http",
                "error": f"HTTP {resp.status_code} ({ctype or 'без content-type'})"}
        break
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
        if "frigate" in hay:
            out.append({"name": a.get("name") or a.get("slug") or "Frigate",
                        "state": a.get("state") or "?",
                        "version": a.get("version") or ""})
    return out


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
