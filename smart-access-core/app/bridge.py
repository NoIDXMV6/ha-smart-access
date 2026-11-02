"""Мост: ОДИН получатель на камеру -> MJPEG для Frigate и HA (v1.3.0).

ПЕРВОПРИЧИНА НЕСТАБИЛЬНОСТИ КАМЕР (до 1.3.0): мост опрашивал камеры
одиночными снимками (camera_proxy), а «снимок» для этих типов камер —
это ПОЛНОЦЕННАЯ СЕССИЯ ПОТОКА:
  * А.М.Видео (prud/govorovo): каждый снимок = новая WSS-сессия MPEG1-TS
    к видеопрокси (интеграция открывает поток, ждёт первый кадр, рвёт);
  * универсальная камера с RTSP без still_image_url (kalitka): каждый
    снимок = новый ffmpeg в HA, каждая попытка запасного пути — новая
    RTSP-сессия к самой камере.
Опрос 1.5 кадра/с = полторы НОВЫЕ сессии в секунду на камеру. Камеры и
облачные прокси на этом захлёбывались: таймауты, «сбоев: 139», fps: 0.
До интеграции Smart Access/Frigate у камеры был один клиент (карточка
HA) — поэтому всё работало стабильно.

АРХИТЕКТУРА v1.3.0 — У КАМЕРЫ ОДИН ПОЛУЧАТЕЛЬ, ЭТОТ МОСТ:
  * mode «rtsp»   — у камеры заполнен rtsp_url: мост сам читает RTSP
                    (TCP) и раздаёт кадры всем. Один клиент камеры;
  * mode «proxy»  — одиночные снимки БЫСТРЫЕ и не плодят сессии
                    (Hikvision still_image_url, ЕГДС-снимки, облачные
                    snapshot-API): опрос с частотой bridge_fps;
  * mode «stream» — ОДНО постоянное соединение к потоку HA
                    (/core/api/camera_proxy_stream, тот же путь, по
                    которому играет карточка): кадры разбираются
                    непрерывно, соединение не рвётся между кадрами.
Выбор режима автоматический и «липкий»: rtsp_url -> rtsp; иначе пробуем
снимок -> проки / поток; три подряд неудачи снимка -> перевод камеры на
«stream» навсегда (без реверса — снимки у таких камер не появляются).

АДРЕС ДЛЯ FRIGATE — http://172.30.32.1:8100/mjpeg/<name> (не 127.0.0.1):
Frigate — мостовой аддон (сеть hassio), хост для него — шлюз 172.30.32.1.
Сайт на :8100 поднимается в main.start_bridge_site.

Чтобы клиентов камеры не стало снова несколько, карточку в HA стоит
переключить на мост (универсальная камера: «URL потока» =
http://172.30.32.1:8100/mjpeg/<name>) — тогда HA тоже читает камеру
через мост. См. README «Архитектура камер: один получатель».

Отказоустойчивость: при недоступности источника поток отдаёт «заглушку»
и продолжает попытки с бэкоффом (1→2→5→10→30 с); Frigate переживает
паузы в потоке.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

import cv2
import numpy as np
from aiohttp import web

from . import ha_api

LOG = logging.getLogger("smart_access.bridge")

STUB = None  # lazy placeholder jpeg


def _placeholder_jpeg() -> bytes:
    global STUB
    if STUB is None:
        img = np.full((360, 640, 3), 24, np.uint8)
        cv2.putText(img, "camera offline", (170, 190), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (90, 90, 90), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        STUB = buf.tobytes() if ok else b""
    return STUB


class Bridge:
    """Один постоянный получатель на камеру + раздача MJPEG-потоков."""

    # период проверки «пора ли погасить получателей без подписчиков»
    WATCH_INTERVAL = 15.0
    # Пока у камеры нет подписчиков дольше GRACE_STOP секунд, её «получатель»
    # останавливается (не держим лишние сессии); первый подписчик запускает снова.
    GRACE_STOP = 90.0

    def __init__(self, fps: float = 1.5, stream_fps: float = 5.0) -> None:
        self.interval = 1.0 / max(0.2, min(5.0, fps))          # период «proxy»-опроса и раздачи
        self.stream_fps = max(1.0, min(15.0, float(stream_fps)))  # темп раздачи кадров RTSP-источника
        self.latest: dict[str, bytes] = {}       # name -> jpeg
        self.clients: dict[str, int] = {}        # name -> количество подписчиков
        self.entities: dict[str, str] = {}       # name -> ha entity ('' допустимо для rtsp)
        self.rtsp_urls: dict[str, str] = {}      # name -> rtsp_url ('' = нет)
        self._task: asyncio.Task | None = None   # совместимость (start/stop)
        self.failures: dict[str, int] = {}
        # телеметрия реального потока (видна в UI: «fps» рядом со «сбоев»)
        self.fps: dict[str, float] = {}          # name -> EMA кадров/с
        self.last_frame_ts: dict[str, float] = {}  # name -> время последнего кадра
        self.frames: dict[str, int] = {}         # name -> всего получено кадров
        # v1.3.0: РЕЖИМ получения («как.camera подключена к мосту») и ПОЧЕМУ кадра нет
        self.mode: dict[str, str] = {}           # '' | 'proxy' | 'stream' | 'rtsp'
        self.source: dict[str, str] = {}         # источник последнего кадра (= mode)
        self.last_error: dict[str, str] = {}     # последняя причина «сбоев»
        # Получатели (по одному на камеру)
        self._feeder: dict[str, asyncio.Task] = {}
        self._stop: dict[str, threading.Event] = {}
        self._last_client_ts: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Конфигурация камер
    # ------------------------------------------------------------------

    def set_cameras(self, cams: list[dict]) -> None:
        """Список из конфига: [{'name', 'ha_entity', 'rtsp_url'?}, ...]."""
        self.entities = {c["name"]: str(c.get("ha_entity") or "") for c in cams}
        self.rtsp_urls = {c["name"]: str(c.get("rtsp_url") or "").strip() for c in cams}
        for name in self.entities:
            self.latest.setdefault(name, _placeholder_jpeg())
            self.failures.setdefault(name, 0)
            self.fps.setdefault(name, 0.0)
            self.frames.setdefault(name, 0)
            self.mode.setdefault(name, "")
            self.source.setdefault(name, "")
            self.last_error.setdefault(name, "")
            self._last_client_ts.setdefault(name, 0.0)
        for gone in [n for n in list(self.fps) if n not in self.entities]:
            self._halt_feeder(gone)
            for d in (self.fps, self.last_frame_ts, self.frames, self.failures,
                      self.latest, self.mode, self.source, self.last_error,
                      self._last_client_ts, self.rtsp_urls):
                d.pop(gone, None)

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:8100/mjpeg/{name}"

    def frigate_url(self, name: str) -> str:
        """Адрес потока С ТОЧКИ ЗРЕНИЯ Frigate.

        Аддон Frigate работает в мостовой Docker-сети hassio (172.30.32.0/23),
        поэтому 127.0.0.1 для него — его собственный контейнер. Аддон Smart
        Access слушает на хосте (host_network), а хост из мостовой сети — это
        шлюз 172.30.32.1 (константа Supervisor: DOCKER_IPV4_NETWORK_MASK[1]).
        """
        return f"http://172.30.32.1:8100/mjpeg/{name}"

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Совместимость: получатели запускаются по подписчикам (_kick)."""
        self._task = asyncio.create_task(self._idle_watch())

    async def stop(self) -> None:
        for name in list(self._stop):
            self._halt_feeder(name)
        if self._task:
            self._task.cancel()

    async def _idle_watch(self) -> None:
        """Гасит получатели камер без подписчиков (GRACE_STOP) — не держим
        сессии, когда Frigate выключен и панель закрыта."""
        while True:
            await asyncio.sleep(self.WATCH_INTERVAL)
            now = time.time()
            for name in list(self._feeder):
                task = self._feeder.get(name)
                if task and not task.done() and self.clients.get(name, 0) <= 0:
                    if now - self._last_client_ts.get(name, 0.0) > self.GRACE_STOP:
                        LOG.info("Мост: подписчиков нет %.0f с — останавливаю получателя «%s»",
                                 now - self._last_client_ts.get(name, 0.0), name)
                        self._halt_feeder(name)

    # ------------------------------------------------------------------
    # Приём кадров (общая телеметрия, потокобезопасно в рамках GIL)
    # ------------------------------------------------------------------

    def _accept(self, name: str, jpeg: bytes, source: str) -> None:
        self.latest[name] = jpeg
        self.failures[name] = 0
        self.mode[name] = source
        self.source[name] = source
        self.last_error[name] = ""
        now = time.time()
        prev = self.last_frame_ts.get(name)
        self.last_frame_ts[name] = now
        self.frames[name] = self.frames.get(name, 0) + 1
        if prev:
            dt = max(0.05, now - prev)
            inst = 1.0 / dt
            # EMA: сглаживает скачки, но быстро видит «0 fps» при сбое
            self.fps[name] = round(0.7 * self.fps.get(name, inst) + 0.3 * inst, 2)

    def _fail(self, name: str, err: str) -> int:
        self.failures[name] = self.failures.get(name, 0) + 1
        self.last_error[name] = err or "нет кадра"
        n = self.failures[name]
        if n in (1, 30, 300):
            LOG.warning("Мост: камера %s (%s) не отвечает — %s (%s подряд)",
                        name, self.entities.get(name) or self.rtsp_urls.get(name) or "?",
                        self.last_error[name], n)
        return n

    def snapshot_for_entity(self, entity: str) -> bytes | None:
        """Последний кадр моста для HA-сущности (для «Обучить»/identify без
        новой сессии к камере). None, если кадра ещё нет (только заглушка)."""
        for name, ent in self.entities.items():
            if ent == entity and self.frames.get(name, 0) > 0:
                return self.latest.get(name)
        return None

    # ------------------------------------------------------------------
    # Получатели (feeders) — один на камеру
    # ------------------------------------------------------------------

    def _kick(self, name: str) -> None:
        """Первый подписчик: запустить получателя камеры."""
        self._last_client_ts[name] = time.time()
        if name in self._feeder and not self._feeder[name].done():
            return
        stop = threading.Event()
        self._stop[name] = stop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        cam = {"name": name,
               "ha_entity": self.entities.get(name, ""),
               "rtsp_url": self.rtsp_urls.get(name, "")}
        self._feeder[name] = loop.create_task(self._feeder_loop(cam, stop))
        LOG.info("Мост: запущен получатель «%s» (сущность %s, rtsp %s)",
                 name, cam["ha_entity"] or "—", "есть" if cam["rtsp_url"] else "нет")

    def _halt_feeder(self, name: str) -> None:
        ev = self._stop.get(name)
        if ev:
            ev.set()
        task = self._feeder.get(name)
        if task and not task.done():
            task.cancel()
        self._feeder.pop(name, None)
        self._stop.pop(name, None)

    async def _feeder_loop(self, cam: dict, stop: threading.Event) -> None:
        """Выбирает режим и держит его. Один вызов = вся жизнь получателя."""
        name = cam["name"]
        loop = asyncio.get_running_loop()
        try:
            rtsp = (cam.get("rtsp_url") or "").strip()
            if rtsp:
                self.mode[name] = "rtsp"
                LOG.info("Мост: «%s» — режим rtsp (мост сам читает камеру, один клиент)", name)
                await loop.run_in_executor(None, self._rtsp_thread, name, rtsp, stop)
                return
            entity = cam.get("ha_entity") or ""
            if not entity:
                self._fail(name, "не заданы ни rtsp_url, ни ha_entity")
                return
            # Проба одиночного снимка: быстрый снимок (nginx/ЕГДС/облако) —
            # режим proxy; иначе — постоянный поток HA.
            jpeg, err = await loop.run_in_executor(
                None, ha_api.get_camera_snapshot_ex, entity, 8)
            if jpeg:
                self.mode[name] = "proxy"
                self._accept(name, jpeg, "proxy")
                LOG.info("Мост: «%s» — режим proxy (быстрые одиночные снимки)", name)
                await self._proxy_loop(name, entity, stop)
            else:
                self.mode[name] = "stream"
                LOG.info("Мост: «%s» — режим stream (одно постоянное соединение к потоку HA; "
                         "одиночный снимок не работает: %s)", name, err)
                await loop.run_in_executor(None, self._stream_thread, name, entity, stop)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Мост: получатель «%s» упал: %s", name, exc)
        finally:
            stop.set()

    async def _proxy_loop(self, name: str, entity: str, stop: threading.Event) -> None:
        """Опрос быстрых одиночных снимков (без порождения сессий)."""
        loop = asyncio.get_running_loop()
        fails = 0
        while not stop.is_set():
            if not self._has_clients(name):
                return  # подписчиков давно нет — получателя погасит _idle_watch
            jpeg, err = await loop.run_in_executor(
                None, ha_api.get_camera_snapshot_ex, entity, 8)
            if jpeg:
                fails = 0
                self._accept(name, jpeg, "proxy")
            else:
                fails += 1
                self._fail(name, err or "снимок не удался")
                if fails >= 3:
                    # Снимки перестали работать (или не работали с самого
                    # начала) — постоянный поток вместо долбёжки снимками.
                    LOG.warning("Мост: «%s» — 3 подряд неудачи снимка, перевожу на "
                                "постоянный поток HA (одна сессия вместо опроса)", name)
                    self.mode[name] = "stream"
                    await loop.run_in_executor(None, self._stream_thread, name, entity, stop)
                    return
            if stop.is_set():
                return
            await asyncio.sleep(self.interval)  # НЕ блокируем event loop

    def _has_clients(self, name: str) -> bool:
        """Есть ли живой интерес к камере (для гашения получателей)."""
        if self.clients.get(name, 0) > 0:
            self._last_client_ts[name] = time.time()
            return True
        return time.time() - self._last_client_ts.get(name, 0.0) <= self.GRACE_STOP

    def _stream_thread(self, name: str, entity: str, stop: threading.Event) -> None:
        """ПОСТОЯННОЕ соединение к потоку HA (одна сессия, как открытая карточка).

        Работает в executor-потоке: читает multipart-MJPEG, вырезает полные
        JPEG (SOI…EOI), обрыв/таймаут -> переподключение с бэкоффом.
        """
        backoff = 1.0
        while not stop.is_set():
            if not self._has_clients(name):
                if stop.wait(5.0):
                    return
                continue
            try:
                got_frame, err = self._stream_once(name, entity, stop)
                if not got_frame and err:
                    self._fail(name, err)
            except Exception as exc:  # noqa: BLE001
                self._fail(name, f"поток HA: {ha_api._short_exc(exc)}")
                got_frame = False
            if stop.is_set():
                return
            if got_frame:
                backoff = 1.0
                continue
            # кадров не было — пауза перед новым соединением
            if stop.wait(backoff):
                return
            backoff = min(30.0, backoff * 2)

    def _stream_once(self, name: str, entity: str,
                     stop: threading.Event) -> tuple[bool, str]:
        """Одна сессия потока HA: (был_кадр, причина_отказа).

        Читает multipart-MJPEG, вырезает полные JPEG (SOI…EOI), обновляет
        latest. 30 с без данных — обрываем сессию и переподключаемся.
        """
        resp = ha_api.open_camera_stream(entity)
        if resp is None:
            return False, "поток HA недоступен"
        try:
            buf = b""
            last_frame = time.time()
            got = False
            for chunk in resp.iter_content(chunk_size=65536):
                if stop.is_set() or not self._has_clients(name):
                    break
                if chunk:
                    buf += chunk
                    jpg = ha_api.first_jpeg(buf)
                    if jpg is not None:
                        eoi = buf.find(b"\xff\xd9", buf.find(b"\xff\xd8\xff") + 3)
                        buf = buf[eoi + 2:]
                        self._accept(name, jpg, "stream")
                        got = True
                        last_frame = time.time()
                    elif len(buf) > 8_000_000:
                        buf = b""
                if time.time() - last_frame > 30.0:
                    return got, "поток HA молчит 30 с"
            return got, ("" if got else "поток не отдал кадра")
        finally:
            resp.close()

    def _rtsp_thread(self, name: str, rtsp_url: str, stop: threading.Event) -> None:
        """Мост сам читает RTSP камеры (TCP) — ЕДИНСТВЕННЫЙ клиент камеры."""
        # TCP-транспорт: UDP у дешёвых камер теряет кадры (проверено на kalitka)
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        backoff = 1.0
        while not stop.is_set():
            if not self._has_clients(name):
                if stop.wait(5.0):
                    return
                continue
            cap = None
            got_frame = False
            try:
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    self._fail(name, "RTSP: соединение не открылось")
                else:
                    last_frame = time.time()
                    while not stop.is_set() and self._has_clients(name):
                        ok, frame = cap.read()
                        if not ok:
                            self._fail(name, "RTSP: кадр не прочитан (поток оборвался)")
                            break
                        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok2:
                            self._accept(name, buf.tobytes(), "rtsp")
                            got_frame = True
                            last_frame = time.time()
                        if time.time() - last_frame > 30.0:
                            self._fail(name, "RTSP: кадров нет 30 с — переподключаюсь")
                            break
            except Exception as exc:  # noqa: BLE001
                self._fail(name, f"RTSP: {ha_api._short_exc(exc)}")
            finally:
                if cap is not None:
                    cap.release()
            if stop.is_set():
                return
            if got_frame:
                backoff = 1.0
            if stop.wait(backoff):
                return
            backoff = min(30.0, backoff * 2)

    # ------------------------------------------------------------------
    # Подсказки для панели
    # ------------------------------------------------------------------

    def hint(self, name: str) -> str:
        """Подсказка для панели, когда мост не может нормально получить кадры."""
        mode = self.mode.get(name, "")
        failures = self.failures.get(name, 0)
        err = self.last_error.get(name, "")
        if failures == 0 and self.fps.get(name, 0) > 0:
            if mode == "stream":
                return ("Камера без быстрых одиночных снимков: мост держит ОДНО "
                        "постоянное соединение к потоку HA — это нормальный режим "
                        "(клиентов камеры: 1). Для независимости от HA впишите "
                        "камере rtsp_url.")
            if mode == "rtsp":
                return ("Мост сам читает RTSP камеры — клиентов камеры: 1. "
                        "Карточку в HA переключите на мост (mjpeg_url "
                        "http://172.30.32.1:8100/mjpeg/" + name + "), чтобы HA не "
                        "открывал вторую сессию.")
            return ""
        if mode == "rtsp" and failures:
            return (f"RTSP недоступен ({err}). Проверьте rtsp_url и доступность "
                    "камеры; мост повторяет попытки с бэкоффом.")
        if failures >= 5:
            return (f"Постоянный поток HA не отвечает ({err}). Проверьте, что "
                    "карточка этой камеры в HA играет. Локальная камера → впишите "
                    "rtsp_url (мост будет читать её напрямую, один клиент); "
                    "универсальной камере можно добавить «URL кадра» (still_image_url).")
        return ""

    # ------------------------------------------------------------------
    # aiohttp
    # ------------------------------------------------------------------

    async def handle_mjpeg(self, request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        if name not in self.entities:
            # частая причина «404» — имя в конфиге Frigate не совпадает с name
            # камеры в Smart Access (например, поток govorovo при камере a_m_video)
            LOG.warning("Мост: запрос НЕИЗВЕСТНОЙ камеры «%s» (известны: %s)",
                        name, ", ".join(sorted(self.entities)) or "нет")
            return web.json_response({"error": "unknown camera"}, status=404)
        first_client = self.clients.get(name, 0) == 0
        self.clients[name] = self.clients.get(name, 0) + 1
        if first_client:
            # первый подписчик: запускаем получателя камеры (одна сессия)
            self._kick(name)
        started = time.time()
        LOG.info("Мост: клиент %s подключился к «%s» (%s), подписчиков: %d",
                 request.remote or "?", name,
                 self.entities.get(name) or self.rtsp_urls.get(name) or "?",
                 self.clients[name])
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=smartaccess",
                "Cache-Control": "no-cache, no-store",
                "Connection": "close",
            },
        )
        await resp.prepare(request)
        try:
            while True:
                self._last_client_ts[name] = time.time()
                jpeg = self.latest.get(name) or _placeholder_jpeg()
                await resp.write(
                    b"--smartaccess\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                )
                # RTSP-источник раздаётся в темпе камеры (stream_fps), остальные
                # — в темпе bridge_fps (кадры всё равно приходят так)
                period = self.interval
                if self.mode.get(name) == "rtsp":
                    period = 1.0 / self.stream_fps
                await asyncio.sleep(period)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.clients[name] = max(0, self.clients.get(name, 1) - 1)
            self._last_client_ts[name] = time.time()
            LOG.info("Мост: клиент %s отключился от «%s» (%.1f с, подписчиков: %d)",
                     request.remote or "?", name, time.time() - started,
                     self.clients[name])
        return resp

    async def handle_latest(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        jpeg = self.latest.get(name) or _placeholder_jpeg()
        return web.Response(body=jpeg, content_type="image/jpeg")

    def register(self, app: web.Application) -> None:
        app.router.add_get("/mjpeg/{name}", self.handle_mjpeg)
        app.router.add_get("/latest/{name}", self.handle_latest)
