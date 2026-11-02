"""Постоянное хранилище базы «своих» (машины и лица) + журнал событий.

Формат БД (JSON, атомарная запись):
  /data/db/cars.json   -> {"<subject>": {"embeds": [[512 float]...], "updated": ts}}
  /data/db/faces.json  -> {"<subject>": {"embeds": [[128 float]...], "updated": ts}}

Оригиналы фото хранятся в /media/smart_access/{cars|faces}/<subject>/*.jpg —
видны в Media-браузере HA, переживают переустановку аддона.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from typing import Any

import numpy as np

from . import config as cfg

LOG = logging.getLogger("smart_access.store")

MAX_EMBEDS_PER_SUBJECT = 240  # окно прототипов (аугментации по несколько за раз)
MAX_EVENTS = 2000             # журнал: хвост событий (фильтруется в UI, архивируется вручную)
IMG_EXT = (".jpg", ".jpeg", ".png")


def safe_subject(subject: str) -> str:
    """Имя subject как ИМЯ каталога: без разделителей пути и ссылок вверх."""
    s = str(subject or "").strip()
    if not s or s in (".", "..") or "/" in s or "\\" in s or "\x00" in s:
        raise ValueError(f"недопустимое имя: {subject!r}")
    return s


class Store:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.db_dir = os.path.join(cfg.DATA_DIR, "db")
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.MEDIA_DIR, "cars"), exist_ok=True)
        os.makedirs(os.path.join(cfg.MEDIA_DIR, "faces"), exist_ok=True)
        os.makedirs(os.path.join(cfg.MEDIA_DIR, "unknown"), exist_ok=True)
        self.cars: dict[str, dict[str, Any]] = self._load("cars.json")
        self.faces: dict[str, dict[str, Any]] = self._load("faces.json")
        self.events: list[dict[str, Any]] = self._load_events()

    # ---------- БД ----------

    def _path(self, name: str) -> str:
        return os.path.join(self.db_dir, name)

    def _load(self, name: str) -> dict[str, dict[str, Any]]:
        try:
            with open(self._path(name), "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return {}

    def _save(self, name: str, data: dict) -> None:
        tmp = self._path(name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, self._path(name))

    @staticmethod
    def _add_subject(db: dict[str, dict[str, Any]], subject: str, embeds: list[list[float]]) -> None:
        rec = db.setdefault(subject, {"embeds": [], "updated": 0})
        rec["embeds"] = (rec["embeds"] + embeds)[-MAX_EMBEDS_PER_SUBJECT:]
        rec["updated"] = time.time()

    def add_car(self, subject: str, embeds: list[list[float]], photo: bytes) -> None:
        with self._lock:
            self._add_subject(self.cars, subject, embeds)
            self._save("cars.json", self.cars)
        self._save_photo("cars", subject, photo)

    def add_face(self, subject: str, embeds: list[list[float]], photo: bytes) -> None:
        with self._lock:
            self._add_subject(self.faces, subject, embeds)
            self._save("faces.json", self.faces)
        self._save_photo("faces", subject, photo)

    def remove_subject(self, kind: str, subject: str) -> bool:
        with self._lock:
            db = self.faces if kind == "face" else self.cars
            if subject not in db:
                return False
            db.pop(subject, None)
            self._save("faces.json" if kind == "face" else "cars.json", db)
        media_dir = os.path.join(cfg.MEDIA_DIR, "faces" if kind == "face" else "cars", subject)
        shutil.rmtree(media_dir, ignore_errors=True)
        return True

    def replace_subject(self, kind: str, subject: str, embeds: list[list[float]]) -> None:
        """Замена всех прототипов subject (переобучение по фото из /media)."""
        with self._lock:
            db = self.faces if kind == "face" else self.cars
            rec = db.setdefault(subject, {"embeds": [], "updated": 0})
            rec["embeds"] = [list(map(float, v)) for v in embeds][-MAX_EMBEDS_PER_SUBJECT:]
            rec["updated"] = time.time()
            self._save("faces.json" if kind == "face" else "cars.json", db)

    def subjects(self, kind: str) -> dict[str, Any]:
        """Список subject с РЕАЛЬНЫМ числом фото (файлы в /media) и миниатюрой."""
        base = os.path.join(cfg.MEDIA_DIR, "faces" if kind == "face" else "cars")
        with self._lock:
            db = self.faces if kind == "face" else self.cars
            out: dict[str, Any] = {}
            for s, r in db.items():
                d = os.path.join(base, s)
                photos, thumb = 0, None
                try:
                    files = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)]
                    photos = len(files)
                    if files:
                        thumb = max(files, key=lambda f: os.path.getmtime(os.path.join(d, f)))
                except OSError:
                    pass
                out[s] = {
                    "photos": photos,
                    "embeds": len(r.get("embeds", [])),
                    "updated": r.get("updated", 0),
                    "thumb": thumb,
                }
            return out

    # ---------- Матчинг ----------

    @staticmethod
    def match(db: dict[str, dict[str, Any]], vec: np.ndarray) -> tuple[str | None, float]:
        """max косинусное сходство по всем прототипам всех subject."""
        best, best_score = None, -1.0
        v = vec / (np.linalg.norm(vec) + 1e-9)
        for subject, rec in db.items():
            arr = np.asarray(rec["embeds"], dtype=np.float32)
            if arr.size == 0:
                continue
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            sims = (arr / norms) @ v
            m = float(np.max(sims))
            if m > best_score:
                best, best_score = subject, m
        return best, best_score

    # ---------- Журнал ----------

    def log_event(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.events.insert(0, {**payload, "ts": time.time()})
            self.events = self.events[:MAX_EVENTS]
            self._save_events()

    def query_events(self, filters: dict[str, Any] | None = None,
                     limit: int = 200) -> list[dict[str, Any]]:
        """Журнал с фильтрами: kind, action, camera, subject, score_min/max,
        from/to (unix ts), limit. События идут от свежих к старым."""
        f = filters or {}
        out: list[dict[str, Any]] = []
        with self._lock:
            for e in self.events:
                if f.get("kind") and e.get("kind") != f["kind"]:
                    continue
                if f.get("action") and e.get("action") != f["action"]:
                    continue
                if f.get("camera") and e.get("camera") != f["camera"]:
                    continue
                if f.get("subject") and (e.get("subject") or "") != f["subject"]:
                    continue
                score = float(e.get("score") or 0)
                if f.get("score_min") is not None and score < float(f["score_min"]):
                    continue
                if f.get("score_max") is not None and score > float(f["score_max"]):
                    continue
                ts = float(e.get("ts") or 0)
                if f.get("from") and ts < float(f["from"]):
                    continue
                if f.get("to") and ts > float(f["to"]):
                    continue
                out.append(e)
                if len(out) >= max(1, int(limit)):
                    break
        return out

    def archive_events(self) -> dict[str, Any]:
        """Архивирует журнал в /media/smart_access/archive/journal-*.json и очищает его."""
        with self._lock:
            if not self.events:
                return {"ok": False, "error": "журнал пуст"}
            adir = os.path.join(cfg.MEDIA_DIR, "archive")
            os.makedirs(adir, exist_ok=True)
            name = "journal-" + time.strftime("%Y%m%d-%H%M%S") + ".json"
            tmp = os.path.join(adir, name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.events, fh, ensure_ascii=False)
            os.replace(tmp, os.path.join(adir, name))
            count = len(self.events)
            self.events = []
            self._save_events()
            LOG.info("Журнал архивирован: %s (%s событий)", name, count)
            return {"ok": True, "name": name, "count": count}

    def clear_events(self) -> int:
        with self._lock:
            count = len(self.events)
            self.events = []
            self._save_events()
            return count

    def list_archives(self) -> list[dict[str, Any]]:
        adir = os.path.join(cfg.MEDIA_DIR, "archive")
        try:
            return [
                {"file": f, "ts": os.path.getmtime(os.path.join(adir, f)),
                 "size": os.path.getsize(os.path.join(adir, f))}
                for f in sorted(os.listdir(adir), reverse=True)
                if f.startswith("journal-") and f.endswith(".json")
            ]
        except OSError:
            return []

    @staticmethod
    def archive_path(name: str) -> str | None:
        """Полный путь архива по имени (без выхода за пределы каталога)."""
        if not name.startswith("journal-") or not name.endswith(".json"):
            return None
        if "/" in name or "\\" in name or ".." in name:
            return None
        return os.path.join(cfg.MEDIA_DIR, "archive", name)

    def _load_events(self) -> list[dict[str, Any]]:
        return self._load("events.json") if os.path.exists(self._path("events.json")) else []

    def _save_events(self) -> None:
        self._save("events.json", self.events)

    # ---------- Фотографии (библиотека распознавания) ----------

    @staticmethod
    def _media_base(kind: str) -> str:
        d = {"car": "cars", "face": "faces"}.get(kind)
        if not d:
            raise ValueError("kind must be car|face")
        return os.path.join(cfg.MEDIA_DIR, d)

    def photos_dir(self, kind: str, subject: str) -> str:
        return os.path.join(self._media_base(kind), safe_subject(subject))

    def list_photos(self, kind: str, subject: str) -> list[dict[str, Any]]:
        """Список фото subject: {'file', 'ts', 'size'} от старых к свежим."""
        d = self.photos_dir(kind, subject)
        out: list[dict[str, Any]] = []
        try:
            for f in os.listdir(d):
                if not f.lower().endswith(IMG_EXT):
                    continue
                p = os.path.join(d, f)
                out.append({"file": f, "ts": os.path.getmtime(p), "size": os.path.getsize(p)})
        except OSError:
            return []
        return sorted(out, key=lambda r: (r["ts"], r["file"]))

    def all_photos(self, kind: str) -> dict[str, list[str]]:
        """{subject: [полные пути к фото]} по всем subject (для переобучения)."""
        base = self._media_base(kind)
        out: dict[str, list[str]] = {}
        try:
            subs = sorted(os.listdir(base))
        except OSError:
            return out
        for s in subs:
            d = os.path.join(base, s)
            if not os.path.isdir(d):
                continue
            files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT)]
            out[s] = sorted(files, key=lambda p: os.path.getmtime(p))
        return out

    def delete_photos(self, kind: str, subject: str, files: list[str]) -> int:
        """Удаляет перечисленные фото (только имена файлов, без путей)."""
        d = self.photos_dir(kind, subject)
        deleted = 0
        for f in files:
            fname = os.path.basename(str(f))
            if not fname.lower().endswith(IMG_EXT):
                continue
            try:
                os.remove(os.path.join(d, fname))
                deleted += 1
            except OSError:
                pass
        return deleted

    def photo_path(self, kind: str, subject: str, fname: str) -> str | None:
        """Путь к фото для раздачи по HTTP (защита от ../)."""
        try:
            d = self.photos_dir(kind, subject)
        except ValueError:
            return None
        fname = os.path.basename(str(fname))
        if not fname.lower().endswith(IMG_EXT):
            return None
        p = os.path.join(d, fname)
        if not os.path.isfile(p):
            return None
        return p

    # ---------- Фото ----------

    @staticmethod
    def _save_photo(kind: str, subject: str, photo: bytes) -> None:
        directory = os.path.join(cfg.MEDIA_DIR, kind, subject)
        os.makedirs(directory, exist_ok=True)
        # uuid в имени: несколько фото за одну миллисекунду (мультизагрузка
        # в панели) больше не перезаписывают друг друга
        fname = time.strftime("%Y%m%d-%H%M%S") + f"-{int((time.time()%1)*1000):03d}" \
            + f"-{uuid.uuid4().hex[:8]}.jpg"
        try:
            with open(os.path.join(directory, fname), "wb") as fh:
                fh.write(photo)
            # ограничиваем архив фото 60 снимками на subject
            files = sorted(os.listdir(directory))
            for old in files[:-60]:
                os.remove(os.path.join(directory, old))
        except OSError as exc:
            LOG.warning("Не сохранил фото (%s/%s): %s", kind, subject, exc)

    def save_unknown(self, kind: str, photo: bytes) -> None:
        directory = os.path.join(cfg.MEDIA_DIR, "unknown")
        fname = f"{kind}-" + time.strftime("%Y%m%d-%H%M%S") + ".jpg"
        try:
            with open(os.path.join(directory, fname), "wb") as fh:
                fh.write(photo)
        except OSError:
            pass

    def cleanup_unknown(self, keep_days: int) -> None:
        cutoff = time.time() - keep_days * 86400
        directory = os.path.join(cfg.MEDIA_DIR, "unknown")
        try:
            for f in os.listdir(directory):
                if os.path.getmtime(os.path.join(directory, f)) < cutoff:
                    os.remove(os.path.join(directory, f))
        except OSError:
            pass
