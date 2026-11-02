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
from typing import Any

import numpy as np

from . import config as cfg

LOG = logging.getLogger("smart_access.store")

MAX_EMBEDS_PER_SUBJECT = 240  # окно прототипов (аугментации по несколько за раз)


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

    def subjects(self, kind: str) -> dict[str, Any]:
        with self._lock:
            db = self.faces if kind == "face" else self.cars
            return {
                s: {"photos": len(r["embeds"]) // 8, "updated": r["updated"]}
                for s, r in db.items()
            }

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
            self.events = self.events[:300]
            self._save_events()

    def _load_events(self) -> list[dict[str, Any]]:
        return self._load("events.json") if os.path.exists(self._path("events.json")) else []

    def _save_events(self) -> None:
        self._save("events.json", self.events)

    # ---------- Фото ----------

    @staticmethod
    def _save_photo(kind: str, subject: str, photo: bytes) -> None:
        directory = os.path.join(cfg.MEDIA_DIR, kind, subject)
        os.makedirs(directory, exist_ok=True)
        fname = time.strftime("%Y%m%d-%H%M%S") + f"-{int((time.time()%1)*1000):03d}.jpg"
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
