"""Распознавание: CLIP ViT-B/32 (int8, CPU) для машин, YuNet+SFace для лиц.

Принцип устойчивости (все модели локальные, CPU, работают без интернета):
  * Машина  -> кроп авто -> аугментации -> CLIP-эмбеддинг 512-d -> max cosine
              к прототипам «своих» машин. CLIP пре-трейнен на ~400 млн
              изображений — устойчив к сезонам, погоде, освещению.
  * Лицо    -> кроп персоны -> YuNet (детекция+5 точек) -> alignCrop 112x112
              -> SFace-эмбеддинг 128-d -> max cosine (порог OpenCV 0.363).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from . import config as cfg

LOG = logging.getLogger("smart_access.models")

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _jpeg_decode(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


class Models:
    """Ленивая загрузка; каждый экземпляр работает в своём потоке (run_predict)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._clip: ort.InferenceSession | None = None
        self._det: cv2.FaceDetectorYN | None = None
        self._rec: cv2.FaceRecognizerSF | None = None
        self.status: dict[str, str] = {"clip": "loading", "face": "loading"}
        self.load()

    # ---------- загрузка ----------

    def load(self) -> None:
        with self._lock:
            mdir = cfg.MODELS_DIR
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.inter_op_num_threads = 1
                path = os.path.join(mdir, "clip_vision_q.onnx")
                self._clip = ort.InferenceSession(
                    path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                self.status["clip"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._clip = None
                self.status["clip"] = f"error: {exc}"
                LOG.error("CLIP не загрузился: %s", exc)
            try:
                self._det = cv2.FaceDetectorYN.create(
                    os.path.join(mdir, "yunet.onnx"), "", (320, 320), 0.6
                )
                self._rec = cv2.FaceRecognizerSF.create(
                    os.path.join(mdir, "sface.onnx"), ""
                )
                self.status["face"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._det = None
                self._rec = None
                self.status["face"] = f"error: {exc}"
                LOG.error("Модели лиц не загрузились: %s", exc)
            LOG.info("Модели: %s", self.status)

    @property
    def ok(self) -> bool:
        return self.status.get("clip") == "ok"

    @property
    def faces_ok(self) -> bool:
        return self.status.get("face") == "ok"

    # ---------- аугментации («пре-трейн» базы под погоду/время суток) ----------

    @staticmethod
    def augment(img: np.ndarray, n_views: int = 8) -> list[np.ndarray]:
        """Сезонно-световые вариации: яркость/контраст/гамма/насыщенность/
        лёгкая расфокусировка/дождевая дымка/отражения. Для повторяемости — seed."""
        rng = np.random.default_rng(42)
        views = [img]
        h, w = img.shape[:2]
        for _ in range(n_views - 1):
            out = img.copy()
            # яркость/контраст
            a = 1.0 + rng.uniform(-0.28, 0.28)
            b = rng.uniform(-38, 38)
            out = np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            # гамма
            gamma = rng.uniform(0.65, 1.5)
            lut = np.array(
                [np.clip(((i / 255.0) ** (1.0 / gamma)) * 255, 0, 255) for i in range(256)],
                dtype=np.uint8,
            )
            out = cv2.LUT(out, lut)
            # насыщенность
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.65, 1.35), 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            # лёгкая дымка (дождь/туман) или расфокус (снег/капли)
            if rng.random() < 0.4:
                out = cv2.GaussianBlur(out, (3, 3), 0)
            if rng.random() < 0.3:
                haze = np.full_like(out, 200, dtype=np.uint8)
                out = cv2.addWeighted(out, 0.82, haze, 0.18, 0)
            # геометрия: отражение + микро-кроп
            if rng.random() < 0.5:
                out = cv2.flip(out, 1)
            sx, sy = rng.uniform(0.9, 1.0, 2)
            dx, dy = int(w * (1 - sx) * rng.random()), int(h * (1 - sy) * rng.random())
            out = out[dy: dy + int(h * sy), dx: dx + int(w * sx)]
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
            views.append(out)
        return views

    # ---------- машины ----------

    def car_embedding(self, jpg: bytes, with_views: bool = False) -> Any:
        """CLIP-эмбеддинг. with_views=True — список эмбеддингов аугментаций."""
        if self._clip is None:
            return None
        img = _jpeg_decode(jpg)
        if img is None:
            return None
        views = self.augment(img) if with_views else [img]
        batch = np.stack([self._clip_pre(v) for v in views]).astype(np.float32)
        out = self._clip.run(["image_embeds"], {"pixel_values": batch})[0]
        return out if with_views else out[0]

    @staticmethod
    def _clip_pre(img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img = (img.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
        return img.transpose(2, 0, 1)

    # ---------- лица ----------

    def face_embeddings(self, jpg_full: bytes, jpg_crop: bytes | None = None) -> list[list[float]]:
        """Возвращает до 2 эмбеддингов лиц с кадра (alignCrop + SFace)."""
        if self._det is None or self._rec is None:
            return []
        result: list[list[float]] = []
        for raw in ([jpg_crop, jpg_full] if jpg_crop else [jpg_full]):
            img = _jpeg_decode(raw) if raw else None
            if img is None:
                continue
            h, w = img.shape[:2]
            if h < 60 or w < 60:
                continue
            det = self._det
            det.setInputSize((w, h))
            faces = det.detect(img)[1]
            if faces is None:
                continue
            for face in faces[:2]:
                aligned = self._rec.alignCrop(img, face)
                feat = self._rec.feature(aligned).flatten()
                result.append([float(x) for x in feat])
            if result:
                break
        return result

    def face_embedding_of_aligned(self, jpg: bytes) -> list[float] | None:
        """Эмбеддинг уже выровненного лица 112x112 (для enrollment из полного кадра)."""
        if self._rec is None:
            return None
        img = _jpeg_decode(jpg)
        if img is None:
            return None
        feat = self._rec.feature(img).flatten()
        return [float(x) for x in feat]

    def detect_face_raw(self, jpg: bytes) -> tuple[np.ndarray | None, int, int]:
        """Детекция лица + возврат выровненного кропа (для веб-UI предпросмотра)."""
        if self._det is None or self._rec is None:
            return None, 0, 0
        img = _jpeg_decode(jpg)
        if img is None:
            return None, 0, 0
        h, w = img.shape[:2]
        det = self._det
        det.setInputSize((w, h))
        faces = det.detect(img)[1]
        if faces is None:
            return None, w, h
        return self._rec.alignCrop(img, faces[0]), w, h
