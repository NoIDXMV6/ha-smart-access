"""Распознавание: CLIP ViT-B/32 (int8, CPU) для машин, YuNet+SFace для лиц,
YOLOv8n-pose (int8, CPU) для жестов.

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
        self._pose: ort.InferenceSession | None = None
        self.status: dict[str, str] = {"clip": "loading", "face": "loading", "pose": "loading"}
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
            try:
                ppath = os.environ.get("SMART_ACCESS_POSE_MODEL") or os.path.join(mdir, "yolov8n-pose.onnx")
                self._pose = ort.InferenceSession(
                    ppath, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                self.status["pose"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._pose = None
                # не ошибка запуска: жесты — опциональная функция
                self.status["pose"] = "off (модель не найдена)"
                LOG.warning("Модель поз не загрузилась (%s) — функционал жестов отключён", exc)
            LOG.info("Модели: %s", self.status)

    @property
    def ok(self) -> bool:
        return self.status.get("clip") == "ok"

    @property
    def faces_ok(self) -> bool:
        return self.status.get("face") == "ok"

    @property
    def gestures_ok(self) -> bool:
        return self._pose is not None

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

    # ---------- жесты (YOLOv8n-pose, 17 ключевых точек COCO) ----------

    # Индексы точек COCO: 0 nose, 5 L-shoulder, 6 R-shoulder,
    #                     7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist
    KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
    KP_L_WRIST, KP_R_WRIST = 9, 10

    def detect_poses(self, jpg: bytes, conf_th: float = 0.35,
                     max_persons: int = 5) -> list[dict[str, Any]]:
        """Детекция людей и их 17 точек на полном кадре.

        Возвращает [{'conf': float, 'kpt': np.ndarray(17, 3)}] — точки
        в координатах ИСХОДНОГО кадра (x, y, score), y растёт вниз.
        """
        if self._pose is None:
            return []
        img = _jpeg_decode(jpg)
        if img is None:
            return []
        h0, w0 = img.shape[:2]
        size = 640
        r = min(size / h0, size / w0)
        nh, nw = max(1, int(h0 * r)), max(1, int(w0 * r))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((size, size, 3), 114, np.uint8)
        dy, dx = (size - nh) // 2, (size - nw) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self._pose.run(None, {"images": x})[0][0]  # (56, 8400)
        conf = out[4]
        cand = np.where(conf > conf_th)[0]
        if cand.size == 0:
            return []
        boxes = out[:4, cand].T  # (N, 4) cx,cy,w,h в сетке 640
        scores = conf[cand]
        keep = self._nms(boxes, scores, iou_th=0.55)
        persons: list[dict[str, Any]] = []
        for i in keep[:max_persons]:
            idx = int(cand[i])
            kpt = out[5:, idx].reshape(17, 3).astype(np.float32)
            kpt[:, 0] = (kpt[:, 0] - dx) / r
            kpt[:, 1] = (kpt[:, 1] - dy) / r
            persons.append({"conf": float(conf[idx]), "kpt": kpt})
        return persons

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float = 0.55) -> list[int]:
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_o = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
            iou = inter / (area_i + area_o - inter + 1e-9)
            order = order[1:][iou <= iou_th]
        return keep

    @classmethod
    def classify_gesture(cls, persons: list[dict[str, Any]], kpt_conf: float = 0.25) -> dict[str, Any]:
        """Правила жестов по ключевым точкам.

        * 'alarm'   — ОБЕ руки подняты выше плеч (классический SOS-жест);
        * 'confirm' — хотя бы ОДНА рука поднята выше плеча.
        Точка считается валидной при score >= kpt_conf.
        """
        res: dict[str, Any] = {"alarm": False, "confirm": False,
                               "persons": len(persons), "available": True}

        def arm_up(kpt: np.ndarray, wrist_i: int, shoulder_i: int) -> bool:
            if kpt.shape[0] <= shoulder_i:
                return False
            wr, sh = kpt[wrist_i], kpt[shoulder_i]
            if wr[2] < kpt_conf or sh[2] < kpt_conf:
                return False
            return float(wr[1]) < float(sh[1])  # y растёт вниз: рука ВЫШЕ плеча

        for p in persons:
            kpt = p.get("kpt")
            if kpt is None:
                continue
            left_up = arm_up(kpt, cls.KP_L_WRIST, cls.KP_L_SHOULDER)
            right_up = arm_up(kpt, cls.KP_R_WRIST, cls.KP_R_SHOULDER)
            if left_up and right_up:
                res["alarm"] = True
            elif left_up or right_up:
                res["confirm"] = True
        return res

    def detect_gesture(self, jpg: bytes) -> dict[str, Any]:
        """Полный путь: кадр -> позы -> жест. Работает только при gestures_ok."""
        if self._pose is None:
            return {"alarm": False, "confirm": False, "persons": 0, "available": False}
        return self.classify_gesture(self.detect_poses(jpg))
