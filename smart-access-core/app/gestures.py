"""Жесты: каталог, геометрические правила (pose) и геометрия кропа кисти.

v1.3.0 — жесты НАСТРАИВАЕМЫЕ, отдельно для каждого лица (как обучение лиц):
у каждого subject свой набор «жест-подтверждение» и «тревожный жест».
Детерминированных правил больше нет: «рука вверх» — лишь один из жестов
каталога, и по умолчанию никому не назначен.

Каталог — два типа:
  * pose   — распознаётся геометрически по 17 ключевым точкам COCO
             (YOLOv8n-pose), обучение не нужно: touch_nose, touch_ear,
             one_hand_up, both_hands_up;
  * hand   — форма кисти (victory, ok, thumbs_up, thumbs_down, palm);
             распознаётся по фото-прототипам (CLIP): 1–3 фото жеста с той же
             камеры -> эмбеддинги кропа кисти -> косинусное сходство.
             Настраивается «аналогично лицам»: показал жест камере — обучил.

Модуль чистый (только numpy): тестируется без ONNX и cv2.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# --- индексы ключевых точек COCO (17) ------------------------------------- #
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10

# --- каталог жестов -------------------------------------------------------- #
# id стабильны (хранятся в gesture_cfg.json / gestures.json), label — для UI.
GESTURES: dict[str, dict[str, str]] = {
    "touch_nose": {
        "label": "Коснуться носа", "type": "pose",
        "hint": "Естественный незаметный жест: рука у лица, внимания не привлекает.",
    },
    "touch_ear": {
        "label": "Коснуться уха", "type": "pose",
        "hint": "Как поправить ухо/волосу — никто не догадается, что это «пароль».",
    },
    "victory": {
        "label": "Знак V (victory)", "type": "hand",
        "hint": "Два пальца — можно показать на уровне пояса, вторая рука занята сумками.",
    },
    "ok": {
        "label": "Знак OK", "type": "hand",
        "hint": "Кольцо из пальцев — обычный жест «всё в порядке».",
    },
    "thumbs_up": {
        "label": "Палец вверх", "type": "hand",
        "hint": "Привычный «класс» — не акцентирует внимание.",
    },
    "thumbs_down": {
        "label": "Палец вниз", "type": "hand",
        "hint": "Обратный «классу» — малозаметный отрицательный жест.",
    },
    "palm": {
        "label": "Открытая ладонь", "type": "hand",
        "hint": "Как будто машешь или показываешь «стоп».",
    },
    "one_hand_up": {
        "label": "Рука вверх", "type": "pose",
        "hint": "Заметный жест (v1.2.0). Оставлен для совместимости — по умолчанию не назначается.",
    },
    "both_hands_up": {
        "label": "Обе руки вверх", "type": "pose",
        "hint": "Классический SOS (v1.2.0): заметен, поэтому по умолчанию не назначается.",
    },
}

POSE_GESTURES = tuple(g for g, v in GESTURES.items() if v["type"] == "pose")
HAND_GESTURES = tuple(g for g, v in GESTURES.items() if v["type"] == "hand")


def is_valid_gesture(gid: str) -> bool:
    return str(gid or "").strip() in GESTURES


def gesture_label(gid: str) -> str:
    return GESTURES.get(str(gid or "").strip(), {}).get("label", str(gid))


def gesture_list_safe(value: Any, fallback: list[str]) -> list[str]:
    """Список id жестов из значения конфига (список или строка 'a,b').

    Некорректные/неизвестные id отбрасываются; пустой результат -> fallback.
    """
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]
    out = [str(x).strip() for x in (value or []) if is_valid_gesture(str(x).strip())]
    return out or list(fallback)


# --- геометрические правила ------------------------------------------------ #

def _conf(kpt: np.ndarray, i: int) -> float:
    return float(kpt[i][2]) if kpt.shape[0] > i else 0.0


def _span(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def face_scale(kpt: np.ndarray, kpt_conf: float = 0.3) -> float | None:
    """Масштаб «размер лица» в пикселях кадра: межушное расстояние.

    Уши часто малоконтрастны — фолбэк: 0.42 ширины плеч (стабильные точки).
    """
    if _conf(kpt, L_EAR) >= kpt_conf and _conf(kpt, R_EAR) >= kpt_conf:
        span = _span(kpt[L_EAR], kpt[R_EAR])
        if span > 1e-3:
            return span
    if _conf(kpt, L_SHOULDER) >= kpt_conf and _conf(kpt, R_SHOULDER) >= kpt_conf:
        return 0.42 * _span(kpt[L_SHOULDER], kpt[R_SHOULDER])
    return None


def pose_gestures(kpt: np.ndarray, kpt_conf: float = 0.3) -> tuple[set[str], dict[str, float]]:
    """Правила pose-жестов для ОДНОГО человека по 17 точкам.

    Возвращает (множество распознанных id, детали-расстояния для журнала).
    Пороги — в долях межушного расстояния (масштабо-независимы):
      * touch_nose — запястье у носа (d <= 0.55 * face_scale);
      * touch_ear  — запястье у уха  (d <= 0.45 * face_scale);
      * one_hand_up / both_hands_up — запястье выше плеча (совместимость
        v1.2.0). Касание носа/уха и «рука вверх» не исключают друг друга:
        рука у лица и так выше плеча, persona решает, какой жест назначен.
    """
    hits: set[str] = set()
    det: dict[str, float] = {}

    if kpt.shape[0] < 11:
        return hits, det

    fs = face_scale(kpt, kpt_conf)
    if fs and fs > 1e-3:
        nose = kpt[NOSE]
        for side, wi in (("L", L_WRIST), ("R", R_WRIST)):
            if _conf(kpt, wi) < kpt_conf:
                continue
            wrist = kpt[wi]
            # кандидаты жеста для этой руки: относительное расстояние до лица
            cands: dict[str, float] = {}
            d_nose = _span(wrist, nose) / fs
            det[f"d_{side}_nose"] = round(d_nose, 2)
            if d_nose <= 0.55:
                cands["touch_nose"] = d_nose
            d_ear_min = None
            for ei in (L_EAR, R_EAR):
                if _conf(kpt, ei) < max(0.1, kpt_conf * 0.5):
                    continue
                d_ear = _span(wrist, kpt[ei]) / fs
                det[f"d_{side}_ear{ei}"] = round(d_ear, 2)
                if d_ear <= 0.45 and (d_ear_min is None or d_ear < d_ear_min):
                    d_ear_min = d_ear
            if d_ear_min is not None:
                cands["touch_ear"] = d_ear_min
            # рука показывает ОДИН жест: берём ближайшую цель
            # (касание носа и уха геометрически близки — без этого
            # оба правила сработали бы одновременно)
            if cands:
                best = min(cands, key=cands.get)  # type: ignore[arg-type]
                hits.add(best)

    # совместимость v1.2.0: запястье выше плеча
    up = 0
    for wi, si in ((L_WRIST, L_SHOULDER), (R_WRIST, R_SHOULDER)):
        if _conf(kpt, wi) < kpt_conf:
            continue
        if _conf(kpt, si) < kpt_conf:
            continue
        if float(kpt[wi][1]) < float(kpt[si][1]):  # y растёт вниз
            up += 1
    if up >= 2:
        hits.add("both_hands_up")
    elif up == 1:
        hits.add("one_hand_up")

    return hits, det


def hand_boxes(kpt: np.ndarray, kpt_conf: float = 0.3) -> list[dict[str, Any]]:
    """Квадратные кропы вокруг кистей одного человека (для CLIP-матчинга).

    Центр — запястье + продолжение предплечья (кисть дальше локтя),
    сторона квадрата — 0.9 ширины плеч, зажата в [40, 220] px.
    Возвращает [{'box': (x1, y1, x2, y2), 'arm': 'L'|'R', 'side': px}].
    """
    out: list[dict[str, Any]] = []
    if kpt.shape[0] < 11:
        return out
    if _conf(kpt, L_SHOULDER) >= kpt_conf and _conf(kpt, R_SHOULDER) >= kpt_conf:
        span = _span(kpt[L_SHOULDER], kpt[R_SHOULDER])
    else:
        return out
    if span < 1e-3:
        return out
    side = float(np.clip(0.9 * span, 40.0, 220.0))

    for arm, wi, ei in (("L", L_WRIST, L_ELBOW), ("R", R_WRIST, R_ELBOW)):
        if _conf(kpt, wi) < kpt_conf:
            continue
        wrist = kpt[wi]
        cx, cy = float(wrist[0]), float(wrist[1])
        if _conf(kpt, ei) >= kpt_conf:
            elbow = kpt[ei]
            v = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]], np.float32)
            n = float(np.linalg.norm(v))
            if n > 1e-3:
                # кисть ~0.15 ширины плеч дальше запястья по направлению руки
                cx += 0.15 * span * float(v[0]) / n
                cy += 0.15 * span * float(v[1]) / n
        half = side / 2.0
        out.append({"box": (cx - half, cy - half, cx + half, cy + half),
                    "arm": arm, "side": side})
    return out


def clamp_box(box: tuple[float, float, float, float],
              w: int, h: int) -> tuple[int, int, int, int]:
    """Обрезка бокса по границам кадра (int, порядок x1<=x2, y1<=y2)."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), w))
    y2 = max(y1 + 1, min(int(round(y2)), h))
    return x1, y1, x2, y2
