"""v1.3.0: настраиваемые жесты — «жесты как лица».

Покрывает:
  * gestures — каталог, pose-правила (touch_nose/touch_ear/руки вверх,
    взаимоисключаемость нос/ухо, масштабо-независимость), геометрия
    кропов кисти (hand_boxes/clamp_box);
  * Store — привязка жестов к лицу, прототипы CLIP, фото жестов,
    очистка при удалении лица;
  * Brain — решение по жестам, настроенным ДЛЯ КОНКРЕТНОГО лица
    (тревога / подтверждение / ожидание), фолбэк старого правила для
    ненастроенных лиц, обучение и переобучение hand-жестов;
  * Config — список жестов из опций (строка/список, фильтрация мусора).

Запуск:  python3 -m pytest smart-access-core/tests/ -v
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

from app import config as cfg            # noqa: E402
from app import gestures as gest         # noqa: E402
from app import ha_api                   # noqa: E402
from app import store as store_mod       # noqa: E402
from app.brain import Brain              # noqa: E402
from app.store import Store              # noqa: E402


# --------------------------------------------------------------------- #
# Фикстуры

@pytest.fixture()
def store_env(tmp_path, monkeypatch):
    """Изолированное окружение: /data, /media — во временном каталоге."""
    data = tmp_path / "data"
    media = tmp_path / "media" / "smart_access"
    data.mkdir(parents=True)
    media.mkdir(parents=True)
    monkeypatch.setattr(cfg, "DATA_DIR", str(data))
    monkeypatch.setattr(cfg, "MEDIA_DIR", str(media))
    return Store()


# --------------------------------------------------------------------- #
# 1. gestures: pose-правила на синтетических точках COCO

def _person(l_wrist, r_wrist, nose=(150, 260), l_ear=(130, 258), r_ear=(170, 258),
            l_sh=(100, 300), r_sh=(200, 300), conf=0.9, ear_conf=None) -> np.ndarray:
    kpt = np.zeros((17, 3), np.float32)
    kpt[:, 2] = conf
    if ear_conf is not None:
        kpt[3][2] = kpt[4][2] = ear_conf
    for i, (x, y) in {0: nose, 3: l_ear, 4: r_ear, 5: l_sh, 6: r_sh,
                      9: l_wrist, 10: r_wrist}.items():
        kpt[i][0], kpt[i][1] = x, y
    return kpt


def test_catalog_and_helpers():
    assert set(gest.POSE_GESTURES) == {"touch_nose", "touch_ear", "one_hand_up", "both_hands_up"}
    assert set(gest.HAND_GESTURES) == {"victory", "ok", "thumbs_up", "thumbs_down", "palm"}
    assert gest.is_valid_gesture("victory") and not gest.is_valid_gesture("middle_finger")
    assert gest.gesture_label("touch_nose") == "Коснуться носа"
    # строка конфига 'victory, ok, мусор' -> только валидные
    assert gest.gesture_list_safe("victory, ok, мусор", ["palm"]) == ["victory", "ok"]
    assert gest.gesture_list_safe("", ["one_hand_up"]) == ["one_hand_up"]
    assert gest.gesture_list_safe(["nope"], ["both_hands_up"]) == ["both_hands_up"]


def test_pose_touch_nose_and_ear_exclusive():
    # запястье у носа: d≈5 при межушном 40 -> touch_nose, не touch_ear
    hits, det = gest.pose_gestures(_person(l_wrist=(150, 265), r_wrist=(400, 400)))
    assert "touch_nose" in hits and "touch_ear" not in hits
    assert det["d_L_nose"] < 0.55

    # запястье у уха: ближайшая цель — ухо, нос исключён
    hits, _ = gest.pose_gestures(_person(l_wrist=(132, 258), r_wrist=(400, 400)))
    assert "touch_ear" in hits and "touch_nose" not in hits

    # обе руки: одна у носа, другая у правого уха
    hits, _ = gest.pose_gestures(_person(l_wrist=(150, 265), r_wrist=(168, 258)))
    assert "touch_nose" in hits and "touch_ear" in hits


def test_pose_arms_up_and_nothing():
    hits, _ = gest.pose_gestures(_person(l_wrist=(90, 280), r_wrist=(210, 380)))
    assert hits == {"one_hand_up"}                       # рука вверх не «трогает лицо»
    hits, _ = gest.pose_gestures(_person(l_wrist=(90, 280), r_wrist=(210, 270)))
    assert hits == {"both_hands_up"}
    hits, _ = gest.pose_gestures(_person(l_wrist=(90, 380), r_wrist=(210, 390)))
    assert hits == set()                                 # руки опущены
    # низкая уверенность запястья — жест не считается
    hits, _ = gest.pose_gestures(_person(l_wrist=(150, 265), r_wrist=(210, 380), conf=0.1))
    assert hits == set()


def test_pose_scale_independent_and_ear_fallback():
    # тот же жест на удвоенном масштабе (человек ближе к камере)
    kpt_near = _person(l_wrist=(300, 205), r_wrist=(800, 800), nose=(300, 210),
                       l_ear=(260, 206), r_ear=(340, 206),
                       l_sh=(200, 300), r_sh=(400, 300))
    hits, _ = gest.pose_gestures(kpt_near)
    assert "touch_nose" in hits
    # уши невидимы (conf=0) — фолбэк на 0.42 ширины плеч
    kpt_noears = _person(l_wrist=(150, 265), r_wrist=(400, 400), ear_conf=0.0)
    hits, _ = gest.pose_gestures(kpt_noears)
    assert "touch_nose" in hits


def test_hand_boxes_geometry():
    kpt = _person(l_wrist=(90, 380), r_wrist=(210, 380))
    boxes = gest.hand_boxes(kpt)
    assert {b["arm"] for b in boxes} == {"L", "R"}
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        assert x2 - x1 == y2 - y1                        # квадрат
        assert b["side"] == pytest.approx(0.9 * 100)     # 0.9 ширины плеч
    # запястье невидимо — кропа нет
    kpt2 = _person(l_wrist=(90, 380), r_wrist=(210, 380), conf=0.1)
    assert gest.hand_boxes(kpt2) == []
    # clamp: бокс у края кадра не выходит за границы
    x1, y1, x2, y2 = gest.clamp_box((-30, -30, 60, 60), 200, 200)
    assert (x1, y1) == (0, 0) and x2 > x1 and y2 > y1
    x1, y1, x2, y2 = gest.clamp_box((180, 180, 260, 260), 200, 200)
    assert (x2, y2) == (200, 200) and x2 > x1 and y2 > y1


def test_legacy_classify_gesture_v120():
    from app.models import Models
    persons_up = [{"conf": 0.9, "kpt": _person(l_wrist=(90, 280), r_wrist=(210, 380))}]
    g = Models.classify_gesture(persons_up)
    assert g["confirm"] is True and g["alarm"] is False
    persons_alarm = [{"conf": 0.9, "kpt": _person(l_wrist=(90, 280), r_wrist=(210, 270))}]
    g = Models.classify_gesture(persons_alarm)
    assert g["alarm"] is True and g["confirm"] is False


# --------------------------------------------------------------------- #
# 2. Store: привязка жестов к лицу, прототипы, фото

def test_store_gesture_cfg_and_protos(store_env):
    s = store_env
    assert s.get_subject_gestures("Я") == {"confirm": [], "alarm": []}

    s.set_subject_gestures("Я", ["victory", "touch_nose"], ["touch_ear"])
    assert s.get_subject_gestures("Я") == {
        "confirm": ["victory", "touch_nose"], "alarm": ["touch_ear"]}
    assert s.gestures_configured_subjects() == ["Я"]

    # пустые списки — настройка снимается
    s.set_subject_gestures("Я", [], [])
    assert s.gestures_configured_subjects() == []

    # прототипы CLIP: добавление + матч
    s.set_subject_gestures("Я", ["victory"], [])
    s.add_gesture_proto("Я", "victory", [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]])
    protos = s.gesture_protos("Я", "victory")
    assert protos is not None and protos.shape == (2, 3)
    s.add_gesture_proto("Я", "ok", [[0.0, 1.0, 0.0]])
    info = s.gesture_info("Я")
    assert info["victory"]["embeds"] == 2 and info["ok"]["embeds"] == 1

    # замена прототипов (переобучение), пустая — жест удаляется
    s.replace_gesture_protos("Я", "victory", [[0.5, 0.5, 0.0]])
    assert s.gesture_protos("Я", "victory").shape == (1, 3)
    s.replace_gesture_protos("Я", "ok", [])
    assert s.gesture_protos("Я", "ok") is None

    # фото жестов: сохранение/список/защита от traversal
    jpg = _jpg_bytes(1)
    s._save_gesture_photo("Я", "victory", jpg)
    photos = s.list_gesture_photos("Я", "victory")
    assert len(photos) == 1
    assert s.gesture_photo_path("Я", "victory", photos[0]["file"]) is not None
    assert s.gesture_photo_path("Я", "victory", "../../etc/passwd") is None
    assert s.gesture_photo_path("Я", "../Я", photos[0]["file"]) is None
    assert s.delete_gesture_photos("Я", "victory", [photos[0]["file"], "../../x"]) == 1
    with pytest.raises(ValueError):
        s.gesture_photos_dir("Я", "a/b")

    # удаление лица чистит и жесты
    s.set_subject_gestures("Я", ["victory"], ["touch_ear"])
    s.add_gesture_proto("Я", "victory", [[1.0, 0.0]])
    s.add_face("Я", [[0.1, 0.2]], _jpg_bytes(2))
    assert s.remove_subject("face", "Я") is True
    assert s.gestures_configured_subjects() == []
    assert s.gesture_protos("Я", "victory") is None
    assert s.list_gesture_photos("Я", "victory") == []


# --------------------------------------------------------------------- #
# 3. Brain: решение по жестам, настроенным для лица

def _jpg_bytes(n: int = 1) -> bytes:
    import cv2
    img = np.full((120, 160, 3), 40 * n % 255, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class _GestModels:
    """Модели-фейки: лица распознаются, жесты — управляемый результат."""

    ok = True
    faces_ok = True
    status = {"clip": "off", "face": "off", "pose": "ok"}

    def __init__(self, pose=None, hands=None, scores=None) -> None:
        self.pose = pose or []
        self.hands = hands or []
        self.scores = scores or {}
        self.last_enrolled = None

    @property
    def gestures_ok(self) -> bool:
        return True

    def face_embeddings(self, jpg_full, jpg_crop=None):
        return [[1.0, 0.0, 0.0, 0.0]]

    def analyze_gestures(self, jpg, enrolled=None, threshold=0.80, kpt_conf=0.3):
        self.last_enrolled = enrolled
        return {"available": True, "persons": 1, "pose": list(self.pose),
                "hand_scores": dict(self.scores), "hands": list(self.hands), "boxes": []}

    def hand_crop_jpegs(self, jpg):
        return [("R", _jpg_bytes(3))]

    def gesture_embedding(self, jpg, n_views=4):
        return [[1.0, 0.0] for _ in range(int(n_views))]

    def car_embedding(self, jpg, with_views=False):
        return None

    def detect_face_raw(self, jpg):
        return None, 0, 0

    def face_embedding_of_aligned(self, jpg):
        return None


@pytest.fixture()
def face_brain(tmp_path, monkeypatch, store_env):
    """Brain с фейк-моделями, лицо «Я» в базе, событие frigate person."""
    st = store_env
    st.add_face("Я", [[1.0, 0.0, 0.0, 0.0]], _jpg_bytes(1))
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(ha_api, "frigate_event_snapshot",
                        lambda url, eid, crop=False: _jpg_bytes(2))
    brain = Brain(st, _GestModels(), on_status=lambda: None)
    return brain, st


def _face_conf(monkeypatch, gestures: dict) -> None:
    conf = types.SimpleNamespace(
        master_enable=True, frigate_url="http://x",
        mqtt={"host": "127.0.0.1"},
        cameras=[{"name": "kalitka", "display_name": "Калитка", "mode": "face",
                  "ha_entity": "camera.kalitka", "executor_service": "", "cooldown": 45}],
        recognition={"face_threshold": 0.363, "min_confirm_frames": 2},
        vlm={"enabled": False},
        gestures=gestures,
        alarm={"enabled": True, "addon": "a", "numbers": "sip:t@pbx",
               "ring_timeout": 10, "sip_account": 1, "language": "ru",
               "message_template": "T", "notify": True},
    )
    conf.camera_by_name = lambda name: next(
        (c for c in conf.cameras if c["name"] == name), None)
    monkeypatch.setattr(cfg, "CONF", conf)


def _face_event(eid: str = "e1") -> bytes:
    return json.dumps({"type": "new",
                       "after": {"id": eid, "label": "person",
                                 "camera": "kalitka"}}).encode()


def test_brain_alarm_gesture_configured_per_face(monkeypatch, face_brain):
    brain, st = face_brain
    # у «Я» тревожный жест — коснуться уха; подтверждение — победа
    st.set_subject_gestures("Я", ["victory"], ["touch_ear"])
    brain.models = _GestModels(pose=["touch_ear"])
    _face_conf(monkeypatch, {"enabled": True, "threshold": 0.8,
                             "default_confirm": ["one_hand_up"],
                             "default_alarm": ["both_hands_up"]})

    brain._handle_frigate_event(_face_event())
    ev = st.query_events({"action": "alarm"})
    assert len(ev) == 1 and ev[0]["subject"] == "Я"
    assert "Коснуться уха" in ev[0]["detail"]
    assert brain.stats["alarms"] == 1 and brain.stats["grants"] == 1


def test_brain_confirm_gesture_configured_per_face(monkeypatch, face_brain):
    brain, st = face_brain
    st.set_subject_gestures("Я", ["touch_nose"], ["touch_ear"])
    brain.models = _GestModels(pose=["touch_nose"])
    _face_conf(monkeypatch, {"enabled": True, "threshold": 0.8})

    brain._handle_frigate_event(_face_event())
    assert not st.query_events({"action": "alarm"})
    ev = st.query_events({"action": "detected"})
    assert len(ev) == 1 and "Коснуться носа" in ev[0]["detail"]


def test_brain_hand_gesture_via_clip_protos(monkeypatch, face_brain):
    brain, st = face_brain
    st.set_subject_gestures("Я", ["victory"], [])
    st.add_gesture_proto("Я", "victory", [[1.0, 0.0]])
    brain.models = _GestModels(hands=["victory"], scores={"victory": 0.87})
    _face_conf(monkeypatch, {"enabled": True, "threshold": 0.8})

    brain._handle_frigate_event(_face_event())
    # прототипы «victory» этого лица реально переданы в матчинг
    assert brain.models.last_enrolled is not None and "victory" in brain.models.last_enrolled
    ev = st.query_events({"action": "detected"})
    assert len(ev) == 1 and "Знак V (victory)" in ev[0]["detail"]


def test_brain_gesture_wait_when_no_configured_gesture(monkeypatch, face_brain):
    brain, st = face_brain
    st.set_subject_gestures("Я", ["victory"], [])
    brain.models = _GestModels(pose=["one_hand_up"])  # рука вверх, но лицу назначен V
    _face_conf(monkeypatch, {"enabled": True, "threshold": 0.8})

    brain._handle_frigate_event(_face_event())
    assert st.query_events({"action": "detected"}) == []
    ev = st.query_events({"action": "gesture_wait"})
    assert len(ev) == 1
    assert "Знак V (victory)" in ev[0]["detail"]          # ждём НАСТРОЕННЫЙ жест
    assert "Рука вверх" in ev[0]["detail"]                # а виден — старый


def test_brain_fallback_legacy_for_unconfigured_face(monkeypatch, face_brain):
    brain, st = face_brain
    brain.models = _GestModels(pose=["one_hand_up"])
    _face_conf(monkeypatch, {"enabled": True, "threshold": 0.8,
                             "default_confirm": ["one_hand_up"],
                             "default_alarm": ["both_hands_up"]})

    brain._handle_frigate_event(_face_event())            # лицо БЕЗ настройки жестов
    ev = st.query_events({"action": "detected"})
    assert len(ev) == 1 and "Рука вверх" in ev[0]["detail"]

    # «обе руки вверх» у ненастроенного лица — тревога (совместимость v1.2.0)
    brain2 = Brain(st, _GestModels(pose=["both_hands_up"]), on_status=lambda: None)
    brain2._handle_frigate_event(_face_event("e2"))
    assert st.query_events({"action": "alarm"})           # и открытие, и звонок


def test_brain_alarm_gesture_wins_over_confirm(monkeypatch, face_brain):
    brain, st = face_brain
    st.set_subject_gestures("Я", ["touch_nose"], ["touch_ear"])
    brain.models = _GestModels(pose=["touch_nose", "touch_ear"])
    _face_conf(monkeypatch, {"enabled": True, "threshold": 0.8})

    brain._handle_frigate_event(_face_event())
    ev = st.query_events({"action": "alarm"})
    assert len(ev) == 1                                    # тревога приоритетнее открытия


def test_brain_gestures_disabled_or_model_off(monkeypatch, face_brain):
    brain, st = face_brain
    st.set_subject_gestures("Я", ["victory"], [])
    brain.models = _GestModels(pose=[])                    # жестов нет
    _face_conf(monkeypatch, {"enabled": False, "threshold": 0.8})

    brain._handle_frigate_event(_face_event())             # выключено — открытие без жеста
    assert st.query_events({"action": "detected"})

    brain2 = Brain(st, _NoPoseModels(), on_status=lambda: None)
    monkeypatch.setattr(cfg, "CONF", cfg.CONF)             # CONF уже стоит
    brain2._handle_frigate_event(_face_event("e3"))
    assert st.query_events({"action": "detected"})         # модель поз недоступна — fail-open


class _NoPoseModels(_GestModels):
    @property
    def gestures_ok(self) -> bool:
        return False


def test_brain_enroll_and_retrain_gesture(monkeypatch, face_brain):
    brain, st = face_brain
    r = brain.enroll_gesture("Я", "victory", _jpg_bytes(1))
    assert r["ok"] and r["added"] == 8 and r["crops"] == 1
    assert st.gesture_protos("Я", "victory").shape == (8, 2)
    assert len(st.list_gesture_photos("Я", "victory")) == 1

    # pose-жесты не обучаются
    r = brain.enroll_gesture("Я", "touch_ear", _jpg_bytes(1))
    assert not r["ok"] and "обучение не нужно" in r["error"]
    # неизвестный жест
    assert not brain.enroll_gesture("Я", "nope", _jpg_bytes(1))["ok"]

    # переобучение по сохранённым кропам
    r = brain.retrain_gesture("Я", "victory")
    assert r["ok"] and r["added"] == 8
    # после пересчёта прототипы на месте
    assert st.gesture_protos("Я", "victory") is not None


def test_brain_test_gesture_report(monkeypatch, face_brain):
    brain, st = face_brain
    st.set_subject_gestures("Я", ["victory"], ["touch_ear"])
    brain.models = _GestModels(pose=["touch_ear"], hands=["victory"],
                               scores={"victory": 0.91})
    _face_conf(monkeypatch, {"enabled": True, "threshold": 0.8})
    r = brain.test_gesture("Я", _jpg_bytes(1))
    assert r["ok"] and r["persons"] == 1
    assert r["hits_alarm"] == ["touch_ear"] and r["hits_confirm"] == ["victory"]
    assert r["hand_scores"] == {"victory": 0.91}
    assert r["labels"]["touch_ear"] == "Коснуться уха"


# --------------------------------------------------------------------- #
# 4. Config: жесты из опций

def test_config_gestures_defaults(tmp_path, monkeypatch):
    options = {
        "master_enable": True, "frigate_url": "http://127.0.0.1:8971",
        "mqtt": {"host": "127.0.0.1", "port": 1883},
        "cameras": [], "recognition": {}, "vlm": {"enabled": False},
        "gestures": {"enabled": True, "threshold": 0.85,
                     "default_confirm": ["touch_ear"], "default_alarm": ["touch_nose"]},
        "alarm": {"enabled": False, "addon": "a", "numbers": "",
                  "sip_account": 1, "ring_timeout": 15, "language": "ru",
                  "message_template": "m", "notify": True},
    }
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "_read_options", lambda: json.loads(json.dumps(options)))
    c = cfg.Config()
    assert c.gestures["enabled"] is True
    assert c.gestures["threshold"] == 0.85
    assert c.gestures["default_confirm"] == ["touch_ear"]
    assert c.gestures["default_alarm"] == ["touch_nose"]
    assert "gestures" in c.as_safe_dict()
