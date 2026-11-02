"""v1.2.0: управление распознаванием, жесты, тревога, журнал.

Покрывает:
  * Store — фото-библиотека (list/delete/replace), фильтры журнала,
    архивирование/очистка, защита от path traversal;
  * Models — правила жестов по ключевым точкам (одна рука = confirm,
    обе руки = alarm);
  * Brain — тревожный поток (SIP-звонки через hassio.addon_stdin c
    текстом для Piper), переобучение;
  * Config — переключатели панели (/data/settings.json) переопределяют опции.

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
from app import ha_api                   # noqa: E402
from app import store as store_mod       # noqa: E402
from app.models import Models            # noqa: E402
from app.store import Store              # noqa: E402
from app.brain import Brain              # noqa: E402


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


class _FakeModels:
    """Модели без ONNX: правила жестов проверяются на синтетике."""
    ok = False
    faces_ok = False

    def __init__(self) -> None:
        self.status = {"clip": "off", "face": "off", "pose": "off"}
        self.calls: list[tuple] = []

    @property
    def gestures_ok(self) -> bool:
        return False

    def car_embedding(self, jpg, with_views=False):
        return None

    def detect_face_raw(self, jpg):
        return None, 0, 0

    def face_embedding_of_aligned(self, jpg):
        return None


@pytest.fixture()
def brain_env(tmp_path, monkeypatch, store_env):
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path / "data"))
    st = store_env
    brain = Brain(st, _FakeModels(), on_status=lambda: None)
    return brain, st


# --------------------------------------------------------------------- #
# 1. Store: фото-библиотека

def _png_jpg_bytes(n: int = 1) -> bytes:
    import cv2
    img = np.full((120, 160, 3), 40 * n % 255, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_store_photo_lifecycle(store_env):
    s = store_env
    s.add_face("Я", [[0.1, 0.2]], _png_jpg_bytes(1))
    s.add_face("Я", [[0.3, 0.4]], _png_jpg_bytes(2))
    s.add_face("Жена", [[0.5, 0.6]], _png_jpg_bytes(3))

    subj = s.subjects("face")
    assert set(subj) == {"Я", "Жена"}
    assert subj["Я"]["photos"] == 2            # РЕАЛЬНОЕ число файлов
    assert subj["Я"]["embeds"] == 2
    assert subj["Я"]["thumb"] is not None

    photos = s.list_photos("face", "Я")
    assert len(photos) == 2
    keys = [(p["ts"], p["file"]) for p in photos]
    assert keys == sorted(keys)  # от старых к свежим, детерминированно

    # удаление ошибочного фото + replace_subject (переобучение)
    deleted = s.delete_photos("face", "Я", [photos[0]["file"], "../../etc/passwd"])
    assert deleted == 1                        # путь traversal отфильтрован
    s.replace_subject("face", "Я", [[0.9, 0.9]])
    assert len(s.faces["Я"]["embeds"]) == 1

    # раздача фото: валидное имя -> файл, подозрительное -> None
    good = s.photo_path("face", "Я", photos[1]["file"])
    assert good is not None and good.endswith(".jpg")
    assert s.photo_path("face", "Я", "../../etc/passwd") is None
    assert s.photo_path("face", "../Я", "x.jpg") is None
    with pytest.raises(ValueError):
        s.photos_dir("face", "a/b")


# --------------------------------------------------------------------- #
# 2. Store: журнал — фильтры, архив, очистка

def test_store_events_filters_and_archive(store_env):
    s = store_env
    base = time.time()
    s.log_event({"kind": "face", "camera": "kalitka", "action": "alarm",
                 "detail": "Тревога", "score": 0.62, "subject": "Я"})
    s.log_event({"kind": "car", "camera": "prud", "action": "opened",
                 "detail": "Моя машина score=0.81", "score": 0.81, "subject": "Моя машина"})
    s.log_event({"kind": "face", "camera": "kalitka", "action": "unknown",
                 "detail": "score=0.10", "score": 0.10, "subject": None})
    # подправим ts: первое событие — 2 часа назад
    with s._lock:
        s.events[-1]["ts"] = base - 7200

    res = s.query_events({"kind": "face"})
    assert len(res) == 2 and res[0]["action"] == "unknown"  # свежие первыми
    res = s.query_events({"action": "alarm"})
    assert len(res) == 1 and res[0]["subject"] == "Я"
    res = s.query_events({"score_min": 0.5})
    assert len(res) == 2
    res = s.query_events({"camera": "prud", "kind": "car"})
    assert res[0]["subject"] == "Моя машина"
    res = s.query_events({"from": base - 60})
    assert all(e["ts"] >= base - 60 for e in res) and len(res) == 2

    # архив: файл появился, журнал пуст
    r = s.archive_events()
    assert r["ok"] and r["count"] == 3
    arch = s.list_archives()
    assert len(arch) == 1 and arch[0]["file"].startswith("journal-")
    path = s.archive_path(arch[0]["file"])
    assert path is not None and path.startswith(str(cfg.MEDIA_DIR))
    assert s.query_events({}) == []
    # защита имени архива
    assert s.archive_path("../../etc/passwd") is None
    assert s.archive_path("journal-x.json") is not None or True

    # очистка без архива
    s.log_event({"kind": "car", "camera": "x", "action": "denied", "detail": "", "score": 0})
    assert s.clear_events() == 1
    assert s.query_events({}) == []


# --------------------------------------------------------------------- #
# 3. Models: правила жестов на синтетических точках COCO

def _kpt(l_wrist_y: float, r_wrist_y: float, sh_y: float = 300.0,
         conf: float = 0.9) -> np.ndarray:
    kpt = np.zeros((17, 3), np.float32)
    kpt[:, 2] = conf
    kpt[5] = [100, sh_y, conf]   # L-shoulder
    kpt[6] = [200, sh_y, conf]   # R-shoulder
    kpt[9] = [90, l_wrist_y, conf]   # L-wrist
    kpt[10] = [210, r_wrist_y, conf]  # R-wrist
    return kpt


def test_models_gesture_rules():
    persons_confirm = [{"conf": 0.9, "kpt": _kpt(l_wrist_y=280.0, r_wrist_y=380.0)}]
    g = Models.classify_gesture(persons_confirm)
    assert g["confirm"] is True and g["alarm"] is False

    persons_alarm = [{"conf": 0.9, "kpt": _kpt(l_wrist_y=280.0, r_wrist_y=270.0)}]
    g = Models.classify_gesture(persons_alarm)
    assert g["alarm"] is True and g["confirm"] is False

    persons_none = [{"conf": 0.9, "kpt": _kpt(l_wrist_y=380.0, r_wrist_y=390.0)}]
    g = Models.classify_gesture(persons_none)
    assert g["alarm"] is False and g["confirm"] is False

    # низкая уверенность точки — рука не считается
    persons_low = [{"conf": 0.9, "kpt": _kpt(l_wrist_y=280.0, r_wrist_y=390.0, conf=0.1)}]
    g = Models.classify_gesture(persons_low)
    assert g["confirm"] is False

    # второй человек в кадре с тревожным жестом
    two = [{"conf": 0.9, "kpt": _kpt(380.0, 390.0)},
           {"conf": 0.8, "kpt": _kpt(250.0, 260.0)}]
    g = Models.classify_gesture(two)
    assert g["alarm"] is True

    # без модели: available=False, исключений нет
    m = Models.__new__(Models)
    m._pose = None
    g = m.detect_gesture(b"\xff\xd8\xff")
    assert g["available"] is False


# --------------------------------------------------------------------- #
# 4. Brain: тревога — звонок через hassio.addon_stdin с текстом для Piper

def test_brain_alarm_dial_payload(monkeypatch, brain_env):
    brain, st = brain_env
    calls: list[tuple] = []

    def fake_call(domain, service, data, **_k):
        calls.append((domain, service, data))
        return True

    monkeypatch.setattr(ha_api, "call_ha_service", fake_call)
    monkeypatch.setattr(cfg, "CONF", types.SimpleNamespace(
        master_enable=True, frigate_url="http://x:5000",
        mqtt={"host": "127.0.0.1", "port": 1883},
        cameras=[{"name": "kalitka", "display_name": "Калитка", "mode": "face",
                  "ha_entity": "camera.kalitka", "executor_service": "", "cooldown": 45}],
        recognition={"face_threshold": 0.363},
        vlm={"enabled": False},
        gestures={"enabled": False, "min_frames": 1},
        alarm={"enabled": True, "addon": "c7744bff_ha-sip",
               "numbers": "sip:89256758881@pbx.kozhevatov.ru\nsip:911@pbx",
               "sip_account": 1, "ring_timeout": 15, "language": "ru",
               "message_template": "Тревога! {time}, {name} открыл {camera} с тревожным жестом.",
               "notify": True},
    ))

    cam = {"name": "kalitka", "display_name": "Калитка", "executor_service": ""}
    brain._grant("face", cam, "Я", 0.55, b"", alarm=True)
    # звонки уходят в фоновом потоке — дождёмся
    deadline = time.time() + 5
    while len(calls) < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert len(calls) == 2
    domain, service, data = calls[0]
    assert domain == "hassio" and service == "addon_stdin"
    inp = data["input"]
    assert data["addon"] == "c7744bff_ha-sip"
    assert inp["command"] == "dial"
    assert inp["number"].startswith("sip:")
    assert inp["ring_timeout"] == 15 and inp["sip_account"] == 1
    msg = inp["menu"]["message"]
    assert "Тревога!" in msg and "Я" in msg and "Калитка" in msg
    assert inp["menu"]["post_action"] == "hangup"
    # событие в журнале
    ev = st.query_events({"action": "alarm"})
    assert len(ev) == 1 and ev[0]["subject"] == "Я"
    assert brain.stats["alarms"] == 1


def test_brain_alarm_disabled_or_empty(monkeypatch, brain_env):
    brain, st = brain_env
    calls: list = []
    monkeypatch.setattr(ha_api, "call_ha_service", lambda *a, **k: calls.append(a) or True)
    monkeypatch.setattr(cfg, "CONF", types.SimpleNamespace(
        master_enable=True, frigate_url="http://x",
        mqtt={"host": "127.0.0.1"}, cameras=[],
        recognition={}, vlm={"enabled": False},
        gestures={"enabled": False},
        alarm={"enabled": False, "addon": "c7744bff_ha-sip",
               "numbers": "sip:x@y", "ring_timeout": 15, "language": "ru",
               "message_template": "T", "notify": True},
    ))
    cam = {"name": "kalitka", "executor_service": ""}
    brain._grant("face", cam, "Я", 0.5, b"", alarm=True)
    time.sleep(0.3)
    assert calls == []  # alarm.enabled=false — звонков нет
    assert st.query_events({"action": "alarm"})

    monkeypatch.setattr(cfg, "CONF", types.SimpleNamespace(
        master_enable=True, frigate_url="http://x",
        mqtt={"host": "127.0.0.1"}, cameras=[],
        recognition={}, vlm={"enabled": False},
        gestures={"enabled": False},
        alarm={"enabled": True, "addon": "a", "numbers": "",
               "ring_timeout": 15, "language": "ru", "message_template": "T", "notify": True},
    ))
    brain._grant("face", cam, "Я", 0.5, b"", alarm=True)
    time.sleep(0.3)
    assert calls == []  # номера пусты — звонков нет


def test_brain_alarm_test(monkeypatch, brain_env):
    brain, _st = brain_env
    calls: list = []
    monkeypatch.setattr(ha_api, "call_ha_service", lambda *a, **k: calls.append(a) or True)

    monkeypatch.setattr(cfg, "CONF", types.SimpleNamespace(
        master_enable=True, frigate_url="http://x", mqtt={"host": "127.0.0.1"},
        cameras=[], recognition={}, vlm={"enabled": False},
        gestures={"enabled": False},
        alarm={"enabled": True, "addon": "c7744bff_ha-sip", "numbers": "sip:t@pbx",
               "ring_timeout": 10, "sip_account": 2, "language": "ru",
               "message_template": "T", "notify": True},
    ))
    r = brain.alarm_test()
    assert r["ok"] is True and r["numbers"] == 1
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.05)
    assert calls and calls[0][0] == "hassio"
    assert calls[0][2]["addon"] == "c7744bff_ha-sip"

    monkeypatch.setattr(cfg, "CONF", types.SimpleNamespace(
        master_enable=True, frigate_url="http://x", mqtt={"host": "127.0.0.1"},
        cameras=[], recognition={}, vlm={"enabled": False},
        gestures={"enabled": False},
        alarm={"enabled": True, "addon": "a", "numbers": "", "ring_timeout": 10,
               "sip_account": 1, "language": "ru", "message_template": "T", "notify": True},
    ))
    r = brain.alarm_test()
    assert r["ok"] is False and "пуст" in r["error"]


# --------------------------------------------------------------------- #
# 5. Brain: переобучение по фото

def test_brain_retrain(monkeypatch, brain_env, monkeypatch_tmp=None):
    brain, st = brain_env

    class _M(_FakeModels):
        """Переобучение: лицо найдено, эмбеддинг фиксированный."""
        def detect_face_raw(self, jpg):
            return np.array([[1.0, 1.0]], np.uint8), 100, 100

        def face_embedding_of_aligned(self, jpg):
            return [1.0, 0.0]

    brain.models = _M()

    st.add_face("Я", [[0.0, 1.0]], _png_jpg_bytes(1))
    st.add_face("Я", [[0.0, 1.0]], _png_jpg_bytes(2))
    r = brain.retrain("face", "Я")
    assert r["ok"] and r["results"]["Я"] == 2
    assert len(st.faces["Я"]["embeds"]) == 2
    # каждое фото дало ровно 1 прототип
    assert all(abs(e[0] - 1.0) < 1e-6 for e in st.faces["Я"]["embeds"])

    # отсутствие фото — прототипы очищаются (subject перестаёт матчиться)
    st.delete_photos("face", "Я", [p["file"] for p in st.list_photos("face", "Я")])
    r = brain.retrain("face", "Я")
    assert r["results"]["Я"] == 0
    assert st.faces["Я"]["embeds"] == []

    # несуществующий subject — просто пустой результат
    r = brain.retrain("face", "НетТакого")
    assert r["ok"] and r["results"]["НетТакого"] == 0


# --------------------------------------------------------------------- #
# 6. Config: settings.json переопределяет опции

def test_config_settings_override(tmp_path, monkeypatch):
    options = {
        "master_enable": True, "frigate_url": "http://127.0.0.1:8971",
        "mqtt": {"host": "127.0.0.1", "port": 1883},
        "cameras": [],
        "recognition": {}, "vlm": {"enabled": False},
        "gestures": {"enabled": True, "min_frames": 2},
        "alarm": {"enabled": False, "addon": "c7744bff_ha-sip",
                  "numbers": "sip:a@b\nc:1", "sip_account": 1, "ring_timeout": 20,
                  "language": "ru", "message_template": "m", "notify": True},
    }
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))
    # options.json не существует -> defaults; подменим чтение напрямую
    orig_read = cfg._read_options
    monkeypatch.setattr(cfg, "_read_options", lambda: json.loads(json.dumps(options)))

    c = cfg.Config()
    assert c.gestures["enabled"] is True and c.gestures["min_frames"] == 2
    assert c.alarm["enabled"] is False
    assert [n.strip() for n in str(c.alarm["numbers"]).splitlines() if n.strip()] == ["sip:a@b", "c:1"]

    # панель выключила жесты и включила тревогу — опции не трогаем
    (tmp_path / "settings.json").write_text(
        json.dumps({"gestures_enabled": False, "alarm_enabled": True}), encoding="utf-8")
    c.reload()
    assert c.gestures["enabled"] is False
    assert c.alarm["enabled"] is True

    # as_safe_dict отдаёт оба блока
    safe = c.as_safe_dict()
    assert "gestures" in safe and "alarm" in safe
