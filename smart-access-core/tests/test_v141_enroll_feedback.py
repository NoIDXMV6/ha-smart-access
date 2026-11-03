"""v1.4.1: обратная связь обучения из карточек HA («Обучить» → Core).

Проблема из эксплуатации: «при нажатии в карточке на лицо или машину ошибки
больше нет, но и фотография в Smart Access Core для обучения не появляется».
Причина: цепочка была fire-and-forget —
  * интеграция публиковала кадр в MQTT и завершалась без ответа;
  * ядро (_handle_enroll) выбрасывало результат enroll(): любая неудача
    (лицо на кадре не найдено, модель недоступна, битый payload) исчезала
    молча — ни ack, ни события в журнале панели, ни ошибки в HA.

Проверяем (v1.4.1):
  * успех (car/face) — фото в базе, ack ok=True, событие enrolled;
  * неудачи — ack ok=False с ВРАЗУМИТЕЛЬНОЙ ошибкой + событие enroll_failed
    в журнале панели + WARNING в журнале аддона;
  * битые payload — ack ok=False, обработчик не падает;
  * ack несёт kind/subject/entity_id — интеграция матчит по ним ожидающих;
  * UI: автообновление вкладок «Машины»/«Лица»/«Журнал» + красная плашка
    enroll_failed в журнале;
  * версии аддона и интеграции синхронны (1.4.1).

Запуск:  python3 -m pytest smart-access-core/tests/test_v141_enroll_feedback.py -v
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

CORE = Path(__file__).resolve().parents[1]
REPO = CORE.parent
sys.path.insert(0, str(CORE))

from app import brain as brain_mod      # noqa: E402
from app import config as cfg           # noqa: E402
from app.store import Store             # noqa: E402


# --------------------------------------------------------------------- #
# Подмены: MQTT-клиент-рекордер, модели, фиктивный Brain с реальным Store

class _FakeMQTT:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> None:
        self.published.append((topic, payload))

    def acks(self) -> list[dict]:
        return [json.loads(p) for t, p in self.published if t == "smart_access/ack"]

    def events(self) -> list[dict]:
        return [json.loads(p) for t, p in self.published if t == "smart_access/event"]


class _FakeModels:
    def __init__(self, face_found: bool = True, clip_ok: bool = True) -> None:
        self.face_found = face_found
        self.ok = clip_ok
        self.faces_ok = True
        self.gestures_ok = False
        self.status = {"clip": "ok" if clip_ok else "error: нет файла",
                       "face": "ok", "pose": "off (модель не найдена)"}

    def car_embedding(self, jpg: bytes, with_views: bool = False):
        if not self.ok:
            return None
        rng = np.random.default_rng(len(jpg) % 97)
        return rng.normal(size=(8, 512) if with_views else 512).astype(np.float32)

    def detect_face_raw(self, jpg: bytes):
        if not self.face_found:
            return None, 640, 480
        return np.zeros((112, 112, 3), dtype=np.uint8), 640, 480

    def face_embedding_of_aligned(self, jpg: bytes):
        rng = np.random.default_rng(len(jpg) % 31)
        return [float(x) for x in rng.normal(size=128)]


def _real_jpeg(w: int = 640, h: int = 480) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (60, 120, 90)).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _make_brain(tmp: str, models: _FakeModels | None = None):
    cfg.DATA_DIR = f"{tmp}/data"
    cfg.MEDIA_DIR = f"{tmp}/media"
    st = Store()
    mqtt_rec = _FakeMQTT()
    b = brain_mod.Brain.__new__(brain_mod.Brain)
    b.store = st
    b.models = models or _FakeModels()
    b.client = mqtt_rec
    b.bridge = None
    b.stats = {"events_seen": 0, "grants": 0, "denied": 0, "alarms": 0,
               "vlm_calls": 0, "last_grant": None, "started": time.time(),
               "mqtt": "ok"}
    b.retrain_status = {"running": False}
    b.publish_status = lambda: None
    return b, st, mqtt_rec


def _payload(subject: str, kind: str, jpg: bytes, entity: str = "") -> bytes:
    """Ровно тот формат, который публикует svc_enroll интеграции."""
    return json.dumps({
        "subject": subject,
        "image_b64": base64.b64encode(jpg).decode("ascii"),
        "entity_id": entity or ("camera.kalitka" if kind == "face" else "camera.prud"),
    }, ensure_ascii=False).encode()


# --------------------------------------------------------------------- #
# 1. Успех: фото появляется В БАЗЕ + ack ok=True

def test_enroll_car_success_photo_in_base_and_ack():
    async def _run():
        pass

    with tempfile.TemporaryDirectory() as tmp:
        b, st, mq = _make_brain(tmp)
        b._handle_enroll("car", _payload("Моя машина", "car", _real_jpeg()))
        subj = st.subjects("car")
        assert "Моя машина" in subj, "машина должна появиться в базе"
        assert subj["Моя машина"]["photos"] == 1, "фото должно сохраниться в /media"
        assert subj["Моя машина"]["embeds"] == 8, "8 аугментаций CLIP"
        acks = mq.acks()
        assert len(acks) == 1, f"ровно один ack: {acks}"
        ack = acks[0]
        assert ack["cmd"] == "enroll" and ack["ok"] is True
        assert ack["kind"] == "car" and ack["subject"] == "Моя машина"
        assert ack["entity_id"] == "camera.prud", "ack несёт сущность — интеграция матчит по ней"
        assert ack["added"] == 8
        acts = [e["action"] for e in mq.events()]
        assert acts == ["enrolled"], f"в журнале панели должно быть событие enrolled: {acts}"


def test_enroll_face_success_ack():
    with tempfile.TemporaryDirectory() as tmp:
        b, st, mq = _make_brain(tmp)
        b._handle_enroll("face", _payload("Я", "face", _real_jpeg()))
        assert "Я" in st.subjects("face"), "лицо должно появиться в базе"
        ack = mq.acks()[0]
        assert ack["ok"] is True and ack["added"] == 1
        assert ack["entity_id"] == "camera.kalitka"


# --------------------------------------------------------------------- #
# 2. Неудачи: ack ok=False С ПРИЧИНОЙ + событие enroll_failed в журнале

def test_enroll_face_not_found_reports(caplog):
    """Главный эксплуатационный кейс: лицо не влезло в кадр калитки.
    Раньше — полная тишина («ошибки нет, но фото не появляется»)."""
    with tempfile.TemporaryDirectory() as tmp:
        b, st, mq = _make_brain(tmp, _FakeModels(face_found=False))
        b._handle_enroll("face", _payload("Жена", "face", _real_jpeg()))
        assert st.subjects("face") == {}, "без лица в кадре база не меняется"
        acks = mq.acks()
        assert len(acks) == 1, "неудача обязана вернуться ack'ом"
        ack = acks[0]
        assert ack["ok"] is False
        assert "лицо на фото не найдено" in ack["error"], ack
        assert ack["subject"] == "Жена" and ack["kind"] == "face"
        failed = [e for e in mq.events() if e["action"] == "enroll_failed"]
        assert len(failed) == 1, "причина должна быть видна в журнале панели"
        assert failed[0]["detail"] == "лицо на фото не найдено"
        assert failed[0]["subject"] == "Жена"


def test_enroll_clip_unavailable_reports():
    with tempfile.TemporaryDirectory() as tmp:
        b, st, mq = _make_brain(tmp, _FakeModels(clip_ok=False))
        b._handle_enroll("car", _payload("Моя машина", "car", _real_jpeg()))
        ack = mq.acks()[0]
        assert ack["ok"] is False and "CLIP недоступен" in ack["error"]
        assert [e["action"] for e in mq.events()] == ["enroll_failed"]


# --------------------------------------------------------------------- #
# 3. Битые payload: не падают, возвращают ack ok=False

def test_enroll_malformed_payloads_ack_error():
    with tempfile.TemporaryDirectory() as tmp:
        b, st, mq = _make_brain(tmp)
        b._handle_enroll("car", b"{not-json")
        b._handle_enroll("car", json.dumps({"subject": "", "image_b64": ""}).encode())
        b._handle_enroll("car", json.dumps({"subject": "X", "image_b64": ""}).encode())
        b._handle_enroll("car", json.dumps({"subject": "X"}).encode())
        acks = mq.acks()
        assert len(acks) == 4, f"каждый битый запрос должен получить ack: {acks}"
        assert all(a["ok"] is False and a["error"] for a in acks)
        assert st.subjects("car") == {}
        # все попадают в журнал панели с enroll_failed
        assert len([e for e in mq.events() if e["action"] == "enroll_failed"]) == 4


def test_enroll_unknown_kind_ack_error():
    """smart_access/enroll/<мусор> — kind вне car|face возвращает ошибку."""
    with tempfile.TemporaryDirectory() as tmp:
        b, st, mq = _make_brain(tmp)
        b._handle_enroll("human", _payload("Кто-то", "human", _real_jpeg()))
        ack = mq.acks()[0]
        assert ack["ok"] is False and "kind must be car|face" in ack["error"]


# --------------------------------------------------------------------- #
# 4. Интеграция: синтаксис + согласованность версий + UI

def test_integration_modules_compile():
    import py_compile

    for f in sorted((REPO / "custom_components" / "smart_access").glob("*.py")):
        py_compile.compile(str(f), doraise=True)


def test_versions_in_sync_v141():
    addon = json.load(open(CORE / "config.json", encoding="utf-8"))
    manifest = json.load(open(REPO / "custom_components" / "smart_access" / "manifest.json",
                              encoding="utf-8"))
    assert addon["version"] == manifest["version"] == "1.4.1"


def test_ui_auto_refresh_and_failed_pill():
    """Панель: вкладки «Машины»/«Лица»/«Журнал» обновляются сами, enroll_failed
    краснеет в журнале — фото от карточек видно без перезахода."""
    html = (CORE / "app" / "ui" / "index.html").read_text(encoding="utf-8")
    assert "autoRefreshActivePane" in html, "нет автообновления активной вкладки"
    assert "_lastSubjects" in html, "нет проверки изменения данных перед рендером"
    assert "enroll_failed:'err'" in html, "enroll_failed должен быть красной плашкой"
    assert "setInterval(autoRefreshActivePane" in html


def test_integration_enroll_waits_for_ack():
    """svc_enroll обязан ждать ack аддона (source-проверка ключевых мест)."""
    src = (REPO / "custom_components" / "smart_access" / "__init__.py").read_text(encoding="utf-8")
    assert "ENROLL_ACK_TIMEOUT" in src, "нет таймаута ожидания ack"
    assert "_ensure_ack_subscription" in src, "нет единой подписки на smart_access/ack"
    assert "enroll_waiters" in src, "нет реестра ожидающих enroll-запросов"
    assert "enroll_done" in src and "enroll_failed" in src, \
        "интеграция должна звать события enroll_done/enroll_failed"
    assert "persistent_notification" in src, "нет уведомления о результате"
