"""v1.4.3: самовосстановление деградирующих сессий потока HA (мост).

Наблюдение на реальном HAOS (prud/tatyanin_park/kalitka): свежая сессия
потока HA даёт 1.3–1.5 fps, но через ~10 минут та же ЖИВАЯ сессия
опускается до 0.1 fps (кадр раз в 7–11 с). При этом:
  * «сбоев: 0» — сторож «молчит 30 с» не срабатывает (байты формально идут);
  * запасной снимок не включается (нет 3 подряд сбоев);
  * подсказка панели уверяет, что это «нормальный режим»;
  * во Frigate — застывшая статическая картинка (мост тиражирует
    устаревший кадр), детекция фактически мертва.

Фикс v1.4.3: мост узнаёт рабочий темп камеры по успешным сессиям
(baseline, автокалибруется) и пересоздаёт соединение, как только темп
живой сессии падает ниже max(FLOOR, 0.35 * baseline). Пересоздание —
НЕ сбой: «сбоев» не растёт, бэкофф не включается, есть антиштормовая
пауза recycle_min и отдельный счётчик пересозданий; подсказка панели
честно называет деградацию деградацией.

Тесты гоняют РЕАЛЬНЫЙ код _stream_once/_stream_thread с фейковым
HTTP-ответом HA (управляемый темп кадров) на коротких временах
(warmup/silence/recycle_min переопределяются как атрибуты экземпляра).

Запуск:  python3 -m pytest smart-access-core/tests/test_v143_bridge_stream_watchdog.py -v
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import sys
CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

from app import ha_api  # noqa: E402
from app.bridge import LOW_FPS_MARK, Bridge, _placeholder_jpeg  # noqa: E402

JPEG = _placeholder_jpeg()  # валидный JPEG (заглушка моста)


class FakeResp:
    """Фейковый multipart-ответ HA: frames кадров с паузой gap, опционально
    финальная тишина tail_silence перед последним куском."""

    def __init__(self, frames: int = 12, gap: float = 0.2,
                 tail_silence: float = 0.0):
        self.frames = frames
        self.gap = gap
        self.tail_silence = tail_silence
        self.closed = False

    def iter_content(self, chunk_size: int = 65536):
        for _ in range(self.frames):
            time.sleep(self.gap)
            yield JPEG
        if self.tail_silence:
            # «хвост» без маркеров SOI — как реальный мусор/пауза в multipart
            time.sleep(self.tail_silence)
            yield b"\x00" * 16

    def close(self) -> None:
        self.closed = True


def _bridge(**kw) -> Bridge:
    """Мост с короткими временами и одним клиентом камеры «prud»."""
    b = Bridge(fps=5.0, stream_fps=5.0)
    b.set_cameras([{"name": "prud", "ha_entity": "camera.x", "rtsp_url": ""}])
    b.clients["prud"] = 1
    # короткие времена для тестов (в бою: warmup 25 c, silence 25 c, min 45 c)
    b.stream_warmup = kw.get("warmup", 1.0)
    b.stream_silence = kw.get("silence", 5.0)
    b.recycle_min = kw.get("recycle_min", 0.5)
    return b


def _serve(monkeypatch, resp: FakeResp) -> None:
    monkeypatch.setattr(ha_api, "open_camera_stream", lambda entity: resp)


# --- 1. Здоровая сессия: кадры принимаются, пересозданий нет -----------------

def test_healthy_session_not_recycled(monkeypatch):
    b = _bridge()
    b.baseline_fps["prud"] = 1.0        # порог = max(0.15, 0.35*1.0) = 0.35
    _serve(monkeypatch, FakeResp(frames=12, gap=0.2))  # темп ~5 fps >> 0.35
    got, err = b._stream_once("prud", "camera.x", threading.Event())
    assert got and err == ""
    assert b.recycles["prud"] == 0
    assert b.failures["prud"] == 0
    assert b.frames["prud"] == 12


def test_healthy_session_calibrates_baseline_up(monkeypatch):
    """Успешная быстрая сессия поднимает baseline (автокалибровка вверх)."""
    b = _bridge()
    b.baseline_fps["prud"] = 1.0
    _serve(monkeypatch, FakeResp(frames=14, gap=0.2))  # ~5 fps
    b._stream_once("prud", "camera.x", threading.Event())
    assert b.baseline_fps["prud"] > 1.5, \
        f"baseline должен вырасти к реальному темпу, а не {b.baseline_fps['prud']}"


# --- 2. Деградировавшая сессия: помечается и пересоздаётся -------------------

def test_degraded_session_marked_low_fps(monkeypatch):
    """Живая сессия с темпом ~5 fps при baseline 50 (порог 17.5) ->
    LOW_FPS_MARK, соединение пересоздаётся. Это ровно реальный случай
    «1.4 fps после старта -> 0.1 fps через 10 минут при сбоев: 0»."""
    b = _bridge()
    b.baseline_fps["prud"] = 50.0
    _serve(monkeypatch, FakeResp(frames=14, gap=0.2))
    got, err = b._stream_once("prud", "camera.x", threading.Event())
    assert got, "сессия жива — кадры были"
    assert err.startswith(LOW_FPS_MARK)


def test_recycle_does_not_count_as_failure(monkeypatch):
    """Пересоздание по деградации НЕ растит «сбоев» и не включает бэкофф —
    панель не должна показывать камеру сломанной, когда она просто
    пересоздала соединение."""
    b = _bridge()
    b.recycle_pause = 0.05
    calls = {"n": 0}

    def fake_once(name, entity, stop):
        calls["n"] += 1
        return True, f"{LOW_FPS_MARK} 0.10 < 0.49 (baseline 1.40)"

    monkeypatch.setattr(b, "_stream_once", fake_once)
    stop = threading.Event()
    th = threading.Thread(target=b._stream_thread, args=("prud", "camera.x", stop),
                          daemon=True)
    th.start()
    deadline = time.time() + 5
    while b.recycles["prud"] < 3 and time.time() < deadline:
        time.sleep(0.01)
    stop.set()
    th.join(2)
    assert calls["n"] >= 3
    assert b.recycles["prud"] == 3
    assert b.failures["prud"] == 0, "пересоздание сессии — не сбой"


def test_baseline_decays_toward_degraded_rate_on_recycle():
    """Если источник стал медленнее навсегда, baseline при пересозданиях
    спадает к наблюдаемому темпу — пересоздания прекращаются, шторма нет."""
    b = _bridge()
    b.baseline_fps["prud"] = 10.0
    b.session_rate["prud"] = 2.0
    b._handle_low_fps("prud", f"{LOW_FPS_MARK} 0.10 < 3.50 (baseline 10.00)")
    assert b.recycles["prud"] == 1
    assert b.baseline_fps["prud"] == round(0.8 * 10.0 + 0.2 * 2.0, 3)  # 8.4
    assert b.baseline_fps["prud"] < 10.0


def test_recycle_storm_is_throttled():
    """Второе пересоздание раньше recycle_min откладывается (антишторм)."""
    b = _bridge(recycle_min=2.0)
    b.last_recycle_ts["prud"] = time.time()  # только что пересоздавали
    pause = b._handle_low_fps("prud", f"{LOW_FPS_MARK} 0.10 < 0.49")
    assert 1.5 <= pause <= 2.0, f"пауза должна покрывать остаток recycle_min, got {pause}"
    # а вне окна антишторма — обычная короткая пауза
    b2 = _bridge()
    assert b2._handle_low_fps("prud", LOW_FPS_MARK) == b2.recycle_pause


def test_baseline_untouched_when_never_established():
    """Без baseline (первая сессия камеры) пересоздание не выдумывает темп."""
    b = _bridge()
    b.session_rate["prud"] = 0.3
    b._handle_low_fps("prud", LOW_FPS_MARK)
    assert b.baseline_fps["prud"] == 0.0
    assert b.recycles["prud"] == 1


# --- 3. Сторож тишины --------------------------------------------------------

def test_silence_watchdog_fires(monkeypatch):
    """Кадры кончились, но байты/соединение формально живы: сессия
    обрывается по stream_silence (в бою — 25 с)."""
    b = _bridge(silence=1.0)
    _serve(monkeypatch, FakeResp(frames=1, gap=0.1, tail_silence=1.6))
    got, err = b._stream_once("prud", "camera.x", threading.Event())
    assert got
    assert "молчит" in err


# --- 4. Недоступность потока (регресс v1.4.0/v1.3.0) ------------------------

def test_unavailable_stream_reports_error(monkeypatch):
    b = _bridge()
    monkeypatch.setattr(ha_api, "open_camera_stream", lambda entity: None)
    got, err = b._stream_once("prud", "camera.x", threading.Event())
    assert not got and err == "поток HA недоступен"


def test_real_jpeg_frames_parsed_from_stream(monkeypatch):
    """Регресс парсера: полный JPEG вырезается из потока, latest обновлён."""
    b = _bridge()
    _serve(monkeypatch, FakeResp(frames=3, gap=0.05))
    got, err = b._stream_once("prud", "camera.x", threading.Event())
    assert got and err == ""
    assert b.latest["prud"] == JPEG
    assert b.mode["prud"] == "stream"


# --- 5. Панель: деградация видна честно --------------------------------------

def _degraded_bridge() -> Bridge:
    b = _bridge()
    b.mode["prud"] = "stream"
    b.fps["prud"] = 0.10
    b.baseline_fps["prud"] = 1.40
    b.recycles["prud"] = 3
    return b


def test_hint_reports_degradation():
    hint = _degraded_bridge().hint("prud")
    assert "деградирует" in hint
    assert "0.10" in hint and "1.40" in hint
    assert "пересоздаёт соединение (3 раз)" in hint


def test_hint_normal_mode_when_no_recycles():
    b = _degraded_bridge()
    b.recycles["prud"] = 0
    b.fps["prud"] = 1.40
    hint = b.hint("prud")
    assert "нормальный режим" in hint


def test_hint_healthy_stream_unchanged():
    b = _bridge()
    b.mode["prud"] = "stream"
    b.fps["prud"] = 1.42
    assert "нормальный режим" in b.hint("prud")


# --- 6. Телеметрия и жизненный цикл ------------------------------------------

def test_set_cameras_initializes_recycle_state():
    b = Bridge(fps=1.5, stream_fps=5.0)
    b.set_cameras([{"name": "prud", "ha_entity": "camera.x"},
                   {"name": "kalitka", "ha_entity": "camera.y"}])
    for name in ("prud", "kalitka"):
        assert b.recycles[name] == 0
        assert b.baseline_fps[name] == 0.0
        assert b.session_rate[name] == 0.0
        assert b.last_recycle_ts[name] == 0.0
    b.set_cameras([{"name": "prud", "ha_entity": "camera.x"}])
    assert "kalitka" not in b.recycles and "kalitka" not in b.baseline_fps
    assert b.recycles["prud"] == 0


def test_session_rate_tracked(monkeypatch):
    b = _bridge()
    _serve(monkeypatch, FakeResp(frames=12, gap=0.2))
    b._stream_once("prud", "camera.x", threading.Event())
    assert b.session_rate["prud"] > 1.0


# --- 7. Версии ----------------------------------------------------------------

def test_versions_are_in_sync_v143():
    """Версии аддона и интеграции синхронны (без пина к конкретной — точная
    версия фиксируется в тесте соответствующего релиза, см. test_v144)."""
    import json
    addon = json.loads((CORE / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((CORE.parent / "custom_components" / "smart_access" /
                           "manifest.json").read_text(encoding="utf-8"))
    assert addon["version"] == manifest["version"]
    assert tuple(int(x) for x in addon["version"].split(".")[:2]) >= (1, 4)
