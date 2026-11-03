"""v1.4.2: «Не удалось сохранить: value must be at most 0.95. Got {...}».

Рецидив того же класса бага, что и в v1.3.1 («value must be at most 3.0»):
в схеме опций аддона (config.json -> "schema") у порогов распознавания
оставались ТЕСНЫЕ ВЕРХНИЕ ГРАНИЦЫ:

    recognition.car_threshold  float(0.5,0.95)   <- max 0.95
    gestures.threshold         float(0.5,0.95)   <- max 0.95
    recognition.face_threshold float(0.2,0.6)    <- max 0.6
    recognition.car_gray_zone  float(0.01,0.2)   <- max 0.2

Supervisor превращает их в vol.Range(max=...), и ОДНО значение выше максимума
(пользователь поднял порог схожести машины до 0.96–1.0, чтобы отсечь ложные
срабатывания) блокировало сохранение ВСЕЙ конфигурации — диалог не давал
сохранить ничего, включая добавленные камеры.

Фикс 1.4.2 (закрывает класс целиком):
  1. В схеме Supervisor'а пороги БЕЗ границ ("float?") — сохранение не может
     быть заблокировано значением порога. ring_timeout int(5,)->int(1,).
  2. Рабочие диапазоны поддерживает ПРИЛОЖЕНИЕ (app/config.py:
     normalize_thresholds): значение > 1 трактуется как проценты
     («76» -> 0.76), затем клэмп в рабочий диапазон + WARNING в журнал.

Тесты: (а) эмуляция валидации Supervisor'а (порт из
test_v131_options_schema.py) на конфиге с «залетевшими» порогами;
(б) документирование первопричины на старой схеме; (в) нормализация
порогов приложением; (г) МЕТА-регресс: в схеме не осталось НИ ОДНОГО
числового ограничения СВЕРХУ — класс «value must be at most …» не может
вернуться; (д) Config.reload() нормализует пороги из options.json.

Запуск:  python3 -m pytest smart-access-core/tests/test_v142_options_schema.py -v
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
from voluptuous import Invalid

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

from test_v131_options_schema import (  # noqa: E402
    RE_SCHEMA_ELEMENT,
    _user_payload,
    validate_app_options,
)

CONFIG_JSON = json.loads((CORE / "config.json").read_text(encoding="utf-8"))
SCHEMA = CONFIG_JSON["schema"]


# --- 1. Реальный сценарий: порог выше 0.95 больше не блокирует сохранение ---

def _user_payload_high_thresholds() -> dict:
    """Конфиг пользователя, дополненный правдоподобной причиной ошибки:
    порог машины поднят выше максимума старой схемы (0.96), порог жестов —
    на максимум (1.0)."""
    payload = _user_payload()
    payload["recognition"]["car_threshold"] = 0.96
    payload["gestures"]["threshold"] = 1.0
    return payload


def test_high_thresholds_pass_supervisor_validation():
    """Диалог «Конфигурация» сохраняет конфиг с car_threshold=0.96 и
    gestures.threshold=1.0 (раньше: «value must be at most 0.95»)."""
    validated = validate_app_options(SCHEMA, _user_payload_high_thresholds())
    assert validated["recognition"]["car_threshold"] == 0.96
    assert validated["gestures"]["threshold"] == 1.0


def test_old_schema_rejected_same_payload_with_at_most_0_95():
    """Документирует первопричину: на схеме 1.4.1 тот же конфиг падает ровно с
    той ошибкой, которую видел пользователь."""
    old_schema = json.loads(json.dumps(SCHEMA))
    old_schema["recognition"]["car_threshold"] = "float(0.5,0.95)?"
    with pytest.raises(Invalid) as exc:
        validate_app_options(old_schema, _user_payload_high_thresholds())
    assert "value must be at most 0.95" in str(exc.value)


def test_thresholds_have_no_bounds_in_schema():
    for sec, key in (("recognition", "car_threshold"),
                     ("recognition", "car_gray_zone"),
                     ("recognition", "face_threshold"),
                     ("gestures", "threshold")):
        assert SCHEMA[sec][key] == "float?", f"{sec}.{key} должен быть без границ"


def test_ring_timeout_accepts_immediate_call():
    """ring_timeout разжат с int(5,) до int(1,): «звонок без ожидания» —
    легитимное значение, не должно блокировать сохранение."""
    validate_app_options({"alarm": SCHEMA["alarm"]}, {"alarm": {"ring_timeout": 1}})


def test_defaults_still_pass_schema():
    validated = validate_app_options(SCHEMA, CONFIG_JSON["options"])
    assert validated["recognition"]["car_threshold"] == 0.76
    assert validated["recognition"]["face_threshold"] == 0.363
    assert validated["gestures"]["threshold"] == 0.80


# --- 2. МЕТА-регресс: у числовых опций нет верхних границ в схеме ------------

def _numeric_leaves(schema, path="schema"):
    """Все int/float-листья схемы: (путь, тип, min, max)."""
    out = []
    if isinstance(schema, dict):
        for k, v in schema.items():
            out += _numeric_leaves(v, f"{path}.{k}")
    elif isinstance(schema, list):
        out += _numeric_leaves(schema[0], f"{path}[]")
    elif isinstance(schema, str):
        m = RE_SCHEMA_ELEMENT.match(schema)
        if m and schema.startswith(("int", "float")):
            lo = m.group("i_min") or m.group("f_min")
            hi = m.group("i_max") or m.group("f_max")
            out.append((path, schema, lo, hi))
    return out


def test_no_numeric_option_has_upper_bound():
    """Класс бага «value must be at most N» не может вернуться: НИ У ОДНОЙ
    int/float-опции в схеме нет конечного максимума. Любые легитимные значения,
    сохранённые прошлыми версиями, проходят валидацию Supervisor'а."""
    leaves = _numeric_leaves(SCHEMA)
    assert leaves, "тест должен реально обходить схему"
    for path, typ, lo, hi in leaves:
        assert hi is None, f"{path} ({typ}): верхняя граница запрещена — " \
                           f"она блокирует сохранение всего конфига (баг v1.3.1/v1.4.2)"
        if lo is not None:
            assert float(lo) <= 1.0, f"{path} ({typ}): нижняя граница {lo} > 1 " \
                                     f"может блокировать сохранение легаси-значений"


# --- 3. Нормализация порогов приложением (app/config.py) ---------------------

from app.config import normalize_thresholds  # noqa: E402


def _sections(**overrides) -> dict:
    sec: dict = {
        "recognition": {
            "car_threshold": 0.76, "car_gray_zone": 0.06, "face_threshold": 0.363,
        },
        "gestures": {"threshold": 0.80},
    }
    for dotted, value in overrides.items():
        s, k = dotted.split("__")
        sec[s][k] = value
    return sec


def test_normalize_keeps_defaults_untouched():
    sec = _sections()
    normalize_thresholds(sec)
    assert sec["recognition"]["car_threshold"] == 0.76
    assert sec["recognition"]["car_gray_zone"] == 0.06
    assert sec["recognition"]["face_threshold"] == 0.363
    assert sec["gestures"]["threshold"] == 0.80


def test_normalize_clamps_above_documented_band():
    sec = _sections(recognition__car_threshold=0.96, gestures__threshold=1.0,
                    recognition__face_threshold=0.7)
    normalize_thresholds(sec)
    assert sec["recognition"]["car_threshold"] == 0.95
    assert sec["gestures"]["threshold"] == 0.95
    assert sec["recognition"]["face_threshold"] == 0.6


def test_normalize_clamps_below_documented_band():
    sec = _sections(recognition__face_threshold=0.05, recognition__car_gray_zone=0.0)
    normalize_thresholds(sec)
    assert sec["recognition"]["face_threshold"] == 0.2
    assert sec["recognition"]["car_gray_zone"] == 0.01


def test_normalize_interprets_percent_values():
    """«76» без ограничений схемы теперь сохранится, а приложение поймёт это
    как 76 % -> 0.76 (а не как бессмысленный порог 76)."""
    sec = _sections(recognition__car_threshold=76, gestures__threshold=90)
    normalize_thresholds(sec)
    assert sec["recognition"]["car_threshold"] == 0.76
    assert sec["gestures"]["threshold"] == 0.9


def test_normalize_non_numeric_uses_band_minimum():
    sec = _sections(recognition__car_threshold="abc")
    normalize_thresholds(sec)
    assert sec["recognition"]["car_threshold"] == 0.5


def test_normalize_missing_sections_and_keys_are_safe():
    normalize_thresholds({})
    normalize_thresholds({"recognition": {}})
    normalize_thresholds({"recognition": "не словарь"})
    sec = {"gestures": {"enabled": False, "min_frames": 1}}  # без threshold
    normalize_thresholds(sec)
    assert "threshold" not in sec["gestures"]


def test_normalize_logs_warning_on_fix(caplog):
    """Effective-значение не меняется молча: каждая правка порога — WARNING."""
    sec = _sections(recognition__car_threshold=0.99)
    with caplog.at_level(logging.WARNING, logger="smart_access.config"):
        normalize_thresholds(sec)
    assert any("car_threshold" in r.message and "0.95" in r.message
               for r in caplog.records), \
        f"WARNING о приведении порога не найден: {[r.message for r in caplog.records]}"


def test_normalize_percent_change_also_warns(caplog):
    sec = _sections(gestures__threshold=90)
    with caplog.at_level(logging.WARNING, logger="smart_access.config"):
        normalize_thresholds(sec)
    assert any("threshold" in r.message and "процент" in r.message.lower()
               for r in caplog.records)


# --- 4. Config.reload() нормализует пороги из options.json -------------------

def test_config_reload_normalizes_thresholds(monkeypatch):
    """Интеграционно: конфиг с «залетевшими» порогами (car_threshold=0.96,
    gestures.threshold=90) после reload даёт рабочие значения."""
    import app.config as cfg_mod

    user_options = {
        "master_enable": True,
        "frigate_url": "http://ccab4aaf-frigate-fa:5000",
        "mqtt": {"host": "localhost", "port": 1883, "user": "mqtt", "password": "mqtt"},
        "cameras": [],
        "recognition": {"car_threshold": 0.96, "face_threshold": 0.363},
        "vlm": {"enabled": False, "base_url": "", "api_key": "", "model": "",
                "mode": "gray_zone", "timeout_sec": 6},
        "gestures": {"enabled": True, "min_frames": 1, "threshold": 90},
        "alarm": {"enabled": False},
    }
    monkeypatch.setattr(cfg_mod, "_read_options", lambda: user_options)
    conf = cfg_mod.Config()
    assert conf.recognition["car_threshold"] == 0.95
    assert conf.recognition["face_threshold"] == 0.363
    assert conf.gestures["threshold"] == 0.9


def test_versions_are_in_sync():
    manifest = json.loads(
        (CORE.parent / "custom_components" / "smart_access" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["version"] == CONFIG_JSON["version"]
    assert tuple(int(x) for x in CONFIG_JSON["version"].split(".")[:2]) >= (1, 4)
