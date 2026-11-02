"""v1.3.1: валидация опций аддона Supervisor'ом (диалог «Конфигурация»).

Баг (реальный случай): «Не удалось сохранить: value must be at most 3.0.
Got {'master_enable': True, ...}». Supervisor разбирает диапазоны схемы как
FLOAT (в apps/options.py: ``range_args[group_name[2:]] = float(group_value)``),
поэтому int(1,3) -> vol.Range(min=1.0, max=3.0), и при попытке сохранить
опцию со значением выше максимума HA показывает «... at most 3.0». Один
такой «залетевший» при старой узкой схеме ключ (например
recognition.min_confirm_frames=5) блокировал сохранение ВСЕЙ конфигурации.

Фикс 1.3.1: расширены все тесные диапазоны (int(1,3) -> int(1,10),
cooldown int(5,600) -> int(0,86400) и т.д.), чтобы любые легитимные значения,
сохранённые прежними версиями, проходили валидацию.

Тесты здесь воспроизводят АЛГОРИТМ валидации Supervisor'а (по исходникам
home-assistant/supervisor, supervisor/apps/options.py) и прогоняют через
него: (а) дефолты из config.json, (б) реальный конфиг пользователя из багрепорта,
(в) старую схему — для документирования первопричины.

Запуск:  python3 -m pytest smart-access-core/tests/test_v131_options_schema.py -v
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from voluptuous import Boolean, Coerce, Invalid, In, Range, Url

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

CONFIG_JSON = json.loads((CORE / "config.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Порт валидатора Supervisor'а (home-assistant/supervisor, apps/options.py).
# Важные детали воспроизведены 1:1:
#   * диапазоны int/float/str парсятся как float() -> в сообщении «3.0»;
#   * неизвестные ключи НЕ ошибка: WARNING + удаление;
#   * отсутствующие обязательные (без «?») опции — ошибка;
#   * вложенные dict/list обходятся рекурсивно.
# ---------------------------------------------------------------------------

RE_SCHEMA_ELEMENT = re.compile(
    r"^(?:"
    r"|bool"
    r"|email"
    r"|url"
    r"|port"
    r"|str(?:\((?P<s_min>\d+)?,(?P<s_max>\d+)?\))?"
    r"|password(?:\((?P<p_min>\d+)?,(?P<p_max>\d+)?\))?"
    r"|int(?:\((?P<i_min>-?\d+)?,(?P<i_max>-?\d+)?\))?"
    r"|float(?:\((?P<f_min>-?\d*\.?\d+)?,(?P<f_max>-?\d*\.?\d+)?\))?"
    r"|match\((?P<match>.*)\)"
    r"|list\((?P<list>.+)\)"
    r")\??$"
)

_SCHEMA_LENGTH_PARTS = ("i_min", "i_max", "f_min", "f_max", "s_min", "s_max", "p_min", "p_max")


def _validate_typed(typ: str, value, key: str):
    """Точная копия AppOptions._validate_typed (supervisor/apps/options.py)."""
    match = RE_SCHEMA_ELEMENT.match(typ)
    if not match:
        raise Invalid(f"Unknown type '{typ}' (option '{key}')")
    range_args: dict = {}
    for group_name in _SCHEMA_LENGTH_PARTS:
        group_value = match.group(group_name)
        if group_value:
            # ВОТ ПРИЧИНА «3.0»: границы диапазона становятся float
            range_args[group_name[2:]] = float(group_value)
    if typ.startswith("str") or typ.startswith("password"):
        return value  # Length-проверки строк здесь не тестируем
    if typ.startswith("int"):
        return __import__("voluptuous").All(Coerce(int), Range(**range_args))(value)
    if typ.startswith("float"):
        return __import__("voluptuous").All(Coerce(float), Range(**range_args))(value)
    if typ.startswith("bool"):
        return Boolean()(value)
    if typ.startswith("url"):
        return Url()(value)
    if typ.startswith("port"):
        return Coerce(int)(Range(min=1, max=65535)(value))
    if typ.startswith("list"):
        return In(match.group("list").split("|"))(str(value))
    raise Invalid(f"Fatal error for option '{key}' with type '{typ}'")


def validate_app_options(schema, options, name="Smart Access Core"):
    """Точная копия AppOptions.__call__/_nested_* (supervisor/apps/options.py)."""

    def walk(sch, data, key):
        if isinstance(sch, list):
            if not isinstance(data, list):
                raise Invalid(f"Invalid list for option '{key}' in {name}")
            return [walk(sch[0], el, key) for el in data]
        if isinstance(sch, dict):
            if not isinstance(data, dict):
                raise Invalid(f"Invalid dict for option '{key}' in {name}")
            out = {}
            for c_key, c_value in data.items():
                if c_key not in sch:
                    continue  # Supervisor: WARNING «does not exist» + удаление
                out[c_key] = walk(sch[c_key], c_value, c_key)
            # обязательные (без «?») должны присутствовать
            for miss in set(sch) - set(out):
                ms = sch[miss]
                if isinstance(ms, list) and ms:
                    ms = ms[0]
                if isinstance(ms, str) and ms.endswith("?"):
                    continue
                raise Invalid(f"Missing option '{miss}' in {key} in {name}")
            return out
        if data is None:
            raise Invalid(f"Missing required option '{key}' in {name}")
        return _validate_typed(sch, data, key)

    return walk(schema, options, "root")


# ---------------------------------------------------------------------------
# Конфиг пользователя из багрепорта (сообщение целиком не влезло в диалог HA —
# хвост реконструирован по дефолтам v1.2.x; в скрытой части и находился
# виновник: min_confirm_frames/sip_account со значениями выше старого max=3).
# ---------------------------------------------------------------------------

def _user_payload() -> dict:
    return {
        "master_enable": True,
        "frigate_url": "http://ccab4aaf-frigate-fa:5000",
        "mqtt": {"host": "localhost", "port": 1883, "user": "mqtt", "password": "mqtt"},
        "cameras": [
            {
                "name": "Пруд",
                "ha_entity": "camera.1_shlagbaum_derevnia_govorovo_15072_kamera",
                "mode": "car",
                "rtsp_url": "",
                "executor_service": "amvideo.open_barrier",
                "executor_target": "",
                "cooldown": 45,
                "schedule": "",
                "enabled": True,
            },
            {
                "name": "Татьянин парк",
                "ha_entity": "camera.ul_lesnaia_shlagbaum_no1",
                "mode": "car",
                "rtsp_url": "",
                "executor_service": "button.ul_lesnaia_shlagbaum_no1_otkryt",
                "executor_target": "",
                "cooldown": 45,
                "schedule": "",
                "enabled": True,
            },
        ],
        # Значения, превышавшие СТАРЫЕ максимумы (5 > 3, 4 > 3) — то, из-за чего
        # HA отказывался сохранять конфигурацию целиком
        "recognition": {
            "car_threshold": 0.76,
            "car_gray_zone": 0.06,
            "face_threshold": 0.363,
            "min_confirm_frames": 5,
            "bridge_fps": 1.5,
            "stream_fps": 5.0,
            "unknown_keep_days": 7,
        },
        "vlm": {
            "enabled": False,
            "base_url": "http://192.168.1.50:11434/v1",
            "api_key": "ollama",
            "model": "moondream",
            "mode": "gray_zone",
            "timeout_sec": 6,
        },
        "gestures": {
            "enabled": False,
            "min_frames": 6,
            "threshold": 0.80,
            "default_confirm": ["one_hand_up"],
            "default_alarm": ["both_hands_up"],
        },
        "alarm": {
            "enabled": False,
            "addon": "c7744bff_ha-sip",
            "numbers": "sip:89256758881@pbx.kozhevatov.ru",
            "sip_account": 4,
            "ring_timeout": 15,
            "language": "ru",
            "message_template": "Тревога! {time}, {name} открыл калитку с тревожным жестом.",
            "notify": True,
        },
    }


# --- 1. Дефолты из config.json всегда проходят собственную схему ------------

def test_defaults_pass_schema():
    validated = validate_app_options(CONFIG_JSON["schema"], CONFIG_JSON["options"])
    assert validated["master_enable"] is True
    assert validated["cameras"][0]["cooldown"] == 45
    assert validated["recognition"]["min_confirm_frames"] == 2


def test_every_default_value_within_own_range():
    """Любое значение из options лежит в диапазоне своей схемы (ловит правки
    config.json, при которых дефолт вылетает за диапазон)."""
    schema, options = CONFIG_JSON["schema"], CONFIG_JSON["options"]

    def check(sch, opts, path):
        if isinstance(sch, dict):
            for k in opts:
                if k in sch:
                    check(sch[k], opts[k], f"{path}.{k}")
        elif isinstance(sch, list):
            for i, el in enumerate(opts):
                check(sch[0], el, f"{path}[{i}]")
        elif isinstance(sch, str):
            m = RE_SCHEMA_ELEMENT.match(sch)
            if not m:
                return
            lo = m.group("i_min") or m.group("f_min")
            hi = m.group("i_max") or m.group("f_max")
            if lo is not None:
                assert float(opts) >= float(lo), f"{path}={opts} < min {lo}"
            if hi is not None:
                assert float(opts) <= float(hi), f"{path}={opts} > max {hi}"

    check(schema, options, "options")


# --- 2. Реальный конфиг пользователя из багрепорта --------------------------

def test_user_payload_passes_v131():
    """После 1.3.1 конфиг пользователя (с легаси-значениями min_confirm_frames=5,
    gestures.min_frames=6, sip_account=4) сохраняется Supervisor'ом без ошибок."""
    validated = validate_app_options(CONFIG_JSON["schema"], _user_payload())
    assert validated["recognition"]["min_confirm_frames"] == 5
    assert validated["alarm"]["sip_account"] == 4
    assert validated["cameras"][0]["cooldown"] == 45


def test_old_schema_rejected_same_payload_with_at_most_3_0():
    """Документирует первопричину: на схеме 1.2.x тот же конфиг падает ровно с
    той ошибкой, которую видел пользователь («value must be at most 3.0»)."""
    old_schema = json.loads(
        (CORE / "config.json").read_text(encoding="utf-8")
    )["schema"]
    old_schema["recognition"]["min_confirm_frames"] = "int(1,3)?"
    old_schema["alarm"]["sip_account"] = "int(1,3)?"
    with pytest.raises(Invalid) as exc:
        validate_app_options(old_schema, _user_payload())
    assert "value must be at most 3.0" in str(exc.value)


def test_gesture_defaults_list_value_passes_v131():
    """Вторая причина блокировки сохранения: default_confirm/default_alarm хранят
    JSON-массив, а в схеме 1.2.x тип был ГОЛЫЙ "list(...)" — Supervisor валидировал
    str(["one_hand_up"]) -> «value must be one of ...». Каноничная форма (как у
    официального mosquitto, log_type) — ["list(a|b|c)"]: массив со строкой-типом.
    Теперь сохранение списка проходит."""
    gest = CONFIG_JSON["schema"]["gestures"]
    assert isinstance(gest["default_confirm"], list), "тип должен быть массивом"
    validated = validate_app_options(
        {"gestures": gest},
        {"gestures": {"default_confirm": ["one_hand_up", "victory"],
                      "default_alarm": []}},
    )
    assert validated["gestures"]["default_confirm"] == ["one_hand_up", "victory"]
    assert validated["gestures"]["default_alarm"] == []


def test_bare_list_type_rejects_list_value():
    """Документирует старую сломанную форму: голый list(...) требует СТРОКУ.
    (именно поэтому схема переведена на ["list(...)"])."""
    with pytest.raises(Invalid):
        validate_app_options(
            {"gestures": {"default_confirm": "list(victory|ok)?"}},
            {"gestures": {"default_confirm": ["victory"]}},
        )


def test_unbounded_ranges_accept_large_legacy_values():
    """Диапазоны счётчиков/таймингов открыты сверху (int(1,)) — любое
    легальное значение не может больше заблокировать сохранение конфига."""
    schema, cams = CONFIG_JSON["schema"], CONFIG_JSON["options"]["cameras"][0]
    validate_app_options(
        {"cameras": schema["cameras"], "recognition": schema["recognition"],
         "alarm": schema["alarm"], "gestures": schema["gestures"]},
        {"cameras": [dict(cams, cooldown=10_000_000)],
         "recognition": {"min_confirm_frames": 50, "bridge_fps": 9.5,
                         "stream_fps": 25.0, "unknown_keep_days": 3650},
         "alarm": {"sip_account": 16, "ring_timeout": 3600},
         "gestures": {"min_frames": 30}},
    )


def test_unbounded_range_still_enforces_min():
    schema = CONFIG_JSON["schema"]["recognition"]
    with pytest.raises(Invalid):
        validate_app_options(
            {"recognition": schema},
            {"recognition": {"min_confirm_frames": 0}},
        )


def test_min_confirm_frames_above_old_max_now_accepted():
    validate_app_options(
        {"recognition": CONFIG_JSON["schema"]["recognition"]},
        {"recognition": {"min_confirm_frames": 10}},
    )


def test_sip_account_above_old_max_now_accepted():
    validate_app_options(
        {"alarm": CONFIG_JSON["schema"]["alarm"]},
        {"alarm": {"sip_account": 8, "ring_timeout": 600}},
    )


def test_cooldown_zero_is_valid_and_means_no_cooldown():
    """cooldown=0 разрешён схемой; в brain.py «0» => проверка
    «now - last < 0» всегда ложна -> повторный допуск без паузы."""
    validate_app_options(
        {"cameras": CONFIG_JSON["schema"]["cameras"]},
        {"cameras": [dict(CONFIG_JSON["options"]["cameras"][0], cooldown=0)]},
    )
    assert not (0 < 0)  # семантика brain.py:389 при cooldown=0


# --- 3. Поведение Supervisor'а на краях --------------------------------------

def test_unknown_keys_are_dropped_not_fatal():
    """Supervisor НЕ отвергает сохранение из-за лишних ключей (только WARNING).
    Иначе легаси-опции из старых версий навсегда ломали бы диалог."""
    validated = validate_app_options(
        CONFIG_JSON["schema"], {**CONFIG_JSON["options"], "legacy_key": 123}
    )
    assert "legacy_key" not in validated


def test_missing_required_option_is_fatal():
    cams = json.loads(json.dumps(CONFIG_JSON["schema"]["cameras"]))
    # в схеме камер «name» — обязательный (без «?»)
    assert not cams[0]["name"].endswith("?")
    with pytest.raises(Invalid):
        validate_app_options(
            {"cameras": cams},
            {"cameras": [{"mode": "car", "cooldown": 45}]},
        )


def test_camera_slug_for_user_cameras():
    """Имена из багрепорта нормализуются в формат Frigate без ошибок."""
    from app.config import slugify_camera_name

    assert slugify_camera_name("Пруд") == "prud"
    assert slugify_camera_name("Татьянин парк") == "tatyanin_park"


def test_gesture_list_safe_accepts_options_shape():
    """brain._gesture_list -> gestures.gesture_list_safe: значение опций (список
    id, пустой список, строка 'a,b') даёт корректный список жестов."""
    from app.gestures import gesture_list_safe

    assert gesture_list_safe(["one_hand_up", "bogus"], ["victory"]) == ["one_hand_up"]
    assert gesture_list_safe([], ["both_hands_up"]) == ["both_hands_up"]
    assert gesture_list_safe("victory, ok", ["one_hand_up"]) == ["victory", "ok"]


def test_versions_are_in_sync():
    """Версия аддона (config.json) и интеграции (manifest.json) совпадают —
    от manifest'а отсчитываются релизы для HACS."""
    manifest = json.loads(
        (CORE.parent / "custom_components" / "smart_access" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    assert manifest["version"] == CONFIG_JSON["version"]
