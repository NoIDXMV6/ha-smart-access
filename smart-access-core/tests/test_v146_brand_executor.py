"""Тесты v1.4.6 — бренд-иконка через executor (ревью HACS egds-ha/amvideo-ha).

Ревью HACS (frenck) на родственные интеграции egds-ha и amvideo-ha
потребовало: блокирующий I/O в async view недопустим. В ha-smart-access
тот же код жил в custom_components/smart_access/brand.py — первый запрос
иконки делал read_bytes() прямо в event loop HA.

Что проверяем:

1. brand.py: BrandIconView принимает hass в конструкторе, блокирующее
   чтение выделено в _load_sync() и вызывается через
   hass.async_add_executor_job; в async get нет прямого read_bytes.
2. __init__.py: регистрация — BrandIconView(hass) (не пустой конструктор).
3. В интеграции не осталось других мест с блокирующим read_bytes.
4. camera.py: предупреждение «/api/status не содержит камер» подставляет
   адрес моста (%s с аргументом), а не печатает литеральный {}.
5. Версии аддона/интеграции синхронны: 1.4.6; CHANGELOG начинается с
   секции версии из manifest (release.yml извлекает заметки по заголовку
   «# X.Y.Z»); README содержит «Что нового в v1.4.6».
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "smart-access-core"
INTEGRATION = REPO / "custom_components" / "smart_access"

BRAND_PY = (INTEGRATION / "brand.py").read_text(encoding="utf-8")
INIT_PY = (INTEGRATION / "__init__.py").read_text(encoding="utf-8")
CAMERA_PY = (INTEGRATION / "camera.py").read_text(encoding="utf-8")


# --- 1. brand.py: executor вместо блокирующего чтения ---------------------------

def test_brand_view_takes_hass():
    assert re.search(r"def __init__\(self, hass:\s*HomeAssistant\)", BRAND_PY), (
        "BrandIconView должен принимать hass в конструкторе — executor "
        "берётся из hass.async_add_executor_job"
    )
    assert "self.hass = hass" in BRAND_PY


def test_brand_read_runs_in_executor():
    # блокирующее чтение выделено в отдельный sync-метод
    assert "def _load_sync(self) -> bytes:" in BRAND_PY, (
        "блокирующее чтение PNG должно быть выделено в _load_sync()"
    )
    # ...и вызывается из async get через executor
    assert re.search(
        r"self\._body\s*=\s*await\s+self\.hass\.async_add_executor_job\(\s*self\._load_sync\s*\)",
        BRAND_PY,
    ), "async get обязан читать PNG через hass.async_add_executor_job"
    # в async get нет прямого read_bytes
    get_body = BRAND_PY.split("async def get")[1]
    assert "read_bytes()" not in get_body, (
        "async get не должен читать файл синхронно (блокировка event loop)"
    )


def test_no_other_blocking_reads_in_integration():
    # единственное допустимое read_bytes — в _load_sync() brand.py
    for name in ("__init__.py", "camera.py", "sensor.py", "config_flow.py", "const.py"):
        src = (INTEGRATION / name).read_text(encoding="utf-8")
        assert "read_bytes()" not in src, (
            f"{name}: блокирующий read_bytes в коде интеграции недопустим"
        )


# --- 2. Регистрация view --------------------------------------------------------

def test_brand_view_registered_with_hass():
    assert "BrandIconView(hass)" in INIT_PY, (
        "BrandIconView регистрируется с hass: hass.http.register_view(BrandIconView(hass))"
    )
    assert "register_view(BrandIconView())" not in INIT_PY


# --- 3. camera.py: косметика журнала ---------------------------------------------

def test_probe_warning_includes_bridge():
    # раньше: "... (bridge: {})" без аргумента — в журнале печатался литеральный {}
    assert "(bridge: {})" not in CAMERA_PY, (
        "литеральный {} без аргумента в LOG.warning — адрес моста должен "
        "подставляться в сообщение"
    )
    assert re.search(r"\(bridge: %s\)", CAMERA_PY)


# --- 4. Версии и заметки релиза ---------------------------------------------------

def test_versions_are_in_sync_v146():
    manifest = json.loads((INTEGRATION / "manifest.json").read_text(encoding="utf-8"))
    addon = json.loads((CORE / "config.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.4.6", manifest["version"]
    assert addon["version"] == "1.4.6", addon["version"]


def test_changelog_starts_with_manifest_version_section():
    manifest = json.loads((INTEGRATION / "manifest.json").read_text(encoding="utf-8"))
    head = (CORE / "CHANGELOG.md").read_text(encoding="utf-8").lstrip()
    assert head.startswith(f"# {manifest['version']}"), (
        "CHANGELOG должен начинаться с секции версии из manifest "
        "(release.yml извлекает заметки по заголовку «# X.Y.Z»)"
    )


def test_readme_has_v146_section():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "## Что нового в v1.4.6" in readme
