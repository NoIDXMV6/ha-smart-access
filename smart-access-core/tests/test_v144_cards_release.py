"""Тесты v1.4.4 — «на гите нет обновлённых карточек» + аудит релиза.

Что проверяем (регресс на реальные жалобы и находки аудита):

1. Карточки в examples/cards/ — ОБНОВЛЁННЫЕ: играют сущности моста
   (camera.smart_access_bridge_*), а не Универсальную камеру (источник
   жалоб «карточка серая/без картинки» в v1.1.0).
2. Кнопки «Обучить» в карточках вызывают smart_access.enroll с ЯВНЫМ kind
   и непустым subject, entity_id — сущность моста (снимок = /snapshot/<имя>).
3. В ЗНАЧЕНИЯХ карточек нет ссылок /mjpeg/… — HA не понимает multipart как
   stream_source (допустимо только в комментариях, они не парсятся).
4. release.yml: пакует smart-access-cards.zip (карточки в релизе),
   чистит __pycache__/*.pyc из интеграционного zip (в v1.4.3 они уезжали
   в релиз), прикладывает оба архива к Release.
5. README: все ссылки examples/cards/*.yaml ведут на существующие файлы;
   путь аддона в разделе «Файлы» — smart-access-core/ (не addon/…).
6. Версии аддона/интеграции синхронны и зафиксированы: 1.4.4.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
CARDS = REPO / "examples" / "cards"
CORE = REPO / "smart-access-core"

# (файл, ожидаемая сущность моста) — реальные камеры пользователя
BRIDGE_CARDS = {
    "amvideo_govorovo.yaml": "camera.smart_access_bridge_prud",
    "egds_lesnaia.yaml": "camera.smart_access_bridge_tatyanin_park",
    "gate_hikvision.yaml": "camera.smart_access_bridge_kalitka",
}


def _load(name: str) -> dict:
    return yaml.safe_load((CARDS / name).read_text(encoding="utf-8"))


def _walk_strings(node) -> list[str]:
    """Все строковые ЗНАЧЕНИЯ карточки (комментарии в YAML не парсятся)."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        out: list[str] = []
        for k, v in node.items():
            out += _walk_strings(k)
            out += _walk_strings(v)
        return out
    if isinstance(node, list):
        return [s for item in node for s in _walk_strings(item)]
    return []


# --- 1. Карточки играют мост ---------------------------------------------------

def test_cards_dir_exists_with_updated_cards():
    files = {p.name for p in CARDS.glob("*.yaml")}
    assert {"amvideo_govorovo.yaml", "egds_lesnaia.yaml",
            "gate_hikvision.yaml", "smart_access_bridge_cameras.yaml"} <= files


def test_legacy_v11_cards_removed():
    leftovers = [p.name for p in CARDS.glob("*_v11*.yaml")]
    assert not leftovers, f"устаревшие карточки v1.1.0 не удалены: {leftovers}"


@pytest.mark.parametrize("fname,bridge", sorted(BRIDGE_CARDS.items()))
def test_card_plays_bridge_entity(fname: str, bridge: str):
    card = _load(fname)
    assert card.get("camera_image") == bridge, (
        f"{fname}: camera_image должен быть сущностью моста {bridge}, "
        "не Универсальной камерой"
    )
    assert card.get("camera_view") == "live"
    # кнопка «во весь экран» тоже смотрит на сущность моста
    more_info = [el for el in card.get("elements", [])
                 if el.get("tap_action", {}).get("action") == "more-info"]
    assert more_info and all(
        el["tap_action"].get("entity") == bridge for el in more_info)


# --- 2. Кнопки «Обучить» -------------------------------------------------------

@pytest.mark.parametrize("fname,bridge", sorted(BRIDGE_CARDS.items()))
def test_card_enroll_button(fname: str, bridge: str):
    card = _load(fname)
    enrolls = [
        el for el in card.get("elements", [])
        if el.get("tap_action", {}).get("perform_action") == "smart_access.enroll"
    ]
    assert len(enrolls) == 1, f"{fname}: ровно одна кнопка «Обучить»"
    data = enrolls[0]["tap_action"]["data"]
    assert data["entity_id"] == bridge, (
        f"{fname}: enroll должен снимать кадр с сущности моста {bridge} "
        "(async_get_image → /snapshot/<имя> моста)"
    )
    assert data["kind"] in ("car", "face")
    assert str(data["subject"]).strip()
    if fname == "gate_hikvision.yaml":
        assert data["kind"] == "face"  # калитка обучает лица


def test_gate_card_keeps_door_relay():
    card = _load("gate_hikvision.yaml")
    opens = [
        el for el in card.get("elements", [])
        if el.get("tap_action", {}).get("perform_action") == "switch.turn_on"
    ]
    assert opens, "карточка калитки потеряла кнопку «Открыть» (switch.turn_on)"
    assert opens[0]["tap_action"]["target"]["entity_id"] == "switch.door_door_relay_0"


def test_barrier_cards_keep_open_buttons():
    for fname, service in (("amvideo_govorovo.yaml", "amvideo.open_barrier"),
                           ("egds_lesnaia.yaml", "egds.open_barrier")):
        card = _load(fname)
        opens = [el for el in card.get("elements", [])
                 if el.get("tap_action", {}).get("perform_action") == service]
        assert opens, f"{fname}: потеряна кнопка «Открыть» ({service})"


# --- 3. Никаких /mjpeg/… в значениях карточек ---------------------------------

@pytest.mark.parametrize("fname", sorted(BRIDGE_CARDS) + ["smart_access_bridge_cameras.yaml"])
def test_no_mjpeg_in_card_values(fname: str):
    strings = _walk_strings(_load(fname))
    bad = [s for s in strings if "/mjpeg/" in s]
    assert not bad, (
        f"{fname}: /mjpeg/… в значениях карточки запрещено — HA открывает "
        f"stream_source как RTSP/видео и не понимает multipart: {bad}"
    )


# --- 4. Релиз: карточки в архиве, без __pycache__ -------------------------------

def test_release_workflow_packs_cards_zip():
    wf = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "smart-access-cards.zip" in wf, "карточки не пакуются в релиз"
    assert "Pack cards zip" in wf
    # оба архива прикладываются к Release одним вызовом gh release create
    wf_idx = wf.index("gh release create")
    create_block = wf[wf_idx:]
    assert "smart-access-hacs.zip" in create_block
    assert "smart-access-cards.zip" in create_block


def test_release_workflow_excludes_pycache():
    wf = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "__pycache__" in wf, "release.yml не чистит __pycache__ перед упаковкой"
    assert "*.py[co]" in wf or "*.pyc" in wf


def test_release_workflow_is_valid_yaml():
    d = yaml.safe_load(
        (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"))
    steps = [s["name"] for s in d["jobs"]["release"]["steps"]]
    assert "Pack integration zip" in steps and "Pack cards zip" in steps


def test_no_tracked_pycache_in_repo():
    import subprocess
    out = subprocess.run(
        ["git", "ls-files", "custom_components", "smart-access-core/app"],
        cwd=REPO, capture_output=True, text=True).stdout
    pyc = [l for l in out.splitlines() if "__pycache__" in l or l.endswith((".pyc", ".pyo"))]
    assert not pyc, f"в git закоммичены скомпилированные файлы: {pyc}"


# --- 5. README ------------------------------------------------------------------

def test_readme_card_references_exist():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    refs = set(re.findall(r"examples/cards/([A-Za-z0-9_.-]+\.yaml)", readme))
    assert refs, "README не ссылается на карточки"
    for name in refs:
        assert (CARDS / name).exists(), f"README ведёт на несуществующую карточку {name}"


def test_readme_files_section_addon_path():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "addon/smart-access-core/" not in readme, (
        "в README остался несуществующий путь addon/smart-access-core/"
    )
    assert "`smart-access-core/` — аддон" in readme


# --- 6. Версии ------------------------------------------------------------------

def test_versions_are_in_sync_v144():
    addon = json.loads((CORE / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((REPO / "custom_components" / "smart_access" /
                           "manifest.json").read_text(encoding="utf-8"))
    assert addon["version"] == manifest["version"] == "1.4.4"


def test_changelog_has_v144_section():
    text = (CORE / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.startswith("# 1.4.4"), "CHANGELOG должен начинаться с секции 1.4.4"
    # release.yml извлекает секцию по заголовку «# 1.4.4» — он обязан существовать
    assert "# 1.4.4" in text
    assert "smart-access-cards.zip" in text
