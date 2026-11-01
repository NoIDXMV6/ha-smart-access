"""Опциональный VLM-контролёр: любой OpenAI-совместимый endpoint (url+token+модель).

Подходит Ollama (http://<pc>:11434/v1), LM Studio, vLLM, OpenAI-совместимые шлюзы.
НЕ обязателен: при недоступности/ошибках решение остаётся за локальными моделями.
"""
from __future__ import annotations

import base64
import logging
import time

import requests

from . import config as cfg

LOG = logging.getLogger("smart_access.vlm")

_last_fail = 0.0  # circuit breaker: после сбоя пауза 60 с


def _breaker_open() -> bool:
    return time.time() - _last_fail < 60


def _encode(jpg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")


def _ask(images: list[bytes], prompt: str) -> str | None:
    global _last_fail
    from .config import CONF

    vlm = CONF.vlm
    if not vlm.get("enabled") or not vlm.get("base_url") or not vlm.get("model"):
        return None
    if _breaker_open():
        return None
    url = vlm["base_url"].rstrip("/") + "/chat/completions"
    content: list[dict] = [{"type": "text", "text": prompt}]
    for jpg in images:
        content.append({"type": "image_url", "image_url": {"url": _encode(jpg)}})
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {vlm.get('api_key') or 'none'}"},
            json={
                "model": vlm["model"],
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
            timeout=float(vlm.get("timeout_sec") or 6),
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return str(text).strip().upper()
    except Exception as exc:  # noqa: BLE001
        _last_fail = time.time()
        LOG.warning("VLM недоступен (%s) — продолжаю только по локальному решению", exc)
        return None


def verify_same_car(reference: bytes | None, candidate: bytes) -> bool | None:
    """True = VLM подтвердил совпадение, False = отверг, None = VLM не помог."""
    if reference is None:
        return None
    answer = _ask(
        [reference, candidate],
        "Image 1 is a reference photo of a specific car. "
        "Image 2 is a new frame from the same camera. "
        "Is image 2 the SAME car as in image 1 (same make, model, color, plates area)? "
        "Ignore lighting and season differences. Answer strictly YES or NO.",
    )
    if answer is None:
        return None
    if answer.startswith("YES"):
        return True
    if answer.startswith("NO"):
        return False
    return None


def verify_same_face(reference: bytes | None, candidate: bytes) -> bool | None:
    if reference is None:
        return None
    answer = _ask(
        [reference, candidate],
        "Image 1 is a reference photo of a specific person. "
        "Image 2 is a new frame from the door camera. "
        "Is image 2 the SAME person as in image 1? Answer strictly YES or NO.",
    )
    if answer is None:
        return None
    if answer.startswith("YES"):
        return True
    if answer.startswith("NO"):
        return False
    return None
