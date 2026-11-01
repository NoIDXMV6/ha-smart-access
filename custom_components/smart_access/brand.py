"""Бренд-иконка Smart Access (щит с замочной скважиной) — без авторизации,
для entity_picture и Lovelace-карточек. С HA 2026.3 файл brand/icon.png
также показывается в «Настройки → Устройства и сервисы»."""
from __future__ import annotations

from importlib import resources

from aiohttp import web
from homeassistant.components.http import HomeAssistantView

from .const import DOMAIN

_BRAND_URL = f"/api/{DOMAIN}/brand/icon.png"


def brand_icon_url() -> str:
    """URL иконки интеграции."""
    return _BRAND_URL


class BrandIconView(HomeAssistantView):
    """PNG с иконкой интеграции (без авторизации, только картинка)."""

    url = _BRAND_URL
    name = f"api:{DOMAIN}:brand_icon"
    requires_auth = False

    def __init__(self) -> None:
        self._body: bytes | None = None

    def _load(self) -> bytes:
        if self._body is None:
            self._body = (
                resources.files(__package__) / "brand" / "icon.png"
            ).read_bytes()
        return self._body

    async def get(self, request: web.Request) -> web.Response:
        return web.Response(
            body=self._load(),
            content_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )
