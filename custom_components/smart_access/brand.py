"""Бренд-иконка Smart Access (щит с замочной скважиной) — без авторизации,
для entity_picture и Lovelace-карточек. С HA 2026.3 файл brand/icon.png
также показывается в «Настройки → Устройства и сервисы».

v1.4.6 — чтение PNG вынесено в executor (ревью HACS на родственные
интеграции egds-ha/amvideo-ha: блокирующий I/O в async view недопустим —
первый запрос иконки подвешивал event loop HA на время чтения файла).
"""
from __future__ import annotations

from importlib import resources

from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant

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

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._body: bytes | None = None

    def _load_sync(self) -> bytes:
        """Блокирующее чтение PNG — выполняется только в executor."""
        return (
            resources.files(__package__) / "brand" / "icon.png"
        ).read_bytes()

    async def get(self, request: web.Request) -> web.Response:
        # Чтение файла — блокирующий I/O, поэтому один раз выполняем
        # его в executor, а дальше отдаём кэшированные байты.
        if self._body is None:
            self._body = await self.hass.async_add_executor_job(self._load_sync)
        return web.Response(
            body=self._body,
            content_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )
