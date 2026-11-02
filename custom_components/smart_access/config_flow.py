"""UI-настройки интеграции (Конфигурация → Интеграции → Smart Access → Настройки)."""
from __future__ import annotations

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback

from .const import (
    CONF_BRIDGE_URL,
    CONF_CAR_SUBJECTS,
    CONF_ENROLL_MAP,
    CONF_FACE_SUBJECTS,
    DEFAULT_BRIDGE_URL,
    DOMAIN,
)


class SmartAccessConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Простое подключение: одна запись на дом (данные приходят по MQTT)."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        if self._async_current_entries():
            return self.async_abort(reason="already_configured")
        if user_input is not None:
            return self.async_create_entry(title="Smart Access", data={})
        return self.async_show_form(step_id="user")

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return SmartAccessOptionsFlow(config_entry)


class SmartAccessOptionsFlow(config_entries.OptionsFlow):
    """Имена «своих» и привязка камер: <camera_entity>=<car|face>:<subject>."""

    def __init__(self, config_entry) -> None:
        self.entry = config_entry

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        opts = self.entry.options
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_CAR_SUBJECTS,
                        default=opts.get(CONF_CAR_SUBJECTS, "Моя машина, Машина жены"),
                    ): str,
                    vol.Required(
                        CONF_FACE_SUBJECTS,
                        default=opts.get(CONF_FACE_SUBJECTS, "Я, Жена, Сын, Дочь"),
                    ): str,
                    vol.Required(
                        CONF_ENROLL_MAP,
                        description={"suggested_value": opts.get(CONF_ENROLL_MAP, "")},
                    ): str,
                    # v1.4.0: адрес моста для сущностей camera.smart_access_*
                    vol.Optional(
                        CONF_BRIDGE_URL,
                        description={
                            "suggested_value": opts.get(CONF_BRIDGE_URL, DEFAULT_BRIDGE_URL)
                        },
                    ): str,
                }
            ),
        )
