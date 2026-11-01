"""Сенсоры Smart Access: статус ядра и последнее событие допуска."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorDeviceClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN, SIGNAL_EVENT, SIGNAL_STATUS

LOG = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    async_add_entities([SmartAccessStatusSensor(hass), SmartAccessLastEventSensor(hass)])


class _Base(SensorEntity):
    _attr_should_poll = False

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._payload: dict[str, Any] = {}

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(self.hass, self._signal, self._on_update)
        )

    @callback
    def _on_update(self, payload: dict[str, Any]) -> None:
        self._payload = payload or {}
        self.async_write_ha_state()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return dict(self._payload)


class SmartAccessStatusSensor(_Base):
    _attr_name = "Smart Access статус"
    _attr_unique_id = "smart_access_status"
    _attr_icon = "mdi:shield-car"

    @property
    def _signal(self):
        return SIGNAL_STATUS

    @property
    def native_value(self) -> str:
        return str(self._payload.get("state", "offline"))

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        attrs = dict(self._payload)
        stats = attrs.pop("stats", None)
        if stats:
            attrs["grants"] = stats.get("grants")
            attrs["events_seen"] = stats.get("events_seen")
        return attrs


class SmartAccessLastEventSensor(_Base):
    _attr_name = "Smart Access последнее событие"
    _attr_unique_id = "smart_access_last_event"
    _attr_icon = "mdi:car-door"

    @property
    def _signal(self):
        return SIGNAL_EVENT

    @property
    def native_value(self) -> str:
        p = self._payload
        if not p:
            return "нет событий"
        subject = p.get("subject") or p.get("detail") or ""
        return f"{p.get('kind','?')}/{p.get('action','?')}: {subject}"
