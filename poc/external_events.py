from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from poc.entities import Side, SourceState

if TYPE_CHECKING:
    from poc.game_state import GameState


class EventType(str, Enum):
    SET_SOURCE_AVAILABLE = "set_source_available"
    SET_SOURCE_EMPTY = "set_source_empty"
    SET_SOURCE_DISTURBED = "set_source_disturbed"
    SET_THERMOMETER_DONE_BLUE = "set_thermometer_done_blue"
    SET_THERMOMETER_DONE_YELLOW = "set_thermometer_done_yellow"
    NOTE = "note"


@dataclass(slots=True, order=True)
class ExternalEvent:
    time: float
    event_type: EventType
    target_id: int | None = None
    origin: str = "scripted"
    note: str = ""
    payload: dict[str, float | int | str | bool] = field(default_factory=dict)


def apply_external_event(state: "GameState", event: ExternalEvent) -> str:
    if event.event_type is EventType.NOTE:
        return event.note or "note"

    if event.event_type in {
        EventType.SET_SOURCE_AVAILABLE,
        EventType.SET_SOURCE_EMPTY,
        EventType.SET_SOURCE_DISTURBED,
    }:
        if event.target_id is None or event.target_id not in state.sources:
            return f"ignored source event {event.event_type.value}"
        source = state.sources[event.target_id]
        items = int(event.payload.get("available_items", source.available_items))
        source.available_items = items
        source.available_from_t = event.time
        if event.event_type is EventType.SET_SOURCE_AVAILABLE:
            source.state = SourceState.UNTOUCHED if items > 0 else SourceState.EMPTY
        elif event.event_type is EventType.SET_SOURCE_EMPTY:
            source.state = SourceState.EMPTY
            source.available_items = 0
        else:
            source.state = SourceState.DISTURBED if items > 0 else SourceState.EMPTY
        return event.note or f"{event.event_type.value}:{event.target_id}"

    if event.event_type is EventType.SET_THERMOMETER_DONE_BLUE:
        state.thermometer.mark_done_for_side(Side.BLUE)
        return event.note or "thermometer_done_blue"

    state.thermometer.mark_done_for_side(Side.YELLOW)
    return event.note or "thermometer_done_yellow"
