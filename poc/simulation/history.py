from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Callable

from poc.domain.entities import Mars, Side
from poc.domain.game_state import GameState


@dataclass(slots=True)
class HistoryEntry:
    time: float
    our_position: tuple[float, float]
    enemy_position: tuple[float, float]
    our_score: int
    enemy_score: int
    our_load: int
    enemy_load: int
    source_states: dict[int, dict[str, object]]
    deposit_states: dict[int, dict[str, object]]
    thermometer_state: str
    thermometer_doing_blue: bool
    thermometer_doing_yellow: bool
    mars_states: dict[str, list[dict[str, object]]]


def plain_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: plain_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(plain_value(key)): plain_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    return value


def build_history_entry(
    state: GameState,
    *,
    mars_position: Callable[[Mars, float], tuple[float, float]],
    mars_has_arrived: Callable[[Mars, float], bool],
    mars_collided: Callable[[Mars, Side | None], bool],
    stopped_mars_names: set[str],
) -> HistoryEntry:
    return HistoryEntry(
        time=round(state.t, 3),
        our_position=state.our_robot.position,
        enemy_position=state.enemy_robot.position,
        our_score=state.score_for_side(state.our_side),
        enemy_score=state.score_for_side(state.enemy_side),
        our_load=state.our_robot.load,
        enemy_load=state.enemy_robot.load,
        source_states={
            source_id: {
                "state": plain_value(source.state),
                "available_items": source.available_items,
                "map_footprint_enabled": source.map_footprint_enabled,
            }
            for source_id, source in state.sources.items()
        },
        deposit_states={
            deposit_id: {
                "blue_items": deposit.blue_items,
                "yellow_items": deposit.yellow_items,
                "map_footprint_enabled": deposit.map_footprint_enabled,
                "push_state": plain_value(deposit.push_state),
                "pushed_owner": plain_value(deposit.pushed_owner),
                "occupied_by": plain_value(deposit.occupied_by),
            }
            for deposit_id, deposit in state.deposits.items()
        },
        thermometer_state=plain_value(state.thermometer.state),
        thermometer_doing_blue=state.thermometer.doing_blue,
        thermometer_doing_yellow=state.thermometer.doing_yellow,
        mars_states={
            side.value: [
                {
                    "name": mars.name,
                    "pantry_id": mars.pantry_id,
                    "position": mars_position(mars, state.t),
                    "released": mars.is_released(state.t),
                    "arrived": mars_has_arrived(mars, state.t),
                    "stopped": mars.name in stopped_mars_names,
                    "collided_by_blue": mars_collided(mars, Side.BLUE),
                    "collided_by_yellow": mars_collided(mars, Side.YELLOW),
                }
                for mars in state.mars_by_side.get(side, ())
            ]
            for side in (Side.BLUE, Side.YELLOW)
        },
    )
