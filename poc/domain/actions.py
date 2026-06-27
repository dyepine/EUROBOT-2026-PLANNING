from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from poc.domain.geometry import Vec2


class ActionType(str, Enum):
    PICK = "pick"
    DEPOSIT = "deposit"
    ATTACK_DEPOSIT = "attack_deposit"
    DO_THERMOMETER = "do_thermometer"
    START_ENDGAME = "start_endgame"
    PLAY_TO_END = "play_to_end"
    WAIT = "wait"


@dataclass(slots=True)
class Action:
    type: ActionType
    target_id: int | None
    label: str
    target_position: Vec2 | None
    waypoints: tuple[Vec2, ...] = ()
    service_duration: float = 0.0
    travel_duration: float = 0.0
    expected_duration: float = 0.0
    expected_reward: float = 0.0
    score: float = 0.0
    risk: float = 0.0
    blocking_penalty: float = 0.0
    swing: float = 0.0
    duration_source: str = "grid_astar+constants"
    metadata: dict[str, object] = field(default_factory=dict)

    def debug_row(self) -> dict[str, float | int | str | bool | None]:
        deposit_count = self.metadata.get("deposit_count")
        return {
            "action": self.label,
            "type": self.type.value,
            "target_id": self.target_id,
            "deposit_count": int(deposit_count) if deposit_count is not None else None,
            "score": round(self.score, 3),
            "reward": round(self.expected_reward, 3),
            "time_cost": round(self.expected_duration, 3),
            "risk": round(self.risk, 3),
            "blocking_penalty": round(self.blocking_penalty, 3),
            "swing": round(self.swing, 3),
            "duration_source": self.duration_source,
        }
