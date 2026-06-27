from __future__ import annotations

from dataclasses import dataclass

from poc.domain.actions import Action
from poc.domain.entities import Side
from poc.domain.game_state import GameState


@dataclass(frozen=True, slots=True)
class RobotObservation:
    position: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PreviousTickObservation:
    t: float
    blue_robot: RobotObservation
    yellow_robot: RobotObservation

    def robot_for_side(self, side: Side) -> RobotObservation:
        return self.blue_robot if side is Side.BLUE else self.yellow_robot


@dataclass(frozen=True, slots=True)
class DecisionObservation:
    state: GameState
    side: Side
    ranked_actions: tuple[Action, ...]
    previous_state: PreviousTickObservation | None
    dt: float

    @property
    def time(self) -> float:
        return self.state.t

    def score_diff(self) -> float:
        return float(self.state.score_for_side(self.side) - self.state.score_for_side(self.side.opponent()))
