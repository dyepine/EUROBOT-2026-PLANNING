from __future__ import annotations

from dataclasses import dataclass, field

from poc.endgame import EndgameConfig
from poc.entities import DepositPoint, DepositType, Robot, Side, SourcePoint, Thermometer
from poc.external_events import ExternalEvent


@dataclass(slots=True)
class GameState:
    t: float
    T_end: float
    our_side: Side
    our_robot: Robot
    enemy_robot: Robot
    sources: dict[int, SourcePoint]
    deposits: dict[int, DepositPoint]
    thermometer: Thermometer
    external_events: list[ExternalEvent]
    semantic_map_name: str
    field_size: tuple[float, float]
    endgame_by_side: dict[Side, EndgameConfig]
    score_blue: int = 0
    score_yellow: int = 0
    endgame_started: dict[Side, bool] = field(
        default_factory=lambda: {Side.BLUE: False, Side.YELLOW: False}
    )

    @property
    def enemy_side(self) -> Side:
        return self.our_side.opponent()

    def robot_for_side(self, side: Side) -> Robot:
        return self.our_robot if side is self.our_side else self.enemy_robot

    def score_for_side(self, side: Side) -> int:
        return self.score_blue if side is Side.BLUE else self.score_yellow

    def add_score(self, side: Side, points: int) -> None:
        if side is Side.BLUE:
            self.score_blue += points
        else:
            self.score_yellow += points

    def friendly_deposits(
        self,
        side: Side,
        include_home: bool = True,
        include_neutral: bool = True,
    ) -> list[DepositPoint]:
        return [
            deposit
            for deposit in self.deposits.values()
            if (deposit.owner is side or include_neutral and deposit.owner is None)
            and (include_home or deposit.kind is not DepositType.HOME)
        ]

    def enemy_deposits(self, side: Side) -> list[DepositPoint]:
        return [deposit for deposit in self.deposits.values() if deposit.owner is side.opponent()]

    def endgame_config_for(self, side: Side) -> EndgameConfig:
        return self.endgame_by_side[side]

    def endgame_started_for(self, side: Side) -> bool:
        return self.endgame_started[side]

    def set_endgame_started(self, side: Side, value: bool) -> None:
        self.endgame_started[side] = value
