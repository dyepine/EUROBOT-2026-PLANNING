from __future__ import annotations

from dataclasses import dataclass, field

from poc.domain.endgame import EndgameConfig
from poc.domain.entities import DepositPoint, DepositType, Mars, Robot, Side, SourcePoint, Thermometer
from poc.simulation.external_events import ExternalEvent


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
    mars_by_side: dict[Side, tuple[Mars, ...]] = field(
        default_factory=lambda: {Side.BLUE: (), Side.YELLOW: ()}
    )
    observed_speed_by_side: dict[Side, float] = field(
        default_factory=lambda: {Side.BLUE: 0.0, Side.YELLOW: 0.0}
    )
    last_motion_start_time_by_side: dict[Side, float | None] = field(
        default_factory=lambda: {Side.BLUE: None, Side.YELLOW: None}
    )
    max_observed_speed_by_side: dict[Side, float] = field(
        default_factory=lambda: {Side.BLUE: 0.0, Side.YELLOW: 0.0}
    )
    source_last_items_delta_by_id: dict[int, float] = field(default_factory=dict)
    source_last_change_time_by_id: dict[int, float] = field(default_factory=dict)
    source_last_change_was_disturb_like_by_id: dict[int, bool] = field(default_factory=dict)
    deposit_last_blue_score_delta_by_id: dict[int, float] = field(default_factory=dict)
    deposit_last_yellow_score_delta_by_id: dict[int, float] = field(default_factory=dict)
    deposit_last_score_change_time_by_id: dict[int, float] = field(default_factory=dict)
    deposit_last_actor_by_id: dict[int, Side | None] = field(default_factory=dict)
    thermometer_last_state_change_time: float = 0.0
    thermometer_lane_clear_by_side: dict[Side, bool] = field(
        default_factory=lambda: {Side.BLUE: False, Side.YELLOW: False}
    )
    thermometer_lane_clear_change_time_by_side: dict[Side, float] = field(
        default_factory=lambda: {Side.BLUE: 0.0, Side.YELLOW: 0.0}
    )
    score_blue: int = 0
    score_yellow: int = 0
    endgame_started: dict[Side, bool] = field(
        default_factory=lambda: {Side.BLUE: False, Side.YELLOW: False}
    )
    play_to_end_started: dict[Side, bool] = field(
        default_factory=lambda: {Side.BLUE: False, Side.YELLOW: False}
    )

    def __post_init__(self) -> None:
        for source_id in self.sources:
            self.source_last_items_delta_by_id.setdefault(source_id, 0.0)
            self.source_last_change_time_by_id.setdefault(source_id, float(self.t))
            self.source_last_change_was_disturb_like_by_id.setdefault(source_id, False)
        for deposit_id in self.deposits:
            self.deposit_last_blue_score_delta_by_id.setdefault(deposit_id, 0.0)
            self.deposit_last_yellow_score_delta_by_id.setdefault(deposit_id, 0.0)
            self.deposit_last_score_change_time_by_id.setdefault(deposit_id, float(self.t))
            self.deposit_last_actor_by_id.setdefault(deposit_id, None)
        for side in (Side.BLUE, Side.YELLOW):
            self.thermometer_lane_clear_by_side.setdefault(side, False)
            self.thermometer_lane_clear_change_time_by_side.setdefault(side, float(self.t))

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
            if deposit.allows_deposit_for_side(side)
            and (include_home or deposit.kind is not DepositType.HOME)
            and (
                include_neutral
                or deposit.owner is side
                or deposit.protected_for is side
            )
        ]

    def enemy_deposits(self, side: Side) -> list[DepositPoint]:
        return [deposit for deposit in self.deposits.values() if deposit.owner is side.opponent()]

    def endgame_config_for(self, side: Side) -> EndgameConfig:
        return self.endgame_by_side[side]

    def endgame_started_for(self, side: Side) -> bool:
        return self.endgame_started[side]

    def set_endgame_started(self, side: Side, value: bool) -> None:
        self.endgame_started[side] = value

    def play_to_end_started_for(self, side: Side) -> bool:
        return self.play_to_end_started[side]

    def set_play_to_end_started(self, side: Side, value: bool) -> None:
        self.play_to_end_started[side] = value
