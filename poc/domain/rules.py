from __future__ import annotations

from poc.domain.entities import DepositPoint, DepositType, Side, SourceState
from poc.domain.game_state import GameState
from poc.domain.scoring import deposit_max_count_for_side


def thermometer_lane_is_clear(state: GameState, side: Side) -> bool:
    blocking_source_id = state.thermometer.blocking_source_id_for_side(side)
    source = state.sources.get(blocking_source_id)
    blocking_source_clear = (
        source is None
        or source.state is SourceState.EMPTY
        or source.available_items <= 0
    )
    zone_10 = state.deposits.get(10)
    zone_10_clear = zone_10 is None or zone_10.total_items() == 0
    blocking_deposit_id = state.thermometer.blocking_deposit_id_for_side(side)
    blocking_deposit = state.deposits.get(blocking_deposit_id)
    blocking_deposit_clear = blocking_deposit is None or blocking_deposit.total_items() == 0
    return blocking_source_clear and zone_10_clear and blocking_deposit_clear


def home_deposit_for_side(state: GameState, side: Side) -> DepositPoint | None:
    for deposit in state.deposits.values():
        if deposit.kind is DepositType.HOME and deposit.owner is side:
            return deposit
    return None


def home_return_blocked(state: GameState, side: Side) -> bool:
    home = home_deposit_for_side(state, side)
    return home is not None and home.total_items() > 0


def deposit_capacity_for_side(deposit: DepositPoint, side: Side, robot_load: int) -> int:
    return deposit_max_count_for_side(deposit, side, robot_load)


def deposit_can_accept_count(deposit: DepositPoint, side: Side, robot_load: int, count: int) -> bool:
    return 0 < count <= deposit_capacity_for_side(deposit, side, robot_load)


def mars_has_pantry_credit(*, arrived: bool, collided: bool) -> bool:
    return arrived and not collided
