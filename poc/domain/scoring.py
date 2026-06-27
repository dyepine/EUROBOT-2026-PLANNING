from __future__ import annotations

from poc.domain.actions import Action, ActionType
from poc.domain.config import DEFAULT_SCORE_CONFIG, ActionTimingConfig, UtilityWeights
from poc.domain.entities import DepositPoint, DepositType, Side, SourceState
from poc.domain.game_state import GameState
from poc.domain.geometry import distance

def blocking_penalty(
    state: GameState,
    side: Side,
    target_position: tuple[float, float] | None,
    timing: ActionTimingConfig,
) -> float:
    if target_position is None:
        return 0.0
    enemy_robot = state.robot_for_side(side.opponent())
    d = distance(enemy_robot.position, target_position)
    if d <= timing.robot_separation_radius:
        return 1.5
    if d <= timing.robot_separation_radius + 0.20:
        return 0.8
    return 0.0


def home_remaining_capacity(deposit: DepositPoint) -> int:
    return max(0, DEFAULT_SCORE_CONFIG.home_capacity - deposit.total_items())


def deposit_can_accept_load(deposit: DepositPoint, load: int) -> bool:
    if load <= 0:
        return False
    if deposit.kind is DepositType.HOME:
        return deposit.total_items() == 0 and home_remaining_capacity(deposit) > 0
    return deposit.total_items() == 0


def deposit_can_accept_load_for_side(deposit: DepositPoint, side: Side, load: int) -> bool:
    return deposit.allows_deposit_for_side(side) and deposit_can_accept_load(deposit, load)


def deposit_max_count(deposit: DepositPoint, available_load: int) -> int:
    if available_load <= 0:
        return 0
    if deposit.kind is DepositType.HOME:
        if deposit.total_items() > 0:
            return 0
        return min(available_load, home_remaining_capacity(deposit))
    if deposit.total_items() > 0:
        return 0
    return available_load


def deposit_max_count_for_side(deposit: DepositPoint, side: Side, available_load: int) -> int:
    if not deposit.allows_deposit_for_side(side):
        return 0
    return deposit_max_count(deposit, available_load)


def deposit_majority_owner(deposit: DepositPoint) -> Side | None:
    if deposit.blue_items > deposit.yellow_items:
        return Side.BLUE
    if deposit.yellow_items > deposit.blue_items:
        return Side.YELLOW
    return None


def deposit_zone_points(deposit: DepositPoint, side: Side) -> int:
    if deposit.kind is DepositType.HOME:
        return DEFAULT_SCORE_CONFIG.home_item_points * min(
            deposit.items_for_side(side),
            DEFAULT_SCORE_CONFIG.home_capacity,
        )

    points = DEFAULT_SCORE_CONFIG.storage_item_points * deposit.items_for_side(side)
    if deposit_majority_owner(deposit) is side and deposit.items_for_side(side) > 0:
        points += DEFAULT_SCORE_CONFIG.storage_majority_bonus
    return points


def deposit_reward_estimate(deposit: DepositPoint, carried_items: int) -> float:
    if carried_items <= 0:
        return 0.0
    if deposit.kind is DepositType.HOME:
        valid_items = min(carried_items, home_remaining_capacity(deposit))
        return float(valid_items * DEFAULT_SCORE_CONFIG.home_item_points)
    return float(
        carried_items * DEFAULT_SCORE_CONFIG.storage_item_points
        + DEFAULT_SCORE_CONFIG.storage_majority_bonus
    )


def evaluate_action(
    state: GameState,
    side: Side,
    action: Action,
    timing: ActionTimingConfig,
    weights: UtilityWeights,
) -> Action:
    robot = state.robot_for_side(side)
    reward = action.expected_reward
    risk = 0.1
    swing = 0.0

    if action.type is ActionType.PICK and action.target_id is not None:
        source = state.sources[action.target_id]
        units = min(source.available_items, robot.capacity - robot.load)
        reward = float(DEFAULT_SCORE_CONFIG.storage_item_points * units)
        risk = 0.9 if source.state is SourceState.DISTURBED else 0.25
    elif action.type is ActionType.DEPOSIT and action.target_id is not None:
        deposit = state.deposits[action.target_id]
        deposit_count = int(action.metadata.get("deposit_count", robot.load))
        if deposit_count <= 0 or deposit_count > deposit_max_count_for_side(deposit, side, robot.load):
            reward = -4.0
            risk = 0.9
            swing = 0.0
        else:
            reward = deposit_reward_estimate(deposit, deposit_count)
            swing = 0.0
            risk = 0.15
    elif action.type is ActionType.ATTACK_DEPOSIT and action.target_id is not None:
        deposit = state.deposits[action.target_id]
        removable = deposit_zone_points(deposit, side.opponent())
        reward = 0.0
        swing = float(removable)
        risk = 0.4
    elif action.type is ActionType.DO_THERMOMETER:
        reward = float(state.thermometer.reward)
        risk = 0.1 if not state.thermometer.is_done_for_side(side) else 1.0
    elif action.type is ActionType.START_ENDGAME:
        reward = float(state.endgame_config_for(side).score.finish_full_points)
        risk = 0.05 if action.expected_duration <= state.T_end - state.t else 0.9
    elif action.type is ActionType.PLAY_TO_END:
        reward = 0.0
        risk = 0.1
    elif action.type is ActionType.WAIT:
        reward = 0.0
        risk = 0.0

    block = blocking_penalty(state, side, action.target_position, timing)
    time_cost = action.expected_duration
    score = (
        weights.reward_weight * reward
        - weights.time_weight * time_cost
        - weights.risk_weight * risk
        - weights.blocking_weight * block
        + weights.swing_weight * swing
    )

    action.expected_reward = reward
    action.risk = risk
    action.blocking_penalty = block
    action.swing = swing
    action.score = score
    return action
