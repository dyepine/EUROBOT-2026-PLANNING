from __future__ import annotations

from dataclasses import dataclass

from poc.actions import Action, ActionType
from poc.entities import DepositType, Side, SourceState, ThermometerState
from poc.game_state import GameState
from poc.geometry import distance

HOME_DEPOSIT_POINTS = 8
STORAGE_DEPOSIT_POINTS = 17


@dataclass(slots=True)
class ActionTimingConfig:
    move_overhead: float = 0.2
    pick_duration: float = 2.5
    deposit_duration: float = 5.0
    thermometer_duration: float = 3.0
    attack_duration: float = 0.0
    wait_duration: float = 1.0
    align_duration: float = 1.0
    grip_rotate_duration: float = 0.2
    obstacle_clearance_radius: float = 0.18
    obstacle_detour_penalty: float = 1.2
    robot_separation_radius: float = 0.24
    interaction_radius: float = 0.08


@dataclass(slots=True)
class UtilityWeights:
    reward_weight: float = 1.6
    time_weight: float = 1.0
    risk_weight: float = 2.2
    blocking_weight: float = 1.3
    swing_weight: float = 1.2


def travel_time(distance_to_target: float, speed: float, timing: ActionTimingConfig) -> float:
    if distance_to_target == 0.0:
        return 0.0
    return distance_to_target / speed + timing.move_overhead


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


def deposit_points(kind: DepositType) -> int:
    return HOME_DEPOSIT_POINTS if kind is DepositType.HOME else STORAGE_DEPOSIT_POINTS


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
        reward = 2.0 * units
        risk = 0.9 if source.state is SourceState.DISTURBED else 0.25
    elif action.type is ActionType.DEPOSIT and action.target_id is not None:
        deposit = state.deposits[action.target_id]
        if deposit.kind is DepositType.STORAGE and deposit.total_items() > 0:
            reward = -4.0
            risk = 0.9
            swing = 0.0
        else:
            reward = float(deposit_points(deposit.kind))
            swing = 0.0
            risk = 0.15
    elif action.type is ActionType.ATTACK_DEPOSIT and action.target_id is not None:
        deposit = state.deposits[action.target_id]
        removable = deposit.items_for_side(side.opponent())
        reward = 0.0
        swing = float(STORAGE_DEPOSIT_POINTS if removable > 0 else 0.0)
        risk = 0.4
    elif action.type is ActionType.DO_THERMOMETER:
        reward = float(state.thermometer.reward)
        risk = 0.1 if not state.thermometer.is_done_for_side(side) else 1.0
    elif action.type is ActionType.START_ENDGAME:
        reward = 10.0
        risk = 0.05 if action.expected_duration <= state.T_end - state.t else 0.9
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
