from __future__ import annotations

from dataclasses import dataclass

from poc.actions import Action, ActionType
from poc.entities import Side, ThermometerState
from poc.game_state import GameState
from poc.planner import UtilityPlanner


@dataclass(slots=True)
class OpponentPolicy:
    name: str

    def choose_action(self, state: GameState, planner: UtilityPlanner, side: Side) -> Action:
        raise NotImplementedError


class NearestGreedyPolicy(OpponentPolicy):
    def __init__(self) -> None:
        super().__init__(name="nearest_greedy")

    def choose_action(self, state: GameState, planner: UtilityPlanner, side: Side) -> Action:
        ranked = planner.rank_actions(state, side)
        robot = state.robot_for_side(side)
        if state.t >= state.endgame_config_for(side).main_pipeline_deadline:
            endgame = [action for action in ranked if action.type is ActionType.START_ENDGAME]
            return endgame[0] if endgame else ranked[0]
        if robot.load > 0:
            deposits = [action for action in ranked if action.type is ActionType.DEPOSIT]
            if deposits:
                return min(deposits, key=lambda action: action.expected_duration)
        picks = [action for action in ranked if action.type is ActionType.PICK]
        if picks:
            return min(picks, key=lambda action: action.expected_duration)
        thermo = [action for action in ranked if action.type is ActionType.DO_THERMOMETER]
        if thermo:
            return thermo[0]
        endgame = [action for action in ranked if action.type is ActionType.START_ENDGAME]
        return endgame[0] if endgame else ranked[0]


class AggressivePolicy(OpponentPolicy):
    def __init__(self) -> None:
        super().__init__(name="aggressive")
        self._fallback = NearestGreedyPolicy()

    def choose_action(self, state: GameState, planner: UtilityPlanner, side: Side) -> Action:
        ranked = planner.rank_actions(state, side)
        attacks = [action for action in ranked if action.type is ActionType.ATTACK_DEPOSIT]
        if attacks:
            return max(attacks, key=lambda action: action.score)
        return self._fallback.choose_action(state, planner, side)


class ThermoFirstPolicy(OpponentPolicy):
    def __init__(self) -> None:
        super().__init__(name="thermo_first")
        self._fallback = NearestGreedyPolicy()

    def choose_action(self, state: GameState, planner: UtilityPlanner, side: Side) -> Action:
        ranked = planner.rank_actions(state, side)
        if not state.thermometer.is_done_for_side(side):
            thermo = [action for action in ranked if action.type is ActionType.DO_THERMOMETER]
            if thermo:
                return thermo[0]
        return self._fallback.choose_action(state, planner, side)


class StorageFirstPolicy(OpponentPolicy):
    def __init__(self) -> None:
        super().__init__(name="storage_first")
        self._fallback = NearestGreedyPolicy()

    def choose_action(self, state: GameState, planner: UtilityPlanner, side: Side) -> Action:
        ranked = planner.rank_actions(state, side)
        robot = state.robot_for_side(side)
        if robot.load > 0:
            deposits = [action for action in ranked if action.type is ActionType.DEPOSIT]
            storage_deposits = [
                action
                for action in deposits
                if action.target_id is not None
                and state.deposits[action.target_id].kind.value == "storage"
            ]
            if storage_deposits:
                return max(storage_deposits, key=lambda action: (action.score, -action.expected_duration))
            if deposits:
                return max(deposits, key=lambda action: (action.score, -action.expected_duration))
        return self._fallback.choose_action(state, planner, side)


class HomeSafePolicy(OpponentPolicy):
    def __init__(self) -> None:
        super().__init__(name="home_safe")
        self._fallback = NearestGreedyPolicy()

    def choose_action(self, state: GameState, planner: UtilityPlanner, side: Side) -> Action:
        ranked = planner.rank_actions(state, side)
        robot = state.robot_for_side(side)
        if state.t >= state.endgame_config_for(side).main_pipeline_deadline:
            endgame = [action for action in ranked if action.type is ActionType.START_ENDGAME]
            if endgame:
                return endgame[0]
        if robot.load > 0:
            deposits = [action for action in ranked if action.type is ActionType.DEPOSIT]
            home_deposits = [
                action
                for action in deposits
                if action.target_id is not None
                and state.deposits[action.target_id].kind.value == "home"
            ]
            if home_deposits:
                return min(home_deposits, key=lambda action: action.expected_duration)
            if deposits:
                return min(deposits, key=lambda action: action.expected_duration)
        return self._fallback.choose_action(state, planner, side)


def build_opponent_policy(name: str) -> OpponentPolicy:
    if name == "aggressive":
        return AggressivePolicy()
    if name == "thermo_first":
        return ThermoFirstPolicy()
    if name == "storage_first":
        return StorageFirstPolicy()
    if name == "home_safe":
        return HomeSafePolicy()
    return NearestGreedyPolicy()
