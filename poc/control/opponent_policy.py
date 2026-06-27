from __future__ import annotations

from dataclasses import dataclass
from math import exp
import random

from poc.domain.actions import Action, ActionType
from poc.domain.entities import DepositType, Side
from poc.domain.game_state import GameState
from poc.domain.scoring import deposit_max_count_for_side


def _split_policy_name(name: str) -> tuple[str, int | None]:
    base, separator, suffix = name.partition("@")
    if not separator:
        return name, None
    try:
        return base, int(suffix)
    except ValueError:
        return name, None


def policy_name_uses_variant_seed(name: str) -> bool:
    base, _ = _split_policy_name(name)
    return base.startswith("randomized_") or base in {"stochastic_planner", "uniform_random"}


def materialize_policy_name(name: str, rng: random.Random) -> str:
    if not policy_name_uses_variant_seed(name):
        return name
    return f"{name}@{rng.randint(1, 1_000_000_000)}"


def _deposit_kind(state: GameState, action: Action) -> DepositType | None:
    if action.target_id is None:
        return None
    deposit = state.deposits.get(action.target_id)
    return None if deposit is None else deposit.kind


def _weighted_choice(
    rng: random.Random,
    adjusted_actions: list[tuple[Action, float]],
    *,
    temperature: float,
) -> Action:
    if len(adjusted_actions) == 1:
        return adjusted_actions[0][0]
    safe_temperature = max(temperature, 1e-3)
    max_score = max(score for _, score in adjusted_actions)
    weights = [exp((score - max_score) / safe_temperature) for _, score in adjusted_actions]
    chosen_index = rng.choices(range(len(adjusted_actions)), weights=weights, k=1)[0]
    return adjusted_actions[chosen_index][0]


@dataclass(slots=True)
class OpponentPolicy:
    name: str

    def choose_action_from_ranked(self, state: GameState, ranked: list[Action], side: Side) -> Action:
        raise NotImplementedError

    def choose_action(self, state: GameState, planner, side: Side) -> Action:
        return self.choose_action_from_ranked(state, planner.rank_actions(state, side), side)


@dataclass(frozen=True, slots=True)
class OrderedActionStep:
    type: ActionType
    target_id: int | None = None
    deposit_count: int | None = None

    def __post_init__(self) -> None:
        if self.deposit_count is not None:
            if self.type is not ActionType.DEPOSIT:
                raise ValueError("deposit_count is only supported for DEPOSIT ordered steps")
            if int(self.deposit_count) <= 0:
                raise ValueError("deposit_count must be a positive integer")


def _wait_action(label: str = "WAIT", duration: float = 0.25) -> Action:
    return Action(
        type=ActionType.WAIT,
        target_id=None,
        label=label,
        target_position=None,
        waypoints=(),
        service_duration=duration,
        travel_duration=0.0,
        expected_duration=duration,
    )


class NearestGreedyPolicy(OpponentPolicy):
    def __init__(self) -> None:
        super().__init__(name="nearest_greedy")

    def choose_action_from_ranked(self, state: GameState, ranked: list[Action], side: Side) -> Action:
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

    def choose_action_from_ranked(self, state: GameState, ranked: list[Action], side: Side) -> Action:
        attacks = [action for action in ranked if action.type is ActionType.ATTACK_DEPOSIT]
        if attacks:
            return max(attacks, key=lambda action: action.score)
        return self._fallback.choose_action_from_ranked(state, ranked, side)


class ThermoFirstPolicy(OpponentPolicy):
    def __init__(self) -> None:
        super().__init__(name="thermo_first")
        self._fallback = NearestGreedyPolicy()

    def choose_action_from_ranked(self, state: GameState, ranked: list[Action], side: Side) -> Action:
        if not state.thermometer.is_done_for_side(side):
            thermo = [action for action in ranked if action.type is ActionType.DO_THERMOMETER]
            if thermo:
                return thermo[0]
        return self._fallback.choose_action_from_ranked(state, ranked, side)


class StorageFirstPolicy(OpponentPolicy):
    def __init__(self) -> None:
        super().__init__(name="storage_first")
        self._fallback = NearestGreedyPolicy()

    def choose_action_from_ranked(self, state: GameState, ranked: list[Action], side: Side) -> Action:
        robot = state.robot_for_side(side)
        if robot.load > 0:
            deposits = [action for action in ranked if action.type is ActionType.DEPOSIT]
            storage_deposits = [
                action
                for action in deposits
                if _deposit_kind(state, action) is DepositType.STORAGE
            ]
            if storage_deposits:
                return max(storage_deposits, key=lambda action: (action.score, -action.expected_duration))
            if deposits:
                return max(deposits, key=lambda action: (action.score, -action.expected_duration))
        return self._fallback.choose_action_from_ranked(state, ranked, side)


class HomeSafePolicy(OpponentPolicy):
    def __init__(self) -> None:
        super().__init__(name="home_safe")
        self._fallback = NearestGreedyPolicy()

    def choose_action_from_ranked(self, state: GameState, ranked: list[Action], side: Side) -> Action:
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
                if _deposit_kind(state, action) is DepositType.HOME
            ]
            if home_deposits:
                return min(home_deposits, key=lambda action: action.expected_duration)
            if deposits:
                return min(deposits, key=lambda action: action.expected_duration)
        return self._fallback.choose_action_from_ranked(state, ranked, side)


class FixedSequencePolicy(OpponentPolicy):
    def __init__(self, *, name: str, steps: tuple[OrderedActionStep, ...]) -> None:
        super().__init__(name=name)
        self.steps = steps
        self._step_index = 0

    def choose_action_from_ranked(self, state: GameState, ranked: list[Action], side: Side) -> Action:
        while self._step_index < len(self.steps):
            step = self.steps[self._step_index]
            if self._step_is_done(step, state, side):
                self._step_index += 1
                continue
            matched = self._match_step(step, ranked)
            if matched is not None:
                self._step_index += 1
                return matched
            if self._step_can_be_skipped(step, state, side):
                self._step_index += 1
                continue
            return _wait_action()
        return ranked[0] if ranked else _wait_action(label="WAIT")

    def _step_is_done(self, step: OrderedActionStep, state: GameState, side: Side) -> bool:
        robot = state.robot_for_side(side)
        if step.type is ActionType.PICK:
            if robot.load > 0:
                return True
            source = state.sources.get(step.target_id)
            return source is None or not source.is_available(state.t)
        if step.type is ActionType.DEPOSIT:
            return robot.load <= 0
        if step.type is ActionType.DO_THERMOMETER:
            return state.thermometer.is_done_for_side(side)
        if step.type is ActionType.START_ENDGAME:
            return state.endgame_started_for(side) or state.play_to_end_started_for(side)
        return False

    def _step_can_be_skipped(self, step: OrderedActionStep, state: GameState, side: Side) -> bool:
        if step.type is ActionType.PICK:
            source = state.sources.get(step.target_id)
            return source is None or not source.is_available(state.t)
        if step.type is ActionType.DEPOSIT:
            robot = state.robot_for_side(side)
            if robot.load <= 0:
                return True
            deposit = state.deposits.get(step.target_id)
            if deposit is None:
                return True
            max_count = deposit_max_count_for_side(deposit, side, robot.load)
            if max_count <= 0:
                return True
            if step.deposit_count is not None and max_count < step.deposit_count:
                return True
            return False
        if step.type is ActionType.DO_THERMOMETER:
            return state.thermometer.is_done_for_side(side)
        return False

    def _match_step(self, step: OrderedActionStep, ranked: list[Action]) -> Action | None:
        matches = [
            action
            for action in ranked
            if action.type is step.type and (step.target_id is None or action.target_id == step.target_id)
        ]
        if not matches:
            return None
        if step.type is ActionType.DEPOSIT:
            if step.deposit_count is not None:
                matches = [
                    action
                    for action in matches
                    if int(action.metadata.get("deposit_count", 0)) == step.deposit_count
                ]
                if matches:
                    return max(matches, key=lambda action: action.score)
            return max(matches, key=lambda action: (int(action.metadata.get("deposit_count", 0)), action.score))
        return matches[0]


class YellowSideFixedSequencePolicy(FixedSequencePolicy):
    def __init__(self) -> None:
        super().__init__(
            name="yellow_side_fixed_sequence",
            steps=(
                OrderedActionStep(ActionType.PICK, 24),
                OrderedActionStep(ActionType.DEPOSIT, 27),
                OrderedActionStep(ActionType.PICK, 23),
                OrderedActionStep(ActionType.DO_THERMOMETER, 900),
                OrderedActionStep(ActionType.DEPOSIT, 26),
                OrderedActionStep(ActionType.PICK, 22),
                OrderedActionStep(ActionType.DEPOSIT, 25),
                OrderedActionStep(ActionType.PICK, 21),
                OrderedActionStep(ActionType.DEPOSIT, 1, 1),
                OrderedActionStep(ActionType.START_ENDGAME, None),
            ),
        )


class UniformRandomPolicy(OpponentPolicy):
    def __init__(self, *, seed: int | None = None, name: str = "uniform_random") -> None:
        super().__init__(name=name)
        self._rng = random.Random(0 if seed is None else seed)

    def choose_action_from_ranked(self, state: GameState, ranked: list[Action], side: Side) -> Action:
        interesting = [action for action in ranked if action.type is not ActionType.WAIT]
        return self._rng.choice(interesting or ranked)


class StochasticPlannerPolicy(OpponentPolicy):
    def __init__(
        self,
        *,
        seed: int | None = None,
        temperature: float = 0.6,
        top_k: int = 4,
        name: str = "stochastic_planner",
    ) -> None:
        super().__init__(name=name)
        self._rng = random.Random(0 if seed is None else seed)
        self.temperature = temperature
        self.top_k = max(1, top_k)

    def choose_action_from_ranked(self, state: GameState, ranked: list[Action], side: Side) -> Action:
        interesting = [action for action in ranked if action.type is not ActionType.WAIT]
        candidates = (interesting or ranked)[: self.top_k]
        adjusted = [(action, action.score) for action in candidates]
        return _weighted_choice(self._rng, adjusted, temperature=self.temperature)


class WeightedStochasticPolicy(OpponentPolicy):
    def __init__(
        self,
        *,
        name: str,
        seed: int | None,
        temperature: float,
        top_k: int,
        duration_penalty: float,
        wait_penalty: float,
        pick_bias: float = 0.0,
        deposit_bias: float = 0.0,
        storage_bias: float = 0.0,
        home_bias: float = 0.0,
        attack_bias: float = 0.0,
        thermometer_bias: float = 0.0,
        endgame_bias: float = 0.0,
    ) -> None:
        super().__init__(name=name)
        self._rng = random.Random(0 if seed is None else seed)
        self.temperature = temperature
        self.top_k = max(1, top_k)
        self.duration_penalty = duration_penalty
        self.wait_penalty = wait_penalty
        self.pick_bias = pick_bias
        self.deposit_bias = deposit_bias
        self.storage_bias = storage_bias
        self.home_bias = home_bias
        self.attack_bias = attack_bias
        self.thermometer_bias = thermometer_bias
        self.endgame_bias = endgame_bias

    def _adjusted_score(self, state: GameState, side: Side, action: Action) -> float:
        adjusted = action.score - self.duration_penalty * action.expected_duration
        if action.type is ActionType.WAIT:
            adjusted -= self.wait_penalty
        elif action.type is ActionType.PICK:
            adjusted += self.pick_bias
        elif action.type is ActionType.DEPOSIT:
            adjusted += self.deposit_bias
            deposit_kind = _deposit_kind(state, action)
            if deposit_kind is DepositType.STORAGE:
                adjusted += self.storage_bias
            elif deposit_kind is DepositType.HOME:
                adjusted += self.home_bias
        elif action.type is ActionType.ATTACK_DEPOSIT:
            adjusted += self.attack_bias
        elif action.type is ActionType.DO_THERMOMETER and not state.thermometer.is_done_for_side(side):
            adjusted += self.thermometer_bias
        elif action.type is ActionType.START_ENDGAME:
            adjusted += self.endgame_bias
        return adjusted

    def choose_action_from_ranked(self, state: GameState, ranked: list[Action], side: Side) -> Action:
        interesting = [action for action in ranked if action.type is not ActionType.WAIT]
        candidates = (interesting or ranked)[: self.top_k]
        adjusted = [(action, self._adjusted_score(state, side, action)) for action in candidates]
        return _weighted_choice(self._rng, adjusted, temperature=self.temperature)


def _randomized_family_policy(base_name: str, seed: int | None, full_name: str) -> OpponentPolicy:
    family_rng = random.Random(0 if seed is None else seed)
    if base_name == "randomized_aggressive":
        return WeightedStochasticPolicy(
            name=full_name,
            seed=seed,
            temperature=family_rng.uniform(0.18, 0.75),
            top_k=family_rng.randint(2, 4),
            duration_penalty=family_rng.uniform(0.15, 0.6),
            wait_penalty=family_rng.uniform(1.0, 3.5),
            attack_bias=family_rng.uniform(2.5, 6.0),
            thermometer_bias=family_rng.uniform(-0.5, 1.5),
            storage_bias=family_rng.uniform(0.0, 1.5),
            endgame_bias=family_rng.uniform(0.0, 1.0),
        )
    if base_name == "randomized_nearest":
        return WeightedStochasticPolicy(
            name=full_name,
            seed=seed,
            temperature=family_rng.uniform(0.12, 0.45),
            top_k=family_rng.randint(2, 3),
            duration_penalty=family_rng.uniform(0.6, 1.3),
            wait_penalty=family_rng.uniform(1.5, 4.0),
            pick_bias=family_rng.uniform(0.0, 1.0),
            deposit_bias=family_rng.uniform(0.0, 1.0),
            attack_bias=family_rng.uniform(-1.0, 0.5),
            thermometer_bias=family_rng.uniform(-0.25, 0.75),
        )
    if base_name == "randomized_thermo":
        return WeightedStochasticPolicy(
            name=full_name,
            seed=seed,
            temperature=family_rng.uniform(0.18, 0.60),
            top_k=family_rng.randint(2, 4),
            duration_penalty=family_rng.uniform(0.2, 0.8),
            wait_penalty=family_rng.uniform(1.0, 3.0),
            thermometer_bias=family_rng.uniform(2.5, 5.5),
            attack_bias=family_rng.uniform(-0.5, 1.0),
            storage_bias=family_rng.uniform(0.0, 1.0),
        )
    if base_name == "randomized_storage":
        return WeightedStochasticPolicy(
            name=full_name,
            seed=seed,
            temperature=family_rng.uniform(0.15, 0.55),
            top_k=family_rng.randint(2, 4),
            duration_penalty=family_rng.uniform(0.2, 0.9),
            wait_penalty=family_rng.uniform(1.0, 3.5),
            deposit_bias=family_rng.uniform(0.5, 1.5),
            storage_bias=family_rng.uniform(2.0, 4.5),
            home_bias=family_rng.uniform(-1.0, 0.25),
            attack_bias=family_rng.uniform(-0.25, 1.0),
        )
    if base_name == "randomized_home":
        return WeightedStochasticPolicy(
            name=full_name,
            seed=seed,
            temperature=family_rng.uniform(0.15, 0.50),
            top_k=family_rng.randint(2, 4),
            duration_penalty=family_rng.uniform(0.2, 0.9),
            wait_penalty=family_rng.uniform(1.0, 3.0),
            deposit_bias=family_rng.uniform(0.5, 1.5),
            storage_bias=family_rng.uniform(-1.0, 0.25),
            home_bias=family_rng.uniform(2.0, 4.0),
            attack_bias=family_rng.uniform(-0.5, 0.75),
            endgame_bias=family_rng.uniform(0.25, 1.25),
        )
    return WeightedStochasticPolicy(
        name=full_name,
        seed=seed,
        temperature=family_rng.uniform(0.12, 0.90),
        top_k=family_rng.randint(2, 5),
        duration_penalty=family_rng.uniform(0.1, 1.2),
        wait_penalty=family_rng.uniform(0.5, 4.0),
        pick_bias=family_rng.uniform(-0.5, 1.0),
        deposit_bias=family_rng.uniform(-0.25, 1.5),
        storage_bias=family_rng.uniform(-1.0, 3.0),
        home_bias=family_rng.uniform(-1.0, 3.0),
        attack_bias=family_rng.uniform(-1.0, 4.0),
        thermometer_bias=family_rng.uniform(-1.0, 4.0),
        endgame_bias=family_rng.uniform(-0.25, 1.25),
    )


def build_opponent_policy(name: str) -> OpponentPolicy:
    base_name, seed = _split_policy_name(name)
    if base_name == "aggressive":
        return AggressivePolicy()
    if base_name == "thermo_first":
        return ThermoFirstPolicy()
    if base_name == "storage_first":
        return StorageFirstPolicy()
    if base_name == "home_safe":
        return HomeSafePolicy()
    if base_name == "yellow_side_fixed_sequence":
        return YellowSideFixedSequencePolicy()
    if base_name == "uniform_random":
        return UniformRandomPolicy(seed=seed, name=name)
    if base_name == "stochastic_planner":
        return StochasticPlannerPolicy(seed=seed, name=name)
    if base_name.startswith("randomized_"):
        return _randomized_family_policy(base_name, seed, name)
    return NearestGreedyPolicy()
