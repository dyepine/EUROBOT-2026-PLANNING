from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from math import hypot
from pathlib import Path

from poc.actions import Action
from poc.entities import DepositPoint, DepositType, Side, SourceState
from poc.game_state import GameState
from poc.policy_mapping import (
    normalized_action_label,
    normalized_deposit_id,
    normalized_source_id,
)
from poc.scoring import deposit_max_count, deposit_zone_points

POLICY_SOURCE_ORDER = (
    "OUR_SOURCE_1",
    "OUR_SOURCE_2",
    "OUR_SOURCE_3",
    "OUR_SOURCE_4",
    "ENEMY_SOURCE_1",
    "ENEMY_SOURCE_2",
    "ENEMY_SOURCE_3",
    "ENEMY_SOURCE_4",
)

POLICY_DEPOSIT_ORDER = (
    "OUR_HOME",
    "ENEMY_HOME",
    "OUR_STORAGE_5",
    "OUR_STORAGE_6",
    "OUR_STORAGE_7",
    "NEUTRAL_STORAGE_1",
    "NEUTRAL_STORAGE_0",
    "ENEMY_STORAGE_5",
    "ENEMY_STORAGE_6",
    "ENEMY_STORAGE_7",
)

POLICY_DEPOSIT_ACTION_TARGETS = (
    "OUR_HOME",
    "OUR_STORAGE_5",
    "OUR_STORAGE_6",
    "OUR_STORAGE_7",
    "NEUTRAL_STORAGE_1",
    "NEUTRAL_STORAGE_0",
    "ENEMY_STORAGE_5",
    "ENEMY_STORAGE_6",
    "ENEMY_STORAGE_7",
)

POLICY_ATTACK_TARGETS = (
    "OUR_STORAGE_5",
    "OUR_STORAGE_6",
    "OUR_STORAGE_7",
    "NEUTRAL_STORAGE_1",
    "NEUTRAL_STORAGE_0",
    "ENEMY_STORAGE_5",
    "ENEMY_STORAGE_6",
    "ENEMY_STORAGE_7",
)

GLOBAL_FEATURE_ORDER = (
    "time_norm",
    "time_remaining_norm",
    "our_x_norm",
    "our_y_norm",
    "enemy_x_norm",
    "enemy_y_norm",
    "our_load_norm",
    "enemy_load_norm",
    "enemy_vel_x_norm",
    "enemy_vel_y_norm",
    "enemy_speed_norm",
    "our_endgame_started",
    "enemy_endgame_started",
    "thermometer_done_for_us",
    "thermometer_done_for_enemy",
    "thermometer_lane_clear_for_us",
    "thermometer_available_for_us",
    "field_width_m",
    "field_height_m",
)

SOURCE_FEATURE_ORDER = (
    "x_norm",
    "y_norm",
    "available_items_norm",
    "available_now",
    "state_untouched",
    "state_disturbed",
    "state_empty",
    "map_footprint_enabled",
)

DEPOSIT_FEATURE_ORDER = (
    "x_norm",
    "y_norm",
    "our_items_norm",
    "enemy_items_norm",
    "total_items_norm",
    "our_points_raw",
    "enemy_points_raw",
    "score_diff_raw",
    "our_points_norm",
    "enemy_points_norm",
    "score_diff_norm",
    "occupied_by_our",
    "occupied_by_enemy",
    "occupied_by_none",
    "kind_home",
    "kind_storage",
    "protected_for_our",
    "protected_for_enemy",
    "map_footprint_enabled",
    "attack_delta_raw",
    "attack_delta_norm",
    "deposit_x1_delta_raw",
    "deposit_x1_delta_norm",
    "deposit_x1_valid",
    "deposit_x2_delta_raw",
    "deposit_x2_delta_norm",
    "deposit_x2_valid",
    "deposit_x3_delta_raw",
    "deposit_x3_delta_norm",
    "deposit_x3_valid",
    "deposit_x4_delta_raw",
    "deposit_x4_delta_norm",
    "deposit_x4_valid",
)


@dataclass(frozen=True, slots=True)
class RLObservationConfig:
    mirror_x_for_yellow: bool = True
    velocity_deadband_mps: float = 0.05
    velocity_normalizer_mps: float = 0.45
    source_items_scale: float = 4.0
    load_scale: float = 4.0
    deposit_items_scale: float = 4.0
    zone_points_scale: float = 17.0


@dataclass(frozen=True, slots=True)
class RLObservation:
    perspective: str
    global_features: dict[str, float]
    source_features: dict[str, dict[str, float]]
    deposit_features: dict[str, dict[str, float]]
    flat_features: dict[str, float]


@dataclass(frozen=True, slots=True)
class RLActionSpace:
    tokens: tuple[str, ...]

    @property
    def index_by_token(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, token: str) -> int:
        return self.index_by_token[token]

    def decode(self, index: int) -> str:
        return self.tokens[index]


@dataclass(frozen=True, slots=True)
class RLCandidate:
    index: int
    policy_action: str
    expected_duration: float
    score: float
    reward: float
    risk: float
    blocking_penalty: float
    swing: float


@dataclass(frozen=True, slots=True)
class RLPolicyStep:
    observation: RLObservation
    action_space: RLActionSpace
    action_mask: tuple[int, ...]
    candidates: tuple[RLCandidate, ...]


@dataclass(frozen=True, slots=True)
class RLTransition:
    side: str
    time: float
    chosen_action: str
    chosen_action_index: int
    action_mask: tuple[int, ...]
    observation: RLObservation
    reward: float
    next_observation: RLObservation
    next_action_mask: tuple[int, ...]
    done: bool
    score_diff_before: float
    score_diff_after: float


DEFAULT_ACTION_SPACE = RLActionSpace(
    tokens=(
        *(f"PICK_{source_id}" for source_id in POLICY_SOURCE_ORDER),
        *(f"DEPOSIT_{target}_X{count}" for target in POLICY_DEPOSIT_ACTION_TARGETS for count in range(1, 5)),
        *(f"ATTACK_{target}" for target in POLICY_ATTACK_TARGETS),
        "THERMOMETER",
        "START_ENDGAME",
        "WAIT",
        "WAIT_FOR_CHILL",
    )
)

DEFAULT_FLAT_FEATURE_KEYS = (
    *(f"global.{key}" for key in GLOBAL_FEATURE_ORDER),
    *(f"source.{source_id}.{key}" for source_id in POLICY_SOURCE_ORDER for key in SOURCE_FEATURE_ORDER),
    *(f"deposit.{deposit_id}.{key}" for deposit_id in POLICY_DEPOSIT_ORDER for key in DEPOSIT_FEATURE_ORDER),
)


def build_rl_policy_step(
    state: GameState,
    side: Side,
    ranked_actions: list[Action],
    previous_state: GameState | None = None,
    dt: float | None = None,
    config: RLObservationConfig | None = None,
    action_space: RLActionSpace = DEFAULT_ACTION_SPACE,
) -> RLPolicyStep:
    observation = build_rl_observation(
        state,
        side,
        previous_state=previous_state,
        dt=dt,
        config=config,
    )
    action_mask = [0] * len(action_space.tokens)
    candidates: list[RLCandidate] = []
    index_by_token = action_space.index_by_token
    for action in ranked_actions:
        token = normalized_action_label(action, side)
        action_index = index_by_token.get(token)
        if action_index is None:
            continue
        action_mask[action_index] = 1
        candidates.append(
            RLCandidate(
                index=action_index,
                policy_action=token,
                expected_duration=action.expected_duration,
                score=action.score,
                reward=action.expected_reward,
                risk=action.risk,
                blocking_penalty=action.blocking_penalty,
                swing=action.swing,
            )
        )
    return RLPolicyStep(
        observation=observation,
        action_space=action_space,
        action_mask=tuple(action_mask),
        candidates=tuple(candidates),
    )


def build_rl_policy_step_from_planner(
    state: GameState,
    side: Side,
    planner: "UtilityPlanner",
    previous_state: GameState | None = None,
    dt: float | None = None,
    config: RLObservationConfig | None = None,
    action_space: RLActionSpace = DEFAULT_ACTION_SPACE,
) -> RLPolicyStep:
    ranked_actions = planner.rank_actions(state, side)
    return build_rl_policy_step(
        state,
        side,
        ranked_actions=ranked_actions,
        previous_state=previous_state,
        dt=dt,
        config=config,
        action_space=action_space,
    )


def build_rl_observation(
    state: GameState,
    side: Side,
    previous_state: GameState | None = None,
    dt: float | None = None,
    config: RLObservationConfig | None = None,
) -> RLObservation:
    cfg = config or RLObservationConfig()
    field_width, field_height = state.field_size
    our_robot = state.robot_for_side(side)
    enemy_robot = state.robot_for_side(side.opponent())

    our_x, our_y = _normalize_point(our_robot.position, state, side, cfg)
    enemy_x, enemy_y = _normalize_point(enemy_robot.position, state, side, cfg)
    enemy_vel_x, enemy_vel_y, enemy_speed_norm = _normalized_velocity(
        previous_state,
        state,
        subject=side.opponent(),
        perspective=side,
        dt=dt,
        cfg=cfg,
    )

    global_features = {
        "time_norm": _safe_div(state.t, state.T_end),
        "time_remaining_norm": _safe_div(state.T_end - state.t, state.T_end),
        "our_x_norm": our_x,
        "our_y_norm": our_y,
        "enemy_x_norm": enemy_x,
        "enemy_y_norm": enemy_y,
        "our_load_norm": _safe_div(our_robot.load, max(1.0, cfg.load_scale)),
        "enemy_load_norm": _safe_div(enemy_robot.load, max(1.0, cfg.load_scale)),
        "enemy_vel_x_norm": enemy_vel_x,
        "enemy_vel_y_norm": enemy_vel_y,
        "enemy_speed_norm": enemy_speed_norm,
        "our_endgame_started": float(state.endgame_started_for(side)),
        "enemy_endgame_started": float(state.endgame_started_for(side.opponent())),
        "thermometer_done_for_us": float(state.thermometer.is_done_for_side(side)),
        "thermometer_done_for_enemy": float(state.thermometer.is_done_for_side(side.opponent())),
        "thermometer_lane_clear_for_us": float(_thermometer_lane_is_clear(state, side)),
        "thermometer_available_for_us": float(
            not state.thermometer.is_done_for_side(side)
            and state.t < state.endgame_config_for(side).main_pipeline_deadline
            and _thermometer_lane_is_clear(state, side)
        ),
        "field_width_m": field_width,
        "field_height_m": field_height,
    }

    source_features: dict[str, dict[str, float]] = {}
    for source_id in state.sources:
        policy_id = normalized_source_id(source_id, side)
        source = state.sources[source_id]
        x_norm, y_norm = _normalize_point(source.position, state, side, cfg)
        source_features[policy_id] = {
            "x_norm": x_norm,
            "y_norm": y_norm,
            "available_items_norm": _safe_div(source.available_items, max(1.0, cfg.source_items_scale)),
            "available_now": float(source.is_available(state.t)),
            "state_untouched": float(source.state is SourceState.UNTOUCHED),
            "state_disturbed": float(source.state is SourceState.DISTURBED),
            "state_empty": float(source.state is SourceState.EMPTY),
            "map_footprint_enabled": float(source.map_footprint_enabled),
        }

    deposit_features: dict[str, dict[str, float]] = {}
    for deposit_id in state.deposits:
        policy_id = normalized_deposit_id(deposit_id, side)
        deposit = state.deposits[deposit_id]
        x_norm, y_norm = _normalize_point(deposit.position, state, side, cfg)
        our_points = float(deposit_zone_points(deposit, side))
        enemy_points = float(deposit_zone_points(deposit, side.opponent()))
        score_diff = our_points - enemy_points
        our_items = float(deposit.items_for_side(side))
        enemy_items = float(deposit.items_for_side(side.opponent()))
        features = {
            "x_norm": x_norm,
            "y_norm": y_norm,
            "our_items_norm": _safe_div(our_items, max(1.0, cfg.deposit_items_scale)),
            "enemy_items_norm": _safe_div(enemy_items, max(1.0, cfg.deposit_items_scale)),
            "total_items_norm": _safe_div(float(deposit.total_items()), max(1.0, cfg.deposit_items_scale)),
            "our_points_raw": our_points,
            "enemy_points_raw": enemy_points,
            "score_diff_raw": score_diff,
            "our_points_norm": _normalize_signed(our_points, cfg.zone_points_scale),
            "enemy_points_norm": _normalize_signed(enemy_points, cfg.zone_points_scale),
            "score_diff_norm": _normalize_signed(score_diff, cfg.zone_points_scale),
            "occupied_by_our": float(deposit.occupied_by is side),
            "occupied_by_enemy": float(deposit.occupied_by is side.opponent()),
            "occupied_by_none": float(deposit.occupied_by is None),
            "kind_home": float(deposit.kind is DepositType.HOME),
            "kind_storage": float(deposit.kind is DepositType.STORAGE),
            "protected_for_our": float(deposit.protected_for is side),
            "protected_for_enemy": float(deposit.protected_for is side.opponent()),
            "map_footprint_enabled": float(deposit.map_footprint_enabled),
            "attack_delta_raw": _attack_score_diff_delta(deposit, side),
            "attack_delta_norm": _normalize_signed(_attack_score_diff_delta(deposit, side), cfg.zone_points_scale),
        }
        for deposit_count in range(1, 5):
            delta_raw = _deposit_score_diff_delta(deposit, side, deposit_count, our_robot.load)
            features[f"deposit_x{deposit_count}_delta_raw"] = delta_raw
            features[f"deposit_x{deposit_count}_delta_norm"] = _normalize_signed(delta_raw, cfg.zone_points_scale)
            features[f"deposit_x{deposit_count}_valid"] = float(
                deposit_count <= our_robot.load and deposit_count <= deposit_max_count(deposit, our_robot.load)
            )
        deposit_features[policy_id] = features

    ordered_source_features = {key: source_features[key] for key in POLICY_SOURCE_ORDER}
    ordered_deposit_features = {key: deposit_features[key] for key in POLICY_DEPOSIT_ORDER}
    flat_features = _flatten_features(global_features, ordered_source_features, ordered_deposit_features)
    return RLObservation(
        perspective=side.value,
        global_features=global_features,
        source_features=ordered_source_features,
        deposit_features=ordered_deposit_features,
        flat_features=flat_features,
    )


def _normalize_point(
    point: tuple[float, float],
    state: GameState,
    perspective: Side,
    cfg: RLObservationConfig,
) -> tuple[float, float]:
    field_width, field_height = state.field_size
    x = point[0]
    if cfg.mirror_x_for_yellow and perspective is Side.YELLOW:
        x = -x
    return (
        _normalize_signed(x, field_width / 2.0),
        _normalize_signed(point[1], field_height / 2.0),
    )


def _normalized_velocity(
    previous_state: GameState | None,
    state: GameState,
    subject: Side,
    perspective: Side,
    dt: float | None,
    cfg: RLObservationConfig,
) -> tuple[float, float, float]:
    if previous_state is None or dt is None or dt <= 0.0:
        return 0.0, 0.0, 0.0
    current = state.robot_for_side(subject).position
    previous = previous_state.robot_for_side(subject).position
    vx = (current[0] - previous[0]) / dt
    vy = (current[1] - previous[1]) / dt
    if cfg.mirror_x_for_yellow and perspective is Side.YELLOW:
        vx = -vx
    speed = hypot(vx, vy)
    if speed < cfg.velocity_deadband_mps:
        return 0.0, 0.0, 0.0
    denom = max(cfg.velocity_normalizer_mps, 1e-6)
    return (
        _clip(vx / denom, -1.0, 1.0),
        _clip(vy / denom, -1.0, 1.0),
        _clip(speed / denom, 0.0, 1.0),
    )


def _deposit_score_diff_delta(
    deposit: DepositPoint,
    side: Side,
    deposit_count: int,
    available_load: int,
) -> float:
    max_count = deposit_max_count(deposit, available_load)
    if deposit_count <= 0 or deposit_count > available_load or deposit_count > max_count:
        return 0.0
    clone = deepcopy(deposit)
    before = float(deposit_zone_points(clone, side) - deposit_zone_points(clone, side.opponent()))
    clone.add_items(side, deposit_count)
    after = float(deposit_zone_points(clone, side) - deposit_zone_points(clone, side.opponent()))
    return after - before


def _attack_score_diff_delta(deposit: DepositPoint, side: Side) -> float:
    if deposit.kind is DepositType.HOME or deposit.protected_for is not None:
        return 0.0
    if deposit.items_for_side(side.opponent()) <= 0:
        return 0.0
    clone = deepcopy(deposit)
    before = float(deposit_zone_points(clone, side) - deposit_zone_points(clone, side.opponent()))
    clone.clear()
    after = float(deposit_zone_points(clone, side) - deposit_zone_points(clone, side.opponent()))
    return after - before


def _thermometer_lane_is_clear(state: GameState, side: Side) -> bool:
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


def _flatten_features(
    global_features: dict[str, float],
    source_features: dict[str, dict[str, float]],
    deposit_features: dict[str, dict[str, float]],
) -> dict[str, float]:
    flat = {f"global.{key}": global_features[key] for key in GLOBAL_FEATURE_ORDER}
    for source_id in POLICY_SOURCE_ORDER:
        features = source_features[source_id]
        for key in SOURCE_FEATURE_ORDER:
            flat[f"source.{source_id}.{key}"] = features[key]
    for deposit_id in POLICY_DEPOSIT_ORDER:
        features = deposit_features[deposit_id]
        for key in DEPOSIT_FEATURE_ORDER:
            flat[f"deposit.{deposit_id}.{key}"] = features[key]
    return flat


def flat_feature_vector(observation: RLObservation | dict[str, float]) -> tuple[float, ...]:
    flat_features = observation.flat_features if isinstance(observation, RLObservation) else observation
    return tuple(float(flat_features[key]) for key in DEFAULT_FLAT_FEATURE_KEYS)


def resolve_policy_action(
    ranked_actions: list[Action],
    side: Side,
    policy_action: str,
) -> Action | None:
    for action in ranked_actions:
        if normalized_action_label(action, side) == policy_action:
            return action
    return None


def _normalize_signed(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return _clip(value / scale, -1.0, 1.0)


def _safe_div(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return value / scale


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def transition_to_record(transition: RLTransition) -> dict[str, object]:
    return {
        "side": transition.side,
        "time": transition.time,
        "chosen_action": transition.chosen_action,
        "chosen_action_index": transition.chosen_action_index,
        "action_mask": list(transition.action_mask),
        "observation": {
            "perspective": transition.observation.perspective,
            "global_features": transition.observation.global_features,
            "source_features": transition.observation.source_features,
            "deposit_features": transition.observation.deposit_features,
            "flat_features": transition.observation.flat_features,
        },
        "reward": transition.reward,
        "next_observation": {
            "perspective": transition.next_observation.perspective,
            "global_features": transition.next_observation.global_features,
            "source_features": transition.next_observation.source_features,
            "deposit_features": transition.next_observation.deposit_features,
            "flat_features": transition.next_observation.flat_features,
        },
        "next_action_mask": list(transition.next_action_mask),
        "done": transition.done,
        "score_diff_before": transition.score_diff_before,
        "score_diff_after": transition.score_diff_after,
    }


def save_transition_dataset(transitions: list[RLTransition], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for transition in transitions:
            handle.write(json.dumps(transition_to_record(transition), ensure_ascii=True))
            handle.write("\n")
    return output
