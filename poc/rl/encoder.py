from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import hypot
import random
from typing import Protocol

from poc.domain.entities import DepositPoint, DepositType, Side, SourceState
from poc.domain.game_state import GameState
from poc.simulation.observations import DecisionObservation
from poc.rl.policy_mapping import normalized_deposit_id, normalized_source_id
from poc.domain.rules import thermometer_lane_is_clear
from poc.domain.scoring import deposit_max_count_for_side, deposit_zone_points
from poc.rl.tokens import POLICY_DEPOSIT_ORDER, POLICY_SOURCE_ORDER

GLOBAL_FEATURE_ORDER = (
    "time_remaining_norm",
    "time_to_main_pipeline_deadline_norm",
    "time_to_chill_end_norm",
    "after_main_pipeline_deadline",
    "after_chill_end",
    "in_last_30s",
    "in_last_20s",
    "in_last_10s",
    "time_bin_0",
    "time_bin_1",
    "time_bin_2",
    "time_bin_3",
    "time_bin_4",
    "time_bin_5",
    "time_bin_6",
    "time_bin_7",
    "time_bin_8",
    "time_bin_9",
    "our_x_norm",
    "our_y_norm",
    "enemy_x_norm",
    "enemy_y_norm",
    "enemy_rel_x_norm",
    "enemy_rel_y_norm",
    "enemy_rel_x_abs_norm",
    "enemy_rel_y_abs_norm",
    "enemy_rel_dist_norm",
    "our_home_rel_x_norm",
    "our_home_rel_y_norm",
    "our_home_rel_x_abs_norm",
    "our_home_rel_y_abs_norm",
    "our_home_rel_dist_norm",
    "our_home_enemy_rel_x_norm",
    "our_home_enemy_rel_y_norm",
    "our_home_enemy_rel_x_abs_norm",
    "our_home_enemy_rel_y_abs_norm",
    "our_home_enemy_rel_dist_norm",
    "enemy_home_rel_x_norm",
    "enemy_home_rel_y_norm",
    "enemy_home_rel_x_abs_norm",
    "enemy_home_rel_y_abs_norm",
    "enemy_home_rel_dist_norm",
    "enemy_home_enemy_rel_x_norm",
    "enemy_home_enemy_rel_y_norm",
    "enemy_home_enemy_rel_x_abs_norm",
    "enemy_home_enemy_rel_y_abs_norm",
    "enemy_home_enemy_rel_dist_norm",
    "our_chill_rel_x_norm",
    "our_chill_rel_y_norm",
    "our_chill_rel_x_abs_norm",
    "our_chill_rel_y_abs_norm",
    "our_chill_rel_dist_norm",
    "our_chill_enemy_rel_x_norm",
    "our_chill_enemy_rel_y_norm",
    "our_chill_enemy_rel_x_abs_norm",
    "our_chill_enemy_rel_y_abs_norm",
    "our_chill_enemy_rel_dist_norm",
    "thermometer_rel_x_norm",
    "thermometer_rel_y_norm",
    "thermometer_rel_x_abs_norm",
    "thermometer_rel_y_abs_norm",
    "thermometer_rel_dist_norm",
    "thermometer_enemy_rel_x_norm",
    "thermometer_enemy_rel_y_norm",
    "thermometer_enemy_rel_x_abs_norm",
    "thermometer_enemy_rel_y_abs_norm",
    "thermometer_enemy_rel_dist_norm",
    "our_load_norm",
    "enemy_vel_x_norm",
    "enemy_vel_y_norm",
    "enemy_speed_norm",
    "enemy_max_speed_seen_norm",
    "our_endgame_started",
    "enemy_endgame_started",
    "thermometer_done_for_us",
    "thermometer_done_for_enemy",
    "thermometer_lane_clear_for_us",
    "thermometer_available_for_us",
    "thermometer_available_for_enemy",
    "time_since_our_lane_clear_change_norm",
    "time_since_enemy_lane_clear_change_norm",
    "time_since_thermometer_state_change_norm",
)

SOURCE_FEATURE_ORDER = (
    "rel_x_norm",
    "rel_y_norm",
    "rel_x_abs_norm",
    "rel_y_abs_norm",
    "rel_dist_norm",
    "enemy_rel_x_norm",
    "enemy_rel_y_norm",
    "enemy_rel_x_abs_norm",
    "enemy_rel_y_abs_norm",
    "enemy_rel_dist_norm",
    "available_items_norm",
    "available_now",
    "state_untouched",
    "state_disturbed",
    "state_empty",
    "map_footprint_enabled",
    "last_items_delta_norm",
    "time_since_last_change_norm",
    "last_change_was_disturb_like",
)

DEPOSIT_FEATURE_ORDER = (
    "rel_x_norm",
    "rel_y_norm",
    "rel_x_abs_norm",
    "rel_y_abs_norm",
    "rel_dist_norm",
    "enemy_rel_x_norm",
    "enemy_rel_y_norm",
    "enemy_rel_x_abs_norm",
    "enemy_rel_y_abs_norm",
    "enemy_rel_dist_norm",
    "our_items_norm",
    "enemy_items_norm",
    "total_items_norm",
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
    "attack_delta_norm",
    "deposit_x1_delta_norm",
    "deposit_x1_valid",
    "deposit_x2_delta_norm",
    "deposit_x2_valid",
    "deposit_x3_delta_norm",
    "deposit_x3_valid",
    "deposit_x4_delta_norm",
    "deposit_x4_valid",
    "last_score_diff_delta_norm",
    "time_since_last_score_change_norm",
    "last_change_by_our",
    "last_change_by_enemy",
)


@dataclass(frozen=True, slots=True)
class RLObservationConfig:
    mirror_x_for_yellow: bool = True
    velocity_deadband_mps: float = 0.02
    velocity_normalizer_mps: float = 0.18
    observation_noise_seed: int = 0
    enemy_velocity_noise_std_mps: float = 0.0
    enemy_velocity_self_motion_leak_fraction: float = 0.0
    enemy_velocity_self_motion_leak_duration_s: float = 0.0
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




class RobotStateView(Protocol):
    position: tuple[float, float]


class PreviousStateView(Protocol):
    t: float

    def robot_for_side(self, side: Side) -> RobotStateView:
        ...


DEFAULT_FLAT_FEATURE_KEYS = (
    *(f"global.{key}" for key in GLOBAL_FEATURE_ORDER),
    *(f"source.{source_id}.{key}" for source_id in POLICY_SOURCE_ORDER for key in SOURCE_FEATURE_ORDER),
    *(f"deposit.{deposit_id}.{key}" for deposit_id in POLICY_DEPOSIT_ORDER for key in DEPOSIT_FEATURE_ORDER),
)


def build_rl_observation(
    observation: DecisionObservation,
    config: RLObservationConfig | None = None,
) -> RLObservation:
    state = observation.state
    side = observation.side
    previous_state = observation.previous_state
    dt = observation.dt
    cfg = config or RLObservationConfig()
    our_robot = state.robot_for_side(side)
    enemy_robot = state.robot_for_side(side.opponent())
    endgame_config = state.endgame_config_for(side)
    enemy_endgame_config = state.endgame_config_for(side.opponent())
    time_remaining = state.T_end - state.t

    our_x, our_y = _normalize_point(our_robot.position, state, side, cfg)
    enemy_x, enemy_y = _normalize_point(enemy_robot.position, state, side, cfg)
    enemy_rel_x, enemy_rel_y, enemy_rel_x_abs, enemy_rel_y_abs, enemy_rel_dist = _normalize_relative_geometry(
        origin=our_robot.position,
        target=enemy_robot.position,
        state=state,
        perspective=side,
        cfg=cfg,
    )
    our_home_rel_x, our_home_rel_y, our_home_rel_x_abs, our_home_rel_y_abs, our_home_rel_dist = _normalize_relative_geometry(
        origin=our_robot.position,
        target=endgame_config.final_home_point,
        state=state,
        perspective=side,
        cfg=cfg,
    )
    our_home_enemy_rel_x, our_home_enemy_rel_y, our_home_enemy_rel_x_abs, our_home_enemy_rel_y_abs, our_home_enemy_rel_dist = _normalize_relative_geometry(
        origin=enemy_robot.position,
        target=endgame_config.final_home_point,
        state=state,
        perspective=side,
        cfg=cfg,
    )
    enemy_home_rel_x, enemy_home_rel_y, enemy_home_rel_x_abs, enemy_home_rel_y_abs, enemy_home_rel_dist = _normalize_relative_geometry(
        origin=our_robot.position,
        target=enemy_endgame_config.final_home_point,
        state=state,
        perspective=side,
        cfg=cfg,
    )
    enemy_home_enemy_rel_x, enemy_home_enemy_rel_y, enemy_home_enemy_rel_x_abs, enemy_home_enemy_rel_y_abs, enemy_home_enemy_rel_dist = _normalize_relative_geometry(
        origin=enemy_robot.position,
        target=enemy_endgame_config.final_home_point,
        state=state,
        perspective=side,
        cfg=cfg,
    )
    our_chill_rel_x, our_chill_rel_y, our_chill_rel_x_abs, our_chill_rel_y_abs, our_chill_rel_dist = _normalize_relative_geometry(
        origin=our_robot.position,
        target=endgame_config.chill_point,
        state=state,
        perspective=side,
        cfg=cfg,
    )
    our_chill_enemy_rel_x, our_chill_enemy_rel_y, our_chill_enemy_rel_x_abs, our_chill_enemy_rel_y_abs, our_chill_enemy_rel_dist = _normalize_relative_geometry(
        origin=enemy_robot.position,
        target=endgame_config.chill_point,
        state=state,
        perspective=side,
        cfg=cfg,
    )
    thermometer_rel_x, thermometer_rel_y, thermometer_rel_x_abs, thermometer_rel_y_abs, thermometer_rel_dist = _normalize_relative_geometry(
        origin=our_robot.position,
        target=state.thermometer.position,
        state=state,
        perspective=side,
        cfg=cfg,
    )
    thermometer_enemy_rel_x, thermometer_enemy_rel_y, thermometer_enemy_rel_x_abs, thermometer_enemy_rel_y_abs, thermometer_enemy_rel_dist = _normalize_relative_geometry(
        origin=enemy_robot.position,
        target=state.thermometer.position,
        state=state,
        perspective=side,
        cfg=cfg,
    )
    enemy_vel_x, enemy_vel_y, enemy_speed_norm = _normalized_velocity(
        previous_state,
        state,
        subject=side.opponent(),
        perspective=side,
        dt=dt,
        cfg=cfg,
    )

    global_features = {
        "time_remaining_norm": _safe_div(time_remaining, state.T_end),
        "time_to_main_pipeline_deadline_norm": _normalize_signed(
            endgame_config.main_pipeline_deadline - state.t,
            state.T_end,
        ),
        "time_to_chill_end_norm": _normalize_signed(
            endgame_config.chill_end - state.t,
            state.T_end,
        ),
        "after_main_pipeline_deadline": float(state.t >= endgame_config.main_pipeline_deadline),
        "after_chill_end": float(state.t >= endgame_config.chill_end),
        "in_last_30s": float(time_remaining <= 30.0),
        "in_last_20s": float(time_remaining <= 20.0),
        "in_last_10s": float(time_remaining <= 10.0),
        **_time_bin_features(state.t, state.T_end, num_bins=10),
        "our_x_norm": our_x,
        "our_y_norm": our_y,
        "enemy_x_norm": enemy_x,
        "enemy_y_norm": enemy_y,
        "enemy_rel_x_norm": enemy_rel_x,
        "enemy_rel_y_norm": enemy_rel_y,
        "enemy_rel_x_abs_norm": enemy_rel_x_abs,
        "enemy_rel_y_abs_norm": enemy_rel_y_abs,
        "enemy_rel_dist_norm": enemy_rel_dist,
        "our_home_rel_x_norm": our_home_rel_x,
        "our_home_rel_y_norm": our_home_rel_y,
        "our_home_rel_x_abs_norm": our_home_rel_x_abs,
        "our_home_rel_y_abs_norm": our_home_rel_y_abs,
        "our_home_rel_dist_norm": our_home_rel_dist,
        "our_home_enemy_rel_x_norm": our_home_enemy_rel_x,
        "our_home_enemy_rel_y_norm": our_home_enemy_rel_y,
        "our_home_enemy_rel_x_abs_norm": our_home_enemy_rel_x_abs,
        "our_home_enemy_rel_y_abs_norm": our_home_enemy_rel_y_abs,
        "our_home_enemy_rel_dist_norm": our_home_enemy_rel_dist,
        "enemy_home_rel_x_norm": enemy_home_rel_x,
        "enemy_home_rel_y_norm": enemy_home_rel_y,
        "enemy_home_rel_x_abs_norm": enemy_home_rel_x_abs,
        "enemy_home_rel_y_abs_norm": enemy_home_rel_y_abs,
        "enemy_home_rel_dist_norm": enemy_home_rel_dist,
        "enemy_home_enemy_rel_x_norm": enemy_home_enemy_rel_x,
        "enemy_home_enemy_rel_y_norm": enemy_home_enemy_rel_y,
        "enemy_home_enemy_rel_x_abs_norm": enemy_home_enemy_rel_x_abs,
        "enemy_home_enemy_rel_y_abs_norm": enemy_home_enemy_rel_y_abs,
        "enemy_home_enemy_rel_dist_norm": enemy_home_enemy_rel_dist,
        "our_chill_rel_x_norm": our_chill_rel_x,
        "our_chill_rel_y_norm": our_chill_rel_y,
        "our_chill_rel_x_abs_norm": our_chill_rel_x_abs,
        "our_chill_rel_y_abs_norm": our_chill_rel_y_abs,
        "our_chill_rel_dist_norm": our_chill_rel_dist,
        "our_chill_enemy_rel_x_norm": our_chill_enemy_rel_x,
        "our_chill_enemy_rel_y_norm": our_chill_enemy_rel_y,
        "our_chill_enemy_rel_x_abs_norm": our_chill_enemy_rel_x_abs,
        "our_chill_enemy_rel_y_abs_norm": our_chill_enemy_rel_y_abs,
        "our_chill_enemy_rel_dist_norm": our_chill_enemy_rel_dist,
        "thermometer_rel_x_norm": thermometer_rel_x,
        "thermometer_rel_y_norm": thermometer_rel_y,
        "thermometer_rel_x_abs_norm": thermometer_rel_x_abs,
        "thermometer_rel_y_abs_norm": thermometer_rel_y_abs,
        "thermometer_rel_dist_norm": thermometer_rel_dist,
        "thermometer_enemy_rel_x_norm": thermometer_enemy_rel_x,
        "thermometer_enemy_rel_y_norm": thermometer_enemy_rel_y,
        "thermometer_enemy_rel_x_abs_norm": thermometer_enemy_rel_x_abs,
        "thermometer_enemy_rel_y_abs_norm": thermometer_enemy_rel_y_abs,
        "thermometer_enemy_rel_dist_norm": thermometer_enemy_rel_dist,
        "our_load_norm": _safe_div(our_robot.load, max(1.0, cfg.load_scale)),
        "enemy_vel_x_norm": enemy_vel_x,
        "enemy_vel_y_norm": enemy_vel_y,
        "enemy_speed_norm": enemy_speed_norm,
        "enemy_max_speed_seen_norm": _normalize_speed_seen(
            state.max_observed_speed_by_side.get(side.opponent(), 0.0),
            cfg,
        ),
        "our_endgame_started": float(state.endgame_started_for(side)),
        "enemy_endgame_started": float(state.endgame_started_for(side.opponent())),
        "thermometer_done_for_us": float(state.thermometer.is_done_for_side(side)),
        "thermometer_done_for_enemy": float(state.thermometer.is_done_for_side(side.opponent())),
        "thermometer_lane_clear_for_us": float(thermometer_lane_is_clear(state, side)),
        "thermometer_available_for_us": float(
            not state.thermometer.is_done_for_side(side)
            and state.t < endgame_config.main_pipeline_deadline
            and thermometer_lane_is_clear(state, side)
        ),
        "thermometer_available_for_enemy": float(
            not state.thermometer.is_done_for_side(side.opponent())
            and state.t < state.endgame_config_for(side.opponent()).main_pipeline_deadline
            and thermometer_lane_is_clear(state, side.opponent())
        ),
        "time_since_our_lane_clear_change_norm": _time_since_change_norm(
            state.t,
            state.T_end,
            state.thermometer_lane_clear_change_time_by_side.get(side, 0.0),
        ),
        "time_since_enemy_lane_clear_change_norm": _time_since_change_norm(
            state.t,
            state.T_end,
            state.thermometer_lane_clear_change_time_by_side.get(side.opponent(), 0.0),
        ),
        "time_since_thermometer_state_change_norm": _time_since_change_norm(
            state.t,
            state.T_end,
            state.thermometer_last_state_change_time,
        ),
    }

    source_features: dict[str, dict[str, float]] = {}
    for source_id in state.sources:
        policy_id = normalized_source_id(source_id, side)
        source = state.sources[source_id]
        source_enemy_rel_x, source_enemy_rel_y, source_enemy_rel_x_abs, source_enemy_rel_y_abs, source_enemy_rel_dist = _normalize_relative_geometry(
            origin=enemy_robot.position,
            target=source.position,
            state=state,
            perspective=side,
            cfg=cfg,
        )
        source_rel_x, source_rel_y, source_rel_x_abs, source_rel_y_abs, source_rel_dist = _normalize_relative_geometry(
            origin=our_robot.position,
            target=source.position,
            state=state,
            perspective=side,
            cfg=cfg,
        )
        source_features[policy_id] = {
            "rel_x_norm": source_rel_x,
            "rel_y_norm": source_rel_y,
            "rel_x_abs_norm": source_rel_x_abs,
            "rel_y_abs_norm": source_rel_y_abs,
            "rel_dist_norm": source_rel_dist,
            "enemy_rel_x_norm": source_enemy_rel_x,
            "enemy_rel_y_norm": source_enemy_rel_y,
            "enemy_rel_x_abs_norm": source_enemy_rel_x_abs,
            "enemy_rel_y_abs_norm": source_enemy_rel_y_abs,
            "enemy_rel_dist_norm": source_enemy_rel_dist,
            "available_items_norm": _safe_div(source.available_items, max(1.0, cfg.source_items_scale)),
            "available_now": float(source.is_available(state.t)),
            "state_untouched": float(source.state is SourceState.UNTOUCHED),
            "state_disturbed": float(source.state is SourceState.DISTURBED),
            "state_empty": float(source.state is SourceState.EMPTY),
            "map_footprint_enabled": float(source.map_footprint_enabled),
            "last_items_delta_norm": _safe_div(
                state.source_last_items_delta_by_id.get(source_id, 0.0),
                max(1.0, cfg.source_items_scale),
            ),
            "time_since_last_change_norm": _time_since_change_norm(
                state.t,
                state.T_end,
                state.source_last_change_time_by_id.get(source_id, 0.0),
            ),
            "last_change_was_disturb_like": float(
                state.source_last_change_was_disturb_like_by_id.get(source_id, False)
            ),
        }

    deposit_features: dict[str, dict[str, float]] = {}
    for deposit_id in state.deposits:
        policy_id = normalized_deposit_id(deposit_id, side)
        deposit = state.deposits[deposit_id]
        deposit_enemy_rel_x, deposit_enemy_rel_y, deposit_enemy_rel_x_abs, deposit_enemy_rel_y_abs, deposit_enemy_rel_dist = _normalize_relative_geometry(
            origin=enemy_robot.position,
            target=deposit.position,
            state=state,
            perspective=side,
            cfg=cfg,
        )
        deposit_rel_x, deposit_rel_y, deposit_rel_x_abs, deposit_rel_y_abs, deposit_rel_dist = _normalize_relative_geometry(
            origin=our_robot.position,
            target=deposit.position,
            state=state,
            perspective=side,
            cfg=cfg,
        )
        our_points = float(deposit_zone_points(deposit, side))
        enemy_points = float(deposit_zone_points(deposit, side.opponent()))
        score_diff = our_points - enemy_points
        our_items = float(deposit.items_for_side(side))
        enemy_items = float(deposit.items_for_side(side.opponent()))
        last_blue_score_delta = float(state.deposit_last_blue_score_delta_by_id.get(deposit_id, 0.0))
        last_yellow_score_delta = float(state.deposit_last_yellow_score_delta_by_id.get(deposit_id, 0.0))
        last_score_diff_delta = (
            last_blue_score_delta - last_yellow_score_delta
            if side is Side.BLUE
            else last_yellow_score_delta - last_blue_score_delta
        )
        last_actor = state.deposit_last_actor_by_id.get(deposit_id)
        features = {
            "rel_x_norm": deposit_rel_x,
            "rel_y_norm": deposit_rel_y,
            "rel_x_abs_norm": deposit_rel_x_abs,
            "rel_y_abs_norm": deposit_rel_y_abs,
            "rel_dist_norm": deposit_rel_dist,
            "enemy_rel_x_norm": deposit_enemy_rel_x,
            "enemy_rel_y_norm": deposit_enemy_rel_y,
            "enemy_rel_x_abs_norm": deposit_enemy_rel_x_abs,
            "enemy_rel_y_abs_norm": deposit_enemy_rel_y_abs,
            "enemy_rel_dist_norm": deposit_enemy_rel_dist,
            "our_items_norm": _safe_div(our_items, max(1.0, cfg.deposit_items_scale)),
            "enemy_items_norm": _safe_div(enemy_items, max(1.0, cfg.deposit_items_scale)),
            "total_items_norm": _safe_div(float(deposit.total_items()), max(1.0, cfg.deposit_items_scale)),
            "our_points_norm": _normalize_zone_points_delta(our_points, cfg),
            "enemy_points_norm": _normalize_zone_points_delta(enemy_points, cfg),
            "score_diff_norm": _normalize_zone_points_delta(score_diff, cfg),
            "occupied_by_our": float(deposit.occupied_by is side),
            "occupied_by_enemy": float(deposit.occupied_by is side.opponent()),
            "occupied_by_none": float(deposit.occupied_by is None),
            "kind_home": float(deposit.kind is DepositType.HOME),
            "kind_storage": float(deposit.kind is DepositType.STORAGE),
            "protected_for_our": float(deposit.protected_for is side),
            "protected_for_enemy": float(deposit.protected_for is side.opponent()),
            "map_footprint_enabled": float(deposit.map_footprint_enabled),
            "attack_delta_norm": _normalize_zone_points_delta(
                _attack_score_diff_delta(deposit, side),
                cfg,
            ),
            "last_score_diff_delta_norm": _normalize_zone_points_delta(last_score_diff_delta, cfg),
            "time_since_last_score_change_norm": _time_since_change_norm(
                state.t,
                state.T_end,
                state.deposit_last_score_change_time_by_id.get(deposit_id, 0.0),
            ),
            "last_change_by_our": float(last_actor is side),
            "last_change_by_enemy": float(last_actor is side.opponent()),
        }
        for deposit_count in range(1, 5):
            delta_raw = _deposit_score_diff_delta(deposit, side, deposit_count, our_robot.load)
            features[f"deposit_x{deposit_count}_delta_norm"] = _normalize_zone_points_delta(delta_raw, cfg)
            features[f"deposit_x{deposit_count}_valid"] = float(
                deposit_count <= our_robot.load
                and deposit_count <= deposit_max_count_for_side(deposit, side, our_robot.load)
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


def _normalize_relative_point(
    *,
    origin: tuple[float, float],
    target: tuple[float, float],
    state: GameState,
    perspective: Side,
    cfg: RLObservationConfig,
) -> tuple[float, float]:
    dx = target[0] - origin[0]
    if cfg.mirror_x_for_yellow and perspective is Side.YELLOW:
        dx = -dx
    dy = target[1] - origin[1]
    field_width, field_height = state.field_size
    return (
        _normalize_signed(dx, field_width / 2.0),
        _normalize_signed(dy, field_height / 2.0),
    )


def _normalize_relative_geometry(
    *,
    origin: tuple[float, float],
    target: tuple[float, float],
    state: GameState,
    perspective: Side,
    cfg: RLObservationConfig,
) -> tuple[float, float, float, float, float]:
    dx = target[0] - origin[0]
    if cfg.mirror_x_for_yellow and perspective is Side.YELLOW:
        dx = -dx
    dy = target[1] - origin[1]
    field_width, field_height = state.field_size
    distance_scale = max(hypot(field_width, field_height), 1e-6)
    return (
        _normalize_signed(dx, field_width / 2.0),
        _normalize_signed(dy, field_height / 2.0),
        _normalize_signed(abs(dx), field_width / 2.0),
        _normalize_signed(abs(dy), field_height / 2.0),
        _clip(hypot(dx, dy) / distance_scale, 0.0, 1.0),
    )


def _normalized_velocity(
    previous_state: PreviousStateView | None,
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
    vx, vy = _apply_enemy_velocity_measurement_noise(
        vx=vx,
        vy=vy,
        previous_state=previous_state,
        state=state,
        subject=subject,
        perspective=perspective,
        dt=dt,
        cfg=cfg,
    )
    speed = hypot(vx, vy)
    if speed < cfg.velocity_deadband_mps:
        return 0.0, 0.0, 0.0
    denom = max(cfg.velocity_normalizer_mps, 1e-6)
    return (
        _clip(vx / denom, -1.0, 1.0),
        _clip(vy / denom, -1.0, 1.0),
        _clip(speed / denom, 0.0, 1.0),
    )


def _normalize_speed_seen(speed_mps: float, cfg: RLObservationConfig) -> float:
    if speed_mps < cfg.velocity_deadband_mps:
        return 0.0
    denom = max(cfg.velocity_normalizer_mps, 1e-6)
    return _clip(speed_mps / denom, 0.0, 1.0)


def _normalize_zone_points_delta(value: float, cfg: RLObservationConfig) -> float:
    denom = max(cfg.zone_points_scale, 1e-6)
    return _clip(value / denom, -1.0, 1.0)


def _apply_enemy_velocity_measurement_noise(
    *,
    vx: float,
    vy: float,
    previous_state: PreviousStateView,
    state: GameState,
    subject: Side,
    perspective: Side,
    dt: float,
    cfg: RLObservationConfig,
) -> tuple[float, float]:
    base_std = max(0.0, cfg.enemy_velocity_noise_std_mps)
    leak_fraction = max(0.0, cfg.enemy_velocity_self_motion_leak_fraction)
    leak_duration = max(0.0, cfg.enemy_velocity_self_motion_leak_duration_s)
    if base_std <= 0.0 and (leak_fraction <= 0.0 or leak_duration <= 0.0):
        return vx, vy

    subject_position = state.robot_for_side(subject).position
    seed = (
        int(cfg.observation_noise_seed) * 1_000_003
        ^ int(round(state.t * 1_000.0)) * 9_176
        ^ int(round(subject_position[0] * 1_000.0)) * 7_919
        ^ int(round(subject_position[1] * 1_000.0)) * 10_213
        ^ (11 if subject is Side.BLUE else 17)
        ^ (23 if perspective is Side.BLUE else 29)
    ) & 0xFFFFFFFF
    rng = random.Random(seed)
    if base_std > 0.0:
        vx += rng.gauss(0.0, base_std)
        vy += rng.gauss(0.0, base_std)
    if leak_fraction > 0.0 and leak_duration > 0.0:
        last_motion_start = state.last_motion_start_time_by_side.get(perspective)
        if last_motion_start is not None and state.t - last_motion_start <= leak_duration:
            current_our = state.robot_for_side(perspective).position
            previous_our = previous_state.robot_for_side(perspective).position
            our_vx = (current_our[0] - previous_our[0]) / dt
            our_vy = (current_our[1] - previous_our[1]) / dt
            if cfg.mirror_x_for_yellow and perspective is Side.YELLOW:
                our_vx = -our_vx
            vx -= leak_fraction * our_vx
            vy -= leak_fraction * our_vy
    return vx, vy


def _deposit_score_diff_delta(
    deposit: DepositPoint,
    side: Side,
    deposit_count: int,
    available_load: int,
) -> float:
    max_count = deposit_max_count_for_side(deposit, side, available_load)
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


def _time_since_change_norm(current_time: float, total_time: float, last_change_time: float) -> float:
    if total_time <= 0.0:
        return 0.0
    delta = max(0.0, current_time - last_change_time)
    return _clip(delta / total_time, 0.0, 1.0)


def _time_bin_features(current_time: float, total_time: float, *, num_bins: int) -> dict[str, float]:
    count = max(1, int(num_bins))
    features = {f"time_bin_{index}": 0.0 for index in range(count)}
    if total_time <= 0.0:
        features["time_bin_0"] = 1.0
        return features
    clamped_time = _clip(current_time, 0.0, total_time)
    bin_index = min(int(clamped_time / total_time * count), count - 1)
    features[f"time_bin_{bin_index}"] = 1.0
    return features


