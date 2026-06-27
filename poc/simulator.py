from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from poc.actions import Action, ActionType
from poc.endgame import EndgameConfig
from poc.entities import DepositType, PushState, Side, SourceState
from poc.external_events import EventType, apply_external_event
from poc.game_state import GameState
from poc.geometry import advance_along_path, distance, interpolate, point_to_segment_distance, segment_to_segment_distance
from poc.opponent_policy import OpponentPolicy
from poc.planner import PlanningDecision, UtilityPlanner
from poc.policy_mapping import (
    normalized_action_label,
    normalized_target_id,
    policy_metadata_for_deposit,
    policy_metadata_for_source,
)
from poc.rl_infra import (
    DEFAULT_ACTION_SPACE,
    RLObservationConfig,
    RLPolicyStep,
    RLTransition,
    build_rl_observation,
    build_rl_policy_step,
    save_transition_dataset,
)
from poc.scoring import deposit_max_count_for_side, deposit_zone_points, home_remaining_capacity


@dataclass(slots=True)
class HistoryEntry:
    time: float
    our_position: tuple[float, float]
    enemy_position: tuple[float, float]
    our_score: int
    enemy_score: int
    our_load: int
    enemy_load: int
    source_states: dict[int, dict[str, object]]
    deposit_states: dict[int, dict[str, object]]
    thermometer_state: str
    thermometer_doing_blue: bool
    thermometer_doing_yellow: bool
    mars_states: dict[str, list[dict[str, object]]]


@dataclass(slots=True)
class ActionLogEntry:
    time: float
    side: Side
    action: str
    policy_action: str
    target_id: int | None
    policy_target_id: str | None
    expected_duration: float
    score: float


@dataclass(slots=True)
class SimulationEvent:
    time: float
    side: Side | None
    kind: str
    note: str


@dataclass(slots=True)
class ActivePhase:
    kind: str
    duration: float
    waypoints: tuple[tuple[float, float], ...] = ()
    anchor: tuple[float, float] | None = None
    start_position: tuple[float, float] | None = None
    clear_source_ids: tuple[int, ...] = ()
    clear_deposit_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class ActiveAction:
    action: Action
    side: Side
    start_time: float
    phases: list[ActivePhase]
    total_duration: float
    contact_registered: bool = False
    completed_phases: int = 0


@dataclass(frozen=True, slots=True)
class RobotTickState:
    position: tuple[float, float]


@dataclass(frozen=True, slots=True)
class TickStateSnapshot:
    t: float
    blue_robot: RobotTickState
    yellow_robot: RobotTickState

    def robot_for_side(self, side: Side) -> RobotTickState:
        return self.blue_robot if side is Side.BLUE else self.yellow_robot


@dataclass(slots=True)
class MatchResult:
    scenario_name: str
    summary: dict[str, float | int | bool | str]
    history: list[HistoryEntry]
    action_log: list[ActionLogEntry]
    rl_transitions: list[RLTransition]
    planner_debug: list[dict[str, object]]
    events: list[SimulationEvent]
    field_size: tuple[float, float]
    sources: dict[int, dict[str, object]]
    deposits: dict[int, dict[str, object]]
    thermometer: dict[str, object]
    endgame: dict[str, dict[str, object]]
    mars: dict[str, list[dict[str, object]]]
    our_side: str


@dataclass(slots=True)
class PendingRLTransition:
    time: float
    step: RLPolicyStep
    chosen_action: str
    chosen_action_index: int
    score_diff_before: float


@dataclass(slots=True)
class StoppedMarsState:
    delay_duration: float = 0.0
    blocked_since: float | None = None


class ActionSelector(Protocol):
    name: str

    def select_action(
        self,
        *,
        state: GameState,
        planner: UtilityPlanner,
        side: Side,
        ranked_actions: list[Action],
        policy_step: RLPolicyStep,
    ) -> Action:
        ...


class Simulator:
    def __init__(
        self,
        state: GameState,
        scenario_name: str,
        opponent_policy: OpponentPolicy,
        planner: UtilityPlanner | None = None,
        dt: float = 0.5,
        action_selectors: dict[Side, ActionSelector] | None = None,
        rl_observation_config: RLObservationConfig | None = None,
        thermometer_reward_bonus: float = 3.0,
        terminal_win_bonus: float = 2.0,
        terminal_draw_bonus: float = 0.0,
        terminal_loss_bonus: float = -2.0,
    ) -> None:
        self.state = state
        self.scenario_name = scenario_name
        self.opponent_policy = opponent_policy
        self.planner = planner or UtilityPlanner()
        self.dt = dt
        self.action_selectors = dict(action_selectors or {})
        self.rl_observation_config = rl_observation_config or RLObservationConfig()
        self.thermometer_reward_bonus = thermometer_reward_bonus
        self.terminal_win_bonus = terminal_win_bonus
        self.terminal_draw_bonus = terminal_draw_bonus
        self.terminal_loss_bonus = terminal_loss_bonus
        self._event_index = 0
        self._active_actions: dict[Side, ActiveAction | None] = {Side.BLUE: None, Side.YELLOW: None}
        self._history: list[HistoryEntry] = []
        self._action_log: list[ActionLogEntry] = []
        self._rl_transitions: list[RLTransition] = []
        self._pending_rl: dict[Side, PendingRLTransition | None] = {Side.BLUE: None, Side.YELLOW: None}
        self._planner_debug: list[dict[str, object]] = []
        self._events: list[SimulationEvent] = []
        self._replan_count = 0
        self._lost_target_count = 0
        self._previous_tick_state: TickStateSnapshot | None = None
        self._last_runtime_replan_time_by_side: dict[Side, float] = {Side.BLUE: -1e9, Side.YELLOW: -1e9}
        self._stopped_mars_by_name: dict[str, StoppedMarsState] = {}
        self._mars_collision_pairs: set[tuple[Side, str]] = set()
        self._initialize_temporal_tracking()

    def run(self) -> MatchResult:
        while self.state.t < self.state.T_end:
            self._apply_pending_events()

            for side in (Side.BLUE, Side.YELLOW):
                self._update_active_action(side)
                if self._active_actions[side] is None and self.state.t < self.state.T_end:
                    self._assign_next_action(side)

            self._update_mars_interactions()
            self._update_observed_speed_tracking()
            self._snapshot()
            self._previous_tick_state = self._capture_tick_state_snapshot()
            self.state.t = round(min(self.state.T_end, self.state.t + self.dt), 6)

        self._apply_pending_events()
        for side in (Side.BLUE, Side.YELLOW):
            self._update_active_action(side)
        self._update_mars_interactions()
        self._finalize_match()
        self._finalize_rl_transitions()
        return MatchResult(
            scenario_name=self.scenario_name,
            summary=self._build_summary(),
            history=self._history,
            action_log=self._action_log,
            rl_transitions=self._rl_transitions,
            planner_debug=self._planner_debug,
            events=self._events,
            field_size=self.state.field_size,
            sources={
                source_id: _serialize_with_policy_metadata(source, policy_metadata_for_source(source_id))
                for source_id, source in self.state.sources.items()
            },
            deposits={
                deposit_id: _serialize_with_policy_metadata(deposit, policy_metadata_for_deposit(deposit_id))
                for deposit_id, deposit in self.state.deposits.items()
            },
            thermometer=_serialize(self.state.thermometer),
            endgame={side.value: _serialize(config) for side, config in self.state.endgame_by_side.items()},
            mars={
                side.value: [
                    {
                        **_serialize(mars),
                        "position": self._mars_position(mars, self.state.T_end),
                        "released": mars.is_released(self.state.T_end),
                        "arrived": self._mars_has_arrived(mars, self.state.T_end),
                        "stopped": mars.name in self._stopped_mars_by_name,
                        "collided_by_blue": self._mars_collided(mars, Side.BLUE),
                        "collided_by_yellow": self._mars_collided(mars, Side.YELLOW),
                    }
                    for mars in self.state.mars_by_side.get(side, ())
                ]
                for side in (Side.BLUE, Side.YELLOW)
            },
            our_side=self.state.our_side.value,
        )

    def _apply_pending_events(self) -> None:
        while self._event_index < len(self.state.external_events):
            event = self.state.external_events[self._event_index]
            if event.time > self.state.t:
                break
            source_before: tuple[int, SourceState] | None = None
            if (
                event.event_type in {
                    EventType.SET_SOURCE_AVAILABLE,
                    EventType.SET_SOURCE_EMPTY,
                    EventType.SET_SOURCE_DISTURBED,
                }
                and event.target_id is not None
                and event.target_id in self.state.sources
            ):
                source = self.state.sources[event.target_id]
                source_before = (source.available_items, source.state)
            thermometer_state_before = self.state.thermometer.state
            note = apply_external_event(self.state, event)
            if source_before is not None and event.target_id is not None:
                self._record_source_change(event.target_id, *source_before)
            self._record_thermometer_state_change(thermometer_state_before)
            if source_before is not None or self.state.thermometer.state is not thermometer_state_before:
                self._refresh_thermometer_lane_clear_tracking()
            self._events.append(SimulationEvent(time=self.state.t, side=None, kind="external_event", note=note))
            self._event_index += 1

    def _initialize_temporal_tracking(self) -> None:
        self._refresh_thermometer_lane_clear_tracking(force=True)

    def _record_source_change(
        self,
        source_id: int,
        previous_items: int,
        previous_state: SourceState,
    ) -> None:
        source = self.state.sources[source_id]
        if source.available_items == previous_items and source.state is previous_state:
            return
        self.state.source_last_items_delta_by_id[source_id] = float(source.available_items - previous_items)
        self.state.source_last_change_time_by_id[source_id] = float(self.state.t)
        self.state.source_last_change_was_disturb_like_by_id[source_id] = source.state is SourceState.DISTURBED

    def _record_deposit_score_change(
        self,
        deposit_id: int,
        before_blue: int,
        before_yellow: int,
        actor: Side | None,
    ) -> None:
        deposit = self.state.deposits[deposit_id]
        after_blue = deposit_zone_points(deposit, Side.BLUE)
        after_yellow = deposit_zone_points(deposit, Side.YELLOW)
        blue_delta = float(after_blue - before_blue)
        yellow_delta = float(after_yellow - before_yellow)
        if blue_delta == 0.0 and yellow_delta == 0.0:
            return
        self.state.deposit_last_blue_score_delta_by_id[deposit_id] = blue_delta
        self.state.deposit_last_yellow_score_delta_by_id[deposit_id] = yellow_delta
        self.state.deposit_last_score_change_time_by_id[deposit_id] = float(self.state.t)
        self.state.deposit_last_actor_by_id[deposit_id] = actor

    def _record_thermometer_state_change(self, previous_state) -> None:
        if self.state.thermometer.state is previous_state:
            return
        self.state.thermometer_last_state_change_time = float(self.state.t)

    def _thermometer_lane_is_clear_for_side(self, side: Side) -> bool:
        blocking_source_id = self.state.thermometer.blocking_source_id_for_side(side)
        source = self.state.sources.get(blocking_source_id)
        blocking_source_clear = (
            source is None
            or source.state is SourceState.EMPTY
            or source.available_items <= 0
        )
        zone_10 = self.state.deposits.get(10)
        zone_10_clear = zone_10 is None or zone_10.total_items() == 0
        blocking_deposit_id = self.state.thermometer.blocking_deposit_id_for_side(side)
        blocking_deposit = self.state.deposits.get(blocking_deposit_id)
        blocking_deposit_clear = blocking_deposit is None or blocking_deposit.total_items() == 0
        return blocking_source_clear and zone_10_clear and blocking_deposit_clear

    def _refresh_thermometer_lane_clear_tracking(self, *, force: bool = False) -> None:
        for side in (Side.BLUE, Side.YELLOW):
            current = self._thermometer_lane_is_clear_for_side(side)
            previous = self.state.thermometer_lane_clear_by_side.get(side, current)
            self.state.thermometer_lane_clear_by_side[side] = current
            if force or current != previous:
                self.state.thermometer_lane_clear_change_time_by_side[side] = float(self.state.t)

    def _capture_tick_state_snapshot(self) -> TickStateSnapshot:
        return TickStateSnapshot(
            t=float(self.state.t),
            blue_robot=RobotTickState(position=self.state.robot_for_side(Side.BLUE).position),
            yellow_robot=RobotTickState(position=self.state.robot_for_side(Side.YELLOW).position),
        )

    def _update_observed_speed_tracking(self) -> None:
        if self._previous_tick_state is None or self.dt <= 0.0:
            return
        motion_start_threshold_mps = 0.02
        for side in (Side.BLUE, Side.YELLOW):
            current = self.state.robot_for_side(side).position
            previous = self._previous_tick_state.robot_for_side(side).position
            speed = distance(current, previous) / self.dt
            previous_speed = self.state.observed_speed_by_side.get(side, 0.0)
            if previous_speed <= motion_start_threshold_mps and speed > motion_start_threshold_mps:
                self.state.last_motion_start_time_by_side[side] = self.state.t
            self.state.observed_speed_by_side[side] = speed
            if speed > self.state.max_observed_speed_by_side[side]:
                self.state.max_observed_speed_by_side[side] = speed

    def _update_active_action(self, side: Side) -> None:
        active = self._active_actions[side]
        if active is None:
            return

        robot = self.state.robot_for_side(side)
        previous_position = robot.position
        elapsed = max(0.0, self.state.t - active.start_time)
        if self._maybe_replan_active_travel(active, elapsed, previous_position):
            return
        robot.position = self._position_during_action(active, elapsed, robot.position)

        if self._should_pause_for_robot_separation(active, previous_position, robot.position):
            robot.position = previous_position
            if self.state.t - self._last_runtime_replan_time_by_side[active.side] < max(self.dt, 0.5):
                active.start_time = round(active.start_time + self.dt, 6)
                return
            self._cancel_active_action_for_runtime_replan(
                active,
                note=f"cancelled {active.action.label} because robot separation blocked progress",
            )
            return

        self._maybe_register_source_contact(active)

        if self._should_cancel_action(active):
            if active.action.type is ActionType.DO_THERMOMETER:
                self.state.thermometer.clear_doing_for_side(side)
            self._active_actions[side] = None
            robot.current_action = None
            robot.current_target_id = None
            self._replan_count += 1 if side is self.state.our_side else 0
            self._lost_target_count += 1 if side is self.state.our_side else 0
            self._events.append(
                SimulationEvent(
                    time=self.state.t,
                    side=side,
                    kind="replan",
                    note=f"cancelled {active.action.label} because target became invalid",
                )
            )
            return

        if self._should_pause_for_contested_pick(active, elapsed):
            active.start_time = round(active.start_time + self.dt, 6)
            return

        self._advance_completed_phases(active, elapsed)

        if elapsed + 1e-9 < active.total_duration:
            return

        robot.position = self._final_position(active, robot.position)
        self._apply_action_effects(side, active.action)
        robot.current_action = None
        robot.current_target_id = None
        self._active_actions[side] = None

    def _maybe_replan_active_travel(
        self,
        active: ActiveAction,
        elapsed: float,
        current_position: tuple[float, float],
    ) -> bool:
        phase_index = self._phase_index_at_elapsed(active, elapsed)
        if phase_index is None:
            return False
        phase = active.phases[phase_index]
        if phase.kind != "travel" or not phase.waypoints:
            return False
        phase_waypoints = self._remaining_travel_waypoints(active, phase_index, current_position)
        lookahead_waypoints = self._truncate_waypoints_to_distance(
            current_position,
            phase_waypoints,
            self.planner.timing.local_replan_distance,
        )
        if not lookahead_waypoints:
            return False
        if self.state.t - self._last_runtime_replan_time_by_side[active.side] < max(self.dt, 0.5):
            return False
        ignored_source_ids, ignored_deposit_ids = self._runtime_obstacle_overrides(active, elapsed)
        enemy_position = self.state.robot_for_side(active.side.opponent()).position
        enemy_blocks_path = (
            distance(current_position, enemy_position) <= self.planner.timing.local_replan_distance
            and self._path_intersects_circle(current_position, lookahead_waypoints, enemy_position, self.planner.timing.robot_separation_radius)
        )
        local_path_blocked = self._local_path_hits_obstacle(
            active.side,
            current_position,
            lookahead_waypoints,
            allow_goal_occupied=self._phase_allows_occupied_goal(active, phase_index),
            ignored_source_ids=ignored_source_ids,
            ignored_deposit_ids=ignored_deposit_ids,
        )
        if not enemy_blocks_path and not local_path_blocked:
            return False
        if enemy_blocks_path and not self._robot_must_yield(active):
            return False
        self._cancel_active_action_for_runtime_replan(
            active,
            note=f"cancelled {active.action.label} because local route became invalid",
        )
        return True

    def _cancel_active_action_for_runtime_replan(self, active: ActiveAction, *, note: str) -> None:
        if active.action.type is ActionType.DO_THERMOMETER:
            self.state.thermometer.clear_doing_for_side(active.side)
        if active.action.type is ActionType.START_ENDGAME:
            self.state.set_endgame_started(active.side, False)
        self._active_actions[active.side] = None
        robot = self.state.robot_for_side(active.side)
        robot.current_action = None
        robot.current_target_id = None
        self._events.append(
            SimulationEvent(
                time=self.state.t,
                side=active.side,
                kind="runtime_replan",
                note=note,
            )
        )
        self._last_runtime_replan_time_by_side[active.side] = self.state.t
        self._replan_count += 1 if active.side is self.state.our_side else 0
        self._lost_target_count += 1 if active.side is self.state.our_side else 0

    def _remaining_travel_waypoints(
        self,
        active: ActiveAction,
        phase_index: int,
        current_position: tuple[float, float],
    ) -> tuple[tuple[float, float], ...]:
        phase = active.phases[phase_index]
        waypoints = tuple(phase.waypoints)
        if len(waypoints) <= 1:
            return waypoints
        if active.action.type is ActionType.START_ENDGAME and phase_index == 2:
            nearest_index = min(
                range(len(waypoints)),
                key=lambda index: distance(current_position, waypoints[index]),
            )
            return waypoints[nearest_index:]
        start_position = phase.start_position or current_position
        return self._remaining_waypoint_tail_from_position(start_position, waypoints, current_position)

    def _remaining_waypoint_tail_from_position(
        self,
        start_position: tuple[float, float],
        waypoints: tuple[tuple[float, float], ...],
        current_position: tuple[float, float],
    ) -> tuple[tuple[float, float], ...]:
        path_points = (start_position, *waypoints)
        next_waypoint_index = 0
        nearest_segment_distance = math.inf
        for path_index in range(len(path_points) - 1):
            segment_distance = point_to_segment_distance(
                current_position,
                path_points[path_index],
                path_points[path_index + 1],
            )
            if segment_distance < nearest_segment_distance:
                nearest_segment_distance = segment_distance
                next_waypoint_index = path_index
        return waypoints[next_waypoint_index:]

    def _assign_next_action(self, side: Side) -> None:
        selector = self.action_selectors.get(side)
        if side is self.state.our_side and selector is None:
            decision = self.planner.plan(self.state, side)
            action = decision.chosen_action
            ranked_actions = decision.ranked_actions
            self._planner_debug.append(decision.debug_payload(self.state.t, side))
        else:
            ranked_actions = self.planner.rank_actions(self.state, side)
            policy_step = build_rl_policy_step(
                self.state,
                side,
                ranked_actions=ranked_actions,
                previous_state=self._previous_tick_state,
                dt=self.dt,
                config=self.rl_observation_config,
                action_space=DEFAULT_ACTION_SPACE,
            )
            if selector is None:
                action = self.opponent_policy.choose_action(self.state, self.planner, side)
            else:
                action = selector.select_action(
                    state=self.state,
                    planner=self.planner,
                    side=side,
                    ranked_actions=ranked_actions,
                    policy_step=policy_step,
                )
                if side is self.state.our_side:
                    decision = PlanningDecision(
                        chosen_action=action,
                        ranked_actions=ranked_actions,
                        reason=f"selector:{selector.name}",
                    )
                    self._planner_debug.append(decision.debug_payload(self.state.t, side))
        if side is self.state.our_side and selector is None:
            policy_step = build_rl_policy_step(
                self.state,
                side,
                ranked_actions=ranked_actions,
                previous_state=self._previous_tick_state,
                dt=self.dt,
                config=self.rl_observation_config,
                action_space=DEFAULT_ACTION_SPACE,
            )
        chosen_policy_action = normalized_action_label(action, side)
        chosen_policy_action_index = DEFAULT_ACTION_SPACE.encode(chosen_policy_action)
        self._finalize_pending_rl_transition(
            side,
            next_step=policy_step,
            score_diff_after=self._score_diff_for_side(side),
            done=False,
        )

        active_action = self._activate_action(side, action)
        self._active_actions[side] = active_action
        robot = self.state.robot_for_side(side)
        robot.current_action = action.type.value
        robot.current_target_id = action.target_id
        if action.type is ActionType.DO_THERMOMETER:
            self.state.thermometer.mark_doing_for_side(side)

        if action.type is ActionType.START_ENDGAME:
            self.state.set_endgame_started(side, True)
        if action.type is ActionType.PLAY_TO_END:
            self.state.set_play_to_end_started(side, True)

        self._action_log.append(
            ActionLogEntry(
                time=self.state.t,
                side=side,
                action=action.label,
                policy_action=normalized_action_label(action, side),
                target_id=action.target_id,
                policy_target_id=normalized_target_id(action.target_id, action.type, side),
                expected_duration=round(action.expected_duration, 3),
                score=round(action.score, 3),
            )
        )
        self._pending_rl[side] = PendingRLTransition(
            time=round(self.state.t, 6),
            step=policy_step,
            chosen_action=chosen_policy_action,
            chosen_action_index=chosen_policy_action_index,
            score_diff_before=self._score_diff_for_side(side),
        )

    def _activate_action(self, side: Side, action: Action) -> ActiveAction:
        robot = self.state.robot_for_side(side)
        phases: list[ActivePhase] = []
        if action.type is ActionType.PLAY_TO_END:
            return ActiveAction(
                action=action,
                side=side,
                start_time=self.state.t,
                phases=[ActivePhase(kind="service", duration=max(action.service_duration, 1e-6), anchor=robot.position)],
                total_duration=max(action.expected_duration, 1e-6),
            )
        if action.type is ActionType.START_ENDGAME:
            endgame = self.state.endgame_config_for(side)
            travel_to_chill = float(action.metadata.get("travel_to_chill", 0.0))
            travel_to_chill_waypoints = _waypoint_tuple(
                action.metadata.get("travel_to_chill_waypoints"),
                fallback=(endgame.chill_point,),
            )
            wait_duration = float(action.metadata.get("wait_duration", 0.0))
            travel_home = float(action.metadata.get("travel_home", 0.0))
            travel_home_waypoints = _waypoint_tuple(
                action.metadata.get("travel_home_waypoints"),
                fallback=endgame.home_waypoints,
            )
            grip_rotate = float(action.metadata.get("grip_rotate", endgame.grip_rotate_duration))
            phases.append(
                ActivePhase(
                    kind="travel",
                    duration=travel_to_chill,
                    waypoints=travel_to_chill_waypoints,
                    start_position=robot.position,
                )
            )
            phases.append(
                ActivePhase(
                    kind="service",
                    duration=wait_duration,
                    anchor=endgame.chill_point,
                )
            )
            phases.append(
                ActivePhase(
                    kind="travel",
                    duration=travel_home,
                    waypoints=travel_home_waypoints,
                    start_position=endgame.chill_point,
                )
            )
            phases.append(
                ActivePhase(
                    kind="service",
                    duration=grip_rotate,
                    anchor=endgame.final_home_point,
                )
            )
        elif action.type is ActionType.WAIT:
            phases.append(ActivePhase(kind="service", duration=action.service_duration, anchor=robot.position))
        else:
            travel_segments = _travel_segments_from_metadata(action.metadata.get("travel_segments"))
            if travel_segments:
                current_start = robot.position
                for segment in travel_segments:
                    phases.append(
                        ActivePhase(
                            kind="travel",
                            duration=segment["duration"],
                            waypoints=segment["waypoints"],
                            start_position=current_start,
                            clear_source_ids=segment["clear_source_ids"],
                            clear_deposit_ids=segment["clear_deposit_ids"],
                        )
                    )
                    if segment["waypoints"]:
                        current_start = segment["waypoints"][-1]
            else:
                phases.append(
                    ActivePhase(
                        kind="travel",
                        duration=action.travel_duration,
                        waypoints=action.waypoints,
                        start_position=robot.position,
                    )
                )
            phases.append(
                ActivePhase(
                    kind="service",
                    duration=action.service_duration,
                    anchor=action.target_position or robot.position,
                )
            )
        total_duration = sum(phase.duration for phase in phases)
        return ActiveAction(action=action, side=side, start_time=self.state.t, phases=phases, total_duration=total_duration)

    def _position_during_action(
        self,
        active: ActiveAction,
        elapsed: float,
        fallback_position: tuple[float, float],
    ) -> tuple[float, float]:
        remaining = elapsed
        current_position = fallback_position
        for phase in active.phases:
            if remaining <= phase.duration + 1e-9:
                if phase.kind == "service":
                    return phase.anchor or current_position
                start = phase.start_position or current_position
                if phase.duration <= 0.0:
                    return phase.waypoints[-1] if phase.waypoints else start
                total_path_distance = sum(
                    distance(a, b)
                    for a, b in zip((start, *phase.waypoints[:-1]), phase.waypoints)
                )
                travelled = total_path_distance * (remaining / phase.duration)
                return advance_along_path(start, phase.waypoints, travelled)

            remaining -= phase.duration
            if phase.kind == "service":
                current_position = phase.anchor or current_position
            else:
                current_position = phase.waypoints[-1] if phase.waypoints else current_position
        return current_position

    def _final_position(self, active: ActiveAction, fallback_position: tuple[float, float]) -> tuple[float, float]:
        for phase in reversed(active.phases):
            if phase.kind == "service" and phase.anchor is not None:
                return phase.anchor
            if phase.kind == "travel" and phase.waypoints:
                return phase.waypoints[-1]
        return fallback_position

    def _should_cancel_action(self, active: ActiveAction) -> bool:
        action = active.action
        if action.type is ActionType.PICK and action.target_id is not None:
            source = self.state.sources[action.target_id]
            return not source.is_available(self.state.t)
        if action.type is ActionType.DEPOSIT and action.target_id is not None:
            deposit = self.state.deposits[action.target_id]
            robot = self.state.robot_for_side(active.side)
            requested = int(active.action.metadata.get("deposit_count", robot.load))
            return requested <= 0 or requested > deposit_max_count_for_side(deposit, active.side, robot.load)
        if action.type is ActionType.DO_THERMOMETER:
            blocking_source_id = self.state.thermometer.blocking_source_id_for_side(active.side)
            blocking_source = self.state.sources.get(blocking_source_id)
            zone_10 = self.state.deposits.get(10)
            blocking_deposit_id = self.state.thermometer.blocking_deposit_id_for_side(active.side)
            blocking_deposit = self.state.deposits.get(blocking_deposit_id)
            thermometer_blocked = (
                blocking_source is not None
                and blocking_source.state is not SourceState.EMPTY
                and blocking_source.available_items > 0
            )
            zone_10_blocked = zone_10 is not None and zone_10.total_items() > 0
            blocking_deposit_blocked = blocking_deposit is not None and blocking_deposit.total_items() > 0
            return (
                self.state.thermometer.is_done_for_side(active.side)
                or thermometer_blocked
                or zone_10_blocked
                or blocking_deposit_blocked
            )
        if action.type is ActionType.ATTACK_DEPOSIT and action.target_id is not None:
            deposit = self.state.deposits[action.target_id]
            return deposit.items_for_side(active.side.opponent()) <= 0
        return False

    def _should_pause_for_robot_separation(
        self,
        active: ActiveAction,
        previous_position: tuple[float, float],
        candidate_position: tuple[float, float],
    ) -> bool:
        if candidate_position == previous_position:
            return False
        opponent_side = active.side.opponent()
        enemy_position = self.state.robot_for_side(opponent_side).position
        enemy_previous_position = enemy_position
        if self._previous_tick_state is not None:
            enemy_previous_position = self._previous_tick_state.robot_for_side(opponent_side).position
        enemy_predicted_position = enemy_position
        enemy_active = self._active_actions.get(opponent_side)
        if enemy_active is not None:
            enemy_elapsed = max(0.0, self.state.t - enemy_active.start_time)
            enemy_predicted_position = self._position_during_action(enemy_active, enemy_elapsed, enemy_position)
        separation_radius = self.planner.timing.robot_separation_radius
        enemy_effectively_stationary = distance(enemy_previous_position, enemy_predicted_position) < 1e-9
        current_enemy_circle_intersection = (
            distance(candidate_position, enemy_position) < separation_radius
            or point_to_segment_distance(enemy_position, previous_position, candidate_position) < separation_radius
        )
        if current_enemy_circle_intersection:
            if self._is_escape_move_from_stationary_enemy(
                previous_position,
                candidate_position,
                enemy_position,
                separation_radius,
            ):
                return False
            return True
        if (
            distance(candidate_position, enemy_position) >= separation_radius
            and distance(candidate_position, enemy_predicted_position) >= separation_radius
            and point_to_segment_distance(enemy_position, previous_position, candidate_position) >= separation_radius
            and point_to_segment_distance(enemy_predicted_position, previous_position, candidate_position) >= separation_radius
            and point_to_segment_distance(candidate_position, enemy_previous_position, enemy_predicted_position) >= separation_radius
            and segment_to_segment_distance(previous_position, candidate_position, enemy_previous_position, enemy_predicted_position) >= separation_radius
        ):
            return False
        return self._robot_must_yield(active)

    def _is_escape_move_from_stationary_enemy(
        self,
        previous_position: tuple[float, float],
        candidate_position: tuple[float, float],
        enemy_position: tuple[float, float],
        separation_radius: float,
    ) -> bool:
        previous_distance = distance(previous_position, enemy_position)
        if previous_distance >= separation_radius:
            return False
        candidate_distance = distance(candidate_position, enemy_position)
        if candidate_distance <= previous_distance + 1e-9:
            return False

        move_dx = candidate_position[0] - previous_position[0]
        move_dy = candidate_position[1] - previous_position[1]
        move_norm = (move_dx * move_dx + move_dy * move_dy) ** 0.5
        if move_norm <= 1e-9:
            return False

        escape_dx = previous_position[0] - enemy_position[0]
        escape_dy = previous_position[1] - enemy_position[1]
        escape_norm = (escape_dx * escape_dx + escape_dy * escape_dy) ** 0.5
        if escape_norm <= 1e-9:
            escape_dx = candidate_position[0] - enemy_position[0]
            escape_dy = candidate_position[1] - enemy_position[1]
            escape_norm = (escape_dx * escape_dx + escape_dy * escape_dy) ** 0.5
        if escape_norm <= 1e-9:
            return False

        alignment = (move_dx * escape_dx + move_dy * escape_dy) / (move_norm * escape_norm)
        cos_threshold = math.cos(math.radians(self.planner.timing.escape_half_angle_deg))
        return alignment >= cos_threshold

    def _should_pause_for_contested_pick(self, active: ActiveAction, elapsed: float) -> bool:
        if active.action.type is not ActionType.PICK or active.action.target_id is None:
            return False
        phase = self._phase_at_elapsed(active, elapsed)
        if phase is None or phase.kind != "service":
            return False
        source = self.state.sources[active.action.target_id]
        enemy_robot = self.state.robot_for_side(active.side.opponent())
        if distance(enemy_robot.position, source.position) >= self.planner.timing.robot_separation_radius:
            return False
        return self._robot_must_yield(active)

    def _maybe_register_source_contact(self, active: ActiveAction) -> None:
        if active.contact_registered or active.action.type is not ActionType.PICK or active.action.target_id is None:
            return
        source = self.state.sources[active.action.target_id]
        robot = self.state.robot_for_side(active.side)
        if distance(robot.position, source.position) > self.planner.timing.interaction_radius:
            return
        active.contact_registered = True
        if source.available_items > 0 and source.state is SourceState.UNTOUCHED:
            source.state = SourceState.DISTURBED

    def _phase_at_elapsed(self, active: ActiveAction, elapsed: float) -> ActivePhase | None:
        remaining = max(elapsed, 0.0)
        for phase in active.phases:
            if remaining <= phase.duration + 1e-9:
                return phase
            remaining -= phase.duration
        return active.phases[-1] if active.phases else None

    def _phase_index_at_elapsed(self, active: ActiveAction, elapsed: float) -> int | None:
        remaining = max(elapsed, 0.0)
        for index, phase in enumerate(active.phases):
            if remaining <= phase.duration + 1e-9:
                return index
            remaining -= phase.duration
        return len(active.phases) - 1 if active.phases else None

    def _completed_phase_count(self, active: ActiveAction, elapsed: float) -> int:
        remaining = max(elapsed, 0.0)
        completed = 0
        for phase in active.phases:
            if remaining + 1e-9 < phase.duration:
                break
            remaining -= phase.duration
            completed += 1
        return completed

    def _advance_completed_phases(self, active: ActiveAction, elapsed: float) -> None:
        completed = self._completed_phase_count(active, elapsed)
        while active.completed_phases < completed:
            phase = active.phases[active.completed_phases]
            self._apply_phase_completion(phase)
            active.completed_phases += 1

    def _apply_phase_completion(self, phase: ActivePhase) -> None:
        for source_id in phase.clear_source_ids:
            source = self.state.sources.get(source_id)
            if source is not None:
                source.map_footprint_enabled = False
        for deposit_id in phase.clear_deposit_ids:
            deposit = self.state.deposits.get(deposit_id)
            if deposit is not None:
                deposit.map_footprint_enabled = False

    def _robot_must_yield(self, active: ActiveAction) -> bool:
        other_active = self._active_actions[active.side.opponent()]
        if other_active is None:
            return True
        if other_active.start_time + 1e-9 < active.start_time:
            return True
        if active.start_time + 1e-9 < other_active.start_time:
            return False
        return active.side is Side.YELLOW

    def _phase_allows_occupied_goal(self, active: ActiveAction, phase_index: int) -> bool:
        if active.action.type is ActionType.ATTACK_DEPOSIT:
            return True
        if active.action.type is ActionType.START_ENDGAME and phase_index == 2:
            return True
        return False

    def _update_mars_interactions(self) -> None:
        score_config = self.state.endgame_config_for(Side.BLUE).score
        interaction_radius = score_config.mars_robot_interaction_radius
        stop_lookahead_distance = score_config.mars_robot_stop_lookahead_distance
        for marses in self.state.mars_by_side.values():
            for mars in marses:
                self._maybe_stop_mars_on_robot_path(mars, interaction_radius, stop_lookahead_distance)
                mars_position = self._mars_position(mars, self.state.t)
                for side in (Side.BLUE, Side.YELLOW):
                    if self._robot_is_in_endgame_home_leg(side):
                        continue
                    pair = (side, mars.name)
                    if pair in self._mars_collision_pairs:
                        continue
                    robot = self.state.robot_for_side(side)
                    if distance(robot.position, mars_position) < interaction_radius:
                        self._mars_collision_pairs.add(pair)
                        self.state.add_score(side, -self.state.endgame_config_for(side).score.mars_collision_penalty)

    def _maybe_stop_mars_on_robot_path(
        self,
        mars,
        interaction_radius: float,
        stop_lookahead_distance: float,
    ) -> None:
        if not mars.is_released(self.state.t) or self._mars_has_arrived(mars, self.state.t):
            return
        blocked = False
        current_position = self._mars_position(mars, self.state.t)
        lookahead_target = advance_along_path(current_position, (mars.target_position,), stop_lookahead_distance)
        for side in (Side.BLUE, Side.YELLOW):
            if self._robot_is_in_endgame_home_leg(side):
                continue
            robot = self.state.robot_for_side(side)
            if point_to_segment_distance(robot.position, current_position, lookahead_target) < interaction_radius:
                blocked = True
                break

        stopped = self._stopped_mars_by_name.get(mars.name)
        if blocked:
            if stopped is None:
                stopped = StoppedMarsState()
                self._stopped_mars_by_name[mars.name] = stopped
            if stopped.blocked_since is None:
                stopped.blocked_since = float(self.state.t)
            return

        if stopped is None or stopped.blocked_since is None:
            return
        stopped.delay_duration += max(0.0, float(self.state.t) - stopped.blocked_since)
        stopped.blocked_since = None

    def _mars_position(self, mars, t: float) -> tuple[float, float]:
        stopped = self._stopped_mars_by_name.get(mars.name)
        effective_t = t
        if stopped is not None:
            effective_t -= stopped.delay_duration
            if stopped.blocked_since is not None and t >= stopped.blocked_since:
                effective_t -= t - stopped.blocked_since
        return mars.position_at(effective_t)

    def _mars_has_arrived(self, mars, t: float) -> bool:
        stopped = self._stopped_mars_by_name.get(mars.name)
        effective_t = t
        if stopped is not None:
            effective_t -= stopped.delay_duration
            if stopped.blocked_since is not None and t >= stopped.blocked_since:
                effective_t -= t - stopped.blocked_since
        return mars.has_arrived(effective_t)

    def _mars_collided(self, mars, side: Side | None = None) -> bool:
        if side is not None:
            return (side, mars.name) in self._mars_collision_pairs
        return (Side.BLUE, mars.name) in self._mars_collision_pairs or (Side.YELLOW, mars.name) in self._mars_collision_pairs

    def _robot_is_in_endgame_home_leg(self, side: Side) -> bool:
        active = self._active_actions.get(side)
        if active is None or active.action.type is not ActionType.START_ENDGAME:
            return False
        elapsed = max(0.0, self.state.t - active.start_time)
        phase_index = self._phase_index_at_elapsed(active, elapsed)
        return phase_index == 2

    def _runtime_obstacle_overrides(self, active: ActiveAction, elapsed: float) -> tuple[set[int], set[int]]:
        completed = self._completed_phase_count(active, elapsed)
        ignored_source_ids: set[int] = set()
        ignored_deposit_ids: set[int] = set()
        for phase in active.phases[:completed]:
            ignored_source_ids.update(phase.clear_source_ids)
            ignored_deposit_ids.update(phase.clear_deposit_ids)
        return ignored_source_ids, ignored_deposit_ids

    def _truncate_waypoints_to_distance(
        self,
        start: tuple[float, float],
        waypoints: tuple[tuple[float, float], ...],
        max_distance: float,
    ) -> tuple[tuple[float, float], ...]:
        if not waypoints or max_distance <= 0.0:
            return ()
        remaining = max_distance
        current = start
        truncated: list[tuple[float, float]] = []
        for waypoint in waypoints:
            segment_length = distance(current, waypoint)
            if segment_length <= 1e-9:
                current = waypoint
                continue
            if segment_length > remaining:
                truncated.append(interpolate(current, waypoint, remaining / segment_length))
                break
            truncated.append(waypoint)
            remaining -= segment_length
            if remaining <= 1e-9:
                break
            current = waypoint
        return tuple(truncated)

    def _path_intersects_circle(
        self,
        start: tuple[float, float],
        waypoints: tuple[tuple[float, float], ...],
        center: tuple[float, float],
        radius: float,
    ) -> bool:
        previous = start
        for waypoint in waypoints:
            if point_to_segment_distance(center, previous, waypoint) < radius:
                return True
            previous = waypoint
        return False

    def _local_path_hits_obstacle(
        self,
        side: Side,
        start: tuple[float, float],
        waypoints: tuple[tuple[float, float], ...],
        *,
        allow_goal_occupied: bool,
        ignored_source_ids: set[int] | None = None,
        ignored_deposit_ids: set[int] | None = None,
    ) -> bool:
        self.planner._sync_grid_navigation(
            self.state,
            side,
            include_enemy_robot=False,
            ignored_source_ids=ignored_source_ids,
            ignored_deposit_ids=ignored_deposit_ids,
        )
        grid_map = self.planner.grid_map
        start_cell = grid_map.world_to_grid(*start)
        start_blocked = grid_map.is_blocked(start_cell[0], start_cell[1], use_inflated=True)
        path_waypoints = waypoints
        escaping_blocked_start = False
        if start_blocked:
            first_free_index = 0
            while first_free_index < len(path_waypoints):
                waypoint_cell = grid_map.world_to_grid(*path_waypoints[first_free_index])
                if not grid_map.is_blocked(waypoint_cell[0], waypoint_cell[1], use_inflated=True):
                    break
                first_free_index += 1
            if first_free_index > 0:
                path_waypoints = path_waypoints[first_free_index:]
            nearest_free = self.planner.grid_planner._find_nearest_free_cell(
                start_cell,
                current_map=grid_map.planning_map,
            )
            if nearest_free is None:
                return True
            exit_point = grid_map.grid_to_world(*nearest_free)
            if distance(start, exit_point) > 1e-9:
                if not path_waypoints or distance(path_waypoints[0], exit_point) > 1e-9:
                    path_waypoints = (exit_point, *path_waypoints)
                escaping_blocked_start = True
        step_m = max(grid_map.config.resolution_m * 0.5, 1e-6)
        previous = start
        for waypoint_index, waypoint in enumerate(path_waypoints):
            segment_length = distance(previous, waypoint)
            samples = max(int(segment_length / step_m), 1)
            for sample_index in range(1, samples + 1):
                if allow_goal_occupied and waypoint_index == len(path_waypoints) - 1 and sample_index == samples:
                    continue
                point = interpolate(previous, waypoint, sample_index / samples)
                row, col = grid_map.world_to_grid(*point)
                blocked = grid_map.is_blocked(row, col, use_inflated=True)
                if escaping_blocked_start:
                    if blocked:
                        continue
                    escaping_blocked_start = False
                if blocked:
                    return True
            previous = waypoint
        return False

    def _apply_action_effects(self, side: Side, action: Action) -> None:
        robot = self.state.robot_for_side(side)

        if action.type is ActionType.PICK and action.target_id is not None:
            source = self.state.sources[action.target_id]
            previous_items = source.available_items
            previous_state = source.state
            picked = min(robot.capacity - robot.load, source.available_items)
            robot.load += picked
            source.available_items -= picked
            if source.available_items <= 0:
                source.available_items = 0
                source.state = SourceState.EMPTY
            else:
                source.state = SourceState.DISTURBED
            self._record_source_change(action.target_id, previous_items, previous_state)
            self._refresh_thermometer_lane_clear_tracking()
            return

        if action.type is ActionType.DEPOSIT and action.target_id is not None:
            deposit = self.state.deposits[action.target_id]
            requested = int(action.metadata.get("deposit_count", robot.load))
            max_count = deposit_max_count_for_side(deposit, side, robot.load)
            if requested <= 0 or max_count <= 0:
                return
            deposited = min(requested, max_count)
            if deposited <= 0:
                return
            before_blue = deposit_zone_points(deposit, Side.BLUE)
            before_yellow = deposit_zone_points(deposit, Side.YELLOW)
            deposit.clear_pushed_state()
            deposit.add_items(side, deposited)
            self._apply_deposit_score_delta(deposit, before_blue, before_yellow, actor=side)
            robot.load -= deposited
            self._refresh_thermometer_lane_clear_tracking()
            return

        if action.type is ActionType.ATTACK_DEPOSIT and action.target_id is not None:
            deposit = self.state.deposits[action.target_id]
            before_blue = deposit_zone_points(deposit, Side.BLUE)
            before_yellow = deposit_zone_points(deposit, Side.YELLOW)
            removed = deposit.items_for_side(side.opponent())
            deposit.clear()
            if removed:
                push_state_raw = str(action.metadata.get("push_state", PushState.CLEAR.value))
                try:
                    push_state = PushState(push_state_raw)
                except ValueError:
                    push_state = PushState.CLEAR
                deposit.set_pushed_state(push_state, side.opponent())
            else:
                deposit.clear_pushed_state()
            self._apply_deposit_score_delta(deposit, before_blue, before_yellow, actor=side)
            self._refresh_thermometer_lane_clear_tracking()
            return

        if action.type is ActionType.DO_THERMOMETER:
            thermometer_state_before = self.state.thermometer.state
            blocking_source_id = self.state.thermometer.blocking_source_id_for_side(side)
            source = self.state.sources.get(blocking_source_id)
            if source is not None:
                previous_items = source.available_items
                previous_state = source.state
                source.available_items = 0
                source.available_from_t = self.state.t
                source.state = SourceState.EMPTY
                source.map_footprint_enabled = False
                self._record_source_change(blocking_source_id, previous_items, previous_state)

            blocking_deposit_id = self.state.thermometer.blocking_deposit_id_for_side(side)
            deposit = self.state.deposits.get(blocking_deposit_id)
            if deposit is not None:
                before_blue = deposit_zone_points(deposit, Side.BLUE)
                before_yellow = deposit_zone_points(deposit, Side.YELLOW)
                deposit.clear()
                self._apply_deposit_score_delta(deposit, before_blue, before_yellow, actor=side)

            self.state.thermometer.mark_done_for_side(side)
            self._record_thermometer_state_change(thermometer_state_before)
            self.state.add_score(side, self.state.thermometer.reward)
            self._refresh_thermometer_lane_clear_tracking()
            return

        if action.type is ActionType.START_ENDGAME:
            self._apply_endgame_home_drop(side)
            self._refresh_thermometer_lane_clear_tracking()
            return

    def _apply_deposit_score_delta(self, deposit, before_blue: int, before_yellow: int, actor: Side | None = None) -> None:
        after_blue = deposit_zone_points(deposit, Side.BLUE)
        after_yellow = deposit_zone_points(deposit, Side.YELLOW)
        if after_blue != before_blue:
            self.state.add_score(Side.BLUE, after_blue - before_blue)
        if after_yellow != before_yellow:
            self.state.add_score(Side.YELLOW, after_yellow - before_yellow)
        self._record_deposit_score_change(deposit.semantic_id, before_blue, before_yellow, actor)

    def _snapshot(self) -> None:
        self._history.append(
            HistoryEntry(
                time=round(self.state.t, 3),
                our_position=self.state.our_robot.position,
                enemy_position=self.state.enemy_robot.position,
                our_score=self.state.score_for_side(self.state.our_side),
                enemy_score=self.state.score_for_side(self.state.enemy_side),
                our_load=self.state.our_robot.load,
                enemy_load=self.state.enemy_robot.load,
                source_states={
                    source_id: {
                        "state": _serialize(source.state),
                        "available_items": source.available_items,
                        "map_footprint_enabled": source.map_footprint_enabled,
                    }
                    for source_id, source in self.state.sources.items()
                },
                deposit_states={
                    deposit_id: {
                        "blue_items": deposit.blue_items,
                        "yellow_items": deposit.yellow_items,
                        "map_footprint_enabled": deposit.map_footprint_enabled,
                        "push_state": _serialize(deposit.push_state),
                        "pushed_owner": _serialize(deposit.pushed_owner),
                        "occupied_by": _serialize(deposit.occupied_by),
                    }
                    for deposit_id, deposit in self.state.deposits.items()
                },
                thermometer_state=_serialize(self.state.thermometer.state),
                thermometer_doing_blue=self.state.thermometer.doing_blue,
                thermometer_doing_yellow=self.state.thermometer.doing_yellow,
                mars_states={
                    side.value: [
                        {
                            "name": mars.name,
                            "pantry_id": mars.pantry_id,
                            "position": self._mars_position(mars, self.state.t),
                            "released": mars.is_released(self.state.t),
                            "arrived": self._mars_has_arrived(mars, self.state.t),
                            "stopped": mars.name in self._stopped_mars_by_name,
                            "collided_by_blue": self._mars_collided(mars, Side.BLUE),
                            "collided_by_yellow": self._mars_collided(mars, Side.YELLOW),
                        }
                        for mars in self.state.mars_by_side.get(side, ())
                    ]
                    for side in (Side.BLUE, Side.YELLOW)
                },
            )
        )

    def _finalize_match(self) -> None:
        for side in (Side.BLUE, Side.YELLOW):
            config = self.state.endgame_config_for(side)
            self._apply_endgame_home_drop(side)
            robot = self.state.robot_for_side(side)
            final_distance = distance(robot.position, config.final_home_point)
            finish_points = config.finish_points(final_distance)
            if finish_points:
                self.state.add_score(side, finish_points)
            self._apply_mars_score(side, config)

    def _apply_endgame_home_drop(self, side: Side) -> None:
        robot = self.state.robot_for_side(side)
        if robot.load <= 0 or not self.state.endgame_started_for(side):
            return
        config = self.state.endgame_config_for(side)
        final_distance = distance(robot.position, config.final_home_point)
        if final_distance > config.home_partial_tolerance:
            return
        home_deposit = self._home_deposit_for_side(side)
        if home_deposit is None:
            return
        before_blue = deposit_zone_points(home_deposit, Side.BLUE)
        before_yellow = deposit_zone_points(home_deposit, Side.YELLOW)
        deposited = min(robot.load, home_remaining_capacity(home_deposit))
        if deposited <= 0:
            return
        home_deposit.add_items(side, deposited)
        self._apply_deposit_score_delta(home_deposit, before_blue, before_yellow, actor=side)
        robot.load -= deposited

    def _apply_mars_score(self, side: Side, config: EndgameConfig) -> None:
        marses = self.state.mars_by_side.get(side, ())
        if not marses:
            return
        pantry_points = 0
        arrived_count = 0
        for mars in marses:
            if self._mars_has_arrived(mars, self.state.T_end) and not self._mars_collided(mars):
                pantry_points += config.score.mars_pantry_points
                arrived_count += 1
        if pantry_points:
            self.state.add_score(side, pantry_points)
        if arrived_count == len(marses):
            self.state.add_score(side, config.score.mars_all_eating_bonus)

    def _home_deposit_for_side(self, side: Side):
        for deposit in self.state.deposits.values():
            if deposit.kind is DepositType.HOME and deposit.owner is side:
                return deposit
        return None

    def _mars_collides_with_robot(self, mars) -> bool:
        return self._mars_collided(mars)

    def _build_summary(self) -> dict[str, float | int | bool | str]:
        our_side = self.state.our_side
        enemy_side = self.state.enemy_side
        our_config = self.state.endgame_config_for(our_side)
        home_distance = distance(self.state.our_robot.position, our_config.final_home_point)
        return {
            "scenario": self.scenario_name,
            "our_side": our_side.value,
            "our_score": self.state.score_for_side(our_side),
            "enemy_score": self.state.score_for_side(enemy_side),
            "score_diff": self.state.score_for_side(our_side) - self.state.score_for_side(enemy_side),
            "win": self.state.score_for_side(our_side) > self.state.score_for_side(enemy_side),
            "successful_return_home": home_distance <= our_config.home_full_tolerance,
            "partial_return_home": home_distance <= our_config.home_partial_tolerance,
            "replan_events": self._replan_count,
            "lost_target_events": self._lost_target_count,
            "thermometer_used": any(log.action == "THERMOMETER" and log.side is our_side for log in self._action_log),
            "start_endgame_used": any(log.action == "START_ENDGAME" and log.side is our_side for log in self._action_log),
            "our_mars_pantry_count": sum(
                1
                for mars in self.state.mars_by_side.get(our_side, ())
                if self._mars_has_arrived(mars, self.state.T_end) and not self._mars_collides_with_robot(mars)
            ),
            "enemy_mars_pantry_count": sum(
                1
                for mars in self.state.mars_by_side.get(enemy_side, ())
                if self._mars_has_arrived(mars, self.state.T_end) and not self._mars_collides_with_robot(mars)
            ),
            "our_mars_all_eating": all(
                self._mars_has_arrived(mars, self.state.T_end) and not self._mars_collides_with_robot(mars)
                for mars in self.state.mars_by_side.get(our_side, ())
            ) if self.state.mars_by_side.get(our_side, ()) else False,
            "enemy_mars_all_eating": all(
                self._mars_has_arrived(mars, self.state.T_end) and not self._mars_collides_with_robot(mars)
                for mars in self.state.mars_by_side.get(enemy_side, ())
            ) if self.state.mars_by_side.get(enemy_side, ()) else False,
            "our_mars_collision_count": sum(
                1 for mars in self.state.mars_by_side.get(our_side, ()) if self._mars_collided(mars)
            ),
            "enemy_mars_collision_count": sum(
                1 for mars in self.state.mars_by_side.get(enemy_side, ()) if self._mars_collided(mars)
            ),
            "history_points": len(self._history),
            "enemy_policy": self._controller_name_for_side(enemy_side),
        }

    def _score_diff_for_side(self, side: Side) -> float:
        return float(self.state.score_for_side(side) - self.state.score_for_side(side.opponent()))

    def _finalize_pending_rl_transition(
        self,
        side: Side,
        next_step: RLPolicyStep,
        score_diff_after: float,
        done: bool,
    ) -> None:
        pending = self._pending_rl[side]
        if pending is None:
            return
        reward = score_diff_after - pending.score_diff_before
        if pending.chosen_action == "THERMOMETER" and self.state.thermometer.is_done_for_side(side):
            reward += self.thermometer_reward_bonus
        if done:
            reward += self._terminal_bonus_for_side(side)
        self._rl_transitions.append(
            RLTransition(
                side=side.value,
                time=pending.time,
                chosen_action=pending.chosen_action,
                chosen_action_index=pending.chosen_action_index,
                action_mask=pending.step.action_mask,
                observation=pending.step.observation,
                reward=reward,
                next_observation=next_step.observation,
                next_action_mask=next_step.action_mask,
                done=done,
                score_diff_before=pending.score_diff_before,
                score_diff_after=score_diff_after,
            )
        )
        self._pending_rl[side] = None

    def _terminal_bonus_for_side(self, side: Side) -> float:
        score_diff = self._score_diff_for_side(side)
        if score_diff > 0.0:
            return self.terminal_win_bonus
        if score_diff < 0.0:
            return self.terminal_loss_bonus
        return self.terminal_draw_bonus

    def _controller_name_for_side(self, side: Side) -> str:
        selector = self.action_selectors.get(side)
        if selector is not None:
            return selector.name
        if side is self.state.our_side:
            return "planner"
        return self.opponent_policy.name

    def _finalize_rl_transitions(self) -> None:
        zero_mask = tuple(0 for _ in DEFAULT_ACTION_SPACE.tokens)
        for side in (Side.BLUE, Side.YELLOW):
            pending = self._pending_rl[side]
            if pending is None:
                continue
            terminal_observation = build_rl_observation(
                self.state,
                side,
                previous_state=self._previous_tick_state,
                dt=self.dt,
                config=self.rl_observation_config,
            )
            terminal_step = RLPolicyStep(
                observation=terminal_observation,
                action_space=DEFAULT_ACTION_SPACE,
                action_mask=zero_mask,
                candidates=(),
            )
            self._finalize_pending_rl_transition(
                side,
                next_step=terminal_step,
                score_diff_after=self._score_diff_for_side(side),
                done=True,
            )


def save_result(result: MatchResult, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_serialize(result), indent=2), encoding="utf-8")
    return output


def load_result(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_rl_dataset(result: MatchResult, path: str | Path, side: Side | None = None) -> Path:
    transitions = result.rl_transitions if side is None else [item for item in result.rl_transitions if item.side == side.value]
    return save_transition_dataset(transitions, path)


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(_serialize(key)): _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _serialize_with_policy_metadata(value: Any, policy_metadata: dict[str, object]) -> dict[str, Any]:
    serialized = _serialize(value)
    assert isinstance(serialized, dict)
    serialized.update(_serialize(policy_metadata))
    return serialized


def _waypoint_tuple(value: Any, fallback: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    if value is None:
        return fallback
    return tuple((float(point[0]), float(point[1])) for point in value)


def _travel_segments_from_metadata(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    segments: list[dict[str, Any]] = []
    for segment in value:
        if not isinstance(segment, dict):
            continue
        segments.append(
            {
                "waypoints": _waypoint_tuple(segment.get("waypoints"), fallback=()),
                "duration": float(segment.get("duration", 0.0)),
                "clear_source_ids": tuple(int(item) for item in segment.get("clear_source_ids", ())),
                "clear_deposit_ids": tuple(int(item) for item in segment.get("clear_deposit_ids", ())),
            }
        )
    return segments
