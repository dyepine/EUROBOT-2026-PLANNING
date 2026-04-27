from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from poc.actions import Action, ActionType
from poc.endgame import EndgameConfig
from poc.entities import DepositType, PushState, Side, SourceState, ThermometerState
from poc.external_events import apply_external_event
from poc.game_state import GameState
from poc.geometry import advance_along_path, distance
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
    RLPolicyStep,
    RLTransition,
    build_rl_observation,
    build_rl_policy_step,
    save_transition_dataset,
)
from poc.scoring import deposit_max_count, deposit_zone_points


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
    our_side: str


@dataclass(slots=True)
class PendingRLTransition:
    time: float
    step: RLPolicyStep
    chosen_action: str
    chosen_action_index: int
    score_diff_before: float


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
        terminal_win_bonus: float = 20.0,
        terminal_draw_bonus: float = 0.0,
        terminal_loss_bonus: float = -20.0,
    ) -> None:
        self.state = state
        self.scenario_name = scenario_name
        self.opponent_policy = opponent_policy
        self.planner = planner or UtilityPlanner()
        self.dt = dt
        self.action_selectors = dict(action_selectors or {})
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
        self._previous_tick_state: GameState | None = None

    def run(self) -> MatchResult:
        while self.state.t < self.state.T_end:
            self._apply_pending_events()

            for side in (Side.BLUE, Side.YELLOW):
                self._update_active_action(side)
                if self._active_actions[side] is None and self.state.t < self.state.T_end:
                    self._assign_next_action(side)

            self._snapshot()
            self._previous_tick_state = deepcopy(self.state)
            self.state.t = round(min(self.state.T_end, self.state.t + self.dt), 6)

        self._apply_pending_events()
        for side in (Side.BLUE, Side.YELLOW):
            self._update_active_action(side)
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
            our_side=self.state.our_side.value,
        )

    def _apply_pending_events(self) -> None:
        while self._event_index < len(self.state.external_events):
            event = self.state.external_events[self._event_index]
            if event.time > self.state.t:
                break
            note = apply_external_event(self.state, event)
            self._events.append(SimulationEvent(time=self.state.t, side=None, kind="external_event", note=note))
            self._event_index += 1

    def _update_active_action(self, side: Side) -> None:
        active = self._active_actions[side]
        if active is None:
            return

        robot = self.state.robot_for_side(side)
        previous_position = robot.position
        elapsed = max(0.0, self.state.t - active.start_time)
        robot.position = self._position_during_action(active, elapsed, robot.position)

        if self._should_pause_for_robot_separation(active, previous_position, robot.position):
            robot.position = previous_position
            active.start_time = round(active.start_time + self.dt, 6)
            return

        self._maybe_register_source_contact(active)

        if self._should_cancel_action(active):
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

        if action.type is ActionType.START_ENDGAME:
            self.state.set_endgame_started(side, True)

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
            return requested <= 0 or requested > deposit_max_count(deposit, robot.load)
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
        enemy_position = self.state.robot_for_side(active.side.opponent()).position
        if distance(candidate_position, enemy_position) >= self.planner.timing.robot_separation_radius:
            return False
        return self._robot_must_yield(active)

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

    def _advance_completed_phases(self, active: ActiveAction, elapsed: float) -> None:
        remaining = max(elapsed, 0.0)
        completed = 0
        for phase in active.phases:
            if remaining + 1e-9 < phase.duration:
                break
            remaining -= phase.duration
            completed += 1
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

    def _apply_action_effects(self, side: Side, action: Action) -> None:
        robot = self.state.robot_for_side(side)

        if action.type is ActionType.PICK and action.target_id is not None:
            source = self.state.sources[action.target_id]
            picked = min(robot.capacity - robot.load, source.available_items)
            robot.load += picked
            source.available_items -= picked
            if source.available_items <= 0:
                source.available_items = 0
                source.state = SourceState.EMPTY
            else:
                source.state = SourceState.DISTURBED
            return

        if action.type is ActionType.DEPOSIT and action.target_id is not None:
            deposit = self.state.deposits[action.target_id]
            requested = int(action.metadata.get("deposit_count", robot.load))
            max_count = deposit_max_count(deposit, robot.load)
            if requested <= 0 or max_count <= 0:
                return
            deposited = min(requested, max_count)
            if deposited <= 0:
                return
            before_blue = deposit_zone_points(deposit, Side.BLUE)
            before_yellow = deposit_zone_points(deposit, Side.YELLOW)
            deposit.clear_pushed_state()
            deposit.add_items(side, deposited)
            self._apply_deposit_score_delta(deposit, before_blue, before_yellow)
            robot.load -= deposited
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
            self._apply_deposit_score_delta(deposit, before_blue, before_yellow)
            return

        if action.type is ActionType.DO_THERMOMETER:
            blocking_source_id = self.state.thermometer.blocking_source_id_for_side(side)
            source = self.state.sources.get(blocking_source_id)
            if source is not None:
                source.available_items = 0
                source.available_from_t = self.state.t
                source.state = SourceState.EMPTY
                source.map_footprint_enabled = False

            blocking_deposit_id = self.state.thermometer.blocking_deposit_id_for_side(side)
            deposit = self.state.deposits.get(blocking_deposit_id)
            if deposit is not None:
                before_blue = deposit_zone_points(deposit, Side.BLUE)
                before_yellow = deposit_zone_points(deposit, Side.YELLOW)
                deposit.clear()
                self._apply_deposit_score_delta(deposit, before_blue, before_yellow)

            self.state.thermometer.mark_done_for_side(side)
            self.state.add_score(side, self.state.thermometer.reward)
            return

    def _apply_deposit_score_delta(self, deposit, before_blue: int, before_yellow: int) -> None:
        after_blue = deposit_zone_points(deposit, Side.BLUE)
        after_yellow = deposit_zone_points(deposit, Side.YELLOW)
        if after_blue != before_blue:
            self.state.add_score(Side.BLUE, after_blue - before_blue)
        if after_yellow != before_yellow:
            self.state.add_score(Side.YELLOW, after_yellow - before_yellow)

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
            )
        )

    def _finalize_match(self) -> None:
        for side in (Side.BLUE, Side.YELLOW):
            config = self.state.endgame_config_for(side)
            robot = self.state.robot_for_side(side)
            final_distance = distance(robot.position, config.final_home_point)
            finish_points = config.finish_points(final_distance)
            if finish_points:
                self.state.add_score(side, finish_points)

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
