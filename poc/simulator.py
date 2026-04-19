from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from poc.actions import Action, ActionType
from poc.endgame import EndgameConfig
from poc.entities import DepositType, Side, SourceState, ThermometerState
from poc.external_events import apply_external_event
from poc.game_state import GameState
from poc.geometry import advance_along_path, distance
from poc.opponent_policy import OpponentPolicy
from poc.planner import UtilityPlanner
from poc.scoring import deposit_points


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
    target_id: int | None
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


@dataclass(slots=True)
class ActiveAction:
    action: Action
    side: Side
    start_time: float
    phases: list[ActivePhase]
    total_duration: float
    contact_registered: bool = False


@dataclass(slots=True)
class MatchResult:
    scenario_name: str
    summary: dict[str, float | int | bool | str]
    history: list[HistoryEntry]
    action_log: list[ActionLogEntry]
    planner_debug: list[dict[str, object]]
    events: list[SimulationEvent]
    field_size: tuple[float, float]
    sources: dict[int, dict[str, object]]
    deposits: dict[int, dict[str, object]]
    thermometer: dict[str, object]
    endgame: dict[str, dict[str, object]]
    our_side: str


class Simulator:
    def __init__(
        self,
        state: GameState,
        scenario_name: str,
        opponent_policy: OpponentPolicy,
        planner: UtilityPlanner | None = None,
        dt: float = 0.5,
    ) -> None:
        self.state = state
        self.scenario_name = scenario_name
        self.opponent_policy = opponent_policy
        self.planner = planner or UtilityPlanner()
        self.dt = dt
        self._event_index = 0
        self._active_actions: dict[Side, ActiveAction | None] = {Side.BLUE: None, Side.YELLOW: None}
        self._history: list[HistoryEntry] = []
        self._action_log: list[ActionLogEntry] = []
        self._planner_debug: list[dict[str, object]] = []
        self._events: list[SimulationEvent] = []
        self._replan_count = 0
        self._lost_target_count = 0

    def run(self) -> MatchResult:
        while self.state.t < self.state.T_end:
            self._apply_pending_events()

            for side in (Side.BLUE, Side.YELLOW):
                self._update_active_action(side)
                if self._active_actions[side] is None and self.state.t < self.state.T_end:
                    self._assign_next_action(side)

            self._snapshot()
            self.state.t = round(min(self.state.T_end, self.state.t + self.dt), 6)

        self._apply_pending_events()
        for side in (Side.BLUE, Side.YELLOW):
            self._update_active_action(side)
        self._finalize_match()
        return MatchResult(
            scenario_name=self.scenario_name,
            summary=self._build_summary(),
            history=self._history,
            action_log=self._action_log,
            planner_debug=self._planner_debug,
            events=self._events,
            field_size=self.state.field_size,
            sources={source_id: _serialize(source) for source_id, source in self.state.sources.items()},
            deposits={deposit_id: _serialize(deposit) for deposit_id, deposit in self.state.deposits.items()},
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

        if elapsed + 1e-9 < active.total_duration:
            return

        robot.position = self._final_position(active, robot.position)
        self._apply_action_effects(side, active.action)
        robot.current_action = None
        robot.current_target_id = None
        self._active_actions[side] = None

    def _assign_next_action(self, side: Side) -> None:
        if side is self.state.our_side:
            decision = self.planner.plan(self.state, side)
            action = decision.chosen_action
            self._planner_debug.append(decision.debug_payload(self.state.t, side))
        else:
            action = self.opponent_policy.choose_action(self.state, self.planner, side)

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
                target_id=action.target_id,
                expected_duration=round(action.expected_duration, 3),
                score=round(action.score, 3),
            )
        )

    def _activate_action(self, side: Side, action: Action) -> ActiveAction:
        robot = self.state.robot_for_side(side)
        phases: list[ActivePhase] = []
        if action.type is ActionType.START_ENDGAME:
            endgame = self.state.endgame_config_for(side)
            travel_to_chill = float(action.metadata.get("travel_to_chill", 0.0))
            wait_duration = float(action.metadata.get("wait_duration", 0.0))
            travel_home = float(action.metadata.get("travel_home", 0.0))
            grip_rotate = float(action.metadata.get("grip_rotate", endgame.grip_rotate_duration))
            phases.append(
                ActivePhase(
                    kind="travel",
                    duration=travel_to_chill,
                    waypoints=(endgame.chill_point,),
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
                    waypoints=endgame.home_waypoints,
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
            return deposit.kind is DepositType.STORAGE and deposit.total_items() > 0
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
            if deposit.kind is DepositType.STORAGE and deposit.total_items() > 0:
                return
            deposited = robot.load
            if deposited <= 0:
                return
            deposit.add_items(side, deposited)
            self.state.add_score(side, deposit_points(deposit.kind))
            robot.load = 0
            return

        if action.type is ActionType.ATTACK_DEPOSIT and action.target_id is not None:
            deposit = self.state.deposits[action.target_id]
            removed = deposit.items_for_side(side.opponent())
            deposit.clear()
            if removed:
                self.state.add_score(side.opponent(), -deposit_points(deposit.kind))
            return

        if action.type is ActionType.DO_THERMOMETER:
            self.state.thermometer.mark_done_for_side(side)
            self.state.add_score(side, self.state.thermometer.reward)
            return

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
                    }
                    for source_id, source in self.state.sources.items()
                },
                deposit_states={
                    deposit_id: {
                        "blue_items": deposit.blue_items,
                        "yellow_items": deposit.yellow_items,
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
            if final_distance <= config.home_full_tolerance:
                self.state.add_score(side, 10)
            elif final_distance <= config.home_partial_tolerance:
                self.state.add_score(side, 5)

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
            "enemy_policy": self.opponent_policy.name,
        }


def save_result(result: MatchResult, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_serialize(result), indent=2), encoding="utf-8")
    return output


def load_result(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
