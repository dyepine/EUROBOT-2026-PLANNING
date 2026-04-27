from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poc.actions import Action, ActionType
from poc.config import ActionTimingConfig, UtilityWeights
from poc.endgame import EndgameConfig
from poc.entities import DepositPoint, DepositType, PushState, RouteOption, Side, SourceState, SourcePoint, Thermometer, ThermometerState
from poc.game_state import GameState
from poc.geometry import distance, point_to_segment_distance
from poc.grid_map import DEFAULT_LAYOUT_PATH, GridOccupancyMap
from poc.grid_planner import GridAStarPlanner
from poc.policy_mapping import normalized_action_label, normalized_target_id
from poc.scoring import deposit_max_count, evaluate_action


@dataclass(slots=True)
class PlanningDecision:
    chosen_action: Action
    ranked_actions: list[Action]
    reason: str = ""

    def debug_payload(self, t: float, side: Side) -> dict[str, object]:
        return {
            "time": round(t, 3),
            "side": side.value,
            "reason": self.reason,
            "chosen_action": _policy_debug_row(self.chosen_action, side),
            "candidates": [_policy_debug_row(action, side) for action in self.ranked_actions],
        }


@dataclass(slots=True)
class PlannedRoute:
    route_name: str
    semantic_waypoints: tuple[tuple[float, float], ...]
    motion_waypoints: tuple[tuple[float, float], ...]
    travel_duration: float
    duration_source: str
    travel_segments: tuple[PlannedTravelSegment, ...] = ()


@dataclass(slots=True)
class PlannedTravelSegment:
    motion_waypoints: tuple[tuple[float, float], ...]
    travel_duration: float
    duration_source: str
    clear_source_ids: tuple[int, ...] = ()
    clear_deposit_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class PlannedEndgameRoute:
    to_chill: PlannedRoute
    home: PlannedRoute
    wait_duration: float
    grip_rotate_duration: float

    @property
    def travel_duration(self) -> float:
        return self.to_chill.travel_duration + self.home.travel_duration

    @property
    def total_duration(self) -> float:
        return self.travel_duration + self.wait_duration + self.grip_rotate_duration


class UtilityPlanner:
    def __init__(
        self,
        timing: ActionTimingConfig | None = None,
        weights: UtilityWeights | None = None,
        layout_path: str | None = None,
    ) -> None:
        self.timing = timing or ActionTimingConfig()
        self.weights = weights or UtilityWeights()
        resolved_layout = DEFAULT_LAYOUT_PATH if layout_path is None else Path(layout_path)
        if not resolved_layout.exists():
            raise FileNotFoundError(f"Grid layout not found: {resolved_layout}")
        self.grid_map = GridOccupancyMap.from_layout(resolved_layout, team_color="all")
        self.grid_planner = GridAStarPlanner(self.grid_map)

    def plan(self, state: GameState, side: Side) -> PlanningDecision:
        ranked = self.rank_actions(state, side)
        chosen = ranked[0] if ranked else self._make_wait_action()
        reason = "utility_max"
        if chosen.type is ActionType.START_ENDGAME:
            reason = "endgame_window"
        elif chosen.label == "WAIT_FOR_CHILL":
            reason = "hold_for_chill"
        return PlanningDecision(chosen_action=chosen, ranked_actions=ranked, reason=reason)

    def rank_actions(self, state: GameState, side: Side) -> list[Action]:
        if state.endgame_started_for(side):
            return [self._score(state, side, self._make_wait_action())]

        candidates = self._generate_candidates(state, side)
        ranked = [self._score(state, side, action) for action in candidates]
        ranked.sort(key=lambda action: action.score, reverse=True)
        return ranked

    def _score(self, state: GameState, side: Side, action: Action) -> Action:
        return evaluate_action(state, side, action, self.timing, self.weights)

    def _generate_candidates(self, state: GameState, side: Side) -> list[Action]:
        robot = state.robot_for_side(side)
        endgame_config = state.endgame_config_for(side)
        endgame_action = self._make_endgame_action(state, side)
        candidates: list[Action] = [self._make_wait_action()]
        if endgame_action is not None:
            candidates.insert(0, endgame_action)

        if state.t >= endgame_config.main_pipeline_deadline:
            if self._must_start_endgame(state, side):
                return candidates
            return [self._make_hold_for_chill_action(state, side)]

        if robot.load == 0:
            for source in state.sources.values():
                if not source.is_available(state.t):
                    continue
                action = self._make_pick_action(state, side, robot.position, robot.speed, source)
                if action is None:
                    continue
                if self._can_fit_before_endgame(state, side, action):
                    candidates.append(action)

        if robot.load > 0:
            for deposit in state.friendly_deposits(side, include_home=True, include_neutral=True):
                max_count = deposit_max_count(deposit, robot.load)
                if max_count <= 0:
                    continue
                for deposit_count in range(1, max_count + 1):
                    action = self._make_deposit_action(
                        state,
                        side,
                        robot.position,
                        robot.speed,
                        deposit,
                        deposit_count,
                    )
                    if action is None:
                        continue
                    if self._can_fit_before_endgame(state, side, action):
                        candidates.append(action)

        if (
            not state.thermometer.is_done_for_side(side)
            and self._thermometer_lane_is_clear(state, side)
        ):
            thermo = self._make_thermometer_action(state, side, robot.position, robot.speed, state.thermometer)
            if thermo is not None and self._can_fit_before_endgame(state, side, thermo):
                candidates.append(thermo)

        if robot.load == 0:
            for deposit in state.deposits.values():
                if deposit.owner is side:
                    continue
                if deposit.protected_for is not None:
                    continue
                if deposit.items_for_side(side.opponent()) <= 0:
                    continue
                action = self._make_attack_action(
                    state,
                    side,
                    robot.position,
                    robot.speed,
                    deposit,
                )
                if action is None:
                    continue
                if self._can_fit_before_endgame(state, side, action):
                    candidates.append(action)

        return candidates

    def _can_fit_before_endgame(self, state: GameState, side: Side, action: Action) -> bool:
        robot = state.robot_for_side(side)
        endgame_config = state.endgame_config_for(side)
        action_end_position = action.waypoints[-1] if action.waypoints else robot.position
        planned_endgame = self._plan_endgame_route(
            state,
            side,
            start=action_end_position,
            now=state.t + action.expected_duration,
        )
        if planned_endgame is None:
            return False
        finishes_before_chill = (
            state.t + action.expected_duration + planned_endgame.to_chill.travel_duration
            <= endgame_config.chill_end - endgame_config.chill_margin
        )
        finishes_home = (
            endgame_config.chill_end + planned_endgame.home.travel_duration + planned_endgame.grip_rotate_duration
            <= state.T_end - endgame_config.home_margin
        )
        return finishes_before_chill and finishes_home

    def _make_pick_action(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
        source: SourcePoint,
    ) -> Action | None:
        planned_route = self._best_route(
            state,
            side,
            start,
            speed,
            source.collection_routes(),
            clear_source_ids=(source.semantic_id,),
        )
        if planned_route is None:
            return None
        travel = planned_route.travel_duration
        service = self.timing.pick_duration + self.timing.align_duration
        return Action(
            type=ActionType.PICK,
            target_id=source.semantic_id,
            label=f"PICK_{source.semantic_id}",
            target_position=planned_route.motion_waypoints[-1],
            waypoints=planned_route.motion_waypoints,
            service_duration=service,
            travel_duration=travel,
            expected_duration=travel + service,
            duration_source=planned_route.duration_source,
            metadata={
                "semantic_position": source.position,
                "route_name": planned_route.route_name,
                "semantic_waypoints": planned_route.semantic_waypoints,
                "travel_segments": self._serialize_travel_segments(list(planned_route.travel_segments)),
            },
        )

    def _make_deposit_action(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
        deposit: DepositPoint,
        deposit_count: int,
    ) -> Action | None:
        planned_route = self._best_route(state, side, start, speed, deposit.deposit_route_candidates())
        if planned_route is None:
            return None
        travel = planned_route.travel_duration
        service = self.timing.deposit_duration
        return Action(
            type=ActionType.DEPOSIT,
            target_id=deposit.semantic_id,
            label=f"DEPOSIT_{deposit.semantic_id}_X{deposit_count}",
            target_position=planned_route.motion_waypoints[-1],
            waypoints=planned_route.motion_waypoints,
            service_duration=service,
            travel_duration=travel,
            expected_duration=travel + service,
            duration_source=planned_route.duration_source,
            metadata={
                "semantic_position": deposit.position,
                "route_name": planned_route.route_name,
                "semantic_waypoints": planned_route.semantic_waypoints,
                "deposit_owner": side.value,
                "deposit_count": deposit_count,
            },
        )

    def _make_attack_action(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
        deposit: DepositPoint,
    ) -> Action | None:
        planned_route = self._best_route(
            state,
            side,
            start,
            speed,
            deposit.attack_route_candidates(side),
            clear_deposit_ids=(deposit.semantic_id,),
            allow_final_goal_occupied=True,
        )
        if planned_route is None:
            return None
        travel = planned_route.travel_duration
        # Deposit destruction resolves immediately on arrival at the zone center.
        service = 0.0
        return Action(
            type=ActionType.ATTACK_DEPOSIT,
            target_id=deposit.semantic_id,
            label=f"ATTACK_{deposit.semantic_id}",
            target_position=planned_route.motion_waypoints[-1],
            waypoints=planned_route.motion_waypoints,
            service_duration=service,
            travel_duration=travel,
            expected_duration=travel + service,
            duration_source=planned_route.duration_source,
            metadata={
                "semantic_position": deposit.position,
                "route_name": planned_route.route_name,
                "semantic_waypoints": planned_route.semantic_waypoints,
                "travel_segments": self._serialize_travel_segments(list(planned_route.travel_segments)),
                "push_state": self._infer_push_state(start, planned_route.semantic_waypoints, deposit.position).value,
                "axis": next(
                    (
                        route.axis
                        for route in deposit.attack_route_candidates(side)
                        if route.name == planned_route.route_name
                    ),
                    "free",
                ) or "free",
            },
        )

    def _make_thermometer_action(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
        thermometer: Thermometer,
    ) -> Action | None:
        route = thermometer.route_for_side(side)
        drag_route = thermometer.drag_route_for_side(side)
        planned_motion = self._plan_motion(
            state,
            side,
            start,
            speed,
            route,
        )
        if planned_motion is None:
            return None
        service = self.timing.thermometer_duration
        return Action(
            type=ActionType.DO_THERMOMETER,
            target_id=thermometer.semantic_id,
            label="THERMOMETER",
            target_position=planned_motion.motion_waypoints[-1],
            waypoints=planned_motion.motion_waypoints,
            service_duration=service,
            travel_duration=planned_motion.travel_duration,
            expected_duration=planned_motion.travel_duration + service,
            duration_source=planned_motion.duration_source,
            metadata={
                "drag_start": drag_route[0] if drag_route else route[0],
                "drag_end": drag_route[-1] if drag_route else route[-1],
                "drag_waypoints": drag_route,
                "blocking_source_id": thermometer.blocking_source_id_for_side(side),
                "semantic_waypoints": route,
            },
        )

    def _make_endgame_action(self, state: GameState, side: Side) -> Action | None:
        robot = state.robot_for_side(side)
        config = state.endgame_config_for(side)
        planned_endgame = self._plan_endgame_route(
            state,
            side,
            start=robot.position,
            now=state.t,
        )
        if planned_endgame is None:
            return None
        waypoints = (*planned_endgame.to_chill.motion_waypoints, *planned_endgame.home.motion_waypoints)
        return Action(
            type=ActionType.START_ENDGAME,
            target_id=None,
            label="START_ENDGAME",
            target_position=config.final_home_point,
            waypoints=waypoints,
            service_duration=planned_endgame.wait_duration + planned_endgame.grip_rotate_duration,
            travel_duration=planned_endgame.travel_duration,
            expected_duration=planned_endgame.total_duration,
            duration_source="grid_astar+constants",
            metadata={
                "travel_to_chill": planned_endgame.to_chill.travel_duration,
                "travel_to_chill_waypoints": planned_endgame.to_chill.motion_waypoints,
                "wait_duration": planned_endgame.wait_duration,
                "travel_home": planned_endgame.home.travel_duration,
                "travel_home_waypoints": planned_endgame.home.motion_waypoints,
                "grip_rotate": planned_endgame.grip_rotate_duration,
                "semantic_waypoints": (config.chill_point, *config.home_waypoints),
            },
        )

    def _make_wait_action(self, label: str = "WAIT", duration: float | None = None) -> Action:
        wait_duration = self.timing.wait_duration if duration is None else duration
        return Action(
            type=ActionType.WAIT,
            target_id=None,
            label=label,
            target_position=None,
            waypoints=(),
            service_duration=wait_duration,
            travel_duration=0.0,
            expected_duration=wait_duration,
        )

    def _make_hold_for_chill_action(self, state: GameState, side: Side) -> Action:
        latest_departure = self._latest_grid_chill_departure_time(state, side)
        if latest_departure is None:
            return self._make_wait_action(label="WAIT_FOR_CHILL")
        remaining_until_departure = max(0.0, latest_departure - state.t)
        wait_duration = max(1e-6, min(self.timing.wait_duration, remaining_until_departure))
        return self._make_wait_action(label="WAIT_FOR_CHILL", duration=wait_duration)

    def _must_start_endgame(self, state: GameState, side: Side) -> bool:
        latest_departure = self._latest_grid_chill_departure_time(state, side)
        if latest_departure is None:
            return True
        return state.t >= latest_departure - 1e-9

    def _best_route(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
        routes: tuple[RouteOption, ...],
        clear_source_ids: tuple[int, ...] = (),
        clear_deposit_ids: tuple[int, ...] = (),
        allow_final_goal_occupied: bool = False,
    ) -> PlannedRoute | None:
        viable_routes = [route for route in routes if self._route_is_available(state, route)]
        best: PlannedRoute | None = None
        for route in viable_routes:
            planned_route = self._plan_candidate_route(
                state,
                side,
                start,
                speed,
                route.waypoints,
                route_name=route.name,
                clear_source_ids=clear_source_ids,
                clear_deposit_ids=clear_deposit_ids,
                allow_final_goal_occupied=allow_final_goal_occupied,
            )
            if planned_route is None:
                continue
            if best is None or planned_route.travel_duration < best.travel_duration:
                best = planned_route
        return best

    def _route_is_available(self, state: GameState, route: RouteOption) -> bool:
        for source_id in route.blocked_by_sources:
            source = state.sources.get(source_id)
            if source is None:
                continue
            if source.is_available(state.t):
                return False
        return True

    def _infer_push_state(
        self,
        start: tuple[float, float],
        semantic_waypoints: tuple[tuple[float, float], ...],
        target: tuple[float, float],
    ) -> PushState:
        previous = semantic_waypoints[-2] if len(semantic_waypoints) >= 2 else start
        dx = target[0] - previous[0]
        dy = target[1] - previous[1]
        if abs(dx) >= abs(dy):
            if dx > 1e-9:
                return PushState.PUSHED_RIGHT
            if dx < -1e-9:
                return PushState.PUSHED_LEFT
        else:
            if dy > 1e-9:
                return PushState.PUSHED_UP
            if dy < -1e-9:
                return PushState.PUSHED_DOWN
        return PushState.CLEAR

    def _plan_candidate_route(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
        semantic_waypoints: tuple[tuple[float, float], ...],
        route_name: str,
        clear_source_ids: tuple[int, ...] = (),
        clear_deposit_ids: tuple[int, ...] = (),
        allow_final_goal_occupied: bool = False,
    ) -> PlannedRoute | None:
        if len(semantic_waypoints) < 2:
            planned_route = self._plan_motion(
                state,
                side,
                start,
                speed,
                semantic_waypoints,
                route_name=route_name,
                allow_final_goal_occupied=allow_final_goal_occupied,
            )
            if planned_route is None:
                return None
            return PlannedRoute(
                route_name=planned_route.route_name,
                semantic_waypoints=planned_route.semantic_waypoints,
                motion_waypoints=planned_route.motion_waypoints,
                travel_duration=planned_route.travel_duration,
                duration_source=planned_route.duration_source,
                travel_segments=(
                    PlannedTravelSegment(
                        motion_waypoints=planned_route.motion_waypoints,
                        travel_duration=planned_route.travel_duration,
                        duration_source=planned_route.duration_source,
                    ),
                ),
            )

        first_route = self._plan_motion(
            state,
            side,
            start,
            speed,
            (semantic_waypoints[0],),
            route_name=f"{route_name}_approach",
        )
        second_route = self._plan_motion(
            state,
            side,
            semantic_waypoints[0],
            speed,
            semantic_waypoints[1:],
            route_name=f"{route_name}_final",
            allow_final_goal_occupied=allow_final_goal_occupied,
            ignored_source_ids=set(clear_source_ids),
            ignored_deposit_ids=set(clear_deposit_ids),
        )
        if first_route is None or second_route is None:
            return None
        return PlannedRoute(
            route_name=route_name,
            semantic_waypoints=semantic_waypoints,
            motion_waypoints=(*first_route.motion_waypoints, *second_route.motion_waypoints),
            travel_duration=first_route.travel_duration + second_route.travel_duration,
            duration_source=second_route.duration_source,
            travel_segments=(
                PlannedTravelSegment(
                    motion_waypoints=first_route.motion_waypoints,
                    travel_duration=first_route.travel_duration,
                    duration_source=first_route.duration_source,
                    clear_source_ids=clear_source_ids,
                    clear_deposit_ids=clear_deposit_ids,
                ),
                PlannedTravelSegment(
                    motion_waypoints=second_route.motion_waypoints,
                    travel_duration=second_route.travel_duration,
                    duration_source=second_route.duration_source,
                ),
            ),
        )

    @staticmethod
    def _serialize_travel_segments(segments: list[PlannedTravelSegment]) -> list[dict[str, object]]:
        return [
            {
                "waypoints": segment.motion_waypoints,
                "duration": segment.travel_duration,
                "duration_source": segment.duration_source,
                "clear_source_ids": list(segment.clear_source_ids),
                "clear_deposit_ids": list(segment.clear_deposit_ids),
            }
            for segment in segments
        ]

    def _plan_endgame_route(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        now: float,
    ) -> PlannedEndgameRoute | None:
        robot = state.robot_for_side(side)
        config = state.endgame_config_for(side)
        to_chill = self._plan_motion(
            state,
            side,
            start,
            robot.speed,
            (config.chill_point,),
            route_name="endgame_to_chill",
        )
        if to_chill is None:
            return None
        home = self._plan_motion(
            state,
            side,
            config.chill_point,
            robot.speed,
            config.home_waypoints,
            route_name="endgame_home",
            allow_final_goal_occupied=True,
        )
        if home is None:
            return None
        wait_duration = max(0.0, config.chill_end - (now + to_chill.travel_duration))
        return PlannedEndgameRoute(
            to_chill=to_chill,
            home=home,
            wait_duration=wait_duration,
            grip_rotate_duration=config.grip_rotate_duration,
        )

    def _latest_grid_chill_departure_time(self, state: GameState, side: Side) -> float | None:
        robot = state.robot_for_side(side)
        config: EndgameConfig = state.endgame_config_for(side)
        to_chill = self._plan_motion(
            state,
            side,
            robot.position,
            robot.speed,
            (config.chill_point,),
            route_name="endgame_to_chill",
        )
        if to_chill is None:
            return None
        return config.chill_end - to_chill.travel_duration

    def _plan_motion(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
        semantic_waypoints: tuple[tuple[float, float], ...],
        route_name: str = "default",
        allow_final_goal_occupied: bool = False,
        ignored_source_ids: set[int] | None = None,
        ignored_deposit_ids: set[int] | None = None,
    ) -> PlannedRoute | None:
        self._sync_grid_navigation(
            state,
            side,
            include_enemy_robot=False,
            ignored_source_ids=ignored_source_ids,
            ignored_deposit_ids=ignored_deposit_ids,
        )
        planned_path = self.grid_planner.plan_through_waypoints(
            start,
            semantic_waypoints,
            allow_final_goal_occupied=allow_final_goal_occupied,
        )
        if not planned_path.success:
            return None
        enemy_robot = state.robot_for_side(side.opponent())
        if (
            distance(start, enemy_robot.position) <= self.timing.route_replan_enemy_distance
            and self._enemy_intersects_path(start, planned_path.waypoints, enemy_robot.position)
        ):
            self._sync_grid_navigation(
                state,
                side,
                include_enemy_robot=True,
                ignored_source_ids=ignored_source_ids,
                ignored_deposit_ids=ignored_deposit_ids,
            )
            replanned_path = self.grid_planner.plan_through_waypoints(
                start,
                semantic_waypoints,
                allow_final_goal_occupied=allow_final_goal_occupied,
            )
            if not replanned_path.success:
                return None
            planned_path = replanned_path
        travel_duration = planned_path.distance_m / speed + self.timing.move_overhead * len(semantic_waypoints)
        return PlannedRoute(
            route_name=route_name,
            semantic_waypoints=semantic_waypoints,
            motion_waypoints=planned_path.waypoints,
            travel_duration=travel_duration,
            duration_source=planned_path.duration_source,
        )

    def _sync_grid_navigation(
        self,
        state: GameState,
        side: Side,
        include_enemy_robot: bool,
        ignored_source_ids: set[int] | None = None,
        ignored_deposit_ids: set[int] | None = None,
    ) -> None:
        assert self.grid_map is not None
        dynamic_circles: list[tuple[float, float, float]] = []
        if include_enemy_robot:
            enemy_robot = state.robot_for_side(side.opponent())
            dynamic_circles.append(
                (
                    enemy_robot.position[0],
                    enemy_robot.position[1],
                    self.timing.robot_separation_radius,
                )
            )
        self.grid_map.sync_semantic_state(
            state.sources,
            state.deposits,
            ignored_source_ids=ignored_source_ids,
            ignored_deposit_ids=ignored_deposit_ids,
            dynamic_circles=dynamic_circles,
        )

    def _enemy_intersects_path(
        self,
        start: tuple[float, float],
        waypoints: tuple[tuple[float, float], ...],
        enemy_position: tuple[float, float],
    ) -> bool:
        if not waypoints:
            return False
        previous = start
        for waypoint in waypoints:
            if point_to_segment_distance(enemy_position, previous, waypoint) <= self.timing.robot_separation_radius:
                return True
            previous = waypoint
        return False

    def _thermometer_lane_is_clear(self, state: GameState, side: Side) -> bool:
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


def _policy_debug_row(action: Action, side: Side) -> dict[str, float | int | str | bool | None]:
    row = action.debug_row()
    row["policy_action"] = normalized_action_label(action, side)
    row["policy_target_id"] = normalized_target_id(action.target_id, action.type, side)
    return row
