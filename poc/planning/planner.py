from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poc.domain.actions import Action, ActionType
from poc.domain.config import ActionTimingConfig, UtilityWeights
from poc.domain.endgame import EndgameConfig
from poc.domain.entities import DepositPoint, DepositType, PushState, RouteOption, Side, SourceState, SourcePoint, Thermometer, ThermometerState
from poc.domain.game_state import GameState
from poc.domain.geometry import distance, point_to_segment_distance
from poc.planning.grid_map import DEFAULT_LAYOUT_PATH, GridOccupancyMap
from poc.planning.grid_planner import GridAStarPlanner
from poc.planning.map_config import DEFAULT_ACTION_MASK_HEURISTICS, ActionMaskHeuristicsConfig
from poc.domain.rules import home_deposit_for_side, home_return_blocked, thermometer_lane_is_clear
from poc.domain.scoring import deposit_max_count_for_side, evaluate_action


@dataclass(slots=True)
class PlanningDecision:
    chosen_action: Action
    ranked_actions: list[Action]
    reason: str = ""

    def debug_payload(self, t: float, side: Side) -> dict[str, object]:
        from poc.io.debug import planning_debug_payload

        return planning_debug_payload(
            time=t,
            side=side,
            reason=self.reason,
            chosen_action=self.chosen_action,
            ranked_actions=self.ranked_actions,
        )


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
        action_mask_heuristics: ActionMaskHeuristicsConfig | None = None,
    ) -> None:
        self.timing = timing or ActionTimingConfig()
        self.weights = weights or UtilityWeights()
        self.action_mask_heuristics = action_mask_heuristics or DEFAULT_ACTION_MASK_HEURISTICS
        resolved_layout = DEFAULT_LAYOUT_PATH if layout_path is None else Path(layout_path)
        if not resolved_layout.exists():
            raise FileNotFoundError(f"Grid layout not found: {resolved_layout}")
        self.grid_map = GridOccupancyMap.from_layout(resolved_layout, team_color="all")
        self.grid_planner = GridAStarPlanner(self.grid_map)
        self._motion_plan_cache: dict[tuple[object, ...], PlannedRoute | None] | None = None
        self._endgame_route_cache: dict[tuple[object, ...], tuple[PlannedRoute, PlannedRoute] | None] | None = None
        self._can_fit_before_endgame_cache: dict[tuple[object, ...], bool] | None = None
        self._latest_chill_departure_cache: dict[tuple[object, ...], float | None] | None = None
        self._point_cell_cache: dict[tuple[float, float], tuple[int, int]] = {}

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
        previous_motion_cache = self._motion_plan_cache
        previous_endgame_route_cache = self._endgame_route_cache
        previous_can_fit_cache = self._can_fit_before_endgame_cache
        previous_latest_chill_cache = self._latest_chill_departure_cache
        self._motion_plan_cache = {}
        self._endgame_route_cache = {}
        self._can_fit_before_endgame_cache = {}
        self._latest_chill_departure_cache = {}
        try:
            if state.endgame_started_for(side):
                return [self._score(state, side, self._make_wait_action())]

            candidates = self._generate_candidates(state, side)
            candidates = self._filter_invalid_candidate_actions(state, side, candidates)
            ranked = [self._score(state, side, action) for action in candidates]
            ranked.sort(key=lambda action: action.score, reverse=True)
            return ranked
        finally:
            self._motion_plan_cache = previous_motion_cache
            self._endgame_route_cache = previous_endgame_route_cache
            self._can_fit_before_endgame_cache = previous_can_fit_cache
            self._latest_chill_departure_cache = previous_latest_chill_cache

    def _score(self, state: GameState, side: Side, action: Action) -> Action:
        return evaluate_action(state, side, action, self.timing, self.weights)

    def _generate_candidates(self, state: GameState, side: Side) -> list[Action]:
        robot = state.robot_for_side(side)
        endgame_config = state.endgame_config_for(side)
        home_return_blocked = self._home_return_blocked(state, side)
        endgame_action = self._make_endgame_action(state, side)
        play_to_end_action = self._make_play_to_end_action(state, side)
        candidates: list[Action] = [self._make_wait_action()]
        if endgame_action is not None:
            candidates.insert(0, endgame_action)
        if play_to_end_action is not None:
            candidates.insert(1 if endgame_action is not None else 0, play_to_end_action)

        if state.t >= endgame_config.main_pipeline_deadline:
            if state.play_to_end_started_for(side):
                home_return_blocked = False
            if home_return_blocked and play_to_end_action is None:
                return candidates
            if not state.play_to_end_started_for(side) and self._must_start_endgame(state, side):
                return candidates
            if not state.play_to_end_started_for(side):
                hold_action = self._make_hold_for_chill_action(state, side)
                if play_to_end_action is not None:
                    return [play_to_end_action, hold_action]
                return [hold_action]

        if robot.load == 0:
            for source in state.sources.values():
                if not source.is_available(state.t):
                    continue
                action = self._make_pick_action(state, side, robot.position, robot.speed, source)
                if action is None:
                    continue
                if state.play_to_end_started_for(side) or self._can_fit_before_endgame(state, side, action):
                    candidates.append(action)

        if robot.load > 0:
            for deposit in state.friendly_deposits(side, include_home=True, include_neutral=True):
                max_count = deposit_max_count_for_side(deposit, side, robot.load)
                if max_count <= 0:
                    continue
                planned_route = self._best_route(
                    state,
                    side,
                    robot.position,
                    robot.speed,
                    deposit.deposit_route_candidates(),
                )
                if planned_route is None:
                    continue
                for deposit_count in range(1, max_count + 1):
                    action = self._build_deposit_action(
                        planned_route=planned_route,
                        side=side,
                        deposit=deposit,
                        deposit_count=deposit_count,
                    )
                    if state.play_to_end_started_for(side) or self._can_fit_before_endgame(state, side, action):
                        candidates.append(action)

        if (
            not state.thermometer.is_done_for_side(side)
            and (
                not self.action_mask_heuristics.require_clear_thermometer_lane
                or self._thermometer_lane_is_clear(state, side)
            )
        ):
            thermo = self._make_thermometer_action(state, side, robot.position, robot.speed, state.thermometer)
            if thermo is not None and (state.play_to_end_started_for(side) or self._can_fit_before_endgame(state, side, thermo)):
                candidates.append(thermo)

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
            if state.play_to_end_started_for(side) or self._can_fit_before_endgame(state, side, action):
                candidates.append(action)

        return candidates

    def _can_fit_before_endgame(self, state: GameState, side: Side, action: Action) -> bool:
        home_return_blocked = self._home_return_blocked(state, side)
        cache_key = self._can_fit_before_endgame_cache_key(
            state=state,
            side=side,
            action=action,
            home_return_blocked=home_return_blocked,
        )
        if self._can_fit_before_endgame_cache is not None and cache_key in self._can_fit_before_endgame_cache:
            return self._can_fit_before_endgame_cache[cache_key]
        if home_return_blocked:
            result = state.t + action.expected_duration <= state.T_end
            if self._can_fit_before_endgame_cache is not None:
                self._can_fit_before_endgame_cache[cache_key] = result
            return result
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
            if self._can_fit_before_endgame_cache is not None:
                self._can_fit_before_endgame_cache[cache_key] = False
            return False
        finishes_before_chill = (
            state.t + action.expected_duration + planned_endgame.to_chill.travel_duration
            <= endgame_config.chill_end - endgame_config.chill_margin
        )
        finishes_home = (
            endgame_config.chill_end + planned_endgame.home.travel_duration + planned_endgame.grip_rotate_duration
            <= state.T_end - endgame_config.home_margin
        )
        result = finishes_before_chill and finishes_home
        if self._can_fit_before_endgame_cache is not None:
            self._can_fit_before_endgame_cache[cache_key] = result
        return result

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
        target_position = self._route_target_position(planned_route, fallback=source.position)
        return Action(
            type=ActionType.PICK,
            target_id=source.semantic_id,
            label=f"PICK_{source.semantic_id}",
            target_position=target_position,
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
        return self._build_deposit_action(
            planned_route=planned_route,
            side=side,
            deposit=deposit,
            deposit_count=deposit_count,
        )

    def _build_deposit_action(
        self,
        *,
        planned_route: PlannedRoute,
        side: Side,
        deposit: DepositPoint,
        deposit_count: int,
    ) -> Action:
        travel = planned_route.travel_duration
        service = self.timing.deposit_duration
        target_position = self._route_target_position(planned_route, fallback=deposit.position)
        return Action(
            type=ActionType.DEPOSIT,
            target_id=deposit.semantic_id,
            label=f"DEPOSIT_{deposit.semantic_id}_X{deposit_count}",
            target_position=target_position,
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
        target_position = self._route_target_position(planned_route, fallback=deposit.position)
        return Action(
            type=ActionType.ATTACK_DEPOSIT,
            target_id=deposit.semantic_id,
            label=f"ATTACK_{deposit.semantic_id}",
            target_position=target_position,
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
        target_position = self._route_target_position(planned_motion, fallback=route[-1])
        return Action(
            type=ActionType.DO_THERMOMETER,
            target_id=thermometer.semantic_id,
            label="THERMOMETER",
            target_position=target_position,
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
        if self._home_return_blocked(state, side):
            return None
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

    def _make_play_to_end_action(self, state: GameState, side: Side) -> Action | None:
        if state.play_to_end_started_for(side):
            return None
        if state.t < state.endgame_config_for(side).main_pipeline_deadline:
            return None
        return Action(
            type=ActionType.PLAY_TO_END,
            target_id=None,
            label="PLAY_TO_END",
            target_position=None,
            waypoints=(),
            service_duration=1e-6,
            travel_duration=0.0,
            expected_duration=1e-6,
            duration_source="mode_switch",
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
        if self._home_return_blocked(state, side):
            return False
        latest_departure = self._latest_grid_chill_departure_time(state, side)
        if latest_departure is None:
            return True
        return state.t >= latest_departure - 1e-9

    def _home_return_blocked(self, state: GameState, side: Side) -> bool:
        return home_return_blocked(state, side)

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

    def _filter_invalid_candidate_actions(
        self,
        state: GameState,
        side: Side,
        candidates: list[Action],
    ) -> list[Action]:
        if not candidates:
            return candidates
        self._sync_grid_navigation(state, side, include_enemy_robot=True)
        valid_candidates: list[Action] = []
        for action in candidates:
            if (
                (
                    not self.action_mask_heuristics.check_semantic_waypoints
                    or self._action_semantic_waypoints_are_valid(action)
                )
                and (
                    not self.action_mask_heuristics.check_attack_enemy_block
                    or not self._attack_target_is_currently_enemy_blocked(state, side, action)
                )
            ):
                valid_candidates.append(action)
        return valid_candidates

    def _action_semantic_waypoints_are_valid(self, action: Action) -> bool:
        assert self.grid_map is not None
        semantic_waypoints = tuple(
            (float(point[0]), float(point[1]))
            for point in action.metadata.get("semantic_waypoints", ())
        )
        if not semantic_waypoints:
            return True
        allow_blocked_final_waypoint = (
            action.type.value in self.action_mask_heuristics.allow_blocked_final_waypoint_for
        )
        last_index = len(semantic_waypoints) - 1
        for index, point in enumerate(semantic_waypoints):
            if allow_blocked_final_waypoint and index == last_index:
                continue
            row, col = self._point_cell(point)
            if self.grid_map.is_blocked(row, col, use_inflated=True):
                return False
        return True

    def _attack_target_is_currently_enemy_blocked(
        self,
        state: GameState,
        side: Side,
        action: Action,
    ) -> bool:
        if action.type is not ActionType.ATTACK_DEPOSIT:
            return False
        semantic_waypoints = tuple(
            (float(point[0]), float(point[1]))
            for point in action.metadata.get("semantic_waypoints", ())
        )
        if not semantic_waypoints:
            return False
        enemy_position = state.robot_for_side(side.opponent()).position
        separation_radius = self.timing.attack_enemy_block_radius
        final_waypoint = semantic_waypoints[-1]
        if (
            self.action_mask_heuristics.check_attack_final_waypoint
            and distance(enemy_position, final_waypoint) < separation_radius
        ):
            return True
        if self.action_mask_heuristics.check_attack_final_segment and len(semantic_waypoints) >= 2:
            approach_waypoint = semantic_waypoints[-2]
            if point_to_segment_distance(enemy_position, approach_waypoint, final_waypoint) < separation_radius:
                return True
        return False

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
        endgame_base = self._plan_endgame_route_base(state, side, start)
        if endgame_base is None:
            return None
        to_chill, home = endgame_base
        config = state.endgame_config_for(side)
        wait_duration = max(0.0, config.chill_end - (now + to_chill.travel_duration))
        return PlannedEndgameRoute(
            to_chill=to_chill,
            home=home,
            wait_duration=wait_duration,
            grip_rotate_duration=config.grip_rotate_duration,
        )

    def _plan_endgame_route_base(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
    ) -> tuple[PlannedRoute, PlannedRoute] | None:
        robot = state.robot_for_side(side)
        config = state.endgame_config_for(side)
        cache_key = self._endgame_route_cache_key(state, side, start, robot.speed)
        if self._endgame_route_cache is not None and cache_key in self._endgame_route_cache:
            return self._endgame_route_cache[cache_key]
        to_chill = self._plan_motion(
            state,
            side,
            start,
            robot.speed,
            (config.chill_point,),
            route_name="endgame_to_chill",
        )
        if to_chill is None:
            if self._endgame_route_cache is not None:
                self._endgame_route_cache[cache_key] = None
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
            if self._endgame_route_cache is not None:
                self._endgame_route_cache[cache_key] = None
            return None
        result = (to_chill, home)
        if self._endgame_route_cache is not None:
            self._endgame_route_cache[cache_key] = result
        return result

    def _latest_grid_chill_departure_time(self, state: GameState, side: Side) -> float | None:
        robot = state.robot_for_side(side)
        cache_key = self._latest_chill_departure_cache_key(state, side, robot.position, robot.speed)
        if self._latest_chill_departure_cache is not None and cache_key in self._latest_chill_departure_cache:
            return self._latest_chill_departure_cache[cache_key]
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
            if self._latest_chill_departure_cache is not None:
                self._latest_chill_departure_cache[cache_key] = None
            return None
        result = config.chill_end - to_chill.travel_duration
        if self._latest_chill_departure_cache is not None:
            self._latest_chill_departure_cache[cache_key] = result
        return result

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
        cache_key = self._motion_cache_key(
            state=state,
            side=side,
            start=start,
            speed=speed,
            semantic_waypoints=semantic_waypoints,
            route_name=route_name,
            allow_final_goal_occupied=allow_final_goal_occupied,
            ignored_source_ids=ignored_source_ids,
            ignored_deposit_ids=ignored_deposit_ids,
        )
        if self._motion_plan_cache is not None and cache_key in self._motion_plan_cache:
            return self._motion_plan_cache[cache_key]
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
            replanning_waypoints = semantic_waypoints
            escape_waypoint = self._enemy_escape_waypoint(start, semantic_waypoints, enemy_robot.position)
            if escape_waypoint is not None:
                replanning_waypoints = (escape_waypoint, *semantic_waypoints)
            self._sync_grid_navigation(
                state,
                side,
                include_enemy_robot=True,
                ignored_source_ids=ignored_source_ids,
                ignored_deposit_ids=ignored_deposit_ids,
            )
            replanned_path = self.grid_planner.plan_through_waypoints(
                start,
                replanning_waypoints,
                allow_final_goal_occupied=allow_final_goal_occupied,
            )
            if not replanned_path.success:
                if self._motion_plan_cache is not None:
                    self._motion_plan_cache[cache_key] = None
                return None
            planned_path = replanned_path
        travel_duration = planned_path.distance_m / speed + self.timing.move_overhead * len(semantic_waypoints)
        planned_route = PlannedRoute(
            route_name=route_name,
            semantic_waypoints=semantic_waypoints,
            motion_waypoints=planned_path.waypoints,
            travel_duration=travel_duration,
            duration_source=planned_path.duration_source,
        )
        if self._motion_plan_cache is not None:
            self._motion_plan_cache[cache_key] = planned_route
        return planned_route

    def _enemy_escape_waypoint(
        self,
        start: tuple[float, float],
        semantic_waypoints: tuple[tuple[float, float], ...],
        enemy_position: tuple[float, float],
    ) -> tuple[float, float] | None:
        distance_to_enemy = distance(start, enemy_position)
        escape_radius = self.timing.route_replan_enemy_distance + self.grid_map.config.resolution_m
        if distance_to_enemy >= escape_radius:
            return None

        dx = start[0] - enemy_position[0]
        dy = start[1] - enemy_position[1]
        norm = (dx * dx + dy * dy) ** 0.5
        if norm <= 1e-9 and semantic_waypoints:
            dx = semantic_waypoints[0][0] - enemy_position[0]
            dy = semantic_waypoints[0][1] - enemy_position[1]
            norm = (dx * dx + dy * dy) ** 0.5
        if norm <= 1e-9:
            dx = 1.0
            dy = 0.0
            norm = 1.0

        scale = escape_radius / norm
        escape_waypoint = (
            enemy_position[0] + dx * scale,
            enemy_position[1] + dy * scale,
        )
        if distance(start, escape_waypoint) <= 1e-9:
            return None
        return escape_waypoint

    def _motion_cache_key(
        self,
        *,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
        semantic_waypoints: tuple[tuple[float, float], ...],
        route_name: str,
        allow_final_goal_occupied: bool,
        ignored_source_ids: set[int] | None,
        ignored_deposit_ids: set[int] | None,
    ) -> tuple[object, ...]:
        assert self.grid_map is not None
        ignored_sources = tuple(sorted(ignored_source_ids or ()))
        ignored_deposits = tuple(sorted(ignored_deposit_ids or ()))
        start_cell = self.grid_map.world_to_grid(*start)
        waypoint_cells = tuple(self._point_cell(point) for point in semantic_waypoints)
        return (
            side.value,
            start_cell,
            waypoint_cells,
            round(speed, 6),
            route_name,
            allow_final_goal_occupied,
            ignored_sources,
            ignored_deposits,
            state.t,
        )

    def _endgame_route_cache_key(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
    ) -> tuple[object, ...]:
        assert self.grid_map is not None
        return (
            side.value,
            self.grid_map.world_to_grid(*start),
            round(speed, 6),
            state.t,
        )

    def _latest_chill_departure_cache_key(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
    ) -> tuple[object, ...]:
        return self._endgame_route_cache_key(state, side, start, speed)

    def _can_fit_before_endgame_cache_key(
        self,
        *,
        state: GameState,
        side: Side,
        action: Action,
        home_return_blocked: bool,
    ) -> tuple[object, ...]:
        assert self.grid_map is not None
        action_end_position = action.waypoints[-1] if action.waypoints else state.robot_for_side(side).position
        return (
            side.value,
            self._point_cell(action_end_position),
            round(action.expected_duration, 6),
            home_return_blocked,
            state.t,
        )

    def _point_cell(self, point: tuple[float, float]) -> tuple[int, int]:
        assert self.grid_map is not None
        cached = self._point_cell_cache.get(point)
        if cached is not None:
            return cached
        cell = self.grid_map.world_to_grid(*point)
        self._point_cell_cache[point] = cell
        return cell

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
        return thermometer_lane_is_clear(state, side)

    @staticmethod
    def _route_target_position(
        planned_route: PlannedRoute,
        fallback: tuple[float, float],
    ) -> tuple[float, float]:
        if planned_route.motion_waypoints:
            return planned_route.motion_waypoints[-1]
        if planned_route.semantic_waypoints:
            return planned_route.semantic_waypoints[-1]
        return fallback
