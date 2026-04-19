from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from poc.actions import Action, ActionType
from poc.endgame import can_finish_scoring_action, estimate_endgame_duration
from poc.entities import DepositPoint, DepositType, RouteOption, Side, SourceState, SourcePoint, Thermometer, ThermometerState
from poc.game_state import GameState
from poc.geometry import distance, path_length, point_to_segment_distance
from poc.scoring import ActionTimingConfig, UtilityWeights, evaluate_action, travel_time


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
            "chosen_action": self.chosen_action.debug_row(),
            "candidates": [action.debug_row() for action in self.ranked_actions],
        }


class UtilityPlanner:
    def __init__(
        self,
        timing: ActionTimingConfig | None = None,
        weights: UtilityWeights | None = None,
    ) -> None:
        self.timing = timing or ActionTimingConfig()
        self.weights = weights or UtilityWeights()

    def plan(self, state: GameState, side: Side) -> PlanningDecision:
        ranked = self.rank_actions(state, side)
        chosen = ranked[0] if ranked else self._make_wait_action()
        reason = "utility_max"
        if chosen.type is ActionType.START_ENDGAME:
            reason = "endgame_window"
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
        candidates: list[Action] = [self._make_endgame_action(state, side), self._make_wait_action()]

        if state.t >= endgame_config.main_pipeline_deadline:
            return candidates

        if robot.load < robot.capacity:
            for source in state.sources.values():
                if not source.is_available(state.t):
                    continue
                action = self._make_pick_action(state, robot.position, robot.speed, source)
                if action is None:
                    continue
                if self._can_fit_before_endgame(state, side, action):
                    candidates.append(action)

        if robot.load > 0:
            for deposit in state.friendly_deposits(side, include_home=True, include_neutral=True):
                if deposit.kind is DepositType.STORAGE and deposit.total_items() > 0:
                    continue
                action = self._make_deposit_action(state, side, robot.position, robot.speed, deposit)
                if action is None:
                    continue
                if self._can_fit_before_endgame(state, side, action):
                    candidates.append(action)

        if (
            not state.thermometer.is_done_for_side(side)
            and self._thermometer_lane_is_clear(state, side)
        ):
            thermo = self._make_thermometer_action(state, robot.position, robot.speed, state.thermometer, side)
            if self._can_fit_before_endgame(state, side, thermo):
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
        return can_finish_scoring_action(
            now=state.t,
            action_duration=action.expected_duration,
            action_end_position=action_end_position,
            speed=robot.speed,
            move_overhead=self.timing.move_overhead,
            config=endgame_config,
            match_end=state.T_end,
        )

    def _make_pick_action(
        self,
        state: GameState,
        start: tuple[float, float],
        speed: float,
        source: SourcePoint,
    ) -> Action | None:
        route = self._best_route(
            state,
            start,
            speed,
            source.collection_routes(),
            ignored_source_ids={source.semantic_id},
        )
        if route is None:
            return None
        travel = self._travel_time_for_waypoints(
            state,
            start,
            speed,
            route.waypoints,
            ignored_source_ids={source.semantic_id},
        )
        service = self.timing.pick_duration + self.timing.align_duration
        return Action(
            type=ActionType.PICK,
            target_id=source.semantic_id,
            label=f"PICK_{source.semantic_id}",
            target_position=route.waypoints[-1],
            waypoints=route.waypoints,
            service_duration=service,
            travel_duration=travel,
            expected_duration=travel + service,
            metadata={
                "semantic_position": source.position,
                "route_name": route.name,
            },
        )

    def _make_deposit_action(
        self,
        state: GameState,
        side: Side,
        start: tuple[float, float],
        speed: float,
        deposit: DepositPoint,
    ) -> Action | None:
        route = self._best_route(state, start, speed, deposit.deposit_route_candidates())
        if route is None:
            return None
        travel = self._travel_time_for_waypoints(state, start, speed, route.waypoints)
        service = self.timing.deposit_duration
        return Action(
            type=ActionType.DEPOSIT,
            target_id=deposit.semantic_id,
            label=f"DEPOSIT_{deposit.semantic_id}",
            target_position=route.waypoints[-1],
            waypoints=route.waypoints,
            service_duration=service,
            travel_duration=travel,
            expected_duration=travel + service,
            metadata={
                "semantic_position": deposit.position,
                "route_name": route.name,
                "deposit_owner": side.value,
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
        route = self._best_route(state, start, speed, deposit.attack_route_candidates(side))
        if route is None:
            return None
        travel = self._travel_time_for_waypoints(state, start, speed, route.waypoints)
        # Deposit destruction resolves immediately on arrival at the zone center.
        service = 0.0
        return Action(
            type=ActionType.ATTACK_DEPOSIT,
            target_id=deposit.semantic_id,
            label=f"ATTACK_{deposit.semantic_id}",
            target_position=route.waypoints[-1],
            waypoints=route.waypoints,
            service_duration=service,
            travel_duration=travel,
            expected_duration=travel + service,
            metadata={
                "semantic_position": deposit.position,
                "route_name": route.name,
                "axis": route.axis or "free",
            },
        )

    def _make_thermometer_action(
        self,
        state: GameState,
        start: tuple[float, float],
        speed: float,
        thermometer: Thermometer,
        side: Side,
    ) -> Action:
        route = thermometer.route_for_side(side)
        travel = self._travel_time_for_waypoints(state, start, speed, route)
        service = self.timing.thermometer_duration
        return Action(
            type=ActionType.DO_THERMOMETER,
            target_id=thermometer.semantic_id,
            label="THERMOMETER",
            target_position=route[-1],
            waypoints=route,
            service_duration=service,
            travel_duration=travel,
            expected_duration=travel + service,
            metadata={
                "drag_start": route[1],
                "drag_end": route[2],
                "blocking_source_id": thermometer.blocking_source_id_for_side(side),
            },
        )

    def _make_endgame_action(self, state: GameState, side: Side) -> Action:
        robot = state.robot_for_side(side)
        config = state.endgame_config_for(side)
        durations = estimate_endgame_duration(
            now=state.t,
            start=robot.position,
            speed=robot.speed,
            move_overhead=self.timing.move_overhead,
            config=config,
        )
        return Action(
            type=ActionType.START_ENDGAME,
            target_id=None,
            label="START_ENDGAME",
            target_position=config.final_home_point,
            waypoints=(config.chill_point, *config.home_waypoints),
            service_duration=durations["wait"] + durations["grip_rotate"],
            travel_duration=durations["to_chill"] + durations["home_travel"],
            expected_duration=durations["total"],
            metadata={
                "travel_to_chill": durations["to_chill"],
                "wait_duration": durations["wait"],
                "travel_home": durations["home_travel"],
                "grip_rotate": durations["grip_rotate"],
            },
        )

    def _make_wait_action(self) -> Action:
        return Action(
            type=ActionType.WAIT,
            target_id=None,
            label="WAIT",
            target_position=None,
            waypoints=(),
            service_duration=self.timing.wait_duration,
            travel_duration=0.0,
            expected_duration=self.timing.wait_duration,
        )

    def _best_route(
        self,
        state: GameState,
        start: tuple[float, float],
        speed: float,
        routes: tuple[RouteOption, ...],
        ignored_source_ids: set[int] | None = None,
    ) -> RouteOption | None:
        viable_routes = [route for route in routes if self._route_is_available(state, route)]
        if not viable_routes:
            return None
        return min(
            viable_routes,
            key=lambda route: self._travel_time_for_waypoints(
                state,
                start,
                speed,
                route.waypoints,
                ignored_source_ids=ignored_source_ids,
            ),
        )

    def _route_is_available(self, state: GameState, route: RouteOption) -> bool:
        for source_id in route.blocked_by_sources:
            source = state.sources.get(source_id)
            if source is None:
                continue
            if source.is_available(state.t):
                return False
        return True

    def _travel_time_for_waypoints(
        self,
        state: GameState,
        start: tuple[float, float],
        speed: float,
        waypoints: tuple[tuple[float, float], ...],
        ignored_source_ids: set[int] | None = None,
    ) -> float:
        if not waypoints:
            return 0.0
        total_distance = path_length((start, *waypoints))
        obstacle_hits = self._count_obstacle_intersections(
            state,
            start,
            waypoints,
            ignored_source_ids=ignored_source_ids or set(),
        )
        return (
            total_distance / speed
            + self.timing.move_overhead * len(waypoints)
            + obstacle_hits * self.timing.obstacle_detour_penalty
        )

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

    def _count_obstacle_intersections(
        self,
        state: GameState,
        start: tuple[float, float],
        waypoints: tuple[tuple[float, float], ...],
        ignored_source_ids: set[int],
    ) -> int:
        points = (start, *waypoints)
        intersections = 0
        for source in state.sources.values():
            if source.semantic_id in ignored_source_ids:
                continue
            if source.available_from_t > state.t:
                continue
            if source.state is SourceState.EMPTY or source.available_items <= 0:
                continue
            if any(
                point_to_segment_distance(source.position, segment_start, segment_end)
                <= self.timing.obstacle_clearance_radius
                for segment_start, segment_end in pairwise(points)
            ):
                intersections += 1
        return intersections
