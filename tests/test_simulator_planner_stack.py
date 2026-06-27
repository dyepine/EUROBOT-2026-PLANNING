from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from copy import deepcopy
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from poc.domain.actions import Action, ActionType
from poc.control.controllers import build_scripted_controller
from poc.domain.entities import DEFAULT_ROBOT_SPEED_MPS, Side, SourceState, Thermometer
from poc.domain.game_state import GameState
from poc.domain.geometry import advance_along_path, distance
from poc.planning.grid_map import DEFAULT_LAYOUT_PATH, GridOccupancyMap
from poc.planning.grid_planner import GridAStarPlanner
from poc.simulation.observations import DecisionObservation, PreviousTickObservation, RobotObservation
from poc.control.opponent_policy import FixedSequencePolicy, OrderedActionStep, build_opponent_policy
from poc.planning.planner import UtilityPlanner
from poc.rl.config import PPOConfig, SelfPlayConfig
from poc.rl.action_space import (
    DEFAULT_ACTION_SPACE,
    build_rl_policy_step,
    resolve_policy_action,
)
from poc.rl.encoder import (
    DEFAULT_FLAT_FEATURE_KEYS,
    RLObservationConfig,
    build_rl_observation,
)
from poc.rl.match import OpponentSpec, classify_opponent_kind, enemy_speed_scale_for_match
from poc.rl.transitions import (
    build_rl_transitions_from_match_result,
)
from poc.rl.ppo import PPORolloutItem, compute_gae_returns
from poc.rl.selfplay import (
    EvaluationSummary,
    _desired_rollout_pending_matches,
    _estimated_rollout_steps_per_match,
    _should_schedule_rollout_match,
)
from poc.rl.workers import PersistentWorkerPools, _build_worker_request, _play_match_worker
from poc.rl_train import (
    aggregate_eval_metrics,
    aggregate_eval_metrics_for_kinds,
    build_parser,
    config_for_resume,
    resolve_output_dir,
    resolve_start_paths,
)
from poc.domain.scoring import deposit_max_count, deposit_max_count_for_side
from poc.simulation.scenarios import build_scenario
from poc.planning.semantic_map import build_default_semantic_map
from poc.simulation.simulator import ActiveAction, ActivePhase, Simulator


def _previous_tick_from_state(state: GameState) -> PreviousTickObservation:
    return PreviousTickObservation(
        t=float(state.t),
        blue_robot=RobotObservation(position=state.robot_for_side(Side.BLUE).position),
        yellow_robot=RobotObservation(position=state.robot_for_side(Side.YELLOW).position),
    )


def _decision_observation(
    state: GameState,
    side: Side,
    *,
    ranked_actions: tuple[Action, ...] | list[Action] = (),
    previous_state: GameState | PreviousTickObservation | None = None,
    dt: float = 1.0,
) -> DecisionObservation:
    previous_tick = _previous_tick_from_state(previous_state) if isinstance(previous_state, GameState) else previous_state
    return DecisionObservation(
        state=state,
        side=side,
        ranked_actions=tuple(ranked_actions),
        previous_state=previous_tick,
        dt=dt,
    )


class SimulatorPlannerStackTests(unittest.TestCase):
    def test_new_scripted_opponents_and_scenarios_are_available(self) -> None:
        self.assertEqual(build_opponent_policy("storage_first").name, "storage_first")
        self.assertEqual(build_opponent_policy("home_safe").name, "home_safe")
        self.assertEqual(build_opponent_policy("yellow_side_fixed_sequence").name, "yellow_side_fixed_sequence")
        self.assertEqual(build_opponent_policy("stochastic_planner@7").name, "stochastic_planner@7")
        self.assertEqual(build_opponent_policy("uniform_random@9").name, "uniform_random@9")
        self.assertEqual(build_opponent_policy("randomized_aggressive@11").name, "randomized_aggressive@11")
        self.assertEqual(build_scenario("storage_first_enemy").default_opponent_policy_name, "storage_first")
        self.assertEqual(build_scenario("home_safe_enemy").default_opponent_policy_name, "home_safe")
        self.assertEqual(
            build_scenario("yellow_side_fixed_sequence_enemy").default_opponent_policy_name,
            "yellow_side_fixed_sequence",
        )
        self.assertEqual(build_scenario("stochastic_enemy").default_opponent_policy_name, "stochastic_planner@1")
        self.assertEqual(build_scenario("uniform_random_enemy").default_opponent_policy_name, "uniform_random@1")
        self.assertEqual(build_scenario("randomized_aggressive_enemy").default_opponent_policy_name, "randomized_aggressive@1")

    def test_yellow_side_fixed_sequence_policy_follows_configured_order(self) -> None:
        scenario = build_scenario("baseline", seed=1, our_side=Side.BLUE)
        state = scenario.game_state
        planner = UtilityPlanner()
        policy = build_opponent_policy("yellow_side_fixed_sequence")
        side = Side.YELLOW

        self.assertEqual(policy.choose_action(state, planner, side).label, "PICK_24")

        state.enemy_robot.load = 4
        state.sources[24].available_items = 0
        state.sources[24].state = SourceState.EMPTY
        self.assertEqual(policy.choose_action(state, planner, side).label, "DEPOSIT_27_X4")

        state.enemy_robot.load = 0
        state.deposits[27].yellow_items = 4
        self.assertEqual(policy.choose_action(state, planner, side).label, "PICK_23")

        state.sources[23].available_items = 0
        state.sources[23].state = SourceState.EMPTY
        self.assertEqual(policy.choose_action(state, planner, side).label, "THERMOMETER")

        state.enemy_robot.load = 4
        state.thermometer.mark_done_for_side(side)
        self.assertEqual(policy.choose_action(state, planner, side).label, "DEPOSIT_26_X4")

    def test_fixed_sequence_policy_supports_explicit_deposit_x2_steps(self) -> None:
        scenario = build_scenario("baseline", seed=1, our_side=Side.BLUE)
        state = scenario.game_state
        planner = UtilityPlanner()
        side = Side.YELLOW
        policy = FixedSequencePolicy(
            name="deposit_x2_test",
            steps=(
                OrderedActionStep(ActionType.DEPOSIT, 27, deposit_count=2),
                OrderedActionStep(ActionType.DEPOSIT, 26, deposit_count=2),
            ),
        )

        state.enemy_robot.load = 4
        self.assertEqual(policy.choose_action(state, planner, side).label, "DEPOSIT_27_X2")

        state.deposits[27].yellow_items = 2
        state.enemy_robot.load = 2
        self.assertEqual(policy.choose_action(state, planner, side).label, "DEPOSIT_26_X2")

    def test_mars_endgame_points_and_summary_are_present(self) -> None:
        scenario_with_mars = build_scenario("baseline", seed=1)
        simulator_with_mars = Simulator(
            state=scenario_with_mars.game_state,
            scenario_name=scenario_with_mars.name,
            opponent_controller=build_scripted_controller(scenario_with_mars.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        result_with_mars = simulator_with_mars.run()
        self.assertEqual(result_with_mars.summary["our_mars_pantry_count"], 3)
        self.assertTrue(result_with_mars.summary["our_mars_all_eating"])
        self.assertIn("blue", result_with_mars.mars)
        self.assertEqual(len(result_with_mars.mars["blue"]), 3)

        scenario_without_mars = build_scenario("baseline", seed=1)
        scenario_without_mars.game_state.mars_by_side = {Side.BLUE: (), Side.YELLOW: ()}
        simulator_without_mars = Simulator(
            state=scenario_without_mars.game_state,
            scenario_name=scenario_without_mars.name,
            opponent_controller=build_scripted_controller(scenario_without_mars.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        result_without_mars = simulator_without_mars.run()
        self.assertEqual(
            result_with_mars.summary["our_score"] - result_without_mars.summary["our_score"],
            25,
        )
        self.assertGreaterEqual(result_with_mars.summary["enemy_mars_pantry_count"], 0)
        self.assertGreaterEqual(result_with_mars.summary["enemy_mars_collision_count"], 0)

    def test_home_deposit_is_blocked_once_occupied_except_for_endgame_arrival(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        our_side = scenario.game_state.our_side
        home = next(
            deposit for deposit in scenario.game_state.deposits.values()
            if deposit.kind.value == "home" and deposit.owner is our_side
        )
        home.add_items(our_side, 2)
        self.assertEqual(deposit_max_count(home, 4), 0)

        scenario.game_state.mars_by_side = {Side.BLUE: (), Side.YELLOW: ()}
        scenario.game_state.t = scenario.game_state.T_end
        scenario.game_state.our_robot.position = scenario.game_state.endgame_config_for(our_side).final_home_point
        scenario.game_state.our_robot.load = 3
        scenario.game_state.set_endgame_started(our_side, True)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        result = simulator.run()
        self.assertEqual(result.summary["our_score"], 14)

    def test_endgame_action_is_not_generated_when_home_already_contains_our_pile(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        our_side = scenario.game_state.our_side
        scenario.game_state.t = 85.0
        home = next(
            deposit for deposit in scenario.game_state.deposits.values()
            if deposit.kind.value == "home" and deposit.owner is our_side
        )
        home.add_items(our_side, 2)
        planner = UtilityPlanner()
        ranked = planner.rank_actions(scenario.game_state, our_side)
        self.assertTrue(ranked)
        self.assertFalse(any(action.type is ActionType.START_ENDGAME for action in ranked))

    def test_actions_before_deadline_remain_available_when_home_is_occupied(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        our_side = scenario.game_state.our_side
        home = next(
            deposit for deposit in scenario.game_state.deposits.values()
            if deposit.kind.value == "home" and deposit.owner is our_side
        )
        home.add_items(our_side, 2)
        planner = UtilityPlanner()
        ranked = planner.rank_actions(scenario.game_state, our_side)
        self.assertTrue(ranked)
        self.assertTrue(any(action.type is not ActionType.WAIT for action in ranked))

    def test_attack_can_be_generated_even_when_robot_carries_load(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.our_robot.load = 2
        scenario.game_state.deposits[17].add_items(Side.YELLOW, 1)
        planner = UtilityPlanner()
        ranked = planner.rank_actions(scenario.game_state, Side.BLUE)
        self.assertTrue(any(action.type is ActionType.ATTACK_DEPOSIT and action.target_id == 17 for action in ranked))

    def test_attack_10_route_is_available_once_blocking_source_is_cleared(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.sources[13].available_items = 0
        scenario.game_state.sources[13].state = SourceState.EMPTY
        scenario.game_state.deposits[10].add_items(Side.YELLOW, 1)
        planner = UtilityPlanner()

        ranked = planner.rank_actions(scenario.game_state, Side.BLUE)

        self.assertTrue(any(action.type is ActionType.ATTACK_DEPOSIT and action.target_id == 10 for action in ranked))

    def test_protected_corner_storages_are_not_attack_candidates(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.deposits[15].add_items(Side.BLUE, 1)
        scenario.game_state.deposits[25].add_items(Side.YELLOW, 1)
        planner = UtilityPlanner()

        blue_ranked = planner.rank_actions(scenario.game_state, Side.BLUE)
        yellow_ranked = planner.rank_actions(scenario.game_state, Side.YELLOW)

        self.assertFalse(any(action.type is ActionType.ATTACK_DEPOSIT and action.target_id == 25 for action in blue_ranked))
        self.assertFalse(any(action.type is ActionType.ATTACK_DEPOSIT and action.target_id == 15 for action in yellow_ranked))

    def test_protected_corner_storages_are_not_deposit_candidates_for_enemy_side(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.our_robot.load = 4
        scenario.game_state.enemy_robot.load = 4
        planner = UtilityPlanner()

        blue_ranked = planner.rank_actions(scenario.game_state, Side.BLUE)
        yellow_ranked = planner.rank_actions(scenario.game_state, Side.YELLOW)

        self.assertTrue(any(action.type is ActionType.DEPOSIT and action.target_id == 15 for action in blue_ranked))
        self.assertTrue(any(action.type is ActionType.DEPOSIT and action.target_id == 25 for action in yellow_ranked))
        self.assertFalse(any(action.type is ActionType.DEPOSIT and action.target_id == 25 for action in blue_ranked))
        self.assertFalse(any(action.type is ActionType.DEPOSIT and action.target_id == 15 for action in yellow_ranked))

    def test_center_lower_storage_deposit_is_reachable_after_yellow_thermometer(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        state = scenario.game_state
        state.enemy_robot.load = 4
        state.sources[23].available_items = 0
        state.sources[23].state = SourceState.EMPTY
        state.thermometer.mark_done_for_side(Side.YELLOW)
        planner = UtilityPlanner()

        ranked = planner.rank_actions(state, Side.YELLOW)

        self.assertTrue(any(action.type is ActionType.DEPOSIT and action.target_id == 10 for action in ranked))

    def test_deposit_max_count_for_side_rejects_enemy_protected_corner_storage(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        deposit_15 = scenario.game_state.deposits[15]
        deposit_25 = scenario.game_state.deposits[25]

        self.assertEqual(deposit_max_count_for_side(deposit_15, Side.BLUE, 4), 4)
        self.assertEqual(deposit_max_count_for_side(deposit_15, Side.YELLOW, 4), 0)
        self.assertEqual(deposit_max_count_for_side(deposit_25, Side.YELLOW, 4), 4)
        self.assertEqual(deposit_max_count_for_side(deposit_25, Side.BLUE, 4), 0)

    def test_apply_action_effects_ignores_illegal_deposit_into_enemy_protected_corner(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.our_robot.load = 4
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        action = Action(
            type=ActionType.DEPOSIT,
            target_id=25,
            label="DEPOSIT_25_X4",
            target_position=scenario.game_state.deposits[25].position,
            metadata={"deposit_count": 4},
        )

        simulator._apply_action_effects(Side.BLUE, action)

        self.assertEqual(scenario.game_state.our_robot.load, 4)
        self.assertEqual(scenario.game_state.deposits[25].blue_items, 0)
        self.assertEqual(scenario.game_state.score_blue, 0)

    def test_start_endgame_immediately_drops_load_in_home_when_robot_arrives(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        our_side = scenario.game_state.our_side
        scenario.game_state.our_robot.load = 3
        scenario.game_state.set_endgame_started(our_side, True)
        scenario.game_state.our_robot.position = scenario.game_state.endgame_config_for(our_side).final_home_point
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        action = Action(
            type=ActionType.START_ENDGAME,
            target_id=None,
            label="START_ENDGAME",
            target_position=scenario.game_state.endgame_config_for(our_side).final_home_point,
        )
        simulator._apply_action_effects(our_side, action)
        home = next(
            deposit for deposit in scenario.game_state.deposits.values()
            if deposit.kind.value == "home" and deposit.owner is our_side
        )
        self.assertEqual(scenario.game_state.our_robot.load, 0)
        self.assertEqual(home.items_for_side(our_side), 3)

    def test_start_endgame_home_leg_metadata_uses_only_final_home_waypoint(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        planner = UtilityPlanner()
        action = planner._make_endgame_action(scenario.game_state, Side.BLUE)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertGreaterEqual(len(tuple(action.metadata.get("travel_home_waypoints", ()))), 1)

    def test_mars_collision_blocks_pantry_credit(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = scenario.game_state.T_end
        scenario.game_state.our_robot.position = (0.7, -0.2)
        scenario.game_state.enemy_robot.position = (-1.12, 0.75)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        result = simulator.run()
        self.assertEqual(result.summary["our_mars_collision_count"], 1)
        self.assertEqual(result.summary["our_mars_pantry_count"], 2)
        self.assertFalse(result.summary["our_mars_all_eating"])

    def test_mars_collision_applies_negative_score_penalty(self) -> None:
        scenario_with_mars = build_scenario("baseline", seed=1)
        scenario_with_mars.game_state.t = scenario_with_mars.game_state.T_end
        scenario_with_mars.game_state.our_robot.position = (0.7, -0.2)
        scenario_with_mars.game_state.enemy_robot.position = (-1.12, 0.75)
        result_with_mars = Simulator(
            state=scenario_with_mars.game_state,
            scenario_name=scenario_with_mars.name,
            opponent_controller=build_scripted_controller(scenario_with_mars.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        ).run()

        scenario_without_mars = build_scenario("baseline", seed=1)
        scenario_without_mars.game_state.t = scenario_without_mars.game_state.T_end
        scenario_without_mars.game_state.our_robot.position = (0.7, -0.2)
        scenario_without_mars.game_state.enemy_robot.position = (-1.12, 0.75)
        scenario_without_mars.game_state.mars_by_side = {Side.BLUE: (), Side.YELLOW: ()}
        result_without_mars = Simulator(
            state=scenario_without_mars.game_state,
            scenario_name=scenario_without_mars.name,
            opponent_controller=build_scripted_controller(scenario_without_mars.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        ).run()

        self.assertEqual(result_with_mars.summary["our_score"] - result_without_mars.summary["our_score"], -40)

    def test_enemy_mars_collision_is_counted_in_enemy_summary_fields(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = scenario.game_state.T_end
        scenario.game_state.our_robot.position = (1.12, 0.75)
        scenario.game_state.enemy_robot.position = (-0.7, -0.2)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        result = simulator.run()
        self.assertGreaterEqual(result.summary["enemy_mars_collision_count"], 1)
        self.assertLess(result.summary["enemy_mars_pantry_count"], 3)

    def test_robot_circle_intersecting_15cm_mars_segment_stops_enemy_mars(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = 90.0
        mars = scenario.game_state.mars_by_side[Side.YELLOW][0]
        mars_position = mars.position_at(scenario.game_state.t)
        scenario.game_state.our_robot.position = advance_along_path(mars_position, (mars.target_position,), 0.10)
        scenario.game_state.enemy_robot.position = (1.12, 0.75)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )

        simulator._update_mars_interactions()

        stopped = simulator._stopped_mars_by_name.get(mars.name)
        self.assertIsNotNone(stopped)
        self.assertIsNotNone(stopped.blocked_since)
        self.assertEqual(simulator._mars_position(mars, scenario.game_state.t), mars_position)

    def test_robot_circle_missed_by_15cm_mars_segment_does_not_stop_enemy_mars(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = 90.0
        mars = scenario.game_state.mars_by_side[Side.YELLOW][0]
        mars_position = mars.position_at(scenario.game_state.t)
        scenario.game_state.our_robot.position = advance_along_path(mars_position, (mars.target_position,), 0.40)
        scenario.game_state.enemy_robot.position = (1.12, 0.75)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )

        simulator._update_mars_interactions()

        self.assertIsNone(simulator._stopped_mars_by_name.get(mars.name))

    def test_enemy_mars_resumes_after_robot_leaves_its_path(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = 90.0
        mars = scenario.game_state.mars_by_side[Side.YELLOW][0]
        mars_position = mars.position_at(scenario.game_state.t)
        blocking_position = advance_along_path(mars_position, (mars.target_position,), 0.10)
        scenario.game_state.our_robot.position = blocking_position
        scenario.game_state.enemy_robot.position = (1.12, 0.75)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )

        simulator._update_mars_interactions()
        self.assertEqual(simulator._mars_position(mars, 91.0), mars_position)

        scenario.game_state.t = 91.0
        scenario.game_state.our_robot.position = (0.0, 0.0)
        simulator._update_mars_interactions()

        self.assertEqual(simulator._mars_position(mars, 91.0), mars_position)
        self.assertGreater(distance(simulator._mars_position(mars, 92.0), mars_position), 1e-6)

    def test_endgame_home_leg_does_not_stop_mars_on_path(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = 90.0
        scenario.game_state.our_robot.position = (-0.7, -0.2)
        scenario.game_state.enemy_robot.position = (1.12, 0.75)
        scenario.game_state.mars_by_side = {Side.BLUE: (), Side.YELLOW: (scenario.game_state.mars_by_side[Side.YELLOW][0],)}
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        simulator._active_actions[Side.BLUE] = ActiveAction(
            action=Action(type=ActionType.START_ENDGAME, target_id=None, label="START_ENDGAME", target_position=(1.12, 0.75)),
            side=Side.BLUE,
            start_time=80.0,
            phases=[
                ActivePhase(kind="travel", duration=1.0, waypoints=((0.55, 0.25),)),
                ActivePhase(kind="service", duration=1.0, anchor=(0.55, 0.25)),
                ActivePhase(kind="travel", duration=20.0, waypoints=((1.05, 0.35), (1.12, 0.75))),
            ],
            total_duration=22.0,
        )

        simulator._update_mars_interactions()

        self.assertEqual(simulator._stopped_mars_by_name, {})

    def test_endgame_home_leg_ignores_mars_collision_penalty(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = scenario.game_state.T_end
        scenario.game_state.our_robot.position = (-0.7, -0.2)
        scenario.game_state.enemy_robot.position = (1.12, 0.75)
        scenario.game_state.mars_by_side = {Side.BLUE: (), Side.YELLOW: (scenario.game_state.mars_by_side[Side.YELLOW][0],)}
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        simulator._active_actions[Side.BLUE] = ActiveAction(
            action=Action(type=ActionType.START_ENDGAME, target_id=None, label="START_ENDGAME", target_position=(1.12, 0.75)),
            side=Side.BLUE,
            start_time=0.0,
            phases=[
                ActivePhase(kind="travel", duration=1.0, waypoints=((0.55, 0.25),)),
                ActivePhase(kind="service", duration=1.0, anchor=(0.55, 0.25)),
                ActivePhase(kind="travel", duration=200.0, waypoints=((1.05, 0.35), (1.12, 0.75))),
            ],
            total_duration=202.0,
        )

        simulator._update_mars_interactions()

        self.assertEqual(simulator.state.score_for_side(Side.BLUE), 0)
        self.assertEqual(simulator._mars_collision_pairs, set())

    def test_start_endgame_home_leg_replan_uses_remaining_waypoint_tail(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=0.5,
        )
        active = ActiveAction(
            action=Action(type=ActionType.START_ENDGAME, target_id=None, label="START_ENDGAME", target_position=(1.12, 0.75)),
            side=Side.BLUE,
            start_time=0.0,
            phases=[
                ActivePhase(kind="travel", duration=1.0, waypoints=((0.55, 0.25),)),
                ActivePhase(kind="service", duration=1.0, anchor=(0.55, 0.25)),
                ActivePhase(kind="travel", duration=10.0, waypoints=((0.6, 0.26), (0.8, 0.26), (1.12, 0.75))),
            ],
            total_duration=12.0,
        )

        remaining = simulator._remaining_travel_waypoints(active, 2, (0.97, 0.51))

        self.assertEqual(remaining, ((1.12, 0.75),))

    def test_regular_travel_replan_drops_already_passed_waypoints(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=0.5,
        )
        active = ActiveAction(
            action=Action(type=ActionType.DO_THERMOMETER, target_id=900, label="THERMOMETER", target_position=(0.63, -0.65)),
            side=Side.BLUE,
            start_time=0.0,
            phases=[
                ActivePhase(
                    kind="travel",
                    duration=10.0,
                    start_position=(0.7, -0.2),
                    waypoints=((0.68, -0.22), (0.48, -0.42), (0.28, -0.42), (0.0, -0.7)),
                ),
            ],
            total_duration=10.0,
        )

        remaining = simulator._remaining_travel_waypoints(active, 0, (0.405, -0.42))

        self.assertEqual(remaining, ((0.28, -0.42), (0.0, -0.7)))

    def test_planner_generates_play_to_end_option(self) -> None:
        planner = UtilityPlanner()
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = 85.0
        ranked = planner.rank_actions(scenario.game_state, Side.BLUE)
        self.assertTrue(any(action.label == "PLAY_TO_END" for action in ranked))

    def test_play_to_end_unlocks_regular_actions_after_deadline(self) -> None:
        planner = UtilityPlanner()
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = 85.0
        scenario.game_state.set_play_to_end_started(Side.BLUE, True)
        ranked = planner.rank_actions(scenario.game_state, Side.BLUE)
        self.assertTrue(any(action.type is ActionType.PICK for action in ranked))
        self.assertFalse(any(action.label == "PLAY_TO_END" for action in ranked))

    def test_mars_targets_are_pantry_centers_not_approach_points(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        for side, marses in scenario.game_state.mars_by_side.items():
            for mars in marses:
                deposit = scenario.game_state.deposits[mars.pantry_id]
                self.assertEqual(mars.target_position, deposit.position)

    def test_endgame_home_route_uses_only_final_home_point(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        blue_endgame = scenario.game_state.endgame_config_for(Side.BLUE)
        yellow_endgame = scenario.game_state.endgame_config_for(Side.YELLOW)

        self.assertEqual(blue_endgame.home_waypoints, (blue_endgame.final_home_point,))
        self.assertEqual(yellow_endgame.home_waypoints, (yellow_endgame.final_home_point,))

    def test_layout_99_is_loaded_as_static_obstacle(self) -> None:
        occupancy = GridOccupancyMap.from_layout(DEFAULT_LAYOUT_PATH, team_color="all")
        self.assertIn("99", occupancy.static_start_ids)
        row, col = occupancy.world_to_grid(0.0, 0.775)
        self.assertTrue(occupancy.is_blocked(row, col, use_inflated=False))

    def test_attack_10_approach_waypoint_is_free_once_blocking_source_is_cleared(self) -> None:
        occupancy = GridOccupancyMap.from_layout(DEFAULT_LAYOUT_PATH, team_color="all")
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.sources[13].available_items = 0
        scenario.game_state.sources[13].state = SourceState.EMPTY
        occupancy.sync_semantic_state(scenario.game_state.sources, scenario.game_state.deposits)
        route = build_default_semantic_map().deposits[10].attack_route_candidates(Side.BLUE)[0]
        row, col = occupancy.world_to_grid(*route.waypoints[0])
        self.assertFalse(occupancy.is_blocked(row, col, use_inflated=True))

    def test_sync_semantic_state_skips_rebuild_for_identical_snapshot(self) -> None:
        class TrackingGridOccupancyMap(GridOccupancyMap):
            rebuild_calls: int = 0

            def rebuild(self) -> None:
                type(self).rebuild_calls += 1
                super().rebuild()

        TrackingGridOccupancyMap.rebuild_calls = 0
        occupancy = TrackingGridOccupancyMap.from_layout(DEFAULT_LAYOUT_PATH, team_color="all", clear_mode=True)
        scenario = build_scenario("baseline", seed=1)
        TrackingGridOccupancyMap.rebuild_calls = 0
        occupancy.sync_semantic_state(scenario.game_state.sources, scenario.game_state.deposits)
        occupancy.sync_semantic_state(scenario.game_state.sources, scenario.game_state.deposits)
        self.assertEqual(TrackingGridOccupancyMap.rebuild_calls, 1)

    def test_dilate_mask_matches_reference_disk_expansion(self) -> None:
        raw = np.zeros((9, 11), dtype=np.uint8)
        mask = np.zeros((9, 11), dtype=bool)
        mask[1, 1] = True
        mask[4, 5] = True
        mask[7, 9] = True

        dilated = GridOccupancyMap._dilate_mask(raw, mask, radius_cells=2)

        expected = raw.copy()
        rows, cols = np.where(mask)
        offsets = [
            (dr, dc)
            for dr in range(-2, 3)
            for dc in range(-2, 3)
            if dr * dr + dc * dc <= 4
        ]
        for row, col in zip(rows.tolist(), cols.tolist()):
            for dr, dc in offsets:
                rr = row + dr
                cc = col + dc
                if 0 <= rr < expected.shape[0] and 0 <= cc < expected.shape[1]:
                    expected[rr, cc] = 100

        np.testing.assert_array_equal(dilated, expected)

    def test_plan_through_waypoints_exits_blocked_start_before_main_goal(self) -> None:
        occupancy = GridOccupancyMap.from_layout(DEFAULT_LAYOUT_PATH, team_color="all")
        planner = GridAStarPlanner(occupancy)
        start = (1.325, 0.2)
        goal = (0.9, 0.3)

        start_cell = occupancy.world_to_grid(*start)
        nearest_free = planner._find_nearest_free_cell(start_cell, use_inflated=True)

        self.assertIsNotNone(nearest_free)
        assert nearest_free is not None
        exit_point = occupancy.grid_to_world(*nearest_free)

        path = planner.plan_through_waypoints(start, (goal,))

        self.assertTrue(path.success)
        self.assertTrue(path.used_obstacle_exit)
        self.assertIn(exit_point, path.waypoints)
        self.assertLess(path.waypoints.index(exit_point), len(path.waypoints) - 1)
        self.assertEqual(path.waypoints[-1], goal)

    def test_plan_motion_uses_cache_for_identical_requests(self) -> None:
        planner = UtilityPlanner()
        scenario = build_scenario("baseline", seed=1)
        state = scenario.game_state
        route = state.sources[11].collection_routes()[0].waypoints

        plan_calls = 0
        original_plan = planner.grid_planner.plan_through_waypoints

        def tracked_plan(*args, **kwargs):
            nonlocal plan_calls
            plan_calls += 1
            return original_plan(*args, **kwargs)

        planner.grid_planner.plan_through_waypoints = tracked_plan
        planner._motion_plan_cache = {}
        try:
            first = planner._plan_motion(
                state,
                Side.BLUE,
                state.our_robot.position,
                state.our_robot.speed,
                route,
                route_name="collect_11",
            )
            calls_after_first = plan_calls
            second = planner._plan_motion(
                state,
                Side.BLUE,
                state.our_robot.position,
                state.our_robot.speed,
                route,
                route_name="collect_11",
            )
        finally:
            planner._motion_plan_cache = None

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(calls_after_first, plan_calls)

    def test_plan_motion_inserts_enemy_escape_waypoint_when_start_is_inside_enemy_zone(self) -> None:
        planner = UtilityPlanner()
        scenario = build_scenario("baseline", seed=1)
        state = scenario.game_state
        state.enemy_robot.position = (0.75, -0.46)
        start = (0.73, -0.42)
        route = ((1.2, -0.10),)

        planned = planner._plan_motion(
            state,
            Side.BLUE,
            start,
            state.our_robot.speed,
            route,
            route_name="enemy_escape_test",
        )

        self.assertIsNotNone(planned)
        assert planned is not None
        self.assertTrue(planned.motion_waypoints)
        enemy = state.enemy_robot.position
        self.assertTrue(
            any(
                ((waypoint[0] - enemy[0]) ** 2 + (waypoint[1] - enemy[1]) ** 2) ** 0.5
                >= planner.timing.route_replan_enemy_distance
                for waypoint in planned.motion_waypoints
            )
        )
        first = planned.motion_waypoints[0]
        enemy = state.enemy_robot.position
        self.assertNotEqual(first, route[0])

    def test_candidate_actions_with_enemy_blocked_semantic_waypoints_are_filtered(self) -> None:
        planner = UtilityPlanner()
        scenario = build_scenario("baseline", seed=1)
        state = scenario.game_state
        state.enemy_robot.position = (0.40, -0.56)
        invalid_pick = Action(
            type=ActionType.PICK,
            target_id=13,
            label="PICK_13",
            target_position=(0.40, -0.56),
            metadata={"semantic_waypoints": ((0.40, -0.56),)},
        )
        wait_action = Action(type=ActionType.WAIT, target_id=None, label="WAIT", target_position=None)

        filtered = planner._filter_invalid_candidate_actions(
            state,
            Side.BLUE,
            [invalid_pick, wait_action],
        )

        self.assertEqual([action.label for action in filtered], ["WAIT"])

    def test_attack_action_uses_extended_enemy_threshold_near_target(self) -> None:
        planner = UtilityPlanner()
        scenario = build_scenario("baseline", seed=1)
        state = scenario.game_state
        state.deposits[27].add_items(Side.YELLOW, 1)
        target = state.deposits[27].position
        state.enemy_robot.position = (
            target[0] + planner.timing.attack_enemy_block_radius - 0.01,
            target[1],
        )

        ranked = planner.rank_actions(state, Side.BLUE)

        self.assertFalse(any(action.label == "ATTACK_27" for action in ranked))

    def test_runtime_requests_new_action_when_enemy_blocks_nearby_path(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        planner = UtilityPlanner()
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=planner,
            dt=0.2,
        )
        action = planner._make_pick_action(
            scenario.game_state,
            Side.BLUE,
            scenario.game_state.our_robot.position,
            scenario.game_state.our_robot.speed,
            scenario.game_state.sources[11],
        )
        self.assertIsNotNone(action)
        assert action is not None
        active = simulator._activate_action(Side.BLUE, action)
        simulator._active_actions[Side.BLUE] = active
        scenario.game_state.enemy_robot.position = (0.7, 0.3)
        replanned = simulator._maybe_replan_active_travel(
            active,
            elapsed=0.0,
            current_position=scenario.game_state.our_robot.position,
        )
        self.assertTrue(replanned)
        self.assertIsNone(simulator._active_actions[Side.BLUE])
        self.assertEqual(simulator._events[-1].kind, "runtime_replan")

    def test_runtime_local_path_allows_escape_from_blocked_start(self) -> None:
        scenario = build_scenario("baseline", seed=1, our_side=Side.BLUE)
        planner = UtilityPlanner()
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=planner,
            dt=0.5,
        )
        state = scenario.game_state
        state.t = 17.0
        state.enemy_robot.position = (-0.4782689121972911, -0.10815597623237845)
        state.deposits[27].yellow_items = 4

        action = planner._make_pick_action(
            state,
            Side.YELLOW,
            state.enemy_robot.position,
            state.enemy_robot.speed,
            state.sources[23],
        )
        self.assertIsNotNone(action)
        assert action is not None
        active = simulator._activate_action(Side.YELLOW, action)
        simulator._active_actions[Side.YELLOW] = active

        replanned = simulator._maybe_replan_active_travel(
            active,
            elapsed=0.0,
            current_position=state.enemy_robot.position,
        )

        self.assertFalse(replanned)
        self.assertIs(simulator._active_actions[Side.YELLOW], active)

    def test_robot_separation_requests_runtime_replan(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        planner = UtilityPlanner()
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=planner,
            dt=0.5,
        )
        action = Action(type=ActionType.ATTACK_DEPOSIT, target_id=27, label="ATTACK_27", target_position=(0.0, 0.0))
        active = ActiveAction(
            action=action,
            side=Side.BLUE,
            start_time=0.0,
            phases=[ActivePhase(kind="travel", duration=2.0, waypoints=((0.6, 0.0),))],
            total_duration=2.0,
        )
        simulator._active_actions[Side.BLUE] = active
        simulator.state.our_robot.current_action = action.label
        simulator.state.our_robot.current_target_id = action.target_id
        simulator._maybe_replan_active_travel = lambda *args, **kwargs: False
        simulator._position_during_action = lambda *args, **kwargs: (0.6, 0.0)
        simulator._should_pause_for_robot_separation = lambda *args, **kwargs: True

        simulator._update_active_action(Side.BLUE)

        self.assertIsNone(simulator._active_actions[Side.BLUE])
        self.assertIsNone(simulator.state.our_robot.current_action)
        self.assertIsNone(simulator.state.our_robot.current_target_id)
        self.assertEqual(simulator._events[-1].kind, "runtime_replan")
        self.assertIn("robot separation blocked progress", simulator._events[-1].note)

    def test_runtime_replan_clears_endgame_started_for_cancelled_start_endgame(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        planner = UtilityPlanner()
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=planner,
            dt=0.5,
        )
        action = Action(type=ActionType.START_ENDGAME, target_id=None, label="START_ENDGAME", target_position=(1.12, 0.75))
        active = ActiveAction(
            action=action,
            side=Side.BLUE,
            start_time=0.0,
            phases=[ActivePhase(kind="travel", duration=2.0, waypoints=((0.6, 0.0),))],
            total_duration=2.0,
        )
        simulator._active_actions[Side.BLUE] = active
        simulator.state.set_endgame_started(Side.BLUE, True)
        simulator.state.our_robot.current_action = action.label
        simulator.state.our_robot.current_target_id = action.target_id

        simulator._cancel_active_action_for_runtime_replan(
            active,
            note="cancelled START_ENDGAME because local route became invalid",
        )

        self.assertFalse(simulator.state.endgame_started_for(Side.BLUE))
        self.assertIsNone(simulator._active_actions[Side.BLUE])
        self.assertIsNone(simulator.state.our_robot.current_action)
        self.assertIsNone(simulator.state.our_robot.current_target_id)
        self.assertEqual(simulator._events[-1].kind, "runtime_replan")

    def test_runtime_obstacle_overrides_follow_completed_travel_phases(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=0.2,
        )
        active = ActiveAction(
            action=Action(
                type=ActionType.ATTACK_DEPOSIT,
                target_id=10,
                label="ATTACK_10",
                target_position=(0.0, -0.9),
            ),
            side=Side.BLUE,
            start_time=0.0,
            phases=[
                ActivePhase(
                    kind="travel",
                    duration=1.0,
                    waypoints=((0.2, -0.9),),
                    clear_deposit_ids=(10,),
                ),
                ActivePhase(
                    kind="travel",
                    duration=1.0,
                    waypoints=((0.0, -0.9),),
                ),
            ],
            total_duration=2.0,
        )

        ignored_sources, ignored_deposits = simulator._runtime_obstacle_overrides(active, elapsed=0.5)
        self.assertEqual(ignored_sources, set())
        self.assertEqual(ignored_deposits, set())

        ignored_sources, ignored_deposits = simulator._runtime_obstacle_overrides(active, elapsed=1.01)
        self.assertEqual(ignored_sources, set())
        self.assertEqual(ignored_deposits, {10})

    def test_robot_separation_checks_enemy_swept_segment(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        planner = UtilityPlanner()
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=planner,
            dt=0.5,
        )
        simulator._previous_tick_state = deepcopy(simulator.state)
        simulator._previous_tick_state.our_robot.position = (0.0, 0.0)
        simulator._previous_tick_state.enemy_robot.position = (1.0, 0.0)
        simulator.state.our_robot.position = (0.0, 0.0)
        simulator.state.enemy_robot.position = (0.4, 0.0)

        active = ActiveAction(
            action=Action(type=ActionType.WAIT, target_id=None, label="MOVE_TEST", target_position=(0.6, 0.0)),
            side=Side.BLUE,
            start_time=0.0,
            phases=[],
            total_duration=0.0,
        )

        should_pause = simulator._should_pause_for_robot_separation(
            active,
            previous_position=(0.0, 0.0),
            candidate_position=(0.6, 0.0),
        )
        self.assertTrue(should_pause)

    def test_robot_separation_does_not_allow_priority_through_stationary_enemy(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        planner = UtilityPlanner()
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=planner,
            dt=0.5,
        )
        simulator._previous_tick_state = deepcopy(simulator.state)
        simulator._previous_tick_state.enemy_robot.position = (0.75, -0.46)
        simulator.state.enemy_robot.position = (0.75, -0.46)
        simulator._active_actions[Side.YELLOW] = ActiveAction(
            action=Action(type=ActionType.WAIT, target_id=None, label="WAIT", target_position=(0.75, -0.46)),
            side=Side.YELLOW,
            start_time=73.0,
            phases=[ActivePhase(kind="service", duration=1.0, anchor=(0.75, -0.46))],
            total_duration=1.0,
        )
        active = ActiveAction(
            action=Action(type=ActionType.PICK, target_id=12, label="PICK_12", target_position=(1.06, -0.60)),
            side=Side.BLUE,
            start_time=68.5,
            phases=[],
            total_duration=0.0,
        )

        should_pause = simulator._should_pause_for_robot_separation(
            active,
            previous_position=(0.6265, -0.42),
            candidate_position=(0.7287, -0.42),
        )
        self.assertTrue(should_pause)

    def test_robot_separation_allows_escape_move_out_of_stationary_deadlock(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        planner = UtilityPlanner()
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=planner,
            dt=0.5,
        )
        simulator._previous_tick_state = deepcopy(simulator.state)
        simulator._previous_tick_state.enemy_robot.position = (0.4, 0.0)
        simulator.state.enemy_robot.position = (0.4, 0.0)
        simulator._active_actions[Side.YELLOW] = ActiveAction(
            action=Action(type=ActionType.WAIT, target_id=None, label="WAIT", target_position=(0.4, 0.0)),
            side=Side.YELLOW,
            start_time=73.0,
            phases=[ActivePhase(kind="service", duration=1.0, anchor=(0.4, 0.0))],
            total_duration=1.0,
        )
        active = ActiveAction(
            action=Action(type=ActionType.PICK, target_id=12, label="PICK_12", target_position=(1.06, -0.60)),
            side=Side.BLUE,
            start_time=68.5,
            phases=[],
            total_duration=0.0,
        )

        should_pause = simulator._should_pause_for_robot_separation(
            active,
            previous_position=(0.0, 0.0),
            candidate_position=(-0.2, 0.0),
        )
        self.assertFalse(should_pause)

    def test_robot_separation_does_not_allow_priority_to_enter_current_enemy_circle(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        planner = UtilityPlanner()
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=planner,
            dt=0.5,
        )
        simulator._previous_tick_state = deepcopy(simulator.state)
        simulator._previous_tick_state.enemy_robot.position = (-0.398, -0.498)
        simulator.state.enemy_robot.position = (-0.398, -0.498)
        simulator._active_actions[Side.YELLOW] = ActiveAction(
            action=Action(type=ActionType.PICK, target_id=23, label="PICK_23", target_position=(0.0, 0.0)),
            side=Side.YELLOW,
            start_time=29.5,
            phases=[],
            total_duration=1.0,
        )
        active = ActiveAction(
            action=Action(type=ActionType.ATTACK_DEPOSIT, target_id=27, label="ATTACK_27", target_position=(0.0, 0.0)),
            side=Side.BLUE,
            start_time=29.0,
            phases=[],
            total_duration=1.0,
        )

        should_pause = simulator._should_pause_for_robot_separation(
            active,
            previous_position=(-0.282, -0.538),
            candidate_position=(-0.354, -0.466),
        )
        self.assertTrue(should_pause)

    def test_deposit_candidates_reuse_one_route_per_target(self) -> None:
        planner = UtilityPlanner()
        scenario = build_scenario("baseline", seed=1)
        state = scenario.game_state
        state.our_robot.load = 4

        deposit_route_calls = 0
        original_best_route = planner._best_route

        def tracked_best_route(*args, **kwargs):
            nonlocal deposit_route_calls
            deposit_route_calls += 1
            return original_best_route(*args, **kwargs)

        planner._best_route = tracked_best_route
        candidates = planner._generate_candidates(state, Side.BLUE)

        expected_targets = sum(
            1
            for deposit in state.friendly_deposits(Side.BLUE, include_home=True, include_neutral=True)
            if deposit_max_count(deposit, state.our_robot.load) > 0
        )
        deposit_actions = [action for action in candidates if action.type is ActionType.DEPOSIT]

        self.assertEqual(deposit_route_calls, expected_targets)
        self.assertTrue(deposit_actions)

    def test_plan_endgame_route_uses_cache_for_identical_requests(self) -> None:
        planner = UtilityPlanner()
        scenario = build_scenario("baseline", seed=1)
        state = scenario.game_state

        plan_motion_calls = 0
        original_plan_motion = planner._plan_motion

        def tracked_plan_motion(*args, **kwargs):
            nonlocal plan_motion_calls
            plan_motion_calls += 1
            return original_plan_motion(*args, **kwargs)

        planner._plan_motion = tracked_plan_motion
        planner._endgame_route_cache = {}
        try:
            first = planner._plan_endgame_route(state, Side.BLUE, state.our_robot.position, state.t)
            calls_after_first = plan_motion_calls
            second = planner._plan_endgame_route(state, Side.BLUE, state.our_robot.position, state.t + 1.0)
        finally:
            planner._endgame_route_cache = None

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(calls_after_first, plan_motion_calls)

    def test_corner_storage_deposit_route_no_longer_uses_safe_upper_detour(self) -> None:
        semantic_map = build_default_semantic_map()
        deposit_15 = semantic_map.deposits[15]
        deposit_25 = semantic_map.deposits[25]
        self.assertEqual(len(deposit_15.deposit_routes), 1)
        self.assertEqual(len(deposit_25.deposit_routes), 1)
        self.assertEqual(deposit_15.deposit_routes[0].waypoints, ((1.13, -0.20),))
        self.assertEqual(deposit_25.deposit_routes[0].waypoints, ((-1.13, -0.20),))

    def test_corner_storage_attack_routes_remain_y_axis_only(self) -> None:
        semantic_map = build_default_semantic_map()
        for deposit_id in (15, 25):
            deposit = semantic_map.deposits[deposit_id]
            for side in (Side.BLUE, Side.YELLOW):
                routes = deposit.attack_route_candidates(side)
                self.assertTrue(routes)
                self.assertTrue(all(route.axis == "y" for route in routes))

    def test_upper_storage_uses_ring_approach_candidates(self) -> None:
        semantic_map = build_default_semantic_map()
        deposit_17 = semantic_map.deposits[17]
        deposit_27 = semantic_map.deposits[27]
        self.assertEqual(deposit_17.deposit_routes, ())
        self.assertEqual(deposit_27.deposit_routes, ())
        self.assertGreater(deposit_17.approach_ring_radius, 0.0)
        self.assertAlmostEqual(deposit_17.approach_ring_radius, deposit_27.approach_ring_radius)
        routes_17 = deposit_17.deposit_route_candidates()
        routes_27 = deposit_27.deposit_route_candidates()
        self.assertEqual(len(routes_17), deposit_17.approach_ring_samples)
        self.assertEqual(len(routes_27), deposit_27.approach_ring_samples)
        self.assertTrue(
            any(math.isclose(point[0], 0.70) and math.isclose(point[1], 0.04) for point in (route.waypoints[0] for route in routes_17))
        )
        self.assertTrue(
            any(math.isclose(point[0], -0.70) and math.isclose(point[1], 0.04) for point in (route.waypoints[0] for route in routes_27))
        )

    def test_lower_storage_deposit_route_no_longer_uses_long_safe_detour(self) -> None:
        semantic_map = build_default_semantic_map()
        deposit_16 = semantic_map.deposits[16]
        deposit_26 = semantic_map.deposits[26]
        self.assertEqual(len(deposit_16.deposit_routes), 1)
        self.assertEqual(len(deposit_26.deposit_routes), 1)
        self.assertEqual(deposit_16.deposit_routes[0].waypoints, ((0.80, -0.65),))
        self.assertEqual(deposit_26.deposit_routes[0].waypoints, ((-0.80, -0.65),))

    def test_source_collect_routes_end_at_bt_working_poses_not_source_centers(self) -> None:
        semantic_map = build_default_semantic_map()
        expected_endpoints = {
            11: {(1.06, 0.20)},
            12: {(1.06, -0.60)},
            13: {(0.40, -0.56)},
            14: {(0.35, 0.06), (0.35, -0.46)},
            21: {(-1.06, 0.20)},
            22: {(-1.06, -0.60)},
            23: {(-0.40, -0.56)},
            24: {(-0.35, 0.06), (-0.35, -0.46)},
        }
        for source_id, endpoints in expected_endpoints.items():
            source = semantic_map.sources[source_id]
            route_endpoints = {route.waypoints[-1] for route in source.collect_routes}
            self.assertEqual(route_endpoints, endpoints)
            self.assertTrue(all(route.waypoints[-1] != source.position for route in source.collect_routes))
            expected_route_count = 2 if source_id in {14, 24} else 1
            self.assertEqual(len(source.collect_routes), expected_route_count)

    def test_randomized_and_stochastic_opponents_choose_legal_actions(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        planner = UtilityPlanner()
        ranked = planner.rank_actions(scenario.game_state, Side.BLUE)
        legal_labels = {action.label for action in ranked}

        stochastic = build_opponent_policy("stochastic_planner@5")
        randomized = build_opponent_policy("randomized_aggressive@6")
        uniform = build_opponent_policy("uniform_random@7")

        self.assertIn(stochastic.choose_action(scenario.game_state, planner, Side.BLUE).label, legal_labels)
        self.assertIn(randomized.choose_action(scenario.game_state, planner, Side.BLUE).label, legal_labels)
        self.assertIn(uniform.choose_action(scenario.game_state, planner, Side.BLUE).label, legal_labels)
