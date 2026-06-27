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


class RLObservationAndTransitionTests(unittest.TestCase):
    def test_action_space_includes_storage_6_and_expected_size(self) -> None:
        self.assertIn("DEPOSIT_OUR_STORAGE_6_X4", DEFAULT_ACTION_SPACE.tokens)
        self.assertIn("DEPOSIT_ENEMY_STORAGE_6_X4", DEFAULT_ACTION_SPACE.tokens)
        self.assertIn("ATTACK_OUR_STORAGE_6", DEFAULT_ACTION_SPACE.tokens)
        self.assertIn("ATTACK_ENEMY_STORAGE_6", DEFAULT_ACTION_SPACE.tokens)
        self.assertIn("PLAY_TO_END", DEFAULT_ACTION_SPACE.tokens)
        self.assertEqual(len(DEFAULT_ACTION_SPACE.tokens), 57)

    def test_policy_action_resolution_covers_main_action_types(self) -> None:
        ranked_actions = [
            Action(type=ActionType.PICK, target_id=11, label="PICK_11", target_position=(0.0, 0.0)),
            Action(
                type=ActionType.DEPOSIT,
                target_id=16,
                label="DEPOSIT_16_X2",
                target_position=(0.0, 0.0),
                metadata={"deposit_count": 2},
            ),
            Action(type=ActionType.ATTACK_DEPOSIT, target_id=26, label="ATTACK_26", target_position=(0.0, 0.0)),
            Action(type=ActionType.DO_THERMOMETER, target_id=None, label="THERMOMETER", target_position=(0.0, 0.0)),
            Action(type=ActionType.WAIT, target_id=None, label="WAIT", target_position=None),
        ]
        expected = {
            "PICK_OUR_SOURCE_1": ActionType.PICK,
            "DEPOSIT_OUR_STORAGE_6_X2": ActionType.DEPOSIT,
            "ATTACK_ENEMY_STORAGE_6": ActionType.ATTACK_DEPOSIT,
            "THERMOMETER": ActionType.DO_THERMOMETER,
            "WAIT": ActionType.WAIT,
        }
        for policy_action, action_type in expected.items():
            resolved = resolve_policy_action(ranked_actions, Side.BLUE, policy_action)
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.type, action_type)

    def test_terminal_bonus_applies_only_to_terminal_transition(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        result = simulator.run()
        our_transitions = build_rl_transitions_from_match_result(result, side=Side(result.our_side))
        self.assertTrue(our_transitions)
        for transition in our_transitions[:-1]:
            self.assertFalse(transition.done)
            self.assertEqual(
                transition.reward,
                transition.score_diff_after - transition.score_diff_before,
            )
        terminal = our_transitions[-1]
        expected_bonus = 2.0 if result.summary["score_diff"] > 0 else -2.0 if result.summary["score_diff"] < 0 else 0.0
        self.assertTrue(terminal.done)
        self.assertEqual(
            terminal.reward,
            terminal.score_diff_after - terminal.score_diff_before + expected_bonus,
        )

    def test_successful_thermometer_receives_shaped_reward_bonus(self) -> None:
        class ThermometerSelector:
            name = "thermo_test"

            def select_action(self, *, observation, ranked_actions):
                del observation
                for action in ranked_actions:
                    if action.type is ActionType.DO_THERMOMETER:
                        return action
                return ranked_actions[0]

        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.our_robot.position = (0.0, -0.7)
        scenario.game_state.sources[13].available_items = 0
        scenario.game_state.deposits[10].clear()
        scenario.game_state.deposits[16].clear()
        bonus = 7.0
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
            controllers={Side.BLUE: ThermometerSelector()},
        )
        result = simulator.run()
        thermo_transitions = [
            item
            for item in build_rl_transitions_from_match_result(
                result,
                side=Side.BLUE,
                thermometer_reward_bonus=bonus,
                terminal_win_bonus=0.0,
                terminal_draw_bonus=0.0,
                terminal_loss_bonus=0.0,
            )
            if item.chosen_action == "THERMOMETER"
        ]
        self.assertTrue(thermo_transitions)
        shaped = thermo_transitions[0]
        self.assertEqual(
            shaped.reward,
            shaped.score_diff_after - shaped.score_diff_before + bonus,
        )

    def test_thermometer_doing_flags_clear_on_done(self) -> None:
        thermometer = Thermometer(semantic_id=1, position=(0.0, 0.0))
        self.assertFalse(thermometer.is_doing_for_side(Side.BLUE))
        self.assertFalse(thermometer.is_doing_for_side(Side.YELLOW))
        thermometer.mark_doing_for_side(Side.BLUE)
        self.assertTrue(thermometer.is_doing_for_side(Side.BLUE))
        self.assertFalse(thermometer.is_done_for_side(Side.BLUE))
        thermometer.mark_done_for_side(Side.BLUE)
        self.assertFalse(thermometer.is_doing_for_side(Side.BLUE))
        self.assertTrue(thermometer.is_done_for_side(Side.BLUE))

    def test_gae_returns_shape_and_finiteness(self) -> None:
        items = [
            PPORolloutItem(
                observation=(0.0,),
                next_observation=(0.1,),
                action_mask=(1,),
                next_action_mask=(1,),
                chosen_action_index=0,
                reward=1.0,
                done=False,
                log_prob=0.0,
                value=0.5,
                entropy=0.1,
                episode_id=0,
                update_id=0,
            ),
            PPORolloutItem(
                observation=(0.1,),
                next_observation=(0.2,),
                action_mask=(1,),
                next_action_mask=(0,),
                chosen_action_index=0,
                reward=2.0,
                done=True,
                log_prob=0.0,
                value=0.25,
                entropy=0.1,
                episode_id=0,
                update_id=0,
            ),
        ]
        advantages, returns = compute_gae_returns(items, gamma=0.99, gae_lambda=0.95)
        self.assertEqual(len(advantages), len(items))
        self.assertEqual(len(returns), len(items))
        self.assertTrue(all(math.isfinite(value) for value in advantages))
        self.assertTrue(all(math.isfinite(value) for value in returns))

    def test_enemy_max_speed_seen_feature_uses_running_observed_max(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.max_observed_speed_by_side[Side.YELLOW] = 0.09
        observation = build_rl_observation(_decision_observation(scenario.game_state, Side.BLUE, dt=1.0))
        self.assertAlmostEqual(observation.global_features["enemy_max_speed_seen_norm"], 0.5)

    def test_enemy_velocity_measurement_noise_is_deterministic_and_can_leak_self_motion(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        previous_state = deepcopy(scenario.game_state)
        previous_state.robot_for_side(Side.BLUE).position = (0.0, 0.0)
        scenario.game_state.robot_for_side(Side.BLUE).position = (0.04, 0.0)
        previous_state.robot_for_side(Side.YELLOW).position = (0.0, 0.0)
        scenario.game_state.robot_for_side(Side.YELLOW).position = (0.09, 0.0)
        scenario.game_state.t = 1.0
        scenario.game_state.last_motion_start_time_by_side[Side.BLUE] = 0.95

        clean = build_rl_observation(
            _decision_observation(
                scenario.game_state,
                Side.BLUE,
                previous_state=previous_state,
                dt=1.0,
            ),
            config=RLObservationConfig(),
        )
        noisy_cfg = RLObservationConfig(
            observation_noise_seed=123,
            enemy_velocity_noise_std_mps=0.01,
            enemy_velocity_self_motion_leak_fraction=1.0,
            enemy_velocity_self_motion_leak_duration_s=0.2,
        )
        noisy_a = build_rl_observation(
            _decision_observation(
                scenario.game_state,
                Side.BLUE,
                previous_state=previous_state,
                dt=1.0,
            ),
            config=noisy_cfg,
        )
        noisy_b = build_rl_observation(
            _decision_observation(
                scenario.game_state,
                Side.BLUE,
                previous_state=previous_state,
                dt=1.0,
            ),
            config=noisy_cfg,
        )
        self.assertEqual(noisy_a.global_features["enemy_vel_x_norm"], noisy_b.global_features["enemy_vel_x_norm"])
        self.assertEqual(noisy_a.global_features["enemy_vel_y_norm"], noisy_b.global_features["enemy_vel_y_norm"])
        self.assertEqual(noisy_a.global_features["enemy_speed_norm"], noisy_b.global_features["enemy_speed_norm"])
        self.assertNotEqual(clean.global_features["enemy_speed_norm"], noisy_a.global_features["enemy_speed_norm"])

    def test_source_temporal_features_track_last_change(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.our_robot.load = 2
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        action = Action(
            type=ActionType.PICK,
            target_id=11,
            label="PICK_11",
            target_position=scenario.game_state.sources[11].position,
        )
        simulator._apply_action_effects(Side.BLUE, action)
        observation = build_rl_observation(_decision_observation(scenario.game_state, Side.BLUE, dt=1.0))
        features = observation.source_features["OUR_SOURCE_1"]
        self.assertAlmostEqual(features["last_items_delta_norm"], -0.5)
        self.assertEqual(features["time_since_last_change_norm"], 0.0)
        self.assertEqual(features["last_change_was_disturb_like"], 1.0)

    def test_deposit_temporal_features_track_last_score_change_and_actor(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.our_robot.load = 2
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        action = Action(
            type=ActionType.DEPOSIT,
            target_id=16,
            label="DEPOSIT_16_X2",
            target_position=scenario.game_state.deposits[16].position,
            metadata={"deposit_count": 2},
        )
        simulator._apply_action_effects(Side.BLUE, action)
        observation = build_rl_observation(_decision_observation(scenario.game_state, Side.BLUE, dt=1.0))
        features = observation.deposit_features["OUR_STORAGE_6"]
        self.assertAlmostEqual(features["last_score_diff_delta_norm"], 11.0 / 17.0)
        self.assertEqual(features["time_since_last_score_change_norm"], 0.0)
        self.assertEqual(features["last_change_by_our"], 1.0)
        self.assertEqual(features["last_change_by_enemy"], 0.0)
        self.assertAlmostEqual(features["our_points_norm"], 11.0 / 17.0)
        self.assertEqual(features["enemy_points_norm"], 0.0)
        self.assertAlmostEqual(features["score_diff_norm"], 11.0 / 17.0)
        self.assertEqual(features["occupied_by_our"], 1.0)
        self.assertEqual(features["occupied_by_enemy"], 0.0)
        self.assertEqual(features["occupied_by_none"], 0.0)
        self.assertEqual(features["kind_storage"], 1.0)
        self.assertEqual(features["kind_home"], 0.0)

    def test_thermometer_temporal_features_track_state_and_lane_changes(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = 5.0
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
            planner=UtilityPlanner(),
            dt=1.0,
        )
        scenario.game_state.t = 10.0
        action = Action(
            type=ActionType.DO_THERMOMETER,
            target_id=None,
            label="THERMOMETER",
            target_position=scenario.game_state.thermometer.position,
        )
        simulator._apply_action_effects(Side.BLUE, action)
        observation = build_rl_observation(_decision_observation(scenario.game_state, Side.BLUE, dt=1.0))
        self.assertEqual(observation.global_features["time_since_thermometer_state_change_norm"], 0.0)
        self.assertEqual(observation.global_features["time_since_our_lane_clear_change_norm"], 0.0)
        self.assertGreater(observation.global_features["time_since_enemy_lane_clear_change_norm"], 0.0)

    def test_time_bucket_and_deadline_features_track_match_phase(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.t = 85.0
        observation = build_rl_observation(_decision_observation(scenario.game_state, Side.BLUE, dt=1.0))
        self.assertEqual(observation.global_features["time_bin_8"], 1.0)
        self.assertEqual(sum(observation.global_features[f"time_bin_{index}"] for index in range(10)), 1.0)
        self.assertEqual(observation.global_features["after_main_pipeline_deadline"], 1.0)
        self.assertEqual(observation.global_features["after_chill_end"], 0.0)
        self.assertEqual(observation.global_features["in_last_30s"], 1.0)
        self.assertEqual(observation.global_features["in_last_20s"], 1.0)
        self.assertEqual(observation.global_features["in_last_10s"], 0.0)
        self.assertLess(observation.global_features["time_to_main_pipeline_deadline_norm"], 0.0)
        self.assertGreater(observation.global_features["time_to_chill_end_norm"], 0.0)

    def test_relative_geometric_features_track_enemy_and_endgame_targets(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.our_robot.position = (0.2, -0.1)
        scenario.game_state.enemy_robot.position = (-0.4, 0.3)
        observation = build_rl_observation(_decision_observation(scenario.game_state, Side.BLUE, dt=1.0))

        self.assertAlmostEqual(observation.global_features["enemy_rel_x_norm"], -0.4)
        self.assertAlmostEqual(observation.global_features["enemy_rel_y_norm"], 0.4)
        self.assertAlmostEqual(observation.global_features["enemy_rel_x_abs_norm"], 0.4)
        self.assertAlmostEqual(observation.global_features["enemy_rel_y_abs_norm"], 0.4)
        self.assertAlmostEqual(observation.global_features["enemy_rel_dist_norm"], 0.2)
        self.assertAlmostEqual(observation.global_features["our_home_rel_x_norm"], 0.6133333333333333)
        self.assertAlmostEqual(observation.global_features["our_home_rel_y_norm"], 0.85)
        self.assertAlmostEqual(observation.global_features["our_home_rel_x_abs_norm"], 0.6133333333333333)
        self.assertAlmostEqual(observation.global_features["our_home_rel_y_abs_norm"], 0.85)
        self.assertAlmostEqual(observation.global_features["our_home_rel_dist_norm"], 0.3473969133205064)
        self.assertAlmostEqual(observation.global_features["enemy_home_rel_x_norm"], -0.88)
        self.assertAlmostEqual(observation.global_features["enemy_home_rel_y_norm"], 0.85)
        self.assertAlmostEqual(observation.global_features["enemy_home_rel_x_abs_norm"], 0.88)
        self.assertAlmostEqual(observation.global_features["enemy_home_rel_y_abs_norm"], 0.85)
        self.assertAlmostEqual(observation.global_features["enemy_home_rel_dist_norm"], 0.4354396540368049)
        self.assertAlmostEqual(observation.global_features["our_chill_rel_x_norm"], 0.23333333333333336)
        self.assertAlmostEqual(observation.global_features["our_chill_rel_y_norm"], 0.35)
        self.assertAlmostEqual(observation.global_features["our_chill_rel_x_abs_norm"], 0.23333333333333336)
        self.assertAlmostEqual(observation.global_features["our_chill_rel_y_abs_norm"], 0.35)
        self.assertAlmostEqual(observation.global_features["our_chill_rel_dist_norm"], 0.13728129459672883)
        self.assertAlmostEqual(observation.global_features["thermometer_rel_x_norm"], -0.13333333333333333)
        self.assertAlmostEqual(observation.global_features["thermometer_rel_y_norm"], -0.9)
        self.assertAlmostEqual(observation.global_features["thermometer_rel_x_abs_norm"], 0.13333333333333333)
        self.assertAlmostEqual(observation.global_features["thermometer_rel_y_abs_norm"], 0.9)
        self.assertAlmostEqual(observation.global_features["thermometer_rel_dist_norm"], 0.2557041559783794)
        self.assertAlmostEqual(observation.global_features["our_home_enemy_rel_x_norm"], 1.0)
        self.assertAlmostEqual(observation.global_features["our_home_enemy_rel_y_norm"], 0.45)
        self.assertAlmostEqual(observation.global_features["our_home_enemy_rel_x_abs_norm"], 1.0)
        self.assertAlmostEqual(observation.global_features["our_home_enemy_rel_y_abs_norm"], 0.45)
        self.assertAlmostEqual(observation.global_features["our_home_enemy_rel_dist_norm"], 0.4396589587396122)
        self.assertAlmostEqual(observation.global_features["enemy_home_enemy_rel_x_norm"], -0.48)
        self.assertAlmostEqual(observation.global_features["enemy_home_enemy_rel_y_norm"], 0.45)
        self.assertAlmostEqual(observation.global_features["enemy_home_enemy_rel_x_abs_norm"], 0.48)
        self.assertAlmostEqual(observation.global_features["enemy_home_enemy_rel_y_abs_norm"], 0.45)
        self.assertAlmostEqual(observation.global_features["enemy_home_enemy_rel_dist_norm"], 0.23548640333116086)
        self.assertAlmostEqual(observation.global_features["our_chill_enemy_rel_x_norm"], 0.6333333333333334)
        self.assertAlmostEqual(observation.global_features["our_chill_enemy_rel_y_norm"], -0.05)
        self.assertAlmostEqual(observation.global_features["our_chill_enemy_rel_x_abs_norm"], 0.6333333333333334)
        self.assertAlmostEqual(observation.global_features["our_chill_enemy_rel_y_abs_norm"], 0.05)
        self.assertAlmostEqual(observation.global_features["our_chill_enemy_rel_dist_norm"], 0.26384727517142303)
        self.assertAlmostEqual(observation.global_features["thermometer_enemy_rel_x_norm"], 0.26666666666666666)
        self.assertAlmostEqual(observation.global_features["thermometer_enemy_rel_y_norm"], -1.0)
        self.assertAlmostEqual(observation.global_features["thermometer_enemy_rel_x_abs_norm"], 0.26666666666666666)
        self.assertAlmostEqual(observation.global_features["thermometer_enemy_rel_y_abs_norm"], 1.0)
        self.assertAlmostEqual(observation.global_features["thermometer_enemy_rel_dist_norm"], 0.3772369180073609)
        self.assertAlmostEqual(observation.source_features["OUR_SOURCE_1"]["rel_x_norm"], 0.75)
        self.assertAlmostEqual(observation.source_features["OUR_SOURCE_1"]["rel_y_norm"], 0.3)
        self.assertAlmostEqual(observation.source_features["OUR_SOURCE_1"]["rel_x_abs_norm"], 0.75)
        self.assertAlmostEqual(observation.source_features["OUR_SOURCE_1"]["rel_y_abs_norm"], 0.3)
        self.assertAlmostEqual(observation.source_features["OUR_SOURCE_1"]["rel_dist_norm"], 0.32292235313438145)
        self.assertAlmostEqual(observation.source_features["OUR_SOURCE_1"]["enemy_rel_x_norm"], 1.0)
        self.assertAlmostEqual(observation.source_features["OUR_SOURCE_1"]["enemy_rel_y_norm"], -0.1)
        self.assertAlmostEqual(observation.source_features["OUR_SOURCE_1"]["enemy_rel_x_abs_norm"], 1.0)
        self.assertAlmostEqual(observation.source_features["OUR_SOURCE_1"]["enemy_rel_y_abs_norm"], 0.1)
        self.assertAlmostEqual(observation.source_features["OUR_SOURCE_1"]["enemy_rel_dist_norm"], 0.47923215828913396)
        self.assertAlmostEqual(observation.deposit_features["OUR_STORAGE_7"]["rel_x_norm"], 0.3333333333333333)
        self.assertAlmostEqual(observation.deposit_features["OUR_STORAGE_7"]["rel_y_norm"], -0.1)
        self.assertAlmostEqual(observation.deposit_features["OUR_STORAGE_7"]["rel_x_abs_norm"], 0.3333333333333333)
        self.assertAlmostEqual(observation.deposit_features["OUR_STORAGE_7"]["rel_y_abs_norm"], 0.1)
        self.assertAlmostEqual(observation.deposit_features["OUR_STORAGE_7"]["rel_dist_norm"], 0.1414213562373095)
        self.assertAlmostEqual(observation.deposit_features["OUR_STORAGE_7"]["enemy_rel_x_norm"], 0.7333333333333334)
        self.assertAlmostEqual(observation.deposit_features["OUR_STORAGE_7"]["enemy_rel_y_norm"], -0.5)
        self.assertAlmostEqual(observation.deposit_features["OUR_STORAGE_7"]["enemy_rel_x_abs_norm"], 0.7333333333333334)
        self.assertAlmostEqual(observation.deposit_features["OUR_STORAGE_7"]["enemy_rel_y_abs_norm"], 0.5)
        self.assertAlmostEqual(observation.deposit_features["OUR_STORAGE_7"]["enemy_rel_dist_norm"], 0.33512339862756874)

    def test_source_empty_flag_is_exposed_in_observation(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        scenario.game_state.sources[11].state = SourceState.EMPTY
        scenario.game_state.sources[11].available_items = 0
        observation = build_rl_observation(_decision_observation(scenario.game_state, Side.BLUE, dt=1.0))
        self.assertEqual(observation.source_features["OUR_SOURCE_1"]["state_empty"], 1.0)

    def test_deposit_normalized_point_and_occupancy_features_are_exposed(self) -> None:
        scenario = build_scenario("baseline", seed=1)
        deposit = scenario.game_state.deposits[101]
        deposit.add_items(Side.BLUE, 2)
        observation = build_rl_observation(_decision_observation(scenario.game_state, Side.BLUE, dt=1.0))
        features = observation.deposit_features["OUR_HOME"]
        self.assertAlmostEqual(features["total_items_norm"], 0.5)
        self.assertAlmostEqual(features["our_points_norm"], 4.0 / 17.0)
        self.assertEqual(features["enemy_points_norm"], 0.0)
        self.assertAlmostEqual(features["score_diff_norm"], 4.0 / 17.0)
        self.assertEqual(features["occupied_by_our"], 1.0)
        self.assertEqual(features["occupied_by_enemy"], 0.0)
        self.assertEqual(features["occupied_by_none"], 0.0)
        self.assertEqual(features["kind_home"], 1.0)
        self.assertEqual(features["kind_storage"], 0.0)

    def test_attack_action_blocked_by_enemy_on_target_is_removed_from_action_mask(self) -> None:
        planner = UtilityPlanner()
        scenario = build_scenario("baseline", seed=1)
        state = scenario.game_state
        state.enemy_robot.position = state.deposits[27].position

        ranked = planner.rank_actions(state, Side.BLUE)

        self.assertFalse(any(action.label == "ATTACK_27" for action in ranked))
        policy_step = build_rl_policy_step(_decision_observation(state, Side.BLUE, ranked_actions=ranked))
        attack_index = DEFAULT_ACTION_SPACE.index_by_token["ATTACK_ENEMY_STORAGE_7"]
        self.assertEqual(policy_step.action_mask[attack_index], 0)
