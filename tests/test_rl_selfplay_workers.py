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


class RLSelfPlayWorkerTests(unittest.TestCase):
    def test_resume_helpers_preserve_saved_config_and_override_selected_fields(self) -> None:
        saved = SelfPlayConfig(
            ppo=PPOConfig(device="cuda", updates=30, dt=1.0),
            thermometer_reward_bonus=3.5,
            terminal_win_bonus=9.0,
            terminal_loss_bonus=-9.0,
            rollout_workers=3,
            eval_workers=2,
            training_scripted_fraction=0.4,
            training_scripted_opponents=("nearest_greedy", "home_safe"),
        )
        args = SimpleNamespace(device="cpu", updates=50, output_dir=None, rollout_workers=4, eval_workers=5)
        resumed = config_for_resume(args, saved)
        self.assertEqual(resumed.ppo.device, "cpu")
        self.assertEqual(resumed.ppo.updates, 50)
        self.assertEqual(resumed.ppo.dt, 1.0)
        self.assertEqual(resumed.thermometer_reward_bonus, 3.5)
        self.assertEqual(resumed.terminal_win_bonus, 9.0)
        self.assertEqual(resumed.terminal_loss_bonus, -9.0)
        self.assertEqual(resumed.rollout_workers, 4)
        self.assertEqual(resumed.eval_workers, 5)
        self.assertEqual(resumed.training_scripted_fraction, 0.4)
        self.assertEqual(resumed.training_scripted_opponents, ("nearest_greedy", "home_safe"))

    def test_resume_output_dir_defaults_to_checkpoint_parent(self) -> None:
        args = SimpleNamespace(output_dir=None)
        resolved = resolve_output_dir(args, Path("/tmp/ppo_run/training_state.pt"))
        self.assertEqual(resolved, Path("/tmp/ppo_run"))

    def test_resume_and_init_from_checkpoint_are_mutually_exclusive(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--resume",
                "runs/a/training_state.pt",
                "--init-from-checkpoint",
                "runs/b/best.pt",
            ]
        )
        with self.assertRaises(ValueError):
            resolve_start_paths(args)

    def test_eval_aggregation_includes_tail_score_metrics(self) -> None:
        summaries = [
            EvaluationSummary(
                opponent="nearest_greedy",
                kind="scripted",
                matches=4,
                winrate=0.75,
                mean_score_diff=6.0,
                median_score_diff=7.0,
                p10_score_diff=-2.0,
                min_score_diff=-5.0,
                mean_our_score=50.0,
                mean_enemy_score=44.0,
                successful_return_home_rate=0.5,
                thermometer_used_rate=0.25,
                invalid_action_count=0,
            ),
            EvaluationSummary(
                opponent="aggressive",
                kind="scripted",
                matches=2,
                winrate=0.5,
                mean_score_diff=3.0,
                median_score_diff=4.0,
                p10_score_diff=-8.0,
                min_score_diff=-12.0,
                mean_our_score=47.0,
                mean_enemy_score=44.0,
                successful_return_home_rate=0.0,
                thermometer_used_rate=0.5,
                invalid_action_count=1,
            ),
        ]
        winrate, score_diff, p10_score_diff, min_score_diff = aggregate_eval_metrics(summaries)
        self.assertAlmostEqual(winrate, (0.75 * 4 + 0.5 * 2) / 6)
        self.assertAlmostEqual(score_diff, (6.0 * 4 + 3.0 * 2) / 6)
        self.assertAlmostEqual(p10_score_diff, (-2.0 * 4 + -8.0 * 2) / 6)
        self.assertEqual(min_score_diff, -12.0)

    def test_eval_kind_classification_and_bucket_aggregation(self) -> None:
        self.assertEqual(classify_opponent_kind("nearest_greedy"), "scripted")
        self.assertEqual(classify_opponent_kind("uniform_random"), "randomized")
        self.assertEqual(classify_opponent_kind("randomized_aggressive@7"), "randomized")
        self.assertEqual(classify_opponent_kind("snapshot_update_12", default_kind="self_play"), "self_play")

        summaries = [
            EvaluationSummary(
                opponent="nearest_greedy",
                kind="scripted",
                matches=4,
                winrate=0.75,
                mean_score_diff=6.0,
                median_score_diff=7.0,
                p10_score_diff=-2.0,
                min_score_diff=-5.0,
                mean_our_score=50.0,
                mean_enemy_score=44.0,
                successful_return_home_rate=0.5,
                thermometer_used_rate=0.25,
                invalid_action_count=0,
            ),
            EvaluationSummary(
                opponent="uniform_random",
                kind="randomized",
                matches=4,
                winrate=0.25,
                mean_score_diff=-6.0,
                median_score_diff=-4.0,
                p10_score_diff=-12.0,
                min_score_diff=-18.0,
                mean_our_score=40.0,
                mean_enemy_score=46.0,
                successful_return_home_rate=0.25,
                thermometer_used_rate=0.0,
                invalid_action_count=1,
            ),
            EvaluationSummary(
                opponent="snapshot_update_8",
                kind="self_play",
                matches=2,
                winrate=0.5,
                mean_score_diff=1.0,
                median_score_diff=1.0,
                p10_score_diff=-3.0,
                min_score_diff=-7.0,
                mean_our_score=45.0,
                mean_enemy_score=44.0,
                successful_return_home_rate=0.5,
                thermometer_used_rate=0.5,
                invalid_action_count=0,
            ),
        ]
        winrate, score_diff, p10_score_diff, min_score_diff = aggregate_eval_metrics_for_kinds(
            summaries,
            {"randomized", "self_play"},
        )
        self.assertAlmostEqual(winrate, (0.25 * 4 + 0.5 * 2) / 6)
        self.assertAlmostEqual(score_diff, (-6.0 * 4 + 1.0 * 2) / 6)
        self.assertAlmostEqual(p10_score_diff, (-12.0 * 4 + -3.0 * 2) / 6)
        self.assertEqual(min_score_diff, -18.0)

    def test_scenario_speed_overrides_and_enemy_speed_jitter_are_deterministic(self) -> None:
        scenario = build_scenario(
            "baseline",
            seed=1,
            our_robot_speed=DEFAULT_ROBOT_SPEED_MPS,
            enemy_robot_speed=DEFAULT_ROBOT_SPEED_MPS * 1.2,
        )
        self.assertAlmostEqual(scenario.game_state.our_robot.speed, DEFAULT_ROBOT_SPEED_MPS)
        self.assertAlmostEqual(scenario.game_state.enemy_robot.speed, DEFAULT_ROBOT_SPEED_MPS * 1.2)

        config = SelfPlayConfig(enemy_speed_jitter_fraction=0.30)
        scale_a = enemy_speed_scale_for_match(config, 12345)
        scale_b = enemy_speed_scale_for_match(config, 12345)
        scale_c = enemy_speed_scale_for_match(config, 54321)
        self.assertAlmostEqual(scale_a, scale_b)
        self.assertGreaterEqual(scale_a, 0.70)
        self.assertLessEqual(scale_a, 1.30)
        self.assertGreaterEqual(scale_c, 0.70)
        self.assertLessEqual(scale_c, 1.30)

    def test_parallel_rollout_scheduler_accounts_for_pending_matches(self) -> None:
        estimated = _estimated_rollout_steps_per_match(
            steps=0,
            matches=0,
            target_steps=128,
            minimum_matches=4,
        )
        self.assertEqual(estimated, 32)
        self.assertFalse(
            _should_schedule_rollout_match(
                steps=34,
                matches=1,
                pending_matches=3,
                target_steps=128,
                minimum_matches=4,
            )
        )
        self.assertEqual(
            _desired_rollout_pending_matches(
                steps=34,
                matches=1,
                pending_matches=2,
                target_steps=128,
                minimum_matches=4,
                worker_count=4,
            ),
            3,
        )
        self.assertEqual(
            _desired_rollout_pending_matches(
                steps=34,
                matches=4,
                pending_matches=2,
                target_steps=128,
                minimum_matches=4,
                worker_count=4,
            ),
            4,
        )
        self.assertTrue(
            _should_schedule_rollout_match(
                steps=34,
                matches=1,
                pending_matches=2,
                target_steps=128,
                minimum_matches=4,
            )
        )
        self.assertFalse(
            _should_schedule_rollout_match(
                steps=100,
                matches=4,
                pending_matches=1,
                target_steps=128,
                minimum_matches=4,
            )
        )
        self.assertEqual(
            _desired_rollout_pending_matches(
                steps=100,
                matches=4,
                pending_matches=1,
                target_steps=128,
                minimum_matches=4,
                worker_count=4,
            ),
            0,
        )

    def test_play_match_worker_reuses_models_and_planner_within_process(self) -> None:
        config = SelfPlayConfig(ppo=PPOConfig(device="cpu"))
        opponent_spec = OpponentSpec(
            name="snapshot_a",
            kind="self_play",
            state_dict={"weights": object()},
            greedy=False,
        )
        request = _build_worker_request(
            config=config,
            learner_state_dict={"weights": object()},
            scenario_name="baseline",
            seed=1,
            update_id=1,
            episode_id=1,
            learner_greedy=False,
            opponent_spec=opponent_spec,
        )

        class FakeModel:
            def __init__(self, tag: str) -> None:
                self.tag = tag
                self.device = None

            def to(self, device):
                self.device = device
                return self

        class FakeSelector:
            def __init__(self, *, model, device, greedy, name) -> None:
                self.model = model
                self.device = device
                self.greedy = greedy
                self.name = name
                self.records: list[object] = []
                self.invalid_action_count = 0

            def reset(self) -> None:
                self.records.clear()
                self.invalid_action_count = 0

        fake_artifacts = SimpleNamespace(
            result=SimpleNamespace(decision_log=[], summary={"score_diff": 0.0}, our_side="blue"),
            learner_records=(),
            invalid_action_count=0,
        )

        with mock.patch("poc.rl.workers._WORKER_RUNTIME", None), \
             mock.patch("poc.rl.workers.torch", SimpleNamespace(
                 set_num_threads=lambda _threads: None,
                 set_num_interop_threads=lambda _threads: None,
                 manual_seed=lambda _seed: None,
                 device=lambda name: name,
             )), \
             mock.patch("poc.rl.workers.UtilityPlanner") as planner_cls, \
             mock.patch("poc.rl.workers.build_model", side_effect=[FakeModel("learner"), FakeModel("opponent")]) as build_model_mock, \
             mock.patch("poc.rl.workers.load_compatible_state_dict") as load_state_mock, \
             mock.patch("poc.rl.workers.TorchPolicySelector", side_effect=FakeSelector), \
             mock.patch("poc.rl.workers.play_match", return_value=fake_artifacts) as play_match_mock:
            first = _play_match_worker(request)
            second = _play_match_worker(request)

        self.assertEqual(build_model_mock.call_count, 2)
        self.assertEqual(planner_cls.call_count, 1)
        self.assertEqual(load_state_mock.call_count, 4)
        self.assertEqual(play_match_mock.call_count, 2)
        self.assertEqual(first.rollout_items, ())
        self.assertEqual(second.rollout_items, ())

    def test_persistent_worker_pools_reuse_executors_and_close_them(self) -> None:
        class FakeExecutor:
            def __init__(self, worker_count: int) -> None:
                self.worker_count = worker_count
                self.shutdown_calls = 0

            def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
                self.shutdown_calls += 1

        created: list[FakeExecutor] = []

        def build_fake_pool(worker_count: int) -> FakeExecutor:
            executor = FakeExecutor(worker_count)
            created.append(executor)
            return executor

        with mock.patch("poc.rl.workers._make_process_pool", side_effect=build_fake_pool):
            pools = PersistentWorkerPools(rollout_workers=2, eval_workers=3)
            rollout_a = pools.rollout_executor()
            rollout_b = pools.rollout_executor()
            eval_a = pools.eval_executor()
            eval_b = pools.eval_executor()
            pools.close()

        self.assertIs(rollout_a, rollout_b)
        self.assertIs(eval_a, eval_b)
        self.assertEqual([executor.worker_count for executor in created], [2, 3])
        self.assertTrue(all(executor.shutdown_calls == 1 for executor in created))
