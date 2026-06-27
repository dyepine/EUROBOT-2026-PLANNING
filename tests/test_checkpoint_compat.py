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


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class TorchMaskTests(unittest.TestCase):
    def test_rng_state_normalization_accepts_legacy_serialized_formats(self) -> None:
        import torch

        from poc.rl.checkpoint import normalize_torch_cuda_rng_state_all, normalize_torch_rng_state

        original_state = torch.get_rng_state()
        normalized_from_tensor = normalize_torch_rng_state(original_state.to(dtype=torch.int64))
        normalized_from_list = normalize_torch_rng_state(original_state.tolist())
        normalized_cuda_states = normalize_torch_cuda_rng_state_all([original_state.tolist()])

        self.assertEqual(normalized_from_tensor.dtype, torch.uint8)
        self.assertEqual(normalized_from_tensor.device.type, "cpu")
        self.assertEqual(normalized_from_tensor.tolist(), original_state.tolist())
        self.assertEqual(normalized_from_list.dtype, torch.uint8)
        self.assertEqual(normalized_from_list.tolist(), original_state.tolist())
        self.assertIsInstance(normalized_cuda_states, list)
        self.assertEqual(len(normalized_cuda_states), 1)
        self.assertEqual(normalized_cuda_states[0].dtype, torch.uint8)
        self.assertEqual(normalized_cuda_states[0].tolist(), original_state.tolist())

    def test_masked_logits_block_invalid_actions(self) -> None:
        import torch

        from poc.rl.checkpoint import load_checkpoint, save_policy_checkpoint
        from poc.rl.model import MaskedPolicyValueNet, greedy_action, load_compatible_state_dict
        from poc.rl.match import action_dim, build_model, observation_dim

        model = MaskedPolicyValueNet(
            observation_dim=observation_dim(),
            action_dim=action_dim(),
            hidden_sizes=SelfPlayConfig().ppo.hidden_sizes,
        )
        observation = torch.zeros((1, observation_dim()), dtype=torch.float32)
        action_mask = torch.zeros((1, action_dim()), dtype=torch.float32)
        action_mask[0, 3] = 1.0
        action_mask[0, 7] = 1.0
        output = model(observation, action_mask)
        action, _, _ = greedy_action(output.masked_logits)
        self.assertIn(int(action.item()), {3, 7})

        config = SelfPlayConfig()
        source_model = build_model(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "init.pt"
            save_policy_checkpoint(
                checkpoint_path,
                model=source_model,
                config=config,
                update_id=7,
                observation_dim=observation_dim(),
                action_dim=action_dim(),
            )
            payload = load_checkpoint(checkpoint_path)
            target_model = build_model(config)
            target_model.load_state_dict(payload["model_state"])
            for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
                self.assertTrue(torch.equal(source_param, target_param))

        legacy_observation_dim = len(DEFAULT_FLAT_FEATURE_KEYS) - 242
        legacy_action_dim = action_dim() - 1
        legacy_model = MaskedPolicyValueNet(
            observation_dim=legacy_observation_dim,
            action_dim=legacy_action_dim,
            hidden_sizes=SelfPlayConfig().ppo.hidden_sizes,
        )
        upgraded_model = build_model(SelfPlayConfig())
        load_compatible_state_dict(upgraded_model, legacy_model.state_dict())
        self.assertTrue(
            torch.equal(
                upgraded_model.backbone[0].weight[:, :legacy_observation_dim],
                legacy_model.backbone[0].weight,
            )
        )
        self.assertTrue(
            torch.equal(
                upgraded_model.backbone[0].weight[:, legacy_observation_dim:],
                torch.zeros_like(upgraded_model.backbone[0].weight[:, legacy_observation_dim:]),
            )
        )
        self.assertTrue(
            torch.equal(
                upgraded_model.policy_head.weight[:legacy_action_dim, :],
                legacy_model.policy_head.weight,
            )
        )
        self.assertTrue(
            torch.equal(
                upgraded_model.policy_head.weight[legacy_action_dim:, :],
                torch.zeros_like(upgraded_model.policy_head.weight[legacy_action_dim:, :]),
            )
        )
        self.assertTrue(
            torch.equal(
                upgraded_model.policy_head.bias[:legacy_action_dim],
                legacy_model.policy_head.bias,
            )
        )
        self.assertTrue(
            torch.equal(
                upgraded_model.policy_head.bias[legacy_action_dim:],
                torch.zeros_like(upgraded_model.policy_head.bias[legacy_action_dim:]),
            )
        )
