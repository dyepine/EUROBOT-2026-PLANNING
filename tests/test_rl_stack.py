from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from poc.actions import Action, ActionType
from poc.entities import Side
from poc.opponent_policy import build_opponent_policy
from poc.planner import UtilityPlanner
from poc.rl_config import SelfPlayConfig
from poc.rl_infra import DEFAULT_ACTION_SPACE, resolve_policy_action
from poc.rl_ppo import PPORolloutItem, compute_gae_returns
from poc.rl_train import config_for_resume, resolve_output_dir
from poc.scenarios import build_scenario
from poc.simulator import Simulator


class RLInfraTests(unittest.TestCase):
    def test_action_space_includes_storage_6_and_expected_size(self) -> None:
        self.assertIn("DEPOSIT_OUR_STORAGE_6_X4", DEFAULT_ACTION_SPACE.tokens)
        self.assertIn("DEPOSIT_ENEMY_STORAGE_6_X4", DEFAULT_ACTION_SPACE.tokens)
        self.assertIn("ATTACK_OUR_STORAGE_6", DEFAULT_ACTION_SPACE.tokens)
        self.assertIn("ATTACK_ENEMY_STORAGE_6", DEFAULT_ACTION_SPACE.tokens)
        self.assertEqual(len(DEFAULT_ACTION_SPACE.tokens), 56)

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
            opponent_policy=scenario.opponent_policy,
            planner=UtilityPlanner(),
            dt=1.0,
        )
        result = simulator.run()
        our_transitions = [item for item in result.rl_transitions if item.side == result.our_side]
        self.assertTrue(our_transitions)
        for transition in our_transitions[:-1]:
            self.assertFalse(transition.done)
            self.assertEqual(
                transition.reward,
                transition.score_diff_after - transition.score_diff_before,
            )
        terminal = our_transitions[-1]
        expected_bonus = 20.0 if result.summary["score_diff"] > 0 else -20.0 if result.summary["score_diff"] < 0 else 0.0
        self.assertTrue(terminal.done)
        self.assertEqual(
            terminal.reward,
            terminal.score_diff_after - terminal.score_diff_before + expected_bonus,
        )

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

    def test_resume_helpers_preserve_saved_config_and_override_selected_fields(self) -> None:
        saved = SelfPlayConfig(
            device="cuda",
            updates=30,
            dt=1.0,
            training_scripted_fraction=0.4,
            training_scripted_opponents=("nearest_greedy", "home_safe"),
        )
        args = SimpleNamespace(device="cpu", updates=50, output_dir=None)
        resumed = config_for_resume(args, saved)
        self.assertEqual(resumed.device, "cpu")
        self.assertEqual(resumed.updates, 50)
        self.assertEqual(resumed.dt, 1.0)
        self.assertEqual(resumed.training_scripted_fraction, 0.4)
        self.assertEqual(resumed.training_scripted_opponents, ("nearest_greedy", "home_safe"))

    def test_resume_output_dir_defaults_to_checkpoint_parent(self) -> None:
        args = SimpleNamespace(output_dir=None)
        resolved = resolve_output_dir(args, Path("/tmp/ppo_run/training_state.pt"))
        self.assertEqual(resolved, Path("/tmp/ppo_run"))

    def test_new_scripted_opponents_and_scenarios_are_available(self) -> None:
        self.assertEqual(build_opponent_policy("storage_first").name, "storage_first")
        self.assertEqual(build_opponent_policy("home_safe").name, "home_safe")
        self.assertEqual(build_scenario("storage_first_enemy").opponent_policy.name, "storage_first")
        self.assertEqual(build_scenario("home_safe_enemy").opponent_policy.name, "home_safe")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class TorchMaskTests(unittest.TestCase):
    def test_masked_logits_block_invalid_actions(self) -> None:
        import torch

        from poc.rl_model import MaskedPolicyValueNet, greedy_action
        from poc.rl_selfplay import action_dim, observation_dim

        model = MaskedPolicyValueNet(
            observation_dim=observation_dim(),
            action_dim=action_dim(),
            hidden_sizes=SelfPlayConfig().hidden_sizes,
        )
        observation = torch.zeros((1, observation_dim()), dtype=torch.float32)
        action_mask = torch.zeros((1, action_dim()), dtype=torch.float32)
        action_mask[0, 3] = 1.0
        action_mask[0, 7] = 1.0
        output = model(observation, action_mask)
        action, _, _ = greedy_action(output.masked_logits)
        self.assertIn(int(action.item()), {3, 7})


if __name__ == "__main__":
    unittest.main()
