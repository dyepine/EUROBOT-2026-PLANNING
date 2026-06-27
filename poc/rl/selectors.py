from __future__ import annotations

from dataclasses import dataclass
import random

from poc.domain.actions import Action
from poc.domain.entities import Side
from poc.rl.policy_mapping import normalized_action_label
from poc.rl.action_space import build_rl_policy_step, resolve_policy_action
from poc.rl.encoder import flat_feature_vector
from poc.rl.model import MaskedPolicyValueNet, greedy_action, sample_action
from poc.rl.torch_compat import require_torch, torch

@dataclass(frozen=True, slots=True)
class PolicyDecisionRecord:
    policy_action: str
    action_index: int
    log_prob: float
    value: float
    entropy: float
    invalid_action: bool


class RandomMaskedPolicySelector:
    def __init__(self, rng: random.Random, name: str = "random_policy") -> None:
        self.rng = rng
        self.name = name

    def select_action(
        self,
        *,
        observation,
        ranked_actions,
    ) -> Action:
        side = observation.side
        policy_step = build_rl_policy_step(observation)
        valid_indices = [index for index, enabled in enumerate(policy_step.action_mask) if enabled]
        if not valid_indices:
            return ranked_actions[0]
        chosen_index = self.rng.choice(valid_indices)
        token = policy_step.action_space.decode(chosen_index)
        action = resolve_policy_action(ranked_actions, side, token)
        return action if action is not None else ranked_actions[0]


class TorchPolicySelector:
    def __init__(
        self,
        *,
        model: MaskedPolicyValueNet,
        device: str,
        greedy: bool,
        name: str,
    ) -> None:
        require_torch(torch)
        self.model = model
        self.device = torch.device(device)
        self.greedy = greedy
        self.name = name
        self.records: list[PolicyDecisionRecord] = []
        self.invalid_action_count = 0

    def reset(self) -> None:
        self.records.clear()
        self.invalid_action_count = 0

    def select_action(
        self,
        *,
        observation,
        ranked_actions,
    ) -> Action:
        side = observation.side
        policy_step = build_rl_policy_step(observation)
        self.model.eval()
        observation_tensor = torch.tensor(
            [flat_feature_vector(policy_step.observation)],
            dtype=torch.float32,
            device=self.device,
        )
        action_mask_tensor = torch.tensor(
            [policy_step.action_mask],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            output = self.model(observation_tensor, action_mask_tensor)
            if self.greedy:
                action_tensor, log_prob_tensor, entropy_tensor = greedy_action(output.masked_logits)
            else:
                action_tensor, log_prob_tensor, entropy_tensor = sample_action(output.masked_logits)
        action_index = int(action_tensor.item())
        token = policy_step.action_space.decode(action_index)
        action = resolve_policy_action(ranked_actions, side, token)
        invalid_action = action is None or not policy_step.action_mask[action_index]
        if invalid_action:
            self.invalid_action_count += 1
            action = ranked_actions[0]
            token = resolve_policy_action_label(action, side)
            action_index = policy_step.action_space.encode(token)
            dist = torch.distributions.Categorical(logits=output.masked_logits)
            fallback_action_tensor = torch.tensor([action_index], dtype=torch.int64, device=self.device)
            log_prob_tensor = dist.log_prob(fallback_action_tensor)
            entropy_tensor = dist.entropy()
        self.records.append(
            PolicyDecisionRecord(
                policy_action=token,
                action_index=action_index,
                log_prob=float(log_prob_tensor.item()),
                value=float(output.value.item()),
                entropy=float(entropy_tensor.item()),
                invalid_action=invalid_action,
            )
        )
        return action


def resolve_policy_action_label(action: Action, side: Side) -> str:
    from poc.rl.policy_mapping import normalized_action_label

    return normalized_action_label(action, side)
