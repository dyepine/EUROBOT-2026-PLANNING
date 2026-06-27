from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import random

from poc.rl.config import PPOConfig
from poc.rl.model import MaskedPolicyValueNet
from poc.rl.torch_compat import require_torch, torch

if torch is not None:
    import torch.nn.functional as F
else:  # pragma: no cover - exercised only when torch is unavailable
    F = None


@dataclass(frozen=True, slots=True)
class PPORolloutItem:
    observation: tuple[float, ...]
    next_observation: tuple[float, ...]
    action_mask: tuple[int, ...]
    next_action_mask: tuple[int, ...]
    chosen_action_index: int
    reward: float
    done: bool
    log_prob: float
    value: float
    entropy: float
    episode_id: int
    update_id: int


@dataclass(frozen=True, slots=True)
class PPORolloutBatch:
    items: tuple[PPORolloutItem, ...]

    @property
    def steps(self) -> int:
        return len(self.items)

    @property
    def episodes(self) -> int:
        return len({item.episode_id for item in self.items})


@dataclass(frozen=True, slots=True)
class PPOTrainingBatch:
    observations: object
    action_masks: object
    actions: object
    old_log_probs: object
    returns: object
    advantages: object

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    steps: int
    episodes: int
    minibatches: int
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float


def compute_gae_returns(
    items: list[PPORolloutItem],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(items)
    returns = [0.0] * len(items)
    next_advantage = 0.0
    next_value = 0.0
    next_episode_id: int | None = None
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        terminal = item.done or next_episode_id is not None and item.episode_id != next_episode_id
        bootstrap_value = 0.0 if terminal else next_value
        delta = item.reward + gamma * bootstrap_value - item.value
        next_advantage = delta if terminal else delta + gamma * gae_lambda * next_advantage
        advantages[index] = next_advantage
        returns[index] = item.value + next_advantage
        next_value = item.value
        next_episode_id = item.episode_id
    return advantages, returns


def build_training_batch(
    items: list[PPORolloutItem],
    config: PPOConfig,
) -> PPOTrainingBatch:
    require_torch(torch)
    advantages, returns = compute_gae_returns(items, config.gamma, config.gae_lambda)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    advantages_mean = advantages_tensor.mean()
    advantages_std = advantages_tensor.std(unbiased=False)
    normalized_advantages = (advantages_tensor - advantages_mean) / max(float(advantages_std), 1e-6)
    return PPOTrainingBatch(
        observations=torch.tensor([item.observation for item in items], dtype=torch.float32),
        action_masks=torch.tensor([item.action_mask for item in items], dtype=torch.float32),
        actions=torch.tensor([item.chosen_action_index for item in items], dtype=torch.int64),
        old_log_probs=torch.tensor([item.log_prob for item in items], dtype=torch.float32),
        returns=torch.tensor(returns, dtype=torch.float32),
        advantages=normalized_advantages,
    )


def ppo_update(
    model: MaskedPolicyValueNet,
    optimizer: object,
    rollout_batch: PPORolloutBatch,
    config: PPOConfig,
    *,
    rng: random.Random | None = None,
) -> PPOUpdateStats:
    require_torch(torch)
    if not rollout_batch.items:
        return PPOUpdateStats(
            steps=0,
            episodes=0,
            minibatches=0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            total_loss=0.0,
        )
    device = torch.device(config.device)
    model.train()
    batch = build_training_batch(list(rollout_batch.items), config)
    observations = batch.observations.to(device)
    action_masks = batch.action_masks.to(device)
    actions = batch.actions.to(device)
    old_log_probs = batch.old_log_probs.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)

    indices = list(range(batch.size))
    minibatches_per_epoch = max(1, ceil(batch.size / config.minibatch_size))
    rng = rng or random.Random(config.seed)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_loss = 0.0
    update_count = 0

    for _ in range(config.epochs_per_update):
        rng.shuffle(indices)
        for start in range(0, batch.size, config.minibatch_size):
            batch_indices = indices[start:start + config.minibatch_size]
            index_tensor = torch.tensor(batch_indices, dtype=torch.int64, device=device)
            output = model(observations.index_select(0, index_tensor), action_masks.index_select(0, index_tensor))
            dist = torch.distributions.Categorical(logits=output.masked_logits)
            new_log_probs = dist.log_prob(actions.index_select(0, index_tensor))
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs.index_select(0, index_tensor))
            batch_advantages = advantages.index_select(0, index_tensor)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * batch_advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(output.value, returns.index_select(0, index_tensor))
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_loss += float(loss.item())
            update_count += 1

    effective_updates = max(update_count, 1)
    return PPOUpdateStats(
        steps=rollout_batch.steps,
        episodes=rollout_batch.episodes,
        minibatches=max(update_count, minibatches_per_epoch),
        policy_loss=total_policy_loss / effective_updates,
        value_loss=total_value_loss / effective_updates,
        entropy=total_entropy / effective_updates,
        total_loss=total_loss / effective_updates,
    )
