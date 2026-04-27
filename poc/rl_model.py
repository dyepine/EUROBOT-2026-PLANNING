from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    from torch.distributions import Categorical
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is unavailable
    torch = None
    Tensor = object
    nn = None
    Categorical = None


def _require_torch() -> None:
    if torch is None or nn is None or Categorical is None:
        raise ModuleNotFoundError("PyTorch is required for self-play PPO. Install project dependencies with torch.")


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    masked_logits: Tensor
    value: Tensor


if nn is None:  # pragma: no cover - exercised only when torch is unavailable

    class MaskedPolicyValueNet:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            _require_torch()

else:

    class MaskedPolicyValueNet(nn.Module):
        def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            hidden_sizes: tuple[int, ...] = (256, 256),
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = observation_dim
            for hidden_dim in hidden_sizes:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.Tanh())
                in_dim = hidden_dim
            self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
            self.policy_head = nn.Linear(in_dim, action_dim)
            self.value_head = nn.Linear(in_dim, 1)

        def forward(self, observation: Tensor, action_mask: Tensor) -> PolicyOutput:
            hidden = self.backbone(observation)
            logits = self.policy_head(hidden)
            masked_logits = self.apply_action_mask(logits, action_mask)
            value = self.value_head(hidden).squeeze(-1)
            return PolicyOutput(masked_logits=masked_logits, value=value)

        @staticmethod
        def apply_action_mask(logits: Tensor, action_mask: Tensor) -> Tensor:
            invalid = action_mask <= 0
            return logits.masked_fill(invalid, -1e9)

        def distribution(self, observation: Tensor, action_mask: Tensor) -> tuple[Categorical, Tensor]:
            output = self.forward(observation, action_mask)
            return Categorical(logits=output.masked_logits), output.value


def sample_action(masked_logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    _require_torch()
    dist = Categorical(logits=masked_logits)
    action = dist.sample()
    return action, dist.log_prob(action), dist.entropy()


def greedy_action(masked_logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    _require_torch()
    dist = Categorical(logits=masked_logits)
    action = torch.argmax(masked_logits, dim=-1)
    return action, dist.log_prob(action), dist.entropy()
