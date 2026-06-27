from __future__ import annotations

from dataclasses import dataclass

from poc.torch_compat import require_torch, torch

if torch is not None:
    import torch
    from torch import Tensor, nn
    from torch.distributions import Categorical
else:  # pragma: no cover - exercised only when torch is unavailable
    Tensor = object
    nn = None
    Categorical = None


def load_compatible_state_dict(model: "MaskedPolicyValueNet", state_dict: dict[str, object]) -> None:
    require_torch(torch)
    target_state = model.state_dict()
    patched_state: dict[str, object] = {}
    for key, target_value in target_state.items():
        source_value = state_dict.get(key)
        if isinstance(source_value, torch.Tensor) and tuple(source_value.shape) == tuple(target_value.shape):
            patched_state[key] = source_value
            continue
        # Backward compatibility for observation feature expansion:
        # preserve existing input weights and zero-init new feature columns.
        if (
            isinstance(source_value, torch.Tensor)
            and isinstance(target_value, torch.Tensor)
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] < target_value.shape[1]
        ):
            expanded = target_value.detach().clone()
            expanded.zero_()
            expanded[:, :source_value.shape[1]] = source_value.to(device=expanded.device, dtype=expanded.dtype)
            patched_state[key] = expanded
            continue
        # Backward compatibility for action-space expansion:
        # preserve existing output rows and zero-init newly-added actions.
        if (
            isinstance(source_value, torch.Tensor)
            and isinstance(target_value, torch.Tensor)
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[1] == target_value.shape[1]
            and source_value.shape[0] < target_value.shape[0]
        ):
            expanded = target_value.detach().clone()
            expanded.zero_()
            expanded[:source_value.shape[0], :] = source_value.to(device=expanded.device, dtype=expanded.dtype)
            patched_state[key] = expanded
            continue
        if (
            isinstance(source_value, torch.Tensor)
            and isinstance(target_value, torch.Tensor)
            and source_value.ndim == 1
            and target_value.ndim == 1
            and source_value.shape[0] < target_value.shape[0]
        ):
            expanded = target_value.detach().clone()
            expanded.zero_()
            expanded[:source_value.shape[0]] = source_value.to(device=expanded.device, dtype=expanded.dtype)
            patched_state[key] = expanded
            continue
        if source_value is not None:
            patched_state[key] = source_value
    model.load_state_dict(patched_state, strict=False)


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    masked_logits: Tensor
    value: Tensor


if nn is None:  # pragma: no cover - exercised only when torch is unavailable

    class MaskedPolicyValueNet:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            require_torch(torch)

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
    require_torch(torch)
    dist = Categorical(logits=masked_logits)
    action = dist.sample()
    return action, dist.log_prob(action), dist.entropy()


def greedy_action(masked_logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    require_torch(torch)
    dist = Categorical(logits=masked_logits)
    action = torch.argmax(masked_logits, dim=-1)
    return action, dist.log_prob(action), dist.entropy()
