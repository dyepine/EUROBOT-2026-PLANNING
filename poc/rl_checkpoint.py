from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is unavailable
    torch = None

from poc.rl_config import SelfPlayConfig


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for self-play PPO. Install project dependencies with torch.")


def clone_state_dict(model: object) -> dict[str, object]:
    _require_torch()
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    name: str
    update_id: int
    state_dict: dict[str, object]


class OpponentPool:
    def __init__(self, max_size: int) -> None:
        self.max_size = max(1, max_size)
        self.latest_policy: PolicySnapshot | None = None
        self._snapshots: deque[PolicySnapshot] = deque()

    def record_latest(self, model: object, update_id: int) -> PolicySnapshot:
        snapshot = PolicySnapshot(
            name=f"latest_update_{update_id}",
            update_id=update_id,
            state_dict=clone_state_dict(model),
        )
        self.latest_policy = snapshot
        return snapshot

    def add_checkpoint_snapshot(self, model: object, update_id: int) -> PolicySnapshot:
        snapshot = PolicySnapshot(
            name=f"snapshot_update_{update_id}",
            update_id=update_id,
            state_dict=clone_state_dict(model),
        )
        self._snapshots.append(snapshot)
        while len(self._snapshots) > self.max_size:
            self._snapshots.popleft()
        return snapshot

    def sample_training_snapshot(self, rng: random.Random) -> PolicySnapshot | None:
        use_latest = self.latest_policy is not None and (not self._snapshots or rng.random() < 0.30)
        if use_latest:
            return self.latest_policy
        if self._snapshots:
            return rng.choice(tuple(self._snapshots))
        return self.latest_policy

    def evaluation_snapshots(self, limit: int = 2) -> tuple[PolicySnapshot, ...]:
        if limit <= 0:
            return ()
        return tuple(list(self._snapshots)[-limit:])

    def to_state(self) -> dict[str, object]:
        return {
            "latest_policy": None if self.latest_policy is None else _snapshot_state(self.latest_policy),
            "snapshots": [_snapshot_state(snapshot) for snapshot in self._snapshots],
        }

    @classmethod
    def from_state(cls, state: dict[str, object], max_size: int) -> "OpponentPool":
        pool = cls(max_size=max_size)
        latest_state = state.get("latest_policy")
        if isinstance(latest_state, dict):
            pool.latest_policy = _snapshot_from_state(latest_state)
        for snapshot_state in state.get("snapshots", []):
            if isinstance(snapshot_state, dict):
                pool._snapshots.append(_snapshot_from_state(snapshot_state))
        while len(pool._snapshots) > pool.max_size:
            pool._snapshots.popleft()
        return pool


def save_policy_checkpoint(
    path: str | Path,
    *,
    model: object,
    config: SelfPlayConfig,
    update_id: int,
    observation_dim: int,
    action_dim: int,
    metadata: dict[str, object] | None = None,
) -> Path:
    _require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": clone_state_dict(model),
            "config": config.to_dict(),
            "update_id": update_id,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "metadata": dict(metadata or {}),
        },
        output,
    )
    return output


def save_training_state(
    path: str | Path,
    *,
    model: object,
    optimizer: object,
    config: SelfPlayConfig,
    update_id: int,
    opponent_pool: OpponentPool,
    observation_dim: int,
    action_dim: int,
    best_winrate: float,
    best_score_diff: float,
    rng: random.Random,
) -> Path:
    _require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": clone_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "update_id": update_id,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "best_winrate": best_winrate,
            "best_score_diff": best_score_diff,
            "opponent_pool": opponent_pool.to_state(),
            "python_random_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        },
        output,
    )
    return output


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, object]:
    _require_torch()
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_training_state(
    path: str | Path,
    *,
    model: object,
    optimizer: object,
    map_location: str = "cpu",
) -> dict[str, object]:
    payload = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _snapshot_state(snapshot: PolicySnapshot) -> dict[str, object]:
    return {
        "name": snapshot.name,
        "update_id": snapshot.update_id,
        "state_dict": snapshot.state_dict,
    }


def _snapshot_from_state(state: dict[str, object]) -> PolicySnapshot:
    return PolicySnapshot(
        name=str(state["name"]),
        update_id=int(state["update_id"]),
        state_dict=dict(state["state_dict"]),
    )
