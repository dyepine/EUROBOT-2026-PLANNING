from __future__ import annotations

from dataclasses import asdict, dataclass, field

from poc.entities import Side

DEFAULT_TRAINING_SCENARIOS = ("baseline", "delayed_sources")
DEFAULT_EVAL_SCENARIOS = ("baseline", "delayed_sources")
DEFAULT_EVAL_OPPONENTS = ("nearest_greedy", "aggressive", "thermo_first")


@dataclass(frozen=True, slots=True)
class PPOConfig:
    seed: int = 1
    device: str = "cpu"
    steps_per_update: int = 1024
    matches_per_update: int = 16
    epochs_per_update: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    side: Side = Side.BLUE
    dt: float = 0.5
    opponent_pool_size: int = 8
    checkpoint_every_updates: int = 1
    eval_every_updates: int = 5
    updates: int = 100
    hidden_sizes: tuple[int, ...] = (256, 256)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["side"] = self.side.value
        return payload


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    seed: int = 1
    device: str = "cpu"
    steps_per_update: int = 1024
    matches_per_update: int = 16
    epochs_per_update: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    side: Side = Side.BLUE
    dt: float = 0.5
    opponent_pool_size: int = 8
    checkpoint_every_updates: int = 1
    eval_every_updates: int = 5
    updates: int = 100
    hidden_sizes: tuple[int, ...] = (256, 256)
    training_scenarios: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TRAINING_SCENARIOS)
    eval_scenarios: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EVAL_SCENARIOS)
    eval_opponents: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EVAL_OPPONENTS)
    eval_matches_per_opponent: int = 4

    def ppo_config(self) -> PPOConfig:
        return PPOConfig(
            seed=self.seed,
            device=self.device,
            steps_per_update=self.steps_per_update,
            matches_per_update=self.matches_per_update,
            epochs_per_update=self.epochs_per_update,
            minibatch_size=self.minibatch_size,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
            learning_rate=self.learning_rate,
            side=self.side,
            dt=self.dt,
            opponent_pool_size=self.opponent_pool_size,
            checkpoint_every_updates=self.checkpoint_every_updates,
            eval_every_updates=self.eval_every_updates,
            updates=self.updates,
            hidden_sizes=self.hidden_sizes,
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["side"] = self.side.value
        payload["ppo"] = self.ppo_config().to_dict()
        return payload


def selfplay_config_from_dict(payload: dict[str, object]) -> SelfPlayConfig:
    return SelfPlayConfig(
        seed=int(payload.get("seed", 1)),
        device=str(payload.get("device", "cpu")),
        steps_per_update=int(payload.get("steps_per_update", 1024)),
        matches_per_update=int(payload.get("matches_per_update", 16)),
        epochs_per_update=int(payload.get("epochs_per_update", 4)),
        minibatch_size=int(payload.get("minibatch_size", 256)),
        gamma=float(payload.get("gamma", 0.99)),
        gae_lambda=float(payload.get("gae_lambda", 0.95)),
        clip_epsilon=float(payload.get("clip_epsilon", 0.2)),
        entropy_coef=float(payload.get("entropy_coef", 0.01)),
        value_coef=float(payload.get("value_coef", 0.5)),
        max_grad_norm=float(payload.get("max_grad_norm", 0.5)),
        learning_rate=float(payload.get("learning_rate", 3e-4)),
        side=Side(str(payload.get("side", Side.BLUE.value))),
        dt=float(payload.get("dt", 0.5)),
        opponent_pool_size=int(payload.get("opponent_pool_size", 8)),
        checkpoint_every_updates=int(payload.get("checkpoint_every_updates", 1)),
        eval_every_updates=int(payload.get("eval_every_updates", 5)),
        updates=int(payload.get("updates", 100)),
        hidden_sizes=tuple(int(value) for value in payload.get("hidden_sizes", (256, 256))),
        training_scenarios=tuple(str(value) for value in payload.get("training_scenarios", DEFAULT_TRAINING_SCENARIOS)),
        eval_scenarios=tuple(str(value) for value in payload.get("eval_scenarios", DEFAULT_EVAL_SCENARIOS)),
        eval_opponents=tuple(str(value) for value in payload.get("eval_opponents", DEFAULT_EVAL_OPPONENTS)),
        eval_matches_per_opponent=int(payload.get("eval_matches_per_opponent", 4)),
    )
