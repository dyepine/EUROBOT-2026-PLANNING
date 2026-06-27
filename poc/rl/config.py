from __future__ import annotations

from dataclasses import asdict, dataclass, field

from poc.domain.entities import Side

DEFAULT_TRAINING_SCENARIOS = ("baseline", "delayed_sources")
DEFAULT_EVAL_SCENARIOS = ("baseline", "delayed_sources")
DEFAULT_EVAL_OPPONENTS = ("nearest_greedy", "aggressive", "thermo_first", "uniform_random")
DEFAULT_TRAINING_SCRIPTED_OPPONENTS = (
    "nearest_greedy",
    "thermo_first",
    "storage_first",
    "home_safe",
    "uniform_random",
    "randomized_aggressive",
    "randomized_nearest",
    "randomized_mixed",
    "stochastic_planner",
)


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
    hidden_sizes: tuple[int, ...] = (256, 256, 256)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["side"] = self.side.value
        return payload


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    ppo: PPOConfig = field(default_factory=PPOConfig)
    thermometer_reward_bonus: float = 3.0
    terminal_win_bonus: float = 2.0
    terminal_draw_bonus: float = 0.0
    terminal_loss_bonus: float = -2.0
    enemy_speed_jitter_fraction: float = 0.30
    enemy_velocity_noise_std_mps: float = 0.01
    enemy_velocity_self_motion_leak_fraction: float = 0.0
    enemy_velocity_self_motion_leak_duration_s: float = 0.0
    rollout_workers: int = 10
    eval_workers: int = 10
    training_scenarios: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TRAINING_SCENARIOS)
    training_scripted_opponents: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TRAINING_SCRIPTED_OPPONENTS)
    training_scripted_fraction: float = 0.30
    eval_scenarios: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EVAL_SCENARIOS)
    eval_opponents: tuple[str, ...] = field(default_factory=lambda: DEFAULT_EVAL_OPPONENTS)
    eval_matches_per_opponent: int = 4

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ppo"] = self.ppo.to_dict()
        return payload


def _ppo_config_from_payload(payload: dict[str, object]) -> PPOConfig:
    return PPOConfig(
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
        side=payload.get("side") if isinstance(payload.get("side"), Side) else Side(str(payload.get("side", Side.BLUE.value))),
        dt=float(payload.get("dt", 0.5)),
        opponent_pool_size=int(payload.get("opponent_pool_size", 8)),
        checkpoint_every_updates=int(payload.get("checkpoint_every_updates", 1)),
        eval_every_updates=int(payload.get("eval_every_updates", 5)),
        updates=int(payload.get("updates", 100)),
        hidden_sizes=tuple(int(value) for value in payload.get("hidden_sizes", (256, 256, 256))),
    )


def selfplay_config_from_dict(payload: dict[str, object]) -> SelfPlayConfig:
    ppo_payload = dict(payload.get("ppo", {}))
    return SelfPlayConfig(
        ppo=_ppo_config_from_payload(ppo_payload),
        thermometer_reward_bonus=float(payload.get("thermometer_reward_bonus", 3.0)),
        terminal_win_bonus=float(payload.get("terminal_win_bonus", 2.0)),
        terminal_draw_bonus=float(payload.get("terminal_draw_bonus", 0.0)),
        terminal_loss_bonus=float(payload.get("terminal_loss_bonus", -2.0)),
        enemy_speed_jitter_fraction=float(payload.get("enemy_speed_jitter_fraction", 0.30)),
        enemy_velocity_noise_std_mps=float(payload.get("enemy_velocity_noise_std_mps", 0.01)),
        enemy_velocity_self_motion_leak_fraction=float(payload.get("enemy_velocity_self_motion_leak_fraction", 0.0)),
        enemy_velocity_self_motion_leak_duration_s=float(payload.get("enemy_velocity_self_motion_leak_duration_s", 0.0)),
        rollout_workers=int(payload.get("rollout_workers", 10)),
        eval_workers=int(payload.get("eval_workers", 10)),
        training_scenarios=tuple(str(value) for value in payload.get("training_scenarios", DEFAULT_TRAINING_SCENARIOS)),
        training_scripted_opponents=tuple(
            str(value) for value in payload.get("training_scripted_opponents", DEFAULT_TRAINING_SCRIPTED_OPPONENTS)
        ),
        training_scripted_fraction=float(payload.get("training_scripted_fraction", 0.30)),
        eval_scenarios=tuple(str(value) for value in payload.get("eval_scenarios", DEFAULT_EVAL_SCENARIOS)),
        eval_opponents=tuple(str(value) for value in payload.get("eval_opponents", DEFAULT_EVAL_OPPONENTS)),
        eval_matches_per_opponent=int(payload.get("eval_matches_per_opponent", 4)),
    )
