from __future__ import annotations

from dataclasses import dataclass
import random

from poc.control.controllers import ActionController, build_scripted_controller
from poc.domain.entities import DEFAULT_ROBOT_SPEED_MPS, Side
from poc.control.opponent_policy import materialize_policy_name, policy_name_uses_variant_seed
from poc.planning.planner import UtilityPlanner
from poc.rl.checkpoint import OpponentPool, PolicySnapshot
from poc.rl.config import SelfPlayConfig
from poc.rl.encoder import DEFAULT_FLAT_FEATURE_KEYS, RLObservationConfig, flat_feature_vector
from poc.rl.model import MaskedPolicyValueNet, load_compatible_state_dict
from poc.rl.ppo import PPORolloutItem
from poc.rl.selectors import PolicyDecisionRecord, RandomMaskedPolicySelector, TorchPolicySelector
from poc.rl.tokens import DEFAULT_ACTION_SPACE
from poc.rl.transitions import RLTransition, build_rl_transitions_from_match_result
from poc.simulation.scenarios import build_scenario
from poc.simulation.simulator import MatchResult, Simulator
from poc.rl.torch_compat import require_torch, torch

@dataclass(frozen=True, slots=True)
class OpponentSpec:
    name: str
    selector: ActionController | None = None
    opponent_policy_name: str = "nearest_greedy"
    kind: str = "scripted"
    state_dict: dict[str, object] | None = None
    greedy: bool = False


@dataclass(frozen=True, slots=True)
class SelfPlayMatchArtifacts:
    result: MatchResult
    learner_records: tuple[PolicyDecisionRecord, ...]
    rollout_items: tuple[PPORolloutItem, ...]
    invalid_action_count: int


def classify_opponent_kind(name: str, *, default_kind: str = "scripted") -> str:
    if default_kind in {"self_play", "random"}:
        return default_kind
    if name == "random_policy":
        return "random"
    return "randomized" if policy_name_uses_variant_seed(name) else "scripted"


def observation_dim() -> int:
    return len(DEFAULT_FLAT_FEATURE_KEYS)


def action_dim() -> int:
    return len(DEFAULT_ACTION_SPACE.tokens)


def build_model(config: SelfPlayConfig) -> MaskedPolicyValueNet:
    return MaskedPolicyValueNet(
        observation_dim=observation_dim(),
        action_dim=action_dim(),
        hidden_sizes=config.ppo.hidden_sizes,
    )


def build_observation_config(config: SelfPlayConfig) -> RLObservationConfig:
    return RLObservationConfig(
        observation_noise_seed=config.ppo.seed,
        enemy_velocity_noise_std_mps=config.enemy_velocity_noise_std_mps,
        enemy_velocity_self_motion_leak_fraction=config.enemy_velocity_self_motion_leak_fraction,
        enemy_velocity_self_motion_leak_duration_s=config.enemy_velocity_self_motion_leak_duration_s,
    )


def selector_from_snapshot(
    snapshot: PolicySnapshot,
    *,
    config: SelfPlayConfig,
    greedy: bool,
) -> TorchPolicySelector:
    require_torch(torch)
    model = build_model(config)
    load_compatible_state_dict(model, snapshot.state_dict)
    model.to(torch.device(config.ppo.device))
    return TorchPolicySelector(
        model=model,
        device=config.ppo.device,
        greedy=greedy,
        name=snapshot.name,
    )


def sample_training_opponent(
    pool: OpponentPool,
    config: SelfPlayConfig,
    rng: random.Random,
    *,
    materialize_selector: bool = True,
) -> OpponentSpec:
    scripted_opponents = config.training_scripted_opponents
    scripted_fraction = min(max(config.training_scripted_fraction, 0.0), 1.0)
    if scripted_opponents and rng.random() < scripted_fraction:
        base_name = rng.choice(scripted_opponents)
        name = materialize_policy_name(base_name, rng)
        return OpponentSpec(
            name=name,
            selector=None,
            opponent_policy_name=name,
            kind=classify_opponent_kind(name),
        )
    snapshot = pool.sample_training_snapshot(rng)
    if snapshot is None:
        selector = RandomMaskedPolicySelector(rng) if materialize_selector else None
        return OpponentSpec(name="random_policy", selector=selector, kind="random")
    selector = selector_from_snapshot(snapshot, config=config, greedy=False) if materialize_selector else None
    return OpponentSpec(
        name=snapshot.name,
        selector=selector,
        kind="self_play",
        state_dict=snapshot.state_dict,
        greedy=False,
    )


def enemy_speed_scale_for_match(config: SelfPlayConfig, seed: int) -> float:
    jitter = max(0.0, float(config.enemy_speed_jitter_fraction))
    if jitter <= 0.0:
        return 1.0
    match_rng = random.Random((seed ^ 0x5DEECE66D) & 0xFFFFFFFF)
    return match_rng.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)


def play_match(
    *,
    config: SelfPlayConfig,
    scenario_name: str,
    seed: int,
    learner_selector: TorchPolicySelector,
    opponent_spec: OpponentSpec,
    planner: UtilityPlanner | None = None,
) -> SelfPlayMatchArtifacts:
    enemy_speed_scale = enemy_speed_scale_for_match(config, seed)
    scenario = build_scenario(
        scenario_name,
        seed=seed,
        our_side=config.ppo.side,
        opponent_policy_name=opponent_spec.opponent_policy_name,
        our_robot_speed=DEFAULT_ROBOT_SPEED_MPS,
        enemy_robot_speed=DEFAULT_ROBOT_SPEED_MPS * enemy_speed_scale,
    )
    selectors = {
        config.ppo.side: learner_selector,
    }
    if opponent_spec.selector is not None:
        selectors[config.ppo.side.opponent()] = opponent_spec.selector
    simulator = Simulator(
        state=scenario.game_state,
        scenario_name=scenario.name,
        planner=planner or UtilityPlanner(),
        dt=config.ppo.dt,
        controllers=selectors,
        opponent_controller=build_scripted_controller(opponent_spec.opponent_policy_name),
    )
    learner_selector.reset()
    if opponent_spec.selector is not None and hasattr(opponent_spec.selector, "reset"):
        opponent_spec.selector.reset()
    result = simulator.run()
    result.summary["our_speed_mps"] = round(scenario.game_state.robot_for_side(config.ppo.side).speed, 4)
    result.summary["enemy_speed_mps"] = round(scenario.game_state.robot_for_side(config.ppo.side.opponent()).speed, 4)
    result.summary["enemy_speed_scale"] = round(enemy_speed_scale, 4)
    learner_transitions = build_rl_transitions_from_match_result(
        result,
        side=config.ppo.side,
        config=build_observation_config(config),
        thermometer_reward_bonus=config.thermometer_reward_bonus,
        terminal_win_bonus=config.terminal_win_bonus,
        terminal_draw_bonus=config.terminal_draw_bonus,
        terminal_loss_bonus=config.terminal_loss_bonus,
    )
    rollout_items = transitions_to_rollout_items(
        learner_transitions,
        learner_selector.records,
        update_id=-1,
        episode_id=seed,
    )
    return SelfPlayMatchArtifacts(
        result=result,
        learner_records=tuple(learner_selector.records),
        rollout_items=tuple(rollout_items),
        invalid_action_count=learner_selector.invalid_action_count,
    )


def transitions_to_rollout_items(
    transitions: list[RLTransition],
    records: list[PolicyDecisionRecord],
    *,
    update_id: int,
    episode_id: int,
) -> list[PPORolloutItem]:
    if len(transitions) != len(records):
        raise ValueError(
            "Mismatch between learner transitions and selector decisions: "
            f"{len(transitions)} transitions vs {len(records)} selector records."
        )
    items: list[PPORolloutItem] = []
    for transition, record in zip(transitions, records):
        items.append(
            PPORolloutItem(
                observation=flat_feature_vector(transition.observation),
                next_observation=flat_feature_vector(transition.next_observation),
                action_mask=transition.action_mask,
                next_action_mask=transition.next_action_mask,
                chosen_action_index=record.action_index,
                reward=transition.reward,
                done=transition.done,
                log_prob=record.log_prob,
                value=record.value,
                entropy=record.entropy,
                episode_id=episode_id,
                update_id=update_id,
            )
        )
    return items
