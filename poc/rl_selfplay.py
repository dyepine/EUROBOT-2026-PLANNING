from __future__ import annotations

from dataclasses import dataclass
import random

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is unavailable
    torch = None

from poc.actions import Action
from poc.entities import Side
from poc.opponent_policy import build_opponent_policy
from poc.planner import UtilityPlanner
from poc.rl_checkpoint import OpponentPool, PolicySnapshot
from poc.rl_config import SelfPlayConfig
from poc.rl_infra import (
    DEFAULT_ACTION_SPACE,
    DEFAULT_FLAT_FEATURE_KEYS,
    RLTransition,
    flat_feature_vector,
    resolve_policy_action,
)
from poc.rl_model import MaskedPolicyValueNet, greedy_action, sample_action
from poc.rl_ppo import PPORolloutBatch, PPORolloutItem
from poc.scenarios import build_scenario
from poc.simulator import ActionSelector, Simulator


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for self-play PPO. Install project dependencies with torch.")


@dataclass(frozen=True, slots=True)
class PolicyDecisionRecord:
    policy_action: str
    action_index: int
    log_prob: float
    value: float
    entropy: float
    invalid_action: bool


@dataclass(frozen=True, slots=True)
class OpponentSpec:
    name: str
    selector: ActionSelector | None = None
    opponent_policy_name: str = "nearest_greedy"


@dataclass(frozen=True, slots=True)
class SelfPlayMatchArtifacts:
    result: MatchResult
    learner_records: tuple[PolicyDecisionRecord, ...]
    rollout_items: tuple[PPORolloutItem, ...]
    invalid_action_count: int


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    opponent: str
    matches: int
    winrate: float
    mean_score_diff: float
    mean_our_score: float
    mean_enemy_score: float
    successful_return_home_rate: float
    thermometer_used_rate: float
    invalid_action_count: int


class RandomMaskedPolicySelector:
    def __init__(self, rng: random.Random, name: str = "random_policy") -> None:
        self.rng = rng
        self.name = name

    def select_action(
        self,
        *,
        state,
        planner,
        side,
        ranked_actions,
        policy_step,
    ) -> Action:
        del state, planner
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
        _require_torch()
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
        state,
        planner,
        side,
        ranked_actions,
        policy_step,
    ) -> Action:
        del state, planner
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
    from poc.policy_mapping import normalized_action_label

    return normalized_action_label(action, side)


def observation_dim() -> int:
    return len(DEFAULT_FLAT_FEATURE_KEYS)


def action_dim() -> int:
    return len(DEFAULT_ACTION_SPACE.tokens)


def build_model(config: SelfPlayConfig) -> MaskedPolicyValueNet:
    return MaskedPolicyValueNet(
        observation_dim=observation_dim(),
        action_dim=action_dim(),
        hidden_sizes=config.hidden_sizes,
    )


def selector_from_snapshot(
    snapshot: PolicySnapshot,
    *,
    config: SelfPlayConfig,
    greedy: bool,
) -> TorchPolicySelector:
    _require_torch()
    model = build_model(config)
    model.load_state_dict(snapshot.state_dict)
    model.to(torch.device(config.device))
    return TorchPolicySelector(
        model=model,
        device=config.device,
        greedy=greedy,
        name=snapshot.name,
    )


def sample_training_opponent(
    pool: OpponentPool,
    config: SelfPlayConfig,
    rng: random.Random,
) -> OpponentSpec:
    snapshot = pool.sample_training_snapshot(rng)
    if snapshot is None:
        return OpponentSpec(name="random_policy", selector=RandomMaskedPolicySelector(rng))
    return OpponentSpec(
        name=snapshot.name,
        selector=selector_from_snapshot(snapshot, config=config, greedy=False),
    )


def play_match(
    *,
    config: SelfPlayConfig,
    scenario_name: str,
    seed: int,
    learner_selector: TorchPolicySelector,
    opponent_spec: OpponentSpec,
    planner: UtilityPlanner | None = None,
) -> SelfPlayMatchArtifacts:
    scenario = build_scenario(
        scenario_name,
        seed=seed,
        our_side=config.side,
        opponent_policy_name=opponent_spec.opponent_policy_name,
    )
    selectors = {
        config.side: learner_selector,
    }
    if opponent_spec.selector is not None:
        selectors[config.side.opponent()] = opponent_spec.selector
    simulator = Simulator(
        state=scenario.game_state,
        scenario_name=scenario.name,
        opponent_policy=build_opponent_policy(opponent_spec.opponent_policy_name),
        planner=planner or UtilityPlanner(),
        dt=config.dt,
        action_selectors=selectors,
    )
    learner_selector.reset()
    if opponent_spec.selector is not None and hasattr(opponent_spec.selector, "reset"):
        opponent_spec.selector.reset()
    result = simulator.run()
    learner_transitions = [item for item in result.rl_transitions if item.side == config.side.value]
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


def build_rollout_batch(
    *,
    config: SelfPlayConfig,
    learner_model: MaskedPolicyValueNet,
    opponent_pool: OpponentPool,
    planner: UtilityPlanner,
    rng: random.Random,
    update_id: int,
) -> tuple[PPORolloutBatch, list[dict[str, object]]]:
    rollout_items: list[PPORolloutItem] = []
    match_summaries: list[dict[str, object]] = []
    episode_counter = 0
    steps = 0
    matches = 0
    learner_selector = TorchPolicySelector(
        model=learner_model,
        device=config.device,
        greedy=False,
        name="learner_policy",
    )
    while steps < config.steps_per_update and matches < config.matches_per_update:
        scenario_name = rng.choice(config.training_scenarios)
        seed = rng.randint(1, 10_000_000)
        opponent_spec = sample_training_opponent(opponent_pool, config, rng)
        artifacts = play_match(
            config=config,
            scenario_name=scenario_name,
            seed=seed,
            learner_selector=learner_selector,
            opponent_spec=opponent_spec,
            planner=planner,
        )
        items = transitions_to_rollout_items(
            [item for item in artifacts.result.rl_transitions if item.side == config.side.value],
            list(artifacts.learner_records),
            update_id=update_id,
            episode_id=episode_counter,
        )
        rollout_items.extend(items)
        steps += len(items)
        matches += 1
        episode_counter += 1
        match_summaries.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "opponent": opponent_spec.name,
                "summary": artifacts.result.summary,
                "invalid_action_count": artifacts.invalid_action_count,
            }
        )
    return PPORolloutBatch(items=tuple(rollout_items)), match_summaries


def evaluate_policy(
    *,
    config: SelfPlayConfig,
    learner_model: MaskedPolicyValueNet,
    planner: UtilityPlanner,
    rng: random.Random,
    opponent_specs: list[OpponentSpec],
) -> list[EvaluationSummary]:
    summaries: list[EvaluationSummary] = []
    for opponent_spec in opponent_specs:
        wins = 0
        total_score_diff = 0.0
        total_our_score = 0.0
        total_enemy_score = 0.0
        total_return_home = 0.0
        total_thermometer = 0.0
        total_invalid = 0
        for match_index in range(config.eval_matches_per_opponent):
            learner_selector = TorchPolicySelector(
                model=learner_model,
                device=config.device,
                greedy=True,
                name="learner_eval",
            )
            if opponent_spec.selector is None:
                eval_opponent = OpponentSpec(
                    name=opponent_spec.name,
                    selector=None,
                    opponent_policy_name=opponent_spec.opponent_policy_name,
                )
            else:
                eval_opponent = opponent_spec
            scenario_name = config.eval_scenarios[match_index % len(config.eval_scenarios)]
            seed = rng.randint(1, 10_000_000)
            artifacts = play_match(
                config=config,
                scenario_name=scenario_name,
                seed=seed,
                learner_selector=learner_selector,
                opponent_spec=eval_opponent,
                planner=planner,
            )
            summary = artifacts.result.summary
            wins += 1 if summary["win"] else 0
            total_score_diff += float(summary["score_diff"])
            total_our_score += float(summary["our_score"])
            total_enemy_score += float(summary["enemy_score"])
            total_return_home += 1.0 if summary["successful_return_home"] else 0.0
            total_thermometer += 1.0 if summary["thermometer_used"] else 0.0
            total_invalid += artifacts.invalid_action_count
        matches = max(config.eval_matches_per_opponent, 1)
        summaries.append(
            EvaluationSummary(
                opponent=opponent_spec.name,
                matches=matches,
                winrate=wins / matches,
                mean_score_diff=total_score_diff / matches,
                mean_our_score=total_our_score / matches,
                mean_enemy_score=total_enemy_score / matches,
                successful_return_home_rate=total_return_home / matches,
                thermometer_used_rate=total_thermometer / matches,
                invalid_action_count=total_invalid,
            )
        )
    return summaries
