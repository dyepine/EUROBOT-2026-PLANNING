from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from math import ceil
import multiprocessing as mp
import random

from poc.actions import Action
from poc.controllers import ActionController, build_scripted_controller
from poc.entities import DEFAULT_ROBOT_SPEED_MPS, Side
from poc.opponent_policy import materialize_policy_name, policy_name_uses_variant_seed
from poc.planner import UtilityPlanner
from poc.rl_checkpoint import OpponentPool, PolicySnapshot, clone_state_dict
from poc.rl_config import SelfPlayConfig, selfplay_config_from_dict
from poc.rl_infra import (
    DEFAULT_ACTION_SPACE,
    DEFAULT_FLAT_FEATURE_KEYS,
    RLObservationConfig,
    RLTransition,
    build_rl_policy_step,
    build_rl_transitions_from_match_result,
    flat_feature_vector,
    resolve_policy_action,
)
from poc.rl_model import MaskedPolicyValueNet, greedy_action, load_compatible_state_dict, sample_action
from poc.rl_ppo import PPORolloutBatch, PPORolloutItem
from poc.scenarios import build_scenario
from poc.simulator import MatchResult, Simulator
from poc.torch_compat import require_torch, torch


def _make_process_pool(max_workers: int) -> ProcessPoolExecutor:
    # `spawn` is safer than the Linux default `fork` when the parent process
    # already owns a CUDA context or imported torch state.
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"))


class PersistentWorkerPools:
    def __init__(self, *, rollout_workers: int, eval_workers: int) -> None:
        self.rollout_workers = max(int(rollout_workers), 1)
        self.eval_workers = max(int(eval_workers), 1)
        self._rollout_executor: ProcessPoolExecutor | None = None
        self._eval_executor: ProcessPoolExecutor | None = None

    def rollout_executor(self) -> ProcessPoolExecutor | None:
        if self.rollout_workers <= 1:
            return None
        if self._rollout_executor is None:
            self._rollout_executor = _make_process_pool(self.rollout_workers)
        return self._rollout_executor

    def eval_executor(self) -> ProcessPoolExecutor | None:
        if self.eval_workers <= 1:
            return None
        if self._eval_executor is None:
            self._eval_executor = _make_process_pool(self.eval_workers)
        return self._eval_executor

    def close(self) -> None:
        if self._rollout_executor is not None:
            self._rollout_executor.shutdown(wait=True, cancel_futures=False)
            self._rollout_executor = None
        if self._eval_executor is not None:
            self._eval_executor.shutdown(wait=True, cancel_futures=False)
            self._eval_executor = None


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


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    opponent: str
    kind: str
    matches: int
    winrate: float
    mean_score_diff: float
    median_score_diff: float
    p10_score_diff: float
    min_score_diff: float
    mean_our_score: float
    mean_enemy_score: float
    successful_return_home_rate: float
    thermometer_used_rate: float
    invalid_action_count: int


def classify_opponent_kind(name: str, *, default_kind: str = "scripted") -> str:
    if default_kind in {"self_play", "random"}:
        return default_kind
    if name == "random_policy":
        return "random"
    return "randomized" if policy_name_uses_variant_seed(name) else "scripted"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    clamped_q = min(max(float(q), 0.0), 1.0)
    ordered = sorted(float(value) for value in values)
    rank = max(1, ceil(clamped_q * len(ordered)))
    return ordered[rank - 1]


def _score_diff_stats(score_diffs: list[float]) -> tuple[float, float, float, float]:
    if not score_diffs:
        return 0.0, 0.0, 0.0, 0.0
    ordered = sorted(float(value) for value in score_diffs)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        median = (ordered[middle - 1] + ordered[middle]) / 2.0
    else:
        median = ordered[middle]
    return (
        sum(ordered) / len(ordered),
        median,
        _quantile(ordered, 0.10),
        ordered[0],
    )


def _estimated_rollout_steps_per_match(
    *,
    steps: int,
    matches: int,
    target_steps: int,
    minimum_matches: int,
) -> int:
    bootstrap = max(1, ceil(target_steps / max(minimum_matches, 1)))
    if matches <= 0 or steps <= 0:
        return bootstrap
    return max(bootstrap, ceil(steps / matches))


def _should_schedule_rollout_match(
    *,
    steps: int,
    matches: int,
    pending_matches: int,
    target_steps: int,
    minimum_matches: int,
) -> bool:
    if matches + pending_matches < minimum_matches:
        return True
    estimated_steps = _estimated_rollout_steps_per_match(
        steps=steps,
        matches=matches,
        target_steps=target_steps,
        minimum_matches=minimum_matches,
    )
    projected_steps = steps + pending_matches * estimated_steps
    return projected_steps < target_steps


def _desired_rollout_pending_matches(
    *,
    steps: int,
    matches: int,
    pending_matches: int,
    target_steps: int,
    minimum_matches: int,
    worker_count: int,
) -> int:
    if matches + pending_matches < minimum_matches:
        return min(worker_count, minimum_matches - matches)
    if _should_schedule_rollout_match(
        steps=steps,
        matches=matches,
        pending_matches=pending_matches,
        target_steps=target_steps,
        minimum_matches=minimum_matches,
    ):
        return worker_count
    return 0


@dataclass(frozen=True, slots=True)
class WorkerMatchRequest:
    config_payload: dict[str, object]
    scenario_name: str
    seed: int
    update_id: int
    episode_id: int
    learner_state_dict: dict[str, object]
    learner_greedy: bool
    opponent_name: str
    opponent_kind: str
    opponent_policy_name: str
    opponent_state_dict: dict[str, object] | None
    opponent_greedy: bool


@dataclass(frozen=True, slots=True)
class WorkerMatchResult:
    rollout_items: tuple[PPORolloutItem, ...]
    match_summary: dict[str, object]


@dataclass(slots=True)
class _WorkerRuntime:
    signature: tuple[object, ...]
    planner: UtilityPlanner
    learner_model: MaskedPolicyValueNet
    opponent_model: MaskedPolicyValueNet


_WORKER_RUNTIME: _WorkerRuntime | None = None


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


def _cpu_worker_config(config: SelfPlayConfig) -> SelfPlayConfig:
    payload = config.to_dict()
    payload["ppo"]["device"] = "cpu"
    return selfplay_config_from_dict(payload)


def _worker_runtime_signature(config: SelfPlayConfig) -> tuple[object, ...]:
    return (
        config.ppo.device,
        tuple(config.ppo.hidden_sizes),
    )


def _get_worker_runtime(config: SelfPlayConfig) -> _WorkerRuntime:
    global _WORKER_RUNTIME
    signature = _worker_runtime_signature(config)
    if _WORKER_RUNTIME is not None and _WORKER_RUNTIME.signature == signature:
        return _WORKER_RUNTIME
    device = torch.device(config.ppo.device)
    learner_model = build_model(config)
    learner_model.to(device)
    opponent_model = build_model(config)
    opponent_model.to(device)
    _WORKER_RUNTIME = _WorkerRuntime(
        signature=signature,
        planner=UtilityPlanner(),
        learner_model=learner_model,
        opponent_model=opponent_model,
    )
    return _WORKER_RUNTIME


def _build_worker_request(
    *,
    config: SelfPlayConfig,
    learner_state_dict: dict[str, object],
    scenario_name: str,
    seed: int,
    update_id: int,
    episode_id: int,
    learner_greedy: bool,
    opponent_spec: OpponentSpec,
) -> WorkerMatchRequest:
    worker_config = _cpu_worker_config(config)
    return WorkerMatchRequest(
        config_payload=worker_config.to_dict(),
        scenario_name=scenario_name,
        seed=seed,
        update_id=update_id,
        episode_id=episode_id,
        learner_state_dict=learner_state_dict,
        learner_greedy=learner_greedy,
        opponent_name=opponent_spec.name,
        opponent_kind=opponent_spec.kind,
        opponent_policy_name=opponent_spec.opponent_policy_name,
        opponent_state_dict=opponent_spec.state_dict,
        opponent_greedy=opponent_spec.greedy,
    )


def _play_match_worker(request: WorkerMatchRequest) -> WorkerMatchResult:
    require_torch(torch)
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch.manual_seed(request.seed)
    config = selfplay_config_from_dict(dict(request.config_payload))
    runtime = _get_worker_runtime(config)
    load_compatible_state_dict(runtime.learner_model, request.learner_state_dict)
    learner_selector = TorchPolicySelector(
        model=runtime.learner_model,
        device=config.ppo.device,
        greedy=request.learner_greedy,
        name="learner_worker",
    )

    opponent_selector: ActionController | None = None
    if request.opponent_kind == "random":
        opponent_selector = RandomMaskedPolicySelector(random.Random(request.seed ^ 0xA5A5A5), name=request.opponent_name)
    elif request.opponent_state_dict is not None:
        load_compatible_state_dict(runtime.opponent_model, request.opponent_state_dict)
        opponent_selector = TorchPolicySelector(
            model=runtime.opponent_model,
            device=config.ppo.device,
            greedy=request.opponent_greedy,
            name=request.opponent_name,
        )

    opponent_spec = OpponentSpec(
        name=request.opponent_name,
        selector=opponent_selector,
        opponent_policy_name=request.opponent_policy_name,
        kind=request.opponent_kind,
        state_dict=request.opponent_state_dict,
        greedy=request.opponent_greedy,
    )
    artifacts = play_match(
        config=config,
        scenario_name=request.scenario_name,
        seed=request.seed,
        learner_selector=learner_selector,
        opponent_spec=opponent_spec,
        planner=runtime.planner,
    )
    items = transitions_to_rollout_items(
        build_rl_transitions_from_match_result(
            artifacts.result,
            side=config.ppo.side,
            config=build_observation_config(config),
            thermometer_reward_bonus=config.thermometer_reward_bonus,
            terminal_win_bonus=config.terminal_win_bonus,
            terminal_draw_bonus=config.terminal_draw_bonus,
            terminal_loss_bonus=config.terminal_loss_bonus,
        ),
        list(artifacts.learner_records),
        update_id=request.update_id,
        episode_id=request.episode_id,
    )
    return WorkerMatchResult(
        rollout_items=tuple(items),
        match_summary={
            "scenario": request.scenario_name,
            "seed": request.seed,
            "opponent": request.opponent_name,
            "opponent_kind": request.opponent_kind,
            "summary": artifacts.result.summary,
            "invalid_action_count": artifacts.invalid_action_count,
        },
    )


def build_rollout_batch(
    *,
    config: SelfPlayConfig,
    learner_model: MaskedPolicyValueNet,
    opponent_pool: OpponentPool,
    planner: UtilityPlanner,
    rng: random.Random,
    update_id: int,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[PPORolloutBatch, list[dict[str, object]]]:
    require_torch(torch)
    rollout_items: list[PPORolloutItem] = []
    match_summaries: list[dict[str, object]] = []
    episode_counter = 0
    steps = 0
    matches = 0
    target_steps = max(config.ppo.steps_per_update, 1)
    minimum_matches = max(config.ppo.matches_per_update, 1)
    worker_count = max(config.rollout_workers, 1)
    learner_state_dict = clone_state_dict(learner_model)
    progress_interval = 4

    print(
        f"[update {update_id}] rollout start target_steps={target_steps} "
        f"minimum_matches={minimum_matches} workers={worker_count}",
        flush=True,
    )

    if worker_count > 1:
        def schedule_request(*, executor: ProcessPoolExecutor, episode_id: int) -> Future[WorkerMatchResult]:
            scenario_name = rng.choice(config.training_scenarios)
            seed = rng.randint(1, 10_000_000)
            opponent_spec = sample_training_opponent(opponent_pool, config, rng, materialize_selector=False)
            request = _build_worker_request(
                config=config,
                learner_state_dict=learner_state_dict,
                scenario_name=scenario_name,
                seed=seed,
                update_id=update_id,
                episode_id=episode_id,
                learner_greedy=False,
                opponent_spec=opponent_spec,
            )
            return executor.submit(_play_match_worker, request)

        owns_executor = executor is None
        active_executor = executor if executor is not None else _make_process_pool(worker_count)
        try:
            in_flight: set[Future[WorkerMatchResult]] = set()
            while len(in_flight) < _desired_rollout_pending_matches(
                steps=steps,
                matches=matches,
                pending_matches=len(in_flight),
                target_steps=target_steps,
                minimum_matches=minimum_matches,
                worker_count=worker_count,
            ):
                in_flight.add(schedule_request(executor=active_executor, episode_id=episode_counter))
                episode_counter += 1
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    rollout_items.extend(result.rollout_items)
                    steps += len(result.rollout_items)
                    matches += 1
                    match_summaries.append(result.match_summary)
                    if matches % progress_interval == 0 or not (steps < target_steps or matches < minimum_matches):
                        print(
                            f"[update {update_id}] rollout progress matches={matches} steps={steps}",
                            flush=True,
                        )
                    while len(in_flight) < _desired_rollout_pending_matches(
                        steps=steps,
                        matches=matches,
                        pending_matches=len(in_flight),
                        target_steps=target_steps,
                        minimum_matches=minimum_matches,
                        worker_count=worker_count,
                    ):
                        in_flight.add(schedule_request(executor=active_executor, episode_id=episode_counter))
                        episode_counter += 1
        finally:
            if owns_executor:
                active_executor.shutdown(wait=True, cancel_futures=False)
        print(
            f"[update {update_id}] rollout done matches={matches} steps={steps}",
            flush=True,
        )
        return PPORolloutBatch(items=tuple(rollout_items)), match_summaries

    learner_selector = TorchPolicySelector(
        model=learner_model,
        device=config.ppo.device,
        greedy=False,
        name="learner_policy",
    )
    while steps < target_steps or matches < minimum_matches:
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
            build_rl_transitions_from_match_result(
                artifacts.result,
                side=config.ppo.side,
                config=build_observation_config(config),
                thermometer_reward_bonus=config.thermometer_reward_bonus,
                terminal_win_bonus=config.terminal_win_bonus,
                terminal_draw_bonus=config.terminal_draw_bonus,
                terminal_loss_bonus=config.terminal_loss_bonus,
            ),
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
                "opponent_kind": opponent_spec.kind,
                "summary": artifacts.result.summary,
                "invalid_action_count": artifacts.invalid_action_count,
            }
        )
        if matches % progress_interval == 0 or not (steps < target_steps or matches < minimum_matches):
            print(
                f"[update {update_id}] rollout progress matches={matches} steps={steps}",
                flush=True,
            )
    print(
        f"[update {update_id}] rollout done matches={matches} steps={steps}",
        flush=True,
    )
    return PPORolloutBatch(items=tuple(rollout_items)), match_summaries


def evaluate_policy(
    *,
    config: SelfPlayConfig,
    learner_model: MaskedPolicyValueNet,
    planner: UtilityPlanner,
    rng: random.Random,
    opponent_specs: list[OpponentSpec],
    executor: ProcessPoolExecutor | None = None,
) -> list[EvaluationSummary]:
    require_torch(torch)
    summaries: list[EvaluationSummary] = []
    worker_count = max(config.eval_workers, 1)
    learner_state_dict = clone_state_dict(learner_model)

    if worker_count > 1:
        owns_executor = executor is None
        active_executor = executor if executor is not None else _make_process_pool(worker_count)
        try:
            for opponent_spec in opponent_specs:
                print(
                    f"[eval] opponent={opponent_spec.name} start matches={config.eval_matches_per_opponent} "
                    f"workers={worker_count}",
                    flush=True,
                )
                wins = 0
                score_diffs: list[float] = []
                total_our_score = 0.0
                total_enemy_score = 0.0
                total_return_home = 0.0
                total_thermometer = 0.0
                total_invalid = 0
                futures: list[Future[WorkerMatchResult]] = []
                for match_index in range(config.eval_matches_per_opponent):
                    scenario_name = config.eval_scenarios[match_index % len(config.eval_scenarios)]
                    seed = rng.randint(1, 10_000_000)
                    request = _build_worker_request(
                        config=config,
                        learner_state_dict=learner_state_dict,
                        scenario_name=scenario_name,
                        seed=seed,
                        update_id=-1,
                        episode_id=match_index,
                        learner_greedy=True,
                        opponent_spec=opponent_spec,
                    )
                    futures.append(active_executor.submit(_play_match_worker, request))

                for future in futures:
                    result = future.result()
                    summary = result.match_summary["summary"]
                    wins += 1 if summary["win"] else 0
                    score_diffs.append(float(summary["score_diff"]))
                    total_our_score += float(summary["our_score"])
                    total_enemy_score += float(summary["enemy_score"])
                    total_return_home += 1.0 if summary["successful_return_home"] else 0.0
                    total_thermometer += 1.0 if summary["thermometer_used"] else 0.0
                    total_invalid += int(result.match_summary["invalid_action_count"])
                matches = max(config.eval_matches_per_opponent, 1)
                mean_score_diff, median_score_diff, p10_score_diff, min_score_diff = _score_diff_stats(score_diffs)
                summaries.append(
                    EvaluationSummary(
                        opponent=opponent_spec.name,
                        kind=opponent_spec.kind,
                        matches=matches,
                        winrate=wins / matches,
                        mean_score_diff=mean_score_diff,
                        median_score_diff=median_score_diff,
                        p10_score_diff=p10_score_diff,
                        min_score_diff=min_score_diff,
                        mean_our_score=total_our_score / matches,
                        mean_enemy_score=total_enemy_score / matches,
                        successful_return_home_rate=total_return_home / matches,
                        thermometer_used_rate=total_thermometer / matches,
                        invalid_action_count=total_invalid,
                    )
                )
                print(
                    f"[eval] opponent={opponent_spec.name} done winrate={wins / matches:.3f} "
                    f"score_diff={mean_score_diff:.3f} p10={p10_score_diff:.3f} min={min_score_diff:.3f}",
                    flush=True,
                )
        finally:
            if owns_executor:
                active_executor.shutdown(wait=True, cancel_futures=False)
        return summaries

    for opponent_spec in opponent_specs:
        print(
            f"[eval] opponent={opponent_spec.name} start matches={config.eval_matches_per_opponent} "
            f"workers={worker_count}",
            flush=True,
        )
        wins = 0
        score_diffs: list[float] = []
        total_our_score = 0.0
        total_enemy_score = 0.0
        total_return_home = 0.0
        total_thermometer = 0.0
        total_invalid = 0
        for match_index in range(config.eval_matches_per_opponent):
            learner_selector = TorchPolicySelector(
                model=learner_model,
                device=config.ppo.device,
                greedy=True,
                name="learner_eval",
            )
            if opponent_spec.selector is None:
                eval_opponent = OpponentSpec(
                    name=opponent_spec.name,
                    selector=None,
                    opponent_policy_name=opponent_spec.opponent_policy_name,
                    kind=opponent_spec.kind,
                    state_dict=opponent_spec.state_dict,
                    greedy=opponent_spec.greedy,
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
            score_diffs.append(float(summary["score_diff"]))
            total_our_score += float(summary["our_score"])
            total_enemy_score += float(summary["enemy_score"])
            total_return_home += 1.0 if summary["successful_return_home"] else 0.0
            total_thermometer += 1.0 if summary["thermometer_used"] else 0.0
            total_invalid += artifacts.invalid_action_count
        matches = max(config.eval_matches_per_opponent, 1)
        mean_score_diff, median_score_diff, p10_score_diff, min_score_diff = _score_diff_stats(score_diffs)
        summaries.append(
            EvaluationSummary(
                opponent=opponent_spec.name,
                kind=opponent_spec.kind,
                matches=matches,
                winrate=wins / matches,
                mean_score_diff=mean_score_diff,
                median_score_diff=median_score_diff,
                p10_score_diff=p10_score_diff,
                min_score_diff=min_score_diff,
                mean_our_score=total_our_score / matches,
                mean_enemy_score=total_enemy_score / matches,
                successful_return_home_rate=total_return_home / matches,
                thermometer_used_rate=total_thermometer / matches,
                invalid_action_count=total_invalid,
            )
        )
        print(
            f"[eval] opponent={opponent_spec.name} done winrate={wins / matches:.3f} "
            f"score_diff={mean_score_diff:.3f} p10={p10_score_diff:.3f} min={min_score_diff:.3f}",
            flush=True,
        )
    return summaries
