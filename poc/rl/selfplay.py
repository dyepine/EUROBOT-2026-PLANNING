from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from math import ceil
import random

from poc.rl.checkpoint import OpponentPool, clone_state_dict
from poc.rl.config import SelfPlayConfig
from poc.rl.match import (
    OpponentSpec,
    SelfPlayMatchArtifacts,
    action_dim,
    build_model,
    build_observation_config,
    classify_opponent_kind,
    enemy_speed_scale_for_match,
    observation_dim,
    play_match,
    sample_training_opponent,
    selector_from_snapshot,
    transitions_to_rollout_items,
)
from poc.rl.model import MaskedPolicyValueNet
from poc.rl.ppo import PPORolloutBatch
from poc.rl.selectors import PolicyDecisionRecord, RandomMaskedPolicySelector, TorchPolicySelector
from poc.rl.transitions import RLTransition, build_rl_transitions_from_match_result
from poc.rl.workers import (
    PersistentWorkerPools,
    WorkerMatchRequest,
    WorkerMatchResult,
    _build_worker_request,
    _make_process_pool,
    _play_match_worker,
)
from poc.rl.torch_compat import require_torch, torch

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
