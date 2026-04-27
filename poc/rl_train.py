from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is unavailable
    torch = None

from poc.rl_checkpoint import OpponentPool, save_policy_checkpoint, save_training_state
from poc.rl_config import DEFAULT_EVAL_OPPONENTS, DEFAULT_TRAINING_SCENARIOS, SelfPlayConfig
from poc.rl_ppo import ppo_update
from poc.rl_selfplay import (
    EvaluationSummary,
    OpponentSpec,
    action_dim,
    build_model,
    build_rollout_batch,
    evaluate_policy,
    observation_dim,
    selector_from_snapshot,
)
from poc.planner import UtilityPlanner


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for self-play PPO. Install project dependencies with torch.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a self-play PPO policy for the Eurobot POC.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "ppo_selfplay")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps-per-update", type=int, default=1024)
    parser.add_argument("--matches-per-update", type=int, default=16)
    parser.add_argument("--epochs-per-update", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--side", choices=["blue", "yellow"], default="blue")
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--opponent-pool-size", type=int, default=8)
    parser.add_argument("--checkpoint-every-updates", type=int, default=1)
    parser.add_argument("--eval-every-updates", type=int, default=5)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--training-scenarios", nargs="+", default=list(DEFAULT_TRAINING_SCENARIOS))
    parser.add_argument("--eval-scenarios", nargs="+", default=["baseline", "delayed_sources"])
    parser.add_argument("--eval-opponents", nargs="+", default=list(DEFAULT_EVAL_OPPONENTS))
    parser.add_argument("--eval-matches-per-opponent", type=int, default=4)
    return parser


def build_config(args: argparse.Namespace) -> SelfPlayConfig:
    from poc.entities import Side

    return SelfPlayConfig(
        seed=args.seed,
        device=args.device,
        steps_per_update=args.steps_per_update,
        matches_per_update=args.matches_per_update,
        epochs_per_update=args.epochs_per_update,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        side=Side(args.side),
        dt=args.dt,
        opponent_pool_size=args.opponent_pool_size,
        checkpoint_every_updates=args.checkpoint_every_updates,
        eval_every_updates=args.eval_every_updates,
        updates=args.updates,
        hidden_sizes=tuple(args.hidden_sizes),
        training_scenarios=tuple(args.training_scenarios),
        eval_scenarios=tuple(args.eval_scenarios),
        eval_opponents=tuple(args.eval_opponents),
        eval_matches_per_opponent=args.eval_matches_per_opponent,
    )


def build_eval_opponents(config: SelfPlayConfig, pool: OpponentPool) -> list[OpponentSpec]:
    opponents = [
        OpponentSpec(name=name, selector=None, opponent_policy_name=name)
        for name in config.eval_opponents
    ]
    for snapshot in pool.evaluation_snapshots(limit=2):
        opponents.append(
            OpponentSpec(
                name=snapshot.name,
                selector=selector_from_snapshot(snapshot, config=config, greedy=True),
            )
        )
    return opponents


def aggregate_eval_metrics(summaries: list[EvaluationSummary]) -> tuple[float, float]:
    if not summaries:
        return 0.0, 0.0
    total_matches = sum(summary.matches for summary in summaries)
    if total_matches <= 0:
        return 0.0, 0.0
    winrate = sum(summary.winrate * summary.matches for summary in summaries) / total_matches
    score_diff = sum(summary.mean_score_diff * summary.matches for summary in summaries) / total_matches
    return winrate, score_diff


def main(argv: list[str] | None = None) -> int:
    _require_torch()
    args = build_parser().parse_args(argv)
    config = build_config(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    planner = UtilityPlanner()
    learner_model = build_model(config).to(torch.device(config.device))
    optimizer = torch.optim.Adam(learner_model.parameters(), lr=config.learning_rate)
    opponent_pool = OpponentPool(config.opponent_pool_size)

    best_winrate = float("-inf")
    best_winrate_tiebreak = float("-inf")
    best_score_diff = float("-inf")

    for update_id in range(1, config.updates + 1):
        rollout_batch, match_summaries = build_rollout_batch(
            config=config,
            learner_model=learner_model,
            opponent_pool=opponent_pool,
            planner=planner,
            rng=rng,
            update_id=update_id,
        )
        update_stats = ppo_update(
            learner_model,
            optimizer,
            rollout_batch,
            config.ppo_config(),
            rng=rng,
        )

        opponent_pool.record_latest(learner_model, update_id)
        save_policy_checkpoint(
            output_dir / "latest.pt",
            model=learner_model,
            config=config,
            update_id=update_id,
            observation_dim=observation_dim(),
            action_dim=action_dim(),
            metadata={"type": "latest"},
        )
        if update_id % config.checkpoint_every_updates == 0:
            opponent_pool.add_checkpoint_snapshot(learner_model, update_id)

        eval_summaries: list[EvaluationSummary] = []
        overall_winrate = 0.0
        overall_score_diff = 0.0
        if update_id % config.eval_every_updates == 0:
            eval_summaries = evaluate_policy(
                config=config,
                learner_model=learner_model,
                planner=planner,
                rng=rng,
                opponent_specs=build_eval_opponents(config, opponent_pool),
            )
            overall_winrate, overall_score_diff = aggregate_eval_metrics(eval_summaries)
            is_best_winrate = (
                overall_winrate > best_winrate
                or overall_winrate == best_winrate and overall_score_diff > best_winrate_tiebreak
            )
            if is_best_winrate:
                best_winrate = overall_winrate
                best_winrate_tiebreak = overall_score_diff
                save_policy_checkpoint(
                    output_dir / "best_winrate.pt",
                    model=learner_model,
                    config=config,
                    update_id=update_id,
                    observation_dim=observation_dim(),
                    action_dim=action_dim(),
                    metadata={"metric": "winrate", "winrate": overall_winrate, "score_diff": overall_score_diff},
                )
            if overall_score_diff > best_score_diff:
                best_score_diff = overall_score_diff
                save_policy_checkpoint(
                    output_dir / "best_score_diff.pt",
                    model=learner_model,
                    config=config,
                    update_id=update_id,
                    observation_dim=observation_dim(),
                    action_dim=action_dim(),
                    metadata={"metric": "score_diff", "winrate": overall_winrate, "score_diff": overall_score_diff},
                )

        save_training_state(
            output_dir / "training_state.pt",
            model=learner_model,
            optimizer=optimizer,
            config=config,
            update_id=update_id,
            opponent_pool=opponent_pool,
            observation_dim=observation_dim(),
            action_dim=action_dim(),
            best_winrate=best_winrate,
            best_score_diff=best_score_diff,
            rng=rng,
        )

        metrics_record = {
            "update": update_id,
            "config": config.to_dict() if update_id == 1 else None,
            "rollout": {
                "steps": update_stats.steps,
                "episodes": update_stats.episodes,
                "minibatches": update_stats.minibatches,
                "policy_loss": update_stats.policy_loss,
                "value_loss": update_stats.value_loss,
                "entropy": update_stats.entropy,
                "total_loss": update_stats.total_loss,
            },
            "training_matches": match_summaries,
            "evaluation": [asdict(summary) for summary in eval_summaries],
            "overall_eval_winrate": overall_winrate,
            "overall_eval_score_diff": overall_score_diff,
            "best_winrate": best_winrate,
            "best_score_diff": best_score_diff,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics_record, ensure_ascii=True))
            handle.write("\n")
        print(
            f"update={update_id} steps={update_stats.steps} episodes={update_stats.episodes} "
            f"loss={update_stats.total_loss:.4f} eval_winrate={overall_winrate:.3f} "
            f"eval_score_diff={overall_score_diff:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
