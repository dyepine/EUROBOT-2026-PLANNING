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

from poc.rl_checkpoint import (
    OpponentPool,
    load_checkpoint,
    load_training_state,
    normalize_torch_cuda_rng_state_all,
    normalize_torch_rng_state,
    save_policy_checkpoint,
    save_training_state,
)
from poc.rl_config import (
    DEFAULT_EVAL_OPPONENTS,
    DEFAULT_TRAINING_SCENARIOS,
    DEFAULT_TRAINING_SCRIPTED_OPPONENTS,
    SelfPlayConfig,
    selfplay_config_from_dict,
)
from poc.rl_ppo import ppo_update
from poc.rl_model import load_compatible_state_dict
from poc.rl_selfplay import (
    EvaluationSummary,
    OpponentSpec,
    PersistentWorkerPools,
    action_dim,
    build_model,
    build_rollout_batch,
    classify_opponent_kind,
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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-from-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
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
    parser.add_argument("--thermometer-reward-bonus", type=float, default=3.0)
    parser.add_argument("--terminal-win-bonus", type=float, default=2.0)
    parser.add_argument("--terminal-draw-bonus", type=float, default=0.0)
    parser.add_argument("--terminal-loss-bonus", type=float, default=-2.0)
    parser.add_argument("--enemy-speed-jitter-fraction", type=float, default=0.30)
    parser.add_argument("--enemy-velocity-noise-std-mps", type=float, default=0.01)
    parser.add_argument("--enemy-velocity-self-motion-leak-fraction", type=float, default=0.0)
    parser.add_argument("--enemy-velocity-self-motion-leak-duration-s", type=float, default=0.0)
    parser.add_argument("--opponent-pool-size", type=int, default=8)
    parser.add_argument("--checkpoint-every-updates", type=int, default=1)
    parser.add_argument("--eval-every-updates", type=int, default=5)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[256, 256, 256])
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--training-scenarios", nargs="+", default=list(DEFAULT_TRAINING_SCENARIOS))
    parser.add_argument("--training-scripted-opponents", nargs="+", default=list(DEFAULT_TRAINING_SCRIPTED_OPPONENTS))
    parser.add_argument("--training-scripted-fraction", type=float, default=0.30)
    parser.add_argument("--eval-scenarios", nargs="+", default=["baseline", "delayed_sources"])
    parser.add_argument("--eval-opponents", nargs="+", default=list(DEFAULT_EVAL_OPPONENTS))
    parser.add_argument("--eval-matches-per-opponent", type=int, default=4)
    return parser


def build_config(args: argparse.Namespace) -> SelfPlayConfig:
    from poc.entities import Side

    return SelfPlayConfig(
        seed=args.seed,
        device=args.device or "cpu",
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
        thermometer_reward_bonus=args.thermometer_reward_bonus,
        terminal_win_bonus=args.terminal_win_bonus,
        terminal_draw_bonus=args.terminal_draw_bonus,
        terminal_loss_bonus=args.terminal_loss_bonus,
        enemy_speed_jitter_fraction=args.enemy_speed_jitter_fraction,
        enemy_velocity_noise_std_mps=args.enemy_velocity_noise_std_mps,
        enemy_velocity_self_motion_leak_fraction=args.enemy_velocity_self_motion_leak_fraction,
        enemy_velocity_self_motion_leak_duration_s=args.enemy_velocity_self_motion_leak_duration_s,
        opponent_pool_size=args.opponent_pool_size,
        checkpoint_every_updates=args.checkpoint_every_updates,
        eval_every_updates=args.eval_every_updates,
        updates=args.updates if args.updates is not None else 100,
        hidden_sizes=tuple(args.hidden_sizes),
        rollout_workers=args.rollout_workers if args.rollout_workers is not None else 1,
        eval_workers=args.eval_workers if args.eval_workers is not None else 1,
        training_scenarios=tuple(args.training_scenarios),
        training_scripted_opponents=tuple(args.training_scripted_opponents),
        training_scripted_fraction=args.training_scripted_fraction,
        eval_scenarios=tuple(args.eval_scenarios),
        eval_opponents=tuple(args.eval_opponents),
        eval_matches_per_opponent=args.eval_matches_per_opponent,
    )


def resolve_output_dir(args: argparse.Namespace, resume_path: Path | None) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if resume_path is not None:
        return resume_path.resolve().parent
    return Path("runs") / "ppo_selfplay"


def resolve_start_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    resume_path = args.resume.resolve() if args.resume is not None else None
    init_checkpoint_path = args.init_from_checkpoint.resolve() if args.init_from_checkpoint is not None else None
    if resume_path is not None and init_checkpoint_path is not None:
        raise ValueError("Use either --resume or --init-from-checkpoint, not both.")
    return resume_path, init_checkpoint_path


def config_for_resume(
    args: argparse.Namespace,
    saved_config: SelfPlayConfig,
) -> SelfPlayConfig:
    payload = saved_config.to_dict()
    payload.pop("ppo", None)
    if args.device is not None:
        payload["device"] = args.device
    if args.updates is not None:
        payload["updates"] = args.updates
    if args.rollout_workers is not None:
        payload["rollout_workers"] = args.rollout_workers
    if args.eval_workers is not None:
        payload["eval_workers"] = args.eval_workers
    return selfplay_config_from_dict(payload)


def build_eval_opponents(config: SelfPlayConfig, pool: OpponentPool) -> list[OpponentSpec]:
    opponents = [
        OpponentSpec(
            name=name,
            selector=None,
            opponent_policy_name=name,
            kind=classify_opponent_kind(name),
        )
        for name in config.eval_opponents
    ]
    for snapshot in pool.evaluation_snapshots(limit=2):
        selector = selector_from_snapshot(snapshot, config=config, greedy=True) if config.eval_workers <= 1 else None
        opponents.append(
            OpponentSpec(
                name=snapshot.name,
                selector=selector,
                kind="self_play",
                state_dict=snapshot.state_dict,
                greedy=True,
            )
        )
    return opponents


def aggregate_eval_metrics(summaries: list[EvaluationSummary]) -> tuple[float, float, float, float]:
    if not summaries:
        return 0.0, 0.0, 0.0, 0.0
    total_matches = sum(summary.matches for summary in summaries)
    if total_matches <= 0:
        return 0.0, 0.0, 0.0, 0.0
    winrate = sum(summary.winrate * summary.matches for summary in summaries) / total_matches
    score_diff = sum(summary.mean_score_diff * summary.matches for summary in summaries) / total_matches
    p10_score_diff = sum(summary.p10_score_diff * summary.matches for summary in summaries) / total_matches
    min_score_diff = min(summary.min_score_diff for summary in summaries)
    return winrate, score_diff, p10_score_diff, min_score_diff


def aggregate_eval_metrics_for_opponents(
    summaries: list[EvaluationSummary],
    opponent_names: set[str],
) -> tuple[float, float, float, float]:
    filtered = [summary for summary in summaries if summary.opponent in opponent_names]
    return aggregate_eval_metrics(filtered)


def aggregate_eval_metrics_for_kinds(
    summaries: list[EvaluationSummary],
    kinds: set[str],
) -> tuple[float, float, float, float]:
    filtered = [summary for summary in summaries if summary.kind in kinds]
    return aggregate_eval_metrics(filtered)


def main(argv: list[str] | None = None) -> int:
    _require_torch()
    args = build_parser().parse_args(argv)
    resume_path, init_checkpoint_path = resolve_start_paths(args)
    config = build_config(args) if resume_path is None else None
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    if init_checkpoint_path is not None and not init_checkpoint_path.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {init_checkpoint_path}")
    output_dir = resolve_output_dir(args, resume_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    if resume_path is None:
        assert config is not None
        rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
    else:
        saved_header = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved_config = selfplay_config_from_dict(dict(saved_header["config"]))
        config = config_for_resume(args, saved_config)
        rng = random.Random(config.seed)
        torch.manual_seed(config.seed)

    planner = UtilityPlanner()
    learner_model = build_model(config).to(torch.device(config.device))
    if init_checkpoint_path is not None:
        init_payload = load_checkpoint(init_checkpoint_path, map_location=config.device)
        load_compatible_state_dict(learner_model, init_payload["model_state"])
    optimizer = torch.optim.Adam(learner_model.parameters(), lr=config.learning_rate)
    opponent_pool = OpponentPool(config.opponent_pool_size)

    best_winrate = float("-inf")
    best_winrate_tiebreak = float("-inf")
    best_score_diff = float("-inf")
    start_update = 1
    if resume_path is not None:
        payload = load_training_state(
            resume_path,
            model=learner_model,
            optimizer=optimizer,
            map_location=config.device,
        )
        pool_state = payload.get("opponent_pool")
        if isinstance(pool_state, dict):
            opponent_pool = OpponentPool.from_state(pool_state, max_size=config.opponent_pool_size)
        best_winrate = float(payload.get("best_winrate", float("-inf")))
        best_winrate_tiebreak = float(payload.get("best_winrate_tiebreak", payload.get("best_score_diff", float("-inf"))))
        best_score_diff = float(payload.get("best_score_diff", float("-inf")))
        python_random_state = payload.get("python_random_state")
        if python_random_state is not None:
            rng.setstate(python_random_state)
        torch_rng_state = payload.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(normalize_torch_rng_state(torch_rng_state))
        cuda_rng_state_all = payload.get("torch_cuda_rng_state_all")
        if cuda_rng_state_all is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(normalize_torch_cuda_rng_state_all(cuda_rng_state_all))
        start_update = int(payload.get("update_id", 0)) + 1
        print(f"resuming_from={resume_path} start_update={start_update} total_updates={config.updates} device={config.device}")
    elif init_checkpoint_path is not None:
        print(f"init_from_checkpoint={init_checkpoint_path} start_update=1 total_updates={config.updates} device={config.device}")
    if start_update > config.updates:
        print(
            f"resume checkpoint already reached update {start_update - 1}, nothing to do for updates={config.updates}",
            flush=True,
        )
        return 0

    print(
        f"training_start start_update={start_update} total_updates={config.updates} "
        f"device={config.device} rollout_workers={config.rollout_workers} eval_workers={config.eval_workers}",
        flush=True,
    )
    worker_pools = PersistentWorkerPools(
        rollout_workers=config.rollout_workers,
        eval_workers=config.eval_workers,
    )

    for update_id in range(start_update, config.updates + 1):
        print(f"[update {update_id}] start", flush=True)
        rollout_batch, match_summaries = build_rollout_batch(
            config=config,
            learner_model=learner_model,
            opponent_pool=opponent_pool,
            planner=planner,
            rng=rng,
            update_id=update_id,
            executor=worker_pools.rollout_executor(),
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
        overall_eval_p10_score_diff = 0.0
        overall_eval_min_score_diff = 0.0
        scripted_eval_winrate = 0.0
        scripted_eval_score_diff = 0.0
        scripted_eval_p10_score_diff = 0.0
        scripted_eval_min_score_diff = 0.0
        randomized_eval_winrate = 0.0
        randomized_eval_score_diff = 0.0
        randomized_eval_p10_score_diff = 0.0
        randomized_eval_min_score_diff = 0.0
        self_play_eval_winrate = 0.0
        self_play_eval_score_diff = 0.0
        self_play_eval_p10_score_diff = 0.0
        self_play_eval_min_score_diff = 0.0
        robust_eval_winrate = 0.0
        robust_eval_score_diff = 0.0
        robust_eval_p10_score_diff = 0.0
        robust_eval_min_score_diff = 0.0
        if update_id % config.eval_every_updates == 0:
            eval_summaries = evaluate_policy(
                config=config,
                learner_model=learner_model,
                planner=planner,
                rng=rng,
                opponent_specs=build_eval_opponents(config, opponent_pool),
                executor=worker_pools.eval_executor(),
            )
            (
                overall_winrate,
                overall_score_diff,
                overall_eval_p10_score_diff,
                overall_eval_min_score_diff,
            ) = aggregate_eval_metrics(eval_summaries)
            (
                scripted_eval_winrate,
                scripted_eval_score_diff,
                scripted_eval_p10_score_diff,
                scripted_eval_min_score_diff,
            ) = aggregate_eval_metrics_for_kinds(
                eval_summaries,
                {"scripted"},
            )
            (
                randomized_eval_winrate,
                randomized_eval_score_diff,
                randomized_eval_p10_score_diff,
                randomized_eval_min_score_diff,
            ) = aggregate_eval_metrics_for_kinds(
                eval_summaries,
                {"randomized", "random"},
            )
            (
                self_play_eval_winrate,
                self_play_eval_score_diff,
                self_play_eval_p10_score_diff,
                self_play_eval_min_score_diff,
            ) = aggregate_eval_metrics_for_kinds(
                eval_summaries,
                {"self_play"},
            )
            (
                robust_eval_winrate,
                robust_eval_score_diff,
                robust_eval_p10_score_diff,
                robust_eval_min_score_diff,
            ) = aggregate_eval_metrics_for_kinds(
                eval_summaries,
                {"randomized", "random", "self_play"},
            )
            is_best_winrate = (
                scripted_eval_winrate > best_winrate
                or scripted_eval_winrate == best_winrate and scripted_eval_p10_score_diff > best_winrate_tiebreak
            )
            if is_best_winrate:
                best_winrate = scripted_eval_winrate
                best_winrate_tiebreak = scripted_eval_p10_score_diff
                save_policy_checkpoint(
                    output_dir / "best_winrate.pt",
                    model=learner_model,
                    config=config,
                    update_id=update_id,
                    observation_dim=observation_dim(),
                    action_dim=action_dim(),
                    metadata={
                        "metric": "scripted_winrate",
                        "scripted_winrate": scripted_eval_winrate,
                        "scripted_score_diff": scripted_eval_score_diff,
                        "scripted_p10_score_diff": scripted_eval_p10_score_diff,
                        "scripted_min_score_diff": scripted_eval_min_score_diff,
                        "randomized_winrate": randomized_eval_winrate,
                        "randomized_score_diff": randomized_eval_score_diff,
                        "randomized_p10_score_diff": randomized_eval_p10_score_diff,
                        "randomized_min_score_diff": randomized_eval_min_score_diff,
                        "self_play_winrate": self_play_eval_winrate,
                        "self_play_score_diff": self_play_eval_score_diff,
                        "self_play_p10_score_diff": self_play_eval_p10_score_diff,
                        "self_play_min_score_diff": self_play_eval_min_score_diff,
                        "robust_winrate": robust_eval_winrate,
                        "robust_score_diff": robust_eval_score_diff,
                        "robust_p10_score_diff": robust_eval_p10_score_diff,
                        "robust_min_score_diff": robust_eval_min_score_diff,
                        "overall_winrate": overall_winrate,
                        "overall_score_diff": overall_score_diff,
                        "overall_p10_score_diff": overall_eval_p10_score_diff,
                        "overall_min_score_diff": overall_eval_min_score_diff,
                    },
                )
            if scripted_eval_score_diff > best_score_diff:
                best_score_diff = scripted_eval_score_diff
                save_policy_checkpoint(
                    output_dir / "best_score_diff.pt",
                    model=learner_model,
                    config=config,
                    update_id=update_id,
                    observation_dim=observation_dim(),
                    action_dim=action_dim(),
                    metadata={
                        "metric": "scripted_score_diff",
                        "scripted_winrate": scripted_eval_winrate,
                        "scripted_score_diff": scripted_eval_score_diff,
                        "scripted_p10_score_diff": scripted_eval_p10_score_diff,
                        "scripted_min_score_diff": scripted_eval_min_score_diff,
                        "randomized_winrate": randomized_eval_winrate,
                        "randomized_score_diff": randomized_eval_score_diff,
                        "randomized_p10_score_diff": randomized_eval_p10_score_diff,
                        "randomized_min_score_diff": randomized_eval_min_score_diff,
                        "self_play_winrate": self_play_eval_winrate,
                        "self_play_score_diff": self_play_eval_score_diff,
                        "self_play_p10_score_diff": self_play_eval_p10_score_diff,
                        "self_play_min_score_diff": self_play_eval_min_score_diff,
                        "robust_winrate": robust_eval_winrate,
                        "robust_score_diff": robust_eval_score_diff,
                        "robust_p10_score_diff": robust_eval_p10_score_diff,
                        "robust_min_score_diff": robust_eval_min_score_diff,
                        "overall_winrate": overall_winrate,
                        "overall_score_diff": overall_score_diff,
                        "overall_p10_score_diff": overall_eval_p10_score_diff,
                        "overall_min_score_diff": overall_eval_min_score_diff,
                    },
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
            best_winrate_tiebreak=best_winrate_tiebreak,
            best_score_diff=best_score_diff,
            rng=rng,
        )

        metrics_record = {
            "update": update_id,
            "config": config.to_dict() if update_id == start_update else None,
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
            "overall_eval_p10_score_diff": overall_eval_p10_score_diff,
            "overall_eval_min_score_diff": overall_eval_min_score_diff,
            "scripted_eval_winrate": scripted_eval_winrate,
            "scripted_eval_score_diff": scripted_eval_score_diff,
            "scripted_eval_p10_score_diff": scripted_eval_p10_score_diff,
            "scripted_eval_min_score_diff": scripted_eval_min_score_diff,
            "randomized_eval_winrate": randomized_eval_winrate,
            "randomized_eval_score_diff": randomized_eval_score_diff,
            "randomized_eval_p10_score_diff": randomized_eval_p10_score_diff,
            "randomized_eval_min_score_diff": randomized_eval_min_score_diff,
            "self_play_eval_winrate": self_play_eval_winrate,
            "self_play_eval_score_diff": self_play_eval_score_diff,
            "self_play_eval_p10_score_diff": self_play_eval_p10_score_diff,
            "self_play_eval_min_score_diff": self_play_eval_min_score_diff,
            "robust_eval_winrate": robust_eval_winrate,
            "robust_eval_score_diff": robust_eval_score_diff,
            "robust_eval_p10_score_diff": robust_eval_p10_score_diff,
            "robust_eval_min_score_diff": robust_eval_min_score_diff,
            "best_winrate": best_winrate,
            "best_score_diff": best_score_diff,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics_record, ensure_ascii=True))
            handle.write("\n")
        print(
            f"update={update_id} steps={update_stats.steps} episodes={update_stats.episodes} "
            f"loss={update_stats.total_loss:.4f} scripted_eval_winrate={scripted_eval_winrate:.3f} "
            f"scripted_eval_score_diff={scripted_eval_score_diff:.3f} "
            f"scripted_eval_p10_score_diff={scripted_eval_p10_score_diff:.3f} "
            f"randomized_eval_winrate={randomized_eval_winrate:.3f} "
            f"robust_eval_winrate={robust_eval_winrate:.3f} "
            f"overall_eval_winrate={overall_winrate:.3f} overall_eval_score_diff={overall_score_diff:.3f}",
            flush=True,
        )

    worker_pools.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
