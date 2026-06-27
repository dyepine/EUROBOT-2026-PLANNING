from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

from poc.planner import UtilityPlanner
from poc.rl_checkpoint import OpponentPool, load_checkpoint
from poc.rl_config import DEFAULT_EVAL_OPPONENTS, selfplay_config_from_dict
from poc.rl_model import load_compatible_state_dict
from poc.rl_selfplay import OpponentSpec, build_model, classify_opponent_kind, evaluate_policy, selector_from_snapshot
from poc.torch_compat import require_torch, torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a self-play PPO checkpoint for the Eurobot POC.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-state", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--matches-per-opponent", type=int, default=None)
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_EVAL_OPPONENTS))
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    require_torch(torch)
    args = build_parser().parse_args(argv)
    payload = load_checkpoint(args.checkpoint)
    config = selfplay_config_from_dict(dict(payload["config"]))
    if args.device is not None:
        config_payload = config.to_dict()
        config_payload["ppo"]["device"] = args.device
        config = selfplay_config_from_dict(config_payload)
    if args.matches_per_opponent is not None:
        config = selfplay_config_from_dict({**config.to_dict(), "eval_matches_per_opponent": args.matches_per_opponent})

    planner = UtilityPlanner()
    model = build_model(config).to(torch.device(config.ppo.device))
    load_compatible_state_dict(model, payload["model_state"])

    opponents = [
        OpponentSpec(
            name=name,
            selector=None,
            opponent_policy_name=name,
            kind=classify_opponent_kind(name),
        )
        for name in args.opponents
    ]
    training_state_path = args.training_state or args.checkpoint.with_name("training_state.pt")
    if training_state_path.exists():
        state_payload = load_checkpoint(training_state_path)
        pool_state = state_payload.get("opponent_pool")
        if isinstance(pool_state, dict):
            pool = OpponentPool.from_state(pool_state, max_size=config.ppo.opponent_pool_size)
            for snapshot in pool.evaluation_snapshots(limit=2):
                opponents.append(
                    OpponentSpec(
                        name=snapshot.name,
                        selector=selector_from_snapshot(snapshot, config=config, greedy=True),
                        kind="self_play",
                    )
                )

    summaries = evaluate_policy(
        config=config,
        learner_model=model,
        planner=planner,
        rng=random.Random(config.ppo.seed),
        opponent_specs=opponents,
    )
    output_payload = {
        "checkpoint": str(args.checkpoint),
        "config": config.to_dict(),
        "summaries": [asdict(summary) for summary in summaries],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    for summary in summaries:
        print(
            f"opponent={summary.opponent} kind={summary.kind} matches={summary.matches} winrate={summary.winrate:.3f} "
            f"score_diff={summary.mean_score_diff:.3f} median={summary.median_score_diff:.3f} "
            f"p10={summary.p10_score_diff:.3f} min={summary.min_score_diff:.3f} "
            f"invalid_actions={summary.invalid_action_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
