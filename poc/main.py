from __future__ import annotations

import argparse
from pathlib import Path

from poc.metrics import summarize_batch
from poc.planner import UtilityPlanner
from poc.scenarios import build_scenario
from poc.simulator import Simulator, save_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Eurobot 2026 planning POC simulator.")
    parser.add_argument("--scenario", default="baseline", choices=["baseline", "delayed_sources", "aggressive_enemy", "thermo_first_enemy"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("runs") / "latest_match.json")
    parser.add_argument("--batch", type=int, default=1, help="Run the scenario multiple times with incrementing seeds.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    planner = UtilityPlanner()
    results = []
    for seed_offset in range(args.batch):
        scenario = build_scenario(args.scenario, seed=args.seed + seed_offset)
        simulator = Simulator(
            state=scenario.game_state,
            scenario_name=scenario.name,
            opponent_policy=scenario.opponent_policy,
            planner=planner,
            dt=args.dt,
        )
        results.append(simulator.run())

    if args.batch == 1:
        save_result(results[0], args.output)
        summary = results[0].summary
        print(f"scenario={summary['scenario']} our_score={summary['our_score']} enemy_score={summary['enemy_score']}")
        print(f"return_home={summary['successful_return_home']} replans={summary['replan_events']} output={args.output}")
        return 0

    summary = summarize_batch(results)
    print(f"batch_runs={args.batch} scenario={args.scenario}")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
