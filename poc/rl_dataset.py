from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from poc.entities import Side
from poc.planner import UtilityPlanner
from poc.rl_infra import transition_to_record
from poc.scenarios import build_scenario
from poc.simulator import Simulator

DEFAULT_SCENARIOS = (
    "baseline",
    "delayed_sources",
    "aggressive_enemy",
    "thermo_first_enemy",
    "storage_first_enemy",
    "home_safe_enemy",
)
DEFAULT_OPPONENT_POLICIES = ("nearest_greedy", "aggressive", "thermo_first", "storage_first", "home_safe")


@dataclass(frozen=True, slots=True)
class DatasetRunSpec:
    scenario_name: str
    seed: int
    opponent_policy_name: str
    our_side: Side
    dt: float

    @property
    def match_id(self) -> str:
        return f"{self.scenario_name}__seed{self.seed}__{self.our_side.value}__vs_{self.opponent_policy_name}"


@dataclass(frozen=True, slots=True)
class DatasetBuildSummary:
    output_path: str
    manifest_path: str
    matches: int
    transitions: int
    scenarios: tuple[str, ...]
    seeds: tuple[int, ...]
    opponent_policies: tuple[str, ...]
    our_side: str
    dt: float


def build_dataset_run_specs(
    scenarios: tuple[str, ...],
    seeds: tuple[int, ...],
    opponent_policies: tuple[str, ...],
    our_side: Side,
    dt: float,
) -> list[DatasetRunSpec]:
    return [
        DatasetRunSpec(
            scenario_name=scenario_name,
            seed=seed,
            opponent_policy_name=opponent_policy_name,
            our_side=our_side,
            dt=dt,
        )
        for scenario_name in scenarios
        for seed in seeds
        for opponent_policy_name in opponent_policies
    ]


def generate_rl_dataset(
    output_path: str | Path,
    *,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
    seeds: tuple[int, ...] = (1,),
    opponent_policies: tuple[str, ...] = DEFAULT_OPPONENT_POLICIES,
    our_side: Side = Side.BLUE,
    dt: float = 0.5,
    manifest_path: str | Path | None = None,
) -> DatasetBuildSummary:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest = Path(manifest_path) if manifest_path is not None else output.with_suffix(".manifest.json")
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)

    planner = UtilityPlanner()
    run_specs = build_dataset_run_specs(
        scenarios=scenarios,
        seeds=seeds,
        opponent_policies=opponent_policies,
        our_side=our_side,
        dt=dt,
    )

    transition_count = 0
    manifest_runs: list[dict[str, object]] = []
    with output.open("w", encoding="utf-8") as handle:
        for run_index, spec in enumerate(run_specs):
            scenario = build_scenario(
                spec.scenario_name,
                seed=spec.seed,
                our_side=spec.our_side,
                opponent_policy_name=spec.opponent_policy_name,
            )
            simulator = Simulator(
                state=scenario.game_state,
                scenario_name=scenario.name,
                opponent_policy=scenario.opponent_policy,
                planner=planner,
                dt=spec.dt,
            )
            result = simulator.run()
            run_transition_count = 0
            for transition_index, transition in enumerate(result.rl_transitions):
                record = transition_to_record(transition)
                record["match"] = {
                    "match_id": spec.match_id,
                    "run_index": run_index,
                    "transition_index": transition_index,
                    "scenario": spec.scenario_name,
                    "seed": spec.seed,
                    "our_side": spec.our_side.value,
                    "opponent_policy": spec.opponent_policy_name,
                    "dt": spec.dt,
                    "our_score_final": result.summary["our_score"],
                    "enemy_score_final": result.summary["enemy_score"],
                    "score_diff_final": result.summary["score_diff"],
                    "win": result.summary["win"],
                }
                handle.write(json.dumps(record, ensure_ascii=True))
                handle.write("\n")
                transition_count += 1
                run_transition_count += 1
            manifest_runs.append(
                {
                    "match_id": spec.match_id,
                    "scenario": spec.scenario_name,
                    "seed": spec.seed,
                    "our_side": spec.our_side.value,
                    "opponent_policy": spec.opponent_policy_name,
                    "dt": spec.dt,
                    "transitions": run_transition_count,
                    "summary": result.summary,
                }
            )

    summary = DatasetBuildSummary(
        output_path=str(output),
        manifest_path=str(resolved_manifest),
        matches=len(run_specs),
        transitions=transition_count,
        scenarios=scenarios,
        seeds=seeds,
        opponent_policies=opponent_policies,
        our_side=our_side.value,
        dt=dt,
    )
    resolved_manifest.write_text(
        json.dumps(
            {
                "summary": {
                    "output_path": summary.output_path,
                    "manifest_path": summary.manifest_path,
                    "matches": summary.matches,
                    "transitions": summary.transitions,
                    "scenarios": list(summary.scenarios),
                    "seeds": list(summary.seeds),
                    "opponent_policies": list(summary.opponent_policies),
                    "our_side": summary.our_side,
                    "dt": summary.dt,
                },
                "runs": manifest_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an offline RL dataset from simulator rollouts.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        choices=list(DEFAULT_SCENARIOS),
        help="Scenario names to include in the Cartesian product.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1],
        help="Seeds to include in the Cartesian product.",
    )
    parser.add_argument(
        "--opponent-policies",
        nargs="+",
        default=list(DEFAULT_OPPONENT_POLICIES),
        choices=list(DEFAULT_OPPONENT_POLICIES),
        help="Opponent policies to include in the Cartesian product.",
    )
    parser.add_argument("--our-side", choices=[Side.BLUE.value, Side.YELLOW.value], default=Side.BLUE.value)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("runs") / "rl_dataset.jsonl")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = generate_rl_dataset(
        output_path=args.output,
        scenarios=tuple(args.scenarios),
        seeds=tuple(args.seeds),
        opponent_policies=tuple(args.opponent_policies),
        our_side=Side(args.our_side),
        dt=args.dt,
        manifest_path=args.manifest,
    )
    print(f"output={summary.output_path}")
    print(f"manifest={summary.manifest_path}")
    print(f"matches={summary.matches}")
    print(f"transitions={summary.transitions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
