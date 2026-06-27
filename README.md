# Eurobot 2026 Planning POC

A lightweight strategy-planning sandbox for the Eurobot 2026 game.

The project models a two-robot match at the semantic-action level: a grid/A*
planner proposes legal actions, a simulator executes them as macro-actions, and
an optional masked PPO loop can train a policy over the planner-generated action
space. The code is intentionally small enough to inspect, run locally, and use
as a research prototype.

## What Is Included

- 2D match simulator with scoring, endgame return logic, and simplified Mars
  pantry scoring.
- Utility-based planner over an occupancy grid with route-specific waypoints.
- Scripted, stochastic, randomized, and fixed-sequence opponent policies.
- Masked discrete PPO self-play stack built on top of planner-generated action
  masks.
- Jupyter notebooks for trajectory inspection and RL result exploration.
- Focused unit tests for planner, simulator, scoring, RL infrastructure, and
  checkpoint compatibility behavior.

## Repository Layout

```text
poc/
  actions.py              # semantic action model
  planner.py              # utility-ranked planner
  simulator.py            # match execution and event history
  scoring.py              # scoring helpers
  rl_*.py                 # observation/action space, PPO, self-play, CLI tools
  data/                   # map and semantic map configuration
docs/
  planning_conditions.md  # planner and simulator behavior reference
  rl_system.md            # RL observation/action/training reference
notebooks/
  poc_results_overview.ipynb
  poc_rl_overview.ipynb
tests/
  test_rl_stack.py
```

## Quick Start

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[rl,notebook]"
```

Run a single simulated match:

```bash
python -m poc.main --scenario baseline --output runs/baseline.json
```

Run a small batch:

```bash
python -m poc.main --scenario aggressive_enemy --batch 5
```

Run the test suite:

```bash
python -m pytest -q
```

## Scenarios

`poc.main` currently supports:

- `baseline`
- `delayed_sources`
- `aggressive_enemy`
- `thermo_first_enemy`
- `storage_first_enemy`
- `home_safe_enemy`
- `yellow_side_fixed_sequence_enemy`
- `stochastic_enemy`
- `uniform_random_enemy`
- `randomized_aggressive_enemy`

## RL Commands

Train a PPO policy:

```bash
python -m poc.rl_train --output-dir runs/ppo_run
```

Evaluate a checkpoint:

```bash
python -m poc.rl_eval --checkpoint runs/ppo_run/latest.pt
```

Generate an offline dataset:

```bash
python -m poc.rl_dataset --output runs/rl_dataset.jsonl
```

PyTorch is kept in the `rl` extra because the base simulator and planner do not
need it.

## Current Modeling Scope

Implemented:

- semantic sources, storage zones, home zones, and thermometer actions;
- endgame sequence through `chill_point -> wait -> home`;
- two large robots with runtime separation handling;
- scenario-driven external events;
- simplified Mars entities for final pantry scoring;
- stochastic enemy speed jitter;
- masked discrete PPO over a fixed semantic action space.

Simplified by design:

- no ROS 2 integration;
- no low-level motion control;
- no rigid-body physics;
- Mars entities are not yet part of dynamic planner avoidance;
- several world-state values are approximate compared with a real robot stack.

## Documentation

- [Planning Conditions](docs/planning_conditions.md)
- [RL System](docs/rl_system.md)

## License

MIT. See [LICENSE](LICENSE).
