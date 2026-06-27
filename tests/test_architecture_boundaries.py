from __future__ import annotations

import ast
from pathlib import Path

from poc.control.controllers import build_scripted_controller
from poc.domain.entities import Side
from poc.rl.encoder import build_rl_observation
from poc.domain.rules import (
    deposit_can_accept_count,
    home_deposit_for_side,
    home_return_blocked,
    mars_has_pantry_credit,
    thermometer_lane_is_clear,
)
from poc.simulation.scenarios import build_scenario
from poc.simulation.simulator import Simulator


REPO_ROOT = Path(__file__).resolve().parents[1]


def _top_level_imports(module_path: str) -> set[str]:
    tree = ast.parse((REPO_ROOT / module_path).read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_core_architecture_boundaries_are_not_crossed() -> None:
    assert "poc.rl.infra" not in _top_level_imports("poc/simulation/simulator.py")
    assert "poc.io.result_export" not in _top_level_imports("poc/simulation/simulator.py")
    assert "poc.rl.policy_mapping" not in _top_level_imports("poc/simulation/simulator.py")
    assert "poc.rl.policy_mapping" not in _top_level_imports("poc/planning/planner.py")
    assert "poc.planning.planner" not in _top_level_imports("poc/control/opponent_policy.py")
    assert "poc.control.opponent_policy" not in _top_level_imports("poc/simulation/scenarios.py")
    assert "poc.rl.selfplay" not in _top_level_imports("poc/rl/encoder.py")
    assert "poc.rl.model" not in _top_level_imports("poc/rl/encoder.py")
    assert "poc.rl.ppo" not in _top_level_imports("poc/rl/encoder.py")
    assert "poc.rl.transitions" not in _top_level_imports("poc/rl/action_space.py")
    assert "poc.rl.workers" not in _top_level_imports("poc/rl/selectors.py")
    assert "poc.rl.selfplay" not in _top_level_imports("poc/rl/workers.py")


def test_rules_cover_thermometer_lane_home_capacity_and_mars_credit() -> None:
    scenario = build_scenario("baseline", seed=1)
    state = scenario.game_state

    assert not thermometer_lane_is_clear(state, Side.BLUE)
    state.sources[13].available_items = 0
    state.sources[13].state = state.sources[13].state.EMPTY
    state.deposits[10].clear()
    state.deposits[16].clear()
    assert thermometer_lane_is_clear(state, Side.BLUE)

    home = home_deposit_for_side(state, Side.BLUE)
    assert home is not None
    assert not home_return_blocked(state, Side.BLUE)
    assert deposit_can_accept_count(home, Side.BLUE, robot_load=4, count=1)
    assert not deposit_can_accept_count(home, Side.BLUE, robot_load=0, count=1)
    home.add_items(Side.BLUE, 1)
    assert home_return_blocked(state, Side.BLUE)

    assert mars_has_pantry_credit(arrived=True, collided=False)
    assert not mars_has_pantry_credit(arrived=False, collided=False)
    assert not mars_has_pantry_credit(arrived=True, collided=True)


def test_simulator_builds_decision_observations_for_decision_points() -> None:
    scenario = build_scenario("baseline", seed=1)
    result = Simulator(
        state=scenario.game_state,
        scenario_name=scenario.name,
        opponent_controller=build_scripted_controller(scenario.default_opponent_policy_name),
        dt=1.0,
    ).run()

    assert result.decision_log
    assert {entry.side for entry in result.decision_log} == {Side.BLUE, Side.YELLOW}
    assert all(entry.observation.ranked_actions for entry in result.decision_log)
    assert any(entry.observation.previous_state is not None for entry in result.decision_log)

    blue_entry = next(entry for entry in result.decision_log if entry.side is Side.BLUE)
    yellow_entry = next(entry for entry in result.decision_log if entry.side is Side.YELLOW)
    blue_observation = build_rl_observation(blue_entry.observation)
    yellow_observation = build_rl_observation(yellow_entry.observation)

    assert blue_observation.perspective == "blue"
    assert yellow_observation.perspective == "yellow"
    assert "enemy_speed_norm" in blue_observation.global_features
    assert "time_since_thermometer_state_change_norm" in blue_observation.global_features
