from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from poc.endgame import build_endgame_config
from poc.entities import Mars, Robot, Side, SourceState
from poc.external_events import EventType, ExternalEvent
from poc.game_state import GameState
from poc.semantic_map import build_default_semantic_map


@dataclass(slots=True)
class Scenario:
    name: str
    description: str
    game_state: GameState
    default_opponent_policy_name: str


def build_scenario(
    name: str,
    seed: int = 1,
    our_side: Side = Side.BLUE,
    opponent_policy_name: str | None = None,
    our_robot_speed: float | None = None,
    enemy_robot_speed: float | None = None,
) -> Scenario:
    variant_seed = max(1, int(seed))

    semantic_map = build_default_semantic_map().clone()
    enemy_side = our_side.opponent()

    blue_home = semantic_map.deposits[101].position
    yellow_home = semantic_map.deposits[201].position
    blue_robot = Robot(name="our_robot" if our_side is Side.BLUE else "enemy_robot", side=Side.BLUE, position=blue_home)
    yellow_robot = Robot(name="our_robot" if our_side is Side.YELLOW else "enemy_robot", side=Side.YELLOW, position=yellow_home)
    our_robot = blue_robot if our_side is Side.BLUE else yellow_robot
    enemy_robot = yellow_robot if our_side is Side.BLUE else blue_robot
    if our_robot_speed is not None:
        our_robot.speed = float(our_robot_speed)
    if enemy_robot_speed is not None:
        enemy_robot.speed = float(enemy_robot_speed)

    external_events: list[ExternalEvent] = []
    scenario_opponent_policy_name = "nearest_greedy"

    if name == "delayed_sources":
        semantic_map.sources[12].available_from_t = 18.0
        semantic_map.sources[12].state = SourceState.EMPTY
        semantic_map.sources[12].available_items = 0
        semantic_map.sources[22].available_from_t = 22.0
        semantic_map.sources[22].state = SourceState.EMPTY
        semantic_map.sources[22].available_items = 0
        external_events = [
            ExternalEvent(
                time=18.0,
                event_type=EventType.SET_SOURCE_AVAILABLE,
                target_id=12,
                origin="scripted",
                note="blue lower source becomes available",
                payload={"available_items": 4},
            ),
            ExternalEvent(
                time=22.0,
                event_type=EventType.SET_SOURCE_AVAILABLE,
                target_id=22,
                origin="scripted",
                note="yellow lower source becomes available",
                payload={"available_items": 4},
            ),
            ExternalEvent(
                time=55.0,
                event_type=EventType.SET_SOURCE_EMPTY,
                target_id=14 if our_side is Side.BLUE else 24,
                origin="scripted",
                note="one upper source becomes unavailable",
            ),
        ]
    elif name == "aggressive_enemy":
        scenario_opponent_policy_name = "aggressive"
    elif name == "thermo_first_enemy":
        scenario_opponent_policy_name = "thermo_first"
    elif name == "storage_first_enemy":
        scenario_opponent_policy_name = "storage_first"
    elif name == "home_safe_enemy":
        scenario_opponent_policy_name = "home_safe"
    elif name == "yellow_side_fixed_sequence_enemy":
        scenario_opponent_policy_name = "yellow_side_fixed_sequence"
    elif name == "stochastic_enemy":
        scenario_opponent_policy_name = f"stochastic_planner@{variant_seed}"
    elif name == "uniform_random_enemy":
        scenario_opponent_policy_name = f"uniform_random@{variant_seed}"
    elif name == "randomized_aggressive_enemy":
        scenario_opponent_policy_name = f"randomized_aggressive@{variant_seed}"
    elif name != "baseline":
        raise ValueError(f"Unknown scenario: {name}")

    game_state = GameState(
        t=0.0,
        T_end=100.0,
        our_side=our_side,
        our_robot=deepcopy(our_robot),
        enemy_robot=deepcopy(enemy_robot),
        sources=deepcopy(semantic_map.sources),
        deposits=deepcopy(semantic_map.deposits),
        thermometer=deepcopy(semantic_map.thermometer),
        external_events=sorted(external_events),
        semantic_map_name=semantic_map.name,
        field_size=semantic_map.field_size,
        endgame_by_side={
            Side.BLUE: build_endgame_config(Side.BLUE),
            Side.YELLOW: build_endgame_config(Side.YELLOW),
        },
        mars_by_side=_build_default_mars_map(semantic_map=semantic_map),
    )

    return Scenario(
        name=name,
        description=_scenario_description(name),
        game_state=game_state,
        default_opponent_policy_name=opponent_policy_name or scenario_opponent_policy_name,
    )


def _scenario_description(name: str) -> str:
    if name == "delayed_sources":
        return "Scenario with delayed source availability and scripted world events."
    if name == "aggressive_enemy":
        return "Scenario with an aggressive enemy policy."
    if name == "thermo_first_enemy":
        return "Scenario where the enemy prioritizes the thermometer."
    if name == "storage_first_enemy":
        return "Scenario where the enemy prioritizes storage deposits."
    if name == "home_safe_enemy":
        return "Scenario where the enemy prefers safe home deposits."
    if name == "yellow_side_fixed_sequence_enemy":
        return "Scenario where the enemy follows the YellowSideTree order with a fixed scripted sequence."
    if name == "stochastic_enemy":
        return "Scenario where the enemy samples from planner-ranked actions."
    if name == "uniform_random_enemy":
        return "Scenario where the enemy chooses uniformly among legal actions."
    if name == "randomized_aggressive_enemy":
        return "Scenario where the enemy uses a randomized aggressive style."
    return "Baseline scenario with a nearest-greedy enemy."


def _build_default_mars_map(*, semantic_map) -> dict[Side, tuple[Mars, ...]]:
    blue_targets = (17, 15, 16)
    yellow_targets = (27, 25, 26)
    blue_starts = ((1.34, 0.50), (1.40, 0.50), (1.46, 0.50))
    yellow_starts = tuple((-point[0], point[1]) for point in blue_starts)
    return {
        Side.BLUE: tuple(
            Mars(
                name=f"blue_mars_{index + 1}",
                side=Side.BLUE,
                pantry_id=deposit_id,
                start_position=start_position,
                target_position=tuple(semantic_map.deposits[deposit_id].position),
            )
            for index, (deposit_id, start_position) in enumerate(zip(blue_targets, blue_starts))
        ),
        Side.YELLOW: tuple(
            Mars(
                name=f"yellow_mars_{index + 1}",
                side=Side.YELLOW,
                pantry_id=deposit_id,
                start_position=start_position,
                target_position=tuple(semantic_map.deposits[deposit_id].position),
            )
            for index, (deposit_id, start_position) in enumerate(zip(yellow_targets, yellow_starts))
        ),
    }
