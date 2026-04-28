from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from poc.endgame import build_endgame_config
from poc.entities import Robot, Side, SourceState
from poc.external_events import EventType, ExternalEvent
from poc.game_state import GameState
from poc.opponent_policy import OpponentPolicy, build_opponent_policy
from poc.semantic_map import build_default_semantic_map


@dataclass(slots=True)
class Scenario:
    name: str
    description: str
    game_state: GameState
    opponent_policy: OpponentPolicy


def build_scenario(
    name: str,
    seed: int = 1,
    our_side: Side = Side.BLUE,
    opponent_policy_name: str | None = None,
) -> Scenario:
    del seed  # scaffold placeholder for reproducible stochastic extensions

    semantic_map = build_default_semantic_map().clone()
    enemy_side = our_side.opponent()

    blue_home = semantic_map.deposits[101].position
    yellow_home = semantic_map.deposits[201].position
    blue_robot = Robot(name="our_robot" if our_side is Side.BLUE else "enemy_robot", side=Side.BLUE, position=blue_home)
    yellow_robot = Robot(name="our_robot" if our_side is Side.YELLOW else "enemy_robot", side=Side.YELLOW, position=yellow_home)
    our_robot = blue_robot if our_side is Side.BLUE else yellow_robot
    enemy_robot = yellow_robot if our_side is Side.BLUE else blue_robot

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
    )

    return Scenario(
        name=name,
        description=_scenario_description(name),
        game_state=game_state,
        opponent_policy=build_opponent_policy(opponent_policy_name or scenario_opponent_policy_name),
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
    return "Baseline scenario with a nearest-greedy enemy."
