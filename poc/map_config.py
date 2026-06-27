from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path

import yaml

from poc.entities import DepositPoint, DepositType, RouteOption, Side, SourcePoint, Thermometer
from poc.geometry import Vec2


FIELD_WIDTH = 3.0
FIELD_HEIGHT = 2.0
DEFAULT_SEMANTIC_MAP_CONFIG_PATH = Path(__file__).resolve().parent / "data" / "semantic_map.yaml"


@dataclass(slots=True)
class SemanticMap:
    name: str
    field_size: tuple[float, float]
    sources: dict[int, SourcePoint]
    deposits: dict[int, DepositPoint]
    thermometer: Thermometer

    def clone(self) -> "SemanticMap":
        return deepcopy(self)


@dataclass(frozen=True, slots=True)
class RouteConfig:
    name: str
    waypoints: tuple[Vec2, ...]
    blocked_by_sources: tuple[int, ...] = ()
    axis: str | None = None
    release_target_footprint_after_waypoint: int | None = None

    def build(self) -> RouteOption:
        return RouteOption(
            name=self.name,
            waypoints=self.waypoints,
            blocked_by_sources=self.blocked_by_sources,
            axis=self.axis,
            release_target_footprint_after_waypoint=self.release_target_footprint_after_waypoint,
        )

    def mirrored(self, suffix: str = "_mirror") -> "RouteConfig":
        return RouteConfig(
            name=f"{self.name}{suffix}",
            waypoints=tuple((-point[0], point[1]) for point in self.waypoints),
            blocked_by_sources=tuple(_mirror_source_id(source_id) for source_id in self.blocked_by_sources),
            axis=self.axis,
            release_target_footprint_after_waypoint=self.release_target_footprint_after_waypoint,
        )


@dataclass(frozen=True, slots=True)
class SourceZoneConfig:
    semantic_id: int
    position: Vec2
    label: str
    map_obstacle_id: str | None = None
    collect_routes: tuple[RouteConfig, ...] = ()
    available_items: int = 4
    available_from_t: float = 0.0
    footprint_enabled: bool = True
    map_footprint_enabled: bool = True

    def build(self) -> SourcePoint:
        return SourcePoint(
            semantic_id=self.semantic_id,
            position=self.position,
            map_obstacle_id=self.map_obstacle_id,
            map_footprint_enabled=self.map_footprint_enabled,
            collect_routes=tuple(route.build() for route in self.collect_routes),
            available_items=self.available_items,
            available_from_t=self.available_from_t,
            footprint_enabled=self.footprint_enabled,
            label=self.label,
        )


@dataclass(frozen=True, slots=True)
class DepositZoneConfig:
    semantic_id: int
    position: Vec2
    kind: DepositType
    label: str
    owner: Side | None = None
    map_obstacle_id: str | None = None
    deposit_routes: tuple[RouteConfig, ...] = ()
    approach_ring_radius: float = 0.0
    approach_ring_samples: int = 16
    attack_routes_by_side: dict[Side, tuple[RouteConfig, ...]] = field(default_factory=dict)
    protected_for: Side | None = None
    map_footprint_enabled: bool = True
    footprint_enabled: bool = True

    def build(self) -> DepositPoint:
        return DepositPoint(
            semantic_id=self.semantic_id,
            position=self.position,
            kind=self.kind,
            owner=self.owner,
            map_obstacle_id=self.map_obstacle_id,
            map_footprint_enabled=self.map_footprint_enabled,
            deposit_routes=tuple(route.build() for route in self.deposit_routes),
            approach_ring_radius=self.approach_ring_radius,
            approach_ring_samples=self.approach_ring_samples,
            attack_routes_by_side={
                side: tuple(route.build() for route in routes)
                for side, routes in self.attack_routes_by_side.items()
            },
            protected_for=self.protected_for,
            footprint_enabled=self.footprint_enabled,
            label=self.label,
        )


@dataclass(frozen=True, slots=True)
class ThermometerFeatureConfig:
    semantic_id: int
    position: Vec2
    approach_point: Vec2
    reward: int = 10
    label: str = "thermometer"
    blue_route: tuple[Vec2, ...] = (
        (0.0, -0.70),
        (0.0, -0.77),
        (0.63, -0.77),
        (0.63, -0.65),
    )
    yellow_route: tuple[Vec2, ...] = (
        (0.0, -0.70),
        (0.0, -0.77),
        (-0.63, -0.77),
        (-0.63, -0.65),
    )
    blue_blocking_source_id: int = 13
    yellow_blocking_source_id: int = 23
    blue_blocking_deposit_id: int = 16
    yellow_blocking_deposit_id: int = 26

    def build(self) -> Thermometer:
        return Thermometer(
            semantic_id=self.semantic_id,
            position=self.position,
            approach_point=self.approach_point,
            reward=self.reward,
            label=self.label,
            blue_route=self.blue_route,
            yellow_route=self.yellow_route,
            blue_blocking_source_id=self.blue_blocking_source_id,
            yellow_blocking_source_id=self.yellow_blocking_source_id,
            blue_blocking_deposit_id=self.blue_blocking_deposit_id,
            yellow_blocking_deposit_id=self.yellow_blocking_deposit_id,
        )


@dataclass(frozen=True, slots=True)
class SemanticMapConfig:
    name: str
    field_size: tuple[float, float]
    sources: tuple[SourceZoneConfig, ...]
    deposits: tuple[DepositZoneConfig, ...]
    thermometer: ThermometerFeatureConfig

    def build(self) -> SemanticMap:
        return SemanticMap(
            name=self.name,
            field_size=self.field_size,
            sources={source.semantic_id: source.build() for source in self.sources},
            deposits={deposit.semantic_id: deposit.build() for deposit in self.deposits},
            thermometer=self.thermometer.build(),
        )


@dataclass(frozen=True, slots=True)
class ActionMaskHeuristicsConfig:
    check_semantic_waypoints: bool = True
    check_attack_enemy_block: bool = True
    require_clear_thermometer_lane: bool = True
    allow_blocked_final_waypoint_for: tuple[str, ...] = ("attack_deposit", "start_endgame")
    check_attack_final_waypoint: bool = True
    check_attack_final_segment: bool = True


@dataclass(frozen=True, slots=True)
class MapPlanningConfig:
    semantic_map: SemanticMapConfig
    action_mask_heuristics: ActionMaskHeuristicsConfig


def load_map_planning_config(path: str | Path = DEFAULT_SEMANTIC_MAP_CONFIG_PATH) -> MapPlanningConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return MapPlanningConfig(
        semantic_map=_semantic_map_config_from_raw(_required_mapping(raw, "semantic_map")),
        action_mask_heuristics=_action_mask_config_from_raw(raw.get("action_mask_heuristics", {})),
    )


def load_semantic_map_config(path: str | Path = DEFAULT_SEMANTIC_MAP_CONFIG_PATH) -> SemanticMapConfig:
    return load_map_planning_config(path).semantic_map


def load_action_mask_heuristics_config(path: str | Path = DEFAULT_SEMANTIC_MAP_CONFIG_PATH) -> ActionMaskHeuristicsConfig:
    return load_map_planning_config(path).action_mask_heuristics


def _semantic_map_config_from_raw(raw: dict[object, object]) -> SemanticMapConfig:
    return SemanticMapConfig(
        name=str(raw["name"]),
        field_size=_vec2(raw["field_size"]),
        sources=tuple(_source_config_from_raw(item) for item in _required_sequence(raw, "sources")),
        deposits=tuple(_deposit_config_from_raw(item) for item in _required_sequence(raw, "deposits")),
        thermometer=_thermometer_config_from_raw(_required_mapping(raw, "thermometer")),
    )


def _source_config_from_raw(raw_item: object) -> SourceZoneConfig:
    raw = _as_mapping(raw_item, "source")
    return SourceZoneConfig(
        semantic_id=int(raw["semantic_id"]),
        position=_vec2(raw["position"]),
        label=str(raw["label"]),
        map_obstacle_id=_optional_str(raw.get("map_obstacle_id")),
        collect_routes=tuple(
            _route_config_from_raw(route)
            for route in raw.get("collect_routes", ())
        ),
        available_items=int(raw.get("available_items", 4)),
        available_from_t=float(raw.get("available_from_t", 0.0)),
        footprint_enabled=bool(raw.get("footprint_enabled", True)),
        map_footprint_enabled=bool(raw.get("map_footprint_enabled", True)),
    )


def _deposit_config_from_raw(raw_item: object) -> DepositZoneConfig:
    raw = _as_mapping(raw_item, "deposit")
    return DepositZoneConfig(
        semantic_id=int(raw["semantic_id"]),
        position=_vec2(raw["position"]),
        kind=DepositType(str(raw["kind"])),
        label=str(raw["label"]),
        owner=_optional_side(raw.get("owner")),
        map_obstacle_id=_optional_str(raw.get("map_obstacle_id")),
        deposit_routes=tuple(
            _route_config_from_raw(route)
            for route in raw.get("deposit_routes", ())
        ),
        approach_ring_radius=float(raw.get("approach_ring_radius", 0.0)),
        approach_ring_samples=int(raw.get("approach_ring_samples", 16)),
        attack_routes_by_side=_attack_routes_from_raw(raw.get("attack_routes_by_side", {})),
        protected_for=_optional_side(raw.get("protected_for")),
        map_footprint_enabled=bool(raw.get("map_footprint_enabled", True)),
        footprint_enabled=bool(raw.get("footprint_enabled", True)),
    )


def _thermometer_config_from_raw(raw: dict[object, object]) -> ThermometerFeatureConfig:
    return ThermometerFeatureConfig(
        semantic_id=int(raw["semantic_id"]),
        position=_vec2(raw["position"]),
        approach_point=_vec2(raw["approach_point"]),
        reward=int(raw.get("reward", 10)),
        label=str(raw.get("label", "thermometer")),
        blue_route=_vec2_tuple(raw.get("blue_route", ())),
        yellow_route=_vec2_tuple(raw.get("yellow_route", ())),
        blue_blocking_source_id=int(raw.get("blue_blocking_source_id", 13)),
        yellow_blocking_source_id=int(raw.get("yellow_blocking_source_id", 23)),
        blue_blocking_deposit_id=int(raw.get("blue_blocking_deposit_id", 16)),
        yellow_blocking_deposit_id=int(raw.get("yellow_blocking_deposit_id", 26)),
    )


def _action_mask_config_from_raw(raw_item: object) -> ActionMaskHeuristicsConfig:
    raw = _as_mapping(raw_item, "action_mask_heuristics")
    return ActionMaskHeuristicsConfig(
        check_semantic_waypoints=bool(raw.get("check_semantic_waypoints", True)),
        check_attack_enemy_block=bool(raw.get("check_attack_enemy_block", True)),
        require_clear_thermometer_lane=bool(raw.get("require_clear_thermometer_lane", True)),
        allow_blocked_final_waypoint_for=tuple(
            str(action_type)
            for action_type in raw.get("allow_blocked_final_waypoint_for", ("attack_deposit", "start_endgame"))
        ),
        check_attack_final_waypoint=bool(raw.get("check_attack_final_waypoint", True)),
        check_attack_final_segment=bool(raw.get("check_attack_final_segment", True)),
    )


def _route_config_from_raw(raw_item: object) -> RouteConfig:
    raw = _as_mapping(raw_item, "route")
    return RouteConfig(
        name=str(raw["name"]),
        waypoints=_vec2_tuple(raw["waypoints"]),
        blocked_by_sources=tuple(int(source_id) for source_id in raw.get("blocked_by_sources", ())),
        axis=_optional_str(raw.get("axis")),
        release_target_footprint_after_waypoint=(
            None
            if raw.get("release_target_footprint_after_waypoint") is None
            else int(raw["release_target_footprint_after_waypoint"])
        ),
    )


def _attack_routes_from_raw(raw_item: object) -> dict[Side, tuple[RouteConfig, ...]]:
    raw = _as_mapping(raw_item, "attack_routes_by_side")
    return {
        Side(str(side)): tuple(_route_config_from_raw(route) for route in routes)
        for side, routes in raw.items()
    }


def _vec2_tuple(raw_item: object) -> tuple[Vec2, ...]:
    return tuple(_vec2(item) for item in _as_sequence(raw_item, "waypoints"))


def _vec2(raw_item: object) -> Vec2:
    raw = _as_sequence(raw_item, "point")
    if len(raw) != 2:
        raise ValueError(f"Point must contain exactly two numbers, got {raw!r}")
    return float(raw[0]), float(raw[1])


def _optional_side(raw_item: object) -> Side | None:
    if raw_item is None:
        return None
    return Side(str(raw_item))


def _optional_str(raw_item: object) -> str | None:
    if raw_item is None:
        return None
    return str(raw_item)


def _required_mapping(raw: dict[object, object], key: str) -> dict[object, object]:
    return _as_mapping(raw[key], key)


def _required_sequence(raw: dict[object, object], key: str) -> tuple[object, ...]:
    return _as_sequence(raw[key], key)


def _as_mapping(raw_item: object, name: str) -> dict[object, object]:
    if raw_item is None:
        return {}
    if not isinstance(raw_item, dict):
        raise ValueError(f"{name} must be a mapping")
    return raw_item


def _as_sequence(raw_item: object, name: str) -> tuple[object, ...]:
    if not isinstance(raw_item, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(raw_item)


def _route(
    name: str,
    *waypoints: Vec2,
    blocked_by_sources: tuple[int, ...] = (),
    axis: str | None = None,
) -> RouteConfig:
    return RouteConfig(name=name, waypoints=tuple(waypoints), blocked_by_sources=blocked_by_sources, axis=axis)


def _mirrored_routes(routes: tuple[RouteConfig, ...], suffix: str = "_mirror") -> tuple[RouteConfig, ...]:
    return tuple(route.mirrored(suffix=suffix) for route in routes)


def _mirror_source_id(source_id: int) -> int:
    if 10 <= source_id < 20:
        return 20 + (source_id - 10)
    if 20 <= source_id < 30:
        return 10 + (source_id - 20)
    return source_id


def build_default_semantic_map_config() -> SemanticMapConfig:
    source_14_route = _route("collect_14", (0.35, 0.15), (0.35, 0.06))
    source_14_back_route = _route(
        "collect_14_back",
        (0.35, -0.55),
        (0.35, -0.46),
        blocked_by_sources=(13,),
    )
    source_13_route = _route("collect_13", (0.40, -0.50), (0.40, -0.56))
    source_12_route = _route("collect_12", (0.99, -0.60), (1.06, -0.60))
    source_11_route = _route("collect_11", (1.00, 0.20), (1.06, 0.20))

    deposit_17_center = (0.70, -0.20)
    deposit_17_bt_pose = (0.70, 0.04)
    deposit_27_center = (-0.70, -0.20)
    upper_storage_ring_radius = hypot(
        deposit_17_bt_pose[0] - deposit_17_center[0],
        deposit_17_bt_pose[1] - deposit_17_center[1],
    )
    deposit_10_center = (0.0, -0.90)
    deposit_15_center = (1.40, -0.20)
    deposit_25_center = (-1.40, -0.20)

    deposit_15_attack_routes_common = (
        _route(
            "attack_15_from_top",
            (deposit_15_center[0], deposit_15_center[1] + 0.20),
            deposit_15_center,
            axis="y",
        ),
        _route(
            "attack_15_from_bottom",
            (deposit_15_center[0], deposit_15_center[1] - 0.20),
            deposit_15_center,
            axis="y",
        ),
    )
    deposit_25_attack_routes_common = _mirrored_routes(deposit_15_attack_routes_common)

    sources = (
        SourceZoneConfig(
            11,
            (1.325, 0.2),
            "blue_source_upper",
            map_obstacle_id="11",
            collect_routes=(source_11_route,),
        ),
        SourceZoneConfig(
            12,
            (1.325, -0.6),
            "blue_source_lower",
            map_obstacle_id="12",
            collect_routes=(source_12_route,),
        ),
        SourceZoneConfig(
            13,
            (0.4, -0.825),
            "blue_source_mid_lower",
            map_obstacle_id="13",
            collect_routes=(source_13_route,),
        ),
        SourceZoneConfig(
            14,
            (0.35, -0.2),
            "blue_source_mid_upper",
            map_obstacle_id="14",
            collect_routes=(source_14_route, source_14_back_route),
        ),
        SourceZoneConfig(
            21,
            (-1.325, 0.2),
            "yellow_source_upper",
            map_obstacle_id="21",
            collect_routes=(source_11_route.mirrored(),),
        ),
        SourceZoneConfig(
            22,
            (-1.325, -0.6),
            "yellow_source_lower",
            map_obstacle_id="22",
            collect_routes=(source_12_route.mirrored(),),
        ),
        SourceZoneConfig(
            23,
            (-0.4, -0.825),
            "yellow_source_mid_lower",
            map_obstacle_id="23",
            collect_routes=_mirrored_routes((source_13_route,)),
        ),
        SourceZoneConfig(
            24,
            (-0.35, -0.2),
            "yellow_source_mid_upper",
            map_obstacle_id="24",
            collect_routes=_mirrored_routes((source_14_route, source_14_back_route)),
        ),
    )
    deposits = (
        DepositZoneConfig(
            1,
            (0.0, -0.2),
            DepositType.STORAGE,
            "center_storage_upper",
            map_obstacle_id="00",
            approach_ring_radius=upper_storage_ring_radius,
        ),
        DepositZoneConfig(
            10,
            deposit_10_center,
            DepositType.STORAGE,
            "center_storage_lower",
            map_obstacle_id="10",
            deposit_routes=(_route("drop_10", (0.0, -0.65)),),
            attack_routes_by_side={
                Side.BLUE: (
                    _route(
                        "attack_10_from_right",
                        (0.24, -0.90),
                        deposit_10_center,
                        blocked_by_sources=(13,),
                        axis="x",
                    ),
                ),
                Side.YELLOW: (
                    _route(
                        "attack_10_from_left",
                        (-0.24, -0.90),
                        deposit_10_center,
                        blocked_by_sources=(23,),
                        axis="x",
                    ),
                ),
            },
        ),
        DepositZoneConfig(
            15,
            deposit_15_center,
            DepositType.STORAGE,
            "right_corner_storage",
            map_obstacle_id="15",
            deposit_routes=(_route("drop_15", (1.13, -0.20)),),
            attack_routes_by_side={
                Side.BLUE: deposit_15_attack_routes_common,
                Side.YELLOW: deposit_15_attack_routes_common,
            },
            protected_for=Side.BLUE,
        ),
        DepositZoneConfig(
            16,
            (0.80, -0.90),
            DepositType.STORAGE,
            "blue_storage_lower",
            owner=Side.BLUE,
            map_obstacle_id="16",
            deposit_routes=(_route("drop_16", (0.80, -0.65)),),
            protected_for=Side.BLUE,
        ),
        DepositZoneConfig(
            17,
            deposit_17_center,
            DepositType.STORAGE,
            "right_upper_storage",
            map_obstacle_id="17",
            approach_ring_radius=upper_storage_ring_radius,
        ),
        DepositZoneConfig(
            25,
            deposit_25_center,
            DepositType.STORAGE,
            "left_corner_storage",
            map_obstacle_id="25",
            deposit_routes=(_route("drop_25", (-1.13, -0.20)),),
            attack_routes_by_side={
                Side.BLUE: deposit_25_attack_routes_common,
                Side.YELLOW: deposit_25_attack_routes_common,
            },
            protected_for=Side.YELLOW,
        ),
        DepositZoneConfig(
            26,
            (-0.80, -0.90),
            DepositType.STORAGE,
            "yellow_storage_lower",
            owner=Side.YELLOW,
            map_obstacle_id="26",
            deposit_routes=(_route("drop_26", (-0.80, -0.65)),),
            protected_for=Side.YELLOW,
        ),
        DepositZoneConfig(
            27,
            deposit_27_center,
            DepositType.STORAGE,
            "left_upper_storage",
            map_obstacle_id="27",
            approach_ring_radius=upper_storage_ring_radius,
        ),
        DepositZoneConfig(
            101,
            (1.12, 0.75),
            DepositType.HOME,
            "blue_home",
            owner=Side.BLUE,
            deposit_routes=(_route("drop_blue_home", (1.12, 0.75)),),
            protected_for=Side.BLUE,
        ),
        DepositZoneConfig(
            201,
            (-1.12, 0.75),
            DepositType.HOME,
            "yellow_home",
            owner=Side.YELLOW,
            deposit_routes=(_route("drop_yellow_home", (-1.12, 0.75)),),
            protected_for=Side.YELLOW,
        ),
    )
    return SemanticMapConfig(
        name="default_semantic_field",
        field_size=(FIELD_WIDTH, FIELD_HEIGHT),
        sources=sources,
        deposits=deposits,
        thermometer=ThermometerFeatureConfig(900, (0.0, -1.0), approach_point=(0.0, -0.77)),
    )


_DEFAULT_MAP_PLANNING_CONFIG = load_map_planning_config()
DEFAULT_SEMANTIC_MAP_CONFIG = _DEFAULT_MAP_PLANNING_CONFIG.semantic_map
DEFAULT_ACTION_MASK_HEURISTICS = _DEFAULT_MAP_PLANNING_CONFIG.action_mask_heuristics
