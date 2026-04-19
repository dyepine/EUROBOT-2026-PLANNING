from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import hypot

from poc.entities import DepositPoint, DepositType, RouteOption, Side, SourcePoint, Thermometer

FIELD_WIDTH = 3.0
FIELD_HEIGHT = 2.0


@dataclass(slots=True)
class SemanticMap:
    name: str
    field_size: tuple[float, float]
    sources: dict[int, SourcePoint]
    deposits: dict[int, DepositPoint]
    thermometer: Thermometer

    def clone(self) -> "SemanticMap":
        return deepcopy(self)


def _route(name: str, *waypoints: tuple[float, float], blocked_by_sources: tuple[int, ...] = ()) -> RouteOption:
    return RouteOption(name=name, waypoints=tuple(waypoints), blocked_by_sources=blocked_by_sources)


def _mirror_route(route: RouteOption, suffix: str = "_mirror") -> RouteOption:
    return RouteOption(
        name=f"{route.name}{suffix}",
        waypoints=tuple((-point[0], point[1]) for point in route.waypoints),
        blocked_by_sources=tuple(20 + (source_id - 10) if 10 <= source_id < 20 else source_id for source_id in route.blocked_by_sources),
        axis=route.axis,
    )


def _mirrored_routes(routes: tuple[RouteOption, ...], suffix: str = "_mirror") -> tuple[RouteOption, ...]:
    return tuple(_mirror_route(route, suffix=suffix) for route in routes)


def build_default_semantic_map() -> SemanticMap:
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

    sources = {
        11: SourcePoint(11, (1.325, 0.2), map_obstacle_id="11", collect_routes=(source_11_route,), label="blue_source_upper"),
        12: SourcePoint(12, (1.325, -0.6), map_obstacle_id="12", collect_routes=(source_12_route,), label="blue_source_lower"),
        13: SourcePoint(13, (0.4, -0.825), map_obstacle_id="13", collect_routes=(source_13_route,), label="blue_source_mid_lower"),
        14: SourcePoint(
            14,
            (0.35, -0.2),
            map_obstacle_id="14",
            collect_routes=(source_14_route, source_14_back_route),
            label="blue_source_mid_upper",
        ),
        21: SourcePoint(21, (-1.325, 0.2), map_obstacle_id="21", collect_routes=(_mirror_route(source_11_route),), label="yellow_source_upper"),
        22: SourcePoint(22, (-1.325, -0.6), map_obstacle_id="22", collect_routes=(_mirror_route(source_12_route),), label="yellow_source_lower"),
        23: SourcePoint(23, (-0.4, -0.825), map_obstacle_id="23", collect_routes=(_mirror_route(source_13_route),), label="yellow_source_mid_lower"),
        24: SourcePoint(
            24,
            (-0.35, -0.2),
            map_obstacle_id="24",
            collect_routes=_mirrored_routes((source_14_route, source_14_back_route)),
            label="yellow_source_mid_upper",
        ),
    }

    deposit_17_center = (0.70, -0.20)
    deposit_17_bt_pose = (0.70, 0.04)
    deposit_27_center = (-0.70, -0.20)
    upper_storage_ring_radius = hypot(
        deposit_17_bt_pose[0] - deposit_17_center[0],
        deposit_17_bt_pose[1] - deposit_17_center[1],
    )
    center_storage_attack_routes = {
        Side.BLUE: (RouteOption(name="attack_01_center", waypoints=((0.0, -0.2),)),),
        Side.YELLOW: (RouteOption(name="attack_01_center", waypoints=((0.0, -0.2),)),),
    }
    deposit_17_attack_routes = {
        Side.BLUE: (RouteOption(name="attack_17_center", waypoints=(deposit_17_center,)),),
        Side.YELLOW: (RouteOption(name="attack_17_center", waypoints=(deposit_17_center,)),),
    }
    deposit_27_attack_routes = {
        Side.BLUE: (RouteOption(name="attack_27_center", waypoints=(deposit_27_center,)),),
        Side.YELLOW: (RouteOption(name="attack_27_center", waypoints=(deposit_27_center,)),),
    }
    deposit_10_center = (0.0, -0.90)
    deposit_15_center = (1.40, -0.20)
    deposit_25_center = (-1.40, -0.20)
    deposit_10_attack_routes = {
        Side.BLUE: (
            RouteOption(
                name="attack_10_from_right",
                waypoints=((0.20, -0.90), deposit_10_center),
                blocked_by_sources=(13,),
                axis="x",
            ),
        ),
        Side.YELLOW: (
            RouteOption(
                name="attack_10_from_left",
                waypoints=((-0.20, -0.90), deposit_10_center),
                blocked_by_sources=(23,),
                axis="x",
            ),
        ),
    }
    deposit_15_attack_routes_common = (
        RouteOption(
            name="attack_15_from_top",
            waypoints=((deposit_15_center[0], deposit_15_center[1] + 0.20), deposit_15_center),
            axis="y",
        ),
        RouteOption(
            name="attack_15_from_bottom",
            waypoints=((deposit_15_center[0], deposit_15_center[1] - 0.20), deposit_15_center),
            axis="y",
        ),
    )
    deposit_15_attack_routes = {
        Side.BLUE: deposit_15_attack_routes_common,
        Side.YELLOW: deposit_15_attack_routes_common,
    }
    deposit_25_attack_routes_common = _mirrored_routes(deposit_15_attack_routes_common)
    deposit_25_attack_routes = {
        Side.BLUE: deposit_25_attack_routes_common,
        Side.YELLOW: deposit_25_attack_routes_common,
    }

    deposits = {
        1: DepositPoint(
            1,
            (0.0, -0.2),
            DepositType.STORAGE,
            None,
            map_obstacle_id="00",
            approach_ring_radius=upper_storage_ring_radius,
            attack_routes_by_side=center_storage_attack_routes,
            label="center_storage_upper",
        ),
        10: DepositPoint(
            10,
            deposit_10_center,
            DepositType.STORAGE,
            None,
            map_obstacle_id="10",
            deposit_routes=(_route("drop_10", deposit_10_center),),
            attack_routes_by_side=deposit_10_attack_routes,
            label="center_storage_lower",
        ),
        15: DepositPoint(
            15,
            deposit_15_center,
            DepositType.STORAGE,
            None,
            map_obstacle_id="15",
            deposit_routes=(_route("drop_15", (1.13, -0.20)),),
            attack_routes_by_side=deposit_15_attack_routes,
            label="right_corner_storage",
        ),
        16: DepositPoint(
            16,
            (0.80, -0.90),
            DepositType.STORAGE,
            Side.BLUE,
            map_obstacle_id="16",
            deposit_routes=(_route("drop_16", (0.80, -0.65)),),
            protected_for=Side.BLUE,
            label="blue_storage_lower",
        ),
        17: DepositPoint(
            17,
            deposit_17_center,
            DepositType.STORAGE,
            None,
            map_obstacle_id="17",
            approach_ring_radius=upper_storage_ring_radius,
            attack_routes_by_side=deposit_17_attack_routes,
            label="right_upper_storage",
        ),
        25: DepositPoint(
            25,
            deposit_25_center,
            DepositType.STORAGE,
            None,
            map_obstacle_id="25",
            deposit_routes=(_route("drop_25", (-1.13, -0.20)),),
            attack_routes_by_side=deposit_25_attack_routes,
            label="left_corner_storage",
        ),
        26: DepositPoint(
            26,
            (-0.80, -0.90),
            DepositType.STORAGE,
            Side.YELLOW,
            map_obstacle_id="26",
            deposit_routes=(_route("drop_26", (-0.80, -0.65)),),
            protected_for=Side.YELLOW,
            label="yellow_storage_lower",
        ),
        27: DepositPoint(
            27,
            deposit_27_center,
            DepositType.STORAGE,
            None,
            map_obstacle_id="27",
            approach_ring_radius=upper_storage_ring_radius,
            attack_routes_by_side=deposit_27_attack_routes,
            label="left_upper_storage",
        ),
        101: DepositPoint(
            101,
            (1.12, 0.75),
            DepositType.HOME,
            Side.BLUE,
            deposit_routes=(_route("drop_blue_home", (1.12, 0.75)),),
            protected_for=Side.BLUE,
            label="blue_home",
        ),
        201: DepositPoint(
            201,
            (-1.12, 0.75),
            DepositType.HOME,
            Side.YELLOW,
            deposit_routes=(_route("drop_yellow_home", (-1.12, 0.75)),),
            protected_for=Side.YELLOW,
            label="yellow_home",
        ),
    }

    thermometer = Thermometer(
        900,
        (0.0, -1.0),
        approach_point=(0.0, -0.77),
    )
    return SemanticMap(
        name="default_semantic_field",
        field_size=(FIELD_WIDTH, FIELD_HEIGHT),
        sources=sources,
        deposits=deposits,
        thermometer=thermometer,
    )
