from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import cos, pi, sin

from poc.geometry import Vec2


class Side(str, Enum):
    BLUE = "blue"
    YELLOW = "yellow"

    def opponent(self) -> "Side":
        return Side.YELLOW if self is Side.BLUE else Side.BLUE


@dataclass(slots=True, frozen=True)
class RouteOption:
    name: str
    waypoints: tuple[Vec2, ...]
    blocked_by_sources: tuple[int, ...] = ()
    axis: str | None = None
    release_target_footprint_after_waypoint: int | None = None


class SourceState(str, Enum):
    UNTOUCHED = "untouched"
    DISTURBED = "disturbed"
    EMPTY = "empty"


class DepositType(str, Enum):
    HOME = "home"
    STORAGE = "storage"


class PushState(str, Enum):
    CLEAR = "clear"
    PUSHED_LEFT = "pushed_left"
    PUSHED_RIGHT = "pushed_right"
    PUSHED_UP = "pushed_up"
    PUSHED_DOWN = "pushed_down"


class ThermometerState(str, Enum):
    NOT_DONE = "not_done"
    DONE_BLUE = "done_blue"
    DONE_YELLOW = "done_yellow"
    DONE_BOTH = "done_both"


@dataclass(slots=True)
class SourcePoint:
    semantic_id: int
    position: Vec2
    map_obstacle_id: str | None = None
    map_footprint_enabled: bool = True
    collect_routes: tuple[RouteOption, ...] = ()
    state: SourceState = SourceState.UNTOUCHED
    available_items: int = 4
    available_from_t: float = 0.0
    footprint_enabled: bool = True
    released_by: str | None = None
    label: str = ""

    def is_available(self, t: float) -> bool:
        return self.available_from_t <= t and self.state is not SourceState.EMPTY and self.available_items > 0

    def collection_routes(self) -> tuple[RouteOption, ...]:
        if self.collect_routes:
            return self.collect_routes
        return (RouteOption(name="default", waypoints=(self.position,)),)


@dataclass(slots=True)
class DepositPoint:
    semantic_id: int
    position: Vec2
    kind: DepositType
    owner: Side | None
    map_obstacle_id: str | None = None
    map_footprint_enabled: bool = True
    deposit_routes: tuple[RouteOption, ...] = ()
    approach_ring_radius: float = 0.0
    approach_ring_samples: int = 16
    attack_routes_by_side: dict[Side, tuple[RouteOption, ...]] = field(default_factory=dict)
    protected_for: Side | None = None
    blue_items: int = 0
    yellow_items: int = 0
    footprint_enabled: bool = True
    push_state: PushState = PushState.CLEAR
    pushed_owner: Side | None = None
    occupied_by: Side | None = None
    was_occupied: bool = False
    label: str = ""

    def items_for_side(self, side: Side) -> int:
        return self.blue_items if side is Side.BLUE else self.yellow_items

    def total_items(self) -> int:
        return self.blue_items + self.yellow_items

    def add_items(self, side: Side, count: int) -> None:
        if count > 0:
            self.was_occupied = True
            self.map_footprint_enabled = True
        if side is Side.BLUE:
            self.blue_items += count
        else:
            self.yellow_items += count
        self._refresh_occupancy()

    def remove_items(self, side: Side, count: int) -> int:
        if side is Side.BLUE:
            removed = min(self.blue_items, count)
            self.blue_items -= removed
            self._refresh_occupancy()
            return removed
        removed = min(self.yellow_items, count)
        self.yellow_items -= removed
        self._refresh_occupancy()
        return removed

    def clear(self) -> None:
        self.blue_items = 0
        self.yellow_items = 0
        self.map_footprint_enabled = False
        self._refresh_occupancy()

    def set_pushed_state(self, push_state: PushState, owner: Side | None) -> None:
        self.push_state = push_state
        self.pushed_owner = owner

    def clear_pushed_state(self) -> None:
        self.push_state = PushState.CLEAR
        self.pushed_owner = None

    def _refresh_occupancy(self) -> None:
        if self.blue_items > 0 and self.yellow_items == 0:
            self.occupied_by = Side.BLUE
        elif self.yellow_items > 0 and self.blue_items == 0:
            self.occupied_by = Side.YELLOW
        else:
            self.occupied_by = None

    def _approach_ring_points(self) -> tuple[Vec2, ...]:
        if self.approach_ring_radius <= 0.0:
            return ()
        return tuple(
            (
                self.position[0] + self.approach_ring_radius * cos(2.0 * pi * index / self.approach_ring_samples),
                self.position[1] + self.approach_ring_radius * sin(2.0 * pi * index / self.approach_ring_samples),
            )
            for index in range(self.approach_ring_samples)
        )

    def deposit_route_candidates(self) -> tuple[RouteOption, ...]:
        if self.deposit_routes:
            return self.deposit_routes
        ring_points = self._approach_ring_points()
        if ring_points:
            return tuple(
                RouteOption(
                    name=f"ring_{self.semantic_id}_{index}",
                    waypoints=(point,),
                )
                for index, point in enumerate(ring_points)
            )
        return (RouteOption(name="default", waypoints=(self.position,)),)

    def attack_route_candidates(self, side: Side) -> tuple[RouteOption, ...]:
        routes = self.attack_routes_by_side.get(side)
        if routes:
            return routes
        ring_points = self._approach_ring_points()
        if ring_points:
            return tuple(
                RouteOption(
                    name=f"attack_ring_{self.semantic_id}_{index}",
                    waypoints=(point, self.position),
                )
                for index, point in enumerate(ring_points)
            )
        return self.deposit_route_candidates()


@dataclass(slots=True)
class Thermometer:
    semantic_id: int
    position: Vec2
    approach_point: Vec2 = (0.0, -0.77)
    reward: int = 10
    state: ThermometerState = ThermometerState.NOT_DONE
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

    def is_done(self) -> bool:
        return self.state is ThermometerState.DONE_BOTH

    def is_done_for_side(self, side: Side) -> bool:
        if side is Side.BLUE:
            return self.state in (ThermometerState.DONE_BLUE, ThermometerState.DONE_BOTH)
        return self.state in (ThermometerState.DONE_YELLOW, ThermometerState.DONE_BOTH)

    def mark_done_for_side(self, side: Side) -> None:
        if side is Side.BLUE:
            if self.state is ThermometerState.DONE_YELLOW:
                self.state = ThermometerState.DONE_BOTH
            else:
                self.state = ThermometerState.DONE_BLUE
            return
        if self.state is ThermometerState.DONE_BLUE:
            self.state = ThermometerState.DONE_BOTH
        else:
            self.state = ThermometerState.DONE_YELLOW

    def route_for_side(self, side: Side) -> tuple[Vec2, ...]:
        return self.blue_route if side is Side.BLUE else self.yellow_route

    def drag_route_for_side(self, side: Side) -> tuple[Vec2, ...]:
        route = self.route_for_side(side)
        if len(route) >= 4:
            return route[1:4]
        if len(route) >= 2:
            return route[1:]
        return route

    def drag_start_for_side(self, side: Side) -> Vec2:
        drag_route = self.drag_route_for_side(side)
        if drag_route:
            return drag_route[0]
        return self.position

    def drag_end_for_side(self, side: Side) -> Vec2:
        drag_route = self.drag_route_for_side(side)
        if drag_route:
            return drag_route[-1]
        return self.position

    def blocking_source_id_for_side(self, side: Side) -> int:
        return self.blue_blocking_source_id if side is Side.BLUE else self.yellow_blocking_source_id

    def blocking_deposit_id_for_side(self, side: Side) -> int:
        return self.blue_blocking_deposit_id if side is Side.BLUE else self.yellow_blocking_deposit_id


@dataclass(slots=True)
class Robot:
    name: str
    side: Side
    position: Vec2
    speed: float = 0.45
    load: int = 0
    capacity: int = 4
    current_action: str | None = None
    current_target_id: int | None = None
    notes: list[str] = field(default_factory=list)
