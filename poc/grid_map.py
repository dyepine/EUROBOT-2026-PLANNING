from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from poc.entities import DepositPoint, SourcePoint


OCCUPIED = 100
FREE = 0
DEFAULT_LAYOUT_PATH = Path("/home/napalkov/coding/EUROBOT_2026_PC/data/configs/map_publisher_config/map_layout.yaml")


@dataclass(slots=True, frozen=True)
class RectangleObstacle:
    obstacle_id: str
    cx: float
    cy: float
    width: float
    height: float


@dataclass(slots=True)
class GridMapConfig:
    width_m: float = 3.0
    height_m: float = 2.0
    resolution_m: float = 0.04
    origin_xy: tuple[float, float] = (-1.5, -1.0)
    obstacle_inflation_m: float = 0.08
    border_inflation_m: float = 0.08
    dynamic_circle_radius_m: float = 0.15


@dataclass(slots=True)
class GridOccupancyMap:
    team_color: str
    config: GridMapConfig
    start_obstacles: dict[str, RectangleObstacle]
    match_obstacles: dict[str, RectangleObstacle]
    width_px: int
    height_px: int
    static_start_ids: set[str] = field(default_factory=set)
    dynamic_start_ids: set[str] = field(default_factory=set)
    active_start_ids: set[str] = field(default_factory=set)
    active_match_ids: set[str] = field(default_factory=set)
    dynamic_circles: list[tuple[float, float, float]] = field(default_factory=list)
    true_map: np.ndarray | None = None
    planning_map: np.ndarray | None = None

    @classmethod
    def from_layout(
        cls,
        layout_path: str | Path,
        team_color: str = "all",
        config: GridMapConfig | None = None,
        clear_mode: bool = False,
    ) -> "GridOccupancyMap":
        if team_color not in {"blue", "yellow", "all"}:
            raise ValueError("team_color must be 'blue', 'yellow', or 'all'")

        grid_config = config or GridMapConfig()
        layout = yaml.safe_load(Path(layout_path).read_text(encoding="utf-8")) or {}

        common_start = cls._convert_rectangles(layout.get("common_start", {}))
        teams = layout.get("teams", {}) or {}

        if team_color == "all":
            start_rectangles = common_start.copy()
            static_start_ids = {"99"} if "99" in start_rectangles else set()
            dynamic_start_ids = set(start_rectangles.keys()) - static_start_ids
            match_rectangles: dict[str, RectangleObstacle] = {}
            for color in ("blue", "yellow"):
                team_layout = teams.get(color, {})
                prefixed_start = cls._convert_rectangles(
                    team_layout.get("start", {}),
                    prefix=f"{color}:start:",
                )
                start_rectangles.update(prefixed_start)
                static_start_ids.update(prefixed_start.keys())
                match_rectangles.update(cls._convert_rectangles(team_layout.get("match", {})))
        else:
            team_layout = teams.get(team_color, {})
            team_start = cls._convert_rectangles(team_layout.get("start", {}))
            match_rectangles = cls._convert_rectangles(team_layout.get("match", {}))
            start_rectangles = common_start.copy()
            start_rectangles.update(team_start)
            static_start_ids = set(team_start.keys())
            if "99" in start_rectangles:
                static_start_ids.add("99")
            dynamic_start_ids = set(start_rectangles.keys()) - static_start_ids

        width_px = int(round(grid_config.width_m / grid_config.resolution_m))
        height_px = int(round(grid_config.height_m / grid_config.resolution_m))

        occupancy = cls(
            team_color=team_color,
            config=grid_config,
            start_obstacles=start_rectangles,
            match_obstacles=match_rectangles,
            width_px=width_px,
            height_px=height_px,
            static_start_ids=static_start_ids,
            dynamic_start_ids=dynamic_start_ids,
            active_start_ids=set() if clear_mode else set(dynamic_start_ids),
            active_match_ids=set(),
        )
        occupancy.rebuild()
        return occupancy

    @staticmethod
    def _convert_rectangles(
        raw_rectangles: dict[object, list[float]],
        prefix: str = "",
    ) -> dict[str, RectangleObstacle]:
        rectangles: dict[str, RectangleObstacle] = {}
        for obstacle_id, values in raw_rectangles.items():
            if len(values) != 4:
                raise ValueError(f'Obstacle "{obstacle_id}" must contain [cx, cy, w, h]')
            obstacle_key = f"{prefix}{obstacle_id}"
            rectangles[obstacle_key] = RectangleObstacle(
                obstacle_id=obstacle_key,
                cx=float(values[0]),
                cy=float(values[1]),
                width=float(values[2]),
                height=float(values[3]),
            )
        return rectangles

    @property
    def shape(self) -> tuple[int, int]:
        return self.height_px, self.width_px

    def clone(self) -> "GridOccupancyMap":
        cloned = GridOccupancyMap(
            team_color=self.team_color,
            config=GridMapConfig(
                width_m=self.config.width_m,
                height_m=self.config.height_m,
                resolution_m=self.config.resolution_m,
                origin_xy=self.config.origin_xy,
                obstacle_inflation_m=self.config.obstacle_inflation_m,
                border_inflation_m=self.config.border_inflation_m,
                dynamic_circle_radius_m=self.config.dynamic_circle_radius_m,
            ),
            start_obstacles=self.start_obstacles.copy(),
            match_obstacles=self.match_obstacles.copy(),
            width_px=self.width_px,
            height_px=self.height_px,
            static_start_ids=set(self.static_start_ids),
            dynamic_start_ids=set(self.dynamic_start_ids),
            active_start_ids=set(self.active_start_ids),
            active_match_ids=set(self.active_match_ids),
            dynamic_circles=list(self.dynamic_circles),
            true_map=None if self.true_map is None else self.true_map.copy(),
            planning_map=None if self.planning_map is None else self.planning_map.copy(),
        )
        return cloned

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        origin_x, origin_y = self.config.origin_xy
        col = int((x - origin_x) / self.config.resolution_m)
        row = int((y - origin_y) / self.config.resolution_m)
        row = max(0, min(row, self.height_px - 1))
        col = max(0, min(col, self.width_px - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        origin_x, origin_y = self.config.origin_xy
        x = origin_x + (col + 0.5) * self.config.resolution_m
        y = origin_y + (row + 0.5) * self.config.resolution_m
        return x, y

    def is_blocked(self, row: int, col: int, use_inflated: bool = True) -> bool:
        current_map = self.planning_map if use_inflated else self.true_map
        if current_map is None:
            self.rebuild()
            current_map = self.planning_map if use_inflated else self.true_map
        assert current_map is not None
        return current_map[row, col] != FREE

    def set_start_obstacle_enabled(self, obstacle_id: int | str, enabled: bool) -> None:
        obstacle_key = str(obstacle_id)
        if obstacle_key not in self.dynamic_start_ids:
            raise KeyError(f"Unknown start obstacle id: {obstacle_key}")
        if enabled:
            self.active_start_ids.add(obstacle_key)
        else:
            self.active_start_ids.discard(obstacle_key)
        self.rebuild()

    def set_match_obstacle_enabled(self, obstacle_id: int | str, enabled: bool) -> None:
        obstacle_key = str(obstacle_id)
        if obstacle_key not in self.match_obstacles:
            raise KeyError(f"Unknown match obstacle id: {obstacle_key}")
        if enabled:
            self.active_match_ids.add(obstacle_key)
        else:
            self.active_match_ids.discard(obstacle_key)
        self.rebuild()

    def apply_layout_event(self, obstacle_id: int | str) -> None:
        obstacle_key = str(obstacle_id)
        # Keep ROS-node behavior: match obstacles are added, start obstacles are removed.
        if obstacle_key in self.match_obstacles:
            self.active_match_ids.add(obstacle_key)
        elif obstacle_key in self.dynamic_start_ids:
            self.active_start_ids.discard(obstacle_key)
        else:
            raise KeyError(f"Unknown obstacle id: {obstacle_key}")
        self.rebuild()

    def set_dynamic_circles(self, circles: list[tuple[float, float, float]]) -> None:
        self.dynamic_circles = list(circles)
        self.rebuild()

    def set_dynamic_circle_points(
        self,
        points: list[tuple[float, float]],
        radius_m: float | None = None,
    ) -> None:
        circle_radius = radius_m if radius_m is not None else self.config.dynamic_circle_radius_m
        self.dynamic_circles = [(x, y, circle_radius) for x, y in points]
        self.rebuild()

    def sync_semantic_state(
        self,
        sources: dict[int, SourcePoint],
        deposits: dict[int, DepositPoint],
    ) -> None:
        active_start_ids = {
            source.map_obstacle_id or str(source_id)
            for source_id, source in sources.items()
            if source.available_items > 0
            and source.state.value != "empty"
            and (source.map_obstacle_id or str(source_id)) in self.dynamic_start_ids
        }
        active_match_ids = {
            deposit.map_obstacle_id or str(deposit_id)
            for deposit_id, deposit in deposits.items()
            if deposit.total_items() > 0
            and (deposit.map_obstacle_id or str(deposit_id)) in self.match_obstacles
        }
        self.active_start_ids = active_start_ids
        self.active_match_ids = active_match_ids
        self.rebuild()

    def rebuild(self) -> None:
        raw = np.zeros(self.shape, dtype=np.uint8)
        self._fill_borders(raw, OCCUPIED)

        for obstacle_id in self.static_start_ids:
            obstacle = self.start_obstacles.get(obstacle_id)
            if obstacle is not None:
                self._add_rectangle(raw, obstacle, OCCUPIED)

        for obstacle_id in self.active_start_ids:
            obstacle = self.start_obstacles.get(obstacle_id)
            if obstacle is not None:
                self._add_rectangle(raw, obstacle, OCCUPIED)

        for obstacle_id in self.active_match_ids:
            obstacle = self.match_obstacles.get(obstacle_id)
            if obstacle is not None:
                self._add_rectangle(raw, obstacle, OCCUPIED)

        for cx, cy, radius_m in self.dynamic_circles:
            self._add_circle(raw, cx, cy, radius_m, OCCUPIED)

        self.true_map = raw
        self.planning_map = self._inflate_binary(raw)

    def _add_rectangle(self, raw: np.ndarray, obstacle: RectangleObstacle, occupied_value: int) -> None:
        half_w = obstacle.width / 2.0
        half_h = obstacle.height / 2.0
        row_min, col_min = self.world_to_grid(obstacle.cx - half_w, obstacle.cy - half_h)
        row_max, col_max = self.world_to_grid(obstacle.cx + half_w, obstacle.cy + half_h)
        r0, r1 = sorted((row_min, row_max))
        c0, c1 = sorted((col_min, col_max))
        raw[r0:r1 + 1, c0:c1 + 1] = occupied_value

    def _add_circle(self, raw: np.ndarray, cx: float, cy: float, radius_m: float, occupied_value: int) -> None:
        center_row, center_col = self.world_to_grid(cx, cy)
        radius_cells = max(1, int(round(radius_m / self.config.resolution_m)))
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                row = center_row + dr
                col = center_col + dc
                if 0 <= row < self.height_px and 0 <= col < self.width_px:
                    raw[row, col] = occupied_value

    def _fill_borders(self, raw: np.ndarray, occupied_value: int) -> None:
        raw[0, :] = occupied_value
        raw[-1, :] = occupied_value
        raw[:, 0] = occupied_value
        raw[:, -1] = occupied_value

    def _inflate_binary(self, raw: np.ndarray) -> np.ndarray:
        inflated = raw.copy()
        obstacle_radius = int(round(self.config.obstacle_inflation_m / self.config.resolution_m))
        border_radius = int(round(self.config.border_inflation_m / self.config.resolution_m))

        obstacle_mask = raw == OCCUPIED
        inflated = self._dilate_mask(inflated, obstacle_mask, obstacle_radius)

        border_mask = np.zeros_like(raw, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        inflated = self._dilate_mask(inflated, border_mask, border_radius)
        return inflated

    @staticmethod
    def _dilate_mask(raw: np.ndarray, mask: np.ndarray, radius_cells: int) -> np.ndarray:
        if radius_cells <= 0:
            return raw
        result = raw.copy()
        rows, cols = np.where(mask)
        offsets = [
            (dr, dc)
            for dr in range(-radius_cells, radius_cells + 1)
            for dc in range(-radius_cells, radius_cells + 1)
            if dr * dr + dc * dc <= radius_cells * radius_cells
        ]
        for row, col in zip(rows.tolist(), cols.tolist()):
            for dr, dc in offsets:
                rr = row + dr
                cc = col + dc
                if 0 <= rr < result.shape[0] and 0 <= cc < result.shape[1]:
                    result[rr, cc] = OCCUPIED
        return result
