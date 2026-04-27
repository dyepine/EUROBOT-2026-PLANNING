from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from math import sqrt

from poc.geometry import Vec2, distance, path_length
from poc.grid_map import GridOccupancyMap


SQRT2 = sqrt(2.0)


@dataclass(slots=True)
class PlannedGridPath:
    success: bool
    waypoints: tuple[Vec2, ...]
    distance_m: float
    traversed_cells: int
    duration_source: str = "grid_astar+constants"
    used_obstacle_exit: bool = False
    used_goal_bridge: bool = False


class GridAStarPlanner:
    DIRECTIONS = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, SQRT2),
        (-1, 1, SQRT2),
        (1, -1, SQRT2),
        (1, 1, SQRT2),
    )

    def __init__(
        self,
        occupancy_map: GridOccupancyMap,
        obstacle_exit_cost: float = 500.0,
    ) -> None:
        self.occupancy_map = occupancy_map
        self.obstacle_exit_cost = obstacle_exit_cost

    def plan(
        self,
        start: Vec2,
        goal: Vec2,
        *,
        use_inflated: bool = True,
        allow_obstacle_exit: bool | None = None,
        allow_goal_occupied: bool = False,
        compress: bool = True,
    ) -> PlannedGridPath:
        start_cell = self.occupancy_map.world_to_grid(*start)
        goal_cell = self.occupancy_map.world_to_grid(*goal)
        start_blocked = self.occupancy_map.is_blocked(*start_cell, use_inflated=use_inflated)
        goal_blocked = self.occupancy_map.is_blocked(*goal_cell, use_inflated=use_inflated)
        if allow_obstacle_exit is None:
            allow_obstacle_exit = start_blocked

        search_goal = goal_cell
        used_goal_bridge = False
        if goal_blocked:
            if not allow_goal_occupied:
                return PlannedGridPath(False, (), 0.0, 0)
            nearest_free = self._find_nearest_free_cell(goal_cell, use_inflated=use_inflated)
            if nearest_free is None:
                return PlannedGridPath(False, (), 0.0, 0)
            search_goal = nearest_free
            used_goal_bridge = search_goal != goal_cell

        if start_cell == search_goal:
            direct_waypoints: list[Vec2] = []
            if distance(start, goal) > 1e-9:
                direct_waypoints.append(goal)
            return PlannedGridPath(
                success=True,
                waypoints=tuple(direct_waypoints),
                distance_m=path_length((start, *direct_waypoints)),
                traversed_cells=0,
                used_obstacle_exit=allow_obstacle_exit,
                used_goal_bridge=used_goal_bridge,
            )

        pending: list[tuple[float, float, tuple[int, int]]] = []
        heappush(pending, (self._heuristic(start_cell, search_goal), 0.0, start_cell))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        best_cost: dict[tuple[int, int], float] = {start_cell: 0.0}

        goal_found = False
        while pending:
            _, current_cost, current = heappop(pending)
            if current == search_goal:
                goal_found = True
                break
            if current_cost > best_cost.get(current, float("inf")):
                continue

            row, col = current
            for d_row, d_col, step_cost in self.DIRECTIONS:
                neighbor = (row + d_row, col + d_col)
                if not self._cell_on_map(neighbor):
                    continue
                if (
                    d_row != 0
                    and d_col != 0
                    and not self.occupancy_map.is_blocked(row, col, use_inflated=use_inflated)
                    and self._diagonal_cuts_blocked_corner(current, d_row, d_col, use_inflated=use_inflated)
                ):
                    continue
                if self.occupancy_map.is_blocked(*neighbor, use_inflated=use_inflated):
                    if not allow_obstacle_exit:
                        continue
                    candidate_cost = current_cost + self.obstacle_exit_cost
                else:
                    candidate_cost = current_cost + step_cost

                if candidate_cost >= best_cost.get(neighbor, float("inf")):
                    continue
                best_cost[neighbor] = candidate_cost
                came_from[neighbor] = current
                priority = candidate_cost + self._heuristic(neighbor, search_goal)
                heappush(pending, (priority, candidate_cost, neighbor))

        if not goal_found:
            return PlannedGridPath(False, (), 0.0, 0)

        full_cell_path = self._reconstruct_path(came_from, start_cell, search_goal)
        cells_without_start = full_cell_path[1:]
        if compress:
            cells_without_start = self._compress_cells(cells_without_start)

        points = [self.occupancy_map.grid_to_world(row, col) for row, col in cells_without_start]
        if used_goal_bridge:
            points.append(goal)
        elif points:
            points[-1] = goal
        else:
            points = [goal]

        return PlannedGridPath(
            success=True,
            waypoints=tuple(points),
            distance_m=path_length((start, *points)),
            traversed_cells=len(full_cell_path),
            used_obstacle_exit=allow_obstacle_exit,
            used_goal_bridge=used_goal_bridge,
        )

    def plan_through_waypoints(
        self,
        start: Vec2,
        goals: tuple[Vec2, ...],
        *,
        use_inflated: bool = True,
        allow_final_goal_occupied: bool = False,
    ) -> PlannedGridPath:
        if not goals:
            return PlannedGridPath(True, (), 0.0, 0)

        current = start
        combined: list[Vec2] = []
        traversed_cells = 0
        used_obstacle_exit = False
        used_goal_bridge = False

        for index, goal in enumerate(goals):
            segment = self.plan(
                current,
                goal,
                use_inflated=use_inflated,
                allow_goal_occupied=allow_final_goal_occupied and index == len(goals) - 1,
            )
            if not segment.success:
                return PlannedGridPath(False, (), 0.0, traversed_cells)
            combined.extend(segment.waypoints)
            traversed_cells += segment.traversed_cells
            used_obstacle_exit = used_obstacle_exit or segment.used_obstacle_exit
            used_goal_bridge = used_goal_bridge or segment.used_goal_bridge
            current = goal

        return PlannedGridPath(
            success=True,
            waypoints=tuple(combined),
            distance_m=path_length((start, *combined)),
            traversed_cells=traversed_cells,
            used_obstacle_exit=used_obstacle_exit,
            used_goal_bridge=used_goal_bridge,
        )

    def _find_nearest_free_cell(
        self,
        start_cell: tuple[int, int],
        *,
        use_inflated: bool,
    ) -> tuple[int, int] | None:
        if not self._cell_on_map(start_cell):
            return None
        if not self.occupancy_map.is_blocked(*start_cell, use_inflated=use_inflated):
            return start_cell

        queue: deque[tuple[int, int]] = deque([start_cell])
        visited = {start_cell}
        while queue:
            row, col = queue.popleft()
            for d_row, d_col, _ in self.DIRECTIONS:
                neighbor = (row + d_row, col + d_col)
                if neighbor in visited or not self._cell_on_map(neighbor):
                    continue
                if not self.occupancy_map.is_blocked(*neighbor, use_inflated=use_inflated):
                    return neighbor
                visited.add(neighbor)
                queue.append(neighbor)
        return None

    def _cell_on_map(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.occupancy_map.height_px and 0 <= col < self.occupancy_map.width_px

    def _diagonal_cuts_blocked_corner(
        self,
        cell: tuple[int, int],
        d_row: int,
        d_col: int,
        *,
        use_inflated: bool,
    ) -> bool:
        row, col = cell
        side_cells = ((row + d_row, col), (row, col + d_col))
        return any(
            not self._cell_on_map(side_cell)
            or self.occupancy_map.is_blocked(*side_cell, use_inflated=use_inflated)
            for side_cell in side_cells
        )

    @staticmethod
    def _heuristic(cell: tuple[int, int], goal: tuple[int, int]) -> float:
        d_row = abs(goal[0] - cell[0])
        d_col = abs(goal[1] - cell[1])
        diagonal = min(d_row, d_col)
        straight = max(d_row, d_col) - diagonal
        return diagonal * SQRT2 + straight

    @staticmethod
    def _reconstruct_path(
        came_from: dict[tuple[int, int], tuple[int, int]],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [goal]
        current = goal
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _compress_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(cells) <= 2:
            return cells
        compressed = [cells[0]]
        prev_step = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
        for index in range(1, len(cells) - 1):
            step = (cells[index + 1][0] - cells[index][0], cells[index + 1][1] - cells[index][1])
            if step != prev_step:
                compressed.append(cells[index])
                prev_step = step
        compressed.append(cells[-1])
        return compressed
