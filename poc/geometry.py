from __future__ import annotations

from itertools import pairwise
from math import hypot
from typing import Sequence

Vec2 = tuple[float, float]


def distance(a: Vec2, b: Vec2) -> float:
    return hypot(b[0] - a[0], b[1] - a[1])


def path_length(points: Sequence[Vec2]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(distance(a, b) for a, b in pairwise(points))


def interpolate(start: Vec2, end: Vec2, ratio: float) -> Vec2:
    ratio = min(max(ratio, 0.0), 1.0)
    return (
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
    )


def advance_along_path(start: Vec2, waypoints: Sequence[Vec2], distance_travelled: float) -> Vec2:
    if not waypoints:
        return start

    points = [start, *waypoints]
    if distance_travelled <= 0.0:
        return start

    remaining = distance_travelled
    for segment_start, segment_end in pairwise(points):
        segment_length = distance(segment_start, segment_end)
        if segment_length == 0.0:
            continue
        if remaining <= segment_length:
            return interpolate(segment_start, segment_end, remaining / segment_length)
        remaining -= segment_length

    return points[-1]


def mirror_x(point: Vec2) -> Vec2:
    return (-point[0], point[1])


def point_to_segment_distance(point: Vec2, start: Vec2, end: Vec2) -> float:
    segment_dx = end[0] - start[0]
    segment_dy = end[1] - start[1]
    segment_length_sq = segment_dx * segment_dx + segment_dy * segment_dy
    if segment_length_sq == 0.0:
        return distance(point, start)

    projection = (
        (point[0] - start[0]) * segment_dx + (point[1] - start[1]) * segment_dy
    ) / segment_length_sq
    projection = min(max(projection, 0.0), 1.0)
    closest = (
        start[0] + projection * segment_dx,
        start[1] + projection * segment_dy,
    )
    return distance(point, closest)
