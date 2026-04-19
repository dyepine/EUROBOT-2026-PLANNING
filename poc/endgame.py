from __future__ import annotations

from dataclasses import dataclass

from poc.entities import Side
from poc.geometry import Vec2, path_length


@dataclass(slots=True)
class EndgameConfig:
    main_pipeline_deadline: float
    chill_end: float
    chill_point: Vec2
    home_waypoints: tuple[Vec2, ...]
    chill_margin: float = 1.0
    home_margin: float = 1.0
    home_full_tolerance: float = 0.12
    home_partial_tolerance: float = 0.25
    grip_rotate_duration: float = 0.2

    @property
    def final_home_point(self) -> Vec2:
        return self.home_waypoints[-1]


def build_endgame_config(side: Side) -> EndgameConfig:
    if side is Side.BLUE:
        return EndgameConfig(
            main_pipeline_deadline=80.0,
            chill_end=90.0,
            chill_point=(0.55, 0.25),
            home_waypoints=((1.05, 0.35), (1.12, 0.75)),
        )
    return EndgameConfig(
        main_pipeline_deadline=80.0,
        chill_end=90.0,
        chill_point=(-0.55, 0.25),
        home_waypoints=((-1.05, 0.35), (-1.12, 0.75)),
    )


def travel_duration_for_waypoints(
    start: Vec2,
    waypoints: tuple[Vec2, ...],
    speed: float,
    move_overhead: float,
) -> float:
    if not waypoints:
        return 0.0
    distance_total = path_length((start, *waypoints))
    return distance_total / speed + move_overhead * len(waypoints)


def estimate_endgame_duration(
    now: float,
    start: Vec2,
    speed: float,
    move_overhead: float,
    config: EndgameConfig,
) -> dict[str, float]:
    to_chill = travel_duration_for_waypoints(start, (config.chill_point,), speed, move_overhead)
    arrival_at_chill = now + to_chill
    wait = max(0.0, config.chill_end - arrival_at_chill)
    home_travel = travel_duration_for_waypoints(
        config.chill_point,
        config.home_waypoints,
        speed,
        move_overhead,
    )
    total = to_chill + wait + home_travel + config.grip_rotate_duration
    return {
        "to_chill": to_chill,
        "wait": wait,
        "home_travel": home_travel,
        "grip_rotate": config.grip_rotate_duration,
        "total": total,
    }


def can_finish_scoring_action(
    now: float,
    action_duration: float,
    action_end_position: Vec2,
    speed: float,
    move_overhead: float,
    config: EndgameConfig,
    match_end: float,
) -> bool:
    to_chill_after_action = travel_duration_for_waypoints(
        action_end_position,
        (config.chill_point,),
        speed,
        move_overhead,
    )
    home_travel = travel_duration_for_waypoints(
        config.chill_point,
        config.home_waypoints,
        speed,
        move_overhead,
    ) + config.grip_rotate_duration
    finishes_before_chill = now + action_duration + to_chill_after_action <= config.chill_end - config.chill_margin
    finishes_home = config.chill_end + home_travel <= match_end - config.home_margin
    return finishes_before_chill and finishes_home
