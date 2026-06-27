from __future__ import annotations

from dataclasses import dataclass

from poc.domain.config import DEFAULT_ENDGAME_PARAMETERS, EndgameParameters, ScoreConfig
from poc.domain.entities import Side
from poc.domain.geometry import Vec2


@dataclass(slots=True)
class EndgameConfig:
    main_pipeline_deadline: float
    chill_end: float
    chill_point: Vec2
    home_waypoints: tuple[Vec2, ...]
    chill_margin: float
    home_margin: float
    grip_rotate_duration: float
    score: ScoreConfig

    @property
    def final_home_point(self) -> Vec2:
        return self.home_waypoints[-1]

    @property
    def home_full_tolerance(self) -> float:
        return self.score.finish_full_tolerance

    @property
    def home_partial_tolerance(self) -> float:
        return self.score.finish_intersection_tolerance

    def finish_points(self, distance_to_home: float) -> int:
        return self.score.finish_points(distance_to_home)


def build_endgame_config(
    side: Side,
    parameters: EndgameParameters = DEFAULT_ENDGAME_PARAMETERS,
) -> EndgameConfig:
    return EndgameConfig(
        main_pipeline_deadline=parameters.main_pipeline_deadline,
        chill_end=parameters.chill_end,
        chill_point=parameters.chill_point_for(side),
        home_waypoints=parameters.home_waypoints_for(side),
        chill_margin=parameters.chill_margin,
        home_margin=parameters.home_margin,
        grip_rotate_duration=parameters.grip_rotate_duration,
        score=parameters.score,
    )
