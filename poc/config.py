from __future__ import annotations

from dataclasses import dataclass, field

from poc.entities import DepositType, Side
from poc.geometry import Vec2


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    home_item_points: int = 2
    storage_item_points: int = 3
    storage_majority_bonus: int = 5
    home_capacity: int = 4
    finish_intersection_points: int = 5
    finish_full_bonus_points: int = 5
    mars_pantry_points: int = 5
    mars_all_eating_bonus: int = 10
    mars_collision_penalty: int = 50
    mars_robot_interaction_radius: float = 0.20
    mars_robot_stop_lookahead_distance: float = 0.15
    finish_full_tolerance: float = 0.12
    finish_intersection_tolerance: float = 0.25

    @property
    def finish_full_points(self) -> int:
        return self.finish_intersection_points + self.finish_full_bonus_points

    def finish_points(self, distance_to_home: float) -> int:
        if distance_to_home <= self.finish_full_tolerance:
            return self.finish_full_points
        if distance_to_home <= self.finish_intersection_tolerance:
            return self.finish_intersection_points
        return 0


@dataclass(frozen=True, slots=True)
class ActionTimingConfig:
    # Calibrated against YellowSideTree.xml so the main pipeline reaches the
    # chill zone at roughly 70 seconds before the final wait-home stage.
    move_overhead: float = 0.18
    pick_duration: float = 2.75
    deposit_duration: float = 5.33
    thermometer_duration: float = 0.35
    attack_duration: float = 0.0
    wait_duration: float = 1.0
    align_duration: float = 0.85
    grip_rotate_duration: float = 0.3
    local_replan_distance: float = 1.0
    robot_separation_radius: float = 0.45
    attack_enemy_extra_margin: float = 0.16
    route_replan_enemy_margin: float = 0.10
    escape_half_angle_deg: float = 45.0
    interaction_radius: float = 0.08

    @property
    def route_replan_enemy_distance(self) -> float:
        return self.robot_separation_radius + self.route_replan_enemy_margin

    @property
    def attack_enemy_block_radius(self) -> float:
        return self.robot_separation_radius + self.attack_enemy_extra_margin


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    reward_weight: float = 1.6
    time_weight: float = 1.0
    risk_weight: float = 2.2
    blocking_weight: float = 1.3
    swing_weight: float = 1.2


@dataclass(frozen=True, slots=True)
class EndgameParameters:
    main_pipeline_deadline: float = 80.0
    chill_end: float = 90.0
    chill_margin: float = 1.0
    home_margin: float = 1.0
    grip_rotate_duration: float = 0.2
    blue_chill_point: Vec2 = (0.55, 0.25)
    blue_home_waypoints: tuple[Vec2, ...] = ((1.12, 0.75),)
    yellow_chill_point: Vec2 = (-0.55, 0.25)
    yellow_home_waypoints: tuple[Vec2, ...] = ((-1.12, 0.75),)
    score: ScoreConfig = field(default_factory=ScoreConfig)

    def chill_point_for(self, side: Side) -> Vec2:
        return self.blue_chill_point if side is Side.BLUE else self.yellow_chill_point

    def home_waypoints_for(self, side: Side) -> tuple[Vec2, ...]:
        if side is Side.BLUE:
            return self.blue_home_waypoints
        return self.yellow_home_waypoints


DEFAULT_SCORE_CONFIG = ScoreConfig()
DEFAULT_ENDGAME_PARAMETERS = EndgameParameters(score=DEFAULT_SCORE_CONFIG)
