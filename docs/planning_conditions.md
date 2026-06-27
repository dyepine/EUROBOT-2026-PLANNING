# Planning Conditions

Last updated: `2026-06-27`

This document describes the current behavior of the planner and simulator.
When this document and the implementation disagree, the implementation is the
source of truth:

- `poc/planner.py`
- `poc/simulator.py`
- `poc/scoring.py`
- `poc/config.py`
- `poc/semantic_map.py`

## 1. Match Model

- A match lasts `100s`.
- Simulation uses a configurable fixed timestep `dt`.
- The planner runs on an occupancy grid and uses 8-direction A*.
- The planner generates only legal semantic actions.
- Each selected action is executed as a macro-action with travel and service
  phases.
- Once endgame starts, regular scoring actions are no longer generated.

## 2. Planner Actions

The planner ranks the following semantic action types:

- `PICK`
- `DEPOSIT`
- `ATTACK_DEPOSIT`
- `DO_THERMOMETER`
- `START_ENDGAME`
- `WAIT`
- `WAIT_FOR_CHILL`

`WAIT_FOR_CHILL` is used after `main_pipeline_deadline` when the scoring phase
is closed but it is still too early to move toward the endgame chill point.

## 3. Endgame

The current endgame rule is deterministic:

- `main_pipeline_deadline = 80.0`
- `chill_end = 90.0`
- after starting endgame, the robot:
  - moves to `chill_point`;
  - waits until `90.0`;
  - returns home through `home_waypoints`;
  - finishes with a short `grip_rotate` action.

Configuration:

- `blue`
  - `chill_point = (0.55, 0.25)`
  - `home_waypoints = ((1.05, 0.35), (1.12, 0.75))`
- `yellow`
  - `chill_point = (-0.55, 0.25)`
  - `home_waypoints = ((-1.05, 0.35), (-1.12, 0.75))`

The planner does not merely penalize late actions. It checks whether an action
can finish before the chill window and whether the robot can still complete the
route home afterward.

## 4. Timing

Current calibrated service timings:

- `move_overhead = 0.18`
- `pick_duration = 2.75`
- `deposit_duration = 5.33`
- `thermometer_duration = 0.35`
- `attack_duration = 0.0`
- `wait_duration = 1.0`
- `align_duration = 0.85`
- `grip_rotate_duration = 0.3`

Geometry parameters:

- `robot_separation_radius = 0.45`
- `interaction_radius = 0.08`

The base robot speed is defined in the entity model rather than in this
document.

## 5. Utility Score

The planner uses:

`score = 1.6 * reward - 1.0 * time_cost - 2.2 * risk - 1.3 * blocking_penalty + 1.2 * swing`

Weights:

- `reward_weight = 1.6`
- `time_weight = 1.0`
- `risk_weight = 2.2`
- `blocking_weight = 1.3`
- `swing_weight = 1.2`

## 6. PICK

`PICK` is proposed only when:

- `t < 80.0`;
- `robot.load == 0`;
- the source is available in time and still has items;
- a valid route exists;
- the action fits into the endgame window.

A source becomes `disturbed` only on real contact, not when the action is
selected.

## 7. DEPOSIT

`DEPOSIT` is proposed only when:

- `t < 80.0`;
- `robot.load > 0`;
- a valid route exists;
- the action fits into the endgame window.

Current zone rules:

- `STORAGE`: depositing into an already occupied storage zone is not allowed.
- `HOME`: regular deposits are allowed only while the home zone is empty.
- `HOME`: once a home pile exists, the planner no longer generates regular
  `DEPOSIT` actions there.

Endgame exception:

- If a robot reaches home through the endgame route while still carrying items,
  the simulator may score a final home drop at arrival.
- If `HOME` already contains our pile, the planner no longer plans another
  return through `START_ENDGAME`.

In short:

- regular runtime deposits into an occupied home are forbidden;
- the endgame arrival may complete the home pile;
- repeated returns into an already occupied home are not planned.

## 8. ATTACK_DEPOSIT

`ATTACK_DEPOSIT` is proposed only when:

- `robot.load == 0`;
- the zone is not protected by `protected_for`;
- the zone actually contains opponent items;
- a valid attack route exists;
- the action fits into the endgame window;
- intermediate `semantic_waypoints` are free in the inflated map;
- the opponent robot is not too close to the target or final approach segment.

Protected zones:

- `16` is protected for `blue`;
- `26` is protected for `yellow`.

These zones cannot be attacked.

`ATTACK_10` also has side-specific route prerequisites:

- for `blue`, source `13` must be removed;
- for `yellow`, source `23` must be removed.

## 9. THERMOMETER

`DO_THERMOMETER` is proposed only when:

- `t < 80.0`;
- this side has not completed the thermometer yet;
- the thermometer lane is clear;
- the action fits into the endgame window.

The current "lane is clear" check means:

- the blocking source (`13` for blue, `23` for yellow) has been removed;
- zone `10` is empty;
- the side-specific blocking storage (`16` or `26`) is empty.

The thermometer has both `done` and `doing` states in the simulator and
visualization.

## 10. Mars Entities

In the current simplified model:

- each side has `3` Mars entities;
- they start near the upper home area;
- they are released late in the match and aim to arrive by `100s`;
- pantry points are awarded in final scoring.

Scoring:

- `+5` for each Mars entity that reaches a pantry;
- `+10` bonus when all Mars entities for a side are eating.

Simplified collision rule:

- If a Mars entity collides with either large robot during the match, that Mars
  entity does not receive pantry credit.

Notes:

- The planner does not yet dynamically avoid Mars entities.
- Mars handling is a simplified scoring interaction, not full physics.

## 11. Home Return and Points

Final points for the large robot:

- `+5` for partial return;
- another `+5` for full return.

Thresholds:

- `partial`: `distance <= 0.25`
- `full`: `distance <= 0.12`

Additional zone scoring:

- `HOME` gives `+2` per item;
- `STORAGE` gives `+3` per item;
- `STORAGE` gives `+5` for majority ownership.

## 12. Runtime Collisions and Waiting

The only hard runtime separation constraint is between the two large robots:

- if a movement step violates `robot_separation_radius`, one robot yields.

Tie-break:

- if actions started at the same time, `yellow` yields.

Mars entities are handled by a separate simplified interaction rule.

## 13. Important Behavioral Constraints

The most important non-obvious rules are:

- home cannot be filled repeatedly during the main match phase;
- final home-drop is allowed only through endgame arrival;
- the thermometer lane is a structural constraint;
- Mars entities affect final score but are not yet part of planner avoidance;
- the planner routes through grid/A* and route-specific waypoints, not through a
  straight-line shortcut.
