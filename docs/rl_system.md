# RL System

Last updated: `2026-06-27`

This document describes the current reinforcement-learning stack in `poc/`.
When this document and the implementation disagree, the implementation is the
source of truth:

- `poc/rl_infra.py`
- `poc/rl_model.py`
- `poc/rl_selfplay.py`
- `poc/rl_train.py`
- `poc/observations.py`

## 1. Architecture

The RL loop is deliberately built on top of the symbolic planner:

- the planner generates legal semantic actions;
- the simulator builds a `DecisionObservation` at each decision point;
- the RL encoder converts that observation and ranked actions into an
  `action_mask`;
- the policy selects an action from a fixed discrete action space;
- the simulator executes the selected action as a semi-MDP macro-action;
- the RL adapter builds transitions from decision observations and score deltas;
- training uses masked discrete PPO.

The policy does not control low-level motion and does not generate routes. It
chooses among planner-approved semantic candidates.

## 2. Observation

The simulator owns decision-point context. It records `DecisionObservation`
snapshots containing the current `GameState`, side perspective, ranked actions,
previous tick snapshot, and `dt`.

`build_rl_observation(...)` encodes a `DecisionObservation` into an
`RLObservation` with:

- `perspective`
- `global_features`
- `source_features`
- `deposit_features`
- `flat_features`

### 2.1. Global Features

The current global block includes:

- `time_remaining_norm`
- `our_x_norm`
- `our_y_norm`
- `enemy_x_norm`
- `enemy_y_norm`
- `our_load_norm`
- `enemy_vel_x_norm`
- `enemy_vel_y_norm`
- `enemy_speed_norm`
- `enemy_max_speed_seen_norm`
- `our_endgame_started`
- `enemy_endgame_started`
- `thermometer_done_for_us`
- `thermometer_done_for_enemy`
- `thermometer_lane_clear_for_us`
- `thermometer_available_for_us`
- `thermometer_available_for_enemy`
- `time_since_our_lane_clear_change_norm`
- `time_since_enemy_lane_clear_change_norm`
- `time_since_thermometer_state_change_norm`

Notes:

- `enemy_max_speed_seen_norm` is a running maximum of observed enemy speed, not
  hidden access to the enemy configuration.
- `enemy_vel_x_norm`, `enemy_vel_y_norm`, and `enemy_speed_norm` may include
  observation noise.
- Enemy velocity can optionally include a short `-our_velocity` component after
  our movement starts, approximating a localization/lidar spike caused by
  self-motion.
- Exact `enemy_load` is not exposed to the policy.
- Coordinates are mirrored along `x` for `yellow`, making the policy
  side-invariant.

### 2.2. Source Features

The order is fixed:

- `OUR_SOURCE_1..4`
- `ENEMY_SOURCE_1..4`

Each source provides:

- `available_items_norm`
- `available_now`
- `state_untouched`
- `state_disturbed`
- `map_footprint_enabled`
- `last_items_delta_norm`
- `time_since_last_change_norm`
- `last_change_was_disturb_like`

### 2.3. Deposit Features

The order is fixed:

- `OUR_HOME`
- `ENEMY_HOME`
- `OUR_STORAGE_5`
- `OUR_STORAGE_6`
- `OUR_STORAGE_7`
- `NEUTRAL_STORAGE_1`
- `NEUTRAL_STORAGE_0`
- `ENEMY_STORAGE_5`
- `ENEMY_STORAGE_6`
- `ENEMY_STORAGE_7`

Each deposit zone provides:

- `our_items_norm`
- `enemy_items_norm`
- `protected_for_our`
- `protected_for_enemy`
- `map_footprint_enabled`
- `attack_delta_raw`
- `deposit_x1_delta_raw`
- `deposit_x1_valid`
- `deposit_x2_delta_raw`
- `deposit_x2_valid`
- `deposit_x3_delta_raw`
- `deposit_x3_valid`
- `deposit_x4_delta_raw`
- `deposit_x4_valid`
- `last_score_diff_delta_raw`
- `time_since_last_score_change_norm`
- `last_change_by_our`
- `last_change_by_enemy`

### 2.4. Missing Observation Inputs

The policy currently does not receive:

- exact `enemy_load`;
- `thermometer_doing_*` status;
- raw `t-1`, `t-2` history stacks;
- explicit Mars-aware features.

Compact temporal and event memory is already present:

- source features store the latest item delta and time since change;
- deposit features store the latest score-diff delta, actor of the latest
  change, and time since change;
- thermometer features store timers for global state and lane-clear changes.

## 3. Action Space

`DEFAULT_ACTION_SPACE` is fixed and includes:

- `PICK_OUR_SOURCE_*`
- `PICK_ENEMY_SOURCE_*`
- `DEPOSIT_<TARGET>_X1..X4`
- `ATTACK_<TARGET>`
- `THERMOMETER`
- `START_ENDGAME`
- `WAIT`
- `WAIT_FOR_CHILL`

The policy should not choose illegal actions. `MaskedPolicyValueNet` receives an
`action_mask` and sets illegal logits to `-1e9`.

## 4. Policy Model

`poc/rl_model.py` defines a compact actor-critic MLP:

- input: flattened `flat_features`;
- backbone: `Linear + Tanh`;
- policy head: logits over the fixed action space;
- value head: scalar value estimate.

This is a masked discrete PPO policy, not a candidate-pair `Q(s, a)` model.

## 5. Transition Semantics

`Simulator.run()` returns a domain `MatchResult` with a `decision_log`, not
RL-specific transitions. `poc.rl_infra.build_rl_transitions_from_match_result`
builds `RLTransition` records in the RL layer.

Each transition contains:

- `observation`
- `action_mask`
- `chosen_action`
- `chosen_action_index`
- `reward`
- `next_observation`
- `next_action_mask`
- `done`
- `score_diff_before`
- `score_diff_after`

These are semi-MDP steps:

- the policy acts at semantic decision points, not every `dt`;
- reward covers everything that happened before the next decision point for the
  same side.

## 6. Reward

Base reward:

- `score_diff_after - score_diff_before`

Terminal bonus:

- `+2` for a win;
- `0` for a draw;
- `-2` for a loss.

Optional shaping:

- `thermometer_reward_bonus`

The shaping bonus is added only after a successful `THERMOMETER`.

## 7. Self-Play Training

Main training command:

```bash
python -m poc.rl_train --output-dir runs/ppo_run
```

Training can use:

- self-play snapshots;
- scripted opponents;
- randomized and stochastic scripted opponents;
- optional enemy speed jitter;
- optional multiprocessing rollout/evaluation workers.

Saved artifacts:

- `latest.pt`
- `best_winrate.pt`
- `best_score_diff.pt`
- `training_state.pt`
- `metrics.jsonl`

Two startup modes are supported:

- `--resume training_state.pt`
  Restores model, optimizer, opponent pool, RNG, and update counter.
- `--init-from-checkpoint best_score_diff.pt`
  Loads only model weights and starts a fresh run with a new optimizer/pool.

## 8. Opponents

Training and evaluation can use:

- `nearest_greedy`
- `aggressive`
- `thermo_first`
- `storage_first`
- `home_safe`
- `uniform_random`
- `stochastic_planner`
- `randomized_aggressive`
- `randomized_nearest`
- `randomized_thermo`
- `randomized_storage`
- `randomized_home`
- `randomized_mixed`

Names with `@seed` materialize distinct behavioral variants.

## 9. RL Assumptions

The environment already exposes several useful structures to the policy,
directly or indirectly:

- side-invariant semantic slots;
- stochastic enemy speed scale;
- thermometer as a distinct strategic action;
- mandatory `WAIT_FOR_CHILL / START_ENDGAME` endgame flow;
- Mars entities in simulator state and final scoring.

Important simplifications:

- Mars entities are not part of dynamic planner avoidance yet;
- Mars interaction is a simplified final/scoring rule;
- the policy does not receive explicit Mars-aware inputs;
- older RL checkpoints can become incompatible when `flat_features` changes.

## 10. Practical Compatibility Notes

Changing `GLOBAL_FEATURE_ORDER`, `SOURCE_FEATURE_ORDER`, or
`DEPOSIT_FEATURE_ORDER` almost always means:

- observation size changes;
- old `best_*.pt` and `latest.pt` checkpoints will not load into the new model
  without migration.

Changing only reward or opponent mix usually leaves old weights structurally
compatible.

## 11. Feature Cleanup Notes

Most baseline cleanup has already been done.

Removed from observation:

- `enemy_load_norm`
- fixed-map source/deposit coordinates;
- redundant normalized score-delta duplicates;
- flags fully derivable from other features.

Current observation size:

- `237`

Future cleanup should be driven by ablation runs rather than intuition alone.
Honest candidates for review:

- `enemy_vel_x_norm`
- `enemy_vel_y_norm`
- `enemy_speed_norm`
- `enemy_max_speed_seen_norm`
- `map_footprint_enabled`
- possible move toward history stacks `t, t-1, t-2`
