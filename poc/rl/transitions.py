from __future__ import annotations

from dataclasses import dataclass

from poc.domain.actions import ActionType
from poc.domain.entities import Side
from poc.rl.policy_mapping import normalized_action_label
from poc.rl.action_space import build_rl_policy_step
from poc.rl.encoder import RLObservation, RLObservationConfig
from poc.rl.tokens import DEFAULT_ACTION_SPACE, RLActionSpace

@dataclass(frozen=True, slots=True)
class RLTransition:
    side: str
    time: float
    chosen_action: str
    chosen_action_index: int
    action_mask: tuple[int, ...]
    observation: RLObservation
    reward: float
    next_observation: RLObservation
    next_action_mask: tuple[int, ...]
    done: bool
    score_diff_before: float
    score_diff_after: float


def build_rl_transitions_from_match_result(
    result,
    *,
    side: Side | None = None,
    config: RLObservationConfig | None = None,
    action_space: RLActionSpace = DEFAULT_ACTION_SPACE,
    thermometer_reward_bonus: float = 3.0,
    terminal_win_bonus: float = 2.0,
    terminal_draw_bonus: float = 0.0,
    terminal_loss_bonus: float = -2.0,
) -> list[RLTransition]:
    transitions: list[RLTransition] = []
    decision_log = list(getattr(result, "decision_log", ()))
    sides = (side,) if side is not None else (Side.BLUE, Side.YELLOW)
    final_score_diff_by_side = _final_score_diff_by_side(result)
    zero_mask = tuple(0 for _ in action_space.tokens)

    for current_side in sides:
        side_decisions = [entry for entry in decision_log if entry.side is current_side]
        for index, entry in enumerate(side_decisions):
            next_entry = side_decisions[index + 1] if index + 1 < len(side_decisions) else None
            current_step = build_rl_policy_step(
                entry.observation,
                config=config,
                action_space=action_space,
            )
            if next_entry is None:
                next_step = None
                done = True
                score_diff_after = final_score_diff_by_side[current_side]
                next_observation = current_step.observation
                next_action_mask = zero_mask
            else:
                next_step = build_rl_policy_step(
                    next_entry.observation,
                    config=config,
                    action_space=action_space,
                )
                done = False
                score_diff_after = next_entry.score_diff_before
                next_observation = next_step.observation
                next_action_mask = next_step.action_mask

            chosen_action = normalized_action_label(entry.chosen_action, current_side)
            chosen_action_index = action_space.encode(chosen_action)
            reward = score_diff_after - entry.score_diff_before
            if _transition_completed_thermometer(entry, next_entry):
                reward += thermometer_reward_bonus
            if done:
                reward += _terminal_bonus(
                    score_diff_after,
                    win_bonus=terminal_win_bonus,
                    draw_bonus=terminal_draw_bonus,
                    loss_bonus=terminal_loss_bonus,
                )
            transitions.append(
                RLTransition(
                    side=current_side.value,
                    time=entry.time,
                    chosen_action=chosen_action,
                    chosen_action_index=chosen_action_index,
                    action_mask=current_step.action_mask,
                    observation=current_step.observation,
                    reward=reward,
                    next_observation=next_observation,
                    next_action_mask=next_action_mask,
                    done=done,
                    score_diff_before=entry.score_diff_before,
                    score_diff_after=score_diff_after,
                )
            )
    return transitions


def _final_score_diff_by_side(result) -> dict[Side, float]:
    summary = getattr(result, "summary", {})
    our_side = Side(getattr(result, "our_side"))
    score_diff = float(summary.get("score_diff", 0.0))
    return {
        our_side: score_diff,
        our_side.opponent(): -score_diff,
    }


def _transition_completed_thermometer(entry, next_entry) -> bool:
    if entry.chosen_action.type is not ActionType.DO_THERMOMETER:
        return False
    next_state = next_entry.observation.state if next_entry is not None else None
    return next_state is not None and next_state.thermometer.is_done_for_side(entry.side)


def _terminal_bonus(
    score_diff: float,
    *,
    win_bonus: float,
    draw_bonus: float,
    loss_bonus: float,
) -> float:
    if score_diff > 0.0:
        return win_bonus
    if score_diff < 0.0:
        return loss_bonus
    return draw_bonus


def transition_to_record(transition: RLTransition) -> dict[str, object]:
    return {
        "side": transition.side,
        "time": transition.time,
        "chosen_action": transition.chosen_action,
        "chosen_action_index": transition.chosen_action_index,
        "action_mask": list(transition.action_mask),
        "observation": {
            "perspective": transition.observation.perspective,
            "global_features": transition.observation.global_features,
            "source_features": transition.observation.source_features,
            "deposit_features": transition.observation.deposit_features,
            "flat_features": transition.observation.flat_features,
        },
        "reward": transition.reward,
        "next_observation": {
            "perspective": transition.next_observation.perspective,
            "global_features": transition.next_observation.global_features,
            "source_features": transition.next_observation.source_features,
            "deposit_features": transition.next_observation.deposit_features,
            "flat_features": transition.next_observation.flat_features,
        },
        "next_action_mask": list(transition.next_action_mask),
        "done": transition.done,
        "score_diff_before": transition.score_diff_before,
        "score_diff_after": transition.score_diff_after,
    }
