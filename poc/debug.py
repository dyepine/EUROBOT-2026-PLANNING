from __future__ import annotations

from poc.actions import Action
from poc.entities import Side
from poc.policy_mapping import normalized_action_label, normalized_target_id


def action_debug_row(action: Action, side: Side) -> dict[str, float | int | str | bool | None]:
    row = action.debug_row()
    row["policy_action"] = normalized_action_label(action, side)
    row["policy_target_id"] = normalized_target_id(action.target_id, action.type, side)
    return row


def planning_debug_payload(
    *,
    time: float,
    side: Side,
    reason: str,
    chosen_action: Action,
    ranked_actions: list[Action],
) -> dict[str, object]:
    return {
        "time": round(time, 3),
        "side": side.value,
        "reason": reason,
        "chosen_action": action_debug_row(chosen_action, side),
        "candidates": [action_debug_row(action, side) for action in ranked_actions],
    }
