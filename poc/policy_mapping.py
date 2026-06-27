from __future__ import annotations

from poc.actions import Action, ActionType
from poc.entities import Side

BLUE_SOURCE_IDS = {11, 12, 13, 14}
YELLOW_SOURCE_IDS = {21, 22, 23, 24}
BLUE_SIDE_STORAGE_IDS = {15, 16, 17}
YELLOW_SIDE_STORAGE_IDS = {25, 26, 27}
NEUTRAL_STORAGE_IDS = {1, 10}
BLUE_HOME_ID = 101
YELLOW_HOME_ID = 201


def normalized_target_id(
    target_id: int | None,
    action_type: ActionType,
    perspective: Side,
) -> str | None:
    if target_id is None:
        return None
    if action_type is ActionType.PICK:
        return normalized_source_id(target_id, perspective)
    if action_type in (ActionType.DEPOSIT, ActionType.ATTACK_DEPOSIT):
        return normalized_deposit_id(target_id, perspective)
    if action_type is ActionType.DO_THERMOMETER:
        return "THERMOMETER"
    return None


def normalized_action_label(action: Action, perspective: Side) -> str:
    if action.type is ActionType.PICK:
        target = normalized_target_id(action.target_id, action.type, perspective)
        return f"PICK_{target}" if target is not None else action.label
    if action.type is ActionType.DEPOSIT:
        target = normalized_target_id(action.target_id, action.type, perspective)
        deposit_count = action.metadata.get("deposit_count")
        if target is None:
            return action.label
        if deposit_count is None:
            return f"DEPOSIT_{target}"
        return f"DEPOSIT_{target}_X{int(deposit_count)}"
    if action.type is ActionType.ATTACK_DEPOSIT:
        target = normalized_target_id(action.target_id, action.type, perspective)
        return f"ATTACK_{target}" if target is not None else action.label
    return action.label


def normalized_source_id(source_id: int, perspective: Side) -> str:
    source_side = _source_side(source_id)
    relation = "OUR" if source_side is perspective else "ENEMY"
    return f"{relation}_SOURCE_{source_id % 10}"


def normalized_deposit_id(deposit_id: int, perspective: Side) -> str:
    if deposit_id in NEUTRAL_STORAGE_IDS:
        return f"NEUTRAL_STORAGE_{deposit_id % 10}"
    if deposit_id in (BLUE_HOME_ID, YELLOW_HOME_ID):
        deposit_side = Side.BLUE if deposit_id == BLUE_HOME_ID else Side.YELLOW
        relation = "OUR" if deposit_side is perspective else "ENEMY"
        return f"{relation}_HOME"
    deposit_side = _deposit_side(deposit_id)
    relation = "OUR" if deposit_side is perspective else "ENEMY"
    return f"{relation}_STORAGE_{deposit_id % 10}"


def raw_source_id(normalized_id: str, perspective: Side) -> int:
    relation, _, slot_text = normalized_id.split("_", maxsplit=2)
    slot = int(slot_text)
    source_side = perspective if relation == "OUR" else perspective.opponent()
    return 10 + slot if source_side is Side.BLUE else 20 + slot


def raw_deposit_id(normalized_id: str, perspective: Side) -> int:
    parts = normalized_id.split("_")
    if len(parts) == 2 and parts[1] == "HOME":
        deposit_side = perspective if parts[0] == "OUR" else perspective.opponent()
        return BLUE_HOME_ID if deposit_side is Side.BLUE else YELLOW_HOME_ID
    relation, _, slot_text = parts
    slot = int(slot_text)
    if relation == "NEUTRAL":
        if slot not in {0, 1}:
            raise ValueError(f"Unsupported neutral storage slot: {slot}")
        return 10 if slot == 0 else 1
    deposit_side = perspective if relation == "OUR" else perspective.opponent()
    if slot not in {5, 6, 7}:
        raise ValueError(f"Unsupported side storage slot: {slot}")
    return 10 + slot if deposit_side is Side.BLUE else 20 + slot


def policy_metadata_for_source(source_id: int) -> dict[str, object]:
    source_side = _source_side(source_id)
    return {
        "policy_slot": source_id % 10,
        "policy_relation_by_side": {
            Side.BLUE.value: "our" if source_side is Side.BLUE else "enemy",
            Side.YELLOW.value: "our" if source_side is Side.YELLOW else "enemy",
        },
        "policy_id_by_side": {
            Side.BLUE.value: normalized_source_id(source_id, Side.BLUE),
            Side.YELLOW.value: normalized_source_id(source_id, Side.YELLOW),
        },
    }


def policy_metadata_for_deposit(deposit_id: int) -> dict[str, object]:
    if deposit_id in NEUTRAL_STORAGE_IDS:
        relation_by_side = {Side.BLUE.value: "neutral", Side.YELLOW.value: "neutral"}
        policy_id_by_side = {
            Side.BLUE.value: normalized_deposit_id(deposit_id, Side.BLUE),
            Side.YELLOW.value: normalized_deposit_id(deposit_id, Side.YELLOW),
        }
        return {
            "policy_slot": deposit_id % 10,
            "policy_relation_by_side": relation_by_side,
            "policy_id_by_side": policy_id_by_side,
        }
    if deposit_id in (BLUE_HOME_ID, YELLOW_HOME_ID):
        deposit_side = Side.BLUE if deposit_id == BLUE_HOME_ID else Side.YELLOW
        return {
            "policy_slot": None,
            "policy_relation_by_side": {
                Side.BLUE.value: "our" if deposit_side is Side.BLUE else "enemy",
                Side.YELLOW.value: "our" if deposit_side is Side.YELLOW else "enemy",
            },
            "policy_id_by_side": {
                Side.BLUE.value: normalized_deposit_id(deposit_id, Side.BLUE),
                Side.YELLOW.value: normalized_deposit_id(deposit_id, Side.YELLOW),
            },
        }
    deposit_side = _deposit_side(deposit_id)
    return {
        "policy_slot": deposit_id % 10,
        "policy_relation_by_side": {
            Side.BLUE.value: "our" if deposit_side is Side.BLUE else "enemy",
            Side.YELLOW.value: "our" if deposit_side is Side.YELLOW else "enemy",
        },
        "policy_id_by_side": {
            Side.BLUE.value: normalized_deposit_id(deposit_id, Side.BLUE),
            Side.YELLOW.value: normalized_deposit_id(deposit_id, Side.YELLOW),
        },
    }


def _source_side(source_id: int) -> Side:
    if source_id in BLUE_SOURCE_IDS:
        return Side.BLUE
    if source_id in YELLOW_SOURCE_IDS:
        return Side.YELLOW
    raise ValueError(f"Unknown source id: {source_id}")


def _deposit_side(deposit_id: int) -> Side:
    if deposit_id in BLUE_SIDE_STORAGE_IDS:
        return Side.BLUE
    if deposit_id in YELLOW_SIDE_STORAGE_IDS:
        return Side.YELLOW
    raise ValueError(f"Unknown side-dependent deposit id: {deposit_id}")
