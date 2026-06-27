from __future__ import annotations

from dataclasses import dataclass

POLICY_SOURCE_ORDER = (
    "OUR_SOURCE_1",
    "OUR_SOURCE_2",
    "OUR_SOURCE_3",
    "OUR_SOURCE_4",
    "ENEMY_SOURCE_1",
    "ENEMY_SOURCE_2",
    "ENEMY_SOURCE_3",
    "ENEMY_SOURCE_4",
)

POLICY_DEPOSIT_ORDER = (
    "OUR_HOME",
    "ENEMY_HOME",
    "OUR_STORAGE_5",
    "OUR_STORAGE_6",
    "OUR_STORAGE_7",
    "NEUTRAL_STORAGE_1",
    "NEUTRAL_STORAGE_0",
    "ENEMY_STORAGE_5",
    "ENEMY_STORAGE_6",
    "ENEMY_STORAGE_7",
)

POLICY_DEPOSIT_ACTION_TARGETS = (
    "OUR_HOME",
    "OUR_STORAGE_5",
    "OUR_STORAGE_6",
    "OUR_STORAGE_7",
    "NEUTRAL_STORAGE_1",
    "NEUTRAL_STORAGE_0",
    "ENEMY_STORAGE_5",
    "ENEMY_STORAGE_6",
    "ENEMY_STORAGE_7",
)

POLICY_ATTACK_TARGETS = (
    "OUR_STORAGE_5",
    "OUR_STORAGE_6",
    "OUR_STORAGE_7",
    "NEUTRAL_STORAGE_1",
    "NEUTRAL_STORAGE_0",
    "ENEMY_STORAGE_5",
    "ENEMY_STORAGE_6",
    "ENEMY_STORAGE_7",
)



@dataclass(frozen=True, slots=True)
class RLActionSpace:
    tokens: tuple[str, ...]

    @property
    def index_by_token(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, token: str) -> int:
        return self.index_by_token[token]

    def decode(self, index: int) -> str:
        return self.tokens[index]


DEFAULT_ACTION_SPACE = RLActionSpace(
    tokens=(
        *(f"PICK_{source_id}" for source_id in POLICY_SOURCE_ORDER),
        *(f"DEPOSIT_{target}_X{count}" for target in POLICY_DEPOSIT_ACTION_TARGETS for count in range(1, 5)),
        *(f"ATTACK_{target}" for target in POLICY_ATTACK_TARGETS),
        "THERMOMETER",
        "START_ENDGAME",
        "PLAY_TO_END",
        "WAIT",
        "WAIT_FOR_CHILL",
    )
)
