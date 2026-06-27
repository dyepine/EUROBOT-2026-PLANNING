from __future__ import annotations

from dataclasses import dataclass

from poc.domain.actions import Action
from poc.domain.entities import Side
from poc.simulation.observations import DecisionObservation
from poc.rl.policy_mapping import normalized_action_label
from poc.rl.encoder import RLObservation, RLObservationConfig, build_rl_observation
from poc.rl.tokens import DEFAULT_ACTION_SPACE, RLActionSpace

@dataclass(frozen=True, slots=True)
class RLCandidate:
    index: int
    policy_action: str
    expected_duration: float
    score: float
    reward: float
    risk: float
    blocking_penalty: float
    swing: float


@dataclass(frozen=True, slots=True)
class RLPolicyStep:
    observation: RLObservation
    action_space: RLActionSpace
    action_mask: tuple[int, ...]
    candidates: tuple[RLCandidate, ...]


def build_rl_policy_step(
    observation: DecisionObservation,
    config: RLObservationConfig | None = None,
    action_space: RLActionSpace = DEFAULT_ACTION_SPACE,
) -> RLPolicyStep:
    side = observation.side
    ranked_actions = observation.ranked_actions
    rl_observation = build_rl_observation(observation, config=config)
    action_mask = [0] * len(action_space.tokens)
    candidates: list[RLCandidate] = []
    index_by_token = action_space.index_by_token
    for action in ranked_actions:
        token = normalized_action_label(action, side)
        action_index = index_by_token.get(token)
        if action_index is None:
            continue
        action_mask[action_index] = 1
        candidates.append(
            RLCandidate(
                index=action_index,
                policy_action=token,
                expected_duration=action.expected_duration,
                score=action.score,
                reward=action.expected_reward,
                risk=action.risk,
                blocking_penalty=action.blocking_penalty,
                swing=action.swing,
            )
        )
    return RLPolicyStep(
        observation=rl_observation,
        action_space=action_space,
        action_mask=tuple(action_mask),
        candidates=tuple(candidates),
    )


def resolve_policy_action(
    ranked_actions: list[Action],
    side: Side,
    policy_action: str,
) -> Action | None:
    for action in ranked_actions:
        if normalized_action_label(action, side) == policy_action:
            return action
    return None


