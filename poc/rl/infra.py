from __future__ import annotations

# Compatibility facade: concrete RL responsibilities live in split modules.
from poc.rl.action_space import RLCandidate, RLPolicyStep, build_rl_policy_step, resolve_policy_action
from poc.rl.encoder import (
    DEFAULT_FLAT_FEATURE_KEYS,
    DEPOSIT_FEATURE_ORDER,
    GLOBAL_FEATURE_ORDER,
    RLObservation,
    RLObservationConfig,
    SOURCE_FEATURE_ORDER,
    flat_feature_vector,
    build_rl_observation,
)
from poc.rl.tokens import (
    DEFAULT_ACTION_SPACE,
    POLICY_ATTACK_TARGETS,
    POLICY_DEPOSIT_ACTION_TARGETS,
    POLICY_DEPOSIT_ORDER,
    POLICY_SOURCE_ORDER,
    RLActionSpace,
)
from poc.rl.transitions import RLTransition, build_rl_transitions_from_match_result, transition_to_record

__all__ = [
    "DEFAULT_ACTION_SPACE",
    "DEFAULT_FLAT_FEATURE_KEYS",
    "DEPOSIT_FEATURE_ORDER",
    "GLOBAL_FEATURE_ORDER",
    "POLICY_ATTACK_TARGETS",
    "POLICY_DEPOSIT_ACTION_TARGETS",
    "POLICY_DEPOSIT_ORDER",
    "POLICY_SOURCE_ORDER",
    "RLCandidate",
    "RLActionSpace",
    "RLObservation",
    "RLObservationConfig",
    "RLPolicyStep",
    "RLTransition",
    "SOURCE_FEATURE_ORDER",
    "build_rl_observation",
    "build_rl_policy_step",
    "build_rl_transitions_from_match_result",
    "flat_feature_vector",
    "resolve_policy_action",
    "transition_to_record",
]
