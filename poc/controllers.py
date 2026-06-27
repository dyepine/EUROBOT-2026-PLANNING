from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from poc.actions import Action
from poc.observations import DecisionObservation
from poc.opponent_policy import OpponentPolicy, build_opponent_policy


class ActionController(Protocol):
    name: str

    def select_action(
        self,
        *,
        observation: DecisionObservation,
        ranked_actions: list[Action],
    ) -> Action:
        ...

@dataclass(slots=True)
class ScriptedOpponentController:
    policy: OpponentPolicy

    @property
    def name(self) -> str:
        return self.policy.name

    def select_action(
        self,
        *,
        observation: DecisionObservation,
        ranked_actions: list[Action],
    ) -> Action:
        return self.policy.choose_action_from_ranked(observation.state, ranked_actions, observation.side)


def build_scripted_controller(name: str) -> ScriptedOpponentController:
    return ScriptedOpponentController(build_opponent_policy(name))
