from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import random

from poc.control.controllers import ActionController
from poc.planning.planner import UtilityPlanner
from poc.rl.config import SelfPlayConfig, selfplay_config_from_dict
from poc.rl.match import (
    OpponentSpec,
    build_model,
    build_observation_config,
    play_match,
    transitions_to_rollout_items,
)
from poc.rl.model import MaskedPolicyValueNet, load_compatible_state_dict
from poc.rl.ppo import PPORolloutItem
from poc.rl.selectors import RandomMaskedPolicySelector, TorchPolicySelector
from poc.rl.transitions import build_rl_transitions_from_match_result
from poc.rl.torch_compat import require_torch, torch

def _make_process_pool(max_workers: int) -> ProcessPoolExecutor:
    # `spawn` is safer than the Linux default `fork` when the parent process
    # already owns a CUDA context or imported torch state.
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"))


class PersistentWorkerPools:
    def __init__(self, *, rollout_workers: int, eval_workers: int) -> None:
        self.rollout_workers = max(int(rollout_workers), 1)
        self.eval_workers = max(int(eval_workers), 1)
        self._rollout_executor: ProcessPoolExecutor | None = None
        self._eval_executor: ProcessPoolExecutor | None = None

    def rollout_executor(self) -> ProcessPoolExecutor | None:
        if self.rollout_workers <= 1:
            return None
        if self._rollout_executor is None:
            self._rollout_executor = _make_process_pool(self.rollout_workers)
        return self._rollout_executor

    def eval_executor(self) -> ProcessPoolExecutor | None:
        if self.eval_workers <= 1:
            return None
        if self._eval_executor is None:
            self._eval_executor = _make_process_pool(self.eval_workers)
        return self._eval_executor

    def close(self) -> None:
        if self._rollout_executor is not None:
            self._rollout_executor.shutdown(wait=True, cancel_futures=False)
            self._rollout_executor = None
        if self._eval_executor is not None:
            self._eval_executor.shutdown(wait=True, cancel_futures=False)
            self._eval_executor = None


@dataclass(frozen=True, slots=True)
class WorkerMatchRequest:
    config_payload: dict[str, object]
    scenario_name: str
    seed: int
    update_id: int
    episode_id: int
    learner_state_dict: dict[str, object]
    learner_greedy: bool
    opponent_name: str
    opponent_kind: str
    opponent_policy_name: str
    opponent_state_dict: dict[str, object] | None
    opponent_greedy: bool


@dataclass(frozen=True, slots=True)
class WorkerMatchResult:
    rollout_items: tuple[PPORolloutItem, ...]
    match_summary: dict[str, object]


@dataclass(slots=True)
class _WorkerRuntime:
    signature: tuple[object, ...]
    planner: UtilityPlanner
    learner_model: MaskedPolicyValueNet
    opponent_model: MaskedPolicyValueNet


_WORKER_RUNTIME: _WorkerRuntime | None = None


def _cpu_worker_config(config: SelfPlayConfig) -> SelfPlayConfig:
    payload = config.to_dict()
    payload["ppo"]["device"] = "cpu"
    return selfplay_config_from_dict(payload)


def _worker_runtime_signature(config: SelfPlayConfig) -> tuple[object, ...]:
    return (
        config.ppo.device,
        tuple(config.ppo.hidden_sizes),
    )


def _get_worker_runtime(config: SelfPlayConfig) -> _WorkerRuntime:
    global _WORKER_RUNTIME
    signature = _worker_runtime_signature(config)
    if _WORKER_RUNTIME is not None and _WORKER_RUNTIME.signature == signature:
        return _WORKER_RUNTIME
    device = torch.device(config.ppo.device)
    learner_model = build_model(config)
    learner_model.to(device)
    opponent_model = build_model(config)
    opponent_model.to(device)
    _WORKER_RUNTIME = _WorkerRuntime(
        signature=signature,
        planner=UtilityPlanner(),
        learner_model=learner_model,
        opponent_model=opponent_model,
    )
    return _WORKER_RUNTIME


def _build_worker_request(
    *,
    config: SelfPlayConfig,
    learner_state_dict: dict[str, object],
    scenario_name: str,
    seed: int,
    update_id: int,
    episode_id: int,
    learner_greedy: bool,
    opponent_spec: OpponentSpec,
) -> WorkerMatchRequest:
    worker_config = _cpu_worker_config(config)
    return WorkerMatchRequest(
        config_payload=worker_config.to_dict(),
        scenario_name=scenario_name,
        seed=seed,
        update_id=update_id,
        episode_id=episode_id,
        learner_state_dict=learner_state_dict,
        learner_greedy=learner_greedy,
        opponent_name=opponent_spec.name,
        opponent_kind=opponent_spec.kind,
        opponent_policy_name=opponent_spec.opponent_policy_name,
        opponent_state_dict=opponent_spec.state_dict,
        opponent_greedy=opponent_spec.greedy,
    )


def _play_match_worker(request: WorkerMatchRequest) -> WorkerMatchResult:
    require_torch(torch)
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch.manual_seed(request.seed)
    config = selfplay_config_from_dict(dict(request.config_payload))
    runtime = _get_worker_runtime(config)
    load_compatible_state_dict(runtime.learner_model, request.learner_state_dict)
    learner_selector = TorchPolicySelector(
        model=runtime.learner_model,
        device=config.ppo.device,
        greedy=request.learner_greedy,
        name="learner_worker",
    )

    opponent_selector: ActionController | None = None
    if request.opponent_kind == "random":
        opponent_selector = RandomMaskedPolicySelector(random.Random(request.seed ^ 0xA5A5A5), name=request.opponent_name)
    elif request.opponent_state_dict is not None:
        load_compatible_state_dict(runtime.opponent_model, request.opponent_state_dict)
        opponent_selector = TorchPolicySelector(
            model=runtime.opponent_model,
            device=config.ppo.device,
            greedy=request.opponent_greedy,
            name=request.opponent_name,
        )

    opponent_spec = OpponentSpec(
        name=request.opponent_name,
        selector=opponent_selector,
        opponent_policy_name=request.opponent_policy_name,
        kind=request.opponent_kind,
        state_dict=request.opponent_state_dict,
        greedy=request.opponent_greedy,
    )
    artifacts = play_match(
        config=config,
        scenario_name=request.scenario_name,
        seed=request.seed,
        learner_selector=learner_selector,
        opponent_spec=opponent_spec,
        planner=runtime.planner,
    )
    items = transitions_to_rollout_items(
        build_rl_transitions_from_match_result(
            artifacts.result,
            side=config.ppo.side,
            config=build_observation_config(config),
            thermometer_reward_bonus=config.thermometer_reward_bonus,
            terminal_win_bonus=config.terminal_win_bonus,
            terminal_draw_bonus=config.terminal_draw_bonus,
            terminal_loss_bonus=config.terminal_loss_bonus,
        ),
        list(artifacts.learner_records),
        update_id=request.update_id,
        episode_id=request.episode_id,
    )
    return WorkerMatchResult(
        rollout_items=tuple(items),
        match_summary={
            "scenario": request.scenario_name,
            "seed": request.seed,
            "opponent": request.opponent_name,
            "opponent_kind": request.opponent_kind,
            "summary": artifacts.result.summary,
            "invalid_action_count": artifacts.invalid_action_count,
        },
    )
