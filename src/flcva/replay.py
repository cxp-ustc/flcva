from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import RunConfig
from .contract import verify_recorded_contract
from .ids import stable_seed
from .io import (
    atomic_write_json,
    atomic_write_jsonl_gz,
    file_sha256,
    iter_jsonl_gz,
    seal_record,
)
from .runner import resolve_run_dir
from .schemas import ReplayPoint


SCHEDULERS = (
    "fixed_1024",
    "round_robin",
    "trace",
    "agreement",
    "prm",
    "anll",
    "entropy",
    "oracle",
)
TRACE_MISSING_POLICY = "sum-available-components-v1"


@dataclass
class Frontier:
    key: tuple[str, int]
    state: dict[str, Any]


@dataclass(frozen=True)
class _DescendingText:
    """Reverse a string's ordering inside a min-heap."""

    value: str

    def __lt__(self, other: "_DescendingText") -> bool:
        return self.value > other.value


def step_auc(
    points: list[ReplayPoint],
    endpoint: int,
    *,
    start_budget: int | None = None,
) -> float:
    """Integrate an observed right-continuous replay curve.

    Accuracy is not observable before the first answer frontier has been
    generated.  When no explicit start is supplied, the first recorded point
    defines the left edge rather than being extrapolated back to budget zero.
    """

    if not points or endpoint <= 0:
        return 0.0
    ordered = sorted(points, key=lambda point: point.budget)
    start = ordered[0].budget if start_budget is None else int(start_budget)
    if endpoint <= start:
        return 0.0
    observed = [point for point in ordered if point.budget <= start]
    if not observed:
        raise ValueError(
            "replay AULC starts before the first observable accuracy point"
        )
    area = 0.0
    previous_budget = start
    previous_accuracy = observed[-1].accuracy
    for point in ordered:
        if point.budget <= start:
            continue
        current_budget = min(endpoint, point.budget)
        if current_budget > previous_budget:
            area += (current_budget - previous_budget) * previous_accuracy
            previous_budget = current_budget
        previous_accuracy = point.accuracy
        if point.budget >= endpoint:
            break
    if previous_budget < endpoint:
        area += (endpoint - previous_budget) * previous_accuracy
    return area


def _accuracy(frontiers: dict[tuple[str, int], Frontier]) -> float:
    return sum(frontier.state["stop_correct"] for frontier in frontiers.values()) / len(
        frontiers
    )


def _score(state: dict[str, Any], scheduler: str) -> float:
    proxies = state.get("proxies") or {}
    if scheduler == "trace":
        persistence = proxies.get("persistence")
        confidence = proxies.get("answer_confidence")
        if persistence is None and confidence is None:
            return -math.inf
        # Persistence is structurally missing at t=256.  Treating the missing
        # value as -inf makes the first advanced frontier monopolize the heap
        # and degenerates TRACE into an ID-ordered serial scheduler.  Sum only
        # the observable components and record this policy in the replay seal.
        return (
            (1.0 - float(persistence)) if persistence is not None else 0.0
        ) + (
            (1.0 - float(confidence)) if confidence is not None else 0.0
        )
    if scheduler == "agreement":
        value = proxies.get("modal_agreement")
        return 1.0 - float(value) if value is not None else -math.inf
    if scheduler == "prm":
        value = proxies.get("prm_min_step")
        return 1.0 - float(value) if value is not None else -math.inf
    if scheduler == "anll":
        value = proxies.get("anll")
        return float(value) if value is not None else -math.inf
    if scheduler == "entropy":
        value = proxies.get("top20_entropy")
        return float(value) if value is not None else -math.inf
    if scheduler == "oracle":
        value = state.get("marginal_value")
        return float(value) if value is not None else -math.inf
    raise ValueError(f"scheduler has no scalar score: {scheduler}")


def _initial_base_cost(
    states: list[dict[str, Any]],
) -> int:
    events: dict[str, tuple[str, int]] = {}
    required_roles = {"prompt", "spine_reasoning", "stop_probe"}
    for state in states:
        rows = state.get("initial_ledger_events")
        if not rows:
            raise ValueError(
                "initial frontier has no complete ledger provenance: "
                f"{state['checkpoint_id']}"
            )
        roles = {str(row["role"]) for row in rows}
        if not required_roles.issubset(roles):
            raise ValueError(
                "initial frontier ledger omits required events: "
                f"{state['checkpoint_id']} missing={sorted(required_roles - roles)}"
            )
        for row in rows:
            event_id = str(row["event_id"])
            value = (str(row["role"]), int(row["logical_tokens"]))
            if value[1] < 0:
                raise ValueError(f"negative initial ledger cost: {event_id}")
            if event_id in events and events[event_id] != value:
                raise ValueError(f"inconsistent duplicated ledger event: {event_id}")
            events[event_id] = value
    return sum(value[1] for value in events.values())


def _initial_proxy_cost(
    states: list[dict[str, Any]],
    scheduler: str,
) -> int:
    if scheduler != "prm":
        return 0
    return sum(
        int((state.get("proxy_costs") or {}).get("prm_min_step", 0))
        for state in states
    )


def _initial_cost(
    states: list[dict[str, Any]],
    scheduler: str,
) -> int:
    """Return the budget at which a scheduler is ready to choose an action."""

    return _initial_base_cost(states) + _initial_proxy_cost(states, scheduler)


def _transition_cost(state: dict[str, Any]) -> int:
    declared = state.get("transition_cost")
    rows = state.get("transition_ledger_events")
    if declared is None or int(declared) <= 0 or not rows:
        raise ValueError(
            f"frontier transition cost is invalid: {state['checkpoint_id']}"
        )
    event_ids: set[str] = set()
    ledger_cost = 0
    for row in rows:
        event_id = str(row["event_id"])
        if event_id in event_ids:
            raise ValueError(f"duplicated transition ledger event: {event_id}")
        event_ids.add(event_id)
        logical_tokens = int(row["logical_tokens"])
        if logical_tokens < 0:
            raise ValueError(f"negative transition ledger cost: {event_id}")
        ledger_cost += logical_tokens
    if ledger_cost != int(declared):
        raise ValueError(
            "frontier transition cost differs from ledger provenance: "
            f"{state['checkpoint_id']} declared={declared} ledger={ledger_cost}"
        )
    return ledger_cost


def _replay_one(
    states: list[dict[str, Any]],
    scheduler: str,
    *,
    checkpoint_interval: int,
    fixed_allotment_tokens: int,
    overridden_values: dict[str, float] | None = None,
) -> list[ReplayPoint]:
    by_id = {state["checkpoint_id"]: dict(state) for state in states}
    first_states = [
        state
        for state in states
        if int(state["depth"]) == checkpoint_interval
    ]
    frontiers = {
        (state["problem_id"], int(state["spine_id"])): Frontier(
            (state["problem_id"], int(state["spine_id"])),
            dict(state),
        )
        for state in first_states
    }
    if not frontiers:
        raise ValueError("replay has no initial frontiers")
    total_frontiers = len(frontiers)
    correct_count = sum(
        int(bool(frontier.state["stop_correct"]))
        for frontier in frontiers.values()
    )
    active_keys = {
        key for key, frontier in frontiers.items() if not frontier.state["terminal"]
    }
    active_count = len(active_keys)
    observation_budget = _initial_base_cost(first_states)
    budget = observation_budget
    points = [
        ReplayPoint(
            scheduler=scheduler,
            budget=budget,
            accuracy=correct_count / total_frontiers,
            active_frontiers=active_count,
            terminal_frontiers=total_frontiers - active_count,
        )
    ]
    scheduler_ready_budget = _initial_cost(first_states, scheduler)
    if scheduler_ready_budget > observation_budget:
        # The t=256 answers remain visible while an auxiliary proxy such as
        # PRM is scored.  Charge that cost as an explicit accuracy plateau.
        budget = scheduler_ready_budget
        points.append(
            ReplayPoint(
                scheduler=scheduler,
                budget=budget,
                accuracy=correct_count / total_frontiers,
                active_frontiers=active_count,
                terminal_frontiers=total_frontiers - active_count,
            )
        )
    ordered_keys = sorted(frontiers)
    queue = deque(ordered_keys)
    fixed_key_index = 0
    fixed_remaining = max(1, fixed_allotment_tokens // checkpoint_interval)
    priority_heap: list[
        tuple[float, int, _DescendingText, str, tuple[str, int]]
    ] = []

    def push_priority(key: tuple[str, int]) -> None:
        state = frontiers[key].state
        if overridden_values is not None and scheduler == "oracle":
            scalar = overridden_values.get(state["checkpoint_id"], -math.inf)
        else:
            scalar = _score(state, scheduler)
        heapq.heappush(
            priority_heap,
            (
                -scalar,
                -int(key[1]),
                _DescendingText(str(key[0])),
                str(state["checkpoint_id"]),
                key,
            ),
        )

    if scheduler not in {"round_robin", "fixed_1024"}:
        for key in active_keys:
            push_priority(key)

    while active_keys:
        if scheduler == "round_robin":
            while queue and queue[0] not in active_keys:
                queue.popleft()
            if not queue:
                queue.extend(sorted(active_keys))
            chosen = queue.popleft()
            queue.append(chosen)
        elif scheduler == "fixed_1024":
            while (
                fixed_key_index < len(ordered_keys)
                and ordered_keys[fixed_key_index] not in active_keys
            ):
                fixed_key_index += 1
                fixed_remaining = max(1, fixed_allotment_tokens // checkpoint_interval)
            if fixed_key_index >= len(ordered_keys):
                fixed_key_index = 0
                continue
            chosen = ordered_keys[fixed_key_index]
            fixed_remaining -= 1
            if fixed_remaining == 0:
                fixed_key_index += 1
                fixed_remaining = max(1, fixed_allotment_tokens // checkpoint_interval)
        else:
            while priority_heap:
                _, _, _, checkpoint_id, candidate = heapq.heappop(priority_heap)
                if (
                    candidate in active_keys
                    and frontiers[candidate].state["checkpoint_id"] == checkpoint_id
                ):
                    chosen = candidate
                    break
            else:
                raise ValueError("priority replay exhausted its active frontier heap")
        current = frontiers[chosen].state
        successor_id = current.get("successor_id")
        if not successor_id or successor_id not in by_id:
            raise ValueError(f"frontier successor is missing: {current['checkpoint_id']}")
        successor = dict(by_id[successor_id])
        budget += _transition_cost(current)
        if scheduler == "prm":
            budget += int(
                (successor.get("proxy_costs") or {}).get("prm_min_step", 0)
            )
        frontiers[chosen].state = successor
        correct_count += int(bool(successor["stop_correct"])) - int(
            bool(current["stop_correct"])
        )
        if successor["terminal"]:
            active_keys.remove(chosen)
            active_count -= 1
        elif scheduler not in {"round_robin", "fixed_1024"}:
            push_priority(chosen)
        points.append(
            ReplayPoint(
                scheduler=scheduler,
                budget=budget,
                accuracy=correct_count / total_frontiers,
                active_frontiers=active_count,
                terminal_frontiers=total_frontiers - active_count,
            )
        )
    return points


def _permuted_oracle(
    states: list[dict[str, Any]],
    *,
    config: RunConfig,
    round_robin_auc: float,
    start_budget: int,
    endpoint: int,
) -> dict[str, Any]:
    eligible = [state for state in states if state.get("marginal_value") is not None]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for state in eligible:
        groups[
            (
                state["model_id"],
                state["dataset_id"],
                state["difficulty"],
                state["depth"],
                state["stop_correct"],
            )
        ].append(state)
    differences: list[float] = []
    effective_groups = sum(
        len(group) >= 2 and len({row["marginal_value"] for row in group}) >= 2
        for group in groups.values()
    )
    for permutation_index in range(config.execution.permutation_replicates):
        assigned: dict[str, float] = {}
        for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
            values = [float(row["marginal_value"]) for row in group]
            if len(group) >= 2 and len(set(values)) >= 2:
                rng = random.Random(
                    stable_seed(config.global_seed, "permutation", permutation_index, *key)
                )
                rng.shuffle(values)
            for row, value in zip(
                sorted(group, key=lambda item: item["checkpoint_id"]),
                values,
            ):
                assigned[row["checkpoint_id"]] = value
        points = _replay_one(
            states,
            "oracle",
            checkpoint_interval=config.protocol.checkpoint_interval,
            fixed_allotment_tokens=config.execution.fixed_allotment_tokens,
            overridden_values=assigned,
        )
        differences.append(
            step_auc(points, endpoint, start_budget=start_budget)
            - round_robin_auc
        )
    array = np.asarray(differences, dtype=float)
    return {
        "replicates": len(differences),
        "effective_groups": effective_groups,
        "mean_difference": float(array.mean()) if len(array) else None,
        "median_difference": float(np.median(array)) if len(array) else None,
        "low": float(np.quantile(array, 0.025)) if len(array) else None,
        "high": float(np.quantile(array, 0.975)) if len(array) else None,
        "differences": differences,
    }


def derive_replay_curves(
    states: list[dict[str, Any]],
    config: RunConfig,
) -> dict[str, list[ReplayPoint]]:
    """Derive deterministic scheduler curves from authenticated frontiers."""

    return {
        scheduler: _replay_one(
            states,
            scheduler,
            checkpoint_interval=config.protocol.checkpoint_interval,
            fixed_allotment_tokens=config.execution.fixed_allotment_tokens,
        )
        for scheduler in SCHEDULERS
    }


def replay_curve_statistics(
    states: list[dict[str, Any]],
    curves: dict[str, list[ReplayPoint]],
    config: RunConfig,
) -> dict[str, Any]:
    """Compute the v2 common-domain AULC and endpoint statistics."""

    if set(curves) != set(SCHEDULERS) or any(not points for points in curves.values()):
        raise ValueError("replay curves are incomplete")
    first_states = [
        state
        for state in states
        if int(state["depth"]) == config.protocol.checkpoint_interval
    ]
    start_budget = _initial_base_cost(first_states)
    endpoint = min(points[-1].budget for points in curves.values())
    if endpoint <= start_budget:
        raise ValueError("replay endpoint does not exceed the observable start")
    aulc = {
        scheduler: step_auc(
            points,
            endpoint,
            start_budget=start_budget,
        )
        for scheduler, points in curves.items()
    }
    return {
        "common_endpoint": endpoint,
        "aulc_start_budget": start_budget,
        "aulc_span": endpoint - start_budget,
        "trace_missing_policy": TRACE_MISSING_POLICY,
        "initial_budget": {
            scheduler: _initial_cost(first_states, scheduler)
            for scheduler in SCHEDULERS
        },
        "aulc": aulc,
        "final_accuracy": {
            scheduler: max(
                (point for point in points if point.budget <= endpoint),
                key=lambda point: point.budget,
            ).accuracy
            for scheduler, points in curves.items()
        },
        "oracle_difference": aulc["oracle"] - aulc["round_robin"],
    }


def replay_run(config: RunConfig, project_root: Path) -> dict[str, Any]:
    run_dir = resolve_run_dir(config, project_root)
    verify_recorded_contract(config, project_root, run_dir)
    all_states = list(iter_jsonl_gz(run_dir / "evaluation" / "frontiers.jsonl.gz"))
    states = [state for state in all_states if state["split"] == "confirmation"]
    split_scope = "confirmation"
    if not states and config.stage in {"mock", "smoke", "canary"}:
        states = all_states
        split_scope = "all_engineering_only"
    if not states:
        raise ValueError("no confirmation frontiers available for scientific replay")
    curves = derive_replay_curves(states, config)
    statistics = replay_curve_statistics(states, curves, config)
    endpoint = int(statistics["common_endpoint"])
    start_budget = int(statistics["aulc_start_budget"])
    aulc = statistics["aulc"]
    permutation = _permuted_oracle(
        states,
        config=config,
        round_robin_auc=aulc["round_robin"],
        start_budget=start_budget,
        endpoint=endpoint,
    )
    original_difference = aulc["oracle"] - aulc["round_robin"]
    exceedances = sum(
        value >= original_difference for value in permutation["differences"]
    )
    permutation["exceedances"] = exceedances
    permutation["fraction_at_least_original"] = (
        exceedances / len(permutation["differences"])
        if permutation["differences"]
        else None
    )
    permutation["p_value_corrected"] = (
        (exceedances + 1) / (len(permutation["differences"]) + 1)
        if permutation["differences"]
        else None
    )
    permutation.pop("differences", None)
    points_payload = [
        point.model_dump(mode="json")
        for scheduler in SCHEDULERS
        for point in curves[scheduler]
        if point.budget <= endpoint
    ]
    replay_dir = run_dir / "replay"
    points_path = replay_dir / "points.jsonl.gz"
    atomic_write_jsonl_gz(points_path, points_payload)
    summary = seal_record(
        {
            "schema_version": "flcva-replay-summary-v2",
            "run_id": config.run_id,
            "split_scope": split_scope,
            **statistics,
            "aulc": aulc,
            "oracle_difference": original_difference,
            "permutation_oracle": permutation,
            "artifacts_sha256": {
                str(points_path.relative_to(replay_dir)): file_sha256(points_path),
            },
        }
    )
    atomic_write_json(replay_dir / "summary.json", summary)
    return summary
