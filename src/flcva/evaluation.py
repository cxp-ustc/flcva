from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import RunConfig
from .contract import verify_recorded_contract
from .data import GoldLabel
from .ids import split_bucket, stable_id, stable_seed
from .io import (
    atomic_write_json,
    atomic_write_json_gz,
    atomic_write_jsonl_gz,
    iter_problem_artifacts,
    read_json_gz,
    seal_record,
    verify_sealed_record,
)
from .parsing import answers_equal
from .labels import load_attested_gold_labels
from .frozen import formal_difficulty_design
from .lineage import build_formal_difficulty_lineage
from .runner import resolve_run_dir
from .schemas import MatchedCell, ProxyRecord, ValueRecord
from .statistics import (
    assign_difficulty_tertiles,
    average_precision,
    bootstrap_cluster_interval,
    brier_score,
    calibration_curve,
    fit_tie_bins,
    jeffreys_bin_probabilities,
    roc_auc,
)

PROXY_NAMES = (
    "persistence",
    "answer_confidence",
    "modal_agreement",
    "prm_min_step",
    "anll",
    "top20_entropy",
)


def marginal_accuracy_value(
    stop_correct: int,
    continuation_correct: list[int],
    continuation_costs: list[int],
) -> float:
    if len(continuation_correct) != 4 or len(continuation_costs) != 4:
        raise ValueError("FLCVA requires exactly four continuation branches")
    if any(cost <= 0 for cost in continuation_costs):
        raise ValueError("continuation ledger costs must be positive")
    mean_change = sum(continuation_correct) / 4.0 - stop_correct
    mean_cost = sum(continuation_costs) / 4.0
    return 1000.0 * mean_change / mean_cost


def _ledger_rows(
    event_ids: Iterable[str],
    events: dict[str, dict[str, Any]],
    *,
    include_predecessors: bool,
) -> list[dict[str, Any]]:
    pending = list(event_ids)
    selected: dict[str, dict[str, Any]] = {}
    while pending:
        event_id = pending.pop()
        if event_id in selected:
            continue
        if event_id not in events:
            raise ValueError(f"frontier references absent ledger event: {event_id}")
        event = events[event_id]
        selected[event_id] = {
            "event_id": event_id,
            "role": event["role"],
            "logical_tokens": int(event["logical_tokens"]),
        }
        predecessor = event.get("causal_predecessor")
        if include_predecessors and predecessor is not None:
            pending.append(str(predecessor))
    return [selected[event_id] for event_id in sorted(selected)]


def _mean_logprob(values: list[float | None]) -> float | None:
    observed = [float(value) for value in values if value is not None]
    return sum(observed) / len(observed) if observed else None


def _top20_entropy(rows: list[dict[str, float]]) -> float | None:
    entropies: list[float] = []
    for row in rows:
        if not row:
            continue
        probabilities = [math.exp(float(value)) for value in row.values()]
        residual = max(0.0, 1.0 - sum(probabilities))
        if residual:
            probabilities.append(residual)
        entropies.append(
            -sum(
                probability * math.log(probability)
                for probability in probabilities
                if probability > 0
            )
        )
    return sum(entropies) / len(entropies) if entropies else None


def _score_payloads(run_dir: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "scores").glob("*.json.gz")):
        payload = read_json_gz(path)
        if not verify_sealed_record(payload):
            raise ValueError(f"score artifact hash mismatch: {path}")
        problem_id = str(payload["problem_id"])
        if problem_id in output:
            raise ValueError(f"duplicate score artifact for problem: {problem_id}")
        output[problem_id] = payload
    return output


def _correctness(
    artifact: dict[str, Any],
    label: GoldLabel,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    checkpoints: dict[str, dict[str, Any]] = {}
    branches: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    for checkpoint in artifact["checkpoints"]:
        correct, parsed, _ = answers_equal(
            artifact["dataset_id"],
            checkpoint["stop_raw_answer"],
            label.answer,
        )
        row = dict(checkpoint)
        row.update(
            {
                "stop_parser_status": parsed.status,
                "stop_canonical_answer": parsed.canonical,
                "stop_correct": int(correct),
            }
        )
        checkpoints[row["checkpoint_id"]] = row
        status_counts[f"{artifact['dataset_id']}:stop:{parsed.status}"] += 1
    for branch in artifact["branches"]:
        correct, parsed, _ = answers_equal(
            artifact["dataset_id"],
            branch["raw_answer"],
            label.answer,
        )
        row = dict(branch)
        row.update(
            {
                "parser_status": parsed.status,
                "canonical_answer": parsed.canonical,
                "correct": int(correct),
            }
        )
        branches[row["branch_id"]] = row
        status_counts[f"{artifact['dataset_id']}:branch:{parsed.status}"] += 1
    return checkpoints, branches, dict(status_counts)


def _t256_scores(
    run_dirs: list[Path],
    labels: dict[str, GoldLabel],
    *,
    expected_problem_ids: set[str] | None = None,
    expected_results_per_problem: int | None = None,
) -> dict[str, float]:
    values: dict[str, list[int]] = defaultdict(list)
    for run_dir in run_dirs:
        for artifact in iter_problem_artifacts(run_dir):
            if artifact["problem_id"] not in labels:
                continue
            checkpoints, _, _ = _correctness(
                artifact,
                labels[artifact["problem_id"]],
            )
            for checkpoint in checkpoints.values():
                if checkpoint["depth"] == 256:
                    values[artifact["problem_id"]].append(checkpoint["stop_correct"])
    if expected_problem_ids is not None and set(values) != expected_problem_ids:
        missing = sorted(expected_problem_ids - set(values))
        unexpected = sorted(set(values) - expected_problem_ids)
        raise ValueError(
            "difficulty problem universe mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    if expected_results_per_problem is not None:
        invalid = {
            problem_id: len(results)
            for problem_id, results in values.items()
            if len(results) != expected_results_per_problem
        }
        if invalid:
            raise ValueError(
                "difficulty pooling has the wrong number of t=256 observations: "
                f"{dict(list(sorted(invalid.items()))[:5])}"
            )
    return {
        problem_id: 1.0 - sum(results) / len(results)
        for problem_id, results in values.items()
        if results
    }


def _modal_agreement(
    checkpoints: Iterable[dict[str, Any]],
) -> dict[str, float | None]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for checkpoint in checkpoints:
        groups[(checkpoint["problem_id"], checkpoint["depth"])].append(checkpoint)
    output: dict[str, float | None] = {}
    for group in groups.values():
        answers = [
            checkpoint["stop_canonical_answer"]
            for checkpoint in group
            if checkpoint["stop_canonical_answer"] is not None
        ]
        if not answers:
            for checkpoint in group:
                output[checkpoint["checkpoint_id"]] = None
            continue
        counts = Counter(answers)
        maximum = max(counts.values())
        mode = sorted(answer for answer, count in counts.items() if count == maximum)[0]
        agreement = counts[mode] / len(group)
        for checkpoint in group:
            output[checkpoint["checkpoint_id"]] = agreement
    return output


def _proxy_rows(
    problem_id: str,
    checkpoints: dict[str, dict[str, Any]],
    prm: dict[str, tuple[float | None, str | None, int]],
    agreements: dict[str, float | None],
) -> tuple[dict[str, dict[str, float | None]], list[ProxyRecord]]:
    by_spine: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for checkpoint in checkpoints.values():
        by_spine[int(checkpoint["spine_id"])].append(checkpoint)
    output: dict[str, dict[str, float | None]] = {}
    records: list[ProxyRecord] = []
    for spine in by_spine.values():
        spine.sort(key=lambda row: row["depth"])
        for index, checkpoint in enumerate(spine):
            checkpoint_id = checkpoint["checkpoint_id"]
            current = checkpoint["stop_canonical_answer"]
            persistence: float | None = None
            persistence_reason: str | None = None
            if current is None:
                persistence_reason = "current_answer_parse_failure"
            elif index == 0:
                persistence_reason = "requires_two_probes"
            else:
                window = spine[max(0, index - 3) : index + 1]
                persistence = sum(
                    item["stop_canonical_answer"] == current for item in window
                ) / len(window)
            mean_logprob = _mean_logprob(checkpoint["stop_token_logprobs"])
            confidence = math.exp(mean_logprob) if mean_logprob is not None else None
            anll = -mean_logprob if mean_logprob is not None else None
            entropy = _top20_entropy(checkpoint["stop_top_logprobs"])
            prm_value, prm_reason, _ = prm.get(
                checkpoint_id,
                (None, "missing_score_artifact", 0),
            )
            values = {
                "persistence": persistence,
                "answer_confidence": confidence,
                "modal_agreement": agreements.get(checkpoint_id),
                "prm_min_step": prm_value,
                "anll": anll,
                "top20_entropy": entropy,
            }
            missing = {
                "persistence": persistence_reason,
                "answer_confidence": (
                    None if confidence is not None else "missing_logprobs"
                ),
                "modal_agreement": (
                    None
                    if agreements.get(checkpoint_id) is not None
                    else "no_parseable_spine_answer"
                ),
                "prm_min_step": prm_reason,
                "anll": None if anll is not None else "missing_logprobs",
                "top20_entropy": (
                    None if entropy is not None else "missing_top20_logprobs"
                ),
            }
            output[checkpoint_id] = values | {
                f"{key}__missing": missing[key] for key in values
            }
            for name, value in values.items():
                records.append(
                    ProxyRecord(
                        checkpoint_id=checkpoint_id,
                        proxy_name=name,
                        raw_value=value,
                        oriented_value=value,
                        missing_reason=missing[name],
                        source_record_ids=[checkpoint_id],
                        ledger_event_ids=[],
                    )
                )
    return output, records


def _analysis_summary(
    records: list[ValueRecord],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for target in ("beneficial", "harmful"):
        target_output: dict[str, Any] = {}
        feasible_stop = 0 if target == "beneficial" else 1
        for proxy in PROXY_NAMES:
            analysis = [
                row
                for row in records
                if row.split == "analysis"
                and row.stop_correct == feasible_stop
                and row.proxies.get(proxy) is not None
            ]
            confirmation = [
                row
                for row in records
                if row.split == "confirmation"
                and row.stop_correct == feasible_stop
                and row.proxies.get(proxy) is not None
            ]
            labels = [int(getattr(row, target)) for row in analysis]
            if not analysis or len(set(labels)) < 2 or not confirmation:
                target_output[proxy] = {
                    "status": "NA",
                    "reason": "insufficient_analysis_classes_or_confirmation_rows",
                    "analysis_n": len(analysis),
                    "confirmation_n": len(confirmation),
                }
                continue
            positive_values = [
                float(row.proxies[proxy])
                for row in analysis
                if int(getattr(row, target)) == 1
            ]
            negative_values = [
                float(row.proxies[proxy])
                for row in analysis
                if int(getattr(row, target)) == 0
            ]
            orientation = (
                1.0
                if sum(positive_values) / len(positive_values)
                >= sum(negative_values) / len(negative_values)
                else -1.0
            )
            oriented_analysis = [
                orientation * float(row.proxies[proxy]) for row in analysis
            ]
            bins = fit_tie_bins(oriented_analysis)
            probabilities = jeffreys_bin_probabilities(
                oriented_analysis,
                labels,
                bins,
            )
            base_rate = (sum(labels) + 0.5) / (len(labels) + 1.0)
            metric_rows: list[dict[str, Any]] = []
            for row in confirmation:
                oriented = orientation * float(row.proxies[proxy])
                metric_rows.append(
                    {
                        "problem_id": row.problem_id,
                        "label": int(getattr(row, target)),
                        "score": oriented,
                        "probability": probabilities.get(
                            bins.assign(oriented),
                            base_rate,
                        ),
                    }
                )
            y = [row["label"] for row in metric_rows]
            scores = [row["score"] for row in metric_rows]
            predicted = [row["probability"] for row in metric_rows]
            target_output[proxy] = {
                "status": "ok",
                "orientation": orientation,
                "analysis_n": len(analysis),
                "confirmation_n": len(confirmation),
                "confirmation_problems": len({row.problem_id for row in confirmation}),
                "auroc": roc_auc(y, scores),
                "auprc": average_precision(y, scores),
                "brier": brier_score(y, predicted),
                "calibration": calibration_curve(y, predicted),
                "auroc_ci": bootstrap_cluster_interval(
                    metric_rows,
                    lambda rows: roc_auc(
                        [row["label"] for row in rows],
                        [row["score"] for row in rows],
                    ),
                    replicates=bootstrap_replicates,
                    seed=stable_seed(seed, "metric", target, proxy),
                ),
                "auprc_ci": bootstrap_cluster_interval(
                    metric_rows,
                    lambda rows: average_precision(
                        [row["label"] for row in rows],
                        [row["score"] for row in rows],
                    ),
                    replicates=bootstrap_replicates,
                    seed=stable_seed(seed, "metric", target, proxy, "auprc"),
                ),
                "brier_ci": bootstrap_cluster_interval(
                    metric_rows,
                    lambda rows: brier_score(
                        [row["label"] for row in rows],
                        [row["probability"] for row in rows],
                    ),
                    replicates=bootstrap_replicates,
                    seed=stable_seed(seed, "metric", target, proxy, "brier"),
                ),
            }
        output[target] = target_output
    return output


def _stratified_proxy_summary(
    records: list[ValueRecord],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target in ("beneficial", "harmful"):
        feasible_stop = 0 if target == "beneficial" else 1
        for proxy in PROXY_NAMES:
            analysis = [
                row
                for row in records
                if row.split == "analysis"
                and row.stop_correct == feasible_stop
                and row.proxies.get(proxy) is not None
            ]
            labels = [int(getattr(row, target)) for row in analysis]
            if not analysis or len(set(labels)) < 2:
                continue
            positive_values = [
                float(row.proxies[proxy])
                for row in analysis
                if int(getattr(row, target)) == 1
            ]
            negative_values = [
                float(row.proxies[proxy])
                for row in analysis
                if int(getattr(row, target)) == 0
            ]
            orientation = (
                1.0
                if sum(positive_values) / len(positive_values)
                >= sum(negative_values) / len(negative_values)
                else -1.0
            )
            oriented_analysis = [
                orientation * float(row.proxies[proxy]) for row in analysis
            ]
            bins = fit_tie_bins(oriented_analysis)
            probabilities = jeffreys_bin_probabilities(
                oriented_analysis,
                labels,
                bins,
            )
            base_rate = (sum(labels) + 0.5) / (len(labels) + 1.0)
            groups: dict[tuple[Any, ...], list[ValueRecord]] = defaultdict(list)
            for row in records:
                if (
                    row.split == "confirmation"
                    and row.stop_correct == feasible_stop
                    and row.proxies.get(proxy) is not None
                ):
                    groups[
                        (
                            row.model_id,
                            row.dataset_id,
                            row.difficulty,
                            row.depth,
                            row.stop_correct,
                        )
                    ].append(row)
            for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
                y = [int(getattr(row, target)) for row in group]
                scores = [orientation * float(row.proxies[proxy]) for row in group]
                predicted = [
                    probabilities.get(bins.assign(score), base_rate) for score in scores
                ]
                output.append(
                    {
                        "target": target,
                        "proxy": proxy,
                        "model_id": key[0],
                        "dataset_id": key[1],
                        "difficulty": key[2],
                        "depth": key[3],
                        "stop_correct": key[4],
                        "checkpoints": len(group),
                        "problems": len({row.problem_id for row in group}),
                        "positive_rate": sum(y) / len(y),
                        "auroc": roc_auc(y, scores),
                        "auprc": average_precision(y, scores),
                        "brier": brier_score(y, predicted),
                    }
                )
    return output


def _matched_cells(
    records: list[ValueRecord],
) -> tuple[list[MatchedCell], dict[str, Any]]:
    cells: list[MatchedCell] = []
    attrition: Counter[str] = Counter()
    for proxy in PROXY_NAMES:
        calibration_values = [
            float(row.proxies[proxy])
            for row in records
            if row.split == "analysis" and row.proxies.get(proxy) is not None
        ]
        bins = fit_tie_bins(calibration_values)
        groups: dict[tuple[Any, ...], list[ValueRecord]] = defaultdict(list)
        for row in records:
            if row.split != "confirmation":
                continue
            value = row.proxies.get(proxy)
            if value is None:
                attrition[f"{proxy}:missing_proxy"] += 1
                continue
            key = (
                row.model_id,
                row.dataset_id,
                row.depth,
                row.difficulty,
                row.stop_correct,
                bins.assign(float(value)),
            )
            groups[key].append(row)
        for key, group in groups.items():
            nonzero = [
                row
                for row in group
                if (row.stop_correct == 0 and row.marginal_value > 0)
                or (row.stop_correct == 1 and row.marginal_value < 0)
            ]
            zero = [row for row in group if row.marginal_value == 0]
            if not nonzero or not zero:
                attrition[f"{proxy}:single_class"] += len(group)
                continue
            values = [float(row.proxies[proxy]) for row in group]
            cells.append(
                MatchedCell(
                    cell_id=stable_id("cell", proxy, *key),
                    proxy_name=proxy,
                    key={
                        "model_id": str(key[0]),
                        "dataset_id": str(key[1]),
                        "depth": int(key[2]),
                        "difficulty": str(key[3]),
                        "stop_correct": int(key[4]),
                        "proxy_bin": int(key[5]),
                    },
                    nonzero_checkpoint_ids=sorted(row.checkpoint_id for row in nonzero),
                    zero_checkpoint_ids=sorted(row.checkpoint_id for row in zero),
                    proxy_min=min(values),
                    proxy_max=max(values),
                    problem_ids=sorted({row.problem_id for row in group}),
                )
            )
    collision_checkpoint_ids = {
        checkpoint_id
        for cell in cells
        for checkpoint_id in (cell.nonzero_checkpoint_ids + cell.zero_checkpoint_ids)
    }
    collision_members = len(collision_checkpoint_ids)
    confirmation = sum(row.split == "confirmation" for row in records)
    gaps: dict[str, dict[str, float | int | None]] = {}
    by_proxy: dict[str, dict[str, float | int | None]] = {}
    for proxy in PROXY_NAMES:
        proxy_cells = [cell for cell in cells if cell.proxy_name == proxy]
        proxy_checkpoint_ids = {
            checkpoint_id
            for cell in proxy_cells
            for checkpoint_id in (
                cell.nonzero_checkpoint_ids + cell.zero_checkpoint_ids
            )
        }
        weighted_sum = 0.0
        weight_sum = 0
        record_map = {row.checkpoint_id: row for row in records}
        for cell in proxy_cells:
            nonzero_values = [
                record_map[checkpoint_id].marginal_value
                for checkpoint_id in cell.nonzero_checkpoint_ids
            ]
            zero_values = [
                record_map[checkpoint_id].marginal_value
                for checkpoint_id in cell.zero_checkpoint_ids
            ]
            weight = min(len(nonzero_values), len(zero_values))
            gap = sum(nonzero_values) / len(nonzero_values) - sum(zero_values) / len(
                zero_values
            )
            weighted_sum += weight * gap
            weight_sum += weight
        gaps[proxy] = {
            "matched_cells": len(proxy_cells),
            "weight": weight_sum,
            "weighted_marginal_value_gap": (
                weighted_sum / weight_sum if weight_sum else None
            ),
        }
        proxy_confirmation = sum(
            row.split == "confirmation" and row.proxies.get(proxy) is not None
            for row in records
        )
        by_proxy[proxy] = {
            "matched_cells": len(proxy_cells),
            "collision_members": len(proxy_checkpoint_ids),
            "eligible_confirmation_checkpoints": proxy_confirmation,
            "collision_rate": (
                len(proxy_checkpoint_ids) / proxy_confirmation
                if proxy_confirmation
                else None
            ),
        }
    return cells, {
        "retained_cells": len(cells),
        "collision_members": collision_members,
        "confirmation_checkpoints": confirmation,
        "collision_rate": collision_members / confirmation if confirmation else None,
        "attrition": dict(sorted(attrition.items())),
        "matched_gaps": gaps,
        "by_proxy": by_proxy,
    }


def _strata_summary(records: list[ValueRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[ValueRecord]] = defaultdict(list)
    for row in records:
        flags = {
            "easy": row.difficulty == "low",
            "middle": row.difficulty == "middle",
            "hard": row.difficulty == "high",
            "initially_wrong": row.initial_stop_correct == 0,
            "initially_correct": row.initial_stop_correct == 1,
            "wrong_to_correct": row.wrong_to_correct,
            "correct_to_wrong": row.correct_to_wrong,
        }
        for stratum, active in flags.items():
            if active:
                groups[
                    (
                        row.split,
                        row.model_id,
                        row.dataset_id,
                        row.depth,
                        row.stop_correct,
                        stratum,
                    )
                ].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        output.append(
            {
                "split": key[0],
                "model_id": key[1],
                "dataset_id": key[2],
                "depth": key[3],
                "stop_correct": key[4],
                "stratum": key[5],
                "checkpoints": len(group),
                "problems": len({row.problem_id for row in group}),
                "mean_marginal_value": sum(row.marginal_value for row in group)
                / len(group),
                "beneficial_rate": sum(row.beneficial for row in group) / len(group),
                "harmful_rate": sum(row.harmful for row in group) / len(group),
            }
        )
    return output


def _headline_test(
    records: list[ValueRecord],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    analysis = [
        row
        for row in records
        if row.split == "analysis"
        and row.stop_correct == 1
        and row.proxies.get("persistence") is not None
    ]
    confirmation = [
        row
        for row in records
        if row.split == "confirmation"
        and row.stop_correct == 1
        and row.proxies.get("persistence") is not None
    ]
    if len(analysis) < 4 or len(confirmation) < 2:
        return {
            "status": "NA",
            "reason": "insufficient_currently_correct_persistence_rows",
            "strong_title_supported": False,
        }
    ordered = sorted(float(row.proxies["persistence"]) for row in analysis)
    low_cut = ordered[max(0, math.ceil(0.25 * len(ordered)) - 1)]
    high_cut = ordered[max(0, math.ceil(0.75 * len(ordered)) - 1)]
    contrast_rows = [
        {
            "problem_id": row.problem_id,
            "group": (
                "high"
                if float(row.proxies["persistence"]) >= high_cut
                else "low" if float(row.proxies["persistence"]) <= low_cut else "middle"
            ),
            "marginal_value": row.marginal_value,
        }
        for row in confirmation
    ]

    def contrast(rows: Iterable[dict[str, Any]]) -> float | None:
        high = [row["marginal_value"] for row in rows if row["group"] == "high"]
        low = [row["marginal_value"] for row in rows if row["group"] == "low"]
        if not high or not low:
            return None
        return sum(high) / len(high) - sum(low) / len(low)

    interval = bootstrap_cluster_interval(
        contrast_rows,
        contrast,
        replicates=replicates,
        seed=stable_seed(seed, "headline", "persistence"),
    )
    supported = (
        interval["estimate"] is not None
        and interval["high"] is not None
        and float(interval["estimate"]) < 0
        and float(interval["high"]) < 0
    )
    return {
        "status": "ok" if interval["estimate"] is not None else "NA",
        "low_persistence_cut": low_cut,
        "high_persistence_cut": high_cut,
        "contrast_high_minus_low": interval,
        "strong_title_supported": supported,
    }


def evaluate_run(
    config: RunConfig,
    project_root: Path,
    *,
    peer_run_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    normalized_peer_dirs = [path.resolve() for path in (peer_run_dirs or [])]
    difficulty_design = (
        formal_difficulty_design(config, project_root)
        if config.stage == "formal"
        else None
    )
    if config.stage == "formal":
        required_peers = int(difficulty_design["required_peer_runs"])
        if len(normalized_peer_dirs) != required_peers:
            qualifier = (
                "one complementary peer run"
                if required_peers == 1
                else f"{required_peers} frozen difficulty-reference runs"
            )
            raise ValueError(f"formal evaluation requires exactly {qualifier}")
    run_dir = resolve_run_dir(config, project_root)
    verify_recorded_contract(config, project_root, run_dir)
    labels, gold_label_attestation = load_attested_gold_labels(
        config,
        project_root,
    )
    difficulty_lineage = (
        build_formal_difficulty_lineage(
            config,
            project_root,
            run_dir,
            normalized_peer_dirs,
        )
        if config.stage == "formal"
        else {
            "schema_version": "flcva-difficulty-lineage-engineering-v1",
            "run_count": 1 + len(normalized_peer_dirs),
        }
    )
    difficulty_run_dirs = (
        (
            [run_dir, *normalized_peer_dirs]
            if bool(difficulty_design["include_current_outcomes"])
            else normalized_peer_dirs
        )
        if config.stage == "formal"
        else [run_dir, *normalized_peer_dirs]
    )
    score_payloads = _score_payloads(run_dir)
    prepared: dict[str, dict[str, Any]] = {}
    parse_counts: Counter[str] = Counter()
    all_checkpoints: list[dict[str, Any]] = []
    for artifact in iter_problem_artifacts(run_dir):
        label = labels.get(artifact["problem_id"])
        if label is None:
            raise ValueError(f"missing gold label: {artifact['problem_id']}")
        checkpoints, branches, counts = _correctness(artifact, label)
        parse_counts.update(counts)
        prepared[artifact["problem_id"]] = {
            "artifact": artifact,
            "checkpoints": checkpoints,
            "branches": branches,
        }
        all_checkpoints.extend(checkpoints.values())
    agreements = _modal_agreement(all_checkpoints)
    t256 = _t256_scores(
        difficulty_run_dirs,
        labels,
        expected_problem_ids=set(labels) if config.stage == "formal" else None,
        expected_results_per_problem=(
            len(difficulty_run_dirs) * config.protocol.spines_per_problem
            if config.stage == "formal"
            else None
        ),
    )
    difficulty = assign_difficulty_tertiles(t256)
    values: list[ValueRecord] = []
    frontiers: list[dict[str, Any]] = []
    proxy_records: list[ProxyRecord] = []
    derived_problem_payloads: list[dict[str, Any]] = []
    for problem_id, item in sorted(prepared.items()):
        artifact = item["artifact"]
        checkpoints = item["checkpoints"]
        branches = item["branches"]
        generation_events = {event["event_id"]: event for event in artifact["ledger"]}
        score_payload = score_payloads.get(problem_id, {})
        score_events = {
            event["event_id"]: event for event in score_payload.get("ledger", [])
        }
        prm: dict[str, tuple[float | None, str | None, int]] = {}
        for proxy in score_payload.get("proxies", []):
            costs = sum(
                int(score_events[event_id]["logical_tokens"])
                for event_id in proxy.get("ledger_event_ids", [])
                if event_id in score_events
            )
            prm[proxy["checkpoint_id"]] = (
                proxy["raw_value"],
                proxy["missing_reason"],
                costs,
            )
        proxy_map, problem_proxy_records = _proxy_rows(
            problem_id,
            checkpoints,
            prm,
            agreements,
        )
        proxy_records.extend(problem_proxy_records)
        by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for branch in branches.values():
            by_checkpoint[branch["checkpoint_id"]].append(branch)
        initial = {
            int(checkpoint["spine_id"]): int(checkpoint["stop_correct"])
            for checkpoint in checkpoints.values()
            if checkpoint["depth"] == config.protocol.checkpoint_interval
        }
        problem_values: list[dict[str, Any]] = []
        value_by_checkpoint: dict[str, ValueRecord] = {}
        for checkpoint in sorted(
            checkpoints.values(),
            key=lambda row: (row["spine_id"], row["depth"]),
        ):
            if not checkpoint["continuation_eligible"]:
                continue
            branch_rows = sorted(
                by_checkpoint.get(checkpoint["checkpoint_id"], []),
                key=lambda row: row["branch_index"],
            )
            if len(branch_rows) != config.protocol.continuation_branches:
                continue
            continuation_correct = [int(row["correct"]) for row in branch_rows]
            continuation_costs = [int(row["logical_tokens"]) for row in branch_rows]
            value = marginal_accuracy_value(
                int(checkpoint["stop_correct"]),
                continuation_correct,
                continuation_costs,
            )
            proxy_values = {
                name: proxy_map[checkpoint["checkpoint_id"]][name]
                for name in PROXY_NAMES
            }
            missing = {
                name: proxy_map[checkpoint["checkpoint_id"]][f"{name}__missing"]
                for name in PROXY_NAMES
            }
            score_cost = prm.get(checkpoint["checkpoint_id"], (None, None, 0))[2]
            record = ValueRecord(
                checkpoint_id=checkpoint["checkpoint_id"],
                problem_id=problem_id,
                dataset_id=artifact["dataset_id"],
                model_id=checkpoint["model_id"],
                spine_id=int(checkpoint["spine_id"]),
                depth=int(checkpoint["depth"]),
                split=(
                    "analysis"
                    if split_bucket(artifact["dataset_id"], problem_id) < 80
                    else "confirmation"
                ),
                difficulty=difficulty[problem_id],
                stop_correct=int(checkpoint["stop_correct"]),
                continuation_correct=continuation_correct,
                continuation_costs=continuation_costs,
                marginal_value=value,
                beneficial=int(value > 0),
                harmful=int(value < 0),
                proxies=proxy_values,
                proxy_costs={
                    "prm_min_step": score_cost,
                    "persistence": 0,
                    "answer_confidence": 0,
                    "modal_agreement": 0,
                    "anll": 0,
                    "top20_entropy": 0,
                },
                proxy_missing_reasons=missing,
                initial_stop_correct=initial[int(checkpoint["spine_id"])],
                wrong_to_correct=(
                    initial[int(checkpoint["spine_id"])] == 0
                    and int(checkpoint["stop_correct"]) == 1
                ),
                correct_to_wrong=(
                    initial[int(checkpoint["spine_id"])] == 1
                    and int(checkpoint["stop_correct"]) == 0
                ),
            )
            values.append(record)
            value_by_checkpoint[checkpoint["checkpoint_id"]] = record
            problem_values.append(record.model_dump(mode="json"))
        for checkpoint in sorted(
            checkpoints.values(),
            key=lambda row: (row["spine_id"], row["depth"]),
        ):
            record = value_by_checkpoint.get(checkpoint["checkpoint_id"])
            initial_ledger_events = (
                _ledger_rows(
                    checkpoint["ledger_event_ids"],
                    generation_events,
                    include_predecessors=True,
                )
                if checkpoint["depth"] == config.protocol.checkpoint_interval
                else []
            )
            on_spine = [
                branch
                for branch in by_checkpoint.get(checkpoint["checkpoint_id"], [])
                if branch["branch_index"] == 1 and branch["on_spine"]
            ]
            transition_ledger_events = (
                _ledger_rows(
                    on_spine[0]["ledger_event_ids"],
                    generation_events,
                    include_predecessors=False,
                )
                if len(on_spine) == 1
                else []
            )
            ledger_transition_cost = (
                sum(row["logical_tokens"] for row in transition_ledger_events)
                if transition_ledger_events
                else None
            )
            if record is not None and (
                ledger_transition_cost is None
                or ledger_transition_cost != int(record.continuation_costs[0])
            ):
                raise ValueError(
                    "on-spine transition cost differs from its ledger events: "
                    f"{checkpoint['checkpoint_id']}"
                )
            frontiers.append(
                {
                    "schema_version": "flcva-frontier-v1",
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "successor_id": checkpoint["successor_id"],
                    "problem_id": problem_id,
                    "dataset_id": artifact["dataset_id"],
                    "model_id": checkpoint["model_id"],
                    "spine_id": int(checkpoint["spine_id"]),
                    "depth": int(checkpoint["depth"]),
                    "difficulty": difficulty[problem_id],
                    "split": (
                        "analysis"
                        if split_bucket(artifact["dataset_id"], problem_id) < 80
                        else "confirmation"
                    ),
                    "stop_correct": int(checkpoint["stop_correct"]),
                    "terminal": not checkpoint["continuation_eligible"],
                    "initial_ledger_events": initial_ledger_events,
                    "transition_ledger_events": transition_ledger_events,
                    "transition_cost": ledger_transition_cost,
                    "marginal_value": (
                        float(record.marginal_value) if record else None
                    ),
                    "proxies": (
                        proxy_map[checkpoint["checkpoint_id"]]
                        if checkpoint["checkpoint_id"] in proxy_map
                        else {name: None for name in PROXY_NAMES}
                    ),
                    "proxy_costs": {
                        "prm_min_step": prm.get(
                            checkpoint["checkpoint_id"],
                            (None, None, 0),
                        )[2],
                    },
                }
            )
        derived_problem_payloads.append(
            {
                "schema_version": "flcva-labeled-problem-v1",
                "problem_id": problem_id,
                "difficulty_score": t256[problem_id],
                "difficulty": difficulty[problem_id],
                "checkpoints": list(checkpoints.values()),
                "branches": list(branches.values()),
                "values": problem_values,
            }
        )
    evaluation_dir = run_dir / "evaluation"
    values_path = evaluation_dir / "values.jsonl.gz"
    proxies_path = evaluation_dir / "proxies.jsonl.gz"
    frontiers_path = evaluation_dir / "frontiers.jsonl.gz"
    cells_path = evaluation_dir / "matched_cells.jsonl.gz"
    atomic_write_jsonl_gz(
        values_path,
        [record.model_dump(mode="json") for record in values],
    )
    atomic_write_jsonl_gz(
        proxies_path,
        [record.model_dump(mode="json") for record in proxy_records],
    )
    atomic_write_jsonl_gz(frontiers_path, frontiers)
    derived_paths: list[Path] = []
    for payload in derived_problem_payloads:
        safe = payload["problem_id"].replace(":", "_")
        path = evaluation_dir / "problems" / f"{safe}.json.gz"
        atomic_write_json_gz(path, payload)
        derived_paths.append(path)
    cells, collision = _matched_cells(values)
    atomic_write_jsonl_gz(
        cells_path,
        [cell.model_dump(mode="json") for cell in cells],
    )
    artifact_paths = [
        values_path,
        proxies_path,
        frontiers_path,
        cells_path,
        *derived_paths,
    ]
    summary = seal_record(
        {
            "schema_version": "flcva-evaluation-summary-v1",
            "run_id": config.run_id,
            "scientific_results": config.stage in {"pilot", "formal"},
            "values": len(values),
            "problems": len(prepared),
            "difficulty_scope_runs": len(difficulty_run_dirs),
            "difficulty_reference_models": (
                list(difficulty_design["reference_models"])
                if difficulty_design is not None
                else sorted({checkpoint["model_id"] for checkpoint in all_checkpoints})
            ),
            "difficulty_current_outcomes_excluded": (
                not bool(difficulty_design["include_current_outcomes"])
                if difficulty_design is not None
                else False
            ),
            "difficulty_lineage": difficulty_lineage,
            "gold_labels": gold_label_attestation,
            "parse_counts": dict(sorted(parse_counts.items())),
            "collision": collision,
            "strata": _strata_summary(values),
            "headline": _headline_test(
                values,
                config.execution.bootstrap_replicates,
                config.global_seed,
            ),
            "metrics": _analysis_summary(
                values,
                config.execution.bootstrap_replicates,
                config.global_seed,
            ),
            "stratified_metrics": _stratified_proxy_summary(values),
            "artifacts": [
                str(path.relative_to(evaluation_dir)) for path in sorted(artifact_paths)
            ],
        }
    )
    atomic_write_json(evaluation_dir / "summary.json", summary)
    return summary
