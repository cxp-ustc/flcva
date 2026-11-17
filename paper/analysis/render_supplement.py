#!/usr/bin/env python3
"""Render supplementary analyses from the two audited formal v8 runs.

Read recorded formal runs and create tables and figures for the paper.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import csv
import gzip
import hashlib
import heapq
import json
import math
import random
import re
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import sympy

from flcva.config import RunConfig
from flcva.contract import verify_run_record
from flcva.ids import stable_seed
from flcva.io import require_artifacts, seal_record, verify_sealed_record
from flcva.replay import (
    _DescendingText,
    _initial_base_cost,
    _initial_cost,
    _replay_one,
    _score,
    _transition_cost,
    step_auc,
)
from flcva.schemas import ReplayPoint, RunRecord
from flcva.statistics import fit_tie_bins, roc_auc

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
OUT = PAPER / "generated"
RUNS = {
    "Qwen3-8B": ROOT / "artifacts" / "primary-qwen3",
    "DeepSeek-R1-Distill-Qwen-7B": (ROOT / "artifacts" / "primary-deepseek"),
}
MODEL_SHORT = {
    "Qwen3-8B": "Qwen3",
    "DeepSeek-R1-Distill-Qwen-7B": "DeepSeek",
}
PROXY_LABELS = {
    "anll": "ANLL",
    "answer_confidence": "Confidence",
    "modal_agreement": "Agreement",
    "persistence": "Persistence",
    "prm_min_step": "PRM",
    "top20_entropy": "Entropy",
}
TARGET_LABELS = {
    "beneficial": "Beneficial among wrong",
    "harmful": "Harmful among correct",
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20251231
NONMYOPIC_DP_SUBSETS = 64
NONMYOPIC_DP_FRONTIERS_PER_SUBSET = 4
NONMYOPIC_DP_NAMESPACE = "flcva-nonmyopic-dp-v1"
SPINE_REPLICATE_NAMESPACE = "flcva-spine-replicate-replay-v1"
FULL_REPLAY_BOOTSTRAP_NAMESPACE = "flcva-full-replay-bootstrap-v1"
MULTI_LEDGER_NAMESPACE = "flcva-multi-ledger-replay-v1"
COLLISION_ROBUSTNESS_NAMESPACE = "flcva-collision-robustness-v1"
DIFFICULTY_DELETION_NAMESPACE = "flcva-difficulty-deletion-v1"
VALUE_RATIO_SENSITIVITY_NAMESPACE = "flcva-value-ratio-sensitivity-v1"
ARCHITECTURES_PATH = PAPER / "analysis" / "frozen_architectures.json"


@dataclass(frozen=True)
class CostReplayPoint:
    budget: float
    accuracy: float


@dataclass(frozen=True)
class CompactReplayContext:
    """A complete single-spine problem lattice for repeated system replay."""

    problem_ids: tuple[str, ...]
    initial_costs: np.ndarray
    stop_correct: np.ndarray
    transition_costs: np.ndarray
    policy_scores: dict[str, np.ndarray]


@dataclass(frozen=True)
class FlopCost:
    generator_reasoning: float = 0.0
    generator_probe: float = 0.0
    prm_prefill: float = 0.0

    def __add__(self, other: "FlopCost") -> "FlopCost":
        return FlopCost(
            self.generator_reasoning + other.generator_reasoning,
            self.generator_probe + other.generator_probe,
            self.prm_prefill + other.prm_prefill,
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def iter_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_inputs() -> dict[str, dict[str, Any]]:
    paper_manifest = read_json(OUT / "manifest.json")
    if not verify_sealed_record(paper_manifest):
        raise ValueError("primary paper manifest seal failed")
    require_artifacts(ROOT, paper_manifest)
    manifest_runs = {row["run_id"]: row for row in paper_manifest["source_runs"]}
    validated: dict[str, dict[str, Any]] = {}
    for model, run_dir in RUNS.items():
        run = read_json(run_dir / "run.json")
        audit = read_json(run_dir / "audit_manifest.json")
        evaluation = read_json(run_dir / "evaluation" / "summary.json")
        replay = read_json(run_dir / "replay" / "summary.json")
        run_manifest = read_json(run_dir / "manifest.json")
        verify_run_record(RunRecord.model_validate(run))
        for label, record in (
            ("run manifest", run_manifest),
            ("audit", audit),
            ("evaluation", evaluation),
            ("replay", replay),
        ):
            if not verify_sealed_record(record):
                raise ValueError(f"{run['run_id']} {label} seal failed")
        for subdirectory, summary in (
            ("evaluation", evaluation),
            ("replay", replay),
        ):
            require_artifacts(run_dir / subdirectory, summary)
        if run["status"] != "sealed":
            raise ValueError(f"{run['run_id']} is not sealed")
        if audit.get("passed") is not True or not all(audit["gates"].values()):
            raise ValueError(f"{run['run_id']} did not pass every audit gate")
        paper_row = manifest_runs.get(run["run_id"])
        if paper_row is None:
            raise ValueError(f"{run['run_id']} is absent from paper manifest")
        if paper_row["result_merkle_root"] != audit["result_merkle_root"]:
            raise ValueError(f"{run['run_id']} paper Merkle root mismatch")
        if evaluation.get("scientific_results") is not True:
            raise ValueError(f"{run['run_id']} is not a scientific evaluation")
        if replay.get("permutation_oracle", {}).get("replicates") != 1000:
            raise ValueError(f"{run['run_id']} permutation oracle is incomplete")
        validated[model] = {
            "run_dir": run_dir,
            "run": run,
            "audit": audit,
            "evaluation": evaluation,
            "replay": replay,
        }
    if len(manifest_runs) != 2:
        raise ValueError("paper manifest must bind exactly two formal runs")
    return validated


def load_values(
    run_dir: Path,
    evaluation: dict[str, Any],
) -> list[dict[str, Any]]:
    path = run_dir / "evaluation" / "values.jsonl.gz"
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl_gz(path):
        rows.append(
            {
                "checkpoint_id": row["checkpoint_id"],
                "problem_id": row["problem_id"],
                "dataset_id": row["dataset_id"],
                "depth": int(row["depth"]),
                "difficulty": row["difficulty"],
                "split": row["split"],
                "stop_correct": int(row["stop_correct"]),
                "beneficial": int(row["beneficial"]),
                "harmful": int(row["harmful"]),
                "marginal_value": float(row["marginal_value"]),
                "persistence": row["proxies"].get("persistence"),
                "proxies": {proxy: row["proxies"].get(proxy) for proxy in PROXY_LABELS},
                "spine_id": int(row["spine_id"]),
                "continuation_correct": tuple(
                    int(value) for value in row["continuation_correct"]
                ),
                "continuation_costs": tuple(
                    int(value) for value in row["continuation_costs"]
                ),
            }
        )
    if len(rows) != 109_140:
        raise ValueError(f"expected 109140 values, found {len(rows)}")
    return rows


def token_normalized_value(
    stop_correct: int,
    continuation_correct: tuple[int, ...] | list[int],
    continuation_costs: tuple[int, ...] | list[int],
) -> float:
    if (
        not continuation_correct
        or len(continuation_correct) != len(continuation_costs)
        or any(int(cost) <= 0 for cost in continuation_costs)
    ):
        raise ValueError("oracle value requires aligned positive branch costs")
    mean_correct = sum(int(value) for value in continuation_correct) / len(
        continuation_correct
    )
    mean_cost = sum(int(value) for value in continuation_costs) / len(
        continuation_costs
    )
    return 1000.0 * (mean_correct - int(stop_correct)) / mean_cost


def oracle_score_maps(
    values: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    maps: dict[str, dict[str, float]] = {
        "mixed": {},
        "hindsight_on": {},
        "off_spine": {},
    }
    for row in values:
        if row["split"] != "confirmation":
            continue
        checkpoint_id = str(row["checkpoint_id"])
        correct = tuple(int(value) for value in row["continuation_correct"])
        costs = tuple(int(value) for value in row["continuation_costs"])
        if len(correct) != 4 or len(costs) != 4:
            raise ValueError(f"{checkpoint_id}: expected four continuation outcomes")
        maps["mixed"][checkpoint_id] = float(row["marginal_value"])
        maps["hindsight_on"][checkpoint_id] = token_normalized_value(
            int(row["stop_correct"]),
            correct[:1],
            costs[:1],
        )
        maps["off_spine"][checkpoint_id] = token_normalized_value(
            int(row["stop_correct"]),
            correct[1:],
            costs[1:],
        )
    expected = 22_320
    if any(len(values_by_id) != expected for values_by_id in maps.values()):
        raise ValueError(
            "oracle score maps must cover all 22,320 confirmation transitions"
        )
    return maps


def load_confirmation_frontiers(
    run_dir: Path,
    evaluation: dict[str, Any],
    values: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    path = run_dir / "evaluation" / "frontiers.jsonl.gz"
    states = [row for row in iter_jsonl_gz(path) if row["split"] == "confirmation"]
    if len(states) != 23_808:
        raise ValueError(f"expected 23,808 confirmation frontiers, found {len(states)}")
    by_id = {str(row["checkpoint_id"]): row for row in states}
    if len(by_id) != len(states):
        raise ValueError("confirmation frontier IDs are not unique")
    checked = 0
    matched = 0
    for row in values:
        if row["split"] != "confirmation":
            continue
        checkpoint_id = str(row["checkpoint_id"])
        current = by_id.get(checkpoint_id)
        if current is None:
            raise ValueError(f"missing replay frontier: {checkpoint_id}")
        successor_id = current.get("successor_id")
        successor = by_id.get(str(successor_id))
        if successor is None:
            raise ValueError(f"missing replay successor: {checkpoint_id}")
        branch_one = int(row["continuation_correct"][0])
        checked += 1
        matched += int(branch_one == int(successor["stop_correct"]))
        transition_cost = int(current["transition_cost"])
        if transition_cost != int(row["continuation_costs"][0]):
            raise ValueError(
                f"on-spine cost mismatch at {checkpoint_id}: "
                f"{transition_cost} != {row['continuation_costs'][0]}"
            )
    if checked != 22_320 or matched != checked:
        raise ValueError(
            "branch-1/successor identity must hold for every confirmation "
            f"transition: matched={matched} checked={checked}"
        )
    return states, {
        "eligible_transitions": checked,
        "branch1_successor_matches": matched,
    }


def leave_one_branch_out_rows(
    values: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        if value["split"] != "confirmation":
            continue
        outcomes = tuple(int(outcome) for outcome in value["continuation_correct"])
        if len(outcomes) != 4:
            raise ValueError("leave-one-out analysis requires four branches")
        stop_correct = int(value["stop_correct"])
        target = "beneficial" if stop_correct == 0 else "harmful"
        for held_out in range(4):
            remaining = outcomes[:held_out] + outcomes[held_out + 1 :]
            if target == "beneficial":
                label = outcomes[held_out]
                probability = sum(remaining) / len(remaining)
            else:
                label = 1 - outcomes[held_out]
                probability = 1.0 - sum(remaining) / len(remaining)
            rows.append(
                {
                    "problem_id": value["problem_id"],
                    "checkpoint_id": value["checkpoint_id"],
                    "target": target,
                    "held_out_branch": held_out + 1,
                    "held_out_role": ("on_spine" if held_out == 0 else "off_spine"),
                    "label": int(label),
                    "probability": float(probability),
                }
            )
    if len(rows) != 89_280:
        raise ValueError(f"expected 89,280 leave-one-out rows, found {len(rows)}")
    return rows


def _discrete_metrics_from_counts(
    counts: dict[float, tuple[float, float]],
) -> dict[str, float | None]:
    positives = sum(value[0] for value in counts.values())
    negatives = sum(value[1] for value in counts.values())
    total = positives + negatives
    if total <= 0:
        return {
            "event_rate": None,
            "auroc": None,
            "auprc": None,
            "brier": None,
        }
    concordant = 0.0
    negatives_below = 0.0
    for score in sorted(counts):
        positive, negative = counts[score]
        concordant += positive * (negatives_below + 0.5 * negative)
        negatives_below += negative
    auroc = (
        concordant / (positives * negatives)
        if positives > 0 and negatives > 0
        else None
    )
    true_positives = 0.0
    predicted_positives = 0.0
    auprc = 0.0
    if positives > 0:
        for score in sorted(counts, reverse=True):
            positive, negative = counts[score]
            true_positives += positive
            predicted_positives += positive + negative
            auprc += positive / positives * (true_positives / predicted_positives)
    else:
        auprc = None
    squared_error = sum(
        positive * (1.0 - score) ** 2 + negative * score**2
        for score, (positive, negative) in counts.items()
    )
    return {
        "event_rate": positives / total,
        "auroc": auroc,
        "auprc": auprc,
        "brier": squared_error / total,
    }


def discrete_problem_bootstrap(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    by_problem: dict[str, dict[float, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    for row in rows:
        score = float(row["probability"])
        label = int(row["label"])
        by_problem[str(row["problem_id"])][score][0 if label else 1] += 1
    problem_ids = sorted(by_problem)
    if len(problem_ids) < 2:
        raise ValueError("leave-one-out bootstrap needs multiple problems")
    observed_counts: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for problem in by_problem.values():
        for score, pair in problem.items():
            observed_counts[score][0] += pair[0]
            observed_counts[score][1] += pair[1]
    observed = _discrete_metrics_from_counts(
        {score: (pair[0], pair[1]) for score, pair in observed_counts.items()}
    )
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        counts: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for problem_id in (rng.choice(problem_ids) for _ in problem_ids):
            for score, pair in by_problem[problem_id].items():
                counts[score][0] += pair[0]
                counts[score][1] += pair[1]
        metrics = _discrete_metrics_from_counts(
            {score: (pair[0], pair[1]) for score, pair in counts.items()}
        )
        for name, metric in metrics.items():
            if metric is not None and math.isfinite(metric):
                samples[name].append(float(metric))
    output: dict[str, dict[str, float | int | None]] = {}
    for name, estimate in observed.items():
        metric_samples = samples[name]
        output[name] = {
            "estimate": estimate,
            "low": (
                float(np.quantile(metric_samples, 0.025)) if metric_samples else None
            ),
            "high": (
                float(np.quantile(metric_samples, 0.975)) if metric_samples else None
            ),
            "replicates": len(metric_samples),
        }
    return output


def cluster_ratio_interval(
    rows: list[dict[str, Any]],
    numerator: Callable[[dict[str, Any]], float],
    denominator: Callable[[dict[str, Any]], float],
    *,
    seed_offset: int,
) -> dict[str, float | int | None]:
    by_problem: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        bucket = by_problem[row["problem_id"]]
        bucket[0] += numerator(row)
        bucket[1] += denominator(row)
    problem_ids = sorted(by_problem)
    values = np.asarray([by_problem[problem_id] for problem_id in problem_ids])
    total_denominator = float(values[:, 1].sum())
    estimate = (
        float(values[:, 0].sum()) / total_denominator if total_denominator else None
    )
    if not problem_ids or estimate is None:
        return {
            "estimate": estimate,
            "low": None,
            "high": None,
            "replicates": 0,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indexes = rng.integers(0, len(problem_ids), size=len(problem_ids))
        sampled = values[indexes].sum(axis=0)
        if sampled[1] > 0:
            samples.append(float(sampled[0] / sampled[1]))
    return {
        "estimate": estimate,
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "replicates": len(samples),
    }


def cluster_contrast_interval(
    rows: list[dict[str, Any]],
    *,
    low_threshold: float,
    seed_offset: int,
) -> dict[str, float | int | None]:
    by_problem: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    high_n = 0
    low_n = 0
    for row in rows:
        persistence = row["persistence"]
        if persistence is None:
            continue
        bucket = by_problem[row["problem_id"]]
        if math.isclose(float(persistence), 1.0):
            bucket[0] += row["marginal_value"]
            bucket[1] += 1
            high_n += 1
        elif float(persistence) <= low_threshold:
            bucket[2] += row["marginal_value"]
            bucket[3] += 1
            low_n += 1
    problem_ids = sorted(by_problem)
    values = np.asarray([by_problem[problem_id] for problem_id in problem_ids])

    def contrast(vector: np.ndarray) -> float | None:
        if vector[1] <= 0 or vector[3] <= 0:
            return None
        return float(vector[0] / vector[1] - vector[2] / vector[3])

    estimate = contrast(values.sum(axis=0))
    if estimate is None:
        return {
            "estimate": None,
            "low": None,
            "high": None,
            "replicates": 0,
            "high_n": high_n,
            "low_n": low_n,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indexes = rng.integers(0, len(problem_ids), size=len(problem_ids))
        value = contrast(values[indexes].sum(axis=0))
        if value is not None:
            samples.append(value)
    return {
        "estimate": estimate,
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "replicates": len(samples),
        "high_n": high_n,
        "low_n": low_n,
    }


def fmt_ci(interval: dict[str, Any], *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    digits = 1 if percent else 3
    estimate = scale * float(interval["estimate"])
    low = scale * float(interval["low"])
    high = scale * float(interval["high"])
    suffix = r"\%" if percent else ""
    return f"{estimate:.{digits}f}{suffix} " f"[{low:.{digits}f}, {high:.{digits}f}]"


def render_rq1(values_by_model: dict[str, list[dict[str, Any]]]) -> list[Path]:
    summary_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    dataset_depth_rows: list[dict[str, Any]] = []
    seed_offset = 100
    for model, all_rows in values_by_model.items():
        confirmation = [row for row in all_rows if row["split"] == "confirmation"]
        for dataset in ("all", "gsm8k", "math500"):
            selected = (
                confirmation
                if dataset == "all"
                else [row for row in confirmation if row["dataset_id"] == dataset]
            )
            wrong = [row for row in selected if row["stop_correct"] == 0]
            correct = [row for row in selected if row["stop_correct"] == 1]
            beneficial = cluster_ratio_interval(
                wrong,
                lambda row: row["beneficial"],
                lambda row: 1.0,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            beneficial_magnitude = cluster_ratio_interval(
                wrong,
                lambda row: (row["marginal_value"] if row["beneficial"] else 0.0),
                lambda row: float(row["beneficial"]),
                seed_offset=seed_offset,
            )
            seed_offset += 1
            harmful = cluster_ratio_interval(
                correct,
                lambda row: row["harmful"],
                lambda row: 1.0,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            harmful_magnitude = cluster_ratio_interval(
                correct,
                lambda row: (-row["marginal_value"] if row["harmful"] else 0.0),
                lambda row: float(row["harmful"]),
                seed_offset=seed_offset,
            )
            seed_offset += 1
            summary_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "wrong_checkpoints": len(wrong),
                    "beneficial_events": sum(row["beneficial"] for row in wrong),
                    "beneficial_rate": beneficial["estimate"],
                    "beneficial_rate_low": beneficial["low"],
                    "beneficial_rate_high": beneficial["high"],
                    "beneficial_magnitude": beneficial_magnitude["estimate"],
                    "beneficial_magnitude_low": beneficial_magnitude["low"],
                    "beneficial_magnitude_high": beneficial_magnitude["high"],
                    "correct_checkpoints": len(correct),
                    "harmful_events": sum(row["harmful"] for row in correct),
                    "harmful_rate": harmful["estimate"],
                    "harmful_rate_low": harmful["low"],
                    "harmful_rate_high": harmful["high"],
                    "harmful_magnitude": harmful_magnitude["estimate"],
                    "harmful_magnitude_low": harmful_magnitude["low"],
                    "harmful_magnitude_high": harmful_magnitude["high"],
                }
            )
        for target, stop_correct, event_field in (
            ("beneficial", 0, "beneficial"),
            ("harmful", 1, "harmful"),
        ):
            for dataset in ("all", "gsm8k", "math500"):
                for depth in range(256, 4096, 256):
                    group = [
                        row
                        for row in confirmation
                        if row["stop_correct"] == stop_correct
                        and row["depth"] == depth
                        and (dataset == "all" or row["dataset_id"] == dataset)
                    ]
                    if not group:
                        continue
                    events = sum(row[event_field] for row in group)
                    dataset_depth_rows.append(
                        {
                            "model": model,
                            "target": target,
                            "dataset": dataset,
                            "depth": depth,
                            "checkpoints": len(group),
                            "events": events,
                            "event_rate": events / len(group),
                        }
                    )
            for difficulty in ("low", "middle", "high"):
                for depth in range(256, 4096, 256):
                    group = [
                        row
                        for row in confirmation
                        if row["stop_correct"] == stop_correct
                        and row["difficulty"] == difficulty
                        and row["depth"] == depth
                    ]
                    if not group:
                        continue
                    events = sum(row[event_field] for row in group)
                    depth_rows.append(
                        {
                            "model": model,
                            "target": target,
                            "difficulty": difficulty,
                            "depth": depth,
                            "checkpoints": len(group),
                            "events": events,
                            "event_rate": events / len(group),
                        }
                    )

    csv_path = OUT / "rq1_summary.csv"
    depth_csv = OUT / "rq1_depth.csv"
    dataset_depth_csv = OUT / "rq1_depth_by_dataset.csv"
    write_csv(csv_path, summary_rows)
    write_csv(depth_csv, depth_rows)
    write_csv(dataset_depth_csv, dataset_depth_rows)

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"Model & Data & Wrong $n$ & Beneficial rate (95\% CI) & "
            r"Correct $n$ & Harmful rate (95\% CI) \\"
        ),
        r"\midrule",
    ]
    for row in summary_rows:
        if row["dataset"] == "all":
            tex_lines.append(r"\addlinespace")
        dataset = {
            "all": "All",
            "gsm8k": "GSM8K",
            "math500": "MATH-500",
        }[row["dataset"]]
        beneficial_interval = {
            "estimate": row["beneficial_rate"],
            "low": row["beneficial_rate_low"],
            "high": row["beneficial_rate_high"],
        }
        harmful_interval = {
            "estimate": row["harmful_rate"],
            "low": row["harmful_rate_low"],
            "high": row["harmful_rate_high"],
        }
        tex_lines.append(
            "{} & {} & {:,} & {} & {:,} & {} \\\\".format(
                MODEL_SHORT[row["model"]],
                dataset,
                row["wrong_checkpoints"],
                fmt_ci(beneficial_interval, percent=True),
                row["correct_checkpoints"],
                fmt_ci(harmful_interval, percent=True),
            )
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Confirmation prevalence of signed continuation "
                r"events. Intervals resample complete problems 2,000 times.}"
            ),
            r"\label{tab:rq1-prevalence}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = OUT / "rq1_table.tex"
    atomic_text(tex_path, "\n".join(tex_lines))

    import matplotlib.pyplot as plt

    colors = {"all": "#111111", "gsm8k": "#1f77b4", "math500": "#d62728"}
    labels = {"all": "All", "gsm8k": "GSM8K", "math500": "MATH-500"}
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.8), sharex=True)
    for row_index, model in enumerate(RUNS):
        for column_index, target in enumerate(("beneficial", "harmful")):
            axis = axes[row_index, column_index]
            for dataset in ("all", "gsm8k", "math500"):
                rows = [
                    row
                    for row in dataset_depth_rows
                    if row["model"] == model
                    and row["target"] == target
                    and row["dataset"] == dataset
                ]
                axis.plot(
                    [row["depth"] for row in rows],
                    [100 * row["event_rate"] for row in rows],
                    marker="o",
                    markersize=3,
                    linewidth=1.6,
                    label=labels[dataset],
                    color=colors[dataset],
                )
            axis.grid(alpha=0.25)
            axis.set_title(
                f"{MODEL_SHORT[model]}: {TARGET_LABELS[target]}",
                fontsize=10,
            )
            axis.set_ylabel("Event rate (%)")
            axis.set_xlabel("Frozen-prefix depth (tokens)")
            axis.set_xticks(range(512, 4097, 512))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figure_path = OUT / "rq1_depth.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return [
        csv_path,
        depth_csv,
        dataset_depth_csv,
        tex_path,
        figure_path,
    ]


def render_protocol_overview() -> list[Path]:
    """Render the intervention, estimand, and two downstream audit paths."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, axis = plt.subplots(figsize=(12.0, 4.15))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4.15)
    axis.axis("off")

    colors = {
        "prefix": "#E8EEF8",
        "action": "#E9F5EC",
        "label": "#FFF2D8",
        "value": "#FDE6E3",
        "audit": "#EEE9FA",
        "ledger": "#F1F3F5",
        "edge": "#34495E",
    }

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        color: str,
        *,
        fontsize: float = 9.0,
        linewidth: float = 1.1,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            facecolor=color,
            edgecolor=colors["edge"],
            linewidth=linewidth,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#17202A",
            linespacing=1.25,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        connectionstyle: str = "arc3,rad=0",
    ) -> None:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.25,
                "color": colors["edge"],
                "shrinkA": 2,
                "shrinkB": 2,
                "connectionstyle": connectionstyle,
            },
        )

    box(
        0.25,
        1.55,
        1.65,
        1.05,
        "Frozen prefix $s$\nidentical prompt +\nreasoning tokens",
        colors["prefix"],
        fontsize=9.2,
    )
    box(
        2.35,
        2.58,
        2.05,
        0.88,
        "Answer now\nfixed operator $\\mathcal{A}$",
        colors["action"],
    )
    box(
        2.35,
        0.72,
        2.05,
        1.12,
        "Continue 256 tokens $\\times 4$\n1 on-spine + 3 off-spine\nthen the same $\\mathcal{A}$",
        colors["action"],
        fontsize=8.7,
    )
    box(
        4.90,
        2.58,
        1.48,
        0.88,
        "Stop label\n$S_s$",
        colors["label"],
    )
    box(
        4.90,
        0.72,
        1.48,
        1.12,
        "Branch outcomes\n$C_{s,k},\\ L_{s,k}$",
        colors["label"],
    )
    box(
        6.88,
        1.48,
        1.68,
        1.22,
        (
            "Signed local value\n"
            "$M_s^{\\mathcal{A}}=1000\\,"
            "\\frac{\\overline{C}_s-S_s}{\\overline{L}_s}$"
        ),
        colors["value"],
        fontsize=9.0,
        linewidth=1.35,
    )
    box(
        9.05,
        2.48,
        2.7,
        1.02,
        "Proxy audit\nprediction, calibration,\nmatching, local information",
        colors["audit"],
        fontsize=8.8,
    )
    box(
        9.05,
        0.66,
        2.7,
        1.02,
        "Equal-ledger replay\non-spine outcomes\naccuracy--cost curves + AULC",
        colors["audit"],
        fontsize=8.8,
    )

    arrow((1.90, 2.08), (2.35, 3.02), connectionstyle="arc3,rad=-0.10")
    arrow((1.90, 2.08), (2.35, 1.28), connectionstyle="arc3,rad=0.10")
    arrow((4.40, 3.02), (4.90, 3.02))
    arrow((4.40, 1.28), (4.90, 1.28))
    arrow((6.38, 3.02), (6.88, 2.40), connectionstyle="arc3,rad=0.10")
    arrow((6.38, 1.28), (6.88, 1.78), connectionstyle="arc3,rad=-0.10")
    arrow((8.56, 2.34), (9.05, 2.90))
    arrow((8.56, 1.82), (9.05, 1.18))

    axis.text(
        5.40,
        3.78,
        "External benchmark labels enter only after generation",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#7B4A12",
    )
    box(
        0.85,
        0.05,
        10.15,
        0.38,
        (
            "Full ledger: prompt + reasoning + answer probes + scoring + "
            "policy observations    |    physical sidecar: GPU-s, memory, FLOPs"
        ),
        colors["ledger"],
        fontsize=8.2,
        linewidth=0.9,
    )

    figure_path = OUT / "protocol_overview.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return [figure_path]


def render_proxy_intervals(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for model, payload in validated.items():
        metrics = payload["evaluation"]["metrics"]
        for target in ("beneficial", "harmful"):
            stop_correct = 0 if target == "beneficial" else 1
            event_field = target
            for proxy, values in metrics[target].items():
                analysis_rows = [
                    row
                    for row in values_by_model[model]
                    if row["split"] == "analysis"
                    and row["stop_correct"] == stop_correct
                    and row["proxies"].get(proxy) is not None
                ]
                confirmation_rows = [
                    row
                    for row in values_by_model[model]
                    if row["split"] == "confirmation"
                    and row["stop_correct"] == stop_correct
                    and row["proxies"].get(proxy) is not None
                ]
                if len(analysis_rows) != int(values["analysis_n"]):
                    raise ValueError(
                        f"{model}/{target}/{proxy}: analysis availability "
                        "does not reproduce sealed metric"
                    )
                if len(confirmation_rows) != int(values["confirmation_n"]):
                    raise ValueError(
                        f"{model}/{target}/{proxy}: confirmation availability "
                        "does not reproduce sealed metric"
                    )
                analysis_prevalence = float(
                    np.mean([float(row[event_field]) for row in analysis_rows])
                )
                confirmation_prevalence = float(
                    np.mean([float(row[event_field]) for row in confirmation_rows])
                )
                brier_reference = float(
                    np.mean(
                        [
                            (float(row[event_field]) - analysis_prevalence) ** 2
                            for row in confirmation_rows
                        ]
                    )
                )
                if confirmation_prevalence <= 0 or brier_reference <= 0:
                    raise ValueError(
                        f"{model}/{target}/{proxy}: rare-event baseline " "is undefined"
                    )
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "proxy": proxy,
                        "analysis_n": values["analysis_n"],
                        "confirmation_n": values["confirmation_n"],
                        "analysis_prevalence": analysis_prevalence,
                        "confirmation_prevalence": confirmation_prevalence,
                        "auroc": values["auroc"],
                        "auroc_low": values["auroc_ci"]["low"],
                        "auroc_high": values["auroc_ci"]["high"],
                        "auprc": values["auprc"],
                        "auprc_low": values["auprc_ci"]["low"],
                        "auprc_high": values["auprc_ci"]["high"],
                        "auprc_lift_over_prevalence": (
                            float(values["auprc"]) / confirmation_prevalence
                        ),
                        "brier": values["brier"],
                        "brier_low": values["brier_ci"]["low"],
                        "brier_high": values["brier_ci"]["high"],
                        "brier_reference_analysis_climatology": (brier_reference),
                        "brier_skill": (1.0 - float(values["brier"]) / brier_reference),
                    }
                )
    csv_path = OUT / "proxy_metrics_with_ci.csv"
    write_csv(csv_path, rows)

    best_rows: list[dict[str, Any]] = []
    for model in RUNS:
        for target in ("beneficial", "harmful"):
            best_rows.append(
                max(
                    (
                        row
                        for row in rows
                        if row["model"] == model and row["target"] == target
                    ),
                    key=lambda row: float(row["auroc"]),
                )
            )

    def interval_text(row: dict[str, Any], metric: str) -> str:
        return (
            f"{float(row[metric]):.3f} "
            f"[{float(row[metric + '_low']):.3f}, "
            f"{float(row[metric + '_high']):.3f}]"
        )

    best_tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        (
            r"Model & Target & Best proxy & Prev. & AUROC (95\% CI) & "
            r"AP (95\% CI) & AP/prev. & Brier & BSS \\"
        ),
        r"\midrule",
    ]
    for row in best_rows:
        best_tex.append(
            "{} & {} & {} & {:.3f} & {} & {} & {:.2f}$\\times$ & "
            "{:.3f} & {:+.3f} \\\\".format(
                MODEL_SHORT[row["model"]],
                row["target"].capitalize(),
                PROXY_LABELS[row["proxy"]],
                row["confirmation_prevalence"],
                interval_text(row, "auroc"),
                interval_text(row, "auprc"),
                row["auprc_lift_over_prevalence"],
                row["brier"],
                row["brier_skill"],
            )
        )
    best_tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Best confirmation proxy per model and feasible "
                r"target, selected by AUROC. Prev.\ is the confirmation "
                r"no-skill AP baseline; AP/prev.\ is average-precision lift. "
                r"Brier skill score (BSS) uses a constant probability frozen "
                r"to the corresponding analysis-set prevalence. Every "
                r"interval uses 2,000 problem-level bootstrap replicates.}"
            ),
            r"\label{tab:best-proxy-ci}",
            r"\end{table*}",
            "",
        ]
    )
    best_path = OUT / "proxy_best_ci_table.tex"
    atomic_text(best_path, "\n".join(best_tex))

    full_tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        (
            r"Model & Target & Proxy & $n_{\rm conf}$ & AUROC "
            r"(95\% CI) & AUPRC (95\% CI) & Brier (95\% CI) \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        full_tex.append(
            "{} & {} & {} & {:,} & {} & {} & {} \\\\".format(
                MODEL_SHORT[row["model"]],
                row["target"].capitalize(),
                PROXY_LABELS[row["proxy"]],
                row["confirmation_n"],
                interval_text(row, "auroc"),
                interval_text(row, "auprc"),
                interval_text(row, "brier"),
            )
        )
    full_tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Complete confirmation discrimination and "
                r"calibration results with problem-bootstrap intervals.}"
            ),
            r"\label{tab:full-proxy-ci}",
            r"\end{table*}",
            "",
        ]
    )
    full_path = OUT / "proxy_full_ci_table.tex"
    atomic_text(full_path, "\n".join(full_tex))

    rare_tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        (
            r"Model & Target & Proxy & Analysis prev. & Conf.\ prev. & "
            r"AP & AP/prev. & Brier ref. & BSS \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        rare_tex.append(
            "{} & {} & {} & {:.3f} & {:.3f} & {:.3f} & "
            "{:.2f}$\\times$ & {:.3f} & {:+.3f} \\\\".format(
                MODEL_SHORT[row["model"]],
                row["target"].capitalize(),
                PROXY_LABELS[row["proxy"]],
                row["analysis_prevalence"],
                row["confirmation_prevalence"],
                row["auprc"],
                row["auprc_lift_over_prevalence"],
                row["brier_reference_analysis_climatology"],
                row["brier_skill"],
            )
        )
    rare_tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Rare-event baselines on each proxy's available-case "
                r"population. Confirmation prevalence is the no-skill AP "
                r"baseline. The Brier reference predicts the analysis-set "
                r"prevalence for every confirmation row; "
                r"$\mathrm{BSS}=1-\mathrm{BS}/\mathrm{BS}_{\rm ref}$.}"
            ),
            r"\label{tab:rare-event-baselines}",
            r"\end{table*}",
            "",
        ]
    )
    rare_path = OUT / "proxy_rare_event_baselines_table.tex"
    atomic_text(rare_path, "\n".join(rare_tex))
    return [csv_path, best_path, full_path, rare_path]


def render_calibration(validated: dict[str, dict[str, Any]]) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for model, payload in validated.items():
        for target, proxy_rows in payload["evaluation"]["metrics"].items():
            for proxy, metrics in proxy_rows.items():
                for calibration in metrics["calibration"]:
                    rows.append(
                        {
                            "model": model,
                            "target": target,
                            "proxy": proxy,
                            **calibration,
                        }
                    )
    csv_path = OUT / "calibration_with_labels.csv"
    write_csv(csv_path, rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True, sharey=True)
    palette = {
        "anll": "#1f77b4",
        "answer_confidence": "#ff7f0e",
        "modal_agreement": "#2ca02c",
        "persistence": "#d62728",
        "prm_min_step": "#9467bd",
        "top20_entropy": "#8c564b",
    }
    for row_index, model in enumerate(RUNS):
        for column_index, target in enumerate(("beneficial", "harmful")):
            axis = axes[row_index, column_index]
            axis.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1)
            for proxy in PROXY_LABELS:
                proxy_rows = [
                    row
                    for row in rows
                    if row["model"] == model
                    and row["target"] == target
                    and row["proxy"] == proxy
                ]
                proxy_rows.sort(key=lambda row: row["mean_probability"])
                axis.plot(
                    [row["mean_probability"] for row in proxy_rows],
                    [row["empirical_rate"] for row in proxy_rows],
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    color=palette[proxy],
                    label=PROXY_LABELS[proxy],
                )
            axis.grid(alpha=0.2)
            axis.set_title(
                f"{MODEL_SHORT[model]}: {target.capitalize()}",
                fontsize=10,
            )
            axis.set_xlabel("Predicted probability")
            axis.set_ylabel("Empirical frequency")
            axis.set_xlim(-0.02, 1.02)
            axis.set_ylim(-0.02, 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figure_path = OUT / "calibration_curves.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return [csv_path, figure_path]


def render_collision(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for model, payload in validated.items():
        value_map = {
            row["checkpoint_id"]: row
            for row in values_by_model[model]
            if row["split"] == "confirmation"
        }
        gap_accumulator: dict[tuple[str, str], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        for cell in iter_jsonl_gz(
            payload["run_dir"] / "evaluation" / "matched_cells.jsonl.gz"
        ):
            key = cell["key"]
            target = "beneficial" if key["stop_correct"] == 0 else "harmful"
            nonzero = [
                value_map[checkpoint_id]["marginal_value"]
                for checkpoint_id in cell["nonzero_checkpoint_ids"]
            ]
            zero = [
                value_map[checkpoint_id]["marginal_value"]
                for checkpoint_id in cell["zero_checkpoint_ids"]
            ]
            weight = min(len(nonzero), len(zero))
            gap = sum(nonzero) / len(nonzero) - sum(zero) / len(zero)
            bucket = gap_accumulator[(cell["proxy_name"], target)]
            bucket[0] += weight * gap
            bucket[1] += weight
        collision = payload["evaluation"]["collision"]
        for proxy, proxy_values in collision["by_proxy"].items():
            beneficial = gap_accumulator[(proxy, "beneficial")]
            harmful = gap_accumulator[(proxy, "harmful")]
            rows.append(
                {
                    "model": model,
                    "proxy": proxy,
                    "collision_rate": proxy_values["collision_rate"],
                    "matched_cells": proxy_values["matched_cells"],
                    "beneficial_gap": (
                        beneficial[0] / beneficial[1] if beneficial[1] else None
                    ),
                    "harmful_gap": (harmful[0] / harmful[1] if harmful[1] else None),
                    "beneficial_weight": int(beneficial[1]),
                    "harmful_weight": int(harmful[1]),
                }
            )
    csv_path = OUT / "collision_by_proxy.csv"
    write_csv(csv_path, rows)
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        (
            r"Model & Proxy & Collision rate & Cells & Beneficial "
            r"$\Delta M_s$ & Harmful $\Delta M_s$ \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            "{} & {} & {:.1f}\\% & {:,} & {:+.3f} & {:+.3f} \\\\".format(
                MODEL_SHORT[row["model"]],
                PROXY_LABELS[row["proxy"]],
                100 * row["collision_rate"],
                row["matched_cells"],
                row["beneficial_gap"],
                row["harmful_gap"],
            )
        )
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Matched-cell collision diagnostics. Gaps compare "
                r"the feasible nonzero class with zero-value controls inside "
                r"cells fixed by model, dataset, difficulty, depth, current "
                r"correctness, proxy, and analysis-learned proxy bin.}"
            ),
            r"\label{tab:collision-detail}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = OUT / "collision_table.tex"
    atomic_text(tex_path, "\n".join(tex))
    return [csv_path, tex_path]


def _proxy_bins(
    analysis_rows: list[dict[str, Any]],
    proxy: str,
    max_bins: int,
) -> Any:
    values = [
        float(row["proxies"][proxy])
        for row in analysis_rows
        if row["proxies"].get(proxy) is not None
    ]
    if proxy in {"persistence", "modal_agreement"}:
        return fit_tie_bins(values, max_bins=max(1, len(set(values))))
    return fit_tie_bins(values, max_bins=max_bins)


def _collision_rate_for_bins(
    confirmation: list[dict[str, Any]],
    *,
    proxy: str,
    bins: Any,
    include_difficulty: bool = True,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    eligible = [row for row in confirmation if row["proxies"].get(proxy) is not None]
    for row in eligible:
        groups[
            (
                *_diagnostic_stratum_key(
                    row,
                    include_difficulty=include_difficulty,
                ),
                bins.assign(float(row["proxies"][proxy])),
            )
        ].append(row)
    collision_ids: set[str] = set()
    matched_cells = 0
    gap_accumulator: dict[str, list[float]] = {
        "beneficial": [0.0, 0.0],
        "harmful": [0.0, 0.0],
    }
    for group in groups.values():
        stop_correct = int(group[0]["stop_correct"])
        target = "beneficial" if stop_correct == 0 else "harmful"
        labels = {int(row[target]) for row in group}
        if labels == {0, 1}:
            matched_cells += 1
            collision_ids.update(str(row["checkpoint_id"]) for row in group)
            if all("marginal_value" in row for row in group):
                nonzero = [
                    float(row["marginal_value"])
                    for row in group
                    if int(row[target]) == 1
                ]
                zero = [
                    float(row["marginal_value"])
                    for row in group
                    if int(row[target]) == 0
                ]
                weight = min(len(nonzero), len(zero))
                gap = float(np.mean(nonzero)) - float(np.mean(zero))
                gap_accumulator[target][0] += weight * gap
                gap_accumulator[target][1] += weight
    return {
        "eligible_checkpoints": len(eligible),
        "matched_cells": matched_cells,
        "collision_members": len(collision_ids),
        "collision_rate": (len(collision_ids) / len(eligible) if eligible else None),
        "beneficial_gap": (
            gap_accumulator["beneficial"][0] / gap_accumulator["beneficial"][1]
            if gap_accumulator["beneficial"][1]
            else None
        ),
        "harmful_gap": (
            gap_accumulator["harmful"][0] / gap_accumulator["harmful"][1]
            if gap_accumulator["harmful"][1]
            else None
        ),
        "beneficial_weight": int(gap_accumulator["beneficial"][1]),
        "harmful_weight": int(gap_accumulator["harmful"][1]),
    }


def _diagnostic_stratum_key(
    row: dict[str, Any],
    *,
    include_difficulty: bool,
) -> tuple[Any, ...]:
    key: list[Any] = [row["dataset_id"]]
    if include_difficulty:
        key.append(row["difficulty"])
    key.extend(
        [
            int(row["depth"]),
            int(row["stop_correct"]),
        ]
    )
    return tuple(key)


def _analysis_orientation(
    rows: list[dict[str, Any]],
    *,
    proxy: str,
    target: str,
) -> float:
    stop_correct = 0 if target == "beneficial" else 1
    selected = [
        row
        for row in rows
        if int(row["stop_correct"]) == stop_correct
        and row["proxies"].get(proxy) is not None
    ]
    positives = [
        float(row["proxies"][proxy]) for row in selected if int(row[target]) == 1
    ]
    negatives = [
        float(row["proxies"][proxy]) for row in selected if int(row[target]) == 0
    ]
    if not positives or not negatives:
        raise ValueError(f"cannot orient {proxy}/{target}")
    return 1.0 if float(np.mean(positives)) >= float(np.mean(negatives)) else -1.0


def _conditional_auroc(
    rows: list[dict[str, Any]],
    *,
    proxy: str,
    target: str,
    orientation: float,
    include_difficulty: bool = True,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["proxies"].get(proxy) is None:
            continue
        groups[
            _diagnostic_stratum_key(
                row,
                include_difficulty=include_difficulty,
            )
        ].append(row)
    concordant_pairs = 0.0
    comparable_pairs = 0
    effective_strata = 0
    for group in groups.values():
        labels = [int(row[target]) for row in group]
        scores = [orientation * float(row["proxies"][proxy]) for row in group]
        positives = sum(labels)
        pairs = positives * (len(labels) - positives)
        auc = roc_auc(labels, scores)
        if auc is None or pairs == 0:
            continue
        concordant_pairs += float(auc) * pairs
        comparable_pairs += pairs
        effective_strata += 1
    return {
        "conditional_auroc": (
            concordant_pairs / comparable_pairs if comparable_pairs else None
        ),
        "comparable_pairs": comparable_pairs,
        "effective_strata": effective_strata,
    }


def _nearest_neighbor_disagreement(
    rows: list[dict[str, Any]],
    *,
    proxy: str,
    target: str,
    orientation: float,
    include_difficulty: bool = True,
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    """Expected disagreement with the closest other-problem proxy neighbor."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["proxies"].get(proxy) is None:
            continue
        groups[
            _diagnostic_stratum_key(
                row,
                include_difficulty=include_difficulty,
            )
        ].append(row)
    per_problem: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    distances: list[float] = []
    eligible = 0
    for group in groups.values():
        buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            buckets[orientation * float(row["proxies"][proxy])].append(row)
        values = sorted(buckets)
        for value_index, value in enumerate(values):
            bucket = buckets[value]
            for row in bucket:
                problem_id = str(row["problem_id"])
                candidates = [
                    candidate
                    for candidate in bucket
                    if str(candidate["problem_id"]) != problem_id
                ]
                distance = 0.0
                if not candidates:
                    alternatives: list[tuple[float, list[dict[str, Any]]]] = []
                    for direction in (-1, 1):
                        neighbor_index = value_index + direction
                        while 0 <= neighbor_index < len(values):
                            neighbor_value = values[neighbor_index]
                            neighbor_candidates = [
                                candidate
                                for candidate in buckets[neighbor_value]
                                if str(candidate["problem_id"]) != problem_id
                            ]
                            if not neighbor_candidates:
                                neighbor_index += direction
                                continue
                            alternatives.append(
                                (
                                    abs(neighbor_value - value),
                                    neighbor_candidates,
                                )
                            )
                            break
                    if not alternatives:
                        continue
                    distance = min(item[0] for item in alternatives)
                    candidates = [
                        candidate
                        for candidate_distance, neighbor_bucket in alternatives
                        if math.isclose(
                            candidate_distance,
                            distance,
                            rel_tol=0.0,
                            abs_tol=1e-15,
                        )
                        for candidate in neighbor_bucket
                    ]
                if not candidates:
                    continue
                disagreement = float(
                    np.mean(
                        [
                            int(candidate[target]) != int(row[target])
                            for candidate in candidates
                        ]
                    )
                )
                per_problem[problem_id][0] += disagreement
                per_problem[problem_id][1] += 1.0
                distances.append(distance)
                eligible += 1
    total_disagreement = sum(value[0] for value in per_problem.values())
    total_rows = sum(value[1] for value in per_problem.values())
    return (
        {
            "nn_disagreement_rate": (
                total_disagreement / total_rows if total_rows else None
            ),
            "nn_eligible": eligible,
            "nn_median_proxy_distance": (
                float(np.median(distances)) if distances else None
            ),
        },
        {problem_id: (value[0], value[1]) for problem_id, value in per_problem.items()},
    )


def _jeffreys_probability_map(
    rows: list[dict[str, Any]],
    *,
    target: str,
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> dict[tuple[Any, ...], float]:
    counts: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = counts[key(row)]
        bucket[0] += int(row[target])
        bucket[1] += 1
    return {
        cell: (positive + 0.5) / (total + 1.0)
        for cell, (positive, total) in counts.items()
    }


def _binary_entropy_bits(labels: list[int]) -> float:
    if not labels:
        return 0.0
    probability = sum(labels) / len(labels)
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(
        1.0 - probability
    )


def _plugin_conditional_entropy(
    rows: list[dict[str, Any]],
    *,
    target: str,
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> float:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(int(row[target]))
    total = len(rows)
    return sum(
        len(labels) / total * _binary_entropy_bits(labels) for labels in groups.values()
    )


def _incremental_cross_entropy(
    analysis: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    *,
    proxy: str,
    target: str,
    bins: Any,
    include_difficulty: bool = True,
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    baseline_key = lambda row: _diagnostic_stratum_key(
        row,
        include_difficulty=include_difficulty,
    )
    proxy_key = lambda row: (
        *baseline_key(row),
        bins.assign(float(row["proxies"][proxy])),
    )
    analysis_available = [
        row for row in analysis if row["proxies"].get(proxy) is not None
    ]
    confirmation_available = [
        row for row in confirmation if row["proxies"].get(proxy) is not None
    ]
    baseline_probabilities = _jeffreys_probability_map(
        analysis_available,
        target=target,
        key=baseline_key,
    )
    augmented_probabilities = _jeffreys_probability_map(
        analysis_available,
        target=target,
        key=proxy_key,
    )
    global_probability = (sum(int(row[target]) for row in analysis_available) + 0.5) / (
        len(analysis_available) + 1.0
    )
    baseline_losses: list[float] = []
    augmented_losses: list[float] = []
    per_problem: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    fallback_cells = 0
    for row in confirmation_available:
        label = int(row[target])
        baseline_probability = baseline_probabilities.get(
            baseline_key(row),
            global_probability,
        )
        augmented_probability = augmented_probabilities.get(proxy_key(row))
        if augmented_probability is None:
            augmented_probability = baseline_probability
            fallback_cells += 1
        baseline_loss = -(
            label * math.log(baseline_probability)
            + (1 - label) * math.log(1.0 - baseline_probability)
        ) / math.log(2.0)
        augmented_loss = -(
            label * math.log(augmented_probability)
            + (1 - label) * math.log(1.0 - augmented_probability)
        ) / math.log(2.0)
        baseline_losses.append(baseline_loss)
        augmented_losses.append(augmented_loss)
        bucket = per_problem[str(row["problem_id"])]
        bucket[0] += baseline_loss - augmented_loss
        bucket[1] += 1.0
    baseline_cross_entropy = float(np.mean(baseline_losses))
    augmented_cross_entropy = float(np.mean(augmented_losses))
    reduction = baseline_cross_entropy - augmented_cross_entropy
    plugin_baseline = _plugin_conditional_entropy(
        confirmation_available,
        target=target,
        key=baseline_key,
    )
    plugin_augmented = _plugin_conditional_entropy(
        confirmation_available,
        target=target,
        key=proxy_key,
    )
    return (
        {
            "confirmation_n": len(confirmation_available),
            "baseline_cross_entropy_bits": baseline_cross_entropy,
            "augmented_cross_entropy_bits": augmented_cross_entropy,
            "cross_entropy_reduction_bits": reduction,
            "relative_cross_entropy_reduction": (
                reduction / baseline_cross_entropy
                if baseline_cross_entropy > 0
                else None
            ),
            "plugin_conditional_entropy_bits": plugin_baseline,
            "plugin_proxy_conditional_entropy_bits": plugin_augmented,
            "plugin_entropy_reduction_bits": (plugin_baseline - plugin_augmented),
            "augmented_fallback_rows": fallback_cells,
        },
        {problem_id: (value[0], value[1]) for problem_id, value in per_problem.items()},
    )


def _cluster_ratio_bootstrap(
    contributions: dict[str, tuple[float, float]],
    *,
    namespace: str,
) -> tuple[float, float]:
    problem_ids = sorted(contributions)
    values = np.asarray(
        [contributions[problem_id] for problem_id in problem_ids],
        dtype=float,
    )
    rng = np.random.default_rng(
        stable_seed(BOOTSTRAP_SEED, COLLISION_ROBUSTNESS_NAMESPACE, namespace)
    )
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(problem_ids), size=len(problem_ids))
        total = values[selected].sum(axis=0)
        samples[index] = total[0] / total[1]
    return (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def render_collision_robustness(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    bin_rows: list[dict[str, Any]] = []
    information_rows: list[dict[str, Any]] = []
    deletion_collision_rows: list[dict[str, Any]] = []
    deletion_information_rows: list[dict[str, Any]] = []
    for model, all_rows in values_by_model.items():
        analysis = [row for row in all_rows if row["split"] == "analysis"]
        confirmation = [row for row in all_rows if row["split"] == "confirmation"]
        for proxy in PROXY_LABELS:
            for max_bins in (5, 10, 20, 50):
                bins = _proxy_bins(analysis, proxy, max_bins)
                values = _collision_rate_for_bins(
                    confirmation,
                    proxy=proxy,
                    bins=bins,
                )
                bin_rows.append(
                    {
                        "model": model,
                        "proxy": proxy,
                        "requested_max_bins": max_bins,
                        "effective_bins": len(bins.upper_bounds) + 1,
                        "matching": (
                            "exact_value"
                            if proxy in {"persistence", "modal_agreement"}
                            else "analysis_equal_frequency_tie_preserving"
                        ),
                        **values,
                    }
                )
            deletion_bins = _proxy_bins(analysis, proxy, 10)
            collision_with_difficulty = _collision_rate_for_bins(
                confirmation,
                proxy=proxy,
                bins=deletion_bins,
                include_difficulty=True,
            )
            collision_without_difficulty = _collision_rate_for_bins(
                confirmation,
                proxy=proxy,
                bins=deletion_bins,
                include_difficulty=False,
            )
            deletion_collision_rows.append(
                {
                    "model": model,
                    "proxy": proxy,
                    "requested_max_bins": 10,
                    "effective_bins": len(deletion_bins.upper_bounds) + 1,
                    **{
                        f"with_difficulty_{key}": value
                        for key, value in collision_with_difficulty.items()
                    },
                    **{
                        f"without_difficulty_{key}": value
                        for key, value in collision_without_difficulty.items()
                    },
                }
            )
            for target, stop_correct in (
                ("beneficial", 0),
                ("harmful", 1),
            ):
                target_analysis = [
                    row for row in analysis if int(row["stop_correct"]) == stop_correct
                ]
                target_confirmation = [
                    row
                    for row in confirmation
                    if int(row["stop_correct"]) == stop_correct
                ]
                orientation = _analysis_orientation(
                    target_analysis,
                    proxy=proxy,
                    target=target,
                )
                bins = _proxy_bins(target_analysis, proxy, 10)
                conditional_auc = _conditional_auroc(
                    target_confirmation,
                    proxy=proxy,
                    target=target,
                    orientation=orientation,
                )
                nearest, nearest_contributions = _nearest_neighbor_disagreement(
                    target_confirmation,
                    proxy=proxy,
                    target=target,
                    orientation=orientation,
                )
                information, information_contributions = _incremental_cross_entropy(
                    target_analysis,
                    target_confirmation,
                    proxy=proxy,
                    target=target,
                    bins=bins,
                )
                nn_low, nn_high = _cluster_ratio_bootstrap(
                    nearest_contributions,
                    namespace=f"{model}/{proxy}/{target}/nn",
                )
                reduction_low, reduction_high = _cluster_ratio_bootstrap(
                    information_contributions,
                    namespace=f"{model}/{proxy}/{target}/logloss",
                )
                information_rows.append(
                    {
                        "model": model,
                        "target": target,
                        "proxy": proxy,
                        "orientation": orientation,
                        **conditional_auc,
                        **nearest,
                        "nn_disagreement_low": nn_low,
                        "nn_disagreement_high": nn_high,
                        **information,
                        "cross_entropy_reduction_low": reduction_low,
                        "cross_entropy_reduction_high": reduction_high,
                    }
                )
                conditional_auc_without = _conditional_auroc(
                    target_confirmation,
                    proxy=proxy,
                    target=target,
                    orientation=orientation,
                    include_difficulty=False,
                )
                nearest_without, nearest_without_contributions = (
                    _nearest_neighbor_disagreement(
                        target_confirmation,
                        proxy=proxy,
                        target=target,
                        orientation=orientation,
                        include_difficulty=False,
                    )
                )
                information_without, information_without_contributions = (
                    _incremental_cross_entropy(
                        target_analysis,
                        target_confirmation,
                        proxy=proxy,
                        target=target,
                        bins=bins,
                        include_difficulty=False,
                    )
                )
                nn_without_low, nn_without_high = _cluster_ratio_bootstrap(
                    nearest_without_contributions,
                    namespace=(f"{model}/{proxy}/{target}/nn/no-difficulty"),
                )
                (
                    reduction_without_low,
                    reduction_without_high,
                ) = _cluster_ratio_bootstrap(
                    information_without_contributions,
                    namespace=(f"{model}/{proxy}/{target}/logloss/no-difficulty"),
                )
                deletion_information_rows.append(
                    {
                        "model": model,
                        "target": target,
                        "proxy": proxy,
                        "orientation": orientation,
                        "with_difficulty_conditional_auroc": (
                            conditional_auc["conditional_auroc"]
                        ),
                        "without_difficulty_conditional_auroc": (
                            conditional_auc_without["conditional_auroc"]
                        ),
                        "with_difficulty_comparable_pairs": (
                            conditional_auc["comparable_pairs"]
                        ),
                        "without_difficulty_comparable_pairs": (
                            conditional_auc_without["comparable_pairs"]
                        ),
                        "with_difficulty_effective_strata": (
                            conditional_auc["effective_strata"]
                        ),
                        "without_difficulty_effective_strata": (
                            conditional_auc_without["effective_strata"]
                        ),
                        "with_difficulty_nn_disagreement_rate": (
                            nearest["nn_disagreement_rate"]
                        ),
                        "with_difficulty_nn_disagreement_low": nn_low,
                        "with_difficulty_nn_disagreement_high": nn_high,
                        "without_difficulty_nn_disagreement_rate": (
                            nearest_without["nn_disagreement_rate"]
                        ),
                        "without_difficulty_nn_disagreement_low": (nn_without_low),
                        "without_difficulty_nn_disagreement_high": (nn_without_high),
                        "with_difficulty_nn_eligible": nearest["nn_eligible"],
                        "without_difficulty_nn_eligible": (
                            nearest_without["nn_eligible"]
                        ),
                        "with_difficulty_cross_entropy_reduction_bits": (
                            information["cross_entropy_reduction_bits"]
                        ),
                        "with_difficulty_cross_entropy_reduction_low": (reduction_low),
                        "with_difficulty_cross_entropy_reduction_high": (
                            reduction_high
                        ),
                        "without_difficulty_cross_entropy_reduction_bits": (
                            information_without["cross_entropy_reduction_bits"]
                        ),
                        "without_difficulty_cross_entropy_reduction_low": (
                            reduction_without_low
                        ),
                        "without_difficulty_cross_entropy_reduction_high": (
                            reduction_without_high
                        ),
                        "with_difficulty_augmented_fallback_rows": (
                            information["augmented_fallback_rows"]
                        ),
                        "without_difficulty_augmented_fallback_rows": (
                            information_without["augmented_fallback_rows"]
                        ),
                    }
                )
    bin_path = OUT / "collision_bin_sensitivity.csv"
    information_path = OUT / "collision_incremental_information.csv"
    deletion_collision_path = OUT / "difficulty_deletion_collision.csv"
    deletion_information_path = OUT / "difficulty_deletion_information.csv"
    write_csv(bin_path, bin_rows)
    write_csv(information_path, information_rows)
    write_csv(deletion_collision_path, deletion_collision_rows)
    write_csv(deletion_information_path, deletion_information_rows)
    bin_table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Proxy & 5 bins & 10 bins & 20 bins & 50 bins \\",
        r"\midrule",
    ]
    for model in RUNS:
        for proxy in PROXY_LABELS:
            selected = {
                int(row["requested_max_bins"]): row
                for row in bin_rows
                if row["model"] == model and row["proxy"] == proxy
            }
            bin_table.append(
                "{} & {} & {:.1f}\\% & {:.1f}\\% & {:.1f}\\% & "
                "{:.1f}\\% \\\\".format(
                    MODEL_SHORT[model],
                    PROXY_LABELS[proxy],
                    *[
                        100.0 * float(selected[count]["collision_rate"])
                        for count in (5, 10, 20, 50)
                    ],
                )
            )
        bin_table.append(r"\addlinespace")
    bin_table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Matched-cell collision sensitivity to proxy "
                r"resolution. Persistence and agreement use exact-value "
                r"matching, so their columns are intentionally identical. "
                r"For continuous proxies, bins are learned only on the "
                r"analysis split and preserve ties. Collision is retained as "
                r"a descriptive coverage diagnostic, not evidence of complete "
                r"local proxy failure.}"
            ),
            r"\label{tab:collision-bin-sensitivity}",
            r"\end{table*}",
            "",
        ]
    )
    bin_table_path = OUT / "collision_bin_sensitivity_table.tex"
    atomic_text(bin_table_path, "\n".join(bin_table))
    information_table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        (
            r"Model & Target & Proxy & Conditional AUROC & NN disagreement & "
            r"$\Delta$ test log-loss, bits \\"
        ),
        r"\midrule",
    ]
    for model in RUNS:
        for target in ("beneficial", "harmful"):
            for proxy in PROXY_LABELS:
                row = next(
                    row
                    for row in information_rows
                    if row["model"] == model
                    and row["target"] == target
                    and row["proxy"] == proxy
                )
                information_table.append(
                    "{} & {} & {} & {:.3f} & {:.1f}\\% "
                    "[{:.1f}, {:.1f}] & {:+.4f} "
                    "[{:+.4f}, {:+.4f}] \\\\".format(
                        MODEL_SHORT[model],
                        target.capitalize(),
                        PROXY_LABELS[proxy],
                        float(row["conditional_auroc"]),
                        100.0 * float(row["nn_disagreement_rate"]),
                        100.0 * float(row["nn_disagreement_low"]),
                        100.0 * float(row["nn_disagreement_high"]),
                        float(row["cross_entropy_reduction_bits"]),
                        float(row["cross_entropy_reduction_low"]),
                        float(row["cross_entropy_reduction_high"]),
                    )
                )
            information_table.append(r"\addlinespace")
    information_table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Local and incremental proxy diagnostics on the "
                r"confirmation split. Conditional AUROC pools comparable "
                r"positive--negative pairs only within dataset, difficulty, "
                r"depth, and current-correctness strata. Nearest neighbors "
                r"must come from another problem in the same stratum; ties "
                r"average over all equally close candidates. Positive "
                r"$\Delta$ test log-loss means that a 10-bin proxy model "
                r"improves over the covariate-only Jeffreys model; bins and "
                r"probabilities are fitted on analysis data. Intervals "
                r"resample problems 2,000 times.}"
            ),
            r"\label{tab:collision-incremental}",
            r"\end{table*}",
            "",
        ]
    )
    information_table_path = OUT / "collision_incremental_table.tex"
    atomic_text(information_table_path, "\n".join(information_table))

    deletion_collision_table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"Model & Proxy & \multicolumn{2}{c}{Collision rate} & "
            r"\multicolumn{2}{c}{Beneficial $\Delta M_s$} & "
            r"\multicolumn{2}{c}{Harmful $\Delta M_s$} \\"
        ),
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        (r" & & $+D$ & $-D$ & $+D$ & $-D$ & $+D$ & $-D$ \\"),
        r"\midrule",
    ]
    for model in RUNS:
        for proxy in PROXY_LABELS:
            row = next(
                item
                for item in deletion_collision_rows
                if item["model"] == model and item["proxy"] == proxy
            )
            deletion_collision_table.append(
                "{} & {} & {:.1f}\\% & {:.1f}\\% & {:+.3f} & "
                "{:+.3f} & {:+.3f} & {:+.3f} \\\\".format(
                    MODEL_SHORT[model],
                    PROXY_LABELS[proxy],
                    100.0 * float(row["with_difficulty_collision_rate"]),
                    100.0 * float(row["without_difficulty_collision_rate"]),
                    float(row["with_difficulty_beneficial_gap"]),
                    float(row["without_difficulty_beneficial_gap"]),
                    float(row["with_difficulty_harmful_gap"]),
                    float(row["without_difficulty_harmful_gap"]),
                )
            )
        deletion_collision_table.append(r"\addlinespace")
    deletion_collision_table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Deletion sensitivity of the matched-collision "
                r"diagnostic at ten proxy bins. $+D$ uses the pooled "
                r"label-aware difficulty stratum; $-D$ removes difficulty "
                r"while retaining dataset, depth, current correctness, the "
                r"same analysis-fitted proxy bins, and exact-value matching "
                r"for persistence and agreement. Coarser $-D$ cells can "
                r"mechanically increase collision coverage, so the signed "
                r"within-cell gaps and proxy ordering are reported alongside "
                r"the rate.}"
            ),
            r"\label{tab:difficulty-deletion-collision}",
            r"\end{table*}",
            "",
        ]
    )
    deletion_collision_table_path = OUT / "difficulty_deletion_collision_table.tex"
    atomic_text(
        deletion_collision_table_path,
        "\n".join(deletion_collision_table),
    )

    deletion_information_table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        (
            r"Model & Target & Proxy & \multicolumn{2}{c}{Conditional "
            r"AUROC} & \multicolumn{2}{c}{NN disagreement} & "
            r"\multicolumn{2}{c}{$\Delta$ test log-loss, bits} \\"
        ),
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r" & & & $+D$ & $-D$ & $+D$ & $-D$ & $+D$ & $-D$ \\",
        r"\midrule",
    ]
    for model in RUNS:
        for target in ("beneficial", "harmful"):
            for proxy in PROXY_LABELS:
                row = next(
                    item
                    for item in deletion_information_rows
                    if item["model"] == model
                    and item["target"] == target
                    and item["proxy"] == proxy
                )
                deletion_information_table.append(
                    "{} & {} & {} & {:.3f} & {:.3f} & {:.1f}\\% & "
                    "{:.1f}\\% & {:+.4f} & {:+.4f} \\\\".format(
                        MODEL_SHORT[model],
                        target.capitalize(),
                        PROXY_LABELS[proxy],
                        float(row["with_difficulty_conditional_auroc"]),
                        float(row["without_difficulty_conditional_auroc"]),
                        100.0 * float(row["with_difficulty_nn_disagreement_rate"]),
                        100.0 * float(row["without_difficulty_nn_disagreement_rate"]),
                        float(row["with_difficulty_" "cross_entropy_reduction_bits"]),
                        float(
                            row["without_difficulty_" "cross_entropy_reduction_bits"]
                        ),
                    )
                )
            deletion_information_table.append(r"\addlinespace")
    deletion_information_table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Deletion sensitivity of local proxy information. "
                r"$+D$ conditions on dataset, pooled difficulty, depth, and "
                r"current correctness; $-D$ deletes difficulty everywhere "
                r"from conditional AUROC strata, nearest-neighbor matching, "
                r"and the analysis-fitted Jeffreys baseline and augmented "
                r"log-loss models. Proxy orientation and bins are held fixed "
                r"from the analysis split. The sealed CSV includes "
                r"problem-bootstrap intervals, eligible counts, comparable "
                r"pairs, and fallback counts for both specifications.}"
            ),
            r"\label{tab:difficulty-deletion-information}",
            r"\end{table*}",
            "",
        ]
    )
    deletion_information_table_path = OUT / "difficulty_deletion_information_table.tex"
    atomic_text(
        deletion_information_table_path,
        "\n".join(deletion_information_table),
    )

    deletion_manifest_path = OUT / "difficulty_deletion_manifest.json"
    deletion_artifacts = (
        deletion_collision_path,
        deletion_information_path,
        deletion_collision_table_path,
        deletion_information_table_path,
    )
    deletion_sealed = seal_record(
        {
            "schema_version": DIFFICULTY_DELETION_NAMESPACE,
            "analysis_status": ("supplementary deletion sensitivity, read-only"),
            "comparison": {
                "with_difficulty": [
                    "dataset_id",
                    "difficulty",
                    "depth",
                    "stop_correct",
                ],
                "without_difficulty": [
                    "dataset_id",
                    "depth",
                    "stop_correct",
                ],
            },
            "held_fixed": [
                "confirmation problem universe",
                "analysis-fitted proxy orientation",
                "analysis-fitted proxy bins",
                "target feasibility condition",
                "same-problem nearest-neighbor exclusion",
            ],
            "collision_bins": 10,
            "bootstrap_unit": "problem_id",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for model, payload in validated.items()
            },
            "artifacts": [path.name for path in deletion_artifacts],
        }
    )
    atomic_text(
        deletion_manifest_path,
        json.dumps(deletion_sealed, indent=2, sort_keys=True) + "\n",
    )

    manifest_path = OUT / "collision_robustness_manifest.json"
    sealed = seal_record(
        {
            "schema_version": COLLISION_ROBUSTNESS_NAMESPACE,
            "analysis_status": "supplementary, label-aware, read-only",
            "bin_sensitivity": [5, 10, 20, 50],
            "discrete_exact_matching": [
                "persistence",
                "modal_agreement",
            ],
            "conditional_covariates": [
                "dataset_id",
                "difficulty",
                "depth",
                "stop_correct",
            ],
            "nearest_neighbor_excludes_same_problem": True,
            "incremental_model_fit_split": "analysis",
            "incremental_model_test_split": "confirmation",
            "bootstrap_unit": "problem_id",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for model, payload in validated.items()
            },
            "artifacts": [
                path.name
                for path in (
                    bin_path,
                    information_path,
                    bin_table_path,
                    information_table_path,
                )
            ],
        }
    )
    atomic_text(
        manifest_path,
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
    )
    return [
        bin_path,
        information_path,
        bin_table_path,
        information_table_path,
        manifest_path,
        *deletion_artifacts,
        deletion_manifest_path,
    ]


def render_persistence_sensitivity(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    thresholds = (0.75, 0.50, 1.0 / 3.0)
    seed_offset = 700
    for model, all_rows in values_by_model.items():
        confirmation = [
            row
            for row in all_rows
            if row["split"] == "confirmation"
            and row["stop_correct"] == 1
            and row["persistence"] is not None
        ]
        for threshold in thresholds:
            interval = cluster_contrast_interval(
                confirmation,
                low_threshold=threshold,
                seed_offset=seed_offset,
            )
            if math.isclose(threshold, 0.75):
                official = validated[model]["evaluation"]["headline"][
                    "contrast_high_minus_low"
                ]
                interval.update(
                    {
                        "estimate": official["estimate"],
                        "low": official["low"],
                        "high": official["high"],
                        "replicates": official["replicates"],
                    }
                )
            seed_offset += 1
            rows.append(
                {
                    "model": model,
                    "high_definition": "persistence = 1",
                    "low_definition": f"persistence <= {threshold:.3f}",
                    "low_threshold": threshold,
                    **interval,
                }
            )
    csv_path = OUT / "persistence_threshold_sensitivity.csv"
    write_csv(csv_path, rows)
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Model & Low group & $n_{\rm high}$ & $n_{\rm low}$ & "
            r"High--low $\Delta M_s$ (95\% CI) \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        label = (
            r"$p\leq0.75$ (tie-resolved)"
            if math.isclose(row["low_threshold"], 0.75)
            else f"$p\\leq{row['low_threshold']:.2f}$"
        )
        tex.append(
            "{} & {} & {:,} & {:,} & {:+.3f} [{:+.3f}, {:+.3f}] \\\\".format(
                MODEL_SHORT[row["model"]],
                label,
                row["high_n"],
                row["low_n"],
                row["estimate"],
                row["low"],
                row["high"],
            )
        )
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Persistence-threshold sensitivity among currently "
                r"correct confirmation checkpoints. Both analysis quartiles "
                r"equal 1.0 because persistence is discrete; the implemented "
                r"tie order therefore compares $p=1$ with $p<1$. Tighter low "
                r"groups leave the conclusion unchanged.}"
            ),
            r"\label{tab:persistence-threshold}",
            r"\end{table}",
            "",
        ]
    )
    tex_path = OUT / "persistence_sensitivity_table.tex"
    atomic_text(tex_path, "\n".join(tex))
    return [csv_path, tex_path]


def subset_event_probability(
    continuation_correct: tuple[int, ...],
    *,
    stop_correct: int,
    k: int,
) -> float:
    event_count = 0
    subsets = list(combinations(range(4), k))
    for subset in subsets:
        selected = [continuation_correct[index] for index in subset]
        event = any(selected) if stop_correct == 0 else not all(selected)
        event_count += int(event)
    return event_count / len(subsets)


def render_k_sensitivity(
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for model, all_rows in values_by_model.items():
        confirmation = [row for row in all_rows if row["split"] == "confirmation"]
        for target, stop_correct in (("beneficial", 0), ("harmful", 1)):
            selected = [
                row for row in confirmation if row["stop_correct"] == stop_correct
            ]
            for k in range(1, 5):
                probabilities = [
                    subset_event_probability(
                        row["continuation_correct"],
                        stop_correct=stop_correct,
                        k=k,
                    )
                    for row in selected
                ]
                full_labels = [
                    int(
                        any(row["continuation_correct"])
                        if stop_correct == 0
                        else not all(row["continuation_correct"])
                    )
                    for row in selected
                ]
                agreement = [
                    probability if full else 1.0
                    for probability, full in zip(
                        probabilities, full_labels, strict=True
                    )
                ]
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "k": k,
                        "checkpoints": len(selected),
                        "expected_event_rate": float(np.mean(probabilities)),
                        "expected_agreement_with_k4": float(np.mean(agreement)),
                    }
                )
    csv_path = OUT / "branch_k_sensitivity.csv"
    write_csv(csv_path, rows)
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (r"Model & Target & $K$ & Expected event rate & " r"Agreement with $K=4$ \\"),
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            "{} & {} & {} & {:.1f}\\% & {:.1f}\\% \\\\".format(
                MODEL_SHORT[row["model"]],
                row["target"].capitalize(),
                row["k"],
                100 * row["expected_event_rate"],
                100 * row["expected_agreement_with_k4"],
            )
        )
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Branch-count sensitivity on confirmation problems. "
                r"For $K<4$, values average over every size-$K$ subset of the "
                r"four observed branches, avoiding dependence on the "
                r"distinguished on-spine branch order.}"
            ),
            r"\label{tab:k-sensitivity}",
            r"\end{table}",
            "",
        ]
    )
    tex_path = OUT / "branch_k_sensitivity_table.tex"
    atomic_text(tex_path, "\n".join(tex))
    return [csv_path, tex_path]


def render_branch_estimands(
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    """Separate a single random transition from four-draw opportunity detection."""

    rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    seed_offset = 2200
    for model, all_rows in values_by_model.items():
        confirmation = [row for row in all_rows if row["split"] == "confirmation"]
        for target, stop_correct in (("beneficial", 0), ("harmful", 1)):
            selected = [
                row for row in confirmation if int(row["stop_correct"]) == stop_correct
            ]
            branch_rows: list[dict[str, Any]] = []
            checkpoint_rows: list[dict[str, Any]] = []
            for row in selected:
                outcomes = tuple(int(value) for value in row["continuation_correct"])
                for branch_index, correct in enumerate(outcomes, start=1):
                    branch_rows.append(
                        {
                            "problem_id": row["problem_id"],
                            "event": (
                                correct if target == "beneficial" else 1 - correct
                            ),
                            "branch_index": branch_index,
                        }
                    )
                event_probability = (
                    sum(outcomes) / len(outcomes)
                    if target == "beneficial"
                    else 1.0 - sum(outcomes) / len(outcomes)
                )
                checkpoint_rows.append(
                    {
                        "problem_id": row["problem_id"],
                        "detected": int(event_probability > 0),
                        "event_probability": event_probability,
                    }
                )
            single_draw = cluster_ratio_interval(
                branch_rows,
                lambda row: float(row["event"]),
                lambda row: 1.0,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            four_draw_detection = cluster_ratio_interval(
                checkpoint_rows,
                lambda row: float(row["detected"]),
                lambda row: 1.0,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            mean_probability = cluster_ratio_interval(
                checkpoint_rows,
                lambda row: float(row["event_probability"]),
                lambda row: 1.0,
                seed_offset=seed_offset,
            )
            seed_offset += 1
            if not math.isclose(
                float(single_draw["estimate"]),
                float(mean_probability["estimate"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("single-draw and mean probability disagree")
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "checkpoints": len(checkpoint_rows),
                    "branch_draws": len(branch_rows),
                    "single_draw_rate": single_draw["estimate"],
                    "single_draw_low": single_draw["low"],
                    "single_draw_high": single_draw["high"],
                    "four_draw_detection_rate": four_draw_detection["estimate"],
                    "four_draw_detection_low": four_draw_detection["low"],
                    "four_draw_detection_high": four_draw_detection["high"],
                    "detection_inflation_percentage_points": (
                        100.0
                        * (
                            float(four_draw_detection["estimate"])
                            - float(single_draw["estimate"])
                        )
                    ),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
            counts: dict[float, int] = defaultdict(int)
            for row in checkpoint_rows:
                counts[float(row["event_probability"])] += 1
            for probability in sorted(counts):
                distribution_rows.append(
                    {
                        "model": model,
                        "target": target,
                        "branch_event_probability": probability,
                        "checkpoints": counts[probability],
                        "fraction": counts[probability] / len(checkpoint_rows),
                    }
                )
    csv_path = OUT / "branch_estimand_summary.csv"
    distribution_path = OUT / "branch_probability_distribution.csv"
    write_csv(csv_path, rows)
    write_csv(distribution_path, distribution_rows)
    table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        (
            r"Model & Event & Single random branch & "
            r"At least one of four branches \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        table.append(
            "{} & {} & \\keyresult{{{:.1f}\\% [{:.1f}, {:.1f}]}} & "
            "{:.1f}\\% [{:.1f}, {:.1f}] \\\\".format(
                MODEL_SHORT[row["model"]],
                "Recovery" if row["target"] == "beneficial" else "Damage",
                100.0 * float(row["single_draw_rate"]),
                100.0 * float(row["single_draw_low"]),
                100.0 * float(row["single_draw_high"]),
                100.0 * float(row["four_draw_detection_rate"]),
                100.0 * float(row["four_draw_detection_low"]),
                100.0 * float(row["four_draw_detection_high"]),
            )
        )
    table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Two distinct continuation estimands on confirmation "
                r"problems. The single-draw rate averages the recovery or "
                r"damage indicator over all four branch positions. The "
                r"four-draw rate is checkpoint-level opportunity detection: "
                r"whether at least one of four sampled continuations exhibits "
                r"the event. Pale green marks the single-continuation rates "
                r"used for claims about one next reasoning block. Intervals "
                r"resample complete problems 2,000 times.}"
            ),
            r"\label{tab:branch-estimands}",
            r"\end{table}",
            "",
        ]
    )
    table_path = OUT / "branch_estimand_table.tex"
    atomic_text(table_path, "\n".join(table))
    return [csv_path, distribution_path, table_path]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return tie-aware one-indexed average ranks."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("average ranks require a finite nonempty vector")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def render_value_ratio_sensitivity(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    """Compare expected-change/expected-cost with mean branch utility."""

    rows: list[dict[str, Any]] = []
    for model, all_rows in values_by_model.items():
        for split in ("analysis", "confirmation"):
            selected = [row for row in all_rows if row["split"] == split]
            ratio_of_means: list[float] = []
            mean_of_ratios: list[float] = []
            stop_labels: list[int] = []
            for row in selected:
                stop = int(row["stop_correct"])
                outcomes = tuple(int(value) for value in row["continuation_correct"])
                costs = tuple(int(value) for value in row["continuation_costs"])
                primary = token_normalized_value(stop, outcomes, costs)
                sealed = float(row["marginal_value"])
                if not math.isclose(
                    primary,
                    sealed,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"{model}/{row['checkpoint_id']}: sealed value "
                        "does not equal the ratio-of-means estimand"
                    )
                alternative = float(
                    np.mean(
                        [
                            1000.0 * (outcome - stop) / cost
                            for outcome, cost in zip(
                                outcomes,
                                costs,
                                strict=True,
                            )
                        ]
                    )
                )
                ratio_of_means.append(primary)
                mean_of_ratios.append(alternative)
                stop_labels.append(stop)
            primary_array = np.asarray(ratio_of_means, dtype=float)
            alternative_array = np.asarray(mean_of_ratios, dtype=float)
            stop_array = np.asarray(stop_labels, dtype=np.int8)
            primary_ranks = _average_ranks(primary_array)
            alternative_ranks = _average_ranks(alternative_array)
            difference = alternative_array - primary_array
            primary_sign = np.sign(primary_array)
            alternative_sign = np.sign(alternative_array)
            wrong = stop_array == 0
            correct = stop_array == 1
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "checkpoints": len(selected),
                    "ratio_of_means_mean": float(primary_array.mean()),
                    "mean_of_ratios_mean": float(alternative_array.mean()),
                    "pearson_correlation": float(
                        np.corrcoef(
                            primary_array,
                            alternative_array,
                        )[0, 1]
                    ),
                    "spearman_correlation": float(
                        np.corrcoef(
                            primary_ranks,
                            alternative_ranks,
                        )[0, 1]
                    ),
                    "sign_agreement": float(np.mean(primary_sign == alternative_sign)),
                    "beneficial_label_agreement": float(
                        np.mean(
                            (primary_array[wrong] > 0) == (alternative_array[wrong] > 0)
                        )
                    ),
                    "harmful_label_agreement": float(
                        np.mean(
                            (primary_array[correct] < 0)
                            == (alternative_array[correct] < 0)
                        )
                    ),
                    "mean_absolute_difference": float(np.mean(np.abs(difference))),
                    "max_absolute_difference": float(np.max(np.abs(difference))),
                }
            )

    csv_path = OUT / "value_ratio_sensitivity.csv"
    write_csv(csv_path, rows)
    table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (r"Model & Pearson & Spearman & Sign agree. & " r"Mean $|\Delta|$ \\"),
        r"\midrule",
    ]
    for row in rows:
        if row["split"] != "confirmation":
            continue
        table.append(
            "{} & {:.5f} & {:.5f} & {:.1f}\\% & {:.4f} \\\\".format(
                MODEL_SHORT[row["model"]],
                row["pearson_correlation"],
                row["spearman_correlation"],
                100.0 * row["sign_agreement"],
                row["mean_absolute_difference"],
            )
        )
    table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Sensitivity of the signed value estimand on "
                r"confirmation problems. The primary ratio of expected "
                r"correctness change to expected logical cost is compared "
                r"with the mean of four branch-specific change-per-cost "
                r"ratios. Both beneficial and harmful event labels agree "
                r"exactly in both models; correlations quantify the small "
                r"remaining magnitude and ranking differences.}"
            ),
            r"\label{tab:value-ratio-sensitivity}",
            r"\end{table}",
            "",
        ]
    )
    table_path = OUT / "value_ratio_sensitivity_table.tex"
    atomic_text(table_path, "\n".join(table))
    manifest_path = OUT / "value_ratio_sensitivity_manifest.json"
    sealed = seal_record(
        {
            "schema_version": "flcva-value-ratio-sensitivity-v1",
            "analysis_status": "supplementary, label-aware, read-only",
            "selection_namespace": VALUE_RATIO_SENSITIVITY_NAMESPACE,
            "primary_estimand": (
                "expected correctness change divided by expected logical "
                "tokens under a uniformly sampled continuation branch"
            ),
            "alternative_estimand": (
                "mean of branch-specific correctness-change-per-logical-" "token ratios"
            ),
            "source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for model, payload in validated.items()
            },
            "artifacts": [csv_path.name, table_path.name],
        }
    )
    atomic_text(
        manifest_path,
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
    )
    return [csv_path, table_path, manifest_path]


def _off_spine_permutation(
    states: list[dict[str, Any]],
    off_spine_values: dict[str, float],
    config: RunConfig,
    *,
    round_robin_auc: float,
    start_budget: int,
    endpoint: int,
    include_difficulty: bool = True,
) -> dict[str, Any]:
    eligible = [
        state for state in states if str(state["checkpoint_id"]) in off_spine_values
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for state in eligible:
        key: list[Any] = [
            state["model_id"],
            state["dataset_id"],
        ]
        if include_difficulty:
            key.append(state["difficulty"])
        key.extend(
            [
                int(state["depth"]),
                int(state["stop_correct"]),
            ]
        )
        groups[tuple(key)].append(state)
    effective_groups = sum(
        len(group) >= 2
        and len({off_spine_values[str(state["checkpoint_id"])] for state in group}) >= 2
        for group in groups.values()
    )
    observed_curve = _replay_one(
        states,
        "oracle",
        checkpoint_interval=config.protocol.checkpoint_interval,
        fixed_allotment_tokens=config.execution.fixed_allotment_tokens,
        overridden_values=off_spine_values,
    )
    observed_difference = (
        step_auc(
            observed_curve,
            endpoint,
            start_budget=start_budget,
        )
        - round_robin_auc
    )
    differences: list[float] = []
    for permutation_index in range(config.execution.permutation_replicates):
        assigned: dict[str, float] = {}
        for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
            ordered = sorted(group, key=lambda row: row["checkpoint_id"])
            values = [off_spine_values[str(row["checkpoint_id"])] for row in ordered]
            if len(ordered) >= 2 and len(set(values)) >= 2:
                rng = random.Random(
                    stable_seed(
                        config.global_seed,
                        (
                            "off-spine-permutation-with-difficulty-v1"
                            if include_difficulty
                            else "off-spine-permutation-no-difficulty-v1"
                        ),
                        permutation_index,
                        *key,
                    )
                )
                rng.shuffle(values)
            assigned.update(
                {
                    str(row["checkpoint_id"]): value
                    for row, value in zip(ordered, values, strict=True)
                }
            )
        curve = _replay_one(
            states,
            "oracle",
            checkpoint_interval=config.protocol.checkpoint_interval,
            fixed_allotment_tokens=config.execution.fixed_allotment_tokens,
            overridden_values=assigned,
        )
        differences.append(
            step_auc(curve, endpoint, start_budget=start_budget) - round_robin_auc
        )
    exceedances = sum(difference >= observed_difference for difference in differences)
    array = np.asarray(differences, dtype=float)
    return {
        "schema_version": "flcva-off-spine-permutation-v1",
        "replicates": len(differences),
        "effective_groups": effective_groups,
        "observed_difference": observed_difference,
        "exceedances": exceedances,
        "p_value_corrected": (
            (exceedances + 1) / (len(differences) + 1) if differences else None
        ),
        "mean_difference": float(array.mean()) if len(array) else None,
        "median_difference": (float(np.median(array)) if len(array) else None),
        "low": (float(np.quantile(array, 0.025)) if len(array) else None),
        "high": (float(np.quantile(array, 0.975)) if len(array) else None),
        "difficulty_included": include_difficulty,
        "strata": (
            [
                "model_id",
                "dataset_id",
                "difficulty",
                "depth",
                "stop_correct",
            ]
            if include_difficulty
            else [
                "model_id",
                "dataset_id",
                "depth",
                "stop_correct",
            ]
        ),
    }


def render_oracle_decomposition(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for model, payload in validated.items():
        run_dir = payload["run_dir"]
        config = RunConfig.model_validate(
            read_json(run_dir / "manifest.json")["config"]
        )
        states, identity = load_confirmation_frontiers(
            run_dir,
            payload["evaluation"],
            values_by_model[model],
        )
        scores = oracle_score_maps(values_by_model[model])
        curves = {
            "round_robin": _replay_one(
                states,
                "round_robin",
                checkpoint_interval=config.protocol.checkpoint_interval,
                fixed_allotment_tokens=(config.execution.fixed_allotment_tokens),
            )
        }
        for oracle_name, values in scores.items():
            curves[oracle_name] = _replay_one(
                states,
                "oracle",
                checkpoint_interval=config.protocol.checkpoint_interval,
                fixed_allotment_tokens=(config.execution.fixed_allotment_tokens),
                overridden_values=values,
            )
        start_budget = curves["round_robin"][0].budget
        endpoint = min(curve[-1].budget for curve in curves.values())
        if endpoint != int(payload["replay"]["common_endpoint"]):
            raise ValueError(f"{model}: oracle decomposition changed the replay domain")
        aulc = {
            name: step_auc(
                curve,
                endpoint,
                start_budget=start_budget,
            )
            for name, curve in curves.items()
        }
        if not math.isclose(
            aulc["mixed"] - aulc["round_robin"],
            float(payload["replay"]["oracle_difference"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{model}: mixed oracle does not reproduce sealed replay")
        round_robin_auc = aulc["round_robin"]
        permutation = _off_spine_permutation(
            states,
            scores["off_spine"],
            config,
            round_robin_auc=round_robin_auc,
            start_budget=start_budget,
            endpoint=endpoint,
        )
        permutation_without_difficulty = _off_spine_permutation(
            states,
            scores["off_spine"],
            config,
            round_robin_auc=round_robin_auc,
            start_budget=start_budget,
            endpoint=endpoint,
            include_difficulty=False,
        )
        for ordering in (
            "round_robin",
            "mixed",
            "hindsight_on",
            "off_spine",
        ):
            curve = curves[ordering]
            area = aulc[ordering]
            final = max(
                (point for point in curve if point.budget <= endpoint),
                key=lambda point: point.budget,
            )
            rows.append(
                {
                    "model": model,
                    "ordering": ordering,
                    "uses_actual_successor_outcome": (
                        ordering in {"mixed", "hindsight_on"}
                    ),
                    "aulc": area,
                    "mean_accuracy": area / (endpoint - start_budget),
                    "aulc_difference_vs_round_robin": (area - round_robin_auc),
                    "relative_aulc_gain_percent": (
                        100.0 * (area - round_robin_auc) / round_robin_auc
                    ),
                    "final_accuracy": final.accuracy,
                    "start_budget": start_budget,
                    "common_endpoint": endpoint,
                    "off_spine_permutation_replicates": (
                        permutation["replicates"] if ordering == "off_spine" else None
                    ),
                    "off_spine_permutation_exceedances": (
                        permutation["exceedances"] if ordering == "off_spine" else None
                    ),
                    "off_spine_permutation_p_value_corrected": (
                        permutation["p_value_corrected"]
                        if ordering == "off_spine"
                        else None
                    ),
                    ("off_spine_permutation_" "p_value_without_difficulty"): (
                        permutation_without_difficulty["p_value_corrected"]
                        if ordering == "off_spine"
                        else None
                    ),
                }
            )
            for point in curve:
                if point.budget <= endpoint:
                    point_rows.append(
                        {
                            "model": model,
                            "ordering": ordering,
                            "budget": point.budget,
                            "normalized_budget": (
                                (point.budget - start_budget)
                                / (endpoint - start_budget)
                            ),
                            "accuracy": point.accuracy,
                            "active_frontiers": point.active_frontiers,
                            "terminal_frontiers": point.terminal_frontiers,
                        }
                    )
        summaries[model] = {
            "run_id": payload["run"]["run_id"],
            "result_merkle_root": payload["audit"]["result_merkle_root"],
            "branch1_successor_identity": identity,
            "aulc_start_budget": start_budget,
            "common_endpoint": endpoint,
            "aulc": aulc,
            "relative_aulc_gain_percent": {
                name: (100.0 * (area - round_robin_auc) / round_robin_auc)
                for name, area in aulc.items()
            },
            "off_spine_permutation": permutation,
            "off_spine_permutation_without_difficulty": (
                permutation_without_difficulty
            ),
        }

    csv_path = OUT / "oracle_decomposition.csv"
    points_path = OUT / "oracle_decomposition_points.csv"
    write_csv(csv_path, rows)
    write_csv(points_path, point_rows)

    labels = {
        "round_robin": "Round-robin",
        "mixed": "Mixed 4-branch",
        "hindsight_on": "On-spine one-step hindsight",
        "off_spine": "Off-spine independent",
    }
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llcrrr}",
        r"\toprule",
        (
            r"Model & Ordering score & Sees successor? & Mean accuracy & "
            r"AULC gain vs.\ RR & Off-spine permutation $p$ ($+D/-D$) \\"
        ),
        r"\midrule",
    ]
    for model in RUNS:
        for row in [row for row in rows if row["model"] == model]:
            p_value = row["off_spine_permutation_p_value_corrected"]
            p_value_without = row["off_spine_permutation_p_value_without_difficulty"]
            tex.append(
                "{} & {} & {} & {:.3f} & {:+.2f}\\% & {} \\\\".format(
                    MODEL_SHORT[model],
                    labels[row["ordering"]],
                    "yes" if row["uses_actual_successor_outcome"] else "no",
                    row["mean_accuracy"],
                    row["relative_aulc_gain_percent"],
                    (
                        f"{p_value:.3f}/{p_value_without:.3f}"
                        if p_value is not None
                        else "--"
                    ),
                )
            )
        tex.append(r"\addlinespace")
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Spine-level myopic score decomposition in the "
                r"pooled diagnostic that admits all four spines of every "
                r"problem simultaneously. Mixed uses all four branches and "
                r"therefore includes the realized on-spine successor; "
                r"one-step hindsight uses only that successor. These greedy "
                r"scores are not global dynamic-programming upper bounds. "
                r"Off-spine "
                r"uses branches 2--4 to order actions evaluated on branch 1. "
                r"The reported permutation test is performed only on the "
                r"off-spine score within the matching strata; $+D$ includes "
                r"pooled difficulty and $-D$ deletes it. Each test evaluates "
                r"checkpoint correspondence, not whether gain versus "
                r"round-robin is positive.}"
            ),
            r"\label{tab:oracle-decomposition}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = OUT / "oracle_decomposition_table.tex"
    atomic_text(tex_path, "\n".join(tex))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5), sharey=True)
    colors = {
        "round_robin": "#444444",
        "mixed": "#9467bd",
        "hindsight_on": "#d62728",
        "off_spine": "#1f77b4",
    }
    for axis, model in zip(axes, RUNS, strict=True):
        for ordering in (
            "round_robin",
            "mixed",
            "hindsight_on",
            "off_spine",
        ):
            selected = [
                row
                for row in point_rows
                if row["model"] == model and row["ordering"] == ordering
            ]
            axis.plot(
                [row["normalized_budget"] for row in selected],
                [row["accuracy"] for row in selected],
                label=labels[ordering],
                color=colors[ordering],
                linewidth=1.3,
                drawstyle="steps-post",
            )
        axis.set_title(MODEL_SHORT[model])
        axis.set_xlabel("Normalized full-ledger budget")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Frontier exact-match accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    figure_path = OUT / "oracle_decomposition.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = OUT / "oracle_decomposition_summary.json"
    summary = seal_record(
        {
            "schema_version": "flcva-oracle-decomposition-v1",
            "analysis_status": "supplementary, label-aware, read-only",
            "score_definitions": {
                "mixed": ("1000 * (mean(C1..C4) - S) / mean(L1..L4)"),
                "hindsight_on": "1000 * (C1 - S) / L1",
                "off_spine": ("1000 * (mean(C2..C4) - S) / mean(L2..L4)"),
            },
            "replay_outcome": (
                "the authenticated on-spine successor, identical to branch 1"
            ),
            "models": summaries,
        }
    )
    atomic_text(
        summary_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )

    row_by_key = {(row["model"], row["ordering"]): row for row in rows}
    macro_lines = [
        "% Auto-generated from the sealed oracle decomposition.",
    ]
    for model, prefix in (
        ("Qwen3-8B", "Qwen"),
        ("DeepSeek-R1-Distill-Qwen-7B", "DeepSeek"),
    ):
        for ordering, suffix in (
            ("mixed", "MixedOracleGain"),
            ("hindsight_on", "OnOracleGain"),
            ("off_spine", "OffOracleGain"),
        ):
            gain = row_by_key[(model, ordering)]["relative_aulc_gain_percent"]
            macro_lines.append(
                f"\\newcommand{{\\{prefix}{suffix}}}" f"{{{gain:+.2f}\\%}}"
            )
        p_value = row_by_key[(model, "off_spine")][
            "off_spine_permutation_p_value_corrected"
        ]
        macro_lines.append(
            f"\\newcommand{{\\{prefix}OffOraclePermutationP}}"
            f"{{{float(p_value):.3f}}}"
        )
        p_value_without = row_by_key[(model, "off_spine")][
            "off_spine_permutation_p_value_without_difficulty"
        ]
        macro_lines.append(
            f"\\newcommand{{\\{prefix}OffOraclePermutationPNoDifficulty}}"
            f"{{{float(p_value_without):.3f}}}"
        )
    macros_path = OUT / "oracle_decomposition_macros.tex"
    atomic_text(macros_path, "\n".join(macro_lines) + "\n")
    return [
        csv_path,
        points_path,
        tex_path,
        figure_path,
        summary_path,
        macros_path,
    ]


def _fmt_metric_interval(interval: dict[str, Any]) -> str:
    return (
        f"{float(interval['estimate']):.3f} "
        f"[{float(interval['low']):.3f}, "
        f"{float(interval['high']):.3f}]"
    )


def render_leave_one_branch_out(
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    output_rows: list[dict[str, Any]] = []
    seed_index = 0
    for model, values in values_by_model.items():
        loo_rows = leave_one_branch_out_rows(values)
        for target in ("beneficial", "harmful"):
            target_rows = [row for row in loo_rows if row["target"] == target]
            scopes: list[tuple[str, list[dict[str, Any]]]] = [
                ("pooled_1_to_4", target_rows)
            ]
            scopes.extend(
                (
                    f"held_out_{branch_index}",
                    [
                        row
                        for row in target_rows
                        if row["held_out_branch"] == branch_index
                    ],
                )
                for branch_index in range(1, 5)
            )
            for scope, selected in scopes:
                intervals = discrete_problem_bootstrap(
                    selected,
                    seed=stable_seed(
                        BOOTSTRAP_SEED,
                        "leave-one-branch-out-v1",
                        model,
                        target,
                        scope,
                        seed_index,
                    ),
                )
                seed_index += 1
                output_rows.append(
                    {
                        "model": model,
                        "target": target,
                        "scope": scope,
                        "held_out_role": (
                            "pooled"
                            if scope == "pooled_1_to_4"
                            else ("on_spine" if scope == "held_out_1" else "off_spine")
                        ),
                        "n": len(selected),
                        "event_rate": intervals["event_rate"]["estimate"],
                        "event_rate_low": intervals["event_rate"]["low"],
                        "event_rate_high": intervals["event_rate"]["high"],
                        "auroc": intervals["auroc"]["estimate"],
                        "auroc_low": intervals["auroc"]["low"],
                        "auroc_high": intervals["auroc"]["high"],
                        "auprc": intervals["auprc"]["estimate"],
                        "auprc_low": intervals["auprc"]["low"],
                        "auprc_high": intervals["auprc"]["high"],
                        "brier": intervals["brier"]["estimate"],
                        "brier_low": intervals["brier"]["low"],
                        "brier_high": intervals["brier"]["high"],
                        "bootstrap_replicates": min(
                            int(intervals[name]["replicates"])
                            for name in (
                                "event_rate",
                                "auroc",
                                "auprc",
                                "brier",
                            )
                        ),
                    }
                )
    csv_path = OUT / "leave_one_branch_out.csv"
    write_csv(csv_path, output_rows)
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        (
            r"Model & Target & Held-out branch & $n$ & Event rate & "
            r"AUROC (95\% CI) & AUPRC (95\% CI) & Brier (95\% CI) \\"
        ),
        r"\midrule",
    ]
    scope_labels = {
        "pooled_1_to_4": "Pooled 1--4",
        "held_out_1": "1 (on-spine)",
        "held_out_2": "2 (off-spine)",
        "held_out_3": "3 (off-spine)",
        "held_out_4": "4 (off-spine)",
    }
    for model in RUNS:
        for target in ("beneficial", "harmful"):
            selected = [
                row
                for row in output_rows
                if row["model"] == model and row["target"] == target
            ]
            for row in selected:
                tex.append(
                    "{} & {} & {} & {:,} & {:.1f}\\% & {} & {} & {} \\\\".format(
                        MODEL_SHORT[model],
                        target.capitalize(),
                        scope_labels[row["scope"]],
                        row["n"],
                        100 * row["event_rate"],
                        _fmt_metric_interval(
                            {
                                "estimate": row["auroc"],
                                "low": row["auroc_low"],
                                "high": row["auroc_high"],
                            }
                        ),
                        _fmt_metric_interval(
                            {
                                "estimate": row["auprc"],
                                "low": row["auprc_low"],
                                "high": row["auprc_high"],
                            }
                        ),
                        _fmt_metric_interval(
                            {
                                "estimate": row["brier"],
                                "low": row["brier_low"],
                                "high": row["brier_high"],
                            }
                        ),
                    )
                )
            tex.append(r"\addlinespace")
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Leave-one-branch-out prediction on confirmation "
                r"checkpoints. The mean event frequency of the other three "
                r"branches predicts the held-out branch event. Beneficial is "
                r"conditioned on $S_s=0$ and harmful on $S_s=1$. Branch 1 is "
                r"the realized replay successor. Intervals resample complete "
                r"problems 2,000 times.}"
            ),
            r"\label{tab:leave-one-branch-out}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = OUT / "leave_one_branch_out_table.tex"
    atomic_text(tex_path, "\n".join(tex))
    return [csv_path, tex_path]


def select_nonmyopic_dp_subsets(
    states: list[dict[str, Any]],
    *,
    model: str,
) -> list[list[tuple[str, int]]]:
    initial = [state for state in states if int(state["depth"]) == 256]
    by_problem: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for state in initial:
        by_problem[str(state["problem_id"])].append(
            (str(state["problem_id"]), int(state["spine_id"]))
        )
    if any(len(keys) != 4 for keys in by_problem.values()):
        raise ValueError("DP sampling expects four initial spines per problem")
    needed = NONMYOPIC_DP_SUBSETS * NONMYOPIC_DP_FRONTIERS_PER_SUBSET
    if len(by_problem) < needed:
        raise ValueError(
            f"DP sampling needs {needed} distinct problems, " f"found {len(by_problem)}"
        )
    ordered_problems = sorted(
        by_problem,
        key=lambda problem_id: hashlib.sha256(
            f"{NONMYOPIC_DP_NAMESPACE}:{model}:{problem_id}".encode()
        ).hexdigest(),
    )[:needed]
    selected: list[tuple[str, int]] = []
    for problem_id in ordered_problems:
        keys = sorted(
            by_problem[problem_id],
            key=lambda key: hashlib.sha256(
                (f"{NONMYOPIC_DP_NAMESPACE}:{model}:" f"{key[0]}:{key[1]}").encode()
            ).hexdigest(),
        )
        selected.append(keys[0])
    subsets = [
        selected[index : index + NONMYOPIC_DP_FRONTIERS_PER_SUBSET]
        for index in range(0, len(selected), NONMYOPIC_DP_FRONTIERS_PER_SUBSET)
    ]
    if len(subsets) != NONMYOPIC_DP_SUBSETS or any(
        len(subset) != NONMYOPIC_DP_FRONTIERS_PER_SUBSET
        or len({key[0] for key in subset}) != NONMYOPIC_DP_FRONTIERS_PER_SUBSET
        for subset in subsets
    ):
        raise ValueError("invalid deterministic DP subset construction")
    return subsets


def exact_nonmyopic_aulc(
    chains: list[list[dict[str, Any]]],
) -> tuple[float, int, int]:
    if len(chains) != NONMYOPIC_DP_FRONTIERS_PER_SUBSET or any(
        len(chain) != 16 for chain in chains
    ):
        raise ValueError("exact DP expects four complete 16-state chains")
    stop_correct = tuple(
        tuple(int(state["stop_correct"]) for state in chain) for chain in chains
    )
    transition_costs = tuple(
        tuple(int(state["transition_cost"]) for state in chain[:-1]) for chain in chains
    )
    if any(
        int(chain[-1]["terminal"]) != 1 or any(cost <= 0 for cost in costs)
        for chain, costs in zip(chains, transition_costs, strict=True)
    ):
        raise ValueError("exact DP received an incomplete or invalid chain")
    terminal_index = len(chains[0]) - 1

    @lru_cache(maxsize=None)
    def optimum(position: tuple[int, ...]) -> float:
        if all(index == terminal_index for index in position):
            return 0.0
        accuracy = sum(
            stop_correct[chain_index][state_index]
            for chain_index, state_index in enumerate(position)
        ) / len(chains)
        candidates: list[float] = []
        for chain_index, state_index in enumerate(position):
            if state_index == terminal_index:
                continue
            next_position = list(position)
            next_position[chain_index] += 1
            candidates.append(
                transition_costs[chain_index][state_index] * accuracy
                + optimum(tuple(next_position))
            )
        return max(candidates)

    initial_position = tuple(0 for _ in chains)
    area = optimum(initial_position)
    span = sum(sum(costs) for costs in transition_costs)
    return area, span, optimum.cache_info().currsize


def _ratio_bootstrap_interval(
    rows: list[dict[str, Any]],
    numerator: Callable[[dict[str, Any]], float],
    denominator: Callable[[dict[str, Any]], float],
    *,
    seed: int,
) -> dict[str, float | int]:
    numerator_values = np.asarray([float(numerator(row)) for row in rows], dtype=float)
    denominator_values = np.asarray(
        [float(denominator(row)) for row in rows], dtype=float
    )
    observed = float(numerator_values.sum() / denominator_values.sum())
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indexes = rng.integers(0, len(rows), size=len(rows))
        sampled_denominator = float(denominator_values[indexes].sum())
        if sampled_denominator > 0:
            samples.append(float(numerator_values[indexes].sum() / sampled_denominator))
    return {
        "estimate": observed,
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
        "replicates": len(samples),
    }


def render_nonmyopic_dp(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    subset_rows: list[dict[str, Any]] = []
    selection_payload: dict[str, Any] = {}
    policy_order = (
        "round_robin",
        "mixed_myopic",
        "on_spine_one_step_hindsight",
        "off_spine_myopic",
        "exact_nonmyopic_hindsight",
    )
    for model, payload in validated.items():
        run_dir = payload["run_dir"]
        config = RunConfig.model_validate(
            read_json(run_dir / "manifest.json")["config"]
        )
        states, _ = load_confirmation_frontiers(
            run_dir,
            payload["evaluation"],
            values_by_model[model],
        )
        all_scores = oracle_score_maps(values_by_model[model])
        subsets = select_nonmyopic_dp_subsets(states, model=model)
        states_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for state in states:
            states_by_key[(str(state["problem_id"]), int(state["spine_id"]))].append(
                state
            )
        for chain in states_by_key.values():
            chain.sort(key=lambda state: int(state["depth"]))
        selection_payload[model] = {
            "subsets": [
                [
                    {"problem_id": problem_id, "spine_id": spine_id}
                    for problem_id, spine_id in subset
                ]
                for subset in subsets
            ],
            "selection_sha256": hashlib.sha256(
                json.dumps(
                    subsets,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        }
        for subset_index, keys in enumerate(subsets):
            chains = [states_by_key[key] for key in keys]
            if any(
                [int(state["depth"]) for state in chain] != list(range(256, 4097, 256))
                for chain in chains
            ):
                raise ValueError(f"{model}/subset-{subset_index}: incomplete DP chain")
            subset_states = [state for chain in chains for state in chain]
            curves = {
                "round_robin": _replay_one(
                    subset_states,
                    "round_robin",
                    checkpoint_interval=config.protocol.checkpoint_interval,
                    fixed_allotment_tokens=(config.execution.fixed_allotment_tokens),
                )
            }
            score_names = {
                "mixed_myopic": "mixed",
                "on_spine_one_step_hindsight": "hindsight_on",
                "off_spine_myopic": "off_spine",
            }
            subset_checkpoint_ids = {
                str(state["checkpoint_id"]) for state in subset_states
            }
            for policy, score_name in score_names.items():
                overrides = {
                    checkpoint_id: score
                    for checkpoint_id, score in all_scores[score_name].items()
                    if checkpoint_id in subset_checkpoint_ids
                }
                curves[policy] = _replay_one(
                    subset_states,
                    "oracle",
                    checkpoint_interval=config.protocol.checkpoint_interval,
                    fixed_allotment_tokens=(config.execution.fixed_allotment_tokens),
                    overridden_values=overrides,
                )
            start_budget = curves["round_robin"][0].budget
            endpoint = min(curve[-1].budget for curve in curves.values())
            policy_aulc = {
                policy: step_auc(
                    curve,
                    endpoint,
                    start_budget=start_budget,
                )
                for policy, curve in curves.items()
            }
            dp_aulc, dp_span, visited_states = exact_nonmyopic_aulc(chains)
            if dp_span != endpoint - start_budget:
                raise ValueError(
                    f"{model}/subset-{subset_index}: DP ledger span mismatch"
                )
            if dp_aulc + 1e-9 < max(policy_aulc.values()):
                raise ValueError(
                    f"{model}/subset-{subset_index}: exact DP is not an upper " "bound"
                )
            policy_aulc["exact_nonmyopic_hindsight"] = dp_aulc
            round_robin_auc = policy_aulc["round_robin"]
            subset_id = f"{MODEL_SHORT[model].lower()}-" f"{subset_index:02d}"
            for policy in policy_order:
                area = policy_aulc[policy]
                subset_rows.append(
                    {
                        "model": model,
                        "subset_id": subset_id,
                        "problem_ids": ";".join(key[0] for key in keys),
                        "spine_ids": ";".join(str(key[1]) for key in keys),
                        "frontiers": len(keys),
                        "states_per_frontier": 16,
                        "policy": policy,
                        "aulc": area,
                        "aulc_span": dp_span,
                        "mean_accuracy": area / dp_span,
                        "relative_gain_vs_round_robin_percent": (
                            100.0 * (area - round_robin_auc) / round_robin_auc
                        ),
                        "relative_gap_to_dp_percent": (
                            100.0 * (dp_aulc - area) / dp_aulc
                        ),
                        "dp_visited_states": visited_states,
                    }
                )
    subset_csv = OUT / "nonmyopic_dp_subsets.csv"
    write_csv(subset_csv, subset_rows)

    summary_rows: list[dict[str, Any]] = []
    for model in RUNS:
        model_rows = [row for row in subset_rows if row["model"] == model]
        by_subset = {
            subset_id: {
                row["policy"]: row
                for row in model_rows
                if row["subset_id"] == subset_id
            }
            for subset_id in sorted({row["subset_id"] for row in model_rows})
        }
        for policy in policy_order:
            selected = [rows_by_policy[policy] for rows_by_policy in by_subset.values()]
            mean_accuracy = _ratio_bootstrap_interval(
                selected,
                lambda row: row["aulc"],
                lambda row: row["aulc_span"],
                seed=stable_seed(
                    BOOTSTRAP_SEED,
                    NONMYOPIC_DP_NAMESPACE,
                    model,
                    policy,
                    "mean-accuracy",
                ),
            )
            gain = _ratio_bootstrap_interval(
                selected,
                lambda row: (
                    row["aulc"] - by_subset[row["subset_id"]]["round_robin"]["aulc"]
                ),
                lambda row: by_subset[row["subset_id"]]["round_robin"]["aulc"],
                seed=stable_seed(
                    BOOTSTRAP_SEED,
                    NONMYOPIC_DP_NAMESPACE,
                    model,
                    policy,
                    "gain",
                ),
            )
            gap = _ratio_bootstrap_interval(
                selected,
                lambda row: (
                    by_subset[row["subset_id"]]["exact_nonmyopic_hindsight"]["aulc"]
                    - row["aulc"]
                ),
                lambda row: by_subset[row["subset_id"]]["exact_nonmyopic_hindsight"][
                    "aulc"
                ],
                seed=stable_seed(
                    BOOTSTRAP_SEED,
                    NONMYOPIC_DP_NAMESPACE,
                    model,
                    policy,
                    "gap",
                ),
            )
            summary_rows.append(
                {
                    "model": model,
                    "policy": policy,
                    "subsets": len(selected),
                    "distinct_problems": len(selected)
                    * NONMYOPIC_DP_FRONTIERS_PER_SUBSET,
                    "mean_accuracy": mean_accuracy["estimate"],
                    "mean_accuracy_low": mean_accuracy["low"],
                    "mean_accuracy_high": mean_accuracy["high"],
                    "relative_gain_vs_round_robin_percent": (100.0 * gain["estimate"]),
                    "relative_gain_low": 100.0 * gain["low"],
                    "relative_gain_high": 100.0 * gain["high"],
                    "relative_gap_to_dp_percent": 100.0 * gap["estimate"],
                    "relative_gap_low": 100.0 * gap["low"],
                    "relative_gap_high": 100.0 * gap["high"],
                    "bootstrap_replicates": min(
                        mean_accuracy["replicates"],
                        gain["replicates"],
                        gap["replicates"],
                    ),
                }
            )
    summary_csv = OUT / "nonmyopic_dp_summary.csv"
    write_csv(summary_csv, summary_rows)

    labels = {
        "round_robin": "Round-robin",
        "mixed_myopic": "Mixed myopic",
        "on_spine_one_step_hindsight": "On-spine one-step hindsight",
        "off_spine_myopic": "Off-spine myopic",
        "exact_nonmyopic_hindsight": "Exact non-myopic hindsight",
    }
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Model & Policy & Mean accuracy & Gain vs.\ RR (95\% CI) & "
            r"Gap to exact DP (95\% CI) \\"
        ),
        r"\midrule",
    ]
    for model in RUNS:
        for row in [row for row in summary_rows if row["model"] == model]:
            tex.append(
                "{} & {} & {:.3f} & {:+.2f}\\% [{:+.2f}, {:+.2f}] & "
                "{:.2f}\\% [{:.2f}, {:.2f}] \\\\".format(
                    MODEL_SHORT[model],
                    labels[row["policy"]],
                    row["mean_accuracy"],
                    row["relative_gain_vs_round_robin_percent"],
                    row["relative_gain_low"],
                    row["relative_gain_high"],
                    row["relative_gap_to_dp_percent"],
                    row["relative_gap_low"],
                    row["relative_gap_high"],
                )
            )
        tex.append(r"\addlinespace")
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Exact non-myopic hindsight scheduling on 64 "
                r"deterministic, non-overlapping subsets per model. Each "
                r"subset contains one complete 16-state frontier from each "
                r"of four distinct confirmation problems. Dynamic programming "
                r"enumerates all $16^4$ precedence-feasible frontier states "
                r"and maximizes full-ledger AULC. Intervals bootstrap the 64 "
                r"disjoint four-problem subsets 2,000 times. The exact DP is "
                r"a genuine upper bound only for these sampled subsets.}"
            ),
            r"\label{tab:nonmyopic-dp}",
            r"\end{table*}",
            "",
        ]
    )
    table_path = OUT / "nonmyopic_dp_table.tex"
    atomic_text(table_path, "\n".join(tex))

    summary_path = OUT / "nonmyopic_dp_manifest.json"
    sealed = seal_record(
        {
            "schema_version": "flcva-nonmyopic-dp-v1",
            "analysis_status": "supplementary, label-aware, read-only",
            "selection_namespace": NONMYOPIC_DP_NAMESPACE,
            "subsets_per_model": NONMYOPIC_DP_SUBSETS,
            "frontiers_per_subset": NONMYOPIC_DP_FRONTIERS_PER_SUBSET,
            "states_per_frontier": 16,
            "distinct_problems_per_model": (
                NONMYOPIC_DP_SUBSETS * NONMYOPIC_DP_FRONTIERS_PER_SUBSET
            ),
            "objective": (
                "maximize right-continuous full-ledger AULC over every "
                "precedence-feasible interleaving of the four recorded "
                "on-spine successor chains"
            ),
            "selection": selection_payload,
            "source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for model, payload in validated.items()
            },
        }
    )
    atomic_text(
        summary_path,
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
    )
    summary_by_key = {(row["model"], row["policy"]): row for row in summary_rows}
    macro_lines = [
        "% Auto-generated from the sealed exact non-myopic DP analysis.",
    ]
    for model, prefix in (
        ("Qwen3-8B", "Qwen"),
        ("DeepSeek-R1-Distill-Qwen-7B", "DeepSeek"),
    ):
        row = summary_by_key[(model, "exact_nonmyopic_hindsight")]
        macro_lines.append(
            f"\\newcommand{{\\{prefix}ExactDPGainWithCI}}{{"
            f"{row['relative_gain_vs_round_robin_percent']:+.2f}\\% "
            f"[{row['relative_gain_low']:+.2f}, "
            f"{row['relative_gain_high']:+.2f}]}}"
        )
    macros_path = OUT / "nonmyopic_dp_macros.tex"
    atomic_text(macros_path, "\n".join(macro_lines) + "\n")
    return [subset_csv, summary_csv, table_path, summary_path, macros_path]


def render_replay_policy_assets() -> list[Path]:
    replay_csv = OUT / "replay.csv"
    points_csv = OUT / "replay_points.csv"
    replay_rows = list(csv.DictReader(replay_csv.open(encoding="utf-8")))
    point_rows = list(csv.DictReader(points_csv.open(encoding="utf-8")))
    if not replay_rows or not point_rows:
        raise ValueError("primary replay assets are absent")
    labels = {
        "fixed_1024": "Fixed 1,024",
        "round_robin": "Round-robin",
        "trace": "TRACE-style",
        "agreement": "Agreement",
        "prm": "PRM",
        "anll": "ANLL",
        "entropy": "Entropy",
        "oracle": "Mixed myopic value",
    }
    table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Model & Scheduler & Mean accuracy & Final accuracy \\",
        r"\midrule",
    ]
    for row in replay_rows:
        table.append(
            "{} & {} & {:.3f} & {:.3f} \\\\".format(
                str(row["model"]).replace("_", r"\_"),
                labels[row["scheduler"]],
                float(row["mean_accuracy"]),
                float(row["final_accuracy"]),
            )
        )
    table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Equal-ledger replay through each model's common "
                r"endpoint. The mixed myopic row is the original four-branch "
                r"label-aware score; it is not a global ceiling. Mean accuracy "
                r"is AULC divided by the observable budget span.}"
            ),
            r"\label{tab:sealed-replay-results}",
            r"\end{table}",
            "",
        ]
    )
    table_path = OUT / "replay_policy_table.tex"
    atomic_text(table_path, "\n".join(table))

    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(row["model"] for row in point_rows))
    fig, axes = plt.subplots(1, len(models), figsize=(9.8, 3.6), sharey=True)
    if len(models) == 1:
        axes = [axes]
    colors = {
        "fixed_1024": "#7f7f7f",
        "round_robin": "#111111",
        "trace": "#ff7f0e",
        "agreement": "#2ca02c",
        "prm": "#9467bd",
        "anll": "#8c564b",
        "entropy": "#e377c2",
        "oracle": "#1f77b4",
    }
    for axis, model in zip(axes, models, strict=True):
        for scheduler in labels:
            selected = [
                row
                for row in point_rows
                if row["model"] == model and row["scheduler"] == scheduler
            ]
            selected.sort(key=lambda row: int(row["budget"]))
            axis.plot(
                [float(row["normalized_budget"]) for row in selected],
                [float(row["accuracy"]) for row in selected],
                label=labels[scheduler],
                color=colors[scheduler],
                linewidth=1.1,
                drawstyle="steps-post",
            )
        axis.set_title(
            "Qwen3" if "Qwen3" in model else "DeepSeek",
        )
        axis.set_xlabel("Normalized full-ledger budget")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Frontier exact-match accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    figure_path = OUT / "replay_policy_curves.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return [table_path, figure_path]


def _replay_one_with_problem_events(
    states: list[dict[str, Any]],
    scheduler: str,
    *,
    checkpoint_interval: int,
    fixed_allotment_tokens: int,
    overridden_values: dict[str, float] | None = None,
    charge_prm: bool = False,
) -> tuple[
    list[ReplayPoint],
    dict[str, int],
    dict[str, list[tuple[int, int]]],
    list[str],
]:
    """Mirror the sealed replay while retaining problem-level change events."""

    by_id = {str(state["checkpoint_id"]): dict(state) for state in states}
    if len(by_id) != len(states):
        raise ValueError("spine-replicate replay received duplicate checkpoints")
    first_states = [
        state for state in states if int(state["depth"]) == checkpoint_interval
    ]
    problem_ids = [str(state["problem_id"]) for state in first_states]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError(
            "spine-replicate replay requires one initial frontier per problem"
        )
    frontiers = {
        (str(state["problem_id"]), int(state["spine_id"])): dict(state)
        for state in first_states
    }
    if not frontiers:
        raise ValueError("spine-replicate replay has no initial frontiers")
    initial_correct = {
        str(state["problem_id"]): int(bool(state["stop_correct"]))
        for state in first_states
    }
    change_events: dict[str, list[tuple[int, int]]] = {
        problem_id: [] for problem_id in initial_correct
    }
    action_checkpoint_ids: list[str] = []
    total_frontiers = len(frontiers)
    correct_count = sum(initial_correct.values())
    active_keys = {key for key, state in frontiers.items() if not state["terminal"]}
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
    cost_scheduler = "prm" if charge_prm else scheduler
    scheduler_ready_budget = _initial_cost(first_states, cost_scheduler)
    if scheduler_ready_budget > observation_budget:
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
    priority_heap: list[tuple[float, int, _DescendingText, str, tuple[str, int]]] = []

    def push_priority(key: tuple[str, int]) -> None:
        state = frontiers[key]
        if overridden_values is not None:
            scalar = overridden_values.get(
                str(state["checkpoint_id"]),
                -math.inf,
            )
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
                fixed_remaining = max(
                    1,
                    fixed_allotment_tokens // checkpoint_interval,
                )
            if fixed_key_index >= len(ordered_keys):
                fixed_key_index = 0
                continue
            chosen = ordered_keys[fixed_key_index]
            fixed_remaining -= 1
            if fixed_remaining == 0:
                fixed_key_index += 1
                fixed_remaining = max(
                    1,
                    fixed_allotment_tokens // checkpoint_interval,
                )
        else:
            while priority_heap:
                _, _, _, checkpoint_id, candidate = heapq.heappop(priority_heap)
                if (
                    candidate in active_keys
                    and str(frontiers[candidate]["checkpoint_id"]) == checkpoint_id
                ):
                    chosen = candidate
                    break
            else:
                raise ValueError("spine-replicate priority heap exhausted")
        current = frontiers[chosen]
        action_checkpoint_ids.append(str(current["checkpoint_id"]))
        successor_id = str(current.get("successor_id") or "")
        if successor_id not in by_id:
            raise ValueError(
                f"spine-replicate successor missing: " f"{current['checkpoint_id']}"
            )
        successor = dict(by_id[successor_id])
        budget += _transition_cost(current)
        if scheduler == "prm" or charge_prm:
            budget += int(
                (successor.get("proxy_costs") or {}).get(
                    "prm_min_step",
                    0,
                )
            )
        old_correct = int(bool(current["stop_correct"]))
        new_correct = int(bool(successor["stop_correct"]))
        frontiers[chosen] = successor
        correct_count += new_correct - old_correct
        if new_correct != old_correct:
            change_events[str(chosen[0])].append((budget, new_correct))
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
    return points, initial_correct, change_events, action_checkpoint_ids


def _assert_replay_equivalent(
    observed: list[ReplayPoint],
    reference: list[ReplayPoint],
) -> None:
    if len(observed) != len(reference):
        raise ValueError("spine-replicate replay point count diverged")
    for left, right in zip(observed, reference, strict=True):
        if (
            left.budget != right.budget
            or left.active_frontiers != right.active_frontiers
            or left.terminal_frontiers != right.terminal_frontiers
            or not math.isclose(
                left.accuracy,
                right.accuracy,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("spine-replicate replay diverged from the sealed core")


def _problem_contributions(
    initial_correct: dict[str, int],
    change_events: dict[str, list[tuple[float, int]]],
    *,
    start_budget: float,
    endpoint: float,
) -> dict[str, dict[str, float | int]]:
    if endpoint <= start_budget:
        raise ValueError("problem contributions require a positive AULC span")
    span = endpoint - start_budget
    rows: dict[str, dict[str, float | int]] = {}
    for problem_id, initial in initial_correct.items():
        current = int(initial)
        previous_budget = start_budget
        area = 0.0
        for budget, new_correct in change_events[problem_id]:
            if budget <= start_budget:
                current = int(new_correct)
                continue
            current_budget = min(endpoint, float(budget))
            if current_budget > previous_budget:
                area += (current_budget - previous_budget) * current
                previous_budget = current_budget
            if budget <= endpoint:
                current = int(new_correct)
            if budget >= endpoint:
                break
        if previous_budget < endpoint:
            area += (endpoint - previous_budget) * current
        rows[problem_id] = {
            "normalized_aulc": area / span,
            "final_correct": current,
        }
    return rows


def _accuracy_at_budget(
    curve: list[ReplayPoint],
    budget: float,
) -> float:
    accuracy = curve[0].accuracy
    for point in curve:
        if point.budget > budget:
            break
        accuracy = point.accuracy
    return float(accuracy)


def _build_compact_replay_context(
    states: list[dict[str, Any]],
    policy_score_maps: dict[str, dict[str, float]],
) -> CompactReplayContext:
    """Validate and compact one spine without changing replay semantics."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        grouped[str(state["problem_id"])].append(state)
    problem_ids = tuple(sorted(grouped))
    if not problem_ids:
        raise ValueError("compact replay requires at least one problem")

    initial_costs: list[int] = []
    correctness: list[list[int]] = []
    transition_costs: list[list[int]] = []
    policy_scores: dict[str, list[list[float]]] = {
        policy: [] for policy in policy_score_maps
    }
    expected_depths: tuple[int, ...] | None = None
    for problem_id in problem_ids:
        chain = sorted(
            grouped[problem_id],
            key=lambda state: int(state["depth"]),
        )
        depths = tuple(int(state["depth"]) for state in chain)
        if expected_depths is None:
            expected_depths = depths
        elif depths != expected_depths:
            raise ValueError(f"{problem_id}: compact replay depth lattice mismatch")
        if len(chain) < 2 or not bool(chain[-1]["terminal"]):
            raise ValueError(f"{problem_id}: compact replay chain is incomplete")
        if any(bool(state["terminal"]) for state in chain[:-1]):
            raise ValueError(
                f"{problem_id}: compact replay terminates before final state"
            )
        for current, successor in zip(chain[:-1], chain[1:], strict=True):
            if str(current.get("successor_id") or "") != str(
                successor["checkpoint_id"]
            ):
                raise ValueError(f"{problem_id}: compact replay successor mismatch")
        initial_costs.append(_initial_base_cost([chain[0]]))
        correctness.append([int(bool(state["stop_correct"])) for state in chain])
        transition_costs.append([_transition_cost(state) for state in chain[:-1]])
        for policy, score_map in policy_score_maps.items():
            values: list[float] = []
            for state in chain[:-1]:
                checkpoint_id = str(state["checkpoint_id"])
                if checkpoint_id not in score_map:
                    raise ValueError(
                        f"{policy}: missing compact replay score "
                        f"for {checkpoint_id}"
                    )
                value = float(score_map[checkpoint_id])
                if not math.isfinite(value):
                    raise ValueError(
                        f"{policy}: non-finite compact replay score "
                        f"for {checkpoint_id}"
                    )
                values.append(value)
            policy_scores[policy].append(values)

    correct_array = np.asarray(correctness, dtype=np.int8)
    transition_array = np.asarray(transition_costs, dtype=np.int64)
    if correct_array.ndim != 2 or transition_array.shape != (
        len(problem_ids),
        correct_array.shape[1] - 1,
    ):
        raise ValueError("compact replay arrays have inconsistent shapes")
    return CompactReplayContext(
        problem_ids=problem_ids,
        initial_costs=np.asarray(initial_costs, dtype=np.int64),
        stop_correct=correct_array,
        transition_costs=transition_array,
        policy_scores={
            policy: np.asarray(rows, dtype=float)
            for policy, rows in policy_scores.items()
        },
    )


def _bootstrap_instance_layout(
    problem_ids: tuple[str, ...],
    sampled_indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic independent identities for duplicate draws."""

    counts: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for source_index in sampled_indexes:
        totals[problem_ids[int(source_index)]] += 1
    clone_ids: list[str] = []
    for source_index in sampled_indexes:
        problem_id = problem_ids[int(source_index)]
        occurrence = counts[problem_id]
        counts[problem_id] += 1
        clone_ids.append(
            problem_id
            if totals[problem_id] == 1
            else f"{problem_id}::bootstrap-copy-{occurrence:04d}"
        )
    ordered = np.asarray(
        sorted(range(len(clone_ids)), key=clone_ids.__getitem__),
        dtype=np.int64,
    )
    tie_rank = np.empty(len(clone_ids), dtype=np.int64)
    tie_rank[ordered] = np.arange(len(clone_ids), dtype=np.int64)
    return ordered, tie_rank


def _compact_system_aulc(
    context: CompactReplayContext,
    sampled_indexes: np.ndarray,
    *,
    policy: str,
    ordered_instances: np.ndarray,
    tie_rank: np.ndarray,
) -> float:
    """Rerun global ordering and return normalized full-ledger AULC."""

    sampled_indexes = np.asarray(sampled_indexes, dtype=np.int64)
    instances = len(sampled_indexes)
    if (
        instances == 0
        or ordered_instances.shape != (instances,)
        or tie_rank.shape != (instances,)
    ):
        raise ValueError("compact replay instance layout is invalid")
    initial_costs = context.initial_costs[sampled_indexes]
    correctness = context.stop_correct[sampled_indexes]
    costs = context.transition_costs[sampled_indexes]
    transitions = costs.shape[1]
    if correctness.shape != (instances, transitions + 1):
        raise ValueError("compact replay sampled lattice is invalid")
    span = int(costs.sum())
    start_budget = int(initial_costs.sum())
    common_endpoint = start_budget + span
    if span <= 0 or common_endpoint <= start_budget:
        raise ValueError("compact replay has a nonpositive budget span")

    correct_count = int(correctness[:, 0].sum())
    area = 0.0
    if policy == "round_robin":
        for depth in range(transitions):
            for instance_value in ordered_instances:
                instance = int(instance_value)
                cost = int(costs[instance, depth])
                area += cost * correct_count / instances
                correct_count += int(
                    correctness[instance, depth + 1] - correctness[instance, depth]
                )
    else:
        if policy not in context.policy_scores:
            raise ValueError(f"compact replay score is missing: {policy}")
        scores = context.policy_scores[policy][sampled_indexes]
        if scores.shape != (instances, transitions):
            raise ValueError("compact replay sampled score matrix is invalid")
        priority_heap: list[tuple[float, int, int, int]] = [
            (
                -float(scores[instance, 0]),
                -int(tie_rank[instance]),
                instance,
                0,
            )
            for instance in range(instances)
        ]
        heapq.heapify(priority_heap)
        while priority_heap:
            _, _, instance, depth = heapq.heappop(priority_heap)
            cost = int(costs[instance, depth])
            area += cost * correct_count / instances
            correct_count += int(
                correctness[instance, depth + 1] - correctness[instance, depth]
            )
            next_depth = depth + 1
            if next_depth < transitions:
                heapq.heappush(
                    priority_heap,
                    (
                        -float(scores[instance, next_depth]),
                        -int(tie_rank[instance]),
                        instance,
                        next_depth,
                    ),
                )
    return area / span


def _paired_full_replay_bootstrap(
    contexts: list[CompactReplayContext],
    *,
    policies: tuple[str, ...],
    replicates: int,
    seed_parts: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Resample the pool and rerun every paired policy and spine."""

    if not contexts or replicates <= 0:
        raise ValueError("full replay bootstrap configuration is invalid")
    problem_ids = contexts[0].problem_ids
    if any(context.problem_ids != problem_ids for context in contexts[1:]):
        raise ValueError("full replay bootstrap problem universes differ")
    if "round_robin" not in policies:
        raise ValueError("full replay bootstrap requires round-robin")
    indexes = np.arange(len(problem_ids), dtype=np.int64)

    def replay_sample(
        sampled: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        ordered, tie_rank = _bootstrap_instance_layout(
            problem_ids,
            sampled,
        )
        values = {
            policy: float(
                np.mean(
                    [
                        _compact_system_aulc(
                            context,
                            sampled,
                            policy=policy,
                            ordered_instances=ordered,
                            tie_rank=tie_rank,
                        )
                        for context in contexts
                    ]
                )
            )
            for policy in policies
        }
        starts = [int(context.initial_costs[sampled].sum()) for context in contexts]
        spans = [int(context.transition_costs[sampled].sum()) for context in contexts]
        return values, {
            "mean_start_budget": float(np.mean(starts)),
            "mean_common_endpoint": float(
                np.mean(
                    [start + span for start, span in zip(starts, spans, strict=True)]
                )
            ),
            "mean_budget_span": float(np.mean(spans)),
        }

    observed, _ = replay_sample(indexes)
    samples = {policy: np.empty(replicates, dtype=float) for policy in policies}
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(
        stable_seed(
            BOOTSTRAP_SEED,
            FULL_REPLAY_BOOTSTRAP_NAMESPACE,
            *seed_parts,
        )
    )
    for replicate in range(replicates):
        sampled = rng.integers(
            0,
            len(problem_ids),
            size=len(problem_ids),
            dtype=np.int64,
        )
        replayed, domain = replay_sample(sampled)
        for policy, value in replayed.items():
            samples[policy][replicate] = value
        shortcut = replayed["correctness_stratified_ev"]
        round_robin = replayed["round_robin"]
        rows.append(
            {
                "replicate": replicate,
                "unique_source_problems": int(len(np.unique(sampled))),
                **domain,
                **{
                    f"{policy}_normalized_aulc": replayed[policy] for policy in policies
                },
                "shortcut_gain_vs_round_robin_percent": (
                    100.0 * (shortcut - round_robin) / round_robin
                ),
                "two_stage_increment_vs_shortcut_percent": (
                    100.0 * (replayed["two_stage_ev"] - shortcut) / shortcut
                ),
                "ridge_increment_vs_shortcut_percent": (
                    100.0 * (replayed["direct_ridge_m"] - shortcut) / shortcut
                ),
            }
        )
    return observed, samples, rows


def render_spine_replicate_replay(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    policy_specs = {
        "fixed_1024": ("fixed_1024", None),
        "round_robin": ("round_robin", None),
        "trace": ("trace", None),
        "agreement": ("agreement", None),
        "prm": ("prm", None),
        "anll": ("anll", None),
        "entropy": ("entropy", None),
        "mixed_myopic": ("oracle", "mixed"),
        "on_spine_one_step_hindsight": ("oracle", "hindsight_on"),
        "off_spine_myopic": ("oracle", "off_spine"),
    }
    seed_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    equivalence_checks = 0
    decomposition_checks = 0
    problem_universe_sha256: dict[str, str] = {}
    for model, payload in validated.items():
        config = RunConfig.model_validate(
            read_json(payload["run_dir"] / "manifest.json")["config"]
        )
        states, _ = load_confirmation_frontiers(
            payload["run_dir"],
            payload["evaluation"],
            values_by_model[model],
        )
        scores = oracle_score_maps(values_by_model[model])
        spine_ids = sorted({int(state["spine_id"]) for state in states})
        if spine_ids != [0, 1, 2, 3]:
            raise ValueError(f"{model}: expected spine IDs 0--3")
        model_problems = sorted(
            {str(state["problem_id"]) for state in states if int(state["depth"]) == 256}
        )
        if len(model_problems) != 372:
            raise ValueError(
                f"{model}: expected 372 confirmation problems, "
                f"found {len(model_problems)}"
            )
        problem_universe_sha256[model] = hashlib.sha256(
            "\n".join(model_problems).encode()
        ).hexdigest()
        curves_by_seed: dict[
            int,
            dict[str, list[ReplayPoint]],
        ] = {}
        domains_by_seed: dict[int, tuple[int, int]] = {}
        for spine_id in spine_ids:
            selected = [state for state in states if int(state["spine_id"]) == spine_id]
            if len(selected) != 372 * 16:
                raise ValueError(f"{model}/spine-{spine_id}: incomplete state lattice")
            initial = [state for state in selected if int(state["depth"]) == 256]
            if {str(state["problem_id"]) for state in initial} != set(model_problems):
                raise ValueError(f"{model}/spine-{spine_id}: problem universe mismatch")
            traces: dict[
                str,
                tuple[
                    list[ReplayPoint],
                    dict[str, int],
                    dict[str, list[tuple[int, int]]],
                    list[str],
                ],
            ] = {}
            for policy, (scheduler, score_name) in policy_specs.items():
                overrides = scores[str(score_name)] if score_name is not None else None
                trace = _replay_one_with_problem_events(
                    selected,
                    scheduler,
                    checkpoint_interval=(config.protocol.checkpoint_interval),
                    fixed_allotment_tokens=(config.execution.fixed_allotment_tokens),
                    overridden_values=overrides,
                )
                reference = _replay_one(
                    selected,
                    scheduler,
                    checkpoint_interval=(config.protocol.checkpoint_interval),
                    fixed_allotment_tokens=(config.execution.fixed_allotment_tokens),
                    overridden_values=overrides,
                )
                _assert_replay_equivalent(trace[0], reference)
                equivalence_checks += 1
                traces[policy] = trace
            start_budget = traces["round_robin"][0][0].budget
            endpoint = min(trace[0][-1].budget for trace in traces.values())
            if any(trace[0][0].budget != start_budget for trace in traces.values()):
                raise ValueError(f"{model}/spine-{spine_id}: start budget mismatch")
            span = endpoint - start_budget
            if span <= 0:
                raise ValueError(f"{model}/spine-{spine_id}: empty replay domain")
            curves_by_seed[spine_id] = {
                policy: trace[0] for policy, trace in traces.items()
            }
            domains_by_seed[spine_id] = (start_budget, endpoint)
            seed_metrics: dict[str, dict[str, float]] = {}
            for policy, (
                curve,
                initial_correct,
                changes,
                _,
            ) in traces.items():
                area = step_auc(
                    curve,
                    endpoint,
                    start_budget=start_budget,
                )
                mean_accuracy = area / span
                final_accuracy = _accuracy_at_budget(curve, endpoint)
                contributions = _problem_contributions(
                    initial_correct,
                    changes,
                    start_budget=start_budget,
                    endpoint=endpoint,
                )
                contribution_mean = float(
                    np.mean([row["normalized_aulc"] for row in contributions.values()])
                )
                final_mean = float(
                    np.mean([row["final_correct"] for row in contributions.values()])
                )
                if not math.isclose(
                    contribution_mean,
                    mean_accuracy,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ) or not math.isclose(
                    final_mean,
                    final_accuracy,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"{model}/spine-{spine_id}/{policy}: "
                        "problem decomposition mismatch"
                    )
                decomposition_checks += 1
                seed_metrics[policy] = {
                    "mean_accuracy": mean_accuracy,
                    "final_accuracy": final_accuracy,
                }
                for problem_id, row in contributions.items():
                    problem_rows.append(
                        {
                            "model": model,
                            "problem_id": problem_id,
                            "spine_id": spine_id,
                            "policy": policy,
                            **row,
                        }
                    )
            round_robin_mean = seed_metrics["round_robin"]["mean_accuracy"]
            for policy, metrics in seed_metrics.items():
                seed_rows.append(
                    {
                        "model": model,
                        "spine_id": spine_id,
                        "policy": policy,
                        "problems": len(model_problems),
                        "initial_frontiers": len(initial),
                        "aulc_start_budget": start_budget,
                        "common_endpoint": endpoint,
                        "aulc_span": span,
                        **metrics,
                        "relative_gain_vs_round_robin_percent": (
                            100.0
                            * (metrics["mean_accuracy"] - round_robin_mean)
                            / round_robin_mean
                        ),
                    }
                )
        grid = np.linspace(0.0, 1.0, 201)
        for policy in policy_specs:
            for normalized_budget in grid:
                accuracies = []
                for spine_id in spine_ids:
                    start_budget, endpoint = domains_by_seed[spine_id]
                    budget = start_budget + normalized_budget * (
                        endpoint - start_budget
                    )
                    accuracies.append(
                        _accuracy_at_budget(
                            curves_by_seed[spine_id][policy],
                            budget,
                        )
                    )
                curve_rows.append(
                    {
                        "model": model,
                        "policy": policy,
                        "normalized_budget": normalized_budget,
                        "mean_problem_accuracy": float(np.mean(accuracies)),
                        "min_seed_accuracy": min(accuracies),
                        "max_seed_accuracy": max(accuracies),
                    }
                )

    seed_csv = OUT / "spine_replicate_replay_seeds.csv"
    problem_csv = OUT / "spine_replicate_replay_problems.csv"
    curve_csv = OUT / "spine_replicate_replay_curves.csv"
    write_csv(seed_csv, seed_rows)
    write_csv(problem_csv, problem_rows)
    write_csv(curve_csv, curve_rows)

    problem_index: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in problem_rows:
        problem_index[(row["model"], row["problem_id"], row["policy"])].append(row)
    clustered: dict[
        tuple[str, str],
        dict[str, dict[str, float]],
    ] = defaultdict(dict)
    for (model, problem_id, policy), rows in problem_index.items():
        if len(rows) != 4 or {int(row["spine_id"]) for row in rows} != {
            0,
            1,
            2,
            3,
        }:
            raise ValueError(f"{model}/{problem_id}/{policy}: incomplete spine cluster")
        clustered[(model, problem_id)][policy] = {
            "mean_accuracy": float(
                np.mean([float(row["normalized_aulc"]) for row in rows])
            ),
            "final_accuracy": float(
                np.mean([float(row["final_correct"]) for row in rows])
            ),
        }
    summary_rows: list[dict[str, Any]] = []
    for model in RUNS:
        problem_ids = sorted(
            problem_id
            for candidate_model, problem_id in clustered
            if candidate_model == model
        )
        if len(problem_ids) != 372:
            raise ValueError(f"{model}: incomplete clustered problem rows")
        for policy in policy_specs:
            policy_values = np.asarray(
                [
                    clustered[(model, problem_id)][policy]["mean_accuracy"]
                    for problem_id in problem_ids
                ],
                dtype=float,
            )
            round_robin_values = np.asarray(
                [
                    clustered[(model, problem_id)]["round_robin"]["mean_accuracy"]
                    for problem_id in problem_ids
                ],
                dtype=float,
            )
            final_values = np.asarray(
                [
                    clustered[(model, problem_id)][policy]["final_accuracy"]
                    for problem_id in problem_ids
                ],
                dtype=float,
            )
            observed_mean = float(policy_values.mean())
            observed_round_robin = float(round_robin_values.mean())
            observed_gain = (
                100.0 * (observed_mean - observed_round_robin) / observed_round_robin
            )
            rng = np.random.default_rng(
                stable_seed(
                    BOOTSTRAP_SEED,
                    SPINE_REPLICATE_NAMESPACE,
                    model,
                    policy,
                )
            )
            boot_mean = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            boot_gain = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            boot_final = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            for replicate in range(BOOTSTRAP_REPLICATES):
                indexes = rng.integers(
                    0,
                    len(problem_ids),
                    size=len(problem_ids),
                )
                sampled_mean = float(policy_values[indexes].mean())
                sampled_round_robin = float(round_robin_values[indexes].mean())
                boot_mean[replicate] = sampled_mean
                boot_gain[replicate] = (
                    100.0 * (sampled_mean - sampled_round_robin) / sampled_round_robin
                )
                boot_final[replicate] = float(final_values[indexes].mean())
            per_seed = [
                row
                for row in seed_rows
                if row["model"] == model and row["policy"] == policy
            ]
            summary_rows.append(
                {
                    "model": model,
                    "policy": policy,
                    "problems": len(problem_ids),
                    "spine_replicates": 4,
                    "mean_problem_accuracy": observed_mean,
                    "mean_problem_accuracy_low": float(np.quantile(boot_mean, 0.025)),
                    "mean_problem_accuracy_high": float(np.quantile(boot_mean, 0.975)),
                    "relative_gain_vs_round_robin_percent": observed_gain,
                    "relative_gain_low": float(np.quantile(boot_gain, 0.025)),
                    "relative_gain_high": float(np.quantile(boot_gain, 0.975)),
                    "final_problem_accuracy": float(final_values.mean()),
                    "final_problem_accuracy_low": float(np.quantile(boot_final, 0.025)),
                    "final_problem_accuracy_high": float(
                        np.quantile(boot_final, 0.975)
                    ),
                    "min_seed_mean_accuracy": min(
                        float(row["mean_accuracy"]) for row in per_seed
                    ),
                    "max_seed_mean_accuracy": max(
                        float(row["mean_accuracy"]) for row in per_seed
                    ),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    summary_csv = OUT / "spine_replicate_replay_summary.csv"
    write_csv(summary_csv, summary_rows)

    labels = {
        "fixed_1024": "Fixed 1,024",
        "round_robin": "Round-robin",
        "trace": "TRACE-style",
        "agreement": "Agreement",
        "prm": "PRM",
        "anll": "ANLL",
        "entropy": "Entropy",
        "mixed_myopic": "Mixed myopic",
        "on_spine_one_step_hindsight": "On-spine hindsight",
        "off_spine_myopic": "Off-spine myopic",
    }
    displayed = [
        "fixed_1024",
        "round_robin",
        "trace",
        "prm",
        "anll",
        "entropy",
    ]
    table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Model & Policy & Mean normalized AULC & "
            r"Rel.\ normalized-AULC gain vs.\ RR (95\% CI) & "
            r"Final problem accuracy \\"
        ),
        r"\midrule",
    ]
    for model in RUNS:
        rows = {row["policy"]: row for row in summary_rows if row["model"] == model}
        for policy in displayed:
            row = rows[policy]
            table.append(
                "{} & {} & {:.3f} & {:+.2f}\\% "
                "[{:+.2f}, {:+.2f}] & {:.3f} \\\\".format(
                    MODEL_SHORT[model],
                    labels[policy],
                    row["mean_problem_accuracy"],
                    row["relative_gain_vs_round_robin_percent"],
                    row["relative_gain_low"],
                    row["relative_gain_high"],
                    row["final_problem_accuracy"],
                )
            )
        table.append(r"\addlinespace")
    table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Problem-level equal-ledger replay with the four "
                r"spine seeds treated as independent repetitions. Each "
                r"repetition allocates budget among 372 distinct problems, "
                r"never among duplicate spines of the same problem. Mean "
                r"normalized AULC averages the normalized area over the four "
                r"repetitions; it is not a fixed-budget accuracy. "
                r"Intervals are paired 2,000-replicate problem-cluster "
                r"bootstraps of per-problem area contributions under the "
                r"fixed full-universe replay order, retaining all four seeds. "
                r"Only non-label-aware baselines are included. The mixed, "
                r"on-spine, and off-spine opportunity diagnostics are reported "
                r"separately in Table~\ref{tab:opportunity-replay}; "
                r"zero-cost cross-spine agreement remains CSV-only, and fully "
                r"charged agreement is reported separately.}"
            ),
            r"\label{tab:spine-replicate-replay}",
            r"\end{table*}",
            "",
        ]
    )
    table_path = OUT / "spine_replicate_replay_table.tex"
    atomic_text(table_path, "\n".join(table))

    diagnostic_displayed = [
        "round_robin",
        "on_spine_one_step_hindsight",
        "mixed_myopic",
        "off_spine_myopic",
    ]
    diagnostic_table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Model & Label-aware ordering & Mean normalized AULC & "
            r"Gain vs.\ RR (95\% CI) & Final problem accuracy \\"
        ),
        r"\midrule",
    ]
    for model in RUNS:
        rows = {row["policy"]: row for row in summary_rows if row["model"] == model}
        for policy in diagnostic_displayed:
            row = rows[policy]
            if policy == "off_spine_myopic":
                diagnostic_table.append(r"\rowcolor{resultgreenbg}")
            diagnostic_table.append(
                "{} & {} & {:.3f} & {:+.2f}\\% "
                "[{:+.2f}, {:+.2f}] & {:.3f} \\\\".format(
                    MODEL_SHORT[model],
                    labels[policy],
                    row["mean_problem_accuracy"],
                    row["relative_gain_vs_round_robin_percent"],
                    row["relative_gain_low"],
                    row["relative_gain_high"],
                    row["final_problem_accuracy"],
                )
            )
        diagnostic_table.append(r"\addlinespace")
    diagnostic_table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Label-aware opportunity diagnostics under the "
                r"single-spine logical ledger. On-spine ordering directly "
                r"observes the realized successor and is pure hindsight; the "
                r"mixed score includes that successor among four labels; the "
                r"off-spine score uses only three independent continuation "
                r"labels. Pale green marks this leakage-free off-spine "
                r"diagnostic; these label-aware orderings are not deployable "
                r"controllers, and the color does not denote an additional "
                r"significance test. Intervals are paired 2,000-"
                r"replicate problem-cluster bootstraps retaining all four "
                r"spine repetitions and conditioning on the fixed "
                r"full-universe replay order; they are conditional "
                r"finite-benchmark intervals.}"
            ),
            r"\label{tab:opportunity-replay}",
            r"\end{table*}",
            "",
        ]
    )
    diagnostic_table_path = OUT / "opportunity_replay_table.tex"
    atomic_text(diagnostic_table_path, "\n".join(diagnostic_table))

    import matplotlib.pyplot as plt

    colors = {
        "fixed_1024": "#7f7f7f",
        "round_robin": "#111111",
        "trace": "#ff7f0e",
        "agreement": "#2ca02c",
        "prm": "#9467bd",
        "anll": "#8c564b",
        "entropy": "#e377c2",
        "mixed_myopic": "#17becf",
        "off_spine_myopic": "#1f77b4",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), sharey=True)
    for axis, model in zip(axes, RUNS, strict=True):
        for policy in displayed:
            selected = [
                row
                for row in curve_rows
                if row["model"] == model and row["policy"] == policy
            ]
            axis.plot(
                [row["normalized_budget"] for row in selected],
                [row["mean_problem_accuracy"] for row in selected],
                label=labels[policy],
                color=colors[policy],
                linewidth=1.1,
                drawstyle="steps-post",
            )
        axis.set_title(MODEL_SHORT[model])
        axis.set_xlabel("Normalized per-repetition ledger budget")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean problem exact-match accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    figure_path = OUT / "spine_replicate_replay.pdf"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    summary_by_key = {(row["model"], row["policy"]): row for row in summary_rows}
    macro_lines = [
        "% Auto-generated from sealed spine-replicate replay.",
    ]
    for model, prefix in (
        ("Qwen3-8B", "Qwen"),
        ("DeepSeek-R1-Distill-Qwen-7B", "DeepSeek"),
    ):
        for policy, suffix in (
            ("mixed_myopic", "ReplicateMixedGainWithCI"),
            ("off_spine_myopic", "ReplicateOffGainWithCI"),
        ):
            row = summary_by_key[(model, policy)]
            macro_lines.append(
                f"\\newcommand{{\\{prefix}{suffix}}}{{"
                f"{row['relative_gain_vs_round_robin_percent']:+.2f}\\% "
                f"[{row['relative_gain_low']:+.2f}, "
                f"{row['relative_gain_high']:+.2f}]}}"
            )
    macros_path = OUT / "spine_replicate_replay_macros.tex"
    atomic_text(macros_path, "\n".join(macro_lines) + "\n")

    manifest_path = OUT / "spine_replicate_replay_manifest.json"
    sealed = seal_record(
        {
            "schema_version": SPINE_REPLICATE_NAMESPACE,
            "analysis_status": "supplementary, label-aware, read-only",
            "estimand": (
                "mean problem-level AULC across four independent single-spine "
                "replay repetitions"
            ),
            "problems_per_model": 372,
            "spine_replicates": [0, 1, 2, 3],
            "simultaneous_frontiers_per_problem": 1,
            "bootstrap_unit": (
                "problem ID paired across policies and retaining all four "
                "spine seeds"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "core_replay_equivalence_checks": equivalence_checks,
            "problem_decomposition_checks": decomposition_checks,
            "problem_universe_sha256": problem_universe_sha256,
            "source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for model, payload in validated.items()
            },
        }
    )
    atomic_text(
        manifest_path,
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
    )
    return [
        seed_csv,
        problem_csv,
        curve_csv,
        summary_csv,
        table_path,
        diagnostic_table_path,
        figure_path,
        macros_path,
        manifest_path,
    ]


def _transformer_chunk_flops(
    architecture: dict[str, Any],
    *,
    new_tokens: int,
    past_tokens: int,
) -> float:
    """Forward FLOPs for a causal transformer chunk with a reusable KV cache."""

    if new_tokens < 0 or past_tokens < 0:
        raise ValueError("token counts must be nonnegative")
    hidden = int(architecture["hidden_size"])
    intermediate = int(architecture["intermediate_size"])
    layers = int(architecture["num_hidden_layers"])
    query_width = int(architecture["num_attention_heads"]) * int(
        architecture["head_dim"]
    )
    kv_width = int(architecture["num_key_value_heads"]) * int(architecture["head_dim"])
    projection_per_token = 2.0 * hidden * (query_width + 2 * kv_width + hidden)
    swiglu_per_token = 6.0 * hidden * intermediate
    causal_pairs = new_tokens * past_tokens + new_tokens * (new_tokens + 1) / 2.0
    attention = 4.0 * query_width * causal_pairs
    return layers * (new_tokens * (projection_per_token + swiglu_per_token) + attention)


def _decode_flops(
    architecture: dict[str, Any],
    *,
    output_tokens: int,
    past_tokens: int,
) -> float:
    transformer = _transformer_chunk_flops(
        architecture,
        new_tokens=output_tokens,
        past_tokens=past_tokens,
    )
    lm_head = (
        2.0
        * output_tokens
        * int(architecture["hidden_size"])
        * int(architecture["vocab_size"])
    )
    return transformer + lm_head


def _prm_prefill_flops(
    architecture: dict[str, Any],
    input_tokens: int,
) -> float:
    transformer = _transformer_chunk_flops(
        architecture,
        new_tokens=input_tokens,
        past_tokens=0,
    )
    reward_head = 4.0 * input_tokens * int(architecture["hidden_size"])
    return transformer + reward_head


class _TwoFeatureFit:
    """Sufficient statistics for a nonnegative, no-intercept two-term fit."""

    def __init__(self) -> None:
        self.count = 0
        self.s11 = 0.0
        self.s12 = 0.0
        self.s22 = 0.0
        self.t1 = 0.0
        self.t2 = 0.0
        self.yy = 0.0
        self.y = 0.0
        self.roles: dict[str, int] = defaultdict(int)

    def add(self, x1: float, x2: float, y: float, role: str) -> None:
        self.count += 1
        self.s11 += x1 * x1
        self.s12 += x1 * x2
        self.s22 += x2 * x2
        self.t1 += x1 * y
        self.t2 += x2 * y
        self.yy += y * y
        self.y += y
        self.roles[role] += 1

    def _sse(self, beta1: float, beta2: float) -> float:
        return (
            self.yy
            - 2.0 * beta1 * self.t1
            - 2.0 * beta2 * self.t2
            + beta1 * beta1 * self.s11
            + 2.0 * beta1 * beta2 * self.s12
            + beta2 * beta2 * self.s22
        )

    def solve(self) -> dict[str, Any]:
        if self.count < 2:
            raise ValueError("insufficient H20 events for calibration")
        candidates = [(0.0, 0.0)]
        matrix = np.asarray(
            [[self.s11, self.s12], [self.s12, self.s22]],
            dtype=float,
        )
        target = np.asarray([self.t1, self.t2], dtype=float)
        try:
            unconstrained = np.linalg.solve(matrix, target)
        except np.linalg.LinAlgError:
            unconstrained = np.linalg.lstsq(matrix, target, rcond=None)[0]
        if np.all(unconstrained >= 0):
            candidates.append((float(unconstrained[0]), float(unconstrained[1])))
        if self.s11 > 0:
            candidates.append((max(0.0, self.t1 / self.s11), 0.0))
        if self.s22 > 0:
            candidates.append((0.0, max(0.0, self.t2 / self.s22)))
        beta1, beta2 = min(candidates, key=lambda pair: self._sse(*pair))
        sse = max(0.0, self._sse(beta1, beta2))
        mean_y = self.y / self.count
        centered_total = max(0.0, self.yy - self.count * mean_y * mean_y)
        total_xx = self.s11 + 2.0 * self.s12 + self.s22
        total_xy = self.t1 + self.t2
        pooled_beta = max(0.0, total_xy / total_xx)
        pooled_sse = max(
            0.0,
            self.yy
            - 2.0 * pooled_beta * total_xy
            + pooled_beta * pooled_beta * total_xx,
        )
        return {
            "events": self.count,
            "seconds_per_total_pflop": pooled_beta,
            "pooled_sse": pooled_sse,
            "pooled_r_squared": (
                1.0 - pooled_sse / centered_total if centered_total > 0 else None
            ),
            "seconds_per_prefill_pflop": beta1,
            "seconds_per_decode_pflop": beta2,
            "sse": sse,
            "r_squared": (1.0 - sse / centered_total if centered_total > 0 else None),
            "roles": dict(sorted(self.roles.items())),
        }


class _OneFeatureFit:
    def __init__(self) -> None:
        self.count = 0
        self.x = 0.0
        self.xx = 0.0
        self.xy = 0.0
        self.yy = 0.0
        self.y = 0.0

    def add(self, x: float, y: float) -> None:
        self.count += 1
        self.x += x
        self.xx += x * x
        self.xy += x * y
        self.yy += y * y
        self.y += y

    def solve(self) -> dict[str, Any]:
        if self.count < 1 or self.xx <= 0:
            raise ValueError("insufficient PRM events for calibration")
        beta = max(0.0, self.xy / self.xx)
        sse = max(0.0, self.yy - 2.0 * beta * self.xy + beta * beta * self.xx)
        mean_y = self.y / self.count
        centered_total = max(0.0, self.yy - self.count * mean_y * mean_y)
        return {
            "events": self.count,
            "aggregate_pflops": self.x,
            "aggregate_seconds": self.y,
            "seconds_per_pflop_aggregate": self.y / self.x,
            "seconds_per_pflop": beta,
            "sse": sse,
            "r_squared": (1.0 - sse / centered_total if centered_total > 0 else None),
        }


def _physical_event_calibration(
    run_dir: Path,
    *,
    generator_architecture: dict[str, Any],
    prm_architecture: dict[str, Any],
    required_event_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    event_lookup: dict[str, dict[str, Any]] = {}
    generation_fit = _TwoFeatureFit()
    reasoning_fit = _OneFeatureFit()
    probe_fit = _OneFeatureFit()
    prm_fit = _OneFeatureFit()
    for path in sorted((run_dir / "problems").glob("*.json.gz")):
        artifact = read_json_gz(path)
        if not verify_sealed_record(artifact):
            raise ValueError(f"generation artifact seal failed: {path}")
        for event in artifact["ledger"]:
            event_id = str(event["event_id"])
            if event_id in required_event_ids:
                if event_id in event_lookup:
                    raise ValueError(f"duplicate required ledger event: {event_id}")
                event_lookup[event_id] = event
            seconds = float(event.get("gpu_seconds") or 0.0)
            input_tokens = int(event.get("input_tokens") or 0)
            output_tokens = int(event.get("output_tokens") or 0)
            if seconds <= 0 or output_tokens <= 0:
                continue
            prefill = (
                _transformer_chunk_flops(
                    generator_architecture,
                    new_tokens=input_tokens,
                    past_tokens=0,
                )
                / 1e15
            )
            decode = (
                _decode_flops(
                    generator_architecture,
                    output_tokens=output_tokens,
                    past_tokens=input_tokens,
                )
                / 1e15
            )
            generation_fit.add(
                prefill,
                decode,
                seconds,
                str(event["role"]),
            )
            role = str(event["role"])
            if role in {"spine_reasoning", "continuation_reasoning"}:
                reasoning_fit.add(prefill + decode, seconds)
            elif role in {"stop_probe", "continuation_probe"}:
                probe_fit.add(prefill + decode, seconds)
            else:
                raise ValueError(f"unexpected timed generator role: {role}")
    missing = required_event_ids - set(event_lookup)
    if missing:
        raise ValueError(f"{len(missing)} required generation events are absent")
    for path in sorted((run_dir / "scores").glob("*.json.gz")):
        artifact = read_json_gz(path)
        if not verify_sealed_record(artifact):
            raise ValueError(f"score artifact seal failed: {path}")
        for event in artifact["ledger"]:
            seconds = float(event.get("gpu_seconds") or 0.0)
            input_tokens = int(event.get("input_tokens") or 0)
            if seconds <= 0 or input_tokens <= 0:
                continue
            prm_fit.add(
                _prm_prefill_flops(
                    prm_architecture,
                    input_tokens,
                )
                / 1e15,
                seconds,
            )
    calibration = {
        "generator": generation_fit.solve(),
        "generator_reasoning": reasoning_fit.solve(),
        "generator_probe": probe_fit.solve(),
        "prm": prm_fit.solve(),
    }
    return event_lookup, calibration


def _events_for_state(
    state: dict[str, Any],
    field: str,
    event_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [event_lookup[str(reference["event_id"])] for reference in state[field]]


def _one_role(
    events: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    selected = [event for event in events if event["role"] == role]
    if len(selected) != 1:
        raise ValueError(f"expected one {role} event, found {len(selected)}")
    return selected[0]


def _state_flop_costs(
    states: list[dict[str, Any]],
    *,
    generator_architecture: dict[str, Any],
    prm_architecture: dict[str, Any],
    event_lookup: dict[str, dict[str, Any]],
    checkpoint_interval: int,
) -> tuple[dict[str, FlopCost], dict[str, FlopCost]]:
    initial: dict[str, FlopCost] = {}
    transitions: dict[str, FlopCost] = {}
    for state in states:
        checkpoint_id = str(state["checkpoint_id"])
        depth = int(state["depth"])
        prm_tokens = int((state.get("proxy_costs") or {}).get("prm_min_step", 0))
        prm_flops = (
            _prm_prefill_flops(prm_architecture, prm_tokens) if prm_tokens > 0 else 0.0
        )
        if depth == checkpoint_interval:
            events = _events_for_state(
                state,
                "initial_ledger_events",
                event_lookup,
            )
            prompt = _one_role(events, "prompt")
            probe = _one_role(events, "stop_probe")
            prompt_tokens = int(prompt["input_tokens"])
            probe_input = int(probe["input_tokens"])
            probe_output = int(probe["output_tokens"])
            cue_tokens = probe_input - prompt_tokens - depth
            if cue_tokens < 0:
                raise ValueError(f"negative initial probe cue: {checkpoint_id}")
            initial[checkpoint_id] = FlopCost(
                generator_reasoning=(
                    _transformer_chunk_flops(
                        generator_architecture,
                        new_tokens=prompt_tokens,
                        past_tokens=0,
                    )
                    + _decode_flops(
                        generator_architecture,
                        output_tokens=depth,
                        past_tokens=prompt_tokens,
                    )
                ),
                generator_probe=(
                    _transformer_chunk_flops(
                        generator_architecture,
                        new_tokens=cue_tokens,
                        past_tokens=prompt_tokens + depth,
                    )
                    + _decode_flops(
                        generator_architecture,
                        output_tokens=probe_output,
                        past_tokens=probe_input,
                    )
                ),
                prm_prefill=prm_flops,
            )
        if state["terminal"]:
            continue
        events = _events_for_state(
            state,
            "transition_ledger_events",
            event_lookup,
        )
        reasoning = _one_role(events, "continuation_reasoning")
        probe = _one_role(events, "stop_probe")
        reasoning_input = int(reasoning["input_tokens"])
        probe_input = int(probe["input_tokens"])
        probe_output = int(probe["output_tokens"])
        successor_depth = depth + checkpoint_interval
        prompt_tokens = reasoning_input - depth
        cue_tokens = probe_input - prompt_tokens - successor_depth
        if prompt_tokens <= 0 or cue_tokens < 0:
            raise ValueError(f"invalid transition lengths: {checkpoint_id}")
        transitions[checkpoint_id] = FlopCost(
            generator_reasoning=_decode_flops(
                generator_architecture,
                output_tokens=checkpoint_interval,
                past_tokens=reasoning_input,
            ),
            generator_probe=(
                _transformer_chunk_flops(
                    generator_architecture,
                    new_tokens=cue_tokens,
                    past_tokens=prompt_tokens + successor_depth,
                )
                + _decode_flops(
                    generator_architecture,
                    output_tokens=probe_output,
                    past_tokens=probe_input,
                )
            ),
            prm_prefill=prm_flops,
        )
    return initial, transitions


def _flop_cost_value(
    cost: FlopCost,
    ledger: str,
    calibration: dict[str, Any],
) -> float:
    if ledger == "estimated_pflops":
        return (
            cost.generator_reasoning + cost.generator_probe + cost.prm_prefill
        ) / 1e15
    if ledger == "calibrated_h20_seconds":
        reasoning = calibration["generator_reasoning"]
        probe = calibration["generator_probe"]
        prm = calibration["prm"]
        return (
            cost.generator_reasoning
            / 1e15
            * float(reasoning["seconds_per_pflop_aggregate"])
            + cost.generator_probe / 1e15 * float(probe["seconds_per_pflop_aggregate"])
            + cost.prm_prefill / 1e15 * float(prm["seconds_per_pflop_aggregate"])
        )
    raise ValueError(f"unsupported cost ledger: {ledger}")


def _replay_action_sequence_with_cost(
    states: list[dict[str, Any]],
    *,
    scheduler: str,
    action_checkpoint_ids: list[str],
    initial_costs: dict[str, FlopCost],
    transition_costs: dict[str, FlopCost],
    ledger: str,
    calibration: dict[str, Any],
    prm_architecture: dict[str, Any],
    checkpoint_interval: int,
) -> tuple[
    list[CostReplayPoint],
    dict[str, int],
    dict[str, list[tuple[float, int]]],
]:
    by_id = {str(state["checkpoint_id"]): state for state in states}
    first = [state for state in states if int(state["depth"]) == checkpoint_interval]
    frontiers = {str(state["problem_id"]): state for state in first}
    if len(frontiers) != len(first):
        raise ValueError("cost replay requires one frontier per problem")
    initial_correct = {
        problem_id: int(bool(state["stop_correct"]))
        for problem_id, state in frontiers.items()
    }
    change_events: dict[str, list[tuple[float, int]]] = {
        problem_id: [] for problem_id in frontiers
    }
    base_cost = FlopCost()
    for state in first:
        state_cost = initial_costs[str(state["checkpoint_id"])]
        base_cost += FlopCost(
            generator_reasoning=state_cost.generator_reasoning,
            generator_probe=state_cost.generator_probe,
        )
    budget = _flop_cost_value(base_cost, ledger, calibration)
    correct_count = sum(initial_correct.values())
    points = [
        CostReplayPoint(
            budget=budget,
            accuracy=correct_count / len(frontiers),
        )
    ]
    if scheduler == "prm":
        prm_cost = FlopCost(
            prm_prefill=sum(
                initial_costs[str(state["checkpoint_id"])].prm_prefill
                for state in first
            )
        )
        budget += _flop_cost_value(prm_cost, ledger, calibration)
        points.append(
            CostReplayPoint(
                budget=budget,
                accuracy=correct_count / len(frontiers),
            )
        )
    for checkpoint_id in action_checkpoint_ids:
        current = by_id[checkpoint_id]
        problem_id = str(current["problem_id"])
        observed = frontiers[problem_id]
        if str(observed["checkpoint_id"]) != checkpoint_id:
            raise ValueError("cost replay action order is not frontier-valid")
        successor_id = str(current.get("successor_id") or "")
        if successor_id not in by_id:
            raise ValueError(f"cost replay successor missing: {checkpoint_id}")
        successor = by_id[successor_id]
        cost = transition_costs[checkpoint_id]
        charged = FlopCost(
            generator_reasoning=cost.generator_reasoning,
            generator_probe=cost.generator_probe,
            prm_prefill=(
                successor.get("proxy_costs", {}).get("prm_min_step", 0)
                and _prm_prefill_flops(
                    prm_architecture,
                    int(successor["proxy_costs"]["prm_min_step"]),
                )
                if scheduler == "prm"
                else 0.0
            ),
        )
        budget += _flop_cost_value(charged, ledger, calibration)
        old_correct = int(bool(current["stop_correct"]))
        new_correct = int(bool(successor["stop_correct"]))
        correct_count += new_correct - old_correct
        frontiers[problem_id] = successor
        if old_correct != new_correct:
            change_events[problem_id].append((budget, new_correct))
        points.append(
            CostReplayPoint(
                budget=budget,
                accuracy=correct_count / len(frontiers),
            )
        )
    return points, initial_correct, change_events


def _step_auc_float(
    curve: list[CostReplayPoint],
    endpoint: float,
    *,
    start_budget: float,
) -> float:
    if endpoint <= start_budget:
        raise ValueError("cost AULC requires a positive domain")
    area = 0.0
    previous_budget = start_budget
    previous_accuracy = _accuracy_at_budget(curve, start_budget)
    for point in curve:
        if point.budget <= start_budget:
            previous_accuracy = point.accuracy
            continue
        current = min(endpoint, point.budget)
        if current > previous_budget:
            area += (current - previous_budget) * previous_accuracy
            previous_budget = current
        previous_accuracy = point.accuracy
        if point.budget >= endpoint:
            break
    if previous_budget < endpoint:
        area += (endpoint - previous_budget) * previous_accuracy
    return area


def _replay_action_sequence_with_budget_map(
    states: list[dict[str, Any]],
    *,
    action_checkpoint_ids: list[str],
    observation_budget: float,
    scheduler_ready_budget: float,
    transition_budgets: dict[str, float],
    checkpoint_interval: int,
) -> tuple[
    list[CostReplayPoint],
    dict[str, int],
    dict[str, list[tuple[float, int]]],
]:
    by_id = {str(state["checkpoint_id"]): state for state in states}
    first = [state for state in states if int(state["depth"]) == checkpoint_interval]
    frontiers = {str(state["problem_id"]): state for state in first}
    initial_correct = {
        problem_id: int(bool(state["stop_correct"]))
        for problem_id, state in frontiers.items()
    }
    changes: dict[str, list[tuple[float, int]]] = {
        problem_id: [] for problem_id in frontiers
    }
    correct_count = sum(initial_correct.values())
    budget = observation_budget
    points = [
        CostReplayPoint(
            budget=budget,
            accuracy=correct_count / len(frontiers),
        )
    ]
    if scheduler_ready_budget < observation_budget:
        raise ValueError("scheduler readiness precedes observation")
    if scheduler_ready_budget > observation_budget:
        budget = scheduler_ready_budget
        points.append(
            CostReplayPoint(
                budget=budget,
                accuracy=correct_count / len(frontiers),
            )
        )
    for checkpoint_id in action_checkpoint_ids:
        current = by_id[checkpoint_id]
        problem_id = str(current["problem_id"])
        if str(frontiers[problem_id]["checkpoint_id"]) != checkpoint_id:
            raise ValueError("budget-map replay action is not frontier-valid")
        successor_id = str(current.get("successor_id") or "")
        if successor_id not in by_id:
            raise ValueError(f"missing replay successor: {checkpoint_id}")
        successor = by_id[successor_id]
        budget += transition_budgets[checkpoint_id]
        old_correct = int(bool(current["stop_correct"]))
        new_correct = int(bool(successor["stop_correct"]))
        correct_count += new_correct - old_correct
        frontiers[problem_id] = successor
        if old_correct != new_correct:
            changes[problem_id].append((budget, new_correct))
        points.append(
            CostReplayPoint(
                budget=budget,
                accuracy=correct_count / len(frontiers),
            )
        )
    return points, initial_correct, changes


def _prompt_flops_for_initial_state(
    state: dict[str, Any],
    *,
    event_lookup: dict[str, dict[str, Any]],
    architecture: dict[str, Any],
) -> float:
    prompt = _one_role(
        _events_for_state(state, "initial_ledger_events", event_lookup),
        "prompt",
    )
    return _transformer_chunk_flops(
        architecture,
        new_tokens=int(prompt["input_tokens"]),
        past_tokens=0,
    )


def render_agreement_accounting(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    """Charge every auxiliary trajectory needed by cross-spine agreement."""

    architecture_bundle = read_json(ARCHITECTURES_PATH)
    architectures = architecture_bundle["models"]
    seed_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    for model, payload in validated.items():
        config = RunConfig.model_validate(
            read_json(payload["run_dir"] / "manifest.json")["config"]
        )
        states, _ = load_confirmation_frontiers(
            payload["run_dir"],
            payload["evaluation"],
            values_by_model[model],
        )
        required_event_ids = {
            str(reference["event_id"])
            for state in states
            for field in (
                "initial_ledger_events",
                "transition_ledger_events",
            )
            for reference in state[field]
        }
        event_lookup, calibration = _physical_event_calibration(
            payload["run_dir"],
            generator_architecture=architectures[model],
            prm_architecture=architectures["Qwen2.5-Math-PRM-7B"],
            required_event_ids=required_event_ids,
        )
        initial_flops, transition_flops = _state_flop_costs(
            states,
            generator_architecture=architectures[model],
            prm_architecture=architectures["Qwen2.5-Math-PRM-7B"],
            event_lookup=event_lookup,
            checkpoint_interval=config.protocol.checkpoint_interval,
        )
        state_index = {
            (
                str(state["problem_id"]),
                int(state["spine_id"]),
                int(state["depth"]),
            ): state
            for state in states
        }
        for spine_id in range(4):
            primary = [state for state in states if int(state["spine_id"]) == spine_id]
            rr_logical = _replay_one_with_problem_events(
                primary,
                "round_robin",
                checkpoint_interval=config.protocol.checkpoint_interval,
                fixed_allotment_tokens=config.execution.fixed_allotment_tokens,
            )
            free_logical = _replay_one_with_problem_events(
                primary,
                "agreement",
                checkpoint_interval=config.protocol.checkpoint_interval,
                fixed_allotment_tokens=config.execution.fixed_allotment_tokens,
            )
            if len(rr_logical[3]) != len(free_logical[3]):
                raise ValueError("agreement and round-robin action counts differ")
            first_primary = [
                state
                for state in primary
                if int(state["depth"]) == config.protocol.checkpoint_interval
            ]
            first_all = [
                state
                for state in states
                if int(state["depth"]) == config.protocol.checkpoint_interval
            ]
            logical_observation = float(_initial_base_cost(first_primary))
            logical_ready = float(_initial_base_cost(first_all))
            logical_parallel_transitions: dict[str, float] = {}
            for state in primary:
                if state["terminal"]:
                    continue
                problem_id = str(state["problem_id"])
                depth = int(state["depth"])
                logical_parallel_transitions[str(state["checkpoint_id"])] = float(
                    sum(
                        _transition_cost(state_index[(problem_id, auxiliary, depth)])
                        for auxiliary in range(4)
                    )
                )
            charged_logical = _replay_action_sequence_with_budget_map(
                primary,
                action_checkpoint_ids=free_logical[3],
                observation_budget=logical_observation,
                scheduler_ready_budget=logical_ready,
                transition_budgets=logical_parallel_transitions,
                checkpoint_interval=config.protocol.checkpoint_interval,
            )
            ledger_traces: dict[
                str,
                dict[
                    str,
                    tuple[
                        list[Any],
                        dict[str, int],
                        dict[str, list[tuple[float, int]]],
                    ],
                ],
            ] = {
                "logical_tokens": {
                    "round_robin_single": rr_logical[:3],
                    "cross_spine_free_diagnostic": free_logical[:3],
                    "parallel_four_spine_charged": charged_logical,
                }
            }
            for ledger in ("estimated_pflops", "calibrated_h20_seconds"):
                rr = _replay_action_sequence_with_cost(
                    primary,
                    scheduler="round_robin",
                    action_checkpoint_ids=rr_logical[3],
                    initial_costs=initial_flops,
                    transition_costs=transition_flops,
                    ledger=ledger,
                    calibration=calibration,
                    prm_architecture=architectures["Qwen2.5-Math-PRM-7B"],
                    checkpoint_interval=config.protocol.checkpoint_interval,
                )
                free = _replay_action_sequence_with_cost(
                    primary,
                    scheduler="agreement",
                    action_checkpoint_ids=free_logical[3],
                    initial_costs=initial_flops,
                    transition_costs=transition_flops,
                    ledger=ledger,
                    calibration=calibration,
                    prm_architecture=architectures["Qwen2.5-Math-PRM-7B"],
                    checkpoint_interval=config.protocol.checkpoint_interval,
                )
                primary_initial_cost = FlopCost()
                all_initial_cost = FlopCost()
                for state in first_primary:
                    primary_initial_cost += initial_flops[str(state["checkpoint_id"])]
                primary_initial_cost = FlopCost(
                    generator_reasoning=(primary_initial_cost.generator_reasoning),
                    generator_probe=primary_initial_cost.generator_probe,
                )
                by_problem_first: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for state in first_all:
                    by_problem_first[str(state["problem_id"])].append(state)
                    all_initial_cost += initial_flops[str(state["checkpoint_id"])]
                shared_prompt_adjustment = 0.0
                for problem_states in by_problem_first.values():
                    if len(problem_states) != 4:
                        raise ValueError("agreement initial spine set incomplete")
                    shared_prompt_adjustment += 3.0 * _prompt_flops_for_initial_state(
                        problem_states[0],
                        event_lookup=event_lookup,
                        architecture=architectures[model],
                    )
                all_initial_cost = FlopCost(
                    generator_reasoning=(
                        all_initial_cost.generator_reasoning - shared_prompt_adjustment
                    ),
                    generator_probe=all_initial_cost.generator_probe,
                )
                observation = _flop_cost_value(
                    primary_initial_cost,
                    ledger,
                    calibration,
                )
                ready = _flop_cost_value(
                    all_initial_cost,
                    ledger,
                    calibration,
                )
                parallel_transitions: dict[str, float] = {}
                for state in primary:
                    if state["terminal"]:
                        continue
                    problem_id = str(state["problem_id"])
                    depth = int(state["depth"])
                    cost = FlopCost()
                    for auxiliary in range(4):
                        auxiliary_state = state_index[(problem_id, auxiliary, depth)]
                        cost += transition_flops[str(auxiliary_state["checkpoint_id"])]
                    parallel_transitions[str(state["checkpoint_id"])] = (
                        _flop_cost_value(cost, ledger, calibration)
                    )
                charged = _replay_action_sequence_with_budget_map(
                    primary,
                    action_checkpoint_ids=free_logical[3],
                    observation_budget=observation,
                    scheduler_ready_budget=ready,
                    transition_budgets=parallel_transitions,
                    checkpoint_interval=config.protocol.checkpoint_interval,
                )
                ledger_traces[ledger] = {
                    "round_robin_single": rr,
                    "cross_spine_free_diagnostic": free,
                    "parallel_four_spine_charged": charged,
                }
            for ledger, traces in ledger_traces.items():
                start_budget = traces["round_robin_single"][0][0].budget
                endpoint = min(trace[0][-1].budget for trace in traces.values())
                span = endpoint - start_budget
                if span <= 0:
                    raise ValueError("agreement replay has empty common domain")
                for policy, (curve, initial_correct, changes) in traces.items():
                    mean_accuracy = (
                        _step_auc_float(
                            curve,
                            endpoint,
                            start_budget=start_budget,
                        )
                        / span
                    )
                    final_accuracy = _accuracy_at_budget(curve, endpoint)
                    contributions = _problem_contributions(
                        initial_correct,
                        changes,
                        start_budget=start_budget,
                        endpoint=endpoint,
                    )
                    seed_rows.append(
                        {
                            "model": model,
                            "spine_id": spine_id,
                            "ledger": ledger,
                            "policy": policy,
                            "start_budget": start_budget,
                            "common_endpoint": endpoint,
                            "mean_problem_accuracy": mean_accuracy,
                            "final_problem_accuracy": final_accuracy,
                        }
                    )
                    for problem_id, values in contributions.items():
                        problem_rows.append(
                            {
                                "model": model,
                                "spine_id": spine_id,
                                "ledger": ledger,
                                "policy": policy,
                                "problem_id": problem_id,
                                **values,
                            }
                        )
    grouped: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in problem_rows:
        grouped[
            (
                row["model"],
                row["ledger"],
                row["policy"],
                row["problem_id"],
            )
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for model in RUNS:
        for ledger in (
            "logical_tokens",
            "estimated_pflops",
            "calibrated_h20_seconds",
        ):
            problem_ids = sorted(
                {key[3] for key in grouped if key[0] == model and key[1] == ledger}
            )
            baseline = np.asarray(
                [
                    np.mean(
                        [
                            float(row["normalized_aulc"])
                            for row in grouped[
                                (
                                    model,
                                    ledger,
                                    "round_robin_single",
                                    problem_id,
                                )
                            ]
                        ]
                    )
                    for problem_id in problem_ids
                ]
            )
            for policy in (
                "round_robin_single",
                "cross_spine_free_diagnostic",
                "parallel_four_spine_charged",
            ):
                values = np.asarray(
                    [
                        np.mean(
                            [
                                float(row["normalized_aulc"])
                                for row in grouped[(model, ledger, policy, problem_id)]
                            ]
                        )
                        for problem_id in problem_ids
                    ]
                )
                gain = (
                    100.0
                    * (float(values.mean()) - float(baseline.mean()))
                    / float(baseline.mean())
                )
                rng = np.random.default_rng(
                    stable_seed(
                        BOOTSTRAP_SEED,
                        "flcva-agreement-accounting-v1",
                        model,
                        ledger,
                        policy,
                    )
                )
                bootstrap = np.empty(BOOTSTRAP_REPLICATES)
                for index in range(BOOTSTRAP_REPLICATES):
                    selected = rng.integers(
                        0,
                        len(problem_ids),
                        size=len(problem_ids),
                    )
                    bootstrap[index] = (
                        100.0
                        * (
                            float(values[selected].mean())
                            - float(baseline[selected].mean())
                        )
                        / float(baseline[selected].mean())
                    )
                summary_rows.append(
                    {
                        "model": model,
                        "ledger": ledger,
                        "policy": policy,
                        "problems": len(problem_ids),
                        "spine_replicates": 4,
                        "mean_problem_accuracy": float(values.mean()),
                        "relative_gain_vs_round_robin_percent": gain,
                        "relative_gain_low": float(np.quantile(bootstrap, 0.025)),
                        "relative_gain_high": float(np.quantile(bootstrap, 0.975)),
                    }
                )
    seed_path = OUT / "agreement_accounting_seeds.csv"
    problem_path = OUT / "agreement_accounting_problems.csv"
    summary_path = OUT / "agreement_accounting_summary.csv"
    write_csv(seed_path, seed_rows)
    write_csv(problem_path, problem_rows)
    write_csv(summary_path, summary_rows)
    table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lllrr}",
        r"\toprule",
        (
            r"Model & Ledger & Agreement treatment & Mean accuracy & "
            r"Gain vs.\ single-spine RR (95\% CI) \\"
        ),
        r"\midrule",
    ]
    ledger_labels = {
        "logical_tokens": "Logical tokens",
        "estimated_pflops": "Estimated FLOPs",
        "calibrated_h20_seconds": "Calibrated H20 seconds",
    }
    policy_labels = {
        "round_robin_single": "Single-spine round-robin",
        "cross_spine_free_diagnostic": "Free cross-spine diagnostic",
        "parallel_four_spine_charged": "Charged four-spine policy",
    }
    for model in RUNS:
        for ledger in ledger_labels:
            for policy in (
                "round_robin_single",
                "cross_spine_free_diagnostic",
                "parallel_four_spine_charged",
            ):
                row = next(
                    row
                    for row in summary_rows
                    if row["model"] == model
                    and row["ledger"] == ledger
                    and row["policy"] == policy
                )
                table.append(
                    "{} & {} & {} & {:.3f} & {:+.2f}\\% "
                    "[{:+.2f}, {:+.2f}] \\\\".format(
                        MODEL_SHORT[model],
                        ledger_labels[ledger],
                        policy_labels[policy],
                        row["mean_problem_accuracy"],
                        row["relative_gain_vs_round_robin_percent"],
                        row["relative_gain_low"],
                        row["relative_gain_high"],
                    )
                )
            table.append(r"\addlinespace")
    table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Agreement accounting repair. The free row reproduces "
                r"the original offline cross-spine diagnostic. The charged "
                r"policy pays the other three reasoning trajectories and "
                r"answer probes initially and whenever agreement is updated; "
                r"the problem's designated spine alone defines accuracy. "
                r"Prompts are deduplicated and their KV state is cloned. Each "
                r"ledger uses a common endpoint across the three rows.}"
            ),
            r"\label{tab:agreement-accounting}",
            r"\end{table*}",
            "",
        ]
    )
    table_path = OUT / "agreement_accounting_table.tex"
    atomic_text(table_path, "\n".join(table))
    policy_table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        (
            r"\begin{tabular}{@{}p{0.14\textwidth}p{0.20\textwidth}"
            r"p{0.15\textwidth}p{0.15\textwidth}p{0.30\textwidth}@{}}"
        ),
        r"\toprule",
        (
            r"Policy & Required observation & Extra generation & "
            r"Extra scoring & Replay charge \\"
        ),
        r"\midrule",
        (
            r"Round-robin / fixed & Current single spine & None & None & "
            r"Prompt, reasoning, answer probe \\"
        ),
        (
            r"TRACE / ANLL / entropy & Current and prior single-spine probes & "
            r"None & None & Same single-spine ledger \\"
        ),
        (
            r"PRM & Current single-spine prefix & None & PRM prefill & "
            r"Single-spine ledger plus every PRM call \\"
        ),
        (
            r"Cross-spine agreement & Four equal-depth spine answers & "
            r"Three synchronized spines & None & Four reasoning/probe paths \\"
        ),
        (
            r"Ridge / two-stage controllers & Single-spine proxies, depth, dataset & "
            r"None & Optional PRM variant & Observable costs plus CPU sidecar \\"
        ),
        (
            r"Mixed/off-spine/DP & Counterfactual correctness labels & "
            r"Stored branches/futures & Gold grading & Offline diagnostic only \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Policy observability and accounting. Cross-spine "
            r"agreement is deployable only in the explicitly charged parallel "
            r"variant; label-aware rows are never described as controllers.}"
        ),
        r"\label{tab:policy-accounting}",
        r"\end{table*}",
        "",
    ]
    policy_table_path = OUT / "policy_accounting_table.tex"
    atomic_text(policy_table_path, "\n".join(policy_table))
    manifest_path = OUT / "agreement_accounting_manifest.json"
    sealed = seal_record(
        {
            "schema_version": "flcva-agreement-accounting-v1",
            "analysis_status": "supplementary, read-only",
            "estimand": (
                "problem accuracy on one designated spine; cross-spine "
                "agreement updates synchronize and charge all four spines"
            ),
            "prompt_accounting": (
                "one logical prompt event and one reusable prompt KV state "
                "per problem"
            ),
            "spine_replicates": [0, 1, 2, 3],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for model, payload in validated.items()
            },
            "artifacts": [
                path.name
                for path in (
                    seed_path,
                    problem_path,
                    summary_path,
                    table_path,
                    policy_table_path,
                )
            ],
        }
    )
    atomic_text(
        manifest_path,
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
    )
    return [
        seed_path,
        problem_path,
        summary_path,
        table_path,
        policy_table_path,
        manifest_path,
    ]


def render_multi_ledger_replay(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    architecture_bundle = read_json(ARCHITECTURES_PATH)
    architectures = architecture_bundle["models"]
    policy_specs = {
        "fixed_1024": ("fixed_1024", None),
        "round_robin": ("round_robin", None),
        "trace": ("trace", None),
        "agreement": ("agreement", None),
        "prm": ("prm", None),
        "anll": ("anll", None),
        "entropy": ("entropy", None),
        "mixed_myopic": ("oracle", "mixed"),
        "off_spine_myopic": ("oracle", "off_spine"),
    }
    seed_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    calibrations: dict[str, Any] = {}
    for model, payload in validated.items():
        config = RunConfig.model_validate(
            read_json(payload["run_dir"] / "manifest.json")["config"]
        )
        states, _ = load_confirmation_frontiers(
            payload["run_dir"],
            payload["evaluation"],
            values_by_model[model],
        )
        required_event_ids = {
            str(reference["event_id"])
            for state in states
            for field in (
                "initial_ledger_events",
                "transition_ledger_events",
            )
            for reference in state[field]
        }
        event_lookup, calibration = _physical_event_calibration(
            payload["run_dir"],
            generator_architecture=architectures[model],
            prm_architecture=architectures["Qwen2.5-Math-PRM-7B"],
            required_event_ids=required_event_ids,
        )
        calibrations[model] = calibration
        initial_costs, transition_costs = _state_flop_costs(
            states,
            generator_architecture=architectures[model],
            prm_architecture=architectures["Qwen2.5-Math-PRM-7B"],
            event_lookup=event_lookup,
            checkpoint_interval=config.protocol.checkpoint_interval,
        )
        score_maps = oracle_score_maps(values_by_model[model])
        for spine_id in range(4):
            selected = [state for state in states if int(state["spine_id"]) == spine_id]
            logical_traces = {}
            for policy, (scheduler, score_name) in policy_specs.items():
                logical_traces[policy] = _replay_one_with_problem_events(
                    selected,
                    scheduler,
                    checkpoint_interval=config.protocol.checkpoint_interval,
                    fixed_allotment_tokens=(config.execution.fixed_allotment_tokens),
                    overridden_values=(
                        score_maps[str(score_name)] if score_name is not None else None
                    ),
                )
            for ledger in (
                "estimated_pflops",
                "calibrated_h20_seconds",
            ):
                traces: dict[
                    str,
                    tuple[
                        list[CostReplayPoint],
                        dict[str, int],
                        dict[str, list[tuple[float, int]]],
                    ],
                ] = {}
                for policy, (scheduler, _) in policy_specs.items():
                    traces[policy] = _replay_action_sequence_with_cost(
                        selected,
                        scheduler=scheduler,
                        action_checkpoint_ids=logical_traces[policy][3],
                        initial_costs=initial_costs,
                        transition_costs=transition_costs,
                        ledger=ledger,
                        calibration=calibration,
                        prm_architecture=architectures["Qwen2.5-Math-PRM-7B"],
                        checkpoint_interval=(config.protocol.checkpoint_interval),
                    )
                start_budget = traces["round_robin"][0][0].budget
                endpoint = min(trace[0][-1].budget for trace in traces.values())
                span = endpoint - start_budget
                for policy, (curve, initial_correct, changes) in traces.items():
                    mean_accuracy = (
                        _step_auc_float(
                            curve,
                            endpoint,
                            start_budget=start_budget,
                        )
                        / span
                    )
                    final_accuracy = _accuracy_at_budget(curve, endpoint)
                    contributions = _problem_contributions(
                        initial_correct,
                        changes,
                        start_budget=start_budget,
                        endpoint=endpoint,
                    )
                    if not math.isclose(
                        mean_accuracy,
                        float(
                            np.mean(
                                [
                                    row["normalized_aulc"]
                                    for row in contributions.values()
                                ]
                            )
                        ),
                        rel_tol=0.0,
                        abs_tol=1e-10,
                    ):
                        raise ValueError("multi-ledger decomposition mismatch")
                    seed_rows.append(
                        {
                            "model": model,
                            "spine_id": spine_id,
                            "ledger": ledger,
                            "policy": policy,
                            "start_budget": start_budget,
                            "common_endpoint": endpoint,
                            "mean_problem_accuracy": mean_accuracy,
                            "final_problem_accuracy": final_accuracy,
                        }
                    )
                    for problem_id, row in contributions.items():
                        problem_rows.append(
                            {
                                "model": model,
                                "spine_id": spine_id,
                                "ledger": ledger,
                                "policy": policy,
                                "problem_id": problem_id,
                                **row,
                            }
                        )
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in problem_rows:
        grouped[
            (
                row["model"],
                row["ledger"],
                row["policy"],
                row["problem_id"],
            )
        ].append(row)
    for model in RUNS:
        for ledger in ("estimated_pflops", "calibrated_h20_seconds"):
            problem_ids = sorted(
                {key[3] for key in grouped if key[0] == model and key[1] == ledger}
            )
            for policy in policy_specs:
                values = np.asarray(
                    [
                        np.mean(
                            [
                                float(row["normalized_aulc"])
                                for row in grouped[(model, ledger, policy, problem_id)]
                            ]
                        )
                        for problem_id in problem_ids
                    ],
                    dtype=float,
                )
                rr = np.asarray(
                    [
                        np.mean(
                            [
                                float(row["normalized_aulc"])
                                for row in grouped[
                                    (
                                        model,
                                        ledger,
                                        "round_robin",
                                        problem_id,
                                    )
                                ]
                            ]
                        )
                        for problem_id in problem_ids
                    ],
                    dtype=float,
                )
                observed = float(values.mean())
                rr_mean = float(rr.mean())
                gain = 100.0 * (observed - rr_mean) / rr_mean
                rng = np.random.default_rng(
                    stable_seed(
                        BOOTSTRAP_SEED,
                        MULTI_LEDGER_NAMESPACE,
                        model,
                        ledger,
                        policy,
                    )
                )
                boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
                for index in range(BOOTSTRAP_REPLICATES):
                    sample = rng.integers(
                        0,
                        len(problem_ids),
                        size=len(problem_ids),
                    )
                    boot[index] = (
                        100.0
                        * (float(values[sample].mean()) - float(rr[sample].mean()))
                        / float(rr[sample].mean())
                    )
                summary_rows.append(
                    {
                        "model": model,
                        "ledger": ledger,
                        "policy": policy,
                        "problems": len(problem_ids),
                        "spine_replicates": 4,
                        "mean_problem_accuracy": observed,
                        "relative_gain_vs_round_robin_percent": gain,
                        "relative_gain_low": float(np.quantile(boot, 0.025)),
                        "relative_gain_high": float(np.quantile(boot, 0.975)),
                    }
                )
    seed_path = OUT / "multi_ledger_replay_seeds.csv"
    problem_path = OUT / "multi_ledger_replay_problems.csv"
    summary_path = OUT / "multi_ledger_replay_summary.csv"
    calibration_path = OUT / "multi_ledger_calibration.json"
    write_csv(seed_path, seed_rows)
    write_csv(problem_path, problem_rows)
    write_csv(summary_path, summary_rows)
    atomic_text(
        calibration_path,
        json.dumps(calibrations, indent=2, sort_keys=True) + "\n",
    )
    logical = {
        (row["model"], row["policy"]): row
        for row in csv.DictReader(
            (OUT / "spine_replicate_replay_summary.csv").open(encoding="utf-8")
        )
    }
    physical = {
        (row["model"], row["ledger"], row["policy"]): row for row in summary_rows
    }
    table_rows: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    for model in RUNS:
        for ledger, label in (
            ("logical_tokens", "Logical tokens"),
            ("estimated_pflops", "Estimated FLOPs"),
            ("calibrated_h20_seconds", "Calibrated H20 seconds"),
        ):
            if ledger == "logical_tokens":
                rr = logical[(model, "round_robin")]
                prm = logical[(model, "prm")]
            else:
                rr = physical[(model, ledger, "round_robin")]
                prm = physical[(model, ledger, "prm")]
            table_rows.append((model, label, rr, prm))
    table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Model & Cost ledger & RR mean accuracy & PRM mean accuracy & "
            r"PRM gain vs.\ RR (95\% CI) \\"
        ),
        r"\midrule",
    ]
    for model, label, rr, prm in table_rows:
        if label == "Logical tokens":
            low = float(prm["relative_gain_low"])
            high = float(prm["relative_gain_high"])
            gain = float(prm["relative_gain_vs_round_robin_percent"])
        else:
            low = float(prm["relative_gain_low"])
            high = float(prm["relative_gain_high"])
            gain = float(prm["relative_gain_vs_round_robin_percent"])
        table.append(
            "{} & {} & {:.3f} & {:.3f} & {:+.2f}\\% "
            "[{:+.2f}, {:+.2f}] \\\\".format(
                MODEL_SHORT[model],
                label,
                float(rr["mean_problem_accuracy"]),
                float(prm["mean_problem_accuracy"]),
                gain,
                low,
                high,
            )
        )
        if label == "Calibrated H20 seconds":
            table.append(r"\addlinespace")
    table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Cost-ledger sensitivity of PRM scheduling. "
                r"Logical tokens are the primary accounting coordinate. "
                r"FLOPs assume reusable prefix KV states and count one "
                r"multiply--add as two operations. H20 seconds apply "
                r"separate aggregate reasoning, answer-probe, and PRM rates "
                r"measured from sealed H20 timings; per-event least-squares "
                r"and a two-term "
                r"prefill/decode fit is also "
                r"released as an identifiability diagnostic. They are "
                r"amortized compute "
                r"estimates, not online wall-clock latency. All ledgers retain "
                r"the same logical-ledger policy order and use a ledger-specific "
                r"common endpoint.}"
            ),
            r"\label{tab:multi-ledger-prm}",
            r"\end{table*}",
            "",
        ]
    )
    table_path = OUT / "multi_ledger_replay_table.tex"
    atomic_text(table_path, "\n".join(table))
    manifest_path = OUT / "multi_ledger_replay_manifest.json"
    sealed = seal_record(
        {
            "schema_version": MULTI_LEDGER_NAMESPACE,
            "analysis_status": "supplementary, read-only",
            "flop_convention": architecture_bundle["flop_convention"],
            "cost_semantics": {
                "logical_tokens": "primary ledger",
                "estimated_pflops": (
                    "online-efficient FLOPs with reusable prefix KV states"
                ),
                "calibrated_h20_seconds": (
                    "estimated FLOPs multiplied by separate aggregate "
                    "reasoning, answer-probe, and PRM seconds-per-FLOP rates "
                    "from sealed request-amortized H20 events"
                ),
            },
            "policy_order": (
                "inherited from the logical-token replay; ledgers "
                "change cost coordinates, not allocation decisions"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for model, payload in validated.items()
            },
            "artifacts": [
                path.name
                for path in (
                    seed_path,
                    problem_path,
                    summary_path,
                    calibration_path,
                    table_path,
                )
            ],
        }
    )
    atomic_text(
        manifest_path,
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
    )
    return [
        seed_path,
        problem_path,
        summary_path,
        calibration_path,
        table_path,
        manifest_path,
    ]


def _ridge_design_fit(
    rows: list[dict[str, Any]],
    raw_features: list[str],
) -> dict[str, Any]:
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for feature in raw_features:
        values = [
            float(row["proxies"][feature])
            for row in rows
            if row["proxies"].get(feature) is not None
        ]
        if not values:
            raise ValueError(f"ridge feature is always missing: {feature}")
        medians[feature] = float(np.median(values))
        filled = np.asarray(
            [
                (
                    float(row["proxies"][feature])
                    if row["proxies"].get(feature) is not None
                    else medians[feature]
                )
                for row in rows
            ],
            dtype=float,
        )
        means[feature] = float(filled.mean())
        scales[feature] = float(filled.std()) or 1.0
    return {
        "raw_features": list(raw_features),
        "medians": medians,
        "means": means,
        "scales": scales,
    }


def _ridge_design_matrix(
    rows: list[dict[str, Any]],
    preprocessing: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    raw_features = list(preprocessing["raw_features"])
    names = ["intercept"]
    columns = [np.ones(len(rows), dtype=float)]
    for feature in raw_features:
        missing = np.asarray(
            [row["proxies"].get(feature) is None for row in rows],
            dtype=float,
        )
        values = np.asarray(
            [
                (
                    float(row["proxies"][feature])
                    if row["proxies"].get(feature) is not None
                    else float(preprocessing["medians"][feature])
                )
                for row in rows
            ],
            dtype=float,
        )
        values = (values - float(preprocessing["means"][feature])) / float(
            preprocessing["scales"][feature]
        )
        names.extend([feature, f"{feature}__missing"])
        columns.extend([values, missing])
    names.extend(["depth_fraction", "dataset_math500"])
    columns.extend(
        [
            np.asarray(
                [float(row["depth"]) / 4096.0 for row in rows],
                dtype=float,
            ),
            np.asarray(
                [row["dataset_id"] == "math500" for row in rows],
                dtype=float,
            ),
        ]
    )
    return np.column_stack(columns), names


def _ridge_solve(
    design: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    matrix = design.T @ design + alpha * penalty
    vector = design.T @ target
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, vector, rcond=None)[0]


def _fit_ridge_controller(
    analysis: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    *,
    model: str,
    raw_features: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    problem_ids = sorted({str(row["problem_id"]) for row in analysis})
    tune_problems = {
        problem_id
        for problem_id in problem_ids
        if stable_seed(
            BOOTSTRAP_SEED,
            "flcva-ridge-controller-tune-v1",
            model,
            problem_id,
        )
        % 5
        == 0
    }
    fit_rows = [row for row in analysis if str(row["problem_id"]) not in tune_problems]
    tune_rows = [row for row in analysis if str(row["problem_id"]) in tune_problems]
    if not fit_rows or not tune_rows:
        raise ValueError("ridge analysis split is empty")
    preprocessing = _ridge_design_fit(fit_rows, raw_features)
    fit_design, feature_names = _ridge_design_matrix(
        fit_rows,
        preprocessing,
    )
    tune_design, tune_names = _ridge_design_matrix(
        tune_rows,
        preprocessing,
    )
    if feature_names != tune_names:
        raise ValueError("ridge feature order mismatch")
    fit_target = np.asarray(
        [float(row["marginal_value"]) for row in fit_rows],
        dtype=float,
    )
    tune_target = np.asarray(
        [float(row["marginal_value"]) for row in tune_rows],
        dtype=float,
    )
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    validation: list[dict[str, float]] = []
    for alpha in candidates:
        coefficients = _ridge_solve(fit_design, fit_target, alpha)
        prediction = tune_design @ coefficients
        validation.append(
            {
                "alpha": alpha,
                "mse": float(np.mean((prediction - tune_target) ** 2)),
            }
        )
    selected_alpha = min(
        validation,
        key=lambda row: (row["mse"], row["alpha"]),
    )["alpha"]
    final_preprocessing = _ridge_design_fit(analysis, raw_features)
    analysis_design, final_names = _ridge_design_matrix(
        analysis,
        final_preprocessing,
    )
    final_target = np.asarray(
        [float(row["marginal_value"]) for row in analysis],
        dtype=float,
    )
    coefficients = _ridge_solve(
        analysis_design,
        final_target,
        selected_alpha,
    )
    start = time.perf_counter()
    confirmation_design, confirmation_names = _ridge_design_matrix(
        confirmation,
        final_preprocessing,
    )
    predictions = confirmation_design @ coefficients
    inference_seconds = time.perf_counter() - start
    if final_names != confirmation_names:
        raise ValueError("ridge confirmation feature order mismatch")
    prediction_map = {
        str(row["checkpoint_id"]): float(prediction)
        for row, prediction in zip(
            confirmation,
            predictions,
            strict=True,
        )
    }
    truth = np.asarray(
        [float(row["marginal_value"]) for row in confirmation],
        dtype=float,
    )
    correlation = float(np.corrcoef(predictions, truth)[0, 1])
    beneficial_rows = [
        index for index, row in enumerate(confirmation) if int(row["stop_correct"]) == 0
    ]
    harmful_rows = [
        index for index, row in enumerate(confirmation) if int(row["stop_correct"]) == 1
    ]
    beneficial_auc = roc_auc(
        [int(confirmation[index]["beneficial"]) for index in beneficial_rows],
        [float(predictions[index]) for index in beneficial_rows],
    )
    harmful_auc = roc_auc(
        [int(confirmation[index]["harmful"]) for index in harmful_rows],
        [-float(predictions[index]) for index in harmful_rows],
    )
    metadata = {
        "schema_version": "flcva-ridge-controller-fit-v1",
        "raw_features": raw_features,
        "excluded_features": [
            "modal_agreement",
            "stop_correct",
            "difficulty",
            "continuation outcomes",
        ],
        "target": "four-branch token-normalized marginal_value",
        "analysis_problems": len(problem_ids),
        "fit_problems": len(problem_ids) - len(tune_problems),
        "tune_problems": len(tune_problems),
        "tune_problem_ids_sha256": hashlib.sha256(
            "\n".join(sorted(tune_problems)).encode()
        ).hexdigest(),
        "alpha_candidates": list(candidates),
        "validation": validation,
        "selected_alpha": selected_alpha,
        "preprocessing": final_preprocessing,
        "feature_names": final_names,
        "coefficients": {
            name: float(value)
            for name, value in zip(
                final_names,
                coefficients,
                strict=True,
            )
        },
        "confirmation_mse": float(np.mean((predictions - truth) ** 2)),
        "confirmation_pearson": correlation,
        "confirmation_beneficial_auroc": beneficial_auc,
        "confirmation_harmful_auroc": harmful_auc,
        "controller_inference_cpu_seconds": inference_seconds,
        "confirmation_predictions": len(predictions),
    }
    return prediction_map, metadata


def render_learned_controller(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    specifications = {
        "ridge_observable": [
            "persistence",
            "answer_confidence",
            "anll",
            "top20_entropy",
        ],
        "ridge_observable_prm": [
            "persistence",
            "answer_confidence",
            "anll",
            "top20_entropy",
            "prm_min_step",
        ],
    }
    fits: dict[str, Any] = {}
    problem_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    predictive_rows: list[dict[str, Any]] = []
    for model, payload in validated.items():
        all_rows = values_by_model[model]
        analysis = [row for row in all_rows if row["split"] == "analysis"]
        confirmation = [row for row in all_rows if row["split"] == "confirmation"]
        predictions: dict[str, dict[str, float]] = {}
        fits[model] = {}
        for policy, features in specifications.items():
            prediction_map, metadata = _fit_ridge_controller(
                analysis,
                confirmation,
                model=model,
                raw_features=features,
            )
            predictions[policy] = prediction_map
            fits[model][policy] = metadata
            predictive_rows.append(
                {
                    "model": model,
                    "policy": policy,
                    "selected_alpha": metadata["selected_alpha"],
                    "confirmation_mse": metadata["confirmation_mse"],
                    "confirmation_pearson": metadata["confirmation_pearson"],
                    "beneficial_auroc": metadata["confirmation_beneficial_auroc"],
                    "harmful_auroc": metadata["confirmation_harmful_auroc"],
                    "inference_cpu_seconds": metadata[
                        "controller_inference_cpu_seconds"
                    ],
                }
            )
        states, _ = load_confirmation_frontiers(
            payload["run_dir"],
            payload["evaluation"],
            all_rows,
        )
        config = RunConfig.model_validate(
            read_json(payload["run_dir"] / "manifest.json")["config"]
        )
        for spine_id in range(4):
            selected_states = [
                state for state in states if int(state["spine_id"]) == spine_id
            ]
            round_robin = _replay_one_with_problem_events(
                selected_states,
                "round_robin",
                checkpoint_interval=config.protocol.checkpoint_interval,
                fixed_allotment_tokens=config.execution.fixed_allotment_tokens,
            )
            for policy in specifications:
                controller = _replay_one_with_problem_events(
                    selected_states,
                    "oracle",
                    checkpoint_interval=config.protocol.checkpoint_interval,
                    fixed_allotment_tokens=(config.execution.fixed_allotment_tokens),
                    overridden_values=predictions[policy],
                    charge_prm=(policy == "ridge_observable_prm"),
                )
                start_budget = round_robin[0][0].budget
                endpoint = min(
                    round_robin[0][-1].budget,
                    controller[0][-1].budget,
                )
                span = endpoint - start_budget
                pair = {
                    "round_robin": round_robin[:3],
                    policy: controller[:3],
                }
                for candidate, (
                    curve,
                    initial_correct,
                    changes,
                ) in pair.items():
                    mean_accuracy = (
                        step_auc(
                            curve,
                            int(endpoint),
                            start_budget=int(start_budget),
                        )
                        / span
                    )
                    contributions = _problem_contributions(
                        initial_correct,
                        changes,
                        start_budget=start_budget,
                        endpoint=endpoint,
                    )
                    seed_rows.append(
                        {
                            "model": model,
                            "spine_id": spine_id,
                            "comparison": policy,
                            "policy": candidate,
                            "start_budget": start_budget,
                            "common_endpoint": endpoint,
                            "mean_problem_accuracy": mean_accuracy,
                            "final_problem_accuracy": _accuracy_at_budget(
                                curve,
                                endpoint,
                            ),
                        }
                    )
                    for problem_id, values in contributions.items():
                        problem_rows.append(
                            {
                                "model": model,
                                "spine_id": spine_id,
                                "comparison": policy,
                                "policy": candidate,
                                "problem_id": problem_id,
                                **values,
                            }
                        )
    grouped: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in problem_rows:
        grouped[
            (
                row["model"],
                row["comparison"],
                row["policy"],
                row["problem_id"],
            )
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for model in RUNS:
        for controller in specifications:
            problem_ids = sorted(
                {key[3] for key in grouped if key[0] == model and key[1] == controller}
            )
            baseline = np.asarray(
                [
                    np.mean(
                        [
                            float(row["normalized_aulc"])
                            for row in grouped[
                                (
                                    model,
                                    controller,
                                    "round_robin",
                                    problem_id,
                                )
                            ]
                        ]
                    )
                    for problem_id in problem_ids
                ]
            )
            values = np.asarray(
                [
                    np.mean(
                        [
                            float(row["normalized_aulc"])
                            for row in grouped[
                                (
                                    model,
                                    controller,
                                    controller,
                                    problem_id,
                                )
                            ]
                        ]
                    )
                    for problem_id in problem_ids
                ]
            )
            gain = (
                100.0
                * (float(values.mean()) - float(baseline.mean()))
                / float(baseline.mean())
            )
            rng = np.random.default_rng(
                stable_seed(
                    BOOTSTRAP_SEED,
                    "flcva-ridge-controller-replay-v1",
                    model,
                    controller,
                )
            )
            bootstrap = np.empty(BOOTSTRAP_REPLICATES)
            for index in range(BOOTSTRAP_REPLICATES):
                sample = rng.integers(
                    0,
                    len(problem_ids),
                    size=len(problem_ids),
                )
                bootstrap[index] = (
                    100.0
                    * (float(values[sample].mean()) - float(baseline[sample].mean()))
                    / float(baseline[sample].mean())
                )
            summary_rows.append(
                {
                    "model": model,
                    "policy": controller,
                    "problems": len(problem_ids),
                    "spine_replicates": 4,
                    "round_robin_mean_accuracy": float(baseline.mean()),
                    "controller_mean_accuracy": float(values.mean()),
                    "relative_gain_vs_round_robin_percent": gain,
                    "relative_gain_low": float(np.quantile(bootstrap, 0.025)),
                    "relative_gain_high": float(np.quantile(bootstrap, 0.975)),
                }
            )
    predictive_path = OUT / "learned_controller_predictive.csv"
    seed_path = OUT / "learned_controller_seeds.csv"
    problem_path = OUT / "learned_controller_problems.csv"
    summary_path = OUT / "learned_controller_summary.csv"
    fit_path = OUT / "learned_controller_fits.json"
    write_csv(predictive_path, predictive_rows)
    write_csv(seed_path, seed_rows)
    write_csv(problem_path, problem_rows)
    write_csv(summary_path, summary_rows)
    atomic_text(
        fit_path,
        json.dumps(fits, indent=2, sort_keys=True) + "\n",
    )
    table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Model & Controller & Pearson $r$ & Beneficial AUROC & "
            r"\shortstack{Harmful AUROC\\($-\widehat M_s$ score)} \\"
        ),
        r"\midrule",
    ]
    labels = {
        "ridge_observable": r"Ridge-$M$ (single-spine)",
        "ridge_observable_prm": r"Ridge-$M$ (+ charged PRM)",
    }
    for model in RUNS:
        for policy in specifications:
            predictive = next(
                row
                for row in predictive_rows
                if row["model"] == model and row["policy"] == policy
            )
            table.append(
                "{} & {} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                    MODEL_SHORT[model],
                    labels[policy],
                    predictive["confirmation_pearson"],
                    predictive["beneficial_auroc"],
                    predictive["harmful_auroc"],
                )
            )
    table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Ridge-$M$ predictive diagnostics. "
                r"Ridge regression "
                r"predicts continuous four-branch $M_s$ from analysis problems "
                r"only; a stable analysis-only tune split selects regularization "
                r"before refitting on all analysis data. Current correctness, "
                r"difficulty, agreement, and continuation outcomes are excluded. "
                r"Harmful AUROC uses damage-risk score $-\widehat M_s$, not "
                r"$+\widehat M_s$. "
                r"The PRM variant pays every score call. Confirmation problems "
                r"are held out from fitting and used only for reported prediction "
                r"and equal-ledger replay. Replay comparisons against both "
                r"round-robin and the correctness shortcut are reported in "
                r"Table~\ref{tab:controller-decomposition-replay}.}"
            ),
            r"\label{tab:learned-controller}",
            r"\end{table*}",
            "",
        ]
    )
    table_path = OUT / "learned_controller_table.tex"
    atomic_text(table_path, "\n".join(table))
    manifest_path = OUT / "learned_controller_manifest.json"
    sealed = seal_record(
        {
            "schema_version": "flcva-learned-controller-v1",
            "analysis_status": (
                "analysis-only fit with confirmation-held-out evaluation"
            ),
            "target": "continuous four-branch marginal_value",
            "gold_or_label_features": [],
            "controller_specs": specifications,
            "bootstrap_unit": "problem ID retaining all four spine repetitions",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for model, payload in validated.items()
            },
            "artifacts": [
                path.name
                for path in (
                    predictive_path,
                    seed_path,
                    problem_path,
                    summary_path,
                    fit_path,
                    table_path,
                )
            ],
        }
    )
    atomic_text(
        manifest_path,
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
    )
    return [
        predictive_path,
        seed_path,
        problem_path,
        summary_path,
        fit_path,
        table_path,
        manifest_path,
    ]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _binomial_objective(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    alpha: float,
) -> float:
    probability = np.clip(
        _sigmoid(design @ coefficients),
        1e-12,
        1.0 - 1e-12,
    )
    loss = -np.sum(
        weights
        * (target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
    )
    return float(loss + 0.5 * alpha * np.sum(coefficients[1:] ** 2))


def _logistic_solve(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Deterministic L2-regularized binomial fit with damped Newton steps."""

    if (
        design.ndim != 2
        or target.shape != (design.shape[0],)
        or weights.shape != target.shape
        or not np.all(weights > 0)
    ):
        raise ValueError("invalid logistic design")
    coefficients = np.zeros(design.shape[1], dtype=float)
    mean_target = float(np.average(target, weights=weights))
    mean_target = min(max(mean_target, 1e-6), 1.0 - 1e-6)
    coefficients[0] = math.log(mean_target / (1.0 - mean_target))
    penalty = np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0
    previous = _binomial_objective(
        design,
        target,
        weights,
        coefficients,
        alpha,
    )
    for _ in range(100):
        probability = _sigmoid(design @ coefficients)
        gradient = design.T @ (weights * (probability - target)) + alpha * (
            penalty @ coefficients
        )
        curvature = np.maximum(
            weights * probability * (1.0 - probability),
            1e-10,
        )
        hessian = design.T @ (design * curvature[:, None]) + alpha * penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        scale = 1.0
        accepted = False
        while scale >= 2.0**-20:
            candidate = coefficients - scale * step
            objective = _binomial_objective(
                design,
                target,
                weights,
                candidate,
                alpha,
            )
            if objective <= previous + 1e-10:
                coefficients = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
        if float(np.max(np.abs(scale * step))) < 1e-8:
            break
        if abs(previous - objective) < 1e-9:
            break
        previous = objective
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("non-finite logistic coefficients")
    return coefficients


def _component_rows_and_targets(
    rows: list[dict[str, Any]],
    component: str,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    if component == "stop_wrong":
        selected = list(rows)
        targets = np.asarray(
            [1.0 - float(row["stop_correct"]) for row in selected],
            dtype=float,
        )
        weights = np.ones(len(selected), dtype=float)
    elif component == "recovery_given_wrong":
        selected = [row for row in rows if int(row["stop_correct"]) == 0]
        targets = np.asarray(
            [float(np.mean(row["continuation_correct"])) for row in selected],
            dtype=float,
        )
        weights = np.full(len(selected), 4.0, dtype=float)
    elif component == "damage_given_correct":
        selected = [row for row in rows if int(row["stop_correct"]) == 1]
        targets = np.asarray(
            [1.0 - float(np.mean(row["continuation_correct"])) for row in selected],
            dtype=float,
        )
        weights = np.full(len(selected), 4.0, dtype=float)
    else:
        raise ValueError(f"unknown two-stage component: {component}")
    if not selected or not np.any(targets > 0) or not np.any(targets < 1):
        raise ValueError(f"degenerate two-stage component: {component}")
    return selected, targets, weights


def _fit_logistic_component(
    analysis: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    *,
    model: str,
    component: str,
    raw_features: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    problem_ids = sorted({str(row["problem_id"]) for row in analysis})
    tune_problems = {
        problem_id
        for problem_id in problem_ids
        if stable_seed(
            BOOTSTRAP_SEED,
            "flcva-two-stage-controller-tune-v1",
            model,
            problem_id,
        )
        % 5
        == 0
    }
    fit_all = [row for row in analysis if str(row["problem_id"]) not in tune_problems]
    tune_all = [row for row in analysis if str(row["problem_id"]) in tune_problems]
    fit_rows, fit_target, fit_weights = _component_rows_and_targets(
        fit_all,
        component,
    )
    tune_rows, tune_target, tune_weights = _component_rows_and_targets(
        tune_all,
        component,
    )
    preprocessing = _ridge_design_fit(fit_rows, raw_features)
    fit_design, feature_names = _ridge_design_matrix(
        fit_rows,
        preprocessing,
    )
    tune_design, tune_names = _ridge_design_matrix(
        tune_rows,
        preprocessing,
    )
    if feature_names != tune_names:
        raise ValueError("two-stage feature order mismatch")
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    validation: list[dict[str, float]] = []
    for alpha in candidates:
        coefficients = _logistic_solve(
            fit_design,
            fit_target,
            fit_weights,
            alpha,
        )
        probability = np.clip(
            _sigmoid(tune_design @ coefficients),
            1e-12,
            1.0 - 1e-12,
        )
        log_loss = -float(
            np.sum(
                tune_weights
                * (
                    tune_target * np.log(probability)
                    + (1.0 - tune_target) * np.log(1.0 - probability)
                )
            )
            / np.sum(tune_weights)
        )
        validation.append({"alpha": alpha, "log_loss": log_loss})
    selected_alpha = min(
        validation,
        key=lambda row: (row["log_loss"], row["alpha"]),
    )["alpha"]
    final_rows, final_target, final_weights = _component_rows_and_targets(
        analysis,
        component,
    )
    final_preprocessing = _ridge_design_fit(final_rows, raw_features)
    final_design, final_names = _ridge_design_matrix(
        final_rows,
        final_preprocessing,
    )
    coefficients = _logistic_solve(
        final_design,
        final_target,
        final_weights,
        selected_alpha,
    )
    confirmation_design, confirmation_names = _ridge_design_matrix(
        confirmation,
        final_preprocessing,
    )
    if final_names != confirmation_names:
        raise ValueError("two-stage confirmation feature order mismatch")
    start = time.perf_counter()
    predictions = _sigmoid(confirmation_design @ coefficients)
    inference_seconds = time.perf_counter() - start
    prediction_map = {
        str(row["checkpoint_id"]): float(prediction)
        for row, prediction in zip(
            confirmation,
            predictions,
            strict=True,
        )
    }
    metadata = {
        "schema_version": "flcva-two-stage-logistic-fit-v1",
        "component": component,
        "raw_features": list(raw_features),
        "excluded_features": [
            "modal_agreement",
            "stop_correct at inference",
            "difficulty",
            "continuation outcomes at inference",
        ],
        "analysis_problems": len(problem_ids),
        "fit_problems": len(problem_ids) - len(tune_problems),
        "tune_problems": len(tune_problems),
        "fit_checkpoints": len(fit_rows),
        "tune_checkpoints": len(tune_rows),
        "final_checkpoints": len(final_rows),
        "tune_problem_ids_sha256": hashlib.sha256(
            "\n".join(sorted(tune_problems)).encode()
        ).hexdigest(),
        "alpha_candidates": list(candidates),
        "validation": validation,
        "selected_alpha": selected_alpha,
        "preprocessing": final_preprocessing,
        "feature_names": final_names,
        "coefficients": {
            name: float(value)
            for name, value in zip(
                final_names,
                coefficients,
                strict=True,
            )
        },
        "controller_inference_cpu_seconds": inference_seconds,
        "confirmation_predictions": len(predictions),
    }
    return prediction_map, metadata


def _stratified_continuation_moments(
    analysis: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, float]], dict[str, Any]]:
    buckets: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {
            "recovery_successes": 0.0,
            "recovery_trials": 0.0,
            "damage_successes": 0.0,
            "damage_trials": 0.0,
            "cost_total": 0.0,
            "cost_branches": 0.0,
        }
    )
    for row in analysis:
        key = (str(row["dataset_id"]), int(row["depth"]))
        bucket = buckets[key]
        correct = tuple(int(value) for value in row["continuation_correct"])
        costs = tuple(float(value) for value in row["continuation_costs"])
        if len(correct) != 4 or len(costs) != 4:
            raise ValueError("two-stage moments require four complete branches")
        if int(row["stop_correct"]) == 0:
            bucket["recovery_successes"] += sum(correct)
            bucket["recovery_trials"] += len(correct)
        else:
            bucket["damage_successes"] += sum(1 - value for value in correct)
            bucket["damage_trials"] += len(correct)
        bucket["cost_total"] += sum(costs)
        bucket["cost_branches"] += len(costs)
    moments: dict[tuple[str, int], dict[str, float]] = {}
    serialized: dict[str, Any] = {}
    for key, bucket in sorted(buckets.items()):
        if (
            bucket["recovery_trials"] <= 0
            or bucket["damage_trials"] <= 0
            or bucket["cost_branches"] <= 0
        ):
            raise ValueError(f"incomplete two-stage stratum: {key}")
        values = {
            "recovery_rate": (bucket["recovery_successes"] + 0.5)
            / (bucket["recovery_trials"] + 1.0),
            "damage_rate": (bucket["damage_successes"] + 0.5)
            / (bucket["damage_trials"] + 1.0),
            "mean_branch_cost": (bucket["cost_total"] / bucket["cost_branches"]),
            **bucket,
        }
        moments[key] = values
        serialized[f"{key[0]}:{key[1]}"] = values
    return moments, serialized


def _bootstrap_relative_gain(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed_parts: tuple[str, ...],
) -> dict[str, float]:
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("invalid paired bootstrap arrays")
    observed = (
        100.0
        * (float(candidate.mean()) - float(baseline.mean()))
        / float(baseline.mean())
    )
    rng = np.random.default_rng(
        stable_seed(
            BOOTSTRAP_SEED,
            "flcva-controller-decomposition-bootstrap-v1",
            *seed_parts,
        )
    )
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(candidate), size=len(candidate))
        samples[index] = (
            100.0
            * (float(candidate[selected].mean()) - float(baseline[selected].mean()))
            / float(baseline[selected].mean())
        )
    return {
        "estimate": observed,
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
    }


def render_controller_decomposition(
    validated: dict[str, dict[str, Any]],
    values_by_model: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    """Separate current-correctness recognition from conditional value."""

    raw_features = [
        "persistence",
        "answer_confidence",
        "anll",
        "top20_entropy",
    ]
    fits: dict[str, Any] = {}
    predictive_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    compact_contexts: dict[str, list[CompactReplayContext]] = defaultdict(list)
    policy_order = (
        "round_robin",
        "wrong_probability_only",
        "correctness_stratified_ev",
        "two_stage_ev",
        "direct_ridge_m",
    )
    for model, payload in validated.items():
        all_rows = values_by_model[model]
        analysis = [row for row in all_rows if row["split"] == "analysis"]
        confirmation = [row for row in all_rows if row["split"] == "confirmation"]
        q_map, q_fit = _fit_logistic_component(
            analysis,
            confirmation,
            model=model,
            component="stop_wrong",
            raw_features=raw_features,
        )
        recovery_map, recovery_fit = _fit_logistic_component(
            analysis,
            confirmation,
            model=model,
            component="recovery_given_wrong",
            raw_features=raw_features,
        )
        damage_map, damage_fit = _fit_logistic_component(
            analysis,
            confirmation,
            model=model,
            component="damage_given_correct",
            raw_features=raw_features,
        )
        ridge_map, ridge_fit = _fit_ridge_controller(
            analysis,
            confirmation,
            model=model,
            raw_features=raw_features,
        )
        moments, serialized_moments = _stratified_continuation_moments(analysis)
        score_maps: dict[str, dict[str, float]] = {
            "wrong_probability_only": {},
            "correctness_stratified_ev": {},
            "two_stage_ev": {},
            "direct_ridge_m": ridge_map,
        }
        for row in confirmation:
            checkpoint_id = str(row["checkpoint_id"])
            q = q_map[checkpoint_id]
            recovery = recovery_map[checkpoint_id]
            damage = damage_map[checkpoint_id]
            stratum = moments[(str(row["dataset_id"]), int(row["depth"]))]
            cost = float(stratum["mean_branch_cost"])
            score_maps["wrong_probability_only"][checkpoint_id] = q
            score_maps["correctness_stratified_ev"][checkpoint_id] = (
                1000.0
                * (
                    q * float(stratum["recovery_rate"])
                    - (1.0 - q) * float(stratum["damage_rate"])
                )
                / cost
            )
            score_maps["two_stage_ev"][checkpoint_id] = (
                1000.0 * (q * recovery - (1.0 - q) * damage) / cost
            )
        wrong_labels = [1 - int(row["stop_correct"]) for row in confirmation]
        wrong_scores = [q_map[str(row["checkpoint_id"])] for row in confirmation]
        wrong_rows = [row for row in confirmation if int(row["stop_correct"]) == 0]
        correct_rows = [row for row in confirmation if int(row["stop_correct"]) == 1]
        recovery_auc = roc_auc(
            [int(row["beneficial"]) for row in wrong_rows],
            [recovery_map[str(row["checkpoint_id"])] for row in wrong_rows],
        )
        damage_auc = roc_auc(
            [int(row["harmful"]) for row in correct_rows],
            [damage_map[str(row["checkpoint_id"])] for row in correct_rows],
        )
        recovery_probabilities = np.asarray(
            [recovery_map[str(row["checkpoint_id"])] for row in wrong_rows],
            dtype=float,
        )
        recovery_frequencies = np.asarray(
            [float(np.mean(row["continuation_correct"])) for row in wrong_rows],
            dtype=float,
        )
        damage_probabilities = np.asarray(
            [damage_map[str(row["checkpoint_id"])] for row in correct_rows],
            dtype=float,
        )
        damage_frequencies = np.asarray(
            [1.0 - float(np.mean(row["continuation_correct"])) for row in correct_rows],
            dtype=float,
        )
        ridge_wrong = np.asarray(
            [ridge_map[str(row["checkpoint_id"])] for row in wrong_rows],
            dtype=float,
        )
        ridge_wrong_truth = np.asarray(
            [float(row["marginal_value"]) for row in wrong_rows],
            dtype=float,
        )
        ridge_correct = np.asarray(
            [ridge_map[str(row["checkpoint_id"])] for row in correct_rows],
            dtype=float,
        )
        ridge_correct_truth = np.asarray(
            [float(row["marginal_value"]) for row in correct_rows],
            dtype=float,
        )
        predictive_rows.append(
            {
                "model": model,
                "stop_wrong_auroc": roc_auc(wrong_labels, wrong_scores),
                "conditional_recovery_auroc": recovery_auc,
                "conditional_damage_auroc": damage_auc,
                "conditional_recovery_frequency_pearson": float(
                    np.corrcoef(
                        recovery_probabilities,
                        recovery_frequencies,
                    )[0, 1]
                ),
                "conditional_damage_frequency_pearson": float(
                    np.corrcoef(
                        damage_probabilities,
                        damage_frequencies,
                    )[0, 1]
                ),
                "conditional_recovery_branch_brier": float(
                    np.mean(
                        [
                            (recovery_map[str(row["checkpoint_id"])] - int(outcome))
                            ** 2
                            for row in wrong_rows
                            for outcome in row["continuation_correct"]
                        ]
                    )
                ),
                "conditional_damage_branch_brier": float(
                    np.mean(
                        [
                            (damage_map[str(row["checkpoint_id"])] - (1 - int(outcome)))
                            ** 2
                            for row in correct_rows
                            for outcome in row["continuation_correct"]
                        ]
                    )
                ),
                "ridge_conditional_pearson_wrong": float(
                    np.corrcoef(ridge_wrong, ridge_wrong_truth)[0, 1]
                ),
                "ridge_conditional_pearson_correct": float(
                    np.corrcoef(ridge_correct, ridge_correct_truth)[0, 1]
                ),
                "ridge_harmful_auroc_score_negative_m": (
                    ridge_fit["confirmation_harmful_auroc"]
                ),
            }
        )
        fits[model] = {
            "features": raw_features,
            "stop_wrong": q_fit,
            "recovery_given_wrong": recovery_fit,
            "damage_given_correct": damage_fit,
            "direct_ridge_m": ridge_fit,
            "stratified_analysis_moments": serialized_moments,
        }
        states, _ = load_confirmation_frontiers(
            payload["run_dir"],
            payload["evaluation"],
            all_rows,
        )
        config = RunConfig.model_validate(
            read_json(payload["run_dir"] / "manifest.json")["config"]
        )
        for spine_id in range(4):
            selected_states = [
                state for state in states if int(state["spine_id"]) == spine_id
            ]
            compact_contexts[model].append(
                _build_compact_replay_context(
                    selected_states,
                    {
                        policy: score_maps[policy]
                        for policy in (
                            "correctness_stratified_ev",
                            "two_stage_ev",
                            "direct_ridge_m",
                        )
                    },
                )
            )
            traces = {
                "round_robin": _replay_one_with_problem_events(
                    selected_states,
                    "round_robin",
                    checkpoint_interval=config.protocol.checkpoint_interval,
                    fixed_allotment_tokens=(config.execution.fixed_allotment_tokens),
                )
            }
            for policy in policy_order[1:]:
                traces[policy] = _replay_one_with_problem_events(
                    selected_states,
                    "oracle",
                    checkpoint_interval=config.protocol.checkpoint_interval,
                    fixed_allotment_tokens=(config.execution.fixed_allotment_tokens),
                    overridden_values=score_maps[policy],
                )
            start_budget = traces["round_robin"][0][0].budget
            endpoint = min(trace[0][-1].budget for trace in traces.values())
            span = endpoint - start_budget
            if span <= 0:
                raise ValueError("controller decomposition replay is empty")
            for normalized_budget in np.linspace(0.0, 1.0, 201):
                budget = start_budget + normalized_budget * span
                for policy in (
                    "round_robin",
                    "correctness_stratified_ev",
                    "two_stage_ev",
                    "direct_ridge_m",
                ):
                    curve_rows.append(
                        {
                            "model": model,
                            "spine_id": spine_id,
                            "policy": policy,
                            "normalized_budget": float(normalized_budget),
                            "problem_accuracy": _accuracy_at_budget(
                                traces[policy][0],
                                budget,
                            ),
                        }
                    )
            for policy in policy_order:
                curve, initial_correct, changes, _ = traces[policy]
                mean_accuracy = (
                    step_auc(
                        curve,
                        endpoint,
                        start_budget=start_budget,
                    )
                    / span
                )
                contributions = _problem_contributions(
                    initial_correct,
                    changes,
                    start_budget=start_budget,
                    endpoint=endpoint,
                )
                if not math.isclose(
                    mean_accuracy,
                    float(
                        np.mean(
                            [row["normalized_aulc"] for row in contributions.values()]
                        )
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError("controller decomposition contribution mismatch")
                seed_rows.append(
                    {
                        "model": model,
                        "spine_id": spine_id,
                        "policy": policy,
                        "start_budget": start_budget,
                        "common_endpoint": endpoint,
                        "mean_problem_accuracy": mean_accuracy,
                        "final_problem_accuracy": _accuracy_at_budget(
                            curve,
                            endpoint,
                        ),
                    }
                )
                for problem_id, contribution in contributions.items():
                    problem_rows.append(
                        {
                            "model": model,
                            "spine_id": spine_id,
                            "policy": policy,
                            "problem_id": problem_id,
                            **contribution,
                        }
                    )
    grouped: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in problem_rows:
        grouped[(row["model"], row["policy"], row["problem_id"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    model_vectors: dict[str, dict[str, np.ndarray]] = {}
    for model in RUNS:
        problem_ids = sorted(
            {
                problem_id
                for candidate_model, _, problem_id in grouped
                if candidate_model == model
            }
        )
        if len(problem_ids) != 372:
            raise ValueError("controller decomposition problem universe mismatch")
        vectors: dict[str, np.ndarray] = {}
        for policy in policy_order:
            vectors[policy] = np.asarray(
                [
                    float(
                        np.mean(
                            [
                                row["normalized_aulc"]
                                for row in grouped[(model, policy, problem_id)]
                            ]
                        )
                    )
                    for problem_id in problem_ids
                ],
                dtype=float,
            )
            if any(
                len(grouped[(model, policy, problem_id)]) != 4
                for problem_id in problem_ids
            ):
                raise ValueError("controller decomposition spine cluster mismatch")
        model_vectors[model] = vectors
        for policy in policy_order:
            versus_round_robin = _bootstrap_relative_gain(
                vectors[policy],
                vectors["round_robin"],
                seed_parts=(model, policy, "round_robin"),
            )
            versus_correctness = _bootstrap_relative_gain(
                vectors[policy],
                vectors["correctness_stratified_ev"],
                seed_parts=(model, policy, "correctness_stratified_ev"),
            )
            summary_rows.append(
                {
                    "model": model,
                    "policy": policy,
                    "problems": len(problem_ids),
                    "spine_replicates": 4,
                    "mean_problem_accuracy": float(vectors[policy].mean()),
                    "gain_vs_round_robin_percent": versus_round_robin["estimate"],
                    "gain_vs_round_robin_low": versus_round_robin["low"],
                    "gain_vs_round_robin_high": versus_round_robin["high"],
                    "incremental_vs_correctness_percent": (
                        versus_correctness["estimate"]
                    ),
                    "incremental_vs_correctness_low": (versus_correctness["low"]),
                    "incremental_vs_correctness_high": (versus_correctness["high"]),
                }
            )

    # The main hierarchy uses a full-system bootstrap rather than resampling
    # contributions from the already-fixed full-universe action order.  Every
    # replicate reconstructs the problem pool (including independent duplicate
    # copies), reruns all four global schedulers, and recomputes AULC.
    hierarchy_order = (
        "round_robin",
        "correctness_stratified_ev",
        "two_stage_ev",
        "direct_ridge_m",
    )
    hierarchy_rows: list[dict[str, Any]] = []
    full_bootstrap_rows: list[dict[str, Any]] = []
    full_bootstrap_metadata: dict[str, Any] = {}
    for model in RUNS:
        print(
            f"full-system replay bootstrap: {model} "
            f"({BOOTSTRAP_REPLICATES} replicates)",
            flush=True,
        )
        observed, bootstrap_samples, replicate_rows = _paired_full_replay_bootstrap(
            compact_contexts[model],
            policies=hierarchy_order,
            replicates=BOOTSTRAP_REPLICATES,
            seed_parts=(model, "controller-hierarchy"),
        )
        print(
            f"full-system replay bootstrap complete: {model}",
            flush=True,
        )
        for replicate_row in replicate_rows:
            full_bootstrap_rows.append({"model": model, **replicate_row})
        expected = {
            policy: float(model_vectors[model][policy].mean())
            for policy in hierarchy_order
        }
        for policy in hierarchy_order:
            if not math.isclose(
                observed[policy],
                expected[policy],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{model}/{policy}: compact full replay does not "
                    "reproduce the authoritative point estimate"
                )
        full_bootstrap_metadata[model] = {
            "problems_per_replicate": len(compact_contexts[model][0].problem_ids),
            "spine_repetitions": len(compact_contexts[model]),
            "replicates": BOOTSTRAP_REPLICATES,
            "unique_source_problems": {
                "min": min(
                    int(row["unique_source_problems"]) for row in replicate_rows
                ),
                "median": float(
                    np.median(
                        [int(row["unique_source_problems"]) for row in replicate_rows]
                    )
                ),
                "max": max(
                    int(row["unique_source_problems"]) for row in replicate_rows
                ),
            },
        }
        round_robin_samples = bootstrap_samples["round_robin"]
        shortcut_samples = bootstrap_samples["correctness_stratified_ev"]
        for policy in hierarchy_order:
            policy_samples = bootstrap_samples[policy]
            gain_samples = (
                100.0 * (policy_samples - round_robin_samples) / round_robin_samples
            )
            shortcut_increment_samples = (
                100.0 * (policy_samples - shortcut_samples) / shortcut_samples
            )
            gain_estimate = (
                100.0
                * (observed[policy] - observed["round_robin"])
                / observed["round_robin"]
            )
            shortcut_increment_estimate = (
                100.0
                * (observed[policy] - observed["correctness_stratified_ev"])
                / observed["correctness_stratified_ev"]
            )
            hierarchy_rows.append(
                {
                    "model": model,
                    "policy": policy,
                    "analysis_status": ("paired full-system problem bootstrap"),
                    "problems": len(compact_contexts[model][0].problem_ids),
                    "spine_replicates": 4,
                    "mean_normalized_aulc": observed[policy],
                    "gain_vs_round_robin_percent": gain_estimate,
                    "gain_vs_round_robin_low": float(np.quantile(gain_samples, 0.025)),
                    "gain_vs_round_robin_high": float(np.quantile(gain_samples, 0.975)),
                    "incremental_vs_shortcut_percent": (shortcut_increment_estimate),
                    "incremental_vs_shortcut_low": float(
                        np.quantile(
                            shortcut_increment_samples,
                            0.025,
                        )
                    ),
                    "incremental_vs_shortcut_high": float(
                        np.quantile(
                            shortcut_increment_samples,
                            0.975,
                        )
                    ),
                }
            )
    predictive_path = OUT / "controller_decomposition_predictive.csv"
    seed_path = OUT / "controller_decomposition_seeds.csv"
    problem_path = OUT / "controller_decomposition_problems.csv"
    summary_path = OUT / "controller_decomposition_summary.csv"
    hierarchy_path = OUT / "controller_hierarchy_summary.csv"
    full_bootstrap_path = OUT / "controller_full_replay_bootstrap.csv"
    curve_path = OUT / "controller_hierarchy_curves.csv"
    fits_path = OUT / "controller_decomposition_fits.json"
    write_csv(predictive_path, predictive_rows)
    write_csv(seed_path, seed_rows)
    write_csv(problem_path, problem_rows)
    write_csv(summary_path, summary_rows)
    write_csv(hierarchy_path, hierarchy_rows)
    write_csv(full_bootstrap_path, full_bootstrap_rows)
    write_csv(curve_path, curve_rows)
    atomic_text(
        fits_path,
        json.dumps(fits, indent=2, sort_keys=True) + "\n",
    )
    predictive_table = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        (r"Model & Current wrong AUROC & Recovery AUROC & " r"Damage AUROC \\"),
        r"\midrule",
    ]
    for row in predictive_rows:
        predictive_table.append(
            "{} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                MODEL_SHORT[row["model"]],
                row["stop_wrong_auroc"],
                row["conditional_recovery_auroc"],
                row["conditional_damage_auroc"],
            )
        )
    predictive_table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Held-out two-stage diagnostics. All three "
                r"probability models are fitted on analysis problems only. "
                r"Recovery and damage models use branch-frequency targets "
                r"within analysis checkpoints with $S_s=0$ and $S_s=1$, "
                r"respectively; AUROC is evaluated on the corresponding "
                r"four-draw checkpoint event. No true stop label is available "
                r"to the replay policy.}"
            ),
            r"\label{tab:controller-decomposition-predictive}",
            r"\end{table}",
            "",
        ]
    )
    predictive_table_path = OUT / "controller_decomposition_predictive_table.tex"
    atomic_text(predictive_table_path, "\n".join(predictive_table))
    replay_table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Model & Policy & Mean normalized AULC & "
            r"Gain vs.\ RR (95\% CI) & "
            r"Increment vs.\ shortcut (95\% CI) \\"
        ),
        r"\midrule",
    ]
    labels = {
        "round_robin": "Round-robin",
        "trace": "TRACE-style",
        "prm": "PRM (charged)",
        "anll": "ANLL",
        "entropy": "Entropy",
        "wrong_probability_only": r"$P(S_s=0\mid x)$ only",
        "correctness_stratified_ev": "Correctness shortcut",
        "two_stage_ev": "Two-stage conditional EV",
        "direct_ridge_m": r"Direct Ridge-$M$",
    }
    for model in RUNS:
        for policy in hierarchy_order:
            row = next(
                item
                for item in hierarchy_rows
                if item["model"] == model and item["policy"] == policy
            )
            gain = (
                f"{row['gain_vs_round_robin_percent']:+.2f}\\% "
                f"[{row['gain_vs_round_robin_low']:+.2f}, "
                f"{row['gain_vs_round_robin_high']:+.2f}]"
            )
            increment = (
                f"{row['incremental_vs_shortcut_percent']:+.2f}\\% "
                f"[{row['incremental_vs_shortcut_low']:+.2f}, "
                f"{row['incremental_vs_shortcut_high']:+.2f}]"
            )
            if policy == "correctness_stratified_ev":
                gain = rf"\keyresult{{{gain}}}"
            if policy in {"two_stage_ev", "direct_ridge_m"}:
                increment = rf"\keyresult{{{increment}}}"
            replay_table.append(
                "{} & {} & {:.3f} & {} & {} \\\\".format(
                    MODEL_SHORT[model],
                    labels[policy],
                    row["mean_normalized_aulc"],
                    gain,
                    increment,
                )
            )
        replay_table.append(r"\addlinespace")
    replay_table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Observable-controller hierarchy under the common "
                r"single-spine logical ledger. Round-robin tests whether "
                r"adaptation helps at all; the correctness shortcut predicts "
                r"only current wrongness and combines it with analysis-only "
                r"dataset--depth mean recovery, damage, and branch cost. The "
                r"final column tests whether a policy extracts "
                r"checkpoint-specific continuation information beyond that "
                r"shortcut. Pale green marks the shortcut gains and the "
                r"two continuation-specific increments discussed in the main "
                r"text; it is an expository aid, not an additional test. "
                r"Label-aware mixed and off-spine diagnostics are excluded. "
                r"Intervals are paired 2,000-replicate full-system problem "
                r"bootstraps: every replicate reconstructs the sampled pool, "
                r"treats duplicate draws as independent copies, reruns global "
                r"policy ordering over all four spine repetitions, and "
                r"recomputes the common endpoint and normalized AULC.}"
            ),
            r"\label{tab:controller-decomposition-replay}",
            r"\end{table*}",
            "",
        ]
    )
    replay_table_path = OUT / "controller_decomposition_replay_table.tex"
    atomic_text(replay_table_path, "\n".join(replay_table))

    import matplotlib.pyplot as plt

    figure_policies = (
        "round_robin",
        "correctness_stratified_ev",
        "two_stage_ev",
        "direct_ridge_m",
    )
    figure_labels = {
        "round_robin": "Round-robin",
        "correctness_stratified_ev": "Correctness shortcut",
        "two_stage_ev": "Two-stage EV",
        "direct_ridge_m": r"Direct Ridge-$M$",
    }
    figure_colors = {
        "round_robin": "#666666",
        "correctness_stratified_ev": "#0072B2",
        "two_stage_ev": "#E69F00",
        "direct_ridge_m": "#009E73",
    }
    figure_styles = {
        "round_robin": "--",
        "correctness_stratified_ev": "-",
        "two_stage_ev": "-.",
        "direct_ridge_m": "-",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.25), sharey=True)
    for axis, model in zip(axes, RUNS, strict=True):
        for policy in figure_policies:
            selected = [
                row
                for row in curve_rows
                if row["model"] == model and row["policy"] == policy
            ]
            by_budget: dict[float, list[float]] = defaultdict(list)
            for row in selected:
                by_budget[float(row["normalized_budget"])].append(
                    float(row["problem_accuracy"])
                )
            budgets = sorted(by_budget)
            means = [float(np.mean(by_budget[budget])) for budget in budgets]
            axis.plot(
                budgets,
                means,
                label=figure_labels[policy],
                color=figure_colors[policy],
                linestyle=figure_styles[policy],
                linewidth=(
                    1.9
                    if policy in {"correctness_stratified_ev", "direct_ridge_m"}
                    else 1.35
                ),
                drawstyle="steps-post",
            )
        ridge_row = next(
            row
            for row in hierarchy_rows
            if row["model"] == model and row["policy"] == "direct_ridge_m"
        )
        axis.text(
            0.035,
            0.055,
            (
                "Ridge vs. shortcut AULC\n"
                f"{ridge_row['incremental_vs_shortcut_percent']:+.2f}% "
                f"[{ridge_row['incremental_vs_shortcut_low']:+.2f}, "
                f"{ridge_row['incremental_vs_shortcut_high']:+.2f}]"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            color="#243746",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#C7D0D8",
                "linewidth": 0.7,
                "alpha": 0.92,
            },
        )
        axis.set_title(MODEL_SHORT[model], fontsize=10)
        axis.set_xlabel("Normalized logical-ledger budget")
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.20, linewidth=0.6)
    axes[0].set_ylabel("Mean problem exact-match accuracy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    hierarchy_figure_path = OUT / "controller_hierarchy_replay.pdf"
    fig.savefig(hierarchy_figure_path, bbox_inches="tight")
    plt.close(fig)

    manifest_path = OUT / "controller_decomposition_manifest.json"
    artifacts = (
        predictive_path,
        seed_path,
        problem_path,
        summary_path,
        hierarchy_path,
        full_bootstrap_path,
        curve_path,
        fits_path,
        predictive_table_path,
        replay_table_path,
        hierarchy_figure_path,
    )
    sealed = seal_record(
        {
            "schema_version": "flcva-controller-decomposition-v3",
            "analysis_status": (
                "analysis-only training and confirmation-held-out evaluation"
            ),
            "interpretive_rule": (
                "positive shortcut-relative gain is necessary evidence for "
                "continuation-specific control; an interval strictly above "
                "zero is stronger descriptive evidence"
            ),
            "confidence_interval_status": (
                "paired full-system problem bootstrap intervals; every "
                "replicate reconstructs the sampled pool, reruns policy "
                "ordering, and recomputes the endpoint and AULC"
            ),
            "conditional_fixed_order_intervals": (
                "retained only in appendix baseline and opportunity tables"
            ),
            "full_replay_bootstrap_namespace": (FULL_REPLAY_BOOTSTRAP_NAMESPACE),
            "full_replay_bootstrap": full_bootstrap_metadata,
            "controller_hierarchy_table_policies": list(hierarchy_order),
            "controller_hierarchy_figure_policies": list(figure_policies),
            "controller_hierarchy_curve_points": len(curve_rows),
            "features": raw_features,
            "confirmation_label_features": [],
            "analysis_only_training_targets": [
                "stop correctness",
                "four-branch recovery frequency conditional on stop wrong",
                "four-branch damage frequency conditional on stop correct",
            ],
            "correctness_baseline": (
                "predicted current wrongness combined with analysis-only "
                "dataset-depth recovery, damage, and branch-cost means"
            ),
            "two_stage_identity": ("E[C-S|x] = q(x) r(x) - (1-q(x)) d(x)"),
            "bootstrap_unit": "problem ID retaining all four spine repetitions",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for model, payload in validated.items()
            },
            "artifacts": [path.name for path in artifacts],
        }
    )
    atomic_text(
        manifest_path,
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
    )
    return [*artifacts, manifest_path]


def _audit_extract_boxed(text: str) -> str | None:
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return text[start:].strip() or None


def _audit_normalized_math_string(value: str) -> str:
    value = value.strip().replace("$", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\,", "").replace("\\!", "")
    return re.sub(r"\s+", "", value)


def _audit_math_expression(value: str) -> Any | None:
    try:
        from sympy.parsing.latex import parse_latex

        return parse_latex(value)
    except Exception:
        pass
    normalized = _audit_normalized_math_string(value)
    fraction = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = fraction.sub(r"(\1)/(\2)", normalized)
    normalized = (
        normalized.replace("^", "**")
        .replace("\\cdot", "*")
        .replace("\\times", "*")
        .replace("\\pi", "pi")
    )
    try:
        return sympy.sympify(normalized, evaluate=True)
    except Exception:
        return None


def _audit_recursive_equal(left: Any, right: Any) -> bool | None:
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            return False
        values = [
            _audit_recursive_equal(a, b) for a, b in zip(left, right, strict=True)
        ]
        if any(value is False for value in values):
            return False
        return True if all(value is True for value in values) else None
    if isinstance(left, set) and isinstance(right, set):
        return left == right
    try:
        difference = sympy.simplify(left - right)
        if difference == 0:
            return True
        equality = difference.equals(0)
        if equality is not None:
            return bool(equality)
    except Exception:
        pass
    try:
        equality = left.equals(right)
        if equality is not None:
            return bool(equality)
    except Exception:
        pass
    return None


def _secondary_answer_equivalence(
    dataset_id: str,
    prediction: str,
    gold: str,
) -> tuple[bool | None, str]:
    if dataset_id == "gsm8k":
        prediction_candidate = _audit_extract_boxed(prediction)
        if prediction_candidate is None:
            numbers = re.findall(
                r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/[+-]?\d+)?",
                prediction,
            )
            prediction_candidate = numbers[-1] if numbers else None
        gold_candidate = gold.rsplit("####", 1)[-1].strip()
        if prediction_candidate is None:
            return None, "gsm8k_no_candidate"
        prediction_numbers = re.findall(
            r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/[+-]?\d+)?",
            prediction_candidate,
        )
        gold_numbers = re.findall(
            r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/[+-]?\d+)?",
            gold_candidate,
        )
        if not prediction_numbers or not gold_numbers:
            return None, "gsm8k_no_numeric_value"
        try:
            left = Fraction(prediction_numbers[-1].replace(",", ""))
            right = Fraction(gold_numbers[-1].replace(",", ""))
            return left == right, "gsm8k_fraction"
        except (ValueError, ZeroDivisionError):
            return None, "gsm8k_fraction_failure"
    prediction_candidate = _audit_extract_boxed(prediction)
    if prediction_candidate is None:
        return None, "math_no_box"
    left = _audit_math_expression(prediction_candidate)
    right = _audit_math_expression(gold)
    if left is not None and right is not None:
        equal = _audit_recursive_equal(left, right)
        if equal is not None:
            return equal, "math_symbolic_equivalence"
    normalized_left = _audit_normalized_math_string(prediction_candidate)
    normalized_right = _audit_normalized_math_string(gold)
    if normalized_left and normalized_right:
        return (
            normalized_left == normalized_right,
            "math_independent_normalized_string",
        )
    return None, "math_unresolved"


def _transition_group(before: int, after: int) -> str:
    if before == 0 and after == 1:
        return "beneficial"
    if before == 1 and after == 0:
        return "harmful"
    if before == 0 and after == 0:
        return "zero_wrong"
    return "zero_correct"


def render_parser_adjudication(
    validated: dict[str, dict[str, Any]],
) -> list[Path]:
    label_path = ROOT / "data" / "sealed" / "combined_labels.jsonl"
    labels: dict[str, str] = {}
    with label_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            labels[str(row["id"])] = str(row["answer"])
    sample_cap = 200
    heaps: dict[
        tuple[str, str, str],
        list[tuple[int, str, dict[str, Any]]],
    ] = defaultdict(list)
    population: dict[tuple[str, str, str], int] = defaultdict(int)
    for model, payload in validated.items():
        confirmation_ids = {
            str(row["problem_id"])
            for row in iter_jsonl_gz(
                payload["run_dir"] / "evaluation" / "values.jsonl.gz"
            )
            if row["split"] == "confirmation"
        }
        for problem_id in sorted(confirmation_ids):
            relative = "problems/" + problem_id.replace(":", "_") + ".json.gz"
            path = payload["run_dir"] / "evaluation" / relative
            artifact = read_json_gz(path)
            checkpoint_map = {
                str(row["checkpoint_id"]): row for row in artifact["checkpoints"]
            }
            dataset_id = problem_id.split(":", 1)[0]
            for branch in artifact["branches"]:
                checkpoint = checkpoint_map[str(branch["checkpoint_id"])]
                before = int(checkpoint["stop_correct"])
                after = int(branch["correct"])
                group = _transition_group(before, after)
                key = (model, dataset_id, group)
                population[key] += 1
                sample_id = hashlib.sha256(
                    (
                        "flcva-parser-adjudication-v1|"
                        + model
                        + "|"
                        + str(branch["branch_id"])
                    ).encode()
                ).hexdigest()[:24]
                candidate = {
                    "sample_id": sample_id,
                    "model": model,
                    "dataset_id": dataset_id,
                    "problem_id": problem_id,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "branch_id": branch["branch_id"],
                    "spine_id": checkpoint["spine_id"],
                    "depth": checkpoint["depth"],
                    "branch_index": branch["branch_index"],
                    "primary_group": group,
                    "primary_before_correct": before,
                    "primary_after_correct": after,
                    "before_raw_answer": checkpoint["stop_raw_answer"],
                    "after_raw_answer": branch["raw_answer"],
                    "gold_answer": labels[problem_id],
                }
                score = stable_seed(
                    BOOTSTRAP_SEED,
                    "flcva-parser-adjudication-v1",
                    sample_id,
                )
                heap = heaps[key]
                item = (-score, sample_id, candidate)
                if len(heap) < sample_cap:
                    heapq.heappush(heap, item)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, item)
    selected = [item[2] for heap in heaps.values() for item in heap]
    selected.sort(
        key=lambda row: (
            row["model"],
            row["dataset_id"],
            row["primary_group"],
            row["sample_id"],
        )
    )
    audit_rows: list[dict[str, Any]] = []
    blinded_rows: list[dict[str, Any]] = []
    for row in selected:
        before, before_method = _secondary_answer_equivalence(
            str(row["dataset_id"]),
            str(row["before_raw_answer"]),
            str(row["gold_answer"]),
        )
        after, after_method = _secondary_answer_equivalence(
            str(row["dataset_id"]),
            str(row["after_raw_answer"]),
            str(row["gold_answer"]),
        )
        secondary_group = (
            _transition_group(int(before), int(after))
            if before is not None and after is not None
            else None
        )
        audit_rows.append(
            {
                **row,
                "secondary_before_correct": before,
                "secondary_after_correct": after,
                "secondary_before_method": before_method,
                "secondary_after_method": after_method,
                "secondary_group": secondary_group,
                "endpoint_labels_agree": (
                    int(before) == int(row["primary_before_correct"])
                    and int(after) == int(row["primary_after_correct"])
                    if before is not None and after is not None
                    else None
                ),
                "transition_group_agrees": (
                    secondary_group == row["primary_group"]
                    if secondary_group is not None
                    else None
                ),
            }
        )
        swap = (
            stable_seed(
                BOOTSTRAP_SEED,
                "flcva-parser-human-packet-v1",
                row["sample_id"],
            )
            % 2
        )
        answers = [
            str(row["before_raw_answer"]),
            str(row["after_raw_answer"]),
        ]
        blinded_rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset_id": row["dataset_id"],
                "gold_answer": row["gold_answer"],
                "answer_a": answers[swap],
                "answer_b": answers[1 - swap],
                "human_a_equivalent": "",
                "human_b_equivalent": "",
                "human_notes": "",
            }
        )
    summary_rows: list[dict[str, Any]] = []
    for group_index, key in enumerate(sorted(heaps)):
        model, dataset_id, group = key
        group_rows = [
            row
            for row in audit_rows
            if row["model"] == model
            and row["dataset_id"] == dataset_id
            and row["primary_group"] == group
        ]
        resolved = [
            row for row in group_rows if row["transition_group_agrees"] is not None
        ]
        agreement = (
            sum(bool(row["transition_group_agrees"]) for row in resolved)
            / len(resolved)
            if resolved
            else None
        )
        interval = cluster_ratio_interval(
            group_rows,
            lambda row: float(row["transition_group_agrees"] is True),
            lambda row: float(row["transition_group_agrees"] is not None),
            seed_offset=3100 + group_index,
        )
        if agreement is not None and not math.isclose(
            agreement,
            float(interval["estimate"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("parser agreement interval estimate mismatch")
        summary_rows.append(
            {
                "model": model,
                "dataset_id": dataset_id,
                "primary_group": group,
                "population_transitions": population[key],
                "sampled_transitions": len(group_rows),
                "secondary_resolved": len(resolved),
                "secondary_coverage": (
                    len(resolved) / len(group_rows) if group_rows else None
                ),
                "transition_label_agreement": agreement,
                "transition_label_agreement_low": interval["low"],
                "transition_label_agreement_high": interval["high"],
                "transition_disagreements": sum(
                    row["transition_group_agrees"] is False for row in resolved
                ),
                "bootstrap_replicates": interval["replicates"],
            }
        )
    audit_path = OUT / "parser_adjudication_samples.csv"
    blinded_path = OUT / "parser_human_review_packet.csv"
    summary_path = OUT / "parser_adjudication_summary.csv"
    write_csv(audit_path, audit_rows)
    write_csv(blinded_path, blinded_rows)
    write_csv(summary_path, summary_rows)
    table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lllrrrl}",
        r"\toprule",
        (
            r"Model & Data & Primary transition & Population & Sample & "
            r"Resolved & Independent agreement (95\% CI) \\"
        ),
        r"\midrule",
    ]
    for row in summary_rows:
        table.append(
            "{} & {} & {} & {:,} & {:,} & {:,} & "
            "{:.1f}\\% [{:.1f}, {:.1f}] \\\\".format(
                MODEL_SHORT[row["model"]],
                "GSM8K" if row["dataset_id"] == "gsm8k" else "MATH-500",
                row["primary_group"].replace("_", " "),
                row["population_transitions"],
                row["sampled_transitions"],
                row["secondary_resolved"],
                100.0 * float(row["transition_label_agreement"]),
                100.0 * float(row["transition_label_agreement_low"]),
                100.0 * float(row["transition_label_agreement_high"]),
            )
        )
    table.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Stratified secondary adjudication of transition "
                r"labels. Up to 200 confirmation transitions per "
                r"model--dataset--primary-label cell are selected by a stable "
                r"hash. The secondary evaluator independently extracts boxed "
                r"answers and applies rational or symbolic equivalence rather "
                r"than comparing stored canonical strings. ``Agreement'' is "
                r"transition-label precision within the primary-selected cell; "
                r"intervals are 2,000-replicate problem-cluster bootstraps. "
                r"The release also includes a condition-blinded two-answer "
                r"packet for optional dual-human review; no unperformed human "
                r"judgment is reported.}"
            ),
            r"\label{tab:parser-adjudication}",
            r"\end{table*}",
            "",
        ]
    )
    table_path = OUT / "parser_adjudication_table.tex"
    atomic_text(table_path, "\n".join(table))
    manifest_path = OUT / "parser_adjudication_manifest.json"
    sealed = seal_record(
        {
            "schema_version": "flcva-parser-adjudication-v1",
            "analysis_status": "supplementary, label-aware, read-only",
            "sample_cap_per_model_dataset_primary_group": sample_cap,
            "sampling": "smallest stable hashes",
            "secondary_evaluator": (
                "independent boxed extraction plus Fraction/SymPy equivalence; "
                "stored canonical answers are not inputs"
            ),
            "human_review_status": (
                "packet generated; no human labels claimed or used"
            ),
            "source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for model, payload in validated.items()
            },
            "artifacts": [
                path.name
                for path in (audit_path, blinded_path, summary_path, table_path)
            ],
        }
    )
    atomic_text(
        manifest_path,
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
    )
    return [
        audit_path,
        blinded_path,
        summary_path,
        table_path,
        manifest_path,
    ]


def aggregate_events(paths: Iterable[Path]) -> dict[str, Any]:
    totals = {
        "events": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "logical_tokens": 0,
        "gpu_seconds": 0.0,
        "peak_gpu_memory_bytes": 0,
    }
    event_ids: set[str] = set()
    for path in paths:
        artifact = read_json_gz(path)
        if not verify_sealed_record(artifact):
            raise ValueError(f"ledger artifact seal failed: {path}")
        for event in artifact["ledger"]:
            event_id = event["event_id"]
            if event_id in event_ids:
                raise ValueError(f"duplicate ledger event {event_id}")
            event_ids.add(event_id)
            totals["events"] += 1
            totals["input_tokens"] += int(event.get("input_tokens") or 0)
            totals["output_tokens"] += int(event.get("output_tokens") or 0)
            totals["logical_tokens"] += int(event.get("logical_tokens") or 0)
            totals["gpu_seconds"] += float(event.get("gpu_seconds") or 0.0)
            totals["peak_gpu_memory_bytes"] = max(
                totals["peak_gpu_memory_bytes"],
                int(event.get("peak_gpu_memory_bytes") or 0),
            )
    return totals


def render_physical_ledger(validated: dict[str, dict[str, Any]]) -> list[Path]:
    rows: list[dict[str, Any]] = []
    for model, payload in validated.items():
        run_dir = payload["run_dir"]
        generation = aggregate_events(sorted((run_dir / "problems").glob("*.json.gz")))
        scoring = aggregate_events(sorted((run_dir / "scores").glob("*.json.gz")))
        audit_totals = payload["audit"]["totals"]
        if generation["events"] != audit_totals["ledger_events"]:
            raise ValueError(f"{model} generation ledger count mismatch")
        if scoring["events"] != audit_totals["prm_ledger_events"]:
            raise ValueError(f"{model} PRM ledger count mismatch")
        merge = read_json(run_dir / "merge_summary.json")
        parent_rows = {
            str(row["run_id"]): row
            for row in read_json(run_dir / "manifest.json")["parent_runs"]
        }
        if len(merge["inputs"]) != len(parent_rows):
            raise ValueError(f"{model} merge input coverage mismatch")
        observed_parent_ids: set[str] = set()
        model_load_seconds = 0.0
        prm_load_seconds = 0.0
        model_load_peak = 0
        prm_load_peak = 0
        for input_path in merge["inputs"]:
            shard = Path(input_path)
            manifest = read_json(shard / "manifest.json")
            if not verify_sealed_record(manifest):
                raise ValueError(f"{model} parent manifest seal failed: {shard}")
            parent = parent_rows.get(str(manifest["run_id"]))
            if parent is None or manifest.get("run_record_sha256") != parent.get(
                "run_record_sha256"
            ):
                raise ValueError(f"{model} unbound physical-ledger parent: {shard}")
            observed_parent_ids.add(str(manifest["run_id"]))
            score_paths = sorted((shard / "scores").glob("*.json.gz"))
            if not score_paths:
                raise ValueError(f"{model} parent has no score artifacts: {shard}")
            score_payload = read_json_gz(score_paths[0])
            if not verify_sealed_record(score_payload):
                raise ValueError(f"{model} score artifact seal failed: {shard}")
            runtime_model = manifest["runtime_model"]
            runtime_scorer = score_payload["runtime_scorer"]
            expected_scorer = parent["runtime_scorer_identity"]
            for key, value in expected_scorer.items():
                if runtime_scorer.get(key) != value:
                    raise ValueError(f"{model} scorer identity mismatch: {shard}/{key}")
            model_load_seconds += runtime_model["load_seconds"]
            prm_load_seconds += runtime_scorer["load_seconds"]
            model_load_peak = max(
                model_load_peak,
                runtime_model["load_peak_gpu_memory_bytes"],
            )
            prm_load_peak = max(
                prm_load_peak,
                runtime_scorer["load_peak_gpu_memory_bytes"],
            )
        if observed_parent_ids != set(parent_rows):
            raise ValueError(f"{model} physical-ledger parents are incomplete")
        rows.append(
            {
                "model": model,
                "generation_logical_tokens": generation["logical_tokens"],
                "generation_physical_output_tokens": generation["output_tokens"],
                "generation_gpu_hours": generation["gpu_seconds"] / 3600,
                "generation_output_tokens_per_gpu_second": (
                    generation["output_tokens"] / generation["gpu_seconds"]
                ),
                "generation_peak_gib": max(
                    generation["peak_gpu_memory_bytes"], model_load_peak
                )
                / (1024**3),
                "model_load_minutes": model_load_seconds / 60,
                "prm_logical_tokens": scoring["logical_tokens"],
                "prm_gpu_hours": scoring["gpu_seconds"] / 3600,
                "prm_tokens_per_gpu_second": (
                    scoring["logical_tokens"] / scoring["gpu_seconds"]
                ),
                "prm_peak_gib": max(scoring["peak_gpu_memory_bytes"], prm_load_peak)
                / (1024**3),
                "prm_load_minutes": prm_load_seconds / 60,
                "total_observed_gpu_hours": (
                    generation["gpu_seconds"]
                    + scoring["gpu_seconds"]
                    + model_load_seconds
                    + prm_load_seconds
                )
                / 3600,
            }
        )
    csv_path = OUT / "physical_ledger.csv"
    write_csv(csv_path, rows)
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Model & Gen. output tokens & Gen. GPU-h & Gen. tok/s & "
            r"Gen. peak GiB & PRM GPU-h & Total observed GPU-h \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        tex.append(
            "{} & {:.2f}M & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\".format(
                MODEL_SHORT[row["model"]],
                row["generation_physical_output_tokens"] / 1_000_000,
                row["generation_gpu_hours"],
                row["generation_output_tokens_per_gpu_second"],
                row["generation_peak_gib"],
                row["prm_gpu_hours"],
                row["total_observed_gpu_hours"],
            )
        )
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Physical-compute sidecar aggregated over all 32 "
                r"formal workers. Total observed GPU-hours add request-level "
                r"generation and PRM time plus separately recorded model-load "
                r"time; logical replay budgets remain the primary ledger.}"
            ),
            r"\label{tab:physical-ledger}",
            r"\end{table*}",
            "",
        ]
    )
    tex_path = OUT / "physical_ledger_table.tex"
    atomic_text(tex_path, "\n".join(tex))
    return [csv_path, tex_path]


def write_manifest(
    validated: dict[str, dict[str, Any]],
    artifacts: list[Path],
) -> Path:
    unique_artifacts = sorted(set(artifacts), key=lambda path: path.name)
    inputs: list[str] = []
    for payload in validated.values():
        for relative in (
            "manifest.json",
            "audit_manifest.json",
            "evaluation/summary.json",
            "evaluation/values.jsonl.gz",
            "evaluation/frontiers.jsonl.gz",
            "evaluation/matched_cells.jsonl.gz",
            "replay/summary.json",
            "merge_summary.json",
        ):
            inputs.append(str((payload["run_dir"] / relative).relative_to(ROOT)))
    inputs.extend(
        str(path.relative_to(ROOT))
        for path in (
            OUT / "replay.csv",
            OUT / "replay_points.csv",
            ARCHITECTURES_PATH,
            ROOT / "data" / "sealed" / "combined_labels.jsonl",
        )
    )
    manifest = seal_record(
        {
            "schema_version": "flcva-paper-supplement-manifest-v2",
            "inputs": sorted(inputs),
            "analysis_status": "supplementary, label-aware, read-only",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "source_runs": [
                {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"]["result_merkle_root"],
                }
                for payload in validated.values()
            ],
            "artifacts": [path.name for path in unique_artifacts],
        }
    )
    manifest_path = OUT / "supplement_manifest.json"
    atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    validated = validate_inputs()
    values_by_model = {
        model: load_values(payload["run_dir"], payload["evaluation"])
        for model, payload in validated.items()
    }
    artifacts: list[Path] = []
    artifacts.extend(render_protocol_overview())
    artifacts.extend(render_rq1(values_by_model))
    artifacts.extend(render_proxy_intervals(validated, values_by_model))
    artifacts.extend(render_calibration(validated))
    artifacts.extend(render_collision(validated, values_by_model))
    artifacts.extend(render_collision_robustness(validated, values_by_model))
    artifacts.extend(render_persistence_sensitivity(validated, values_by_model))
    artifacts.extend(render_k_sensitivity(values_by_model))
    artifacts.extend(render_branch_estimands(values_by_model))
    artifacts.extend(render_value_ratio_sensitivity(validated, values_by_model))
    artifacts.extend(render_oracle_decomposition(validated, values_by_model))
    artifacts.extend(render_leave_one_branch_out(values_by_model))
    artifacts.extend(render_nonmyopic_dp(validated, values_by_model))
    artifacts.extend(render_replay_policy_assets())
    artifacts.extend(render_spine_replicate_replay(validated, values_by_model))
    artifacts.extend(render_agreement_accounting(validated, values_by_model))
    artifacts.extend(render_multi_ledger_replay(validated, values_by_model))
    artifacts.extend(render_learned_controller(validated, values_by_model))
    artifacts.extend(render_controller_decomposition(validated, values_by_model))
    artifacts.extend(render_parser_adjudication(validated))
    artifacts.extend(render_physical_ledger(validated))
    manifest = write_manifest(validated, artifacts)
    print(manifest.relative_to(ROOT))


if __name__ == "__main__":
    main()
