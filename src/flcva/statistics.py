from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(y_true) != len(scores) or not y_true:
        return None
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and scores[order[end]] == scores[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = average_rank
        index = end
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, y_true) if label == 1
    )
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(y_true) != len(scores) or not y_true:
        return None
    positives = sum(y_true)
    if positives == 0:
        return None
    thresholds: dict[float, list[int]] = defaultdict(list)
    for label, score in zip(y_true, scores):
        thresholds[float(score)].append(int(label))
    true_positives = 0
    predicted_positives = 0
    area = 0.0
    for score in sorted(thresholds, reverse=True):
        labels = thresholds[score]
        block_positives = sum(labels)
        true_positives += block_positives
        predicted_positives += len(labels)
        precision = true_positives / predicted_positives
        recall_increment = block_positives / positives
        area += recall_increment * precision
    return area


def brier_score(y_true: Sequence[int], probabilities: Sequence[float]) -> float | None:
    if len(y_true) != len(probabilities) or not y_true:
        return None
    return sum(
        (float(probability) - int(label)) ** 2
        for label, probability in zip(y_true, probabilities)
    ) / len(y_true)


def calibration_curve(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    bins: int = 10,
) -> list[dict[str, float | int]]:
    rows: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for label, probability in zip(y_true, probabilities):
        index = min(bins - 1, max(0, int(float(probability) * bins)))
        rows[index].append((int(label), float(probability)))
    output: list[dict[str, float | int]] = []
    for index, row in enumerate(rows):
        if not row:
            continue
        output.append(
            {
                "bin": index,
                "n": len(row),
                "mean_probability": sum(value for _, value in row) / len(row),
                "empirical_rate": sum(label for label, _ in row) / len(row),
            }
        )
    return output


@dataclass(frozen=True)
class TieBins:
    upper_bounds: tuple[float, ...]

    def assign(self, value: float) -> int:
        for index, bound in enumerate(self.upper_bounds):
            if value <= bound:
                return index
        return len(self.upper_bounds)


def fit_tie_bins(values: Sequence[float], max_bins: int = 10) -> TieBins:
    if not values:
        return TieBins(())
    distinct = sorted(set(float(value) for value in values))
    if len(distinct) <= max_bins:
        return TieBins(tuple(distinct[:-1]))
    sorted_values = sorted(float(value) for value in values)
    boundaries: list[float] = []
    for index in range(1, max_bins):
        position = math.ceil(index * len(sorted_values) / max_bins) - 1
        candidate = sorted_values[position]
        if not boundaries or candidate > boundaries[-1]:
            boundaries.append(candidate)
    if boundaries and boundaries[-1] == sorted_values[-1]:
        boundaries.pop()
    return TieBins(tuple(boundaries))


def jeffreys_bin_probabilities(
    values: Sequence[float],
    labels: Sequence[int],
    bins: TieBins,
) -> dict[int, float]:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for value, label in zip(values, labels):
        key = bins.assign(value)
        counts[key][0] += int(label)
        counts[key][1] += 1
    return {
        key: (positive + 0.5) / (total + 1.0)
        for key, (positive, total) in counts.items()
    }


def bootstrap_cluster_interval(
    rows: Sequence[dict],
    metric: Callable[[Sequence[dict]], float | None],
    *,
    replicates: int,
    seed: int,
    cluster_key: str = "problem_id",
) -> dict[str, float | int | None]:
    observed = metric(rows)
    clusters: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        clusters[str(row[cluster_key])].append(row)
    keys = sorted(clusters)
    if len(keys) < 2 or observed is None:
        return {
            "estimate": observed,
            "low": None,
            "high": None,
            "replicates": 0,
            "reason": "insufficient_clusters_or_undefined_metric",
        }
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sample: list[dict] = []
        for key in (rng.choice(keys) for _ in keys):
            sample.extend(clusters[key])
        value = metric(sample)
        if value is not None and math.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return {
            "estimate": observed,
            "low": None,
            "high": None,
            "replicates": 0,
            "reason": "all_bootstrap_metrics_undefined",
        }
    low, high = np.quantile(np.asarray(estimates), [0.025, 0.975]).tolist()
    return {
        "estimate": float(observed),
        "low": float(low),
        "high": float(high),
        "replicates": len(estimates),
        "reason": None,
    }


def assign_difficulty_tertiles(
    scores: dict[str, float],
) -> dict[str, str]:
    """Assign intact equal-score blocks near one-third boundaries."""
    groups: list[tuple[float, list[str]]] = []
    for score in sorted(set(scores.values())):
        groups.append((score, sorted(key for key, value in scores.items() if value == score)))
    total = len(scores)
    targets = (total / 3, 2 * total / 3)
    cumulative = 0
    cuts: list[int] = []
    for target in targets:
        candidates: list[tuple[float, int]] = []
        running = 0
        for index, (_, ids) in enumerate(groups, start=1):
            running += len(ids)
            candidates.append((abs(running - target), index))
        _, cut = min(candidates, key=lambda item: (item[0], item[1]))
        if cuts and cut < cuts[-1]:
            cut = cuts[-1]
        cuts.append(cut)
    output: dict[str, str] = {}
    for index, (_, ids) in enumerate(groups):
        label = "low" if index < cuts[0] else "middle" if index < cuts[1] else "high"
        for problem_id in ids:
            output[problem_id] = label
    return output
