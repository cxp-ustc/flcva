from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RunConfig
from .data import GoldLabel, load_gold_labels, load_public_problems
from .ids import payload_sha256


def load_attested_gold_labels(
    config: RunConfig,
    project_root: Path,
) -> tuple[dict[str, GoldLabel], dict[str, Any]]:
    if not config.dataset.labels_path:
        raise ValueError("attested evaluation requires a labels_path")
    path = config.resolve_path(project_root, config.dataset.labels_path)
    if not path.is_file():
        raise ValueError(f"sealed label file is missing: {path}")
    canonical_required = config.backend == "vllm"
    labels = load_gold_labels(config, project_root)
    public = {
        problem.id: problem.dataset_id
        for problem in load_public_problems(config, project_root)
    }
    if set(labels) != set(public):
        raise ValueError("gold-label and public problem universes differ")
    dataset_mismatches = {
        problem_id: {
            "public": public[problem_id],
            "label": labels[problem_id].dataset_id,
        }
        for problem_id in labels
        if labels[problem_id].dataset_id != public[problem_id]
    }
    if dataset_mismatches:
        raise ValueError(
            "gold-label dataset identities differ from public records: "
            f"{dict(list(dataset_mismatches.items())[:5])}"
        )
    attestation = {
        "schema_version": "flcva-gold-label-attestation-v1",
        "dataset_id": config.dataset.id,
        "canonical": canonical_required,
        "problem_count": len(labels),
        "problem_ids_sha256": payload_sha256(sorted(labels)),
    }
    return labels, attestation


def verify_gold_label_attestation(
    config: RunConfig,
    project_root: Path,
    recorded: dict[str, Any],
) -> bool:
    _, observed = load_attested_gold_labels(config, project_root)
    # Compare semantic fields when reading summaries from earlier releases.
    if any(recorded.get(key) != value for key, value in observed.items()):
        raise ValueError(
            "evaluation gold-label attestation differs from canonical labels"
        )
    return True
