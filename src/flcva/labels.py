from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import RunConfig
from .data import GoldLabel, load_gold_labels, load_public_problems
from .ids import payload_sha256
from .io import file_sha256


CANONICAL_SEALED_LABEL_SHA256 = {
    "gsm8k": "6bd6c93376e4cb5ac8caf88e6e6a918252c9e2288589cc2df89a35984f9a329f",
    "math500": "5292eb053ae0e7aa799a376a2b5f9c107dc81c3f8ca6e817bbaa1e0796911a4a",
    "combined": "06a9a319dabefbfb6c869f055376cc38f98b7c15285f4d8a051654630637f169",
}


def _manifest_label_sha256(dataset_id: str, project_root: Path) -> str:
    manifest = json.loads(
        (project_root / "data" / "frozen" / "MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    row = (
        manifest.get("combined", {})
        if dataset_id == "combined"
        else manifest.get("datasets", {}).get(dataset_id, {})
    )
    observed = row.get("sealed_labels_sha256")
    expected = CANONICAL_SEALED_LABEL_SHA256.get(dataset_id)
    if expected is None:
        raise ValueError(f"dataset has no canonical sealed-label digest: {dataset_id}")
    if observed != expected:
        raise ValueError(
            "frozen data manifest sealed-label digest mismatch: "
            f"dataset={dataset_id} expected={expected} actual={observed}"
        )
    return expected


def load_attested_gold_labels(
    config: RunConfig,
    project_root: Path,
) -> tuple[dict[str, GoldLabel], dict[str, Any]]:
    if not config.dataset.labels_path:
        raise ValueError("attested evaluation requires a labels_path")
    path = config.resolve_path(project_root, config.dataset.labels_path)
    if not path.is_file():
        raise ValueError(f"sealed label file is missing: {path}")
    observed_sha256 = file_sha256(path)
    canonical_required = config.backend == "vllm"
    canonical_sha256 = None
    if canonical_required:
        canonical_sha256 = _manifest_label_sha256(config.dataset.id, project_root)
        if observed_sha256 != canonical_sha256:
            raise ValueError(
                "sealed gold-label digest mismatch: "
                f"dataset={config.dataset.id} expected={canonical_sha256} "
                f"actual={observed_sha256}"
            )
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
        "sha256": observed_sha256,
        "canonical_sha256": canonical_sha256,
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
    if recorded != observed:
        raise ValueError(
            "evaluation gold-label attestation differs from canonical labels"
        )
    return True
