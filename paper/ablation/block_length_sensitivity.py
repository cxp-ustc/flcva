#!/usr/bin/env python3
"""Run a label-blind 128/256/512-token continuation-length audit.

The same stored 256-token branches anchor every condition.  The 128-token arm
truncates each branch, the 256-token arm keeps the stored branch, and the
512-token arm extends that branch by another 256 tokens.  All three arms rerun
the canonical answer operator together in matched batches, so answer-execution
drift is not confounded with continuation length.  Sealed formal answers remain
read-only parity controls.  Gold labels enter only in the local ``evaluate``
stage.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from flcva.backends.base import GenerationRequest
from flcva.backends.vllm import build_backend
from flcva.config import RunConfig
from flcva.ids import canonical_json_bytes, payload_sha256, stable_seed
from flcva.io import (
    BlobStore,
    file_sha256,
    read_json_gz,
    seal_record,
    verify_sealed_record,
)
from flcva.parsing import answers_equal
from flcva.statistics import roc_auc


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "paper" / "ablation" / "block_length_inputs"
GENERATED_DIR = ROOT / "paper" / "generated"
LABELS_PATH = ROOT / "data" / "sealed" / "combined_labels.jsonl"
DATA_MANIFEST_PATH = ROOT / "data" / "frozen" / "MANIFEST.json"
STUDY_ID = "flcva-block-length-sensitivity-v4"
SELECTION_NAMESPACE = "flcva-block-length-sensitivity-v1"
WORKERS = 8
LENGTHS = (128, 256, 512)
COMMON_DEPTHS = tuple(range(256, 3585, 256))
SPLIT_QUOTAS = {
    "analysis": {"gsm8k": 150, "math500": 50},
    "confirmation": {"gsm8k": 150, "math500": 50},
}
CANONICAL_CUE = r"Therefore, the final answer is \boxed{"
ANSWER_BATCH = 24
EXTENSION_BATCH = 42
ANSWER_MAX_TOKENS = 32
PROBE_TOP_LOGPROBS = 20
BOOTSTRAP_REPLICATES = 2_000
RANK_SUBSAMPLE_REPLICATES = 64
RIDGE_ALPHA = 10.0
MIN_256_BRANCH_CORRECTNESS_PARITY = 0.99
MIN_256_CHECKPOINT_SIGN_PARITY = 0.98
ADAPTIVE_FEATURES = ("depth", "dataset")
PROXIES = (
    "persistence",
    "answer_confidence",
    "modal_agreement",
    "prm_min_step",
    "anll",
    "top20_entropy",
)
RUNTIME_PACKAGE_NAMES = (
    "PyYAML",
    "antlr4-python3-runtime",
    "huggingface-hub",
    "numpy",
    "pydantic",
    "requests",
    "sympy",
    "tokenizers",
    "torch",
    "transformers",
    "urllib3",
    "vllm",
)
SNAPSHOT_IDENTITY_KEYS = (
    "expected_revision",
    "revision_evidence",
    "transport_requested_revision",
    "transport_resolution_mode",
    "config_sha256",
    "tokenizer_config_sha256",
)
TRANSPORT_PROVENANCE_KEYS = (
    "transport_attestation_sha256",
    "transport_metadata_sha256",
)
SOURCE_RUNS = {
    "qwen3": ROOT / "artifacts" / "primary-qwen3",
    "deepseek": ROOT / "artifacts" / "primary-deepseek",
}
CONFIGS = {
    "qwen3": ROOT / "configs" / "formal-qwen3.yaml",
    "deepseek": ROOT / "configs" / "formal-deepseek.yaml",
}
EXECUTION_FILES = (
    "paper/ablation/block_length_sensitivity.py",
    "src/flcva/backends/base.py",
    "src/flcva/backends/mock.py",
    "src/flcva/backends/vllm.py",
    "src/flcva/config.py",
    "src/flcva/ids.py",
    "src/flcva/io.py",
    "src/flcva/model_resolver.py",
    "src/flcva/parsing.py",
    "src/flcva/statistics.py",
    "configs/formal-common.yaml",
    "configs/formal-qwen3.yaml",
    "configs/formal-deepseek.yaml",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ).encode()
            )
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_gz(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(canonical_json_bytes(value))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_gz(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                for row in rows:
                    stream.write(canonical_json_bytes(row) + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _safe_problem_name(problem_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in problem_id
    )


def _execution_bundle() -> dict[str, Any]:
    files = {
        relative: file_sha256(ROOT / relative)
        for relative in EXECUTION_FILES
    }
    return {
        "files_sha256": files,
        "bundle_sha256": payload_sha256(sorted(files.items())),
    }


def _snapshot_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in SNAPSHOT_IDENTITY_KEYS}


def _transport_provenance(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in TRANSPORT_PROVENANCE_KEYS
    }


def _runtime_contract_differences(
    runtime: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}
    for field in (
        "model_snapshot",
        "packages",
        "cuda",
        "device_family",
    ):
        observed = runtime.get(field)
        expected = contract.get(field)
        if observed != expected:
            differences[field] = {
                "expected": expected,
                "observed": observed,
            }
    return differences


def _effective_generation_config(config: RunConfig) -> dict[str, Any]:
    return {
        "backend": config.backend,
        "global_seed": config.global_seed,
        "model": config.model.model_dump(mode="json"),
        "sampling": config.sampling.model_dump(mode="json"),
        "answer_cue": CANONICAL_CUE,
        "answer_max_tokens": ANSWER_MAX_TOKENS,
        "answer_batch": ANSWER_BATCH,
        "extension_batch": EXTENSION_BATCH,
        "probe_top_logprobs": PROBE_TOP_LOGPROBS,
    }


def _runtime_identity(runtime_model: dict[str, Any]) -> dict[str, Any]:
    import torch

    devices = [
        str(torch.cuda.get_device_name(index))
        for index in range(torch.cuda.device_count())
    ]
    if not devices or not all("H20" in name.upper() for name in devices):
        raise RuntimeError(
            f"block-length runtime requires NVIDIA H20: {devices}"
        )
    payload = {
        "schema_version": "flcva-block-length-runtime-v4",
        "model_snapshot": _snapshot_identity(runtime_model),
        "packages": {
            name: package_metadata.version(name)
            for name in RUNTIME_PACKAGE_NAMES
        },
        "cuda": str(torch.version.cuda),
        "device_family": "NVIDIA H20",
    }
    return {**payload, "identity_sha256": payload_sha256(payload)}


def _assert_audited(run_dir: Path) -> dict[str, Any]:
    audit = _read_json(run_dir / "audit_manifest.json")
    if (
        not verify_sealed_record(audit)
        or audit.get("passed") is not True
        or not all((audit.get("gates") or {}).values())
    ):
        raise ValueError(f"source run is not audit-passed: {run_dir}")
    return audit


def _problem_universe(
    run_dir: Path,
) -> dict[str, dict[str, set[str]]]:
    output: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in _iter_jsonl_gz(
        run_dir / "evaluation" / "values.jsonl.gz"
    ):
        output[str(row["split"])][str(row["dataset_id"])].add(
            str(row["problem_id"])
        )
    return output


def _sample_problem_ids() -> dict[str, dict[str, list[str]]]:
    universes = [
        _problem_universe(run_dir) for run_dir in SOURCE_RUNS.values()
    ]
    selected: dict[str, dict[str, list[str]]] = {}
    for split, quotas in SPLIT_QUOTAS.items():
        selected[split] = {}
        for dataset_id, quota in quotas.items():
            shared = set.intersection(
                *(
                    universe[split][dataset_id]
                    for universe in universes
                )
            )
            ordered = sorted(
                shared,
                key=lambda problem_id: hashlib.sha256(
                    (
                        f"{SELECTION_NAMESPACE}:{split}:"
                        f"{dataset_id}:{problem_id}"
                    ).encode()
                ).hexdigest(),
            )
            if len(ordered) < quota:
                raise ValueError(
                    f"{split}/{dataset_id}: need {quota}, found {len(ordered)}"
                )
            selected[split][dataset_id] = ordered[:quota]
    return selected


def _value_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl_gz(
        run_dir / "evaluation" / "values.jsonl.gz"
    ):
        output[str(row["checkpoint_id"])] = {
            "split": str(row["split"]),
            "proxies": {
                proxy: row["proxies"].get(proxy) for proxy in PROXIES
            },
        }
    return output


def _extract_problem(
    run_dir: Path,
    problem_id: str,
    *,
    expected_model_id: str,
    values: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = (
        run_dir
        / "problems"
        / f"{_safe_problem_name(problem_id)}.json.gz"
    )
    artifact = read_json_gz(path)
    if not verify_sealed_record(artifact):
        raise ValueError(f"problem artifact seal failed: {problem_id}")
    blobs = BlobStore(run_dir / "blobs")
    by_spine_depth: dict[tuple[int, int], dict[str, Any]] = {}
    branches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in artifact["branches"]:
        branches[str(row["checkpoint_id"])].append(row)
    for checkpoint in artifact["checkpoints"]:
        if checkpoint["model_id"] != expected_model_id:
            raise ValueError(f"{problem_id}: model identity mismatch")
        by_spine_depth[
            (int(checkpoint["spine_id"]), int(checkpoint["depth"]))
        ] = checkpoint

    spines: list[dict[str, Any]] = []
    for spine_id in range(4):
        first = by_spine_depth[(spine_id, 256)]
        prompt_tokens = blobs.get(str(first["prompt_blob"]))
        spine_tokens = blobs.get(str(first["spine_blob"]))
        checkpoints: list[dict[str, Any]] = []
        for depth in COMMON_DEPTHS:
            checkpoint = by_spine_depth[(spine_id, depth)]
            successor_512 = by_spine_depth[(spine_id, depth + 512)]
            value = values[str(checkpoint["checkpoint_id"])]
            branch_rows = sorted(
                branches[str(checkpoint["checkpoint_id"])],
                key=lambda row: int(row["branch_index"]),
            )
            if [int(row["branch_index"]) for row in branch_rows] != [
                1,
                2,
                3,
                4,
            ]:
                raise ValueError(
                    f"{checkpoint['checkpoint_id']}: incomplete branches"
                )
            extracted_branches: list[dict[str, Any]] = []
            for branch in branch_rows:
                reasoning_tokens = blobs.get(str(branch["reasoning_blob"]))
                if (
                    int(branch["reasoning_tokens"]) != 256
                    or len(reasoning_tokens) != 256
                ):
                    raise ValueError("stored continuation is not 256 tokens")
                extracted_branches.append(
                    {
                        "branch_id": branch["branch_id"],
                        "branch_index": int(branch["branch_index"]),
                        "reasoning_tokens_256": reasoning_tokens,
                        "answer_tokens_256": branch["answer_tokens"],
                    }
                )
            checkpoints.append(
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "depth": depth,
                    "split": value["split"],
                    "proxies": value["proxies"],
                    "stop_answer_tokens": checkpoint["stop_answer_tokens"],
                    "on_spine_answer_tokens_512": successor_512[
                        "stop_answer_tokens"
                    ],
                    "branches": extracted_branches,
                }
            )
        spines.append(
            {
                "spine_id": spine_id,
                "prompt_tokens": prompt_tokens,
                "spine_tokens": spine_tokens,
                "checkpoints": checkpoints,
            }
        )
    splits = {
        checkpoint["split"]
        for spine in spines
        for checkpoint in spine["checkpoints"]
    }
    if len(splits) != 1:
        raise ValueError(f"{problem_id}: split differs across checkpoints")
    output = {
        "schema_version": "flcva-block-length-input-problem-v4",
        "problem_id": problem_id,
        "dataset_id": artifact["dataset_id"],
        "split": next(iter(splits)),
        "source_problem_record_sha256": artifact["record_sha256"],
        "spines": spines,
        "record_sha256": "",
    }
    output["record_sha256"] = payload_sha256(output)
    return output


def prepare_inputs(
    model_names: tuple[str, ...] | None = None,
) -> None:
    selected = _sample_problem_ids()
    selected_ids = sorted(
        problem_id
        for split_rows in selected.values()
        for dataset_rows in split_rows.values()
        for problem_id in dataset_rows
    )
    if len(selected_ids) != 400 or len(set(selected_ids)) != 400:
        raise ValueError("block-length study requires 400 unique problems")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = _execution_bundle()
    selected_models = model_names or tuple(SOURCE_RUNS)
    if not selected_models or any(
        model_name not in SOURCE_RUNS for model_name in selected_models
    ):
        raise ValueError("invalid block-length input model selection")
    for model_name in selected_models:
        run_dir = SOURCE_RUNS[model_name]
        audit = _assert_audited(run_dir)
        contract = _read_json(run_dir / "contract.json")
        config = RunConfig.load(CONFIGS[model_name])
        if (
            config.backend != "vllm"
            or config.model.id != contract["model_id"]
            or config.model.revision != contract["model_revision"]
            or config.model.tokenizer_revision
            != contract["tokenizer_revision"]
            or config.sampling.model_dump(mode="json")
            != contract["sampling"]
            or config.global_seed != contract["global_seed"]
        ):
            raise ValueError("length config differs from formal source contract")
        source_manifest = _read_json(run_dir / "manifest.json")
        if not verify_sealed_record(source_manifest):
            raise ValueError("source merged manifest seal failed")
        model_identities = {
            payload_sha256(row["runtime_model_identity"]): row[
                "runtime_model_identity"
            ]
            for row in source_manifest["parent_runs"]
        }
        merge = _read_json(run_dir / "merge_summary.json")
        package_sets: dict[str, dict[str, Any]] = {}
        hardware_sets: dict[str, dict[str, Any]] = {}
        for source_shard in merge["inputs"]:
            source_run = _read_json(Path(source_shard) / "run.json")
            packages = {
                name: source_run["packages"].get(name)
                for name in RUNTIME_PACKAGE_NAMES
            }
            package_sets[payload_sha256(packages)] = packages
            hardware = {
                "cuda": str(source_run["hardware"]["cuda"]),
                "device_family": "NVIDIA H20",
            }
            hardware_sets[payload_sha256(hardware)] = hardware
        if (
            len(model_identities) != 1
            or len(package_sets) != 1
            or len(hardware_sets) != 1
        ):
            raise ValueError("formal runtime identity is not unique")
        values = _value_index(run_dir)
        input_path = INPUT_DIR / f"{model_name}.jsonl.gz"
        _write_jsonl_gz(
            input_path,
            (
                _extract_problem(
                    run_dir,
                    problem_id,
                    expected_model_id=str(contract["model_id"]),
                    values=values,
                )
                for problem_id in selected_ids
            ),
        )
        manifest = seal_record(
            {
                "schema_version": "flcva-block-length-input-manifest-v4",
                "study_id": STUDY_ID,
                "model_name": model_name,
                "model_id": contract["model_id"],
                "model_revision": contract["model_revision"],
                "tokenizer_revision": contract["tokenizer_revision"],
                "sampling": contract["sampling"],
                "global_seed": contract["global_seed"],
                "generation_contract": _effective_generation_config(config),
                "source_run_id": audit["run_id"],
                "source_contract_sha256": audit["contract_sha256"],
                "source_result_merkle_root": audit["result_merkle_root"],
                "source_audit_manifest_sha256": file_sha256(
                    run_dir / "audit_manifest.json"
                ),
                "selection": {
                    "namespace": SELECTION_NAMESPACE,
                    "quotas": SPLIT_QUOTAS,
                    "problem_ids": selected,
                    "problem_ids_sha256": payload_sha256(selected_ids),
                },
                "lengths": list(LENGTHS),
                "common_depths": list(COMMON_DEPTHS),
                "arms": {
                    "128": (
                        "truncate each stored 256-token branch after token "
                        "128 and rerun the canonical answer operator in the "
                        "matched three-length schedule"
                    ),
                    "256": (
                        "reuse the sealed formal reasoning branch, rerun the "
                        "canonical answer operator in the matched schedule, "
                        "and retain the formal answer only as a parity control"
                    ),
                    "512": (
                        "extend the same stored branch by 256 tokens; reuse "
                        "the verified on-spine reasoning extension, newly "
                        "sample off-spine extensions, and rerun every answer "
                        "operator in the matched schedule"
                    ),
                },
                "answer_schedule": {
                    "capacity": ANSWER_BATCH,
                    "construction": (
                        "eight prefix-branch contexts per batch, with all "
                        "three lengths adjacent for each context"
                    ),
                    "formal_answers_used_as_outcomes": False,
                },
                "evaluation_gates": {
                    "formal_256_stop_correctness_identity": 1.0,
                    "formal_256_branch_correctness_identity_min": (
                        MIN_256_BRANCH_CORRECTNESS_PARITY
                    ),
                    "formal_256_checkpoint_sign_identity_min": (
                        MIN_256_CHECKPOINT_SIGN_PARITY
                    ),
                },
                "spines_per_problem": 4,
                "branches_per_checkpoint": 4,
                "workers": WORKERS,
                "labels_included": False,
                "proxies": list(PROXIES),
                "input_file": input_path.name,
                "input_sha256": file_sha256(input_path),
                "execution_bundle": bundle,
                "runtime_contract": {
                    "model_snapshot": _snapshot_identity(
                        next(iter(model_identities.values()))
                    ),
                    "source_transport_provenance": _transport_provenance(
                        next(iter(model_identities.values()))
                    ),
                    "packages": next(iter(package_sets.values())),
                    "device_family": "NVIDIA H20",
                    "cuda": next(iter(hardware_sets.values()))["cuda"],
                },
            }
        )
        _atomic_json(INPUT_DIR / f"{model_name}.manifest.json", manifest)
        print(
            json.dumps(
                {
                    "model": model_name,
                    "problems": len(selected_ids),
                    "input_sha256": manifest["input_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        # The full source value index is large.  Release it before constructing
        # the peer model index so input preparation remains reproducible on a
        # memory-constrained analysis host.
        del values
        gc.collect()


def prepare_all_inputs() -> None:
    """Isolate each full source value index in its own process."""
    for model_name in SOURCE_RUNS:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "prepare-model",
                "--model",
                model_name,
            ],
            check=True,
        )
    validate_inputs()


def _load_input(
    model_name: str,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    manifest_path = INPUT_DIR / f"{model_name}.manifest.json"
    manifest = _read_json(manifest_path)
    if (
        not verify_sealed_record(manifest)
        or manifest.get("schema_version")
        != "flcva-block-length-input-manifest-v4"
        or manifest.get("study_id") != STUDY_ID
        or manifest.get("model_name") != model_name
        or manifest.get("labels_included") is not False
        or manifest.get("lengths") != list(LENGTHS)
        or manifest.get("common_depths") != list(COMMON_DEPTHS)
        or manifest.get("workers") != WORKERS
        or manifest.get("proxies") != list(PROXIES)
        or (manifest.get("answer_schedule") or {}).get("capacity")
        != ANSWER_BATCH
        or manifest.get("evaluation_gates")
        != {
            "formal_256_stop_correctness_identity": 1.0,
            "formal_256_branch_correctness_identity_min": (
                MIN_256_BRANCH_CORRECTNESS_PARITY
            ),
            "formal_256_checkpoint_sign_identity_min": (
                MIN_256_CHECKPOINT_SIGN_PARITY
            ),
        }
        or manifest.get("execution_bundle") != _execution_bundle()
    ):
        raise ValueError(f"invalid block-length input: {manifest_path}")
    input_path = INPUT_DIR / str(manifest["input_file"])
    if file_sha256(input_path) != manifest["input_sha256"]:
        raise ValueError("block-length input digest mismatch")
    rows = list(_iter_jsonl_gz(input_path))
    if len(rows) != 400:
        raise ValueError("block-length input must contain 400 problems")
    observed_ids: set[str] = set()
    for row in rows:
        problem_id = str(row.get("problem_id"))
        unhashed = dict(row)
        recorded = str(unhashed.pop("record_sha256", ""))
        unhashed["record_sha256"] = ""
        if (
            row.get("schema_version")
            != "flcva-block-length-input-problem-v4"
            or payload_sha256(unhashed) != recorded
            or problem_id in observed_ids
        ):
            raise ValueError("invalid sealed block-length input problem")
        observed_ids.add(problem_id)
        _expected_problem_rows(row)
    selected_ids = sorted(
        problem_id
        for split_rows in manifest["selection"]["problem_ids"].values()
        for dataset_rows in split_rows.values()
        for problem_id in dataset_rows
    )
    if (
        observed_ids != set(selected_ids)
        or manifest["selection"]["problem_ids_sha256"]
        != payload_sha256(selected_ids)
    ):
        raise ValueError("block-length input selection mismatch")
    return manifest, input_path, rows


def _generation_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "flcva-block-length-generation-identity-v4",
        "study_id": STUDY_ID,
        "input_manifest_record_sha256": manifest["record_sha256"],
        "input_sha256": manifest["input_sha256"],
        "execution_bundle": manifest["execution_bundle"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "sampling": manifest["sampling"],
        "global_seed": manifest["global_seed"],
        "generation_contract": manifest["generation_contract"],
        "lengths": list(LENGTHS),
        "common_depths": list(COMMON_DEPTHS),
        "answer_batch": ANSWER_BATCH,
        "extension_batch": EXTENSION_BATCH,
    }
    return {**payload, "identity_sha256": payload_sha256(payload)}


def validate_inputs() -> None:
    problem_sets: list[set[str]] = []
    for model_name in SOURCE_RUNS:
        manifest, _, rows = _load_input(model_name)
        ids = {str(row["problem_id"]) for row in rows}
        if len(ids) != 400:
            raise ValueError("block-length input has duplicate problems")
        for row in rows:
            unhashed = dict(row)
            recorded = str(unhashed.pop("record_sha256"))
            unhashed["record_sha256"] = ""
            if payload_sha256(unhashed) != recorded:
                raise ValueError("block-length input problem seal failed")
        problem_sets.append(ids)
        print(
            json.dumps(
                {
                    "model": model_name,
                    "problems": len(ids),
                    "input_sha256": manifest["input_sha256"],
                },
                sort_keys=True,
            )
        )
    if problem_sets[0] != problem_sets[1]:
        raise ValueError("model block-length samples differ")


def _worker_bucket(problem_id: str) -> int:
    digest = hashlib.sha256(
        f"flcva-block-length-worker-v4:{problem_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % WORKERS


def _chunks(rows: list[Any], size: int) -> Iterator[list[Any]]:
    if len(rows) % size:
        raise ValueError(
            f"request count {len(rows)} is not divisible by batch {size}"
        )
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _generated_metadata(output: Any) -> dict[str, Any]:
    return {
        "tokens": output.token_ids,
        "text": output.text,
        "output_tokens": len(output.token_ids),
        "gpu_seconds": float(output.gpu_seconds or 0.0),
        "peak_gpu_memory_bytes": int(
            output.peak_gpu_memory_bytes or 0
        ),
        "batch_size": int(output.batch_size),
        "batch_utilization": float(output.batch_utilization or 0.0),
        "finish_reason": output.finish_reason,
    }


def _decode_binding(
    token_ids: list[int],
    decoded_text: str,
) -> str:
    return payload_sha256(
        {
            "token_ids": token_ids,
            "decoded_text": decoded_text,
        }
    )


def _run_problem(
    problem: dict[str, Any],
    backend: Any,
    sampling: Any,
    *,
    global_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cue_tokens = backend.encode(CANONICAL_CUE)
    base_rows: dict[tuple[str, int], dict[str, Any]] = {}
    pending_extension: list[
        tuple[tuple[str, int], GenerationRequest]
    ] = []
    results: list[dict[str, Any]] = []

    for spine in problem["spines"]:
        spine_id = int(spine["spine_id"])
        prompt = list(spine["prompt_tokens"])
        spine_tokens = list(spine["spine_tokens"])
        for checkpoint in spine["checkpoints"]:
            depth = int(checkpoint["depth"])
            prefix = prompt + spine_tokens[:depth]
            stop_answer_tokens = list(checkpoint["stop_answer_tokens"])
            stop_decoded_text = backend.decode(stop_answer_tokens)
            stop_raw = CANONICAL_CUE + stop_decoded_text
            stop_decode_binding = _decode_binding(
                stop_answer_tokens,
                stop_decoded_text,
            )
            for branch in checkpoint["branches"]:
                branch_index = int(branch["branch_index"])
                key = (str(checkpoint["checkpoint_id"]), branch_index)
                common = {
                    "problem_id": problem["problem_id"],
                    "dataset_id": problem["dataset_id"],
                    "split": problem["split"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "spine_id": spine_id,
                    "depth": depth,
                    "branch_id": branch["branch_id"],
                    "branch_index": branch_index,
                    "stop_answer_token_ids": stop_answer_tokens,
                    "stop_decoded_text": stop_decoded_text,
                    "stop_decode_binding_sha256": stop_decode_binding,
                    "stop_raw_answer": stop_raw,
                    "proxies": checkpoint["proxies"],
                }
                first_256 = list(branch["reasoning_tokens_256"])
                if (
                    branch_index == 1
                    and first_256 != spine_tokens[depth : depth + 256]
                ):
                    raise ValueError(
                        "on-spine branch does not match the stored successor"
                    )
                base_rows[key] = {
                    **common,
                    "first_256": first_256,
                    "prefix": prefix,
                    "spine_tokens": spine_tokens,
                    "answer_tokens_256": list(
                        branch["answer_tokens_256"]
                    ),
                    "on_spine_answer_tokens_512": list(
                        checkpoint["on_spine_answer_tokens_512"]
                    ),
                    "on_spine_second_256": list(
                        spine_tokens[depth + 256 : depth + 512]
                    ),
                }
                if branch_index > 1:
                    pending_extension.append(
                        (
                            key,
                            GenerationRequest(
                                prompt_token_ids=prefix + first_256,
                                max_tokens=256,
                                seed=stable_seed(
                                    global_seed,
                                    STUDY_ID,
                                    checkpoint["checkpoint_id"],
                                    branch_index,
                                    512,
                                    "extension",
                                ),
                                purpose="reasoning",
                                batch_capacity=EXTENSION_BATCH,
                            ),
                        )
                    )

    if len(base_rows) != 224:
        raise ValueError("expected 224 branch contexts per problem")
    physical = {
        "output_tokens": 0.0,
        "gpu_seconds": 0.0,
        "peak_gpu_memory_bytes": 0.0,
        "requests": 0.0,
        "batches": 0.0,
        "answer_backend_text_mismatches": 0.0,
    }

    extensions: dict[tuple[str, int], dict[str, Any]] = {}
    for batch in _chunks(pending_extension, EXTENSION_BATCH):
        batch_index = int(physical["batches"])
        generated = backend.generate_batch(
            [request for _, request in batch],
            sampling=sampling,
        )
        physical["batches"] += 1
        for batch_position, ((key, request), output) in enumerate(
            zip(batch, generated, strict=True)
        ):
            metadata = _generated_metadata(output)
            if len(metadata["tokens"]) != 256:
                raise ValueError("512 arm extension is not 256 tokens")
            metadata["batch_index"] = batch_index
            metadata["batch_position"] = batch_position
            metadata["seed"] = request.seed
            extensions[key] = metadata
            physical["requests"] += 1
            physical["output_tokens"] += metadata["output_tokens"]
            physical["gpu_seconds"] += metadata["gpu_seconds"]
            physical["peak_gpu_memory_bytes"] = max(
                physical["peak_gpu_memory_bytes"],
                metadata["peak_gpu_memory_bytes"],
            )

    continuations: dict[tuple[str, int, int], list[int]] = {}
    pending_answers: list[
        tuple[tuple[str, int, int], GenerationRequest]
    ] = []
    for key in sorted(base_rows):
        base = base_rows[key]
        first_256 = list(base["first_256"])
        if int(base["branch_index"]) == 1:
            second_256 = list(base["on_spine_second_256"])
        else:
            second_256 = list(extensions[key]["tokens"])
        by_length = {
            128: first_256[:128],
            256: first_256,
            512: first_256 + second_256,
        }
        for length in LENGTHS:
            continuation = by_length[length]
            if len(continuation) != length:
                raise ValueError("continuation length construction failed")
            request_key = (key[0], key[1], length)
            continuations[request_key] = continuation
            pending_answers.append(
                (
                    request_key,
                    GenerationRequest(
                        prompt_token_ids=(
                            base["prefix"] + continuation + cue_tokens
                        ),
                        max_tokens=ANSWER_MAX_TOKENS,
                        seed=stable_seed(
                            global_seed,
                            STUDY_ID,
                            key[0],
                            key[1],
                            length,
                            "answer",
                        ),
                        purpose="answer",
                        greedy=True,
                        top_logprobs=PROBE_TOP_LOGPROBS,
                        batch_capacity=ANSWER_BATCH,
                    ),
                )
            )
    if len(pending_answers) != 672:
        raise ValueError("matched answer schedule has wrong cardinality")

    answers: dict[tuple[str, int, int], dict[str, Any]] = {}
    for batch in _chunks(pending_answers, ANSWER_BATCH):
        batch_index = int(physical["batches"])
        generated = backend.generate_batch(
            [request for _, request in batch],
            sampling=sampling,
        )
        physical["batches"] += 1
        for batch_position, ((key, request), output) in enumerate(
            zip(batch, generated, strict=True)
        ):
            metadata = _generated_metadata(output)
            decoded_text = backend.decode(list(metadata["tokens"]))
            metadata["batch_index"] = batch_index
            metadata["batch_position"] = batch_position
            metadata["seed"] = request.seed
            metadata["decoded_text"] = decoded_text
            metadata["backend_text_matches_decode"] = (
                metadata["text"] == decoded_text
            )
            physical["answer_backend_text_mismatches"] += int(
                not metadata["backend_text_matches_decode"]
            )
            metadata["decode_binding_sha256"] = _decode_binding(
                list(metadata["tokens"]),
                decoded_text,
            )
            answers[key] = metadata
            physical["requests"] += 1
            physical["output_tokens"] += metadata["output_tokens"]
            physical["gpu_seconds"] += metadata["gpu_seconds"]
            physical["peak_gpu_memory_bytes"] = max(
                physical["peak_gpu_memory_bytes"],
                metadata["peak_gpu_memory_bytes"],
            )

    for key in sorted(base_rows):
        base = base_rows[key]
        shared = {
            name: base[name]
            for name in (
                "problem_id",
                "dataset_id",
                "split",
                "checkpoint_id",
                "spine_id",
                "depth",
                "branch_id",
                "branch_index",
                "stop_answer_token_ids",
                "stop_decoded_text",
                "stop_decode_binding_sha256",
                "stop_raw_answer",
                "proxies",
            )
        }
        for length in LENGTHS:
            request_key = (key[0], key[1], length)
            answer = answers[request_key]
            continuation = continuations[request_key]
            extension = (
                extensions.get(key)
                if length == 512 and int(base["branch_index"]) > 1
                else None
            )
            formal_tokens: list[int] | None
            if length == 256:
                formal_tokens = list(base["answer_tokens_256"])
            elif length == 512 and int(base["branch_index"]) == 1:
                formal_tokens = list(base["on_spine_answer_tokens_512"])
            else:
                formal_tokens = None
            if extension is None:
                physical_reasoning = 0
                extension_tokens = None
                extension_seed = None
                extension_batch_index = None
                extension_batch_position = None
                extension_batch_size = None
                extension_gpu_seconds = 0.0
                extension_peak_memory = 0
            else:
                physical_reasoning = int(extension["output_tokens"])
                extension_tokens = list(extension["tokens"])
                extension_seed = int(extension["seed"])
                extension_batch_index = int(extension["batch_index"])
                extension_batch_position = int(
                    extension["batch_position"]
                )
                extension_batch_size = int(extension["batch_size"])
                extension_gpu_seconds = float(extension["gpu_seconds"])
                extension_peak_memory = int(
                    extension["peak_gpu_memory_bytes"]
                )
            physical_source = {
                128: "stored branch prefix plus new matched probe",
                256: "stored formal branch plus new matched probe",
                512: (
                    "stored on-spine continuation plus new matched probe"
                    if int(base["branch_index"]) == 1
                    else "stored first block plus new extension and matched probe"
                ),
            }[length]
            answer_tokens = list(answer["tokens"])
            results.append(
                {
                    **shared,
                    "length": length,
                    "reasoning_tokens": length,
                    "reasoning_token_sha256": payload_sha256(continuation),
                    "extension_tokens_256": extension_tokens,
                    "extension_seed": extension_seed,
                    "extension_batch_capacity": (
                        EXTENSION_BATCH if extension is not None else None
                    ),
                    "extension_batch_index": extension_batch_index,
                    "extension_batch_position": extension_batch_position,
                    "extension_batch_size": extension_batch_size,
                    "answer_token_ids": answer_tokens,
                    "answer_decoded_text": answer["decoded_text"],
                    "answer_backend_text_matches_decode": answer[
                        "backend_text_matches_decode"
                    ],
                    "answer_decode_binding_sha256": answer[
                        "decode_binding_sha256"
                    ],
                    "answer_tokens": len(answer_tokens),
                    "answer_seed": int(answer["seed"]),
                    "answer_batch_capacity": ANSWER_BATCH,
                    "answer_batch_index": int(answer["batch_index"]),
                    "answer_batch_position": int(
                        answer["batch_position"]
                    ),
                    "answer_batch_size": int(answer["batch_size"]),
                    "answer_batch_utilization": float(
                        answer["batch_utilization"]
                    ),
                    "answer_finish_reason": answer["finish_reason"],
                    "raw_answer": (
                        CANONICAL_CUE + answer["decoded_text"]
                    ),
                    "formal_answer_token_match": (
                        answer_tokens == formal_tokens
                        if formal_tokens is not None
                        else None
                    ),
                    "logical_tokens": length + len(answer_tokens),
                    "physical_source": physical_source,
                    "physical_reasoning_tokens": physical_reasoning,
                    "physical_answer_tokens": len(answer_tokens),
                    "physical_request_count": (
                        1 + int(extension is not None)
                    ),
                    "gpu_seconds": (
                        float(answer["gpu_seconds"])
                        + extension_gpu_seconds
                    ),
                    "peak_gpu_memory_bytes": max(
                        int(answer["peak_gpu_memory_bytes"]),
                        extension_peak_memory,
                    ),
                }
            )
    if len(results) != 672:
        raise ValueError("block-length problem output cardinality mismatch")
    if (
        int(physical["requests"]) != 840
        or int(physical["batches"]) != 32
    ):
        raise ValueError("block-length physical schedule mismatch")
    return results, physical


def _resume_worker_payloads_are_valid(
    worker: dict[str, Any],
    worker_dir: Path,
    *,
    model_name: str,
    worker_index: int,
    expected_problem_ids: list[str],
    problem_map: dict[str, dict[str, Any]],
    input_manifest: dict[str, Any],
    generation_identity: dict[str, Any],
) -> bool:
    """Verify every child artifact before accepting a completed worker."""
    try:
        runtime = worker["runtime_identity"]
        runtime_contract = input_manifest["runtime_contract"]
        if (
            not _identity_is_valid(runtime)
            or _runtime_contract_differences(
                runtime,
                runtime_contract,
            )
            or _snapshot_identity(worker["runtime_model_identity"])
            != runtime_contract["model_snapshot"]
        ):
            return False
        problem_files = worker.get("problem_files")
        if (
            not isinstance(problem_files, list)
            or len(problem_files) != len(expected_problem_ids)
        ):
            return False
        observed: list[str] = []
        rows_total = 0
        accounting_total: dict[str, float] = defaultdict(float)
        for item in problem_files:
            if not isinstance(item, dict):
                return False
            relative = Path(str(item.get("path", "")))
            if (
                not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                return False
            path = worker_dir / relative
            if (
                not path.is_file()
                or file_sha256(path) != item.get("sha256")
            ):
                return False
            payload = read_json_gz(path)
            if (
                not verify_sealed_record(payload)
                or payload.get("schema_version")
                != "flcva-block-length-output-problem-v4"
                or payload.get("study_id") != STUDY_ID
                or payload.get("model_name") != model_name
                or payload.get("model_id") != input_manifest["model_id"]
                or payload.get("model_revision")
                != input_manifest["model_revision"]
                or payload.get("worker_index") != worker_index
                or payload.get("input_sha256")
                != input_manifest["input_sha256"]
                or payload.get("generation_identity_sha256")
                != generation_identity["identity_sha256"]
                or payload.get("runtime_identity_sha256")
                != runtime["identity_sha256"]
            ):
                return False
            problem_id = str(payload.get("problem_id"))
            if (
                problem_id not in problem_map
                or item.get("problem_id") != problem_id
                or payload.get("input_problem_record_sha256")
                != problem_map[problem_id]["record_sha256"]
                or payload.get("dataset_id")
                != problem_map[problem_id]["dataset_id"]
                or payload.get("split")
                != problem_map[problem_id]["split"]
            ):
                return False
            rows = payload.get("results")
            if (
                not isinstance(rows, list)
                or len(rows) != 672
                or item.get("rows") != 672
            ):
                return False
            recomputed = _validate_problem_rows(
                rows,
                problem_map[problem_id],
                global_seed=int(input_manifest["global_seed"]),
            )
            if not _numeric_maps_equal(
                recomputed,
                payload.get("physical_accounting", {}),
            ):
                return False
            for key, value in recomputed.items():
                if key == "peak_gpu_memory_bytes":
                    accounting_total[key] = max(
                        accounting_total[key],
                        value,
                    )
                else:
                    accounting_total[key] += value
            observed.append(problem_id)
            rows_total += len(rows)
        return (
            sorted(observed) == expected_problem_ids
            and len(set(observed)) == len(observed)
            and worker.get("problem_ids_sha256")
            == payload_sha256(expected_problem_ids)
            and worker.get("problems") == len(expected_problem_ids)
            and worker.get("rows") == rows_total
            and _numeric_maps_equal(
                dict(accounting_total),
                worker.get("physical_accounting", {}),
            )
        )
    except (EOFError, KeyError, OSError, TypeError, ValueError):
        return False


def generate(
    model_name: str,
    worker_index: int,
    output_dir: Path,
) -> None:
    if not 0 <= worker_index < WORKERS:
        raise ValueError("worker index must be in [0,7]")
    manifest, _, problems = _load_input(model_name)
    config = RunConfig.load(CONFIGS[model_name])
    if (
        _effective_generation_config(config)
        != manifest["generation_contract"]
    ):
        raise ValueError("model config differs from sealed length input")
    generation_identity = _generation_identity(manifest)
    expected = sorted(
        problem["problem_id"]
        for problem in problems
        if _worker_bucket(str(problem["problem_id"])) == worker_index
    )
    problem_map = {str(row["problem_id"]): row for row in problems}
    worker_dir = (
        output_dir.resolve() / "workers" / f"worker-{worker_index:02d}"
    )
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_manifest_path = worker_dir / "manifest.json"
    if worker_manifest_path.exists():
        existing = _read_json(worker_manifest_path)
        compatible = (
            verify_sealed_record(existing)
            and existing.get("schema_version")
            == "flcva-block-length-worker-manifest-v4"
            and existing.get("complete") is True
            and existing.get("study_id") == STUDY_ID
            and existing.get("model_name") == model_name
            and existing.get("model_id") == manifest["model_id"]
            and existing.get("model_revision")
            == manifest["model_revision"]
            and existing.get("worker_index") == worker_index
            and existing.get("workers") == WORKERS
            and existing.get("problem_ids_sha256")
            == payload_sha256(expected)
            and existing.get("input_sha256") == manifest["input_sha256"]
            and existing.get("generation_identity")
            == generation_identity
        )
        if not compatible:
            raise ValueError("incompatible existing worker manifest")
        if _resume_worker_payloads_are_valid(
            existing,
            worker_dir,
            model_name=model_name,
            worker_index=worker_index,
            expected_problem_ids=expected,
            problem_map=problem_map,
            input_manifest=manifest,
            generation_identity=generation_identity,
        ):
            print(json.dumps({"resume": "complete", "worker": worker_index}))
            return
        print(
            json.dumps(
                {
                    "resume": "repair",
                    "worker": worker_index,
                    "reason": (
                        "completed manifest has missing, corrupt, or "
                        "scientifically invalid child artifacts"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    backend = build_backend(config.backend, config.model)
    runtime = _runtime_identity(backend.runtime_identity)
    contract = manifest["runtime_contract"]
    runtime_differences = _runtime_contract_differences(
        runtime,
        contract,
    )
    if runtime_differences:
        raise ValueError(
            "runtime differs from sealed formal runtime: "
            + json.dumps(runtime_differences, sort_keys=True)
        )
    print(
        json.dumps(
            {
                "runtime_contract": "verified",
                "worker": worker_index,
                "runtime_identity_sha256": runtime["identity_sha256"],
                "transport_provenance": _transport_provenance(
                    backend.runtime_identity
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.time()
    problem_files: list[dict[str, Any]] = []
    totals: dict[str, float] = defaultdict(float)
    for problem_id in expected:
        rows, accounting = _run_problem(
            problem_map[problem_id],
            backend,
            config.sampling,
            global_seed=config.global_seed,
        )
        payload = seal_record(
            {
                "schema_version": "flcva-block-length-output-problem-v4",
                "study_id": STUDY_ID,
                "model_name": model_name,
                "model_id": manifest["model_id"],
                "model_revision": manifest["model_revision"],
                "worker_index": worker_index,
                "input_sha256": manifest["input_sha256"],
                "generation_identity_sha256": generation_identity[
                    "identity_sha256"
                ],
                "runtime_identity_sha256": runtime["identity_sha256"],
                "problem_id": problem_id,
                "dataset_id": problem_map[problem_id]["dataset_id"],
                "split": problem_map[problem_id]["split"],
                "input_problem_record_sha256": problem_map[problem_id][
                    "record_sha256"
                ],
                "results": rows,
                "physical_accounting": accounting,
            }
        )
        relative = Path("problems") / f"{_safe_problem_name(problem_id)}.json.gz"
        path = worker_dir / relative
        _atomic_json_gz(path, payload)
        problem_files.append(
            {
                "problem_id": problem_id,
                "path": str(relative),
                "sha256": file_sha256(path),
                "rows": len(rows),
            }
        )
        for key, value in accounting.items():
            if key == "peak_gpu_memory_bytes":
                totals[key] = max(totals[key], value)
            else:
                totals[key] += value
        print(
            json.dumps(
                {
                    "worker": worker_index,
                    "problem": problem_id,
                    "completed": len(problem_files),
                    "expected": len(expected),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    worker_manifest = seal_record(
        {
            "schema_version": "flcva-block-length-worker-manifest-v4",
            "complete": True,
            "study_id": STUDY_ID,
            "model_name": model_name,
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "worker_index": worker_index,
            "workers": WORKERS,
            "input_sha256": manifest["input_sha256"],
            "generation_identity": generation_identity,
            "problem_ids_sha256": payload_sha256(expected),
            "problems": len(expected),
            "rows": 672 * len(expected),
            "physical_accounting": dict(totals),
            "wall_seconds": time.time() - started,
            "runtime_identity": runtime,
            "runtime_model_identity": backend.runtime_identity,
            "problem_files": problem_files,
        }
    )
    _atomic_json(worker_manifest_path, worker_manifest)
    print(json.dumps(worker_manifest, sort_keys=True), flush=True)


def _load_labels() -> dict[str, tuple[str, str]]:
    manifest = _read_json(DATA_MANIFEST_PATH)
    if (
        file_sha256(LABELS_PATH)
        != manifest["combined"]["sealed_labels_sha256"]
    ):
        raise ValueError("sealed label digest mismatch")
    output: dict[str, tuple[str, str]] = {}
    with LABELS_PATH.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            output[str(row["id"])] = (
                str(row["dataset_id"]),
                str(row["answer"]),
            )
    return output


def _identity_is_valid(value: dict[str, Any]) -> bool:
    payload = dict(value)
    recorded = payload.pop("identity_sha256", None)
    return isinstance(recorded, str) and payload_sha256(payload) == recorded


def _numeric_maps_equal(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        math.isclose(
            float(left[key]),
            float(right[key]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        for key in left
    )


def _expected_problem_rows(
    problem: dict[str, Any],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    expected: dict[tuple[str, int, int], dict[str, Any]] = {}
    for spine in problem["spines"]:
        spine_id = int(spine["spine_id"])
        spine_tokens = list(spine["spine_tokens"])
        for checkpoint in spine["checkpoints"]:
            checkpoint_id = str(checkpoint["checkpoint_id"])
            depth = int(checkpoint["depth"])
            for branch in checkpoint["branches"]:
                branch_index = int(branch["branch_index"])
                first_256 = list(branch["reasoning_tokens_256"])
                for length in LENGTHS:
                    key = (checkpoint_id, branch_index, length)
                    if key in expected:
                        raise ValueError("duplicate expected block-length row")
                    expected[key] = {
                        "problem_id": problem["problem_id"],
                        "dataset_id": problem["dataset_id"],
                        "split": problem["split"],
                        "checkpoint_id": checkpoint_id,
                        "spine_id": spine_id,
                        "depth": depth,
                        "branch_id": branch["branch_id"],
                        "branch_index": branch_index,
                        "length": length,
                        "first_256": first_256,
                        "on_spine_second_256": list(
                            spine_tokens[depth + 256 : depth + 512]
                        ),
                        "formal_answer_tokens_256": list(
                            branch["answer_tokens_256"]
                        ),
                        "formal_answer_tokens_512": (
                            list(checkpoint["on_spine_answer_tokens_512"])
                            if branch_index == 1
                            else None
                        ),
                        "stop_answer_tokens": list(
                            checkpoint["stop_answer_tokens"]
                        ),
                        "proxies": checkpoint["proxies"],
                    }
    if len(expected) != 672:
        raise ValueError("sealed input does not define 672 expected rows")
    return expected


def _validate_problem_rows(
    rows: list[dict[str, Any]],
    problem: dict[str, Any],
    *,
    global_seed: int,
) -> dict[str, float]:
    expected = _expected_problem_rows(problem)
    observed: set[tuple[str, int, int]] = set()
    stop_answers: dict[str, tuple[tuple[int, ...], str, str]] = {}
    answer_batches: dict[
        int,
        list[tuple[int, str, int, int]],
    ] = defaultdict(list)
    extension_batches: dict[
        int,
        list[tuple[int, str, int]],
    ] = defaultdict(list)
    total_gpu_seconds = 0.0
    peak_gpu_memory = 0.0
    output_tokens = 0
    requests = 0
    answer_backend_text_mismatches = 0
    for row in rows:
        key = (
            str(row.get("checkpoint_id")),
            int(row.get("branch_index")),
            int(row.get("length")),
        )
        if key not in expected or key in observed:
            raise ValueError("unexpected or duplicate block-length result row")
        observed.add(key)
        reference = expected[key]
        for field in (
            "problem_id",
            "dataset_id",
            "split",
            "checkpoint_id",
            "spine_id",
            "depth",
            "branch_id",
            "branch_index",
            "length",
            "proxies",
        ):
            if row.get(field) != reference[field]:
                raise ValueError(
                    f"block-length row differs from input field {field}"
                )

        length = int(row["length"])
        branch_index = int(row["branch_index"])
        if int(row.get("reasoning_tokens", -1)) != length:
            raise ValueError("invalid logical reasoning-token count")
        extension_expected = length == 512 and branch_index > 1
        extension_tokens = row.get("extension_tokens_256")
        if extension_expected:
            if (
                not isinstance(extension_tokens, list)
                or len(extension_tokens) != 256
                or not all(isinstance(token, int) for token in extension_tokens)
            ):
                raise ValueError("off-spine 512 extension is not auditable")
            continuation = (
                list(reference["first_256"]) + list(extension_tokens)
            )
            expected_extension_seed = stable_seed(
                global_seed,
                STUDY_ID,
                reference["checkpoint_id"],
                branch_index,
                512,
                "extension",
            )
            if (
                row.get("extension_seed") != expected_extension_seed
                or row.get("extension_batch_capacity") != EXTENSION_BATCH
                or row.get("extension_batch_size") != EXTENSION_BATCH
                or not 0
                <= int(row.get("extension_batch_index", -1))
                < 4
                or not 0
                <= int(row.get("extension_batch_position", -1))
                < EXTENSION_BATCH
            ):
                raise ValueError("off-spine extension provenance mismatch")
            extension_batches[int(row["extension_batch_index"])].append(
                (
                    int(row["extension_batch_position"]),
                    str(reference["checkpoint_id"]),
                    branch_index,
                )
            )
        else:
            if any(
                row.get(field) is not None
                for field in (
                    "extension_tokens_256",
                    "extension_seed",
                    "extension_batch_capacity",
                    "extension_batch_index",
                    "extension_batch_position",
                    "extension_batch_size",
                )
            ):
                raise ValueError("non-extension row charges an extension")
            if length == 128:
                continuation = list(reference["first_256"])[:128]
            elif length == 256:
                continuation = list(reference["first_256"])
            else:
                continuation = list(reference["first_256"]) + list(
                    reference["on_spine_second_256"]
                )
        if (
            len(continuation) != length
            or row.get("reasoning_token_sha256")
            != payload_sha256(continuation)
        ):
            raise ValueError("reasoning continuation digest mismatch")

        answer_token_ids = row.get("answer_token_ids")
        if (
            not isinstance(answer_token_ids, list)
            or len(answer_token_ids) > ANSWER_MAX_TOKENS
            or not all(isinstance(token, int) for token in answer_token_ids)
            or row.get("answer_tokens") != len(answer_token_ids)
        ):
            raise ValueError("answer-token accounting mismatch")
        expected_answer_seed = stable_seed(
            global_seed,
            STUDY_ID,
            reference["checkpoint_id"],
            branch_index,
            length,
            "answer",
        )
        if (
            row.get("answer_seed") != expected_answer_seed
            or row.get("answer_batch_capacity") != ANSWER_BATCH
            or row.get("answer_batch_size") != ANSWER_BATCH
            or not math.isclose(
                float(row.get("answer_batch_utilization", -1.0)),
                1.0,
            )
            or not 4 <= int(row.get("answer_batch_index", -1)) < 32
            or not 0
            <= int(row.get("answer_batch_position", -1))
            < ANSWER_BATCH
        ):
            raise ValueError("matched answer-probe provenance mismatch")
        answer_batches[int(row["answer_batch_index"])].append(
            (
                int(row["answer_batch_position"]),
                str(reference["checkpoint_id"]),
                branch_index,
                length,
            )
        )
        formal_tokens = (
            reference["formal_answer_tokens_256"]
            if length == 256
            else (
                reference["formal_answer_tokens_512"]
                if length == 512 and branch_index == 1
                else None
            )
        )
        expected_match = (
            answer_token_ids == formal_tokens
            if formal_tokens is not None
            else None
        )
        if row.get("formal_answer_token_match") != expected_match:
            raise ValueError("formal answer parity bit mismatch")
        if (
            row.get("logical_tokens") != length + len(answer_token_ids)
            or row.get("physical_reasoning_tokens")
            != (256 if extension_expected else 0)
            or row.get("physical_answer_tokens") != len(answer_token_ids)
            or row.get("physical_request_count")
            != 1 + int(extension_expected)
        ):
            raise ValueError("row ledger accounting mismatch")
        decoded_text = row.get("answer_decoded_text")
        if (
            not isinstance(decoded_text, str)
            or not isinstance(
                row.get("answer_backend_text_matches_decode"),
                bool,
            )
            or row.get("answer_decode_binding_sha256")
            != _decode_binding(answer_token_ids, decoded_text)
            or row.get("raw_answer") != CANONICAL_CUE + decoded_text
        ):
            raise ValueError("answer text/token decode binding mismatch")
        answer_backend_text_mismatches += int(
            not row["answer_backend_text_matches_decode"]
        )
        checkpoint_id = str(row["checkpoint_id"])
        stop_token_ids = row.get("stop_answer_token_ids")
        stop_decoded_text = row.get("stop_decoded_text")
        stop_raw = row.get("stop_raw_answer")
        if (
            stop_token_ids != reference["stop_answer_tokens"]
            or not isinstance(stop_decoded_text, str)
            or row.get("stop_decode_binding_sha256")
            != _decode_binding(stop_token_ids, stop_decoded_text)
            or stop_raw != CANONICAL_CUE + stop_decoded_text
        ):
            raise ValueError("stop text/token decode binding mismatch")
        stop_identity = (
            tuple(stop_token_ids),
            stop_decoded_text,
            str(row["stop_decode_binding_sha256"]),
        )
        previous_stop = stop_answers.setdefault(
            checkpoint_id,
            stop_identity,
        )
        if previous_stop != stop_identity:
            raise ValueError("stop answer varies within a checkpoint")
        gpu_seconds = float(row.get("gpu_seconds", -1.0))
        peak_memory = float(row.get("peak_gpu_memory_bytes", -1.0))
        if gpu_seconds < 0 or peak_memory < 0:
            raise ValueError("negative physical-compute measurement")
        total_gpu_seconds += gpu_seconds
        peak_gpu_memory = max(peak_gpu_memory, peak_memory)
        output_tokens += int(row["physical_reasoning_tokens"]) + int(
            row["physical_answer_tokens"]
        )
        requests += int(row["physical_request_count"])
    if observed != set(expected):
        raise ValueError("block-length row coverage mismatch")
    if set(extension_batches) != set(range(4)):
        raise ValueError("extension batch coverage mismatch")
    for batch_index, batch_rows in extension_batches.items():
        positions = [row[0] for row in batch_rows]
        contexts = [(row[1], row[2]) for row in batch_rows]
        if (
            len(batch_rows) != EXTENSION_BATCH
            or sorted(positions) != list(range(EXTENSION_BATCH))
            or len(set(contexts)) != EXTENSION_BATCH
        ):
            raise ValueError(
                f"extension batch {batch_index} structure mismatch"
            )
    if set(answer_batches) != set(range(4, 32)):
        raise ValueError("answer batch coverage mismatch")
    matched_contexts: dict[tuple[str, int], int] = {}
    for batch_index, batch_rows in answer_batches.items():
        ordered = sorted(batch_rows)
        if (
            len(ordered) != ANSWER_BATCH
            or [row[0] for row in ordered]
            != list(range(ANSWER_BATCH))
        ):
            raise ValueError(
                f"answer batch {batch_index} structure mismatch"
            )
        for start in range(0, ANSWER_BATCH, len(LENGTHS)):
            group = ordered[start : start + len(LENGTHS)]
            contexts = {(row[1], row[2]) for row in group}
            lengths = [row[3] for row in group]
            if len(contexts) != 1 or lengths != list(LENGTHS):
                raise ValueError(
                    "128/256/512 probes are not matched in one batch"
                )
            context = next(iter(contexts))
            if context in matched_contexts:
                raise ValueError(
                    "answer context occurs in multiple matched batches"
                )
            matched_contexts[context] = batch_index
    expected_contexts = {
        (checkpoint_id, branch_index)
        for checkpoint_id, branch_index, _ in expected
    }
    if set(matched_contexts) != expected_contexts:
        raise ValueError("matched answer context coverage mismatch")
    return {
        "output_tokens": float(output_tokens),
        "gpu_seconds": total_gpu_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "requests": float(requests),
        "batches": float(
            len(extension_batches) + len(answer_batches)
        ),
        "answer_backend_text_mismatches": float(
            answer_backend_text_mismatches
        ),
    }


def _load_outputs(
    model_name: str,
    result_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    input_manifest, _, problems = _load_input(model_name)
    expected = {str(row["problem_id"]): row for row in problems}
    source_run = SOURCE_RUNS[model_name]
    source_audit = _assert_audited(source_run)
    if (
        source_audit["run_id"] != input_manifest["source_run_id"]
        or source_audit["contract_sha256"]
        != input_manifest["source_contract_sha256"]
        or source_audit["result_merkle_root"]
        != input_manifest["source_result_merkle_root"]
        or file_sha256(source_run / "audit_manifest.json")
        != input_manifest["source_audit_manifest_sha256"]
    ):
        raise ValueError("sealed length input no longer authenticates its source")
    for problem_id, problem in expected.items():
        source_problem = read_json_gz(
            source_run
            / "problems"
            / f"{_safe_problem_name(problem_id)}.json.gz"
        )
        if (
            not verify_sealed_record(source_problem)
            or source_problem["record_sha256"]
            != problem["source_problem_record_sha256"]
        ):
            raise ValueError("length input source problem identity mismatch")
    outputs: list[dict[str, Any]] = []
    worker_manifests: list[dict[str, Any]] = []
    observed: set[str] = set()
    runtime_ids: set[str] = set()
    generation_identity = _generation_identity(input_manifest)
    runtime_contract = input_manifest["runtime_contract"]
    for worker_index in range(WORKERS):
        worker_dir = result_dir / "workers" / f"worker-{worker_index:02d}"
        worker = _read_json(worker_dir / "manifest.json")
        if (
            not verify_sealed_record(worker)
            or worker.get("schema_version")
            != "flcva-block-length-worker-manifest-v4"
            or worker.get("complete") is not True
            or worker.get("study_id") != STUDY_ID
            or worker.get("model_name") != model_name
            or worker.get("model_id") != input_manifest["model_id"]
            or worker.get("model_revision")
            != input_manifest["model_revision"]
            or worker.get("worker_index") != worker_index
            or worker.get("workers") != WORKERS
            or worker.get("input_sha256") != input_manifest["input_sha256"]
            or worker.get("generation_identity") != generation_identity
        ):
            raise ValueError("invalid block-length worker manifest")
        runtime = worker["runtime_identity"]
        if (
            not _identity_is_valid(runtime)
            or _runtime_contract_differences(
                runtime,
                runtime_contract,
            )
            or _snapshot_identity(worker["runtime_model_identity"])
            != runtime_contract["model_snapshot"]
        ):
            raise ValueError("worker runtime is not the sealed runtime")
        runtime_ids.add(str(runtime["identity_sha256"]))
        worker_ids: list[str] = []
        worker_rows = 0
        worker_accounting: dict[str, float] = defaultdict(float)
        for item in worker["problem_files"]:
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe block-length result path")
            path = worker_dir / relative
            if file_sha256(path) != item["sha256"]:
                raise ValueError("block-length result digest mismatch")
            payload = read_json_gz(path)
            if (
                not verify_sealed_record(payload)
                or payload.get("schema_version")
                != "flcva-block-length-output-problem-v4"
                or payload.get("study_id") != STUDY_ID
                or payload.get("model_name") != model_name
                or payload.get("model_id") != input_manifest["model_id"]
                or payload.get("model_revision")
                != input_manifest["model_revision"]
                or payload.get("worker_index") != worker_index
                or payload.get("input_sha256")
                != input_manifest["input_sha256"]
                or payload.get("generation_identity_sha256")
                != generation_identity["identity_sha256"]
                or payload.get("runtime_identity_sha256")
                != runtime["identity_sha256"]
            ):
                raise ValueError("invalid block-length problem result")
            problem_id = str(payload["problem_id"])
            if (
                problem_id not in expected
                or problem_id in observed
                or item.get("problem_id") != problem_id
                or payload.get("input_problem_record_sha256")
                != expected[problem_id]["record_sha256"]
                or payload.get("dataset_id")
                != expected[problem_id]["dataset_id"]
                or payload.get("split") != expected[problem_id]["split"]
            ):
                raise ValueError("block-length result coverage mismatch")
            rows = payload["results"]
            if len(rows) != 672 or item.get("rows") != 672:
                raise ValueError("block-length result row count mismatch")
            recomputed = _validate_problem_rows(
                rows,
                expected[problem_id],
                global_seed=int(input_manifest["global_seed"]),
            )
            if not _numeric_maps_equal(
                recomputed,
                payload["physical_accounting"],
            ):
                raise ValueError("problem physical accounting mismatch")
            for key, value in recomputed.items():
                if key == "peak_gpu_memory_bytes":
                    worker_accounting[key] = max(
                        worker_accounting[key], value
                    )
                else:
                    worker_accounting[key] += value
            outputs.extend(rows)
            observed.add(problem_id)
            worker_ids.append(problem_id)
            worker_rows += len(rows)
        expected_worker = sorted(
            problem_id
            for problem_id in expected
            if _worker_bucket(problem_id) == worker_index
        )
        if (
            sorted(worker_ids) != expected_worker
            or worker["problem_ids_sha256"]
            != payload_sha256(expected_worker)
            or worker.get("problems") != len(expected_worker)
            or worker["rows"] != worker_rows
            or not _numeric_maps_equal(
                dict(worker_accounting),
                worker["physical_accounting"],
            )
        ):
            raise ValueError("block-length worker coverage mismatch")
        worker_manifests.append(worker)
    if observed != set(expected) or len(runtime_ids) != 1:
        raise ValueError("block-length model output is incomplete")
    return input_manifest, outputs, worker_manifests


def _checkpoint_values(
    model_name: str,
    outputs: list[dict[str, Any]],
    labels: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in outputs:
        grouped[
            (
                str(row["problem_id"]),
                str(row["checkpoint_id"]),
                int(row["length"]),
            )
        ].append(row)
    values: list[dict[str, Any]] = []
    for (problem_id, checkpoint_id, length), rows in grouped.items():
        if len(rows) != 4:
            raise ValueError("length checkpoint lacks four branches")
        dataset_id, gold = labels[problem_id]
        stop_correct, _, _ = answers_equal(
            dataset_id,
            CANONICAL_CUE + str(rows[0]["stop_decoded_text"]),
            gold,
        )
        continuation_correct: list[int] = []
        costs: list[int] = []
        for row in sorted(rows, key=lambda item: int(item["branch_index"])):
            correct, _, _ = answers_equal(
                dataset_id,
                CANONICAL_CUE + str(row["answer_decoded_text"]),
                gold,
            )
            continuation_correct.append(int(correct))
            costs.append(int(row["logical_tokens"]))
        stop = int(stop_correct)
        mean_correct = float(np.mean(continuation_correct))
        mean_cost = float(np.mean(costs))
        values.append(
            {
                "model": model_name,
                "problem_id": problem_id,
                "dataset_id": dataset_id,
                "split": rows[0]["split"],
                "checkpoint_id": checkpoint_id,
                "spine_id": int(rows[0]["spine_id"]),
                "depth": int(rows[0]["depth"]),
                "length": length,
                "stop_correct": stop,
                "continuation_correct": continuation_correct,
                "continuation_costs": costs,
                "formal_answer_token_match": [
                    row["formal_answer_token_match"]
                    for row in sorted(
                        rows, key=lambda item: int(item["branch_index"])
                    )
                ],
                "marginal_value": (
                    1000.0 * (mean_correct - stop) / mean_cost
                ),
                "single_draw_recovery": (
                    mean_correct if stop == 0 else None
                ),
                "single_draw_damage": (
                    1.0 - mean_correct if stop == 1 else None
                ),
                "recovery_detected": (
                    int(any(continuation_correct)) if stop == 0 else None
                ),
                "damage_detected": (
                    int(not all(continuation_correct))
                    if stop == 1
                    else None
                ),
                "proxies": rows[0]["proxies"],
            }
        )
    expected = 400 * 4 * len(COMMON_DEPTHS) * len(LENGTHS)
    if len(values) != expected:
        raise ValueError(
            f"expected {expected} checkpoint values, observed {len(values)}"
        )
    return values


def _sign(value: float) -> int:
    return int(value > 0) - int(value < 0)


def _formal_256_parity(
    model_name: str,
    values: list[dict[str, Any]],
) -> dict[str, Any]:
    formal = {
        str(row["checkpoint_id"]): row
        for row in _iter_jsonl_gz(
            SOURCE_RUNS[model_name] / "evaluation" / "values.jsonl.gz"
        )
    }
    rerun = {
        str(row["checkpoint_id"]): row
        for row in values
        if int(row["length"]) == 256
    }
    if not rerun or not set(rerun).issubset(formal):
        raise ValueError("256-token parity checkpoint universe mismatch")
    stop_matches = 0
    branch_matches = 0
    branch_total = 0
    sign_matches = 0
    token_matches = 0
    token_total = 0
    absolute_value_delta: list[float] = []
    for checkpoint_id, row in rerun.items():
        source = formal[checkpoint_id]
        stop_matches += int(
            int(row["stop_correct"]) == int(source["stop_correct"])
        )
        rerun_correct = [int(value) for value in row["continuation_correct"]]
        source_correct = [
            int(value) for value in source["continuation_correct"]
        ]
        if len(rerun_correct) != 4 or len(source_correct) != 4:
            raise ValueError("256-token parity branch cardinality mismatch")
        branch_matches += sum(
            int(left == right)
            for left, right in zip(
                rerun_correct, source_correct, strict=True
            )
        )
        branch_total += 4
        sign_matches += int(
            _sign(float(row["marginal_value"]))
            == _sign(float(source["marginal_value"]))
        )
        parity_bits = row["formal_answer_token_match"]
        if len(parity_bits) != 4 or any(
            not isinstance(value, bool) for value in parity_bits
        ):
            raise ValueError("256-token answer parity bits are incomplete")
        token_matches += sum(int(value) for value in parity_bits)
        token_total += 4
        absolute_value_delta.append(
            abs(
                float(row["marginal_value"])
                - float(source["marginal_value"])
            )
        )
    checkpoints = len(rerun)
    stop_rate = stop_matches / checkpoints
    branch_rate = branch_matches / branch_total
    sign_rate = sign_matches / checkpoints
    token_rate = token_matches / token_total
    passed = (
        stop_rate == 1.0
        and branch_rate >= MIN_256_BRANCH_CORRECTNESS_PARITY
        and sign_rate >= MIN_256_CHECKPOINT_SIGN_PARITY
    )
    return {
        "schema_version": "flcva-block-length-256-parity-v4",
        "model": model_name,
        "checkpoints": checkpoints,
        "branches": branch_total,
        "stop_correctness_identity": stop_rate,
        "branch_correctness_identity": branch_rate,
        "checkpoint_sign_identity": sign_rate,
        "answer_token_identity": token_rate,
        "mean_absolute_marginal_value_delta": float(
            np.mean(absolute_value_delta)
        ),
        "thresholds": {
            "stop_correctness_identity": 1.0,
            "branch_correctness_identity": (
                MIN_256_BRANCH_CORRECTNESS_PARITY
            ),
            "checkpoint_sign_identity": MIN_256_CHECKPOINT_SIGN_PARITY,
        },
        "passed": passed,
    }


def _feature_state(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(
        [
            [
                float(row["depth"]) / 4096.0,
                float(row["dataset_id"] == "math500"),
            ]
            for row in rows
        ],
        dtype=float,
    )
    medians = np.nanmedian(raw, axis=0)
    missing = np.isnan(raw).astype(float)
    filled = np.where(np.isnan(raw), medians, raw)
    scale = filled.std(axis=0)
    scale[scale == 0] = 1.0
    center = filled.mean(axis=0)
    matrix = np.column_stack(
        [
            np.ones(len(rows)),
            (filled - center) / scale,
            missing,
        ]
    )
    return matrix, medians, center, scale


def _transform_features(
    rows: list[dict[str, Any]],
    medians: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(
        [
            [
                float(row["depth"]) / 4096.0,
                float(row["dataset_id"] == "math500"),
            ]
            for row in rows
        ],
        dtype=float,
    )
    missing = np.isnan(raw).astype(float)
    filled = np.where(np.isnan(raw), medians, raw)
    return np.column_stack(
        [np.ones(len(rows)), (filled - center) / scale, missing]
    )


def _adaptive_rows(
    values: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    analysis = [row for row in values if row["split"] == "analysis"]
    confirmation = [
        row for row in values if row["split"] == "confirmation"
    ]
    by_length_analysis = {
        length: sorted(
            [row for row in analysis if row["length"] == length],
            key=lambda row: str(row["checkpoint_id"]),
        )
        for length in LENGTHS
    }
    by_length_confirmation = {
        length: sorted(
            [row for row in confirmation if row["length"] == length],
            key=lambda row: str(row["checkpoint_id"]),
        )
        for length in LENGTHS
    }
    reference_ids = [
        str(row["checkpoint_id"])
        for row in by_length_confirmation[256]
    ]
    predictions: dict[int, dict[str, float]] = {}
    fits: dict[str, Any] = {}
    for length in LENGTHS:
        train = by_length_analysis[length]
        test = by_length_confirmation[length]
        design, medians, center, scale = _feature_state(train)
        target = np.asarray(
            [float(row["marginal_value"]) for row in train],
            dtype=float,
        )
        penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ target,
        )
        predicted = (
            _transform_features(test, medians, center, scale)
            @ coefficients
        )
        ids = [str(row["checkpoint_id"]) for row in test]
        if ids != reference_ids:
            raise ValueError("adaptive length checkpoint order mismatch")
        predictions[length] = dict(zip(ids, predicted, strict=True))
        fits[str(length)] = {
            "alpha": RIDGE_ALPHA,
            "analysis_rows": len(train),
            "confirmation_rows": len(test),
            "features": list(ADAPTIVE_FEATURES),
            "coefficients": coefficients.tolist(),
            "medians": medians.tolist(),
            "center": center.tolist(),
            "scale": scale.tolist(),
        }
    lookup = {
        (str(row["checkpoint_id"]), int(row["length"])): row
        for row in confirmation
    }
    adaptive: list[dict[str, Any]] = []
    choices: dict[int, int] = defaultdict(int)
    for checkpoint_id in reference_ids:
        chosen = max(
            LENGTHS,
            key=lambda length: (
                predictions[length][checkpoint_id],
                -length,
            ),
        )
        choices[chosen] += 1
        source = dict(lookup[(checkpoint_id, chosen)])
        source["source_length"] = chosen
        source["length"] = "adaptive"
        source["predicted_values"] = {
            str(length): predictions[length][checkpoint_id]
            for length in LENGTHS
        }
        adaptive.append(source)
    fits["selection"] = {
        "rule": "argmax predicted token-normalized marginal value",
        "tie_break": "shorter length",
        "confirmation_labels_used": False,
        "audited_proxies_used": False,
        "interpretation": (
            "context-only adaptive arm; excluding audited proxies prevents "
            "policy-conditioned proxy-ranking circularity"
        ),
        "choices": dict(sorted(choices.items())),
    }
    return adaptive, fits


def _percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _cluster_interval(
    rows: list[dict[str, Any]],
    field: str,
    *,
    seed_parts: tuple[str, ...],
) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            grouped[str(row["problem_id"])].append(float(value))
    keys = sorted(grouped)
    observed = float(
        np.mean([value for key in keys for value in grouped[key]])
    )
    rng = random.Random(stable_seed(20251231, STUDY_ID, *seed_parts))
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [rng.choice(keys) for _ in keys]
        estimates.append(
            float(
                np.mean(
                    [
                        value
                        for key in sampled
                        for value in grouped[key]
                    ]
                )
            )
        )
    return observed, _percentile(estimates, 0.025), _percentile(
        estimates, 0.975
    )


def _paired_cluster_difference(
    treatment: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    field: str,
    *,
    seed_parts: tuple[str, ...],
) -> tuple[float, float, float]:
    def aggregate(
        rows: list[dict[str, Any]],
    ) -> dict[str, tuple[float, int]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = row.get(field)
            if value is not None:
                grouped[str(row["problem_id"])].append(float(value))
        return {
            problem_id: (sum(values), len(values))
            for problem_id, values in grouped.items()
        }

    left = aggregate(treatment)
    right = aggregate(baseline)
    keys = sorted(set(left) & set(right))
    if set(left) != set(right) or not keys:
        raise ValueError("paired length comparison problem sets differ")

    def difference(sampled: list[str]) -> float:
        left_sum = sum(left[key][0] for key in sampled)
        left_count = sum(left[key][1] for key in sampled)
        right_sum = sum(right[key][0] for key in sampled)
        right_count = sum(right[key][1] for key in sampled)
        return left_sum / left_count - right_sum / right_count

    observed = difference(keys)
    rng = random.Random(stable_seed(20251231, STUDY_ID, *seed_parts))
    estimates = [
        difference([rng.choice(keys) for _ in keys])
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return observed, _percentile(estimates, 0.025), _percentile(
        estimates, 0.975
    )


def _event_metrics(
    model_name: str,
    values: list[dict[str, Any]],
    adaptive: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    confirmation = [
        row for row in values if row["split"] == "confirmation"
    ]
    conditions: dict[str, list[dict[str, Any]]] = {
        str(length): [
            row for row in confirmation if row["length"] == length
        ]
        for length in LENGTHS
    }
    conditions["adaptive"] = adaptive
    output: list[dict[str, Any]] = []
    for condition, rows in conditions.items():
        for target, stop, field, detected in (
            (
                "recovery",
                0,
                "single_draw_recovery",
                "recovery_detected",
            ),
            ("damage", 1, "single_draw_damage", "damage_detected"),
        ):
            eligible = [
                row for row in rows if int(row["stop_correct"]) == stop
            ]
            estimate, low, high = _cluster_interval(
                eligible,
                field,
                seed_parts=(model_name, condition, target, "single"),
            )
            detection, detection_low, detection_high = _cluster_interval(
                eligible,
                detected,
                seed_parts=(model_name, condition, target, "detected"),
            )
            baseline = [
                row
                for row in conditions["256"]
                if int(row["stop_correct"]) == stop
            ]
            if condition == "256":
                difference = difference_low = difference_high = 0.0
            else:
                (
                    difference,
                    difference_low,
                    difference_high,
                ) = _paired_cluster_difference(
                    eligible,
                    baseline,
                    field,
                    seed_parts=(
                        model_name,
                        condition,
                        target,
                        "difference-vs-256",
                    ),
                )
            output.append(
                {
                    "model": model_name,
                    "condition": condition,
                    "target": target,
                    "checkpoints": len(eligible),
                    "single_draw_rate": estimate,
                    "single_draw_low": low,
                    "single_draw_high": high,
                    "difference_vs_256": difference,
                    "difference_vs_256_low": difference_low,
                    "difference_vs_256_high": difference_high,
                    "four_draw_detection_rate": detection,
                    "four_draw_detection_low": detection_low,
                    "four_draw_detection_high": detection_high,
                }
            )
    return output


def _average_ranks(values: list[float], *, descending: bool) -> list[float]:
    array = np.asarray(values, dtype=float)
    if descending:
        array = -array
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks.tolist()


def _proxy_metrics(
    model_name: str,
    values: list[dict[str, Any]],
    adaptive: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    orientations: dict[tuple[str, str], int] = {}
    analysis_256 = [
        row
        for row in values
        if row["split"] == "analysis" and row["length"] == 256
    ]
    for target, stop, label_field in (
        ("recovery", 0, "recovery_detected"),
        ("damage", 1, "damage_detected"),
    ):
        eligible = [
            row for row in analysis_256 if row["stop_correct"] == stop
        ]
        for proxy in PROXIES:
            available = [
                row
                for row in eligible
                if row["proxies"].get(proxy) is not None
            ]
            auc = roc_auc(
                [int(row[label_field]) for row in available],
                [float(row["proxies"][proxy]) for row in available],
            )
            orientations[(target, proxy)] = 1 if auc >= 0.5 else -1

    confirmation = [
        row for row in values if row["split"] == "confirmation"
    ]
    conditions: dict[str, list[dict[str, Any]]] = {
        str(length): [
            row for row in confirmation if row["length"] == length
        ]
        for length in LENGTHS
    }
    conditions["adaptive"] = adaptive
    rows: list[dict[str, Any]] = []
    for condition, condition_rows in conditions.items():
        for target, stop, label_field in (
            ("recovery", 0, "recovery_detected"),
            ("damage", 1, "damage_detected"),
        ):
            eligible = [
                row
                for row in condition_rows
                if int(row["stop_correct"]) == stop
            ]
            target_rows: list[dict[str, Any]] = []
            for proxy in PROXIES:
                available = [
                    row
                    for row in eligible
                    if row["proxies"].get(proxy) is not None
                ]
                orientation = orientations[(target, proxy)]
                target_rows.append(
                    {
                        "model": model_name,
                        "condition": condition,
                        "target": target,
                        "proxy": proxy,
                        "orientation": orientation,
                        "available": len(available),
                        "auroc": roc_auc(
                            [int(row[label_field]) for row in available],
                            [
                                orientation
                                * float(row["proxies"][proxy])
                                for row in available
                            ],
                        ),
                    }
                )
            ranks = _average_ranks(
                [float(row["auroc"]) for row in target_rows],
                descending=True,
            )
            for row, rank in zip(target_rows, ranks, strict=True):
                row["rank"] = rank
                rows.append(row)

    stability: list[dict[str, Any]] = []
    for target in ("recovery", "damage"):
        baseline = {
            str(row["proxy"]): float(row["rank"])
            for row in rows
            if row["condition"] == "256" and row["target"] == target
        }
        for condition in ("128", "512", "adaptive"):
            selected = {
                str(row["proxy"]): float(row["rank"])
                for row in rows
                if row["condition"] == condition
                and row["target"] == target
            }
            proxies = list(PROXIES)
            baseline_ranks = np.asarray(
                [baseline[proxy] for proxy in proxies], dtype=float
            )
            selected_ranks = np.asarray(
                [selected[proxy] for proxy in proxies], dtype=float
            )
            top_256 = min(proxies, key=baseline.__getitem__)
            top_condition = min(proxies, key=selected.__getitem__)
            stability.append(
                {
                    "model": model_name,
                    "target": target,
                    "condition": condition,
                    "spearman_rank_correlation_vs_256": float(
                        np.corrcoef(
                            baseline_ranks,
                            selected_ranks,
                        )[0, 1]
                    ),
                    "top_proxy_256": top_256,
                    "top_proxy_condition": top_condition,
                    "top_proxy_stable": int(
                        top_256 == top_condition
                    ),
                }
            )

    stability_index = {
        (str(row["target"]), str(row["condition"])): row
        for row in stability
    }
    problem_ids = sorted(
        {
            str(row["problem_id"])
            for row in conditions["256"]
        }
    )
    if len(problem_ids) < 4:
        raise ValueError("too few confirmation problems for rank stability")
    half_size = len(problem_ids) // 2
    robustness: dict[tuple[str, str], dict[str, list[float]]] = {
        (target, condition): {"rho": [], "top": []}
        for target in ("recovery", "damage")
        for condition in ("128", "512", "adaptive")
    }

    def subset_ranks(
        condition: str,
        *,
        target: str,
        selected_ids: set[str],
    ) -> dict[str, float]:
        stop = 0 if target == "recovery" else 1
        label_field = (
            "recovery_detected"
            if target == "recovery"
            else "damage_detected"
        )
        eligible = [
            row
            for row in conditions[condition]
            if int(row["stop_correct"]) == stop
            and str(row["problem_id"]) in selected_ids
        ]
        aucs: dict[str, float] = {}
        for proxy in PROXIES:
            available = [
                row
                for row in eligible
                if row["proxies"].get(proxy) is not None
            ]
            auc = roc_auc(
                [int(row[label_field]) for row in available],
                [
                    orientations[(target, proxy)]
                    * float(row["proxies"][proxy])
                    for row in available
                ],
            )
            if auc is None or not math.isfinite(float(auc)):
                raise ValueError(
                    "proxy rank is undefined in a deterministic half-sample"
                )
            aucs[proxy] = float(auc)
        ranks = _average_ranks(
            [aucs[proxy] for proxy in PROXIES],
            descending=True,
        )
        return dict(zip(PROXIES, ranks, strict=True))

    for replicate in range(RANK_SUBSAMPLE_REPLICATES):
        selected_ids = set(
            sorted(
                problem_ids,
                key=lambda problem_id: hashlib.sha256(
                    (
                        f"{STUDY_ID}:rank-half:{model_name}:"
                        f"{replicate}:{problem_id}"
                    ).encode()
                ).hexdigest(),
            )[:half_size]
        )
        for target in ("recovery", "damage"):
            baseline = subset_ranks(
                "256",
                target=target,
                selected_ids=selected_ids,
            )
            baseline_array = np.asarray(
                [baseline[proxy] for proxy in PROXIES], dtype=float
            )
            baseline_top = min(PROXIES, key=baseline.__getitem__)
            for condition in ("128", "512", "adaptive"):
                selected = subset_ranks(
                    condition,
                    target=target,
                    selected_ids=selected_ids,
                )
                selected_array = np.asarray(
                    [selected[proxy] for proxy in PROXIES], dtype=float
                )
                rho = float(
                    np.corrcoef(baseline_array, selected_array)[0, 1]
                )
                if not math.isfinite(rho):
                    raise ValueError(
                        "proxy rank correlation is undefined in a half-sample"
                    )
                cell = robustness[(target, condition)]
                cell["rho"].append(rho)
                cell["top"].append(
                    float(
                        baseline_top
                        == min(PROXIES, key=selected.__getitem__)
                    )
                )
    for key, cell in robustness.items():
        row = stability_index[key]
        row["half_sample_replicates"] = RANK_SUBSAMPLE_REPLICATES
        row["half_sample_rho_low"] = _percentile(
            cell["rho"], 0.025
        )
        row["half_sample_rho_high"] = _percentile(
            cell["rho"], 0.975
        )
        row["half_sample_top_proxy_stability"] = float(
            np.mean(cell["top"])
        )
    return rows, stability


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(
            descriptor, "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_tables(
    event_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    event_path = GENERATED_DIR / "block_length_event_table.tex"
    stability_path = GENERATED_DIR / "block_length_proxy_table.tex"
    model_labels = {"qwen3": "Qwen3", "deepseek": "DeepSeek"}
    conditions = ("128", "256", "512", "adaptive")
    event_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Model & Length & Recovery & Damage \\",
        r"\midrule",
    ]
    for model in SOURCE_RUNS:
        for condition in conditions:
            recovery = next(
                row
                for row in event_rows
                if row["model"] == model
                and row["condition"] == condition
                and row["target"] == "recovery"
            )
            damage = next(
                row
                for row in event_rows
                if row["model"] == model
                and row["condition"] == condition
                and row["target"] == "damage"
            )
            event_lines.append(
                "{} & {} & {:.1f}\\% [{:.1f}, {:.1f}] & "
                "{:.1f}\\% [{:.1f}, {:.1f}] \\\\".format(
                    model_labels[model],
                    (
                        "Adaptive"
                        if condition == "adaptive"
                        else f"{condition} tokens"
                    ),
                    100 * recovery["single_draw_rate"],
                    100 * recovery["single_draw_low"],
                    100 * recovery["single_draw_high"],
                    100 * damage["single_draw_rate"],
                    100 * damage["single_draw_low"],
                    100 * damage["single_draw_high"],
                )
            )
        event_lines.append(r"\addlinespace")
    event_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Continuation-length sensitivity on the common "
                r"depth lattice. Rates describe one uniformly sampled branch; "
                r"intervals resample complete problems 2,000 times. The "
                r"context-only adaptive length is selected by an analysis-only "
                r"Ridge model using depth and dataset, without confirmation "
                r"labels or any audited proxy.}"
            ),
            r"\label{tab:block-length-events}",
            r"\end{table}",
            "",
        ]
    )
    _atomic_text(event_path, "\n".join(event_lines))

    stability_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        (
            r"Model & Target & Condition & Rank $\rho$ vs.\ 256 "
            r"& Top proxy (half-sample stability) \\"
        ),
        r"\midrule",
    ]
    for row in stability_rows:
        stability_lines.append(
            "{} & {} & {} & {:.2f} [{:.2f}, {:.2f}] & {} ({:.0f}\\%) "
            "\\\\".format(
                model_labels[str(row["model"])],
                str(row["target"]).capitalize(),
                (
                    "Adaptive"
                    if row["condition"] == "adaptive"
                    else f"{row['condition']} tokens"
                ),
                row["spearman_rank_correlation_vs_256"],
                row["half_sample_rho_low"],
                row["half_sample_rho_high"],
                str(row["top_proxy_condition"]).replace("_", r"\_"),
                100 * row["half_sample_top_proxy_stability"],
            )
        )
    stability_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Stability of the six-proxy AUROC ranking. Proxy "
                r"orientations are selected once on the 256-token analysis "
                r"arm and frozen for all confirmation conditions. The adaptive "
                r"condition excludes all six audited proxies from its selector, "
                r"avoiding policy-conditioned ranking circularity. Brackets and "
                r"parentheses summarize 64 deterministic paired half-problem "
                r"subsamples and are robustness ranges, not confidence intervals.}"
            ),
            r"\label{tab:block-length-proxies}",
            r"\end{table}",
            "",
        ]
    )
    _atomic_text(stability_path, "\n".join(stability_lines))
    return event_path, stability_path


def evaluate(qwen_results: Path, deepseek_results: Path) -> None:
    labels = _load_labels()
    result_dirs = {
        "qwen3": qwen_results,
        "deepseek": deepseek_results,
    }
    all_values: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_proxy_rows: list[dict[str, Any]] = []
    all_stability: list[dict[str, Any]] = []
    parities: dict[str, dict[str, Any]] = {}
    fits: dict[str, Any] = {}
    source: dict[str, Any] = {}
    for model_name, result_dir in result_dirs.items():
        manifest, outputs, worker_manifests = _load_outputs(
            model_name,
            result_dir,
        )
        values = _checkpoint_values(model_name, outputs, labels)
        parities[model_name] = _formal_256_parity(model_name, values)
        adaptive, model_fits = _adaptive_rows(values)
        events = _event_metrics(model_name, values, adaptive)
        proxy_rows, stability = _proxy_metrics(
            model_name,
            values,
            adaptive,
        )
        all_values.extend(values)
        all_values.extend(adaptive)
        all_events.extend(events)
        all_proxy_rows.extend(proxy_rows)
        all_stability.extend(stability)
        fits[model_name] = model_fits
        source[model_name] = {
            "input_manifest_record_sha256": manifest["record_sha256"],
            "source_result_merkle_root": manifest[
                "source_result_merkle_root"
            ],
            "source_values_sha256": file_sha256(
                SOURCE_RUNS[model_name]
                / "evaluation"
                / "values.jsonl.gz"
            ),
            "worker_manifest_record_sha256": sorted(
                row["record_sha256"] for row in worker_manifests
            ),
        }

    parity_path = GENERATED_DIR / "block_length_256_parity.json"
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    parity_manifest = seal_record(
        {
            "schema_version": "flcva-block-length-256-parity-manifest-v4",
            "study_id": STUDY_ID,
            "models": parities,
            "passed": all(row["passed"] for row in parities.values()),
        }
    )
    _atomic_json(parity_path, parity_manifest)
    if not parity_manifest["passed"]:
        raise ValueError(
            "256-token rerun failed the frozen formal parity gate; "
            f"inspect {parity_path}"
        )

    values_path = GENERATED_DIR / "block_length_values.jsonl.gz"
    events_path = GENERATED_DIR / "block_length_event_metrics.csv"
    proxies_path = GENERATED_DIR / "block_length_proxy_metrics.csv"
    stability_path = GENERATED_DIR / "block_length_proxy_stability.csv"
    fits_path = GENERATED_DIR / "block_length_adaptive_fits.json"
    _write_jsonl_gz(values_path, all_values)
    _write_csv(events_path, all_events)
    _write_csv(proxies_path, all_proxy_rows)
    _write_csv(stability_path, all_stability)
    _atomic_json(fits_path, fits)
    event_table, proxy_table = _render_tables(
        all_events, all_stability
    )
    artifacts = (
        values_path,
        events_path,
        proxies_path,
        stability_path,
        fits_path,
        event_table,
        proxy_table,
        parity_path,
    )
    paper_manifest = seal_record(
        {
            "schema_version": "flcva-block-length-paper-manifest-v4",
            "study_id": STUDY_ID,
            "analysis_status": (
                "post-seal length robustness; analysis-only adaptive fit"
            ),
            "lengths": list(LENGTHS),
            "common_depths": list(COMMON_DEPTHS),
            "adaptive_policy": (
                "per-length context-only Ridge(alpha=10), using depth and "
                "dataset but none of the audited proxies, fitted on analysis "
                "problems; choose maximum predicted token-normalized value"
            ),
            "confirmation_labels_used_for_fit": False,
            "audited_proxies_used_for_adaptive_selection": False,
            "proxy_orientation": (
                "selected on 256-token analysis arm and frozen"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "proxy_rank_half_sample_replicates": (
                RANK_SUBSAMPLE_REPLICATES
            ),
            "formal_256_parity_record_sha256": parity_manifest[
                "record_sha256"
            ],
            "sealed_labels_sha256": file_sha256(LABELS_PATH),
            "source": source,
            "artifacts_sha256": {
                path.name: file_sha256(path) for path in artifacts
            },
        }
    )
    _atomic_json(
        GENERATED_DIR / "block_length_sensitivity_manifest.json",
        paper_manifest,
    )
    print(
        json.dumps(
            {
                "manifest": "paper/generated/"
                "block_length_sensitivity_manifest.json",
                "record_sha256": paper_manifest["record_sha256"],
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    prepare_model_parser = subparsers.add_parser("prepare-model")
    prepare_model_parser.add_argument(
        "--model", choices=tuple(SOURCE_RUNS), required=True
    )
    subparsers.add_parser("validate-inputs")
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--model", choices=tuple(SOURCE_RUNS), required=True
    )
    generate_parser.add_argument("--worker-index", type=int, required=True)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--qwen-results", type=Path, required=True)
    evaluate_parser.add_argument(
        "--deepseek-results", type=Path, required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_all_inputs()
    elif args.command == "prepare-model":
        prepare_inputs((args.model,))
    elif args.command == "validate-inputs":
        validate_inputs()
    elif args.command == "generate":
        generate(args.model, args.worker_index, args.output_dir)
    elif args.command == "evaluate":
        evaluate(args.qwen_results, args.deepseek_results)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
