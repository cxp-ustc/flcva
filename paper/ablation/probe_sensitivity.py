#!/usr/bin/env python3
"""Prepare, run, and evaluate a post-seal answer-probe sensitivity study.

The GPU phase is deliberately label-blind.  ``prepare`` extracts only frozen
token prefixes and the original greedy probe outputs from two audited runs.
``generate`` reruns three answer cues on the same stop and continuation
contexts.  ``evaluate`` is local and label-aware; it never runs on the
generation workers.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

from flcva.backends.base import GenerationRequest
from flcva.backends.vllm import build_backend
from flcva.config import RunConfig
from flcva.contract import source_tree_fingerprint
from flcva.ids import canonical_json_bytes, payload_sha256, stable_seed
from flcva.io import (
    BlobStore,
    file_sha256,
    read_json_gz,
    seal_record,
    verify_sealed_record,
)
from flcva.parsing import answers_equal, parse_answer


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "paper" / "ablation" / "inputs"
GENERATED_DIR = ROOT / "paper" / "generated"
LABELS_PATH = ROOT / "data" / "sealed" / "combined_labels.jsonl"
DATA_MANIFEST_PATH = ROOT / "data" / "frozen" / "MANIFEST.json"
BOOTSTRAP_REPLICATES = 2_000
SAMPLE_QUOTAS = {"gsm8k": 150, "math500": 50}
WORKERS = 8
STUDY_ID = "flcva-probe-sensitivity-v5"
SELECTION_NAMESPACE = "flcva-probe-sensitivity-v1"
FORMAL_STOP_BATCH_SIZE = 16
FORMAL_BRANCH_BATCH_SIZE = 45
PROBE_TOP_LOGPROBS = 20
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
    "transport_attestation_sha256",
    "transport_requested_revision",
    "transport_resolution_mode",
    "transport_metadata_sha256",
    "config_sha256",
    "tokenizer_config_sha256",
)
SOURCE_RUNS = {
    "qwen3": ROOT / "artifacts" / "primary-qwen3",
    "deepseek": ROOT / "artifacts" / "primary-deepseek",
}
CONFIGS = {
    "qwen3": ROOT / "configs" / "formal-qwen3.yaml",
    "deepseek": ROOT / "configs" / "formal-deepseek.yaml",
}
CUES = {
    "canonical": r"Therefore, the final answer is \boxed{",
    "direct": "\nNow give only the final answer in " + r"\boxed{",
    "terse": "\nFinal answer: " + r"\boxed{",
}
PROBE_GATE_PATH = GENERATED_DIR / "probe_gate.tex"
PENDING_PROBE_GATE = (
    "% Default fail-closed gate. The sealed probe evaluator replaces this file.\n"
    "\\newif\\ifprobevalidated\n"
    "\\probevalidatedfalse\n"
)
PROBE_EXECUTION_FILES = (
    "paper/ablation/probe_sensitivity.py",
    "src/flcva/backends/base.py",
    "src/flcva/backends/vllm.py",
    "src/flcva/model_resolver.py",
)


def _json_line(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _safe_problem_name(problem_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in problem_id
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2
            ).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
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
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                handle.write(canonical_json_bytes(value))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
                for row in rows:
                    handle.write(_json_line(row))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _snapshot_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in SNAPSHOT_IDENTITY_KEYS}


def _probe_execution_bundle() -> dict[str, Any]:
    files = {
        relative: file_sha256(ROOT / relative)
        for relative in PROBE_EXECUTION_FILES
    }
    return {
        "files_sha256": files,
        "bundle_sha256": payload_sha256(sorted(files.items())),
    }


def _generation_identity(
    manifest: dict[str, Any],
    config: RunConfig,
    batch_size: int,
) -> dict[str, Any]:
    if batch_size != FORMAL_BRANCH_BATCH_SIZE:
        raise ValueError(
            "probe continuation-answer batch size is frozen at "
            f"{FORMAL_BRANCH_BATCH_SIZE}"
        )
    payload = {
        "schema_version": "flcva-probe-generation-identity-v4",
        "study_id": manifest["study_id"],
        "model_name": manifest["model_name"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "sampling": manifest["sampling"],
        "global_seed": manifest["global_seed"],
        "input_sha256": manifest["input_sha256"],
        "script_sha256": manifest["script_sha256"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "source_tree_file_count": manifest["source_tree_file_count"],
        "execution_bundle": manifest["execution_bundle"],
        "cues": manifest["cues"],
        "workers": manifest["workers"],
        "batch_size": batch_size,
        "batch_schedule": {
            "stop_probe_batch_size": FORMAL_STOP_BATCH_SIZE,
            "continuation_answer_batch_size": FORMAL_BRANCH_BATCH_SIZE,
            "answer_top_logprobs": PROBE_TOP_LOGPROBS,
            "on_spine_successor_reuse": True,
            "stored_frozen_contexts_only": True,
        },
        "effective_model_config": config.model.model_dump(mode="json"),
        "effective_sampling_config": config.sampling.model_dump(mode="json"),
    }
    return {
        **payload,
        "identity_sha256": payload_sha256(payload),
    }


def _runtime_execution_identity(
    runtime_model: dict[str, Any],
) -> dict[str, Any]:
    import torch

    devices = [
        str(torch.cuda.get_device_name(index))
        for index in range(torch.cuda.device_count())
    ]
    if not devices or not all("H20" in name.upper() for name in devices):
        raise RuntimeError(f"probe runtime requires only NVIDIA H20 devices: {devices}")
    payload = {
        "schema_version": "flcva-probe-runtime-identity-v1",
        "model_snapshot": _snapshot_identity(runtime_model),
        "packages": {
            name: package_metadata.version(name)
            for name in RUNTIME_PACKAGE_NAMES
        },
        "cuda": str(torch.version.cuda),
        "device_family": "NVIDIA H20",
    }
    return {
        **payload,
        "identity_sha256": payload_sha256(payload),
    }


def _verify_runtime_contract(
    manifest: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    expected = manifest["runtime_contract"]
    if runtime.get("model_snapshot") != expected["model_snapshot"]:
        raise ValueError("probe model snapshot differs from sealed source runtime")
    if runtime.get("packages") != expected["packages"]:
        raise ValueError(
            "probe package runtime differs from the sealed source runtime: "
            f"expected={expected['packages']} actual={runtime.get('packages')}"
        )
    if runtime.get("cuda") != expected["cuda"]:
        raise ValueError(
            "probe CUDA runtime differs from the sealed source runtime: "
            f"expected={expected['cuda']} actual={runtime.get('cuda')}"
        )
    if runtime.get("device_family") != "NVIDIA H20":
        raise ValueError("probe runtime is not NVIDIA H20")


def _verify_problem_output(
    payload: dict[str, Any],
    *,
    manifest: dict[str, Any],
    generation_identity: dict[str, Any],
    runtime_identity: dict[str, Any],
    worker_index: int,
    problem: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    expected = {
        "schema_version": "flcva-probe-output-problem-v5",
        "study_id": manifest["study_id"],
        "model_name": manifest["model_name"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "worker_index": worker_index,
        "input_sha256": manifest["input_sha256"],
        "generation_script_sha256": manifest["script_sha256"],
        "generation_identity_sha256": generation_identity["identity_sha256"],
        "runtime_identity_sha256": runtime_identity["identity_sha256"],
    }
    if not verify_sealed_record(payload):
        raise ValueError("probe problem output seal failed")
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"probe problem output {field} mismatch: "
                f"expected={value!r} actual={payload.get(field)!r}"
            )
    if problem is not None:
        if payload.get("problem_id") != problem["problem_id"]:
            raise ValueError("probe problem output problem_id mismatch")
        if payload.get("dataset_id") != problem["dataset_id"]:
            raise ValueError("probe problem output dataset_id mismatch")
        if (
            payload.get("input_problem_record_sha256")
            != problem["record_sha256"]
        ):
            raise ValueError("probe problem output input record mismatch")
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != 900:
        raise ValueError(
            "probe problem output must contain exactly 900 logical rows"
        )
    physical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        physical[str(row["physical_request_id"])].append(row)
    if len(physical) != 732:
        raise ValueError(
            "probe answer output must contain exactly 732 physical requests"
        )
    accounted_rows: list[dict[str, Any]] = []
    for references in physical.values():
        reference_count = len(references)
        if (
            {int(row["physical_reference_count"]) for row in references}
            != {reference_count}
            or sorted(
                int(row["physical_reference_index"]) for row in references
            )
            != list(range(reference_count))
        ):
            raise ValueError("probe physical-reference accounting mismatch")
        accounted = [
            row for row in references if row.get("physical_accounted") is True
        ]
        if (
            len(accounted) != 1
            or int(accounted[0]["physical_reference_index"]) != 0
        ):
            raise ValueError("probe physical request must be accounted once")
        for row in references:
            if row.get("physical_accounted") is not True and (
                int(row.get("physical_output_tokens") or 0) != 0
                or float(row.get("gpu_seconds") or 0.0) != 0.0
                or int(row.get("peak_gpu_memory_bytes") or 0) != 0
            ):
                raise ValueError("duplicate logical reference charges compute")
        accounted_rows.append(accounted[0])
    batches: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in accounted_rows:
        batches[int(row["batch_index"])].append(row)
    if sorted(batches) != list(range(24)):
        raise ValueError("probe problem output has non-contiguous batch indices")
    for batch_rows in batches.values():
        capacities = {int(row["batch_capacity"]) for row in batch_rows}
        declared_sizes = {int(row["batch_size"]) for row in batch_rows}
        utilizations = {
            float(row["batch_utilization"]) for row in batch_rows
        }
        if (
            len(capacities) != 1
            or declared_sizes != {len(batch_rows)}
            or utilizations != {len(batch_rows) / next(iter(capacities))}
            or capacities
            not in (
                {FORMAL_STOP_BATCH_SIZE},
                {FORMAL_BRANCH_BATCH_SIZE},
            )
        ):
            raise ValueError("probe problem output batch accounting mismatch")
    capacity_counts = {
        capacity: sum(
            1
            for batch_rows in batches.values()
            if int(batch_rows[0]["batch_capacity"]) == capacity
        )
        for capacity in (FORMAL_STOP_BATCH_SIZE, FORMAL_BRANCH_BATCH_SIZE)
    }
    if capacity_counts != {
        FORMAL_STOP_BATCH_SIZE: 12,
        FORMAL_BRANCH_BATCH_SIZE: 12,
    }:
        raise ValueError("probe problem output batch schedule mismatch")
    return rows


def _physical_accounting(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    accounted = [
        row for row in rows if row.get("physical_accounted") is True
    ]
    physical_events = accounted
    batches = {
        int(row["batch_index"]): int(row["batch_capacity"])
        for row in physical_events
    }
    return {
        "logical_rows": float(len(rows)),
        "physical_requests": float(len(physical_events)),
        "output_tokens": float(
            sum(
                int(row["physical_output_tokens"])
                for row in physical_events
            )
        ),
        "gpu_seconds": sum(
            float(row.get("gpu_seconds") or 0.0)
            for row in physical_events
        ),
        "peak_gpu_memory_bytes": float(
            max(
                (
                    int(row.get("peak_gpu_memory_bytes") or 0)
                    for row in physical_events
                ),
                default=0,
            )
        ),
        "batches": float(len(batches)),
        "declared_batch_slots": float(sum(batches.values())),
    }


def _assert_audited(run_dir: Path) -> dict[str, Any]:
    audit = _read_json(run_dir / "audit_manifest.json")
    if audit.get("passed") is not True:
        raise ValueError(f"source run is not audit-passed: {run_dir}")
    gates = audit.get("gates")
    if not isinstance(gates, dict) or not gates or not all(gates.values()):
        raise ValueError(f"source audit gates are incomplete: {run_dir}")
    if not verify_sealed_record(audit):
        raise ValueError(f"source audit manifest seal failed: {run_dir}")
    return audit


def _confirmation_ids(run_dir: Path) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in _iter_jsonl_gz(run_dir / "evaluation" / "values.jsonl.gz"):
        if row.get("split") == "confirmation":
            output[str(row["dataset_id"])].add(str(row["problem_id"]))
    return output


def _sample_problem_ids() -> dict[str, list[str]]:
    universes = [_confirmation_ids(path) for path in SOURCE_RUNS.values()]
    selected: dict[str, list[str]] = {}
    for dataset_id, quota in SAMPLE_QUOTAS.items():
        shared = set.intersection(
            *(universe[dataset_id] for universe in universes)
        )
        ordered = sorted(
            shared,
            key=lambda problem_id: hashlib.sha256(
                f"{SELECTION_NAMESPACE}:{problem_id}".encode()
            ).hexdigest(),
        )
        if len(ordered) < quota:
            raise ValueError(
                f"{dataset_id}: need {quota} confirmation problems, "
                f"found {len(ordered)}"
            )
        selected[dataset_id] = ordered[:quota]
    return selected


def _problem_path(run_dir: Path, problem_id: str) -> Path:
    return run_dir / "problems" / f"{_safe_problem_name(problem_id)}.json.gz"


def _extract_problem(
    run_dir: Path,
    problem_id: str,
    *,
    expected_model_id: str,
) -> dict[str, Any]:
    artifact = read_json_gz(_problem_path(run_dir, problem_id))
    if not verify_sealed_record(artifact):
        raise ValueError(f"problem artifact seal failed: {problem_id}")
    blobs = BlobStore(run_dir / "blobs")
    checkpoints_by_spine: dict[int, list[dict[str, Any]]] = defaultdict(list)
    branches_by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for branch in artifact["branches"]:
        branches_by_checkpoint[str(branch["checkpoint_id"])].append(branch)
    for checkpoint in artifact["checkpoints"]:
        if checkpoint["model_id"] != expected_model_id:
            raise ValueError(
                f"{problem_id}: unexpected model {checkpoint['model_id']}"
            )
        if checkpoint["continuation_eligible"]:
            checkpoints_by_spine[int(checkpoint["spine_id"])].append(checkpoint)

    spines: list[dict[str, Any]] = []
    for spine_id in sorted(checkpoints_by_spine):
        checkpoints = sorted(
            checkpoints_by_spine[spine_id],
            key=lambda row: int(row["depth"]),
        )
        if len(checkpoints) != 15:
            raise ValueError(
                f"{problem_id}/spine-{spine_id}: expected 15 checkpoints"
            )
        first = checkpoints[0]
        prompt_tokens = blobs.get(str(first["prompt_blob"]))
        spine_tokens = blobs.get(str(first["spine_blob"]))
        rows: list[dict[str, Any]] = []
        for checkpoint in checkpoints:
            branches = sorted(
                branches_by_checkpoint[str(checkpoint["checkpoint_id"])],
                key=lambda row: int(row["branch_index"]),
            )
            if [int(row["branch_index"]) for row in branches] != [1, 2, 3, 4]:
                raise ValueError(
                    f"{checkpoint['checkpoint_id']}: incomplete branches"
                )
            rows.append(
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "depth": checkpoint["depth"],
                    "prefix_sha256": checkpoint["prefix_sha256"],
                    "expected_stop_tokens": checkpoint["stop_answer_tokens"],
                    "branches": [
                        {
                            "branch_id": branch["branch_id"],
                            "branch_index": branch["branch_index"],
                            "reasoning_tokens": blobs.get(
                                str(branch["reasoning_blob"])
                            ),
                            "reasoning_token_count": branch["reasoning_tokens"],
                            "expected_answer_tokens": branch["answer_tokens"],
                        }
                        for branch in branches
                    ],
                }
            )
        spines.append(
            {
                "spine_id": spine_id,
                "prompt_tokens": prompt_tokens,
                "spine_tokens": spine_tokens,
                "checkpoints": rows,
            }
        )
    if len(spines) != 4:
        raise ValueError(f"{problem_id}: expected four spines")
    output = {
        "schema_version": "flcva-probe-input-problem-v1",
        "problem_id": problem_id,
        "dataset_id": artifact["dataset_id"],
        "source_problem_record_sha256": artifact["record_sha256"],
        "spines": spines,
        "record_sha256": "",
    }
    output["record_sha256"] = payload_sha256(output)
    return output


def prepare_inputs() -> None:
    selected = _sample_problem_ids()
    selected_ids = sorted(
        problem_id
        for dataset_ids in selected.values()
        for problem_id in dataset_ids
    )
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_tree_sha256, source_tree_file_count = source_tree_fingerprint(ROOT)
    execution_bundle = _probe_execution_bundle()
    for model_name, run_dir in SOURCE_RUNS.items():
        audit = _assert_audited(run_dir)
        contract = _read_json(run_dir / "contract.json")
        source_manifest = _read_json(run_dir / "manifest.json")
        if not verify_sealed_record(source_manifest):
            raise ValueError(f"source manifest seal failed: {run_dir}")
        model_identities = {
            payload_sha256(row["runtime_model_identity"]): row[
                "runtime_model_identity"
            ]
            for row in source_manifest["parent_runs"]
        }
        if len(model_identities) != 1:
            raise ValueError(f"source model identity is not unique: {run_dir}")
        merge = _read_json(run_dir / "merge_summary.json")
        package_sets: dict[str, dict[str, Any]] = {}
        hardware_sets: dict[str, dict[str, Any]] = {}
        for source_shard in merge["inputs"]:
            source_run = _read_json(Path(source_shard) / "run.json")
            if not verify_sealed_record(source_run):
                raise ValueError(f"source shard run seal failed: {source_shard}")
            packages = {
                name: source_run["packages"].get(name)
                for name in RUNTIME_PACKAGE_NAMES
            }
            package_sets[payload_sha256(packages)] = packages
            hardware = {
                "cuda": str(source_run["hardware"]["cuda"]),
                "device_family": "NVIDIA H20",
            }
            if not all(
                "H20" in str(device).upper()
                for device in source_run["hardware"]["devices"]
            ):
                raise ValueError(f"source shard is not H20: {source_shard}")
            hardware_sets[payload_sha256(hardware)] = hardware
        if len(package_sets) != 1:
            raise ValueError(f"source package runtime is not unique: {run_dir}")
        if len(hardware_sets) != 1:
            raise ValueError(f"source hardware runtime is not unique: {run_dir}")
        input_path = INPUT_DIR / f"{model_name}.jsonl.gz"
        rows = (
            _extract_problem(
                run_dir,
                problem_id,
                expected_model_id=str(contract["model_id"]),
            )
            for problem_id in selected_ids
        )
        _write_jsonl_gz(input_path, rows)
        manifest = seal_record({
            "schema_version": "flcva-probe-input-manifest-v5",
            "study_id": STUDY_ID,
            "model_name": model_name,
            "model_id": contract["model_id"],
            "model_revision": contract["model_revision"],
            "tokenizer_revision": contract["tokenizer_revision"],
            "sampling": contract["sampling"],
            "global_seed": contract["global_seed"],
            "source_run_id": audit["run_id"],
            "source_contract_sha256": audit["contract_sha256"],
            "source_problem_merkle_root": audit["problem_merkle_root"],
            "source_result_merkle_root": audit["result_merkle_root"],
            "source_audit_manifest_sha256": file_sha256(
                run_dir / "audit_manifest.json"
            ),
            "selection": {
                "split": "confirmation",
                "rule": (
                    f"smallest sha256({SELECTION_NAMESPACE}:problem_id)"
                ),
                "namespace_frozen_from": SELECTION_NAMESPACE,
                "quotas": SAMPLE_QUOTAS,
                "problem_ids": selected,
                "problem_ids_sha256": payload_sha256(selected_ids),
            },
            "cues": CUES,
            "contexts_per_checkpoint": 5,
            "nonterminal_checkpoints_per_spine": 15,
            "spines_per_problem": 4,
            "workers": WORKERS,
            "labels_included": False,
            "input_file": input_path.name,
            "input_sha256": file_sha256(input_path),
            "script_sha256": file_sha256(Path(__file__)),
            "source_tree_sha256": source_tree_sha256,
            "source_tree_file_count": source_tree_file_count,
            "execution_bundle": execution_bundle,
            "runtime_contract": {
                "model_snapshot": _snapshot_identity(
                    next(iter(model_identities.values()))
                ),
                "packages": next(iter(package_sets.values())),
                "device_family": "NVIDIA H20",
                "cuda": next(iter(hardware_sets.values()))["cuda"],
            },
        })
        _atomic_json(INPUT_DIR / f"{model_name}.manifest.json", manifest)
        print(
            json.dumps(
                {
                    "model": model_name,
                    "problems": len(selected_ids),
                    "input": str(input_path),
                    "sha256": manifest["input_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _load_input(model_name: str) -> tuple[dict[str, Any], Path]:
    manifest_path = INPUT_DIR / f"{model_name}.manifest.json"
    manifest = _read_json(manifest_path)
    if not verify_sealed_record(manifest):
        raise ValueError(f"input manifest seal failed: {manifest_path}")
    if manifest.get("schema_version") != "flcva-probe-input-manifest-v5":
        raise ValueError(f"input manifest schema mismatch: {manifest_path}")
    if manifest.get("study_id") != STUDY_ID:
        raise ValueError(f"input study mismatch: {manifest_path}")
    input_path = INPUT_DIR / str(manifest["input_file"])
    if file_sha256(input_path) != manifest["input_sha256"]:
        raise ValueError(f"input digest mismatch: {input_path}")
    if manifest.get("labels_included") is not False:
        raise ValueError("GPU input manifest must explicitly exclude labels")
    if manifest.get("cues") != CUES:
        raise ValueError("probe cue manifest mismatch")
    if manifest.get("script_sha256") != file_sha256(Path(__file__)):
        raise ValueError(
            "probe implementation differs from the immutable input manifest; "
            "prepare a new study version"
        )
    source_tree_sha256, source_tree_file_count = source_tree_fingerprint(ROOT)
    if (
        manifest.get("source_tree_sha256") != source_tree_sha256
        or manifest.get("source_tree_file_count") != source_tree_file_count
    ):
        raise ValueError(
            "probe source tree differs from the immutable input manifest; "
            "prepare a new study version"
        )
    if manifest.get("execution_bundle") != _probe_execution_bundle():
        raise ValueError(
            "probe execution bundle differs from the immutable input manifest; "
            "prepare a new study version"
        )
    return manifest, input_path


def validate_inputs() -> None:
    problem_sets: list[set[str]] = []
    for model_name in SOURCE_RUNS:
        manifest, input_path = _load_input(model_name)
        rows = list(_iter_jsonl_gz(input_path))
        ids = {str(row["problem_id"]) for row in rows}
        if len(rows) != 200 or len(ids) != 200:
            raise ValueError(f"{model_name}: expected 200 unique problems")
        for row in rows:
            recorded = row.get("record_sha256")
            unhashed = dict(row)
            unhashed["record_sha256"] = ""
            if recorded != payload_sha256(unhashed):
                raise ValueError(
                    f"{model_name}/{row['problem_id']}: record digest mismatch"
                )
        problem_sets.append(ids)
        print(
            json.dumps(
                {
                    "model": model_name,
                    "problems": len(ids),
                    "input_sha256": manifest["input_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if problem_sets[0] != problem_sets[1]:
        raise ValueError("model inputs do not share the same problem sample")


def _worker_bucket(problem_id: str) -> int:
    digest = hashlib.sha256(
        f"flcva-probe-worker-v1:{problem_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % WORKERS


def _context_rows(
    problem: dict[str, Any],
    backend: Any,
    sampling: Any,
    *,
    batch_size: int,
    global_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if batch_size != FORMAL_BRANCH_BATCH_SIZE:
        raise ValueError(
            "probe continuation-answer batch size is frozen at "
            f"{FORMAL_BRANCH_BATCH_SIZE}"
    )
    results: list[dict[str, Any]] = []
    totals = {
        "logical_rows": 0.0,
        "physical_requests": 0.0,
        "output_tokens": 0.0,
        "gpu_seconds": 0.0,
        "peak_gpu_memory_bytes": 0.0,
        "batches": 0.0,
        "declared_batch_slots": 0.0,
    }
    cue_tokens = {
        cue_name: backend.encode(cue)
        for cue_name, cue in CUES.items()
    }
    global_seed = int(global_seed)

    def run_batch(
        pending: list[tuple[list[dict[str, Any]], GenerationRequest]],
        *,
        capacity: int,
    ) -> None:
        if len(pending) != capacity:
            raise ValueError(
                f"formal-equivalent probe batch requires {capacity} requests, "
                f"got {len(pending)}"
            )
        generated = backend.generate_batch(
            [request for _, request in pending],
            sampling=sampling,
        )
        if len(generated) != len(pending):
            raise RuntimeError("probe batch returned the wrong number of rows")
        batch_index = int(totals["batches"])
        totals["batches"] += 1
        totals["declared_batch_slots"] += capacity
        for (references, request), output in zip(
            pending, generated, strict=True
        ):
            if len(references) not in (1, 2):
                raise ValueError("physical probe request has invalid references")
            physical_request_id = payload_sha256(
                {
                    "problem_id": problem["problem_id"],
                    "batch_index": batch_index,
                    "prompt_token_ids": request.prompt_token_ids,
                    "seed": request.seed,
                    "purpose": request.purpose,
                }
            )
            for reference_index, metadata in enumerate(references):
                expected = metadata.pop("expected_canonical_tokens")
                source_answer_text = backend.decode(expected)
                accounted = reference_index == 0
                row = {
                    **metadata,
                    "generation_seed": request.seed,
                    "answer_tokens": output.token_ids,
                    "answer_text": output.text,
                    "raw_answer": (
                        CUES[metadata["cue_name"]] + output.text
                    ),
                    "source_raw_answer": CUES["canonical"] + source_answer_text,
                    "finish_reason": output.finish_reason,
                    "output_tokens": len(output.token_ids),
                    "physical_output_tokens": (
                        len(output.token_ids) if accounted else 0
                    ),
                    "gpu_seconds": (
                        output.gpu_seconds if accounted else 0.0
                    ),
                    "peak_gpu_memory_bytes": (
                        output.peak_gpu_memory_bytes if accounted else 0
                    ),
                    "batch_size": output.batch_size,
                    "batch_capacity": capacity,
                    "batch_index": batch_index,
                    "batch_utilization": output.batch_utilization,
                    "physical_request_id": physical_request_id,
                    "physical_reference_index": reference_index,
                    "physical_reference_count": len(references),
                    "physical_accounted": accounted,
                    "canonical_control_token_match": (
                        output.token_ids == expected
                        if metadata["cue_name"] == "canonical"
                        else None
                    ),
                    "canonical_control_text_match": (
                        output.text == source_answer_text
                        if metadata["cue_name"] == "canonical"
                        else None
                    ),
                }
                results.append(row)
                totals["logical_rows"] += 1
            totals["physical_requests"] += 1
            totals["output_tokens"] += len(output.token_ids)
            totals["gpu_seconds"] += float(output.gpu_seconds or 0.0)
            totals["peak_gpu_memory_bytes"] = max(
                totals["peak_gpu_memory_bytes"],
                float(output.peak_gpu_memory_bytes or 0),
            )

    for cue_name in CUES:
        for spine in problem["spines"]:
            prompt = list(spine["prompt_tokens"])
            spine_tokens = list(spine["spine_tokens"])
            checkpoints = sorted(
                spine["checkpoints"], key=lambda row: int(row["depth"])
            )
            if [int(row["depth"]) for row in checkpoints] != list(
                range(256, 4096, 256)
            ):
                raise ValueError("probe input checkpoint depths are incomplete")
            by_depth = {
                int(checkpoint["depth"]): checkpoint
                for checkpoint in checkpoints
            }
            stop_pending: list[
                tuple[list[dict[str, Any]], GenerationRequest]
            ] = []
            for stop_depth in range(256, 4097, 256):
                references: list[dict[str, Any]] = []
                if stop_depth < 4096:
                    checkpoint = by_depth[stop_depth]
                    references.append(
                        {
                            "checkpoint_id": checkpoint["checkpoint_id"],
                            "spine_id": spine["spine_id"],
                            "depth": stop_depth,
                            "context_kind": "stop",
                            "context_id": (
                                f"stop:{checkpoint['checkpoint_id']}"
                            ),
                            "branch_id": None,
                            "branch_index": None,
                            "reasoning_token_count": 0,
                            "cue_name": cue_name,
                            "expected_canonical_tokens": checkpoint[
                                "expected_stop_tokens"
                            ],
                        }
                    )
                if stop_depth > 256:
                    predecessor = by_depth[stop_depth - 256]
                    on_spine = next(
                        branch
                        for branch in predecessor["branches"]
                        if int(branch["branch_index"]) == 1
                    )
                    references.append(
                        {
                            "checkpoint_id": predecessor["checkpoint_id"],
                            "spine_id": spine["spine_id"],
                            "depth": stop_depth - 256,
                            "context_kind": "branch",
                            "context_id": (
                                f"branch:{on_spine['branch_id']}"
                            ),
                            "branch_id": on_spine["branch_id"],
                            "branch_index": 1,
                            "reasoning_token_count": on_spine[
                                "reasoning_token_count"
                            ],
                            "cue_name": cue_name,
                            "expected_canonical_tokens": on_spine[
                                "expected_answer_tokens"
                            ],
                        }
                    )
                expected_sets = {
                    tuple(reference["expected_canonical_tokens"])
                    for reference in references
                }
                if len(expected_sets) != 1:
                    raise ValueError(
                        "on-spine branch does not match successor stop output"
                    )
                stop_pending.append(
                    (
                        references,
                        GenerationRequest(
                            prompt_token_ids=(
                                prompt
                                + spine_tokens[:stop_depth]
                                + cue_tokens[cue_name]
                            ),
                            max_tokens=32,
                            seed=stable_seed(
                                global_seed,
                                problem["problem_id"],
                                spine["spine_id"],
                                stop_depth,
                                "stop",
                            ),
                            purpose="answer",
                            greedy=True,
                            top_logprobs=PROBE_TOP_LOGPROBS,
                            batch_capacity=FORMAL_STOP_BATCH_SIZE,
                        ),
                    )
                )
            run_batch(stop_pending, capacity=FORMAL_STOP_BATCH_SIZE)

            branch_pending: list[
                tuple[list[dict[str, Any]], GenerationRequest]
            ] = []
            for checkpoint in checkpoints:
                depth = int(checkpoint["depth"])
                prefix = prompt + spine_tokens[:depth]
                off_spine = sorted(
                    (
                        branch
                        for branch in checkpoint["branches"]
                        if int(branch["branch_index"]) > 1
                    ),
                    key=lambda row: int(row["branch_index"]),
                )
                if [int(row["branch_index"]) for row in off_spine] != [2, 3, 4]:
                    raise ValueError("probe input off-spine branches are incomplete")
                for branch in off_spine:
                    branch_index = int(branch["branch_index"])
                    metadata = {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "spine_id": spine["spine_id"],
                        "depth": depth,
                        "context_kind": "branch",
                        "context_id": f"branch:{branch['branch_id']}",
                        "branch_id": branch["branch_id"],
                        "branch_index": branch_index,
                        "reasoning_token_count": branch[
                            "reasoning_token_count"
                        ],
                        "cue_name": cue_name,
                        "expected_canonical_tokens": branch[
                            "expected_answer_tokens"
                        ],
                    }
                    branch_pending.append(
                        (
                            [metadata],
                            GenerationRequest(
                                prompt_token_ids=(
                                    prefix
                                    + list(branch["reasoning_tokens"])
                                    + cue_tokens[cue_name]
                                ),
                                max_tokens=32,
                                seed=stable_seed(
                                    global_seed,
                                    checkpoint["checkpoint_id"],
                                    branch_index,
                                    "answer",
                                ),
                                purpose="answer",
                                greedy=True,
                                top_logprobs=PROBE_TOP_LOGPROBS,
                                batch_capacity=FORMAL_BRANCH_BATCH_SIZE,
                            ),
                        )
                    )
            run_batch(branch_pending, capacity=FORMAL_BRANCH_BATCH_SIZE)
    if len(results) != 900 or int(totals["physical_requests"]) != 732:
        raise ValueError("probe logical/physical request cardinality mismatch")
    return results, totals


def generate(
    model_name: str,
    worker_index: int,
    output_dir: Path,
    batch_size: int,
) -> None:
    if not 0 <= worker_index < WORKERS:
        raise ValueError(f"worker index must be in [0,{WORKERS - 1}]")
    if batch_size != FORMAL_BRANCH_BATCH_SIZE:
        raise ValueError(
            "probe continuation-answer batch size is frozen at "
            f"{FORMAL_BRANCH_BATCH_SIZE}"
        )
    manifest, input_path = _load_input(model_name)
    config = RunConfig.load(CONFIGS[model_name])
    if config.model.id != manifest["model_id"]:
        raise ValueError("model configuration does not match input manifest")
    if config.model.revision != manifest["model_revision"]:
        raise ValueError("model revision does not match input manifest")
    if config.model.tokenizer_revision != manifest["tokenizer_revision"]:
        raise ValueError("tokenizer revision does not match input manifest")
    generation_identity = _generation_identity(manifest, config, batch_size)
    expected_problem_records = {
        str(row["problem_id"]): {
            "problem_id": row["problem_id"],
            "dataset_id": row["dataset_id"],
            "record_sha256": row["record_sha256"],
        }
        for row in _iter_jsonl_gz(input_path)
        if _worker_bucket(str(row["problem_id"])) == worker_index
    }
    expected_ids = sorted(expected_problem_records)
    output_dir = output_dir.resolve()
    worker_dir = output_dir / "workers" / f"worker-{worker_index:02d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_manifest_path = worker_dir / "manifest.json"
    if worker_manifest_path.exists():
        existing = _read_json(worker_manifest_path)
        expected_worker_fields = {
            "schema_version": "flcva-probe-worker-manifest-v5",
            "complete": True,
            "study_id": manifest["study_id"],
            "model_name": model_name,
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "tokenizer_revision": manifest["tokenizer_revision"],
            "worker_index": worker_index,
            "workers": WORKERS,
            "input_sha256": manifest["input_sha256"],
            "source_result_merkle_root": manifest[
                "source_result_merkle_root"
            ],
            "generation_script_sha256": manifest["script_sha256"],
            "generation_identity_sha256": generation_identity[
                "identity_sha256"
            ],
            "problem_ids_sha256": payload_sha256(expected_ids),
            "problems": len(expected_ids),
            "requested_batch_size": batch_size,
        }
        compatible = verify_sealed_record(existing)
        compatible = compatible and all(
            existing.get(field) == value
            for field, value in expected_worker_fields.items()
        )
        compatible = compatible and (
            existing.get("generation_identity") == generation_identity
        )
        runtime_identity = existing.get("runtime_identity")
        compatible = compatible and isinstance(runtime_identity, dict)
        if compatible:
            _verify_runtime_contract(manifest, runtime_identity)
            observed_ids: list[str] = []
            logical_row_count = 0
            physical_request_count = 0
            observed_batches = 0
            observed_batch_slots = 0
            for item in existing.get("problem_files", []):
                relative = Path(str(item["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"unsafe completed worker output path: {relative}"
                    )
                path = worker_dir / relative
                if file_sha256(path) != item["sha256"]:
                    raise ValueError(
                        f"completed worker output is corrupted: {path}"
                    )
                problem_id = str(item["problem_id"])
                if problem_id not in expected_problem_records:
                    raise ValueError(
                        f"unexpected completed problem output: {problem_id}"
                    )
                problem_payload = read_json_gz(path)
                rows = _verify_problem_output(
                    problem_payload,
                    manifest=manifest,
                    generation_identity=generation_identity,
                    runtime_identity=runtime_identity,
                    worker_index=worker_index,
                    problem=expected_problem_records[problem_id],
                )
                accounting = _physical_accounting(rows)
                if (
                    item.get("logical_rows") != len(rows)
                    or item.get("physical_requests")
                    != int(accounting["physical_requests"])
                ):
                    raise ValueError(
                        f"completed row count mismatch: {path}"
                    )
                observed_ids.append(problem_id)
                logical_row_count += len(rows)
                physical_request_count += int(
                    accounting["physical_requests"]
                )
                observed_batches += int(accounting["batches"])
                observed_batch_slots += int(
                    accounting["declared_batch_slots"]
                )
            if (
                sorted(observed_ids) != expected_ids
                or existing.get("logical_rows") != logical_row_count
                or existing.get("physical_requests")
                != physical_request_count
                or existing.get("batches") != observed_batches
                or existing.get("declared_batch_slots")
                != observed_batch_slots
                or existing.get("request_batch_fill")
                != physical_request_count / observed_batch_slots
            ):
                raise ValueError("completed worker coverage mismatch")
            print(f"worker {worker_index}: already complete", flush=True)
            return
        raise ValueError(
            f"existing incompatible worker manifest: {worker_manifest_path}; "
            "use a new output directory"
        )

    backend = build_backend("vllm", config.model)
    runtime_identity = _runtime_execution_identity(backend.runtime_identity)
    _verify_runtime_contract(manifest, runtime_identity)
    problem_files: list[dict[str, Any]] = []
    total_logical_rows = 0
    total_physical_requests = 0
    total_output_tokens = 0
    total_gpu_seconds = 0.0
    peak_gpu_memory_bytes = 0
    total_batches = 0
    total_declared_batch_slots = 0
    started = time.time()
    for problem in _iter_jsonl_gz(input_path):
        if _worker_bucket(str(problem["problem_id"])) != worker_index:
            continue
        output_path = (
            worker_dir
            / "problems"
            / f"{_safe_problem_name(str(problem['problem_id']))}.json.gz"
        )
        if output_path.exists():
            existing = read_json_gz(output_path)
            try:
                rows = _verify_problem_output(
                    existing,
                    manifest=manifest,
                    generation_identity=generation_identity,
                    runtime_identity=runtime_identity,
                    worker_index=worker_index,
                    problem=problem,
                )
            except ValueError as error:
                raise ValueError(
                    f"existing incompatible problem output: {output_path}; "
                    "use a new output directory"
                ) from error
            else:
                problem_files.append(
                    {
                        "path": str(output_path.relative_to(worker_dir)),
                        "sha256": file_sha256(output_path),
                        "problem_id": problem["problem_id"],
                        "logical_rows": len(rows),
                        "physical_requests": int(
                            _physical_accounting(rows)["physical_requests"]
                        ),
                    }
                )
                accounting = _physical_accounting(rows)
                total_logical_rows += len(rows)
                total_physical_requests += int(
                    accounting["physical_requests"]
                )
                total_output_tokens += int(accounting["output_tokens"])
                total_gpu_seconds += float(accounting["gpu_seconds"])
                peak_gpu_memory_bytes = max(
                    peak_gpu_memory_bytes,
                    int(accounting["peak_gpu_memory_bytes"]),
                )
                total_batches += int(accounting["batches"])
                total_declared_batch_slots += int(
                    accounting["declared_batch_slots"]
                )
                print(
                    f"worker {worker_index}: resume {problem['problem_id']}",
                    flush=True,
                )
                continue
        rows, totals = _context_rows(
            problem,
            backend,
            config.sampling,
            batch_size=batch_size,
            global_seed=int(manifest["global_seed"]),
        )
        if len(rows) != 900:
            raise ValueError(
                f"{problem['problem_id']}: expected 900 results, got {len(rows)}"
            )
        payload = seal_record({
            "schema_version": "flcva-probe-output-problem-v5",
            "study_id": manifest["study_id"],
            "model_name": model_name,
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "tokenizer_revision": manifest["tokenizer_revision"],
            "worker_index": worker_index,
            "problem_id": problem["problem_id"],
            "dataset_id": problem["dataset_id"],
            "input_sha256": manifest["input_sha256"],
            "input_problem_record_sha256": problem["record_sha256"],
            "generation_script_sha256": manifest["script_sha256"],
            "generation_identity_sha256": generation_identity[
                "identity_sha256"
            ],
            "runtime_identity_sha256": runtime_identity["identity_sha256"],
            "results": rows,
        })
        _atomic_json_gz(output_path, payload)
        problem_files.append(
            {
                "path": str(output_path.relative_to(worker_dir)),
                "sha256": file_sha256(output_path),
                "problem_id": problem["problem_id"],
                "logical_rows": len(rows),
                "physical_requests": int(totals["physical_requests"]),
            }
        )
        total_logical_rows += int(totals["logical_rows"])
        total_physical_requests += int(totals["physical_requests"])
        total_output_tokens += int(totals["output_tokens"])
        total_gpu_seconds += float(totals["gpu_seconds"])
        peak_gpu_memory_bytes = max(
            peak_gpu_memory_bytes,
            int(totals["peak_gpu_memory_bytes"]),
        )
        total_batches += int(totals["batches"])
        total_declared_batch_slots += int(totals["declared_batch_slots"])
        print(
            f"worker {worker_index}: complete {problem['problem_id']}",
            flush=True,
        )

    observed_ids = sorted(str(row["problem_id"]) for row in problem_files)
    if observed_ids != expected_ids:
        raise ValueError("worker problem coverage mismatch")
    worker_manifest = seal_record({
        "schema_version": "flcva-probe-worker-manifest-v5",
        "study_id": manifest["study_id"],
        "complete": True,
        "model_name": model_name,
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "worker_index": worker_index,
        "workers": WORKERS,
        "input_sha256": manifest["input_sha256"],
        "source_result_merkle_root": manifest["source_result_merkle_root"],
        "generation_script_sha256": manifest["script_sha256"],
        "generation_identity": generation_identity,
        "generation_identity_sha256": generation_identity["identity_sha256"],
        "problem_ids_sha256": payload_sha256(expected_ids),
        "problems": len(expected_ids),
        "requested_batch_size": batch_size,
        "logical_rows": total_logical_rows,
        "physical_requests": total_physical_requests,
        "output_tokens": total_output_tokens,
        "gpu_seconds": total_gpu_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "batches": total_batches,
        "declared_batch_slots": total_declared_batch_slots,
        "request_batch_fill": (
            total_physical_requests / total_declared_batch_slots
            if total_declared_batch_slots
            else None
        ),
        "wall_seconds": time.time() - started,
        "runtime_identity": runtime_identity,
        "runtime_model_identity": backend.runtime_identity,
        "problem_files": sorted(
            problem_files, key=lambda row: str(row["problem_id"])
        ),
    })
    _atomic_json(worker_manifest_path, worker_manifest)
    print(json.dumps(worker_manifest, sort_keys=True), flush=True)


def _load_labels() -> dict[str, tuple[str, str]]:
    data_manifest = _read_json(DATA_MANIFEST_PATH)
    expected = data_manifest["combined"]["sealed_labels_sha256"]
    if file_sha256(LABELS_PATH) != expected:
        raise ValueError("sealed combined-label digest mismatch")
    labels: dict[str, tuple[str, str]] = {}
    with LABELS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            labels[str(row["id"])] = (
                str(row["dataset_id"]),
                str(row["answer"]),
            )
    return labels


def _load_outputs(
    model_name: str,
    result_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    input_manifest, input_path = _load_input(model_name)
    config = RunConfig.load(CONFIGS[model_name])
    expected_problem_records = {
        str(row["problem_id"]): {
            "problem_id": row["problem_id"],
            "dataset_id": row["dataset_id"],
            "record_sha256": row["record_sha256"],
        }
        for row in _iter_jsonl_gz(input_path)
    }
    expected_ids = set(expected_problem_records)
    outputs: list[dict[str, Any]] = []
    worker_manifests: list[dict[str, Any]] = []
    worker_manifest_paths = [
        result_dir / "workers" / f"worker-{worker_index:02d}" / "manifest.json"
        for worker_index in range(WORKERS)
    ]
    preloaded_worker_manifests = [_read_json(path) for path in worker_manifest_paths]
    requested_batch_sizes = {
        int(row["requested_batch_size"])
        for row in preloaded_worker_manifests
        if isinstance(row.get("requested_batch_size"), int)
    }
    if len(requested_batch_sizes) != 1:
        raise ValueError(
            f"{model_name}: workers disagree on requested batch size"
        )
    requested_batch_size = next(iter(requested_batch_sizes))
    if requested_batch_size <= 0:
        raise ValueError(f"{model_name}: invalid requested batch size")
    generation_identity = _generation_identity(
        input_manifest,
        config,
        requested_batch_size,
    )
    observed_ids: set[str] = set()
    runtime_identities: dict[str, dict[str, Any]] = {}
    for worker_index in range(WORKERS):
        worker_dir = result_dir / "workers" / f"worker-{worker_index:02d}"
        worker_manifest = preloaded_worker_manifests[worker_index]
        runtime_identity = worker_manifest.get("runtime_identity")
        if (
            not verify_sealed_record(worker_manifest)
            or worker_manifest.get("schema_version")
            != "flcva-probe-worker-manifest-v5"
            or worker_manifest.get("complete") is not True
            or worker_manifest.get("study_id") != input_manifest["study_id"]
            or worker_manifest.get("model_name") != model_name
            or worker_manifest.get("model_id") != input_manifest["model_id"]
            or worker_manifest.get("model_revision")
            != input_manifest["model_revision"]
            or worker_manifest.get("tokenizer_revision")
            != input_manifest["tokenizer_revision"]
            or worker_manifest.get("worker_index") != worker_index
            or worker_manifest.get("workers") != WORKERS
            or worker_manifest.get("requested_batch_size")
            != requested_batch_size
            or worker_manifest.get("input_sha256")
            != input_manifest["input_sha256"]
            or worker_manifest.get("source_result_merkle_root")
            != input_manifest["source_result_merkle_root"]
            or worker_manifest.get("generation_script_sha256")
            != input_manifest["script_sha256"]
            or worker_manifest.get("generation_identity")
            != generation_identity
            or worker_manifest.get("generation_identity_sha256")
            != generation_identity["identity_sha256"]
            or not isinstance(runtime_identity, dict)
        ):
            raise ValueError(
                f"{model_name}: incompatible worker {worker_index} manifest"
            )
        _verify_runtime_contract(input_manifest, runtime_identity)
        runtime_identities[runtime_identity["identity_sha256"]] = (
            runtime_identity
        )
        worker_manifests.append(worker_manifest)
        worker_ids: list[str] = []
        worker_logical_rows = 0
        worker_physical_requests = 0
        worker_batches = 0
        worker_batch_slots = 0
        for item in worker_manifest["problem_files"]:
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe result path: {relative}")
            path = worker_dir / relative
            if file_sha256(path) != item["sha256"]:
                raise ValueError(f"result file digest mismatch: {path}")
            payload = read_json_gz(path)
            problem_id = str(payload["problem_id"])
            if problem_id not in expected_problem_records:
                raise ValueError(f"unexpected result problem: {problem_id}")
            rows = _verify_problem_output(
                payload,
                manifest=input_manifest,
                generation_identity=generation_identity,
                runtime_identity=runtime_identity,
                worker_index=worker_index,
                problem=expected_problem_records[problem_id],
            )
            accounting = _physical_accounting(rows)
            if (
                item.get("problem_id") != problem_id
                or item.get("logical_rows") != len(rows)
                or item.get("physical_requests")
                != int(accounting["physical_requests"])
            ):
                raise ValueError(f"result manifest entry mismatch: {path}")
            if problem_id in observed_ids:
                raise ValueError(f"duplicate result problem: {problem_id}")
            observed_ids.add(problem_id)
            worker_ids.append(problem_id)
            worker_logical_rows += len(rows)
            worker_physical_requests += int(accounting["physical_requests"])
            worker_batches += int(accounting["batches"])
            worker_batch_slots += int(accounting["declared_batch_slots"])
            for row in rows:
                outputs.append(
                    {
                        **row,
                        "problem_id": problem_id,
                        "dataset_id": payload["dataset_id"],
                    }
                )
        expected_worker_ids = sorted(
            problem_id
            for problem_id in expected_ids
            if _worker_bucket(problem_id) == worker_index
        )
        if (
            sorted(worker_ids) != expected_worker_ids
            or worker_manifest.get("problem_ids_sha256")
            != payload_sha256(expected_worker_ids)
            or worker_manifest.get("problems") != len(expected_worker_ids)
            or worker_manifest.get("logical_rows") != worker_logical_rows
            or worker_manifest.get("physical_requests")
            != worker_physical_requests
            or worker_manifest.get("batches") != worker_batches
            or worker_manifest.get("declared_batch_slots")
            != worker_batch_slots
            or worker_manifest.get("request_batch_fill")
            != worker_physical_requests / worker_batch_slots
        ):
            raise ValueError(
                f"{model_name}: worker {worker_index} coverage mismatch"
            )
    if len(runtime_identities) != 1:
        raise ValueError(
            f"{model_name}: workers used non-identical runtime identities"
        )
    if observed_ids != expected_ids:
        raise ValueError(
            f"{model_name}: output coverage mismatch "
            f"({len(observed_ids)} vs {len(expected_ids)})"
        )
    if len(outputs) != 180_000:
        raise ValueError(
            f"{model_name}: expected 180000 outputs, got {len(outputs)}"
        )
    runtime = {
        "gpu_seconds": sum(
            float(row.get("gpu_seconds") or 0.0) for row in worker_manifests
        ),
        "wall_seconds_max": max(
            float(row.get("wall_seconds") or 0.0) for row in worker_manifests
        ),
        "output_tokens": sum(
            int(row["output_tokens"]) for row in worker_manifests
        ),
        "logical_rows": sum(
            int(row["logical_rows"]) for row in worker_manifests
        ),
        "physical_requests": sum(
            int(row["physical_requests"]) for row in worker_manifests
        ),
        "peak_gpu_memory_bytes": max(
            int(row.get("peak_gpu_memory_bytes") or 0)
            for row in worker_manifests
        ),
        "batches": sum(int(row["batches"]) for row in worker_manifests),
        "declared_batch_slots": sum(
            int(row["declared_batch_slots"]) for row in worker_manifests
        ),
        "runtime_identity": next(iter(runtime_identities.values())),
        "worker_manifest_record_sha256": sorted(
            str(row["record_sha256"]) for row in worker_manifests
        ),
        "result_files_sha256": sorted(
            str(item["sha256"])
            for row in worker_manifests
            for item in row["problem_files"]
        ),
    }
    runtime["worker_manifest_set_sha256"] = payload_sha256(
        runtime["worker_manifest_record_sha256"]
    )
    runtime["result_file_set_sha256"] = payload_sha256(
        runtime["result_files_sha256"]
    )
    runtime["request_batch_fill"] = (
        runtime["physical_requests"] / runtime["declared_batch_slots"]
        if runtime["declared_batch_slots"]
        else None
    )
    return input_manifest, outputs, runtime


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _cluster_interval(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    seed: int,
) -> tuple[float, float, float]:
    by_problem: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_problem[str(row["problem_id"])].append(float(row[metric]))
    keys = sorted(by_problem)
    if not keys:
        raise ValueError(f"bootstrap metric {metric} has no eligible rows")
    aggregates = {
        key: (sum(values), len(values))
        for key, values in by_problem.items()
    }
    observed = sum(value for value, _ in aggregates.values()) / sum(
        count for _, count in aggregates.values()
    )
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [rng.choice(keys) for _ in keys]
        numerator = sum(aggregates[key][0] for key in selected)
        denominator = sum(aggregates[key][1] for key in selected)
        estimates.append(numerator / denominator)
    return observed, _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _scientific_rows(
    model_name: str,
    outputs: list[dict[str, Any]],
    labels: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    branches_by_checkpoint: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    control_token_matches: list[bool] = []
    control_text_matches: list[bool] = []
    control_answer_matches: list[bool] = []
    parse_rows: list[dict[str, Any]] = []
    for row in outputs:
        key = (
            str(row["problem_id"]),
            str(row["context_id"]),
            str(row["cue_name"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate probe output: {key}")
        dataset_id, gold = labels[str(row["problem_id"])]
        correct, parsed, _ = answers_equal(
            dataset_id, str(row["raw_answer"]), gold
        )
        source_parsed = parse_answer(
            dataset_id, str(row["source_raw_answer"])
        )
        enriched = {
            **row,
            "correct": int(correct),
            "canonical_answer": parsed.canonical,
            "parsed": int(parsed.canonical is not None),
            "source_canonical_answer": source_parsed.canonical,
            "source_answer_replay": int(
                row["cue_name"] == "canonical"
                and parsed.canonical is not None
                and source_parsed.canonical is not None
                and parsed.canonical == source_parsed.canonical
            ),
        }
        indexed[key] = enriched
        parse_rows.append(enriched)
        if enriched["context_kind"] == "branch":
            branches_by_checkpoint[
                (
                    str(enriched["problem_id"]),
                    str(enriched["checkpoint_id"]),
                    str(enriched["cue_name"]),
                )
            ].append(enriched)
        if row["cue_name"] == "canonical":
            control_token_matches.append(
                bool(enriched["canonical_control_token_match"])
            )
            control_text_matches.append(
                bool(enriched["canonical_control_text_match"])
            )
            control_answer_matches.append(
                bool(enriched["source_answer_replay"])
            )
    if not control_token_matches:
        raise ValueError(f"{model_name}: canonical replay has no contexts")
    for row in parse_rows:
        canonical = indexed[
            (
                str(row["problem_id"]),
                str(row["context_id"]),
                "canonical",
            )
        ]
        row["answer_pair_eligible"] = int(
            row["canonical_answer"] is not None
            and canonical["canonical_answer"] is not None
        )
        row["answer_agreement"] = int(
            row["answer_pair_eligible"] == 1
            and row["canonical_answer"] == canonical["canonical_answer"]
        )

    value_rows: list[dict[str, Any]] = []
    stop_rows = [
        row
        for row in outputs
        if row["context_kind"] == "stop"
        and row["cue_name"] == "canonical"
    ]
    for stop in stop_rows:
        problem_id = str(stop["problem_id"])
        checkpoint_id = str(stop["checkpoint_id"])
        for cue_name in CUES:
            stop_value = indexed[
                (problem_id, f"stop:{checkpoint_id}", cue_name)
            ]
            branch_values = sorted(
                branches_by_checkpoint[
                    (problem_id, checkpoint_id, cue_name)
                ],
                key=lambda row: int(row["branch_index"]),
            )
            if len(branch_values) != 4:
                raise ValueError(
                    f"{problem_id}/{checkpoint_id}/{cue_name}: "
                    "expected four branches"
                )
            stop_correct = int(stop_value["correct"])
            continuation_correct = [
                int(row["correct"]) for row in branch_values
            ]
            costs = [
                int(row["reasoning_token_count"])
                + int(row["output_tokens"])
                for row in branch_values
            ]
            marginal = 1000.0 * (
                sum(continuation_correct) / len(continuation_correct)
                - stop_correct
            ) / (sum(costs) / len(costs))
            value_rows.append(
                {
                    "model": model_name,
                    "problem_id": problem_id,
                    "dataset_id": stop["dataset_id"],
                    "checkpoint_id": checkpoint_id,
                    "spine_id": stop["spine_id"],
                    "depth": stop["depth"],
                    "cue_name": cue_name,
                    "stop_correct": stop_correct,
                    "branch_accuracy": (
                        sum(continuation_correct)
                        / len(continuation_correct)
                    ),
                    "marginal_value": marginal,
                    "beneficial": int(marginal > 0),
                    "harmful": int(marginal < 0),
                }
            )
    by_key = {
        (row["problem_id"], row["checkpoint_id"], row["cue_name"]): row
        for row in value_rows
    }
    for row in value_rows:
        canonical = by_key[
            (row["problem_id"], row["checkpoint_id"], "canonical")
        ]
        row["canonical_stop_correct"] = canonical["stop_correct"]
        row["beneficial_agreement"] = int(
            row["beneficial"] == canonical["beneficial"]
        )
        row["harmful_agreement"] = int(
            row["harmful"] == canonical["harmful"]
        )
        row["sign_agreement"] = int(
            (row["marginal_value"] > 0)
            - (row["marginal_value"] < 0)
            == (canonical["marginal_value"] > 0)
            - (canonical["marginal_value"] < 0)
        )
    diagnostics = {
        "canonical_control_replay_rate": (
            sum(control_token_matches) / len(control_token_matches)
        ),
        "canonical_control_text_replay_rate": (
            sum(control_text_matches) / len(control_text_matches)
        ),
        "canonical_control_answer_replay_rate": (
            sum(control_answer_matches) / len(control_answer_matches)
        ),
        "canonical_control_contexts": len(control_token_matches),
        "parse_rows": parse_rows,
    }
    return value_rows, diagnostics


def _metric_rows(
    model_name: str,
    values: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    parse_rows = diagnostics["parse_rows"]
    for cue_name in CUES:
        cue_values = [row for row in values if row["cue_name"] == cue_name]
        cue_parse = [
            row for row in parse_rows if row["cue_name"] == cue_name
        ]
        answer_pairs = [
            row for row in cue_parse if row["answer_pair_eligible"] == 1
        ]
        wrong = [row for row in cue_values if row["stop_correct"] == 0]
        correct = [row for row in cue_values if row["stop_correct"] == 1]
        canonical_wrong = [
            row
            for row in cue_values
            if row["canonical_stop_correct"] == 0
        ]
        canonical_correct = [
            row
            for row in cue_values
            if row["canonical_stop_correct"] == 1
        ]
        metrics = {
            "parse_rate": (cue_parse, "parsed"),
            "answer_agreement": (answer_pairs, "answer_agreement"),
            "beneficial_given_wrong": (wrong, "beneficial"),
            "harmful_given_correct": (correct, "harmful"),
            "beneficial_label_agreement": (
                canonical_wrong,
                "beneficial_agreement",
            ),
            "harmful_label_agreement": (
                canonical_correct,
                "harmful_agreement",
            ),
            "marginal_sign_agreement": (cue_values, "sign_agreement"),
        }
        if cue_name == "canonical":
            metrics.update(
                {
                    "source_token_replay": (
                        cue_parse,
                        "canonical_control_token_match",
                    ),
                    "source_text_replay": (
                        cue_parse,
                        "canonical_control_text_match",
                    ),
                    "source_answer_replay": (
                        cue_parse,
                        "source_answer_replay",
                    ),
                }
            )
        for metric_name, (rows, field) in metrics.items():
            estimate, low, high = _cluster_interval(
                rows,
                field,
                seed=stable_seed(20250701, model_name, cue_name, metric_name),
            )
            output.append(
                {
                    "model": model_name,
                    "cue": cue_name,
                    "metric": metric_name,
                    "n": len(rows),
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return output


def _write_evaluation_artifacts(
    metric_rows: list[dict[str, Any]],
    value_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = GENERATED_DIR / "probe_sensitivity_metrics.csv"
    metrics_buffer = io.StringIO(newline="")
    metrics_writer = csv.DictWriter(
        metrics_buffer, fieldnames=list(metric_rows[0])
    )
    metrics_writer.writeheader()
    metrics_writer.writerows(metric_rows)
    _atomic_bytes(metrics_path, metrics_buffer.getvalue().encode("utf-8"))

    values_path = GENERATED_DIR / "probe_sensitivity_values.csv.gz"
    values_buffer = io.StringIO(newline="")
    values_writer = csv.DictWriter(
        values_buffer, fieldnames=list(value_rows[0])
    )
    values_writer.writeheader()
    values_writer.writerows(value_rows)
    _atomic_bytes(
        values_path,
        gzip.compress(values_buffer.getvalue().encode("utf-8"), mtime=0),
    )

    table_path = GENERATED_DIR / "probe_sensitivity_table.tex"
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"Model & Cue & Parse rate & Answer agreement & "
            r"Benefit agreement & Harm agreement \\"
        ),
        r"\midrule",
    ]
    index = {
        (row["model"], row["cue"], row["metric"]): row
        for row in metric_rows
    }

    def interval(row: dict[str, Any]) -> str:
        return (
            f"{100 * row['estimate']:.1f}\\% "
            f"[{100 * row['ci_low']:.1f}, {100 * row['ci_high']:.1f}]"
        )

    model_labels = {"qwen3": "Qwen3", "deepseek": "DeepSeek"}
    cue_labels = {
        "canonical": "Canonical",
        "direct": "Direct",
        "terse": "Terse",
    }
    for model_name in ("qwen3", "deepseek"):
        for cue_name in CUES:
            lines.append(
                "{} & {} & {} & {} & {} & {} \\\\".format(
                    model_labels[model_name],
                    cue_labels[cue_name],
                    interval(index[(model_name, cue_name, "parse_rate")]),
                    interval(
                        index[
                            (model_name, cue_name, "answer_agreement")
                        ]
                    ),
                    interval(
                        index[
                            (
                                model_name,
                                cue_name,
                                "beneficial_label_agreement",
                            )
                        ]
                    ),
                    interval(
                        index[
                            (
                                model_name,
                                cue_name,
                                "harmful_label_agreement",
                            )
                        ]
                    ),
                )
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _atomic_bytes(
        table_path, ("\n".join(lines) + "\n").encode("utf-8")
    )
    replay_table_path = (
        GENERATED_DIR / "probe_sensitivity_replay_table.tex"
    )
    replay_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Token identity & Text identity & Normalized-answer identity \\",
        r"\midrule",
    ]
    for model_name in ("qwen3", "deepseek"):
        replay_lines.append(
            "{} & {} & {} & {} \\\\".format(
                model_labels[model_name],
                interval(
                    index[
                        (model_name, "canonical", "source_token_replay")
                    ]
                ),
                interval(
                    index[
                        (model_name, "canonical", "source_text_replay")
                    ]
                ),
                interval(
                    index[
                        (model_name, "canonical", "source_answer_replay")
                    ]
                ),
            )
        )
    replay_lines.extend([r"\bottomrule", r"\end{tabular}"])
    _atomic_bytes(
        replay_table_path,
        ("\n".join(replay_lines) + "\n").encode("utf-8"),
    )
    artifacts = [
        metrics_path,
        values_path,
        table_path,
        replay_table_path,
    ]
    manifest["artifacts_sha256"] = {
        path.name: file_sha256(path) for path in artifacts
    }
    manifest["script_sha256"] = file_sha256(Path(__file__))
    manifest_path = GENERATED_DIR / "probe_sensitivity_manifest.json"
    sealed_manifest = seal_record(manifest)
    _atomic_json(manifest_path, sealed_manifest)
    gate = (
        "% Generated only after sealed probe outputs pass all hard gates.\n"
        f"% manifest_record_sha256={sealed_manifest['record_sha256']}\n"
        "\\newif\\ifprobevalidated\n"
        "\\probevalidatedtrue\n"
    )
    _atomic_bytes(PROBE_GATE_PATH, gate.encode("utf-8"))


def evaluate(qwen_results: Path, deepseek_results: Path) -> None:
    # Invalidate any older included table before reading new outputs.  A
    # failed or interrupted evaluation must leave the paper fail-closed.
    _atomic_bytes(PROBE_GATE_PATH, PENDING_PROBE_GATE.encode("utf-8"))
    labels = _load_labels()
    all_values: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    source_results: dict[str, Any] = {}
    for model_name, result_dir in (
        ("qwen3", qwen_results),
        ("deepseek", deepseek_results),
    ):
        input_manifest, outputs, runtime = _load_outputs(
            model_name, result_dir.resolve()
        )
        values, diagnostics = _scientific_rows(
            model_name, outputs, labels
        )
        metrics = _metric_rows(model_name, values, diagnostics)
        all_values.extend(values)
        all_metrics.extend(metrics)
        source_results[model_name] = {
            "result_dir": str(result_dir.resolve()),
            "input_manifest_record_sha256": input_manifest["record_sha256"],
            "input_sha256": input_manifest["input_sha256"],
            "source_tree_sha256": input_manifest["source_tree_sha256"],
            "execution_bundle_sha256": input_manifest[
                "execution_bundle"
            ]["bundle_sha256"],
            "source_result_merkle_root": input_manifest[
                "source_result_merkle_root"
            ],
            "canonical_control_replay_rate": diagnostics[
                "canonical_control_replay_rate"
            ],
            "canonical_control_text_replay_rate": diagnostics[
                "canonical_control_text_replay_rate"
            ],
            "canonical_control_answer_replay_rate": diagnostics[
                "canonical_control_answer_replay_rate"
            ],
            "runtime": runtime,
        }
    manifest = {
        "schema_version": "flcva-probe-sensitivity-manifest-v5",
        "study_id": STUDY_ID,
        "analysis_status": "post-seal, label-aware, independent ablation",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "sealed_labels_sha256": file_sha256(LABELS_PATH),
        "source_results": source_results,
    }
    _write_evaluation_artifacts(
        all_metrics, all_values, manifest
    )
    print(
        json.dumps(
            {
                "models": 2,
                "value_rows": len(all_values),
                "metric_rows": len(all_metrics),
                "manifest": str(
                    GENERATED_DIR / "probe_sensitivity_manifest.json"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("validate-inputs")
    generation = subparsers.add_parser("generate")
    generation.add_argument("--model", choices=tuple(SOURCE_RUNS), required=True)
    generation.add_argument("--worker-index", type=int, required=True)
    generation.add_argument("--output-dir", type=Path, required=True)
    generation.add_argument(
        "--batch-size", type=int, default=FORMAL_BRANCH_BATCH_SIZE
    )
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--qwen-results", type=Path, required=True)
    evaluation.add_argument("--deepseek-results", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_inputs()
    elif args.command == "validate-inputs":
        validate_inputs()
    elif args.command == "generate":
        generate(
            args.model,
            args.worker_index,
            args.output_dir,
            args.batch_size,
        )
    elif args.command == "evaluate":
        evaluate(args.qwen_results, args.deepseek_results)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
