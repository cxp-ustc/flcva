from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import RunConfig
from .ids import payload_sha256
from .io import (
    atomic_write_json,
    file_sha256,
    seal_record,
    verify_sealed_record,
)
from .frozen import validate_formal_config
from .schemas import ExperimentContract, RunRecord


SOURCE_ROOTS = ("src", "scripts")
SOURCE_FILES = (
    "pyproject.toml",
    "README.md",
)


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def source_tree_fingerprint(project_root: Path) -> tuple[str, int]:
    entries: list[tuple[str, str]] = []
    paths: list[Path] = []
    for name in SOURCE_ROOTS:
        root = project_root / name
        if root.is_dir():
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not any(
                    part.endswith(".egg-info") for part in path.parts
                )
                and path.suffix not in {".pyc", ".pyo"}
                and not path.name.endswith(
                    (".aux", ".log", ".out", ".bbl", ".blg", ".fls", ".fdb_latexmk")
                )
            )
    paths.extend(
        project_root / name
        for name in SOURCE_FILES
        if (project_root / name).is_file()
    )
    for path in sorted(set(paths), key=lambda item: str(item.relative_to(project_root))):
        entries.append((str(path.relative_to(project_root)), file_sha256(path)))
    return payload_sha256(entries), len(entries)


def _git(project_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_contract(config: RunConfig, project_root: Path) -> ExperimentContract:
    frozen_campaign_sha256 = (
        validate_formal_config(config, project_root)
        if config.stage == "formal"
        else None
    )
    config_payload = config.model_dump(mode="json")
    config_sha = payload_sha256(config_payload)
    source_sha, _ = source_tree_fingerprint(project_root)
    prompt_template = {
        "system": (
            "Solve the problem by reasoning step by step. Keep reasoning until "
            "explicitly asked for a final answer, then put only the final answer "
            "inside \\boxed{}."
        ),
        "answer_cue": config.protocol.answer_cue,
    }
    parent_sha = None
    if config.execution.parent_contract:
        parent = config.resolve_path(project_root, config.execution.parent_contract)
        parent_sha = file_sha256(parent)
    if config.dataset.source == "jsonl":
        public_path = config.resolve_path(
            project_root,
            str(config.dataset.public_path),
        )
        public_data_sha256 = file_sha256(public_path)
    else:
        public_data_sha256 = config.dataset.revision
    data_manifest_path = project_root / "data" / "frozen" / "MANIFEST.json"
    public_data_manifest_sha256 = None
    dataset_components: dict[str, str] = {}
    if data_manifest_path.is_file():
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        component_rows = data_manifest.get("datasets", {})
        selected_components = (
            sorted(component_rows)
            if config.dataset.id == "combined"
            else [config.dataset.id]
        )
        for dataset_id in selected_components:
            row = component_rows.get(dataset_id)
            if row:
                dataset_components[dataset_id] = (
                    f"{row['repository']}@{row['revision']}"
                )
        public_component_manifest = {
            dataset_id: {
                key: value
                for key, value in component_rows[dataset_id].items()
                if key != "sealed_labels_sha256"
            }
            for dataset_id in selected_components
            if dataset_id in component_rows
        }
        if public_component_manifest:
            public_data_manifest_sha256 = payload_sha256(
                public_component_manifest
            )
    if not dataset_components:
        dataset_components[config.dataset.id] = (
            f"{config.dataset.repo_id or config.dataset.id}@"
            f"{config.dataset.revision}"
        )
    payload: dict[str, Any] = {
        "schema_version": "flcva-experiment-contract-v2",
        "protocol_id": config.protocol_id,
        "run_id": config.run_id,
        "stage": config.stage,
        "config_sha256": config_sha,
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "tokenizer_revision": config.model.tokenizer_revision,
        "dataset_id": config.dataset.id,
        "dataset_revision": config.dataset.revision,
        "dataset_components": dataset_components,
        "public_data_manifest_sha256": public_data_manifest_sha256,
        "public_data_sha256": public_data_sha256,
        "data_split": config.dataset.split,
        "scorer_id": config.scorer.id,
        "scorer_revision": config.scorer.revision,
        "sampling": config.sampling.model_dump(mode="json"),
        "prompt_sha256": payload_sha256(prompt_template),
        "global_seed": config.global_seed,
        "spines_per_problem": config.protocol.spines_per_problem,
        "checkpoint_interval": config.protocol.checkpoint_interval,
        "reasoning_max_tokens": config.protocol.reasoning_max_tokens,
        "continuation_branches": config.protocol.continuation_branches,
        "continuation_tokens": config.protocol.continuation_tokens,
        "answer_max_tokens": config.protocol.answer_max_tokens,
        "split_rule": "sha256(dataset_id:problem_id)%100<80",
        "source_tree_sha256": source_sha,
        "parent_contract_sha256": parent_sha,
        "frozen_campaign_sha256": frozen_campaign_sha256,
        "contract_sha256": "",
    }
    payload["contract_sha256"] = payload_sha256(payload)
    return ExperimentContract.model_validate(payload)


def materialize_contract(
    config: RunConfig,
    project_root: Path,
    run_dir: Path,
) -> ExperimentContract:
    contract = build_contract(config, project_root)
    path = run_dir / "contract.json"
    if path.exists():
        existing = ExperimentContract.model_validate_json(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                "run_id already has a different experiment contract; choose a new run_id"
            )
    else:
        atomic_write_json(path, contract.model_dump(mode="json"))
    return contract


def load_recorded_contract(run_dir: Path) -> ExperimentContract:
    """Load and authenticate a historical contract without replaying its code.

    This is appropriate for read-only post-seal analysis: a later analysis-code
    change must not make an otherwise self-consistent historical generation
    lineage unverifiable.
    """

    path = run_dir / "contract.json"
    if not path.is_file():
        raise ValueError(f"run has no experiment contract: {run_dir}")
    contract = ExperimentContract.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    payload = contract.model_dump(mode="json")
    recorded_sha256 = payload["contract_sha256"]
    payload["contract_sha256"] = ""
    if payload_sha256(payload) != recorded_sha256:
        raise ValueError("recorded experiment contract self-hash mismatch")
    return contract


def verify_recorded_contract(
    config: RunConfig,
    project_root: Path,
    run_dir: Path,
) -> ExperimentContract:
    contract = load_recorded_contract(run_dir)
    if build_contract(config, project_root) != contract:
        raise ValueError(
            "current config/source tree does not match the recorded experiment contract"
        )
    return contract


def hardware_snapshot() -> dict[str, Any]:
    output: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        output = {
            "cuda_available": bool(torch.cuda.is_available()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        pass
    return output


RUNTIME_PACKAGES = (
    "antlr4-python3-runtime",
    "datasets",
    "huggingface-hub",
    "numpy",
    "huggingface-hub",
    "pydantic",
    "PyYAML",
    "requests",
    "sympy",
    "tokenizers",
    "torch",
    "transformers",
    "urllib3",
    "vllm",
)


def runtime_environment_snapshot() -> dict[str, Any]:
    """Capture the exact scientific runtime, excluding per-task identities."""
    packages = {name: _version(name) for name in RUNTIME_PACKAGES}
    hardware = hardware_snapshot()
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": packages,
        "hardware": hardware,
        "execution": {
            "vllm_enable_v1_multiprocessing": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ),
            "container_image": (
                os.environ.get("pouch_container_image")
                or os.environ.get("POD_IMAGE")
                or os.environ.get("CONTAINER_IMAGE")
            ),
            "cluster_job_name": os.environ.get("CLUSTER_JOB_NAME"),
            "scheduler_queue": os.environ.get("SCHEDULER_QUEUE"),
        },
    }


def reseal_run_record(record: RunRecord, **updates: Any) -> RunRecord:
    payload = record.model_dump(mode="json")
    payload.update(updates)
    draft = RunRecord.model_validate(payload)
    return RunRecord.model_validate(
        seal_record(draft.model_dump(mode="json"))
    )


def verify_run_record(record: RunRecord) -> None:
    payload = record.model_dump(mode="json")
    if not verify_sealed_record(payload):
        raise ValueError("run record self-hash mismatch")
    expected_runtime_sha256 = payload_sha256(record.runtime_environment)
    if record.runtime_environment_sha256 != expected_runtime_sha256:
        raise ValueError("run record runtime environment hash mismatch")
    if record.packages != record.runtime_environment.get("packages"):
        raise ValueError("run record package snapshot is inconsistent")
    if record.hardware != record.runtime_environment.get("hardware"):
        raise ValueError("run record hardware snapshot is inconsistent")


def bind_run_record(
    manifest: dict[str, Any],
    record: RunRecord,
) -> dict[str, Any]:
    verify_run_record(record)
    output = dict(manifest)
    output["runtime_environment_sha256"] = record.runtime_environment_sha256
    output["run_record_sha256"] = record.record_sha256
    return seal_record(output)


def new_run_record(
    config: RunConfig,
    contract: ExperimentContract,
) -> RunRecord:
    task_id = (
        os.environ.get("CLUSTER_TASK_ID")
        or os.environ.get("TASK_ID")
    )
    application_id = os.environ.get("APPLICATION_ID") or os.environ.get("APP_ID")
    task_scope = (
        os.environ.get("FLCVA_TASK_SCOPE")
        or application_id
        or task_id
        or socket.gethostname()
    )
    runtime_environment = runtime_environment_snapshot()
    payload = {
        "schema_version": "flcva-run-record-v2",
        "run_id": config.run_id,
        "protocol_id": config.protocol_id,
        "stage": config.stage,
        "status": "running",
        "backend": config.backend,
        "started_at": datetime.now(timezone.utc),
        "host": socket.gethostname(),
        "python": sys.version,
        "packages": runtime_environment["packages"],
        "hardware": runtime_environment["hardware"],
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": payload_sha256(runtime_environment),
        "contract_sha256": contract.contract_sha256,
        "parent_run_id": None,
        "task_id": task_id,
        "application_id": application_id,
        "task_scope": task_scope,
        "parent_task_id": os.environ.get("FLCVA_PARENT_TASK_ID"),
        "record_sha256": "",
    }
    draft = RunRecord.model_validate(payload)
    return RunRecord.model_validate(
        seal_record(draft.model_dump(mode="json"))
    )


def finish_run_record(record: RunRecord, status: str) -> RunRecord:
    return reseal_run_record(
        record,
        status=status,
        completed_at=datetime.now(timezone.utc),
    )


def build_manifest(
    config: RunConfig,
    contract: ExperimentContract,
    project_root: Path,
    problem_ids: list[str],
) -> dict[str, Any]:
    source_sha, source_count = source_tree_fingerprint(project_root)
    return {
        "schema_version": "flcva-manifest-v1",
        "run_id": config.run_id,
        "protocol_id": config.protocol_id,
        "contract_sha256": contract.contract_sha256,
        "config": config.model_dump(mode="json"),
        "problem_ids": sorted(problem_ids),
        "problem_ids_sha256": payload_sha256(sorted(problem_ids)),
        "problem_count": len(problem_ids),
        "source_tree_sha256": source_sha,
        "source_tree_file_count": source_count,
        "git_commit": _git(project_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(project_root, "status", "--porcelain", "--", ".")),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "vllm_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
        },
        "hardware": hardware_snapshot(),
    }
