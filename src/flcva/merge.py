from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import RunConfig
from .contract import (
    bind_run_record,
    build_manifest,
    finish_run_record,
    materialize_contract,
    new_run_record,
    reseal_run_record,
    verify_run_record,
)
from .data import load_public_problems
from .frozen import validate_formal_config, validate_formal_manifest_config
from .ids import payload_sha256
from .io import (
    atomic_write_json,
    iter_problem_artifacts,
    problem_artifact_path,
    read_json_gz,
    verify_sealed_record,
)
from .runner import resolve_run_dir
from .schemas import ExperimentContract
from .schemas import RunRecord


def _invariants(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in (
            "protocol_id",
            "stage",
            "backend",
            "global_seed",
            "model",
            "dataset",
            "sampling",
            "protocol",
            "scorer",
        )
    }


SNAPSHOT_IDENTITY_KEYS = (
    "expected_revision",
    "revision_evidence",
    "transport_requested_revision",
    "transport_resolution_mode",
)


def _snapshot_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in SNAPSHOT_IDENTITY_KEYS}


def merge_runs(
    config: RunConfig,
    project_root: Path,
    input_dirs: list[Path],
) -> dict[str, Any]:
    if len(input_dirs) < 2:
        raise ValueError("merge requires at least two input run directories")
    if config.stage == "formal":
        validate_formal_config(config, project_root)
    manifests = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for path in input_dirs
    ]
    for input_dir, manifest in zip(input_dirs, manifests, strict=True):
        if not verify_sealed_record(manifest):
            raise ValueError(f"parent manifest hash mismatch: {input_dir}")
    parent_records = [
        RunRecord.model_validate_json((path / "run.json").read_text(encoding="utf-8"))
        for path in input_dirs
    ]
    for input_dir, manifest, record in zip(
        input_dirs,
        manifests,
        parent_records,
        strict=True,
    ):
        verify_run_record(record)
        if record.status not in {"complete", "sealed"}:
            raise ValueError(f"parent run is not complete: {input_dir}")
        if record.run_id != manifest.get("run_id"):
            raise ValueError(f"parent run/manifest identity mismatch: {input_dir}")
        if record.contract_sha256 != manifest.get("contract_sha256"):
            raise ValueError(f"parent run/manifest contract mismatch: {input_dir}")
        if manifest.get("run_record_sha256") != record.record_sha256:
            raise ValueError(f"parent manifest does not bind run record: {input_dir}")
        if (
            manifest.get("runtime_environment_sha256")
            != record.runtime_environment_sha256
        ):
            raise ValueError(
                f"parent manifest does not bind runtime environment: {input_dir}"
            )
    runtime_digests = {record.runtime_environment_sha256 for record in parent_records}
    if len(runtime_digests) != 1:
        raise ValueError("input shards were produced by different runtime environments")
    formal_preflights: dict[Path, dict[str, Any]] = {}
    if config.stage == "formal":
        for input_dir, manifest, record in zip(
            input_dirs,
            manifests,
            parent_records,
            strict=True,
        ):
            if not record.task_id or not record.application_id:
                raise ValueError(
                    f"formal parent lacks cluster task/application identity: {input_dir}"
                )
            preflight_path = input_dir / "preflight.json"
            if not preflight_path.is_file():
                raise ValueError(f"formal parent lacks GPU preflight: {input_dir}")
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            if not verify_sealed_record(preflight):
                raise ValueError(f"formal parent preflight hash mismatch: {input_dir}")
            if (
                preflight.get("schema_version") != "flcva-preflight-v2"
                or not preflight.get("passed")
                or preflight.get("run_id") != record.run_id
                or preflight.get("contract_sha256") != record.contract_sha256
            ):
                raise ValueError(
                    f"formal parent preflight identity mismatch: {input_dir}"
                )
            runtime = preflight.get("runtime", {})
            if runtime.get(
                "environment_sha256"
            ) != record.runtime_environment_sha256 or not runtime.get(
                "gpu_model_checks"
            ):
                raise ValueError(
                    f"formal parent preflight/runtime mismatch: {input_dir}"
                )
            gpu_checks = runtime["gpu_model_checks"]
            if _snapshot_identity(
                manifest.get("runtime_model", {})
            ) != _snapshot_identity(gpu_checks.get("model", {})):
                raise ValueError(
                    f"formal generation model differs from preflight: {input_dir}"
                )
            formal_preflights[input_dir] = preflight
            frozen_campaign_sha256 = validate_formal_manifest_config(
                manifest.get("config", {}),
                project_root,
            )
            parent_contract = ExperimentContract.model_validate_json(
                (input_dir / "contract.json").read_text(encoding="utf-8")
            )
            contract_payload = parent_contract.model_dump(mode="json")
            recorded_contract_sha = contract_payload["contract_sha256"]
            contract_payload["contract_sha256"] = ""
            if payload_sha256(contract_payload) != recorded_contract_sha:
                raise ValueError(f"parent contract self-hash mismatch: {input_dir}")
            if (
                manifest.get("contract_sha256") != recorded_contract_sha
                or parent_contract.config_sha256
                != payload_sha256(manifest.get("config", {}))
                or parent_contract.frozen_campaign_sha256 != frozen_campaign_sha256
            ):
                raise ValueError(
                    f"parent run differs from frozen campaign contract: {input_dir}"
                )
    expected = _invariants(manifests[0]["config"])
    for manifest in manifests[1:]:
        if _invariants(manifest["config"]) != expected:
            raise ValueError("cannot merge runs with different scientific invariants")
    shard_pairs = {
        (
            int(manifest["config"]["execution"]["shard_index"]),
            int(manifest["config"]["execution"]["num_shards"]),
        )
        for manifest in manifests
    }
    num_shards = {pair[1] for pair in shard_pairs}
    if len(num_shards) != 1:
        raise ValueError("input runs disagree on num_shards")
    expected_indices = set(range(next(iter(num_shards))))
    observed_indices = {pair[0] for pair in shard_pairs}
    if observed_indices != expected_indices:
        raise ValueError(
            f"merge needs every shard exactly once: expected={expected_indices}, "
            f"observed={observed_indices}"
        )
    declared_problem_ids: list[str] = [
        str(problem_id)
        for manifest in manifests
        for problem_id in manifest["problem_ids"]
    ]
    if len(declared_problem_ids) != len(set(declared_problem_ids)):
        raise ValueError("input manifests contain duplicate problem IDs")
    expected_problem_ids = {
        problem.id for problem in load_public_problems(config, project_root)
    }
    if set(declared_problem_ids) != expected_problem_ids:
        missing = sorted(expected_problem_ids - set(declared_problem_ids))
        unexpected = sorted(set(declared_problem_ids) - expected_problem_ids)
        raise ValueError(
            "input shard manifests do not exactly cover the merged problem set: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    if config.stage == "formal":
        for input_dir, manifest in zip(input_dirs, manifests, strict=True):
            score_payloads = [
                read_json_gz(path)
                for path in sorted((input_dir / "scores").glob("*.json.gz"))
            ]
            if any(not verify_sealed_record(payload) for payload in score_payloads):
                raise ValueError(
                    f"formal input has a corrupt score artifact: {input_dir}"
                )
            score_ids = {str(payload["problem_id"]) for payload in score_payloads}
            if score_ids != set(manifest["problem_ids"]):
                raise ValueError(f"formal input score set is incomplete: {input_dir}")
            expected_scorer = formal_preflights[input_dir]["runtime"][
                "gpu_model_checks"
            ]["scorer"]
            if any(
                _snapshot_identity(payload.get("runtime_scorer", {}))
                != _snapshot_identity(expected_scorer)
                for payload in score_payloads
            ):
                raise ValueError(
                    f"formal scorer snapshot differs from preflight: {input_dir}"
                )
    output_dir = resolve_run_dir(config, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = materialize_contract(config, project_root, output_dir)
    parent_task_ids = sorted(
        {str(record.task_id) for record in parent_records if record.task_id}
    )
    run_record = reseal_run_record(
        new_run_record(config, contract),
        parent_run_ids=[str(item["run_id"]) for item in manifests],
        parent_task_ids=parent_task_ids,
    )
    seen: set[str] = set()
    problem_ids: list[str] = []
    for input_dir in input_dirs:
        for source in sorted((input_dir / "blobs").rglob("*.json.gz")):
            relative = source.relative_to(input_dir / "blobs")
            target = output_dir / "blobs" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            elif source.read_bytes() != target.read_bytes():
                raise ValueError(f"blob path collision: {relative}")
        for artifact in iter_problem_artifacts(input_dir):
            problem_id = artifact["problem_id"]
            if problem_id in seen:
                raise ValueError(f"duplicate problem across shards: {problem_id}")
            seen.add(problem_id)
            problem_ids.append(problem_id)
            source_path = problem_artifact_path(input_dir, problem_id)
            target_path = problem_artifact_path(output_dir, problem_id)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        for source in sorted((input_dir / "scores").glob("*.json.gz")):
            payload = read_json_gz(source)
            if not verify_sealed_record(payload):
                raise ValueError(f"score artifact hash mismatch: {source}")
            target = output_dir / "scores" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"duplicate score artifact: {source.name}")
            shutil.copy2(source, target)
    manifest = build_manifest(config, contract, project_root, problem_ids)
    manifest["parent_runs"] = [
        {
            "run_id": item["run_id"],
            "contract_sha256": item["contract_sha256"],
            "problem_ids_sha256": item["problem_ids_sha256"],
            "task_id": record.task_id,
            "application_id": record.application_id,
            "run_record_sha256": record.record_sha256,
            "runtime_environment_sha256": record.runtime_environment_sha256,
            "preflight_record_sha256": (
                json.loads(
                    (input_dir / "preflight.json").read_text(encoding="utf-8")
                ).get("record_sha256")
                if (input_dir / "preflight.json").is_file()
                else None
            ),
            "runtime_model_identity": (
                _snapshot_identity(item.get("runtime_model", {}))
                if config.stage == "formal"
                else None
            ),
            "runtime_scorer_identity": (
                _snapshot_identity(
                    formal_preflights[input_dir]["runtime"]["gpu_model_checks"][
                        "scorer"
                    ]
                )
                if config.stage == "formal"
                else None
            ),
        }
        for input_dir, item, record in zip(
            input_dirs,
            manifests,
            parent_records,
            strict=True,
        )
    ]
    completed_record = finish_run_record(run_record, "complete")
    atomic_write_json(
        output_dir / "run.json",
        completed_record.model_dump(mode="json"),
    )
    manifest = bind_run_record(manifest, completed_record)
    atomic_write_json(output_dir / "manifest.json", manifest)
    summary = {
        "schema_version": "flcva-merge-summary-v1",
        "run_id": config.run_id,
        "inputs": [str(path) for path in input_dirs],
        "problems": len(problem_ids),
        "parent_runs": [item["run_id"] for item in manifests],
    }
    atomic_write_json(output_dir / "merge_summary.json", summary)
    return summary
