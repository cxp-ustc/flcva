from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import RunConfig
from .contract import (
    bind_run_record,
    build_contract,
    reseal_run_record,
    verify_run_record,
)
from .frozen import formal_difficulty_design, validate_formal_manifest_config
from .ids import payload_sha256
from .io import (
    BlobStore,
    atomic_write_json,
    file_sha256,
    iter_problem_artifacts,
    read_json_gz,
    seal_record,
    verify_sealed_record,
)
from .labels import verify_gold_label_attestation
from .lineage import verify_formal_difficulty_lineage
from .runner import resolve_run_dir
from .schemas import ExperimentContract, RunRecord


SNAPSHOT_IDENTITY_KEYS = (
    "expected_revision",
    "revision_evidence",
    "transport_attestation_sha256",
    "transport_requested_revision",
    "transport_resolution_mode",
    "transport_metadata_sha256",
    "config_sha256",
    "tokenizer_config_sha256",
    "weight_layout_sha256",
)


def _snapshot_identity(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in SNAPSHOT_IDENTITY_KEYS}


def _merkle_root(digests: list[str]) -> str:
    nodes = [bytes.fromhex(value) for value in sorted(digests)]
    if not nodes:
        return hashlib.sha256(b"").hexdigest()
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(nodes[index] + nodes[index + 1]).digest()
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def audit_run(config: RunConfig, project_root: Path) -> dict[str, Any]:
    run_dir = resolve_run_dir(config, project_root)
    contract = ExperimentContract.model_validate_json(
        (run_dir / "contract.json").read_text(encoding="utf-8")
    )
    contract_payload = contract.model_dump(mode="json")
    recorded_contract_sha = contract_payload["contract_sha256"]
    contract_payload["contract_sha256"] = ""
    if payload_sha256(contract_payload) != recorded_contract_sha:
        raise ValueError("contract self-hash mismatch")
    if build_contract(config, project_root) != contract:
        raise ValueError(
            "current config/source tree does not match the recorded experiment contract"
        )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if not verify_sealed_record(manifest):
        raise ValueError("manifest self-hash mismatch")
    run_path = run_dir / "run.json"
    record = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
    verify_run_record(record)
    if (
        record.run_id != config.run_id
        or record.contract_sha256 != contract.contract_sha256
    ):
        raise ValueError("run record identity mismatch")
    if manifest.get("run_record_sha256") != record.record_sha256:
        raise ValueError("manifest does not bind the run record")
    if (
        manifest.get("runtime_environment_sha256")
        != record.runtime_environment_sha256
    ):
        raise ValueError("manifest does not bind the runtime environment")
    if manifest.get("contract_sha256") != contract.contract_sha256:
        raise ValueError("manifest contract mismatch")
    if manifest.get("source_tree_sha256") != contract.source_tree_sha256:
        raise ValueError("manifest source tree mismatch")
    manifest_config = manifest.get("config")
    if manifest_config != config.model_dump(mode="json"):
        raise ValueError("manifest config differs from the audit config")
    if payload_sha256(manifest_config) != contract.config_sha256:
        raise ValueError("manifest config digest differs from the contract")
    frozen_campaign_match = True
    runtime_lineage_verified = True
    formal_preflight: dict[str, Any] | None = None
    if config.stage == "formal":
        frozen_campaign_match = (
            validate_formal_manifest_config(manifest_config, project_root)
            == contract.frozen_campaign_sha256
        )
        parent_rows = manifest.get("parent_runs", [])
        if parent_rows:
            parent_runtime_digests = {
                str(row.get("runtime_environment_sha256", ""))
                for row in parent_rows
            }
            runtime_lineage_verified = (
                len(parent_runtime_digests) == 1
                and "" not in parent_runtime_digests
                and len(
                    {
                        payload_sha256(row.get("runtime_model_identity", {}))
                        for row in parent_rows
                    }
                )
                == 1
                and len(
                    {
                        payload_sha256(row.get("runtime_scorer_identity", {}))
                        for row in parent_rows
                    }
                )
                == 1
                and all(
                    row.get("task_id")
                    and row.get("application_id")
                    and row.get("run_record_sha256")
                    and row.get("preflight_record_sha256")
                    and row.get("runtime_model_identity", {}).get(
                        "expected_revision"
                    )
                    == contract.model_revision
                    and row.get("runtime_model_identity", {}).get(
                        "config_sha256"
                    )
                    and row.get("runtime_scorer_identity", {}).get(
                        "expected_revision"
                    )
                    == contract.scorer_revision
                    and row.get("runtime_scorer_identity", {}).get(
                        "config_sha256"
                    )
                    for row in parent_rows
                )
            )
        else:
            preflight_path = run_dir / "preflight.json"
            runtime_lineage_verified = preflight_path.is_file()
            if runtime_lineage_verified:
                preflight = json.loads(
                    preflight_path.read_text(encoding="utf-8")
                )
                formal_preflight = preflight
                gpu_checks = preflight.get("runtime", {}).get(
                    "gpu_model_checks"
                ) or {}
                runtime_lineage_verified = (
                    verify_sealed_record(preflight)
                    and preflight.get("schema_version") == "flcva-preflight-v2"
                    and preflight.get("passed") is True
                    and preflight.get("run_id") == config.run_id
                    and preflight.get("contract_sha256")
                    == contract.contract_sha256
                    and preflight.get("runtime", {}).get(
                        "environment_sha256"
                    )
                    == record.runtime_environment_sha256
                    and bool(
                        gpu_checks
                    )
                    and _snapshot_identity(manifest.get("runtime_model", {}))
                    == _snapshot_identity(gpu_checks.get("model", {}))
                    and bool(record.task_id)
                    and bool(record.application_id)
                )
    allowed_contracts = {
        parent["contract_sha256"]
        for parent in manifest.get("parent_runs", [])
    } or {contract.contract_sha256}
    blobs = BlobStore(run_dir / "blobs")
    problem_hashes: list[str] = []
    result_hashes: list[str] = []
    totals: Counter[str] = Counter()
    attrition: Counter[str] = Counter()
    prefix_checks = 0
    prefix_matches = 0
    complete_branches = 0
    eligible_checkpoints = 0
    problem_ids: list[str] = []
    for artifact in iter_problem_artifacts(run_dir):
        problem_ids.append(artifact["problem_id"])
        problem_hashes.append(artifact["record_sha256"])
        checkpoints = artifact["checkpoints"]
        branches = artifact["branches"]
        event_ids = {event["event_id"] for event in artifact["ledger"]}
        by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for branch in branches:
            by_checkpoint[branch["checkpoint_id"]].append(branch)
            missing_events = set(branch["ledger_event_ids"]) - event_ids
            if missing_events:
                raise ValueError(
                    f"branch references absent ledger events: {sorted(missing_events)}"
                )
            blobs.verify(branch["reasoning_blob"])
            if branch["reasoning_tokens"] != config.protocol.continuation_tokens:
                raise ValueError("branch reasoning length violates the fixed intervention")
            totals["branches"] += 1
            totals["logical_branch_tokens"] += int(branch["logical_tokens"])
        for checkpoint in checkpoints:
            prompt = blobs.get(checkpoint["prompt_blob"])
            spine = blobs.get(checkpoint["spine_blob"])
            prefix = prompt + spine[: checkpoint["depth"]]
            prefix_checks += 1
            if (
                payload_sha256(prefix) == checkpoint["prefix_sha256"]
                and len(prefix) == checkpoint["prefix_token_count"]
            ):
                prefix_matches += 1
            else:
                raise ValueError(
                    f"checkpoint prefix reconstruction failed: {checkpoint['checkpoint_id']}"
                )
            if checkpoint["continuation_eligible"]:
                eligible_checkpoints += 1
                rows = by_checkpoint.get(checkpoint["checkpoint_id"], [])
                if (
                    len(rows) == config.protocol.continuation_branches
                    and sorted(row["branch_index"] for row in rows)
                    == list(range(1, config.protocol.continuation_branches + 1))
                ):
                    complete_branches += 1
                else:
                    attrition["incomplete_branch_set"] += 1
                on_spine = [row for row in rows if row["on_spine"]]
                if len(on_spine) != 1 or on_spine[0]["branch_index"] != 1:
                    raise ValueError("each checkpoint must have one k=1 on-spine branch")
                expected = spine[
                    checkpoint["depth"] : checkpoint["depth"]
                    + config.protocol.continuation_tokens
                ]
                observed = blobs.get(on_spine[0]["reasoning_blob"])
                if expected != observed:
                    raise ValueError("on-spine branch does not match successor token segment")
            totals["checkpoints"] += 1
            for event_id in checkpoint["ledger_event_ids"]:
                if event_id not in event_ids:
                    raise ValueError("checkpoint references an absent ledger event")
        forbidden_keys = {"gold", "gold_answer", "label", "target", "reference_answer"}
        for key in forbidden_keys:
            if key in artifact:
                raise ValueError(f"generation artifact leaks a forbidden label key: {key}")
        totals["problems"] += 1
        result_hashes.append(artifact["record_sha256"])
        for event in artifact["ledger"]:
            totals["ledger_events"] += 1
            totals["ledger_logical_tokens"] += int(event["logical_tokens"])
            totals["physical_output_tokens"] += int(event["output_tokens"])
    for artifact in iter_problem_artifacts(run_dir):
        if artifact["contract_sha256"] not in allowed_contracts:
            raise ValueError(
                "problem artifact is outside the declared contract lineage: "
                f"{artifact['problem_id']}"
            )
    if sorted(problem_ids) != sorted(manifest["problem_ids"]):
        raise ValueError("manifest and artifact problem sets differ")
    score_problem_ids: set[str] = set()
    score_runtime_identities: set[str] = set()
    for path in sorted((run_dir / "scores").glob("*.json.gz")):
        payload = read_json_gz(path)
        if not verify_sealed_record(payload):
            raise ValueError(f"score artifact hash mismatch: {path}")
        if payload["contract_sha256"] not in allowed_contracts:
            raise ValueError(f"score artifact is outside declared lineage: {path}")
        if (
            payload.get("scorer_id") != contract.scorer_id
            or payload.get("scorer_revision") != contract.scorer_revision
        ):
            raise ValueError(f"score artifact scorer differs from contract: {path}")
        problem_id = str(payload["problem_id"])
        if problem_id in score_problem_ids:
            raise ValueError(f"duplicate score artifact: {problem_id}")
        score_problem_ids.add(problem_id)
        score_runtime_identities.add(
            payload_sha256(_snapshot_identity(payload.get("runtime_scorer", {})))
        )
        result_hashes.append(payload["record_sha256"])
        totals["score_artifacts"] += 1
        totals["prm_ledger_events"] += len(payload.get("ledger", []))
    if config.stage == "formal":
        if manifest.get("parent_runs"):
            parent_scorer_identities = {
                payload_sha256(row.get("runtime_scorer_identity", {}))
                for row in manifest["parent_runs"]
            }
            runtime_lineage_verified = (
                runtime_lineage_verified
                and len(score_runtime_identities) == 1
                and score_runtime_identities == parent_scorer_identities
            )
        elif formal_preflight is not None:
            expected_scorer = formal_preflight["runtime"]["gpu_model_checks"][
                "scorer"
            ]
            runtime_lineage_verified = (
                runtime_lineage_verified
                and score_runtime_identities
                == {payload_sha256(_snapshot_identity(expected_scorer))}
            )
    runtime_model = manifest.get("runtime_model", {})
    totals["model_load_seconds"] = float(
        runtime_model.get("load_seconds") or 0.0
    )
    totals["model_load_peak_gpu_memory_bytes"] = int(
        runtime_model.get("load_peak_gpu_memory_bytes") or 0
    )
    score_summary_path = run_dir / "score_summary.json"
    if score_summary_path.is_file():
        score_summary = json.loads(score_summary_path.read_text(encoding="utf-8"))
        runtime_scorer = score_summary.get("runtime_scorer", {})
        totals["prm_model_load_seconds"] = float(
            runtime_scorer.get("load_seconds") or 0.0
        )
        totals["prm_model_load_peak_gpu_memory_bytes"] = int(
            runtime_scorer.get("load_peak_gpu_memory_bytes") or 0
        )
    prefix_rate = prefix_matches / prefix_checks if prefix_checks else 0.0
    branch_rate = complete_branches / eligible_checkpoints if eligible_checkpoints else 0.0
    parse_rates: dict[str, dict[str, float | None]] = {}
    evaluation_summary_path = run_dir / "evaluation" / "summary.json"
    evaluation_valid = False
    gold_labels_match = False
    difficulty_peer_match = config.stage != "formal"
    if evaluation_summary_path.exists():
        evaluation = json.loads(evaluation_summary_path.read_text(encoding="utf-8"))
        if not verify_sealed_record(evaluation):
            raise ValueError("evaluation summary hash mismatch")
        if evaluation.get("run_id") != config.run_id:
            raise ValueError("evaluation summary run ID mismatch")
        for relative, expected in evaluation.get("artifacts_sha256", {}).items():
            target = run_dir / "evaluation" / relative
            if not target.is_file() or file_sha256(target) != expected:
                raise ValueError(f"evaluation artifact hash mismatch: {relative}")
        evaluation_valid = (
            int(evaluation.get("problems", -1)) == len(problem_ids)
            and bool(evaluation.get("artifacts_sha256"))
            and (
                (
                    int(evaluation.get("difficulty_scope_runs", -1))
                    == len(
                        formal_difficulty_design(
                            config,
                            project_root,
                        )["reference_models"]
                    )
                    and sorted(
                        evaluation.get(
                            "difficulty_reference_models",
                            evaluation.get(
                                "difficulty_lineage",
                                {},
                            ).get("models", []),
                        )
                    )
                    == sorted(
                        formal_difficulty_design(
                            config,
                            project_root,
                        )["reference_models"]
                    )
                    and bool(
                        evaluation.get(
                            "difficulty_current_outcomes_excluded",
                            False,
                        )
                    )
                    is not bool(
                        formal_difficulty_design(
                            config,
                            project_root,
                        )["include_current_outcomes"]
                    )
                )
                if config.stage == "formal"
                else True
            )
        )
        gold_labels_match = verify_gold_label_attestation(
            config,
            project_root,
            evaluation.get("gold_labels", {}),
        )
        if config.stage == "formal":
            difficulty_peer_match = verify_formal_difficulty_lineage(
                config,
                project_root,
                run_dir,
                evaluation.get("difficulty_lineage", {}),
            )
        result_hashes.append(evaluation["record_sha256"])
        parse_counts = evaluation["parse_counts"]
        datasets = sorted({key.split(":", 1)[0] for key in parse_counts})
        for dataset_id in datasets:
            parse_rates[dataset_id] = {}
            for scope in ("stop", "branch"):
                prefix = f"{dataset_id}:{scope}:"
                relevant = {
                    key: value
                    for key, value in parse_counts.items()
                    if key.startswith(prefix)
                }
                total = sum(relevant.values())
                success = sum(
                    value
                    for key, value in relevant.items()
                    if "parse_failure" not in key
                )
                parse_rates[dataset_id][scope] = success / total if total else None
    expected_datasets = {
        artifact["dataset_id"] for artifact in iter_problem_artifacts(run_dir)
    }
    parser_gate = evaluation_valid and expected_datasets == set(parse_rates)
    for dataset_id in expected_datasets:
        rates = parse_rates.get(dataset_id, {})
        threshold = 0.90 if dataset_id == "math500" else 0.95
        parser_gate = parser_gate and (
            rates.get("stop") is not None
            and float(rates["stop"]) >= threshold
            and rates.get("branch") is not None
            and float(rates["branch"]) >= threshold
        )
    replay_summary_path = run_dir / "replay" / "summary.json"
    replay_valid = False
    if replay_summary_path.exists():
        replay = json.loads(replay_summary_path.read_text(encoding="utf-8"))
        if not verify_sealed_record(replay):
            raise ValueError("replay summary hash mismatch")
        if replay.get("run_id") != config.run_id:
            raise ValueError("replay summary run ID mismatch")
        for relative, expected in replay.get("artifacts_sha256", {}).items():
            target = run_dir / "replay" / relative
            if not target.is_file() or file_sha256(target) != expected:
                raise ValueError(f"replay artifact hash mismatch: {relative}")
        replay_valid = bool(replay.get("artifacts_sha256"))
        result_hashes.append(replay["record_sha256"])
    gates = {
        "prefix_replay_100pct": prefix_rate == 1.0,
        "branch_completeness_ge_99_5pct": branch_rate >= 0.995,
        "score_count_matches_manifest": score_problem_ids == set(problem_ids),
        "evaluation_present_and_sealed": evaluation_valid,
        "gold_labels_match": gold_labels_match,
        "difficulty_peer_match": difficulty_peer_match,
        "replay_present_and_sealed": replay_valid,
        "parser_rate_gate": parser_gate,
        "artifact_count_matches_manifest": len(problem_ids) == manifest["problem_count"],
        "frozen_campaign_match": (
            frozen_campaign_match
            and contract.frozen_campaign_sha256 is not None
            if config.stage == "formal"
            else True
        ),
        "runtime_lineage_verified": runtime_lineage_verified,
    }
    audit = seal_record(
        {
            "schema_version": "flcva-audit-manifest-v1",
            "run_id": config.run_id,
            "protocol_id": config.protocol_id,
            "contract_sha256": contract.contract_sha256,
            "problem_merkle_root": _merkle_root(problem_hashes),
            "result_merkle_root": _merkle_root(result_hashes),
            "problem_ids_sha256": payload_sha256(sorted(problem_ids)),
            "totals": dict(sorted(totals.items())),
            "prefix_replay_rate": prefix_rate,
            "branch_completeness_rate": branch_rate,
            "parse_rates": parse_rates,
            "attrition": dict(sorted(attrition.items())),
            "gates": gates,
            "passed": all(gates.values()),
        }
    )
    atomic_write_json(run_dir / "audit_manifest.json", audit)
    if audit["passed"]:
        sealed_record = reseal_run_record(record, status="sealed")
        atomic_write_json(
            run_path,
            sealed_record.model_dump(mode="json"),
        )
        atomic_write_json(
            run_dir / "manifest.json",
            bind_run_record(manifest, sealed_record),
        )
    return audit
