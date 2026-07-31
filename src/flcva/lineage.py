from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import RunConfig
from .contract import load_recorded_contract
from .data import load_public_problems
from .frozen import (
    FROZEN_CAMPAIGN_SHA256,
    formal_difficulty_design,
    load_formal_campaign,
    validate_formal_manifest_config,
)
from .ids import payload_sha256
from .io import iter_problem_artifacts, verify_sealed_record


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


def _generation_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Hash immutable generation identity, excluding audit-state bindings."""
    stable = {
        key: value
        for key, value in manifest.items()
        if key not in {"record_sha256", "run_record_sha256"}
    }
    return payload_sha256(stable)


def generation_lineage(run_dir: Path, project_root: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not verify_sealed_record(manifest):
        raise ValueError(f"formal generation manifest seal mismatch: {run_dir}")
    config = RunConfig.model_validate(manifest.get("config", {}))
    if config.stage != "formal":
        raise ValueError(f"difficulty peer is not a formal run: {run_dir}")
    if run_dir.name != config.run_id:
        raise ValueError(f"formal run directory does not match run_id: {run_dir}")
    frozen_campaign_sha256 = validate_formal_manifest_config(
        manifest["config"],
        project_root,
    )
    contract = load_recorded_contract(run_dir)
    if (
        manifest.get("run_id") != config.run_id
        or manifest.get("protocol_id") != config.protocol_id
        or manifest.get("contract_sha256") != contract.contract_sha256
        or manifest.get("source_tree_sha256") != contract.source_tree_sha256
        or payload_sha256(manifest["config"]) != contract.config_sha256
        or contract.frozen_campaign_sha256 != frozen_campaign_sha256
    ):
        raise ValueError(f"formal generation manifest lineage mismatch: {run_dir}")
    declared_ids = [str(value) for value in manifest.get("problem_ids", [])]
    if (
        len(declared_ids) != len(set(declared_ids))
        or len(declared_ids) != int(manifest.get("problem_count", -1))
        or payload_sha256(sorted(declared_ids))
        != manifest.get("problem_ids_sha256")
    ):
        raise ValueError(f"formal generation manifest problem set is invalid: {run_dir}")
    allowed_contracts = {
        str(parent["contract_sha256"])
        for parent in manifest.get("parent_runs", [])
    } or {contract.contract_sha256}
    artifact_ids: list[str] = []
    artifact_hashes: list[str] = []
    for artifact in iter_problem_artifacts(run_dir):
        problem_id = str(artifact["problem_id"])
        artifact_ids.append(problem_id)
        artifact_hashes.append(str(artifact["record_sha256"]))
        if artifact.get("contract_sha256") not in allowed_contracts:
            raise ValueError(
                f"formal difficulty artifact is outside contract lineage: {problem_id}"
            )
        checkpoints = artifact.get("checkpoints", [])
        if not checkpoints or any(
            checkpoint.get("model_id") != config.model.id
            for checkpoint in checkpoints
        ):
            raise ValueError(
                f"formal difficulty artifact has the wrong model identity: {problem_id}"
            )
    if sorted(artifact_ids) != sorted(declared_ids):
        raise ValueError(
            f"formal difficulty artifacts differ from manifest problem set: {run_dir}"
        )
    return {
        "run_id": config.run_id,
        "protocol_id": config.protocol_id,
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "contract_sha256": contract.contract_sha256,
        "frozen_campaign_sha256": frozen_campaign_sha256,
        "source_tree_sha256": contract.source_tree_sha256,
        "generation_manifest_sha256": _generation_manifest_sha256(manifest),
        "problem_count": len(artifact_ids),
        "problem_ids_sha256": payload_sha256(sorted(artifact_ids)),
        "problem_merkle_root": _merkle_root(artifact_hashes),
    }


def _full_problem_universe(
    config: RunConfig,
    project_root: Path,
) -> tuple[int, str]:
    execution = config.execution.model_copy(
        update={"shard_index": 0, "num_shards": 1, "limit": None}
    )
    full = config.model_copy(update={"execution": execution})
    problem_ids = sorted(
        problem.id for problem in load_public_problems(full, project_root)
    )
    return len(problem_ids), payload_sha256(problem_ids)


def build_formal_difficulty_lineage(
    config: RunConfig,
    project_root: Path,
    run_dir: Path,
    peer_run_dirs: list[Path],
) -> dict[str, Any]:
    if config.stage != "formal":
        raise ValueError("formal difficulty lineage requires stage=formal")
    design = formal_difficulty_design(config, project_root)
    required_peers = int(design["required_peer_runs"])
    if len(peer_run_dirs) != required_peers:
        qualifier = (
            "one complementary peer run"
            if required_peers == 1
            else f"{required_peers} frozen difficulty-reference runs"
        )
        raise ValueError(
            f"formal evaluation requires exactly {qualifier}"
        )
    current_dir = run_dir.resolve()
    peer_dirs = [path.resolve() for path in peer_run_dirs]
    if current_dir in peer_dirs:
        raise ValueError("formal difficulty peer cannot be the current run")
    if len(peer_dirs) != len(set(peer_dirs)):
        raise ValueError("formal difficulty lineage contains a duplicated run")
    if any(path.parent != current_dir.parent for path in peer_dirs):
        raise ValueError(
            "formal difficulty peer must be a sibling in the same result root"
        )
    current_row = generation_lineage(current_dir, project_root)
    peer_rows = [
        generation_lineage(path, project_root)
        for path in peer_dirs
    ]
    rows = [current_row, *peer_rows]
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("formal difficulty lineage contains a duplicated run")
    expected_count, expected_ids_sha256 = _full_problem_universe(
        config,
        project_root,
    )
    for row in rows:
        if (
            row["problem_count"] != expected_count
            or row["problem_ids_sha256"] != expected_ids_sha256
        ):
            raise ValueError(
                "formal difficulty run does not cover the complete frozen "
                f"problem universe: {row['run_id']}"
            )

    expected_reference_models = set(design["reference_models"])
    if bool(design["include_current_outcomes"]):
        observed_models = {row["model_id"] for row in rows}
        if observed_models != expected_reference_models:
            raise ValueError(
                "formal difficulty requires exactly the two complementary frozen "
                f"models: expected={sorted(expected_reference_models)} "
                f"actual={sorted(observed_models)}"
            )
        if len({row["source_tree_sha256"] for row in rows}) != 1:
            raise ValueError(
                "formal difficulty peers were generated by different source trees"
            )
        lineage = {
            "schema_version": "flcva-difficulty-lineage-v1",
            "protocol_id": config.protocol_id,
            "frozen_campaign_sha256": FROZEN_CAMPAIGN_SHA256,
            "problem_count": expected_count,
            "problem_ids_sha256": expected_ids_sha256,
            "models": sorted(expected_reference_models),
            "runs": sorted(rows, key=lambda row: row["model_id"]),
            "lineage_sha256": "",
        }
        lineage["lineage_sha256"] = payload_sha256(lineage)
        return lineage

    campaign, campaign_sha256 = load_formal_campaign(
        project_root,
        config.protocol_id,
    )
    expected_current_models = {
        str(row["id"])
        for row in campaign["models"].values()
    }
    if (
        {current_row["model_id"]} != expected_current_models
        or current_row["protocol_id"] != config.protocol_id
        or current_row["frozen_campaign_sha256"] != campaign_sha256
    ):
        raise ValueError(
            "external replication current run differs from its frozen campaign"
        )
    observed_reference_models = {row["model_id"] for row in peer_rows}
    if observed_reference_models != expected_reference_models:
        raise ValueError(
            "external replication difficulty references the wrong model set: "
            f"expected={sorted(expected_reference_models)} "
            f"actual={sorted(observed_reference_models)}"
        )
    for row in peer_rows:
        if (
            row["protocol_id"] != design["reference_protocol_id"]
            or row["frozen_campaign_sha256"]
            != design["reference_campaign_sha256"]
        ):
            raise ValueError(
                "external replication difficulty reference differs from the "
                f"sealed primary campaign: {row['run_id']}"
            )
    if len({row["source_tree_sha256"] for row in peer_rows}) != 1:
        raise ValueError(
            "formal difficulty reference runs were generated by different "
            "source trees"
        )
    lineage = {
        "schema_version": "flcva-difficulty-lineage-v2",
        "protocol_id": config.protocol_id,
        "frozen_campaign_sha256": campaign_sha256,
        "difficulty_design": design,
        "current_outcomes_excluded": True,
        "problem_count": expected_count,
        "problem_ids_sha256": expected_ids_sha256,
        "models": sorted(expected_reference_models),
        "difficulty_run_ids": sorted(row["run_id"] for row in peer_rows),
        "runs": sorted(rows, key=lambda row: row["model_id"]),
        "lineage_sha256": "",
    }
    lineage["lineage_sha256"] = payload_sha256(lineage)
    return lineage


def verify_formal_difficulty_lineage(
    config: RunConfig,
    project_root: Path,
    run_dir: Path,
    recorded: dict[str, Any],
) -> bool:
    unhashed = dict(recorded)
    recorded_sha256 = unhashed.get("lineage_sha256")
    unhashed["lineage_sha256"] = ""
    if recorded_sha256 != payload_sha256(unhashed):
        raise ValueError("formal difficulty lineage self-hash mismatch")
    rows = recorded.get("runs", [])
    own = [row for row in rows if row.get("run_id") == config.run_id]
    peers = [row for row in rows if row.get("run_id") != config.run_id]
    design = formal_difficulty_design(config, project_root)
    required_peers = int(design["required_peer_runs"])
    if len(own) != 1 or len(peers) != required_peers:
        expected = (
            "one current and one peer run"
            if required_peers == 1
            else f"one current and {required_peers} reference runs"
        )
        raise ValueError(
            f"formal difficulty lineage does not contain {expected}"
        )
    peer_run_ids = [str(row.get("run_id", "")) for row in peers]
    if any(
        not run_id or Path(run_id).name != run_id
        for run_id in peer_run_ids
    ):
        raise ValueError("formal difficulty peer run_id is unsafe")
    observed = build_formal_difficulty_lineage(
        config,
        project_root,
        run_dir,
        [
            run_dir.resolve().parent / run_id
            for run_id in peer_run_ids
        ],
    )
    if observed != recorded:
        raise ValueError(
            "recorded formal difficulty lineage differs from current peer artifacts"
        )
    return True
