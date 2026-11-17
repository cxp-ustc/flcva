from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import RunConfig

# Stable campaign IDs retained for compatibility with recorded runs.
FROZEN_CAMPAIGN_SHA256 = (
    "992b4d8b277f20d14d70b0a5ce1de1ed3693f9691206d89302e68bd32685e958"
)
LLAMA_REPLICATION_CAMPAIGN_SHA256 = (
    "6b6b20fb7f93eff41643fd08041ccf2ab39e8fefbafa001254fabd5554a805dd"
)
PRIMARY_PROTOCOL_ID = "flcva-2025-cutoff-v3"
LLAMA_REPLICATION_PROTOCOL_ID = "flcva-llama-architecture-replication-v1"
FORMAL_CAMPAIGNS = {
    PRIMARY_PROTOCOL_ID: (
        "configs/frozen.yaml",
        FROZEN_CAMPAIGN_SHA256,
    ),
    LLAMA_REPLICATION_PROTOCOL_ID: (
        "configs/replication-llama.yaml",
        LLAMA_REPLICATION_CAMPAIGN_SHA256,
    ),
}
PROTOCOL_FIELDS = (
    "spines_per_problem",
    "reasoning_max_tokens",
    "checkpoint_interval",
    "continuation_branches",
    "continuation_tokens",
    "answer_max_tokens",
    "answer_cue",
)
EXECUTION_FIELDS = (
    "bootstrap_replicates",
    "permutation_replicates",
    "fixed_allotment_tokens",
)


def load_formal_campaign(
    project_root: Path,
    protocol_id: str,
) -> tuple[dict[str, Any], str]:
    spec = FORMAL_CAMPAIGNS.get(protocol_id)
    if spec is None:
        raise ValueError(f"unknown formal protocol_id: {protocol_id}")
    relative_path, campaign_id = spec
    path = project_root / relative_path
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{relative_path} must be a mapping")
    if payload.get("protocol_id") != protocol_id:
        raise ValueError(f"{relative_path} protocol_id differs from its registry key")
    return payload, campaign_id


def load_frozen_campaign(project_root: Path) -> dict[str, Any]:
    """Load the immutable two-model primary campaign."""

    campaign, _ = load_formal_campaign(project_root, PRIMARY_PROTOCOL_ID)
    return campaign


def formal_difficulty_design(
    config: RunConfig,
    project_root: Path,
) -> dict[str, Any]:
    """Return the authenticated model set used only for difficulty strata."""

    campaign, campaign_sha256 = load_formal_campaign(
        project_root,
        config.protocol_id,
    )
    reference = campaign.get("difficulty_reference")
    if reference is None:
        models = sorted(str(row["id"]) for row in campaign["models"].values())
        return {
            "schema_version": "flcva-difficulty-design-v1",
            "campaign_sha256": campaign_sha256,
            "reference_campaign_sha256": campaign_sha256,
            "reference_protocol_id": config.protocol_id,
            "reference_models": models,
            "include_current_outcomes": True,
            "required_peer_runs": len(models) - 1,
        }

    models = sorted(str(value) for value in reference.get("models", []))
    if not models or len(models) != len(set(models)):
        raise ValueError(
            "formal difficulty reference models must be non-empty and unique"
        )
    include_current = bool(reference.get("include_current_outcomes"))
    if include_current:
        raise ValueError(
            "external replication difficulty must exclude current outcomes"
        )
    if config.model.id in models:
        raise ValueError(
            "external replication model cannot be a difficulty reference model"
        )
    if reference.get("campaign_sha256") != FROZEN_CAMPAIGN_SHA256:
        raise ValueError(
            "external replication must bind the sealed primary campaign digest"
        )
    if reference.get("protocol_id") != PRIMARY_PROTOCOL_ID:
        raise ValueError("external replication must bind the sealed primary protocol")
    primary, primary_sha256 = load_formal_campaign(
        project_root,
        PRIMARY_PROTOCOL_ID,
    )
    primary_models = sorted(str(row["id"]) for row in primary["models"].values())
    if (
        primary_sha256 != reference["campaign_sha256"]
        or models != primary_models
        or campaign.get("parent_campaign_sha256") != primary_sha256
        or campaign.get("parent_protocol_id") != PRIMARY_PROTOCOL_ID
    ):
        raise ValueError(
            "external replication difficulty design differs from the "
            "canonical primary campaign"
        )
    return {
        "schema_version": "flcva-difficulty-design-v1",
        "campaign_sha256": campaign_sha256,
        "reference_campaign_sha256": str(reference["campaign_sha256"]),
        "reference_protocol_id": str(reference["protocol_id"]),
        "reference_models": models,
        "include_current_outcomes": False,
        "required_peer_runs": len(models),
    }


def _mismatches(
    expected: Any,
    observed: Any,
    path: str = "",
) -> dict[str, dict[str, Any]]:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return {path or "<root>": {"expected": expected, "actual": observed}}
        output: dict[str, dict[str, Any]] = {}
        for key, value in expected.items():
            child = f"{path}.{key}" if path else str(key)
            if key not in observed:
                output[child] = {"expected": value, "actual": "<missing>"}
            else:
                output.update(_mismatches(value, observed[key], child))
        return output
    if expected != observed:
        return {path or "<root>": {"expected": expected, "actual": observed}}
    return {}


def _expected_formal_projection(
    campaign: dict[str, Any], model_id: str
) -> dict[str, Any]:
    model_rows = list(campaign["models"].values())
    matches = [row for row in model_rows if row["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"model is not in the frozen campaign: {model_id}")
    model = {**matches[0], **campaign["model_runtime"]}
    combined = campaign["datasets"]["combined"]
    dataset = {
        "id": combined["id"],
        "source": combined["source"],
        "repo_id": None,
        "revision": combined["revision"],
        "subset": None,
        "split": combined["split"],
        "public_path": combined["public_path"],
        "labels_path": combined["labels_path"],
    }
    return {
        "protocol_id": campaign["protocol_id"],
        "stage": "formal",
        "backend": "vllm",
        "global_seed": campaign["global_seed"],
        "model": model,
        "dataset": dataset,
        "sampling": campaign["sampling"],
        "protocol": {key: campaign["protocol"][key] for key in PROTOCOL_FIELDS},
        "scorer": campaign["scorer"],
        "execution": {key: campaign["analysis"][key] for key in EXECUTION_FIELDS},
    }


def _observed_formal_projection(config: RunConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    return {
        "protocol_id": payload["protocol_id"],
        "stage": payload["stage"],
        "backend": payload["backend"],
        "global_seed": payload["global_seed"],
        "model": payload["model"],
        "dataset": payload["dataset"],
        "sampling": payload["sampling"],
        "protocol": {key: payload["protocol"][key] for key in PROTOCOL_FIELDS},
        "scorer": payload["scorer"],
        "execution": {key: payload["execution"][key] for key in EXECUTION_FIELDS},
    }


def _validate_public_data(campaign: dict[str, Any], project_root: Path) -> None:
    combined = campaign["datasets"]["combined"]
    public_path = project_root / combined["public_path"]
    if not public_path.is_file():
        raise ValueError(f"frozen public data is missing: {public_path}")
    manifest_path = project_root / "data" / "frozen" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        dataset_id: {
            "repository": campaign["datasets"][dataset_id]["id"],
            "revision": campaign["datasets"][dataset_id]["revision"],
            "split": campaign["datasets"][dataset_id]["split"],
            "problems": campaign["datasets"][dataset_id]["problems"],
        }
        for dataset_id in ("gsm8k", "math500")
    }
    observed_manifest = {
        dataset_id: {
            key: manifest["datasets"][dataset_id].get(key)
            for key in expected_manifest[dataset_id]
        }
        for dataset_id in expected_manifest
    }
    manifest_mismatches = _mismatches(expected_manifest, observed_manifest)
    combined_mismatches = _mismatches(
        {
            "problems": combined["problems"],
        },
        {
            "problems": manifest.get("combined", {}).get("problems"),
        },
        "combined",
    )
    if manifest_mismatches or combined_mismatches:
        raise ValueError(
            "frozen data manifest differs from the canonical campaign: "
            f"{manifest_mismatches | combined_mismatches}"
        )


def validate_formal_config(config: RunConfig, project_root: Path) -> str:
    if config.stage != "formal":
        raise ValueError("frozen campaign validation requires stage=formal")
    campaign, campaign_sha256 = load_formal_campaign(
        project_root,
        config.protocol_id,
    )
    expected = _expected_formal_projection(campaign, config.model.id)
    observed = _observed_formal_projection(config)
    mismatches = _mismatches(expected, observed)
    if mismatches:
        raise ValueError(f"formal config differs from frozen campaign: {mismatches}")
    _validate_public_data(campaign, project_root)
    formal_difficulty_design(config, project_root)
    return campaign_sha256


def validate_formal_manifest_config(
    payload: dict[str, Any],
    project_root: Path,
) -> str:
    return validate_formal_config(RunConfig.model_validate(payload), project_root)
