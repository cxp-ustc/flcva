import json
import shutil
from pathlib import Path

import pytest

from flcva.audit import audit_run
from flcva.io import (
    atomic_write_json,
    iter_jsonl_gz,
    iter_problem_artifacts,
    seal_record,
    verify_sealed_record,
)
from flcva.runner import generate_run

ROOT = Path(__file__).resolve().parents[1]


def test_mock_pipeline_is_complete_and_sealed(mock_pipeline) -> None:
    assert mock_pipeline["generation"]["problems"] == 2
    assert mock_pipeline["scoring"]["completed_now"] == 2
    assert mock_pipeline["evaluation"]["values"] == 24
    assert mock_pipeline["audit"]["passed"]
    assert mock_pipeline["audit"]["prefix_replay_rate"] == 1.0
    assert mock_pipeline["audit"]["branch_completeness_rate"] == 1.0
    assert verify_sealed_record(mock_pipeline["audit"])
    run = json.loads((mock_pipeline["run_dir"] / "run.json").read_text())
    assert run["status"] == "sealed"


def test_sealed_generation_resume_is_read_only(mock_pipeline) -> None:
    summary = generate_run(mock_pipeline["config"], ROOT)
    assert summary["completed_now"] == 0
    assert summary["resumed"] == 2
    run = json.loads((mock_pipeline["run_dir"] / "run.json").read_text())
    assert run["status"] == "sealed"


def test_generation_artifacts_are_label_blind(mock_pipeline) -> None:
    forbidden = {"gold", "gold_answer", "label", "target", "reference_answer"}
    for artifact in iter_problem_artifacts(mock_pipeline["run_dir"]):
        assert forbidden.isdisjoint(artifact)


def test_generation_uses_audited_request_batches(mock_pipeline) -> None:
    observed_batch_sizes = set()
    observed_utilization = set()
    for artifact in iter_problem_artifacts(mock_pipeline["run_dir"]):
        for event in artifact["ledger"]:
            if event["role"] in {
                "stop_probe",
                "continuation_reasoning",
                "continuation_probe",
            }:
                observed_batch_sizes.add(event["batch_size"])
                if event["utilization"] is not None:
                    observed_utilization.add(event["utilization"])
    assert max(observed_batch_sizes) > 1
    # This is explicitly request-batch fill against a caller-declared
    # capacity, not GPU SM utilization.
    assert observed_utilization == {1.0}


def test_targets_respect_reachable_signs(mock_pipeline) -> None:
    values = list(
        iter_jsonl_gz(mock_pipeline["run_dir"] / "evaluation" / "values.jsonl.gz")
    )
    for row in values:
        if row["stop_correct"] == 0:
            assert row["marginal_value"] >= 0
            assert row["harmful"] == 0
        else:
            assert row["marginal_value"] <= 0
            assert row["beneficial"] == 0


def test_replay_uses_a_common_endpoint(mock_pipeline) -> None:
    summary = mock_pipeline["replay"]
    assert summary["common_endpoint"] > 0
    assert set(summary["aulc"]) == {
        "fixed_1024",
        "round_robin",
        "trace",
        "agreement",
        "prm",
        "anll",
        "entropy",
        "oracle",
    }
    assert summary["permutation_oracle"]["replicates"] == 20


def test_replay_initial_budget_uses_complete_deduplicated_ledger(
    mock_pipeline,
) -> None:
    frontiers = list(
        iter_jsonl_gz(mock_pipeline["run_dir"] / "evaluation" / "frontiers.jsonl.gz")
    )
    if mock_pipeline["replay"]["split_scope"] == "confirmation":
        frontiers = [row for row in frontiers if row["split"] == "confirmation"]
    initial = [
        row
        for row in frontiers
        if row["depth"] == mock_pipeline["config"].protocol.checkpoint_interval
    ]
    events = {
        event["event_id"]: event
        for row in initial
        for event in row["initial_ledger_events"]
    }
    expected = sum(int(event["logical_tokens"]) for event in events.values())
    assert mock_pipeline["replay"]["initial_budget"]["round_robin"] == expected
    assert mock_pipeline["replay"]["aulc_start_budget"] == expected
    assert (
        expected > len(initial) * mock_pipeline["config"].protocol.checkpoint_interval
    )
    for row in frontiers:
        if row["terminal"]:
            continue
        assert row["transition_cost"] == sum(
            int(event["logical_tokens"]) for event in row["transition_ledger_events"]
        )


def test_replay_charges_prm_before_it_can_schedule(mock_pipeline) -> None:
    points = [
        row
        for row in iter_jsonl_gz(
            mock_pipeline["run_dir"] / "replay" / "points.jsonl.gz"
        )
        if row["scheduler"] == "prm"
    ]
    assert len(points) >= 2
    assert points[0]["budget"] == mock_pipeline["replay"]["aulc_start_budget"]
    assert points[1]["budget"] == mock_pipeline["replay"]["initial_budget"]["prm"]
    assert points[1]["accuracy"] == points[0]["accuracy"]


def test_permutation_p_value_uses_finite_sample_correction(mock_pipeline) -> None:
    permutation = mock_pipeline["replay"]["permutation_oracle"]
    assert permutation["p_value_corrected"] == pytest.approx(
        (permutation["exceedances"] + 1) / (permutation["replicates"] + 1)
    )


def test_collision_rates_are_probabilities(mock_pipeline) -> None:
    collision = mock_pipeline["evaluation"]["collision"]
    if collision["collision_rate"] is not None:
        assert 0.0 <= collision["collision_rate"] <= 1.0
    for row in collision["by_proxy"].values():
        if row["collision_rate"] is not None:
            assert 0.0 <= row["collision_rate"] <= 1.0


def test_audit_rejects_missing_derived_artifact(
    mock_pipeline,
    tmp_path,
    monkeypatch,
) -> None:
    config = mock_pipeline["config"]
    monkeypatch.setenv("FLCVA_RESULT_ROOT", str(tmp_path))
    copied = tmp_path / config.run_id
    shutil.copytree(mock_pipeline["run_dir"], copied)
    points = copied / "replay" / "points.jsonl.gz"
    points.unlink()
    with pytest.raises(FileNotFoundError, match="missing artifact"):
        audit_run(config, ROOT)


def test_audit_rejects_manifest_config_drift(
    mock_pipeline,
    tmp_path,
    monkeypatch,
) -> None:
    config = mock_pipeline["config"]
    monkeypatch.setenv("FLCVA_RESULT_ROOT", str(tmp_path))
    copied = tmp_path / config.run_id
    shutil.copytree(mock_pipeline["run_dir"], copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["global_seed"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest self-hash mismatch"):
        audit_run(config, ROOT)


def test_audit_rejects_gold_label_attestation_drift(
    mock_pipeline,
    tmp_path,
    monkeypatch,
) -> None:
    config = mock_pipeline["config"]
    monkeypatch.setenv("FLCVA_RESULT_ROOT", str(tmp_path))
    copied = tmp_path / config.run_id
    shutil.copytree(mock_pipeline["run_dir"], copied)
    summary_path = copied / "evaluation" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["gold_labels"]["problem_count"] += 1
    atomic_write_json(summary_path, seal_record(summary))
    with pytest.raises(ValueError, match="gold-label attestation differs"):
        audit_run(config, ROOT)
