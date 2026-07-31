#!/usr/bin/env python3
"""Evaluate primary-trained observable controllers on Llama without refitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from flcva.config import RunConfig
from flcva.frozen import (
    LLAMA_REPLICATION_CAMPAIGN_SHA256,
    LLAMA_REPLICATION_PROTOCOL_ID,
    validate_formal_manifest_config,
)
from flcva.ids import payload_sha256
from flcva.io import (
    file_sha256,
    seal_record,
    verify_sealed_record,
)
from paper.analysis import render_supplement as supplement


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "generated" / "llama_replication"
MODEL = "DeepSeek-R1-Distill-Llama-8B"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"
TRANSFER_ID = "flcva-primary-to-llama-controller-transfer-v2"
POLICIES = (
    "round_robin",
    "correctness_stratified_ev",
    "two_stage_ev",
    "direct_ridge_m",
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected an object")
    return payload


def _validate_llama(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = _read(run_dir / "manifest.json")
    audit = _read(run_dir / "audit_manifest.json")
    evaluation = _read(run_dir / "evaluation" / "summary.json")
    if not all(
        verify_sealed_record(row)
        for row in (manifest, audit, evaluation)
    ):
        raise ValueError("Llama replication seal verification failed")
    config = RunConfig.model_validate(manifest["config"])
    campaign_sha256 = validate_formal_manifest_config(
        manifest["config"],
        ROOT,
    )
    if (
        config.protocol_id != LLAMA_REPLICATION_PROTOCOL_ID
        or config.model.id != MODEL_ID
        or config.model.revision != MODEL_REVISION
        or campaign_sha256 != LLAMA_REPLICATION_CAMPAIGN_SHA256
        or audit.get("passed") is not True
        or not all(audit.get("gates", {}).values())
        or evaluation.get("difficulty_current_outcomes_excluded") is not True
    ):
        raise ValueError("run is not an audited external Llama replication")
    for relative, expected in evaluation["artifacts_sha256"].items():
        path = run_dir / "evaluation" / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Llama evaluation artifact mismatch: {path}")
    return {
        "run_dir": run_dir,
        "run": {"run_id": config.run_id},
        "audit": audit,
        "evaluation": evaluation,
        "config": config,
    }


def _score_maps(
    primary_analysis: list[dict[str, Any]],
    llama_confirmation: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    raw_features = [
        "persistence",
        "answer_confidence",
        "anll",
        "top20_entropy",
    ]
    model_key = "primary-pooled-to-llama"
    q_map, q_fit = supplement._fit_logistic_component(
        primary_analysis,
        llama_confirmation,
        model=model_key,
        component="stop_wrong",
        raw_features=raw_features,
    )
    recovery_map, recovery_fit = supplement._fit_logistic_component(
        primary_analysis,
        llama_confirmation,
        model=model_key,
        component="recovery_given_wrong",
        raw_features=raw_features,
    )
    damage_map, damage_fit = supplement._fit_logistic_component(
        primary_analysis,
        llama_confirmation,
        model=model_key,
        component="damage_given_correct",
        raw_features=raw_features,
    )
    ridge_map, ridge_fit = supplement._fit_ridge_controller(
        primary_analysis,
        llama_confirmation,
        model=model_key,
        raw_features=raw_features,
    )
    moments, serialized_moments = (
        supplement._stratified_continuation_moments(primary_analysis)
    )
    maps: dict[str, dict[str, float]] = {
        "correctness_stratified_ev": {},
        "two_stage_ev": {},
        "direct_ridge_m": ridge_map,
    }
    for row in llama_confirmation:
        checkpoint_id = str(row["checkpoint_id"])
        q = q_map[checkpoint_id]
        stratum = moments[
            (str(row["dataset_id"]), int(row["depth"]))
        ]
        cost = float(stratum["mean_branch_cost"])
        maps["correctness_stratified_ev"][checkpoint_id] = (
            1000.0
            * (
                q * float(stratum["recovery_rate"])
                - (1.0 - q) * float(stratum["damage_rate"])
            )
            / cost
        )
        maps["two_stage_ev"][checkpoint_id] = (
            1000.0
            * (
                q * recovery_map[checkpoint_id]
                - (1.0 - q) * damage_map[checkpoint_id]
            )
            / cost
        )
    return maps, {
        "raw_features": raw_features,
        "llama_rows_used_for_fit": 0,
        "stop_wrong": q_fit,
        "recovery_given_wrong": recovery_fit,
        "damage_given_correct": damage_fit,
        "direct_ridge_m": ridge_fit,
        "primary_stratified_moments": serialized_moments,
    }


def _replay(
    payload: dict[str, Any],
    llama_rows: list[dict[str, Any]],
    score_maps: dict[str, dict[str, float]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[supplement.CompactReplayContext],
]:
    states, _ = supplement.load_confirmation_frontiers(
        payload["run_dir"],
        payload["evaluation"],
        llama_rows,
    )
    config = payload["config"]
    seed_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    compact_contexts: list[supplement.CompactReplayContext] = []
    for spine_id in range(4):
        selected = [
            state
            for state in states
            if int(state["spine_id"]) == spine_id
        ]
        compact_contexts.append(
            supplement._build_compact_replay_context(
                selected,
                {
                    policy: score_maps[policy]
                    for policy in POLICIES[1:]
                },
            )
        )
        traces = {
            "round_robin": supplement._replay_one_with_problem_events(
                selected,
                "round_robin",
                checkpoint_interval=config.protocol.checkpoint_interval,
                fixed_allotment_tokens=(
                    config.execution.fixed_allotment_tokens
                ),
            )
        }
        for policy in POLICIES[1:]:
            traces[policy] = supplement._replay_one_with_problem_events(
                selected,
                "oracle",
                checkpoint_interval=config.protocol.checkpoint_interval,
                fixed_allotment_tokens=(
                    config.execution.fixed_allotment_tokens
                ),
                overridden_values=score_maps[policy],
            )
        start_budget = traces["round_robin"][0][0].budget
        endpoint = min(trace[0][-1].budget for trace in traces.values())
        span = endpoint - start_budget
        if span <= 0:
            raise ValueError("cross-model transfer replay is empty")
        for policy, trace in traces.items():
            curve, initial_correct, changes, _ = trace
            normalized_aulc = (
                supplement.step_auc(
                    curve,
                    endpoint,
                    start_budget=start_budget,
                )
                / span
            )
            seed_rows.append(
                {
                    "model": MODEL,
                    "spine_id": spine_id,
                    "policy": policy,
                    "normalized_aulc": normalized_aulc,
                    "final_accuracy": supplement._accuracy_at_budget(
                        curve,
                        endpoint,
                    ),
                }
            )
            contributions = supplement._problem_contributions(
                initial_correct,
                changes,
                start_budget=start_budget,
                endpoint=endpoint,
            )
            for problem_id, contribution in contributions.items():
                problem_rows.append(
                    {
                        "model": MODEL,
                        "spine_id": spine_id,
                        "policy": policy,
                        "problem_id": problem_id,
                        **contribution,
                    }
                )
    return seed_rows, problem_rows, compact_contexts


def _summarize(
    problem_rows: list[dict[str, Any]],
    compact_contexts: list[supplement.CompactReplayContext],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in problem_rows:
        grouped[
            (str(row["policy"]), str(row["problem_id"]))
        ].append(float(row["normalized_aulc"]))
    problem_ids = sorted(
        {
            problem_id
            for _, problem_id in grouped
        }
    )
    if len(problem_ids) != 372:
        raise ValueError(
            f"expected 372 confirmation problems, observed {len(problem_ids)}"
        )
    vectors: dict[str, np.ndarray] = {}
    for policy in POLICIES:
        if any(
            len(grouped[(policy, problem_id)]) != 4
            for problem_id in problem_ids
        ):
            raise ValueError("transfer replay lacks four spine repetitions")
        vectors[policy] = np.asarray(
            [
                float(np.mean(grouped[(policy, problem_id)]))
                for problem_id in problem_ids
            ],
            dtype=float,
        )
    print(
        "full-system replay bootstrap: Llama "
        f"({supplement.BOOTSTRAP_REPLICATES} replicates)",
        flush=True,
    )
    observed, bootstrap_samples, bootstrap_rows = (
        supplement._paired_full_replay_bootstrap(
            compact_contexts,
            policies=POLICIES,
            replicates=supplement.BOOTSTRAP_REPLICATES,
            seed_parts=(TRANSFER_ID, "controller-hierarchy"),
        )
    )
    print("full-system replay bootstrap complete: Llama", flush=True)
    for policy in POLICIES:
        expected = float(vectors[policy].mean())
        if not np.isclose(
            observed[policy],
            expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{policy}: compact full replay does not reproduce "
                "the authoritative Llama point estimate"
            )

    round_robin_samples = bootstrap_samples["round_robin"]
    shortcut_samples = bootstrap_samples["correctness_stratified_ev"]
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        policy_samples = bootstrap_samples[policy]
        versus_rr = (
            100.0
            * (policy_samples - round_robin_samples)
            / round_robin_samples
        )
        versus_shortcut = (
            100.0
            * (policy_samples - shortcut_samples)
            / shortcut_samples
        )
        point_vs_rr = (
            100.0
            * (observed[policy] - observed["round_robin"])
            / observed["round_robin"]
        )
        point_vs_shortcut = (
            100.0
            * (
                observed[policy]
                - observed["correctness_stratified_ev"]
            )
            / observed["correctness_stratified_ev"]
        )
        rows.append(
            {
                "model": MODEL,
                "policy": policy,
                "analysis_status": (
                    "paired full-system problem bootstrap"
                ),
                "training_models": [
                    "Qwen3-8B",
                    "DeepSeek-R1-Distill-Qwen-7B",
                ],
                "llama_fit_rows": 0,
                "problems": len(problem_ids),
                "spine_repetitions": 4,
                "mean_normalized_aulc": observed[policy],
                "gain_vs_round_robin_percent": point_vs_rr,
                "gain_vs_round_robin_low": float(
                    np.quantile(versus_rr, 0.025)
                ),
                "gain_vs_round_robin_high": float(
                    np.quantile(versus_rr, 0.975)
                ),
                "increment_vs_shortcut_percent": point_vs_shortcut,
                "increment_vs_shortcut_low": float(
                    np.quantile(versus_shortcut, 0.025)
                ),
                "increment_vs_shortcut_high": float(
                    np.quantile(versus_shortcut, 0.975)
                ),
            }
        )
    metadata = {
        "namespace": supplement.FULL_REPLAY_BOOTSTRAP_NAMESPACE,
        "replicates": supplement.BOOTSTRAP_REPLICATES,
        "problems_per_replicate": len(problem_ids),
        "spine_repetitions": len(compact_contexts),
        "duplicate_draw_semantics": (
            "independent replay identities with shared sampled source data"
        ),
        "unique_source_problems": {
            "min": min(
                int(row["unique_source_problems"])
                for row in bootstrap_rows
            ),
            "median": float(
                np.median(
                    [
                        int(row["unique_source_problems"])
                        for row in bootstrap_rows
                    ]
                )
            ),
            "max": max(
                int(row["unique_source_problems"])
                for row in bootstrap_rows
            ),
        },
    }
    return rows, bootstrap_rows, metadata


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(llama_run: Path) -> dict[str, Any]:
    primary = supplement.validate_inputs()
    primary_values = {
        model: supplement.load_values(
            payload["run_dir"],
            payload["evaluation"],
        )
        for model, payload in primary.items()
    }
    primary_analysis = [
        row
        for rows in primary_values.values()
        for row in rows
        if row["split"] == "analysis"
    ]
    llama = _validate_llama(llama_run)
    llama_values = supplement.load_values(
        llama["run_dir"],
        llama["evaluation"],
    )
    llama_confirmation = [
        row for row in llama_values if row["split"] == "confirmation"
    ]
    score_maps, fits = _score_maps(
        primary_analysis,
        llama_confirmation,
    )
    seed_rows, problem_rows, compact_contexts = _replay(
        llama,
        llama_values,
        score_maps,
    )
    summary_rows, bootstrap_rows, bootstrap_metadata = _summarize(
        problem_rows,
        compact_contexts,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    seed_path = OUT / "controller_transfer_seeds.csv"
    problem_path = OUT / "controller_transfer_problems.csv"
    summary_path = OUT / "controller_transfer_summary.csv"
    bootstrap_path = OUT / "controller_transfer_full_replay_bootstrap.csv"
    fits_path = OUT / "controller_transfer_fits.json"
    table_path = OUT / "controller_transfer_table.tex"
    manifest_path = OUT / "controller_transfer_manifest.json"
    _write_csv(seed_path, seed_rows)
    _write_csv(problem_path, problem_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(bootstrap_path, bootstrap_rows)
    supplement.atomic_text(
        fits_path,
        json.dumps(fits, indent=2, sort_keys=True) + "\n",
    )
    labels = {
        "round_robin": "Round-robin",
        "correctness_stratified_ev": "Correctness shortcut",
        "two_stage_ev": "Two-stage EV",
        "direct_ridge_m": r"Direct Ridge-$M$",
    }
    table_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        (
            r"Policy & Gain vs.\ RR (95\% CI) & "
            r"Increment vs.\ shortcut (95\% CI) \\"
        ),
        r"\midrule",
    ]
    for row in summary_rows:
        gain = (
            f"{row['gain_vs_round_robin_percent']:+.2f}\\% "
            f"[{row['gain_vs_round_robin_low']:+.2f}, "
            f"{row['gain_vs_round_robin_high']:+.2f}]"
        )
        increment = (
            f"{row['increment_vs_shortcut_percent']:+.2f}\\% "
            f"[{row['increment_vs_shortcut_low']:+.2f}, "
            f"{row['increment_vs_shortcut_high']:+.2f}]"
        )
        if row["policy"] == "correctness_stratified_ev":
            table_lines.append(r"\rowcolor{resultgreenbg}")
        if row["policy"] in {"two_stage_ev", "direct_ridge_m"}:
            increment = rf"\keyresult{{{increment}}}"
        table_lines.append(
            "{} & {} & {} \\\\".format(
                labels[str(row["policy"])],
                gain,
                increment,
            )
        )
    table_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Zero-refit transfer from the two primary models "
                r"to DeepSeek-R1-Distill-Llama-8B. No Llama outcome is used "
                r"for fitting or hyperparameter selection. Intervals are "
                r"paired 2,000-replicate full-system problem bootstraps: "
                r"every replicate reconstructs the sampled problem pool, "
                r"treats duplicate draws as independent copies, reruns global "
                r"policy ordering over all four spine repetitions, and "
                r"recomputes the ledger endpoint and normalized AULC. Pale "
                r"green marks the winning shortcut row and the "
                r"shortcut-relative increments that support the transfer "
                r"interpretation; it is an expository aid only.}"
            ),
            r"\label{tab:llama-controller-transfer}",
            r"\end{table}",
            "",
        ]
    )
    supplement.atomic_text(table_path, "\n".join(table_lines))
    manifest = seal_record(
        {
            "schema_version": "flcva-controller-transfer-manifest-v2",
            "transfer_id": TRANSFER_ID,
            "analysis_status": (
                "post-seal external architecture replication"
            ),
            "training_models": sorted(primary),
            "target_model": MODEL,
            "target_run_id": llama["run"]["run_id"],
            "llama_fit_rows": 0,
            "confidence_interval_status": (
                "paired full-system problem bootstrap intervals; every "
                "replicate reconstructs the sampled pool, reruns policy "
                "ordering, and recomputes the endpoint and AULC"
            ),
            "full_replay_bootstrap": bootstrap_metadata,
            "features": [
                "persistence",
                "answer_confidence",
                "anll",
                "top20_entropy",
                "depth",
                "dataset",
            ],
            "excluded_target_information": [
                "Llama analysis outcomes",
                "Llama confirmation correctness",
                "Llama continuation outcomes",
                "difficulty",
                "cross-spine agreement",
            ],
            "primary_source_runs": {
                model: {
                    "run_id": payload["run"]["run_id"],
                    "result_merkle_root": payload["audit"][
                        "result_merkle_root"
                    ],
                }
                for model, payload in primary.items()
            },
            "target_result_merkle_root": llama["audit"][
                "result_merkle_root"
            ],
            "artifacts_sha256": {
                path.name: file_sha256(path)
                for path in (
                    seed_path,
                    problem_path,
                    summary_path,
                    bootstrap_path,
                    fits_path,
                    table_path,
                )
            },
            "source_sha256": file_sha256(Path(__file__)),
            "summary_rows_sha256": payload_sha256(summary_rows),
        }
    )
    supplement.atomic_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return {
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "record_sha256": manifest["record_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-run", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(render(args.llama_run), sort_keys=True))


if __name__ == "__main__":
    main()
