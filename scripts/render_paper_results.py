#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from pathlib import Path
from typing import Any

from flcva.ids import payload_sha256
from flcva.config import RunConfig
from flcva.contract import verify_run_record
from flcva.frozen import validate_formal_manifest_config
from flcva.io import (
    atomic_write_json,
    file_sha256,
    iter_jsonl_gz,
    seal_record,
    verify_sealed_record,
)
from flcva.labels import (
    CANONICAL_SEALED_LABEL_SHA256,
    verify_gold_label_attestation,
)
from flcva.lineage import verify_formal_difficulty_lineage
from flcva.replay import (
    SCHEDULERS,
    derive_replay_curves,
    replay_curve_statistics,
)
from flcva.schemas import RunRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_AUDIT_GATES = {
    "artifact_count_matches_manifest",
    "branch_completeness_ge_99_5pct",
    "evaluation_present_and_sealed",
    "parser_rate_gate",
    "prefix_replay_100pct",
    "replay_present_and_sealed",
    "score_count_matches_manifest",
    "frozen_campaign_match",
    "gold_labels_match",
    "difficulty_peer_match",
    "runtime_lineage_verified",
}


def _verify_summary_tree(
    run_dir: Path,
    subdirectory: str,
    summary: dict[str, Any],
) -> None:
    if not verify_sealed_record(summary):
        raise ValueError(f"{subdirectory} summary hash mismatch: {run_dir}")
    artifacts = summary.get("artifacts_sha256", {})
    if not artifacts:
        raise ValueError(f"{subdirectory} summary has no artifact hashes: {run_dir}")
    for relative, expected in artifacts.items():
        target = run_dir / subdirectory / relative
        if not target.is_file() or file_sha256(target) != expected:
            raise ValueError(
                f"{subdirectory} artifact hash mismatch: {run_dir} / {relative}"
            )


def _load(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    audit = json.loads((run_dir / "audit_manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads(
        (run_dir / "evaluation" / "summary.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (run_dir / "replay" / "summary.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (run_dir / "contract.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    run_record = RunRecord.model_validate(run)
    verify_run_record(run_record)
    if not verify_sealed_record(manifest):
        raise ValueError(f"manifest hash mismatch: {run_dir}")
    if (
        manifest.get("run_record_sha256") != run_record.record_sha256
        or manifest.get("runtime_environment_sha256")
        != run_record.runtime_environment_sha256
    ):
        raise ValueError(f"manifest/run runtime binding mismatch: {run_dir}")
    unhashed_contract = dict(contract)
    recorded_contract_sha = unhashed_contract.get("contract_sha256")
    unhashed_contract["contract_sha256"] = ""
    if recorded_contract_sha != payload_sha256(unhashed_contract):
        raise ValueError(f"contract hash mismatch: {run_dir}")
    if not verify_sealed_record(audit):
        raise ValueError(f"audit manifest hash mismatch: {run_dir}")
    identities = {
        (
            item.get("run_id"),
            item.get("protocol_id"),
            item.get("contract_sha256"),
        )
        for item in (run, audit, contract, manifest)
    }
    if len(identities) != 1:
        raise ValueError(f"run/contract/manifest/audit lineage mismatch: {run_dir}")
    manifest_config = manifest.get("config", {})
    config = RunConfig.model_validate(manifest_config)
    frozen_campaign_sha256 = validate_formal_manifest_config(
        manifest_config,
        PROJECT_ROOT,
    )
    if (
        contract.get("stage") != "formal"
        or manifest_config.get("stage") != "formal"
        or manifest_config.get("model", {}).get("id") != contract.get("model_id")
        or contract.get("schema_version") != "flcva-experiment-contract-v2"
        or contract.get("frozen_campaign_sha256") != frozen_campaign_sha256
    ):
        raise ValueError(f"formal frozen campaign contract mismatch: {run_dir}")
    if run["status"] != "sealed" or not audit["passed"]:
        raise ValueError(f"run is not sealed: {run_dir}")
    gates = audit.get("gates", {})
    if not REQUIRED_AUDIT_GATES.issubset(gates) or not all(
        gates[name] for name in REQUIRED_AUDIT_GATES
    ):
        raise ValueError(f"run lacks the complete fail-closed audit gates: {run_dir}")
    if not audit.get("result_merkle_root"):
        raise ValueError(f"run has no result Merkle root: {run_dir}")
    _verify_summary_tree(run_dir, "evaluation", evaluation)
    _verify_summary_tree(run_dir, "replay", replay)
    if run["stage"] != "formal" or not evaluation["scientific_results"]:
        raise ValueError(f"run is not a formal scientific result: {run_dir}")
    if (
        evaluation.get("gold_labels", {}).get("sha256")
        != CANONICAL_SEALED_LABEL_SHA256["combined"]
        or evaluation.get("gold_labels", {}).get("canonical_sha256")
        != CANONICAL_SEALED_LABEL_SHA256["combined"]
        or not evaluation.get("gold_labels", {}).get("canonical")
    ):
        raise ValueError(f"formal result is not bound to canonical gold labels: {run_dir}")
    verify_gold_label_attestation(
        config,
        PROJECT_ROOT,
        evaluation.get("gold_labels", {}),
    )
    verify_formal_difficulty_lineage(
        config,
        PROJECT_ROOT,
        run_dir,
        evaluation.get("difficulty_lineage", {}),
    )
    return run, audit, evaluation, replay


def _best(metrics: list[dict[str, Any]], target: str) -> str:
    values: list[float] = []
    for summary in metrics:
        for row in summary["metrics"][target].values():
            value = row.get("auroc")
            if row.get("status") == "ok" and value is not None:
                values.append(float(value))
    return f"{max(values):.3f}" if values else "--"


def _tex(value: Any) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("_", r"\_")
        .replace("%", r"\%")
    )


def _number(value: Any, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])


def _proxy_rows(
    entries: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, evaluation, _ in entries:
        for target, proxies in evaluation["metrics"].items():
            for proxy, metrics in proxies.items():
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "proxy": proxy,
                        "status": metrics.get("status"),
                        "analysis_n": metrics.get("analysis_n"),
                        "confirmation_n": metrics.get("confirmation_n"),
                        "auroc": metrics.get("auroc"),
                        "auprc": metrics.get("auprc"),
                        "brier": metrics.get("brier"),
                        "auroc_low": (metrics.get("auroc_ci") or {}).get("low"),
                        "auroc_high": (metrics.get("auroc_ci") or {}).get("high"),
                    }
                )
    return rows


def _write_proxy_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Model & Target & Proxy & $n_{\rm conf}$ & AUROC & AUPRC & Brier \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex(row["model"]),
                    _tex(row["target"]),
                    _tex(row["proxy"]),
                    _tex(row["confirmation_n"] or 0),
                    _number(row["auroc"]),
                    _number(row["auprc"]),
                    _number(row["brier"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Confirmation-set proxy discrimination and calibration. "
                r"Orientation and probability bins are learned on analysis problems only.}"
            ),
            r"\label{tab:sealed-proxy-results}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _replay_rows(
    entries: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, _, replay in entries:
        endpoint = int(replay["common_endpoint"])
        start = int(replay["aulc_start_budget"])
        span = endpoint - start
        if span <= 0:
            raise ValueError(f"{model}: invalid replay AULC span")
        for scheduler, aulc in replay["aulc"].items():
            rows.append(
                {
                    "model": model,
                    "scheduler": scheduler,
                    "common_endpoint": endpoint,
                    "aulc_start_budget": start,
                    "aulc_span": span,
                    "aulc": aulc,
                    "mean_accuracy": float(aulc) / span,
                    "final_accuracy": replay["final_accuracy"][scheduler],
                }
            )
    return rows


def _write_replay_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Model & Scheduler & Mean accuracy & Final accuracy \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _tex(row["model"]),
                    _tex(row["scheduler"]),
                    _number(row["mean_accuracy"]),
                    _number(row["final_accuracy"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Equal-ledger replay through each model's common endpoint. "
                r"Mean accuracy is AULC divided by the observable budget span "
                r"from the first complete answer frontier to that endpoint.}"
            ),
            r"\label{tab:sealed-replay-results}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_replay_points(
    run_dir: Path,
    model: str,
    endpoint: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(
        run_dir / "replay" / "points.jsonl.gz",
        "rt",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            point = json.loads(line)
            budget = int(point["budget"])
            if budget <= endpoint:
                rows.append(
                    {
                        "model": model,
                        "scheduler": point["scheduler"],
                        "budget": budget,
                        "normalized_budget": budget / endpoint,
                        "accuracy": point["accuracy"],
                    }
                )
    return rows


def _derive_replay_v2(
    run_dir: Path,
    evaluation: dict[str, Any],
    sealed_replay_v1: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    config = RunConfig.model_validate(manifest["config"])
    frontiers_path = run_dir / "evaluation" / "frontiers.jsonl.gz"
    expected_frontiers = evaluation["artifacts_sha256"].get(
        "frontiers.jsonl.gz"
    )
    if (
        expected_frontiers is None
        or file_sha256(frontiers_path) != expected_frontiers
    ):
        raise ValueError(f"authenticated frontier digest mismatch: {run_dir}")
    states = [
        row
        for row in iter_jsonl_gz(frontiers_path)
        if row.get("split") == "confirmation"
    ]
    if not states:
        raise ValueError(f"no confirmation frontiers: {run_dir}")
    curves = derive_replay_curves(states, config)
    statistics = replay_curve_statistics(states, curves, config)
    if int(statistics["common_endpoint"]) != int(
        sealed_replay_v1["common_endpoint"]
    ):
        raise ValueError(
            f"replay-v2 endpoint changed unexpectedly: {run_dir}"
        )
    if not math.isclose(
        float(statistics["oracle_difference"]),
        float(sealed_replay_v1["oracle_difference"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "removing the common pre-observation rectangle changed the "
            f"oracle/round-robin difference: {run_dir}"
        )
    old_permutation = sealed_replay_v1["permutation_oracle"]
    replicates = int(old_permutation["replicates"])
    raw_fraction = old_permutation.get("fraction_at_least_original")
    exceedances = (
        int(round(float(raw_fraction) * replicates))
        if raw_fraction is not None
        else None
    )
    permutation = {
        **old_permutation,
        "exceedances": exceedances,
        "p_value_corrected": (
            (exceedances + 1) / (replicates + 1)
            if exceedances is not None
            else None
        ),
        "reuse_justification": (
            "oracle-minus-round-robin removes the same unobservable initial "
            "rectangle from both curves; TRACE policy does not enter the "
            "permuted-oracle statistic"
        ),
        "source_replay_schema": sealed_replay_v1["schema_version"],
    }
    derived = {
        "schema_version": "flcva-postseal-replay-v2",
        "run_id": manifest["run_id"],
        "split_scope": "confirmation",
        **statistics,
        "permutation_oracle": permutation,
        "source": {
            "result_merkle_root": json.loads(
                (run_dir / "audit_manifest.json").read_text(encoding="utf-8")
            )["result_merkle_root"],
            "frontiers_sha256": expected_frontiers,
            "evaluation_summary_sha256": file_sha256(
                run_dir / "evaluation" / "summary.json"
            ),
            "sealed_replay_v1_summary_sha256": file_sha256(
                run_dir / "replay" / "summary.json"
            ),
        },
    }
    endpoint = int(statistics["common_endpoint"])
    points = [
        point.model_dump(mode="json")
        for scheduler in SCHEDULERS
        for point in curves[scheduler]
        if point.budget <= endpoint
    ]
    return derived, points


def _write_replay_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    models = sorted({row["model"] for row in rows})
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(7.0, 2.7),
        sharey=True,
        squeeze=False,
    )
    for axis, model in zip(axes[0], models, strict=True):
        model_rows = [row for row in rows if row["model"] == model]
        for scheduler in sorted({row["scheduler"] for row in model_rows}):
            curve = sorted(
                (row for row in model_rows if row["scheduler"] == scheduler),
                key=lambda row: row["normalized_budget"],
            )
            axis.step(
                [row["normalized_budget"] for row in curve],
                [row["accuracy"] for row in curve],
                where="post",
                linewidth=1.1,
                label=scheduler.replace("_", " "),
            )
        axis.set_title(model)
        axis.set_xlabel("Fraction of common ledger budget")
        axis.grid(alpha=0.2)
    axes[0][0].set_ylabel("Exact-match accuracy")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-run", type=Path, required=True)
    parser.add_argument("--deepseek-run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/results.tex"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/generated"),
    )
    args = parser.parse_args()
    loaded = [_load(args.qwen_run), _load(args.deepseek_run)]
    protocols = {item[0]["protocol_id"] for item in loaded}
    if protocols != {"flcva-2025-cutoff-v3"}:
        raise ValueError(f"unexpected protocol set: {protocols}")
    models = {
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))["config"][
            "model"
        ]["id"]
        for path in (args.qwen_run, args.deepseek_run)
    }
    expected_models = {
        "Qwen/Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    if models != expected_models:
        raise ValueError(f"unexpected model set: {models}")
    audits = [item[1] for item in loaded]
    evaluations = [item[2] for item in loaded]
    sealed_replays_v1 = [item[3] for item in loaded]
    if evaluations[0].get("difficulty_lineage") != evaluations[1].get(
        "difficulty_lineage"
    ):
        raise ValueError(
            "formal evaluations were not computed from the same authenticated "
            "two-model difficulty lineage"
        )
    if evaluations[0].get("gold_labels") != evaluations[1].get("gold_labels"):
        raise ValueError("formal evaluations disagree on gold-label attestation")
    run_paths = [args.qwen_run, args.deepseek_run]
    model_names = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))["config"][
            "model"
        ]["id"].split("/")[-1]
        for path in run_paths
    ]
    replay_v2_payloads = [
        _derive_replay_v2(run_dir, evaluation, sealed_replay)
        for run_dir, evaluation, sealed_replay in zip(
            run_paths,
            evaluations,
            sealed_replays_v1,
            strict=True,
        )
    ]
    replays = [payload[0] for payload in replay_v2_payloads]
    entries = list(zip(model_names, evaluations, replays, strict=True))
    total_problems = sum(item["totals"]["problems"] for item in audits)
    total_values = sum(item["values"] for item in evaluations)
    strong_title = all(
        item.get("headline", {}).get("strong_title_supported", False)
        for item in evaluations
    )
    headline = (
        "Stable Answers Can Be the Worst Place to Spend the Next Token"
        if strong_title
        else "A Full-Ledger Counterfactual Audit of Adaptive Reasoning"
    )
    oracle_gains = [
        100.0
        * (item["aulc"]["oracle"] - item["aulc"]["round_robin"])
        / item["aulc"]["round_robin"]
        for item in replays
        if item["aulc"]["round_robin"]
    ]
    oracle_gain_by_model = {
        model: (
            100.0
            * (replay["aulc"]["oracle"] - replay["aulc"]["round_robin"])
            / replay["aulc"]["round_robin"]
        )
        for model, _, replay in entries
        if replay["aulc"]["round_robin"]
    }
    permutation_p = [
        float(item["permutation_oracle"]["p_value_corrected"])
        for item in replays
        if item["permutation_oracle"]["p_value_corrected"] is not None
    ]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_rows = _proxy_rows(entries)
    replay_rows = _replay_rows(entries)
    replay_points = []
    for model, replay, (_, points) in zip(
        model_names,
        replays,
        replay_v2_payloads,
        strict=True,
    ):
        endpoint = int(replay["common_endpoint"])
        start = int(replay["aulc_start_budget"])
        replay_points.extend(
            {
                "model": model,
                "scheduler": point["scheduler"],
                "budget": int(point["budget"]),
                "normalized_budget": (
                    (int(point["budget"]) - start) / (endpoint - start)
                ),
                "accuracy": point["accuracy"],
            }
            for point in points
        )
    calibration_rows = [
        {
            "model": model,
            "target": target,
            "proxy": proxy,
            **bin_row,
        }
        for model, evaluation, _ in entries
        for target, proxies in evaluation["metrics"].items()
        for proxy, metrics in proxies.items()
        for bin_row in metrics.get("calibration", [])
    ]
    stratified_rows = [
        {"model_run": model, **row}
        for model, evaluation, _ in entries
        for row in evaluation.get("stratified_metrics", [])
    ]
    proxy_table = output_dir / "proxy_table.tex"
    replay_table = output_dir / "replay_table.tex"
    replay_figure = output_dir / "replay_curves.pdf"
    _write_proxy_table(proxy_table, proxy_rows)
    _write_replay_table(replay_table, replay_rows)
    _write_replay_figure(replay_figure, replay_points)
    _write_csv(output_dir / "proxy_metrics.csv", proxy_rows)
    _write_csv(output_dir / "replay.csv", replay_rows)
    _write_csv(output_dir / "replay_points.csv", replay_points)
    _write_csv(output_dir / "calibration.csv", calibration_rows)
    _write_csv(output_dir / "stratified_metrics.csv", stratified_rows)
    replay_v2_path = output_dir / "replay_v2_summary.json"
    atomic_write_json(
        replay_v2_path,
        seal_record(
            {
                "schema_version": "flcva-paper-replay-v2-bundle-v1",
                "trace_missing_policy": replays[0]["trace_missing_policy"],
                "runs": replays,
            }
        ),
    )
    relative_generated = Path(
        os.path.relpath(output_dir, start=args.output.parent)
    )
    content = "\n".join(
        [
            "% Auto-generated from sealed formal artifacts.",
            "\\newif\\ifsealedresults",
            "\\sealedresultstrue",
            "\\newcommand{\\ResultStatus}{\\textsc{Sealed Formal Result}}",
            f"\\newcommand{{\\HeadlineTitle}}{{{headline}}}",
            f"\\newcommand{{\\TotalEvaluatedProblems}}{{{total_problems:,}}}",
            f"\\newcommand{{\\TotalEligibleCheckpoints}}{{{total_values:,}}}",
            f"\\newcommand{{\\BestProxyBeneficialAUROC}}{{{_best(evaluations, 'beneficial')}}}",
            f"\\newcommand{{\\BestProxyHarmfulAUROC}}{{{_best(evaluations, 'harmful')}}}",
            (
                "\\newcommand{\\OracleAULCGain}{"
                + (f"{sum(oracle_gains) / len(oracle_gains):.2f}\\%" if oracle_gains else "--")
                + "}"
            ),
            (
                "\\newcommand{\\QwenOracleAULCGain}{"
                + (
                    f"{oracle_gain_by_model.get('Qwen3-8B', float('nan')):.2f}\\%"
                    if "Qwen3-8B" in oracle_gain_by_model
                    else "--"
                )
                + "}"
            ),
            (
                "\\newcommand{\\DeepSeekOracleAULCGain}{"
                + (
                    f"{oracle_gain_by_model.get('DeepSeek-R1-Distill-Qwen-7B', float('nan')):.2f}\\%"
                    if "DeepSeek-R1-Distill-Qwen-7B" in oracle_gain_by_model
                    else "--"
                )
                + "}"
            ),
            (
                "\\newcommand{\\PermutationPValue}{"
                + (f"{max(permutation_p):.3f}" if permutation_p else "--")
                + "}"
            ),
            (
                "\\newcommand{\\GeneratedProxyTable}{"
                + str(relative_generated / proxy_table.name)
                + "}"
            ),
            (
                "\\newcommand{\\GeneratedReplayTable}{"
                + str(relative_generated / replay_table.name)
                + "}"
            ),
            (
                "\\newcommand{\\GeneratedReplayFigure}{"
                + str(relative_generated / replay_figure.name)
                + "}"
            ),
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    generated_paths = [
        proxy_table,
        replay_table,
        replay_figure,
        output_dir / "proxy_metrics.csv",
        output_dir / "replay.csv",
        output_dir / "replay_points.csv",
        output_dir / "calibration.csv",
        output_dir / "stratified_metrics.csv",
        replay_v2_path,
    ]
    artifact_paths = [
        PROJECT_ROOT / "paper" / "main.tex",
        PROJECT_ROOT / "paper" / "references.bib",
        args.output.resolve(),
        *generated_paths,
    ]
    def artifact_key(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    manifest = seal_record({
        "schema_version": "flcva-paper-result-manifest-v2",
        "analysis_schema": "flcva-postseal-replay-v2",
        "renderer_sha256": file_sha256(Path(__file__).resolve()),
        "source_runs": [
            {
                "path": str(path),
                "run_id": run["run_id"],
                "result_merkle_root": audit["result_merkle_root"],
                "evaluation_summary_sha256": file_sha256(
                    path / "evaluation" / "summary.json"
                ),
                "sealed_replay_v1_summary_sha256": file_sha256(
                    path / "replay" / "summary.json"
                ),
            }
            for path, (run, audit, _, _) in zip(
                run_paths,
                loaded,
                strict=True,
            )
        ],
        "artifacts_sha256": {
            artifact_key(path): file_sha256(path)
            for path in artifact_paths
        },
    })
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(args.output)


if __name__ == "__main__":
    main()
