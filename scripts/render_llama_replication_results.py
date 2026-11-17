#!/usr/bin/env python3
"""Render an audited Llama replication result without mutating the primary paper."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from flcva.config import RunConfig
from flcva.frozen import (
    LLAMA_REPLICATION_CAMPAIGN_SHA256,
    LLAMA_REPLICATION_PROTOCOL_ID,
    validate_formal_manifest_config,
)
from flcva.io import (
    require_artifacts,
    atomic_write_json,
    seal_record,
    verify_sealed_record,
)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"


def _read_sealed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_sealed_record(payload):
        raise ValueError(f"sealed record verification failed: {path}")
    return payload


def _verify_artifacts(root: Path, summary: dict[str, Any]) -> None:
    require_artifacts(root, summary)


def _best_proxy(metrics: dict[str, Any], target: str) -> dict[str, Any]:
    rows = [
        (name, payload)
        for name, payload in metrics[target].items()
        if payload.get("status") == "ok" and payload.get("auroc") is not None
    ]
    if not rows:
        raise ValueError(f"no valid proxy metric for {target}")
    name, payload = max(rows, key=lambda row: float(row[1]["auroc"]))
    interval = payload["auroc_ci"]
    return {
        "target": target,
        "proxy": name,
        "auroc": float(payload["auroc"]),
        "auroc_low": float(interval["low"]),
        "auroc_high": float(interval["high"]),
        "auprc": float(payload["auprc"]),
        "brier": float(payload["brier"]),
        "confirmation_n": int(payload["confirmation_n"]),
    }


def _relative_gain(aulc: dict[str, Any], policy: str) -> float:
    reference = float(aulc["round_robin"])
    if reference <= 0:
        raise ValueError("round-robin AULC must be positive")
    return 100.0 * (float(aulc[policy]) - reference) / reference


def _branch_estimands(values_path: Path) -> dict[str, dict[str, Any]]:
    accumulators = {
        "beneficial": {
            "checkpoints": 0,
            "branch_events": 0,
            "branch_trials": 0,
            "opportunities": 0,
        },
        "harmful": {
            "checkpoints": 0,
            "branch_events": 0,
            "branch_trials": 0,
            "opportunities": 0,
        },
    }
    with gzip.open(values_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("split") != "confirmation":
                continue
            continuation = row.get("continuation_correct")
            if (
                not isinstance(continuation, list)
                or len(continuation) != 4
                or any(value not in (0, 1) for value in continuation)
            ):
                raise ValueError(
                    "invalid four-branch correctness vector at "
                    f"{values_path}:{line_number}"
                )
            stop_correct = row.get("stop_correct")
            continuation_correct = sum(int(value) for value in continuation)
            if stop_correct == 0:
                target = "beneficial"
                branch_events = continuation_correct
                opportunity = int(continuation_correct > 0)
                if int(row.get("beneficial", -1)) != opportunity:
                    raise ValueError(
                        "beneficial label disagrees with branch outcomes at "
                        f"{values_path}:{line_number}"
                    )
            elif stop_correct == 1:
                target = "harmful"
                branch_events = 4 - continuation_correct
                opportunity = int(continuation_correct < 4)
                if int(row.get("harmful", -1)) != opportunity:
                    raise ValueError(
                        "harmful label disagrees with branch outcomes at "
                        f"{values_path}:{line_number}"
                    )
            else:
                raise ValueError(
                    f"invalid stop correctness at {values_path}:{line_number}"
                )
            accumulator = accumulators[target]
            accumulator["checkpoints"] += 1
            accumulator["branch_events"] += branch_events
            accumulator["branch_trials"] += 4
            accumulator["opportunities"] += opportunity

    rendered: dict[str, dict[str, Any]] = {}
    for target, accumulator in accumulators.items():
        checkpoints = int(accumulator["checkpoints"])
        branch_trials = int(accumulator["branch_trials"])
        if checkpoints <= 0 or branch_trials != 4 * checkpoints:
            raise ValueError(f"missing confirmation estimand population: {target}")
        rendered[target] = {
            **accumulator,
            "single_draw_rate": (float(accumulator["branch_events"]) / branch_trials),
            "four_draw_opportunity_rate": (
                float(accumulator["opportunities"]) / checkpoints
            ),
        }
    return rendered


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render(run_dir: Path, project_root: Path, output_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = _read_sealed(run_dir / "manifest.json")
    config = RunConfig.model_validate(manifest["config"])
    campaign_sha256 = validate_formal_manifest_config(
        manifest["config"],
        project_root,
    )
    if (
        config.protocol_id != LLAMA_REPLICATION_PROTOCOL_ID
        or config.model.id != MODEL_ID
        or config.model.revision != MODEL_REVISION
        or campaign_sha256 != LLAMA_REPLICATION_CAMPAIGN_SHA256
    ):
        raise ValueError("run is not the frozen Llama architecture replication")

    audit = _read_sealed(run_dir / "audit_manifest.json")
    if (
        audit.get("run_id") != config.run_id
        or audit.get("passed") is not True
        or not all(audit.get("gates", {}).values())
    ):
        raise ValueError("Llama replication audit did not pass every gate")
    evaluation = _read_sealed(run_dir / "evaluation" / "summary.json")
    replay = _read_sealed(run_dir / "replay" / "summary.json")
    _verify_artifacts(run_dir / "evaluation", evaluation)
    _verify_artifacts(run_dir / "replay", replay)
    if (
        evaluation.get("scientific_results") is not True
        or evaluation.get("difficulty_current_outcomes_excluded") is not True
        or evaluation.get("difficulty_lineage", {}).get("current_outcomes_excluded")
        is not True
    ):
        raise ValueError(
            "replication difficulty is not authenticated as outcome-external"
        )

    proxy_rows = [
        _best_proxy(evaluation["metrics"], target)
        for target in ("beneficial", "harmful")
    ]
    branch_estimands = _branch_estimands(run_dir / "evaluation" / "values.jsonl.gz")
    persistence_contrast = evaluation.get("headline", {}).get("contrast_high_minus_low")
    if (
        not isinstance(persistence_contrast, dict)
        or persistence_contrast.get("estimate") is None
        or persistence_contrast.get("low") is None
        or persistence_contrast.get("high") is None
    ):
        raise ValueError("sealed persistence contrast is missing")
    policies = [
        policy
        for policy in ("trace", "prm", "anll", "entropy", "oracle")
        if policy in replay["aulc"]
    ]
    replay_rows = [
        {
            "policy": policy,
            "relative_aulc_gain_percent": _relative_gain(
                replay["aulc"],
                policy,
            ),
            "final_accuracy": float(replay["final_accuracy"][policy]),
        }
        for policy in policies
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_by_target = {row["target"]: row for row in proxy_rows}
    table_lines = [
        "% Auto-generated from the sealed Llama replication.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Target & Best proxy & AUROC (95\\% CI) & AUPRC & Brier \\\\",
        "\\midrule",
    ]
    for target in ("beneficial", "harmful"):
        row = proxy_by_target[target]
        table_lines.append(
            f"{target.capitalize()} & "
            f"{str(row['proxy']).replace('_', ' ')} & "
            f"{row['auroc']:.3f} "
            f"[{row['auroc_low']:.3f}, {row['auroc_high']:.3f}] & "
            f"{row['auprc']:.3f} & {row['brier']:.3f} \\\\"
        )
    table_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{External DeepSeek-R1-Distill-Llama-8B architecture "
            "replication. Difficulty strata are frozen from the two primary "
            "models and exclude Llama outcomes.}",
            "\\label{tab:llama-replication}",
            "\\end{table}",
        ]
    )
    table_path = output_dir / "table.tex"
    _atomic_text(table_path, "\n".join(table_lines) + "\n")

    estimand_lines = [
        "% Auto-generated from the sealed Llama replication.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Event & Single draw & Four-draw detection & Persistence AUROC "
        "(95\\% CI) \\\\",
        "\\midrule",
    ]
    proxy_by_target = {row["target"]: row for row in proxy_rows}
    for target, label in (
        ("beneficial", "Recovery when currently wrong"),
        ("harmful", "Damage when currently correct"),
    ):
        estimand = branch_estimands[target]
        proxy = proxy_by_target[target]
        estimand_lines.append(
            f"{label} & "
            f"{100.0 * estimand['single_draw_rate']:.1f}\\% & "
            f"{100.0 * estimand['four_draw_opportunity_rate']:.1f}\\% & "
            f"{proxy['auroc']:.3f} "
            f"[{proxy['auroc_low']:.3f}, {proxy['auroc_high']:.3f}] \\\\"
        )
    estimand_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Full-protocol external Llama-8B replication on "
            "confirmation problems. Four-draw detection reports whether at "
            "least one of four sampled continuations changes correctness; it "
            "is not a single-continuation probability.}",
            "\\label{tab:llama-estimands}",
            "\\end{table}",
        ]
    )
    estimand_table_path = output_dir / "estimand_table.tex"
    _atomic_text(
        estimand_table_path,
        "\n".join(estimand_lines) + "\n",
    )

    contrast = {
        "estimate": float(persistence_contrast["estimate"]),
        "low": float(persistence_contrast["low"]),
        "high": float(persistence_contrast["high"]),
        "replicates": int(persistence_contrast["replicates"]),
        "strong_title_supported": bool(
            evaluation.get("headline", {}).get(
                "strong_title_supported",
                False,
            )
        ),
    }
    macro_lines = [
        "% Auto-generated from the sealed Llama replication.",
        "\\newcommand{\\LlamaPersistenceContrast}{" f"{contrast['estimate']:+.3f}}}",
        "\\newcommand{\\LlamaPersistenceContrastCI}{"
        f"[{contrast['low']:.3f},{contrast['high']:.3f}]}}",
    ]
    macros_path = output_dir / "macros.tex"
    _atomic_text(macros_path, "\n".join(macro_lines) + "\n")

    rendered_artifacts = [
        path.name for path in (table_path, estimand_table_path, macros_path)
    ]
    try:
        source_run = str(run_dir.relative_to(project_root))
    except ValueError as error:
        raise ValueError(
            "Llama replication source must be inside the project root"
        ) from error
    summary = seal_record(
        {
            "schema_version": "flcva-llama-replication-paper-v2",
            "analysis_status": ("separately sealed external architecture replication"),
            "source_run": source_run,
            "run_id": config.run_id,
            "model_id": config.model.id,
            "model_revision": config.model.revision,
            "frozen_campaign_sha256": campaign_sha256,
            "result_merkle_root": audit["result_merkle_root"],
            "current_outcomes_excluded_from_difficulty": True,
            "difficulty_reference_models": evaluation["difficulty_reference_models"],
            "problems": int(evaluation["problems"]),
            "branch_estimands": branch_estimands,
            "persistence_high_minus_low": contrast,
            "best_proxies": proxy_rows,
            "replay": replay_rows,
            "artifacts": rendered_artifacts,
        }
    )
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    return {
        "summary": str(summary_path),
        "table": str(table_path),
        "estimand_table": str(estimand_table_path),
        "macros": str(macros_path),
        "record_sha256": summary["record_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/generated/llama_replication"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            render(
                args.run_dir,
                project_root,
                args.output_dir,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
