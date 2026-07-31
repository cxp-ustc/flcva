#!/usr/bin/env python3
"""Correct the v5 probe replay text diagnostic without rerunning the GPUs.

The v5 GPU evaluator compared vLLM's context-aware ``output.text`` with
``tokenizer.decode(answer_token_ids)`` performed outside the original prompt
context.  Some tokenizers render boundary whitespace or punctuation
differently in those two situations, so that diagnostic is not raw-text
identity even when the answer-token sequence is identical.

This post-seal analysis compares each canonical rerun's ``raw_answer`` with
the corresponding ``stop_raw_answer`` or branch ``raw_answer`` stored in the
sealed formal problem artifact.  It changes only the replay diagnostic and
its table, preserves every GPU output and scientific value row, and records
the correction lineage in the resealed v5 manifest.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from flcva.ids import payload_sha256, stable_seed
from flcva.io import (
    file_sha256,
    read_json_gz,
    seal_record,
    verify_sealed_record,
)


ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = ROOT / "paper" / "ablation" / "probe_sensitivity.py"
GENERATED_DIR = ROOT / "paper" / "generated"
MANIFEST_PATH = GENERATED_DIR / "probe_sensitivity_manifest.json"
METRICS_PATH = GENERATED_DIR / "probe_sensitivity_metrics.csv"
REPLAY_TABLE_PATH = GENERATED_DIR / "probe_sensitivity_replay_table.tex"
GATE_PATH = GENERATED_DIR / "probe_gate.tex"
EXPECTED_SCHEMA = "flcva-probe-sensitivity-manifest-v5"
EXPECTED_STUDY = "flcva-probe-sensitivity-v5"
RESULT_DIRS = {
    "qwen3": ROOT / "downloads" / "flcva-probe-sensitivity-v5-qwen3",
    "deepseek": ROOT / "downloads" / "flcva-probe-sensitivity-v5-deepseek",
}
MODEL_LABELS = {"qwen3": "Qwen3", "deepseek": "DeepSeek"}


def _load_probe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "flcva_probe_sensitivity_v5", PROBE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PROBE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_raw_answers(
    probe: ModuleType,
    model_name: str,
    problem_ids: set[str],
) -> tuple[dict[tuple[str, str], str], str]:
    source_run = probe.SOURCE_RUNS[model_name]
    answers: dict[tuple[str, str], str] = {}
    for problem_id in sorted(problem_ids):
        artifact = read_json_gz(probe._problem_path(source_run, problem_id))
        if not verify_sealed_record(artifact):
            raise ValueError(
                f"{model_name}/{problem_id}: formal problem seal failed"
            )
        if str(artifact.get("problem_id")) != problem_id:
            raise ValueError(
                f"{model_name}/{problem_id}: formal problem ID mismatch"
            )
        for checkpoint in artifact["checkpoints"]:
            key = (problem_id, f"stop:{checkpoint['checkpoint_id']}")
            if key in answers:
                raise ValueError(f"duplicate formal context: {key}")
            answers[key] = str(checkpoint["stop_raw_answer"])
        for branch in artifact["branches"]:
            key = (problem_id, f"branch:{branch['branch_id']}")
            if key in answers:
                raise ValueError(f"duplicate formal context: {key}")
            answers[key] = str(branch["raw_answer"])
    digest_rows = [
        [problem_id, context_id, raw_answer]
        for (problem_id, context_id), raw_answer in sorted(answers.items())
    ]
    return answers, payload_sha256(digest_rows)


def _corrected_rows(
    probe: ModuleType,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, outputs, _ = probe._load_outputs(
        model_name, RESULT_DIRS[model_name].resolve()
    )
    canonical = [row for row in outputs if row["cue_name"] == "canonical"]
    if len(canonical) != 60_000:
        raise ValueError(
            f"{model_name}: expected 60,000 canonical contexts, "
            f"found {len(canonical)}"
        )
    problem_ids = {str(row["problem_id"]) for row in canonical}
    if len(problem_ids) != 200:
        raise ValueError(
            f"{model_name}: expected 200 sampled problems, "
            f"found {len(problem_ids)}"
        )
    source_answers, source_digest = _source_raw_answers(
        probe, model_name, problem_ids
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for output in canonical:
        key = (str(output["problem_id"]), str(output["context_id"]))
        if key in seen:
            raise ValueError(f"{model_name}: duplicate canonical context {key}")
        seen.add(key)
        if key not in source_answers:
            raise ValueError(f"{model_name}: missing formal context {key}")
        rows.append(
            {
                "problem_id": key[0],
                "raw_text_replay": int(
                    str(output["raw_answer"]) == source_answers[key]
                ),
            }
        )
    estimate, low, high = probe._cluster_interval(
        rows,
        "raw_text_replay",
        seed=stable_seed(
            20250701, model_name, "canonical", "source_text_replay"
        ),
    )
    return rows, {
        "n": len(rows),
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "formal_raw_answer_map_sha256": source_digest,
        "problem_count": len(problem_ids),
    }


def _read_metrics() -> tuple[list[str], list[dict[str, str]]]:
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("probe metrics have no header")
        return list(reader.fieldnames), list(reader)


def _write_metrics(
    probe: ModuleType,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    probe._atomic_bytes(METRICS_PATH, buffer.getvalue().encode("utf-8"))


def _render_replay_table(
    probe: ModuleType,
    metrics: list[dict[str, str]],
) -> None:
    index = {
        (row["model"], row["cue"], row["metric"]): row for row in metrics
    }

    def interval(row: dict[str, str]) -> str:
        return (
            f"{100 * float(row['estimate']):.1f}\\% "
            f"[{100 * float(row['ci_low']):.1f}, "
            f"{100 * float(row['ci_high']):.1f}]"
        )

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        (
            r"Model & Token identity & Raw-text identity & "
            r"Normalized-answer identity \\"
        ),
        r"\midrule",
    ]
    for model_name in ("qwen3", "deepseek"):
        lines.append(
            "{} & {} & {} & {} \\\\".format(
                MODEL_LABELS[model_name],
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
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    probe._atomic_bytes(
        REPLAY_TABLE_PATH, ("\n".join(lines) + "\n").encode("utf-8")
    )


def main() -> None:
    probe = _load_probe_module()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != EXPECTED_SCHEMA
        or manifest.get("study_id") != EXPECTED_STUDY
        or not verify_sealed_record(manifest)
    ):
        raise ValueError("expected a sealed v5 probe sensitivity manifest")
    if manifest.get("script_sha256") != file_sha256(PROBE_SCRIPT):
        raise ValueError("sealed GPU evaluator no longer matches v5 source")
    prior_attestation = manifest.get("postseal_raw_text_correction", {})
    source_manifest_record = prior_attestation.get(
        "source_manifest_record_sha256", manifest["record_sha256"]
    )

    corrected: dict[str, dict[str, Any]] = {}
    for model_name in ("qwen3", "deepseek"):
        _, corrected[model_name] = _corrected_rows(probe, model_name)

    fieldnames, metric_rows = _read_metrics()
    replaced: set[str] = set()
    for row in metric_rows:
        model_name = row["model"]
        if (
            model_name in corrected
            and row["cue"] == "canonical"
            and row["metric"] == "source_text_replay"
        ):
            values = corrected[model_name]
            row["n"] = str(values["n"])
            row["estimate"] = repr(values["estimate"])
            row["ci_low"] = repr(values["ci_low"])
            row["ci_high"] = repr(values["ci_high"])
            replaced.add(model_name)
    if replaced != set(corrected):
        raise ValueError(
            f"did not replace both source_text_replay rows: {replaced}"
        )
    _write_metrics(probe, fieldnames, metric_rows)
    _render_replay_table(probe, metric_rows)

    for model_name, values in corrected.items():
        manifest["source_results"][model_name][
            "canonical_control_text_replay_rate"
        ] = values["estimate"]
    manifest["postseal_raw_text_correction"] = {
        "schema_version": "flcva-probe-raw-text-correction-v1",
        "source_manifest_record_sha256": source_manifest_record,
        "analysis_script_sha256": file_sha256(Path(__file__)),
        "metric_semantics": (
            "exact equality between rerun raw_answer and the corresponding "
            "raw answer stored in the sealed formal problem artifact"
        ),
        "reason": (
            "v5's isolated tokenizer.decode comparison was not equivalent "
            "to context-aware raw output text"
        ),
        "gpu_outputs_changed": False,
        "scientific_value_rows_changed": False,
        "models": corrected,
    }
    artifact_names = set(manifest["artifacts_sha256"])
    if artifact_names != {
        "probe_sensitivity_metrics.csv",
        "probe_sensitivity_values.csv.gz",
        "probe_sensitivity_table.tex",
        "probe_sensitivity_replay_table.tex",
    }:
        raise ValueError("unexpected v5 probe artifact set")
    manifest["artifacts_sha256"] = {
        name: file_sha256(GENERATED_DIR / name)
        for name in sorted(artifact_names)
    }
    manifest["record_sha256"] = ""
    sealed = seal_record(manifest)
    probe._atomic_json(MANIFEST_PATH, sealed)
    gate = (
        "% Generated only after sealed probe outputs pass all hard gates.\n"
        f"% manifest_record_sha256={sealed['record_sha256']}\n"
        "\\newif\\ifprobevalidated\n"
        "\\probevalidatedtrue\n"
    )
    probe._atomic_bytes(GATE_PATH, gate.encode("utf-8"))
    print(
        json.dumps(
            {
                "manifest_record_sha256": sealed["record_sha256"],
                "models": {
                    name: {
                        key: values[key]
                        for key in ("n", "estimate", "ci_low", "ci_high")
                    }
                    for name, values in corrected.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
