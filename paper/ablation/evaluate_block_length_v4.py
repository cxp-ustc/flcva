#!/usr/bin/env python3
"""Evaluate the sealed v4 block-length run under the frozen answer operator.

The v4 generation artifacts bind answer token IDs to their deterministic
tokenizer decoding and record whether vLLM's returned ``output.text`` matched
that decoding.  With a string stop at ``}``, vLLM can stop inside a composite
token: its returned text includes the first ``}`` but excludes any suffix from
the same token.  The generation artifacts deliberately retain enough
information to reconstruct that text exactly.

This evaluator is separate from ``block_length_sensitivity.py`` because the
latter is part of the sealed generation bundle and must remain byte-identical
to the code that produced the GPU artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from flcva.io import read_json_gz, seal_record
from flcva.parsing import answers_equal

from paper.ablation import block_length_sensitivity as base

ANALYSIS_ID = "flcva-block-length-analysis-v6"
ANSWER_OPERATOR = (
    "frozen vLLM output-text semantics: when a string stop occurs inside a "
    "composite token, retain text through the first closing brace and discard "
    "the remaining decoded-token suffix"
)
_OPERATOR_AUDIT: dict[str, dict[str, int]] = {}
_SEALED_RENDER_TABLES = base._render_tables


def _render_tables(
    event_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Render tables, then clarify the post-seal adaptive policy in the caption."""
    event_path, proxy_path = _SEALED_RENDER_TABLES(
        event_rows,
        stability_rows,
    )
    text = event_path.read_text(encoding="utf-8")
    old = (
        r"\caption{Continuation-length sensitivity on the common "
        r"depth lattice. Rates describe one uniformly sampled branch; "
        r"intervals resample complete problems 2,000 times. The "
        r"context-only adaptive length is selected by an analysis-only "
        r"Ridge model using depth and dataset, without confirmation "
        r"labels or any audited proxy.}"
    )
    new = (
        r"\caption{Continuation-length sensitivity on the 200-problem "
        r"confirmation half of the stable-hash ablation sample and its "
        r"common depth lattice. Rates describe one uniformly sampled "
        r"branch; intervals resample complete problems 2,000 times. "
        r"For each length, an analysis-only Ridge model predicts "
        r"four-branch signed correctness change per mean logical token "
        r"from dataset and depth; the context-only adaptive row chooses "
        r"the largest predicted value, breaking ties toward the shorter "
        r"length. Confirmation labels and audited proxies are excluded.}"
    )
    if text.count(old) != 1:
        raise ValueError("unexpected sealed event-table caption")
    base._atomic_text(event_path, text.replace(old, new))
    return event_path, proxy_path


def _backend_answer_raw(row: dict[str, Any]) -> str:
    """Reconstruct the answer string returned by the frozen vLLM operator."""
    decoded = row.get("answer_decoded_text")
    matches = row.get("answer_backend_text_matches_decode")
    if not isinstance(decoded, str) or not isinstance(matches, bool):
        raise ValueError("answer operator provenance is incomplete")
    if matches:
        operator_text = decoded
    else:
        if row.get("answer_finish_reason") != "stop" or "}" not in decoded:
            raise ValueError(
                "backend/decode mismatch is not a reconstructible stop-boundary " "case"
            )
        operator_text = decoded[: decoded.index("}") + 1]
    return base.CANONICAL_CUE + operator_text


def _source_operator_index(
    model_name: str,
    problem_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    checkpoints: dict[str, dict[str, Any]] = {}
    branches: dict[tuple[str, int], dict[str, Any]] = {}
    source_dir = base.SOURCE_RUNS[model_name] / "evaluation" / "problems"
    for problem_id in sorted(problem_ids):
        path = source_dir / f"{base._safe_problem_name(problem_id)}.json.gz"
        payload = read_json_gz(path)
        if str(payload.get("problem_id")) != problem_id:
            raise ValueError("formal answer-operator problem identity mismatch")
        for row in payload.get("checkpoints", []):
            checkpoint_id = str(row["checkpoint_id"])
            if checkpoint_id in checkpoints:
                raise ValueError("duplicate formal checkpoint identity")
            checkpoints[checkpoint_id] = row
        for row in payload.get("branches", []):
            key = (str(row["checkpoint_id"]), int(row["branch_index"]))
            if key in branches:
                raise ValueError("duplicate formal branch identity")
            branches[key] = row
    return checkpoints, branches


def _checkpoint_values(
    model_name: str,
    outputs: list[dict[str, Any]],
    labels: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    """Compute length values while preserving the frozen answer operator."""
    grouped: dict[
        tuple[str, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    problem_ids: set[str] = set()
    for row in outputs:
        problem_id = str(row["problem_id"])
        problem_ids.add(problem_id)
        grouped[
            (
                problem_id,
                str(row["checkpoint_id"]),
                int(row["length"]),
            )
        ].append(row)
    source_checkpoints, source_branches = _source_operator_index(
        model_name,
        problem_ids,
    )
    audit = {
        "backend_decode_mismatches": 0,
        "formal_256_exact_token_rows": 0,
        "formal_256_exact_raw_rows": 0,
        "formal_256_exact_correctness_rows": 0,
    }
    values: list[dict[str, Any]] = []
    for (problem_id, checkpoint_id, length), rows in grouped.items():
        if len(rows) != 4:
            raise ValueError("length checkpoint lacks four branches")
        dataset_id, gold = labels[problem_id]
        source_checkpoint = source_checkpoints.get(checkpoint_id)
        if source_checkpoint is None:
            raise ValueError("block-length checkpoint lacks formal lineage")
        stop = int(source_checkpoint["stop_correct"])
        continuation_correct: list[int] = []
        costs: list[int] = []
        sorted_rows = sorted(rows, key=lambda item: int(item["branch_index"]))
        for row in sorted_rows:
            raw_answer = _backend_answer_raw(row)
            audit["backend_decode_mismatches"] += int(
                not row["answer_backend_text_matches_decode"]
            )
            correct, _, _ = answers_equal(dataset_id, raw_answer, gold)
            branch_correct = int(correct)
            continuation_correct.append(branch_correct)
            costs.append(int(row["logical_tokens"]))

            if length == 256 and row["formal_answer_token_match"] is True:
                audit["formal_256_exact_token_rows"] += 1
                source_branch = source_branches.get(
                    (checkpoint_id, int(row["branch_index"]))
                )
                if source_branch is None:
                    raise ValueError("exact 256 branch lacks formal lineage")
                if raw_answer == source_branch["raw_answer"]:
                    audit["formal_256_exact_raw_rows"] += 1
                if branch_correct == int(source_branch["correct"]):
                    audit["formal_256_exact_correctness_rows"] += 1

        mean_correct = float(np.mean(continuation_correct))
        mean_cost = float(np.mean(costs))
        values.append(
            {
                "model": model_name,
                "problem_id": problem_id,
                "dataset_id": dataset_id,
                "split": rows[0]["split"],
                "checkpoint_id": checkpoint_id,
                "spine_id": int(rows[0]["spine_id"]),
                "depth": int(rows[0]["depth"]),
                "length": length,
                "stop_correct": stop,
                "stop_label_source": "frozen formal answer-now operator",
                "continuation_correct": continuation_correct,
                "continuation_costs": costs,
                "answer_operator": ANSWER_OPERATOR,
                "formal_answer_token_match": [
                    row["formal_answer_token_match"] for row in sorted_rows
                ],
                "marginal_value": (1000.0 * (mean_correct - stop) / mean_cost),
                "single_draw_recovery": (mean_correct if stop == 0 else None),
                "single_draw_damage": (1.0 - mean_correct if stop == 1 else None),
                "recovery_detected": (
                    int(any(continuation_correct)) if stop == 0 else None
                ),
                "damage_detected": (
                    int(not all(continuation_correct)) if stop == 1 else None
                ),
                "proxies": rows[0]["proxies"],
            }
        )
    expected = 400 * 4 * len(base.COMMON_DEPTHS) * len(base.LENGTHS)
    if len(values) != expected:
        raise ValueError(
            f"expected {expected} checkpoint values, observed {len(values)}"
        )
    exact = audit["formal_256_exact_token_rows"]
    if (
        exact <= 0
        or audit["formal_256_exact_raw_rows"] != exact
        or audit["formal_256_exact_correctness_rows"] != exact
    ):
        raise ValueError(
            "frozen answer-operator reconstruction failed exact-token parity"
        )
    _OPERATOR_AUDIT[model_name] = audit
    return values


def _upgrade_manifest() -> dict[str, Any]:
    path = base.GENERATED_DIR / "block_length_sensitivity_manifest.json"
    old = base._read_json(path)
    old_record = str(old["record_sha256"])
    payload = {
        key: value
        for key, value in old.items()
        if key not in {"record_sha256", "schema_version"}
    }
    payload.update(
        {
            "schema_version": "flcva-block-length-paper-manifest-v6",
            "analysis_id": ANALYSIS_ID,
            "generation_study_id": base.STUDY_ID,
            "analysis_status": (
                "post-seal length robustness with frozen answer-operator "
                "reconstruction; analysis-only adaptive fit"
            ),
            "answer_operator": ANSWER_OPERATOR,
            "stop_labels": (
                "reused from the authenticated frozen answer-now evaluation; "
                "the answer-now action is unchanged across length arms"
            ),
            "operator_reconstruction_audit": _OPERATOR_AUDIT,
            "superseded_v4_analysis_record_sha256": old_record,
        }
    )
    manifest = seal_record(payload)
    base._atomic_json(path, manifest)
    return manifest


def evaluate(qwen_results: Path, deepseek_results: Path) -> None:
    # The sealed generation module remains byte-identical.  Only its
    # post-generation value construction is replaced for this analysis call.
    base._checkpoint_values = _checkpoint_values
    base._render_tables = _render_tables
    base.evaluate(qwen_results, deepseek_results)
    manifest = _upgrade_manifest()
    print(
        json.dumps(
            {
                "analysis_id": ANALYSIS_ID,
                "manifest": ("paper/generated/block_length_sensitivity_manifest.json"),
                "record_sha256": manifest["record_sha256"],
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-results", type=Path, required=True)
    parser.add_argument("--deepseek-results", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.qwen_results, args.deepseek_results)


if __name__ == "__main__":
    main()
