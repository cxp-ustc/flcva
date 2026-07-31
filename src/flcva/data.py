from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import DatasetSpec, RunConfig
from .ids import split_bucket


@dataclass(frozen=True)
class PublicProblem:
    id: str
    dataset_id: str
    question: str


@dataclass(frozen=True)
class GoldLabel:
    id: str
    dataset_id: str
    answer: str


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} is not an object: {path}")
            yield payload


def _select_shard(
    records: Iterable[PublicProblem],
    config: RunConfig,
) -> list[PublicProblem]:
    selected = [
        item
        for item in records
        if split_bucket(item.dataset_id, item.id, config.execution.num_shards)
        == config.execution.shard_index
    ]
    selected.sort(key=lambda item: (item.dataset_id, item.id))
    if config.execution.limit is not None:
        selected = selected[: config.execution.limit]
    return selected


def _hf_rows(spec: DatasetSpec) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": spec.repo_id,
        "split": spec.split,
        "revision": spec.revision,
    }
    if spec.subset:
        kwargs["name"] = spec.subset
    return load_dataset(**kwargs)


def _row_id(spec: DatasetSpec, index: int, row: dict[str, Any]) -> str:
    native = row.get("id") or row.get("problem_id")
    return str(native) if native is not None else f"{spec.id}:{spec.split}:{index:04d}"


def _question(spec: DatasetSpec, row: dict[str, Any]) -> str:
    for field in ("question", "problem"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"dataset row has no question/problem field: {spec.id}")


def _answer(spec: DatasetSpec, row: dict[str, Any]) -> str:
    value = row.get("answer") or row.get("solution")
    if not isinstance(value, str):
        raise ValueError(f"dataset row has no answer/solution field: {spec.id}")
    return value


def load_public_problems(config: RunConfig, project_root: Path) -> list[PublicProblem]:
    spec = config.dataset
    if spec.source == "jsonl":
        path = config.resolve_path(project_root, str(spec.public_path))
        records = (
            PublicProblem(
                id=str(row["id"]),
                dataset_id=str(row.get("dataset_id") or spec.id),
                question=str(row["question"]),
            )
            for row in _jsonl(path)
        )
    else:
        records = (
            PublicProblem(
                id=_row_id(spec, index, dict(row)),
                dataset_id=spec.id,
                question=_question(spec, dict(row)),
            )
            for index, row in enumerate(_hf_rows(spec))
        )
    output = _select_shard(records, config)
    if len({item.id for item in output}) != len(output):
        raise ValueError("duplicate public problem ids")
    return output


def load_gold_labels(config: RunConfig, project_root: Path) -> dict[str, GoldLabel]:
    spec = config.dataset
    if spec.labels_path:
        path = config.resolve_path(project_root, spec.labels_path)
        labels = [
            GoldLabel(
                id=str(row["id"]),
                dataset_id=str(row.get("dataset_id") or spec.id),
                answer=str(row["answer"]),
            )
            for row in _jsonl(path)
        ]
    elif spec.source == "huggingface":
        labels = [
            GoldLabel(
                id=_row_id(spec, index, dict(row)),
                dataset_id=spec.id,
                answer=_answer(spec, dict(row)),
            )
            for index, row in enumerate(_hf_rows(spec))
        ]
    else:
        raise ValueError("labels_path is required for JSONL evaluation")
    all_labels = {label.id: label for label in labels}
    if len(all_labels) != len(labels):
        raise ValueError("duplicate gold label ids")
    selected_ids = {item.id for item in load_public_problems(config, project_root)}
    if selected_ids != set(all_labels) & selected_ids:
        raise ValueError("gold labels are missing selected public problem ids")
    return {key: all_labels[key] for key in sorted(selected_ids)}


def materialize_dataset(
    spec: DatasetSpec,
    public_path: Path,
    labels_path: Path,
) -> dict[str, int]:
    if spec.source != "huggingface":
        raise ValueError("only Hugging Face datasets can be materialized")
    public_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(_hf_rows(spec))
    public_lines: list[str] = []
    label_lines: list[str] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        problem_id = _row_id(spec, index, row)
        public_lines.append(
            json.dumps(
                {
                    "id": problem_id,
                    "dataset_id": spec.id,
                    "question": _question(spec, row),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        label_lines.append(
            json.dumps(
                {
                    "id": problem_id,
                    "dataset_id": spec.id,
                    "answer": _answer(spec, row),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    public_path.write_text("\n".join(public_lines) + "\n", encoding="utf-8")
    labels_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return {"problems": len(rows), "labels": len(rows)}

