#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from flcva.config import DatasetSpec
from flcva.data import materialize_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    public_root = ROOT / "data" / "frozen"
    label_root = ROOT / "data" / "sealed"
    specs = (
        DatasetSpec(
            id="gsm8k",
            source="huggingface",
            repo_id="openai/gsm8k",
            revision="740312add88f781978c0658806c59bc2815b9866",
            subset="main",
            split="test",
        ),
        DatasetSpec(
            id="math500",
            source="huggingface",
            repo_id="HuggingFaceH4/MATH-500",
            revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
            split="test",
        ),
    )
    outputs: list[dict] = []
    for spec in specs:
        public = public_root / f"{spec.id}_public.jsonl"
        labels = label_root / f"{spec.id}_labels.jsonl"
        counts = materialize_dataset(spec, public, labels)
        outputs.append(
            {
                "dataset": spec.id,
                "public_file": str(public),
                "labels_file": str(labels),
                "counts": counts,
            }
        )
    public_lines: list[str] = []
    label_lines: list[str] = []
    for spec in specs:
        public_lines.extend(
            (public_root / f"{spec.id}_public.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        label_lines.extend(
            (label_root / f"{spec.id}_labels.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    combined_public = public_root / "combined_public.jsonl"
    combined_labels = label_root / "combined_labels.jsonl"
    combined_public.write_text("\n".join(public_lines) + "\n", encoding="utf-8")
    combined_labels.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "datasets": outputs,
                "combined": {
                    "public_file": str(combined_public),
                    "labels_file": str(combined_labels),
                    "problems": len(public_lines),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
