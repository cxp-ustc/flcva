#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from flcva.config import RunConfig
from flcva.data import materialize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize public questions and sealed labels from a pinned dataset"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--public-out", type=Path, required=True)
    parser.add_argument("--labels-out", type=Path, required=True)
    args = parser.parse_args()
    config = RunConfig.load(args.config)
    counts = materialize_dataset(
        config.dataset,
        args.public_out,
        args.labels_out,
    )
    print(
        {
            "public_file": str(args.public_out),
            "labels_file": str(args.labels_out),
            "counts": counts,
        }
    )


if __name__ == "__main__":
    main()
