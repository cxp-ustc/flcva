from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit import audit_run
from .config import RunConfig
from .evaluation import evaluate_run
from .merge import merge_runs
from .preflight import run_preflight
from .replay import replay_run
from .runner import generate_run
from .scoring import score_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flcva")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "generate", "score", "evaluate", "replay", "audit"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "preflight":
            command.add_argument("--require-gpu", action="store_true")
        if name == "evaluate":
            command.add_argument(
                "--peer-run-dir",
                type=Path,
                action="append",
                default=[],
            )
    merge = subparsers.add_parser("merge")
    merge.add_argument("--config", type=Path, required=True)
    merge.add_argument("--input", type=Path, action="append", required=True)
    return parser


def _load(path: Path) -> tuple[RunConfig, Path]:
    resolved = path.resolve()
    project_root = Path(__file__).resolve().parents[2]
    return RunConfig.load(resolved), project_root


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config, project_root = _load(args.config)
    if args.command == "preflight":
        result = run_preflight(
            config,
            project_root,
            require_gpu=args.require_gpu,
        )
    elif args.command == "generate":
        result = generate_run(config, project_root)
    elif args.command == "score":
        result = score_run(config, project_root)
    elif args.command == "merge":
        result = merge_runs(
            config,
            project_root,
            [path.resolve() for path in args.input],
        )
    elif args.command == "evaluate":
        result = evaluate_run(
            config,
            project_root,
            peer_run_dirs=[path.resolve() for path in args.peer_run_dir],
        )
    elif args.command == "replay":
        result = replay_run(config, project_root)
    elif args.command == "audit":
        result = audit_run(config, project_root)
    else:
        raise AssertionError(args.command)
    _print(result)
