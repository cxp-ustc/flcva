#!/usr/bin/env python3
"""Fail if the review artifact contains common identity or credential traces."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_SUFFIXES = {".gz", ".png", ".pdf", ".pyc"}
PATTERNS = {
    "personal absolute path": re.compile(r"/Users/|[A-Za-z]:\\\\Users\\\\"),
    "email address": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
    ),
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "cloud access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
}


def main() -> int:
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() in SKIP_SUFFIXES
            or ".git" in path.parts
            or path == Path(__file__).resolve()
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{path.relative_to(ROOT)}: {label}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("anonymous artifact scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
