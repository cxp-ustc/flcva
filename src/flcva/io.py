from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from .ids import canonical_json_bytes, payload_sha256, sha256_bytes


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    _atomic_replace(path, encoded)


def atomic_write_json_gz(path: Path, payload: Any) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _atomic_replace(path, gzip.compress(encoded, mtime=0))


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_jsonl_gz(path: Path, records: Iterable[Any]) -> None:
    lines = b"".join(
        canonical_json_bytes(record) + b"\n" for record in records
    )
    _atomic_replace(path, gzip.compress(lines, mtime=0))


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} is not an object: {path}")
            yield payload


def file_sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BlobStore:
    """Content-addressed compressed JSON blobs."""

    def __init__(self, root: Path):
        self.root = root

    def put(self, payload: Any) -> str:
        encoded = canonical_json_bytes(payload)
        digest = sha256_bytes(encoded)
        relative = Path(digest[:2]) / f"{digest}.json.gz"
        target = self.root / relative
        if not target.exists():
            atomic_write_json_gz(target, payload)
        elif self.get(str(relative)) != payload:
            raise ValueError(f"blob collision or corruption: {target}")
        return str(relative)

    def get(self, reference: str) -> Any:
        return read_json_gz(self.root / reference)

    def verify(self, reference: str) -> str:
        payload = self.get(reference)
        observed = payload_sha256(payload)
        expected = Path(reference).stem.split(".")[0]
        if observed != expected:
            raise ValueError(f"blob digest mismatch: {reference}")
        return observed


def problem_artifact_path(run_dir: Path, problem_id: str) -> Path:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in problem_id)
    return run_dir / "problems" / f"{safe}.json.gz"


def iter_problem_artifacts(run_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted((run_dir / "problems").glob("*.json.gz")):
        payload = read_json_gz(path)
        if not verify_sealed_record(payload):
            raise ValueError(f"problem artifact hash mismatch: {path}")
        yield payload


def verify_sealed_record(
    payload: dict[str, Any],
    *,
    field: str = "record_sha256",
) -> bool:
    recorded = payload.get(field)
    unhashed = dict(payload)
    unhashed[field] = ""
    return isinstance(recorded, str) and recorded == payload_sha256(unhashed)


def seal_record(payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output["record_sha256"] = ""
    output["record_sha256"] = payload_sha256(output)
    return output
