from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def stable_id(namespace: str, *parts: object, length: int = 24) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode())
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode())
    return f"{namespace}-{digest.hexdigest()[:length]}"


def stable_seed(global_seed: int, *parts: object) -> int:
    digest = hashlib.sha256(str(global_seed).encode())
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode())
    return int.from_bytes(digest.digest()[:8], "big") & 0x7FFF_FFFF


def split_bucket(dataset_id: str, problem_id: str, modulo: int = 100) -> int:
    digest = hashlib.sha256(f"{dataset_id}:{problem_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def without_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value.pop("record_sha256", None)
    return value

