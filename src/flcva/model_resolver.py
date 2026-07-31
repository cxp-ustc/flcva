from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .ids import payload_sha256
from .io import file_sha256


REVISION_MARKERS = (
    ".flcva_revision",
    "FLCVA_REVISION",
    "revision.txt",
)


def _local_revision_evidence(path: Path, revision: str) -> str | None:
    """Return deterministic evidence that ``path`` is the requested snapshot."""

    if revision in path.parts or path.name == revision:
        return "snapshot_path"
    for name in REVISION_MARKERS:
        marker = path / name
        if (
            marker.is_file()
            and marker.read_text(encoding="utf-8").strip().lower()
            == revision.lower()
        ):
            return name
    return None


def _validate_local_revision(path: Path, revision: str) -> str:
    evidence = _local_revision_evidence(path, revision)
    if evidence is not None:
        return evidence
    if os.environ.get("FLCVA_ALLOW_UNVERIFIED_LOCAL_SNAPSHOT") == "1":
        return "explicit_environment_override"
    raise RuntimeError(
        f"local snapshot {path} does not prove revision {revision}; use a "
        "Hugging Face snapshots/<commit> directory or add a .flcva_revision "
        "file containing the exact commit"
    )


def resolve_model_path(
    model_id: str,
    revision: str,
    *,
    override_env: str,
    uri_env: str | None = None,
) -> str:
    """Resolve an exactly pinned local or Hugging Face model snapshot.

    ``uri_env`` is accepted for compatibility with the experiment runner but
    is intentionally ignored in this anonymous, platform-neutral release.
    """

    del uri_env
    override = os.environ.get(override_env)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"{override_env} is not a model directory: {path}"
            )
        _validate_local_revision(path, revision)
        return str(path)

    direct = Path(model_id).expanduser()
    if direct.is_dir():
        path = direct.resolve()
        _validate_local_revision(path, revision)
        return str(path)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required to resolve remote model snapshots"
        ) from error

    path = Path(
        snapshot_download(repo_id=model_id, revision=revision)
    ).resolve()
    _validate_local_revision(path, revision)
    return str(path)


def runtime_snapshot_identity(
    path: str,
    *,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    model_path = Path(path).resolve()

    def digest(name: str) -> str | None:
        target = model_path / name
        return file_sha256(target) if target.is_file() else None

    revision_evidence = (
        _validate_local_revision(model_path, expected_revision)
        if expected_revision is not None
        else None
    )
    weight_files = sorted(
        file
        for pattern in ("*.safetensors", "*.bin", "*.pt")
        for file in model_path.glob(pattern)
        if file.is_file()
    )
    weight_layout = [
        {"name": file.name, "size_bytes": file.stat().st_size}
        for file in weight_files
    ]
    weight_indexes = {
        name: digest(name)
        for name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if (model_path / name).is_file()
    }
    config_path = model_path / "config.json"
    config_payload: dict[str, Any] | None = None
    if config_path.is_file():
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
            config_payload = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            config_payload = None
    return {
        "path_sha256": hashlib.sha256(
            str(model_path).encode("utf-8")
        ).hexdigest(),
        "expected_revision": expected_revision,
        "revision_evidence": revision_evidence,
        "config_sha256": digest("config.json"),
        "tokenizer_config_sha256": digest("tokenizer_config.json"),
        "generation_config_sha256": digest("generation_config.json"),
        "weight_layout_sha256": payload_sha256(weight_layout),
        "weight_index_sha256": payload_sha256(weight_indexes),
        "architectures": (
            config_payload.get("architectures")
            if config_payload is not None
            else None
        ),
        "model_type": (
            config_payload.get("model_type")
            if config_payload is not None
            else None
        ),
    }
