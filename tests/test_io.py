from pathlib import Path

import pytest

from flcva.io import (
    BlobStore,
    atomic_write_json_gz,
    listed_artifacts,
    read_json_gz,
    require_artifacts,
)


def test_blob_store_roundtrip_and_tamper(tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "blobs")
    reference = store.put([1, 2, 3])
    assert store.get(reference) == [1, 2, 3]
    assert store.verify(reference)
    atomic_write_json_gz(store.root / reference, [9])
    with pytest.raises(ValueError, match="digest mismatch"):
        store.verify(reference)


def test_atomic_gzip_json(tmp_path: Path) -> None:
    path = tmp_path / "record.json.gz"
    atomic_write_json_gz(path, {"x": 1})
    assert read_json_gz(path) == {"x": 1}


def test_historical_artifact_paths_do_not_require_file_checksums(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.json").write_text('{"value": 1}\n')
    historical = {"artifacts_sha256": {"result.json": "obsolete-byte-checksum"}}
    assert require_artifacts(tmp_path, historical) == ["result.json"]
    (tmp_path / "result.json").write_text('{ "value" : 1 }\n')
    assert require_artifacts(tmp_path, historical) == ["result.json"]


def test_new_artifact_paths_are_validated(tmp_path: Path) -> None:
    assert listed_artifacts({"artifacts": ["missing.json"]}) == ["missing.json"]
    with pytest.raises(FileNotFoundError, match="missing artifact"):
        require_artifacts(tmp_path, {"artifacts": ["missing.json"]})
    with pytest.raises(ValueError, match="list of strings"):
        listed_artifacts({"artifacts": "result.json"})
