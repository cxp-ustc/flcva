from pathlib import Path

import pytest

from flcva.io import BlobStore, atomic_write_json_gz, read_json_gz


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

