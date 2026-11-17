import gzip
import json
import shutil
from pathlib import Path

from flcva.audit import audit_run
from flcva.config import RunConfig
from flcva.contract import (
    build_contract,
    materialize_contract,
    verify_recorded_contract,
)
from flcva.frozen import validate_formal_config
from flcva.ids import payload_sha256
from flcva.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]


def test_documentation_and_json_format_do_not_change_contract(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "data", tmp_path / "data")
    config = RunConfig.load(tmp_path / "configs" / "formal-qwen3.yaml")
    config = config.model_copy(update={"stage": "formal"})
    before = build_contract(config, tmp_path)
    (tmp_path / "README.md").write_text("Updated public documentation.\n")
    public = tmp_path / config.dataset.public_path
    rows = [json.loads(line) for line in public.read_text().splitlines()]
    public.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n"
    )
    assert build_contract(config, tmp_path) == before
    assert validate_formal_config(config, tmp_path) == before.frozen_campaign_sha256


def test_recompressing_derived_output_preserves_audit(
    mock_pipeline, tmp_path, monkeypatch
):
    config = mock_pipeline["config"]
    monkeypatch.setenv("FLCVA_RESULT_ROOT", str(tmp_path))
    copied = tmp_path / config.run_id
    shutil.copytree(mock_pipeline["run_dir"], copied)
    points = copied / "replay" / "points.jsonl.gz"
    original = points.read_bytes()
    decoded = gzip.decompress(original)
    points.write_bytes(gzip.compress(decoded, mtime=12345))
    assert points.read_bytes() != original
    assert gzip.decompress(points.read_bytes()) == decoded
    assert audit_run(config, ROOT)["passed"]


def test_historical_contract_reuses_its_record_id(tmp_path):
    config = RunConfig.load(ROOT / "configs" / "mock.yaml")
    historical = build_contract(config, ROOT).model_dump(mode="json")
    historical["source_tree_sha256"] = "old-source-checksum"
    historical["public_data_sha256"] = "old-data-checksum"
    historical["public_data_manifest_sha256"] = "old-manifest-checksum"
    historical["contract_sha256"] = ""
    historical["contract_sha256"] = payload_sha256(historical)
    atomic_write_json(tmp_path / "contract.json", historical)
    assert (
        materialize_contract(config, ROOT, tmp_path).contract_sha256
        == historical["contract_sha256"]
    )
    assert (
        verify_recorded_contract(config, ROOT, tmp_path).contract_sha256
        == historical["contract_sha256"]
    )
