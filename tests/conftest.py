from __future__ import annotations

from pathlib import Path

import pytest

from flcva.audit import audit_run
from flcva.config import RunConfig
from flcva.evaluation import evaluate_run
from flcva.replay import replay_run
from flcva.runner import generate_run, resolve_run_dir
from flcva.scoring import score_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def mock_pipeline(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("flcva-mock")
    config = RunConfig.load(PROJECT_ROOT / "configs" / "mock.yaml")
    config = config.model_copy(
        update={
            "run_id": "flcva-test-mock-v3",
            "execution": config.execution.model_copy(
                update={"result_root": str(root)}
            ),
        }
    )
    generation = generate_run(config, PROJECT_ROOT)
    scoring = score_run(config, PROJECT_ROOT)
    evaluation = evaluate_run(config, PROJECT_ROOT)
    replay = replay_run(config, PROJECT_ROOT)
    audit = audit_run(config, PROJECT_ROOT)
    return {
        "config": config,
        "run_dir": resolve_run_dir(config, PROJECT_ROOT),
        "generation": generation,
        "scoring": scoring,
        "evaluation": evaluation,
        "replay": replay,
        "audit": audit,
    }
