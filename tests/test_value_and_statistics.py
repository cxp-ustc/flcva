import pytest

from flcva.evaluation import marginal_accuracy_value
from flcva.replay import _replay_one, _score, step_auc
from flcva.schemas import ReplayPoint
from flcva.statistics import (
    assign_difficulty_tertiles,
    average_precision,
    brier_score,
    fit_tie_bins,
    roc_auc,
)


def test_marginal_value_sign_constraints() -> None:
    beneficial = marginal_accuracy_value(0, [1, 0, 1, 0], [256] * 4)
    harmful = marginal_accuracy_value(1, [1, 0, 1, 0], [256] * 4)
    neutral = marginal_accuracy_value(0, [0, 0, 0, 0], [256] * 4)
    assert beneficial > 0
    assert harmful < 0
    assert neutral == 0


def test_metrics_and_ties() -> None:
    assert roc_auc([0, 1], [0.1, 0.9]) == 1.0
    assert average_precision([0, 1], [0.1, 0.9]) == 1.0
    assert brier_score([0, 1], [0.1, 0.9]) == pytest.approx(0.01)
    bins = fit_tie_bins([0.0] * 5 + [1.0] * 5, max_bins=10)
    assert bins.assign(0.0) != bins.assign(1.0)


def test_average_precision_is_tie_aware_and_order_invariant() -> None:
    assert average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert average_precision([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert average_precision([1, 0, 1], [0.9, 0.5, 0.5]) == pytest.approx(
        5 / 6
    )


def test_difficulty_does_not_split_ties() -> None:
    scores = {"a": 0.0, "b": 0.0, "c": 0.5, "d": 1.0, "e": 1.0}
    labels = assign_difficulty_tertiles(scores)
    assert labels["a"] == labels["b"]
    assert labels["d"] == labels["e"]


def test_replay_auc_does_not_extrapolate_before_first_observation() -> None:
    points = [
        ReplayPoint(
            scheduler="round_robin",
            budget=100,
            accuracy=0.25,
            active_frontiers=1,
            terminal_frontiers=0,
        ),
        ReplayPoint(
            scheduler="round_robin",
            budget=200,
            accuracy=0.75,
            active_frontiers=0,
            terminal_frontiers=1,
        ),
    ]
    assert step_auc(points, 300) == pytest.approx(100 * 0.25 + 100 * 0.75)
    with pytest.raises(ValueError, match="before the first observable"):
        step_auc(points, 300, start_budget=0)


def test_trace_uses_available_confidence_when_persistence_is_missing() -> None:
    state = {
        "proxies": {
            "persistence": None,
            "answer_confidence": 0.8,
        }
    }
    assert _score(state, "trace") == pytest.approx(0.2)


def test_trace_first_action_is_confidence_ranked_not_id_serialized() -> None:
    def initial(problem: str, confidence: float, successor: str) -> dict:
        return {
            "checkpoint_id": f"{problem}-256",
            "problem_id": problem,
            "spine_id": 0,
            "depth": 256,
            "stop_correct": 0,
            "terminal": False,
            "successor_id": successor,
            "proxies": {
                "persistence": None,
                "answer_confidence": confidence,
            },
            "initial_ledger_events": [
                {
                    "event_id": f"{problem}-{role}",
                    "role": role,
                    "logical_tokens": 1,
                }
                for role in ("prompt", "spine_reasoning", "stop_probe")
            ],
            "transition_cost": 1,
            "transition_ledger_events": [
                {
                    "event_id": f"{problem}-transition",
                    "role": "continuation",
                    "logical_tokens": 1,
                }
            ],
        }

    states = [
        initial("a", 0.9, "a-512"),
        initial("b", 0.1, "b-512"),
        {
            "checkpoint_id": "a-512",
            "problem_id": "a",
            "spine_id": 0,
            "depth": 512,
            "stop_correct": 1,
            "terminal": True,
            "successor_id": None,
            "proxies": {},
        },
        {
            "checkpoint_id": "b-512",
            "problem_id": "b",
            "spine_id": 0,
            "depth": 512,
            "stop_correct": 0,
            "terminal": True,
            "successor_id": None,
            "proxies": {},
        },
    ]
    points = _replay_one(
        states,
        "trace",
        checkpoint_interval=256,
        fixed_allotment_tokens=1024,
    )
    # b has lower confidence and must be advanced first; its answer remains
    # wrong. ID-serialized scheduling would advance a and jump to 0.5 here.
    assert points[1].accuracy == 0.0
