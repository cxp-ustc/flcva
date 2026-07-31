from __future__ import annotations

from paper.analysis.render_supplement import (
    _discrete_metrics_from_counts,
    _problem_contributions,
    exact_nonmyopic_aulc,
    token_normalized_value,
)
from flcva.statistics import average_precision, brier_score, roc_auc


def test_token_normalized_oracle_values_use_selected_branch_costs() -> None:
    stop = 0
    correct = [1, 0, 1]
    costs = [250, 260, 270]
    expected = 1000.0 * (2.0 / 3.0) / 260.0
    assert token_normalized_value(stop, correct, costs) == expected


def test_discrete_count_metrics_match_tie_aware_reference() -> None:
    labels = [1, 0, 1, 0, 1, 0, 0]
    scores = [1.0, 1.0, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 0.0]
    counts = {
        0.0: (0.0, 1.0),
        1 / 3: (1.0, 1.0),
        2 / 3: (1.0, 1.0),
        1.0: (1.0, 1.0),
    }
    metrics = _discrete_metrics_from_counts(counts)
    assert metrics["auroc"] == roc_auc(labels, scores)
    assert metrics["auprc"] == average_precision(labels, scores)
    assert metrics["brier"] == brier_score(labels, scores)
    assert metrics["event_rate"] == sum(labels) / len(labels)


def test_exact_nonmyopic_dp_enumerates_all_four_chain_states() -> None:
    chains = []
    for chain_index in range(4):
        chains.append(
            [
                {
                    "stop_correct": int(chain_index == 0),
                    "transition_cost": 1,
                    "terminal": int(state_index == 15),
                }
                for state_index in range(16)
            ]
        )

    area, span, visited_states = exact_nonmyopic_aulc(chains)

    assert area == 15.0
    assert span == 60
    assert visited_states == 16**4


def test_problem_contributions_integrate_each_problem_before_averaging() -> None:
    rows = _problem_contributions(
        {"a": 0, "b": 1},
        {
            "a": [(4, 1)],
            "b": [(7, 0)],
        },
        start_budget=0,
        endpoint=10,
    )

    assert rows["a"] == {
        "normalized_aulc": 0.6,
        "final_correct": 1,
    }
    assert rows["b"] == {
        "normalized_aulc": 0.7,
        "final_correct": 0,
    }
