from __future__ import annotations

import math

from paper.analysis.render_supplement import (
    _OneFeatureFit,
    _collision_rate_for_bins,
    _incremental_cross_entropy,
    _nearest_neighbor_disagreement,
    _proxy_bins,
    _transformer_chunk_flops,
)


ARCHITECTURE = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1000,
}


def _row(
    checkpoint: str,
    problem: str,
    split: str,
    value: float,
    label: int,
) -> dict:
    return {
        "checkpoint_id": checkpoint,
        "problem_id": problem,
        "dataset_id": "gsm8k",
        "difficulty": "low",
        "depth": 512,
        "stop_correct": 0,
        "beneficial": label,
        "harmful": 0,
        "split": split,
        "proxies": {
            "persistence": value,
            "modal_agreement": value,
        },
    }


def test_flops_increase_with_tokens_and_context() -> None:
    one = _transformer_chunk_flops(
        ARCHITECTURE,
        new_tokens=1,
        past_tokens=0,
    )
    two = _transformer_chunk_flops(
        ARCHITECTURE,
        new_tokens=2,
        past_tokens=0,
    )
    cached = _transformer_chunk_flops(
        ARCHITECTURE,
        new_tokens=1,
        past_tokens=100,
    )
    assert one > 0
    assert two > 2 * one
    assert cached > one


def test_aggregate_h20_rate_reconstructs_observed_total() -> None:
    fit = _OneFeatureFit()
    fit.add(1.0, 3.0)
    fit.add(4.0, 5.0)
    result = fit.solve()
    assert result["aggregate_pflops"] == 5.0
    assert result["aggregate_seconds"] == 8.0
    assert result["seconds_per_pflop_aggregate"] == 1.6


def test_discrete_proxy_matching_is_exact_at_every_resolution() -> None:
    analysis = [
        _row("a0", "a", "analysis", 0.0, 0),
        _row("a1", "b", "analysis", 0.5, 1),
        _row("a2", "c", "analysis", 1.0, 1),
    ]
    confirmation = [
        _row("c0", "d", "confirmation", 1.0, 0),
        _row("c1", "e", "confirmation", 1.0, 1),
        _row("c2", "f", "confirmation", 0.5, 0),
    ]
    low_resolution = _proxy_bins(analysis, "persistence", 5)
    high_resolution = _proxy_bins(analysis, "persistence", 50)
    assert low_resolution.upper_bounds == high_resolution.upper_bounds
    assert _collision_rate_for_bins(
        confirmation,
        proxy="persistence",
        bins=low_resolution,
    ) == _collision_rate_for_bins(
        confirmation,
        proxy="persistence",
        bins=high_resolution,
    )


def test_nearest_neighbor_excludes_same_problem() -> None:
    rows = [
        _row("a", "same", "confirmation", 0.0, 0),
        _row("b", "same", "confirmation", 0.0, 1),
        _row("c", "other", "confirmation", 1.0, 0),
    ]
    summary, _ = _nearest_neighbor_disagreement(
        rows,
        proxy="persistence",
        target="beneficial",
        orientation=1.0,
    )
    assert summary["nn_eligible"] == 3
    assert math.isclose(
        float(summary["nn_disagreement_rate"]),
        0.5,
    )


def test_incremental_information_is_fit_only_on_analysis() -> None:
    analysis = [
        _row(f"a{i}", f"a{i}", "analysis", float(i >= 4), int(i >= 4))
        for i in range(8)
    ]
    confirmation = [
        _row(f"c{i}", f"c{i}", "confirmation", float(i >= 4), int(i >= 4))
        for i in range(8)
    ]
    bins = _proxy_bins(analysis, "persistence", 10)
    summary, contributions = _incremental_cross_entropy(
        analysis,
        confirmation,
        proxy="persistence",
        target="beneficial",
        bins=bins,
    )
    assert summary["cross_entropy_reduction_bits"] > 0
    assert summary["augmented_fallback_rows"] == 0
    assert set(contributions) == {f"c{i}" for i in range(8)}
