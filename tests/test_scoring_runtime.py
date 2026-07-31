from __future__ import annotations

import sys
from types import SimpleNamespace

from flcva.config import ScorerSpec
from flcva.scoring import QwenPRMScorer, _forward_prm_logits


def test_single_h20_prm_load_does_not_require_accelerate(monkeypatch) -> None:
    observed_kwargs: dict[str, object] = {}

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert text == "<extra_0>"
            assert not add_special_tokens
            return [42]

    class FakeModel:
        device = "cuda:0"

        def __init__(self) -> None:
            self.target: str | None = None
            self.eval_called = False
            self.config = SimpleNamespace(use_cache=True)

        def to(self, target: str) -> "FakeModel":
            self.target = target
            return self

        def eval(self) -> "FakeModel":
            self.eval_called = True
            return self

    fake_model = FakeModel()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeModel:
            observed_kwargs.update(kwargs)
            return fake_model

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> FakeTokenizer:
            return FakeTokenizer()

    fake_cuda = SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 123,
    )
    fake_torch = SimpleNamespace(cuda=fake_cuda, bfloat16="bfloat16")
    fake_transformers = SimpleNamespace(
        AutoModel=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "flcva.scoring.resolve_model_path",
        lambda *args, **kwargs: "/frozen/prm",
    )
    monkeypatch.setattr(
        "flcva.scoring.runtime_snapshot_identity",
        lambda *args, **kwargs: {"expected_revision": "a" * 40},
    )

    scorer = QwenPRMScorer(
        ScorerSpec(
            backend="qwen_prm",
            id="Qwen/PRM",
            revision="a" * 40,
        )
    )

    assert "device_map" not in observed_kwargs
    assert observed_kwargs["torch_dtype"] == "bfloat16"
    assert fake_model.target == "cuda:0"
    assert fake_model.eval_called
    assert fake_model.config.use_cache is False
    assert scorer.separator_id == 42


def test_prm_forward_explicitly_disables_incompatible_kv_cache() -> None:
    observed: dict[str, object] = {}

    class FakeModel:
        def __call__(self, **kwargs: object) -> tuple[str]:
            observed.update(kwargs)
            return ("logits",)

    assert _forward_prm_logits(FakeModel(), {"input_ids": "tokens"}) == "logits"
    assert observed == {"input_ids": "tokens", "use_cache": False}
