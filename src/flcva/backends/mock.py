from __future__ import annotations

import hashlib
import math
import random
from typing import Any

from flcva.config import SamplingSpec

from .base import GeneratedSequence, GenerationBackend


class MockBackend(GenerationBackend):
    """Deterministic, label-blind backend for engineering tests only."""

    @property
    def runtime_identity(self) -> dict[str, Any]:
        return {
            "backend": "mock",
            "snapshot_name": "label-blind-v1",
            "load_seconds": 0.0,
            "load_peak_gpu_memory_bytes": 0,
        }

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        values = bytes(token % 256 for token in token_ids)
        return values.decode("utf-8", errors="replace")

    def render_prompt(self, question: str, dataset_id: str) -> list[int]:
        return self.encode(
            "System: Solve step by step. Do not use an external judge.\n"
            f"Dataset: {dataset_id}\nQuestion: {question}\nAssistant: <think>\n"
        )

    def generate(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        sampling: SamplingSpec,
        seed: int,
        purpose: str,
        greedy: bool = False,
        top_logprobs: int = 0,
    ) -> GeneratedSequence:
        material = bytes(token % 256 for token in prompt_token_ids[-2048:])
        derived = int.from_bytes(
            hashlib.sha256(material + str(seed).encode() + purpose.encode()).digest()[:8],
            "big",
        )
        rng = random.Random(derived)
        if purpose == "answer":
            answer = str(1 + derived % 10) + "}"
            tokens = self.encode(answer)[:max_tokens]
            logprobs = [-0.05 - 0.02 * rng.random() for _ in tokens]
            top = []
            for selected in tokens:
                candidates = {
                    str(selected): logprobs[len(top)],
                    str((selected + 1) % 256): -1.2,
                    str((selected + 2) % 256): -2.0,
                }
                top.append(candidates if top_logprobs else {})
            return GeneratedSequence(
                token_ids=tokens,
                text=self.decode(tokens),
                token_logprobs=logprobs,
                top_logprobs=top,
                finish_reason="stop",
                gpu_seconds=0.0,
                peak_gpu_memory_bytes=0,
            )
        tokens = [rng.randrange(32, 127) for _ in range(max_tokens)]
        return GeneratedSequence(
            token_ids=tokens,
            text=self.decode(tokens),
            token_logprobs=[-math.log(95.0)] * len(tokens),
            top_logprobs=[{} for _ in tokens],
            finish_reason="length",
            gpu_seconds=0.0,
            peak_gpu_memory_bytes=0,
        )
