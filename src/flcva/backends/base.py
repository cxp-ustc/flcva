from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from flcva.config import SamplingSpec


@dataclass(frozen=True)
class GeneratedSequence:
    token_ids: list[int]
    text: str
    token_logprobs: list[float | None]
    top_logprobs: list[dict[str, float]]
    finish_reason: str
    gpu_seconds: float | None = None
    peak_gpu_memory_bytes: int | None = None
    batch_size: int = 1
    batch_utilization: float | None = None


@dataclass(frozen=True)
class GenerationRequest:
    prompt_token_ids: list[int]
    max_tokens: int
    seed: int
    purpose: str
    greedy: bool = False
    top_logprobs: int = 0
    batch_capacity: int | None = None


class GenerationBackend(ABC):
    @property
    def runtime_identity(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def render_prompt(self, question: str, dataset_id: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    def generate_batch(
        self,
        requests: list[GenerationRequest],
        *,
        sampling: SamplingSpec,
    ) -> list[GeneratedSequence]:
        """Compatibility batch; accelerated backends should override this method."""
        if not requests:
            return []
        batch_size = len(requests)
        capacities = {
            request.batch_capacity
            for request in requests
            if request.batch_capacity is not None
        }
        if len(capacities) > 1:
            raise ValueError("generation requests disagree on batch capacity")
        capacity = next(iter(capacities), None)
        if capacity is not None and capacity < batch_size:
            raise ValueError("generation batch exceeds its declared capacity")
        output = [
            self.generate(
                request.prompt_token_ids,
                max_tokens=request.max_tokens,
                sampling=sampling,
                seed=request.seed,
                purpose=request.purpose,
                greedy=request.greedy,
                top_logprobs=request.top_logprobs,
            )
            for request in requests
        ]
        return [
            replace(
                result,
                batch_size=batch_size,
                batch_utilization=(
                    batch_size / capacity if capacity is not None else None
                ),
            )
            for result in output
        ]
