from __future__ import annotations

import os
import time
from typing import Any

from flcva.config import ModelSpec, SamplingSpec
from flcva.model_resolver import resolve_model_path, runtime_snapshot_identity

from .base import GeneratedSequence, GenerationBackend, GenerationRequest


class VLLMBackend(GenerationBackend):
    def __init__(self, model: ModelSpec):
        if model.tokenizer_revision != model.revision:
            raise RuntimeError(
                "vLLM generation requires tokenizer_revision to equal model revision"
            )
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        self.model_id = model.id
        self.runtime_model = resolve_model_path(
            model.id,
            model.revision,
            override_env="FLCVA_MODEL_PATH",
            uri_env="FLCVA_MODEL_URI",
        )
        self._runtime_identity = runtime_snapshot_identity(
            self.runtime_model,
            expected_revision=model.revision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.runtime_model,
            trust_remote_code=model.trust_remote_code,
        )
        self.llm = LLM(
            model=self.runtime_model,
            trust_remote_code=model.trust_remote_code,
            tensor_parallel_size=model.tensor_parallel_size,
            gpu_memory_utilization=model.gpu_memory_utilization,
            max_model_len=model.max_model_len,
            enable_prefix_caching=True,
        )
        self._runtime_identity.update(
            {
                "load_seconds": time.perf_counter() - load_started,
                "load_peak_gpu_memory_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
            }
        )

    @property
    def runtime_identity(self) -> dict[str, Any]:
        return self._runtime_identity

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def render_prompt(self, question: str, dataset_id: str) -> list[int]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the problem by reasoning step by step. Keep reasoning "
                    "until explicitly asked for a final answer, then put only the "
                    "final answer inside \\boxed{}."
                ),
            },
            {
                "role": "user",
                "content": f"[{dataset_id}] {question}",
            },
        ]
        kwargs: dict[str, Any] = {
            "conversation": messages,
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if "Qwen3" in self.model_id:
            kwargs["enable_thinking"] = True
        return list(self.tokenizer.apply_chat_template(**kwargs))

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
        return self.generate_batch(
            [
                GenerationRequest(
                    prompt_token_ids=prompt_token_ids,
                    max_tokens=max_tokens,
                    seed=seed,
                    purpose=purpose,
                    greedy=greedy,
                    top_logprobs=top_logprobs,
                    batch_capacity=1,
                )
            ],
            sampling=sampling,
        )[0]

    def generate_batch(
        self,
        requests: list[GenerationRequest],
        *,
        sampling: SamplingSpec,
    ) -> list[GeneratedSequence]:
        import torch
        from vllm import SamplingParams, TokensPrompt

        if not requests:
            return []
        capacities = {
            request.batch_capacity
            for request in requests
            if request.batch_capacity is not None
        }
        if len(capacities) > 1:
            raise ValueError("generation requests disagree on batch capacity")
        declared_capacity = next(iter(capacities), None)
        if declared_capacity is not None and declared_capacity < len(requests):
            raise ValueError("generation batch exceeds its declared capacity")
        prompts = [
            TokensPrompt(prompt_token_ids=request.prompt_token_ids)
            for request in requests
        ]
        params = [
            SamplingParams(
                max_tokens=request.max_tokens,
                temperature=0.0 if request.greedy else sampling.temperature,
                top_p=1.0 if request.greedy else sampling.top_p,
                repetition_penalty=sampling.repetition_penalty,
                seed=request.seed,
                ignore_eos=request.purpose != "answer",
                stop=(
                    ["}"]
                    if request.purpose == "answer"
                    else None
                ),
                include_stop_str_in_output=True,
                logprobs=request.top_logprobs or None,
            )
            for request in requests
        ]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        outputs = self.llm.generate(
            prompts,
            params,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - started
        if len(outputs) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(requests)} requests"
            )
        peak_memory = int(torch.cuda.max_memory_allocated())
        batch_size = len(requests)
        results: list[GeneratedSequence] = []
        for output in outputs:
            candidate = output.outputs[0]
            top_rows: list[dict[str, float]] = []
            selected_rows: list[float | None] = []
            raw_logprobs = candidate.logprobs or [
                {} for _ in candidate.token_ids
            ]
            for token_id, candidates in zip(candidate.token_ids, raw_logprobs):
                row = {
                    str(candidate_id): float(value.logprob)
                    for candidate_id, value in (candidates or {}).items()
                }
                top_rows.append(row)
                selected_rows.append(row.get(str(token_id)))
            results.append(
                GeneratedSequence(
                    token_ids=list(candidate.token_ids),
                    text=candidate.text,
                    token_logprobs=selected_rows,
                    top_logprobs=top_rows,
                    finish_reason=str(candidate.finish_reason),
                    gpu_seconds=elapsed / batch_size,
                    peak_gpu_memory_bytes=peak_memory,
                    batch_size=batch_size,
                    # Request-batch fill, not SM utilization.  The caller
                    # declares the intended batch capacity explicitly.
                    batch_utilization=(
                        batch_size / declared_capacity
                        if declared_capacity is not None
                        else None
                    ),
                )
            )
        return results


def build_backend(kind: str, model: ModelSpec) -> GenerationBackend:
    if kind == "mock":
        from .mock import MockBackend

        return MockBackend()
    if kind == "vllm":
        return VLLMBackend(model)
    raise ValueError(f"unsupported backend: {kind}")
