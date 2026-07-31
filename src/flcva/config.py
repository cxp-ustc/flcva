from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(FrozenModel):
    id: str
    revision: str
    tokenizer_revision: str
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192


class DatasetSpec(FrozenModel):
    id: Literal["gsm8k", "math500", "combined", "smoke"]
    source: Literal["huggingface", "jsonl"]
    repo_id: str | None = None
    revision: str
    subset: str | None = None
    split: str = "test"
    public_path: str | None = None
    labels_path: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "DatasetSpec":
        if self.source == "huggingface" and not self.repo_id:
            raise ValueError("Hugging Face datasets require repo_id")
        if self.source == "jsonl" and not self.public_path:
            raise ValueError("JSONL datasets require public_path")
        return self


class SamplingSpec(FrozenModel):
    temperature: float = 0.6
    top_p: float = 0.95
    repetition_penalty: float = 1.0


class ProtocolSpec(FrozenModel):
    spines_per_problem: int = 4
    checkpoint_interval: int = 256
    reasoning_max_tokens: int = 4096
    continuation_branches: int = 4
    continuation_tokens: int = 256
    answer_max_tokens: int = 32
    answer_cue: str = "Therefore, the final answer is \\boxed{"

    @model_validator(mode="after")
    def validate_lattice(self) -> "ProtocolSpec":
        positive = (
            self.spines_per_problem,
            self.checkpoint_interval,
            self.reasoning_max_tokens,
            self.continuation_branches,
            self.continuation_tokens,
            self.answer_max_tokens,
        )
        if min(positive) <= 0:
            raise ValueError("protocol integer fields must be positive")
        if self.reasoning_max_tokens % self.checkpoint_interval:
            raise ValueError("reasoning_max_tokens must divide into checkpoints")
        if self.continuation_tokens != self.checkpoint_interval:
            raise ValueError("continuation_tokens must equal checkpoint_interval")
        if self.continuation_branches != 4:
            raise ValueError("frozen protocol requires exactly four branches")
        return self


class ScorerSpec(FrozenModel):
    backend: Literal["mock", "qwen_prm"]
    id: str
    revision: str
    batch_size: int = 4
    step_separator: str = "<extra_0>"
    system_prompt: str = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

    @model_validator(mode="after")
    def validate_batch_size(self) -> "ScorerSpec":
        if self.batch_size <= 0:
            raise ValueError("scorer batch_size must be positive")
        return self


class ExecutionSpec(FrozenModel):
    result_root: str
    shard_index: int = 0
    num_shards: int = 1
    limit: int | None = None
    bootstrap_replicates: int = 2000
    permutation_replicates: int = 1000
    fixed_allotment_tokens: int = 1024
    parent_contract: str | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> "ExecutionSpec":
        if self.num_shards <= 0 or not 0 <= self.shard_index < self.num_shards:
            raise ValueError("invalid shard assignment")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        if min(
            self.bootstrap_replicates,
            self.permutation_replicates,
            self.fixed_allotment_tokens,
        ) <= 0:
            raise ValueError("execution counts must be positive")
        return self


class RunConfig(FrozenModel):
    schema_version: Literal["flcva-config-v1"] = "flcva-config-v1"
    protocol_id: str
    run_id: str
    stage: Literal["mock", "smoke", "canary", "pilot", "formal"]
    backend: Literal["mock", "vllm"]
    global_seed: int
    model: ModelSpec
    dataset: DatasetSpec
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    protocol: ProtocolSpec = Field(default_factory=ProtocolSpec)
    scorer: ScorerSpec
    execution: ExecutionSpec

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        path = path.resolve()

        def read(current: Path, stack: tuple[Path, ...] = ()) -> dict:
            if current in stack:
                raise ValueError(f"cyclic config inheritance: {current}")
            with current.open(encoding="utf-8") as handle:
                value = yaml.safe_load(handle)
            if not isinstance(value, dict):
                raise ValueError("configuration must be a YAML mapping")
            parent_name = value.pop("extends", None)
            if parent_name is None:
                return value
            parent = Path(str(parent_name))
            if not parent.is_absolute():
                parent = (current.parent / parent).resolve()
            base = read(parent, stack + (current,))

            def merge(left: dict, right: dict) -> dict:
                output = dict(left)
                for key, item in right.items():
                    if (
                        key in output
                        and isinstance(output[key], dict)
                        and isinstance(item, dict)
                    ):
                        output[key] = merge(output[key], item)
                    else:
                        output[key] = item
                return output

            return merge(base, value)

        payload = read(path)
        if not isinstance(payload, dict):
            raise ValueError("configuration must be a YAML mapping")
        return cls.model_validate(payload)

    def resolve_path(self, project_root: Path, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path
