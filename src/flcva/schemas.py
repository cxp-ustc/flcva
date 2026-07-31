from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentContract(StrictModel):
    schema_version: Literal[
        "flcva-experiment-contract-v1",
        "flcva-experiment-contract-v2",
    ] = (
        "flcva-experiment-contract-v2"
    )
    protocol_id: str
    run_id: str
    stage: str
    config_sha256: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    dataset_id: str
    dataset_revision: str
    dataset_components: dict[str, str]
    public_data_manifest_sha256: str | None = None
    public_data_sha256: str
    data_split: str
    scorer_id: str
    scorer_revision: str
    sampling: dict[str, float]
    prompt_sha256: str
    global_seed: int
    spines_per_problem: int
    checkpoint_interval: int
    reasoning_max_tokens: int
    continuation_branches: int
    continuation_tokens: int
    answer_max_tokens: int
    split_rule: str
    source_tree_sha256: str
    parent_contract_sha256: str | None = None
    frozen_campaign_sha256: str | None = None
    contract_sha256: str


class RunRecord(StrictModel):
    schema_version: Literal["flcva-run-record-v2"] = "flcva-run-record-v2"
    run_id: str
    protocol_id: str
    stage: str
    status: Literal["running", "complete", "failed", "sealed"]
    backend: str
    started_at: datetime
    completed_at: datetime | None = None
    host: str
    python: str
    packages: dict[str, str | None]
    hardware: dict[str, Any]
    runtime_environment: dict[str, Any]
    runtime_environment_sha256: str
    contract_sha256: str
    parent_run_id: str | None = None
    parent_run_ids: list[str] = Field(default_factory=list)
    task_id: str | None = None
    application_id: str | None = None
    task_scope: str | None = None
    parent_task_id: str | None = None
    parent_task_ids: list[str] = Field(default_factory=list)
    record_sha256: str


class LedgerEvent(StrictModel):
    schema_version: Literal["flcva-ledger-event-v1"] = "flcva-ledger-event-v1"
    event_id: str
    run_id: str
    problem_id: str
    checkpoint_id: str | None = None
    branch_id: str | None = None
    scope: Literal["problem", "spine", "checkpoint", "branch", "campaign"]
    role: Literal[
        "prompt",
        "spine_reasoning",
        "stop_probe",
        "continuation_reasoning",
        "continuation_probe",
        "prm_scoring",
        "proxy_scoring",
        "scheduler",
    ]
    input_tokens: int = 0
    output_tokens: int = 0
    logical_tokens: int = 0
    gpu_seconds: float | None = None
    peak_gpu_memory_bytes: int | None = None
    batch_size: int | None = None
    utilization: float | None = None
    causal_predecessor: str | None = None

    @model_validator(mode="after")
    def validate_costs(self) -> "LedgerEvent":
        if min(self.input_tokens, self.output_tokens, self.logical_tokens) < 0:
            raise ValueError("ledger token counts must be non-negative")
        if self.gpu_seconds is not None and self.gpu_seconds < 0:
            raise ValueError("gpu_seconds must be non-negative")
        return self


class CheckpointRecord(StrictModel):
    schema_version: Literal["flcva-checkpoint-v1"] = "flcva-checkpoint-v1"
    checkpoint_id: str
    run_id: str
    problem_id: str
    dataset_id: str
    model_id: str
    spine_id: int
    depth: int
    prompt_blob: str
    spine_blob: str
    prefix_token_count: int
    prefix_sha256: str
    continuation_eligible: bool
    successor_id: str | None = None
    stop_raw_answer: str
    stop_answer_tokens: list[int]
    stop_token_logprobs: list[float | None]
    stop_top_logprobs: list[dict[str, float]]
    stop_parser_status: str | None = None
    stop_canonical_answer: str | None = None
    stop_correct: int | None = None
    ledger_event_ids: list[str] = Field(default_factory=list)


class BranchRecord(StrictModel):
    schema_version: Literal["flcva-branch-v1"] = "flcva-branch-v1"
    branch_id: str
    checkpoint_id: str
    branch_index: int
    on_spine: bool
    seed: int
    prefix_sha256: str
    reasoning_blob: str
    reasoning_tokens: int
    raw_answer: str
    answer_tokens: list[int]
    answer_token_logprobs: list[float | None]
    answer_top_logprobs: list[dict[str, float]]
    finish_reason: str
    logical_tokens: int
    parser_status: str | None = None
    canonical_answer: str | None = None
    correct: int | None = None
    ledger_event_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def positive_cost(self) -> "BranchRecord":
        if self.logical_tokens <= 0:
            raise ValueError("branch logical_tokens must be positive")
        return self


class ProxyRecord(StrictModel):
    schema_version: Literal["flcva-proxy-v1"] = "flcva-proxy-v1"
    checkpoint_id: str
    proxy_name: Literal[
        "persistence",
        "answer_confidence",
        "modal_agreement",
        "prm_min_step",
        "anll",
        "top20_entropy",
    ]
    raw_value: float | None
    oriented_value: float | None
    missing_reason: str | None = None
    source_record_ids: list[str] = Field(default_factory=list)
    ledger_event_ids: list[str] = Field(default_factory=list)


class ValueRecord(StrictModel):
    schema_version: Literal["flcva-value-v1"] = "flcva-value-v1"
    checkpoint_id: str
    problem_id: str
    dataset_id: str
    model_id: str
    spine_id: int
    depth: int
    split: Literal["analysis", "confirmation"]
    difficulty: Literal["low", "middle", "high"]
    stop_correct: int
    continuation_correct: list[int]
    continuation_costs: list[int]
    marginal_value: float
    beneficial: int
    harmful: int
    proxies: dict[str, float | None]
    proxy_costs: dict[str, int] = Field(default_factory=dict)
    proxy_missing_reasons: dict[str, str | None]
    initial_stop_correct: int
    wrong_to_correct: bool
    correct_to_wrong: bool


class MatchedCell(StrictModel):
    schema_version: Literal["flcva-matched-cell-v1"] = "flcva-matched-cell-v1"
    cell_id: str
    proxy_name: str
    key: dict[str, str | int]
    nonzero_checkpoint_ids: list[str]
    zero_checkpoint_ids: list[str]
    proxy_min: float
    proxy_max: float
    problem_ids: list[str]


class ReplayPoint(StrictModel):
    schema_version: Literal["flcva-replay-point-v1"] = "flcva-replay-point-v1"
    scheduler: str
    budget: int
    accuracy: float
    active_frontiers: int
    terminal_frontiers: int


class ProblemArtifact(StrictModel):
    schema_version: Literal["flcva-problem-artifact-v1"] = (
        "flcva-problem-artifact-v1"
    )
    run_id: str
    contract_sha256: str
    problem_id: str
    dataset_id: str
    question: str
    checkpoints: list[CheckpointRecord]
    branches: list[BranchRecord]
    ledger: list[LedgerEvent]
    record_sha256: str
