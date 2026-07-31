from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Any

from .backends.base import (
    GeneratedSequence,
    GenerationBackend,
    GenerationRequest,
)
from .backends.vllm import build_backend
from .config import RunConfig
from .contract import (
    bind_run_record,
    build_manifest,
    finish_run_record,
    materialize_contract,
    new_run_record,
    reseal_run_record,
    verify_run_record,
)
from .data import PublicProblem, load_public_problems
from .ids import payload_sha256, stable_id, stable_seed
from .io import (
    BlobStore,
    atomic_write_json,
    atomic_write_json_gz,
    problem_artifact_path,
    read_json_gz,
    seal_record,
    verify_sealed_record,
)
from .schemas import (
    BranchRecord,
    CheckpointRecord,
    LedgerEvent,
    ProblemArtifact,
    RunRecord,
)


def resolve_run_dir(config: RunConfig, project_root: Path) -> Path:
    override = os.environ.get("FLCVA_RESULT_ROOT")
    root = (
        Path(override).expanduser().resolve()
        if override
        else config.resolve_path(project_root, config.execution.result_root)
    )
    return root / config.run_id


def _event(
    config: RunConfig,
    problem: PublicProblem,
    *,
    role: str,
    scope: str,
    ordinal: str,
    checkpoint_id: str | None = None,
    branch_id: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    logical_tokens: int = 0,
    result: GeneratedSequence | None = None,
    predecessor: str | None = None,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=stable_id("evt", config.run_id, problem.id, ordinal),
        run_id=config.run_id,
        problem_id=problem.id,
        checkpoint_id=checkpoint_id,
        branch_id=branch_id,
        scope=scope,
        role=role,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        logical_tokens=logical_tokens,
        gpu_seconds=result.gpu_seconds if result else 0.0,
        peak_gpu_memory_bytes=result.peak_gpu_memory_bytes if result else 0,
        batch_size=result.batch_size if result else 1,
        utilization=result.batch_utilization if result else None,
        causal_predecessor=predecessor,
    )


def _generate_problem(
    config: RunConfig,
    contract_sha256: str,
    problem: PublicProblem,
    backend: GenerationBackend,
    blobs: BlobStore,
) -> ProblemArtifact:
    protocol = config.protocol
    prompt_tokens = backend.render_prompt(problem.question, problem.dataset_id)
    prompt_blob = blobs.put(prompt_tokens)
    checkpoints: list[CheckpointRecord] = []
    branches: list[BranchRecord] = []
    ledger: list[LedgerEvent] = []
    prompt_event = _event(
        config,
        problem,
        role="prompt",
        scope="problem",
        ordinal="prompt",
        input_tokens=len(prompt_tokens),
        logical_tokens=len(prompt_tokens),
    )
    ledger.append(prompt_event)
    cue_tokens = backend.encode(protocol.answer_cue)

    for spine_id in range(protocol.spines_per_problem):
        spine_seed = stable_seed(config.global_seed, problem.id, spine_id, "spine")
        spine = backend.generate(
            prompt_tokens,
            max_tokens=protocol.reasoning_max_tokens,
            sampling=config.sampling,
            seed=spine_seed,
            purpose="reasoning",
        )
        if len(spine.token_ids) != protocol.reasoning_max_tokens:
            raise RuntimeError(
                f"spine must contain exactly {protocol.reasoning_max_tokens} tokens"
            )
        spine_blob = blobs.put(spine.token_ids)
        spine_event = _event(
            config,
            problem,
            role="spine_reasoning",
            scope="spine",
            ordinal=f"spine:{spine_id}",
            input_tokens=len(prompt_tokens),
            output_tokens=len(spine.token_ids),
            logical_tokens=protocol.checkpoint_interval,
            result=spine,
            predecessor=prompt_event.event_id,
        )
        ledger.append(spine_event)
        checkpoint_ids = {
            depth: stable_id("ckpt", config.run_id, problem.id, spine_id, depth)
            for depth in range(
                protocol.checkpoint_interval,
                protocol.reasoning_max_tokens + 1,
                protocol.checkpoint_interval,
            )
        }
        stop_results: dict[int, GeneratedSequence] = {}
        stop_events: dict[int, LedgerEvent] = {}
        stop_jobs = [
            (
                depth,
                checkpoint_id,
                prompt_tokens + spine.token_ids[:depth],
                stable_seed(
                    config.global_seed,
                    problem.id,
                    spine_id,
                    depth,
                    "stop",
                ),
            )
            for depth, checkpoint_id in checkpoint_ids.items()
        ]
        stop_outputs = backend.generate_batch(
            [
                GenerationRequest(
                    prompt_token_ids=prefix + cue_tokens,
                    max_tokens=protocol.answer_max_tokens,
                    seed=stop_seed,
                    purpose="answer",
                    greedy=True,
                    top_logprobs=20,
                    batch_capacity=len(stop_jobs),
                )
                for _, _, prefix, stop_seed in stop_jobs
            ],
            sampling=config.sampling,
        )
        for (depth, checkpoint_id, prefix, _), stop in zip(
            stop_jobs,
            stop_outputs,
            strict=True,
        ):
            stop_results[depth] = stop
            stop_event = _event(
                config,
                problem,
                role="stop_probe",
                scope="checkpoint",
                ordinal=f"spine:{spine_id}:depth:{depth}:stop",
                checkpoint_id=checkpoint_id,
                input_tokens=len(prefix) + len(cue_tokens),
                output_tokens=len(stop.token_ids),
                logical_tokens=len(stop.token_ids),
                result=stop,
                predecessor=spine_event.event_id,
            )
            stop_events[depth] = stop_event
            ledger.append(stop_event)
            checkpoints.append(
                CheckpointRecord(
                    checkpoint_id=checkpoint_id,
                    run_id=config.run_id,
                    problem_id=problem.id,
                    dataset_id=problem.dataset_id,
                    model_id=config.model.id,
                    spine_id=spine_id,
                    depth=depth,
                    prompt_blob=prompt_blob,
                    spine_blob=spine_blob,
                    prefix_token_count=len(prefix),
                    prefix_sha256=payload_sha256(prefix),
                    continuation_eligible=depth < protocol.reasoning_max_tokens,
                    successor_id=checkpoint_ids.get(depth + protocol.checkpoint_interval),
                    stop_raw_answer=protocol.answer_cue + stop.text,
                    stop_answer_tokens=stop.token_ids,
                    stop_token_logprobs=stop.token_logprobs,
                    stop_top_logprobs=stop.top_logprobs,
                    ledger_event_ids=[stop_event.event_id],
                )
            )

        continuation_jobs: list[
            tuple[int, str, int, str, int, list[int]]
        ] = []
        for depth, checkpoint_id in checkpoint_ids.items():
            if depth >= protocol.reasoning_max_tokens:
                continue
            prefix = prompt_tokens + spine.token_ids[:depth]
            for branch_index in range(2, protocol.continuation_branches + 1):
                branch_id = stable_id(
                    "branch",
                    config.run_id,
                    problem.id,
                    spine_id,
                    depth,
                    branch_index,
                )
                seed = stable_seed(
                    config.global_seed,
                    checkpoint_id,
                    branch_index,
                    "continue",
                )
                continuation_jobs.append(
                    (
                        depth,
                        checkpoint_id,
                        branch_index,
                        branch_id,
                        seed,
                        prefix,
                    )
                )
        continuation_outputs = backend.generate_batch(
            [
                GenerationRequest(
                    prompt_token_ids=prefix,
                    max_tokens=protocol.continuation_tokens,
                    seed=seed,
                    purpose="continuation",
                    batch_capacity=len(continuation_jobs),
                )
                for _, _, _, _, seed, prefix in continuation_jobs
            ],
            sampling=config.sampling,
        )
        continuation_map = {
            (depth, branch_index): result
            for (
                depth,
                _,
                branch_index,
                _,
                _,
                _,
            ), result in zip(
                continuation_jobs,
                continuation_outputs,
                strict=True,
            )
        }
        for result in continuation_map.values():
            if len(result.token_ids) != protocol.continuation_tokens:
                raise RuntimeError("continuation did not reach the fixed token block")
        answer_jobs: list[tuple[int, int, list[int], int]] = []
        for depth, checkpoint_id, branch_index, _, _, prefix in continuation_jobs:
            continuation_tokens = continuation_map[
                (depth, branch_index)
            ].token_ids
            answer_seed = stable_seed(
                config.global_seed,
                checkpoint_id,
                branch_index,
                "answer",
            )
            answer_jobs.append(
                (
                    depth,
                    branch_index,
                    prefix + continuation_tokens + cue_tokens,
                    answer_seed,
                )
            )
        answer_outputs = backend.generate_batch(
            [
                GenerationRequest(
                    prompt_token_ids=answer_prompt,
                    max_tokens=protocol.answer_max_tokens,
                    seed=answer_seed,
                    purpose="answer",
                    greedy=True,
                    top_logprobs=20,
                    batch_capacity=len(answer_jobs),
                )
                for _, _, answer_prompt, answer_seed in answer_jobs
            ],
            sampling=config.sampling,
        )
        answer_map = {
            (depth, branch_index): result
            for (depth, branch_index, _, _), result in zip(
                answer_jobs,
                answer_outputs,
                strict=True,
            )
        }

        for depth, checkpoint_id in checkpoint_ids.items():
            if depth >= protocol.reasoning_max_tokens:
                continue
            prefix = prompt_tokens + spine.token_ids[:depth]
            for branch_index in range(1, protocol.continuation_branches + 1):
                branch_id = stable_id(
                    "branch",
                    config.run_id,
                    problem.id,
                    spine_id,
                    depth,
                    branch_index,
                )
                seed = stable_seed(
                    config.global_seed,
                    checkpoint_id,
                    branch_index,
                    "continue",
                )
                on_spine = branch_index == 1
                if on_spine:
                    continuation_tokens = spine.token_ids[
                        depth : depth + protocol.continuation_tokens
                    ]
                    continuation = None
                else:
                    continuation = continuation_map[(depth, branch_index)]
                    continuation_tokens = continuation.token_ids
                continuation_blob = blobs.put(continuation_tokens)
                continuation_event = _event(
                    config,
                    problem,
                    role="continuation_reasoning",
                    scope="branch",
                    ordinal=(
                        f"spine:{spine_id}:depth:{depth}:branch:{branch_index}:reasoning"
                    ),
                    checkpoint_id=checkpoint_id,
                    branch_id=branch_id,
                    input_tokens=len(prefix),
                    output_tokens=0 if on_spine else len(continuation_tokens),
                    logical_tokens=len(continuation_tokens),
                    result=continuation,
                    predecessor=spine_event.event_id,
                )
                ledger.append(continuation_event)
                if on_spine:
                    answer = stop_results[depth + protocol.continuation_tokens]
                    answer_event = stop_events[depth + protocol.continuation_tokens]
                else:
                    answer = answer_map[(depth, branch_index)]
                    answer_event = _event(
                        config,
                        problem,
                        role="continuation_probe",
                        scope="branch",
                        ordinal=(
                            f"spine:{spine_id}:depth:{depth}:branch:{branch_index}:answer"
                        ),
                        checkpoint_id=checkpoint_id,
                        branch_id=branch_id,
                        input_tokens=(
                            len(prefix) + len(continuation_tokens) + len(cue_tokens)
                        ),
                        output_tokens=len(answer.token_ids),
                        logical_tokens=len(answer.token_ids),
                        result=answer,
                        predecessor=continuation_event.event_id,
                    )
                    ledger.append(answer_event)
                branches.append(
                    BranchRecord(
                        branch_id=branch_id,
                        checkpoint_id=checkpoint_id,
                        branch_index=branch_index,
                        on_spine=on_spine,
                        seed=seed,
                        prefix_sha256=payload_sha256(prefix),
                        reasoning_blob=continuation_blob,
                        reasoning_tokens=len(continuation_tokens),
                        raw_answer=protocol.answer_cue + answer.text,
                        answer_tokens=answer.token_ids,
                        answer_token_logprobs=answer.token_logprobs,
                        answer_top_logprobs=answer.top_logprobs,
                        finish_reason=answer.finish_reason,
                        logical_tokens=len(continuation_tokens) + len(answer.token_ids),
                        ledger_event_ids=[
                            continuation_event.event_id,
                            answer_event.event_id,
                        ],
                    )
                )
    payload: dict[str, Any] = {
        "schema_version": "flcva-problem-artifact-v1",
        "run_id": config.run_id,
        "contract_sha256": contract_sha256,
        "problem_id": problem.id,
        "dataset_id": problem.dataset_id,
        "question": problem.question,
        "checkpoints": [
            checkpoint.model_dump(mode="json") for checkpoint in checkpoints
        ],
        "branches": [branch.model_dump(mode="json") for branch in branches],
        "ledger": [event.model_dump(mode="json") for event in ledger],
        "record_sha256": "",
    }
    return ProblemArtifact.model_validate(seal_record(payload))


def generate_run(
    config: RunConfig,
    project_root: Path,
    *,
    backend_factory: Callable[[str, Any], GenerationBackend] = build_backend,
) -> dict[str, Any]:
    run_dir = resolve_run_dir(config, project_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = materialize_contract(config, project_root, run_dir)
    problems = load_public_problems(config, project_root)
    run_path = run_dir / "run.json"
    existing_record = (
        RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
        if run_path.exists()
        else None
    )
    if existing_record is not None:
        verify_run_record(existing_record)
        existing_manifest_path = run_dir / "manifest.json"
        if not existing_manifest_path.is_file():
            raise ValueError("existing run record has no manifest")
        existing_manifest = json.loads(
            existing_manifest_path.read_text(encoding="utf-8")
        )
        if not verify_sealed_record(existing_manifest):
            raise ValueError("existing run manifest hash mismatch")
        if (
            existing_manifest.get("run_record_sha256")
            != existing_record.record_sha256
            or existing_manifest.get("runtime_environment_sha256")
            != existing_record.runtime_environment_sha256
        ):
            raise ValueError("existing manifest/run record binding mismatch")
    if (
        existing_record is not None
        and existing_record.contract_sha256 != contract.contract_sha256
    ):
        raise ValueError("run record contract mismatch")
    if existing_record is not None and existing_record.status == "sealed":
        for problem in problems:
            target = problem_artifact_path(run_dir, problem.id)
            if not target.is_file():
                raise ValueError(f"sealed run is missing a problem artifact: {target}")
            payload = read_json_gz(target)
            if not verify_sealed_record(payload):
                raise ValueError(f"sealed problem artifact hash mismatch: {target}")
            artifact = ProblemArtifact.model_validate(payload)
            if (
                artifact.contract_sha256 != contract.contract_sha256
                or artifact.problem_id != problem.id
            ):
                raise ValueError(f"sealed artifact identity mismatch: {target}")
        return {
            "schema_version": "flcva-generation-summary-v1",
            "run_id": config.run_id,
            "problems": len(problems),
            "completed_now": 0,
            "resumed": len(problems),
            "contract_sha256": contract.contract_sha256,
        }
    manifest = build_manifest(
        config,
        contract,
        project_root,
        [problem.id for problem in problems],
    )
    record = (
        reseal_run_record(
            existing_record,
            status="running",
            completed_at=None,
        )
        if existing_record is not None
        else new_run_record(config, contract)
    )
    atomic_write_json(run_path, record.model_dump(mode="json"))
    manifest = bind_run_record(manifest, record)
    atomic_write_json(run_dir / "manifest.json", manifest)
    backend = backend_factory(config.backend, config.model)
    manifest["runtime_model"] = backend.runtime_identity
    manifest = seal_record(manifest)
    atomic_write_json(run_dir / "manifest.json", manifest)
    blobs = BlobStore(run_dir / "blobs")
    completed = 0
    skipped = 0
    try:
        for problem in problems:
            target = problem_artifact_path(run_dir, problem.id)
            if target.exists():
                payload = read_json_gz(target)
                if not verify_sealed_record(payload):
                    raise ValueError(f"resume artifact hash mismatch: {target}")
                existing = ProblemArtifact.model_validate(payload)
                if (
                    existing.contract_sha256 != contract.contract_sha256
                    or existing.problem_id != problem.id
                ):
                    raise ValueError(f"resume artifact identity mismatch: {target}")
                skipped += 1
                continue
            artifact = _generate_problem(
                config,
                contract.contract_sha256,
                problem,
                backend,
                blobs,
            )
            atomic_write_json_gz(target, artifact.model_dump(mode="json"))
            completed += 1
    except Exception:
        failed_record = finish_run_record(record, "failed")
        atomic_write_json(
            run_dir / "run.json",
            failed_record.model_dump(mode="json"),
        )
        atomic_write_json(
            run_dir / "manifest.json",
            bind_run_record(manifest, failed_record),
        )
        raise
    record = finish_run_record(record, "complete")
    atomic_write_json(run_path, record.model_dump(mode="json"))
    manifest = bind_run_record(manifest, record)
    atomic_write_json(run_dir / "manifest.json", manifest)
    summary = {
        "schema_version": "flcva-generation-summary-v1",
        "run_id": config.run_id,
        "problems": len(problems),
        "completed_now": completed,
        "resumed": skipped,
        "contract_sha256": contract.contract_sha256,
    }
    atomic_write_json(run_dir / "generation_summary.json", summary)
    return summary
