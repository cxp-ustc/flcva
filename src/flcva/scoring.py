from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import RunConfig, ScorerSpec
from .contract import verify_recorded_contract
from .ids import stable_id
from .io import (
    BlobStore,
    atomic_write_json,
    atomic_write_json_gz,
    iter_problem_artifacts,
    read_json_gz,
    seal_record,
    verify_sealed_record,
)
from .model_resolver import resolve_model_path, runtime_snapshot_identity
from .runner import resolve_run_dir
from .schemas import LedgerEvent, ProxyRecord


@dataclass(frozen=True)
class PRMResult:
    score: float | None
    input_tokens: int
    step_scores: list[float]
    elapsed_seconds: float
    peak_gpu_memory_bytes: int = 0
    missing_reason: str | None = None


class PRMScorer:
    def score_many(
        self,
        items: list[tuple[str, str]],
    ) -> list[PRMResult]:
        raise NotImplementedError


class MockPRMScorer(PRMScorer):
    def score_many(self, items: list[tuple[str, str]]) -> list[PRMResult]:
        output: list[PRMResult] = []
        for question, response in items:
            digest = hashlib.sha256((question + "\x00" + response).encode()).digest()
            value = int.from_bytes(digest[:4], "big") / 0xFFFF_FFFF
            steps = [part for part in response.split("\n\n") if part.strip()]
            output.append(
                PRMResult(
                    score=value,
                    input_tokens=len(question.encode()) + len(response.encode()),
                    step_scores=[value for _ in steps] or [value],
                    elapsed_seconds=0.0,
                )
            )
        return output


def _forward_prm_logits(model, batch):
    # The frozen Qwen PRM is a one-shot token classifier, so a KV cache has no
    # value here.  More importantly, its frozen remote code targets the older
    # Transformers Cache API while the pytorch280 image carries Transformers
    # 4.55.  Passing this explicitly prevents that code from constructing an
    # incompatible DynamicCache.
    return model(**batch, use_cache=False)[0]


class QwenPRMScorer(PRMScorer):
    def __init__(self, spec: ScorerSpec):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        self.torch = torch
        self.spec = spec
        self.runtime_model = resolve_model_path(
            spec.id,
            spec.revision,
            override_env="FLCVA_SCORER_PATH",
            uri_env="FLCVA_SCORER_URI",
        )
        self.runtime_identity = runtime_snapshot_identity(
            self.runtime_model,
            expected_revision=spec.revision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.runtime_model,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.runtime_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda:0").eval()
        self.model.config.use_cache = False
        self.runtime_identity.update(
            {
                "load_seconds": time.perf_counter() - load_started,
                "load_peak_gpu_memory_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
            }
        )
        encoded_separator = self.tokenizer.encode(
            spec.step_separator,
            add_special_tokens=False,
        )
        if len(encoded_separator) != 1:
            raise ValueError("Qwen PRM step separator must encode to one token")
        self.separator_id = encoded_separator[0]

    def _conversation(self, question: str, response: str) -> str:
        steps = [part.strip() for part in response.split("\n\n") if part.strip()]
        if not steps:
            steps = [response.strip()]
        messages = [
            {"role": "system", "content": self.spec.system_prompt},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": self.spec.step_separator.join(steps)
                + self.spec.step_separator,
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def score_many(self, items: list[tuple[str, str]]) -> list[PRMResult]:
        torch = self.torch
        conversations = [
            self._conversation(question, response)
            for question, response in items
        ]
        batch = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(self.model.device)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            logits = _forward_prm_logits(self.model, batch)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        elapsed = time.perf_counter() - started
        peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
        masks = batch["input_ids"] == self.separator_id
        results: list[PRMResult] = []
        per_item_elapsed = elapsed / max(1, len(items))
        for index in range(len(items)):
            step_logits = probabilities[index][masks[index]]
            scores = (
                step_logits[:, 1].float().cpu().tolist()
                if step_logits.numel()
                else []
            )
            input_tokens = int(batch["attention_mask"][index].sum().item())
            results.append(
                PRMResult(
                    score=min(scores) if scores else None,
                    input_tokens=input_tokens,
                    step_scores=scores,
                    elapsed_seconds=per_item_elapsed,
                    peak_gpu_memory_bytes=peak_gpu_memory_bytes,
                    missing_reason=None if scores else "no_prm_step_separator",
                )
            )
        return results


def build_scorer(spec: ScorerSpec) -> PRMScorer:
    if spec.backend == "mock":
        return MockPRMScorer()
    if spec.backend == "qwen_prm":
        return QwenPRMScorer(spec)
    raise ValueError(f"unsupported scorer backend: {spec.backend}")


def _chunks(values: list, size: int) -> Iterable[list]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def score_run(config: RunConfig, project_root: Path) -> dict[str, int | str]:
    run_dir = resolve_run_dir(config, project_root)
    verify_recorded_contract(config, project_root, run_dir)
    scorer = build_scorer(config.scorer)
    if config.backend == "mock":
        decode = lambda token_ids: "".join(chr(token % 256) for token in token_ids)
    else:
        from transformers import AutoTokenizer

        tokenizer_override_env = (
            "FLCVA_TOKENIZER_PATH"
            if os.environ.get("FLCVA_TOKENIZER_PATH")
            else "FLCVA_MODEL_PATH"
        )
        tokenizer_source = resolve_model_path(
            config.model.id,
            config.model.tokenizer_revision,
            override_env=tokenizer_override_env,
            uri_env="FLCVA_MODEL_URI",
        )
        generation_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=config.model.trust_remote_code,
        )
        decode = lambda token_ids: generation_tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )
    blobs = BlobStore(run_dir / "blobs")
    score_dir = run_dir / "scores"
    completed = 0
    skipped = 0
    for artifact in iter_problem_artifacts(run_dir):
        target = score_dir / f"{artifact['problem_id'].replace(':', '_')}.json.gz"
        if target.exists():
            payload = read_json_gz(target)
            if not verify_sealed_record(payload):
                raise ValueError(f"score resume record hash mismatch: {target}")
            if payload.get("contract_sha256") != artifact["contract_sha256"]:
                raise ValueError(f"score resume contract mismatch: {target}")
            skipped += 1
            continue
        checkpoints = artifact["checkpoints"]
        work: list[tuple[dict, tuple[str, str]]] = []
        for checkpoint in checkpoints:
            spine_tokens = blobs.get(checkpoint["spine_blob"])
            response_tokens = spine_tokens[: checkpoint["depth"]]
            response = decode(response_tokens)
            work.append(
                (
                    checkpoint,
                    (artifact["question"], response),
                )
            )
        proxies: list[dict] = []
        ledger: list[dict] = []
        for chunk in _chunks(work, config.scorer.batch_size):
            results = scorer.score_many([item[1] for item in chunk])
            for (checkpoint, _), result in zip(chunk, results):
                event_id = stable_id(
                    "evt",
                    config.run_id,
                    checkpoint["checkpoint_id"],
                    "prm",
                )
                event = LedgerEvent(
                    event_id=event_id,
                    run_id=config.run_id,
                    problem_id=artifact["problem_id"],
                    checkpoint_id=checkpoint["checkpoint_id"],
                    scope="checkpoint",
                    role="prm_scoring",
                    input_tokens=result.input_tokens,
                    output_tokens=0,
                    logical_tokens=result.input_tokens,
                    gpu_seconds=result.elapsed_seconds,
                    peak_gpu_memory_bytes=result.peak_gpu_memory_bytes,
                    batch_size=len(chunk),
                    utilization=len(chunk) / config.scorer.batch_size,
                )
                proxy = ProxyRecord(
                    checkpoint_id=checkpoint["checkpoint_id"],
                    proxy_name="prm_min_step",
                    raw_value=result.score,
                    oriented_value=result.score,
                    missing_reason=result.missing_reason,
                    source_record_ids=[checkpoint["checkpoint_id"]],
                    ledger_event_ids=[event_id],
                )
                proxies.append(proxy.model_dump(mode="json"))
                ledger.append(event.model_dump(mode="json"))
        payload = seal_record(
            {
                "schema_version": "flcva-score-artifact-v1",
                "run_id": config.run_id,
                "contract_sha256": artifact["contract_sha256"],
                "problem_id": artifact["problem_id"],
                "scorer_id": config.scorer.id,
                "scorer_revision": config.scorer.revision,
                "runtime_scorer": getattr(
                    scorer,
                    "runtime_identity",
                    {"backend": "mock"},
                ),
                "proxies": proxies,
                "ledger": ledger,
            }
        )
        atomic_write_json_gz(target, payload)
        completed += 1
    summary = {
        "schema_version": "flcva-score-summary-v1",
        "run_id": config.run_id,
        "completed_now": completed,
        "resumed": skipped,
        "runtime_scorer": getattr(
            scorer,
            "runtime_identity",
            {
                "backend": "mock",
                "load_seconds": 0.0,
                "load_peak_gpu_memory_bytes": 0,
            },
        ),
    }
    atomic_write_json(run_dir / "score_summary.json", summary)
    return summary
