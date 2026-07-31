from __future__ import annotations

import os
import re
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import RunConfig
from .contract import (
    build_contract,
    hardware_snapshot,
    runtime_environment_snapshot,
)
from .data import load_public_problems
from .frozen import validate_formal_config
from .ids import payload_sha256
from .io import atomic_write_json, seal_record
from .model_resolver import resolve_model_path, runtime_snapshot_identity
from .runner import resolve_run_dir


IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
VLLM_0_11_SERIES = re.compile(r"^0\.11(?:\.|$)")


def is_vllm_0_11_series(version: str) -> bool:
    """Accept the frozen vLLM 0.11 API series while recording the exact build."""
    return VLLM_0_11_SERIES.match(version) is not None


def _release_tuple(version: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", version)
    if match is None:
        raise RuntimeError(f"cannot parse package version: {version}")
    return tuple(int(part) for part in match.group(1).split("."))


def _require_package_version(
    name: str,
    *,
    minimum: tuple[int, ...] | None = None,
    maximum: tuple[int, ...] | None = None,
    exact: str | None = None,
) -> str:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required runtime package is absent: {name}") from error
    release = _release_tuple(version)
    if exact is not None and version != exact:
        raise RuntimeError(f"{name} must be {exact}, observed {version}")
    if minimum is not None and release < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, observed {version}")
    if maximum is not None and release >= maximum:
        raise RuntimeError(f"{name} must be < {maximum}, observed {version}")
    return version


def _render_prompt_tokens(tokenizer: Any, config: RunConfig, question: str) -> list[int]:
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
            "content": f"[{config.dataset.id}] {question}",
        },
    ]
    kwargs: dict[str, Any] = {
        "conversation": messages,
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if "Qwen3" in config.model.id:
        kwargs["enable_thinking"] = True
    return list(tokenizer.apply_chat_template(**kwargs))


def _gpu_model_preflight(
    config: RunConfig,
    problems: list[Any],
) -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise RuntimeError("frozen runtime requires Python 3.10 or newer")
    torch_version = _require_package_version(
        "torch",
        minimum=(2, 8),
        maximum=(2, 9),
    )
    vllm_version = _require_package_version(
        "vllm",
        minimum=(0, 11),
        maximum=(0, 12),
    )
    if not is_vllm_0_11_series(vllm_version):
        raise RuntimeError(
            "frozen runtime requires the vLLM 0.11 series, "
            f"observed {vllm_version}"
        )
    package_versions = {
        "torch": torch_version,
        "vllm": vllm_version,
        "transformers": _require_package_version(
            "transformers",
            minimum=(4, 55),
            maximum=(5,),
        ),
        "numpy": _require_package_version(
            "numpy",
            minimum=(1, 26),
            maximum=(3,),
        ),
        "huggingface-hub": _require_package_version(
            "huggingface-hub",
            minimum=(0, 34),
        ),
    }
    if config.model.tokenizer_revision != config.model.revision:
        raise RuntimeError(
            "vLLM generation requires tokenizer_revision to equal model revision"
        )
    model_path = resolve_model_path(
        config.model.id,
        config.model.revision,
        override_env="FLCVA_MODEL_PATH",
        uri_env="FLCVA_MODEL_URI",
    )
    scorer_path = resolve_model_path(
        config.scorer.id,
        config.scorer.revision,
        override_env="FLCVA_SCORER_PATH",
        uri_env="FLCVA_SCORER_URI",
    )
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt

    model_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    scorer_tokenizer = AutoTokenizer.from_pretrained(
        scorer_path,
        trust_remote_code=True,
    )
    longest_prompt = max(
        len(_render_prompt_tokens(model_tokenizer, config, problem.question))
        for problem in problems
    )
    answer_cue_tokens = len(
        model_tokenizer.encode(
            config.protocol.answer_cue,
            add_special_tokens=False,
        )
    )
    required_context = (
        longest_prompt
        + config.protocol.reasoning_max_tokens
        + answer_cue_tokens
        + config.protocol.answer_max_tokens
    )
    if required_context > config.model.max_model_len:
        raise RuntimeError(
            "frozen prompt can exceed max_model_len: "
            f"required={required_context}, configured={config.model.max_model_len}"
        )
    separator_ids = scorer_tokenizer.encode(
        config.scorer.step_separator,
        add_special_tokens=False,
    )
    if len(separator_ids) != 1:
        raise RuntimeError(
            "Qwen PRM step separator must encode to exactly one token"
        )
    TokensPrompt(prompt_token_ids=[int(model_tokenizer.eos_token_id or 0)])
    SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=config.sampling.repetition_penalty,
        seed=config.global_seed,
        ignore_eos=True,
        stop=["}"],
        include_stop_str_in_output=True,
        logprobs=20,
    )
    return {
        "packages": package_versions,
        "model": runtime_snapshot_identity(
            model_path,
            expected_revision=config.model.revision,
        ),
        "scorer": runtime_snapshot_identity(
            scorer_path,
            expected_revision=config.scorer.revision,
        ),
        "longest_prompt_tokens": longest_prompt,
        "required_context_tokens": required_context,
        "prm_step_separator_token_id": int(separator_ids[0]),
        "vllm_api_constructed": True,
    }


def expected_scale(config: RunConfig, problems: int) -> dict[str, int]:
    protocol = config.protocol
    checkpoints_per_spine = (
        protocol.reasoning_max_tokens // protocol.checkpoint_interval
    )
    eligible_per_spine = checkpoints_per_spine - 1
    spines = problems * protocol.spines_per_problem
    checkpoints = spines * checkpoints_per_spine
    eligible = spines * eligible_per_spine
    branches = eligible * protocol.continuation_branches
    base_reasoning = spines * protocol.reasoning_max_tokens
    stop_answer_max = checkpoints * protocol.answer_max_tokens
    off_spine_reasoning = (
        eligible
        * (protocol.continuation_branches - 1)
        * protocol.continuation_tokens
    )
    off_spine_answer_max = (
        eligible
        * (protocol.continuation_branches - 1)
        * protocol.answer_max_tokens
    )
    return {
        "problems": problems,
        "spines": spines,
        "checkpoints": checkpoints,
        "eligible_checkpoints": eligible,
        "branch_references": branches,
        "base_reasoning_tokens": base_reasoning,
        "stop_answer_tokens_max": stop_answer_max,
        "off_spine_reasoning_tokens": off_spine_reasoning,
        "off_spine_answer_tokens_max": off_spine_answer_max,
        "generated_tokens_max": (
            base_reasoning
            + stop_answer_max
            + off_spine_reasoning
            + off_spine_answer_max
        ),
    }


def run_preflight(
    config: RunConfig,
    project_root: Path,
    *,
    require_gpu: bool = False,
) -> dict[str, Any]:
    if config.backend == "vllm":
        for label, revision in (
            ("model revision", config.model.revision),
            ("tokenizer revision", config.model.tokenizer_revision),
            ("dataset revision", config.dataset.revision),
            ("scorer revision", config.scorer.revision),
        ):
            if not IMMUTABLE_REVISION.fullmatch(revision):
                raise ValueError(f"{label} must be an immutable hexadecimal commit")
    if config.stage == "formal":
        model_overrides = {
            name: os.environ.get(name)
            for name in (
                "FLCVA_MODEL_PATH",
                "FLCVA_SCORER_PATH",
                "FLCVA_MODEL_URI",
                "FLCVA_SCORER_URI",
            )
            if os.environ.get(name)
        }
        if model_overrides:
            raise ValueError(
                "formal runs forbid model path/URI overrides; frozen IDs must "
                f"resolve automatically: {sorted(model_overrides)}"
            )
        unsafe_snapshot_flags = {
            name: os.environ.get(name)
            for name in (
                "FLCVA_ALLOW_UNVERIFIED_LOCAL_SNAPSHOT",
            )
            if os.environ.get(name) == "1"
        }
        if unsafe_snapshot_flags:
            raise ValueError(
                "formal runs forbid unverified model snapshots: "
                f"{sorted(unsafe_snapshot_flags)}"
            )
        frozen_campaign_sha256 = validate_formal_config(config, project_root)
    else:
        frozen_campaign_sha256 = None
    problems = load_public_problems(config, project_root)
    run_dir = resolve_run_dir(config, project_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".flcva-write-probe.",
        dir=run_dir,
        delete=False,
    ) as handle:
        probe_path = Path(handle.name)
        handle.write(b"flcva-preflight")
        handle.flush()
        os.fsync(handle.fileno())
    probe_path.unlink()
    hardware = hardware_snapshot()
    vllm_version: str | None = None
    gpu_model_checks: dict[str, Any] | None = None
    if require_gpu:
        if not hardware.get("cuda_available"):
            raise RuntimeError("GPU preflight requires CUDA")
        devices = [str(value) for value in hardware.get("devices", [])]
        if not any("H20" in value.upper() for value in devices):
            raise RuntimeError(f"expected NVIDIA H20, observed devices={devices}")
        gpu_model_checks = _gpu_model_preflight(config, problems)
        vllm_version = str(gpu_model_checks["packages"]["vllm"])
    contract = build_contract(config, project_root)
    runtime_environment = runtime_environment_snapshot()
    payload = seal_record({
        "schema_version": "flcva-preflight-v2",
        "run_id": config.run_id,
        "passed": True,
        "contract_sha256": contract.contract_sha256,
        "scale": expected_scale(config, len(problems)),
        "hardware": hardware,
        "runtime": {
            "vllm": vllm_version,
            "environment": runtime_environment,
            "environment_sha256": payload_sha256(runtime_environment),
            "gpu_model_checks": gpu_model_checks,
        },
        "checks": {
            "public_problem_ids_unique": len({item.id for item in problems})
            == len(problems),
            "labels_not_loaded": True,
            "immutable_revisions": config.backend != "vllm"
            or all(
                IMMUTABLE_REVISION.fullmatch(value)
                for value in (
                    config.model.revision,
                    config.model.tokenizer_revision,
                    config.dataset.revision,
                    config.scorer.revision,
                )
            ),
            "gpu_required": require_gpu,
            "vllm_0_11_series": (
                is_vllm_0_11_series(vllm_version)
                if vllm_version is not None
                else None
            ),
            "frozen_campaign_sha256": frozen_campaign_sha256,
            "frozen_campaign_match": (
                frozen_campaign_sha256 is not None
                if config.stage == "formal"
                else None
            ),
        },
    })
    atomic_write_json(run_dir / "preflight.json", payload)
    return payload
