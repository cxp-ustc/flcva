# FLCVA: anonymous review artifact

This repository implements the **Full-Ledger Counterfactual Value Audit
(FLCVA)**. For each frozen reasoning prefix, it compares an immediate
answer with four fixed-length continuations, grades every endpoint with the
same answer operator, and accounts for all observations used by a policy.

## Contents

- `src/flcva/`: generation, scoring, evaluation, replay, statistics, and audit
  implementation.
- `configs/`: frozen campaign specification plus local mock and GPU examples.
- `data/`: frozen public benchmark prompts and deterministic gold labels.
- `paper/analysis/`: controller, replay, bootstrap, collision, and cost
  analyses.
- `paper/ablation/`: answer-cue and continuation-length robustness analyses.
- `tests/`: focused tests for parsing, signed value, statistics, ledger
  accounting, replay, and the end-to-end mock pipeline.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -r requirements.txt
```

GPU generation additionally requires the frozen vLLM API series:

```bash
python -m pip install -r requirements-gpu.txt
```

These requirement files are thin entry points to the dependency groups in
`pyproject.toml`, which remains the single source of dependency versions.

Model and dataset revisions are pinned in `configs/frozen.yaml`. Remote model
snapshots are resolved from Hugging Face at the specified commit. A local
snapshot may instead be supplied through `FLCVA_MODEL_PATH` and
`FLCVA_SCORER_PATH`; it must be a commit-named snapshot directory or contain a
`.flcva_revision` file with the exact commit hash.

## Fast verification

The mock pipeline uses no GPU and produces no paper claim:

```bash
flcva preflight --config configs/mock.yaml
flcva generate --config configs/mock.yaml
flcva score --config configs/mock.yaml
flcva evaluate --config configs/mock.yaml
flcva replay --config configs/mock.yaml
flcva audit --config configs/mock.yaml
pytest
```

## GPU protocol

The local examples retain the paper's model revisions, sampling parameters,
four-spine design, 256-token checkpoints, four continuation branches, answer
operator, and PRM:

```bash
flcva preflight --config configs/formal-qwen3.yaml --require-gpu
flcva generate --config configs/formal-qwen3.yaml
flcva score --config configs/formal-qwen3.yaml
```

Formal execution is intentionally not started by this repository. Users should
shard and schedule the same configuration using their own compute environment,
then use `flcva merge`, `flcva evaluate`, `flcva replay`, and `flcva audit`.

## Analysis inputs

The main analysis renderer reads the two audited primary run directories shown
in `ARTIFACT_SCOPE.md`. The Llama transfer renderer accepts an explicit
replication directory. Full generated traces and model weights are omitted
because of their size; the included mock pipeline exercises the same artifact
schemas and stage boundaries.

## Anonymity

This review copy contains no author names, affiliations, personal paths,
credentials, private cluster configuration, or version-control history.
