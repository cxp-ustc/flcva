# Artifact scope

This repository contains the executable FLCVA protocol, deterministic answer
parsers, ledger construction, proxy statistics, replay policies, robustness
analyses, frozen public benchmark inputs, and tests.

The full experiment produced hundreds of millions of generated tokens and is
not copied into the review repository. Analysis scripts expect audited run
directories under:

- `artifacts/primary-qwen3`
- `artifacts/primary-deepseek`
- an explicitly supplied Llama replication directory

The repository intentionally omits cluster-specific launch wrappers,
credentials, model weights, caches, and raw execution logs. Those omissions do
not change the scientific definitions implemented by the package.
