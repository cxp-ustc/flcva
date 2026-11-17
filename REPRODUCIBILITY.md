# Reproducibility

This repository contains the executable FLCVA protocol, deterministic answer
parsers, ledger construction, proxy statistics, replay policies, robustness
analyses, frozen public benchmark inputs, and tests.

The full experiment produced hundreds of millions of generated tokens and is
not bundled with this repository. Analysis scripts expect audited run
directories under:

- `artifacts/primary-qwen3`
- `artifacts/primary-deepseek`
- an explicitly supplied Llama replication directory

Model weights, caches, raw execution logs, and cluster-specific launch scripts
are not included. Use the configuration files and commands in the README to
run the protocol in your own environment.
