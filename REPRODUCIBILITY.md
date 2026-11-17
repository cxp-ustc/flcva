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

## Determinism and result compatibility

SHA-256 is retained for deterministic problem splits, random seeds, sampling,
and content-based record IDs. These definitions keep the experimental
assignments stable and let records refer to the same prefixes and events.
The repository does not compute or compare file-byte checksums.

New result summaries list their output paths under `artifacts`. Readers also
accept the path keys in older `artifacts_sha256` mappings without checking the
stored file digests. Protocol, problem-set, prefix, ledger, and record-ID
consistency checks remain part of the pipeline.
