# Small Models Society

Phase 1 is a reproducible, four-domain benchmark harness for studying adaptive
coordination among small language models. It prepares balanced benchmark data,
validates model-independent prediction records, scores each domain, runs generated
Python only in a constrained Docker container, and writes machine-readable and
Markdown reports.

This phase intentionally contains no model inference, fine-tuning, routing, RL,
dashboard, or cloud deployment.

## Prerequisites

- Windows 10 or 11
- Python 3.11
- Docker Desktop using Linux containers, required only for MBPP evaluation and
  Docker integration tests

## Setup on Windows

From PowerShell in the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
```

Start Docker Desktop, then build and verify the sandbox:

```powershell
.\.venv\Scripts\sms.exe doctor --build-sandbox
```

`doctor` returns a nonzero exit code until Python, the benchmark configuration,
the Docker daemon, and the sandbox image are ready.

## Prepare Data

The default configuration samples 100 examples from each pinned source:

```powershell
.\.venv\Scripts\sms.exe data prepare
```

For a quick 20-row reproducibility check:

```powershell
.\.venv\Scripts\sms.exe data prepare `
  --sample-per-domain 5 `
  --output-dir data/processed/five-per-domain
```

Preparation writes `benchmark.jsonl` and `manifest.json`. The manifest records
the immutable source revisions, source and sample row counts, seed, and SHA-256
of the exact normalized JSONL bytes. Repeating the same configuration produces
the same benchmark hash.

Downloaded Hugging Face caches and normalized datasets are not committed. Source
licenses, configurations, revisions, and citations are listed in
[docs/datasets.md](docs/datasets.md).

## Validate Evaluation

Create oracle predictions for the committed synthetic benchmark:

```powershell
.\.venv\Scripts\sms.exe fixtures oracle `
  --benchmark tests/fixtures/benchmark.jsonl `
  --output data/processed/fixture-oracle.jsonl
```

Evaluate them:

```powershell
.\.venv\Scripts\sms.exe evaluate `
  --benchmark tests/fixtures/benchmark.jsonl `
  --predictions data/processed/fixture-oracle.jsonl `
  --output-dir reports/fixture-oracle
```

The output directory contains:

- `results.jsonl`: one status and metric record per benchmark example
- `summary.json`: domain, macro, micro, status, latency, token, and cost metrics
- `report.md`: a compact human-readable summary

Missing predictions, abstentions, and prediction errors remain in the denominator
with zero primary score. A Docker infrastructure failure is reported separately
from an incorrect code solution.

## Prediction Contract

Each JSONL row follows `PredictionRecord`:

```json
{"completion_tokens":3,"cost_usd":0.0,"domain":"math","example_id":"gsm8k-00001","latency_ms":42.5,"metadata":{},"model_id":"baseline-small","prompt_tokens":31,"response":"42","status":"ok"}
```

Gold references are absent from this contract and forbidden in nested metadata.
Phase 2 model adapters must emit this format without receiving the benchmark
`reference` object.

## Sandbox Boundary

MBPP candidate code is sent as JSON over standard input to Docker. It is never
passed to a host Python executable and no host directory is mounted. Each run uses:

- no network and no IPC namespace
- a read-only root filesystem and bounded, `noexec` temporary storage
- UID/GID `65532`, all Linux capabilities dropped, and no new privileges
- CPU, memory, process, file descriptor, file size, output, and wall-time limits
- host-side timeout termination and Docker `--rm` cleanup

The image base is pinned by digest in [docker/sandbox/Dockerfile](docker/sandbox/Dockerfile).
These controls reduce risk but do not make Docker equivalent to a hardened VM.
Run the harness only on a maintained Docker engine and do not inject secrets into
the sandbox environment.

## Development Checks

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src/small_models_society
.\.venv\Scripts\python.exe -m pytest -m "not docker"
```

With Docker Desktop running:

```powershell
.\.venv\Scripts\python.exe -m pytest -m docker
```
