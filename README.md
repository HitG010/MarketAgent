# Small Models Society

Small Models Society is a reproducible research harness for studying adaptive
coordination among small language models. Phase 1 prepares balanced four-domain
benchmarks, validates predictions, scores answers, executes generated Python in a
constrained Docker container, and writes reports. Phase 2 adds pinned local model
inference plus one general and four prompt-specialist baselines. Phase 3 adds
leakage-aware LoRA training and cross-domain validation for four learned adapters.
Phase 4 adds a policy-gated candidate workflow laboratory with an exact calculator,
split-specific BM25 retrieval, local base/conditional LoRA execution, verified
strong-model replay, and full-information Pareto/oracle analysis.

Prompt profiles still share one model and remain separate from the learned adapters.
Phase 4 measures candidate outcomes but does not create router labels or train a
router. Live paid APIs, online exploration, RL, dashboards, and cloud deployment
remain outside the current phase.

## Prerequisites

- Windows 10 or 11; Apple silicon with macOS 14+ for the MPS training path
- Python 3.11
- Docker Desktop using Linux containers, required only for MBPP evaluation and
  Docker integration tests
- For local inference: approximately 8 GB system RAM or 4.5 GB NVIDIA VRAM is
  recommended; CPU inference is supported but can be slow
- For the LoRA pilot: 16 GB NVIDIA VRAM or 16 GB Apple unified memory is
  recommended; 12 GB is the enforced minimum accelerator-memory estimate

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

## Local Inference Setup

Keep the lightweight Phase 1 environment and model environment separate. The
inference lock includes all baseline, development, PyTorch, Transformers,
Safetensors, and hardware-diagnostic dependencies:

```powershell
py -3.11 -m venv .venv-inference
.\.venv-inference\Scripts\python.exe -m pip install --upgrade pip
.\.venv-inference\Scripts\python.exe -m pip install -r requirements-inference.lock
.\.venv-inference\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
```

Check package, device, dtype, memory, and model-cache readiness without loading or
downloading the model:

```powershell
.\.venv-inference\Scripts\sms.exe inference doctor
```

Device selection is automatic: CUDA with BF16 when supported, CUDA with FP16
otherwise, Apple MPS with FP16, and CPU with FP32 as the fallback. The project does
not silently enable quantization. See [docs/models.md](docs/models.md) for the pinned
model and memory details.

On Windows, the generic PyPI lock may install a CPU-only Torch wheel. If this machine
has an NVIDIA GPU but `inference doctor` reports `cuda_available: false`, use the
[official PyTorch installer](https://pytorch.org/get-started/locally/) to replace
Torch with the CUDA-enabled build for the pinned Torch release, then rerun
`python -m pip check` and `sms inference doctor`. Do not infer CUDA support only from
the presence of an NVIDIA driver.

## Candidate Workflow Setup

Create a separate Python 3.11 environment for Phase 4. The routing lock includes the
local inference stack, PEFT adapter support, and pinned `rank-bm25==0.2.2`:

```powershell
py -3.11 -m venv .venv-routing
.\.venv-routing\Scripts\python.exe -m pip install --upgrade pip
.\.venv-routing\Scripts\python.exe -m pip install -r requirements-routing.lock
.\.venv-routing\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv-routing\Scripts\python.exe -m pip check
```

On Apple Silicon, use `python3.11`, `.venv-routing/bin/python`, and
`.venv-routing/bin/sms`. The runtime should select MPS with FP16. Full commands and
the claims boundary are in [docs/routing-research.md](docs/routing-research.md).

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
Inference backends must emit this format without receiving the benchmark
`reference` object.

## Generate Local Predictions

Start with the deterministic 20-row benchmark before attempting the default
400-row run:

```powershell
.\.venv\Scripts\sms.exe data prepare `
  --sample-per-domain 5 `
  --output-dir data/processed/five-per-domain
```

Run the general prompt profile. The first online run downloads the pinned model
snapshot to the Hugging Face cache:

```powershell
.\.venv-inference\Scripts\sms.exe inference predict `
  --benchmark data/processed/five-per-domain/benchmark.jsonl `
  --output data/processed/predictions/general.jsonl `
  --profile general
```

Available profiles are `general`, `math`, `code`, `logic`, and `knowledge`.
`--domain` may be repeated, and `--limit` applies after domain filtering:

```powershell
.\.venv-inference\Scripts\sms.exe inference predict `
  --benchmark data/processed/five-per-domain/benchmark.jsonl `
  --output data/processed/predictions/math-smoke.jsonl `
  --profile math `
  --domain math `
  --limit 2
```

Prediction output is collision-safe. Existing artifacts require one explicit policy:

```powershell
# Continue only when benchmark, model, prompts, filters, packages, and hardware match.
.\.venv-inference\Scripts\sms.exe inference predict <same arguments> --resume

# Discard the old run and start again.
.\.venv-inference\Scripts\sms.exe inference predict <same arguments> --overwrite
```

`--local-files-only` prevents network access and fails if the exact pinned snapshot
is not cached. `--fail-fast` stops at the first recoverable per-example error;
without it, failures become explicit `error` prediction rows and the run continues.
The inference lock includes Hugging Face's Xet transfer client for the multi-gigabyte
weight file. A broken first transfer can be retried; the cache remains outside Git.

Each prediction file has a sibling `.manifest.json` containing benchmark, model,
prompt, configuration, package, filter, device, and dtype fingerprints. Checkpoints
are atomically rewritten after the configured number of examples. An interrupted run
therefore leaves valid ordered JSONL and can be resumed safely.

Evaluate predictions with the unchanged Phase 1 scorer:

```powershell
.\.venv-inference\Scripts\sms.exe evaluate `
  --benchmark data/processed/five-per-domain/benchmark.jsonl `
  --predictions data/processed/predictions/general.jsonl `
  --output-dir reports/general
```

Docker must be running when the selected benchmark includes code examples.

## Prompt Specialization Matrix

Run every profile over the same selected examples while loading the model only once:

```powershell
.\.venv-inference\Scripts\sms.exe experiment prompt-matrix `
  --benchmark data/processed/five-per-domain/benchmark.jsonl `
  --output-dir reports/prompt-matrix
```

The output contains one prediction manifest and standard evaluation directory per
profile, plus:

- `profile_results.jsonl`: one profile/example score per line
- `specialization_summary.json`: profile-by-domain scores, general deltas,
  own-domain lift, off-domain degradation, and observed oracle opportunity
- `specialization_report.md`: compact human-readable tables

The observed prompt-profile oracle chooses the best measured profile separately for
each example. Its improvement over `general` estimates the routing opportunity
available to a future controller; it is an upper bound for these observed profiles,
not a deployable router and not evidence of trained specialization.

## Train LoRA Specialists

Create a separate Python 3.11 environment containing the pinned inference and
training stack:

```powershell
py -3.11 -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install --upgrade pip
.\.venv-training\Scripts\python.exe -m pip install -r requirements-training.lock
.\.venv-training\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv-training\Scripts\python.exe -m pip check
```

Check readiness. CPU requires an explicit override and is intended only for tiny
debug runs:

```powershell
.\.venv-training\Scripts\sms.exe training doctor
```

Prepare 96 training and 24 validation examples per domain from pinned source
training splits. The Phase 1 benchmark and manifest must already exist at the paths
in `configs/training.yaml`:

```powershell
.\.venv-training\Scripts\sms.exe training prepare --local-files-only
```

Train one adapter, or train all four in isolated sequential processes:

```powershell
.\.venv-training\Scripts\sms.exe training train `
  --specialist math `
  --local-files-only

.\.venv-training\Scripts\sms.exe training train-all --local-files-only
```

Interrupted runs retain checkpoints under `artifacts/adapters/.<domain>.work` and
require `--resume`. Existing runs are never reused or replaced implicitly.

Evaluate base weights and all four adapters over the same benchmark examples with
the same general prompt:

```powershell
.\.venv-training\Scripts\sms.exe experiment lora-matrix `
  --benchmark data/processed/benchmark.jsonl `
  --output-dir reports/lora-matrix `
  --prompt-summary reports/prompt-matrix/specialization_summary.json `
  --local-files-only
```

This produces aggregate adapter-by-domain scores, own-domain lift, off-domain
degradation, retention, and observed oracle opportunity. It deliberately does not
emit per-example oracle router labels. See [docs/training.md](docs/training.md) for
the full hardware, artifact, recovery, licensing, and interpretation guide.

## Candidate Workflow Matrix

Phase 4 requires the verified Phase 1 benchmark and Phase 3 selected source-data
artifacts for leakage exclusion. Learned adapter weights and strong-model replay rows
remain optional.

Prepare opaque development/test requests, hidden evaluator rows, calculator fixtures,
and one HotpotQA corpus per split:

```powershell
.\.venv-routing\Scripts\sms.exe routing prepare --local-files-only
.\.venv-routing\Scripts\sms.exe routing inspect
```

Run the development matrix. Omit replay arguments when no verified replay dataset
exists:

```powershell
.\.venv-routing\Scripts\sms.exe experiment workflow-matrix `
  --data-dir data/routing `
  --split development `
  --output-dir reports/workflow-matrix/development `
  --local-files-only
```

The eight configured actions cover the bounded calculator, local Qwen base, four
conditional LoRA adapters, BM25-RAG plus local base, and provider-neutral strong
replay. Every request/action pair receives one explicit completed, unsupported,
blocked, unavailable, or error status. Policy is evaluated before runtime readiness.

Outputs include independent `actions/<action-id>/outcomes.jsonl` checkpoints plus
`action_outcomes.jsonl`, `workflow_matrix_summary.json`, and
`workflow_matrix_report.md`. `--resume` can add newly approved LoRA or replay rows
without rerunning unchanged actions. Aggregates are removed at the beginning of an
attempt and republished only after the full selected grid and hidden-reference
scoring succeed.

The report includes action coverage, quality, safety feasibility, latency, known
provider fee, energy-known rate, retrieval metrics, deterministic bootstrap
intervals, rule baselines, constrained oracles, and Pareto frontiers. Unknown local
compute cost and energy remain null. Energy comparisons never mix measurement
sources. Local and RAG semantic safety remains unknown by default, so those outcomes
are measured but are not oracle-feasible unless policy explicitly permits unknown
safety.

After development choices are frozen, run the complete untouched test split. A test
run with `--limit` is labeled exploratory, not confirmatory.

Strong-model observations are imported and inspected offline with `sms routing
replay-import` and `sms routing replay-inspect`. Phase 4 never calls a paid model API
or stores provider credentials. See
[docs/routing-research.md](docs/routing-research.md) for capture contracts, policy
rules, split semantics, replay commands, and interpretation gates.

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
.\.venv\Scripts\python.exe -m pytest -m "not docker and not model and not training and not workflow"
```

With Docker Desktop running:

```powershell
.\.venv\Scripts\python.exe -m pytest -m docker
```

Run the full-weight model smoke manually after installing inference dependencies,
starting Docker, and allowing the pinned model download:

```powershell
$env:SMS_RUN_MODEL_TESTS = "1"
.\.venv-inference\Scripts\python.exe -m pytest -m model -v
Remove-Item Env:SMS_RUN_MODEL_TESTS
```

Run the offline one-step tiny-Qwen LoRA smoke after installing the training lock:

```powershell
$env:SMS_RUN_TRAINING_TESTS = "1"
.\.venv-training\Scripts\python.exe -m pytest -m training -v
Remove-Item Env:SMS_RUN_TRAINING_TESTS
```

Run the production candidate workflow smoke after installing the routing lock and
caching the pinned model:

```powershell
$env:SMS_RUN_WORKFLOW_TESTS = "1"
.\.venv-routing\Scripts\python.exe -m pytest -m workflow -v
Remove-Item Env:SMS_RUN_WORKFLOW_TESTS
```

Normal CI never downloads model weights or trains a model. It uses fake model
backends for inference/training orchestration and runs Docker sandbox tests
separately. The production workflow smoke is also opt-in.

## Inference Troubleshooting

- **Missing Torch/Transformers:** install `requirements-inference.lock` in the
  Python 3.11 inference environment and rerun `sms inference doctor`.
- **Model not cached:** allow one online run, or remove `--local-files-only`.
- **CUDA requested but unavailable:** use `device: auto` or `cpu` in
  `configs/inference.yaml` and inspect the doctor output.
- **Out of memory:** reduce `max_input_tokens` or per-domain `max_new_tokens`, or
  use a machine with more RAM/VRAM. Phase 2 deliberately excludes quantization.
- **Slow CPU inference:** begin with `--limit 1` or the 20-row benchmark. A full
  five-profile matrix performs five generations per example.
- **Code evaluation fails before scoring:** start Docker Desktop, then run
  `sms doctor --build-sandbox`.
- **Resume mismatch:** use the original model, prompt, filters, hardware, and
  package stack, or choose `--overwrite` for a new run.
