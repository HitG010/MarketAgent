# LoRA Specialist Training

Phase 3 trains four independent adapters for math, code, logic, and knowledge on the
pinned Qwen2.5-1.5B-Instruct base. The purpose is to measure whether learned weights
create useful differentiation. It does not assume that training will produce a
positive result, and it does not generate router labels.

## Environment

Use Python 3.11 and `requirements-training.lock`. Keep this environment separate
from the lightweight benchmark environment.

### Windows or NVIDIA CUDA

```powershell
py -3.11 -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install --upgrade pip
.\.venv-training\Scripts\python.exe -m pip install -r requirements-training.lock
.\.venv-training\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv-training\Scripts\python.exe -m pip check
```

The generic Windows PyPI Torch wheel may be CPU-only. If an NVIDIA GPU exists but
the doctor reports `cuda_available: false`, replace Torch using the official PyTorch
selector for the pinned release. Rerun `pip check` and the doctor afterward.

CUDA selection prefers BF16 and uses FP16 when BF16 is unavailable.

### Apple Silicon MPS

Use Apple silicon, macOS 14 or later, Python 3.11, and an MPS-enabled PyTorch build.
Install Xcode command-line tools if needed:

```bash
xcode-select --install
python3.11 -m venv .venv-training
./.venv-training/bin/python -m pip install --upgrade pip
./.venv-training/bin/python -m pip install -r requirements-training.lock
./.venv-training/bin/python -m pip install --no-build-isolation --no-deps -e .
./.venv-training/bin/python -m pip check
```

MPS uses FP16 and unified system memory. PyTorch still labels MPS as beta. Run the
one-step smoke before a real adapter. The project reports
`PYTORCH_ENABLE_MPS_FALLBACK=1` but never sets it automatically, because silent CPU
operations make runtime and memory behavior harder to interpret.

## Readiness

```powershell
.\.venv-training\Scripts\sms.exe training doctor --local-files-only
```

The doctor checks:

- Torch, Transformers, Safetensors, Accelerate, PEFT, and TRL imports and versions
- CUDA and MPS availability, dtype, and detected memory
- complete pinned model cache
- writable adapter root
- explicit CPU-only status

Full training is blocked below 12 GB detected accelerator memory. Between 12 and
16 GB is classified as tight. CPU is a debug path and requires `--allow-cpu`; this
override does not make a full 1.5B pilot practical.

## Prepare Training Data

First prepare the Phase 1 benchmark at the paths named in `configs/training.yaml`:

```powershell
.\.venv\Scripts\sms.exe data prepare
```

Then prepare source and model-facing SFT artifacts:

```powershell
.\.venv-training\Scripts\sms.exe training prepare --local-files-only
```

Preparation downloads only pinned source training splits when they are not cached.
Remove `--local-files-only` for the first online run. Existing output requires
`--overwrite`; partial old data is never accepted implicitly.

The output layout is:

```text
data/training/
  train.jsonl
  validation.jsonl
  manifest.json
  sft/
    train.jsonl
    validation.jsonl
    manifest.json
```

The source records retain normalized references for formatting and provenance. The
SFT files contain only `system`/`user` prompts and one assistant completion. Every
prompt uses the fixed `general` system profile. Prompt tokens are masked from loss;
overlength examples are excluded before selection rather than truncating answers.

## Train Adapters

Train one domain:

```powershell
.\.venv-training\Scripts\sms.exe training train `
  --specialist math `
  --local-files-only
```

Train all four sequentially:

```powershell
.\.venv-training\Scripts\sms.exe training train-all --local-files-only
```

`train-all` launches one child process per domain in enum order. This ensures the
base model and accelerator allocations are released between adapters. It does not
train adapters concurrently.

Completed outputs use this layout:

```text
artifacts/adapters/
  math/
    adapter_model.safetensors
    adapter_config.json
    manifest.json
    metrics.json
    log_history.json
  code/
  logic/
  knowledge/
```

The backend rejects an adapter output containing full base-model weights. The final
directory is published only after adapter/config/hash validation succeeds.

## Resume and Overwrite

An interrupted run keeps `artifacts/adapters/.<domain>.work`, its running manifest,
and TRL checkpoints. Continue only with identical model, configuration, SFT hashes,
prompt catalog, package/hardware identity, and source IDs:

```powershell
.\.venv-training\Scripts\sms.exe training train `
  --specialist math `
  --local-files-only `
  --resume
```

Use `--overwrite` to start a replacement run. A failed replacement does not remove
the previously completed adapter. Per-domain advisory locks reject concurrent
writers.

## Evaluate Learned Weights

Start with a small benchmark slice, then use the full 100-per-domain benchmark after
runtime and outputs look sound:

```powershell
.\.venv-training\Scripts\sms.exe experiment lora-matrix `
  --benchmark data/processed/benchmark.jsonl `
  --output-dir reports/lora-matrix `
  --adapter-root artifacts/adapters `
  --prompt-summary reports/prompt-matrix/specialization_summary.json `
  --local-files-only
```

The experiment loads the base once, verifies and attaches all four adapters, and
runs five weight states across every selected domain:

- base weights with adapters disabled
- math adapter
- code adapter
- logic adapter
- knowledge adapter

All five states use the same general prompt. This isolates learned weight changes
from Phase 2 prompt-profile effects. Docker is required when code examples are
selected.

Outputs include per-variant prediction/evaluation directories plus:

- `adapter_results.jsonl`: one aggregate-evaluation row per variant and example
- `lora_specialization_summary.json`: matrix, deltas, retention, and oracle metrics
- `lora_specialization_report.md`: human-readable tables

The optional prompt summary contributes only its aggregate routing opportunity. No
per-example prompt or adapter oracle selections are written as router labels.

## Interpretation Gates

Pipeline success means deterministic data, valid adapters, safe publication,
switchable inference, and reproducible evaluation all work. It does not mean the
adapters specialized successfully.

The initial research targets are:

- at least two adapters have positive own-domain lift over base/general
- mean own-domain lift is positive
- mean off-domain degradation is no worse than 0.10 absolute primary score
- learned-adapter oracle opportunity exceeds the Phase 2 prompt-profile opportunity

These are research targets, not merge assertions. If differentiation is weak, stop
before router training and revise data, formatting, LoRA targets, or hyperparameters.

## Validation and Troubleshooting

Run the offline one-step smoke:

```powershell
$env:SMS_RUN_TRAINING_TESTS = "1"
.\.venv-training\Scripts\python.exe -m pytest -m training -v
Remove-Item Env:SMS_RUN_TRAINING_TESTS
```

- **Missing training packages:** install `requirements-training.lock`, then run
  `python -m pip check` and `sms training doctor`.
- **CPU-only on an NVIDIA host:** install the matching CUDA-enabled Torch wheel.
- **MPS unavailable:** verify Apple silicon, macOS 14+, and an MPS-enabled Torch.
- **Out of memory:** stop the run and create an explicit 384-token config. Do not
  alter the existing run's fingerprint or silently switch to CPU.
- **Resume mismatch:** use the original data/config/software/hardware inputs or
  start a separate run with `--overwrite`.
- **Adapter catalog rejection:** inspect base revision, LoRA fields, manifest, and
  `adapter_model.safetensors` SHA-256 before evaluating.
- **Code matrix failure before scoring:** start Docker and build the sandbox image
  with `sms doctor --build-sandbox`.

Training data and adapters remain gitignored. Review [datasets.md](datasets.md)
before publishing adapters derived from attributed or share-alike datasets.