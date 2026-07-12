# Local Model, Prompt Profiles, and LoRA Adapters

## Pinned Model

| Property | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Hugging Face revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Architecture | `Qwen2ForCausalLM` |
| Parameters | 1,543,714,304 BF16 parameters |
| License | Apache 2.0 |
| Project input limit | 4,096 tokens by default |
| Decoding | Greedy, batch size 1, sampling disabled |

The model ID alone is not reproducible because repository contents may change. Every
load passes the immutable revision, `trust_remote_code=false`, and
`use_safetensors=true`. Model weights and Hugging Face caches are not committed.

The tokenizer-owned Qwen chat template formats system and user messages. Generation
starts at the assistant turn and normally stops on `<|im_end|>`. The backend records
whether generation stopped on EOS, reached its token limit, or ended for another
reason.

Model source and license:

- <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
- <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE>

## Memory Expectations

The raw BF16 parameters require approximately 3.1 GB. CPU FP32 parameters require
approximately 6.2 GB before tokenizer, activations, attention cache, Python, and
framework overhead. The runtime therefore warns below 4.5 GB VRAM or 8 GB system
RAM; these are practical starting points, not guarantees.

`device: auto` selects:

1. CUDA BF16 when the GPU reports BF16 support.
2. CUDA FP16 on older CUDA devices.
3. Apple MPS FP16 when CUDA is unavailable and MPS is available.
4. CPU FP32 when no accelerator is available.

Selection depends on the installed Torch build. The generic Windows PyPI wheel may
be CPU-only even on an NVIDIA machine. In that case, install the CUDA-enabled wheel
for the pinned Torch release using PyTorch's official platform selector and rerun the
doctor. The project does not install or guess a CUDA toolkit build automatically.

Explicit CPU FP16/BF16 and MPS BF16 are rejected for broad operator compatibility.
Phase 3 does not use bitsandbytes, GGUF, ONNX, or another quantized runtime. Adding
quantization later requires separate accuracy, portability, and reproducibility
experiments.

## LoRA Training Defaults

Phase 3 keeps the 1.5B backbone frozen and trains four independent adapters. The
default pilot uses rank 8, alpha 16, dropout 0.05, and Qwen attention projections
`q_proj`, `k_proj`, `v_proj`, and `o_proj`. It uses completion-only loss, a 512-token
limit, batch size 1, gradient accumulation 8, learning rate `1e-4`, and three epochs.
The lowest validation-loss checkpoint is restored before adapter publication.

Standard LoRA is used instead of QLoRA so the same configuration works on CUDA and
Apple MPS. Full pilot training requires at least 12 GB detected accelerator memory;
16 GB is recommended. MPS uses system memory as its unified-memory estimate and is
still considered a beta PyTorch backend. An out-of-memory error stops the run. It
never silently falls back to CPU; the first explicit fallback is a separately
fingerprinted 384-token configuration.

Each completed adapter contains Safetensors adapter weights rather than a copy of
the base model. Its manifest binds the adapter to the exact base revision, training
configuration, SFT files, prompt catalog, source IDs, package versions, hardware,
and adapter SHA-256. Inference validates those fields before loading any adapter.

## Determinism

The project fixes the model revision, prompts, random seed, generation settings,
token budgets, and package versions. Sampling is disabled. This makes repeated runs
on one stable software/hardware stack as deterministic as the underlying kernels
permit.

Bit-for-bit output equality is not promised across CPU and GPU, different GPU
families, driver versions, Torch releases, or Transformers releases. Every run
manifest records the selected device/dtype and package versions so such differences
are visible and incompatible runs cannot be resumed together.

Input tokenization happens before generation. Inputs longer than the configured
limit are truncated on the right. Domain prompts therefore place the task before
long evidence. Prompt-token counts report the post-truncation input; metadata also
records the original count and truncation flag.

## Prompt Profiles

The committed profiles are versioned in `configs/prompt_profiles.yaml`:

- `general`: general-purpose baseline
- `math`: quantitative reasoning role
- `code`: Python generation/debugging role
- `logic`: deduction and constraint role
- `knowledge`: evidence-grounded QA role

Every profile answers every domain. This produces a full profile-by-domain matrix
and reveals both own-domain lift and off-domain cost. Domain user formatting remains
constant across profiles; only the system role changes. Therefore measured deltas
primarily reflect role prompting rather than different questions or output formats.

These profiles are not separate models, adapters, or trained specialists. They remain
the prompt-only comparator for learned LoRA adapters.

## Observed Oracle and Routing Opportunity

For each benchmark example, the observed prompt-profile oracle selects the highest
score achieved by any of the five profiles. Its aggregate score is compared with the
general profile:

$$
\text{routing opportunity}
=
\text{observed profile oracle score}
-
\text{general profile score}
$$

This is an empirical upper bound over only the tested profiles and examples. A real
router does not know correctness at decision time, so it should be expected to score
below this oracle. If the opportunity is close to zero, prompt specialization has not
created enough differentiation to justify router training.

## Phase Boundary

Phase 3 includes deterministic specialist training data, four LoRA adapters,
resumable sequential training, adapter switching, and a fixed-general-prompt weight
matrix. It reports aggregate observed oracle opportunity but excludes per-example
router labels, router training, verification, parallel collaboration, hosted-model
escalation, contextual bandits, sequential RL, dashboards, and deployment.