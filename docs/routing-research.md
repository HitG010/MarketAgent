# Candidate Workflow Research Protocol

Phase 4 measures a fixed set of candidate workflows before training a router. It
asks which policy-eligible workflow can satisfy a request's quality requirement
while minimizing observed provider fee, latency, or comparable measured energy.

Phase 4 is a full-information laboratory. It runs every available action for every
selected request and keeps blocked and unavailable cells explicit. It does not use
a learned selector, live paid API, online exploration, contextual bandit, or RL.

## Claims Boundary

Phase 4 can establish:

- deterministic, leakage-aware routing development and test data
- hard pre-execution policy gates
- exact calculator behavior and abstention coverage
- deterministic BM25 retrieval quality and downstream RAG behavior
- local base-model and conditionally approved LoRA outcomes
- verified provider-neutral strong-model replay outcomes
- action coverage, quality, safety, latency, provider fee, and energy-known rates
- constrained oracle opportunity and aggregate Pareto comparisons

Phase 4 does not establish:

- that a deployable router has learned the oracle decisions
- that local or RAG model outputs are semantically safe
- that a large model is superior when no verified replay row exists
- that missing compute cost or energy is zero
- that energy values from different meters or system boundaries are comparable
- that a partial test run is confirmatory evidence
- that LoRA specialization succeeded before the M2 matrix is reviewed

No per-request oracle action is written as a router-training label.

## Environment

Use the Python 3.11 routing lock in a separate environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv-routing
.\.venv-routing\Scripts\python.exe -m pip install --upgrade pip
.\.venv-routing\Scripts\python.exe -m pip install -r requirements-routing.lock
.\.venv-routing\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv-routing\Scripts\python.exe -m pip check
```

Apple Silicon:

```bash
python3.11 -m venv .venv-routing
./.venv-routing/bin/python -m pip install --upgrade pip
./.venv-routing/bin/python -m pip install -r requirements-routing.lock
./.venv-routing/bin/python -m pip install --no-build-isolation --no-deps -e .
./.venv-routing/bin/python -m pip check
```

The generic lock installs platform-appropriate Torch wheels. On Apple Silicon,
`inference doctor` should select MPS with FP16. The exact pinned Qwen snapshot must
be cached for `--local-files-only` workflow runs.

```bash
./.venv-routing/bin/sms inference doctor --config configs/inference.yaml
```

Docker is required when a matrix includes MBPP code requests because candidate code
is scored in the existing sandbox. It is not needed for the calculator, retrieval,
replay, or non-code local-model paths.

## Candidate Actions

| Action ID | Workflow | Initial state |
|---|---|---|
| `tool.calculator.v1` | Bounded exact arithmetic AST | Enabled |
| `local.qwen-base.v1` | Pinned local Qwen base | Enabled when runtime is ready |
| `local.qwen-lora-math.v1` | Qwen plus math adapter | Unapproved by default |
| `local.qwen-lora-code.v1` | Qwen plus code adapter | Unapproved by default |
| `local.qwen-lora-logic.v1` | Qwen plus logic adapter | Unapproved by default |
| `local.qwen-lora-knowledge.v1` | Qwen plus knowledge adapter | Unapproved by default |
| `rag.bm25-qwen-base.v1` | Split corpus BM25 plus local Qwen base | Enabled when corpus and local runtime are ready |
| `remote.strong-replay.reference.v1` | Verified offline provider observation | Available only where a row exists |

A route is a composable workflow identity. The contracts separately represent the
tool, retriever, corpus, provider, model, adapter, and token budget. Executors never
silently fall back to another action.

## Research Partitions

The default routing dataset contains 50 development and 50 untouched test requests
per domain. It uses the pinned evaluation source splits and seed 42.

Preparation refuses to proceed unless it can verify:

- the Phase 1 benchmark bytes against its manifest
- the Phase 3 source train and validation bytes against their manifest
- source IDs and normalized input-content hashes

Rows used by Phase 1 or Phase 3 are excluded. Development and test selections are
also disjoint by qualified source ID and normalized content hash.

Action-visible request files contain opaque request IDs, messages, output contracts,
and policy context. They do not contain benchmark domain, references, answers,
supporting facts, MBPP tests, or HotpotQA context. Hidden evaluator files retain the
normalized example and source provenance under the same request ID.

Knowledge requests expose only the question as `retrieval_query`. The no-retrieval
local action therefore sees the question without evidence. The RAG action is the
only action that receives retrieved passages.

The calculator companion suite contains six development and six test expressions.
Its exact expected outputs are stored separately and are never included in the
four-domain aggregate.

## Prepare and Inspect

Prepare Phase 1 and Phase 3 source artifacts first. The formal adapters are optional,
but the Phase 3 selected source-data files are required for leakage exclusions.

```powershell
.\.venv\Scripts\sms.exe data prepare
.\.venv-training\Scripts\sms.exe training prepare --local-files-only
```

Prepare routing requests, hidden evaluator records, calculator fixtures, and one
HotpotQA corpus per split. Omit `--local-files-only` on the first source download.

```powershell
.\.venv-routing\Scripts\sms.exe routing prepare --local-files-only
.\.venv-routing\Scripts\sms.exe routing inspect
```

`routing inspect` verifies request/evaluator hashes and exact ID pairing, verifies
each corpus against the correct split and evaluator hash, and prints current action
fingerprints for replay capture.

Preparation is collision-safe. Use `--overwrite` only to intentionally replace all
prepared routing and retrieval artifacts under the configured data directory.

## Strict Calculator

The calculator accepts only numeric constants, parentheses, unary plus/minus, and
configured add, subtract, multiply, divide, and integer-power operations. It uses
exact rational arithmetic and never calls `eval`, `exec`, Python source, or a shell.

Configured limits cover expression length, AST depth, operation count, intermediate
magnitude, decimal exponent, and power exponent. Calls, names, attributes, strings,
collections, booleans, complex values, floor division, modulo, division by zero,
fractional powers, and out-of-range values are unsupported.

Unsupported input is an abstention with a stable reason, not an attempted wrong
answer. Calculator accuracy and coverage are reported separately.

## BM25 Retrieval and RAG

Each routing split has its own corpus. Documents are NFKC-normalized, deduplicated by
full title/body content hash, and ordered by content-addressed document ID. Corpus
documents contain no request IDs or relevance labels.

The tokenizer is deterministic Unicode NFKC normalization, lowercase conversion,
and a regex word extractor. It does not download stopwords, stemmers, or language
models. BM25 uses pinned `rank-bm25==0.2.2`; score ties break by document ID.

Retrieval is evaluated at `k = 1, 3, 5, 10`. RAG generation sees only the first five
ranked passages. Reports include recall@k, MRR, empty-query rate, error rate, mean
retrieval latency, corpus coverage, and observation coverage when policy blocks some
knowledge requests.

Supporting titles remain evaluator-only. Unresolved titles reduce corpus coverage
and remain in the retrieval recall denominator.

## Local Base and LoRA Actions

The local workflow reuses the Phase 2 Hugging Face backend. If an approved verified
adapter is actionable, it reuses the Phase 3 PEFT backend and switches adapters on
one loaded base model. RAG uses the same backend with adapters disabled.

LoRA requires two independent gates:

1. The action is explicitly `approved: true` in `configs/routing.yaml`.
2. All four adapter artifacts pass existing manifest/config/weight verification.

The M2 LoRA matrix should be reviewed before changing approval. Artifact existence
alone is not evidence of useful specialization. Approving a new adapter does not
invalidate unchanged base, calculator, retrieval, or replay action artifacts.

Each local observation binds the action, inference config, prepared prompt catalog,
hardware/runtime identity, and adapter SHA/run fingerprint. Historical observations
are reused only when this action-specific identity can still be verified.

## Strong-Model Replay

Phase 4 makes no paid API calls and handles no provider credentials. A replay capture
is an externally acquired observation bound to:

- request ID and request fingerprint
- action ID and action fingerprint
- provider, model, and model version
- response and prompt/completion tokens
- observed latency
- provider fee and pricing schedule
- verified provider safety status
- optional measured/replayed energy and measurement source
- UTC capture time and non-sensitive capture metadata

Use `routing inspect` to obtain the current action fingerprint. The request JSONL
already contains its request fingerprint. The committed files under
`tests/fixtures/routing/` are a synthetic schema example.

Import captures for one request split:

```powershell
.\.venv-routing\Scripts\sms.exe routing replay-import `
  --requests data/routing/development.requests.jsonl `
  --captures C:\captures\development.jsonl `
  --pricing C:\captures\pricing.json `
  --output data/routing/replay/development.rows.jsonl
```

Inspect verified coverage:

```powershell
.\.venv-routing\Scripts\sms.exe routing replay-inspect `
  --requests data/routing/development.requests.jsonl `
  --pricing C:\captures\pricing.json `
  --rows data/routing/replay/development.rows.jsonl
```

Import recomputes provider fees from the pinned pricing catalog and rejects stale
request/action fingerprints, duplicate rows, unknown schedules, excess completion
tokens, unverified safety, invalid energy provenance, evaluator fields, and likely
credential fields. Runtime loading rechecks the import manifest and every row.

A missing replay row makes only that request/action cell unavailable. Replay never
substitutes a fabricated response and never contacts a provider.

## Policy and Safety

Policy is checked before runtime availability:

- confidential and restricted requests cannot use remote replay
- network-disabled requests cannot use remote actions
- RAG requires the configured corpus in the request allowlist
- calculator requires the configured tool in the request allowlist
- LoRA requires explicit research approval

Blocked, unavailable, unsupported, error, and completed are distinct states.

Calculator output is marked safe under its deterministic syntax boundary. Verified
replay uses the provider safety status captured in the row. Local and RAG semantic
safety remains unknown because Phase 4 includes no learned moderation system.

The default policy has `allow_unknown_output_safety: false`. Local and RAG quality
is still measured, but those outputs are not considered feasible by constrained
oracles until an explicit policy allows unknown safety or a later verified safety
assessment is added.

## Telemetry Semantics

Telemetry separates:

- provider fee
- optional compute-cost estimate
- total cost, only when both components are known

Local provider fee is zero, but local compute cost and total cost remain null. They
are never imputed as zero. Replay provider fee is verified against its price sheet.

Energy has one provenance: `measured`, `replay`, `estimated`, or `unavailable`.
Primary energy comparisons accept only measured/replay values and compare actions
only within the same provenance and measurement source. Missing energy is null.
Phase 4 does not claim energy optimization.

## Run the Workflow Matrix

Start with development data. Docker must be ready if code requests are selected.

```powershell
.\.venv-routing\Scripts\sms.exe experiment workflow-matrix `
  --data-dir data/routing `
  --split development `
  --output-dir reports/workflow-matrix/development `
  --replay-rows data/routing/replay/development.rows.jsonl `
  --pricing C:\captures\pricing.json `
  --local-files-only
```

Omit replay arguments when no verified replay dataset exists. Missing local model,
corpus, adapter, or replay rows become explicit unavailable cells; malformed
artifacts remain fatal.

After development decisions are frozen, run the complete untouched test split:

```powershell
.\.venv-routing\Scripts\sms.exe experiment workflow-matrix `
  --data-dir data/routing `
  --split test `
  --output-dir reports/workflow-matrix/test `
  --local-files-only
```

Using `--limit` on the test split labels the result
`exploratory_partial_test`, not confirmatory. Only the complete frozen split is
labeled `confirmatory_untouched_test`.

## Outputs and Resume

```text
reports/workflow-matrix/development/
  actions/
    <action-id>/
      manifest.json
      outcomes.jsonl
  action_outcomes.jsonl
  workflow_matrix_summary.json
  workflow_matrix_report.md
```

Each action directory is independently resumable. Its manifest binds the action
fingerprint and exact request IDs/fingerprints. `--resume` reuses current rows,
reruns newly available actions, and republishes aggregates. Adding an approved LoRA
or replay row therefore extends the matrix without rerunning unchanged actions.

An interrupted or failed attempt preserves valid per-action checkpoints but removes
old aggregate files before runtime/planning/execution begins. Aggregates are
republished only after every action has one status row for every selected request
and hidden-reference scoring succeeds.

Historical local, RAG, and replay observations are never reused when their
action-specific runtime identity cannot be verified. Use `--overwrite` to start a
new action set intentionally.

Infrastructure errors abort aggregate publication. Wrong answers, policy blocks,
runtime unavailability, and supported abstentions remain research data.

## Analysis

The summary reports action coverage, status, quality, safety, latency, known fee,
energy-known rate, retrieval metrics, and domain scores. It evaluates configured
quality floors while also enforcing every request's hard `required_quality`.

It reports:

- aggregate Pareto frontiers only for full-coverage, safely feasible, fully priced
  actions
- energy Pareto frontiers separately by measurement provenance/source
- best-quality oracle
- cheapest feasible oracle
- fastest feasible oracle
- energy-aware oracle on one common measurement boundary
- always-local, calculator-first, RAG-where-eligible, replay-where-available, and
  static tool/RAG/local baselines
- deterministic bootstrap intervals for coverage, baseline quality, and paired
  constrained oracle gaps

Provider fee totals remain null when any selected fee is unknown. Sparse actions do
not dominate full-coverage actions through incomparable averages.

## Opt-In Production Smoke

After the pinned model is cached, run the production calculator, local base, BM25
RAG, replay, and policy paths together without network access:

Windows:

```powershell
$env:SMS_RUN_WORKFLOW_TESTS = "1"
.\.venv-routing\Scripts\python.exe -m pytest -m workflow -v
Remove-Item Env:SMS_RUN_WORKFLOW_TESTS
```

Apple Silicon:

```bash
SMS_RUN_WORKFLOW_TESTS=1 ./.venv-routing/bin/python -m pytest -m workflow -v
```

Normal CI excludes `workflow`, `model`, and `training`; it never downloads model
weights or calls a model provider.

## Product Gate After Phase 4

Proceed to a learned router only if the complete action matrix has enough verified
coverage and nontrivial constrained oracle opportunity. Phase 5 should freeze the
matrix, extract only pre-decision features, train calibrated per-action success and
latency predictors, and compare against every Phase 4 rule baseline.

Report Brier score, calibration error, risk-coverage, policy/quality constraint
violations, bootstrap intervals, and oracle gap. Contextual bandits remain later
work and require logged context, available actions, chosen action, action
probability, reward, and policy version.