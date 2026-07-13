# Dataset Sources and Attribution

The harness downloads datasets through Hugging Face `datasets` at immutable Git
commit revisions. Raw downloads and normalized benchmark rows are gitignored.
Users remain responsible for complying with each source license and any upstream
terms that apply to their use.

| Domain | Dataset and configuration | Split | Pinned revision | Dataset card license |
|---|---|---|---|---|
| Math | `openai/gsm8k` (`main`) | `test` | `740312add88f781978c0658806c59bc2815b9866` | MIT |
| Code | `google-research-datasets/mbpp` (`sanitized`) | `test` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | CC BY 4.0 |
| Logic | `allenai/ai2_arc` (`ARC-Challenge`) | `test` | `210d026faf9955653af8916fad021475a3f00453` | CC BY-SA 4.0 |
| Knowledge | `hotpotqa/hotpot_qa` (`distractor`) | `validation` | `1908d6afbbead072334abe2965f91bd2709910ab` | CC BY-SA 4.0 |

License labels above are the metadata published by the pinned Hugging Face dataset
cards. Consult the upstream repositories and papers before redistribution.

## Phase 3 Training Splits

The LoRA pilot uses the same pinned dataset revisions but only their source training
splits. It selects exactly 120 eligible rows per domain and deterministically assigns
96 to adapter training and 24 to internal validation with seed 42.

| Domain | Configuration | Training source split | Pilot rows | Train | Validation |
|---|---|---|---:|---:|---:|
| Math | `openai/gsm8k` (`main`) | `train` | 120 | 96 | 24 |
| Code | `google-research-datasets/mbpp` (`sanitized`) | `train` | 120 | 96 | 24 |
| Logic | `allenai/ai2_arc` (`ARC-Challenge`) | `train` | 120 | 96 | 24 |
| Knowledge | `hotpotqa/hotpot_qa` (`distractor`) | `train` | 120 | 96 | 24 |

Rows are hash-ranked after normalization. Selection rejects duplicate normalized
content, benchmark source IDs, benchmark input-content hashes, and examples whose
fully templated prompt plus completion exceeds 512 tokens. Training and validation
files have disjoint source IDs and content hashes. Sanitized MBPP tests and imports
are never included in model-facing prompts or completions.

Adapters are derived artifacts. Before publishing or redistributing them, review the
source licenses and attribution/share-alike obligations, especially CC BY 4.0 and
CC BY-SA 4.0. This repository records provenance but does not provide legal advice.

## Phase 4 Routing Splits

The routing laboratory samples unused rows from the same pinned evaluation source
splits. It selects 50 development and 50 untouched test examples per domain with
seed 42 after excluding:

- every Phase 1 qualified source ID and normalized input-content hash
- every selected Phase 3 train/validation source ID and content hash
- duplicate normalized content within the routing source
- overlap between routing development and routing test

Action-visible rows use opaque request IDs. Domain, references, accepted answers,
supporting facts, MBPP tests, and source IDs remain in separate hidden evaluator
records. The manifest binds both sides by file hash, row count, and exact request ID.

For HotpotQA, every action-visible request contains only the question. Phase 4 builds
one content-addressed BM25 corpus per routing split from hidden context passages.
Corpus documents contain title/body text only; request associations and supporting
title relevance labels are stored separately. Corpus loading verifies the exact
split and evaluator hash, preventing development evidence from being used in the
untouched test run.

The six-development/six-test calculator companion suite is synthetic and remains
outside four-domain aggregate quality.

## Citations

### GSM8K

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz
Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano,
Christopher Hesse, and John Schulman. *Training Verifiers to Solve Math Word
Problems*. 2021. <https://arxiv.org/abs/2110.14168>

### MBPP

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski,
David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton.
*Program Synthesis with Large Language Models*. 2021.
<https://arxiv.org/abs/2108.07732>

### ARC

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa
Schoenick, and Oyvind Tafjord. *Think You Have Solved Question Answering? Try
ARC, the AI2 Reasoning Challenge*. 2018. <https://arxiv.org/abs/1803.05457>

### HotpotQA

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. *HotpotQA: A Dataset for Diverse,
Explainable Multi-hop Question Answering*. EMNLP 2018.
<https://arxiv.org/abs/1809.09600>

## Normalization

- GSM8K stores the question as model input and separates the final answer and
  rationale into the reference object.
- Sanitized MBPP stores the prompt and inferred entry point as input. Tests,
  imports, and canonical code remain references and are executed only in Docker.
- ARC stores the question and labeled choices as input and the answer label as a
  reference.
- HotpotQA stores the distractor context and question as input. Accepted answers
  and supporting-fact indices remain references.

The Phase 4 projection deliberately removes HotpotQA context from workflow requests.
Only the BM25-RAG action can recover evidence from the separately verified corpus.

The benchmark JSONL contains references because it is an evaluator artifact.
Inference adapters must receive only each example's `input`, `id`, and permitted
metadata; prediction records reject reference fields.
