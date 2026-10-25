# GRPO implementation and verification report

The Kaggle notebook and supporting implementation are ready for a GPU smoke
attempt. **The current iteration has not been executed on Kaggle. No genuine GRPO optimizer
update, pretrained-model tool use, or T4 memory measurement was verified on the
development host. The previous user-supplied run is analyzed separately in
[the seed-41 report](seed41_results_analysis.md); it is not validation of this iteration.** Local execution used an Apple Silicon CPU with no CUDA.
The saved notebook outputs explicitly record these limits.

## Files

| File | Purpose |
|---|---|
| `notebooks/02_grpo_training_kaggle.ipynb` | All 13 requested sections; saved outputs from CPU preflight |
| `src/kg_rlvr/training/environment.py` | Existing graph access, isolated discovery state, typed tools, public prompts, private answer verifier |
| `src/kg_rlvr/training/rollouts.py` | Real Transformers generation, multi-turn orchestration, token preservation, observation masks, evaluation and logs |
| `src/kg_rlvr/training/grpo.py` | Pinned TRL/QLoRA construction, metrics, genuine-update checks, checkpoints, export/reload |
| `src/kg_rlvr/training/__init__.py` | Optional training package |
| `requirements-grpo.txt` | Separate pinned optional training dependencies |
| `tests/test_training_environment.py` | CPU graph, reward, isolation, leakage, episode, token-mask and checkpoint checks |
| `tests/test_grpo_pinned_api.py` | Optional tests against the real pinned TRL, Qwen tokenizer and PEFT APIs |
| `scripts/run_grpo_notebook_checks.py` | Execute notebook CPU sections with the selected Python interpreter |
| `notebooks/reports/grpo_api_validation.json` | Machine-readable CPU API evidence and explicit unverified GPU flags |
| `artifacts/grpo/runs/kg_grpo_seed41/` | Actual local configuration, selected task IDs, memory snapshot and execution evidence |
| `README.md`, `src/kg_rlvr/__init__.py` | Usage, optional-training scope and limitations |

The graph backend now uses RDFLib 7.6.0 and the saved graphs have been migrated
to Turtle. The one-hop extension appends questions while all original question and
conversational records retain their original bytes, IDs, answers and paths. See [the RDF migration report](rdf_migration_report.md).
The notebook fingerprints its selected inputs before and after execution and
reuses `GraphIndex`, `NavigationSession`, the question reader and in-memory
validator. Graph inspection is through Protégé; custom HTML graph views have
been removed. Statistical and training-metric charts remain.

## Selected stack

| Library | Pin | Local verification |
|---|---|---|
| RDFLib / pyparsing | 7.6.0 / 3.2.3 | Turtle load/save, OWL schema, forward/inverse navigation and unchanged QA execution |
| PyTorch | 2.6.0; cu124 wheel on Kaggle | CPU wheel imported; CUDA wheel not executed |
| Transformers | 5.3.0 | Real Qwen tokenizer and tiny random Qwen architecture |
| TRL | 1.8.0 | Config, actual RepeatSampler and custom-rollout mask boundary |
| PEFT | 0.18.1 | Actual adapter construction, serialization and reload on the tiny CPU architecture |
| Accelerate | 1.10.1 | Imported with the pinned stack |
| bitsandbytes | 0.49.2 | Linux dependency only; NF4 CUDA loading unverified |
| datasets | 4.8.3 | Imported with the pinned stack |
| tokenizers / huggingface-hub | 0.22.2 / 1.33.0 | Tokenizer-only download and local use |
| NumPy / pandas / PyArrow | 1.26.4 / 2.2.3 / 21.0.0 | Imported together |
| Matplotlib / psutil / jmespath | 3.9.4 / 7.2.2 / 1.0.1 | Imported with the pinned stack |

Qwen3 uses a BPE tokenizer; SentencePiece is unnecessary. vLLM, FlashAttention,
Reward models and an LLM judge are not installed by this workflow. Supervised
fine-tuning uses the existing pinned Transformers Trainer and LoRA adapter.
An attempted PEFT 0.19.0 adapter construction failed with Torch 2.6 because it
referenced an unavailable float8 dtype; 0.18.1 passed the actual adapter test.
PyArrow 26 also failed with the selected NumPy; the tested pin is 21.0.0.
The pins are API-tested on CPU, not a claim of end-to-end T4 compatibility.

The implementation follows the [released TRL 1.8.0 GRPO documentation](https://huggingface.co/docs/trl/v1.8.0/en/grpo_trainer)
and inspects the [matching trainer source](https://github.com/huggingface/trl/blob/v1.8.0/trl/trainer/grpo_trainer.py).
Installation uses the [official Torch cu124 wheel instructions](https://pytorch.org/get-started/previous-versions/).
The selected bitsandbytes release documents NF4 hardware support and CUDA wheel
coverage in its [installation guide](https://huggingface.co/docs/bitsandbytes/v0.49.2/en/installation).
Actual driver/kernel compatibility still needs the Kaggle run.

## Model and training configuration

The default model is [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
at immutable revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
The explicitly selected fallback is Qwen3-0.6B at
`c1899de289a04d12100db370d81485cdf75e47ca`; the code never silently switches.

The base is loaded once on GPU 0 in double-quantized NF4. LoRA uses rank 8,
alpha 16, dropout 0 and no bias on `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj`, checked against the actual architecture.
Only LoRA weights are trainable. Non-reentrant gradient checkpointing is enabled;
training KV caching is disabled. Bounded inference enables its own cache.
T4 uses FP16; the BF16 test excludes emulation. Generation explicitly uses the
same CUDA AMP dtype as training. These GPU paths remain unexecuted locally.

The pinned TRL constructor unconditionally casts trainable quantized-model
parameters to BF16, even for FP16 training. This caused the reported CUDA
GradScaler failure on BF16 gradients. The integration now snapshots the small
FP32 adapter before construction, restores FP32 storage and exact adapter values
after construction, and checks precision before optimizer creation. Frozen NF4
weights are untouched. Checks run again at training start after preparation or
checkpoint loading and before each step. The notebook displays actual flags and
adapter dtypes, with records in `precision_checks.jsonl`. Existing gradients cause
repair to fail with a fresh-run instruction instead of being silently discarded.
The behavior is confirmed in the [TRL 1.8.0 trainer source](https://github.com/huggingface/trl/blob/v1.8.0/trl/trainer/grpo_trainer.py);
FP32 adapter storage follows [PEFT's AMP guidance](https://huggingface.co/docs/peft/developer_guides/troubleshooting#dtype-related-issues).

AdamW uses learning rate 5e-6, constant scheduling, no weight decay and gradient
clipping at 1. The TRL objective is explicitly `loss_type="grpo"`, group-scaled
rewards, `num_iterations=1`, and `beta=0`. No KL/reference-model copy is needed;
KL is unavailable, not reported as a measured zero.

Microbatch 1, accumulation 8, one GPU and G=8 give eight trajectories for one
question per optimizer update. Accumulation 16 would give two question groups. The
generation batch equals accumulation, which must be a positive multiple of G.
Defaults are eight smoke update attempts and an opt-in 30-update full phase.
Training hops default to `[1, 2, 3]`; any nonempty combination of `[1, 2, 3, 4]` is allowed.
Set exposure weights alongside the selected training hops. Both smoke and full
training use a deterministic weighted-deficit schedule, defaulting to 20%/40%/40%
one/two/three-hop question groups. TRL shuffling is disabled for the scheduled
dataset; TRL still supplies generation/buffering repetitions. Thirty full updates
use 6/12/12 groups; eight smoke updates round to 2/3/3. Within each hop, questions
are shuffled and cycled as necessary. The schedule has an extension-stable prefix
for checkpoint resumption, and planned/actual hops are exported.

`SINGLE_ANSWER_ONLY=True` filters complete gold sets before deduplication/caps and
uses reviewed singular templates for all baseline/training/evaluation phases.
Retained original IDs and split assignments are unchanged. Before deduplication,
the full source provides 2,449 train and 983 test questions. Source
files remain unchanged; active filtered records and selection counts are exported
under each run. Equivalent source paraphrases resolve to the first original ID
chosen by unique-instance selection. No gold answer ID or path enters prompts.
All four variants use the same 30 fixed test questions, balanced across the
configured evaluation hops. `TEST_EVAL_SIZE=None` uses every eligible unique
test instance. Fixed IDs are saved before model loading; no validation split remains.

Limits default to 12 tool calls, 128 model tokens per turn, 1,536 total completion
tokens including observations, 1,024 prompt tokens and 3,072 total context tokens.
Responses allow at most 16 neighbors and 4,000 characters. Oversized neighbor
responses return an explicit error instead of silently hiding candidates.
Eight calls can constrain a four-hop episode that requests relation listings at
each step; this is a configurable experimental limit, not guaranteed sufficiency.

## Multi-turn correctness

The native TRL environment loop in this release does not consult the environment's
completed/submitted flag before generating again. The implementation therefore
uses TRL's supported experimental `rollout_func` interface. It does not implement
advantages, clipping, losses or optimizer updates itself; TRL owns those operations.
No native `tools` or `environment_factory` is passed alongside the custom hook,
which would run a second tool loop.

The inspected RepeatSampler already repeats each prompt G times. The hook asserts
that each group contains the same public question and creates a new graph session
for every trajectory. It executes actual graph operations, preserves discoveries
and stops on submission, invalid formatting or explicit limits. No generated
action is repaired into an invented call.

Qwen's real chat template renders tools and observations with thinking disabled.
Sampled action token IDs are preserved exactly. Observation and injected assistant
prefix tokens receive `env_mask=0`; action tokens receive 1. The actual TRL API
test confirms that this becomes its `tool_mask`; the release's objective excludes
those observation positions while retaining them as attention context. Sampling
uses temperature 0.8 with no top-k or top-p filtering, matching the policy
probabilities TRL computes. Greedy evaluation uses the same graph tools and limits.

Only question/start fields, public rules, schemas and real observations enter the
policy context. Gold programs, answer sets and reference paths stay in the private
task store. The verifier uses the environment's actual submitted entity ID:
`float(submitted_answer in gold_answer_ids)`. Any one correct member earns 1;
wrong or missing submissions earn 0. This remains the evaluation reward/accuracy.
Training adds a once-per-episode bonus of 0.2 when a successful executed
`get_neighbors` observation contains a gold entity. The private verifier checks
that successful observations match the actual graph query; failed calls, relation
lists, prose, acknowledgments and repeated retrievals earn no additional bonus.
The reward function recomputes both components from engine-supplied events, rather
than accepting model claims or a supplied score. Acknowledgments and tool outputs
do not expose scores or gold metadata. No format, efficiency or path rewards are
added. Per-episode and summary logs separate binary correctness, retrieval success,
retrieval bonus and combined training reward. Homogeneous-group metrics use the
combined rewards actually supplied to GRPO, including groups homogeneous at 1.2.
Run manifests fingerprint reward, single-answer selection and exposure settings
so old incompatible checkpoints are rejected.

## Verification performed

| Check | Result |
|---|---|
| Full local pytest suite | **118 passed, 9 skipped**; skips are optional pinned-stack tests in the ordinary environment |
| Isolated pinned-stack full suite | **124 passed, 1 skipped in the isolated pinned environment**, including all 6 GRPO API checks and supervised training checks; real TRL config/weighted sampler/reward/mask boundaries, real Qwen BPE template, tiny CPU PEFT adapter save/reload. Notebook-setup module skipped because nbformat is absent |
| Notebook execution | **20 code cells, zero error outputs**, CPU preflight using a real Jupyter kernel |
| Kaggle package setup regression suite | **3 passed**; reproduces missing TRL/bitsandbytes and version drift, verifies the section-1 gate and recovery instructions |
| Model progress and evaluation reward checks | **8 passed** with scripted CPU fixtures; real tqdm counters, evaluation/trajectory totals, unchanged episode results/RNG, and partial progress on errors |
| Precision regression | Real TRL constructor's quantized-model dtype branch exercised on a small CPU architecture with a metadata flag; exact FP32 adapter restoration and late-recast rejection verified. Three CPU parameter/GradScaler tests pass; CUDA NF4/GRPO remains unverified |
| Graph and QA loading | Existing 2,000 entities, 5,000 triples and 3,734 questions validated in memory |
| Source preservation | All original QA/conversational records unchanged; graph format migrated to Turtle |
| Python/notebook syntax | Supporting modules and every code cell compile |
| Pretrained Qwen inference/tool calls | **UNVERIFIED** |
| Multiple real model trajectories per question | **UNVERIFIED** |
| CUDA NF4 loading / genuine GRPO update | **UNVERIFIED** |
| T4 peak VRAM | **Not measured** (`null` in evidence) |
| Full trainer checkpoint resume | File/manifest guard tested; actual resume **UNVERIFIED** |
| Pretrained trained-adapter reload/tool evaluation | **UNVERIFIED** |
| Kaggle execution | **Not performed** |

Scripted unit fixtures only test environment orchestration and formatting. Their
actions may use reference paths to verify graph operations; they are never used
as a baseline, training result or actual model-generated example. The tiny random
CPU Qwen adapter test performs no optimization and validates neither the 1.7B
pretrained policy nor CUDA quantization. No reward/loss chart or three model
trajectories is fabricated for the saved CPU run.

The notebook's GPU smoke gate requires actual finite trainer metrics, positive
gradient norm, changed LoRA tensors, graph-neighbor execution and a checkpoint
containing adapter, optimizer, scheduler, trainer and RNG state. A homogeneous
combined-training-reward group has zero advantage; if all sampled groups are homogeneous,
the notebook fails the update gate and records the failure. The default now
samples eight groups (64 episodes at G=8 and accumulation 8). The shared prompt emphasizes completing all relationships and
checking the requested answer type, using only public question text and tool
observations. Saved group rewards, metrics and failed trajectories are displayed
in per-variant smoke logs and standalone HTML, without requiring a live
trainer. More samples cannot guarantee reward variation. Recompute the vanilla
baseline with the same revised prompt; full training does not follow a failed gate.

## Exact Kaggle execution procedure

`SHOW_PROGRESS=True` enables native model download/loading bars, prompt-processing
counts, question counters for all evaluations and trajectory counters for GRPO
generation batches. Tool-turn, executed-call and model-token status updates during
each episode. Completed evaluation episodes update measured running accuracy and
submission counts. The Trainer's optimizer-step bar is explicitly enabled;
trajectory counts and optimizer steps have separate displays. Periodic evaluation
uses the same question progress bar. Setting the flag to `False` hides the bars.
The shared tqdm dependency selects notebook or terminal rendering automatically.
Counters and error cleanup were checked with real tqdm text rendering and scripted
CPU graph fixtures on tqdm 4.67.1 and 4.70.1. No fixture is reported as model output;
Kaggle widget/GPU rendering remains unverified locally.

`INSTALL_DEPENDENCIES=False` skips installation; Kaggle's preinstalled packages
are not assumed to match the pins. If setup (or section 5 in an older notebook)
reports a mismatch, set the flag to `True`, rerun configuration and setup, wait
for pip to finish, restart the Python kernel, then set the flag back to `False`
and run from the beginning. Keep smoke/full training disabled to review only
the vanilla baseline. A new session/image needs its own setup. The current
notebook checks installed metadata for every requirements pin before importing
the graph/ML backend, so this failure is reported in section 1. Kaggle documents
its changing base environment and package installation in the
[notebook guide](https://www.kaggle.com/docs/notebooks).

1. Create a Kaggle Python 3.11/3.12 notebook and import
   `notebooks/02_grpo_training_kaggle.ipynb`. Select GPU T4 acceleration and enable
   internet for GitHub, installation and the pinned model download. Only one T4
   is used.
2. Leave `CLONE_REPOSITORY=True`, `REPOSITORY_BRANCH="master"`, and
   `PROJECT_ROOT_OVERRIDE=None`. The configuration cell obtains
   `https://github.com/AlirezaAK2000/kg-navigation-rlvr.git` directly in the current
   working directory, normally `/kaggle/working/`. It creates no repository-named
   subfolder. Empty folders use `git clone ... .`; nonempty folders use a temporary
   clone followed by a conflict-checked copy including `.git`. Unrelated files
   remain intact, conflicting files cause an error before copying, and reruns
   reuse an existing project without pulling automatically.
3. `PROJECT_ROOT` resolves to that folder and `DATA_ROOT` to its saved `data/`.
   For separately attached data, set `DATA_ROOT_OVERRIDE`. To use a mounted
   repository instead, disable cloning and set `PROJECT_ROOT_OVERRIDE` to its
   directory containing `src/`. Keep outputs under `/kaggle/working/`.
4. In a fresh session set `INSTALL_DEPENDENCIES=True` if the pins are missing.
   Run setup, which installs the official Torch cu124 wheel then the requirements.
   It can remove unused audio/vision packages to avoid compiled-extension clashes.
   If setup reports already-imported libraries, restart the session, set installation
   back to `False`, then Run All. Do not continue with mixed in-memory versions.
5. Choose a new `RUN_NAME`. Leave `RUN_GPU`, `RUN_BASELINE` and `RUN_SMOKE_TEST`
   enabled and `RUN_FULL_TRAINING=False`. Run all cells. Confirm actual baseline
   tools, changed LoRA tensors, finite metrics, full smoke checkpoint and measured
   memory. A CPU skip is not a successful training run.
6. For a complete training run, enable full training before Run All; the smoke
   gate still runs first. After a completed smoke-only Run All, restart the
   session and use a new `RUN_NAME` with full training enabled, because export
   has already released trainers and reloaded the adapter for inference.
   Configure hops, pool size, G, accumulation, token/call budgets, learning
   rate, steps, checkpoint interval and evaluation interval near the beginning.
7. Run SFT + GRPO test evaluation, export/reload and trajectory
   cells. Inspect measured base-versus-trained comparisons without assuming
   improvement. The final cell packages adapter and diagnostics for download.
8. Save a Kaggle version with outputs or download the ZIP. Download full trainer
   checkpoints separately if continued training is needed; the ZIP excludes them
   and the large base-model cache.

Outputs are `/kaggle/working/checkpoints/<RUN_NAME>/{smoke,full}/`,
`/kaggle/working/runs/<RUN_NAME>/`, `RUN_ROOT/variants/<variant>/adapter/`, and
`/kaggle/working/<RUN_NAME>_adapter_and_diagnostics.zip`.
For resume in a new session, attach the original complete full-phase checkpoint
and its parent `run_manifest.json`, use a fresh output `RUN_NAME`, enable full
training, and set `RESUME_FROM_CHECKPOINT` to that saved checkpoint. The new smoke
gate runs before resumption; the full checkpoint then restores the full-phase
weights and state. The step target is absolute and must exceed its saved step.
Model, dependencies, dataset/source hashes, task IDs and rollout settings must
match. Adapter-only exports are for inference and do not restore training state.
Bitwise-identical continuation across CUDA sessions is not promised.

On OOM, reduce completion/context lengths and optional memory use, then adjust
tool/group budgets while preserving GRPO divisibility. If necessary explicitly
select Qwen3-0.6B and rerun the baseline with the final limits. T4 fit, wall time,
working-disk consumption and useful reward frequency require measurement.

For later inference, reload the recorded model/revision once with the same NF4
configuration and load `PeftModel.from_pretrained(base, adapter_directory,
is_trainable=False)`. Load the tokenizer from the adapter directory and use the
repository's `evaluate_policy` for real graph-tool interaction. The notebook
includes the corresponding loading instructions and a reload evaluation cell.

## Recorded trajectory inspection

Inspection cells follow baseline evaluation, training, and post-training evaluation.
They use colored HTML cards with escaped text for each input message, raw model
output, parsed tool call and tool response. Full native Qwen input (including
tool schemas and chat tokens) is expandable. Each response is marked according
to whether it was appended to the model context. `INSPECT_EXAMPLES` and
`INSPECT_OFFSET` select a consecutive stored slice, without success-based ranking
or additional inference. Training traces load from saved JSONL, so a live trainer
is unnecessary. Empty CPU-only runs display an explicit absence of model traces.
Renderer unit fixtures check escaping, turn order, error responses, selection,
and trace immutability; the actual pinned tokenizer is checked separately.

The current one-hop extension and inspection checks are recorded in
[`one_hop_extension_validation.json`](../notebooks/reports/one_hop_extension_validation.json);
earlier API and RDF migration reports retain their historical scope.

## Reward reporting at training-time evaluation

`EVALUATE_EVERY_STEPS=1` enables greedy evaluation of the same six fixed test
questions after each smoke/full optimizer step. Larger positive intervals reduce
inference cost; zero disables periodic evaluation. The callback saves one summary
and one reward row per completed question, tagged with phase and optimizer step,
and prints mean reward, total reward, evaluated count and per-hop means. Progress
shows the latest question reward and running mean. Mean reward equals accuracy
under the binary final-answer verifier. Separate columns report retrieval success,
retrieval bonus and combined training reward; these diagnostics do not change
answer accuracy or contribute evaluation rewards to optimization.
Evaluation preserves the training/evaluation mode and PyTorch RNG state and its
rewards never enter the optimizer. The notebook displays a separate test-monitoring
reward table/curve and exports consolidated JSONL in the run directory. Each
baseline/final/reload evaluation also writes label-specific reward JSONL; streaming
flushes preserve completed-question rewards if a later generation fails.

## Next-iteration verification

The single-answer/retrieval-reward/exposure iteration passed 110 host tests
(6 optional API skips) and 113 tests in the pinned environment (1 optional
notebook-setup module skip). All 18 notebook code cells executed on CPU with
zero errors; GPU phases were explicitly skipped. All 3,432 retained question
prompts fit the native Qwen tokenizer budget at 601–619 tokens (limit 1,024).
The real TRL sampler preserves 6/12/12 question groups over 30 updates at G=8;
its real reward-function boundary receives independently verified 1.2/0.2/0.0
fixture scores. These are CPU correctness checks, not model-performance results.
See `notebooks/reports/training_iteration_validation.json`.

The retrieval-success signal is adapted from
[KG-R1, Section 3.3](https://arxiv.org/html/2509.26383v5#S3.SS3).
The 0.2 bonus is this project's experimental weight. The current implementation
uses one scalar reward per episode; it does not implement KG-R1's turn-level
advantage assignment. Correctness remains the primary evaluation metric.

## Supervised warm start with separate training data

The notebook now runs vanilla evaluation, assistant-only SFT, SFT test evaluation,
and GRPO in that order. The required SFT stage reserves 200 distinct train-split instances
(40/80/80 by hop), independently of the GRPO pool cap. The remaining 733 unique
one/two/three-hop instances are used by both GRPO smoke and full training.
Reservation occurs before caps; all paraphrases of a reserved query remain out
of GRPO. Runtime checks compare IDs, query-instance signatures and prompts across
SFT, GRPO and test, and the GRPO constructor checks SFT exclusion.
Graph entities and facts may be shared; no disjoint-graph claim is made.

`training/sft.py` builds demonstrations from train-only witness paths, executing
relation discovery, neighbor retrieval and final submission with the actual
GraphEnvironment. Its encoder replays them through the same Qwen codec, token
ledger and tool limits as inference. Initial prompts, observations, injected
prefixes and padding have `labels=-100`; only actual assistant call/EOS tokens
receive causal-LM loss. All 200 default demonstrations fit the existing budgets
at 816–1,238 total tokens; no truncation or invented observations is used.
Reference data is explicitly labeled as supervised demonstrations, not model
trajectories or performance evidence.

Default SFT uses one epoch, microbatch one, accumulation eight and learning rate
2e-5 (25 optimizer steps). FP32 LoRA storage and hardware-selected AMP compute are
retained. Finite gradients, changed adapter tensors and a resumable checkpoint
must verify before GRPO. SFT optimizer state is released between stages. SFT
adapter fingerprints and partition metadata enter the GRPO warm-start manifest;
resume requires the matching supervised initialization. SFT-only losses, adapter,
evaluation and formatted tool traces are separate from final GRPO artifacts.
The SFT objective is assistant-token cross-entropy; the GRPO objective and reward
remain unchanged. SFT + GRPO requires a verified SFT result; independent GRPO only restores the
pre-SFT adapter and uses its own smoke/full phases. Real pretrained SFT/GRPO on CUDA remains unverified
on the CPU development host.

The supervised-stage validation passed 118 host tests (9 optional pinned skips)
and 124 pinned-stack tests (1 optional notebook-setup module skip). A further
targeted check trained the tiny CPU adapter, released its supervised optimizer
and constructed the actual GRPO trainer, confirming exact preservation of its
updated LoRA values. All 20 notebook code cells completed CPU preflight. See
`notebooks/reports/sft_grpo_validation.json` for that stage’s evidence.


## Train/test and four-variant evaluation update

The full dataset now contains 2,667 train and 1,067 test questions; the pilot has
94 train and 41 test questions. Every former validation row moves to test with
only its split label changed. Original train files and graph files are preserved.
Each dataset records `metadata/split_migration.json`; fixed-seed verification
reconstructs the original records and applies the same merge. Explicit pilot hop
quotas preserve the original per-hop rounding rather than changing questions.
New generation produces only train/test files. Optional composition holdouts are
marked inside test and count toward its configured total.

The notebook selects `test_ids` once and uses the same ordered list for vanilla,
SFT, SFT + GRPO and independent GRPO only. Both RL variants use the same GRPO
pool, excluding the reserved SFT instances, and have matching hop schedules,
seeds, reward, decoding limits and budgets. The independent variant restores the
saved initial adapter onto the same frozen quantized base and verifies an exact
value fingerprint before creating its new optimizer. SFT + GRPO still requires
verified SFT; each full phase requires its own verified smoke.

`training/artifacts.py` persists per-variant metrics, raw/per-question rewards
and standalone HTML cards immediately after evaluation. It consolidates each RL
phase's live diagnostics into episodic returns, raw/HTML trajectories and PNG
curves on both success and failure. `training/experiments.py` orchestrates the
same smoke/full path for both RL variants, with separate checkpoint directories
and directly streamed diagnostic logs. Evaluation now flushes completed raw
trajectories alongside per-question rewards, preserving both after interruption.

Artifacts live under `RUN_ROOT/variants/{vanilla,sft,sft_grpo,grpo_only}/`.
`comparison.json` is refreshed after every completed variant; incomplete results
have no aggregate score. The final notebook code cell only ZIPs these saved
artifacts and displays a link, without requiring model or trainer state. Large
optimizer checkpoints and Hub caches remain excluded. See README for the exact
layout and `notebooks/reports/four_variant_validation.json` for current checks.
Earlier JSON reports describe their respective historical implementations.
