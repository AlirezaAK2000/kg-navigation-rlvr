# GRPO implementation and verification report

The Kaggle notebook and supporting implementation are ready for a GPU smoke
attempt. **They have not been executed on Kaggle. No genuine GRPO optimizer
update, pretrained-model tool use, or T4 memory measurement was verified on the
development host.** Local execution used an Apple Silicon CPU with no CUDA.
The saved notebook outputs explicitly record these limits.

## Files

| File | Purpose |
|---|---|
| `notebooks/02_grpo_training_kaggle.ipynb` | All 13 requested sections; saved outputs from CPU preflight |
| `src/kg_rlvr/training/environment.py` | Existing graph access, isolated discovery state, typed tools, public prompts, private answer verifier |
| `src/kg_rlvr/training/rollouts.py` | Real Transformers generation, multi-turn orchestration, token preservation, observation masks, evaluation and logs |
| `src/kg_rlvr/training/grpo.py` | Pinned TRL/QLoRA construction, metrics, genuine-update checks, checkpoints, export/reload and observed-trajectory HTML |
| `src/kg_rlvr/training/__init__.py` | Optional training package |
| `requirements-grpo.txt` | Separate pinned optional training dependencies |
| `tests/test_training_environment.py` | CPU graph, reward, isolation, leakage, episode, token-mask and checkpoint checks |
| `tests/test_grpo_pinned_api.py` | Optional tests against the real pinned TRL, Qwen tokenizer and PEFT APIs |
| `scripts/run_grpo_notebook_checks.py` | Execute notebook CPU sections with the selected Python interpreter |
| `notebooks/reports/grpo_api_validation.json` | Machine-readable CPU API evidence and explicit unverified GPU flags |
| `artifacts/grpo/runs/kg_grpo_seed41/` | Actual local configuration, selected task IDs, memory snapshot and execution evidence |
| `README.md`, `src/kg_rlvr/__init__.py` | Usage, optional-training scope and limitations |

No source graph or question file was changed. All 24 files across `data/` and
`pilot_data/` were compared with their initial SHA-256 fingerprints after the
final checks. The notebook additionally fingerprints its selected dataset before
and after execution. It reuses `GraphIndex`, `NavigationSession`, the existing
question reader, in-memory validator and SVG/HTML visualization template.

## Selected stack

| Library | Pin | Local verification |
|---|---|---|
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
SFT, reward models and an LLM judge are not installed by this workflow.
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

AdamW uses learning rate 5e-6, constant scheduling, no weight decay and gradient
clipping at 1. The TRL objective is explicitly `loss_type="grpo"`, group-scaled
rewards, `num_iterations=1`, and `beta=0`. No KL/reference-model copy is needed;
KL is unavailable, not reported as a measured zero.

Microbatch 1, accumulation 4, one GPU and G=4 give four trajectories for one
question per optimizer update. Accumulation 8 gives two question groups. The
generation batch equals accumulation, which must be a positive multiple of G.
Defaults are two smoke updates and an opt-in 30-update full phase.
Training hops default to `[2, 3]`; any combination of `[2, 3, 4]` is allowed.
Validation and test default to 30 separately selected questions each, balanced
across the configured evaluation hops. Fixed IDs are saved before model loading.

Limits default to 8 tool calls, 128 model tokens per turn, 1,536 total completion
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
wrong or missing submissions earn 0. There are no intermediate, efficiency,
invalid-action or path rewards. Acknowledgments do not expose correctness.

## Verification performed

| Check | Result |
|---|---|
| Full local pytest suite | **72 passed, 3 skipped**; skips are optional pinned-stack tests in the ordinary environment |
| Separate pinned-stack API suite | **3 passed**; real TRL config/sampler/mask boundary, real Qwen BPE template, tiny CPU PEFT adapter save/reload |
| Notebook execution | **14 code cells, zero error outputs**, CPU preflight using a real Jupyter kernel |
| Graph and QA loading | Existing 2,000 entities, 5,000 triples and 2,800 questions validated in memory |
| Source preservation | All 24 original data/pilot files unchanged |
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
binary-reward group has zero advantage; if all sampled groups are homogeneous,
the notebook fails the update gate and records the failure. Two steps can be
insufficient for sparse rewards. Diagnose real logs before increasing smoke
sampling or changing public limits; full training does not follow a failed gate.

## Exact Kaggle execution procedure

1. Upload the repository as a Kaggle input dataset, retaining `src/kg_rlvr/`,
   `requirements-grpo.txt`, and the saved `data/graph`, `data/rl`, configuration
   and metadata files. A separate graph/QA input mount is also supported.
2. Create a Kaggle Python 3.11/3.12 notebook and import
   `notebooks/02_grpo_training_kaggle.ipynb`. Select GPU T4 acceleration and enable
   internet for installation and the pinned model download. Only one T4 is used.
3. In the configuration cell, set `PROJECT_ROOT_OVERRIDE` to the directory with
   `src/`, and `DATA_ROOT_OVERRIDE` to the existing dataset directory. Keep working
   outputs under `/kaggle/working/`. No dataset slug is hardcoded.
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
7. Run post-training validation/test evaluation, export/reload and trajectory
   cells. Inspect measured base-versus-trained comparisons without assuming
   improvement. The final cell packages adapter and diagnostics for download.
8. Save a Kaggle version with outputs or download the ZIP. Download full trainer
   checkpoints separately if continued training is needed; the ZIP excludes them
   and the large base-model cache.

Outputs are `/kaggle/working/checkpoints/<RUN_NAME>/{smoke,full}/`,
`/kaggle/working/runs/<RUN_NAME>/`, `/kaggle/working/grpo_adapter/`, and
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
