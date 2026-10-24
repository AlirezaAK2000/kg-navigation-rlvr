# Synthetic knowledge graph navigation data

A deterministic, CPU-only dataset-generation and inspection pipeline for studying
multi-turn graph navigation. It produces a fictional academic ecosystem, typed
2/3/4-hop set-valued questions, private verifier metadata, leakage-aware splits,
statistics, and standalone interactive HTML inspectors. An optional Kaggle
notebook adds real multi-turn GRPO training with Qwen3 and NF4 QLoRA; its
GPU-dependent validation status is documented below.

Python **3.11+** is required. Generation, validation, conversion, and visualization
use only the standard library. `pytest` is the only development dependency. The
same graph loads on Kaggle without a GPU, external database, or network access.

## Included outputs

- `data/`: full dataset, **2,000 train + 300 validation + 500 test**.
- `pilot_data/`: **100 questions**, split 70/10/20, using the same full-sized graph.
- `visualizations/`: overview, entity neighborhood, and question inspectors for
  all three hop categories, including a multi-answer example.
- `data/metadata/samples.md`: ten seeded random questions with complete answers,
  reference paths, and fresh executable verification.
- `data/metadata/report.md` and `statistics.json`: readable and machine-readable reports.
- `data/metadata/validation.json`: independent validation and regeneration checks.
- `conversations/`: public, prompt-only conversational JSONL exports.

Open `visualizations/index.html` in a browser to select an inspection mode.
See `VERIFICATION.md` for completed checks and the remaining browser inspection
blocked by this session's unavailable computer-use permission.

## Reproduce

Run from the repository root. Use fresh output directories: the generator refuses
to overwrite an existing nonempty dataset. The checked-in artifacts already use
`data` and `pilot_data`, so these commands use new directory names.

```bash
python3.11 -m pip install -r requirements.txt  # only needed for pytest
python3.11 -m pytest -q

python3.11 scripts/generate_dataset.py --config configs/pilot.json --output pilot_reproduced
python3.11 scripts/validate_dataset.py --data pilot_reproduced --check-reproducibility

python3.11 scripts/generate_dataset.py --config configs/default.json --output data_reproduced
python3.11 scripts/validate_dataset.py --data data_reproduced --check-reproducibility
python3.11 scripts/sample_questions.py --data data_reproduced --count 10 --seed 41
python3.11 scripts/convert_conversations.py --data data_reproduced --output conversations_reproduced
python3.11 scripts/inspection_bundle.py --data data_reproduced --output visualizations_reproduced
```

Validation returns a nonzero exit code on any failure. Fixed-seed regeneration
compares all entities, triples, questions, answers, paths, templates, difficulty
metadata, IDs, and split assignments. The independent validator also checks the
exported catalog and split filenames. It updates the validation section of the
saved reports. Reproduction is scoped to the same source/catalog version and
configuration; there are no timestamps or environment-dependent graph facts.

Direct imports work after `python3.11 -m pip install -e .`, or by adding `src` to
`PYTHONPATH`. The scripts bootstrap their own import path and can be invoked from
any working directory with appropriate input/output paths.

## Data layout

```text
data/
  config.json                    # fully resolved configuration
  graph/
    entities.json                # ID -> {name, type}
    relations.json               # typed ontology, cardinalities and constraints
    triples.tsv                  # headerless head<TAB>relation<TAB>tail
  rl/
    train.jsonl                  # PRIVATE records, including verifier metadata
    validation.jsonl
    test.jsonl
  metadata/
    query_programs.jsonl         # reviewed program/template catalog
    statistics.json
    report.md
    validation.json
    samples.md
```

All paths in records are entity IDs. Relations and inverse operation IDs remain
separate from readable names. Source modules are under `src/kg_rlvr/`; command
line entry points are under `scripts/`. Configurations are JSON to avoid a YAML
dependency.

## Graph construction and ontology

The default graph has **2,000 entities, exactly 5,000 stored triples, eight entity
types, and 16 stored relation types**. The default population is 700 researchers,
100 universities, 300 projects, 400 publications, 150 fields, 100 funders, 200
cities, and 50 countries. Every name is constructed from fictional syllables and
a unique within-type suffix; no real-world facts or name lists are imported.

The backbone covers geography, employment, specialties, leadership,
contributions, sponsorship, authorship, publication provenance, and an acyclic
field hierarchy. A collaboration spine joins representatives of all universities.
Publications have an author who leads their producing project and a field
investigated by that project. Every leader contributes to the led project.
Additional constrained edges supply coauthors, contributors, funders, research
fields, collaborations, university partnerships, and citations. Extra triples
are not drawn independently of these rules.

Coverage assignments are balanced then **independently shuffled**, so numeric
entity/name suffixes do not encode employer, city, country, funder, or producing
project IDs. This reduces simple arithmetic shortcuts without claiming to
prevent fact memorization. The same seed/configuration preserves IDs and names.

Cardinalities below are from head to tail; “many:one” means a functional
relation. Mandatory participation is independent of cardinality. Full logical
constraints are exported in `relations.json` and checked by `graph_errors`.

| Relation | Domain → range | Cardinality | Participation / constraint |
|---|---|---|---|
| works_at | researcher → university | many:one | Exactly one employer |
| led_by | project → researcher | many:many | At least one leader, all contribute |
| authored_by | publication → researcher | many:many | At least one author, includes a project leader |
| investigates | project → field | many:many | At least one field, includes a leader specialty |
| funded_by | project → funder | many:many | At least one funder |
| located_in_city | university → city | many:one | Exactly one city |
| city_in_country | city → country | many:one | Exactly one country |
| collaborates_with | researcher → researcher | many:many | Symmetric, no self edges |
| contributes_to | researcher → project | many:many | Includes all leaders |
| subfield_of | field → field | many:one | Optional parent; acyclic forest |
| produced_by | publication → project | many:one | Exactly one producing project |
| publication_field | publication → field | many:one | Exactly one field, shared by its project |
| based_in | funder → city | many:one | Exactly one home city |
| specializes_in | researcher → field | many:one | Exactly one primary specialty |
| cites | publication → publication | many:many | Earlier-ID publications only, acyclic |
| partners_with | university → university | many:many | Symmetric, no self edges |

All relations except the two symmetric ones permit inverse traversal, with
operation IDs such as `inv:works_at`. Symmetric relations store both directions
and use their forward operation in either direction; their `inv:` aliases are
forbidden. The **5,000 triple count includes reciprocal symmetric triples**;
inverse adjacency entries for other relations are derived, not stored triples.
Distinct IDs avoid polymorphic `located_in` ambiguity. Cardinality limits and
hierarchy/citation constraints also apply when inverse navigation is enabled.

## Navigation contract

```python
from kg_rlvr.graph_index import GraphIndex, NavigationSession, execute

graph = GraphIndex.load("data/graph")
graph.get_entity("researcher_000042")        # {id, name, type}
graph.get_relations("researcher_000042")     # sorted available operation IDs
graph.get_neighbors("researcher_000042", "works_at")  # all sorted neighbor IDs

session = NavigationSession(graph, "researcher_000042")
university = session.get_neighbors("researcher_000042", "works_at")[0]
session.get_relations(university)             # newly discovered entity
session.get_relations("researcher_000042")    # previous entities stay accessible
session.submit_answer([university])           # canonical ID envelope only

result = execute(graph, "researcher_000042", ["works_at", "located_in_city"])
print(result.answers, result.paths, result.difficulty)
```

Unknown entities/relations, forbidden inverses, ill-typed operations, and access
to undiscovered session entities raise `ValueError`. A valid but absent relation
returns `[]`. Duplicate submitted IDs are canonicalized. `submit_answer` validates
IDs/discovery only; it provides **no correctness judgment, score, or reward**.
Neighbor responses are complete and never silently truncated.

The reference executor performs `get_relations` on each frontier entity, then
`get_neighbors` if the desired operation is available. It unions all results,
retains one deterministic witness per answer, counts all program-conforming
walks, and finally calls `submit_answer`. It uses the same discovery gate intended
for future tool use. It knows the gold program and is solely a solvability check,
not a language-model baseline.

## Question construction and hop semantics

`question_templates.py` contains 26 typed relation compositions, each with three
versioned, manually reviewed natural-language templates. The catalog includes
forward and inverse traversal, citations, project leadership, employment,
funding, and research fields. Set/existential wording avoids implying uniqueness
when multiple answers exist. Some ontology relations currently serve only as
distractors; the report lists them explicitly.

Programs are executed first, yielding **complete answer sets**. Empty answers,
oversized answer sets, low branching, over-budget gold executions, and oversized
gold neighbor responses are rejected with explicit reason counts. No target is
silently reduced and no constraint is automatically relaxed. An impossible
request raises an error identifying the split, hop quota, and produced count.
An infeasible graph triple budget reports the mandatory backbone size.

Each eligible instance also undergoes an exhaustive shorter-program check over
**all permitted operations**, including programs absent from the template
catalog. Dynamic programming enumerates distinct reachable frontier sets at
depths zero through three. Any program returning the identical complete answer
set in fewer operations disqualifies the candidate. Equal frontiers are merged
because their future transitions are identical; empty frontiers are terminal.

This is a conservative **instance-level, extensional** shortcut screen. A
coincidental equality of answer sets can cause rejection even if meanings differ.
It does not establish minimality across arbitrary natural-language logical
queries, inferential rules, unions, intersections, or external knowledge.

The schema deliberately separates:

- `hop_count`: number of operations in the intended executable program.
- `shortest_equivalent_program_length`: the accepted program length after the
  exhaustive smaller-answer-set-program screen.
- `shortest_path_hops`: unlabelled shortest navigable distance to each answer;
  may be smaller because a different relation sequence may reach just that node.
- `difficulty.reference_tool_calls`: actual calls of the specified gold executor,
  including relation discovery and final submission. Calls to fetch entity names
  or an uninformed agent's exploration are not included. This is a configured
  **reference-execution** budget, not a guarantee for every possible agent.

Difficulty also records mean navigation degree, mean available relation count,
mean query-relation fanout, frontier widths, maximum neighbor response, and
distractor edge occurrences at visited frontier nodes. Distractor counts can
count an edge again if its node is revisited at another depth. Path counts refer
to program-conforming walks; one reference path is not asserted to be unique.

## Splitting and configuration

`config.py` defines all defaults. Override selected values in a JSON file; unknown
keys are rejected. Entity/triple counts, explicit per-type populations, seeds,
leaders, coauthor/field/funder caps, collaboration degree targets, extra-relation
weights, question counts, hop weights, branching/answer/response/tool budgets,
paraphrase counts, splitting policy, and visualization limits are configurable.
Entity counts scale the default proportions unless `graph.type_counts` is given.
At least 100 entities are supported; explicit populations must meet documented
coverage constraints and fit the requested coherent backbone budget.
`collaboration_degree_target` is an upper cap on per-researcher collaboration
fanout; it does not guarantee that every researcher reaches that degree. Overall
degree density is controlled by entity/triple counts and extra-relation weights.

```json
{
  "seed": 2027,
  "graph": {"entity_count": 3000, "triple_count": 8500},
  "questions": {
    "split_counts": {"train": 2000, "validation": 300, "test": 500},
    "hop_weights": {"2": 1, "3": 1, "4": 1},
    "min_branching_factor": 3.0,
    "max_answer_set_size": 6,
    "max_tool_calls": 60,
    "max_neighbors_per_response": 32
  }
}
```

Hop quotas use largest-remainder allocation separately per split. By default,
up to two paraphrases share a query instance; a quota's final group may contain
one. Instances are keyed by starting entity, complete relation program, and
sorted complete answer set. A group belongs to exactly one split, and identical
question text is rejected even within a split.

- **Default `instance` strategy:** tests new query instances on the same graph.
  Entities, facts, templates, relations and compositions can overlap. It does
  not prevent general fact memorization or template exploitation.
- **`start_entity` strategy:** starting IDs are deterministically assigned to
  disjoint ordinary splits. Intermediate/answer entities, facts, and templates
  may still overlap. See `configs/entity_split.json`.
- **Optional `composition_test`:** reserves whole relation compositions, with
  all their templates, from every ordinary split. This tests composition
  generalization, while entities and individual relations may overlap. Start
  IDs may overlap the optional holdout even under `start_entity`. See
  `configs/composition_holdout.json`; it adds 150 questions beyond the base 2,800.

```bash
python3.11 scripts/generate_dataset.py --config configs/composition_holdout.json --output composition_data
python3.11 scripts/validate_dataset.py --data composition_data --check-reproducibility
```

`statistics.json` reports exact shared composition paths, starting-entity overlap,
reference-entity overlap, and operation overlap for every split pair. Additional
statistics cover degree, components, answer sizes, branching, tool calls, path
multiplicity, templates, relations, duplicates, rejection reasons, and warnings.

## Interactive inspection

These are standalone, offline HTML files. Double-click a file, drag it into a
browser, or use your browser's Open File command. No server, CDN, database,
JavaScript package installation, or frontend framework is needed. A small native
SVG renderer is used instead of PyVis to keep offline packaging self-contained.
`inspection_bundle.py` reproducibly creates the index and all modes, choosing
examples with the largest complete answer sets for each hop category.

```bash
# Mode A: bounded, representative overview
python3.11 scripts/visualize.py --data data --max-nodes 120 --max-edges 220

# Mode B: local neighborhood, with optional type and relation filters
python3.11 scripts/visualize.py --data data --entity researcher_000042 --radius 2
python3.11 scripts/visualize.py --data data --entity researcher_000042 --radius 1 \
  --entity-types university --relation-types works_at --output visualizations/employment.html

# Mode C: question (replace the ID with any ID from the saved question files)
python3.11 scripts/visualize.py --data data --question-id q_000001 --paths-per-answer 3
```

The inspectors support mouse-wheel zoom, toolbar zoom/fit, background panning,
node dragging, name/ID search, node inspection, type/relation filters, and optional
edge labels. Colors identify entity types. Rings distinguish start, intermediate,
and answer nodes; cyan edges highlight the reference traversals. Hover over an
edge or node for its full label. Inverse highlighted edges are drawn in traversal
direction and explicitly labeled `inv:...`; context edges use stored directions.

Question mode displays the question, program, answer IDs, hop count, shortest
distances, branching and tool statistics. The path selector includes witnesses
for **every answer**, with up to three actual paths per answer by default. It
reports total program-conforming walk counts and does not imply that displayed
witnesses exhaust all solutions. Reference paths survive context filters and
limits at generation time; insufficient limits raise an error rather than omit
an answer. Interactive filters intentionally let the inspector hide nodes/edges.

The overview is a breadth-first bounded projection, not a uniform sample of graph
statistics; use the full-data report for those. Neighborhood radius and limits
bound context. Edge caps can prune isolated context nodes; every gold path is
retained. HTML embeds a derived view of the canonical saved graph, not a second
editable graph store. The Python visualizer always reads the saved input files.

## Hugging Face / future TRL integration

```bash
python3.11 scripts/convert_conversations.py --data data --output conversations
```

Every exported row has only `prompt`, containing system/user chat messages.
The user message contains the question and public start ID/name/type. Public
instructions describe the navigation tools. **No gold answer, query program,
template ID, hop label, witness path, or difficulty record is copied.** Keep the
original `rl/` records private to a future verifier; do not pass them wholesale
to a model. Correspondence with verifier records is by split and row order.

If Hugging Face `datasets` is installed in a future training environment:

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files={
    "train": "conversations/train.jsonl",
    "validation": "conversations/validation.jsonl",
    "test": "conversations/test.jsonl",
})
assert dataset["train"].column_names == ["prompt"]
```

This utility creates a conversational dataset artifact only. The separate GRPO
notebook uses a single-submitted-answer task and its own allowlisted prompts,
graph environment, binary verifier, and multi-turn rollout adapter.

## Verification coverage and limitations

Tests cover graph determinism and constraints, inverse traversal, discovery
gating, complete multi-answer execution, convergent alternative paths, all hop
categories, semantic template linkage, shorter-program enumeration, grouped
splitting, unseen compositions, disjoint starting IDs, corruption detection,
impossible constraints, prompt isolation, export/reload/regeneration, and all
visualization modes. Semantic correspondence ultimately rests on the reviewed
template definitions: automatic checks enforce exact binding to those definitions
and cannot prove that a newly authored English template expresses its program.

The finite grammar and fixed shared graph remain potential shortcuts. For stronger
future evaluations, use new graph seeds and independently reviewed novel
compositions/wording, and compare navigation behavior rather than answer accuracy
alone. High branching here measures available choices, not a validated empirical
difficulty scale for a language model.

## Exploratory analysis notebook

Open [`notebooks/01_dataset_eda.ipynb`](notebooks/01_dataset_eda.ipynb) in Jupyter,
VS Code, or Kaggle and run all cells with Python 3.11+. The saved notebook includes
executed tables, 16 charts, five verified question examples per hop category,
interactive graph exports, quality checks, and conclusions based on the loaded
data. Its size is about 2 MB.

The core generation pipeline remains stdlib-only. Install the optional analysis
dependencies separately:

```bash
python3.11 -m pip install -r requirements-eda.txt
# Optional live notebook controls:
python3.11 -m pip install 'ipywidgets>=8,<9'
```

For headless execution with the same interpreter and saved outputs:

```bash
python3.11 scripts/run_eda_notebook.py
# Alternate dataset, writable output location, and no widget dependency:
python3.11 scripts/run_eda_notebook.py --data pilot_data \
  --output notebooks/pilot_eda.ipynb --output-base /tmp/pilot_eda --no-widgets
```

The first notebook cell exposes `PROJECT_ROOT_OVERRIDE`, `DATA_DIR_OVERRIDE`,
`OUTPUT_BASE_OVERRIDE`, seed, sampling, validation, and display limits. Environment
variables `KG_RLVR_PROJECT_ROOT`, `KG_RLVR_DATA_DIR`, and
`KG_RLVR_EDA_OUTPUT_DIR` are equivalent path overrides. The runner sets these
variables and starts a local Jupyter kernel; it needs permission for local
loopback sockets. It never invokes graph generation or the validator CLI that
writes into `data/`.

On Kaggle:

1. Upload/mount the repository including `src/kg_rlvr/` and the existing saved
   graph/QA files. No editable installation or internet connection is required.
2. Import the notebook and select a Python 3.11+ CPU environment. Set
   `PROJECT_ROOT_OVERRIDE` to the mounted directory containing `src/`, and set
   `DATA_DIR_OVERRIDE` if graph data is mounted separately.
3. Keep outputs under `/kaggle/working/kg_rlvr_eda/` (the automatic Kaggle default),
   then run all cells. The first import cell lists installed package versions;
   missing analysis dependencies must be available in the notebook environment.
4. Trust the notebook to use embedded HTML/JavaScript. If the frontend strips
   iframe scripts or does not support widgets, use `ENTITY_ID`, `ENTITY_RADIUS`,
   and `QUESTION_ID`, rerun their cells, and download/open the standalone HTML
   exports from the Kaggle output panel. A running kernel is needed for widget
   callbacks; exported HTML runs independently.

Local derived artifacts are saved separately under `visualizations/eda/`:

```text
figures/    # 16 PNG charts, including a static bounded graph preview
html/       # offline overview, entity neighborhood, question-path inspectors
reports/    # summary.md, quality checks, split overlaps, question metrics,
            # execution_report.json with input SHA-256 fingerprints
```

The notebook reuses `GraphIndex`, the discovery-gated reference executor, the
existing in-memory validator, and the existing SVG/HTML renderer. Small reusable
analysis helpers live in `src/kg_rlvr/eda.py`. The input dataset is fingerprinted
before and after execution, including existing metadata reports. Output paths
inside the dataset are refused. Missing optional metadata is reported; difficulty
and paths are recomputed in memory where possible. The full catalog validator is
marked unavailable if its required verifier metadata is absent.

Leakage analysis separates exact query/text collisions from expected shared
entities, answers, grammar, and compositions. A bounded, seeded lexical candidate
screen avoids all-pairs comparisons and explicitly does not claim exhaustive
paraphrase detection. Tool-call plots show measured gold-executor calls, not an
invented estimate of model exploration. Browser rendering cannot be proved by
headless Python execution: the notebook tests the question-control callback and
exported answer/path coverage, while actual frontend rendering remains subject
to the browser's permissions and notebook trust policy.

## Kaggle GRPO training notebook

[`notebooks/02_grpo_training_kaggle.ipynb`](notebooks/02_grpo_training_kaggle.ipynb)
implements genuine TRL GRPO with real graph interaction, isolated episodes,
single-entity binary rewards, baseline/post-training tool-use evaluation,
checkpointing, and adapter export/reload. Training uses Qwen/Qwen3-1.7B, NF4
double quantization, LoRA rank 8 / alpha 16 / dropout 0, non-reentrant gradient
checkpointing, microbatch 1 and four generations per question. T4 computation
uses FP16; emulated BF16 is explicitly excluded. No SFT, reward shaping, vLLM,
or LLM judge is implemented.

**Validation status:** non-GPU notebook cells, CPU graph/environment tests, and
the pinned TRL/Qwen-tokenizer integration were executed locally. A tiny random
CPU Qwen architecture is used only for an adapter serialization test. The
pretrained model's real tool use, CUDA NF4 loading, genuine GRPO optimizer step,
T4 peak memory, full checkpoint resumption, and trained-model reload/evaluation
remain **unverified on Kaggle**. The notebook's smoke gate will refuse to claim
success unless LoRA tensors actually change and full checkpoint/metric checks
pass. See [the implementation report](docs/grpo_training_report.md).

Use a fresh Kaggle Python 3.11/3.12 notebook:

1. Attach the repository (including `src/kg_rlvr`, `requirements-grpo.txt`, and
   saved `data/`) as an input dataset. Separate graph/QA mounts are supported.
2. Import the notebook, select a T4 GPU and enable internet for dependencies and
   the pinned model download. The notebook exposes only one GPU; two T4 memories
   are not pooled.
3. Set `PROJECT_ROOT_OVERRIDE` and `DATA_ROOT_OVERRIDE` in the configuration cell.
   Outputs default to `/kaggle/working/` without a hardcoded dataset slug.
4. If the pinned packages are absent, set `INSTALL_DEPENDENCIES=True` and run
   setup. It installs Torch 2.6.0 from the official cu124 index followed by
   `requirements-grpo.txt`. In this fresh language-model session it optionally
   removes unused audio/vision packages whose compiled extensions can conflict
   with Torch. Restart if instructed, set installation back to `False`, then
   Run All.
5. Defaults evaluate 30 validation questions and attempt two genuine GPU smoke
   steps. `RUN_FULL_TRAINING=False` prevents an automatic long experiment. Enable
   it explicitly for the initial 30 full-phase updates. Training defaults to
   hops `[2, 3]`; `[4]` or any combination remains configurable. Evaluation
   defaults to `[2, 3, 4]`.
   To follow a completed smoke-only Run All with full training, restart with a
   new `RUN_NAME` and enable full training before Run All; export/reload has
   already released the previous trainers and made that adapter inference-only.
6. Inspect actual rewards, malformed calls, submissions, truncations, homogeneous
   groups and LoRA deltas. Sparse binary rewards can produce no learning signal;
   an all-zero/all-one smoke run is diagnosed rather than reported as success.
7. Download the adapter/diagnostics ZIP or save a Kaggle version with outputs.
   Save trainer checkpoints separately if you need resumption across sessions.

The pinned training stack is Torch **2.6.0/cu124**, Transformers **5.3.0**, TRL
**1.8.0**, PEFT **0.18.1**, Accelerate **1.10.1**, bitsandbytes **0.49.2**,
datasets **4.8.3**, with the remaining pins in `requirements-grpo.txt`.
PEFT 0.19.0 was rejected after its real adapter construction failed on Torch 2.6;
the selected 0.18.1 adapter path was checked instead. Generation/inspection
dependencies stay independent of the optional GPU stack.

The rollout integration deliberately uses `rollout_func` rather than
`environment_factory`: the inspected TRL 1.8.0 native tool loop does not consult
our episode's submission state before generating again. The custom adapter stops
at submission/budget limits and passes `env_mask` into TRL. Model token IDs are
preserved, observation/prefill tokens are masked from the policy objective, and
all GRPO loss/advantage/optimizer code remains in TRL. The release's RepeatSampler
already repeats each prompt four times; assertions prevent repeating it twice or
mixing questions in a comparison group. A single GPU, accumulation 4, microbatch
1 and G=4 means one question/group and four trajectories per optimizer update.

Only question/start information, tool schemas, public rules and real observations
reach the model. The binary verifier reads the environment's actual submitted
ID, independently of generated claims of correctness. Returning any one member
of the complete gold set earns 1; wrong/no submissions earn 0. This is distinct
from the earlier complete-set conversational export. Paths and tools logs are
private diagnostics, never policy prompt fields.

Outputs on Kaggle:

```text
/kaggle/working/
  checkpoints/<RUN_NAME>/smoke/   # real smoke checkpoints + raw rollout/trainer logs
  checkpoints/<RUN_NAME>/full/    # full trainer checkpoints, optimizer/scheduler/RNG
  runs/<RUN_NAME>/                # fixed task IDs, metrics, figures, evaluations,
                                 # actual trajectories and run_evidence.json
  grpo_adapter/                   # adapter weights, tokenizer and training manifest
  <RUN_NAME>_adapter_and_diagnostics.zip
  hf_cache/                       # downloaded base model cache, excluded from ZIP
```

`RESUME_FROM_CHECKPOINT` accepts a complete trainer checkpoint with the original
`run_manifest.json`; model, versions, source hashes, training rows and rollout
settings must match. Full-phase `INITIAL_TRAINING_STEPS` is an absolute target.
Adapter-only exports do not restore optimizer/scheduler/RNG state. The constant
scheduler permits extending the target, but bitwise-identical continuation is not
promised across CUDA sessions. Use a new `RUN_NAME` for unrelated experiments.

Local CPU preflight (does not install training packages or download model weights):

```bash
python3.11 -m pytest -q
python3.11 scripts/run_grpo_notebook_checks.py
```

The CPU runner refuses a CUDA host to avoid accidentally starting the notebook's
GPU smoke phase. Optional tests in `tests/test_grpo_pinned_api.py` run only when
the exact training API stack is installed; the tokenizer tests additionally use
`KG_GRPO_TOKENIZER` pointing to a downloaded tokenizer-only directory. No hidden
environment variable is needed for normal notebook training: user configuration
is in its first code cell.

On OOM, explicitly lower response/context lengths, reduce optional memory use and
then adjust call/group budgets while preserving GRPO divisibility. If needed,
set `MODEL_NAME="Qwen/Qwen3-0.6B"`; the matching immutable revision is printed.
The notebook never silently substitutes a smaller model. Re-run the baseline
with the final limits so the post-training comparison remains meaningful.
