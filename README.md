# Knowledge graph navigation with verifiable rewards

A deterministic, CPU-only dataset-generation and inspection pipeline for studying
multi-turn graph navigation. It produces a fictional academic ecosystem, typed
1/2/3/4-hop set-valued questions, private verifier metadata, leakage-aware splits,
statistics, and RDF/Turtle graph and ontology files for Protégé. An optional Kaggle
notebook adds real multi-turn GRPO training with Qwen3 and NF4 QLoRA; its
GPU-dependent validation status is documented below.

Only `train` and `test` are dataset splits. The split-only migration preserves all
3,734 questions and their content; the 400 former validation questions now belong
to test. The pilot similarly contains 94 train and 41 test questions. Fixed-seed
checks replay the recorded original configuration and the merge to verify this
preservation. `hop_counts_by_split`, when set, records exact hop quotas (used by
the pilot to preserve rounding from its former splits).

Python **3.11+**, **RDFLib 7.6.0**, and **pyparsing 3.2.3** are required. The graph backend uses RDFLib
for triple storage, parsing, serialization and forward/inverse lookup. `pytest`
is used for development tests. A saved graph loads locally or on Kaggle without
a GPU, external database, or network access once dependencies are installed.

## Included outputs

- `data/`: full dataset, **2,667 train + 1,067 test** (3,734 total).
- `pilot_data/`: **135 questions**, split 94 train / 41 test, using the same full-sized graph.
- `data/graph/graph.ttl`: self-contained OWL ontology, labeled individuals and graph facts.
- `data/metadata/samples.md`: ten seeded random questions with complete answers,
  reference paths, and fresh executable verification.
- `data/metadata/report.md` and `statistics.json`: readable and machine-readable reports.
- `data/metadata/validation.json`: independent validation and regeneration checks.
- `data/metadata/split_migration.json`: provenance for merging the former validation
  questions into test while retaining every question ID and the original train file.
- `conversations/`: public, prompt-only conversational JSONL exports.

Open **`data/graph/graph.ttl` in Protégé** to inspect classes, properties and
individuals. See `VERIFICATION.md` and [the RDF migration report](docs/rdf_migration_report.md)
for validation evidence.

## Reproduce

Run from the repository root. Use fresh output directories: the generator refuses
to overwrite an existing nonempty dataset. The checked-in artifacts already use
`data` and `pilot_data`, so these commands use new directory names.

```bash
python3.11 -m pip install -r requirements.txt
python3.11 -m pytest -q

python3.11 scripts/generate_dataset.py --config configs/pilot.json --output pilot_reproduced
python3.11 scripts/validate_dataset.py --data pilot_reproduced --check-reproducibility

python3.11 scripts/generate_dataset.py --config configs/default.json --output data_reproduced
python3.11 scripts/validate_dataset.py --data data_reproduced --check-reproducibility
python3.11 scripts/sample_questions.py --data data_reproduced --count 10 --seed 41
python3.11 scripts/convert_conversations.py --data data_reproduced --output conversations_reproduced
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
    graph.ttl                    # OWL schema + individuals + asserted graph edges
  rl/
    train.jsonl                  # PRIVATE records, including verifier metadata
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
constraints are exported as annotations in `graph.ttl` and checked by `graph_errors`.

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
inverse traversal for other relations queries incoming edges without adding stored triples.
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

`question_templates.py` contains 46 typed relation programs, each with three
versioned, manually reviewed natural-language templates. The catalog includes
forward and inverse traversal, citations, project leadership, employment,
funding, and research fields. Set/existential wording avoids implying uniqueness
when multiple answers exist. The one-hop catalog includes all 16 forward
relations and four inverse operations; reports show the coverage actually sampled.

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
paraphrase counts and splitting policy are configurable.
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
    "split_counts": {"train": 2667, "test": 1067},
    "hop_weights": {"1": 1, "2": 1, "3": 1, "4": 1},
    "min_branching_factor": 3.0,
    "max_answer_set_size": 6,
    "max_tool_calls": 60,
    "max_neighbors_per_response": 32
  }
}
```

Hop quotas use largest-remainder allocation separately per split. The checked-in
one-hop extension adds 934 questions (667 train, 267 now in test) while
preserving every original 2–4-hop row, ID, split assignment and graph byte. Its
20 reviewed one-hop programs cover all 16 forward relations and four inverse
operations, with three language templates per program. New one-hop rows are
appended after the original question IDs; independent RNG streams preserve the
multi-hop sample when its quotas stay unchanged. The pilot similarly adds 35
one-hop questions. Existing conversational prompt rows remain unchanged prefixes.

`hop_weights` replaces the default hop mapping: use `{"1": 1}` for one-hop-only
generation or `{"2": 1, "3": 1, "4": 1}` for the previous multi-hop selection.
No graph or reward shaping changes are needed to generate one-hop tasks.
 By default,
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
- **Optional composition holdout:** reserves whole relation compositions, with
  all their templates, from the ordinary training/test sample. These questions
  live in `test.jsonl` with `evaluation_category="composition_holdout"`. This tests composition
  generalization, while entities and individual relations may overlap. Start
  IDs may overlap the optional holdout even under `start_entity`. See
  `configs/composition_holdout.json`; it reserves 150 of the configured test questions.
  The output still has only train and test splits, with unchanged total quotas.

```bash
python3.11 scripts/generate_dataset.py --config configs/composition_holdout.json --output composition_data
python3.11 scripts/validate_dataset.py --data composition_data --check-reproducibility
```

`statistics.json` reports exact shared composition paths, starting-entity overlap,
reference-entity overlap, and operation overlap for every split pair. Additional
statistics cover degree, components, answer sizes, branching, tool calls, path
multiplicity, templates, relations, duplicates, rejection reasons, and warnings.

## RDF graph and Protégé inspection

Open `data/graph/graph.ttl` using **File → Open** in Protégé. The file embeds the
OWL ontology and all 2,000 individuals, readable labels, stable entity IDs and
5,000 asserted graph edges. It needs no external ontology imports. Use Classes,
Object properties and Individuals to inspect it. The ontology and graph data
are kept together in this single file. The custom HTML/SVG graph renderer and generated graph views have
been removed; EDA distributions and training-metric plots remain in the notebooks.

The namespace is `https://example.org/kg-navigation-rlvr/ontology#` (`kg:`), with
individuals under `https://example.org/kg-navigation-rlvr/entity/` (`entity:`).
These are identifiers for fictional data; loading does not dereference them.
Eight `owl:Class` declarations represent entity types. Relations are
`owl:ObjectProperty` declarations with domains/ranges, functional or symmetric
characteristics where applicable, inverse-property declarations, and annotations
preserving the generation constraints and permitted inverse traversal settings.
The inverse properties are schema declarations; the training backend queries
incoming asserted edges without materializing inferred facts.

`graph.ttl` is the single authoritative file for the ontology and graph data.
Edit this file when editing a
dataset in Protégé, save as Turtle, then validate it. The loader expects the
project's stable IDs and schema annotations. The validator checks the versioned
ontology and closed-world constraints; OWL domain/range and functional-property
axioms do not replace those checks. Automatic reasoning is not run in training.
The RDF file contains 13,232 triples in total: 5,000 navigation facts plus entity
metadata and ontology declarations. Navigation exposes only the 16 declared
relation IDs and their permitted inverse operations, excluding labels and schema
predicates. QA answers, query programs and reference paths stay in private JSONL.

```python
from rdflib import Graph
from kg_rlvr.rdf import ENTITY, KG

rdf = Graph().parse("data/graph/graph.ttl", format="turtle")
universities = list(rdf.objects(ENTITY.researcher_000042, KG.works_at))
researchers = list(rdf.subjects(KG.works_at, universities[0]))
```

`GraphIndex` preserves the existing string-ID tool API and uses RDFLib's
`objects`, `subjects` and `predicates` queries. `NavigationSession` retains
per-episode discovery state. Reference execution still counts every valid walk
and records witnesses, branching and tool usage. The backend no longer builds a
custom adjacency index. Existing JSON/TSV graph mounts remain readable as a
compatibility input; new generation and `GraphIndex.save()` write Turtle.

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
impossible constraints, prompt isolation, export/reload/regeneration, RDF round trips, OWL declarations, and RDFLib
property-path/SPARQL agreement. Semantic correspondence ultimately rests on the reviewed
template definitions: automatic checks enforce exact binding to those definitions
and cannot prove that a newly authored English template expresses its program.

The finite grammar and fixed shared graph remain potential shortcuts. For stronger
future evaluations, use new graph seeds and independently reviewed novel
compositions/wording, and compare navigation behavior rather than answer accuracy
alone. High branching here measures available choices, not a validated empirical
difficulty scale for a language model.

## Exploratory analysis notebook

Open [`notebooks/01_dataset_eda.ipynb`](notebooks/01_dataset_eda.ipynb) in Jupyter,
VS Code or Kaggle. It provides 15 statistical charts, five verified question
examples per hop category, schema/quality checks, leakage analysis and findings
computed from the saved dataset. Graph inspection uses Protégé and the Turtle
files described above.

```bash
python3.11 -m pip install -r requirements-eda.txt
python3.11 scripts/run_eda_notebook.py
# Alternate saved data and derived-output directory:
python3.11 scripts/run_eda_notebook.py --data pilot_data \
  --output notebooks/pilot_eda.ipynb --output-base /tmp/pilot_eda
```

Configuration exposes `PROJECT_ROOT_OVERRIDE`, `DATA_DIR_OVERRIDE`,
`OUTPUT_BASE_OVERRIDE`, seed, sampling and validation choices. Equivalent path
variables are `KG_RLVR_PROJECT_ROOT`, `KG_RLVR_DATA_DIR` and
`KG_RLVR_EDA_OUTPUT_DIR`. Outputs go under `artifacts/eda/{figures,reports}/`
locally or `/kaggle/working/kg_rlvr_eda/` on Kaggle. The runner starts a local
Jupyter kernel and needs local loopback sockets. It never regenerates source
data or invokes the validator CLI that writes reports into the input directory.

On Kaggle, mount the repository and saved graph/QA data, install the optional
analysis requirements, set the project/data overrides as needed, and run all
cells with Python 3.11+. Statistical NetworkX projections are used only for
connectivity/degree analysis; the navigation backend uses RDFLib.

The notebook fingerprints every input file before and after execution. Outputs
inside the dataset are refused. Optional metadata is recomputed in memory where
possible; the full validator is marked unavailable when required verification
metadata is missing. Near-duplicate screening is a bounded lexical heuristic,
not an exhaustive semantic comparison. Tool-call charts describe the gold
executor, not measured model exploration.

## Kaggle SFT and GRPO training notebook

[`notebooks/02_grpo_training_kaggle.ipynb`](notebooks/02_grpo_training_kaggle.ipynb)
implements assistant-only SFT followed by genuine TRL GRPO with real graph interaction, isolated episodes,
single-answer questions, correctness-plus-retrieval training rewards, baseline/post-training tool-use evaluation,
checkpointing, and adapter export/reload. Training uses Qwen/Qwen3-1.7B, NF4
double quantization, LoRA rank 8 / alpha 16 / dropout 0, non-reentrant gradient
checkpointing, microbatch 1 and eight generations per question. T4 computation
uses FP16; emulated BF16 is explicitly excluded. SFT uses the pinned Transformers
Trainer on the same LoRA adapter. No vLLM or LLM judge is implemented.

**Validation status:** non-GPU notebook cells, CPU graph/environment tests, and
the pinned TRL/Qwen-tokenizer integration were executed locally. A tiny random
CPU Qwen architecture verifies supervised optimizer updates, resumable checkpoint
saving and preservation of the updated adapter at the GRPO constructor boundary. The
pretrained model's real tool use, CUDA NF4 loading, genuine GRPO optimizer step,
T4 peak memory, full checkpoint resumption, and trained-model reload/evaluation
remain **unverified on Kaggle**. The notebook's smoke gate will refuse to claim
success unless LoRA tensors actually change and full checkpoint/metric checks
pass. See [the implementation report](docs/grpo_training_report.md).

Use a fresh Kaggle Python 3.11/3.12 notebook:

1. Import the notebook, select a T4 GPU and enable internet for GitHub,
   dependencies and the pinned model download. The notebook exposes only one
   GPU; two T4 memories are not pooled.
2. Leave `CLONE_REPOSITORY=True` and `PROJECT_ROOT_OVERRIDE=None`. On Kaggle the
   configuration cell clones `https://github.com/AlirezaAK2000/kg-navigation-rlvr.git`
   directly into the current folder, normally `/kaggle/working/`, with no
   `kg-navigation-rlvr/` subfolder. A nonempty working folder is handled using a
   temporary clone and a conflict-checked copy of its contents. Unrelated files
   are preserved; existing project code is reused on reruns without pulling.
3. Code and saved `data/` are obtained from GitHub. Set `DATA_ROOT_OVERRIDE` only
   for a separately attached dataset. Alternatively disable cloning and set
   `PROJECT_ROOT_OVERRIDE` to a mounted repository. Outputs default to
   `/kaggle/working/` without a hardcoded dataset slug.
4. If the pinned packages are absent, set `INSTALL_DEPENDENCIES=True` and run
   setup. It installs Torch 2.6.0 from the official cu124 index followed by
   `requirements-grpo.txt`. In this fresh language-model session it optionally
   removes unused audio/vision packages whose compiled extensions can conflict
   with Torch. Restart if instructed, set installation back to `False`, then
   Run All.
5. Defaults evaluate vanilla on 30 fixed test questions, train SFT for one pass
   over 200 verified demonstrations, evaluate SFT on the same questions, and
   attempt eight genuine GRPO smoke steps before evaluating SFT + GRPO. `RUN_FULL_TRAINING=False` prevents an automatic long experiment. Enable
   it explicitly for the initial 30 full-phase updates. Training defaults to
   hops `[1, 2, 3]`; any nonempty combination of `[1, 2, 3, 4]` is supported.
   Evaluation defaults to `[1, 2, 3, 4]`. `SINGLE_ANSWER_ONLY=True` removes questions
   with multiple final answers from every model phase before deduplication/caps,
   and uses reviewed singular wording. Original source data and split assignments
   are preserved; active records are exported under the run's `selected_dataset/rl/`.
   Eligible counts are 2,449 train and 983 test before deduplication.
   `TEST_EVAL_SIZE=30` selects the shared test subset once; `None` uses the full
   eligible unique test pool. There is no separate validation split.
   Actual smoke/full training exposure uses `TRAIN_HOP_EXPOSURE={1:0.2,2:0.4,3:0.4}`.
   The ordered schedule gives one/two/two question groups per five updates;
   the default 30-step full run has 6/12/12 groups and eight samples per group.
   Partial blocks are rounded by interleaving (eight smoke groups: 2/3/3).
   Questions within each hop are shuffled, cycling if needed. Update exposure
   weights alongside `TRAIN_HOPS` when changing categories. The tool-call budget
   is 12, allowing discovery at all four hops
   plus submission; token and response limits can still terminate an episode.
   To follow a completed smoke-only Run All with full training, restart with a
   new `RUN_NAME` and enable full training before Run All; export/reload has
   already released the previous trainers and made that adapter inference-only.
   To review vanilla-model performance first, run through section 6 and stop
   before continuing to the required SFT stage. The baseline runs
   with the LoRA adapter disabled; it is saved before any optimizer update.
6. Inspect actual rewards, malformed calls, submissions, truncations, homogeneous
   groups and LoRA deltas. Homogeneous combined rewards can produce no learning
   signal; such a smoke run is diagnosed rather than reported as success.
   The default smoke samples 64 episodes across eight groups. On failure it
   displays group rewards, trainer metrics and failed trajectories, and saves
   `smoke_diagnostics.json` under the run directory. The shared prompt emphasizes
   completing the relationships in the question and checking the answer type.
   After a prompt change, recompute the vanilla baseline with a fresh `RUN_NAME`.
7. Download the adapter/diagnostics ZIP or save a Kaggle version with outputs.
   Save trainer checkpoints separately if you need resumption across sessions.

The required supervised stage is configured with `SFT_EXAMPLES=200`,
`SFT_EPOCHS=1`, `SFT_LEARNING_RATE=2e-5`, and
`SFT_GRADIENT_ACCUMULATION_STEPS=8`. The 200 distinct demonstrations comprise
40 one-hop, 80 two-hop and 80 three-hop questions, drawn only from the source
training split. Complete query instances and their paraphrases are reserved for
SFT **before** applying `MAX_TRAIN_EXAMPLES` to GRPO. With the saved dataset, 733
of the original 933 unique one/two/three-hop instances remain available to GRPO.
Both GRPO smoke and full training use that remaining pool. Test
instances are excluded from both training stages. The graph is shared; separation
is by query instance rather than by graph entity or relation.

SFT targets are executable reference trajectories, not model-generated answers:
`get_relations` / `get_neighbors` at each hop, followed by a correct submission.
Every observation comes from the actual graph tools. Tokenization replays the
same Qwen codec and runtime limits used by model rollouts. Only assistant action
and EOS tokens receive loss; prompts, tool observations, injected prefixes and
padding are masked. No observations are fabricated, and no sequence is silently
truncated to make it fit. SFT must verify finite gradients, changed LoRA weights
and a complete checkpoint before GRPO begins. Its optimizer is released between
stages; the SFT adapter values are preserved when constructing GRPO.

The notebook saves `stage_partition.json`, separate `training_partitions/sft.jsonl`
and `grpo.jsonl`, and `sft/demonstrations.jsonl` under the run directory. It checks
stage separation by question ID, semantic query instance and public prompt, and
the GRPO constructor checks its pool against reserved SFT instances again.
`SFT_EXAMPLES` is independent of the GRPO example cap. The SFT adapter, losses,
update-verification result and SFT-only model evaluation are saved separately;
formatted SFT-only output inspection uses the same test questions as the
vanilla and SFT + GRPO models. The download ZIP includes SFT artifacts even for an
SFT-only run (`RUN_SMOKE_TEST=False`, `RUN_FULL_TRAINING=False`).

The notebook compares exactly **vanilla, SFT, and SFT + GRPO**. GRPO smoke and
full training require the verified SFT result and continue that adapter; the
GRPO-only switch has been removed. The default run name is
`kg_sft_grpo_seed41_train_test`. At one SFT epoch the default gives 25 supervised
updates, followed by eight GRPO smoke updates and, when enabled, 30 full GRPO
updates. The final comparison is saved as
`evaluation/vanilla_sft_sft_grpo_test_comparison.json`; individual stages use
`vanilla_test_*`, `sft_test_*`, and `sft_grpo_test_*` filenames.

The pinned training stack is Torch **2.6.0/cu124**, Transformers **5.3.0**, TRL
**1.8.0**, PEFT **0.18.1**, Accelerate **1.10.1**, bitsandbytes **0.49.2**,
datasets **4.8.3**, with the remaining pins in `requirements-grpo.txt`.
PEFT 0.19.0 was rejected after its real adapter construction failed on Torch 2.6;
the selected 0.18.1 adapter path was checked instead. Generation/inspection
dependencies stay independent of the optional GPU stack.

If an older notebook stops in section 5 with `Pinned training stack mismatch`,
the installation step was skipped or did not complete. `INSTALL_DEPENDENCIES=False`
does not install anything. Set it to `True`, rerun configuration and setup in
section 1, wait for pip to finish, restart the Python kernel, then set it back
to `False` and run from the beginning. A new Kaggle session/image needs its own
dependency setup. The current notebook checks every pin in section 1 before
importing the graph/ML backend, so missing TRL/bitsandbytes and version differences
are reported there. For vanilla-baseline-only evaluation, use `RUN_BASELINE=True`,
`RUN_SMOKE_TEST=False`, and `RUN_FULL_TRAINING=False`.

During smoke and full training, `EVALUATE_EVERY_STEPS=1` evaluates the same
`PERIODIC_EVAL_SIZE=6` fixed test questions after each optimizer step.
These monitoring scores never enter gradients, choose checkpoints, trigger early
stopping or alter the fixed schedule. All three stage evaluations reuse the
same test IDs and decoding/runtime limits.
Increase the interval to reduce evaluation time, or set it to 0 to disable it.
Evaluation progress shows each completed question's reward and the running mean;
each evaluation prints mean/total reward and per-hop means. A separate table and
curve show measured test rewards by optimizer step. Per-question records
include phase, step, question ID, hop, submitted ID and terminal reward. Rewards
remain binary final-answer checks. Separate fields report `gold_entity_retrieved`,
`retrieval_reward` and combined `training_reward`, with aggregate retrieval-success
rate and mean training reward. These diagnostics do not change evaluation accuracy.
Each phase saves `evaluation_metrics.jsonl` and `evaluation_rewards.jsonl`, with
per-step files under `periodic_evaluation/`. Section 10 also exports consolidated
copies under the run directory. Baseline, final evaluation and reload additionally
save label-specific `_rewards.jsonl` files. Completed-question rewards are flushed
to disk even if a later evaluation episode fails; a partial run has no new
aggregate success claim. Validation rewards never contribute to GRPO updates.

The notebook includes formatted trajectory inspection immediately after baseline
evaluation, training, and post-training evaluation. Cards show the input messages,
expandable full Qwen input with tool schemas, raw assistant text, parsed tool
arguments, and actual tool responses in order. Training inspection reads saved
rollouts; evaluation inspection uses existing traces. Neither invokes the model.
Use `INSPECT_EXAMPLES` and `INSPECT_OFFSET` to browse consecutive episodes, with
matching offsets for vanilla/SFT/SFT + GRPO test outputs. Failures are not filtered out.

`SHOW_PROGRESS=True` displays progress throughout model use. Downloads and weight
loading use native Hugging Face bars; prompt checks and evaluations show completed
questions, elapsed time and ETA. During each episode the status shows its question
ID, generation turn, executed tool calls and generated-token count. Evaluation
also displays measured running accuracy and submissions. GRPO sampling counts
completed trajectories in each batch while the Trainer separately counts optimizer
steps. The same display applies to periodic test monitoring, held-out test and adapter
reload evaluation. Set `SHOW_PROGRESS=False` to disable these bars. `tqdm`, already
provided by TRL/datasets, chooses notebook widgets or a terminal display.

If smoke training reports `_amp_foreach_non_finite_check_and_unscale_cuda` with
`BFloat16`, the FP16 scaler encountered BF16 adapter gradients. TRL 1.8.0 casts
trainable parameters on quantized models to BF16 in its constructor, even when
FP16 is selected. `build_trainer` now restores FP32 LoRA parameters **after** that
constructor and **before** optimizer creation, preserving the original adapter
values exactly. The frozen NF4 base and hardware-selected compute dtype remain
unchanged. Checks after Trainer preparation/checkpoint loading and before each
step catch later dtype changes. Actual flags/dtypes appear in the notebook and
`precision_checks.jsonl`. To retry a failed run, restart the Python kernel with
the updated source and use a fresh `RUN_NAME`; old gradients/scaler state must
not be reused. Keep FP16 computation on a T4.

The rollout integration deliberately uses `rollout_func` rather than
`environment_factory`: the inspected TRL 1.8.0 native tool loop does not consult
our episode's submission state before generating again. The custom adapter stops
at submission/budget limits and passes `env_mask` into TRL. Model token IDs are
preserved, observation/prefill tokens are masked from the policy objective, and
all GRPO loss/advantage/optimizer code remains in TRL. The release's RepeatSampler
already repeats each prompt G times; assertions prevent repeating it twice or
mixing questions in a comparison group. A single GPU, accumulation 8, microbatch
1 and G=8 means one question/group and eight trajectories per optimizer update.
An ordered weighted-deficit schedule controls actual training exposure instead
of relying on pool proportions. TRL shuffling is disabled for this pre-shuffled
schedule, whose prefix is preserved when extending a resumed run. Planned group
IDs/hops and actual trajectory hops are saved separately for audit.

Only question/start information, tool schemas, public rules and real observations
reach the model. Binary answer correctness is computed from the environment's
actual submitted ID and remains the evaluation reward/accuracy. The optimizer
receives `answer_correct + 0.2 * gold_entity_retrieved`. Retrieval earns its bonus
once per episode if a successful executed `get_neighbors` response contains a
gold ID, independently rechecked against the graph. Relation lists, failed calls,
model prose, submission acknowledgments and repeated retrievals earn no extra
bonus. Correctness, retrieval bonus and combined reward are logged separately;
zero-variance group metrics use the actual combined rewards. Gold metadata and
scores never appear in tool observations. The active experiment contains only
one-answer questions; the earlier complete-set source/conversational data remains
available for reproducibility. Run manifests fingerprint the reward, selection
and exposure policy so incompatible old checkpoints cannot be resumed silently.

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
