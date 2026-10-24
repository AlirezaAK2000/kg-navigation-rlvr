# Verification record

The public graph now uses RDFLib 7.6.0 and Turtle. Custom graph inspectors and
their generated views have been removed. Open `data/graph/graph.ttl` in Protégé
for inspection; actual Protégé frontend behavior was not tested in this session.

| Check | Result |
|---|---|
| Core test environment | 69 passed, 3 optional pinned-training API tests skipped |
| Isolated pinned training/API environment | 72 passed, including all 3 real TRL/tokenizer/PEFT API tests |
| RDF/SPARQL dependencies | RDFLib 7.6.0, pyparsing 3.2.3 |
| RDF navigation | Forward/inverse queries agree with original graph facts; discovery and budget tests pass |
| Turtle round trip | Entity IDs, names, types, relation schema and asserted edges preserved |
| OWL schema | Eight classes, typed object properties, functional/symmetric characteristics, inverse declarations |
| RDFLib queries | Property paths and SPARQL agree with complete QA answers |
| Full saved dataset | 2,800/2,800 questions validate; fixed-seed regeneration matches |
| Pilot saved dataset | 100/100 questions validate; fixed-seed regeneration matches |
| Source QA and conversation preservation | All original JSONL bytes unchanged |
| EDA notebook | 22 code cells, zero errors, 15 statistical charts; no graph-renderer dependency |
| GRPO notebook | 14 code cells, zero errors in CPU preflight; no graph-renderer dependency |
| Kaggle setup regression tests | 3 passed; missing/mismatched packages stop before backend imports; matching packages and local CPU preflight are accepted |
| Model progress integration | 3 passed; evaluation counts, GRPO trajectory counts, unchanged episode results/RNG and honest failure completion, using scripted CPU unit fixtures |
| QLoRA dtype regression | Real TRL constructor's quantized-model branch reproduced on a CPU metadata fixture; FP32 adapter values restored exactly; 3 CPU precision/GradScaler checks passed. Actual NF4/CUDA retry unverified |
| Pretrained CUDA tool use, GRPO updates, VRAM and trained-adapter evaluation | Unverified; no CUDA GPU available |

Each saved graph contains 2,000 entities and 5,000 navigation facts. Its complete
Turtle document contains 13,232 RDF triples including metadata and ontology
declarations. The two counts describe different things; navigation excludes
schema predicates. No reasoner or inferred-edge materialization runs in training.

See [the migration report](docs/rdf_migration_report.md), [the training report](docs/grpo_training_report.md)
and [README](README.md) for details and execution instructions.
