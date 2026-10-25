# Verification record

The public graph now uses RDFLib 7.6.0 and Turtle. Custom graph inspectors and
their generated views have been removed. Open `data/graph/graph.ttl` in Protégé
for inspection; actual Protégé frontend behavior was not tested in this session.

| Check | Result |
|---|---|
| Core test environment | 97 passed, 4 optional pinned-training API tests skipped |
| Isolated pinned training/API environment | 98 passed, including all 4 real TRL/tokenizer/PEFT API tests; 1 optional notebook-setup module skipped (nbformat absent) |
| RDF/SPARQL dependencies | RDFLib 7.6.0, pyparsing 3.2.3 |
| RDF navigation | Forward/inverse queries agree with original graph facts; discovery and budget tests pass |
| Turtle round trip | Entity IDs, names, types, relation schema and asserted edges preserved |
| OWL schema | Eight classes, typed object properties, functional/symmetric characteristics, inverse declarations |
| RDFLib queries | Property paths and SPARQL agree with complete QA answers |
| Full saved dataset | 3,734/3,734 questions validate; fixed-seed regeneration matches |
| Pilot saved dataset | 135/135 questions validate; fixed-seed regeneration matches |
| Source QA and conversation preservation | All original 2,800 full and 100 pilot question rows and conversational rows remain unchanged prefixes; 934 full and 35 pilot one-hop rows appended |
| EDA notebook | 22 code cells, zero errors, 15 statistical charts; no graph-renderer dependency |
| GRPO notebook | 18 code cells, zero errors in CPU preflight; no graph-renderer dependency |
| Kaggle setup regression tests | 3 passed; missing/mismatched packages stop before backend imports; matching packages and local CPU preflight are accepted |
| Model progress and evaluation rewards | 8 passed; evaluation counts, GRPO trajectory counts, unchanged episode results/RNG honest failure completion, binary per-question evaluation rewards, scheduled callbacks, and partial evaluation persistence, using scripted CPU unit fixtures |
| QLoRA dtype regression | Real TRL constructor's quantized-model branch reproduced on a CPU metadata fixture; FP32 adapter values restored exactly; 3 CPU precision/GradScaler checks passed. Actual NF4/CUDA retry unverified |
| One-hop extension | 934 full and 35 pilot one-hop questions appended; 8 regression checks cover complete answers, original-row preservation, grouped splits, optional split strategies and regeneration. Mixed training/smoke selection includes one-hop tasks |
| Recorded model/tool inspection | 3 HTML renderer checks passed; escaping, turn order, tool errors, stored selection and no trace mutation. Real pinned Qwen tokenizer rendering checked. 3 new notebook cells execute on CPU with honest empty-trace output |
| Public prompt budget | All 3,734 native Qwen prompts checked: 604–625 tokens against a 1,024-token limit |
| Pretrained CUDA tool use, GRPO updates, VRAM and trained-adapter evaluation | Unverified; no CUDA GPU available |

Each saved graph contains 2,000 entities and 5,000 navigation facts. Its complete
Turtle document contains 13,232 RDF triples including metadata and ontology
declarations. The two counts describe different things; navigation excludes
schema predicates. No reasoner or inferred-edge materialization runs in training.

See [the migration report](docs/rdf_migration_report.md), [the training report](docs/grpo_training_report.md)
and [README](README.md) for details and execution instructions.

Current machine-readable evidence: [one-hop extension and inspection validation](notebooks/reports/one_hop_extension_validation.json).
