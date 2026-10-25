# Verification record

The public graph now uses RDFLib 7.6.0 and Turtle. Custom graph inspectors and
their generated views have been removed. Open `data/graph/graph.ttl` in Protégé
for inspection; actual Protégé frontend behavior was not tested in this session.

| Check | Result |
|---|---|
| Core test environment | 129 passed, 10 optional pinned-training/SFT tests skipped |
| Isolated pinned training/API environment | 136 passed, including real TRL/tokenizer/PEFT, SFT and independent adapter-reset checks; 1 optional notebook-setup module skipped (nbformat absent) |
| RDF/SPARQL dependencies | RDFLib 7.6.0, pyparsing 3.2.3 |
| RDF navigation | Forward/inverse queries agree with original graph facts; discovery and budget tests pass |
| Turtle round trip | Entity IDs, names, types, relation schema and asserted edges preserved |
| OWL schema | Eight classes, typed object properties, functional/symmetric characteristics, inverse declarations |
| RDFLib queries | Property paths and SPARQL agree with complete QA answers |
| Full saved dataset | 3,734/3,734 questions validate; fixed-seed regeneration matches |
| Pilot saved dataset | 135/135 questions validate; fixed-seed regeneration matches |
| Source QA and conversation preservation | All 3,734 full and 135 pilot IDs and question content preserved; only former validation labels changed to test. Train QA/conversation bytes unchanged; conversation exports have only train/test files |
| EDA notebook | 22 code cells, zero errors, 15 statistical charts; no graph-renderer dependency |
| Four-variant training notebook | 26 code cells, zero errors in CPU preflight; no graph-renderer dependency |
| Kaggle setup regression tests | 3 passed; missing/mismatched packages stop before backend imports; matching packages and local CPU preflight are accepted |
| Model progress and evaluation rewards | 8 passed; evaluation counts, GRPO trajectory counts, unchanged episode results/RNG honest failure completion, binary per-question evaluation rewards, scheduled callbacks, and partial evaluation persistence, using scripted CPU unit fixtures |
| QLoRA dtype regression | Real TRL constructor's quantized-model branch reproduced on a CPU metadata fixture; FP32 adapter values restored exactly; 3 CPU precision/GradScaler checks passed. Actual NF4/CUDA retry unverified |
| One-hop extension | 934 full and 35 pilot one-hop questions appended; 8 regression checks cover complete answers, original-row preservation, grouped splits, optional split strategies and regeneration. Mixed training/smoke selection includes one-hop tasks |
| Recorded model/tool inspection | 3 HTML renderer checks passed; escaping, turn order, tool errors, stored selection and no trace mutation. Real pinned Qwen tokenizer rendering checked. 3 new notebook cells execute on CPU with honest empty-trace output |
| Public prompt budget | All 3,432 active single-answer native Qwen prompts checked: 601–619 tokens against a 1,024-token limit |
| Next iteration | 13 regression tests cover single-answer selection/wording/isolation, verified retrieval bonus, 20/40/40 schedules and manifests; real pinned TRL sampling and reward boundaries checked |
| Episodic-return plotting | 1 regression check executes the measured-rollout plotting branch and verifies each panel title matches its plotted series |
| Train/test and four-variant comparison | Full 2,667/1,067 and pilot 94/41; source data unchanged in this iteration. One shared test selection for vanilla/SFT/SFT + GRPO/GRPO only. Both RL variants use identical GRPO pools and budgets |
| Independent GRPO-only initialization | Actual tiny CPU PEFT adapter reset restores its exact initial fingerprint, keeps base tensors unchanged and restores trainable FP32 LoRA; actual CUDA/NF4 reset remains unverified |
| Per-variant artifacts and recovery | Real evaluation flushes partial raw trajectories and rewards; fixture checks preserve completed variants across later failure, save independent RL returns/raw+HTML traces/charts at phase boundaries, and ZIP partial artifacts without checkpoints |
| SFT warm start | 200 verified demonstrations, 733 disjoint GRPO instances; all 200 native Qwen loss masks and budgets checked; tiny CPU adapter takes 2 verified SFT steps, saves a complete checkpoint and preserves its learned weights at GRPO initialization |
| Current-iteration pretrained CUDA tool use, GRPO updates, VRAM and trained-adapter evaluation | Unverified; no CUDA GPU available |

Each saved graph contains 2,000 entities and 5,000 navigation facts. Its complete
Turtle document contains 13,232 RDF triples including metadata and ontology
declarations. The two counts describe different things; navigation excludes
schema predicates. No reasoner or inferred-edge materialization runs in training.

See [the migration report](docs/rdf_migration_report.md), [the training report](docs/grpo_training_report.md)
and [README](README.md) for details and execution instructions.

Current machine-readable evidence: [four-variant validation](notebooks/reports/four_variant_validation.json). Previous train/test evidence: [train/test pipeline validation](notebooks/reports/train_test_pipeline_validation.json). Previous SFT evidence: [SFT and GRPO validation](notebooks/reports/sft_grpo_validation.json). Earlier iteration evidence remains in [training iteration validation](notebooks/reports/training_iteration_validation.json). Historical one-hop evidence remains in [one-hop extension and inspection validation](notebooks/reports/one_hop_extension_validation.json). The previous user-supplied GPU run is analyzed separately in [the seed-41 report](docs/seed41_results_analysis.md).
