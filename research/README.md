# Literature review: learning to navigate knowledge graphs

[Back to the project](../README.md)

**Reviewed through 28 October 2026.** This is a focused narrative review of
multi-hop knowledge-graph question answering (KGQA), with emphasis on language-model
agents trained through executable tools and verifiable rewards. Sources are
original papers, conference proceedings and official project resources. Searches
covered graph-navigation RL, agentic KGQA, SFT followed by RL, reward design and
recent benchmark results. This is a curated review rather than an exhaustive
systematic survey. Reported scores are authors' results, not reproductions here.

**The closest research direction is learning an interactive graph-querying policy.**
Recent systems train a compact language model to select graph operations, inspect
execution feedback and produce an answer. Strong results come from several
approaches: grounded demonstrations followed by RL, richer intermediate guidance,
and better retrieval or search. There is no single meaningful winner across all
of these settings. This repository provides a controlled tutorial-scale
comparison within that broader research direction.

## 1. The problem and its boundaries

A knowledge graph stores facts as directed triples, such as
`researcher → works_at → university`. A multi-hop question requires combining
facts connected by two or more relations. An interactive agent must map language
to relations, select useful neighbors, retain earlier evidence, recover from
unproductive choices and decide when to answer. The graph executes its queries,
so the generated actions can be checked against actual results. This framing
appears directly in [KG-Agent](https://aclanthology.org/2025.acl-long.468/) and
[KBQA-R1](https://arxiv.org/html/2512.10999v4).

Three related problems need to be distinguished when reading the literature:

| Setting | What the system must do | Relationship to this project |
| --- | --- | --- |
| Query-conditioned graph walking | Starting from an entity and a query relation, choose edges to reach an answer. | Establishes the RL formulation, but often uses symbolic queries rather than natural-language tool calls. |
| Agentic KGQA | Interpret a question and interact with a structured knowledge base through executable actions. | The closest match to the model–tool–observation loop implemented here. |
| Graph-based retrieval-augmented generation | Retrieve graph or graph-linked textual evidence, then generate an answer. | Relevant to retrieval and reward design, but may use extracted document graphs and free-text answers. |

These categories follow the different interfaces in
[MINERVA](https://arxiv.org/abs/1711.05851),
[KG-R1](https://arxiv.org/html/2509.26383v5) and
[Graph-R1](https://arxiv.org/html/2507.21892v2).
A method can combine categories. In particular, using a graph somewhere in a
pipeline does not mean the language model itself learns to traverse it.

## 2. How the field developed

| Work and publication | Main contribution | Significance for this tutorial |
| --- | --- | --- |
| [DeepPath — Xiong et al., EMNLP 2017](https://aclanthology.org/D17-1060/) | Learns multi-hop relational paths with policy gradients and rewards for accuracy, diversity and efficiency. | Shows that learning graph exploration and shaping its reward predate language-model agents. Its relation-reasoning setup differs from natural-language QA. |
| [MINERVA — Das et al., ICLR 2018](https://arxiv.org/abs/1711.05851) | Learns to navigate from a known source entity to an unknown answer, conditioned on a query relation. | The clearest early ancestor of start-entity-to-answer navigation. It uses a dedicated neural policy rather than an instruction-following LLM. |
| [Think-on-Graph — Sun et al., ICLR 2024](https://arxiv.org/abs/2307.07697) | Uses an LLM to guide iterative beam search over entities and relations without additional model training. | Establishes an important prompting-based alternative to training a navigation policy. Search width and inference cost matter alongside accuracy. |
| [Reasoning on Graphs (RoG) — Luo et al., ICLR 2024](https://arxiv.org/abs/2310.01061) | Generates relation-path plans, retrieves matching graph paths, and reasons over the retrieved evidence. | Separates planning, retrieval and answering. A good plan can guide a graph executor without requiring an LLM decision at every hop. |
| [KG-Agent — Jiang et al., ACL 2025](https://aclanthology.org/2025.acl-long.468/) | Fine-tunes an autonomous agent using code-based instructions, a toolbox, an executor and knowledge memory. | A close supervised predecessor: tool selection and observation handling are learned from demonstrations. |
| [Think-on-Graph 2.0 — Ma et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/830b1abc6d2da85f23d41169fa44d185-Abstract-Conference.html) | Alternates structured graph retrieval and document-context retrieval without model training. | Extends the evidence available to the agent beyond graph labels and triples. |
| [GNN-RAG — Mavromatis and Karypis, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.856/) | Uses a graph neural network to identify candidates and retrieves connecting paths for the LLM. | Demonstrates that graph exploration can be delegated to a specialized retriever. |
| [SubgraphRAG — Li et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/11e1900e680f5fe1893a8e27362dbe2c-Abstract-Conference.html) | Uses lightweight triple scoring and structural features to retrieve a size-controlled subgraph for an LLM. | A useful comparison for the accuracy–cost trade-off of repeated model-driven traversal. |
| [CoG — Liu et al., ACL 2026](https://aclanthology.org/2026.acl-long.708/) | Guides training-free exploration with relational blueprints and responds to failures with reflection and controlled backtracking. | Shows that planning and recovery mechanisms can improve navigation without changing the model through RL. |

The progression is from specialized graph policies, through prompted search and
retrieval pipelines, toward language models trained to manage the interaction
loop. The next section concentrates on that last group.

## 3. The closest recent RL approaches

### KG-R1: a unified querying policy

[KG-R1 — Efficient and Transferable Agentic Knowledge Graph RAG via Reinforcement Learning](https://arxiv.org/html/2509.26383v5)
was first posted in 2025 and revised in May 2026. A single agent interacts with a
KG server and learns through multi-turn GRPO. Its reward combines turn-level
structure and query-validity signals with episode-level answer F1 and retrieval
coverage. It also evaluates transfer by changing the graph backend.

The overlap with this project is substantial: iterative graph queries, trusted
execution results and private answer-based scoring. Important differences are its
richer rewards, answer-set evaluation and larger-scale transfer experiments.
Its published reward should not be described as simply binary answer accuracy.
The [official implementation](https://github.com/junhongmit/KG-R1) is a useful
companion to the paper.

### KBQA-R1: executable reasoning actions and a grounded warm start

[KBQA-R1 — Sun et al., ICML 2026](https://proceedings.mlr.press/v306/sun26h.html)
combines referenced rejection sampling for SFT with subsequent GRPO. Its actions
construct executable logical expressions rather than only moving to neighboring
entities. In the [revised method](https://arxiv.org/html/2512.10999v4), operations
include relation traversal, intersection, ordering, comparison, counting and time
constraints. Execution feedback and relation retrieval help prevent schema
hallucination.

This is a particularly close reference for the SFT + GRPO pipeline. Its broader
action language also explains why general KGQA is harder than the single-answer
relation-chain task used here: some questions require combining sets or applying
operators, not merely extending a path.

### Explore-on-Graph: path information as a training signal

[Explore-on-Graph (EoG) — Yan et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d2e24df9cfaad3189833b819c40b392-Abstract-Conference.html)
uses SFT followed by RL, first with answer-based rewards and then with an
additional path-refined signal. It tests whether rewarding properties of the
reasoning path improves exploration beyond final-answer supervision.

Its [paper](https://arxiv.org/html/2602.21728v1) is relevant to process rewards,
but its retrieved graph-context setting differs from live tool-based traversal.
Path guidance is also additional training information. An outcome-only agent and
a path-supervised agent should therefore be compared with that distinction made
explicit.

### GraphWalker: exploration and recovery before sparse-reward RL

[GraphWalker — Xu et al., COLM 2026](https://arxiv.org/html/2603.28533v3)
uses diverse synthetic trajectories followed by expert recovery demonstrations,
then GRPO with binary exact-match rewards. It navigates full Freebase with gold
topic entities and relation reranking.

This is a counterpoint to increasingly elaborate rewards: a strong navigation
prior can make sparse rewards effective. It does not establish their sufficiency
for an unprepared base model.

### Search-on-Graph-R1: a closely matched four-variant study

[Search-on-Graph-R1 — Sun et al., July 2026 preprint](https://arxiv.org/html/2607.18481v1)
trains an 8B live-search agent and compares base, SFT, RL-only and SFT + RL
variants. Its reward multiplies format validity, correctness and a tool-count
efficiency factor, so incorrect answers receive zero. Its training stages share a
question pool, unlike this repository's separate SFT and GRPO instances.

### GAPD: local guidance during on-policy RL

[GAPD — Sun et al., Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.2459/)
adds gold-action policy distillation to outcome-based RL. It aligns student
exploration states with gold execution states using intermediate entity sets.
A gold-action-conditioned version of the current model supplies detached
teacher scores on generated action tokens. Gold actions are used for training,
not inference.

The [paper's matched ablations](https://aclanthology.org/2026.findings-emnlp.2459.pdf)
compare against the same SFT initialization and outcome-only RL. They offer
recent evidence that intermediate credit assignment can improve beyond a warm
start plus final-answer rewards. This requires logical-form annotations and
alignment machinery that the tutorial currently does not use.

## 4. What is the current state of the art?

**As of the review date, the strongest relevant evidence is distributed across
live-query agents, graph-context reasoning systems and executable logical-form
agents.** The following snapshot records recent results with their original
metric labels. It is not a unified leaderboard: different rows use different
retrieval support, supervision, data splits and answer criteria.

| System and source | Backbone | Author-reported results (%) | Evaluation context |
| --- | --- | --- | --- |
| [KG-R1, v5, Table 1 and §4.5](https://arxiv.org/html/2509.26383v5) | Qwen2.5-3B | WebQSP **82.75 Hits@1**; CWQ **65.31 Hits@1** | KG-server interaction, with separate cross-KG transfer experiments. |
| [EoG, Table 12](https://arxiv.org/html/2602.21728v1) | Llama-3.1-8B | WebQSP **92.8 Hit@1 / 81.3 F1**; CWQ **82.6 Hit@1 / 73.9 F1** | Retrieved graph context and path-refined training rewards. Table 12 is used because the CWQ values differ from the main table. |
| [GraphWalker, v3, Table 2](https://arxiv.org/html/2603.28533v3) | Qwen2.5-7B-Instruct | WebQSP **91.5 EM / 88.6 F1**; CWQ **79.6 EM / 74.2 F1** | Full-graph interaction, gold topic entities and relation reranking; trained on CWQ. |
| [Search-on-Graph-R1, Table 2](https://arxiv.org/html/2607.18481v1) | Llama-3.1-8B | WebQSP **91.7**; CWQ **88.7**; GrailQA **90.0** | Paper labels these “exact match accuracy” and also describes the metric as Hits@1; live graph retrieval. |
| [KBQA-R1, v4, Table 6](https://arxiv.org/html/2512.10999v4) | Llama-3.1-8B | WebQSP **83.4 F1**; GrailQA **86.1 F1**; GraphQ **53.8 F1** | Executable logical-form actions; GrailQA uses its development set. |
| [GAPD, Tables 1–4](https://aclanthology.org/2026.findings-emnlp.2459.pdf) | Llama-3.1-8B | WebQSP **86.1 F1**; GrailQA **89.5 F1**; GraphQ **55.1 F1** | Extends the KBQA-R1 setting with gold-action guidance during RL; GrailQA development evaluation. |

Within the reviewed sources, **GraphWalker and Search-on-Graph-R1 are especially
relevant references for live graph navigation**, while **KBQA-R1 and GAPD provide
a strong progression for action-based logical reasoning**. GAPD improves on its
matched outcome-only baseline across all three reported datasets. The
Search-on-Graph-R1 preprint leads its own CWQ comparison, rather than establishing
a universal lead. [GraphWalker](https://arxiv.org/html/2603.28533v3),
[Search-on-Graph-R1](https://arxiv.org/html/2607.18481v1),
[GAPD](https://aclanthology.org/2026.findings-emnlp.2459.pdf).

Interpret exact-match labels carefully. A single predicted entity belonging to
the gold set, an exactly recovered answer set, and an equivalent logical form are
different success criteria. For example, the
[original GrailQA evaluation](https://dki-lab.github.io/GrailQA/) distinguishes
logical-form semantic exact match from answer F1. Some later papers use EM and
Hits@1 terminology differently. Search-on-Graph-R1 also has table/prose
inconsistencies in its stage ablations. Its main-results table is used here.

## 5. Rewards, verifiers and optimization

The literature does not establish one universally best graph-navigation reward.
**Reward usefulness depends on the starting policy, available annotations and
what the environment exposes.** Three findings are particularly relevant.

First, sparse correctness rewards can work after extensive navigation training,
as [GraphWalker](https://arxiv.org/html/2603.28533v3) demonstrates. More detailed
feedback is another viable route: [KG-R1](https://arxiv.org/html/2509.26383v5)
uses turn-level rewards, while [GAPD](https://aclanthology.org/2026.findings-emnlp.2459/)
uses aligned gold-action guidance. These are alternative ways to improve the
learning signal, with different annotation and implementation costs.

Second, retrieval incentives must distinguish progress from activity.
[GraphRAG-R1 — Yu et al., WWW 2026](https://arxiv.org/html/2507.23581v2)
uses progressively attenuated retrieval rewards and cost-aware answer F1 to
address both insufficient retrieval and unnecessary repeated calls. It operates
with hybrid graph–text retrieval rather than this project's entity-ID interface.
[Graph-R1 — Luo et al., ICML 2026](https://arxiv.org/html/2507.21892v2)
likewise learns multi-turn retrieval, but over a constructed knowledge hypergraph,
with format and answer-F1 rewards on document-QA benchmarks. These papers inform
reward design without providing directly comparable KGQA scores.

Third, group-relative optimization needs variation among sampled returns.
[DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced GRPO.
[DAPO — Yu et al., NeurIPS 2025](https://arxiv.org/html/2503.14476v1)
addresses uninformative groups through dynamic sampling: groups with identical
outcomes are filtered while more samples are collected. For this project's
combined reward, the relevant diagnostic is variance of the **total return**,
not just answer correctness. Identical total returns give zero group-centered
advantages. Dynamic sampling would change the effective training distribution
and sampling cost, so it should be evaluated as an explicit intervention.

A separate concern is assigning gradients to the right tokens.
[Search-R1](https://arxiv.org/html/2503.09516v5) studies multi-turn search with RL
and masks externally retrieved tokens from the policy loss. The same principle
applies to graph tools: the policy should learn its generated actions, while
tool responses remain observations. In this repository, rewards are computed
from actual environment events and the stored gold ID, rather than trusting a
model's textual claim that it retrieved or submitted something.

## 6. Valid actions and constrained generation

Validity and task success are different objectives. A JSON call can be
well-formed yet select an irrelevant relation. A real neighbor can still be the
wrong answer. [KBQA-R1](https://arxiv.org/html/2512.10999v4) grounds proposed
relations through retrieval and confidence gating.
[XGrammar-2 — Li et al., 2026](https://arxiv.org/abs/2601.04426)
provides efficient structured generation for dynamic agent workloads, building
on grammar-constrained decoding.

There are also graph-specific precedents.
[Graph-constrained Reasoning (GCR) — Luo et al., ICML 2025](https://proceedings.mlr.press/v267/luo25t.html)
encodes graph paths in a KG-Trie to constrain generation, then combines the
generated paths with a general LLM's reasoning.
[Decoding on Graphs (DoG) — Li et al., ACL 2025](https://aclanthology.org/2025.acl-long.1186.pdf)
uses graph-aware constrained decoding to generate connected chains of real
triples without additional model training. These approaches constrain graph
content, going beyond syntactically valid JSON. A connected factual chain still
needs to answer the question's intended relations and constraints.

**Application to this repository, as an inference from these methods:** a
per-turn schema could enumerate only discovered entity IDs and available
relations. That would reduce invalid arguments without revealing which valid
choice reaches the answer. This requires environment-derived constraints in
addition to generic JSON formatting. It is an extension, not something the
current implementation already does. Any RL version would also need rollout
sampling and policy-probability calculations to respect the same constraints.

## 7. Benchmarks and credible comparisons

| Benchmark | What it tests | Why it matters here |
| --- | --- | --- |
| [WebQSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763) | Natural questions annotated with executable Freebase semantic parses and answers. | An established next step beyond a small synthetic graph, with an explicit entity-linking and schema-matching problem. |
| [ComplexWebQuestions (CWQ)](https://aclanthology.org/N18-1059/) | Complex questions composed from simpler queries. | Tests whether navigation handles more than a short relation chain. The [repartitioning study](https://arxiv.org/abs/1807.09623) also makes data-split design important. |
| [GrailQA](https://dki-lab.github.io/GrailQA/) | I.i.d., compositional and zero-shot generalization, with logical forms and answers. | Separates familiar query patterns from new compositions and unseen schema domains. |

For a meaningful comparison, report the graph snapshot, access to gold starting
entities, full graph versus retrieved subgraph, tools and auxiliary retrievers,
training annotations, decoding budget, and exact evaluation split. These factors
change what the model must solve. The contrast between
[GraphWalker's full-graph setting](https://arxiv.org/html/2603.28533v3) and
[EoG's graph-context setting](https://arxiv.org/html/2602.21728v1) illustrates why
model size and final accuracy alone are insufficient descriptors.

For this project, accuracy by hop count should be read alongside submission
rate, tool-call validity, answer retrieval, budget exhaustion and calls per
successful episode. Report training return separately from accuracy because the
retrieval bonus can increase without a correct submission. Confidence intervals
within a run describe a different uncertainty from variation across independently
trained seeds. These are evaluation recommendations for this repository, not
claims of a new benchmark standard.

## 8. Where this repository fits

| Dimension | Current tutorial | Broader research setting |
| --- | --- | --- |
| Policy | Qwen3-1.7B with LoRA | Many close comparisons use 3B–8B policies and sometimes additional retrievers or teachers. |
| Graph | Fixed synthetic RDF graph with 2,000 entities and 5,000 navigation facts | Papers commonly evaluate on Freebase, Wikidata or constructed document graphs. |
| Input | Question plus a supplied starting entity ID, name and type | Entity linking may be supplied, predicted or handled by a separate module. |
| Actions | List relations, retrieve neighbors, submit one discovered ID | Other interfaces include set operations, aggregation, temporal filters or batched search. |
| Supervision | Separate SFT and GRPO training instances | Stage-specific data use varies; some pipelines reuse the same question pool. |
| Reward | Correct submission + a once-per-episode 0.2 answer-retrieval bonus | Outcome-only, process-shaped and annotation-guided approaches all appear in the literature. |
| Evidence | Four variants on the same 30 test questions in one recorded run | A research comparison needs larger fixed evaluations, repeated runs and matched access to evidence. |

Implementation details and the recorded results are documented in the
[main README](../README.md). The research-side comparisons above are supported
by the papers cited throughout this review.

The defensible contribution is **a reproducible, inspectable tutorial for
studying how supervised demonstrations and verifiable-reward optimization change
a compact model's graph-tool behavior**. It is not a claim to introduce a new RL
algorithm or establish state-of-the-art benchmark accuracy. The four-variant
comparison makes the separate contributions of SFT and GRPO visible, while the
saved trajectories make failures such as premature submission or continued
exploration after finding the answer available for analysis.

Our synthesis of the literature is that **interaction quality and the learning
signal deserve attention before algorithm substitution alone**. Warm-start
quality, recovery behavior, grounded action selection and intermediate credit
assignment recur across recent methods. These are hypotheses to test under
matched budgets, not guarantees that any one technique will improve this
particular run. In particular, this project's 66.7% SFT + GRPO accuracy should
only be compared with its own matched variants, not with percentages in the
external benchmark table.

## Suggested reading order

For the most direct route through the topic, read
[MINERVA](https://arxiv.org/abs/1711.05851) for the navigation formulation,
[Think-on-Graph](https://arxiv.org/abs/2307.07697) for prompted exploration,
[KG-Agent](https://aclanthology.org/2025.acl-long.468/) for supervised tool use,
[KG-R1](https://arxiv.org/html/2509.26383v5) and
[KBQA-R1](https://arxiv.org/html/2512.10999v4) for interactive RL,
[GraphWalker](https://arxiv.org/html/2603.28533v3) for curriculum and recovery,
and [GAPD](https://aclanthology.org/2026.findings-emnlp.2459/) for recent
intermediate-action guidance. Read
[Search-on-Graph-R1](https://arxiv.org/html/2607.18481v1) alongside this project's
four-variant experiment, keeping its preprint status and metric caveats in mind.
