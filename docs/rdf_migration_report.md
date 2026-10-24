# RDFLib and Turtle migration

The graph storage and navigation backend now uses **RDFLib 7.6.0**. Both saved
datasets use Turtle. The HTML/SVG graph renderer, inspection scripts, related
tests, interactive notebook graph widgets, trajectory graph exports and old
`visualizations/` directory have been removed. Statistical EDA and GRPO metric
plots remain available in the notebooks.

## Files to open in Protégé

- `data/graph/graph.ttl`: self-contained OWL schema plus all labeled individuals
  and asserted edges. Open this file for the full graph.
- `data/graph/ontology.ttl`: schema only.
- `pilot_data/graph/`: the same two exports for the pilot dataset.

Use **File → Open**, then inspect Classes, Object properties and Individuals.
No remote ontology imports are required. The documents were parsed and queried
with RDFLib; a live Protégé UI session was not available for frontend validation.

Classes represent the eight existing entity types. Properties carry domains,
ranges, cardinality/constraint annotations and inverse-navigation permissions.
Many-to-one properties are declared functional, symmetric relations are declared
symmetric, and permitted inverses have explicit `owl:inverseOf` declarations.
Entity IRIs preserve the original string IDs and labels. There are 16 base
relations; inverse declarations add schema properties without adding navigation
facts. The full document contains 13,232 RDF triples, including 5,000 asserted
navigation edges and 2,000 individuals' metadata.

The graph uses fictional identifiers under
`https://example.org/kg-navigation-rlvr/`. These are RDF identifiers, not network
resources that need to be fetched. The self-contained `graph.ttl` is authoritative;
`ontology.ttl` is the separate schema export. When editing a dataset in Protégé,
save the complete graph as Turtle and rerun validation. The project expects its
stable IDs and schema annotations. Its closed-world constraints are still
checked by Python; OWL reasoning is not a replacement for missing-value,
cardinality, cycle or cross-relation validation.

## Implementation

`src/kg_rlvr/rdf.py` defines the namespaces, OWL schema and conversion between
the public string-ID API and RDF terms. `GraphIndex` loads and saves Turtle and
uses RDFLib `predicates`, `objects` and `subjects` queries for relation and
neighbor lookup. The custom adjacency index has been removed. Bounded standard
library caches retain immutable lookup results for repeated verification and
rollouts. The loaded dataset view is read-only: edit Turtle and reload it to
apply changes. Public methods return fresh lists to protect cached results.

`NavigationSession`, the RL environment and reference executor retain the
experiment's discovery rules, call budgets, actual submissions, complete answer
sets, witness/path counts and difficulty measurements. Metadata predicates such
as `rdf:type` and `rdfs:label` are excluded from model navigation. Inverse calls
query incoming asserted edges while preserving the existing `inv:` permissions.
No inference, reward shaping or automatic answer shortcuts were introduced.

Graph generation now writes `graph.ttl` and `ontology.ttl`. The loader still
accepts old JSON/TSV graph mounts as a compatibility input. The six legacy graph
files in the repository were replaced by four Turtle files; the obsolete graph
inspection status and visualization configuration were removed. QA records and
public conversational exports retain their original bytes. Graph IDs, names,
types, relation specifications, all 5,000 facts, question answers, reference
paths, difficulty values and split membership were preserved.

README and both notebooks now direct graph inspection to Protégé. The EDA
notebook saves statistical figures/reports under `artifacts/eda/` by default
(its validation run used `/tmp/kg_rdf_eda_check`). The training notebook retains
the GitHub clone into the current Kaggle folder, the same tool API, and actual
JSONL trajectory logs. Dependency pins and training manifests include RDFLib
and pyparsing. Older trainer manifests will fail compatibility checks after
this data/backend migration; start a new experiment with the new source/data.

## Verification

- Core environment: **69 tests passed, 3 optional API tests skipped**.
- Isolated pinned training environment: **72 tests passed**, including the three
  real TRL/Qwen-tokenizer/PEFT API checks. Those API fixtures do not establish a
  pretrained model's GPU tool use or training success.
- RDF navigation agrees with the original facts in forward and permitted inverse
  directions. Tests check complete multi-neighbor results, metadata exclusion,
  cache-result isolation, corrupt RDF rejection and legacy-input migration.
- RDFLib property paths and a SPARQL query agree with complete question answers
  across the hop categories. RDF round trips preserve the schema and graph data.
- In-memory validation and full fixed-seed regeneration passed for **2,800 full
  questions and 100 pilot questions**. Neither check rewrote saved QA records.
- The EDA notebook executed **22 code cells** with zero errors and generated
  **15 statistical charts**. The GRPO notebook executed **14 code cells** in CPU
  preflight with zero errors. Neither notebook references the deleted renderer.
- QA and conversation files were compared with the pre-migration SHA-256
  snapshot. No record changed. Source datasets are also fingerprinted around
  notebook execution.

The first SPARQL check exposed an incompatibility between RDFLib 7.6.0 and the
host's old pyparsing 3.0.9. The explicit **pyparsing 3.2.3** pin fixes the checked
API. Verification used temporary dependency environments instead of changing
the host's global Python installation. Initial uncached verification was slow;
bounded query caching restored practical repeated-query performance.

Kaggle GPU execution, pretrained model tool calls, genuine GRPO optimizer
updates, trained-model adapter evaluation and peak VRAM remain unverified on
this CPU-only development host. Migration does not change that status.

The implementation follows the [RDFLib Graph API](https://rdflib.readthedocs.io/en/stable/apidocs/rdflib.graph/),
[Turtle specification](https://www.w3.org/TR/turtle/) and
[OWL 2 ontology model](https://www.w3.org/TR/owl2-primer/).
