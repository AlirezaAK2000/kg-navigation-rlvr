# Verification record

The implementation and generated artifacts are present. **Interactive browser
visual inspection is still pending** because this session has no computer-use
permission and the in-app browser is unavailable. No claim of completed browser
visual QA is made.

| Acceptance check | Result |
|---|---|
| 100-question pilot spanning 2/3/4 hops | Passed; 70 train, 10 validation, 20 test |
| Automated graph/question validation | Passed for all 100 pilot and 2,800 full questions |
| Unit tests | 43 passed with Python 3.11 |
| Fixed-seed full and pilot regeneration | Passed; independent comparison of graph and question records |
| Interactive overview generated | `visualizations/graph_overview.html` |
| Browser overview inspection | Blocked by missing computer-use permission |
| Entity neighborhood generated | `visualizations/entity_researcher_000042.html` |
| 2-hop question visualization | `q_000201`, five answers |
| 3-hop question visualization | `q_001240`, six answers |
| 4-hop question visualization | `q_001457`, five answers |
| Visualization projection/path tests | Passed; all answers and saved witnesses retained |
| Exported JavaScript syntax | Passed with Node.js `--check`; not a browser rendering test |
| Ten random samples displayed and re-executed | Passed; seed 41, `data/metadata/samples.md` |
| Full 2,800-question dataset | Passed; 2,000 train, 300 validation, 500 test |
| Statistical summaries and overlap checks | Saved in `data/metadata/` and `pilot_data/metadata/` |
| Prompt-only conversational conversion | Saved in `conversations/` |
| Reproduction instructions | `README.md` |

The full graph contains 2,000 entities, 5,000 stored triples, 16 relations and one
weakly connected component. Questions cover 26 relation compositions, including
250 multi-answer questions. Mean measured navigation branching is 5.42; maximum
reference execution cost is 35 public tool calls. There are zero duplicate
question texts and zero query-instance groups shared across splits. Facts,
entities, templates, and compositions can overlap in the ordinary split.

To finish visual QA, open `visualizations/index.html` in a standard browser.
Inspect the overview and each question view; exercise zoom, panning, node
dragging, node details, search, type/relation filters, metadata disclosure, and
the per-answer path selector. Machine-readable inspection status is stored in
`data/metadata/inspection_status.json`.
