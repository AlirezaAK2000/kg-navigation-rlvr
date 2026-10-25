# Synthetic KG dataset report

Seed: 1729. Validation: 100.0% question checks passed; dataset valid: True.

2,000 entities; 5,000 stored triples; 1 weak component(s).
Unique-neighbor node degree: mean 4.58, median 4.0, maximum 19.

## Entity counts

| Type | Count |
|---|---:|
| city | 200 |
| country | 50 |
| field | 150 |
| funder | 100 |
| project | 300 |
| publication | 400 |
| researcher | 700 |
| university | 100 |

## Stored triples

| Relation | Count |
|---|---:|
| authored_by | 411 |
| based_in | 100 |
| cites | 402 |
| city_in_country | 200 |
| collaborates_with | 220 |
| contributes_to | 307 |
| funded_by | 304 |
| investigates | 303 |
| led_by | 300 |
| located_in_city | 100 |
| partners_with | 28 |
| produced_by | 400 |
| publication_field | 400 |
| specializes_in | 700 |
| subfield_of | 125 |
| works_at | 700 |

## Questions

| Split | 1 hop | 2 hops | 3 hops | 4 hops | Total |
|---|---:|---:|---:|---:|---:|
| test | 11 | 11 | 10 | 9 | 41 |
| train | 24 | 24 | 23 | 23 | 94 |

71 unique query instances; 29 relation compositions; 7 multi-answer questions.
Answer-set sizes: mean 1.07, median 1, maximum 3.
Branching: mean 5.22, median 5.50, maximum 7.67.
Reference public tool calls: mean 6.23, maximum 15.
Duplicate question texts: 0; cross-split query-instance groups: 0.

## Cross-split overlap

| Splits | Shared compositions | Shared starts | Shared reference entities |
|---|---:|---:|---:|
| test__train | 13 | 0 | 8 |

## Generation filters

Rejected candidates are counted by the first failed constraint; requested quotas are never silently reduced.

- `1hop:empty_answer_set`: 1879
- `2hop:empty_answer_set`: 412
- `2hop:shorter_equivalent_answer_set`: 295
- `3hop:empty_answer_set`: 18
- `3hop:shorter_equivalent_answer_set`: 9
- `4hop:empty_answer_set`: 412
- `4hop:shorter_equivalent_answer_set`: 313

## Warnings and interpretation

- Stored relation frequencies differ by more than 10x; inspect the ontology and query coverage.
- Template-generated language is finite. Ordinary splits share grammar, facts, entities and compositions; they do not rule out memorization or template exploitation.
- Relations used as distractors but not in the current question catalog: collaborates_with, partners_with, publication_field, subfield_of.

The intended hop count is the length of the executable relation program. Shorter equivalent answer-set programs are exhaustively excluded per start instance. This conservative extensional check is not a proof of arbitrary natural-language logical minimality. Unlabelled shortest graph distances and reference tool costs are stored separately. See statistics.json for full distributions and composition paths.
