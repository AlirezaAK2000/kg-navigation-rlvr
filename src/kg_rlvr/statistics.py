"""Inspectable statistics and overlap reports without numerical dependencies."""

from collections import Counter, defaultdict
from itertools import combinations
import statistics as stats


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "max": 0, "p90": 0, "histogram": {}}
    return {"count": len(values), "mean": stats.mean(values), "median": stats.median(values), "max": max(values),
            "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
            "histogram": dict(sorted(Counter(round(x, 1) for x in values).items()))}


def component_sizes(graph):
    parent = {e: e for e in graph.entities}
    def find(entity):
        while parent[entity] != entity:
            parent[entity] = parent[parent[entity]]
            entity = parent[entity]
        return entity
    for head, _, tail in graph.triples:
        parent[find(head)] = find(tail)
    return sorted(Counter(find(e) for e in graph.entities).values(), reverse=True)


def build_statistics(graph, questions, validation, config, generation):
    counts = Counter(e["type"] for e in graph.entities.values())
    triples = Counter(r for _, r, _ in graph.triples)
    # Undirected unique neighbors (not doubled for reciprocal symmetric triples).
    neighbors = defaultdict(set)
    for head, _, tail in graph.triples:
        neighbors[head].add(tail)
        neighbors[tail].add(head)
    degree = distribution([len(neighbors[e]) for e in graph.entities])
    components = component_sizes(graph)
    split_hops, programs, starts, mentioned, relations = {}, {}, {}, {}, {}
    for split in sorted({q["split"] for q in questions}):
        rows = [q for q in questions if q["split"] == split]
        split_hops[split] = dict(sorted(Counter(q["hop_count"] for q in rows).items()))
        programs[split] = {tuple(q["relation_path"]) for q in rows}
        starts[split] = {q["start_entity"] for q in rows}
        mentioned[split] = {e for q in rows for path in q["gold_entity_paths"] for e in path}
        relations[split] = {r for q in rows for r in q["relation_path"]}
    overlaps = {}
    for left, right in combinations(sorted(programs), 2):
        overlaps[f"{left}__{right}"] = {
            "shared_compositions": len(programs[left] & programs[right]),
            "shared_composition_paths": [list(path) for path in sorted(programs[left] & programs[right])],
            "shared_start_entities": len(starts[left] & starts[right]),
            "shared_reference_entities": len(mentioned[left] & mentioned[right]),
            "shared_relation_operations": len(relations[left] & relations[right]),
        }
    template_counts = Counter(q["template_id"] for q in questions)
    question_relations = Counter(r for q in questions for r in q["relation_path"])
    answer_counts = Counter(tuple(q["answer_ids"]) for q in questions)
    branch = distribution([q["difficulty"]["branching_factor"] for q in questions])
    all_response_sizes = [len(ns) for rels in graph.adjacency.values() for ns in rels.values()]
    warnings = []
    if len(components) > 1:
        warnings.append(f"Graph has {len(components)} weakly connected components.")
    if branch["median"] < 3:
        warnings.append("Median navigation branching is below three choices.")
    if triples and max(triples.values()) > 10 * min(triples.values()):
        warnings.append("Stored relation frequencies differ by more than 10x; inspect the ontology and query coverage.")
    if all_response_sizes and max(all_response_sizes) > config["questions"]["max_neighbors_per_response"]:
        warnings.append("Some graph neighbor lists exceed the question response limit; these are excluded only when encountered by a gold execution.")
    if questions and max(template_counts.values(), default=0) / len(questions) > 0.1:
        warnings.append("A single exact template accounts for over 10% of questions.")
    if questions and max(answer_counts.values(), default=0) / len(questions) > 0.05:
        warnings.append("More than 5% of questions share an identical answer set.")
    # A limited reviewed grammar is expected, and must be acknowledged even if balanced.
    warnings.append("Template-generated language is finite. Ordinary splits share grammar, facts, entities and compositions; they do not rule out memorization or template exploitation.")
    unused = sorted(set(graph.relations) - {r.removeprefix("inv:") for r in question_relations})
    if unused:
        warnings.append("Relations used as distractors but not in the current question catalog: " + ", ".join(unused) + ".")
    exact_texts = Counter(q["question"] for q in questions)
    instances = defaultdict(set)
    for q in questions:
        instances[q["query_instance_id"]].add(q["split"])
    return {
        "schema_version": 1, "seed": config["seed"],
        "graph": {"entity_count": len(graph.entities), "entity_counts_by_type": dict(sorted(counts.items())),
                  "triple_count": len(graph.triples), "triple_counts_by_relation": dict(sorted(triples.items())),
                  "node_degree_undirected_unique": degree,
                  "navigation_neighbor_list_sizes": distribution(all_response_sizes),
                  "weak_components": {"count": len(components), "sizes": components}},
        "questions": {"count": len(questions), "counts_by_split_and_hop": split_hops,
                      "unique_query_instances": len(instances), "unique_compositions": len(set().union(*programs.values())) if programs else 0,
                      "compositions_per_split": {s: len(ps) for s, ps in programs.items()},
                      "answer_set_sizes": distribution([len(q["answer_ids"]) for q in questions]),
                      "multi_answer_questions": sum(len(q["answer_ids"]) > 1 for q in questions),
                      "branching_factors": branch,
                      "reference_tool_calls": distribution([q["difficulty"]["reference_tool_calls"] for q in questions]),
                      "shortest_answer_distances": distribution([d for q in questions for d in q["shortest_path_hops"].values()]),
                      "questions_with_multiple_program_walks": sum(sum(q["answer_path_counts"].values()) > len(q["answer_ids"]) for q in questions),
                      "template_counts": dict(sorted(template_counts.items())),
                      "query_operation_counts": dict(sorted(question_relations.items())),
                      "most_common_answer_sets": [{"answers": list(a), "count": c} for a, c in answer_counts.most_common(10)]},
        "overlap": overlaps,
        "duplicates": {"duplicate_question_texts": sum(c - 1 for c in exact_texts.values()),
                       "cross_split_instance_groups": sum(len(s) > 1 for s in instances.values())},
        "validation": validation, "generation": generation, "warnings": warnings,
    }


def markdown_report(report):
    g, q = report["graph"], report["questions"]
    lines = ["# Synthetic KG dataset report", "", f"Seed: {report['seed']}. Validation: {report['validation']['pass_percentage']:.1f}% question checks passed; dataset valid: {report['validation']['valid']}.", "",
             f"{g['entity_count']:,} entities; {g['triple_count']:,} stored triples; {g['weak_components']['count']} weak component(s).",
             f"Unique-neighbor node degree: mean {g['node_degree_undirected_unique']['mean']:.2f}, median {g['node_degree_undirected_unique']['median']}, maximum {g['node_degree_undirected_unique']['max']}.", "",
             "## Entity counts", "", "| Type | Count |", "|---|---:|"]
    lines += [f"| {t} | {n} |" for t, n in g["entity_counts_by_type"].items()]
    lines += ["", "## Stored triples", "", "| Relation | Count |", "|---|---:|"]
    lines += [f"| {r} | {n} |" for r, n in g["triple_counts_by_relation"].items()]
    lines += ["", "## Questions", "", "| Split | 2 hops | 3 hops | 4 hops | Total |", "|---|---:|---:|---:|---:|"]
    for split, hops in q["counts_by_split_and_hop"].items():
        get = lambda h: hops.get(h, hops.get(str(h), 0))
        lines.append(f"| {split} | {get(2)} | {get(3)} | {get(4)} | {sum(hops.values())} |")
    lines += ["", f"{q['unique_query_instances']} unique query instances; {q['unique_compositions']} relation compositions; {q['multi_answer_questions']} multi-answer questions.",
              f"Answer-set sizes: mean {q['answer_set_sizes']['mean']:.2f}, median {q['answer_set_sizes']['median']}, maximum {q['answer_set_sizes']['max']}.",
              f"Branching: mean {q['branching_factors']['mean']:.2f}, median {q['branching_factors']['median']:.2f}, maximum {q['branching_factors']['max']:.2f}.",
              f"Reference public tool calls: mean {q['reference_tool_calls']['mean']:.2f}, maximum {q['reference_tool_calls']['max']}.",
              f"Duplicate question texts: {report['duplicates']['duplicate_question_texts']}; cross-split query-instance groups: {report['duplicates']['cross_split_instance_groups']}.",
              "", "## Cross-split overlap", "", "| Splits | Shared compositions | Shared starts | Shared reference entities |", "|---|---:|---:|---:|"]
    for pair, overlap in report["overlap"].items():
        lines.append(f"| {pair} | {overlap['shared_compositions']} | {overlap['shared_start_entities']} | {overlap['shared_reference_entities']} |")
    lines += ["", "## Generation filters", "", "Rejected candidates are counted by the first failed constraint; requested quotas are never silently reduced.", ""]
    lines += [f"- `{reason}`: {count}" for reason, count in report["generation"].get("candidate_rejections", {}).items()]
    lines += ["", "## Warnings and interpretation", ""] + [f"- {warning}" for warning in report["warnings"]]
    lines += ["", "The intended hop count is the length of the executable relation program. Shorter equivalent answer-set programs are exhaustively excluded per start instance. This conservative extensional check is not a proof of arbitrary natural-language logical minimality. Unlabelled shortest graph distances and reference tool costs are stored separately. See statistics.json for full distributions and composition paths.", ""]
    return "\n".join(lines)
