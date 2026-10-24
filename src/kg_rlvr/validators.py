"""Exhaustive deterministic verification of graph and question invariants."""

from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path

from .graph_index import GraphIndex, execute
from .ontology import ENTITY_TYPES, RELATIONS, program_end_type
from .query_generator import instance_id
from .question_templates import PROGRAMS, PROGRAM_BY_ID


def graph_errors(graph, config=None):
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(graph.relations == RELATIONS, "Ontology differs from the versioned implementation")
    check(all(e.get("type") in ENTITY_TYPES and isinstance(e.get("name"), str) and e["name"] for e in graph.entities.values()), "Invalid entity type or name")
    check(len({e["name"] for e in graph.entities.values()}) == len(graph.entities), "Entity names are not unique")
    required = {"researcher": ("works_at", "specializes_in"), "university": ("located_in_city",),
                "city": ("city_in_country",), "funder": ("based_in",),
                "project": ("led_by", "investigates", "funded_by"),
                "publication": ("authored_by", "produced_by", "publication_field")}
    for entity, spec in graph.entities.items():
        for relation in required.get(spec["type"], ()):
            count = len(graph.get_neighbors(entity, relation))
            check(count >= 1, f"Missing required {relation} for {entity}")
            if RELATIONS[relation]["cardinality"] == "many_to_one":
                check(count == 1, f"Functional relation violated: {entity}/{relation}")
        if spec["type"] == "field":
            check(len(graph.get_neighbors(entity, "subfield_of")) <= 1, f"Multiple field parents: {entity}")
        if spec["type"] == "project":
            leaders = graph.get_neighbors(entity, "led_by")
            for leader in leaders:
                check(entity in graph.get_neighbors(leader, "contributes_to"), f"Leader does not contribute: {entity}/{leader}")
            specialties = {f for leader in leaders for f in graph.get_neighbors(leader, "specializes_in")}
            check(bool(specialties.intersection(graph.get_neighbors(entity, "investigates"))), f"Project lacks a leader specialty: {entity}")
        if spec["type"] == "publication":
            for project in graph.get_neighbors(entity, "produced_by"):
                check(bool(set(graph.get_neighbors(entity, "authored_by")).intersection(graph.get_neighbors(project, "led_by"))), f"Publication lacks an author who leads its project: {entity}")
                check(set(graph.get_neighbors(entity, "publication_field")).issubset(graph.get_neighbors(project, "investigates")), f"Publication/project field mismatch: {entity}")
    edges = set(graph.triples)
    for head, relation, tail in graph.triples:
        if RELATIONS[relation]["symmetric"]:
            check(head != tail and (tail, relation, head) in edges, f"Symmetry or self-edge violation: {head}/{relation}/{tail}")
        if relation == "cites":
            check(head > tail, f"Citations must point to an earlier publication ID: {head}/{tail}")
    # Iterative Kahn test: does not depend on numeric field ID ordering.
    fields = [e for e, spec in graph.entities.items() if spec["type"] == "field"]
    indegree, children = Counter(), defaultdict(list)
    for head, relation, tail in graph.triples:
        if relation == "subfield_of":
            children[tail].append(head)
            indegree[head] += 1
    pending = [f for f in fields if not indegree[f]]
    visited = 0
    while pending:
        field = pending.pop()
        visited += 1
        for child in children[field]:
            indegree[child] -= 1
            if not indegree[child]:
                pending.append(child)
    check(visited == len(fields), "Field hierarchy contains a cycle")
    if config:
        g = config["graph"]
        check(len(graph.entities) == g["entity_count"], "Configured entity count mismatch")
        check(len(graph.triples) == g["triple_count"], "Configured triple count mismatch")
        if g["type_counts"]:
            check(dict(Counter(e["type"] for e in graph.entities.values())) == g["type_counts"], "Configured type counts mismatch")
        limits = {"authored_by": g["max_authors"], "investigates": g["max_project_fields"],
                  "funded_by": g["max_project_funders"], "collaborates_with": g["collaboration_degree_target"]}
        for entity, relations in graph.adjacency.items():
            for relation, limit in limits.items():
                check(len(relations.get(relation, ())) <= limit, f"Degree cap violated: {entity}/{relation}")
            if graph.entities[entity]["type"] == "project":
                check(len(relations.get("led_by", ())) == g["leaders_per_project"], f"Leader count mismatch: {entity}")
    return errors


def validate_dataset(graph, questions, config=None):
    errors = [{"id": "graph", "error": error} for error in graph_errors(graph, config)]
    ids, text_splits, instance_splits, start_splits = set(), {}, {}, defaultdict(set)
    program_splits = defaultdict(set)
    bad_ids = set()

    @lru_cache(maxsize=256)
    def shorter(start):
        return graph.shorter_results(start, 3)

    @lru_cache(maxsize=4096)
    def reference(start, path):
        return execute(graph, start, path)

    @lru_cache(maxsize=256)
    def distances(start):
        return graph.shortest_distances(start)

    for q in questions:
        qid = q.get("id", "<missing>")
        def check(condition, message):
            if not condition:
                raise ValueError(message)
        try:
            check(qid not in ids, "Duplicate question ID")
            ids.add(qid)
            start, path, answers = q["start_entity"], q["relation_path"], q["answer_ids"]
            check(start in graph.entities, "Unknown starting entity")
            check(isinstance(answers, list) and answers == sorted(set(answers)) and bool(answers), "Answers must be nonempty, unique, sorted")
            check(all(a in graph.entities for a in answers), "Unknown answer entity")
            end_type = program_end_type(graph.entities[start]["type"], path, graph.relations)
            check(all(graph.entities[a]["type"] == end_type for a in answers), "Answer type mismatch")
            check(q["hop_count"] == len(path) and len(path) in {2, 3, 4}, "Hop count mismatch")
            program = PROGRAM_BY_ID[q["program_id"]]
            check(program.start_type == graph.entities[start]["type"] and list(program.relation_path) == path, "Program/template semantics mismatch")
            templates = {f"{program.id}:v{i + 1}": t for i, t in enumerate(program.templates)}
            check(q["template_id"] in templates, "Unknown template")
            check(q["question"] == templates[q["template_id"]].format(start=graph.entities[start]["name"]), "Question does not match its reviewed template")
            result = reference(start, tuple(path))
            check(result.answers == answers, "Recorded answer set is incomplete or incorrect")
            check(result.path_counts == q["answer_path_counts"], "Alternative path counts mismatch")
            paths = q["gold_entity_paths"]
            check(len(paths) == len(answers) and {p[-1] for p in paths if p} == set(answers), "Missing answer witnesses")
            for witness in paths:
                check(len(witness) == len(path) + 1 and witness[0] == start, "Invalid witness length/start")
                for head, relation, tail in zip(witness, path, witness[1:]):
                    check(tail in graph.get_neighbors(head, relation), "Invalid witness edge")
            check(result.difficulty == q["difficulty"], "Branching/tool statistics mismatch")
            check(q["shortest_path_hops"] == {a: distances(start)[a] for a in answers}, "Shortest distance mismatch")
            equivalent = shorter(start).get(frozenset(answers))
            check(equivalent is None or len(equivalent) >= len(path), "A shorter program returns the same complete answer set")
            check(q["shortest_equivalent_program_length"] == len(path), "Equivalent program metadata mismatch")
            iid = instance_id(start, path, answers)
            check(q["query_instance_id"] == iid, "Query instance hash mismatch")
            check(instance_splits.get(iid, q["split"]) == q["split"], "Query instance/paraphrases leak across splits")
            instance_splits[iid] = q["split"]
            normalized_text = " ".join(q["question"].casefold().split())
            check(normalized_text not in text_splits, "Duplicate question text")
            text_splits[normalized_text] = q["split"]
            start_splits[start].add(q["split"])
            program_splits[program.id].add(q["split"])
            if config:
                settings = config["questions"]
                d = result.difficulty
                check(len(answers) <= settings["max_answer_set_size"], "Answer set exceeds configured limit")
                check(d["branching_factor"] >= settings["min_branching_factor"], "Branching below configured minimum")
                check(d["reference_tool_calls"] <= settings["max_tool_calls"], "Tool budget exceeded")
                check(d["max_neighbor_response"] <= settings["max_neighbors_per_response"], "Neighbor response exceeds configured limit")
        except (ValueError, KeyError, TypeError, IndexError) as error:
            errors.append({"id": qid, "error": str(error)})
            bad_ids.add(qid)
    if config:
        from .config import allocate
        counts = dict(config["questions"]["split_counts"])
        if config["questions"]["composition_test_count"]:
            counts["composition_test"] = config["questions"]["composition_test_count"]
        actual = Counter(q.get("split") for q in questions)
        if any(actual[split] != count for split, count in counts.items()) or set(actual) - set(counts):
            errors.append({"id": "splits", "error": f"Split count mismatch: expected {counts}, got {dict(actual)}"})
        for split, count in counts.items():
            expected = {int(h): n for h, n in allocate(count, config["questions"]["hop_weights"]).items() if n}
            got = dict(Counter(q["hop_count"] for q in questions if q["split"] == split))
            if expected != got:
                errors.append({"id": split, "error": f"Hop quota mismatch: expected {expected}, got {got}"})
        if config["questions"]["split_strategy"] == "start_entity":
            if any(len(splits - {"composition_test"}) > 1 for splits in start_splits.values()):
                errors.append({"id": "splits", "error": "Starting entities leak across ordinary splits"})
    if any("composition_test" in splits and len(splits) > 1 for splits in program_splits.values()):
        errors.append({"id": "splits", "error": "Held-out relation composition leaks into another split"})
    return {"question_count": len(questions), "passed": len(questions) - len(bad_ids),
            "pass_percentage": 100 * (len(questions) - len(bad_ids)) / len(questions) if questions else 0,
            "valid": not errors, "errors": errors,
            "reproducibility_checked": False}


def validate_saved_dataset(directory, check_reproducibility=False):
    from .config import load_config
    from .dataset_builder import read_questions, split_candidates
    from .graph_generator import generate_graph
    from .query_generator import generate_candidates
    directory = Path(directory)
    graph = GraphIndex.load(directory / "graph")
    config = load_config(directory / "config.json")
    questions = read_questions(directory)
    result = validate_dataset(graph, questions, config)
    catalog = [json.loads(line) for line in (directory / "metadata" / "query_programs.jsonl").read_text().splitlines()]
    if catalog != [p.as_dict() for p in PROGRAMS]:
        result["errors"].append({"id": "catalog", "error": "Exported template/program catalog mismatch"})
    for path in (directory / "rl").glob("*.jsonl"):
        if any(json.loads(line)["split"] != path.stem for line in path.read_text().splitlines() if line.strip()):
            result["errors"].append({"id": str(path), "error": "Question stored in the wrong split file"})
    if check_reproducibility:
        regenerated = generate_graph(config)
        candidates, _ = generate_candidates(regenerated, config)
        rebuilt, _ = split_candidates(regenerated, candidates, config)
        matches = (regenerated.entities == graph.entities and regenerated.triples == graph.triples
                   and sorted(rebuilt, key=lambda q: q["id"]) == sorted(questions, key=lambda q: q["id"]))
        if not matches:
            result["errors"].append({"id": "reproducibility", "error": "Fixed-seed regeneration differs"})
        result["reproducibility_checked"] = True
        result["reproducible"] = matches
    result["valid"] = not result["errors"]
    return result
