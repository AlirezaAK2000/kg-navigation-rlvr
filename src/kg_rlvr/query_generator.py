"""Program-first candidate execution and explicit constraint rejection accounting."""

from collections import Counter, defaultdict
import hashlib
import json
import random

from .graph_index import execute
from .question_templates import PROGRAMS


def instance_id(start, path, answers):
    payload = json.dumps([start, list(path), sorted(answers)], separators=(",", ":"))
    return "instance_" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def generate_candidates(graph, config):
    settings = config["questions"]
    programs_by_type = defaultdict(list)
    for program in PROGRAMS:
        programs_by_type[program.start_type].append(program)
    candidates, rejected = [], Counter()
    starts = sorted(graph.entities)
    random.Random(config["seed"] + 1).shuffle(starts)
    for start in starts:
        programs = programs_by_type[graph.entities[start]["type"]]
        if not programs:
            continue
        shorter = graph.shorter_results(start, 3)
        distances = None
        for program in programs:
            result = execute(graph, start, program.relation_path)
            reason = None
            if not result.answers:
                reason = "empty_answer_set"
            elif len(result.answers) > settings["max_answer_set_size"]:
                reason = "answer_set_too_large"
            elif result.difficulty["reference_tool_calls"] > settings["max_tool_calls"]:
                reason = "tool_budget_exceeded"
            elif result.difficulty["max_neighbor_response"] > settings["max_neighbors_per_response"]:
                reason = "neighbor_response_too_large"
            elif result.difficulty["branching_factor"] < settings["min_branching_factor"]:
                reason = "insufficient_branching"
            else:
                equivalent = shorter.get(frozenset(result.answers))
                if equivalent is not None and len(equivalent) < len(program.relation_path):
                    reason = "shorter_equivalent_answer_set"
            if reason:
                rejected[f"{len(program.relation_path)}hop:{reason}"] += 1
                continue
            if distances is None:
                distances = graph.shortest_distances(start)
            candidates.append({
                "query_instance_id": instance_id(start, program.relation_path, result.answers),
                "program_id": program.id,
                "start_entity": start,
                "answer_ids": result.answers,
                "hop_count": len(program.relation_path),
                "relation_path": list(program.relation_path),
                "gold_entity_paths": result.paths,
                "answer_path_counts": result.path_counts,
                "shortest_path_hops": {a: distances[a] for a in result.answers},
                "shortest_equivalent_program_length": len(program.relation_path),
                "difficulty": result.difficulty,
            })
    return candidates, dict(sorted(rejected.items()))
