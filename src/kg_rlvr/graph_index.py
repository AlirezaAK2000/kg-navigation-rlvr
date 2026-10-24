"""Public navigation API and deterministic reference execution.

get_relations(id) -> sorted available operation IDs (including permitted inverses).
get_neighbors(id, operation) -> ALL sorted neighbor IDs; never silently truncated.
get_entity(id) -> public ID, type and readable name.
submit_answer(ids) -> canonical ID envelope only; no correctness/reward judgment.
NavigationSession permits operations only on previously discovered entities.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path

from .ontology import operation_schema


class GraphIndex:
    def __init__(self, entities, relations, triples):
        self.entities = entities
        self.relations = relations
        rows = list(map(tuple, triples))
        if len(rows) != len(set(rows)):
            raise ValueError("Duplicate triples")
        self.triples = tuple(sorted(rows))
        adjacency = defaultdict(lambda: defaultdict(set))
        for head, rel, tail in self.triples:
            if head not in entities or tail not in entities:
                raise ValueError("Triple refers to an unknown entity")
            domain, range_ = operation_schema(rel, relations)
            if entities[head]["type"] != domain or entities[tail]["type"] != range_:
                raise ValueError(f"Ill-typed triple: {(head, rel, tail)}")
            adjacency[head][rel].add(tail)
            if relations[rel]["inverse_supported"]:
                adjacency[tail]["inv:" + rel].add(head)
        self.adjacency = {e: {r: tuple(sorted(ns)) for r, ns in rs.items()} for e, rs in adjacency.items()}

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with (directory / "triples.tsv").open() as stream:
            triples = [line.rstrip("\n").split("\t") for line in stream if line.strip()]
        return cls(json.loads((directory / "entities.json").read_text()), json.loads((directory / "relations.json").read_text()), triples)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "entities.json").write_text(json.dumps(self.entities, indent=2, sort_keys=True) + "\n")
        (directory / "relations.json").write_text(json.dumps(self.relations, indent=2, sort_keys=True) + "\n")
        (directory / "triples.tsv").write_text("".join("\t".join(row) + "\n" for row in self.triples))

    def get_entity(self, entity_id):
        if entity_id not in self.entities:
            raise ValueError(f"Unknown entity: {entity_id}")
        return {"id": entity_id, **self.entities[entity_id]}

    def get_relations(self, entity_id):
        self.get_entity(entity_id)
        return sorted(self.adjacency.get(entity_id, {}))

    def get_neighbors(self, entity_id, relation_id):
        self.get_entity(entity_id)
        domain, _ = operation_schema(relation_id, self.relations)
        if self.entities[entity_id]["type"] != domain:
            raise ValueError(f"Operation {relation_id} does not apply to {entity_id}")
        return list(self.adjacency.get(entity_id, {}).get(relation_id, ()))

    def submit_answer(self, entity_ids):
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        for entity_id in entity_ids:
            self.get_entity(entity_id)
        return {"answer_ids": sorted(set(entity_ids))}

    def shortest_distances(self, start):
        """Unlabelled shortest distances using only permitted navigation operations."""
        result, pending = {start: 0}, deque([start])
        while pending:
            entity = pending.popleft()
            for neighbors in self.adjacency.get(entity, {}).values():
                for neighbor in neighbors:
                    if neighbor not in result:
                        result[neighbor] = result[entity] + 1
                        pending.append(neighbor)
        return result

    def shorter_results(self, start, max_depth):
        """All distinct nonempty answer sets obtainable in <= max_depth steps.

        Dynamic programming merges equal frontier sets: subsequent transitions are
        identical. This is exhaustive over available typed relation operations,
        includes zero hops, and is NOT restricted to the question template catalog.
        Empty frontiers can never recover and are safely pruned.
        """
        initial = frozenset([start])
        seen = {initial: ()}
        frontier = [(initial, ())]
        for _ in range(max_depth):
            next_frontier = []
            for members, path in frontier:
                choices = set().union(*(self.adjacency.get(e, {}).keys() for e in members))
                for rel in sorted(choices):
                    result = frozenset(n for e in members for n in self.adjacency.get(e, {}).get(rel, ()))
                    if result and result not in seen:
                        new_path = path + (rel,)
                        seen[result] = new_path
                        next_frontier.append((result, new_path))
            frontier = next_frontier
        return seen


class NavigationSession:
    def __init__(self, graph, start_entity):
        graph.get_entity(start_entity)
        self.graph = graph
        self.discovered = {start_entity}
        self.tool_calls = 0
        self.max_response_size = 0

    def _check(self, entity):
        if entity not in self.discovered:
            raise ValueError(f"Entity has not been discovered: {entity}")

    def get_entity(self, entity):
        self._check(entity)
        self.tool_calls += 1
        return self.graph.get_entity(entity)

    def get_relations(self, entity):
        self._check(entity)
        self.tool_calls += 1
        return self.graph.get_relations(entity)

    def get_neighbors(self, entity, relation):
        self._check(entity)
        self.tool_calls += 1
        result = self.graph.get_neighbors(entity, relation)
        self.max_response_size = max(self.max_response_size, len(result))
        self.discovered.update(result)
        return result

    def submit_answer(self, entity_ids):
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        for entity in entity_ids:
            self._check(entity)
        self.tool_calls += 1
        return self.graph.submit_answer(entity_ids)


@dataclass
class Execution:
    answers: list
    paths: list
    path_counts: dict
    difficulty: dict


def execute(graph, start, relation_path):
    """Set-valued query via the PUBLIC discovery-gated interface, without truncation.

    Keeps one deterministic witness per answer and counts every program-conforming
    walk. Tool cost includes get_relations/get_neighbors per frontier node and one
    final submit_answer. It does not measure an uninformed model's search cost.
    """
    from .ontology import program_end_type
    program_end_type(graph.get_entity(start)["type"], relation_path, graph.relations)
    session = NavigationSession(graph, start)
    paths, counts = {start: [start]}, {start: 1}
    degrees, relation_options, query_neighbors = [], [], []
    distractors, widths = 0, [1]
    for relation in relation_path:
        next_paths, next_counts = {}, defaultdict(int)
        for entity in sorted(paths):
            available = session.get_relations(entity)
            options = graph.adjacency.get(entity, {})
            relation_options.append(len(available))
            degrees.append(sum(len(v) for v in options.values()))
            distractors += sum(len(v) for r, v in options.items() if r != relation)
            neighbors = session.get_neighbors(entity, relation) if relation in available else []
            query_neighbors.append(len(neighbors))
            for neighbor in neighbors:
                next_paths.setdefault(neighbor, paths[entity] + [neighbor])
                next_counts[neighbor] += counts[entity]
        paths, counts = next_paths, dict(next_counts)
        widths.append(len(paths))
    answers = session.submit_answer(sorted(paths))["answer_ids"]
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return Execution(answers, [paths[a] for a in answers], {a: counts[a] for a in answers}, {
        "branching_factor": mean(degrees),
        "relation_options_mean": mean(relation_options),
        "query_neighbors_mean": mean(query_neighbors),
        "distractor_count": distractors,
        "frontier_widths": widths,
        "reference_tool_calls": session.tool_calls,
        "max_neighbor_response": session.max_response_size,
        "program_conforming_walks": sum(counts.values()),
    })
