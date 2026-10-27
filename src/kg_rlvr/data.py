"""Load saved questions and keep private verifier metadata outside model prompts."""

import copy
import hashlib
import json
from pathlib import Path
import random

from .graph_index import GraphIndex
from .training.rewards import score_episode


def read_questions(directory):
    return [
        json.loads(line)
        for split in ("train", "test")
        for line in (Path(directory) / "rl" / f"{split}.jsonl").read_text().splitlines()
        if line.strip()
    ]


def public_prompt(graph, record, max_tool_calls=8):
    """Only the question and starting entity enter the model-visible input."""
    entity = graph.get_entity(record["start_entity"])
    instruction = (
        "Navigate the fictional research graph to answer the question. Submit ONE correct entity ID using submit_answer; "
        "The question has one correct answer. "
        "Use only discovered entities, beginning with the starting ID. "
        "Identify the entity type requested by the question. Follow every relationship needed to answer the whole question; "
        "a discovered neighbor may be an intermediate entity that requires further traversal. "
        "Before submitting, check that the selected entity's type and the traversed relationships satisfy the question. "
        "get_relations lists available operations; get_neighbors returns all neighboring IDs, names and types. "
        "Use get_relations on intermediate entities to discover how to continue. "
        "inv: means permitted inverse traversal. Previously discovered entities remain accessible. "
        f"You have at most {max_tool_calls} tool calls, including invalid calls and submission. "
        "Issue exactly one tool call per assistant turn using the supplied tool format. Stop after submitting."
    )
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": f"Question: {record['question']}\nStarting entity: {entity['id']} ({entity['name']}; {entity['type']})",
        },
    ]


def prompt_key(prompt):
    return hashlib.sha256(
        json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class TaskStore:
    def __init__(self, graph, records, max_tool_calls=12):
        self.graph = graph
        self._records = {q["id"]: copy.deepcopy(q) for q in records}
        self._prompts = {
            qid: public_prompt(graph, q, max_tool_calls) for qid, q in self._records.items()
        }
        self._prompt_lookup = {}
        for qid in sorted(self._prompts):
            self._prompt_lookup.setdefault(prompt_key(self._prompts[qid]), qid)
        self.selection_statistics = []
        for split in ("train", "test"):
            for hop in (1, 2, 3, 4):
                rows = [q for q in records if q["split"] == split and q["hop_count"] == hop]
                keys = {
                    (q["start_entity"], tuple(q["relation_path"]), q["answer_id"]) for q in rows
                }
                self.selection_statistics.append(
                    {
                        "split": split,
                        "hop_count": hop,
                        "questions": len(rows),
                        "unique_instances": len(keys),
                    }
                )

    def prompt(self, task_id):
        return copy.deepcopy(self._prompts[task_id])

    def record(self, task_id):
        return copy.deepcopy(self._records[task_id])

    def resolve_prompt(self, prompt):
        return self._prompt_lookup[prompt_key(prompt)]

    def reward(self, task_id, submitted_answer):
        return float(submitted_answer == self._records[task_id]["answer_id"])

    def verify_episode(self, task_id, submitted_answer, tool_events):
        return score_episode(submitted_answer, self._records[task_id]["answer_id"], tool_events)

    def select(
        self, split, hops=(1, 2, 3), count=None, seed=41, unique_instances=True, balanced=False
    ):
        if not hops or set(hops) - {1, 2, 3, 4}:
            raise ValueError("Choose at least one hop category from 1, 2, 3, 4")
        rows = sorted(
            (q for q in self._records.values() if q["split"] == split and q["hop_count"] in hops),
            key=lambda q: q["id"],
        )
        if unique_instances:
            grouped = {}
            for q in rows:
                key = (q["start_entity"], tuple(q["relation_path"]), q["answer_id"])
                grouped.setdefault(key, q)
            rows = list(grouped.values())
        rng = random.Random(seed)
        rng.shuffle(rows)
        if balanced:
            buckets = {h: [q for q in rows if q["hop_count"] == h] for h in sorted(hops)}
            rows = []
            while any(buckets.values()):
                for h in sorted(hops):
                    if buckets[h]:
                        rows.append(buckets[h].pop())
        if count is not None:
            if count < 1:
                raise ValueError("Example counts must be positive")
            rows = rows[:count]
        if not rows:
            raise ValueError(f"No tasks for split={split}, hops={hops}")
        return [q["id"] for q in rows]


def load_tasks(data_root, max_tool_calls=12):
    root = Path(data_root)
    graph = GraphIndex.load(root / "graph")
    return graph, TaskStore(graph, read_questions(root), max_tool_calls)
