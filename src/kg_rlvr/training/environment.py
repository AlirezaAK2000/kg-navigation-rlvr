"""Real graph tools, private answer/retrieval verifiers and strict Qwen parsing.

This module uses the existing RDFLib-backed graph/dataset utilities.
Gold information never enters the prompt builder or graph environment.
"""

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import random
import re

from ..dataset_builder import read_questions
from ..graph_index import GraphIndex, NavigationSession
from ..validators import validate_dataset
from ..question_templates import SINGLE_ANSWER_TEMPLATES


RETRIEVAL_BONUS = 0.2
REWARD_SPEC = {"version": "answer_plus_retrieval_v1", "answer_weight": 1.0,
               "retrieval_bonus": RETRIEVAL_BONUS, "retrieval_source": "successful_get_neighbors",
               "evaluation_reward": "binary_answer_membership"}


@dataclass(frozen=True)
class EpisodeLimits:
    max_tool_calls: int = 8
    max_neighbors: int = 16
    max_observation_chars: int = 4000
    max_turn_tokens: int = 128
    max_completion_tokens: int = 1536  # includes environment tokens
    max_prompt_tokens: int = 1024
    max_context_tokens: int = 3072

    def __post_init__(self):
        if any(value < 1 for value in vars(self).values()):
            raise ValueError("All episode limits must be positive")
        if self.max_turn_tokens > self.max_completion_tokens:
            raise ValueError("A turn cannot exceed the episode completion budget")
        if self.max_prompt_tokens + self.max_completion_tokens > self.max_context_tokens:
            raise ValueError("Prompt + completion budgets must fit the context limit")


def _schema(name, description, parameters):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {key: {"type": "string", "description": value}
                           for key, value in parameters.items()}, "required": list(parameters), "additionalProperties": False}}}


TOOL_SCHEMAS = [
    _schema("get_relations", "List available directed relation IDs for a discovered graph entity. inv: denotes a permitted inverse traversal.",
            {"entity_id": "A previously discovered entity ID, including the starting entity."}),
    _schema("get_neighbors", "Traverse one available relation from a discovered entity. Returns ALL neighbors with IDs, names and types; they remain accessible later.",
            {"entity_id": "A previously discovered entity ID.", "relation_id": "A relation operation ID, including inv: when explicitly available."}),
    _schema("submit_answer", "Submit ONE discovered entity ID and end the episode. The response acknowledges submission without revealing correctness.",
            {"entity_id": "The single entity ID you select as an answer."}),
]


def public_prompt(graph, record, max_tool_calls=8, single_answer_only=False):
    """Explicit allowlist. Deliberately differs from the complete-set EDA prompt."""
    entity = graph.get_entity(record["start_entity"])
    instruction = (
        "Navigate the fictional research graph to answer the question. Submit ONE correct entity ID using submit_answer; "
        + ("The question has one correct answer. " if single_answer_only else "If several answers exist, any one is accepted. ")
        + "Use only discovered entities, beginning with the starting ID. "
        "Identify the entity type requested by the question. Follow every relationship needed to answer the whole question; "
        "a discovered neighbor may be an intermediate entity that requires further traversal. "
        "Before submitting, check that the selected entity's type and the traversed relationships satisfy the question. "
        "get_relations lists available operations; get_neighbors returns all neighboring IDs, names and types. "
        "Use get_relations on intermediate entities to discover how to continue. "
        "inv: means permitted inverse traversal. Previously discovered entities remain accessible. "
        f"You have at most {max_tool_calls} tool calls, including invalid calls and submission. "
        "Issue exactly one tool call per assistant turn using the supplied tool format. Stop after submitting."
    )
    return [{"role": "system", "content": instruction}, {"role": "user", "content":
            f"Question: {record['question']}\nStarting entity: {entity['id']} ({entity['name']}; {entity['type']})"}]


def prompt_key(prompt):
    return hashlib.sha256(json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class TaskStore:
    """Private verifier metadata. The model receives only allowlisted prompts."""
    def __init__(self, graph, records, max_tool_calls=8, single_answer_only=False):
        self.graph = graph
        self.single_answer_only = single_answer_only
        self.selection_statistics = []
        for split in sorted({q["split"] for q in records if "split" in q}):
            for hop in sorted({q["hop_count"] for q in records if q.get("split") == split}):
                rows = [q for q in records if q.get("split") == split and q["hop_count"] == hop]
                eligible = [q for q in rows if not single_answer_only or len(q["answer_ids"]) == 1]
                keys = {(q["start_entity"], tuple(q["relation_path"]), tuple(sorted(q["answer_ids"]))) for q in eligible}
                self.selection_statistics.append({"split": split, "hop_count": hop, "source_questions": len(rows),
                    "eligible_questions": len(eligible), "removed_questions": len(rows) - len(eligible),
                    "eligible_unique_instances": len(keys)})
        if single_answer_only:
            records = [copy.deepcopy(q) for q in records if len(q["answer_ids"]) == 1]
            for q in records:
                template = SINGLE_ANSWER_TEMPLATES[q["program_id"]]
                q["question"] = template.format(start=graph.get_entity(q["start_entity"])["name"])
                q["wording_template_id"] = q["program_id"] + ":single_v1"
        self._records = {q["id"]: copy.deepcopy(q) for q in records}
        if len(self._records) != len(records):
            raise ValueError("Duplicate task IDs")
        self._prompts = {qid: public_prompt(graph, q, max_tool_calls, single_answer_only) for qid, q in self._records.items()}
        self._prompt_lookup = {}
        for qid in sorted(self._prompts):
            key = prompt_key(self._prompts[qid])
            if key in self._prompt_lookup:
                other = self._records[self._prompt_lookup[key]]
                current = self._records[qid]
                # Singular wording collapses source paraphrases. Resolve those
                # aliases to the same first ID chosen by select(unique_instances=True).
                fields = ("start_entity", "relation_path", "answer_ids", "split")
                if not single_answer_only or any(current.get(f) != other.get(f) for f in fields):
                    raise ValueError("Public prompts must uniquely identify tasks")
            else:
                self._prompt_lookup[key] = qid

    def prompt(self, task_id):
        return copy.deepcopy(self._prompts[task_id])

    def resolve_prompt(self, prompt):
        try:
            return self._prompt_lookup[prompt_key(prompt)]
        except KeyError as error:
            raise ValueError("Rollout prompt is not in the validated task store") from error

    def record(self, task_id):
        return copy.deepcopy(self._records[task_id])

    def reward(self, task_id, submitted_answer):
        return float(submitted_answer in self._records[task_id]["answer_ids"])

    def verify_episode(self, task_id, submitted_answer, tool_events):
        """Private scoring from executed observations; never parse model claims.

        Successful retrieval observations are independently checked against the
        graph. Failed calls, relation lists and submission acknowledgments cannot
        earn the once-per-episode bonus. No score is exposed through graph tools.
        """
        retrieved = set()
        for event in tool_events:
            observation = event["observation"]
            if event["name"] != "get_neighbors" or observation.get("ok") is not True:
                continue
            arguments = event["arguments"]
            head, relation = arguments["entity_id"], arguments["relation_id"]
            ids = [entity["id"] for entity in observation["entities"]]
            expected = self.graph.get_neighbors(head, relation)
            if (observation.get("entity_id") != head or observation.get("relation_id") != relation
                    or sorted(ids) != sorted(expected)):
                raise ValueError("Retrieval verifier: observation does not match the executed graph query")
            retrieved.update(ids)
        correct = self.reward(task_id, submitted_answer)
        hit = bool(retrieved.intersection(self._records[task_id]["answer_ids"]))
        bonus = RETRIEVAL_BONUS * hit
        return {"answer_correct": correct, "gold_entity_retrieved": hit,
                "retrieval_reward": bonus, "training_reward": correct + bonus}

    def select(self, split, hops=(1, 2, 3), count=None, seed=41, unique_instances=True, balanced=False):
        if not hops or set(hops) - {1, 2, 3, 4}:
            raise ValueError("Choose at least one hop category from 1, 2, 3, 4")
        rows = sorted((q for q in self._records.values() if q["split"] == split and q["hop_count"] in hops), key=lambda q: q["id"])
        if unique_instances:
            grouped = {}
            for q in rows:
                key = (q["start_entity"], tuple(q["relation_path"]), tuple(sorted(q["answer_ids"])))
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
            raise ValueError(f"No eligible tasks for split={split}, hops={hops}")
        return [q["id"] for q in rows]


def load_tasks(data_root, max_tool_calls=8, validate=True, single_answer_only=False):
    root = Path(data_root)
    graph = GraphIndex.load(root / "graph")
    records = read_questions(root)
    required = {"id", "question", "start_entity", "answer_ids", "hop_count", "relation_path", "split", "gold_entity_paths"}
    for q in records:
        missing = sorted(k for k in required if k not in q or q[k] is None)
        if missing:
            raise ValueError(f"Task {q.get('id', '<unknown>')} missing critical fields: {missing}")
        if not q["answer_ids"]:
            raise ValueError(f"Task {q['id']} has an empty gold answer set")
    if not records:
        raise ValueError(f"No QA records under {root / 'rl'}")
    if {q["split"] for q in records} - {"train", "test"}:
        raise ValueError("Only train and test splits are supported; migrate older validation records into test")
    for split in ("train", "test"):
        if not any(q["split"] == split for q in records):
            raise ValueError(f"Required split is missing: {split}")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    validation = validate_dataset(graph, records, config) if validate else None
    if validation is not None and not validation["valid"]:
        raise ValueError("Dataset validation failed: " + json.dumps(validation["errors"][:10]))
    return graph, TaskStore(graph, records, max_tool_calls, single_answer_only), validation


class GraphEnvironment:
    """One real isolated graph-navigation episode; has no gold-answer metadata."""
    def __init__(self, graph, start_entity, limits=None):
        self._graph = graph
        self._limits = limits or EpisodeLimits()
        self._start = start_entity
        self.reset()

    def reset(self):
        self._session = NavigationSession(self._graph, self._start)
        self.calls = 0
        self.invalid_calls = 0
        self.done = False
        self.submitted_answer = None
        self.termination_reason = None
        self.trajectory = []

    def _finish(self, reason):
        if not self.done:
            self.done, self.termination_reason = True, reason

    def _known(self, entity_id):
        if not isinstance(entity_id, str) or entity_id not in self._session.discovered:
            raise ValueError("Use a discovered entity ID")

    def _invoke(self, name, arguments, operation):
        if self.done:
            return {"ok": False, "error": "episode_complete"}
        if self.calls >= self._limits.max_tool_calls:
            self._finish("tool_budget")
            return {"ok": False, "error": "tool_budget"}
        self.calls += 1
        try:
            result = {"ok": True, **operation()}
        except (ValueError, TypeError, KeyError) as error:
            self.invalid_calls += 1
            result = {"ok": False, "error": str(error)}
        self.trajectory.append({"index": self.calls, "name": name, "arguments": copy.deepcopy(arguments), "observation": result})
        if not self.done and self.calls == self._limits.max_tool_calls:
            self._finish("tool_budget")
        return result

    def get_relations(self, entity_id: str) -> dict:
        """List available directed operations for a discovered entity.

        Args:
            entity_id: A previously discovered entity ID.
        """
        def operation():
            self._known(entity_id)
            return {"entity_id": entity_id, "relations": self._session.get_relations(entity_id)}
        return self._invoke("get_relations", {"entity_id": entity_id}, operation)

    def get_neighbors(self, entity_id: str, relation_id: str) -> dict:
        """Traverse a directed relation and reveal all neighboring IDs and labels.

        Args:
            entity_id: A previously discovered entity ID.
            relation_id: An available forward or explicitly permitted inverse operation.
        """
        def operation():
            self._known(entity_id)
            if not isinstance(relation_id, str):
                raise ValueError("relation_id must be a string")
            neighbors = self._graph.get_neighbors(entity_id, relation_id)
            observation = {"entity_id": entity_id, "relation_id": relation_id,
                           "entities": [self._graph.get_entity(e) for e in neighbors]}
            if len(neighbors) > self._limits.max_neighbors or len(json.dumps(observation)) > self._limits.max_observation_chars:
                raise ValueError("Observation exceeds configured response limit; no neighbors were hidden or truncated")
            self._session.get_neighbors(entity_id, relation_id)
            return observation
        return self._invoke("get_neighbors", {"entity_id": entity_id, "relation_id": relation_id}, operation)

    def submit_answer(self, entity_id: str) -> dict:
        """Submit one discovered ID and end the episode without scoring it.

        Args:
            entity_id: The single previously discovered entity ID selected as the answer.
        """
        def operation():
            self._known(entity_id)
            self._session.submit_answer(entity_id)
            self.submitted_answer = entity_id
            self._finish("submitted")
            return {"submitted": entity_id, "episode_complete": True}
        return self._invoke("submit_answer", {"entity_id": entity_id}, operation)

    def _invalid(self, code, arguments=None):
        def invalid():
            raise ValueError(code)
        return self._invoke("invalid_tool_call", arguments or {}, invalid)

    def _dispatch(self, name, arguments):
        if name not in {s["function"]["name"] for s in TOOL_SCHEMAS}:
            return self._invalid("unknown_tool", {"name": name})
        schema = next(s["function"]["parameters"] for s in TOOL_SCHEMAS if s["function"]["name"] == name)
        if not isinstance(arguments, dict) or set(arguments) != set(schema["required"]) or any(not isinstance(v, str) for v in arguments.values()):
            return self._invalid("invalid_arguments")
        return getattr(self, name)(**arguments)


def parse_qwen_action(text):
    """Parse ONE Qwen/Hermes JSON tool call. No eval, repair or invented actions."""
    calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S)
    if not calls:
        raise ValueError("malformed_call" if "<tool_call" in text else "no_tool_call")
    if len(calls) != 1 or text.count("<tool_call>") != 1:
        raise ValueError("multiple_calls_in_turn")
    try:
        call = json.loads(calls[0])
    except json.JSONDecodeError as error:
        raise ValueError("malformed_call") from error
    if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
        raise ValueError("invalid_call_structure")
    return call["name"], call["arguments"]


def verify_group_prompts(prompts, num_generations):
    """TRL 1.8.0 RepeatSampler ALREADY repeats rows; never duplicate them again."""
    if num_generations < 2 or len(prompts) % num_generations:
        raise ValueError("Rollout batch must contain complete GRPO groups")
    for offset in range(0, len(prompts), num_generations):
        group = prompts[offset:offset + num_generations]
        if len({prompt_key(p) for p in group}) != 1:
            raise ValueError("A GRPO group contains different questions or initial prompts")


def make_reward(store):
    def graph_answer_reward(completions, task_ids, submitted_answers, tool_events, **kwargs):
        # These values are supplied by the trusted rollout engine, never parsed
        # from model prose. Recompute both components from trusted engine data.
        if not len(completions) == len(task_ids) == len(submitted_answers) == len(tool_events):
            raise ValueError("Reward/rollout length mismatch")
        return [store.verify_episode(task_id, answer, events)["training_reward"]
                for task_id, answer, events in zip(task_ids, submitted_answers, tool_events)]
    return graph_answer_reward
