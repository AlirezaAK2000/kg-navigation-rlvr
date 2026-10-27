"""Stateful graph tools and parsing of one Qwen tool call per turn."""

from dataclasses import dataclass
import copy
import json
import re

from .rewards import REWARD_SPEC


@dataclass(frozen=True)
class EpisodeLimits:
    max_tool_calls: int = 8
    max_neighbors: int = 16
    max_observation_chars: int = 4000
    max_turn_tokens: int = 128
    max_completion_tokens: int = 1536
    max_prompt_tokens: int = 1024
    max_context_tokens: int = 3072


def _schema(name, description, parameters):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "string", "description": value}
                    for key, value in parameters.items()
                },
                "required": list(parameters),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema(
        "get_relations",
        "List available directed relation IDs for a discovered graph entity. inv: denotes a permitted inverse traversal.",
        {"entity_id": "A previously discovered entity ID, including the starting entity."},
    ),
    _schema(
        "get_neighbors",
        "Traverse one available relation from a discovered entity. Returns ALL neighbors with IDs, names and types; they remain accessible later.",
        {
            "entity_id": "A previously discovered entity ID.",
            "relation_id": "A relation operation ID, including inv: when explicitly available.",
        },
    ),
    _schema(
        "submit_answer",
        "Submit ONE discovered entity ID and end the episode. The response acknowledges submission without revealing correctness.",
        {"entity_id": "The single entity ID you select as an answer."},
    ),
]


class GraphEnvironment:
    """One real isolated graph-navigation episode; has no gold-answer metadata."""

    def __init__(self, graph, start_entity, limits=None):
        self._graph = graph
        self._limits = limits or EpisodeLimits()
        self._start = start_entity
        self.reset()

    def reset(self):
        self.discovered = {self._start}
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
        if not isinstance(entity_id, str) or entity_id not in self.discovered:
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
        self.trajectory.append(
            {
                "index": self.calls,
                "name": name,
                "arguments": copy.deepcopy(arguments),
                "observation": result,
            }
        )
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
            return {"entity_id": entity_id, "relations": self._graph.get_relations(entity_id)}

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
            observation = {
                "entity_id": entity_id,
                "relation_id": relation_id,
                "entities": [self._graph.get_entity(e) for e in neighbors],
            }
            if (
                len(neighbors) > self._limits.max_neighbors
                or len(json.dumps(observation)) > self._limits.max_observation_chars
            ):
                raise ValueError(
                    "Observation exceeds configured response limit; no neighbors were hidden or truncated"
                )
            self.discovered.update(neighbors)
            return observation

        return self._invoke(
            "get_neighbors", {"entity_id": entity_id, "relation_id": relation_id}, operation
        )

    def submit_answer(self, entity_id: str) -> dict:
        """Submit one discovered ID and end the episode without scoring it.

        Args:
            entity_id: The single previously discovered entity ID selected as the answer.
        """

        def operation():
            self._known(entity_id)
            self.submitted_answer = entity_id
            self._finish("submitted")
            return {"submitted": entity_id, "episode_complete": True}

        return self._invoke("submit_answer", {"entity_id": entity_id}, operation)

    def _invalid(self, code, arguments=None):
        def invalid():
            raise ValueError(code)

        return self._invoke("invalid_tool_call", arguments or {}, invalid)

    def step(self, name, arguments):
        if name not in {s["function"]["name"] for s in TOOL_SCHEMAS}:
            return self._invalid("unknown_tool", {"name": name})
        schema = next(
            s["function"]["parameters"] for s in TOOL_SCHEMAS if s["function"]["name"] == name
        )
        if (
            not isinstance(arguments, dict)
            or set(arguments) != set(schema["required"])
            or any(not isinstance(v, str) for v in arguments.values())
        ):
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
    if (
        not isinstance(call, dict)
        or set(call) != {"name", "arguments"}
        or not isinstance(call["name"], str)
        or not isinstance(call["arguments"], dict)
    ):
        raise ValueError("invalid_call_structure")
    return call["name"], call["arguments"]
