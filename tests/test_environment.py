import copy
import json

import pytest

from kg_rlvr.data import public_prompt, read_questions
from kg_rlvr.training.environment import EpisodeLimits, GraphEnvironment, parse_qwen_action
from kg_rlvr.training.rewards import score_episode
from kg_rlvr.training.sampling import DEFAULT_HOP_EXPOSURE, exposure_schedule


def test_saved_graph_data_and_split_separation(tasks):
    graph, store = tasks
    assert len(graph.entities) == 2000 and len(graph.triples) == 5000
    assert {q["split"] for q in read_questions("data")} == {"train", "test"}
    train, test = store.select("train", [1, 2, 3, 4]), store.select("test", [1, 2, 3, 4])
    assert set(train).isdisjoint(test)
    assert len(store.select("train")) == 933 and len(test) == 493
    assert all(len(store.record(q)["answer_ids"]) == 1 for q in train + test)
    key = lambda q: (q["start_entity"], tuple(q["relation_path"]), tuple(q["answer_ids"]))
    assert {key(store.record(q)) for q in train}.isdisjoint(key(store.record(q)) for q in test)


def test_prompt_contains_only_public_inputs(tasks):
    graph, store = tasks
    record = copy.deepcopy(store.record(store.select("train", [2], count=1)[0]))
    original = public_prompt(graph, record)
    record.update(
        answer_ids=["PRIVATE"], relation_path=["PRIVATE"], gold_entity_paths=[["PRIVATE"]]
    )
    assert public_prompt(graph, record) == original
    assert "PRIVATE" not in json.dumps(original)


def test_actual_traversal_requires_discovery_and_explicit_submission(tasks):
    graph, store = tasks
    q = store.record(store.select("train", [2], count=1)[0])
    env = GraphEnvironment(graph, q["start_entity"])
    assert not env.step("submit_answer", {"entity_id": q["answer_ids"][0]})["ok"]
    path = q["gold_entity_paths"][0]
    for head, relation, tail in zip(path, q["relation_path"], path[1:]):
        result = env.step("get_neighbors", {"entity_id": head, "relation_id": relation})
        assert result["ok"] and tail in [e["id"] for e in result["entities"]]
    score = score_episode(None, q["answer_ids"], env.trajectory)
    assert score["answer_correct"] == 0 and score["training_reward"] == 0.2
    result = env.step("submit_answer", {"entity_id": path[-1]})
    assert result["ok"] and env.done
    assert "reward" not in result and "answer_ids" not in result
    assert (
        score_episode(env.submitted_answer, q["answer_ids"], env.trajectory)["training_reward"]
        == 1.2
    )
    env.reset()
    assert env.discovered == {q["start_entity"]} and not env.done


def test_tool_budget_and_bad_arguments(tasks):
    graph, store = tasks
    q = store.record(store.select("train", count=1)[0])
    env = GraphEnvironment(graph, q["start_entity"], EpisodeLimits(max_tool_calls=2))
    assert not env.step("unknown", {})["ok"]
    assert not env.step("get_neighbors", {"entity_id": q["start_entity"], "relation_id": 3})["ok"]
    assert env.calls == env.invalid_calls == 2 and env.termination_reason == "tool_budget"


def test_inverse_navigation_is_the_reverse_of_forward_navigation(tasks):
    graph, _ = tasks
    researcher = next(q for q, spec in graph.entities.items() if spec["type"] == "researcher")
    university = graph.get_neighbors(researcher, "works_at")[0]
    assert researcher in graph.get_neighbors(university, "inv:works_at")
    assert "inv:works_at" in graph.get_relations(university)


def test_parser_accepts_one_call_and_rejects_multiple_or_malformed_calls():
    text = '<tool_call>{"name":"submit_answer","arguments":{"entity_id":"x"}}</tool_call><|im_end|>'
    assert parse_qwen_action(text) == ("submit_answer", {"entity_id": "x"})
    for bad in [text + text, "<tool_call>{invalid}</tool_call>", "ordinary text"]:
        with pytest.raises(ValueError):
            parse_qwen_action(bad)


def test_hop_exposure_matches_requested_groups(tasks):
    _, store = tasks
    plan = exposure_schedule(store, store.select("train"), 30, DEFAULT_HOP_EXPOSURE)
    from collections import Counter

    assert Counter(store.record(q)["hop_count"] for q in plan) == {1: 6, 2: 12, 3: 12}
