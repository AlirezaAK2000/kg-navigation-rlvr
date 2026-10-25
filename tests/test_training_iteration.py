"""Single-answer selection, private retrieval verification and actual exposure.

Scripted graph fixtures and sampler checks, never model-performance evidence.
"""

from collections import Counter
import copy
import json

import pytest

from kg_rlvr.question_templates import PROGRAM_BY_ID, SINGLE_ANSWER_TEMPLATES
from kg_rlvr.training.environment import GraphEnvironment, TaskStore, load_tasks, make_reward, public_prompt
from kg_rlvr.training.rollouts import summarize_episodes
from kg_rlvr.training.sampling import DEFAULT_HOP_EXPOSURE, exposure_schedule


@pytest.fixture(scope="module")
def single_store():
    return load_tasks("data", single_answer_only=True)[1]


def test_single_answer_subset_filters_before_caps_and_preserves_source(single_store):
    from kg_rlvr.dataset_builder import read_questions
    source = read_questions("data")
    by_id = {q["id"]: q for q in source}
    assert set(SINGLE_ANSWER_TEMPLATES) == set(PROGRAM_BY_ID)
    for split, expected in [("train", 2449), ("validation", 364), ("test", 619)]:
        ids = single_store.select(split, [1, 2, 3, 4], unique_instances=False)
        assert len(ids) == expected
        assert set(ids) == {q["id"] for q in source if q["split"] == split and len(q["answer_ids"]) == 1}
        for qid in ids:
            q, old = single_store.record(qid), by_id[qid]
            assert q["split"] == split and q["answer_ids"] == old["answer_ids"]
            assert q["relation_path"] == old["relation_path"] and q["gold_entity_paths"] == old["gold_entity_paths"]
            assert q["question"] == SINGLE_ANSWER_TEMPLATES[q["program_id"]].format(
                start=single_store.graph.entities[q["start_entity"]]["name"])
    limited = single_store.select("train", [1, 2, 3], count=100)
    assert len(limited) == 100 and all(len(single_store.record(q)["answer_ids"]) == 1 for q in limited)
    report = single_store.selection_statistics
    assert sum(r["source_questions"] for r in report) == 3734
    assert sum(r["removed_questions"] for r in report) == 302
    # Wording is produced in a separate store; loading never mutates source rows.
    assert source == read_questions("data")


def test_single_answer_prompt_never_uses_private_answers(single_store):
    q = single_store.record(single_store.select("train", [2], count=1)[0])
    changed = copy.deepcopy(q)
    changed.update(answer_ids=["PRIVATE"], relation_path=["PRIVATE"], gold_entity_paths=[["PRIVATE"]])
    assert public_prompt(single_store.graph, q, single_answer_only=True) == public_prompt(
        single_store.graph, changed, single_answer_only=True)
    assert "one correct answer" in single_store.prompt(q["id"])[0]["content"]
    assert "PRIVATE" not in json.dumps(single_store.prompt(q["id"]))


def reward_fixture(graph):
    start = "researcher_000000"
    university = graph.get_neighbors(start, "works_at")[0]
    city = graph.get_neighbors(university, "located_in_city")[0]
    record = {"id": "fixture", "start_entity": start, "question": "Which city hosts the employer?", "answer_ids": [city]}
    return TaskStore(graph, [record]), record, university, city


def test_retrieval_bonus_is_once_and_independent_of_final_submission(graph):
    store, q, university, city = reward_fixture(graph)
    env = GraphEnvironment(graph, q["start_entity"])
    env.get_neighbors(q["start_entity"], "works_at")
    assert store.verify_episode(q["id"], university, env.trajectory)["training_reward"] == 0.0
    env.get_neighbors(university, "located_in_city")
    env.get_neighbors(university, "located_in_city")
    env.get_relations(city)
    assert store.verify_episode(q["id"], None, env.trajectory) == {
        "answer_correct": 0.0, "gold_entity_retrieved": True, "retrieval_reward": 0.2, "training_reward": 0.2}
    ack = env.submit_answer(city)
    verified = store.verify_episode(q["id"], env.submitted_answer, env.trajectory)
    assert verified["answer_correct"] == 1.0 and verified["training_reward"] == 1.2
    assert not {"reward", "answer_correct", "gold_entity_retrieved"}.intersection(ack)
    fresh = GraphEnvironment(graph, q["start_entity"])
    assert store.verify_episode(q["id"], None, fresh.trajectory)["training_reward"] == 0.0


def test_verifier_ignores_failed_calls_and_rejects_fabricated_observations(graph):
    store, q, _, city = reward_fixture(graph)
    fake = {"name": "get_neighbors", "arguments": {"entity_id": q["start_entity"], "relation_id": "works_at"},
            "observation": {"ok": False, "entities": [{"id": city}]}}
    assert store.verify_episode(q["id"], None, [fake])["training_reward"] == 0.0
    fake["observation"].update(ok=True, entity_id=q["start_entity"], relation_id="works_at")
    with pytest.raises(ValueError, match="does not match"):
        store.verify_episode(q["id"], None, [fake])
    fake["name"] = "get_relations"
    assert store.verify_episode(q["id"], None, [fake])["retrieval_reward"] == 0.0
    score = make_reward(store)
    # Prose claiming a gold ID never supplies retrieval evidence.
    assert score([f"I retrieved {city}"], [q["id"]], [None], [[]]) == [0.0]
    with pytest.raises(ValueError, match="length mismatch"):
        score(["x"], [q["id"]], [None], [])


def test_actual_reward_function_recomputes_combined_reward(graph):
    store, q, university, city = reward_fixture(graph)
    env = GraphEnvironment(graph, q["start_entity"])
    env.get_neighbors(q["start_entity"], "works_at")
    env.get_neighbors(university, "located_in_city")
    score = make_reward(store)
    assert score(["x"] * 3, [q["id"]] * 3, [None, city, university],
                 [env.trajectory] * 3) == [0.2, 1.2, 0.2]


@pytest.mark.parametrize("groups,counts", [(5, {1: 1, 2: 2, 3: 2}), (8, {1: 2, 2: 3, 3: 3}),
                                          (30, {1: 6, 2: 12, 3: 12}), (100, {1: 20, 2: 40, 3: 40})])
def test_exposure_controls_sampled_groups_and_resume_prefix(single_store, groups, counts):
    pool = single_store.select("train", [1, 2, 3], count=100)
    plan = exposure_schedule(single_store, pool, groups, DEFAULT_HOP_EXPOSURE)
    hops = [single_store.record(q)["hop_count"] for q in plan]
    assert dict(Counter(hops)) == counts
    assert plan == exposure_schedule(single_store, pool, groups + 40, DEFAULT_HOP_EXPOSURE)[:groups]
    assert plan == exposure_schedule(single_store, pool, groups, DEFAULT_HOP_EXPOSURE)
    other_seed = exposure_schedule(single_store, pool, groups, DEFAULT_HOP_EXPOSURE, seed=99)
    assert [single_store.record(q)["hop_count"] for q in other_seed] == hops and other_seed != plan
    assert set(plan) <= set(pool)
    for offset in range(0, groups - 4, 5):
        assert Counter(hops[offset:offset + 5]) == {1: 1, 2: 2, 3: 2}


def test_exposure_cycles_existing_questions_without_relaxing_missing_hops(single_store):
    pool = [single_store.select("train", [h], count=1)[0] for h in (1, 2, 3)]
    assert Counter(exposure_schedule(single_store, pool, 100, DEFAULT_HOP_EXPOSURE)) == {
        pool[0]: 20, pool[1]: 40, pool[2]: 40}
    with pytest.raises(ValueError, match="No selected training questions"):
        exposure_schedule(single_store, pool[:2], 10, DEFAULT_HOP_EXPOSURE)
    for weights in [{1: -1}, {1: float("nan")}, {5: 1}, {}]:
        with pytest.raises(ValueError):
            exposure_schedule(single_store, pool, 10, weights)


def test_summary_keeps_accuracy_binary_when_training_rewards_exceed_one():
    common = {"hop_count": 2, "tool_calls": 3, "invalid_tool_calls": 0,
              "submitted": True, "termination_reason": "submitted", "gold_entity_retrieved": True, "retrieval_reward": 0.2}
    metrics = summarize_episodes([{**common, "reward": 1.0, "training_reward": 1.2},
                                 {**common, "reward": 0.0, "training_reward": 0.2}])
    assert metrics["accuracy"] == metrics["mean_reward"] == 0.5
    assert metrics["retrieval_success_rate"] == 1 and metrics["mean_training_reward"] == 0.7
    assert metrics["mean_training_reward_by_hop"] == {"2": 0.7}


def test_notebook_defaults_and_identical_selection_across_phases():
    from pathlib import Path
    nb = json.loads(Path("notebooks/02_grpo_training_kaggle.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    parameters = next(s for s in code if "NUM_GENERATIONS =" in s)
    assert "NUM_GENERATIONS = 8" in parameters and "GRADIENT_ACCUMULATION_STEPS = 8" in parameters
    assert "TRAIN_HOP_EXPOSURE = {1: 0.2, 2: 0.4, 3: 0.4}" in parameters
    assert "SINGLE_ANSWER_ONLY = True" in parameters
    for source in code:
        compile(source, "notebook", "exec")


def test_manifest_roundtrip_retains_reward_filter_and_exposure():
    from kg_rlvr.training.environment import EpisodeLimits, REWARD_SPEC
    from kg_rlvr.training.grpo import training_manifest
    manifest = training_manifest("fixture", "revision", EpisodeLimits(), ["fixture"], 8, 8, 5e-6, 41,
                                 hop_weights=DEFAULT_HOP_EXPOSURE, single_answer_only=True)
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["reward_spec"] == REWARD_SPEC and manifest["single_answer_only"]
    assert manifest["training_exposure"]["hop_weights"] == {"1": 0.2, "2": 0.4, "3": 0.4}
