import copy
import json
import pytest

from kg_rlvr.training.environment import (
    EpisodeLimits, GraphEnvironment, TaskStore, TOOL_SCHEMAS, load_tasks,
    make_reward, parse_qwen_action, public_prompt, verify_group_prompts,
)
from kg_rlvr.training.rollouts import TokenLedger, run_episode, summarize_episodes


def test_existing_graph_and_question_loading():
    graph, store, report = load_tasks("data")
    assert len(graph.entities) == 2000 and len(graph.triples) == 5000
    assert report["valid"] and report["question_count"] == 3734
    assert len(store.select("validation", [1, 2, 3, 4], count=30, balanced=True)) == 30


def test_real_tools_multiple_neighbors_and_inverse(graph):
    env = GraphEnvironment(graph, "researcher_000000")
    result = env.get_relations("researcher_000000")
    assert result["ok"] and "works_at" in result["relations"]
    result = env.get_neighbors("researcher_000000", "works_at")
    uni = result["entities"][0]["id"]
    assert result["entities"][0]["name"] == graph.entities[uni]["name"]
    result = env.get_neighbors(uni, "inv:works_at")
    assert len(result["entities"]) > 1
    assert "researcher_000000" in [n["id"] for n in result["entities"]]
    assert env.get_relations("researcher_000000")["ok"]


def test_invalid_entities_relations_arguments_and_discovery(graph):
    env = GraphEnvironment(graph, "researcher_000000")
    assert not env.get_relations("missing")["ok"]
    assert not env.get_neighbors("researcher_000000", "missing")["ok"]
    assert not env.get_neighbors("researcher_000000", "inv:collaborates_with")["ok"]
    assert not env._dispatch("get_relations", {"entity_id": 4})["ok"]
    assert not env._dispatch("get_reward", {})["ok"]
    assert not env.submit_answer("country_000000")["ok"]  # exists, undiscovered
    assert env.submitted_answer is None


def test_call_budget_reset_and_episode_isolation(graph):
    limits = EpisodeLimits(max_tool_calls=2)
    first = GraphEnvironment(graph, "researcher_000000", limits)
    second = GraphEnvironment(graph, "researcher_000000", limits)
    first.get_relations("researcher_000000")
    first.get_relations("missing")
    assert first.done and first.termination_reason == "tool_budget"
    assert first.calls == 2 and second.calls == 0 and not second.done
    assert not first.submit_answer("researcher_000000")["ok"]
    assert first.calls == 2 and first.submitted_answer is None
    first.reset()
    assert not first.done and first.calls == 0 and not first.trajectory


def test_submission_at_last_call_is_recorded_without_correctness(graph):
    env = GraphEnvironment(graph, "researcher_000000", EpisodeLimits(max_tool_calls=1))
    result = env.submit_answer("researcher_000000")
    assert result == {"ok": True, "submitted": "researcher_000000", "episode_complete": True}
    assert env.done and env.termination_reason == "submitted"
    assert "reward" not in result and "correct" not in result


def test_complete_neighbor_lists_are_not_silently_truncated(graph):
    researcher = "researcher_000000"
    uni = graph.get_neighbors(researcher, "works_at")[0]
    env = GraphEnvironment(graph, uni, EpisodeLimits(max_neighbors=1))
    result = env.get_neighbors(uni, "inv:works_at")
    assert not result["ok"] and "truncated" in result["error"]
    assert env._session.discovered == {uni}


def test_binary_rewards_use_actual_single_submissions(graph, questions):
    store = TaskStore(graph, questions)
    q = next(q for q in questions if len(q["answer_ids"]) > 1)
    assert all(store.reward(q["id"], answer) == 1.0 for answer in q["answer_ids"])
    assert store.reward(q["id"], None) == 0.0
    assert store.reward(q["id"], q["start_entity"]) == 0.0
    function = make_reward(store)
    assert function(["claimed correct", "claimed wrong"], [q["id"], q["id"]],
                    [None, q["answer_ids"][0]], [[], []]) == [0.0, 1.0]


def test_intermediate_submission_is_accepted_but_not_rewarded(graph):
    # Graph/verifier regression fixture, not a model trajectory or training result.
    start = "researcher_000000"
    university = graph.get_neighbors(start, "works_at")[0]
    city = graph.get_neighbors(university, "located_in_city")[0]
    record = {"id": "unit_multihop", "start_entity": start,
              "question": "Which city hosts the university where this researcher works?",
              "answer_ids": [city]}
    store = TaskStore(graph, [record])
    early = GraphEnvironment(graph, start)
    early.get_neighbors(start, "works_at")
    assert early.submit_answer(university)["ok"]
    assert store.reward(record["id"], early.submitted_answer) == 0.0
    complete = GraphEnvironment(graph, start)
    complete.get_neighbors(start, "works_at")
    assert "located_in_city" in complete.get_relations(university)["relations"]
    complete.get_neighbors(university, "located_in_city")
    assert complete.submit_answer(city)["ok"]
    assert store.reward(record["id"], complete.submitted_answer) == 1.0


def test_private_verifier_changes_cannot_affect_prompts_or_tools(graph, questions):
    q = copy.deepcopy(questions[0])
    hidden = copy.deepcopy(q)
    hidden.update(answer_ids=["VERIFIER_SECRET_ANSWER"], relation_path=["VERIFIER_SECRET_PROGRAM"],
                  gold_entity_paths=[["VERIFIER_SECRET_PATH"]], difficulty={"private": "VERIFIER_SECRET_DIFFICULTY"},
                  answer_type="VERIFIER_SECRET_TYPE")
    assert public_prompt(graph, q) == public_prompt(graph, hidden)
    original = GraphEnvironment(graph, q["start_entity"])
    changed = GraphEnvironment(graph, hidden["start_entity"])
    left = original.get_relations(q["start_entity"])
    right = changed.get_relations(hidden["start_entity"])
    assert left == right
    relation = graph.get_relations(q["start_entity"])[0]
    assert original.get_neighbors(q["start_entity"], relation) == changed.get_neighbors(q["start_entity"], relation)
    visible = json.dumps([public_prompt(graph, hidden), original.trajectory, TOOL_SCHEMAS])
    assert "VERIFIER_SECRET" not in visible
    assert not any(key in visible for key in ("answer_ids", "gold_entity_paths", "relation_path", "difficulty"))


@pytest.mark.parametrize("text", ["plain answer", '<tool_call>{bad}</tool_call>', '<tool_call>{"name":"x"}</tool_call>',
    '<tool_call>{"name":"x","arguments":{}}</tool_call><tool_call>{"name":"x","arguments":{}}</tool_call>'])
def test_malformed_and_non_tool_outputs_are_rejected(text):
    with pytest.raises(ValueError):
        parse_qwen_action(text)


def test_qwen_tool_format_and_typed_schemas():
    text = '<tool_call>\n{"name":"get_neighbors","arguments":{"entity_id":"r","relation_id":"works_at"}}\n</tool_call><|im_end|>'
    assert parse_qwen_action(text) == ("get_neighbors", {"entity_id": "r", "relation_id": "works_at"})
    assert {s["function"]["name"] for s in TOOL_SCHEMAS} == {"get_relations", "get_neighbors", "submit_answer"}
    assert all(p["type"] == "string" for s in TOOL_SCHEMAS for p in s["function"]["parameters"]["properties"].values())


def test_grpo_groups_must_be_same_question_and_are_not_repeated_twice(graph, questions):
    first, second = public_prompt(graph, questions[0]), public_prompt(graph, questions[2])
    verify_group_prompts([first] * 4, 4)
    with pytest.raises(ValueError):
        verify_group_prompts([first] * 3 + [second], 4)
    with pytest.raises(ValueError):
        verify_group_prompts([first] * 3, 4)


def test_ledger_preserves_actions_and_masks_environment_tokens():
    ledger = TokenLedger([1, 2], EpisodeLimits())
    ledger.append([8, 9, 10], True)
    ledger.append([33, 34], False)
    ledger.append([11], True)
    assert ledger.completion_ids == [8, 9, 10, 33, 34, 11]
    assert ledger.env_mask == [1, 1, 1, 0, 0, 1]
    assert ledger.context_ids == [1, 2, 8, 9, 10, 33, 34, 11]


def test_t4_uses_fp16_even_if_bf16_emulation_is_available(monkeypatch):
    torch = pytest.importorskip("torch")
    from kg_rlvr.training.grpo import native_bf16_supported
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=0: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kwargs: True)
    assert not native_bf16_supported()


def test_adapter_only_directory_cannot_be_used_for_resume(tmp_path):
    from kg_rlvr.training.grpo import check_resume
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="complete trainer checkpoint"):
        check_resume(adapter, {})


def test_resume_checks_rng_and_exact_manifest(tmp_path):
    from kg_rlvr.training.grpo import check_resume
    checkpoint = tmp_path / "full/checkpoint-2"
    checkpoint.mkdir(parents=True)
    for name in ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json", "adapter_model.safetensors"]:
        (checkpoint / name).write_text("unit file-presence fixture; not a real trainer state")
    manifest = {"source_sha256": {"input": "aaa"}, "seed": 41}
    (checkpoint.parent / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="RNG"):
        check_resume(checkpoint, manifest)
    (checkpoint / "rng_state.pth").write_text("unit fixture")
    assert check_resume(checkpoint, manifest) == str(checkpoint)
    with pytest.raises(ValueError, match="Incompatible"):
        check_resume(checkpoint, {**manifest, "seed": 42})


class FixtureCodec:
    """Unit fixture only, never substituted for a model in the notebook."""
    tokenizer = type("Tokenizer", (), {"eos_token_id": 0})()
    def __init__(self, texts):
        self.texts = texts
    def prompt_ids(self, prompt):
        return [100]
    def decode(self, tokens):
        return self.texts[tokens[0]]
    def feedback_ids(self, name, observation):
        return [200, 201]


def test_real_episode_execution_completion_feedback_and_independent_state(graph, questions):
    q = questions[0]
    store = TaskStore(graph, [q])
    witness = q["gold_entity_paths"][0]
    actions = []
    for head, relation in zip(witness, q["relation_path"]):
        actions.append(("get_neighbors", {"entity_id": head, "relation_id": relation}))
    actions.append(("submit_answer", {"entity_id": witness[-1]}))
    texts = {i + 1: '<tool_call>' + json.dumps({"name": name, "arguments": args}) + '</tool_call>' for i, (name, args) in enumerate(actions)}
    def rollout_once():
        sequence = iter(range(1, len(actions) + 1))
        contexts = []
        progress_events = []
        def sample(ids, budget):
            contexts.append(ids[:])
            return [next(sequence), 0]
        ledger, trace = run_episode(store, q["id"], FixtureCodec(texts), sample, EpisodeLimits(),
                                    on_turn=lambda **event: progress_events.append(event))
        return ledger, trace, contexts, progress_events
    ledger, trace, contexts, events = rollout_once()
    _, second, _, _ = rollout_once()
    assert trace["reward"] == second["reward"] == 1.0
    assert trace["submitted_answer"] == witness[-1] and trace["termination_reason"] == "submitted"
    assert len(contexts) == len(actions)  # no generation after terminal submission
    assert contexts[1] == [100, 1, 0, 200, 201]
    assert ledger.env_mask[:4] == [1, 1, 0, 0]
    assert summarize_episodes([trace])["accuracy"] == 1.0
    assert [e["turn"] for e in events if e["status"] == "generating"] == list(range(1, len(actions) + 1))
    assert events[-1] == {"task": q["id"], "turn": len(actions), "calls": len(actions),
                          "tokens": trace["model_tokens"], "status": "submitted"}
    assert not any("answer_ids" in event or "gold_entity_paths" in event for event in events)


@pytest.mark.parametrize("tokens,reason", [([1], "completion_limit"), ([1, 0], "no_tool_call")])
def test_truncated_and_premature_episodes_never_invent_submission(graph, questions, tokens, reason):
    store = TaskStore(graph, [questions[0]])
    _, trace = run_episode(store, questions[0]["id"], FixtureCodec({1: "no action"}), lambda ids, budget: tokens, EpisodeLimits())
    assert trace["submitted_answer"] is None and trace["reward"] == 0.0
    assert trace["termination_reason"] == reason
