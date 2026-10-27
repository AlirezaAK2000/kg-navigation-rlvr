from kg_rlvr.training.rewards import correctness_reward, retrieval_reward, score_episode


def event(name="get_neighbors", ok=True):
    return {"name": name, "observation": {"ok": ok, "entities": [{"id": "gold"}]}}


def test_standard_trl_rewards_use_executed_outcomes_and_private_dataset_columns():
    completions = ["I know gold", "wrong prose", "submit gold"]
    gold = ["gold"] * 3
    answers = [None, "gold", "wrong"]
    events = [[], [event()], [event(), event()]]
    correct = correctness_reward(completions, gold, answers)
    retrieved = retrieval_reward(completions, gold, events)
    assert correct == [0.0, 1.0, 0.0]
    assert score_episode("gol", "gold", [])["answer_correct"] == 0
    assert score_episode(None, "gold_extra", [event()])["retrieval_reward"] == 0
    assert retrieved == [0.0, 1.0, 1.0]
    for i in range(3):
        assert (
            score_episode(answers[i], gold[i], events[i])["training_reward"]
            == correct[i] + 0.2 * retrieved[i]
        )


def test_failed_calls_relation_lists_and_model_claims_do_not_earn_retrieval_reward():
    assert retrieval_reward(
        ["gold"] * 3, ["gold"] * 3, [[event(ok=False)], [event(name="get_relations")], []]
    ) == [0.0, 0.0, 0.0]


def test_identical_rewards_are_valid_scores():
    assert correctness_reward(["x"] * 8, ["gold"] * 8, ["gold"] * 8) == [1.0] * 8
    assert correctness_reward(["x"] * 8, ["gold"] * 8, [None] * 8) == [0.0] * 8
