from collections import Counter
import pytest
from kg_rlvr.training.environment import EpisodeLimits
from kg_rlvr.training.sampling import partition_training_tasks, check_stage_separation
from kg_rlvr.training.sft import build_demonstration


def test_saved_sft_and_grpo_partitions_remain_separate(tasks):
    _, store = tasks
    limits = EpisodeLimits(max_tool_calls=12)
    sft, grpo, report = partition_training_tasks(
        store,
        [1, 2, 3],
        sft_count=100,
        hop_weights={1: 0.2, 2: 0.4, 3: 0.4},
        max_grpo_examples=1000,
        sft_candidate_check=lambda q: build_demonstration(store, q, limits),
    )
    assert len(sft) == 100 and len(grpo) == 833
    assert Counter(store.record(q)["hop_count"] for q in sft) == {1: 20, 2: 40, 3: 40}
    test = store.select("test", [1, 2, 3, 4], 30, 42, balanced=True)
    assert check_stage_separation(store, {"sft": sft, "grpo": grpo, "test": test})["disjoint"]
    with pytest.raises(ValueError, match="overlap"):
        check_stage_separation(store, {"sft": sft, "grpo": [sft[0]]})


def test_reference_demonstration_uses_real_tools_and_submits(tasks):
    _, store = tasks
    qid = store.select("train", [3], 1)[0]
    demo = build_demonstration(store, qid, EpisodeLimits(max_tool_calls=12))
    assert demo["reward"] == 1 and demo["tool_calls"] == 7
    assert all(event["observation"]["ok"] for event in demo["tool_events"])
    assert demo["turns"][-1]["tool_call"]["name"] == "submit_answer"
