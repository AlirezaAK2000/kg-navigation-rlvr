"""Persisted-log unit fixtures, never model performance or training evidence."""

import json

from kg_rlvr.training.grpo import smoke_diagnostics


def test_missing_smoke_logs_before_first_rollout(tmp_path):
    report = smoke_diagnostics(tmp_path)
    assert report["episodes"] == report["groups"] == report["mixed_reward_groups"] == 0
    assert report["trainer_metrics"] == report["failed_examples"] == []


def test_saved_diagnostics_distinguish_homogeneous_and_mixed_groups(tmp_path):
    # Replay the reported two all-zero groups and one mixed synthetic group.
    traces = []
    for group, rewards in [("zero1", [0, 0, 0, 0]), ("zero2", [0, 0, 0, 0]),
                           ("mixed", [0, 1, 0, 0]), ("all_success", [1, 1, 1, 1])]:
        for reward in rewards:
            traces.append({"group_id": group, "reward": reward, "termination_reason": "submitted",
                           "question": "unit fixture", "start_entity": "r", "submitted_answer": "u",
                           "gold_answer_ids": ["c"], "turns": [{"raw_assistant": "unit fixture"}]})
    (tmp_path / "trajectories.jsonl").write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    (tmp_path / "trainer_metrics.jsonl").write_text('{"step": 1, "loss": 0, "grad_norm": 0}\n')
    report = smoke_diagnostics(tmp_path)
    assert report["episodes"] == 16 and report["groups"] == 4
    assert report["mixed_reward_groups"] == 1
    assert report["group_rewards"]["zero1"] == [0, 0, 0, 0]
    assert report["termination_reasons"] == {"submitted": 16}
    assert report["trainer_metrics"][0]["grad_norm"] == 0
    assert len(report["failed_examples"]) == 2
    assert report["failed_examples"][0]["turns"] == traces[0]["turns"]
    # Entire report remains serializable without any live trainer/model objects.
    assert json.loads(json.dumps(report)) == report
