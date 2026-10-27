import json
import zipfile

import pytest

from kg_rlvr.training.artifacts import evaluate_variant, save_json, zip_run_artifacts


def test_failed_evaluation_keeps_earlier_variant_and_partial_trajectories(tmp_path, monkeypatch):
    from kg_rlvr.training import rollouts

    trace = {"task_id": "unit", "question": "<unit>", "prompt": [], "turns": [], "reward": 0}

    def evaluate(model, tokenizer, store, ids, limits, directory, label, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "test_trajectories.jsonl").write_text(json.dumps(trace) + "\n")
        if directory.parent.name == "sft_grpo":
            raise RuntimeError("unit interruption")
        save_json(directory / "test_metrics.json", {"accuracy": 1})
        return {"accuracy": 1}, [trace]

    monkeypatch.setattr(rollouts, "evaluate_policy", evaluate)
    evaluate_variant(None, None, None, ["unit"], None, tmp_path, "vanilla")
    original = (tmp_path / "variants/vanilla/evaluation/test_metrics.json").read_bytes()
    with pytest.raises(RuntimeError):
        evaluate_variant(None, None, None, ["unit"], None, tmp_path, "sft_grpo")
    assert (tmp_path / "variants/vanilla/evaluation/test_metrics.json").read_bytes() == original
    assert (tmp_path / "variants/sft_grpo/evaluation/partial_trajectories.html").exists()
    status = json.loads((tmp_path / "variants/sft_grpo/status.json").read_text())
    assert not status["evaluation_complete"]


def test_zip_excludes_optimizer_checkpoints(tmp_path):
    save_json(tmp_path / "run/variants/sft/summary.json", {"accuracy": 1})
    save_json(tmp_path / "run/checkpoint-1/trainer_state.json", {})
    output = zip_run_artifacts(tmp_path / "run", tmp_path / "results.zip")
    with zipfile.ZipFile(output) as archive:
        assert any(n.endswith("summary.json") for n in archive.namelist())
        assert not any("checkpoint-1" in n for n in archive.namelist())
