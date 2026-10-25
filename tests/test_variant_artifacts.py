"""Per-policy durability, failure isolation and independent initial adapters."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from kg_rlvr.training.artifacts import (
    evaluate_variant, persist_rl_diagnostics, read_rows, save_json, save_rows, zip_run_artifacts,
)


def trace(qid="unit", reward=1):
    return {"task_id": qid, "question": "<unit question>", "prompt": [{"role": "user", "content": "unit input"}],
        "phase": "smoke", "step": 1, "group_id": "unit:1", "hop_count": 1,
        "reward": reward, "training_reward": reward + .2, "retrieval_reward": .2,
        "turns": [{"raw_assistant": "<tool_call>unit output</tool_call>", "model_tokens": 5,
                   "observation": {"ok": True, "entities": [{"id": "unit entity"}]}}]}


def test_phase_returns_and_pretty_traces_are_saved_separately_before_zip(tmp_path):
    run = tmp_path / "run"
    for variant, reward in [("sft_grpo", 1), ("grpo_only", 0)]:
        checkpoint = tmp_path / "checkpoints" / variant / "smoke"
        episode = trace(reward=reward)
        save_rows(checkpoint / "trajectories.jsonl", [episode])
        save_rows(checkpoint / "rollout_metrics.jsonl", [{"phase": "smoke", "step": 1,
            "mean_episode_reward": episode["training_reward"]}])
        save_json(checkpoint / "checkpoint-1/trainer_state.json", {"fixture": "large checkpoint excluded"})
        persist_rl_diagnostics(run / "variants" / variant, [checkpoint])
    for variant, expected in [("sft_grpo", 1.2), ("grpo_only", .2)]:
        root = run / "variants" / variant
        assert read_rows(root / "episodic_returns.jsonl")[0]["episodic_return"] == expected
        assert (root / "figures/rollout_metrics.png").exists()
        html = (root / "training_trajectories.html").read_text()
        assert "unit input" in html and "unit output" in html and "unit entity" in html
        assert "&lt;unit question&gt;" in html and "&lt;tool_call&gt;" in html
    output = zip_run_artifacts(run, tmp_path / "results.zip")
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert any("sft_grpo/episodic_returns.jsonl" in n for n in names)
        assert any("grpo_only/training_trajectories.html" in n for n in names)
        assert not any("checkpoint-" in n for n in names)
    with pytest.raises(ValueError, match="outside"):
        zip_run_artifacts(run, run / "recursive.zip")


def test_failed_later_evaluation_keeps_prior_policy_files_and_partial_traces(tmp_path, monkeypatch):
    from kg_rlvr.training import rollouts
    def evaluate(model, tokenizer, store, ids, limits, directory, name, **kwargs):
        save_rows(directory / "test_trajectories.jsonl", [trace(ids[0])])
        if directory.parent.name == "grpo_only":
            raise RuntimeError("unit interrupted evaluation")
        metrics = {"accuracy": 1., "accuracy_by_hop": {"1": 1.}}
        save_json(directory / "test_metrics.json", metrics)
        return metrics, [trace(ids[0])]
    monkeypatch.setattr(rollouts, "evaluate_policy", evaluate)
    evaluate_variant(None, None, None, ["unit"], None, tmp_path, "vanilla")
    prior = (tmp_path / "variants/vanilla/summary.json").read_bytes()
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_variant(None, None, None, ["unit", "uncompleted"], None, tmp_path, "grpo_only")
    assert (tmp_path / "variants/vanilla/summary.json").read_bytes() == prior
    assert (tmp_path / "variants/grpo_only/evaluation/partial_trajectories.html").exists()
    rows = json.loads((tmp_path / "comparison.json").read_text())
    assert next(r for r in rows if r["metric"] == "accuracy")["GRPO only"] is None
    output = zip_run_artifacts(tmp_path / "variants", tmp_path / "partial.zip")
    assert output.exists()


def test_rl_phase_failure_persists_logs_before_returning_control(tmp_path, monkeypatch):
    from kg_rlvr.training import experiments
    def build(*args, **kwargs):
        # Fake optimizer failure after a completed rollout; no performance claim.
        def train(**kw):
            save_rows(kwargs["diagnostics_dir"] / "trajectories.jsonl", [trace(reward=0)])
            raise RuntimeError("unit optimizer failure")
        return SimpleNamespace(train=train), None
    monkeypatch.setattr(experiments, "build_trainer", build)
    monkeypatch.setattr(experiments, "lora_snapshot", lambda model: {})
    monkeypatch.setattr(experiments, "memory_snapshot", lambda stage: {"stage": stage})
    variant = tmp_path / "variants/grpo_only"
    with pytest.raises(RuntimeError, match="optimizer failure"):
        experiments.run_rl_phase(None, None, None, ["unit"], None, variant_dir=variant,
            checkpoint_root=tmp_path / "checkpoints/grpo_only", phase="smoke", steps=1, manifest={},
            generations=8, accumulation=8, learning_rate=5e-6, save_steps=1, seed=41, test_ids=[],
            eval_every=0, show_progress=False, hop_weights={1: 1}, excluded_sft_task_ids=[])
    assert json.loads((variant / "status.json").read_text())["status"] == "training_failed"
    assert (variant / "training/smoke/failure.json").exists()
    assert read_rows(variant / "episodic_returns.jsonl")[0]["episodic_return"] == .2
    assert (variant / "training_trajectories.html").exists()


def test_real_peft_reset_removes_prior_training_and_restores_trainable_initial_adapter(tmp_path):
    pytest.importorskip("torch")
    trl = pytest.importorskip("trl")
    if trl.__version__ != "1.8.0":
        pytest.skip("Requires pinned PEFT/Transformers stack")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from kg_rlvr.training.grpo import LORA_TARGETS, lora_fingerprint, reload_adapter_in_place, assert_lora_fp32
    torch.manual_seed(41)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    initial = lora_fingerprint(model)
    frozen_base = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
    model.save_pretrained(tmp_path / "initial", safe_serialization=True, save_embedding_layers=False)
    # Stand in for learned weights to verify exact reset; no training result.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.add_(.125)
    assert lora_fingerprint(model) != initial
    model = reload_adapter_in_place(model, tmp_path / "initial", is_trainable=True)
    assert lora_fingerprint(model) == initial
    assert assert_lora_fp32(model)["trainable_parameters"] > 0
    assert all(torch.equal(p, frozen_base[n]) for n, p in model.named_parameters() if "lora_" not in n)


def test_zip_is_the_last_notebook_cell_and_does_not_run_or_save_models():
    nb = json.loads(Path("notebooks/02_grpo_training_kaggle.ipynb").read_text())
    last = nb["cells"][-1]
    assert last["id"] == "zip-artifacts"
    source = "".join(last["source"])
    assert "zip_run_artifacts(" in source
    assert all(name not in source for name in ("evaluate_policy(", "train(", "export_adapter(", "RUN_MODEL"))
