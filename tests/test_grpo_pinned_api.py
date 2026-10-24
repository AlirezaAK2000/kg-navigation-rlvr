"""Optional CPU API checks against the pinned stack and a real Qwen tokenizer.

No pretrained model is loaded and these are not training/evaluation results.
Set KG_GRPO_TOKENIZER to a locally downloaded Qwen tokenizer directory.
"""
import inspect
import os
from pathlib import Path
import pytest


def pinned():
    trl = pytest.importorskip("trl")
    if trl.__version__ != "1.8.0":
        pytest.skip("Pinned integration test requires TRL 1.8.0")
    return trl


def test_real_installed_trainer_and_configuration_api(tmp_path):
    trl = pinned()
    from kg_rlvr.training.environment import EpisodeLimits
    from kg_rlvr.training.grpo import assert_training_stack, make_grpo_args
    assert_training_stack(require_cuda=False)
    config = make_grpo_args(tmp_path, 2, EpisodeLimits(), cpu=True, save_steps=2)
    assert isinstance(config, trl.GRPOConfig)
    assert config.generation_batch_size == config.num_generations == 4
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 4
    assert "rollout_func" in inspect.signature(trl.GRPOTrainer).parameters
    assert "tool_mask" in inspect.getsource(trl.GRPOTrainer._compute_loss)
    from types import SimpleNamespace
    from itertools import islice
    probe = SimpleNamespace(args=config, train_dataset=list(range(8)), num_generations=4,
                            num_iterations=1, shuffle_dataset=True)
    sampler = trl.GRPOTrainer._get_train_sampler(probe)
    assert len(set(islice(iter(sampler), 4))) == 1  # release actually repeats rows before rollout_func


def test_real_qwen_chat_template_boundary_and_token_mask(graph, questions):
    pinned()
    directory = os.environ.get("KG_GRPO_TOKENIZER")
    if not directory or not Path(directory).exists():
        pytest.skip("Set KG_GRPO_TOKENIZER to the downloaded tokenizer for this offline test")
    from transformers import AutoTokenizer
    from kg_rlvr.training.environment import TaskStore, EpisodeLimits, parse_qwen_action
    from kg_rlvr.training.rollouts import QwenChatCodec, run_episode
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    codec = QwenChatCodec(tokenizer)
    q = questions[0]
    store = TaskStore(graph, [q])
    path = q["gold_entity_paths"][0]
    names_args = [("get_neighbors", {"entity_id": head, "relation_id": relation}) for head, relation in zip(path, q["relation_path"])]
    names_args.append(("submit_answer", {"entity_id": path[-1]}))
    import json
    actions = ['<tool_call>\n' + json.dumps({"name": name, "arguments": args}) + '\n</tool_call><|im_end|>' for name, args in names_args]
    action_ids = [tokenizer.encode(text, add_special_tokens=False) for text in actions]
    script = iter(action_ids)
    ledger, trace = run_episode(store, q["id"], codec, lambda context, limit: next(script), EpisodeLimits())
    assert trace["reward"] == 1.0
    assert len(ledger.prompt_ids) <= EpisodeLimits().max_prompt_tokens
    assert any(value == 0 for value in ledger.env_mask)
    reconstructed_actions = [ledger.completion_ids[s["begin"]:s["end"]] for s in ledger.segments if s["source"] == "model"]
    assert reconstructed_actions == action_ids
    for segment in ledger.segments:
        expected = 1 if segment["source"] == "model" else 0
        assert set(ledger.env_mask[segment["begin"]:segment["end"]]) == {expected}
    feedback = codec.decode(codec.feedback_ids("get_neighbors", {"ok": True, "entities": [{"id": "x", "name": "Fictional X"}]}))
    assert "<tool_response>" in feedback and '"id":"x"' in feedback
    assert feedback.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # Exercise the real TRL custom-rollout boundary with a fixture payload. This
    # proves env_mask reaches its tool mask; it is not a model or optimizer test.
    import torch
    from collections import defaultdict
    from types import SimpleNamespace
    from trl import GRPOTrainer
    payload = {"prompt_ids": [ledger.prompt_ids] * 4, "completion_ids": [ledger.completion_ids] * 4,
               "env_mask": [ledger.env_mask] * 4, "logprobs": None}
    probe = SimpleNamespace(
        accelerator=SimpleNamespace(device=torch.device("cpu"), gather=lambda value: value),
        model=SimpleNamespace(training=True), rollout_func=lambda prompts, trainer: payload,
        use_vllm=False, tools=[], _tokenizer=tokenizer, processing_class=tokenizer,
        state=SimpleNamespace(num_input_tokens_seen=0), _metrics={"train": defaultdict(list)},
    )
    output = GRPOTrainer._generate(probe, [store.prompt(q["id"])] * 4)
    assert output[2] == [ledger.env_mask] * 4


def test_real_peft_adapter_save_reload_on_tiny_cpu_fixture(tmp_path):
    pinned()
    directory = os.environ.get("KG_GRPO_TOKENIZER")
    if not directory:
        pytest.skip("Tokenizer fixture directory not provided")
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    from kg_rlvr.training.grpo import LORA_TARGETS, export_adapter, reload_adapter_in_place
    # Random, very small architecture fixture. No model rollouts, training or
    # optimization occurs, and this does not validate a GPU/quantized checkpoint.
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    before = {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" in name}
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    output = export_adapter(model, tokenizer, tmp_path / "adapter", {"fixture": "tiny random CPU architecture; not trained"})
    assert not (output / "model.safetensors").exists()
    reloaded = reload_adapter_in_place(model, output)
    after = {name: p.detach() for name, p in reloaded.named_parameters() if "lora_" in name}
    assert set(after) == set(before)
    assert all(torch.equal(before[name], after[name]) for name in before)
