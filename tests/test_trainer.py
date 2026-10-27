"""Optional real TRL checks on a tiny random CPU architecture, not model performance."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("all_correct", [False, True])
def test_standard_rewards_reach_grpo_and_zero_variance_does_not_abort(
    tasks, tmp_path, monkeypatch, all_correct
):
    trl = pytest.importorskip("trl")
    if trl.__version__ != "1.8.0":
        pytest.skip("Use the tutorial's pinned training stack")
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    from kg_rlvr.training import grpo
    from kg_rlvr.training.environment import EpisodeLimits

    torch.manual_seed(41)
    vocab = {"[PAD]": 0, "<|im_end|>": 1, "[UNK]": 2, **{f"w{i}": i for i in range(3, 32)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
    )
    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model = get_peft_model(
        base, LoraConfig(r=2, lora_alpha=4, target_modules=grpo.LORA_TARGETS, task_type="CAUSAL_LM")
    )
    _, store = tasks

    def scripted_rollout(prompts, trainer):
        answers = [
            store.record(store.resolve_prompt(p))["answer_ids"][0]
            if all_correct or i % 2 == 0
            else None
            for i, p in enumerate(prompts)
        ]
        return {
            "prompt_ids": [[3, 4] for _ in prompts],
            "completion_ids": [[5 + i % 2, 7, 1] for i, p in enumerate(prompts)],
            "env_mask": [[1, 0, 1] for _ in prompts],
            "logprobs": None,
            "submitted_answers": answers,
            "tool_events": [[] for _ in prompts],
        }

    monkeypatch.setattr(grpo, "GRPORollout", lambda *a, **kw: scripted_rollout)
    args = trl.GRPOConfig(
        output_dir=str(tmp_path / "checkpoints"),
        max_steps=2,
        num_generations=2,
        gradient_accumulation_steps=2,
        generation_batch_size=2,
        per_device_train_batch_size=1,
        use_cpu=True,
        bf16=False,
        fp16=False,
        beta=0.0,
        reward_weights=[1.0, 0.2],
        learning_rate=1e-3,
        save_strategy="no",
        logging_steps=1,
        remove_unused_columns=False,
        shuffle_dataset=False,
        gradient_checkpointing=False,
        report_to=[],
        disable_tqdm=True,
    )
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    if not all_correct:
        base.is_loaded_in_4bit = True  # metadata only, not a claim of CPU NF4 execution
    monkeypatch.setattr(grpo, "make_grpo_args", lambda *a, **kw: args)
    trainer, _ = grpo.build_trainer(
        model,
        tokenizer,
        store,
        store.select("train", [2], count=1),
        EpisodeLimits(),
        tmp_path / "checkpoints",
        "full",
        2,
        generations=2,
        accumulation=2,
        hop_weights={2: 1},
        eval_every=0,
        show_progress=False,
    )
    assert all(
        p.dtype == torch.float32 and torch.equal(p, initial[n])
        for n, p in model.named_parameters()
        if n in initial
    )
    base.is_loaded_in_4bit = False
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    result = trainer.train()
    assert trainer.state.global_step == 2
    assert torch.isfinite(torch.tensor(result.training_loss))
    gradients = [r["grad_norm"] for r in trainer.state.log_history if "grad_norm" in r]
    assert all(
        r["completions/mean_length"] == 2
        for r in trainer.state.log_history
        if "completions/mean_length" in r
    )
    changed = any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before)
    assert any(g > 0 for g in gradients) == (not all_correct)
    assert changed == (not all_correct)


def test_real_qwen_codec_keeps_tool_observation_tokens_out_of_loss(tasks):
    trl = pytest.importorskip("trl")
    if trl.__version__ != "1.8.0":
        pytest.skip("Use the tutorial's pinned stack")
    import os

    directory = os.environ.get("KG_GRPO_TOKENIZER")
    if not directory:
        pytest.skip("Set KG_GRPO_TOKENIZER to an offline Qwen tokenizer")
    import json
    from transformers import AutoTokenizer
    from kg_rlvr.training.rollouts import QwenChatCodec, run_episode
    from kg_rlvr.training.environment import EpisodeLimits

    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    _, store = tasks
    qid = store.select("train", [2], count=1)[0]
    q = store.record(qid)
    path = q["gold_entity_paths"][0]
    actions = [
        ("get_neighbors", {"entity_id": h, "relation_id": r})
        for h, r in zip(path, q["relation_path"])
    ]
    actions += [("submit_answer", {"entity_id": path[-1]})]
    sampled = [
        tokenizer.encode(
            "<tool_call>\n" + json.dumps({"name": n, "arguments": a}) + "\n</tool_call><|im_end|>",
            add_special_tokens=False,
        )
        for n, a in actions
    ]
    script = iter(sampled)
    ledger, trace = run_episode(
        store, qid, QwenChatCodec(tokenizer), lambda context, limit: next(script), EpisodeLimits()
    )
    assert trace["answer_correct"] == 1
    assert [token for token, mask in zip(ledger.completion_ids, ledger.env_mask) if mask] == [
        t for turn in sampled for t in turn
    ]
    assert 0 in ledger.env_mask


def test_sft_updates_adapter_with_assistant_only_labels(tmp_path):
    pytest.importorskip("trl")
    import torch
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    from kg_rlvr.training.sft import build_sft_trainer

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel(
                {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "q": 3, "a": 4}, unk_token="[UNK]"
            )
        ),
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    )
    model = get_peft_model(
        base,
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    row = {"input_ids": [3, 3, 4, 1], "attention_mask": [1] * 4, "labels": [-100, -100, 4, 1]}
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    trainer = build_sft_trainer(
        model,
        tokenizer,
        [row, row],
        tmp_path / "sft",
        accumulation=2,
        cpu=True,
        show_progress=False,
    )
    result = trainer.train()
    assert trainer.state.global_step == 1 and torch.isfinite(torch.tensor(result.training_loss))
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before)
