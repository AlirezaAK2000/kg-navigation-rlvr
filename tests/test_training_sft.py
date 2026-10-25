"""SFT partition, executable demonstrations, masks and tiny CPU optimizer checks."""

from collections import Counter
import json
import os
from pathlib import Path

import pytest

from kg_rlvr.training.environment import EpisodeLimits, load_tasks
from kg_rlvr.training.sampling import (
    DEFAULT_HOP_EXPOSURE, assert_disjoint_tasks, partition_training_tasks, task_instance_key,
)
from kg_rlvr.training.sft import AssistantOnlyCollator, build_demonstration, encode_demonstration


@pytest.fixture(scope="module")
def sft_store():
    return load_tasks("data", single_answer_only=True)[1]


@pytest.fixture(scope="module")
def partition(sft_store):
    limits = EpisodeLimits(max_tool_calls=12)
    return partition_training_tasks(sft_store, [1, 2, 3], hop_weights=DEFAULT_HOP_EXPOSURE,
        sft_candidate_check=lambda q: build_demonstration(sft_store, q, limits))


def test_default_partition_is_deterministic_disjoint_and_weighted(sft_store, partition):
    sft_ids, grpo_ids, report = partition
    assert len(sft_ids) == 200 and len(grpo_ids) == 733
    assert Counter(sft_store.record(q)["hop_count"] for q in sft_ids) == {1: 40, 2: 80, 3: 80}
    assert len({task_instance_key(sft_store.record(q)) for q in sft_ids}) == 200
    assert all(sft_store.record(q)["split"] == "train" for q in sft_ids + grpo_ids)
    test = sft_store.select("test", [1, 2, 3, 4])
    assert assert_disjoint_tasks(sft_store, {"sft": sft_ids, "grpo": grpo_ids,
        "test": test})["disjoint"]
    smaller_sft, smaller_grpo, _ = partition_training_tasks(sft_store, [1, 2, 3],
        hop_weights=DEFAULT_HOP_EXPOSURE, max_grpo_examples=100)
    assert smaller_sft == sft_ids and smaller_grpo == grpo_ids[:100]
    assert report["source_train_unique_instances"] == 933
    assert report["sft_candidate_rejections"] == {}


def test_partition_rejects_paraphrase_overlap_not_just_matching_ids(sft_store):
    rows = sft_store.select("train", [1, 2, 3], unique_instances=False)
    seen = {}
    for qid in rows:
        key = task_instance_key(sft_store.record(qid))
        if key in seen:
            with pytest.raises(ValueError, match="query instances"):
                assert_disjoint_tasks(sft_store, {"sft": [seen[key]], "grpo": [qid]})
            return
        seen[key] = qid
    pytest.fail("Expected source paraphrase instances")


def test_partition_rejects_impossible_quotas_and_can_disable_reservation(sft_store):
    with pytest.raises(ValueError, match="Cannot reserve"):
        partition_training_tasks(sft_store, [1, 2, 3], sft_count=5000)
    with pytest.raises(ValueError, match="SFT hop weights"):
        partition_training_tasks(sft_store, [1, 2, 3], hop_weights={1: 1})
    sft, grpo, _ = partition_training_tasks(sft_store, [1, 2, 3], sft_count=0)
    assert not sft and len(grpo) == 933


def test_reference_demonstrations_execute_all_hops_before_submission(sft_store, partition):
    for qid in partition[0]:
        demo = build_demonstration(sft_store, qid, EpisodeLimits(max_tool_calls=12))
        assert demo["kind"] == "verified_reference_demonstration"
        assert demo["tool_calls"] == 2 * demo["hop_count"] + 1
        assert demo["answer_correct"] == 1 and demo["termination_reason"] == "submitted"
        assert demo["turns"][-1]["tool_call"]["name"] == "submit_answer"
        assert demo["submitted_answer"] in sft_store.record(qid)["answer_ids"]
        for turn in demo["turns"]:
            assert turn["observation"]["ok"]
            assert "answer_correct" not in turn["observation"] and "reward" not in turn["observation"]


def test_reference_builder_refuses_heldout_data_and_runtime_budget_failures(sft_store, partition):
    for split in ("test",):
        qid = sft_store.select(split, [1], count=1)[0]
        with pytest.raises(ValueError, match="source training split"):
            build_demonstration(sft_store, qid, EpisodeLimits())
    with pytest.raises(ValueError, match="tool-call budget"):
        build_demonstration(sft_store, partition[0][0], EpisodeLimits(max_tool_calls=2))
    with pytest.raises(ValueError, match="SFT reference tool failure"):
        build_demonstration(sft_store, partition[0][0], EpisodeLimits(max_observation_chars=1))


def test_collator_masks_padding_without_changing_supervised_tokens():
    torch = pytest.importorskip("torch")
    rows = [{"input_ids": [4, 5, 6], "attention_mask": [1] * 3, "labels": [-100, 5, 6]},
            {"input_ids": [7], "attention_mask": [1], "labels": [7]}]
    batch = AssistantOnlyCollator(0)(rows)
    assert batch["input_ids"].tolist() == [[4, 5, 6], [7, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 5, 6], [7, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].dtype == torch.long


def require_pinned():
    trl = pytest.importorskip("trl")
    if trl.__version__ != "1.8.0":
        pytest.skip("Requires pinned training environment")


def test_real_qwen_demo_masks_and_budgets_for_all_reserved_examples(sft_store, partition):
    require_pinned()
    directory = os.environ.get("KG_GRPO_TOKENIZER")
    if not directory:
        pytest.skip("Set KG_GRPO_TOKENIZER for offline Qwen tokenization")
    from transformers import AutoTokenizer
    from kg_rlvr.training.rollouts import QwenChatCodec
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    codec = QwenChatCodec(tokenizer)
    limits = EpisodeLimits(max_tool_calls=12)
    padding_side = tokenizer.padding_side
    for qid in partition[0]:
        demo = build_demonstration(sft_store, qid, limits)
        row = encode_demonstration(sft_store, demo, tokenizer, limits)
        prompt_length = len(codec.prompt_ids(sft_store.prompt(qid)))
        assert row["labels"][:prompt_length] == [-100] * prompt_length
        expected = [token for turn in demo["turns"]
                    for token in tokenizer.encode(turn["raw_assistant"], add_special_tokens=False)]
        assert [token for token in row["labels"] if token != -100] == expected
        assert len(row["input_ids"]) <= limits.max_context_tokens
        assert len(row["input_ids"]) - prompt_length <= limits.max_completion_tokens
    assert tokenizer.padding_side == padding_side


def test_grpo_constructor_rejects_reserved_sft_instances_before_model_use(sft_store, partition, tmp_path):
    require_pinned()
    from kg_rlvr.training.grpo import build_trainer
    with pytest.raises(ValueError, match="Training partition overlap"):
        build_trainer(None, None, sft_store, [partition[0][0]], EpisodeLimits(), tmp_path,
                      "overlap_fixture", 1, excluded_sft_task_ids=partition[0])


def test_real_supervised_trainer_changes_lora_and_saves_resumable_checkpoint(tmp_path, graph, questions, monkeypatch):
    require_pinned()
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    from kg_rlvr.training.grpo import LORA_TARGETS, lora_snapshot, verify_optimizer_update
    from kg_rlvr.training.sft import build_sft_trainer
    # A tiny random CPU architecture and artificial token rows: optimizer/mask
    # evidence only. These are not pretrained-policy or CUDA performance results.
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
             **{f"w{i}": i for i in range(4, 64)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
                                         eos_token="[EOS]", unk_token="[UNK]")
    torch.manual_seed(41)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False))
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=LORA_TARGETS,
                                           task_type="CAUSAL_LM"))
    rows = [{"input_ids": [1, 4, 5, 6, 7, 8, 9, 2], "attention_mask": [1] * 8,
             "labels": [-100, -100, -100, 6, 7, -100, 9, 2]} for _ in range(4)]
    before = lora_snapshot(model)
    trainer = build_sft_trainer(model, tokenizer, rows, tmp_path / "sft", accumulation=2,
                                learning_rate=1e-3, show_progress=False, cpu=True)
    trainer.train()
    verified = verify_optimizer_update(model, before, trainer, tmp_path / "sft", required_steps=2,
                                       training_kind="SFT")
    assert verified["optimizer_steps"] == 2 and verified["changed_lora_tensors"] > 0
    assert verified["max_lora_weight_delta"] > 0 and verified["finite_losses"]
    directory = os.environ.get("KG_GRPO_TOKENIZER")
    if directory:
        # Constructor-only handoff: the tiny architecture is not used to generate
        # native Qwen tokens. Real tokenizer/GRPO initialization must preserve the
        # actual supervised weight changes before any RL optimizer is created.
        from transformers import AutoTokenizer
        from kg_rlvr.training.environment import TaskStore
        from kg_rlvr.training import grpo
        after_sft = lora_snapshot(model)
        trainer.optimizer = None; trainer.lr_scheduler = None
        trainer.accelerator.free_memory()
        model.zero_grad(set_to_none=True)
        make_args = grpo.make_grpo_args
        monkeypatch.setattr(grpo, "make_grpo_args", lambda *a, **kw: make_args(*a, cpu=True, **kw))
        native_tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        store = TaskStore(graph, questions)
        rl_trainer, _ = grpo.build_trainer(model, native_tokenizer, store, store.select("train", [2, 3]),
            EpisodeLimits(), tmp_path / "grpo", "handoff_fixture", 1, generations=8, accumulation=8,
            show_progress=False, hop_weights={2: 0.5, 3: 0.5})
        assert rl_trainer.optimizer is None
        assert all(torch.equal(value, rl_trainer.model.get_parameter(name)) for name, value in after_sft.items())


def test_notebook_orders_baseline_sft_evaluation_and_disjoint_grpo():
    notebook = json.loads(Path("notebooks/02_grpo_training_kaggle.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    order = [c["id"] for c in notebook["cells"]]
    assert order.index("grpo-013") < order.index("sft-training") < order.index("sft-evaluation") < order.index("grpo-017")
    assert "RUN_SFT" not in cells["grpo-002"] and "SFT_EXAMPLES = 200" in cells["grpo-002"]
    assert "partition_training_tasks(" in cells["grpo-007"]
    assert "sft_ids + train_ids" in cells["grpo-011"]
    for cell in ("grpo-017", "grpo-019"):
        assert "**RL_KWARGS" in cells[cell]
        assert '"excluded_sft_task_ids": sft_ids' in cells["grpo-015"]
        assert "if SFT_RESULT is None:" in cells[cell]
