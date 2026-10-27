"""Tool-use demonstrations and assistant-only causal-LM fine-tuning.

Reference actions are explicitly supervised demonstrations, never represented
as model-generated episodes. Observations are always executed by the real graph
environment. Initial prompts and all environment tokens are excluded from loss.
"""

import json

from .environment import GraphEnvironment
from .rollouts import QwenChatCodec, progress_bar, run_episode


def build_demonstration(store, task_id, limits):
    record = store.record(task_id)
    if record["split"] != "train":
        raise ValueError("SFT demonstrations must come only from the source training split")
    if len(record["answer_ids"]) != 1:
        raise ValueError("SFT demonstrations require one complete gold answer")
    path, relations = record["gold_entity_paths"][0], record["relation_path"]
    if (
        len(path) != len(relations) + 1
        or path[0] != record["start_entity"]
        or path[-1] not in record["answer_ids"]
    ):
        raise ValueError("Invalid supervised witness path")
    env = GraphEnvironment(store.graph, record["start_entity"], limits)
    turns = []

    def call(name, arguments):
        if env.done:
            raise ValueError("SFT reference execution exceeds the tool-call budget")
        observation = env.step(name, arguments)
        if not observation.get("ok"):
            raise ValueError("SFT reference tool failure: " + observation["error"])
        raw = (
            "<tool_call>\n"
            + json.dumps({"name": name, "arguments": arguments})
            + "\n</tool_call><|im_end|>"
        )
        turns.append(
            {
                "raw_assistant": raw,
                "tool_call": {"name": name, "arguments": arguments},
                "observation": observation,
                "observation_in_context": not env.done,
            }
        )
        return observation

    for head, relation, tail in zip(path, relations, path[1:]):
        available = call("get_relations", {"entity_id": head})
        if relation not in available["relations"]:
            raise ValueError("SFT witness requires an unavailable relation")
        neighbors = call("get_neighbors", {"entity_id": head, "relation_id": relation})
        if tail not in {e["id"] for e in neighbors["entities"]}:
            raise ValueError("SFT witness edge is absent from the executed observation")
    call("submit_answer", {"entity_id": path[-1]})
    verified = store.verify_episode(task_id, env.submitted_answer, env.trajectory)
    if verified["answer_correct"] != 1:
        raise ValueError("SFT reference did not submit the correct answer")
    return {
        "task_id": task_id,
        "kind": "verified_reference_demonstration",
        "split": "train",
        "question": record["question"],
        "hop_count": record["hop_count"],
        "start_entity": record["start_entity"],
        "prompt": store.prompt(task_id),
        "turns": turns,
        "tool_events": env.trajectory,
        "submitted_answer": env.submitted_answer,
        "tool_calls": env.calls,
        "termination_reason": env.termination_reason,
        "reward": verified["answer_correct"],
        **verified,
    }


def encode_demonstration(store, demonstration, tokenizer, limits):
    """Reuse the exact runtime codec, EOS handling, observation masks and limits."""
    codec = QwenChatCodec(tokenizer)
    actions = [
        list(tokenizer.encode(t["raw_assistant"], add_special_tokens=False))
        for t in demonstration["turns"]
    ]
    cursor = 0

    def scripted_turn(context, max_new_tokens):
        nonlocal cursor
        if cursor >= len(actions):
            raise ValueError("SFT reference requires an unexpected extra action")
        action = actions[cursor]
        if len(action) > max_new_tokens:
            raise ValueError(
                "SFT reference action exceeds runtime token limits; no truncation applied"
            )
        cursor += 1
        return action

    ledger, replay = run_episode(store, demonstration["task_id"], codec, scripted_turn, limits)
    input_ids = ledger.context_ids
    labels = [-100] * len(ledger.prompt_ids) + [
        token if mask else -100 for token, mask in zip(ledger.completion_ids, ledger.env_mask)
    ]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def encode_demonstrations(store, demonstrations, tokenizer, limits, show_progress=True):
    rows = []
    with progress_bar(
        len(demonstrations), "Preparing SFT tool demonstrations", enabled=show_progress
    ) as progress:
        for demo in demonstrations:
            rows.append(encode_demonstration(store, demo, tokenizer, limits))
            progress.update(1)
    return rows


class AssistantOnlyCollator:
    """Right-pad training tensors without changing left-padding used for generation."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        length = max(len(row["input_ids"]) for row in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in features:
            padding = length - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(values, dtype=torch.long) for key, values in batch.items()}


def build_sft_trainer(
    model,
    tokenizer,
    rows,
    output_dir,
    epochs=1,
    accumulation=8,
    learning_rate=2e-5,
    seed=41,
    show_progress=True,
    cpu=False,
    diagnostics_dir=None,
):
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    from .grpo import make_metrics_callback, native_bf16_supported

    if not rows or epochs <= 0 or accumulation < 1 or learning_rate <= 0:
        raise ValueError(
            "SFT requires examples and positive epochs, accumulation and learning rate"
        )
    bf16 = not cpu and native_bf16_supported()
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=-1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=accumulation,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        optim="adamw_torch",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=not cpu and not bf16,
        bf16=bf16,
        use_cpu=cpu,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=False,
        logging_steps=1,
        logging_nan_inf_filter=False,
        report_to=[],
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        disable_tqdm=not show_progress,
    )
    callback = make_metrics_callback(diagnostics_dir or output_dir)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=[callback],
    )
    return trainer
