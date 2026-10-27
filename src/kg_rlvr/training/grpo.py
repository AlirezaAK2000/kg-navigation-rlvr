"""QLoRA loading, TRL training and compact experiment metadata."""

from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from .rewards import correctness_reward, retrieval_reward, REWARD_SPEC
from .sampling import exposure_schedule
from .rollouts import GRPORollout, append_jsonl, evaluate_policy, evaluation_reward_record


PINS = {
    "torch": "2.6.0",
    "transformers": "5.3.0",
    "trl": "1.8.0",
    "peft": "0.18.1",
    "accelerate": "1.10.1",
    "bitsandbytes": "0.49.2",
    "datasets": "4.8.3",
    "jmespath": "1.0.1",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "pyarrow": "21.0.0",
    "rdflib": "7.6.0",
    "pyparsing": "3.2.3",
}
MODEL_REVISIONS = {
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def native_bf16_supported():
    """Exclude emulated BF16; a T4 must take the FP16 path."""
    import torch

    return bool(
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
        and torch.cuda.is_bf16_supported(including_emulation=False)
    )


def dependency_versions():
    versions = {}
    for name in [
        *PINS,
        "tokenizers",
        "huggingface-hub",
        "numpy",
        "pandas",
        "matplotlib",
        "psutil",
        "tqdm",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def fingerprint_inputs(directory):
    directory = Path(directory)
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def memory_snapshot(stage):
    import torch
    import psutil

    row = {
        "stage": stage,
        "system_available_gib": psutil.virtual_memory().available / 2**30,
        "cuda_available": torch.cuda.is_available(),
        "allocated_gib": None,
        "reserved_gib": None,
        "peak_allocated_gib": None,
        "peak_reserved_gib": None,
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        row.update(
            allocated_gib=torch.cuda.memory_allocated() / 2**30,
            reserved_gib=torch.cuda.memory_reserved() / 2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
    return row


def lora_precision(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return {
        "trainable_dtype": str(trainable[0].dtype) if trainable else None,
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(p.numel() for p in trainable),
    }


def restore_lora_fp32(model):
    """Adapter storage stays FP32; quantized base weights keep their own dtype."""
    import torch

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name and parameter.requires_grad:
                parameter.data = parameter.data.float()
    return lora_precision(model)


def load_qlora(model_name, revision, lora_r=8, lora_alpha=16, show_progress=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.utils import logging as transformers_logging
    from huggingface_hub.utils import enable_progress_bars, disable_progress_bars
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if show_progress:
        enable_progress_bars()
        transformers_logging.enable_progress_bar()
    else:
        disable_progress_bars()
        transformers_logging.disable_progress_bar()
    dtype = torch.bfloat16 if native_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=False,
        dtype=dtype,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        autocast_adapter_dtype=True,
    )
    restore_lora_fp32(model)
    model.print_trainable_parameters()
    return model, tokenizer, dtype


def make_grpo_args(
    output_dir,
    steps,
    limits,
    generations=4,
    accumulation=4,
    learning_rate=5e-6,
    save_steps=10,
    seed=41,
    cpu=False,
    show_progress=True,
    shuffle_dataset=True,
):
    from trl import GRPOConfig

    if accumulation < generations or accumulation % generations:
        raise ValueError(
            "With microbatch=1, gradient accumulation must be a positive multiple of num_generations"
        )
    import torch

    bf16 = not cpu and native_bf16_supported()
    kwargs = dict(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=accumulation,
        generation_batch_size=accumulation,
        num_generations=generations,
        max_completion_length=limits.max_completion_tokens,
        temperature=0.8,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",
        reward_weights=[1.0, 0.2],
        num_iterations=1,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=not cpu and not bf16,
        bf16=bf16,
        use_cpu=cpu,
        use_vllm=False,
        mask_truncated_completions=False,
        remove_unused_columns=False,
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        save_only_model=False,
        eval_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        log_completions=False,
        disable_tqdm=not show_progress,
        shuffle_dataset=shuffle_dataset,
    )
    return GRPOConfig(**kwargs)


def make_metrics_callback(
    output_dir,
    store=None,
    tokenizer=None,
    eval_task_ids=None,
    limits=None,
    eval_every=0,
    show_progress=True,
):
    from transformers import TrainerCallback

    directory = Path(output_dir)

    class MetricsCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, model=None, **kwargs):
            memory = memory_snapshot("optimizer_step")
            append_jsonl(
                directory / "step_metrics.jsonl",
                {
                    "step": state.global_step,
                    "step_seconds": time.perf_counter() - self.started,
                    **memory,
                },
            )
            if eval_every and state.global_step % eval_every == 0 and eval_task_ids:
                label = f"step_{state.global_step}"
                metrics, traces = evaluate_policy(
                    model,
                    tokenizer,
                    store,
                    eval_task_ids,
                    limits,
                    directory / "periodic_evaluation",
                    label,
                    show_progress=show_progress,
                    progress_position=1,
                    training_step=state.global_step,
                )
                append_jsonl(
                    directory / "evaluation_metrics.jsonl",
                    {"phase": directory.name, "step": state.global_step, **metrics},
                )
                for index, trace in enumerate(traces, 1):
                    append_jsonl(
                        directory / "evaluation_rewards.jsonl",
                        evaluation_reward_record(trace, index, label, state.global_step),
                    )

        def on_log(self, args, state, control, logs=None, **kwargs):
            append_jsonl(
                directory / "trainer_metrics.jsonl", {"step": state.global_step, **(logs or {})}
            )

    return MetricsCallback()


def build_trainer(
    model,
    tokenizer,
    store,
    task_ids,
    limits,
    output_dir,
    phase,
    steps,
    generations=4,
    accumulation=4,
    learning_rate=5e-6,
    save_steps=10,
    seed=41,
    eval_task_ids=None,
    eval_every=0,
    show_progress=True,
    hop_weights=None,
    excluded_sft_task_ids=None,
    diagnostics_dir=None,
):
    from datasets import Dataset
    from trl import GRPOTrainer

    diagnostics_dir = Path(diagnostics_dir or output_dir)
    args = make_grpo_args(
        output_dir,
        steps,
        limits,
        generations,
        accumulation,
        learning_rate,
        save_steps,
        seed,
        show_progress=show_progress,
        shuffle_dataset=hop_weights is None,
    )
    scheduled_ids = (
        exposure_schedule(store, task_ids, steps * (accumulation // generations), hop_weights, seed)
        if hop_weights is not None
        else list(task_ids)
    )
    if hop_weights is not None:
        # RepeatSampler retains this order and supplies G actual samples/group;
        # repetitions needed for buffered gradient accumulation remain TRL's job.
        plan = [
            {
                "group": i + 1,
                "optimizer_step": i // (accumulation // generations) + 1,
                "task_id": qid,
                "hop_count": store.record(qid)["hop_count"],
            }
            for i, qid in enumerate(scheduled_ids)
        ]
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        (diagnostics_dir / "exposure_schedule.json").write_text(json.dumps(plan, indent=2) + "\n")
    dataset = Dataset.from_list(
        [
            {"prompt": store.prompt(qid), "answer_ids": store.record(qid)["answer_ids"]}
            for qid in scheduled_ids
        ]
    )
    rollout = GRPORollout(
        store, tokenizer, limits, diagnostics_dir, phase, show_progress=show_progress
    )
    callback = make_metrics_callback(
        diagnostics_dir,
        store,
        tokenizer,
        eval_task_ids,
        limits,
        eval_every,
        show_progress=show_progress,
    )
    adapter_before_init = lora_snapshot(model)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=dataset,
        reward_funcs=[correctness_reward, retrieval_reward],
        rollout_func=rollout,
        callbacks=[callback],
    )
    # TRL 1.8 casts quantized-model adapters to BF16. Restore the exact FP32
    # warm start before optimizer creation; T4 FP16 AMP needs FP32 gradients.
    import torch

    restore_lora_fp32(trainer.model)
    with torch.no_grad():
        for name, parameter in trainer.model.named_parameters():
            if name in adapter_before_init:
                parameter.copy_(adapter_before_init[name])
    return trainer, rollout


def lora_snapshot(model):
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and "lora_" in name
    }


def summarize_training(model, before, trainer):
    """Report measured updates without blocking homogeneous-reward training."""
    deltas = [
        float((p.detach().cpu() - before[name]).abs().max())
        for name, p in model.named_parameters()
        if name in before
    ]
    return {
        "optimizer_steps": trainer.state.global_step,
        "changed_lora_tensors": sum(delta > 0 for delta in deltas),
        "max_lora_weight_delta": max(deltas, default=0),
        "precision": lora_precision(model),
    }


def training_manifest(
    model_name,
    revision,
    limits,
    task_ids,
    generations,
    accumulation,
    learning_rate,
    seed,
    r=8,
    alpha=16,
    hop_weights=None,
    single_answer_only=False,
    supervised_warm_start=None,
):
    return {
        "model_name": model_name,
        "revision": revision,
        "limits": asdict(limits),
        "task_ids": task_ids,
        "num_generations": generations,
        "gradient_accumulation_steps": accumulation,
        "learning_rate": learning_rate,
        "seed": seed,
        "lora_r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "lora_targets": LORA_TARGETS,
        "beta": 0.0,
        "trainable_parameter_dtype": "torch.float32",
        "loss_type": "grpo",
        "versions": dependency_versions(),
        "reward_spec": dict(REWARD_SPEC),
        "single_answer_only": single_answer_only,
        "wording_version": "single_v1" if single_answer_only else "source",
        "supervised_warm_start": supervised_warm_start,
        "training_exposure": {
            "algorithm": "weighted_deficit_v1",
            "hop_weights": {str(h): float(w) for h, w in hop_weights.items()},
        }
        if hop_weights is not None
        else {"algorithm": "trl_shuffle"},
    }


def export_adapter(model, tokenizer, directory, manifest, extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Embeddings are frozen and never resized in this pipeline. Explicitly
    # exclude them instead of PEFT's Hub-based "auto" vocabulary comparison.
    model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(directory)
    (directory / "training_manifest.json").write_text(
        json.dumps({**manifest, **(extra or {})}, indent=2) + "\n"
    )
    return directory


def reload_adapter_in_place(model, directory, *, is_trainable=False):
    """Use the same quantized base; do not allocate another 1.7B model copy."""
    from peft import PeftModel

    base = model.unload()
    if hasattr(base, "peft_config"):
        delattr(base, "peft_config")  # residual metadata after PEFT 0.18.1 unload
    gc.collect()
    reloaded = PeftModel.from_pretrained(base, directory, is_trainable=is_trainable)
    if is_trainable:
        restore_lora_fp32(reloaded)
    return reloaded


def lora_fingerprint(model):
    """Fingerprint adapter values independently of their requires_grad flags."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" in name:
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()
