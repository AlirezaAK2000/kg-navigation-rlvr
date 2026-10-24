"""Optional pinned-library model/trainer construction and measured run artifacts."""

from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import time

from .environment import make_reward
from .rollouts import GRPORollout, append_jsonl, evaluate_policy

PINS = {"torch": "2.6.0", "transformers": "5.3.0", "trl": "1.8.0", "peft": "0.18.1",
        "accelerate": "1.10.1", "bitsandbytes": "0.49.2", "datasets": "4.8.3", "jmespath": "1.0.1",
        "numpy": "1.26.4", "pandas": "2.2.3", "pyarrow": "21.0.0"}
MODEL_REVISIONS = {"Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                   "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca"}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def native_bf16_supported():
    """Exclude emulated BF16; a T4 must take the FP16 path."""
    import torch
    return bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
                and torch.cuda.is_bf16_supported(including_emulation=False))


def dependency_versions():
    versions = {}
    for name in [*PINS, "tokenizers", "huggingface-hub", "numpy", "pandas", "matplotlib", "psutil"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def assert_training_stack(require_cuda=True):
    versions = dependency_versions()
    names = set(PINS) - ({"bitsandbytes"} if not require_cuda else set())
    mismatch = {name: {"expected": PINS[name], "actual": versions[name]} for name in names
                if (versions[name] or "").split("+")[0] != PINS[name]}
    if mismatch:
        raise RuntimeError("Pinned training stack mismatch; run setup in a fresh kernel: " + json.dumps(mismatch))
    import torch
    from trl import GRPOConfig, GRPOTrainer
    required = {"rollout_func", "processing_class", "peft_config"}
    if not required <= inspect.signature(GRPOTrainer.__init__).parameters.keys():
        raise RuntimeError("Installed GRPOTrainer API differs from the inspected release")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the QLoRA training smoke test")
    if require_cuda and torch.version.cuda != "12.4":
        raise RuntimeError(f"Expected the pinned PyTorch cu124 build; found CUDA runtime {torch.version.cuda}")
    return versions


def fingerprint_inputs(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def memory_snapshot(stage):
    import torch
    import psutil
    row = {"stage": stage, "system_available_gib": psutil.virtual_memory().available / 2**30,
           "cuda_available": torch.cuda.is_available(), "allocated_gib": None, "reserved_gib": None,
           "peak_allocated_gib": None, "peak_reserved_gib": None}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        row.update(allocated_gib=torch.cuda.memory_allocated() / 2**30,
                   reserved_gib=torch.cuda.memory_reserved() / 2**30,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    return row


def load_qlora(model_name, revision, lora_r=8, lora_alpha=16):
    assert_training_stack()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU before importing torch; T4 memories are not combined")
    dtype = torch.bfloat16 if native_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, trust_remote_code=False,
              dtype=dtype, quantization_config=quantization, device_map={"": 0}, attn_implementation="sdpa")
    suffixes = {name.split(".")[-1] for name, _ in model.named_modules()}
    if not set(LORA_TARGETS) <= suffixes:
        raise RuntimeError(f"Qwen architecture is missing LoRA targets: {set(LORA_TARGETS) - suffixes}")
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                   target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    if any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
        raise RuntimeError("A non-LoRA parameter unexpectedly requires gradients")
    model.print_trainable_parameters()
    return model, tokenizer, dtype


def make_grpo_args(output_dir, steps, limits, generations=4, accumulation=4, learning_rate=5e-6,
                   save_steps=10, seed=41, cpu=False):
    from trl import GRPOConfig
    if steps < 1 or save_steps < 1:
        raise ValueError("Training and checkpoint intervals must be positive")
    if accumulation < generations or accumulation % generations:
        raise ValueError("With microbatch=1, gradient accumulation must be a positive multiple of num_generations")
    import torch
    bf16 = not cpu and native_bf16_supported()
    kwargs = dict(output_dir=str(output_dir), max_steps=steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=accumulation, generation_batch_size=accumulation,
        num_generations=generations, max_completion_length=limits.max_completion_tokens,
        temperature=0.8, top_p=1.0, top_k=0, beta=0.0, loss_type="grpo", scale_rewards="group",
        num_iterations=1, learning_rate=learning_rate, lr_scheduler_type="constant", warmup_steps=0,
        optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=not cpu and not bf16, bf16=bf16, use_cpu=cpu, use_vllm=False,
        mask_truncated_completions=False, remove_unused_columns=False,
        logging_steps=1, logging_nan_inf_filter=False, save_strategy="steps", save_steps=save_steps,
        save_total_limit=2, save_only_model=False,
        eval_strategy="no", report_to=[], seed=seed, data_seed=seed, dataloader_num_workers=0,
        log_completions=False)
    unsupported = set(kwargs) - GRPOConfig.__dataclass_fields__.keys()
    if unsupported:
        raise RuntimeError(f"Unsupported pinned GRPOConfig arguments: {unsupported}")
    return GRPOConfig(**kwargs)


def make_metrics_callback(output_dir, store=None, tokenizer=None, eval_task_ids=None, limits=None, eval_every=0):
    from transformers import TrainerCallback
    output_dir = Path(output_dir)
    class MetricsCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, model=None, **kwargs):
            append_jsonl(output_dir / "step_metrics.jsonl", {"step": state.global_step,
                "step_seconds": time.perf_counter() - self.started, **memory_snapshot("optimizer_step")})
            if eval_every and state.global_step % eval_every == 0 and eval_task_ids:
                evaluate_policy(model, tokenizer, store, eval_task_ids, limits,
                                output_dir / "periodic_evaluation", f"step_{state.global_step}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            for key, value in logs.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite trainer metric {key}: {value}")
            append_jsonl(output_dir / "trainer_metrics.jsonl", {"step": state.global_step, **logs})
    return MetricsCallback()


def build_trainer(model, tokenizer, store, task_ids, limits, output_dir, phase, steps,
                  generations=4, accumulation=4, learning_rate=5e-6, save_steps=10, seed=41,
                  eval_task_ids=None, eval_every=0):
    from datasets import Dataset
    from trl import GRPOTrainer
    if len(task_ids) < accumulation // generations:
        raise ValueError("Not enough unique training rows for one complete GRPO generation batch")
    args = make_grpo_args(output_dir, steps, limits, generations, accumulation, learning_rate, save_steps, seed)
    dataset = Dataset.from_list([{"prompt": store.prompt(task_id)} for task_id in task_ids])
    assert dataset.column_names == ["prompt"]
    rollout = GRPORollout(store, tokenizer, limits, output_dir, phase)
    callback = make_metrics_callback(output_dir, store, tokenizer, eval_task_ids, limits, eval_every)
    # No `tools` / `environment_factory`: those would trigger TRL's independent
    # tool loop and could execute our already-executed actions a second time.
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, args=args, train_dataset=dataset,
                reward_funcs=make_reward(store), rollout_func=rollout, callbacks=[callback])
    return trainer, rollout


def lora_snapshot(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad and "lora_" in name}


def verify_optimizer_update(model, before, trainer, output_dir, required_steps=1):
    import torch
    changed, max_change = [], 0.0
    for name, p in model.named_parameters():
        if name in before:
            delta = float((p.detach().cpu() - before[name]).abs().max())
            if not math.isfinite(delta):
                raise FloatingPointError(f"Non-finite LoRA weight delta: {name}")
            if delta:
                changed.append(name)
                max_change = max(max_change, delta)
    losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
    if trainer.state.global_step < required_steps or not losses or not all(math.isfinite(x) for x in losses):
        raise RuntimeError("No verified finite GRPO optimizer steps/losses were recorded")
    if not changed:
        raise RuntimeError("No LoRA parameters changed. Inspect zero_reward_group_fraction and zero_variance_group_fraction: binary rewards with homogeneous groups produce zero advantages. Increase real smoke groups / tool budget or diagnose tool formatting; no successful update is claimed.")
    if not any(row.get("grad_norm", 0) > 0 for row in trainer.state.log_history):
        raise RuntimeError("No positive finite gradient norm was logged")
    checkpoints = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    if not checkpoints or not all((checkpoints[-1] / name).exists() for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json", "adapter_model.safetensors")):
        raise RuntimeError("Smoke test did not save a complete resumable adapter/trainer checkpoint")
    if not list(checkpoints[-1].glob("rng_state*.pth")):
        raise RuntimeError("Smoke checkpoint has no RNG state for resumption")
    return {"optimizer_steps": trainer.state.global_step, "changed_lora_tensors": len(changed),
            "max_lora_weight_delta": max_change, "finite_losses": True,
            "checkpoint": str(checkpoints[-1]), **memory_snapshot("verified_smoke_update")}


def training_manifest(model_name, revision, limits, task_ids, generations, accumulation, learning_rate, seed, r=8, alpha=16):
    return {"model_name": model_name, "revision": revision, "limits": asdict(limits),
            "task_ids": task_ids, "num_generations": generations, "gradient_accumulation_steps": accumulation,
            "learning_rate": learning_rate, "seed": seed, "lora_r": r, "lora_alpha": alpha,
            "lora_dropout": 0.0, "lora_targets": LORA_TARGETS, "beta": 0.0,
            "loss_type": "grpo", "versions": dependency_versions(),
            "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(Path(__file__).parent.glob("*.py"))}}


def check_resume(checkpoint, manifest):
    checkpoint = Path(checkpoint)
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json"]
    if (any(not (checkpoint / name).is_file() for name in required)
            or not (checkpoint / "adapter_model.safetensors").is_file()):
        raise ValueError("Resume requires a complete trainer checkpoint, not an adapter-only directory")
    if not list(checkpoint.glob("rng_state*.pth")):
        raise ValueError("Resume checkpoint has no saved RNG state")
    prior_path = checkpoint.parent / "run_manifest.json"
    if not prior_path.exists():
        raise ValueError("Resume requires the original run_manifest.json alongside the checkpoints")
    prior = json.loads(prior_path.read_text())
    if prior != manifest:
        differences = [key for key in manifest if prior.get(key) != manifest[key]]
        raise ValueError(f"Incompatible resume configuration: {differences}")
    return str(checkpoint)


def export_adapter(model, tokenizer, directory, manifest, extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Embeddings are frozen and never resized in this pipeline. Explicitly
    # exclude them instead of PEFT's Hub-based "auto" vocabulary comparison.
    model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(directory)
    (directory / "training_manifest.json").write_text(json.dumps({**manifest, **(extra or {})}, indent=2) + "\n")
    if not (directory / "adapter_model.safetensors").exists():
        raise RuntimeError("Adapter export did not create adapter_model.safetensors")
    if (directory / "model.safetensors").exists():
        raise RuntimeError("Unexpected full base-model copy in the adapter export")
    return directory


def reload_adapter_in_place(model, directory):
    """Use the same quantized base; do not allocate another 1.7B model copy."""
    from peft import PeftModel
    base = model.unload()
    if any("lora_" in name for name, _ in base.named_parameters()):
        raise RuntimeError("Unloading left adapter parameters on the base model")
    if hasattr(base, "peft_config"):
        delattr(base, "peft_config")  # residual metadata after PEFT 0.18.1 unload
    gc.collect()
    return PeftModel.from_pretrained(base, directory, is_trainable=False)


def observed_trajectory_html(graph, trace, output, max_nodes=80, max_edges=140):
    """Visualize only model-observed traversals; never use a gold path highlight."""
    import html
    from ..visualization import HTML
    observed, edges = {trace["start_entity"]}, {}
    for turn in trace["turns"]:
        call, result = turn.get("tool_call", {}), turn.get("observation", {})
        if call.get("name") != "get_neighbors" or not result.get("ok") or not turn.get("observation_in_context"):
            continue
        head, rel = call["arguments"]["entity_id"], call["arguments"]["relation_id"]
        observed.update([head, *(e["id"] for e in result.get("entities", []))])
        for entity in result.get("entities", []):
            edges[head, rel, entity["id"]] = {"from": head, "relation": rel, "to": entity["id"], "gold": False}
    if len(observed) > max_nodes or len(edges) > max_edges:
        raise ValueError("Increase trajectory display limits; observed traversals are never silently omitted")
    from ..visualization import COLORS
    nodes = [{"id": e, **graph.entities[e], "color": COLORS[graph.entities[e]["type"]], "layer": None,
              "role": "start" if e == trace["start_entity"] else "answer" if e == trace["submitted_answer"] else "context"} for e in sorted(observed)]
    payload = {"mode": "Model-observed traversal (no gold path)", "model_question": trace["question"], "start": trace["start_entity"], "nodes": nodes,
               "edges": list(edges.values()), "paths": [], "question": None, "colors": COLORS,
               "graph_counts": {"nodes": len(graph.entities), "edges": len(graph.triples)}, "limits": {"nodes": max_nodes, "edges": max_edges}}
    raw = json.dumps(payload).replace("<", "\\u003c").replace(">", "\\u003e")
    document = HTML.replace("__TITLE__", html.escape("Model-observed graph traversal")).replace("__PAYLOAD__", raw)
    document = document.replace('data.question?data.question.question:data.nodes.find(n=>n.id===data.start).name',
                                'data.question?data.question.question:(data.model_question || data.nodes.find(n=>n.id===data.start).name)')
    document = document.replace("Start: cyan ring · intermediate: violet ring · answers: amber ring. Other nodes provide context and distractors.",
                                "Cyan ring: starting entity. Amber ring: submitted entity, regardless of correctness. Edges are observed tool results; no gold path is shown.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(document)
    return payload
