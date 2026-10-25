"""Bounded multi-turn Qwen rollouts, passed to the actual TRL GRPOTrainer.

Only orchestration lives here: GRPO advantages, clipping, loss, gradients and
optimizer updates are implemented by TRL. Each invocation gets fresh graph state.
"""

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
import time

from .environment import GraphEnvironment, TOOL_SCHEMAS, parse_qwen_action, verify_group_prompts


def progress_bar(total, description, *, unit="question", enabled=True, position=0, leave=True):
    """Use notebook widgets when available and terminal bars otherwise."""
    from tqdm.auto import tqdm
    return tqdm(total=total, desc=description, unit=unit, disable=not enabled,
                dynamic_ncols=True, position=position, leave=leave)


class QwenChatCodec:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if tokenizer.convert_tokens_to_ids("<|im_end|>") != tokenizer.eos_token_id:
            raise ValueError("This rollout codec requires the pinned Qwen3 chat template/EOS")
        template = tokenizer.get_chat_template()
        if "<tool_call>" not in template or "<tool_response>" not in template:
            raise ValueError("The model chat template does not support the required tools")

    def prompt_ids(self, prompt):
        return list(self.tokenizer.apply_chat_template(prompt, tools=TOOL_SCHEMAS, tokenize=True,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False))

    def feedback_ids(self, name, observation):
        # Render the real model template; never re-tokenize sampled action text.
        # Qwen3 changes how it renders the last assistant's reasoning block when
        # a tool message is appended, so naive prefix subtraction is unsafe.
        content = json.dumps(observation, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")
        dummy = [{"role": "user", "content": "tool-format-probe"},
                 {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": {}}}]},
                 {"role": "tool", "name": name, "content": content}]
        full = list(self.tokenizer.apply_chat_template(dummy, tools=TOOL_SCHEMAS, tokenize=True,
                    return_dict=False, add_generation_prompt=True, enable_thinking=False))
        ends = [i for i, token in enumerate(full) if token == self.tokenizer.eos_token_id]
        if len(ends) != 4:  # system tools, dummy user, assistant action, tool/user observation
            raise ValueError("Unexpected pinned Qwen3 tool template boundaries")
        return full[ends[-2] + 1:]

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


@dataclass
class TokenLedger:
    prompt_ids: list
    limits: object
    completion_ids: list = field(default_factory=list)
    env_mask: list = field(default_factory=list)
    segments: list = field(default_factory=list)

    @property
    def context_ids(self):
        return self.prompt_ids + self.completion_ids

    @property
    def remaining(self):
        return min(self.limits.max_completion_tokens - len(self.completion_ids),
                   self.limits.max_context_tokens - len(self.context_ids))

    def append(self, ids, model_generated):
        ids = list(ids)
        if len(ids) > self.remaining:
            raise ValueError("Token segment exceeds episode limits; no silent truncation")
        begin = len(self.completion_ids)
        self.completion_ids.extend(ids)
        self.env_mask.extend([int(model_generated)] * len(ids))
        self.segments.append({"begin": begin, "end": len(self.completion_ids), "source": "model" if model_generated else "environment"})
        assert len(self.env_mask) == len(self.completion_ids)


def run_episode(store, task_id, codec, sample_turn, limits, on_turn=None):
    """sample_turn(context_ids, max_new_tokens) must return ACTUAL sampled IDs.

    Dependency injection supports scripted unit fixtures; those are never used
    as evaluation/training results. The notebook supplies Transformers generation.
    """
    record = store.record(task_id)
    prompt = store.prompt(task_id)
    initial = codec.prompt_ids(prompt)
    if len(initial) > limits.max_prompt_tokens:
        raise ValueError(f"Public prompt for {task_id} has {len(initial)} tokens > max_prompt_tokens={limits.max_prompt_tokens}; increase the explicit prompt/context budget")
    env = GraphEnvironment(store.graph, record["start_entity"], limits)
    ledger, turns = TokenLedger(initial, limits), []
    def report(turn, status):
        if on_turn is not None:
            on_turn(task=task_id, turn=turn, calls=env.calls,
                    tokens=sum(ledger.env_mask), status=status)

    started = time.perf_counter()
    for _ in range(limits.max_tool_calls):
        if env.done:
            break
        budget = min(limits.max_turn_tokens, ledger.remaining)
        if budget <= 0:
            env._finish("completion_limit")
            break
        report(len(turns) + 1, "generating")
        sampled = list(sample_turn(ledger.context_ids, budget))
        if not sampled:
            raise RuntimeError("Generation backend returned no action tokens; cannot construct a GRPO sample without fabricating tokens")
        if len(sampled) > budget:
            raise ValueError("Generation backend ignored max_new_tokens")
        # Stop at the first model EOS; do not append generation padding.
        if codec.tokenizer.eos_token_id in sampled:
            sampled = sampled[:sampled.index(codec.tokenizer.eos_token_id) + 1]
        ledger.append(sampled, model_generated=True)
        text = codec.decode(sampled)
        turns.append({"raw_assistant": text, "model_tokens": len(sampled)})
        report(len(turns), "generated")
        if sampled[-1] != codec.tokenizer.eos_token_id:
            env._finish("turn_token_limit" if len(sampled) == limits.max_turn_tokens else "completion_limit")
            break  # incomplete action is never executed
        try:
            name, arguments = parse_qwen_action(text)
        except ValueError as error:
            if str(error) != "no_tool_call":
                env._invalid(str(error))
            env._finish(str(error))
            break
        observation = env._dispatch(name, arguments)
        turns[-1].update(tool_call={"name": name, "arguments": arguments}, observation=observation,
                         observation_in_context=False)
        if env.done:
            break  # terminal acknowledgment need not be fed into another generation
        feedback = codec.feedback_ids(name, observation)
        if len(feedback) >= ledger.remaining:
            env._finish("completion_limit")
            break  # never splice a partial observation into the context
        ledger.append(feedback, model_generated=False)
        turns[-1]["observation_in_context"] = True
    if not env.done:
        env._finish("turn_limit")
    report(len(turns), env.termination_reason)
    reward = store.reward(task_id, env.submitted_answer)
    verified = store.verify_episode(task_id, env.submitted_answer, env.trajectory)
    trace = {"task_id": task_id, "question": record["question"], "start_entity": record["start_entity"],
             "hop_count": record["hop_count"], "split": record["split"], "prompt": prompt,
             "turns": turns, "tool_events": copy.deepcopy(env.trajectory),
             "submitted_answer": env.submitted_answer, "gold_answer_ids": record["answer_ids"],
             "reward": reward, **verified, "tool_calls": env.calls, "invalid_tool_calls": env.invalid_calls,
             "termination_reason": env.termination_reason, "submitted": env.submitted_answer is not None,
             "model_tokens": sum(ledger.env_mask), "environment_tokens": len(ledger.env_mask) - sum(ledger.env_mask),
             "elapsed_seconds": time.perf_counter() - started, "token_segments": ledger.segments}
    return ledger, trace


@contextmanager
def generation_mode(model):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def transformer_sampler(model, tokenizer, temperature=0.8, sample=True, use_cache=True):
    """No top-k/p warping: matches TRL's temperature-scaled policy log-probs."""
    import torch
    from transformers import GenerationConfig
    device = model.get_input_embeddings().weight.device
    # The baseline runs before Accelerate wraps the model, while training runs
    # after wrapping. Apply the same native CUDA AMP precision explicitly in
    # both phases, including the FP32 modules retained by k-bit preparation.
    amp_dtype = None
    if device.type == "cuda":
        native_bf16 = (torch.cuda.get_device_capability(device)[0] >= 8
                       and torch.cuda.is_bf16_supported(including_emulation=False))
        amp_dtype = torch.bfloat16 if native_bf16 else torch.float16
    # Start with fresh defaults: a Hub generation_config must not silently apply
    # top-k/p or another logits warper absent from TRL's policy log-probabilities.
    generation_config = GenerationConfig(do_sample=sample, use_cache=use_cache,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        **({"temperature": temperature, "top_p": 1.0, "top_k": 0} if sample else {}))
    def sample_turn(context_ids, max_new_tokens):
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        precision = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None else nullcontext()
        with torch.inference_mode(), precision:
            output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        generation_config=generation_config, max_new_tokens=max_new_tokens)
        return output[0, len(context_ids):].detach().cpu().tolist()
    return sample_turn


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def summarize_episodes(traces):
    n = len(traces)
    calls = sum(t["tool_calls"] for t in traces)
    invalid = sum(t["invalid_tool_calls"] for t in traces)
    correct = [t for t in traces if t["reward"] == 1.0]
    by_hop = {str(h): sum(t["reward"] for t in traces if t["hop_count"] == h) / sum(t["hop_count"] == h for t in traces)
              for h in sorted({t["hop_count"] for t in traces})}
    total_reward = sum(t["reward"] for t in traces)
    training_rewards = [t.get("training_reward", t["reward"]) for t in traces]
    training_by_hop = {str(h): sum(t.get("training_reward", t["reward"]) for t in traces if t["hop_count"] == h)
                      / sum(t["hop_count"] == h for t in traces)
                      for h in sorted({t["hop_count"] for t in traces})}
    return {"episodes": n, "mean_reward": total_reward / n if n else None,
            "total_reward": total_reward, "mean_reward_by_hop": by_hop,
            "mean_training_reward": sum(training_rewards) / n if n else None,
            "total_training_reward": sum(training_rewards), "mean_training_reward_by_hop": training_by_hop,
            "retrieval_success_rate": sum(t.get("gold_entity_retrieved", False) for t in traces) / n if n else None,
            "mean_retrieval_reward": sum(t.get("retrieval_reward", 0.0) for t in traces) / n if n else None,
            "accuracy": total_reward / n if n else None,
            "submission_rate": sum(t["submitted"] for t in traces) / n if n else None,
            "valid_tool_call_rate": (calls - invalid) / calls if calls else 0.0,
            "episodes_with_valid_calls_rate": sum(t["tool_calls"] > t["invalid_tool_calls"] for t in traces) / n if n else None,
            "episodes_all_calls_valid_rate": sum(t["tool_calls"] > 0 and t["invalid_tool_calls"] == 0 for t in traces) / n if n else None,
            "average_tool_calls": calls / n if n else None,
            "average_tool_calls_correct": sum(t["tool_calls"] for t in correct) / len(correct) if correct else None,
            "invalid_tool_call_rate": invalid / calls if calls else 0.0,
            "accuracy_by_hop": by_hop, "termination_reasons": dict(Counter(t["termination_reason"] for t in traces))}


class GRPORollout:
    """Pinned TRL 1.8.0 rollout_func protocol, one output per ALREADY repeated row."""
    def __init__(self, store, tokenizer, limits, output_dir, phase, show_progress=True):
        self.store, self.codec, self.limits = store, QwenChatCodec(tokenizer), limits
        self.output_dir, self.phase = Path(output_dir), phase
        self.batch_count = 0
        self.last_traces = []
        self.show_progress = show_progress

    def __call__(self, prompts, trainer):
        if trainer.accelerator.num_processes != 1:
            raise ValueError("This bounded-memory integration supports one GPU/process only")
        group_size = trainer.num_generations
        verify_group_prompts(prompts, group_size)
        model = trainer.accelerator.unwrap_model(trainer.model)
        sampler = transformer_sampler(model, self.codec.tokenizer, temperature=trainer.temperature, sample=True)
        ledgers, traces = [], []
        step = trainer.state.global_step + 1
        with generation_mode(model), progress_bar(
                len(prompts), f"{self.phase} step {step}: sampling", unit="trajectory",
                enabled=self.show_progress, position=1, leave=False) as progress:
            for prompt in prompts:
                task_id = self.store.resolve_prompt(prompt)
                ledger, trace = run_episode(self.store, task_id, self.codec, sampler, self.limits,
                                           on_turn=progress.set_postfix if self.show_progress else None)
                ledgers.append(ledger)
                traces.append(trace)
                progress.set_postfix(completed=f"{len(traces)}/{len(prompts)}",
                    training_reward=f"{sum(t['training_reward'] for t in traces) / len(traces):.3f}", refresh=False)
                progress.update(1)
        self.batch_count += 1
        for group_number, offset in enumerate(range(0, len(traces), group_size)):
            group = traces[offset:offset + group_size]
            assert len({t["task_id"] for t in group}) == 1
            group_id = f"{self.phase}:{step}:{self.batch_count}:{group_number}"
            for trace in group:
                append_jsonl(self.output_dir / "trajectories.jsonl", {"phase": self.phase, "step": step, "group_id": group_id, **trace})
        grouped_rewards = [[t["training_reward"] for t in traces[i:i + group_size]]
                           for i in range(0, len(traces), group_size)]
        grouped_answers = [[t["answer_correct"] for t in traces[i:i + group_size]]
                           for i in range(0, len(traces), group_size)]
        metrics = summarize_episodes(traces)
        append_jsonl(self.output_dir / "rollout_metrics.jsonl", {"phase": self.phase, "step": step,
            "mean_episode_reward": metrics["mean_training_reward"], "success_fraction": metrics["accuracy"],
            "zero_reward_group_fraction": sum(all(r == 0 for r in group) for group in grouped_rewards) / len(grouped_rewards),
            "zero_variance_group_fraction": sum(len(set(group)) == 1 for group in grouped_rewards) / len(grouped_rewards),
            "answer_zero_variance_group_fraction": sum(len(set(group)) == 1 for group in grouped_answers) / len(grouped_answers),
            **metrics})
        self.last_traces = traces
        return {"prompt_ids": [l.prompt_ids for l in ledgers], "completion_ids": [l.completion_ids for l in ledgers],
                "env_mask": [l.env_mask for l in ledgers], "logprobs": None,
                "task_ids": [t["task_id"] for t in traces],
                "submitted_answers": [t["submitted_answer"] for t in traces],
                "tool_events": [t["tool_events"] for t in traces]}


def evaluation_reward_record(trace, index, label, training_step=None):
    """One terminal reward per evaluated question; no intermediate tool rewards."""
    return {"evaluation": label, "training_step": training_step, "evaluation_index": index,
            "task_id": trace["task_id"], "hop_count": trace["hop_count"],
            "reward": trace["reward"], "answer_correct": trace["answer_correct"],
            "gold_entity_retrieved": trace["gold_entity_retrieved"],
            "retrieval_reward": trace["retrieval_reward"], "training_reward": trace["training_reward"],
            "submitted_answer": trace["submitted_answer"],
            "tool_calls": trace["tool_calls"], "termination_reason": trace["termination_reason"]}


def evaluate_policy(model, tokenizer, store, task_ids, limits, output_dir, label,
                    show_progress=True, progress_position=0, training_step=None):
    """Same real tools/limits for base, trained, and reloaded adapters; no gold guidance."""
    import torch
    codec = QwenChatCodec(tokenizer)
    sampler = transformer_sampler(model, tokenizer, sample=False)
    traces = []
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cuda_devices = [model.get_input_embeddings().weight.device.index] if model.get_input_embeddings().weight.is_cuda else []
    with (directory / f"{label}_rewards.jsonl").open("w") as reward_file, torch.random.fork_rng(devices=cuda_devices), generation_mode(model), progress_bar(
            len(task_ids), label.replace("_", " "), enabled=show_progress,
            position=progress_position) as progress:
        for task_id in task_ids:
            _, trace = run_episode(store, task_id, codec, sampler, limits,
                                  on_turn=progress.set_postfix if show_progress else None)
            traces.append(trace)
            reward_file.write(json.dumps(evaluation_reward_record(trace, len(traces), label, training_step),
                                         allow_nan=False) + "\n")
            reward_file.flush()  # retain completed-question rewards if a later episode fails
            progress.set_postfix(completed=f"{len(traces)}/{len(task_ids)}",
                reward=f"{trace['reward']:.0f}",
                mean_reward=f"{sum(t['reward'] for t in traces) / len(traces):.3f}",
                accuracy=f"{sum(t['reward'] for t in traces) / len(traces):.1%}",
                submitted=f"{sum(t['submitted'] for t in traces)}/{len(traces)}", refresh=False)
            progress.update(1)
    # Label-specific evaluation files are replaced on rerun, never source datasets.
    (directory / f"{label}_trajectories.jsonl").write_text("".join(json.dumps(t) + "\n" for t in traces))
    summary = summarize_episodes(traces)
    (directory / f"{label}_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary, traces
