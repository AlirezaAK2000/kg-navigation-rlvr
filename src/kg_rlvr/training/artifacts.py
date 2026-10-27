"""Persist each policy's measured results before another policy starts."""

import json
from pathlib import Path
import shutil
import zipfile

from .inspection import render_traces_html


VARIANTS = ("vanilla", "sft", "sft_grpo", "grpo_only")
LABELS = {"vanilla": "Vanilla", "sft": "SFT", "sft_grpo": "SFT + GRPO", "grpo_only": "GRPO only"}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path):
    path = Path(path)
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    temporary.replace(path)


def save_trajectory_html(path, traces, label, tokenizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = render_traces_html(traces, label=label, count=max(1, len(traces)), tokenizer=tokenizer)
    path.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "</head><body>" + body + "</body></html>"
    )


def save_comparison(run_root):
    """Refresh from completed evaluations only; skipped/failed scores stay null."""
    run_root = Path(run_root)
    measured = {}
    for variant in VARIANTS:
        directory = run_root / "variants" / variant
        status = directory / "status.json"
        metrics = directory / "evaluation/test_metrics.json"
        if (
            status.exists()
            and json.loads(status.read_text()).get("evaluation_complete")
            and metrics.exists()
        ):
            measured[variant] = json.loads(metrics.read_text())
    metrics = [
        "accuracy",
        "mean_reward",
        "mean_training_reward",
        "retrieval_success_rate",
        "valid_tool_call_rate",
        "episodes_with_valid_calls_rate",
        "episodes_all_calls_valid_rate",
        "invalid_tool_call_rate",
        "average_tool_calls",
        "average_tool_calls_correct",
        "submission_rate",
    ]
    hops = sorted(
        {h for result in measured.values() for h in result.get("accuracy_by_hop", {})}, key=int
    )
    rows = [
        {
            "metric": f"{hop}-hop test accuracy",
            **{
                LABELS[v]: measured.get(v, {}).get("accuracy_by_hop", {}).get(hop) for v in VARIANTS
            },
        }
        for hop in hops
    ]
    rows += [
        {"metric": metric, **{LABELS[v]: measured.get(v, {}).get(metric) for v in VARIANTS}}
        for metric in metrics
    ]
    save_json(run_root / "comparison.json", rows)
    return rows


def evaluate_variant(
    model,
    tokenizer,
    store,
    task_ids,
    limits,
    run_root,
    variant,
    show_progress=True,
    training_result=None,
):
    """Save metrics, per-question rewards and pretty traces immediately."""
    from .rollouts import evaluate_policy

    if variant not in VARIANTS:
        raise ValueError("Unknown policy variant")
    directory = Path(run_root) / "variants" / variant
    status = {
        "variant": variant,
        "evaluation_split": "test",
        "evaluation_task_ids": list(task_ids),
        "training_result": training_result,
        "evaluation_complete": False,
        "status": "evaluating",
    }
    save_json(directory / "status.json", status)
    save_json(directory / "summary.json", status)
    (directory / "evaluation/test_metrics.json").unlink(missing_ok=True)
    try:
        metrics, traces = evaluate_policy(
            model,
            tokenizer,
            store,
            task_ids,
            limits,
            directory / "evaluation",
            "test",
            show_progress=show_progress,
        )
        if [trace["task_id"] for trace in traces] != list(task_ids):
            raise RuntimeError("Variant evaluation did not return the requested ordered test IDs")
        save_trajectory_html(
            directory / "evaluation/trajectories.html", traces, LABELS[variant], tokenizer
        )
        save_json(
            directory / "summary.json",
            {**status, "metrics": metrics, "evaluation_complete": True, "status": "complete"},
        )
        save_json(
            directory / "status.json", {**status, "evaluation_complete": True, "status": "complete"}
        )
    except Exception as error:
        save_json(
            directory / "status.json",
            {**status, "status": "evaluation_failed", "error": repr(error)},
        )
        save_json(
            directory / "summary.json",
            {**status, "status": "evaluation_failed", "error": repr(error)},
        )
        # The evaluator flushes completed episodes before starting the next one.
        partial = read_rows(directory / "evaluation/test_trajectories.jsonl")
        if partial:
            save_trajectory_html(
                directory / "evaluation/partial_trajectories.html",
                partial,
                LABELS[variant] + " · incomplete evaluation",
                tokenizer,
            )
        raise
    finally:
        save_comparison(run_root)
    return metrics, traces


def collect_training_logs(variant_dir, phase_dirs):
    """Copy diagnostics, never optimizer checkpoints, into the policy directory."""
    variant_dir = Path(variant_dir)
    for source in map(Path, phase_dirs):
        if not source.exists():
            continue
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if (
                not path.is_file()
                or path.suffix not in {".json", ".jsonl"}
                or any(part.startswith("checkpoint-") for part in relative.parts)
            ):
                continue
            target = variant_dir / "training" / source.name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def plot_measured(ax, rows, column, title, ylabel):
    groups = {}
    for row in rows:
        if row.get(column) is not None:
            groups.setdefault(row.get("phase", "training"), []).append(row)
    if not groups:
        ax.text(0.5, 0.5, "Not measured", transform=ax.transAxes, ha="center")
    for phase, group in groups.items():
        ax.plot([r["step"] for r in group], [r[column] for r in group], marker=".", label=phase)
    if groups:
        ax.legend()
    ax.set(title=title, xlabel="Optimizer step (within phase)", ylabel=ylabel)


def persist_rl_diagnostics(variant_dir, phase_dirs, tokenizer=None):
    """Called after EACH phase, including failure, rather than at notebook end."""
    variant_dir = Path(variant_dir)
    collect_training_logs(variant_dir, phase_dirs)
    phases = [variant_dir / "training" / Path(p).name for p in phase_dirs]
    collected = {}
    for filename in [
        "rollout_metrics",
        "trainer_metrics",
        "step_metrics",
        "evaluation_metrics",
        "evaluation_rewards",
    ]:
        collected[filename] = [
            {**row, "phase": phase.name, "variant": variant_dir.name}
            for phase in phases
            for row in read_rows(phase / (filename + ".jsonl"))
        ]
        if collected[filename]:
            save_rows(variant_dir / (filename + ".jsonl"), collected[filename])
    traces = [
        {**row, "phase": phase.name, "variant": variant_dir.name}
        for phase in phases
        for row in read_rows(phase / "trajectories.jsonl")
    ]
    if traces:
        save_rows(variant_dir / "training_trajectories.jsonl", traces)
        save_trajectory_html(
            variant_dir / "training_trajectories.html",
            traces,
            variant_dir.name + " · sampled training",
            tokenizer,
        )
        save_rows(
            variant_dir / "episodic_returns.jsonl",
            [
                {
                    "episode_index": index,
                    "variant": variant_dir.name,
                    **{
                        k: trace.get(k)
                        for k in (
                            "phase",
                            "step",
                            "group_id",
                            "task_id",
                            "hop_count",
                            "reward",
                            "answer_correct",
                            "retrieval_reward",
                            "training_reward",
                            "tool_calls",
                            "invalid_tool_calls",
                            "termination_reason",
                        )
                    },
                    "episodic_return": trace.get("training_reward", trace["reward"]),
                }
                for index, trace in enumerate(traces, 1)
            ],
        )
    rollout, trainer, steps, evaluation = (
        collected[k]
        for k in ["rollout_metrics", "trainer_metrics", "step_metrics", "evaluation_metrics"]
    )
    if not any([rollout, trainer, steps, evaluation]):
        return collected
    import matplotlib.pyplot as plt

    figures = variant_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, column, title in zip(
        axes.flat,
        [
            "mean_episode_reward",
            "zero_reward_group_fraction",
            "zero_variance_group_fraction",
            "average_tool_calls",
            "invalid_tool_call_rate",
            "submission_rate",
        ],
        [
            "Mean episodic return (answer + retrieval)",
            "All-zero reward groups",
            "Zero-variance comparison groups",
            "Tool calls per episode",
            "Invalid tool-call fraction",
            "Answer submission fraction",
        ],
    ):
        plot_measured(
            ax,
            rollout,
            column,
            title,
            "Calls" if column == "average_tool_calls" else "Fraction / mean reward",
        )
    fig.savefig(figures / "rollout_metrics.png", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, rows, column, title, ylabel in [
        (axes[0, 0], trainer, "loss", "Actual GRPO loss", "Loss"),
        (axes[0, 1], trainer, "grad_norm", "Gradient norm", "Norm"),
        (axes[0, 2], steps, "step_seconds", "Measured optimizer step duration", "Seconds"),
        (axes[1, 0], steps, "allocated_gib", "CUDA allocated memory", "GiB"),
        (axes[1, 1], steps, "peak_allocated_gib", "CUDA peak allocated within phase", "GiB"),
        (
            axes[1, 2],
            rollout,
            "episodes_all_calls_valid_rate",
            "Episodes with only valid calls",
            "Fraction",
        ),
    ]:
        plot_measured(ax, rows, column, title, ylabel)
    fig.savefig(figures / "optimizer_metrics.png", bbox_inches="tight")
    plt.close(fig)
    if evaluation:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        plot_measured(
            ax,
            evaluation,
            "mean_reward",
            "Measured test reward during training",
            "Mean terminal reward",
        )
        ax.set_ylim(-0.05, 1.05)
        fig.savefig(figures / "evaluation_rewards.png", bbox_inches="tight")
        plt.close(fig)
    return collected


def zip_run_artifacts(run_root, output):
    """Package already saved, including partial, results; no model work here."""
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if run_root == output or run_root in output.parents:
        raise ValueError("ZIP output must be outside the run directory")
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_root.rglob("*")):
            relative = path.relative_to(run_root)
            if path.is_file() and not any(
                p.startswith("checkpoint-") or p in {"checkpoints", "hf_cache"}
                for p in relative.parts
            ):
                archive.write(path, Path(run_root.name) / relative)
    return output
