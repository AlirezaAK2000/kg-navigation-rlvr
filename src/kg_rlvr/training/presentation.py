"""Load the recorded experiment for the notebook's unchanged slide plots."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VARIANTS = {
    "vanilla": ("Vanilla", "#64748B"),
    "sft": ("SFT", "#3B82F6"),
    "sft_grpo": ("SFT + GRPO", "#0D9488"),
    "grpo_only": ("GRPO only", "#D97706"),
}
RL_VARIANTS = ["sft_grpo", "grpo_only"]
rng = np.random.default_rng(41)
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": "#334155",
    "text.color": "#0F172A",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


def read_jsonl(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(s) for s in path.read_text().splitlines() if s.strip())


def first_rows(paths):
    for path in paths:
        frame = read_jsonl(path)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def numeric(frame, *names):
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def prepare(frame):
    frame = frame.copy()
    frame["answer"] = numeric(frame, "answer_correct", "reward")
    frame["retrieval"] = numeric(frame, "retrieval_reward")  # Already weighted.
    frame["return"] = numeric(frame, "training_reward", "episodic_return")
    frame["return"] = frame["return"].fillna(frame["answer"] + frame["retrieval"])
    if "hop_count" in frame:
        frame["hop_count"] = pd.to_numeric(frame["hop_count"], errors="coerce")
    return frame


def return_ci(frame):
    # Resample entire questions, keeping their related trajectories together.
    data = frame.dropna(subset=["return"]).copy()
    if data.empty:
        return np.nan, np.nan, np.nan, 0
    key = "task_id" if "task_id" in data and data["task_id"].notna().all() else "group_id"
    if key not in data or data[key].isna().any():
        return data["return"].mean(), np.nan, np.nan, 0
    groups = data.groupby(key)["return"].agg(["sum", "count"])
    n = len(groups)
    mean = data["return"].mean()
    if n < 2:
        return mean, np.nan, np.nan, n
    sums, counts = groups["sum"].to_numpy(), groups["count"].to_numpy()
    indices = rng.integers(n, size=(2000, n))
    samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return mean, lo, hi, n


def finish_axis(ax):
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=8)


def export(fig, name, directory):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            Path(directory) / f"{name}.{extension}", dpi=300, bbox_inches="tight", facecolor="white"
        )
    plt.show()
    plt.close(fig)


def load_results(root, window_steps=5):
    root = Path(root)
    evaluations, training = {}, {}
    for variant in VARIANTS:
        folder = root / "variants" / variant
        if (folder / "evaluation/test_metrics.json").exists():
            frame = first_rows(
                [
                    folder / "evaluation/test_rewards.jsonl",
                    folder / "evaluation/test_trajectories.jsonl",
                ]
            )
            if not frame.empty:
                evaluations[variant] = prepare(frame)
        if variant not in RL_VARIANTS:
            continue
        frame = first_rows(
            [folder / "episodic_returns.jsonl", folder / "training_trajectories.jsonl"]
        )
        if frame.empty:
            continue
        if "phase" in frame:
            frame = frame[frame["phase"].fillna("").str.lower() != "smoke"].copy()
        frame = prepare(frame)
        frame["step"] = numeric(frame, "step")
        frame = frame.dropna(subset=["step", "return"]).copy()
        if not frame.empty:
            frame["window"] = ((frame["step"] - 1) // window_steps).astype(int)
            training[variant] = frame
    return evaluations, training
