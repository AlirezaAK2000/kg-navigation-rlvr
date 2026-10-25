"""Exercise the measured-rollout plotting branch with explicit unit fixtures."""

import json
from pathlib import Path
import shutil

import pytest


def test_notebook_return_and_behavior_panels_use_matching_series(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["mean_episode_reward", "zero_reward_group_fraction", "zero_variance_group_fraction",
               "average_tool_calls", "invalid_tool_call_rate", "submission_rate"]
    values = [1.2, 0.1, 0.3, 5.0, 0.05, 0.8]
    row = {"phase": "full", "step": 2, "accuracy": 1.0,
           "retrieval_success_rate": 1.0, "mean_retrieval_reward": 0.2, **dict(zip(columns, values))}
    checkpoint = tmp_path / "checkpoints/full"
    checkpoint.mkdir(parents=True)
    (checkpoint / "rollout_metrics.jsonl").write_text(json.dumps(row) + "\n")
    run = tmp_path / "run"
    run.mkdir()
    plotted = []
    original_save = plt.Figure.savefig
    def capture_plot(figure, *args, **kwargs):
        plotted.append([(ax.get_title(), [list(line.get_ydata()) for line in ax.lines])
                        for ax in figure.axes])
        return original_save(figure, *args, **kwargs)
    monkeypatch.setattr(plt.Figure, "savefig", capture_plot)
    from kg_rlvr.training.artifacts import persist_rl_diagnostics
    persist_rl_diagnostics(run, [checkpoint])
    titles = ["Mean episodic return (answer + retrieval)", "All-zero reward groups",
              "Zero-variance comparison groups", "Tool calls per episode",
              "Invalid tool-call fraction", "Answer submission fraction"]
    assert plotted[0] == [(title, [[value]]) for title, value in zip(titles, values)]
    assert (run / "figures/rollout_metrics.png").is_file()
