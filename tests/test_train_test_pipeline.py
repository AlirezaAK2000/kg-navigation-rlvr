"""Train/test migration and the three-stage notebook evaluation contract."""

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kg_rlvr.config import load_config
from kg_rlvr.dataset_builder import read_questions


def notebook_cells():
    notebook = json.loads(Path("notebooks/02_grpo_training_kaggle.ipynb").read_text())
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"]}


@pytest.mark.parametrize("directory,counts", [
    ("data", {"train": 2667, "test": 1067}),
    ("pilot_data", {"train": 94, "test": 41}),
])
def test_split_migration_keeps_train_bytes_and_moves_every_validation_id(directory, counts):
    root = Path(directory)
    records = read_questions(root)
    assert Counter(q["split"] for q in records) == counts
    assert {p.stem for p in (root / "rl").glob("*.jsonl")} == {"train", "test"}
    provenance = json.loads((root / "metadata/split_migration.json").read_text())
    assert hashlib.sha256((root / "rl/train.jsonl").read_bytes()).hexdigest() == provenance["original_files_sha256"]["rl/train.jsonl"]
    by_id = {q["id"]: q for q in records}
    assert len(by_id) == sum(counts.values())
    assert all(by_id[q]["split"] == "test" for q in provenance["moved_validation_ids"])
    assert load_config(root / "config.json")["questions"]["split_counts"] == counts


def test_configuration_rejects_a_third_split(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"questions": {"split_counts": {"train": 10, "validation": 2, "test": 4}}}))
    with pytest.raises(ValueError, match="only train and test"):
        load_config(config)


def test_all_three_policies_evaluate_the_same_ordered_test_ids(tmp_path, monkeypatch):
    # Inference is stubbed to exercise notebook routing, adapter state and export
    # columns. These fixtures are not model-performance measurements.
    pd = pytest.importorskip("pandas")
    from kg_rlvr.training import rollouts

    class Model:
        adapter_enabled = True

        @contextmanager
        def disable_adapter(self):
            self.adapter_enabled = False
            try:
                yield
            finally:
                self.adapter_enabled = True

    model = Model()
    calls = []

    def evaluate(policy, tokenizer, store, ids, limits, directory, name, **kwargs):
        calls.append((name, list(ids), policy.adapter_enabled))
        return {"accuracy": 0.5, "accuracy_by_hop": {"1": 0.5}, "termination_reasons": {"submitted": 2}}, [
            {"tool_events": [{"name": "get_neighbors", "observation": {"ok": True}}]}]

    monkeypatch.setattr(rollouts, "evaluate_policy", evaluate)
    cells = notebook_cells()
    namespace = {"RUN_MODEL": True, "model": model, "tokenizer": None, "store": object(),
        "test_ids": ["test_b", "test_a"], "LIMITS": object(), "EVALUATION_DIR": tmp_path,
        "SHOW_PROGRESS": False, "pd": pd, "display": lambda *a: None, "MEMORY": [],
        "memory_snapshot": lambda *a: {}, "SFT_RESULT": {"verified": True},
        "HTML": lambda value: value, "render_traces_html": lambda *a, **kw: "fixture",
        "INSPECT_EXAMPLES": 2, "INSPECT_OFFSET": 0, "SMOKE_RESULT": {"verified": True},
        "final_trainer": SimpleNamespace(model=model, accelerator=SimpleNamespace(unwrap_model=lambda m: m)),
        "EVAL_HOPS": [1]}
    for cell in ("grpo-013", "sft-evaluation", "grpo-023"):
        exec(cells[cell], namespace)
    assert calls == [("vanilla_test", ["test_b", "test_a"], False),
                     ("sft_test", ["test_b", "test_a"], True),
                     ("sft_grpo_test", ["test_b", "test_a"], True)]
    assert list(namespace["comparison"].columns) == ["metric", "vanilla", "SFT", "SFT + GRPO"]
    assert (tmp_path / "vanilla_sft_sft_grpo_test_comparison.json").is_file()


@pytest.mark.parametrize("cell,switch", [("grpo-017", "RUN_SMOKE_TEST"), ("grpo-019", "RUN_FULL_TRAINING")])
def test_grpo_cannot_bypass_verified_sft(cell, switch):
    with pytest.raises(RuntimeError, match="SFT stage"):
        exec(notebook_cells()[cell], {"RUN_MODEL": True, switch: True, "SFT_RESULT": None})


def test_notebook_has_one_test_selection_and_no_grpo_only_switch():
    cells = notebook_cells()
    assert "RUN_SFT" not in cells["grpo-002"] and "RUN_BASELINE" not in cells["grpo-002"]
    assert 'store.select("test"' in cells["grpo-007"]
    assert 'store.select("validation"' not in "\n".join(cells.values())
    assert '"sft": sft_ids, "grpo": train_ids, "test": test_ids' in cells["grpo-007"]
    for cell in ("grpo-017", "grpo-019"):
        assert "eval_task_ids=test_ids[:PERIODIC_EVAL_SIZE]" in cells[cell]
