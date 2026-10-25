"""Progress integration using scripted unit fixtures, never model results."""

import io
import itertools
import json
from contextlib import redirect_stdout
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tqdm")

from kg_rlvr.training.environment import EpisodeLimits, TaskStore
from kg_rlvr.training import rollouts


@pytest.fixture
def progress_fixture(graph, questions, monkeypatch):
    q = questions[0]
    store = TaskStore(graph, [q])
    path = q["gold_entity_paths"][0]
    actions = [("get_neighbors", {"entity_id": h, "relation_id": r})
               for h, r in zip(path, q["relation_path"])]
    actions.append(("submit_answer", {"entity_id": path[-1]}))
    texts = {i: '<tool_call>' + json.dumps({"name": name, "arguments": args}) + '</tool_call>'
             for i, (name, args) in enumerate(actions, 1)}

    class Codec:
        tokenizer = SimpleNamespace(eos_token_id=0)
        def prompt_ids(self, prompt):
            return [100]
        def feedback_ids(self, name, observation):
            return [200, 201]
        def decode(self, ids):
            return texts[ids[0]]

    class Model:
        training = True
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.empty(1))
        def eval(self):
            self.training = False
        def train(self, mode=True):
            self.training = mode

    def sampler(*args, **kwargs):
        sequence = itertools.cycle(texts)
        return lambda context, limit: [next(sequence), 0]

    monkeypatch.setattr(rollouts, "QwenChatCodec", lambda tokenizer: Codec())
    monkeypatch.setattr(rollouts, "transformer_sampler", sampler)
    # Exercise the real tqdm counters/text backend, independent of widgets.
    from tqdm.std import tqdm
    bars = []
    def recording_bar(**kwargs):
        stream = io.StringIO()
        bar = tqdm(**kwargs, file=stream, mininterval=0)
        bars.append((bar, stream))
        return bar
    monkeypatch.setattr("tqdm.auto.tqdm", recording_bar)
    return store, q, Model(), bars


def test_evaluation_progress_counts_finished_questions_without_changing_results(progress_fixture, tmp_path):
    store, q, model, bars = progress_fixture
    state = torch.random.get_rng_state().clone()
    visible, traces = rollouts.evaluate_policy(model, None, store, [q["id"]] * 2,
        EpisodeLimits(), tmp_path / "visible", "fixture_validation", show_progress=True)
    bar, stream = bars[-1]
    assert bar.n == bar.total == 2
    assert "100%" in stream.getvalue() and q["id"] in stream.getvalue()
    assert "accuracy=100.0%" in stream.getvalue()
    quiet, quiet_traces = rollouts.evaluate_policy(model, None, store, [q["id"]] * 2,
        EpisodeLimits(), tmp_path / "quiet", "fixture_validation", show_progress=False)
    assert bars[-1][1].getvalue() == ""
    assert visible == quiet
    assert [t["tool_events"] for t in traces] == [t["tool_events"] for t in quiet_traces]
    assert [t["model_tokens"] for t in traces] == [t["model_tokens"] for t in quiet_traces]
    assert torch.equal(state, torch.random.get_rng_state()) and model.training


def test_grpo_progress_counts_trajectories_in_the_actual_group(progress_fixture, tmp_path):
    store, q, model, bars = progress_fixture
    trainer = SimpleNamespace(model=model, num_generations=4, temperature=0.8,
        state=SimpleNamespace(global_step=2),
        accelerator=SimpleNamespace(num_processes=1, unwrap_model=lambda model: model))
    rollout = rollouts.GRPORollout(store, None, EpisodeLimits(), tmp_path, "fixture")
    output = rollout([store.prompt(q["id"])] * 4, trainer)
    bar, _ = bars[-1]
    assert bar.n == bar.total == 4 and bar.unit == "trajectory"
    assert bar.desc == "fixture step 3: sampling"
    assert len(output["completion_ids"]) == 4 and len(output["env_mask"]) == 4
    assert output["task_ids"] == [q["id"]] * 4
    metrics = json.loads((tmp_path / "rollout_metrics.jsonl").read_text().splitlines()[-1])
    assert metrics["mean_episode_reward"] == metrics["mean_training_reward"] == 1.2
    assert metrics["accuracy"] == 1.0 and metrics["retrieval_success_rate"] == 1.0
    assert metrics["zero_variance_group_fraction"] == 1.0  # homogeneous at 1.2, not binary 1
    assert len(output["tool_events"]) == 4
    assert model.training


def test_generation_failure_closes_progress_without_claiming_completion(progress_fixture, tmp_path, monkeypatch):
    store, q, model, bars = progress_fixture
    def fail(context, limit):
        raise RuntimeError("unit generation failure")
    monkeypatch.setattr(rollouts, "transformer_sampler", lambda *args, **kwargs: fail)
    with pytest.raises(RuntimeError, match="unit generation failure"):
        rollouts.evaluate_policy(model, None, store, [q["id"]] * 2,
            EpisodeLimits(), tmp_path, "fixture_failure")
    bar, stream = bars[-1]
    assert bar.n == 0 and bar.disable  # real tqdm closes its display on exception
    assert "100%" not in stream.getvalue() and model.training


@pytest.mark.parametrize("correct", [True, False])
def test_evaluation_records_actual_terminal_rewards_and_training_step(progress_fixture, tmp_path, monkeypatch, correct):
    store, q, model, bars = progress_fixture
    if not correct:
        codec = rollouts.QwenChatCodec(None)
        original_decode = codec.decode
        # A deliberately wrong but valid submission in a scripted unit fixture.
        codec.decode = lambda ids: original_decode(ids).replace(q["gold_entity_paths"][0][-1], q["start_entity"])
        monkeypatch.setattr(rollouts, "QwenChatCodec", lambda tokenizer: codec)
    summary, traces = rollouts.evaluate_policy(model, None, store, [q["id"]] * 2,
        EpisodeLimits(), tmp_path, "step_7", training_step=7)
    rows = [json.loads(line) for line in (tmp_path / "step_7_rewards.jsonl").read_text().splitlines()]
    expected = float(correct)
    assert [r["reward"] for r in rows] == [t["reward"] for t in traces] == [expected, expected]
    assert [r["evaluation_index"] for r in rows] == [1, 2]
    assert all(r["training_step"] == 7 and r["task_id"] == q["id"] for r in rows)
    assert summary["mean_reward"] == summary["accuracy"] == expected
    assert summary["total_reward"] == 2 * expected
    assert summary["mean_reward_by_hop"] == {str(q["hop_count"]): expected}
    assert all(r["answer_correct"] == expected and r["gold_entity_retrieved"] for r in rows)
    assert all(r["retrieval_reward"] == 0.2 and r["training_reward"] == expected + 0.2 for r in rows)
    assert f"reward={int(expected)}" in bars[-1][1].getvalue()
    assert f"mean_reward={expected:.3f}" in bars[-1][1].getvalue()


@pytest.mark.parametrize("interval", [0, 2])
def test_periodic_callback_logs_only_scheduled_evaluations(progress_fixture, tmp_path, interval, monkeypatch):
    from kg_rlvr.training.grpo import make_metrics_callback
    store, q, model, _ = progress_fixture
    # HF lazily subclasses tqdm during import; import with the actual class,
    # then restore the recording counter used for this unit fixture's episodes.
    from tqdm.std import tqdm
    with monkeypatch.context() as import_patch:
        import_patch.setattr("tqdm.auto.tqdm", tqdm)
        callback = make_metrics_callback(tmp_path / "smoke", store, None, [q["id"]] * 2,
                                         EpisodeLimits(), eval_every=interval, show_progress=False)
    state_before = torch.random.get_rng_state().clone()
    stream = io.StringIO()
    with redirect_stdout(stream):
        for step in range(1, 5):
            callback.started = time.perf_counter()
            callback.on_step_end(None, SimpleNamespace(global_step=step), None, model=model)
    summary_path = tmp_path / "smoke/evaluation_metrics.jsonl"
    if not interval:
        assert not summary_path.exists() and "evaluation |" not in stream.getvalue()
    else:
        summaries = [json.loads(line) for line in summary_path.read_text().splitlines()]
        assert [s["step"] for s in summaries] == [2, 4]
        assert all(s["mean_reward"] == 1 and s["total_reward"] == 2 for s in summaries)
        rewards = [json.loads(line) for line in (tmp_path / "smoke/evaluation_rewards.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rewards] == [2, 2, 4, 4]
        assert all(r["phase"] == "smoke" and r["training_step"] == r["step"] for r in rewards)
        assert "optimizer step 2" in stream.getvalue() and "mean_reward=1.0000" in stream.getvalue()
        assert len(list((tmp_path / "smoke/periodic_evaluation").glob("*_rewards.jsonl"))) == 2
    assert model.training and torch.equal(state_before, torch.random.get_rng_state())


def test_partial_evaluation_retains_completed_rewards_without_claiming_full_metrics(progress_fixture, tmp_path, monkeypatch):
    store, q, model, _ = progress_fixture
    original_sampler = rollouts.transformer_sampler(model, None)
    calls = 0
    def fail_in_second_episode(context, limit):
        nonlocal calls
        if calls == len(q["relation_path"]) + 1:
            raise RuntimeError("unit failure after one completed evaluation episode")
        calls += 1
        return original_sampler(context, limit)
    monkeypatch.setattr(rollouts, "transformer_sampler", lambda *args, **kwargs: fail_in_second_episode)
    with pytest.raises(RuntimeError, match="after one completed"):
        rollouts.evaluate_policy(model, None, store, [q["id"]] * 2, EpisodeLimits(),
                                 tmp_path, "partial", training_step=3)
    rows = [json.loads(line) for line in (tmp_path / "partial_rewards.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["reward"] == 1 and rows[0]["training_step"] == 3
    assert not (tmp_path / "partial_metrics.json").exists()
    saved_traces = [json.loads(line) for line in (tmp_path / "partial_trajectories.jsonl").read_text().splitlines()]
    assert len(saved_traces) == 1 and saved_traces[0]["task_id"] == q["id"]
    assert model.training
