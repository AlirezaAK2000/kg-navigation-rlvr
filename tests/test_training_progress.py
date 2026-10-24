"""Progress integration using scripted unit fixtures, never model results."""

import io
import itertools
import json
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
