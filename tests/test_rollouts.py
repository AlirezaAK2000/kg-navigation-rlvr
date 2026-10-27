import itertools
import json
from types import SimpleNamespace

import pytest

from kg_rlvr.training.environment import EpisodeLimits
from kg_rlvr.training import rollouts


@pytest.fixture
def episode_fixture(tasks, monkeypatch):
    graph, store = tasks
    q = store.record(store.select("train", [2], count=1)[0])
    path = q["gold_entity_paths"][0]
    actions = [
        ("get_neighbors", {"entity_id": h, "relation_id": r})
        for h, r in zip(path, q["relation_path"])
    ]
    actions += [("submit_answer", {"entity_id": path[-1]})]
    texts = {
        i: "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"
        for i, (name, args) in enumerate(actions, 1)
    }

    class Codec:
        tokenizer = SimpleNamespace(eos_token_id=0)

        def prompt_ids(self, prompt):
            return [20]

        def feedback_ids(self, name, observation):
            return [30, 31]

        def decode(self, ids):
            return texts[ids[0]]

    codec = Codec()
    monkeypatch.setattr(rollouts, "QwenChatCodec", lambda tokenizer: codec)
    monkeypatch.setattr(
        rollouts,
        "transformer_sampler",
        lambda *a, **kw: lambda context, limit, script=itertools.cycle(texts): [next(script), 0],
    )
    return store, q, codec


def test_episode_masks_observations_and_preserves_sampled_action_ids(episode_fixture):
    store, q, codec = episode_fixture
    script = iter(range(1, len(q["relation_path"]) + 2))
    ledger, trace = rollouts.run_episode(
        store, q["id"], codec, lambda context, limit: [next(script), 0], EpisodeLimits()
    )
    assert trace["reward"] == 1 and trace["training_reward"] == 1.2
    assert ledger.env_mask.count(0) == 4 and ledger.env_mask.count(1) == 6
    assert len(ledger.env_mask) == len(ledger.completion_ids)
    for segment in ledger.segments:
        mask = ledger.env_mask[segment["begin"] : segment["end"]]
        assert set(mask) == ({1} if segment["source"] == "model" else {0})


def test_partial_evaluation_persists_complete_episodes(episode_fixture, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    store, q, codec = episode_fixture

    class Model:
        training = True

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.empty(1))

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

    script = iter(range(1, len(q["relation_path"]) + 2))

    def sample(context, limit):
        try:
            return [next(script), 0]
        except StopIteration:
            raise RuntimeError("unit interruption")

    monkeypatch.setattr(rollouts, "transformer_sampler", lambda *a, **kw: sample)
    with pytest.raises(RuntimeError, match="unit interruption"):
        rollouts.evaluate_policy(
            Model(), None, store, [q["id"]] * 2, EpisodeLimits(), tmp_path, "test", False
        )
    assert len((tmp_path / "test_trajectories.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "test_rewards.jsonl").read_text().splitlines()) == 1
    assert not (tmp_path / "test_metrics.json").exists()
