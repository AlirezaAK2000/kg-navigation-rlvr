"""One-hop generation, preservation, split isolation and training integration."""

from collections import Counter, defaultdict
import copy
import json

import pytest

from kg_rlvr.config import load_config
from kg_rlvr.dataset_builder import build_dataset, split_candidates
from kg_rlvr.graph_index import execute
from kg_rlvr.ontology import RELATIONS
from kg_rlvr.query_generator import generate_candidates
from kg_rlvr.question_templates import ONE_HOP_PROGRAMS
from kg_rlvr.training.environment import TaskStore
from kg_rlvr.validators import validate_dataset, validate_saved_dataset


@pytest.fixture(scope="module")
def expanded(graph, config):
    settings = copy.deepcopy(config)
    settings["questions"].update(split_counts={"train": 56, "validation": 8, "test": 16},
                                  hop_weights={"1": 1, "2": 1, "3": 1, "4": 1})
    candidates, _ = generate_candidates(graph, settings)
    questions, _ = split_candidates(graph, candidates, settings)
    return settings, candidates, questions


def test_one_hop_catalog_covers_forward_and_inverse_relations():
    assert {p.relation_path[0] for p in ONE_HOP_PROGRAMS if not p.relation_path[0].startswith("inv:")} == set(RELATIONS)
    assert any(p.relation_path[0].startswith("inv:") for p in ONE_HOP_PROGRAMS)
    assert all(len(p.relation_path) == 1 and len(p.templates) == 3 for p in ONE_HOP_PROGRAMS)


def test_adding_one_hop_preserves_existing_rows_and_split_quotas(graph, questions, expanded):
    config, _, mixed = expanded
    assert mixed[:len(questions)] == questions  # includes IDs, wording and all metadata
    assert Counter(q["hop_count"] for q in mixed) == {1: 20, 2: 20, 3: 20, 4: 20}
    assert Counter(q["split"] for q in mixed if q["hop_count"] == 1) == {"train": 14, "validation": 2, "test": 4}
    assert validate_dataset(graph, mixed, config)["valid"]
    grouped = defaultdict(set)
    for q in mixed:
        grouped[q["query_instance_id"]].add(q["split"])
        if q["hop_count"] == 1:
            result = execute(graph, q["start_entity"], q["relation_path"])
            assert result.answers == q["answer_ids"]
            assert all(len(witness) == 2 for witness in q["gold_entity_paths"])
            assert set(q["shortest_path_hops"].values()) == {1}
    assert all(len(splits) == 1 for splits in grouped.values())


def test_one_hop_paraphrases_cannot_cross_splits(graph, expanded):
    _, _, mixed = expanded
    one = next(q for q in mixed if q["hop_count"] == 1)
    variant = copy.deepcopy(next(q for q in mixed if q["query_instance_id"] == one["query_instance_id"] and q["id"] != one["id"]))
    variant["split"] = "test" if one["split"] != "test" else "train"
    result = validate_dataset(graph, [one, variant])
    assert any("leak across splits" in error["error"] for error in result["errors"])


def test_one_hop_only_config_and_candidate_selection(tmp_path, graph, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"questions": {"hop_weights": {"1": 1},
                               "split_counts": {"train": 12, "validation": 4, "test": 4}}}))
    settings = load_config(path)
    assert settings["questions"]["hop_weights"] == {"1": 1}
    settings["graph"] = copy.deepcopy(config["graph"])
    candidates, _ = generate_candidates(graph, settings)
    questions, _ = split_candidates(graph, candidates, settings)
    assert {q["hop_count"] for q in questions} == {1}
    assert validate_dataset(graph, questions, settings)["valid"]


def test_one_hop_export_and_fixed_seed_regeneration(tmp_path, expanded):
    settings, _, _ = expanded
    directory = tmp_path / "mixed"
    build_dataset(settings, directory, progress=lambda _: None)
    result = validate_saved_dataset(directory, check_reproducibility=True)
    assert result["valid"] and result["reproducible"]
    assert "| 1 hop |" in (directory / "metadata/report.md").read_text()


def test_mixed_training_and_smoke_selection(graph, expanded):
    _, _, questions = expanded
    store = TaskStore(graph, questions)
    ids = store.select("train", [1, 2, 3], count=8, seed=41, balanced=True)
    assert Counter(store.record(i)["hop_count"] for i in ids) == {1: 3, 2: 3, 3: 2}
    assert len({store.record(i)["query_instance_id"] for i in ids}) == len(ids)
    assert all(store.record(i)["split"] == "train" for i in ids)
    assert {store.record(i)["hop_count"] for i in store.select("train", [1])} == {1}


@pytest.mark.parametrize("strategy", ["start_entity", "composition_holdout"])
def test_one_hop_respects_optional_split_strategies(graph, expanded, strategy):
    settings, candidates, _ = expanded
    settings = copy.deepcopy(settings)
    if strategy == "start_entity":
        settings["questions"]["split_strategy"] = strategy
    else:
        settings["questions"]["composition_test_count"] = 12
    questions, _ = split_candidates(graph, candidates, settings)
    result = validate_dataset(graph, questions, settings)
    assert result["valid"], result["errors"]
    if strategy == "composition_holdout":
        assert any(q["hop_count"] == 1 and q["split"] == "composition_test" for q in questions)
