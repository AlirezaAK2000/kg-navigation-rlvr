from collections import Counter, defaultdict
import copy
import json
import pytest

from kg_rlvr.config import allocate, load_config
from kg_rlvr.conversion import conversational_row, PUBLIC_INSTRUCTIONS
from kg_rlvr.dataset_builder import GenerationError, build_dataset, read_questions, split_candidates
from kg_rlvr.graph_index import execute
from kg_rlvr.query_generator import generate_candidates
from kg_rlvr.question_templates import PROGRAMS
from kg_rlvr.validators import validate_dataset, validate_saved_dataset


def test_all_hops_templates_and_complete_answers(graph, questions):
    assert {q["hop_count"] for q in questions} == {2, 3, 4}
    assert Counter(q["split"] for q in questions) == {"train": 42, "validation": 6, "test": 12}
    for q in questions:
        assert execute(graph, q["start_entity"], q["relation_path"]).answers == q["answer_ids"]
    assert all(len(p.templates) == 3 for p in PROGRAMS)


def test_validation_passes(graph, questions, config):
    result = validate_dataset(graph, questions, config)
    assert result["valid"], result["errors"]
    assert result["pass_percentage"] == 100


def test_paraphrases_grouped_and_split_reproducible(graph, candidates, questions, config):
    groups = defaultdict(set)
    for q in questions:
        groups[q["query_instance_id"]].add(q["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert len(groups) < len(questions)
    assert split_candidates(graph, candidates, config)[0] == questions


def test_unseen_composition_split(graph, candidates, config):
    config = copy.deepcopy(config)
    config["questions"]["composition_test_count"] = 12
    questions, info = split_candidates(graph, candidates, config)
    ordinary = {q["program_id"] for q in questions if q["split"] != "composition_test"}
    unseen = {q["program_id"] for q in questions if q["split"] == "composition_test"}
    assert unseen and ordinary.isdisjoint(unseen)
    assert unseen.issubset(info["heldout_program_ids"])
    assert validate_dataset(graph, questions, config)["valid"]


def test_start_entity_split(graph, candidates, config):
    config = copy.deepcopy(config)
    config["questions"].update(split_strategy="start_entity", split_counts={"train": 30, "validation": 12, "test": 18})
    questions, _ = split_candidates(graph, candidates, config)
    owners = defaultdict(set)
    for q in questions:
        owners[q["start_entity"]].add(q["split"])
    assert all(len(splits) == 1 for splits in owners.values())
    assert validate_dataset(graph, questions, config)["valid"]


@pytest.mark.parametrize("field,value", [
    ("answer_ids", ["missing"]), ("answer_ids", []), ("hop_count", 99),
    ("question", "An unrelated question?"), ("start_entity", "missing"),
    ("relation_path", ["led_by", "city_in_country"]),
    ("gold_entity_paths", [["missing"]]), ("difficulty", {}),
    ("shortest_equivalent_program_length", 1), ("query_instance_id", "bad"),
    ("answer_path_counts", {}), ("shortest_path_hops", {}),
])
def test_validation_rejects_corrupted_metadata(graph, questions, field, value):
    changed = copy.deepcopy(questions[:1])
    changed[0][field] = value
    assert not validate_dataset(graph, changed)["valid"]


def test_incomplete_multi_answer_set(graph, questions):
    question = next(q for q in questions if len(q["answer_ids"]) > 1)
    changed = copy.deepcopy(question)
    changed["answer_ids"] = changed["answer_ids"][:1]
    result = validate_dataset(graph, [changed])
    assert any("incomplete" in e["error"] for e in result["errors"])


def test_cross_split_paraphrase_leakage(graph, questions):
    first = copy.deepcopy(questions[0])
    second = copy.deepcopy(next(q for q in questions if q["query_instance_id"] == first["query_instance_id"] and q["id"] != first["id"]))
    second["split"] = "test"
    assert not validate_dataset(graph, [first, second])["valid"]


def test_candidate_filters_do_not_relax(graph, config):
    config = copy.deepcopy(config)
    config["questions"]["max_tool_calls"] = 1
    candidates, rejected = generate_candidates(graph, config)
    assert not candidates
    assert any("tool_budget_exceeded" in key for key in rejected)
    with pytest.raises(GenerationError, match="No filters were relaxed"):
        split_candidates(graph, candidates, config)


def test_prompt_export_allowlist(graph, questions):
    for q in questions:
        row = conversational_row(graph, q)
        assert set(row) == {"prompt"}
        assert row["prompt"] == [
            {"role": "system", "content": PUBLIC_INSTRUCTIONS},
            {"role": "user", "content": f"Question: {q['question']}\nStarting entity: {q['start_entity']} ({graph.entities[q['start_entity']]['name']}; {graph.entities[q['start_entity']]['type']})"},
        ]
        assert not any(key in json.dumps(row) for key in ("answer_ids", "gold_entity_paths", "relation_path", "program_id"))


def test_saved_dataset_reproducibility_and_manifest(tmp_path, config):
    output = tmp_path / "dataset"
    _, questions, stats = build_dataset(config, output, progress=lambda _: None)
    assert len(read_questions(output)) == len(questions)
    result = validate_saved_dataset(output, check_reproducibility=True)
    assert result["valid"] and result["reproducible"]
    assert stats["duplicates"]["cross_split_instance_groups"] == 0
    assert stats["questions"]["counts_by_split_and_hop"]["train"] == {2: 14, 3: 14, 4: 14}
    with pytest.raises(GenerationError, match="not empty"):
        build_dataset(config, output, progress=lambda _: None)
    catalog = output / "metadata" / "query_programs.jsonl"
    catalog.write_text("{}\n")
    assert not validate_saved_dataset(output)["valid"]


def test_configuration_and_quotas(tmp_path):
    assert allocate(7, {"2": 1, "3": 1, "4": 1}) == {"2": 3, "3": 2, "4": 2}
    path = tmp_path / "config.json"
    path.write_text('{"questions":{"hop_weights":{"2":1,"3":0,"4":1}}}')
    with pytest.raises(ValueError, match="positive weights"):
        load_config(path)
    path.write_text('{"typo": 1}')
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path)
