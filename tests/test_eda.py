import copy
import pytest

pytest.importorskip("pandas")
pytest.importorskip("networkx")

from kg_rlvr.eda import (
    OPTIONAL, fingerprint, graph_views,
    navigation_audit, near_duplicate_screen, output_directory, question_frame,
    schema_audit, split_overlaps, stratified_samples,
)


def test_eda_reuses_graph_and_validates_metadata(graph, questions, config):
    audit, trace, full = navigation_audit(graph, questions, config=config)
    assert audit.core_valid.all()
    assert full["valid"]
    assert len(trace) > 0
    q = questions[0]
    group = trace[trace.instance_key == q["query_instance_id"]]
    assert group.navigation_branches.mean() == q["difficulty"]["branching_factor"]
    assert group.candidate_neighbors.mean() == q["difficulty"]["query_neighbors_mean"]
    assert group.distractor_branches.sum() == q["difficulty"]["distractor_count"]


def test_missing_optional_metadata_is_measured_not_invented(graph, questions, tmp_path):
    records = [{k: v for k, v in q.items() if k not in OPTIONAL} for q in questions]
    audit, _, full = navigation_audit(graph, records)
    assert audit.core_valid.all()
    assert not audit.difficulty_metadata_present.any()
    assert not full["available"]
    frame = question_frame(graph, records)
    overlaps, _ = split_overlaps(frame)
    assert overlaps.loc[overlaps.item == "query_instance", "shared"].sum() == 0
    assert "gold_entity_paths" not in records[0]  # no mutation of input dictionaries


def test_difficulty_corruption_is_detected(graph, questions):
    records = copy.deepcopy(questions[:1])
    records[0]["difficulty"]["branching_factor"] += 1
    audit, _, _ = navigation_audit(graph, records, run_full_validation=False)
    assert not audit.core_valid.iloc[0]
    assert audit.difficulty_mismatches.iloc[0] == 1


def test_sampling_and_near_duplicate_screen_are_deterministic(graph, questions):
    frame = question_frame(graph, questions)
    assert stratified_samples(frame).groupby("hop_count").size().to_dict() == {2: 5, 3: 5, 4: 5}
    first = near_duplicate_screen(frame, candidates_per_row=2)
    second = near_duplicate_screen(frame, candidates_per_row=2)
    assert first.equals(second)
    assert len(first) <= len(frame) * 2 * 2


def test_graph_views_count_stored_triples_and_permitted_navigation(graph):
    stored, simple, navigation, entities, triples, branch = graph_views(graph)
    assert stored.number_of_edges() == len(graph.triples) == len(triples)
    assert len(entities) == len(graph.entities)
    assert simple.number_of_edges() <= stored.number_of_edges()
    for row in branch.itertuples():
        assert row.neighbor_count == len(graph.get_neighbors(row.entity_id, row.operation))
    assert all(navigation.has_edge(t, h) for h, r, t in graph.triples if graph.relations[r]["inverse_supported"])


def test_output_guard_and_source_fingerprint(tmp_path):
    source = tmp_path / "data"
    source.mkdir()
    (source / "input.json").write_text('{}')
    before = fingerprint(source)
    output_directory(tmp_path / "outputs", source)
    assert fingerprint(source) == before
    with pytest.raises(ValueError, match="outside"):
        output_directory(source / "output", source)
    (source / "input.json").write_text('{"changed":true}')
    assert fingerprint(source) != before


def test_schema_audit_reports_invalid_references(graph, questions):
    records = copy.deepcopy(questions[:1])
    records[0]["answer_ids"] = ["missing"]
    result = schema_audit(graph, records).set_index("check")
    assert result.loc["Unknown answer references", "count"] == 1
