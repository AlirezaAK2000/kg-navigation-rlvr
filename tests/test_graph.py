import copy
import pytest

from kg_rlvr.graph_generator import generate_graph
from kg_rlvr.graph_index import GraphIndex, NavigationSession, execute
from kg_rlvr.ontology import RELATIONS, program_end_type
from kg_rlvr.validators import graph_errors


def test_generator_constraints_and_exact_size(graph, config):
    assert len(graph.entities) == 240
    assert len(graph.triples) == 800
    assert set(r for _, r, _ in graph.triples) == set(RELATIONS)
    assert graph_errors(graph, config) == []
    assert len(set(e["type"] for e in graph.entities.values())) == 8
    assert len(graph.shortest_distances("researcher_000000")) == len(graph.entities)


def test_deterministic_graph(config, graph):
    again = generate_graph(config)
    assert again.entities == graph.entities
    assert again.triples == graph.triples
    other = copy.deepcopy(config)
    other["seed"] += 1
    assert generate_graph(other).triples != graph.triples


def test_entity_suffixes_do_not_encode_employers(graph):
    rows = [(h, t) for h, r, t in graph.triples if r == "works_at"]
    university_count = sum(e["type"] == "university" for e in graph.entities.values())
    arithmetic_matches = sum(int(h.split("_")[-1]) % university_count == int(t.split("_")[-1]) for h, t in rows)
    assert arithmetic_matches / len(rows) < 0.25


def test_graph_roundtrip(tmp_path, graph):
    graph.save(tmp_path)
    loaded = GraphIndex.load(tmp_path)
    assert loaded.triples == graph.triples
    assert loaded.adjacency == graph.adjacency
    assert loaded.entities == graph.entities


def test_inverse_and_multiple_neighbors(graph):
    researcher = "researcher_000000"
    uni = graph.get_neighbors(researcher, "works_at")[0]
    assert researcher in graph.get_neighbors(uni, "inv:works_at")
    assert len(graph.get_neighbors(uni, "inv:works_at")) > 1
    with pytest.raises(ValueError, match="not permitted"):
        graph.get_neighbors(researcher, "inv:collaborates_with")
    with pytest.raises(ValueError, match="does not apply"):
        graph.get_neighbors(uni, "works_at")


def test_discovery_gating_and_answer_contract(graph):
    session = NavigationSession(graph, "researcher_000000")
    with pytest.raises(ValueError, match="not been discovered"):
        session.get_relations("country_000000")
    uni = session.get_neighbors("researcher_000000", "works_at")[0]
    assert session.get_entity(uni)["type"] == "university"
    assert session.submit_answer([uni, uni]) == {"answer_ids": [uni]}
    assert session.get_relations("researcher_000000")  # previously discovered remains accessible
    assert "reward" not in session.submit_answer([])


def test_invalid_graph_inputs(graph):
    with pytest.raises(ValueError, match="Duplicate"):
        GraphIndex(graph.entities, RELATIONS, [graph.triples[0], graph.triples[0]])
    with pytest.raises(ValueError, match="unknown entity"):
        GraphIndex(graph.entities, RELATIONS, [("missing", "works_at", "university_000000")])
    with pytest.raises(ValueError, match="Ill-typed"):
        GraphIndex(graph.entities, RELATIONS, [("university_000000", "works_at", "researcher_000000")])


def test_exact_answer_sets_and_alternative_walks():
    entities = {"p": {"type": "project", "name": "Project Zha"},
                "r1": {"type": "researcher", "name": "Dr. Vex"}, "r2": {"type": "researcher", "name": "Dr. Qor"},
                "u1": {"type": "university", "name": "University Lym"}, "u2": {"type": "university", "name": "University Xel"}}
    triples = [("p", "led_by", "r1"), ("p", "led_by", "r2"), ("r1", "works_at", "u1"), ("r2", "works_at", "u2")]
    graph = GraphIndex(entities, RELATIONS, triples)
    result = execute(graph, "p", ["led_by", "works_at"])
    assert result.answers == ["u1", "u2"]
    assert result.paths == [["p", "r1", "u1"], ["p", "r2", "u2"]]
    assert result.difficulty["reference_tool_calls"] == 7
    # Both branches merge at u1 but the complete walk count is preserved.
    merged = GraphIndex(entities, RELATIONS, triples[:-1] + [("r2", "works_at", "u1")])
    assert execute(merged, "p", ["led_by", "works_at"]).path_counts == {"u1": 2}


def test_shorter_equivalence_includes_inverse_and_zero_hop(graph):
    found = graph.shorter_results("researcher_000000", 3)
    assert found[frozenset(["researcher_000000"])] == ()
    uni = graph.get_neighbors("researcher_000000", "works_at")
    assert found[frozenset(uni)] == ("works_at",)
    for answer_set, path in found.items():
        assert frozenset(execute(graph, "researcher_000000", path).answers) == answer_set


def test_impossible_graph_budget_is_explicit(config):
    config = copy.deepcopy(config)
    config["graph"]["triple_count"] = 100
    with pytest.raises(ValueError, match="backbone requires"):
        generate_graph(config)


def test_hierarchy_cycles_are_rejected(graph):
    triples = [t for t in graph.triples if not (t[1] == "subfield_of" and t[0] in {"field_000000", "field_000001"})]
    triples += [("field_000000", "subfield_of", "field_000001"), ("field_000001", "subfield_of", "field_000000")]
    corrupted = GraphIndex(graph.entities, RELATIONS, triples)
    assert "Field hierarchy contains a cycle" in graph_errors(corrupted)


def test_typed_program_validation():
    with pytest.raises(ValueError, match="Invalid composition"):
        program_end_type("project", ["led_by", "city_in_country"])
