from collections import defaultdict
import json

import pytest
from rdflib import Graph
from rdflib.namespace import OWL, RDF, RDFS

from kg_rlvr.graph_index import GraphIndex, execute
from kg_rlvr.rdf import ENTITY, KG, ONTOLOGY


def test_rdflib_navigation_matches_all_original_edges(graph):
    expected = defaultdict(lambda: defaultdict(set))
    for head, relation, tail in graph.triples:
        expected[head][relation].add(tail)
        if graph.relations[relation]["inverse_supported"]:
            expected[tail]["inv:" + relation].add(head)
    assert isinstance(graph.rdf, Graph)
    assert not hasattr(graph, "adjacency")
    for entity in graph.entities:
        assert graph.get_relations(entity) == sorted(expected[entity])
        for operation, neighbors in expected[entity].items():
            assert graph.get_neighbors(entity, operation) == sorted(neighbors)
    assert all(not r.startswith("http") for e in graph.entities for r in graph.get_relations(e))


def test_callers_cannot_mutate_cached_lookup_results(graph):
    researcher = "researcher_000000"
    expected = graph.get_neighbors(researcher, "works_at")
    graph.get_neighbors(researcher, "works_at").clear()
    graph.get_relations(researcher).clear()
    assert graph.get_neighbors(researcher, "works_at") == expected
    assert "works_at" in graph.get_relations(researcher)


def test_turtle_contains_owl_schema_and_original_facts(graph, tmp_path):
    graph.save(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"graph.ttl"}
    rdf = Graph().parse(tmp_path / "graph.ttl", format="turtle")
    assert (ONTOLOGY, RDF.type, OWL.Ontology) in rdf
    assert len(set(rdf.subjects(RDF.type, OWL.NamedIndividual))) == len(graph.entities)
    assert len(set(rdf.subjects(RDF.type, OWL.Class))) == 8
    assert (KG.works_at, RDFS.domain, KG.researcher) in rdf
    assert (KG.works_at, RDF.type, OWL.FunctionalProperty) in rdf
    assert (KG.collaborates_with, RDF.type, OWL.SymmetricProperty) in rdf
    assert (KG.inverse_works_at, OWL.inverseOf, KG.works_at) in rdf
    assert (ENTITY.researcher_000000, RDFS.label, None) in rdf
    assert sum(len(list(rdf.triples((None, KG[r], None)))) for r in graph.relations) == len(graph.triples)
    assert not any(term in (tmp_path / "graph.ttl").read_text() for term in ("answer_ids", "gold_entity_paths", "relation_path"))


def test_rdflib_property_paths_and_sparql_match_complete_answers(graph, questions):
    for hop in (2, 3, 4):
        q = next(q for q in questions if q["hop_count"] == hop)
        parts = [~KG[r[4:]] if r.startswith("inv:") else KG[r] for r in q["relation_path"]]
        path = parts[0]
        for part in parts[1:]:
            path = path / part
        found = sorted(str(e).removeprefix(str(ENTITY)) for e in graph.rdf.objects(ENTITY[q["start_entity"]], path))
        assert sorted(set(found)) == q["answer_ids"]
    q = questions[0]
    path = "/".join(("^" if r.startswith("inv:") else "") + KG[r.removeprefix("inv:")].n3() for r in q["relation_path"])
    result = graph.rdf.query(f"SELECT DISTINCT ?answer WHERE {{ {ENTITY[q['start_entity']].n3()} {path} ?answer . }}")
    assert sorted(str(row.answer).removeprefix(str(ENTITY)) for row in result) == q["answer_ids"]


def test_legacy_input_can_be_read_and_migrated_without_changing_ids(graph, tmp_path):
    (tmp_path / "entities.json").write_text(json.dumps(graph.entities))
    (tmp_path / "relations.json").write_text(json.dumps(graph.relations))
    (tmp_path / "triples.tsv").write_text("".join("\t".join(row) + "\n" for row in graph.triples))
    loaded = GraphIndex.load(tmp_path)
    loaded.save(tmp_path)
    # Turtle is authoritative even when an old mount contains legacy files too.
    (tmp_path / "triples.tsv").write_text("invalid old format\n")
    migrated = GraphIndex.load(tmp_path)
    assert migrated.entities == graph.entities and migrated.relations == graph.relations
    assert migrated.triples == graph.triples


def test_corrupt_rdf_entities_and_schema_are_rejected(graph, tmp_path):
    graph.save(tmp_path)
    rdf = Graph().parse(tmp_path / "graph.ttl", format="turtle")
    rdf.remove((ENTITY.researcher_000000, KG.entityId, None))
    rdf.serialize(destination=tmp_path / "graph.ttl", format="turtle")
    with pytest.raises(ValueError, match="unknown entity"):
        GraphIndex.load(tmp_path)
