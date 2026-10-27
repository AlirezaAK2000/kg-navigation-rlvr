"""Read schema, entity labels and navigation facts from the saved Turtle graph."""

from rdflib import Namespace
from rdflib.namespace import OWL, RDF, RDFS

BASE = "https://example.org/kg-navigation-rlvr/"
KG = Namespace(BASE + "ontology#")
ENTITY = Namespace(BASE + "entity/")


def decode_graph(graph):
    entities, by_uri = {}, {}
    classes = set(graph.subjects(RDF.type, OWL.Class))
    for subject, identifier in graph.subject_objects(KG.entityId):
        kind = next(t for t in graph.objects(subject, RDF.type) if t in classes)
        entity_id = str(identifier)
        entities[entity_id] = {
            "type": str(kind).removeprefix(str(KG)),
            "name": str(graph.value(subject, RDFS.label)),
        }
        by_uri[subject] = entity_id

    relations = {}
    for predicate, identifier in graph.subject_objects(KG.relationId):
        relations[str(identifier)] = {
            "domain": str(graph.value(predicate, RDFS.domain)).removeprefix(str(KG)),
            "range": str(graph.value(predicate, RDFS.range)).removeprefix(str(KG)),
            "inverse_supported": bool(graph.value(predicate, KG.inverseSupported).toPython()),
            "symmetric": (predicate, RDF.type, OWL.SymmetricProperty) in graph,
            "cardinality": str(graph.value(predicate, KG.cardinality)),
        }
    triples = [
        (by_uri[h], name, by_uri[t])
        for name in relations
        for h, t in graph.subject_objects(KG[name])
    ]
    return entities, relations, triples
