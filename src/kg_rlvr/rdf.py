"""RDF vocabulary and Turtle interchange for the public graph (never QA gold data)."""

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from .ontology import ENTITY_TYPES

BASE = "https://example.org/kg-navigation-rlvr/"
KG = Namespace(BASE + "ontology#")
ENTITY = Namespace(BASE + "entity/")
ONTOLOGY = URIRef(BASE + "ontology")


def ontology_graph(relations):
    graph = Graph()
    for prefix, namespace in (("kg", KG), ("entity", ENTITY), ("owl", OWL), ("rdfs", RDFS)):
        graph.bind(prefix, namespace)
    graph.add((ONTOLOGY, RDF.type, OWL.Ontology))
    graph.add((ONTOLOGY, RDFS.label, Literal("Knowledge graph navigation RLVR")))
    graph.add((ONTOLOGY, RDFS.comment, Literal(
        "Fictional research ecosystem. Closed-world generation constraints are checked by the Python validator; no inference is applied during navigation.")))
    for annotation in ("entityId", "relationId", "cardinality", "inverseSupported", "constraints"):
        graph.add((KG[annotation], RDF.type, OWL.AnnotationProperty))
    for kind in ENTITY_TYPES:
        graph.add((KG[kind], RDF.type, OWL.Class))
        graph.add((KG[kind], RDFS.label, Literal(kind)))
    for name, spec in sorted(relations.items()):
        predicate = KG[name]
        graph.add((predicate, RDF.type, OWL.ObjectProperty))
        graph.add((predicate, RDFS.label, Literal(name)))
        graph.add((predicate, KG.relationId, Literal(name)))
        graph.add((predicate, RDFS.domain, KG[spec["domain"]]))
        graph.add((predicate, RDFS.range, KG[spec["range"]]))
        graph.add((predicate, KG.cardinality, Literal(spec["cardinality"])))
        graph.add((predicate, KG.inverseSupported, Literal(spec["inverse_supported"])))
        graph.add((predicate, KG.constraints, Literal(spec["constraints"])))
        if spec["cardinality"] == "many_to_one":
            graph.add((predicate, RDF.type, OWL.FunctionalProperty))
        if spec["symmetric"]:
            graph.add((predicate, RDF.type, OWL.SymmetricProperty))
        if spec["inverse_supported"]:
            inverse = KG["inverse_" + name]
            graph.add((inverse, RDF.type, OWL.ObjectProperty))
            graph.add((inverse, RDFS.label, Literal("inv:" + name)))
            graph.add((inverse, OWL.inverseOf, predicate))
            graph.add((inverse, RDFS.domain, KG[spec["range"]]))
            graph.add((inverse, RDFS.range, KG[spec["domain"]]))
    return graph


def encode_graph(entities, relations, triples):
    graph = ontology_graph(relations)
    for entity_id, spec in sorted(entities.items()):
        subject = ENTITY[entity_id]
        graph.add((subject, RDF.type, OWL.NamedIndividual))
        graph.add((subject, RDF.type, KG[spec["type"]]))
        graph.add((subject, KG.entityId, Literal(entity_id)))
        graph.add((subject, RDFS.label, Literal(spec["name"])))
    for head, relation, tail in triples:
        graph.add((ENTITY[head], KG[relation], ENTITY[tail]))
    return graph


def decode_graph(graph):
    """Read the explicit project schema and facts, independently of triple order."""
    def one(subject, predicate):
        values = list(graph.objects(subject, predicate))
        if len(values) != 1:
            raise ValueError(f"Expected exactly one {predicate} on {subject}; got {len(values)}")
        return values[0]

    entities, by_uri = {}, {}
    for subject, identifier in graph.subject_objects(KG.entityId):
        entity_id = str(identifier)
        if entity_id in entities or subject != ENTITY[entity_id]:
            raise ValueError(f"Duplicate or noncanonical entity ID: {entity_id}")
        types = [kind for kind in ENTITY_TYPES if (subject, RDF.type, KG[kind]) in graph]
        if len(types) != 1:
            raise ValueError(f"Entity {entity_id} requires exactly one supported entity type")
        entities[entity_id] = {"type": types[0], "name": str(one(subject, RDFS.label))}
        by_uri[subject] = entity_id
    relations = {}
    for predicate, identifier in graph.subject_objects(KG.relationId):
        name = str(identifier)
        if name in relations or predicate != KG[name]:
            raise ValueError(f"Duplicate or noncanonical relation ID: {name}")
        domain, range_ = one(predicate, RDFS.domain), one(predicate, RDFS.range)
        type_names = {KG[kind]: kind for kind in ENTITY_TYPES}
        if domain not in type_names or range_ not in type_names:
            raise ValueError(f"Unsupported domain/range for relation {name}")
        inverse = one(predicate, KG.inverseSupported).toPython()
        if not isinstance(inverse, bool):
            raise ValueError(f"inverseSupported must be a boolean on {name}")
        relations[name] = {"domain": type_names[domain], "range": type_names[range_],
            "cardinality": str(one(predicate, KG.cardinality)), "inverse_supported": inverse,
            "symmetric": (predicate, RDF.type, OWL.SymmetricProperty) in graph,
            "constraints": str(one(predicate, KG.constraints))}
    if not entities or not relations:
        raise ValueError("Turtle graph is missing entity IDs or the project relation schema")
    triples = []
    for name in relations:
        for head, tail in graph.subject_objects(KG[name]):
            if head not in by_uri or tail not in by_uri:
                raise ValueError(f"Triple refers to an unknown entity: {(head, name, tail)}")
            triples.append((by_uri[head], name, by_uri[tail]))
    return entities, relations, triples
