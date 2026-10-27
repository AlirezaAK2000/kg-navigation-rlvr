"""Read-only RDFLib navigation over the saved graph."""

from functools import lru_cache
from pathlib import Path
from rdflib import Graph
from .rdf import ENTITY, KG, decode_graph


class GraphIndex:
    def __init__(self, rdf):
        self.rdf = rdf
        self.entities, self.relations, triples = decode_graph(rdf)
        self.triples = tuple(sorted(triples))
        self._relation_ids = {KG[r]: r for r in self.relations}
        self._inverse_ids = {
            KG[r]: "inv:" + r for r, spec in self.relations.items() if spec["inverse_supported"]
        }
        self._available = lru_cache(maxsize=4096)(self._query_relations)
        self._neighbors = lru_cache(maxsize=32768)(self._query_neighbors)

    @classmethod
    def load(cls, directory):
        return cls(Graph().parse(Path(directory) / "graph.ttl", format="turtle"))

    def get_entity(self, entity_id):
        if entity_id not in self.entities:
            raise ValueError(f"Unknown entity: {entity_id}")
        return {"id": entity_id, **self.entities[entity_id]}

    def get_relations(self, entity_id):
        self.get_entity(entity_id)
        return list(self._available(entity_id))

    def _query_relations(self, entity_id):
        node = ENTITY[entity_id]
        forward = {
            self._relation_ids[p] for p in self.rdf.predicates(node) if p in self._relation_ids
        }
        reverse = {
            self._inverse_ids[p] for p in self.rdf.predicates(object=node) if p in self._inverse_ids
        }
        return tuple(sorted(forward | reverse))

    def get_neighbors(self, entity_id, relation_id):
        self.get_entity(entity_id)
        inverse = relation_id.startswith("inv:")
        name = relation_id[4:] if inverse else relation_id
        if name not in self.relations or (
            inverse and not self.relations[name]["inverse_supported"]
        ):
            raise ValueError(f"Unknown relation operation: {relation_id}")
        spec = self.relations[name]
        domain = spec["range"] if inverse else spec["domain"]
        if self.entities[entity_id]["type"] != domain:
            raise ValueError(f"Operation {relation_id} does not apply to {entity_id}")
        return list(self._neighbors(entity_id, relation_id))

    def _query_neighbors(self, entity_id, relation_id):
        inverse = relation_id.startswith("inv:")
        predicate = KG[relation_id[4:] if inverse else relation_id]
        nodes = (
            self.rdf.subjects(predicate, ENTITY[entity_id])
            if inverse
            else self.rdf.objects(ENTITY[entity_id], predicate)
        )
        return tuple(sorted(str(node)[len(str(ENTITY)) :] for node in nodes))

    def neighbor_map(self, entity_id):
        """Permitted navigation operations only; RDF schema/label triples stay private to the store."""
        return {r: self.get_neighbors(entity_id, r) for r in self.get_relations(entity_id)}
