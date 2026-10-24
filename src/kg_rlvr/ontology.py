"""Closed, typed ontology. Inverse operations use a separate `inv:` namespace."""

ENTITY_TYPES = ("researcher", "university", "project", "publication", "field", "funder", "city", "country")


def relation(domain, range_, cardinality, constraint, *, symmetric=False):
    return {
        "domain": domain, "range": range_, "cardinality": cardinality,
        "inverse_supported": not symmetric, "symmetric": symmetric,
        "constraints": constraint,
    }


RELATIONS = {
    "works_at": relation("researcher", "university", "many_to_one", "Exactly one employer per researcher."),
    "led_by": relation("project", "researcher", "many_to_many", "At least one leader; every leader contributes to the project."),
    "authored_by": relation("publication", "researcher", "many_to_many", "At least one author; the primary author leads the producing project."),
    "investigates": relation("project", "field", "many_to_many", "At least one field; includes the primary leader's specialty."),
    "funded_by": relation("project", "funder", "many_to_many", "At least one funder per project."),
    "located_in_city": relation("university", "city", "many_to_one", "Exactly one city per university."),
    "city_in_country": relation("city", "country", "many_to_one", "Exactly one country per city."),
    "collaborates_with": relation("researcher", "researcher", "many_to_many", "Symmetric, reciprocal stored triples, no self edges.", symmetric=True),
    "contributes_to": relation("researcher", "project", "many_to_many", "Includes all project leaders; researchers may join multiple projects."),
    "subfield_of": relation("field", "field", "many_to_one", "Acyclic forest; zero or one immediate parent per field."),
    "produced_by": relation("publication", "project", "many_to_one", "Exactly one producing project per publication."),
    "publication_field": relation("publication", "field", "many_to_one", "Exactly one field, investigated by its producing project."),
    "based_in": relation("funder", "city", "many_to_one", "Exactly one city per funding organization."),
    "specializes_in": relation("researcher", "field", "many_to_one", "Exactly one primary specialty per researcher."),
    "cites": relation("publication", "publication", "many_to_many", "No self citations; citations form a DAG from later to earlier publication IDs."),
    "partners_with": relation("university", "university", "many_to_many", "Symmetric, reciprocal stored triples, no self edges.", symmetric=True),
}


def operation_schema(operation, relations=RELATIONS):
    inverse = operation.startswith("inv:")
    key = operation[4:] if inverse else operation
    if key not in relations:
        raise ValueError(f"Unknown relation operation: {operation}")
    item = relations[key]
    if inverse and not item["inverse_supported"]:
        raise ValueError(f"Inverse traversal not permitted: {operation}")
    return (item["range"], item["domain"]) if inverse else (item["domain"], item["range"])


def operations(relations=RELATIONS):
    return sorted(list(relations) + [f"inv:{r}" for r, spec in relations.items() if spec["inverse_supported"]])


def program_end_type(start_type, path, relations=RELATIONS):
    current = start_type
    for operation in path:
        domain, range_ = operation_schema(operation, relations)
        if current != domain:
            raise ValueError(f"Invalid composition: {current} --{operation}--> {range_}")
        current = range_
    return current
