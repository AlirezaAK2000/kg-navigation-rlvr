"""Generate a coherent graph backbone, then add constrained distractor edges."""

from collections import Counter
import random

from .config import allocate
from .graph_index import GraphIndex
from .ontology import ENTITY_TYPES, RELATIONS

DEFAULT_COUNTS = dict(zip(ENTITY_TYPES, [700, 100, 300, 400, 150, 100, 200, 50]))


def generate_graph(config):
    rng = random.Random(config["seed"])
    settings = config["graph"]
    counts = settings["type_counts"] or allocate(settings["entity_count"], DEFAULT_COUNTS)
    if set(counts) != set(ENTITY_TYPES) or any(not isinstance(n, int) or n < 2 for n in counts.values()):
        raise ValueError("type_counts must specify at least two entities of each of the eight types")
    if sum(counts.values()) != settings["entity_count"]:
        raise ValueError("type_counts must sum to entity_count")
    if counts["researcher"] < max(counts["university"], counts["field"]):
        raise ValueError("Need at least as many researchers as universities and fields for coverage")
    if counts["city"] < counts["country"]:
        raise ValueError("Need at least one city per country")
    if settings["leaders_per_project"] > counts["researcher"]:
        raise ValueError("Too many leaders requested")
    by_type = {kind: [f"{kind}_{i:06d}" for i in range(n)] for kind, n in counts.items()}
    prefix = {"researcher": "Dr.", "university": "University of", "project": "Project", "publication": "Study",
              "field": "Discipline", "funder": "Research Trust", "city": "City of", "country": "Republic of"}
    syllables = ["zha", "vex", "qor", "lym", "thae", "nuv", "xel", "orv", "ith", "zae", "kyr", "pha"]
    entities = {}
    for kind in ENTITY_TYPES:
        for index, entity_id in enumerate(by_type[kind]):
            word = "".join(rng.choice(syllables) for _ in range(3)).capitalize()
            entities[entity_id] = {"type": kind, "name": f"{prefix[kind]} {word} {index + 1:04d}"}
    triples, outgoing = set(), Counter()

    def add(head, rel, tail):
        edge = (head, rel, tail)
        if edge in triples:
            return
        triples.add(edge)
        outgoing[head, rel] += 1
        if RELATIONS[rel]["symmetric"]:
            triples.add((tail, rel, head))
            outgoing[tail, rel] += 1

    researchers, universities = by_type["researcher"], by_type["university"]
    fields, cities, countries = by_type["field"], by_type["city"], by_type["country"]
    projects, publications, funders = by_type["project"], by_type["publication"], by_type["funder"]
    def covered_assignments(values, count):
        # Balanced coverage is shuffled independently for each relation. Numeric
        # entity/name suffixes therefore do not encode employers or answers.
        assigned = [values[i % len(values)] for i in range(count)]
        rng.shuffle(assigned)
        return assigned

    country_assignment = covered_assignments(countries, len(cities))
    for city, country in zip(cities, country_assignment):
        add(city, "city_in_country", country)
    city_order = cities[:]
    rng.shuffle(city_order)
    for i, uni in enumerate(universities):
        add(uni, "located_in_city", city_order[i % len(cities)])
    for i, funder in enumerate(funders):
        add(funder, "based_in", city_order[(len(universities) + i) % len(cities)])
    employer = dict(zip(researchers, covered_assignments(universities, len(researchers))))
    specialty = dict(zip(researchers, covered_assignments(fields, len(researchers))))
    for researcher in researchers:
        add(researcher, "works_at", employer[researcher])
        add(researcher, "specializes_in", specialty[researcher])
    # A cross-university collaboration spine connects the institutional ecosystem.
    representatives = [next(r for r in researchers if employer[r] == uni) for uni in universities]
    for i in range(len(universities) - 1):
        add(representatives[i], "collaborates_with", representatives[i + 1])
    for i in range(min(10, len(universities) - 1)):
        add(universities[i], "partners_with", universities[i + 1])
    roots = max(1, len(fields) // 6)
    for i in range(roots, len(fields)):
        add(fields[i], "subfield_of", fields[rng.randrange(i)])
    primary_leader, primary_field = {}, {}
    leaders_order = researchers[:]
    rng.shuffle(leaders_order)
    funder_assignment = covered_assignments(funders, len(projects))
    for i, project in enumerate(projects):
        leader = leaders_order[i % len(researchers)]
        primary_leader[project] = leader
        primary_field[project] = specialty[leader]
        chosen = [leader] + rng.sample([r for r in researchers if r != leader], settings["leaders_per_project"] - 1)
        for person in chosen:
            add(project, "led_by", person)
            add(person, "contributes_to", project)
        add(project, "investigates", primary_field[project])
        add(project, "funded_by", funder_assignment[i])
    project_assignment = covered_assignments(projects, len(publications))
    for i, pub in enumerate(publications):
        project = project_assignment[i]
        add(pub, "produced_by", project)
        add(pub, "authored_by", primary_leader[project])
        add(pub, "publication_field", primary_field[project])
        if i:
            add(pub, "cites", publications[rng.randrange(i)])
    target = settings["triple_count"]
    if len(triples) > target:
        raise ValueError(f"Coherent backbone requires {len(triples)} triples; requested {target}. Increase triple_count; constraints were not relaxed.")
    weights = settings["extra_relation_weights"]
    allowed = {"authored_by", "contributes_to", "collaborates_with", "investigates", "funded_by", "cites", "partners_with"}
    if set(weights) - allowed:
        raise ValueError(f"Unsupported extra relations: {set(weights) - allowed}")
    attempts = 0
    while len(triples) < target:
        attempts += 1
        if attempts > max(10000, target * 100):
            raise ValueError(f"Could only create {len(triples)}/{target} triples under the configured degree limits")
        rel = rng.choices(list(weights), list(weights.values()))[0]
        spec = RELATIONS[rel]
        head = rng.choice(by_type[spec["domain"]])
        tail = rng.choice(by_type[spec["range"]])
        if head == tail or (head, rel, tail) in triples:
            continue
        cost = 2 if spec["symmetric"] else 1
        if len(triples) + cost > target:
            continue
        limits = {"authored_by": settings["max_authors"], "investigates": settings["max_project_fields"],
                  "funded_by": settings["max_project_funders"], "collaborates_with": settings["collaboration_degree_target"]}
        if rel in limits and (outgoing[head, rel] >= limits[rel] or (spec["symmetric"] and outgoing[tail, rel] >= limits[rel])):
            continue
        if rel == "cites" and head <= tail:
            continue
        add(head, rel, tail)
    return GraphIndex(entities, RELATIONS, triples)
