"""Reviewed semantic templates, colocated with their typed executable programs.

All templates use existential/set wording, allowing complete multi-answer sets.
Add a definition here to extend the catalog; template IDs are versioned metadata.
No runtime language model or network access is used.
"""

from dataclasses import dataclass

from .ontology import program_end_type


@dataclass(frozen=True)
class Program:
    id: str
    start_type: str
    relation_path: tuple
    templates: tuple

    def as_dict(self):
        return {"id": self.id, "start_type": self.start_type, "answer_type": program_end_type(self.start_type, self.relation_path),
                "relation_path": list(self.relation_path), "hop_count": len(self.relation_path),
                "templates": [{"id": f"{self.id}:v{i + 1}", "text": text, "answer_quantifier": "set"} for i, text in enumerate(self.templates)]}


def p(id_, type_, path, *templates):
    program_end_type(type_, path.split())
    if len(templates) != 3 or any(t.count("{start}") != 1 for t in templates):
        raise ValueError(f"Program {id_} must have three templates with one start placeholder")
    return Program(id_, type_, tuple(path.split()), tuple(templates))


PROGRAMS = (
    p("p02_leader_employer", "project", "led_by works_at",
      "Which universities employ at least one leader of {start}?",
      "At which universities do the leaders of {start} work?",
      "Name the universities employing researchers who lead {start}."),
    p("p02_funder_city", "project", "funded_by based_in",
      "In which cities are the organizations funding {start} based?",
      "Which cities host a funding organization that sponsors {start}?",
      "Name the home cities of the funders of {start}."),
    p("p02_author_employer", "publication", "authored_by works_at",
      "Which universities employ at least one author of {start}?",
      "At which universities do the authors of {start} work?",
      "Name the universities employing researchers who authored {start}."),
    p("p02_publication_funder", "publication", "produced_by funded_by",
      "Which organizations fund the project that produced {start}?",
      "Name the funders of the research project behind {start}.",
      "Which funding organizations sponsor the project responsible for {start}?"),
    p("p02_university_country", "university", "located_in_city city_in_country",
      "Which countries contain the city where {start} is located?",
      "Name the countries containing the home city of {start}.",
      "In which countries is the city hosting {start}?"),
    p("p02_funder_country", "funder", "based_in city_in_country",
      "Which countries contain the city where {start} is based?",
      "Name the countries containing the home city of {start}.",
      "In which countries is the city hosting {start}?"),
    p("p02_field_funder", "field", "inv:investigates funded_by",
      "Which organizations fund projects investigating {start}?",
      "Name the funders of research projects that investigate {start}.",
      "Which funding organizations sponsor at least one project studying {start}?"),
    p("p02_university_projects", "university", "inv:works_at inv:led_by",
      "Which projects have at least one leader employed by {start}?",
      "Name the projects led by researchers who work at {start}.",
      "Which research projects are led by at least one researcher working at {start}?"),
    p("p02_author_projects", "researcher", "inv:authored_by produced_by",
      "Which projects produced publications authored by {start}?",
      "Name the research projects behind publications with {start} as an author.",
      "Which projects are responsible for at least one publication written by {start}?"),
    p("p02_cited_authors", "publication", "cites authored_by",
      "Who authored at least one publication cited by {start}?",
      "Which researchers wrote publications that {start} cites?",
      "Name the authors of publications cited in {start}."),
    p("p02_researcher_city", "researcher", "works_at located_in_city",
      "In which cities is the university employing {start} located?",
      "Which cities host the university where {start} works?",
      "Name the home cities of the university that employs {start}."),
    p("p03_researcher_country", "researcher", "works_at located_in_city city_in_country",
      "Which countries contain the city hosting the university where {start} works?",
      "In which countries is the university employing {start}, through its home city?",
      "Name the countries containing the home city of the university employing {start}."),
    p("p03_leader_city", "project", "led_by works_at located_in_city",
      "Which cities host universities employing at least one leader of {start}?",
      "In which cities are the universities where the leaders of {start} work?",
      "Name the home cities of universities employing researchers who lead {start}."),
    p("p03_funder_country", "project", "funded_by based_in city_in_country",
      "Which countries contain the home cities of organizations funding {start}?",
      "In which countries are the cities where the funders of {start} are based?",
      "Name the countries containing a city that hosts a funder of {start}."),
    p("p03_publication_funder_city", "publication", "produced_by funded_by based_in",
      "In which cities are the funders of the project behind {start} based?",
      "Which cities host organizations funding the project that produced {start}?",
      "Name the home cities of funders sponsoring the research project responsible for {start}."),
    p("p03_author_city", "publication", "authored_by works_at located_in_city",
      "Which cities host universities employing at least one author of {start}?",
      "In which cities are the universities where the authors of {start} work?",
      "Name the home cities of universities employing researchers who authored {start}."),
    p("p03_university_funders", "university", "inv:works_at inv:led_by funded_by",
      "Which organizations fund projects led by researchers employed at {start}?",
      "Name the funders of projects with at least one leader working at {start}.",
      "Which funding organizations sponsor projects led by at least one employee of {start}?"),
    p("p03_field_universities", "field", "inv:investigates led_by works_at",
      "Which universities employ leaders of projects investigating {start}?",
      "Name the universities employing at least one leader of a project studying {start}.",
      "At which universities do leaders of research projects investigating {start} work?"),
    p("p03_cited_author_employer", "publication", "cites authored_by works_at",
      "Which universities employ authors of publications cited by {start}?",
      "At which universities do authors of at least one publication cited in {start} work?",
      "Name the universities employing researchers who wrote publications that {start} cites."),
    p("p04_author_country", "publication", "authored_by works_at located_in_city city_in_country",
      "Which countries contain cities hosting universities employing authors of {start}?",
      "Name the countries containing the home cities of universities where authors of {start} work.",
      "In which countries are the home cities of universities employing at least one author of {start}?"),
    p("p04_leader_country", "project", "led_by works_at located_in_city city_in_country",
      "Which countries contain cities hosting universities employing leaders of {start}?",
      "Name the countries containing the home cities of universities where leaders of {start} work.",
      "In which countries are the home cities of universities employing at least one leader of {start}?"),
    p("p04_publication_funder_country", "publication", "produced_by funded_by based_in city_in_country",
      "Which countries contain the home cities of funders of the project behind {start}?",
      "Name the countries containing cities where funders of the project that produced {start} are based.",
      "In which countries are the home cities of organizations funding the project responsible for {start}?"),
    p("p04_university_funder_city", "university", "inv:works_at inv:led_by funded_by based_in",
      "Which cities host funders of projects led by researchers working at {start}?",
      "Name the home cities of organizations funding projects with a leader employed by {start}.",
      "In which cities are funders of projects led by at least one researcher employed at {start} based?"),
    p("p04_field_university_city", "field", "inv:investigates led_by works_at located_in_city",
      "Which cities host universities employing leaders of projects investigating {start}?",
      "Name the home cities of universities where leaders of projects studying {start} work.",
      "In which cities are universities employing at least one leader of a project investigating {start}?"),
    p("p04_author_funder_city", "researcher", "inv:authored_by produced_by funded_by based_in",
      "Which cities host funders of projects that produced publications authored by {start}?",
      "Name the home cities of organizations funding projects behind publications with {start} as an author.",
      "In which cities are funders of projects responsible for publications written by {start} based?"),
    p("p04_cited_author_city", "publication", "cites authored_by works_at located_in_city",
      "Which cities host universities employing authors of publications cited by {start}?",
      "Name the home cities of universities where authors of publications cited in {start} work.",
      "In which cities are universities employing researchers who authored publications that {start} cites?"),
)

PROGRAM_BY_ID = {p.id: p for p in PROGRAMS}
