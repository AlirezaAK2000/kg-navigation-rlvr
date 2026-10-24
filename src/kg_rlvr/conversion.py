"""Export private-metadata-free conversational JSONL, loadable with HF datasets."""

from .dataset_builder import write_jsonl

PUBLIC_INSTRUCTIONS = (
    "Answer the question by navigating the supplied fictional research graph. "
    "Return the complete set of entity IDs. You may inspect previously discovered entities. "
    "get_relations(entity_id) lists available operation IDs; get_neighbors(entity_id, relation_id) "
    "returns all neighboring IDs; get_entity(entity_id) reveals its name and type. "
    "inv: denotes a permitted inverse relation. submit_answer(entity_ids) submits a set of discovered IDs."
)


def conversational_row(graph, question):
    entity = graph.get_entity(question["start_entity"])
    # Explicit allowlist: never copy the source record or embed verifier metadata.
    return {"prompt": [
        {"role": "system", "content": PUBLIC_INSTRUCTIONS},
        {"role": "user", "content": f"Question: {question['question']}\nStarting entity: {entity['id']} ({entity['name']}; {entity['type']})"},
    ]}


def export_conversations(graph, questions, directory):
    from pathlib import Path
    directory = Path(directory)
    for split in sorted({q["split"] for q in questions}):
        write_jsonl(directory / f"{split}.jsonl", (conversational_row(graph, q) for q in questions if q["split"] == split))
