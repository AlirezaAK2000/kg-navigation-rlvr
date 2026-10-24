import _bootstrap  # noqa: F401
import argparse
from pathlib import Path

from kg_rlvr.dataset_builder import read_questions
from kg_rlvr.graph_index import GraphIndex
from kg_rlvr.visualization import render_visualization


def main():
    parser = argparse.ArgumentParser(description="Offline interactive graph, neighborhood, or question inspector")
    parser.add_argument("--data", default="data")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--entity")
    group.add_argument("--question-id")
    parser.add_argument("--radius", type=int, default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--entity-types", nargs="+")
    parser.add_argument("--relation-types", nargs="+")
    parser.add_argument("--paths-per-answer", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    import json
    limits = json.loads((Path(args.data) / "config.json").read_text())["visualization"]
    graph = GraphIndex.load(Path(args.data) / "graph")
    question = None
    if args.question_id:
        question = next((q for q in read_questions(args.data) if q["id"] == args.question_id), None)
        if question is None:
            parser.error(f"Unknown question ID: {args.question_id}")
    name = f"question_{args.question_id}" if question else f"entity_{args.entity}" if args.entity else "graph_overview"
    output = args.output or f"visualizations/{name}.html"
    payload = render_visualization(graph, output, question=question, entity=args.entity,
                                  radius=limits["radius"] if args.radius is None else args.radius,
                                  max_nodes=limits["max_nodes"] if args.max_nodes is None else args.max_nodes,
                                  max_edges=limits["max_edges"] if args.max_edges is None else args.max_edges,
                                  entity_types=args.entity_types, relation_types=args.relation_types,
                                  paths_per_answer=args.paths_per_answer)
    print(f"Saved {output}: {len(payload['nodes'])} nodes, {len(payload['edges'])} edges. Open directly in a browser; no server or network needed.")


if __name__ == "__main__":
    main()
