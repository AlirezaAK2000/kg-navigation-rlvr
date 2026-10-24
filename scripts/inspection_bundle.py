import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from kg_rlvr.dataset_builder import read_questions
from kg_rlvr.graph_index import GraphIndex
from kg_rlvr.visualization import inspection_bundle


def main():
    parser = argparse.ArgumentParser(description="Generate all inspection modes and one multi-answer example per hop")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="visualizations")
    args = parser.parse_args()
    graph = GraphIndex.load(Path(args.data) / "graph")
    selected = inspection_bundle(graph, read_questions(args.data), args.output)
    print(json.dumps({"index": str(Path(args.output) / "index.html"), "question_examples": selected}, indent=2))


if __name__ == "__main__":
    main()
