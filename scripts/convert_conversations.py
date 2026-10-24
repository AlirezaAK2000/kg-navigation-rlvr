import _bootstrap  # noqa: F401
import argparse

from kg_rlvr.conversion import export_conversations
from kg_rlvr.dataset_builder import read_questions
from kg_rlvr.graph_index import GraphIndex


def main():
    parser = argparse.ArgumentParser(description="Export prompt-only HF-compatible conversational JSONL")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="conversations")
    args = parser.parse_args()
    export_conversations(GraphIndex.load(f"{args.data}/graph"), read_questions(args.data), args.output)
    print(f"Saved public prompts under {args.output}; gold metadata remains in {args.data}/rl.")


if __name__ == "__main__":
    main()
