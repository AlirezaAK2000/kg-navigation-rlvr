import _bootstrap  # noqa: F401
import argparse
import random
from pathlib import Path

from kg_rlvr.dataset_builder import read_questions
from kg_rlvr.graph_index import GraphIndex, execute


def sample_report(graph, questions, count=10, seed=41):
    selected = random.Random(seed).sample(sorted(questions, key=lambda q: q["id"]), min(count, len(questions)))
    lines = ["# Randomly sampled and re-executed questions", "", f"Sampling seed: {seed}. Every sample below was checked against the executable graph interface.", ""]
    for q in selected:
        result = execute(graph, q["start_entity"], q["relation_path"])
        if result.answers != q["answer_ids"]:
            raise ValueError(f"Sample verification failed: {q['id']}")
        lines += [f"## {q['id']} · {q['hop_count']} hops · {q['split']}", "", q["question"], "",
                  "Answers: " + "; ".join(f"{graph.entities[a]['name']} (`{a}`)" for a in result.answers) + ".", "", "Verified reference paths:", ""]
        for path in result.paths:
            pieces = [f"{graph.entities[path[0]]['name']} (`{path[0]}`)"]
            for relation, entity in zip(q["relation_path"], path[1:]):
                pieces.append(f"— **{relation}** → {graph.entities[entity]['name']} (`{entity}`)")
            lines.append("- " + " ".join(pieces))
        lines += ["", f"Complete answer set verified. Navigation branching: {q['difficulty']['branching_factor']:.2f}; reference tool calls: {q['difficulty']['reference_tool_calls']}.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Display reproducible random samples with verified answers and witnesses")
    parser.add_argument("--data", default="data")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    report = sample_report(GraphIndex.load(Path(args.data) / "graph"), read_questions(args.data), args.count, args.seed)
    path = Path(args.output) if args.output else Path(args.data) / "metadata" / "samples.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
