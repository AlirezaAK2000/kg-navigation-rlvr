import _bootstrap  # noqa: F401
import argparse
import json

from kg_rlvr.config import load_config
from kg_rlvr.dataset_builder import build_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate and exhaustively validate a synthetic KG dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="data")
    args = parser.parse_args()
    _, questions, statistics = build_dataset(load_config(args.config), args.output, progress=lambda text: print(text, flush=True))
    print(json.dumps({"output": args.output, "question_count": len(questions), "validation": statistics["validation"]}, indent=2))


if __name__ == "__main__":
    main()
