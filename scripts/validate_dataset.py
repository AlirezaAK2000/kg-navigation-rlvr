import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from kg_rlvr.validators import validate_saved_dataset


def main():
    parser = argparse.ArgumentParser(description="Verify all graph and QA invariants")
    parser.add_argument("--data", default="data")
    parser.add_argument("--check-reproducibility", action="store_true")
    args = parser.parse_args()
    result = validate_saved_dataset(args.data, args.check_reproducibility)
    path = Path(args.data) / "metadata" / "validation.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    statistics_path = Path(args.data) / "metadata" / "statistics.json"
    if statistics_path.exists():
        from kg_rlvr.statistics import markdown_report
        statistics = json.loads(statistics_path.read_text())
        statistics["validation"] = result
        statistics_path.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n")
        (statistics_path.parent / "report.md").write_text(markdown_report(statistics))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
