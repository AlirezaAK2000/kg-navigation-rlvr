"""Deterministic grouping, exact quotas, export and an optional composition holdout."""

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

from .config import generation_quotas, split_hop_quotas
from .graph_generator import generate_graph
from .query_generator import generate_candidates
from .question_templates import PROGRAMS, PROGRAM_BY_ID


class GenerationError(ValueError):
    pass


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def read_questions(directory):
    rows = []
    for path in sorted((Path(directory) / "rl").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def split_candidates(graph, candidates, config):
    rng = random.Random(config["seed"] + 2)
    one_hop_rng = random.Random(config["seed"] + 3)
    settings = config["questions"]
    heldout = []
    if settings["composition_test_count"]:
        for hop in (2, 3, 4, 1):
            if str(hop) not in settings["hop_weights"]:
                continue
            ids = [p.id for p in PROGRAMS if len(p.relation_path) == hop]
            (one_hop_rng if hop == 1 else rng).shuffle(ids)
            heldout.extend(ids[:max(1, int(len(ids) * settings["holdout_program_fraction"]))])
    pools = defaultdict(list)
    for candidate in candidates:
        if str(candidate["hop_count"]) in settings["hop_weights"]:
            pools[candidate["hop_count"]].append(candidate)
    for hop, pool in pools.items():
        (one_hop_rng if hop == 1 else rng).shuffle(pool)
    used, questions, quotas = set(), [], {}
    base_counts = settings["split_counts"]

    def owner(start):
        digest = hashlib.sha256(f"{config['seed']}:{start}".encode()).hexdigest()
        point = int(digest[:16], 16) % max(1, sum(base_counts.values()))
        for split, count in base_counts.items():
            if point < count:
                return split
            point -= count
        raise AssertionError("Unreachable allocation")

    quotas = generation_quotas(settings)
    # Keep multi-hop RNG consumption and IDs stable when adding one-hop quotas.
    # One-hop records are appended only after every original multi-hop split.
    for one_hop in (False, True):
        for split in quotas:
            for hop_label, quota in quotas[split].items():
                hop = int(hop_label)
                if (hop == 1) != one_hop:
                    continue
                remaining = quota
                available = 0
                for candidate in pools[hop]:
                    if not remaining:
                        break
                    iid = candidate["query_instance_id"]
                    is_holdout = candidate["program_id"] in heldout
                    if iid in used or is_holdout != (split == "composition_test"):
                        continue
                    if settings["split_strategy"] == "start_entity" and split != "composition_test" and owner(candidate["start_entity"]) != split:
                        continue
                    available += 1
                    used.add(iid)
                    program = PROGRAM_BY_ID[candidate["program_id"]]
                    variants = list(range(len(program.templates)))
                    (one_hop_rng if hop == 1 else rng).shuffle(variants)
                    take = min(settings["paraphrases_per_instance"], remaining)
                    for variant in variants[:take]:
                        question = copy.deepcopy(candidate)
                        question.update({"id": f"q_{len(questions) + 1:06d}", "split": "test" if split == "composition_test" else split,
                                         "template_id": f"{program.id}:v{variant + 1}",
                                         "question": program.templates[variant].format(start=graph.entities[candidate["start_entity"]]["name"])})
                        questions.append(question)
                        if split == "composition_test":
                            question["evaluation_category"] = "composition_holdout"
                    remaining -= take
                if remaining:
                    raise GenerationError(f"Cannot fill {split} {hop}-hop quota: requested {quota}, produced {quota - remaining}; {available} eligible unused groups. No filters were relaxed. Adjust graph size, counts, or explicit filters.")
    return questions, {"quotas": split_hop_quotas(settings),
                       "composition_holdout_quotas": quotas.get("composition_test", {}),
                       "heldout_program_ids": sorted(heldout),
                       "eligible_candidates_by_hop": dict(sorted(Counter(c["hop_count"] for c in candidates).items()))}


def build_dataset(config, directory, progress=print):
    """Write only after exact requested counts have been satisfied and verified."""
    from .statistics import build_statistics, markdown_report
    from .validators import validate_dataset
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise GenerationError(f"Output directory is not empty: {directory}. Choose a fresh directory; existing data is never overwritten.")
    progress("Generating the coherent graph...")
    graph = generate_graph(config)
    progress(f"Executing typed programs on {len(graph.entities)} entities...")
    candidates, rejections = generate_candidates(graph, config)
    progress(f"Splitting {len(candidates)} eligible query instances...")
    questions, generation = split_candidates(graph, candidates, config)
    generation["candidate_rejections"] = rejections
    progress(f"Validating all {len(questions)} questions...")
    validation = validate_dataset(graph, questions, config)
    if validation["errors"]:
        raise GenerationError(json.dumps(validation, indent=2))
    stats = build_statistics(graph, questions, validation, config, generation)
    # A fresh directory avoids accidental mixing with files from an earlier run.
    graph.save(directory / "graph")
    for split in generation["quotas"]:
        write_jsonl(directory / "rl" / f"{split}.jsonl", (q for q in questions if q["split"] == split))
    write_jsonl(directory / "metadata" / "query_programs.jsonl", (p.as_dict() for p in PROGRAMS))
    (directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (directory / "metadata" / "statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    (directory / "metadata" / "report.md").write_text(markdown_report(stats))
    return graph, questions, stats
