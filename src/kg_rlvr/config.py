"""JSON configuration with explicit validation and reproducible defaults."""

import copy
import json
from pathlib import Path


DEFAULT = {
    "seed": 1729,
    "graph": {
        "entity_count": 2000,
        "triple_count": 5000,
        "type_counts": None,
        "leaders_per_project": 1,
        "max_authors": 4,
        "max_project_fields": 3,
        "max_project_funders": 3,
        "collaboration_degree_target": 4,
        "extra_relation_weights": {
            "authored_by": 4, "contributes_to": 4, "collaborates_with": 3,
            "investigates": 1, "funded_by": 2, "cites": 2, "partners_with": 1,
        },
    },
    "questions": {
        "split_counts": {"train": 2667, "test": 1067},
        "hop_weights": {"1": 1, "2": 1, "3": 1, "4": 1},
        "hop_counts_by_split": None,
        "paraphrases_per_instance": 2,
        "min_branching_factor": 2.0,
        "max_answer_set_size": 8,
        "max_tool_calls": 80,
        "max_neighbors_per_response": 64,
        "split_strategy": "instance",
        "composition_test_count": 0,
        "holdout_program_fraction": 0.25,
    },
}


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key not in result:
            raise ValueError(f"Unknown configuration key: {key}")
        # Hop weights are an explicit selection, not a patch to default hops.
        result[key] = (copy.deepcopy(value) if key in {"hop_weights", "split_counts"} else
                       merge(result[key], value) if isinstance(value, dict) and isinstance(result[key], dict) else value)
    return result


def load_config(path=None):
    config = merge(DEFAULT, json.loads(Path(path).read_text())) if path else copy.deepcopy(DEFAULT)
    q, g = config["questions"], config["graph"]
    if set(q["split_counts"]) != {"train", "test"}:
        raise ValueError("split_counts must contain only train and test")
    q["split_counts"] = {s: q["split_counts"][s] for s in ("train", "test")}
    if any(not isinstance(v, int) or v < 0 for v in q["split_counts"].values()):
        raise ValueError("Split counts must be nonnegative integers")
    if not q["hop_weights"] or set(q["hop_weights"]) - {"1", "2", "3", "4"} or any(v <= 0 for v in q["hop_weights"].values()):
        raise ValueError("Selected hop categories must be from 1, 2, 3, 4 with positive weights")
    if q["hop_counts_by_split"] is not None:
        if set(q["hop_counts_by_split"]) != set(q["split_counts"]):
            raise ValueError("Explicit hop counts must specify train and test")
        for split, counts in q["hop_counts_by_split"].items():
            if (set(counts) != set(q["hop_weights"]) or
                    any(not isinstance(n, int) or n < 0 for n in counts.values()) or
                    sum(counts.values()) != q["split_counts"][split]):
                raise ValueError("Explicit hop counts must match the selected hops and split totals")
    if q["split_strategy"] not in {"instance", "start_entity"}:
        raise ValueError("split_strategy must be instance or start_entity")
    if not 0 < q["holdout_program_fraction"] < 1:
        raise ValueError("holdout_program_fraction must be between zero and one")
    if not 1 <= q["paraphrases_per_instance"] <= 3:
        raise ValueError("paraphrases_per_instance must be 1, 2, or 3")
    for key in ("max_answer_set_size", "max_tool_calls", "max_neighbors_per_response"):
        if q[key] < 1:
            raise ValueError(f"{key} must be positive")
    if q["composition_test_count"] < 0 or q["min_branching_factor"] < 0:
        raise ValueError("Question counts and branching thresholds must be nonnegative")
    if q["composition_test_count"] > q["split_counts"]["test"]:
        raise ValueError("composition_test_count must fit inside the test split")
    if g["entity_count"] < 100 or g["triple_count"] < 1:
        raise ValueError("Use at least 100 entities and a positive triple count")
    if g["leaders_per_project"] < 1 or any(g[k] < 1 for k in ("max_authors", "max_project_fields", "max_project_funders")):
        raise ValueError("Graph cardinality limits must be positive")
    if not g["extra_relation_weights"] or any(v <= 0 for v in g["extra_relation_weights"].values()):
        raise ValueError("Extra relation weights must be positive")
    return config


def allocate(total, weights):
    """Largest-remainder allocation, with stable tie breaking."""
    denom = sum(weights.values())
    raw = {k: total * v / denom for k, v in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for key in sorted(raw, key=lambda k: (-(raw[k] - counts[k]), str(k)))[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def split_hop_quotas(settings):
    """Optional exact quotas preserve hop counts during a split-only migration."""
    return copy.deepcopy(settings.get("hop_counts_by_split") or {
        split: allocate(count, settings["hop_weights"])
        for split, count in settings["split_counts"].items()})


def generation_quotas(settings):
    """Keep composition holdouts inside the configured test split and hop quotas."""
    quotas = split_hop_quotas(settings)
    if settings["composition_test_count"]:
        heldout = allocate(settings["composition_test_count"], settings["hop_weights"])
        quotas["test"] = {h: n - heldout[h] for h, n in quotas["test"].items()}
        if any(n < 0 for n in quotas["test"].values()):
            raise ValueError("Composition holdout exceeds the test hop quotas")
        quotas["composition_test"] = heldout
    return quotas
