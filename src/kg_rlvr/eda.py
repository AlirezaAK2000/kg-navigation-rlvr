"""Read-only EDA helpers. Optional pandas/networkx dependencies live in the EDA extra.

No generation or on-disk validation utilities are called here: saved input files
are never written. Public RDFLib-backed graph execution is reused.
"""

from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import html
import json
from pathlib import Path
import random
import re

import networkx as nx
import pandas as pd

from .dataset_builder import read_questions
from .graph_index import GraphIndex, execute
from .ontology import program_end_type
from .query_generator import instance_id
from .validators import validate_dataset

REQUIRED = {"id", "question", "split", "start_entity", "answer_ids", "hop_count", "relation_path"}
OPTIONAL = {"difficulty", "gold_entity_paths", "template_id", "program_id", "query_instance_id",
            "answer_path_counts", "shortest_path_hops", "shortest_equivalent_program_length"}
WORD = re.compile(r"\b\w+(?:[-']\w+)*\b", re.UNICODE)


def fingerprint(directory):
    """Include every file and its relative path; detects changes AND added files."""
    root = Path(directory).resolve()
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def output_directory(path, data_directory):
    """Protect read-only inputs, including symlink-resolved output locations."""
    output, source = Path(path).resolve(), Path(data_directory).resolve()
    if output == source or source in output.parents:
        raise ValueError("EDA output must be outside the source dataset directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def words(text):
    return WORD.findall(str(text).casefold())


def normalize(text):
    return " ".join(str(text).casefold().split())


def masked_question(record, graph):
    start = record.get("start_entity")
    text = record.get("question", "")
    if start in graph.entities:
        text = re.sub(re.escape(graph.entities[start]["name"]), " ENTITY ", text, flags=re.I)
        text = re.sub(re.escape(start), " ENTITY ", text, flags=re.I)
    return " ".join(words(text))


def load_data(directory):
    """Reuse the canonical graph loader and question reader; retain source split."""
    directory = Path(directory)
    graph = GraphIndex.load(directory / "graph")
    records = read_questions(directory)
    if not records:
        raise ValueError(f"No questions found under {directory / 'rl'}")
    source_splits = [p.stem for p in sorted((directory / "rl").glob("*.jsonl"))
                     for line in p.read_text().splitlines() if line.strip()]
    records = [{**q, "_source_split": source} for q, source in zip(records, source_splits)]
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    return graph, records, config


def schema_audit(graph, records):
    rows = []
    def add(check, count, importance, explanation):
        rows.append(dict(check=check, count=int(count), importance=importance, explanation=explanation))
    for field in sorted(REQUIRED | OPTIONAL):
        missing = sum(field not in q or q[field] is None for q in records)
        add(f"Missing {field}", missing, "required" if field in REQUIRED else "optional", "Missing optional metadata is reported and recomputed where possible.")
    add("Blank question/ID", sum(not str(q.get("id") or "").strip() or not str(q.get("question") or "").strip() for q in records), "required", "Blank identifiers or question text cannot support QA inspection.")
    add("Unknown starting entity", sum(q.get("start_entity") not in graph.entities for q in records), "required", "Starting entities must exist.")
    add("Unknown answer references", sum(a not in graph.entities for q in records for a in (q.get("answer_ids") or [])), "required", "Counts individual invalid answer IDs.")
    add("Unknown witness references", sum(e not in graph.entities for q in records for p in (q.get("gold_entity_paths") or []) for e in p), "required", "Counts individual invalid path entity IDs.")
    add("Split/file mismatch", sum(q.get("split") != q.get("_source_split", q.get("split")) for q in records), "required", "The split label should match the JSONL filename.")
    add("Duplicate question IDs", len(records) - len({q.get("id") for q in records}), "required", "Question IDs should be unique.")
    typed_errors = 0
    for q in records:
        try:
            program_end_type(graph.entities[q["start_entity"]]["type"], q["relation_path"], graph.relations)
        except (ValueError, KeyError, TypeError):
            typed_errors += 1
    add("Invalid typed compositions", typed_errors, "required", "Checks existing ontology, inverse permission, and type compatibility.")
    add("Hop/program length mismatch", sum(q.get("hop_count") != len(q.get("relation_path") or []) for q in records), "required", "The intended hop count is the executable program length.")
    add("Empty/duplicate answer sets", sum(not q.get("answer_ids") or len(q["answer_ids"]) != len(set(q["answer_ids"])) for q in records), "required", "All answers must be present and unique.")
    return pd.DataFrame(rows)


def question_frame(graph, records):
    rows = []
    for q in records:
        answers = q.get("answer_ids") or []
        start = q.get("start_entity")
        path = tuple(q.get("relation_path") or [])
        rows.append({
            **q, "composition": path, "composition_label": " → ".join(path),
            "instance_key": instance_id(start, path, answers),
            "text_key": normalize(q.get("question", "")),
            "template_pattern": masked_question(q, graph),
            "start_type": graph.entities.get(start, {}).get("type", "UNKNOWN"),
            "answer_count": len(answers), "answer_key": tuple(sorted(answers)),
            "answer_types": tuple(sorted({graph.entities.get(a, {}).get("type", "UNKNOWN") for a in answers})),
            "word_count": len(words(q.get("question", ""))), "char_count": len(q.get("question", "")),
        })
    return pd.DataFrame(rows)


def graph_views(graph):
    """Separate stored directed triples, collapsed endpoint pairs, and navigation."""
    stored = nx.MultiDiGraph()
    stored.add_nodes_from((e, spec) for e, spec in graph.entities.items())
    stored.add_edges_from((h, t, {"relation": r}) for h, r, t in graph.triples)
    simple = nx.DiGraph(stored)
    navigation = nx.DiGraph()
    navigation.add_nodes_from(stored.nodes)
    navigation.add_edges_from((e, n) for e in graph.entities for neighbors in graph.neighbor_map(e).values() for n in neighbors)
    entities = pd.DataFrame.from_dict(graph.entities, orient="index").rename_axis("entity_id").reset_index()
    entities["in_degree"] = entities.entity_id.map(dict(stored.in_degree()))
    entities["out_degree"] = entities.entity_id.map(dict(stored.out_degree()))
    entities["total_degree"] = entities.in_degree + entities.out_degree
    entities["unique_neighbors"] = entities.entity_id.map(dict(simple.to_undirected().degree()))
    entities["navigation_degree"] = entities.entity_id.map(dict(navigation.out_degree()))
    triples = pd.DataFrame(graph.triples, columns=["head", "relation", "tail"])
    branch = pd.DataFrame([{"entity_id": e, "operation": r, "neighbor_count": len(ns),
                           "entity_type": graph.entities[e]["type"], "inverse": r.startswith("inv:")}
                          for e in graph.entities for r, ns in graph.neighbor_map(e).items()])
    return stored, simple, navigation, entities, triples, branch


def _trace(graph, start, path, key):
    frontier, rows = {start}, []
    for depth, relation in enumerate(path):
        following = set()
        for entity in sorted(frontier):
            operations = graph.get_relations(entity)
            actual = graph.get_neighbors(entity, relation) if relation in operations else []
            total = sum(len(graph.get_neighbors(entity, r)) for r in operations)
            rows.append({"instance_key": key, "hop_count": len(path), "step": depth + 1,
                         "entity_id": entity, "entity_type": graph.entities[entity]["type"],
                         "role": "start" if depth == 0 else "intermediate", "operation": relation,
                         "available_relations": len(operations), "navigation_branches": total,
                         "candidate_neighbors": len(actual), "distracting_relations": len(operations) - int(relation in operations),
                         "distractor_branches": total - len(actual)})
            following.update(actual)
        frontier = following
    return rows


def navigation_audit(graph, records, run_full_validation=True, config=None):
    """Recompute existing metrics, BFS from used starts only, and recheck witnesses.

    Traces contain one row per unique query-instance frontier visit, so duplicated
    paraphrases do not inflate the intermediate-node measurements.
    """
    results, distance_cache, traces, rows = {}, {}, [], []
    for q in records:
        start, path = q.get("start_entity"), tuple(q.get("relation_path") or [])
        key = (start, path)
        row = {"id": q.get("id"), "execution_error": None, "witness_errors": 0,
               "difficulty_mismatches": 0, "shortest_metadata_mismatch": False,
               "difficulty_metadata_present": isinstance(q.get("difficulty"), dict),
               "witness_metadata_present": bool(q.get("gold_entity_paths"))}
        try:
            if key not in results:
                result = execute(graph, start, path)
                results[key] = result
                traces.extend(_trace(graph, start, path, instance_id(start, path, result.answers)))
            result = results[key]
            if start not in distance_cache:
                distance_cache[start] = graph.shortest_distances(start)
            distances = distance_cache[start]
            observed = {a: distances[a] for a in result.answers}
            stored = q.get("difficulty") or {}
            mismatches = [k for k, v in stored.items() if k in result.difficulty and v != result.difficulty[k]]
            row.update(result.difficulty)
            row.update({"complete_answers_valid": result.answers == q.get("answer_ids"),
                        "difficulty_mismatches": len(mismatches),
                        "shortest_metadata_mismatch": q.get("shortest_path_hops") is not None and q["shortest_path_hops"] != observed,
                        "shortest_min": min(observed.values(), default=float("nan")),
                        "shortest_mean": sum(observed.values()) / len(observed) if observed else float("nan"),
                        "shortest_max": max(observed.values(), default=float("nan")),
                        "recomputed_paths": result.paths})
            for witness in q.get("gold_entity_paths") or []:
                if len(witness) != len(path) + 1 or witness[0] != start or witness[-1] not in result.answers:
                    row["witness_errors"] += 1
                    continue
                try:
                    row["witness_errors"] += sum(t not in graph.get_neighbors(h, r) for h, r, t in zip(witness, path, witness[1:]))
                except (KeyError, ValueError):
                    row["witness_errors"] += 1
            if q.get("gold_entity_paths") and {p[-1] for p in q["gold_entity_paths"] if p} != set(result.answers):
                row["witness_errors"] += 1
            row["core_valid"] = bool(row["complete_answers_valid"] and not row["witness_errors"] and not row["difficulty_mismatches"] and not row["shortest_metadata_mismatch"])
        except (KeyError, ValueError, TypeError, IndexError) as error:
            row.update(execution_error=str(error), complete_answers_valid=False, core_valid=False)
        rows.append(row)
    full = {"available": False, "reason": "Full validation disabled or optional verifier metadata missing."}
    if run_full_validation and all(REQUIRED | OPTIONAL <= q.keys() and all(q[k] is not None for k in OPTIONAL) for q in records):
        full = {"available": True, **validate_dataset(graph, records, config)}
    return pd.DataFrame(rows), pd.DataFrame(traces), full


def split_overlaps(frame):
    """Exact pairwise split intersections; sharing facts/templates is not failure."""
    sets = {}
    for split, part in frame.groupby("split"):
        paths = {(tuple(path), tuple(q.relation_path)) for q in part.itertuples()
                 for path in (getattr(q, "gold_entity_paths", None) or [])}
        sets[split] = {"question_text": set(part.text_key), "query_instance": set(part.instance_key),
                       "start_entity": set(part.start_entity), "answer_entity": {a for answers in part.answer_ids for a in answers},
                       "composition": set(part.composition), "template_pattern": set(part.template_pattern),
                       "template_id": set(part.template_id.dropna()) if "template_id" in part else set(), "reference_path": paths}
    rows = []
    for left, right in combinations(sorted(sets), 2):
        for key in sets[left]:
            a, b = sets[left][key], sets[right][key]
            common = len(a & b)
            rows.append({"left": left, "right": right, "item": key, "shared": common,
                         "left_unique": len(a), "right_unique": len(b),
                         "jaccard": common / len(a | b) if a | b else 0,
                         "right_covered_pct": 100 * common / len(b) if b else 0})
    return pd.DataFrame(rows), sets


def near_duplicate_screen(frame, seed=41, threshold=0.82, candidates_per_row=6):
    """Bounded lexical screen, not an exhaustive or semantic duplicate detector.

    Block by entity-masked text and composition; sample at most k candidates per
    row per block. Explicit cross-split candidates keep split boundaries visible.
    Actual (unmasked) word-set Jaccard scores are reported. Cost is O(n*k), not n².
    """
    rng = random.Random(seed)
    tokens = {q.id: set(words(q.question)) for q in frame.itertuples()}
    by_id = frame.set_index("id")
    split_for = dict(zip(frame.id, frame.split))
    pairs = set()
    for column in ("template_pattern", "composition"):
        for _, group in frame.groupby(column, sort=False):
            ids = sorted(group.id)
            split_ids = {s: sorted(g.id) for s, g in group.groupby("split")}
            pools = {s: [q for other, qs in split_ids.items() if other != s for q in qs] or ids for s in split_ids}
            for left in ids:
                pool = pools[split_for[left]]
                for right in rng.sample(pool, min(candidates_per_row, len(pool))):
                    if left != right:
                        pairs.add(tuple(sorted([left, right])))
    rows = []
    for left, right in sorted(pairs):
        a, b = tokens[left], tokens[right]
        similarity = len(a & b) / len(a | b) if a | b else 1.0
        rows.append({"left_id": left, "right_id": right, "left_split": by_id.loc[left, "split"],
                     "right_split": by_id.loc[right, "split"], "jaccard": similarity,
                     "near_duplicate": similarity >= threshold,
                     "same_instance": by_id.loc[left, "instance_key"] == by_id.loc[right, "instance_key"],
                     "same_template_pattern": by_id.loc[left, "template_pattern"] == by_id.loc[right, "template_pattern"],
                     "left_question": by_id.loc[left, "question"], "right_question": by_id.loc[right, "question"]})
    return pd.DataFrame(rows, columns=["left_id", "right_id", "left_split", "right_split", "jaccard", "near_duplicate", "same_instance", "same_template_pattern", "left_question", "right_question"])


def lexical_summary(frame, seed=41, token_budget=1500):
    rng = random.Random(seed)
    collections = {s: [w for text in group.question for w in words(text)] for s, group in frame.groupby("split")}
    budget = min(token_budget, min(map(len, collections.values())))
    rows = []
    for split, group in frame.groupby("split"):
        raw = collections[split]
        masked = [w for text in group.template_pattern for w in words(text)]
        matched = rng.sample(raw, budget)
        # Both raw and masked comparisons have the same budget across splits.
        rows.append({"split": split, "tokens": len(raw), "vocabulary": len(set(raw)),
                     "raw_type_token_ratio": len(set(raw)) / max(1, len(raw)),
                     "matched_raw_tokens": budget, "matched_raw_ttr": len(set(matched)) / max(1, budget),
                     "masked_tokens": len(masked), "masked_vocabulary": len(set(masked)),
                     "masked_raw_ttr": len(set(masked)) / max(1, len(masked))})
    masked_budget = min(token_budget, min(sum(len(words(t)) for t in group.template_pattern) for _, group in frame.groupby("split")))
    for row in rows:
        texts = frame.loc[frame.split == row["split"], "template_pattern"]
        sampled = rng.sample([w for t in texts for w in words(t)], masked_budget)
        row.update(matched_masked_tokens=masked_budget, matched_masked_ttr=len(set(sampled)) / max(1, masked_budget))
    return pd.DataFrame(rows)


def stratified_samples(frame, per_hop=5, seed=41):
    sampled = [part.sort_values("id").sample(n=min(per_hop, len(part)), random_state=seed + int(hop))
               for hop, part in frame.groupby("hop_count")]
    return pd.concat(sampled).sort_values(["hop_count", "id"])


def question_html(graph, record, audit_row=None):
    """Escaped, readable question and all answer witnesses; private EDA only."""
    result = execute(graph, record["start_entity"], record["relation_path"])
    name = lambda e: f"{html.escape(graph.entities[e]['name'])} <code>{html.escape(e)}</code>"
    answers = "<br>".join(name(a) for a in result.answers)
    paths = []
    for path in result.paths:
        pieces = [name(path[0])]
        for operation, node in zip(record["relation_path"], path[1:]):
            pieces.append(f" <b>— {html.escape(operation)} →</b> {name(node)}")
        paths.append("<li>" + "".join(pieces) + "</li>")
    verified = result.answers == record["answer_ids"]
    status = "PASS" if verified and (audit_row is None or audit_row.get("core_valid", True)) else "FAIL"
    d = result.difficulty
    return (f'<article style="border:1px solid #bbc9d4;border-radius:7px;padding:16px;margin:12px 0;font-size:13px">'
            f'<h4>{html.escape(record["id"])} · {record["hop_count"]} hops · {html.escape(record["split"])}</h4>'
            f'<p><strong>{html.escape(record["question"])}</strong></p><p>Start: {name(record["start_entity"])}</p>'
            f'<p>Complete answer set:<br>{answers}</p><p>Program: <code>{html.escape(" → ".join(record["relation_path"]))}</code></p>'
            f'<ol>{"".join(paths)}</ol><p>Branching {d["branching_factor"]:.2f} · available relations {d["relation_options_mean"]:.2f} '
            f'· query fanout {d["query_neighbors_mean"]:.2f} · distractor occurrences {d["distractor_count"]} '
            f'· reference calls {d["reference_tool_calls"]} · verification: <b>{status}</b></p></article>')


def quality_assessment(graph, frame, audit, schema, overlaps, config, full):
    rows = []
    def add(check, observed, expectation, status, explanation):
        rows.append(dict(check=check, observed=observed, expectation=expectation, status=status, explanation=explanation))
    bad_schema = int(schema.loc[schema.importance == "required", "count"].sum())
    add("Required schema/reference issues", bad_schema, "0", "PASS" if not bad_schema else "FAIL", "Invalid identifiers, types, or schema undermine executable verification.")
    invalid = int((~audit.core_valid).sum())
    add("Question execution / metadata failures", invalid, "0", "PASS" if not invalid else "FAIL", "Recomputed complete answers, saved witnesses, and available difficulty/distance fields.")
    full_errors = len(full.get("errors", []))
    add("Full existing validator", full_errors if full.get("available") else "not available", "0 errors", ("PASS" if not full_errors else "FAIL") if full.get("available") else "UNAVAILABLE", "Includes ontology constraints, template/program binding, grouping, and shorter-equivalent-program checks; missing optional fields limit this audit.")
    for key, label in [("question_text", "Cross-split identical text"), ("query_instance", "Cross-split query instances")]:
        count = int(overlaps.loc[overlaps.item == key, "shared"].sum()) if not overlaps.empty else 0
        add(label, count, "0 shared pairwise keys", "PASS" if count == 0 else "FAIL", "Counts shared keys per split pair; one key can contribute to multiple pairs.")
    _, simple, _, _, _, branch = graph_views(graph)
    components = nx.number_weakly_connected_components(simple)
    add("Weak components", components, "Prefer a connected navigation ecosystem", "PASS" if components == 1 else "REVIEW", "Disconnected islands can constrain reachable tasks; not automatically invalid.")
    isolates = len(list(nx.isolates(simple)))
    add("Isolated entities", isolates, "Prefer 0", "PASS" if not isolates else "REVIEW", "An isolated entity cannot support a positive-length traversal task.")
    relation_counts = Counter(r for _, r, _ in graph.triples)
    ratio = max(relation_counts.values()) / max(1, min(relation_counts.values()))
    add("Stored relation imbalance", f"{ratio:.1f}× max/min", "No universal balance requirement", "REVIEW", "Mandatory functional relations and sparse partnerships have different expected frequencies; inspect query coverage separately.")
    add("Entity-masked language patterns", f"{frame.template_pattern.nunique()} patterns / {len(frame)} questions", "No universal cutoff", "REVIEW", "Finite grammar can support template matching; shared patterns alone are not query-instance leakage.")
    add("Most reused starting entity", int(frame.start_entity.value_counts().max()), "No universal cutoff", "REVIEW", "Frequent starts can encourage memorization. Compare with train/test starting-entity overlap.")
    settings = (config or {}).get("questions", {})
    limits = [("branching_factor", "min_branching_factor", ">="), ("reference_tool_calls", "max_tool_calls", "<="),
              ("max_neighbor_response", "max_neighbors_per_response", "<=")]
    for metric, key, comparison in limits:
        if key not in settings:
            add(metric, "threshold unavailable", "User-configured", "UNAVAILABLE", "Values are measured, but no saved generation threshold exists.")
            continue
        values = frame[metric]
        count = int((values < settings[key]).sum()) if comparison == ">=" else int((values > settings[key]).sum())
        add(f"Configured {metric} violations", count, f"{metric} {comparison} {settings[key]}", "PASS" if count == 0 else "FAIL", "Applies to the specified gold executor/visited frontiers, not arbitrary agent exploration.")
    cap = settings.get("max_answer_set_size")
    count = int((frame.answer_count > cap).sum()) if cap is not None else "unknown"
    add("Configured answer-set cap violations", count, f"≤ {cap}" if cap else "not configured", "PASS" if count == 0 else "REVIEW", "Complete sets are required; cardinality also affects response and verification cost.")
    four = frame[frame.hop_count == 4]
    easy = int(((four.query_neighbors_mean <= 1) & (four.reference_tool_calls == 9)).sum())
    add("Single-frontier four-hop executions", easy, "Review; not an automatic failure", "REVIEW", "Nine calls is 2 calls × 4 stages + submission. Single query neighbors do not remove choices among distracting relations.")
    shorter = int((four.shortest_min < four.hop_count).sum())
    add("Four-hop questions with a shorter unlabelled path", shorter, "Review relation semantics", "REVIEW" if shorter else "PASS", "Reaching an answer via another relation sequence does not necessarily answer the same question.")
    add("Largest graph neighbor response", int(branch.neighbor_count.max()), "Compare with context limits", "INFO", "Includes all available operations, not only gold-traversed responses; no truncation was applied.")
    return pd.DataFrame(rows)
