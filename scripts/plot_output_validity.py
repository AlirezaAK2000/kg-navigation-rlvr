"""Compare vanilla and SFT output validity from saved test trajectories.

Run against a notebook results ZIP or its extracted run directory. No model is run.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

VARIANTS = {"vanilla": ("Vanilla", "#64748B"), "sft": ("After SFT", "#2563EB")}
ERROR_LABELS = {
    "multiple_call_blocks": "Multiple tool-call blocks in one turn",
    "turn_token_limit": "Generation cut off at token limit",
    "unknown_relation": "Unknown relation ID",
    "relation_type_mismatch": "Relation incompatible with entity type",
    "undiscovered_entity": "Entity ID not previously discovered",
}


def graph_error(message):
    if message.startswith("Unknown relation operation:"):
        return "unknown_relation"
    if message.startswith("Operation ") and "does not apply" in message:
        return "relation_type_mismatch"
    if message == "Use a discovered entity ID":
        return "undiscovered_entity"
    return "other_tool_error"


def load_evaluations(source):
    suffix = "variants/{variant}/evaluation/test_trajectories.jsonl"
    data, provenance = {}, {}
    if source.is_file():
        with ZipFile(source) as archive:
            for variant in VARIANTS:
                matches = [n for n in archive.namelist() if n.endswith(suffix.format(variant=variant))]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {variant} test trajectory file, found {matches}")
                raw = archive.read(matches[0])
                provenance[variant] = {"member": matches[0], "sha256": hashlib.sha256(raw).hexdigest()}
                data[variant] = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    else:
        for variant in VARIANTS:
            path = source / suffix.format(variant=variant)
            raw = path.read_bytes()
            provenance[variant] = {"member": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
            data[variant] = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    by_id = {v: {r["task_id"]: r for r in rows} for v, rows in data.items()}
    for variant, rows in data.items():
        if not rows or len(by_id[variant]) != len(rows):
            raise ValueError(f"Empty or duplicate test questions for {variant}")
    if set(by_id["vanilla"]) != set(by_id["sft"]):
        raise ValueError("Variants must be evaluated on the same test questions")
    for task_id, record in by_id["vanilla"].items():
        other = by_id["sft"][task_id]
        for key in ("question", "start_entity", "hop_count", "prompt"):
            if record[key] != other[key]:
                raise ValueError(f"Unmatched {key} for {task_id}")
    return data, provenance


def analyze(data):
    episodes, turns = [], []
    for variant, records in data.items():
        for record in records:
            current = []
            for index, turn in enumerate(record["turns"], 1):
                observation = turn.get("observation")
                error = observation.get("error", "") if observation else ""
                if observation is not None:
                    outcome = "accepted" if observation.get("ok") is True else graph_error(error)
                else:
                    if index != len(record["turns"]):
                        raise ValueError("An unexecuted turn occurs before the terminal turn")
                    outcome = record["termination_reason"]
                row = {
                    "variant": variant, "task_id": record["task_id"], "hop_count": record["hop_count"],
                    "turn": index, "outcome": outcome, "error": error,
                    "single_call_reached_executor": "tool_call" in turn,
                    "accepted": outcome == "accepted",
                    # Inspect raw text even if generation ended before parsing/execution.
                    "multiple_call_blocks": turn["raw_assistant"].count("<tool_call>") > 1,
                    "model_tokens": turn["model_tokens"],
                }
                turns.append(row)
                current.append(row)
            if not current:
                raise ValueError(f"No generated turns for {record['task_id']}")
            episodes.append({
                "variant": variant, "task_id": record["task_id"], "hop_count": record["hop_count"],
                "turns": len(current), "accepted_turns": sum(t["accepted"] for t in current),
                "all_outputs_valid": all(t["accepted"] for t in current),
                "termination_reason": record["termination_reason"],
                "submitted": bool(record["submitted"]), "answer_correct": bool(record["reward"]),
                **{key: any(t["multiple_call_blocks"] if key == "multiple_call_blocks"
                            else t["outcome"] == key for t in current) for key in ERROR_LABELS},
            })
            # Reconcile against saved counters, allowing parser errors not stored as turn observations.
            if sum(e["observation"].get("ok") is True for e in record["tool_events"]) != sum(t["accepted"] for t in current):
                raise ValueError(f"Executed-event mismatch for {record['task_id']}")
            if len(record["tool_events"]) != record["tool_calls"]:
                raise ValueError(f"Tool counter mismatch for {record['task_id']}")
    summaries, errors = [], []
    for variant in VARIANTS:
        es = [r for r in episodes if r["variant"] == variant]
        ts = [r for r in turns if r["variant"] == variant]
        summaries.append({
            "variant": variant, "episodes": len(es), "generated_turns": len(ts),
            "fully_valid_episodes": sum(r["all_outputs_valid"] for r in es),
            "fully_valid_episode_rate": sum(r["all_outputs_valid"] for r in es) / len(es),
            "single_call_turns": sum(r["single_call_reached_executor"] for r in ts),
            "single_call_turn_rate": sum(r["single_call_reached_executor"] for r in ts) / len(ts),
            "accepted_turns": sum(r["accepted"] for r in ts),
            "accepted_turn_rate": sum(r["accepted"] for r in ts) / len(ts),
            "episodes_with_multiple_call_blocks": sum(r["multiple_call_blocks"] for r in es),
            "cut_off_turns": sum(r["outcome"] == "turn_token_limit" for r in ts),
            "tool_validation_errors": sum(r["error"] != "" for r in ts),
            "submitted_episodes": sum(r["submitted"] for r in es),
            "correct_answers": sum(r["answer_correct"] for r in es),
            "tool_budget_episodes": sum(r["termination_reason"] == "tool_budget" for r in es),
        })
        for key, label in ERROR_LABELS.items():
            count = sum(r[key] for r in es)
            errors.append({"variant": variant, "error": key, "label": label,
                           "episodes": count, "denominator": len(es), "episode_rate": count / len(es),
                           "turns": sum(r["multiple_call_blocks"] if key == "multiple_call_blocks"
                                        else r["outcome"] == key for r in ts)})
    return summaries, errors, episodes, turns


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(summaries, errors, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": "#14263D", "axes.labelcolor": "#53657B",
                         "xtick.color": "#53657B", "ytick.color": "#14263D",
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    top = fig.add_gridspec(1, 2, left=.055, right=.94, top=.78, bottom=.58, wspace=.35)
    fig.text(.055, .94, "Tool-call validity before and after SFT", fontsize=24, weight="bold")
    fig.text(.055, .89, f"Same {summaries[0]['episodes']} test questions · vanilla vs. supervised fine-tuning · saved trajectories", fontsize=13, color="#53657B")

    for panel, (rate, count, denom, title) in enumerate([
        ("fully_valid_episode_rate", "fully_valid_episodes", "episodes", "A   Episodes with only valid outputs"),
        ("accepted_turn_rate", "accepted_turns", "generated_turns", "B   Accepted generated turns"),
    ]):
        ax = fig.add_subplot(top[0, panel])
        for y, row in enumerate(summaries):
            name, color = VARIANTS[row["variant"]]
            value = row[rate] * 100
            ax.barh(y, 100, height=.42, color="#F1F5F9", zorder=1)
            ax.barh(y, value, height=.42, color=color, zorder=2)
            ax.text(0, y-.31, name, fontsize=12, weight="bold", va="bottom")
            ax.text(min(value+2, 101), y, f"{value:.1f}%  ({row[count]}/{row[denom]})",
                    fontsize=12, va="center", weight="bold", color=color, clip_on=False)
        ax.set(xlim=(0, 133), ylim=(1.55, -.65), yticks=[], xticks=[])
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=22)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax = fig.add_axes([.26, .18, .68, .29])
    for index, (key, label) in enumerate(ERROR_LABELS.items()):
        for j, variant in enumerate(VARIANTS):
            row = next(r for r in errors if r["variant"] == variant and r["error"] == key)
            y = index + (j-.5)*.31
            name, color = VARIANTS[variant]
            value = row["episode_rate"]*100
            ax.barh(y, value, height=.24, color=color, zorder=3)
            ax.text(value+1.0, y, f"{row['episodes']}/{row['denominator']}", color=color,
                    va="center", fontsize=11, weight="bold")
    ax.set(yticks=range(len(ERROR_LABELS)), yticklabels=list(ERROR_LABELS.values()),
           xlim=(0, 82), xticks=[0, 20, 40, 60, 80], xlabel="Episodes affected (%)")
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%"])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E2E8F0", zorder=0)
    ax.tick_params(axis="both", length=0, pad=9)
    fig.text(.055, .50, "C   Which output failures changed after SFT?", fontsize=14, weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=[Patch(facecolor=color, label=name) for name, color in VARIANTS.values()],
              loc="lower right", frameon=False, fontsize=12)
    fig.text(.055, .045,
             "Valid = one completed tool call accepted by the environment. Validity does not imply a correct answer or submission.\n"
             "Failure categories can overlap. Every generated turn counts, including outputs cut off before tool execution.",
             fontsize=10.5, color="#53657B", linespacing=1.6)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"sft_output_validity_comparison.{extension}", dpi=240, facecolor="white")
    plt.close(fig)


def report(summaries, errors, episodes, turns, provenance, source, output):
    summary = {r["variant"]: r for r in summaries}
    paired = {}
    for row in episodes:
        paired.setdefault(row["task_id"], {})[row["variant"]] = row["all_outputs_valid"]
    transitions = Counter((r["vanilla"], r["sft"]) for r in paired.values())
    result = {
        "source": str(source.resolve()), "files": provenance,
        "matched_questions": len(paired), "summary": summaries, "errors": errors,
        "paired_validity": {"improved": transitions[(False, True)], "regressed": transitions[(True, False)],
                            "valid_both": transitions[(True, True)], "invalid_both": transitions[(False, False)]},
        "definitions": {
            "accepted_turn": "A generated turn with a recorded observation whose ok field is true.",
            "fully_valid_episode": "Every generated turn has a successful tool observation; submission and correctness are not required.",
            "single_call_turn": "A completed, parsed call reached the executor, whether or not its arguments were accepted.",
            "multiple_call_blocks": "More than one <tool_call> opening tag in a generated turn, including unexecuted truncated text.",
            "turn_token_limit": "Recorded runtime termination before executing the final generated turn.",
            "error_rates": "Fraction of matched test episodes affected; categories may overlap.",
        },
    }
    (output / "statistics.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, rows in [("summary", summaries), ("errors", errors), ("episodes", episodes), ("turns", turns)]:
        write_csv(output / f"{name}.csv", rows)
    vanilla, sft = summary["vanilla"], summary["sft"]
    vanilla_multi = [t for t in turns if t["variant"] == "vanilla" and t["multiple_call_blocks"]]
    multi_stops = Counter(t["outcome"] for t in vanilla_multi)
    sft_error_turns = [t for t in turns if t["variant"] == "sft" and t["error"]]
    sft_error_counts = Counter(t["outcome"] for t in sft_error_turns)
    affected_sft = len({t["task_id"] for t in sft_error_turns})
    valid_sft_budget = sum(e["variant"] == "sft" and e["all_outputs_valid"]
                           and e["termination_reason"] == "tool_budget" for e in episodes)
    lines = ["# Generated-output validity: vanilla vs. SFT", "",
             f"Paired comparison on the same {len(paired)} test questions, read directly from `{source.name}`. No inference was run.", "",
             "![Output validity comparison](sft_output_validity_comparison.png)", "",
             "| Metric | Vanilla | After SFT |", "| --- | ---: | ---: |"]
    for title, key, denominator in [("Episodes with only valid outputs", "fully_valid_episodes", "episodes"),
                                    ("Completed single-call turns reaching executor", "single_call_turns", "generated_turns"),
                                    ("Accepted generated turns", "accepted_turns", "generated_turns")]:
        cells = [f"{r[key]}/{r[denominator]} ({100*r[key]/r[denominator]:.1f}%)" for r in summaries]
        lines.append(f"| {title} | {' | '.join(cells)} |")
    lines += ["", "Each generated turn counts once, including outputs cut off before tool execution. "
              "The notebook's saved call-validity rate excludes those truncated outputs, so it is not the denominator used in panel B.", "",
              f"The paired change is {transitions[(False, True)]} questions improving from invalid to fully valid, "
              f"{transitions[(True, False)]} regressing, {transitions[(True, True)]} valid in both and "
              f"{transitions[(False, False)]} invalid in both.", "",
              "## Error categories", "", "| Error | Vanilla episodes | SFT episodes | Vanilla turns | SFT turns |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for key, label in ERROR_LABELS.items():
        rows = [next(r for r in errors if r["variant"] == v and r["error"] == key) for v in VARIANTS]
        lines.append(f"| {label} | {rows[0]['episodes']} | {rows[1]['episodes']} | {rows[0]['turns']} | {rows[1]['turns']} |")
    lines += ["", "The categories overlap. Multiple-call counts inspect the raw assistant text; "
              f"they are not limited to the runtime parser's termination labels. Of {len(vanilla_multi)} vanilla multi-call outputs, "
              f"{multi_stops['turn_token_limit']} first hit the generation token limit and "
              f"{multi_stops['multiple_calls_in_turn']} were explicitly rejected by the parser. "
              "The unexecuted blocks were not counted as additional executed calls or separately validated for arguments.", "",
              f"SFT has {sft['episodes_with_multiple_call_blocks']} episodes containing multiple call blocks, "
              f"compared with {vanilla['episodes_with_multiple_call_blocks']} for vanilla. "
              f"Its {len(sft_error_turns)} rejected tool calls affect {affected_sft} episodes and include "
              f"{sft_error_counts['unknown_relation']} unknown-relation errors and "
              f"{sft_error_counts['relation_type_mismatch']} entity-type/relation mismatches.", "",
              "## Interpretation", "",
              f"The fully valid episode rate changes from {100*vanilla['fully_valid_episode_rate']:.1f}% "
              f"to {100*sft['fully_valid_episode_rate']:.1f}% after SFT. "
              f"Of the {sft['fully_valid_episodes']} fully valid SFT episodes, {valid_sft_budget} exhaust "
              f"the tool budget without submitting. SFT has {sft['tool_budget_episodes']} tool-budget endings overall. "
              "A wrong submitted answer can also be a valid tool call. This is a validity comparison, not a replacement for accuracy.", "",
              f"The comparison uses one recorded run and {len(paired)} paired questions. Turn-level rates have different denominators because "
              "SFT sustains longer interactions; episode-level rates provide a complementary comparison on the same questions.", "",
              "## Files", "", "- `summary.csv`: variant-level counts and rates.",
              "- `errors.csv`: episode and turn counts by error category.",
              "- `episodes.csv`: task-level validity flags and outcomes.",
              "- `turns.csv`: each generated turn's observed acceptance or rejection.",
              "- `statistics.json`: definitions, paired transitions and source-file hashes.",
              "- PNG, SVG and PDF: the same slide-ready comparison figure.", "",
              "Reproduce from the project root:", "", "```bash",
              f"python scripts/plot_output_validity.py {source.name}", "```", ""]
    (output / "README.md").write_text("\n".join(lines))
    for r in summaries:
        print(f"{r['variant']}: fully valid episodes {r['fully_valid_episodes']}/{r['episodes']}; "
              f"accepted turns {r['accepted_turns']}/{r['generated_turns']}")
    print(f"Files saved to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Notebook results ZIP or extracted run directory")
    parser.add_argument("--output", type=Path, default=Path("artifacts/output_validity"))
    args = parser.parse_args()
    data, provenance = load_evaluations(args.source)
    summaries, errors, episodes, turns = analyze(data)
    args.output.mkdir(parents=True, exist_ok=True)
    draw(summaries, errors, args.output)
    report(summaries, errors, episodes, turns, provenance, args.source, args.output)


if __name__ == "__main__":
    main()
