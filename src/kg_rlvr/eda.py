"""Presentation figures computed from the saved graph and question files."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath
import networkx as nx
import numpy as np
import pandas as pd

from .data import read_questions

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CBD5E1",
    "text.color": "#0F172A",
    "axes.labelcolor": "#475569",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}
COLORS = {1: "#0D9488", 2: "#3B82F6", 3: "#8B5CF6", 4: "#D97706"}


def question_tables(graph, data_root):
    rows = []
    for record in read_questions(data_root):
        rows.append(
            {
                **record,
                "answer_count": len(record["answer_ids"]),
                "instance": (
                    record["start_entity"],
                    tuple(record["relation_path"]),
                    tuple(sorted(record["answer_ids"])),
                ),
                "start_type": graph.entities[record["start_entity"]]["type"],
                "relation_options": record["difficulty"]["relation_options_mean"],
                "reference_calls": record["difficulty"]["reference_tool_calls"],
            }
        )
    source = pd.DataFrame(rows)
    eligible = source[source.answer_count == 1].copy()
    unique = eligible.drop_duplicates(["split", "instance"]).copy()
    return source, eligible, unique


def graph_statistics(graph):
    network = nx.Graph()
    network.add_nodes_from(graph.entities)
    network.add_edges_from((head, tail) for head, _, tail in graph.triples)
    return pd.Series(
        {
            "Entities": len(graph.entities),
            "Stored navigation facts": len(graph.triples),
            "Entity types": len({e["type"] for e in graph.entities.values()}),
            "Relation types": len(graph.relations),
            "Weakly connected components": nx.number_connected_components(network),
            "Isolated entities": len(list(nx.isolates(network))),
        },
        name="Saved graph",
    )


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            directory / f"{name}.{extension}", dpi=240, bbox_inches="tight", facecolor="white"
        )


def style_axis(ax, grid="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color="#E2E8F0", linewidth=0.8)
    ax.tick_params(length=0, pad=8)


def ontology_figure(graph):
    """Eight class nodes and all stored relation types, read from Turtle."""
    counts = Counter(entity["type"] for entity in graph.entities.values())
    positions = {
        "researcher": (0, 3),
        "project": (3, 3),
        "publication": (6, 3),
        "field": (9, 3),
        "university": (0, 0),
        "funder": (3, 0),
        "city": (6, 0),
        "country": (9, 0),
    }
    # Curves separate the long relations and the two researcher/project directions.
    routes = {
        "works_at": (0, (-0.25, 1.5)),
        "led_by": (0, (1.5, 2.77)),
        "contributes_to": (-0.30, (1.5, 3.55)),
        "authored_by": (0.50, (3, 4.55)),
        "investigates": (-0.28, (6, 3.90)),
        "funded_by": (0, (3.12, 1.5)),
        "located_in_city": (0.27, (3, -0.85)),
        "city_in_country": (0, (7.5, -0.20)),
        "produced_by": (0, (4.5, 2.77)),
        "publication_field": (0, (7.5, 2.77)),
        "based_in": (0, (4.5, -0.20)),
        "specializes_in": (-0.55, (4.5, 5.45)),
    }
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.84)
    fig.suptitle("The research-world ontology", fontsize=25, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.91,
        f"{len(positions)} entity types · {len(graph.relations)} relation types · counts show saved instances",
        ha="center",
        color="#64748B",
        fontsize=13,
    )
    boxes = {}
    for kind, (x, y) in positions.items():
        fill = "#E0F2FE" if kind in {"city", "country", "university"} else "#CCFBF1"
        box = FancyBboxPatch(
            (x - 0.72, y - 0.31),
            1.44,
            0.62,
            boxstyle="round,pad=.08,rounding_size=.13",
            facecolor=fill,
            edgecolor="#94A3B8",
            linewidth=1.1,
            zorder=4,
        )
        boxes[kind] = box
        ax.add_patch(box)
        ax.text(
            x,
            y + 0.08,
            kind.title(),
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=12,
            zorder=5,
        )
        ax.text(
            x,
            y - 0.15,
            f"{counts[kind]:,} entities",
            ha="center",
            va="center",
            fontsize=10,
            color="#475569",
            zorder=5,
        )

    for name, spec in graph.relations.items():
        head, tail = spec["domain"], spec["range"]
        x, y = positions[head]
        if head == tail:
            if head in {"researcher", "university"}:
                vertices = [
                    (x - 0.65, y + 0.20),
                    (x - 2.0, y + 1),
                    (x - 2.0, y - 1),
                    (x - 0.65, y - 0.20),
                ]
                label = (x - 1.65, y)
            elif head == "field":
                vertices = [
                    (x + 0.65, y + 0.20),
                    (x + 2, y + 1),
                    (x + 2, y - 1),
                    (x + 0.65, y - 0.20),
                ]
                label = (x + 1.65, y)
            else:
                vertices = [
                    (x - 0.35, y - 0.27),
                    (x - 1, y - 1.65),
                    (x + 1, y - 1.65),
                    (x + 0.35, y - 0.27),
                ]
                label = (x, y - 1.1)
            path = PlotPath(vertices, [PlotPath.MOVETO] + [PlotPath.CURVE4] * 3)
            arrow = FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=13,
                color="#64748B",
                linewidth=1.3,
                zorder=1,
            )
        else:
            radius, label = routes[name]
            arrow = FancyArrowPatch(
                positions[head],
                positions[tail],
                patchA=boxes[head],
                patchB=boxes[tail],
                connectionstyle=f"arc3,rad={radius}",
                arrowstyle="-|>",
                mutation_scale=13,
                color="#64748B",
                linewidth=1.3,
                zorder=1,
            )
        ax.add_patch(arrow)
        ax.text(
            *label,
            name,
            ha="center",
            va="center",
            fontsize=10,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
            zorder=3,
        )
    ax.set(xlim=(-2.35, 11.25), ylim=(-1.4, 6))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(
        0.5,
        0.035,
        "Arrows show stored predicates between classes. Permitted inverse traversal uses inv:<relation>.\n"
        "Self-loops connect different instances of the same class; the full schema remains in graph.ttl.",
        ha="center",
        fontsize=11,
        color="#64748B",
    )
    return fig


def composition_figure(graph):
    entities = pd.Series(Counter(e["type"] for e in graph.entities.values())).sort_values()
    relations = pd.Series(Counter(r for _, r, _ in graph.triples)).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 1.45]})
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.11, top=0.79, wspace=0.55)
    fig.suptitle("What is inside the graph?", fontsize=24, fontweight="bold", y=0.96)
    fig.text(
        0.5,
        0.90,
        f"{len(graph.entities):,} entities · {len(graph.triples):,} stored navigation facts · fictional research ecosystem",
        ha="center",
        color="#64748B",
    )
    for ax, values, title, color in zip(
        axes,
        [entities, relations],
        ["Entities by type", "Facts by relation"],
        ["#0D9488", "#3B82F6"],
    ):
        bars = ax.barh(values.index, values.values, color=color, height=0.66)
        ax.bar_label(bars, labels=[f"{n:,}" for n in values.values], padding=5, fontsize=10)
        ax.set_title(title, loc="left", fontweight="bold", pad=15)
        ax.set_xlim(0, values.max() * 1.22)
        ax.set_xlabel("Count")
        style_axis(ax)
    return fig


def questions_figure(source, eligible, unique):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 7.5))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.17, top=0.78, wspace=0.28)
    fig.suptitle(
        "From saved questions to training instances", fontsize=23, fontweight="bold", y=0.96
    )
    fig.text(
        0.5,
        0.90,
        "Single-answer filtering is followed by paraphrase deduplication",
        ha="center",
        color="#64748B",
    )
    counts = pd.crosstab(source.hop_count, source.split).reindex([1, 2, 3, 4])
    x = np.arange(4)
    for offset, split, color in [(-0.19, "train", "#0D9488"), (0.19, "test", "#3B82F6")]:
        bars = ax0.bar(x + offset, counts[split], 0.34, color=color, label=split.title())
        ax0.bar_label(bars, padding=4, fontsize=11)
    ax0.set_xticks(x, [f"{h} hop" for h in [1, 2, 3, 4]])
    ax0.set(title="Source questions by hop", ylabel="Question rows")
    ax0.set_ylim(0, counts.max().max() * 1.20)
    ax0.legend(frameon=False)
    stages = [source, eligible, unique]
    labels = ["Source\nquestions", "Single-answer\nquestions", "Unique query\ninstances"]
    for offset, split, color in [(-0.19, "train", "#0D9488"), (0.19, "test", "#3B82F6")]:
        values = [int((frame.split == split).sum()) for frame in stages]
        bars = ax1.bar(np.arange(3) + offset, values, 0.34, color=color, label=split.title())
        ax1.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4, fontsize=11)
    ax1.set_xticks(range(3), labels)
    ax1.set(title="Eligibility and deduplication", ylabel="Count")
    ax1.set_ylim(0, len(source[source.split == "train"]) * 1.20)
    for ax in (ax0, ax1):
        style_axis(ax, "y")
    fig.text(
        0.5,
        0.04,
        "Training uses hops 1–3. Evaluation includes hops 1–4. Counts here describe all saved data, before experiment caps.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    return fig


def navigation_figure(unique, tool_budget=12):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 7.5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.78, wspace=0.28)
    fig.suptitle("What makes navigation difficult?", fontsize=24, fontweight="bold", y=0.96)
    fig.text(
        0.5,
        0.90,
        "Single-answer query instances can still offer distracting relations and branches",
        ha="center",
        color="#64748B",
    )
    hops = [1, 2, 3, 4]
    values = [unique.loc[unique.hop_count == h, "relation_options"].to_numpy() for h in hops]
    boxes = ax0.boxplot(
        values,
        labels=[f"{h} hop" for h in hops],
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops={"color": "#0F172A", "linewidth": 2},
    )
    for box, h in zip(boxes["boxes"], hops):
        box.set(facecolor=COLORS[h], alpha=0.55, edgecolor="none")
    ax0.set(
        title="Relation choices along reference routes",
        ylabel="Mean available operations per visited entity",
    )
    bars = ax1.bar(hops, [2 * h + 1 for h in hops], color=[COLORS[h] for h in hops], width=0.55)
    ax1.bar_label(bars, padding=5, fontsize=13, fontweight="bold")
    ax1.axhline(
        tool_budget,
        color="#DC5A5A",
        linestyle="--",
        linewidth=2,
        label=f"Episode budget: {tool_budget}",
    )
    ax1.set_xticks(hops, [f"{h} hop" for h in hops])
    ax1.set(title="Reference tool calls", ylabel="Calls per episode", ylim=(0, tool_budget + 2))
    ax1.legend(frameon=False, loc="upper left")
    for ax in (ax0, ax1):
        style_axis(ax, "y")
    fig.text(
        0.5,
        0.045,
        "Reference protocol: get_relations + get_neighbors at each hop, followed by submit_answer.\n"
        "Actual model trajectories can skip relation lookups, repeat actions, explore branches or stop early.",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    return fig
