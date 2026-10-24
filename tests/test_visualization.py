import json
import re
import pytest

from kg_rlvr.visualization import inspection_bundle, render_visualization, visualization_payload


def test_graph_overview_and_limits(graph, tmp_path):
    path = tmp_path / "overview.html"
    data = render_visualization(graph, path, max_nodes=30, max_edges=40)
    assert 1 < len(data["nodes"]) <= 30
    assert len(data["edges"]) <= 40
    document = path.read_text()
    assert "https://" not in document and 'src="http' not in document
    assert json.loads(re.search(r'id="data">(.*?)</script>', document, re.S).group(1)) == data
    assert "pointerdown" in document and "wheel" in document


def test_neighborhood_filters(graph):
    data = visualization_payload(graph, entity="researcher_000000", radius=1, entity_types=["university"], relation_types=["works_at"])
    assert {n["type"] for n in data["nodes"]} == {"researcher", "university"}
    assert all(e["relation"] == "works_at" for e in data["edges"])


@pytest.mark.parametrize("hop", [2, 3, 4])
def test_question_highlighting_and_all_answers(graph, questions, hop):
    q = next(q for q in questions if q["hop_count"] == hop)
    data = visualization_payload(graph, question=q)
    nodes = {n["id"]: n for n in data["nodes"]}
    assert nodes[q["start_entity"]]["role"] == "start"
    assert set(q["answer_ids"]).issubset(nodes)
    assert {p["answer"] for p in data["paths"]} == set(q["answer_ids"])
    assert all(len(p["entities"]) == hop + 1 for p in data["paths"])
    assert all(path in [p["entities"] for p in data["paths"]] for path in q["gold_entity_paths"])
    assert sum(e["gold"] for e in data["edges"]) >= hop


def test_reference_paths_not_silently_cut(graph, questions):
    with pytest.raises(ValueError, match="Reference paths require"):
        visualization_payload(graph, question=questions[0], max_nodes=1)
    with pytest.raises(ValueError, match="Reference paths require"):
        visualization_payload(graph, question=questions[0], max_edges=1)


def test_invalid_filters(graph):
    with pytest.raises(ValueError, match="Unknown entity type"):
        visualization_payload(graph, entity_types=["not-a-type"])


def test_inspection_bundle_all_modes(graph, questions, tmp_path):
    selected = inspection_bundle(graph, questions, tmp_path)
    assert {q["hop_count"] for q in selected} == {2, 3, 4}
    for q in selected:
        assert (tmp_path / q["file"]).exists()
        assert q["file"] in (tmp_path / "index.html").read_text()
    assert (tmp_path / "graph_overview.html").exists()
