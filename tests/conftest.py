import copy
import pytest

from kg_rlvr.config import DEFAULT
from kg_rlvr.graph_generator import generate_graph
from kg_rlvr.query_generator import generate_candidates
from kg_rlvr.dataset_builder import split_candidates


@pytest.fixture(scope="session")
def config():
    result = copy.deepcopy(DEFAULT)
    result["graph"].update(entity_count=240, triple_count=800)
    result["questions"]["split_counts"] = {"train": 42, "validation": 6, "test": 12}
    return result


@pytest.fixture(scope="session")
def graph(config):
    return generate_graph(config)


@pytest.fixture(scope="session")
def candidates(graph, config):
    return generate_candidates(graph, config)[0]


@pytest.fixture(scope="session")
def questions(graph, candidates, config):
    return split_candidates(graph, candidates, config)[0]
