import pytest
from kg_rlvr.data import load_tasks


@pytest.fixture(scope="session")
def tasks():
    return load_tasks("data", max_tool_calls=12)
