"""Regression checks for fresh Kaggle packages, before model/backend imports."""

import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

nbformat = pytest.importorskip("nbformat")
ROOT = Path(__file__).resolve().parents[1]


def setup_namespace(versions, kaggle=True):
    def version(name):
        if versions.get(name) is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]
    return {"INSTALL_DEPENDENCIES": False, "IS_KAGGLE": kaggle, "RUN_GPU": True,
        "PROJECT_ROOT": ROOT, "MODEL_NAME": "Qwen/Qwen3-1.7B", "MODEL_REVISION_OVERRIDE": None,
        "sys": sys, "json": json, "importlib": SimpleNamespace(metadata=SimpleNamespace(
            version=version, PackageNotFoundError=importlib.metadata.PackageNotFoundError))}


def setup_code():
    notebook = nbformat.read(ROOT / "notebooks/02_grpo_training_kaggle.ipynb", 4)
    return next(c.source for c in notebook.cells if c.cell_type == "code" and "setup_mismatches" in c.source)


def test_fresh_kaggle_stops_in_setup_before_importing_backend():
    # Reported Kaggle image: several different APIs and no TRL/bitsandbytes.
    namespace = setup_namespace({"torch": "2.11.0+cu128", "transformers": "5.16.1",
        "datasets": "4.8.5", "peft": "0.20.0", "accelerate": "1.14.0"})
    with pytest.raises(RuntimeError) as raised:
        exec(setup_code(), namespace)
    message = str(raised.value)
    assert "section 1" in message and "INSTALL_DEPENDENCIES=True" in message
    assert "restart" in message and "Stop after section 6" in message
    assert namespace["setup_mismatches"]["trl"]["actual"] is None
    assert namespace["setup_mismatches"]["bitsandbytes"]["actual"] is None
    assert "MODEL_REVISIONS" not in namespace  # stopped before the backend import


def test_matching_kaggle_package_metadata_passes_setup():
    versions = {}
    for line in (ROOT / "requirements-grpo.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, pinned = line.split("==", 1)
            versions[name] = pinned.split(";", 1)[0].strip()
    versions["torch"] += "+cu124"
    namespace = setup_namespace(versions)
    exec(setup_code(), namespace)
    assert namespace["setup_mismatches"] == {}
    assert namespace["MODEL_REVISION"]


def test_local_cpu_preflight_does_not_require_gpu_packages():
    namespace = setup_namespace({}, kaggle=False)
    exec(setup_code(), namespace)
    assert "setup_mismatches" not in namespace
    assert namespace["MODEL_REVISION"]
