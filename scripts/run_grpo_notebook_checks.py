"""Execute non-GPU notebook cells; never start a long model-training experiment."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--notebook", default=str(root / "notebooks/02_grpo_training_kaggle.ipynb"))
    parser.add_argument("--output", help="Saved executed notebook; defaults to input path")
    args = parser.parse_args()
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("The CPU-check runner refuses a CUDA host because RUN_GPU may be enabled. Use Jupyter and explicit notebook switches for a real GPU smoke/training run.")
    with tempfile.TemporaryDirectory(prefix="kg_grpo_notebook_") as runtime:
        for key, suffix in [("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter"),
                            ("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "cache")]:
            os.environ.setdefault(key, str(Path(runtime) / suffix))
        os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        notebook = nbformat.read(args.notebook, as_version=4)
        parameter_cells = [c for c in notebook.cells if "parameters" in c.metadata.get("tags", [])]
        if len(parameter_cells) != 1 or "INSTALL_DEPENDENCIES = False" not in parameter_cells[0].source:
            raise RuntimeError("CPU preflight requires the reviewed configuration cell with dependency installation disabled")
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=300, allow_errors=False,
                                resources={"metadata": {"path": str(root)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        output = Path(args.output or args.notebook)
        nbformat.write(notebook, output)
        errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
        print(json.dumps({"notebook": str(output), "executed_code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                          "error_outputs": len(errors), "mode": "CPU preflight; GPU tests UNVERIFIED",
                          "size_bytes": output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
