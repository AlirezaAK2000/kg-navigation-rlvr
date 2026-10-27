"""Execute the EDA notebook with this interpreter, preserving read-only inputs."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=str(root / "notebooks/01_dataset_eda.ipynb"))
    parser.add_argument("--data", default=str(root / "data"))
    parser.add_argument(
        "--output", help="Executed notebook path; defaults to overwriting the notebook"
    )
    parser.add_argument(
        "--output-base", help="Writable directory for statistical figures and derived reports"
    )
    args = parser.parse_args()
    # Avoid writing Jupyter state into a read-only user home or input mount.
    with tempfile.TemporaryDirectory(prefix="kg_eda_runtime_") as runtime:
        for key, suffix in [
            ("IPYTHONDIR", "ipython"),
            ("JUPYTER_RUNTIME_DIR", "jupyter"),
            ("MPLCONFIGDIR", "matplotlib"),
            ("XDG_CACHE_HOME", "cache"),
        ]:
            os.environ.setdefault(key, str(Path(runtime) / suffix))
        os.environ["KG_RLVR_PROJECT_ROOT"] = str(root)
        os.environ["KG_RLVR_DATA_DIR"] = str(Path(args.data).resolve())
        if args.output_base:
            os.environ["KG_RLVR_EDA_OUTPUT_DIR"] = str(Path(args.output_base).resolve())
        os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager

        notebook = nbformat.read(args.notebook, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=300,
            allow_errors=False,
            resources={"metadata": {"path": str(root)}},
        )
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        output = Path(args.output or args.notebook)
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
        errors = [
            out
            for cell in notebook.cells
            if cell.cell_type == "code"
            for out in cell.get("outputs", [])
            if out.output_type == "error"
        ]
        png_outputs = sum(
            "image/png" in out.get("data", {})
            for cell in notebook.cells
            if cell.cell_type == "code"
            for out in cell.get("outputs", [])
        )
        print(
            json.dumps(
                {
                    "executed_notebook": str(output),
                    "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                    "error_outputs": len(errors),
                    "inline_png_outputs": png_outputs,
                    "size_bytes": output.stat().st_size,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
