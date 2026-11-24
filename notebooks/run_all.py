"""Run all notebooks sequentially; save outputs in place, including failures."""

from pathlib import Path
from importlib.util import find_spec
import sys
import traceback

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
import nbformat
from nbclient import NotebookClient


class CurrentPythonKernelSpecManager(KernelSpecManager):
    """Supply an in-memory kernel spec without requiring kernel registration."""

    def get_kernel_spec(self, kernel_name):
        return KernelSpec(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name="Current Python",
            language="python",
        )


def main():
    if find_spec("ipykernel") is None:
        raise SystemExit(
            "Missing notebook kernel. Install it in this environment with:\n"
            "  python -m pip install ipykernel"
        )
    root = Path(__file__).resolve().parent
    experiments = root / "experiments"
    notebooks = sorted(
        [*experiments.glob("*.ipynb"), *(root / "statistics").glob("*.ipynb")],
        # Run producers before the figure notebooks that load their saved data.
        key=lambda path: ("figures" in path.stem, path.as_posix()),
    )
    if not notebooks:
        raise SystemExit("No notebooks found.")

    failures = []
    for index, path in enumerate(notebooks, 1):
        print(f"[{index}/{len(notebooks)}] {path.relative_to(root)}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None

        # Use this script's Python environment in a fresh kernel for each notebook.
        manager = KernelManager(
            kernel_name="python3", kernel_spec_manager=CurrentPythonKernelSpecManager()
        )
        client = NotebookClient(
            notebook, km=manager, timeout=None,
            resources={"metadata": {"path": str(path.parent)}},
        )
        try:
            client.execute(cleanup_kc=True)
        except Exception:
            failures.append(path)
            traceback.print_exc()
        finally:
            nbformat.write(notebook, path)

    print(f"Finished: {len(notebooks) - len(failures)} passed, {len(failures)} failed.")
    for path in failures:
        print(f"FAILED: {path.relative_to(root)}")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
