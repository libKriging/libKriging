#!/usr/bin/env python3
"""Execute Jupyter notebooks and fail if any cell raises.

Used by CI (.github/workflows/notebooks.yml) to keep the worked notebooks
runnable, and to refresh their stored outputs (`--inplace`).

The binding notebooks start with an "Installation" section of `%%bash` cells that
create a virtual environment, install requirements and build libKriging. Those
cells provision an environment and are not what the notebooks demonstrate: with
`--skip-setup` no `%%bash` cell is run (they keep their stored content, and no
local path ends up in the outputs), and the notebook is executed against the
libKriging already installed in the kernel's environment.

Usage:
  python3 tools/notebooks/run_notebooks.py --skip-setup bindings/Python/*.ipynb
  python3 tools/notebooks/run_notebooks.py --skip-setup --inplace --kernel python3 nb.ipynb

Requires nbclient, nbformat and ipykernel.
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def is_setup_cell(cell):
    """The installation cells of the binding notebooks are all `%%bash` cells."""
    return cell.cell_type == "code" and cell.source.lstrip().startswith("%%bash")


def run(path, kernel, timeout, skip_setup, inplace):
    nb = nbformat.read(path, as_version=4)
    work = copy.deepcopy(nb)
    skipped = set()
    if skip_setup:
        for i, cell in enumerate(work.cells):
            if is_setup_cell(cell):
                skipped.add(i)
                cell.source = "pass"  # keeps the numbering of the cells

    client = NotebookClient(
        work,
        kernel_name=kernel or nb.metadata.get("kernelspec", {}).get("name", "python3"),
        timeout=timeout,
        resources={"metadata": {"path": str(path.parent)}},
        record_timing=False,
    )
    start = time.time()
    try:
        client.execute()
    except CellExecutionError as exc:
        return False, f"{exc}".strip()[-1500:], time.time() - start

    if inplace:
        # Merge the outputs back, leaving the skipped setup cells as they were.
        for i, (orig, done) in enumerate(zip(nb.cells, work.cells)):
            if orig.cell_type == "code" and i not in skipped:
                orig.outputs = done.outputs
                orig.execution_count = done.execution_count
        nbformat.write(nb, path)
    return True, f"{len(skipped)} setup cell(s) skipped" if skipped else "", time.time() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--kernel", help="kernel name to use instead of the notebook's own kernelspec")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds (default 1800)")
    parser.add_argument("--skip-setup", action="store_true", help="do not run the %%bash installation cells")
    parser.add_argument("--inplace", action="store_true", help="write the outputs back into the notebooks")
    args = parser.parse_args(argv)

    failures = []
    for path in args.notebooks:
        ok, info, seconds = run(path, args.kernel, args.timeout, args.skip_setup, args.inplace)
        print(f"{'ok  ' if ok else 'FAIL'} {path} ({seconds:.0f}s){': ' + info if info else ''}", flush=True)
        if not ok:
            failures.append(path)
    if failures:
        print(f"\n{len(failures)} notebook(s) failed:", *map(str, failures), sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
