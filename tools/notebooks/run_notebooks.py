#!/usr/bin/env python3
"""Execute Jupyter notebooks and fail if any cell raises.

Used by CI (.github/workflows/notebooks.yml) to keep the worked notebooks
runnable, and to refresh their stored outputs (`--inplace`).

The binding notebooks start with an "Installation" section of `%%bash` cells that
create a virtual environment, install requirements and build libKriging. Those
cells provision an environment and are not what the notebooks demonstrate: with
`--skip-setup` no `%%bash` cell (nor the cell that puts a built mlibkriging on the
Octave path) is run: they keep their stored content, no local path ends up in the
outputs, and the notebook is executed against the libKriging already available to
the kernel (Python: installed; Julia: `--project`; Octave: `OCTAVE_PATH`).

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


# Python notebooks: the installation cells are `%%bash` cells. Octave notebooks: one cell puts
# `build/installed` on the path, and errors out when libKriging is not there.
SETUP_PREFIXES = ("%%bash", "% Add mlibkriging to path")


def is_setup_cell(cell):
    return cell.cell_type == "code" and cell.source.lstrip().startswith(SETUP_PREFIXES)


def output_errors(nb):
    """Octave's kernel reports a failing statement as plain text ("error: ..."), not as an error output."""
    if nb.metadata.get("kernelspec", {}).get("language") != "octave":
        return []
    found = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            for line in out.get("text", "").splitlines():
                if line.startswith("error: "):
                    found.append(line.strip())
    return found


def run(path, kernel, timeout, skip_setup, inplace):
    nb = nbformat.read(path, as_version=4)
    work = copy.deepcopy(nb)
    skipped = {i for i, cell in enumerate(nb.cells) if skip_setup and is_setup_cell(cell)}
    kept = [i for i in range(len(nb.cells)) if i not in skipped]
    work.cells = [work.cells[i] for i in kept]  # the skipped cells are not sent to the kernel

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

    errors = output_errors(work)
    if errors:
        return False, "\n".join(sorted(set(errors))[:10]), time.time() - start

    if inplace:
        # Merge the outputs back, leaving the skipped setup cells as they were.
        for i, done in zip(kept, work.cells):
            if done.cell_type == "code":
                nb.cells[i].outputs = done.outputs
                nb.cells[i].execution_count = done.execution_count
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
