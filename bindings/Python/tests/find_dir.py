# prefer pathlib over os.path when you can
# https://towardsdatascience.com/dont-use-python-os-library-any-more-when-pathlib-can-do-141fefb6bdb5
# https://docs.python.org/3/library/pathlib.html
from pathlib import Path
import sys


def find_reference_dir():
    path = Path.cwd()
    found = False
    # while (! is.null(path) and !found):
    while not found:
        testpath = path / ".git" / ".." / "tests" / "references"
        if testpath.exists():
            return testpath
        else:
            parent = path.parent
            if parent == path:
                print("Cannot find reference test directory", file=sys.stderr)
                sys.exit(1)
            path = parent
