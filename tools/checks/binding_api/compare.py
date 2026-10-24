#!/usr/bin/env python3
"""Compares the methods exposed by the Python, R, Octave and Julia bindings
with each other (issue #419).

  compare.py [--report-only] [--markdown FILE] DUMP...

A dump is a text file of "<scope>\\t<name>" lines written by the dump_*
script of this directory, the binding being given by its file name
(python.txt, r.txt, octave.txt or julia.txt). Names are matched across
bindings after normalization of the naming conventions: case and "_" are
ignored, as are Julia's trailing "!", the deprecated Julia "get_" prefix and
R's trailing "_" (F_, T_): leaveOneOut, leave_one_out and fit / fit! match.

Every method missing from some bindings, and every binding without a dump,
is an error (exit status 1, a GitHub error annotation when run in Actions);
--report-only makes them warnings (exit status 0). --markdown writes the
correspondence table, e.g. to $GITHUB_STEP_SUMMARY.
"""

import argparse
import collections
import os
import re
import sys

BINDINGS = ("python", "r", "octave", "julia")

# language idioms with no counterpart to expect in the other bindings
IDIOMS = {
    "r": {"print", "show", "as.list"},
    "octave": {"disp"},
    "julia": {"show"},
}


def normalize(lang, name):
    if lang == "julia":
        name = name.rstrip("!")
        name = re.sub(r"^get_", "", name)
    return name.replace("_", "").lower()


def read_dump(path):
    dump = collections.defaultdict(set)
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            scope, sep, name = line.partition("\t")
            if not sep or not name:
                sys.exit("%s:%d: expected '<scope>\\t<name>', got %r" % (path, n, line))
            dump[scope].add(name)
    return dump


def compare(dumps):
    """dumps: {lang: {scope: names}} -> (table, gaps)

    table: {scope: {key: {lang: [names]}}}; gaps: list of messages."""
    langs = [lang for lang in BINDINGS if lang in dumps]
    table = collections.defaultdict(lambda: collections.defaultdict(dict))
    for lang in langs:
        for scope, names in dumps[lang].items():
            for name in names:
                if name in IDIOMS.get(lang, ()):
                    continue
                table[scope][normalize(lang, name)].setdefault(lang, []).append(name)
    gaps = []
    for scope in sorted(table):
        having = [lang for lang in langs if scope in dumps[lang]]
        for lang in langs:
            if lang not in having:
                gaps.append("%s: %s not exposed (%s)" % (lang, scope, ", ".join(having)))
        for key in sorted(table[scope]):
            entry = table[scope][key]
            missing = [lang for lang in having if lang not in entry]
            if missing:
                found = ", ".join("%s: %s" % (lang, "/".join(sorted(entry[lang]))) for lang in langs if lang in entry)
                gaps.append("%s.%s missing in %s (%s)" % (scope, key, ", ".join(missing), found))
    return langs, table, gaps


def markdown(langs, table, dumps):
    out = ["## Binding methods (%s)" % ", ".join(langs)]
    for scope in sorted(table):
        out.append("\n### %s\n" % scope)
        out.append("| " + " | ".join(langs) + " |")
        out.append("|---" * len(langs) + "|")
        for key in sorted(table[scope]):
            entry = table[scope][key]
            cells = []
            for lang in langs:
                if lang in entry:
                    cells.append(" ".join("`%s`" % n for n in sorted(entry[lang])))
                else:
                    cells.append("**—**" if scope in dumps[lang] else "")
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-only", action="store_true",
                        help="report the missing methods as warnings, exit status 0")
    parser.add_argument("--markdown", help="write the correspondence table to this file (appended)")
    parser.add_argument("dumps", nargs="+")
    args = parser.parse_args(argv)

    dumps = {}
    for path in args.dumps:
        lang = os.path.splitext(os.path.basename(path))[0].lower()
        if lang not in BINDINGS:
            sys.exit("%s: file name must be one of %s" % (path, ", ".join(b + ".txt" for b in BINDINGS)))
        dumps[lang] = read_dump(path)
    langs, table, gaps = compare(dumps)
    gaps += ["%s: no dump, binding not compared" % lang for lang in BINDINGS if lang not in dumps]

    if args.markdown:
        with open(args.markdown, "a") as f:
            f.write(markdown(langs, table, dumps))
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for gap in gaps:
        print(("::%s::" % ("warning" if args.report_only else "error") if in_actions else "") + gap)
    print("%d difference(s) between the bindings (%s)" % (len(gaps), ", ".join(langs)))
    return 1 if gaps and not args.report_only else 0


if __name__ == "__main__":
    sys.exit(main())
