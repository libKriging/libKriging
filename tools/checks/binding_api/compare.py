#!/usr/bin/env python3
"""Compares the methods exposed by the Python, R, Octave and Julia bindings
with each other (issue #419).

  compare.py [--report-only] [--markdown FILE] DUMP...

A dump is a text file of "<scope>\\t<name>[\\t<args>]" lines written by the
dump_* script of this directory, the binding being given by its file name
(python.txt, r.txt, octave.txt or julia.txt). Names are matched across
bindings after normalization of the naming conventions: case and "_" are
ignored, as are Julia's trailing "!", the deprecated Julia "get_" prefix of
the model getters and R's trailing "_" (F_, T_): leaveOneOut, leave_one_out
and fit / fit! match. A few names are the same method under another name
(ALIASES).

Every method missing from some bindings, and every binding without a dump,
is an error (exit status 1, a GitHub error annotation when run in Actions);
--report-only makes them warnings (exit status 0).

<args>, when the binding can list them (not Octave, whose methods all take
varargin), are the arguments of the method, the object excluded, separated
by ",": "name" (required), "name=" (with a default) or "*" (*args, "...",
kwargs..., ignored here); a name with several lines has overloads, the one
with the most arguments is compared. The arguments of a method are compared
across the bindings listing them, names normalized as the methods (and
ARG_ALIASES), and every difference is a warning, never an error: argument
missing from some bindings, required in some and optional in others, or
another number of arguments when a binding has no argument names (pybind11
arg0, arg1...). The order is not compared (Julia keyword arguments), nor
the name of the argument when every binding has a single one (setters), nor
the arguments of one binding only by design (ARG_IDIOMS).

The log ends with the synthesis of all the compared methods (one line per
method, then the counts per scope); --markdown writes it too, e.g. to
$GITHUB_STEP_SUMMARY.
"""

import argparse
import collections
import os
import re
import sys

BINDINGS = ("python", "r", "octave", "julia")

# language idioms with no counterpart to expect in the other bindings
IDIOMS = {
    "r": {"print", "show", "str", "as.list"},
    "octave": {"disp"},
    "julia": {"show"},
}

# the same method under another name: {lang: {name or (scope, name): name in
# the other bindings}}
ALIASES = {
    # as.list.Kriging: all the fields of the model (other classes: the base
    # conversion of the object, an idiom)
    "r": {("Kriging", "as.list"): "model"},
    "julia": {
        "is_normalize": "normalize",  # deprecated getter
        # static load(filename) of each class (load(filename) dispatches on the file)
        "load_kriging": "load",
        "load_warp_kriging": "load",
        "load_mlp_kriging": "load",
        "load_nested_kriging": "load",
    },
}

# the same argument under another name: {lang: {argument: name in the other bindings}}
ARG_ALIASES = {
    "julia": {"X_n": "X"},  # the prediction points
}

# arguments of one binding only, by design: {lang: {method: arguments}}
ARG_IDIOMS = {
    "julia": {
        # keyword shortcuts of the entries of `parameters`
        "fit!": {"sigma2", "is_sigma2_estim", "theta", "is_theta_estim", "beta", "is_beta_estim",
                 "nugget", "is_nugget_estim", "warp_params"},
        # Kriging nugget model: with_nugget=b is with_noise=b (Python / R)
        "simulate": {"with_nugget"},
    },
}

# scopes of the model classes, whose Julia getters had a "get_" prefix
MODELS = ("Kriging", "WarpKriging", "MLPKriging", "NestedKriging")


def normalize(lang, scope, name):
    aliases = ALIASES.get(lang, {})
    name = aliases.get((scope, name), aliases.get(name, name))
    if lang == "julia":
        name = name.rstrip("!")
        if scope in MODELS:
            name = re.sub(r"^get_", "", name)
    return name.replace("_", "").lower()


def normalize_arg(lang, arg):
    return ARG_ALIASES.get(lang, {}).get(arg, arg).replace("_", "").lower()


def read_dump(path):
    """-> ({scope: names}, {(scope, name): [args]}), args a list of (arg, required)"""
    dump = collections.defaultdict(set)
    sigs = collections.defaultdict(list)
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) not in (2, 3) or not fields[1]:
                sys.exit("%s:%d: expected '<scope>\\t<name>[\\t<args>]', got %r" % (path, n, line))
            scope, name = fields[:2]
            dump[scope].add(name)
            if len(fields) == 3:
                args = [a for a in fields[2].split(",") if a and a != "*"]
                sigs[(scope, name)].append([(a.rstrip("="), not a.endswith("=")) for a in args])
    return dump, sigs


def compare(dumps):
    """dumps: {lang: {scope: names}} -> (langs, table, gaps)

    table: {scope: {key: {lang: [names]}}}; gaps: list of messages."""
    langs = [lang for lang in BINDINGS if lang in dumps]
    table = collections.defaultdict(lambda: collections.defaultdict(dict))
    for lang in langs:
        for scope, names in dumps[lang].items():
            for name in names:
                if name in IDIOMS.get(lang, ()) and (scope, name) not in ALIASES.get(lang, {}):
                    continue
                table[scope][normalize(lang, scope, name)].setdefault(lang, []).append(name)
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


def compare_args(langs, table, sigs):
    """-> ({(scope, key): "ok" | "differ"}, warnings), for the methods whose
    arguments at least two bindings list"""
    status, warnings = {}, []
    for scope in sorted(table):
        for key in sorted(table[scope]):
            entry = table[scope][key]
            chosen = {}
            for lang in langs:
                overloads = [args for name in entry.get(lang, []) for args in sigs[lang].get((scope, name), [])]
                if overloads:
                    chosen[lang] = max(overloads, key=len)
            if len(chosen) < 2:
                continue
            anonymous = {lang for lang, args in chosen.items() if args and all(re.match(r"^arg\d+$", a) for a, _ in args)}
            # a single argument in every binding (setters): its name is not compared
            if all(len(args) == 1 for args in chosen.values()):
                anonymous = set(chosen)
            idioms = {lang: set().union(*[ARG_IDIOMS.get(lang, {}).get(name, set()) for name in entry.get(lang, [])])
                      for lang in chosen}
            named = {lang: {normalize_arg(lang, a): (a, req) for a, req in args if a not in idioms[lang]}
                     for lang, args in chosen.items() if lang not in anonymous}
            problems = []
            union = sorted({a for args in named.values() for a in args})
            for a in union:
                lacking = [lang for lang in named if a not in named[lang]]
                spelled = "/".join(sorted({named[lang][a][0] for lang in named if a in named[lang]}))
                if lacking:
                    problems.append("%s missing in %s" % (spelled, ", ".join(lacking)))
                    continue
                required = sorted(lang for lang in named if named[lang][a][1])
                if required and len(required) < len(named):
                    problems.append("%s required in %s only" % (spelled, ", ".join(required)))
            if len(anonymous) == len(chosen):
                if len({args[0][1] for args in chosen.values()}) > 1:
                    problems.append("the argument is required in %s only"
                                    % ", ".join(sorted(lang for lang, args in chosen.items() if args[0][1])))
            for lang in sorted(anonymous):
                counts = {len(named[other]) for other in named}
                if counts and len(chosen[lang]) not in counts:
                    problems.append("%d arguments in %s, %s elsewhere"
                                    % (len(chosen[lang]), lang, "/".join(str(c) for c in sorted(counts))))
            status[(scope, key)] = "differ" if problems else "ok"
            for p in problems:
                warnings.append("%s.%s arguments: %s" % (scope, key, p))
    return status, warnings


def cell(lang, scope, entry, dumps, missing):
    if lang in entry:
        return " ".join(sorted(entry[lang]))
    return missing if scope in dumps[lang] else ""


def counts(langs, table, dumps, status):
    """[(scope, methods, complete, missing per lang, args differ)] and the totals row"""
    rows = []
    for scope in sorted(table):
        having = [lang for lang in langs if scope in dumps[lang]]
        entries = table[scope].values()
        complete = sum(all(lang in entry for lang in having) for entry in entries)
        missing = [sum(lang not in entry for entry in entries) if lang in having else None for lang in langs]
        differ = sum(status.get((scope, key)) == "differ" for key in table[scope])
        rows.append((scope, len(entries), complete, missing, differ))
    total = ("total", sum(r[1] for r in rows), sum(r[2] for r in rows),
             [sum(r[3][i] or 0 for r in rows) for i in range(len(langs))], sum(r[4] for r in rows))
    return rows, total


def text(langs, table, dumps, status):
    """the synthesis printed at the end of the log: one line per method
    ("-": missing, blank: scope not exposed; args: "ok", "differ" or blank
    when not compared), then the counts per scope"""
    header = ["scope", "method"] + list(langs) + ["args"]
    lines = [[scope, key] + [cell(lang, scope, table[scope][key], dumps, "-") for lang in langs]
             + [status.get((scope, key), "")]
             for scope in sorted(table) for key in sorted(table[scope])]
    rows, total = counts(langs, table, dumps, status)
    summary_header = ["scope", "methods", "complete"] + ["missing " + lang for lang in langs] + ["args differ"]
    summary = [[scope, str(n), str(c)] + ["" if m is None else str(m) for m in missing] + [str(d)]
               for scope, n, c, missing, d in rows + [total]]

    def render(header, lines):
        widths = [max(len(row[i]) for row in [header] + lines) for i in range(len(header))]
        fmt = lambda row: "  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip()
        return [fmt(header), fmt(["-" * w for w in widths])] + [fmt(row) for row in lines]

    return "\n".join(render(header, lines) + [""] + render(summary_header, summary)) + "\n"


def markdown(langs, table, dumps, status):
    out = ["## Binding methods (%s)" % ", ".join(langs), ""]
    rows, total = counts(langs, table, dumps, status)
    out.append("| scope | methods | complete | " + " | ".join("missing in " + lang for lang in langs)
               + " | arguments differ |")
    out.append("|---" * (4 + len(langs)) + "|")
    for scope, n, c, missing, d in rows + [total]:
        bold = "**%s**" if scope == "total" else "%s"
        cells = [bold % scope, str(n), str(c)] + ["" if m is None else ("**%d**" % m if m else "0") for m in missing]
        out.append("| " + " | ".join(cells + [str(d)]) + " |")
    for scope in sorted(table):
        out.append("\n### %s\n" % scope)
        out.append("| " + " | ".join(langs) + " | arguments |")
        out.append("|---" * (len(langs) + 1) + "|")
        for key in sorted(table[scope]):
            entry = table[scope][key]
            cells = []
            for lang in langs:
                if lang in entry:
                    cells.append(" ".join("`%s`" % n for n in sorted(entry[lang])))
                else:
                    cells.append("**—**" if scope in dumps[lang] else "")
            cells.append({"differ": "**differ**"}.get(status.get((scope, key)), status.get((scope, key), "")))
            out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-only", action="store_true",
                        help="report the missing methods as warnings, exit status 0")
    parser.add_argument("--markdown", help="write the correspondence table to this file (appended)")
    parser.add_argument("dumps", nargs="+")
    args = parser.parse_args(argv)

    dumps, sigs = {}, {}
    for path in args.dumps:
        lang = os.path.splitext(os.path.basename(path))[0].lower()
        if lang not in BINDINGS:
            sys.exit("%s: file name must be one of %s" % (path, ", ".join(b + ".txt" for b in BINDINGS)))
        dumps[lang], sigs[lang] = read_dump(path)
    langs, table, gaps = compare(dumps)
    gaps += ["%s: no dump, binding not compared" % lang for lang in BINDINGS if lang not in dumps]
    status, arg_warnings = compare_args(langs, table, sigs)

    if args.markdown:
        with open(args.markdown, "a") as f:
            f.write(markdown(langs, table, dumps, status))
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for gap in gaps:
        print(("::%s::" % ("warning" if args.report_only else "error") if in_actions else "") + gap)
    for warning in arg_warnings:
        print(("::warning::" if in_actions else "") + warning)
    print("\n" + text(langs, table, dumps, status))
    print("%d difference(s) between the bindings (%s), %d argument difference(s) (warnings)"
          % (len(gaps), ", ".join(langs), len(arg_warnings)))
    return 1 if gaps and not args.report_only else 0


if __name__ == "__main__":
    sys.exit(main())
