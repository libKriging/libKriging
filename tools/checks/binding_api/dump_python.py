# Lists the names exposed by pylibkriging, one "<scope>\t<name>[\t<args>]"
# line each (one per overload), for compare.py, which compares the bindings
# with each other. <args>: the arguments, self excluded, "name" (required) or
# "name=" (with a default) separated by ",", "*" for *args / **kwargs.
# Usage: python3 dump_python.py <output file>
import inspect
import re
import sys

import pylibkriging as lk

# scope -> object exposing it
SCOPES = {
    "Kriging": lk.Kriging,
    "WarpKriging": lk.WarpKriging,
    "MLPKriging": lk.MLPKriging,
    "NestedKriging": lk.NestedKriging,
    "Optim": lk.Optim,
    "LinearAlgebra": lk.LinearAlgebra,
}


def public(names):
    return sorted(n for n in names if not n.startswith("_"))


def split_top(s):
    """splits s on the commas outside brackets and quotes"""
    parts, depth, quote, start = [], 0, None, 0
    for i, c in enumerate(s):
        if quote:
            if c == quote and s[i - 1] != "\\":
                quote = None
        elif c in "'\"":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return [p.strip() for p in parts if p.strip()]


def pybind_signatures(name, doc):
    """the argument lists of the pybind11 signature lines "name(a: T = v, ...) -> R" of doc"""
    sigs = []
    for line in (doc or "").splitlines():
        m = re.match(r"^(?:\d+\. )?%s\((.*)\)(?: -> .*)?$" % re.escape(name), line.strip())
        if not m:
            continue
        args = []
        for p in split_top(m.group(1)):
            if p.startswith("*"):
                args.append("*")
                continue
            arg = p.split(":")[0].split("=")[0].strip()
            if arg == "self":
                continue
            args.append(arg + ("=" if "=" in p else ""))
        sigs.append(args)
    return sigs


def python_signature(f):
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return None
    args = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            args.append("*")
        else:
            args.append(p.name + ("" if p.default is p.empty else "="))
    return args


def signatures(obj, name):
    f = getattr(obj, name)
    if not callable(f):
        return []
    sigs = pybind_signatures(name, getattr(f, "__doc__", None))
    if sigs or type(f).__name__ in ("builtin_function_or_method", "instancemethod"):
        return sigs
    sig = python_signature(f)
    return [sig] if sig is not None else []


lines = []
for scope, obj in SCOPES.items():
    for n in public(dir(obj)):
        sigs = signatures(obj, n)
        if sigs:
            lines.extend("%s\t%s\t%s" % (scope, n, ",".join(args)) for args in sigs)
        else:
            lines.append("%s\t%s" % (scope, n))

with open(sys.argv[1], "w") as f:
    f.write("".join(line + "\n" for line in lines))
