# Lists the names exposed by pylibkriging, one "<scope>\t<name>" line each,
# for check.py, which compares them to bindings/api.yaml.
# Usage: python3 dump_python.py <output file>
import inspect
import sys

import pylibkriging as lk

# scope in bindings/api.yaml -> object exposing it
SCOPES = {
    "Kriging": lk.Kriging,
    "WarpKriging": lk.WarpKriging,
    "MLPKriging": lk.MLPKriging,
    "NestedKriging": lk.NestedKriging,
    "Optim": lk.Optim,
}


def public(names):
    return sorted(n for n in names if not n.startswith("_"))


lines = [("module", n) for n in public(dir(lk)) if not inspect.ismodule(getattr(lk, n))]
for scope, obj in SCOPES.items():
    lines.extend((scope, n) for n in public(dir(obj)))

with open(sys.argv[1], "w") as f:
    f.write("".join("%s\t%s\n" % line for line in lines))
