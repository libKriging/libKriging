# Lists the names exposed by pylibkriging, one "<scope>\t<name>" line each,
# for tools/checks/binding_api.py, which compares them to bindings/api.yaml.
# Written to $LK_API_DUMP_DIR/python.txt when that variable is set (ctest sets
# it to <build>/binding-api), only checked for non-emptiness otherwise.
import inspect
import os

import pylibkriging as lk

# scope in bindings/api.yaml -> object exposing it
SCOPES = {
    "Kriging": lk.Kriging,
    "WarpKriging": lk.WarpKriging,
    "MLPKriging": lk.MLPKriging,
    "NestedKriging": lk.NestedKriging,
    "Optim": lk.Optim,
}


def _public(names):
    return sorted(n for n in names if not n.startswith("_"))


def api_dump():
    lines = []
    for name in _public(dir(lk)):
        if not inspect.ismodule(getattr(lk, name)):
            lines.append(("module", name))
    for scope, obj in SCOPES.items():
        lines.extend((scope, name) for name in _public(dir(obj)))
    return lines


def test_api_dump():
    lines = api_dump()
    assert all(any(s == scope for s, _ in lines) for scope in SCOPES)
    out_dir = os.environ.get("LK_API_DUMP_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "python.txt"), "w") as f:
            f.write("".join("%s\t%s\n" % line for line in lines))
