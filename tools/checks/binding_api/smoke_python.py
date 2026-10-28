# Smoke calls of pylibkriging (issue #435): each family of methods is called
# once on a tiny model (n = 10, d = 2) with the argument types a user would
# pass (an int and a float), and the settings of Optim / LinearAlgebra are
# checked by a get / set / get round trip (smoke_setters.tsv). Prints one
# "ok" / "FAIL" line per call, exits with status 1 on any failure.
# Usage: python3 smoke_python.py [<markdown file to append to>]
import os
import sys
import tempfile
import traceback

import numpy as np
import pylibkriging as lk

HERE = os.path.dirname(os.path.abspath(__file__))
failures, calls = [], 0


class Skip(Exception):
    """a call that needs arguments: not a getter"""


def check(label, fn):
    global calls
    calls += 1
    try:
        fn()
        print("ok    " + label)
    except Skip:
        calls -= 1
        print("skip  " + label)
    except Exception as e:  # noqa: BLE001 - any error is a finding
        failures.append(label)
        print("FAIL  %s: %s: %s" % (label, type(e).__name__, str(e).splitlines()[0] if str(e) else ""))
        if os.environ.get("SMOKE_TRACEBACK"):
            traceback.print_exc()


def same(a, b):
    return bool(np.all(np.isclose(a, b, rtol=1e-12, atol=0)))


def setter_roundtrip(scope, setter, getter, kind, value):
    cls = getattr(lk, scope)
    if not hasattr(cls, setter):
        raise AttributeError("%s.%s not exposed" % (scope, setter))
    if getter == "-":
        getattr(cls, setter)(value.lower() == "true")
        return
    get = getattr(cls, getter)
    old = get()
    new = {"int": lambda: int(value), "float": lambda: float(value), "bool": lambda: not old}[kind]()
    try:
        getattr(cls, setter)(new)
        assert same(get(), new), "set %r then get %r" % (new, get())
    finally:
        getattr(cls, setter)(old)
    assert same(get(), old), "restore %r then get %r" % (old, get())


def settings():
    with open(os.path.join(HERE, "smoke_setters.tsv")) as f:
        for line in f:
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                yield line.split("\t")


# zero-argument members of a fitted model, called whatever their name
def getters(label, model, skip=()):
    for name in sorted(n for n in dir(model) if not n.startswith("_") and n not in skip):
        member = getattr(model, name)
        if not callable(member):
            continue

        def call(member=member):
            try:
                member()
            except TypeError as e:
                if "incompatible function arguments" not in str(e) and "required positional" not in str(e):
                    raise  # a type error inside a call without arguments
                raise Skip()

        check("%s.%s()" % (label, name), call)


rng = np.random.default_rng(1)
X = rng.uniform(size=(10, 2))
y = np.sin(3 * X[:, 0]) + X[:, 1]
Xn = rng.uniform(size=(3, 2))
Xu, yu = rng.uniform(size=(2, 2)), np.array([0.5, 0.7])
tmp = tempfile.mkdtemp()

for scope, setter, getter, kind, value in settings():
    check("%s.%s" % (scope, setter), lambda a=(scope, setter, getter, kind, value): setter_roundtrip(*a))

models = {}


def update_simulate(model):
    copy = model.copy()
    copy.simulate(2, 123, Xn, True)  # update_simulate needs the simulation to update
    copy.update_simulate(yu, Xu)



def build(label, make):
    def run():
        models[label] = make()
    check("%s fit" % label, run)


build("Kriging", lambda: lk.Kriging(y, X, "gauss"))
build("WarpKriging", lambda: (lambda m: (m.fit(y, X), m)[1])(lk.WarpKriging(["none", "none"], "gauss")))
build("MLPKriging", lambda: (lambda m: (m.fit(y, X), m)[1])(lk.MLPKriging([2], 2, "selu", "gauss")))
build("NestedKriging", lambda: lk.NestedKriging(y, X, "gauss", 2))

for label, model in models.items():
    getters(label, model, skip={"fit", "save", "copy", "load", "update", "update_simulate", "simulate"})
    check("%s.predict" % label, lambda m=model: m.predict(Xn, True))
    if label != "NestedKriging":
        check("%s.simulate" % label, lambda m=model: m.simulate(2, 123, Xn))
        check("%s.update" % label, lambda m=model: m.copy().update(yu, Xu))
        check("%s.update_simulate" % label, lambda m=model: update_simulate(m))
    path = os.path.join(tmp, label + ".h5")
    check("%s.save/load" % label, lambda m=model, p=path: (m.save(p), lk.load(p)))

k = models.get("Kriging")
if k is not None:
    check("Kriging.logLikelihood", lambda: k.logLikelihood())
    check("Kriging.leaveOneOut", lambda: k.leaveOneOut())
    check("Kriging.logMargPost", lambda: k.logMargPost())
    check("Kriging.covMat", lambda: k.covMat(Xn, Xn))
    check("Kriging.logLikelihoodFun", lambda: k.logLikelihoodFun(np.array([0.5, 0.5]), True, False))
    check("Kriging.leaveOneOutFun", lambda: k.leaveOneOutFun(np.array([0.5, 0.5]), True, False))
    check("Kriging.logMargPostFun", lambda: k.logMargPostFun(np.array([0.5, 0.5]), True, False))
    check("Kriging.leaveOneOutVec", lambda: k.leaveOneOutVec(np.array([0.5, 0.5])))
    check("Kriging.subsetOfData", lambda: lk.Kriging.subsetOfData(X, 5))

print("\n%d call(s), %d failure(s)" % (calls, len(failures)))
if len(sys.argv) > 1:
    with open(sys.argv[1], "a") as f:
        f.write("\n## Smoke calls: Python\n\n%d call(s), **%d failure(s)**\n\n" % (calls, len(failures)))
        f.writelines("- `%s`\n" % label for label in failures)
sys.exit(1 if failures else 0)
