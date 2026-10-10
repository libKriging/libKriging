import os
import re

import numpy as np
import pylibkriging as m
import pytest


def _expected_version():
    """Read the version from the single source of truth (cmake/version.cmake)
    so this test does not need updating on every release."""
    here = os.path.dirname(os.path.abspath(__file__))
    vfile = os.path.normpath(
        os.path.join(here, "..", "..", "..", "..", "cmake", "version.cmake"))
    with open(vfile) as f:
        data = f.read()

    def part(key):
        return re.search(r"^set\(KRIGING_VERSION_%s (\d+)\)$" % key, data, re.M).group(1)

    return "%s.%s.%s" % (part("MAJOR"), part("MINOR"), part("PATCH"))


def test_version():
    assert m.__version__ == _expected_version()


def test_generic_load_dispatches_classes():
    X = np.linspace(0.01, 0.99, 8).reshape(-1, 1)
    y = 1 - 0.5 * (np.sin(12 * X[:, 0]) / (1 + X[:, 0]) + 2 * np.cos(7 * X[:, 0]) * X[:, 0] ** 5 + 0.7)
    filenames = ["loading_test_k.json", "loading_test_wk.json", "loading_test_mlp.json",
                 "loading_test_nuk.json", "loading_test_nok.json", "loading_test_nk.json"]

    try:
        k = m.Kriging(y, X, "gauss")
        k.save(filenames[0])
        assert isinstance(m.load(filenames[0]), m.Kriging)

        wk = m.WarpKriging(y, X, ["kumaraswamy"], "gauss")
        wk.save(filenames[1])
        assert isinstance(m.load(filenames[1]), m.WarpKriging)

        mk = m.MLPKriging(y, X, [8, 4], 2, "selu", "gauss")
        mk.save(filenames[2])
        assert isinstance(m.load(filenames[2]), m.MLPKriging)

        # Kriging with a nugget / noise channel: same class, described as
        # NuggetKriging / NoiseKriging by the loader
        nuk = m.Kriging(y, X, "gauss", noise="nugget")
        nuk.save(filenames[3])
        assert isinstance(m.load(filenames[3]), m.Kriging)

        nok = m.Kriging(y, X, "gauss", noise=np.full(len(y), 0.01))
        nok.save(filenames[4])
        assert isinstance(m.load(filenames[4]), m.Kriging)

        X2 = np.random.default_rng(1).uniform(size=(40, 2))
        y2 = np.sin(3 * X2[:, 0]) + X2[:, 1]
        nk = m.NestedKriging(y2, X2, "gauss", 2)
        nk.save(filenames[5])
        assert isinstance(m.load(filenames[5]), m.NestedKriging)
    finally:
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)


def test_pickle_roundtrip_all_classes():
    import pickle

    rng = np.random.default_rng(3)
    X = rng.uniform(size=(40, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    Xt = rng.uniform(size=(7, 2))
    models = [
        m.Kriging(y, X, "gauss"),
        m.WarpKriging(y, X, ["kumaraswamy", "none"], "gauss"),
        m.MLPKriging(y, X, [4], 2, "selu", "gauss"),
        m.NestedKriging(y, X, "gauss", 2),
    ]
    for k in models:
        k2 = pickle.loads(pickle.dumps(k))
        assert type(k2) is type(k)
        np.testing.assert_array_equal(k2.theta(), k.theta())
        if isinstance(k, m.NestedKriging):
            p1, p2 = k.predict(Xt, True), k2.predict(Xt, True)
        else:
            p1, p2 = k.predict(Xt, True, False, False)[:2], k2.predict(Xt, True, False, False)[:2]
        for a, b in zip(p1, p2):
            assert a.shape == (7,)  # vector outputs are 1-D for every class
            np.testing.assert_allclose(b, a, rtol=0, atol=1e-12)
