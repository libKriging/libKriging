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
    filenames = ["loading_test_k.json", "loading_test_wk.json", "loading_test_mlp.json"]

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
    finally:
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)
