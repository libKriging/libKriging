import numpy as np
import pylibkriging as lk
import pytest


@pytest.debug_only
def test_trivial_np():
    xs = np.arange(12)
    print(xs)

    print(lk.add_arrays(xs, xs))


if __name__ == '__main__':
    test_trivial_np()
