import numpy as np
import pylibkriging as lk


def test_trivial_np():
    xs = np.arange(12)
    print(xs)

    print(lk.add_arrays(xs, xs))


if __name__ == '__main__':
    test_trivial_np()
