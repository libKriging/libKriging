import numpy as np
import pylibkriging as lk

if __name__ == '__main__':
    xs = np.arange(12)
    print(xs)

    print(lk.add_arrays(xs, xs))
