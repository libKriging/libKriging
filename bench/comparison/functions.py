"""Classic response surfaces for surrogate benchmarking.

All functions accept X of shape (n, d) in their *native* domain and return
y of shape (n,). Domains are given by `DOMAINS[name]` as (d, 2) arrays.
References: Surjanovic & Bingham, https://www.sfu.ca/~ssurjano/
"""
import numpy as np

def branin(X):
    x1, x2 = X[:, 0], X[:, 1]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

_H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]], float)
_H3_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470],
                         [1091, 8732, 5547], [381, 5743, 8828]], float)

def hartmann3(X):
    diff2 = (X[:, None, :] - _H3_P[None, :, :]) ** 2          # (n, 4, 3)
    expo = np.exp(-np.sum(_H3_A[None, :, :] * diff2, axis=2))  # (n, 4)
    return -expo @ _H3_ALPHA

_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]], float)
_H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]], float)

def hartmann6(X):
    diff2 = (X[:, None, :] - _H6_P[None, :, :]) ** 2           # (n, 4, 6)
    expo = np.exp(-np.sum(_H6_A[None, :, :] * diff2, axis=2))  # (n, 4)
    return -expo @ _H6_ALPHA

def borehole(X):
    rw, r, Tu, Hu, Tl, Hl, L, Kw = (X[:, i] for i in range(8))
    lnr = np.log(r / rw)
    return (2 * np.pi * Tu * (Hu - Hl)) / (
        lnr * (1 + 2 * L * Tu / (lnr * rw**2 * Kw) + Tu / Tl))

FUNCTIONS = {"branin": branin, "hartmann3": hartmann3,
             "hartmann6": hartmann6, "borehole": borehole}

DOMAINS = {
    "branin": np.array([[-5.0, 10.0], [0.0, 15.0]]),
    "hartmann3": np.array([[0.0, 1.0]] * 3),
    "hartmann6": np.array([[0.0, 1.0]] * 6),
    "borehole": np.array([[0.05, 0.15], [100.0, 50000.0], [63070.0, 115600.0],
                          [990.0, 1110.0], [63.1, 116.0], [700.0, 820.0],
                          [1120.0, 1680.0], [9855.0, 12045.0]]),
}
