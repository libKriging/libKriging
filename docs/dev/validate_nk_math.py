"""Numerical validation of the NK aggregation math implemented in NestedKriging.cpp.

Mirrors predict_nk() line by line (same formulas, same jitter) against an
exact simple-kriging reference. Checks:
  1. p=1  ==> NK == exact simple kriging (mean & variance)
  2. NK interpolates the design points
  3. NK stays close to the exact full GP for p>1
  4. NK variance in [0, sigma2]; reverts to prior far from data
"""
import numpy as np

rng = np.random.default_rng(123)
JITTER = 1e-10


def corr_matern52(X1, X2, theta):
    d = np.abs(X1[:, None, :] - X2[None, :, :]) / theta[None, None, :]
    s = np.sqrt(5.0) * d
    return np.prod((1 + s + s**2 / 3.0) * np.exp(-s), axis=2)


def exact_sk(X, y, Xt, theta, sigma2, beta0):
    R = corr_matern52(X, X, theta) + JITTER * np.eye(len(X))
    r = corr_matern52(X, Xt, theta)
    Ri = np.linalg.inv(R)
    mean = beta0 + r.T @ Ri @ (y - beta0)
    var = sigma2 * (1 - np.einsum("ij,ik,kj->j", r, Ri, r))
    return mean, np.maximum(var, 0)


def nk_predict(X, y, groups, Xt, theta, sigma2, beta0):
    """Transcription of NestedKriging::predict_nk."""
    p, q = len(groups), len(Xt)
    L, alpha = [], []
    for g in groups:
        Rg = corr_matern52(X[g], X[g], theta) + JITTER * np.eye(len(g))
        Lg = np.linalg.cholesky(Rg)
        L.append(Lg)
        alpha.append(np.linalg.solve(Lg.T, np.linalg.solve(Lg, y[g] - beta0)))
    M, Kdiag, W = np.zeros((p, q)), np.zeros((p, q)), []
    for i, g in enumerate(groups):
        C = corr_matern52(X[g], Xt, theta)
        U = np.linalg.solve(L[i], C)
        W.append(np.linalg.solve(L[i].T, U))
        M[i] = beta0 + C.T @ alpha[i]
        Kdiag[i] = (U * U).sum(axis=0)
    cross = np.zeros((p, p, q))
    for i in range(p):
        for j in range(i + 1, p):
            Rij = corr_matern52(X[groups[i]], X[groups[j]], theta)
            cij = (W[i] * (Rij @ W[j])).sum(axis=0)
            cross[i, j] = cross[j, i] = cij
    mean, var = np.zeros(q), np.zeros(q)
    for t in range(q):
        KM = cross[:, :, t].copy()
        np.fill_diagonal(KM, Kdiag[:, t] + JITTER)
        kM = np.diag(KM).copy()
        w = np.linalg.solve(KM, kM)
        mean[t] = beta0 + w @ (M[:, t] - beta0)
        var[t] = sigma2 * max(0.0, 1.0 - w @ kM)
    return mean, var


# --- data --------------------------------------------------------------------
n, d = 128, 2
X = rng.uniform(size=(n, d))
y = np.sin(3 * X[:, 0]) + np.cos(5 * X[:, 1]) + X[:, 0] * X[:, 1]
Xt = rng.uniform(size=(64, d))
theta = np.array([0.35, 0.35])
sigma2 = float(np.var(y))
beta0 = float(np.mean(y))

# --- 1. p=1 equivalence ------------------------------------------------------
m_ref, v_ref = exact_sk(X, y, Xt, theta, sigma2, beta0)
m_nk1, v_nk1 = nk_predict(X, y, [np.arange(n)], Xt, theta, sigma2, beta0)
err_m, err_v = np.abs(m_nk1 - m_ref).max(), np.abs(v_nk1 - v_ref).max()
print(f"[1] p=1 vs exact SK : max|dmean|={err_m:.2e}  max|dvar|={err_v:.2e}")
assert err_m < 1e-8 and err_v < 1e-8

# --- 2. interpolation --------------------------------------------------------
groups = np.array_split(rng.permutation(n), 4)
m_int, v_int = nk_predict(X, y, groups, X, theta, sigma2, beta0)
print(f"[2] interpolation   : max|mean-y|={np.abs(m_int - y).max():.2e}  max stdev={np.sqrt(v_int).max():.2e}")
assert np.abs(m_int - y).max() < 1e-4 and np.sqrt(v_int).max() < 1e-3

# --- 3. accuracy vs exact full GP (p=4) ---------------------------------------
m_nk, v_nk = nk_predict(X, y, groups, Xt, theta, sigma2, beta0)
diff = np.abs(m_nk - m_ref).mean()
print(f"[3] p=4 vs exact GP : mean|dmean|={diff:.2e}  (sd(y)={np.std(y):.2f})")
assert diff < 0.05 * np.std(y)

# --- 4. variance bounds & prior reversion -------------------------------------
assert (v_nk >= 0).all() and (v_nk <= sigma2 * (1 + 1e-9)).all()
Xfar = np.full((5, d), 10.0)
m_far, v_far = nk_predict(X, y, groups, Xfar, theta, sigma2, beta0)
print(f"[4] far point       : var/sigma2={v_far.min()/sigma2:.4f}  |mean-beta0|={np.abs(m_far-beta0).max():.2e}")
assert v_far.min() > 0.99 * sigma2 and np.abs(m_far - beta0).max() < 1e-6

print("\nAll NK math checks passed.")
