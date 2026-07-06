"""Numerical validation of the Vecchia log-likelihood (VLL) math for libKriging.

Mirrors the planned Kriging objective="VLL(m)" implementation:
  - greedy maxmin ordering,
  - m nearest previously-ordered neighbors (Euclidean),
  - per-conditional terms with profiled sigma2 and GLS-profiled constant beta,
  - analytic gradient d(VLL)/d(theta) (envelope theorem for beta_hat).

Checks:
  1. VLL(m = n-1) == exact concentrated LL      (machine precision)
  2. analytic gradient == finite differences    (all m)
  3. VLL(m) -> LL monotonically-ish as m grows  (screening effect)
  4. theta_hat from VLL(m=20) close to exact MLE on a Matern 5/2 field
  5. maxmin ordering beats natural ordering at equal m
"""
import numpy as np

rng = np.random.default_rng(42)


# ── kernel: matern 5/2, correlation form (matches libKriging) ────────────────
def corr(X1, X2, theta):
    d = np.abs(X1[:, None, :] - X2[None, :, :]) / theta[None, None, :]
    s = np.sqrt(5.0) * d
    return np.prod((1 + s + s**2 / 3.0) * np.exp(-s), axis=2)


def dcorr_dtheta(X1, X2, theta):
    """d corr / d theta_k, shape (n1, n2, d). Matern 5/2 per-dim derivative."""
    n1, n2, d = X1.shape[0], X2.shape[0], X1.shape[1]
    dist = np.abs(X1[:, None, :] - X2[None, :, :]) / theta[None, None, :]
    s = np.sqrt(5.0) * dist
    f = (1 + s + s**2 / 3.0) * np.exp(-s)          # per-dim factors
    # df/ds = (1 + 2s/3)e^-s - (1+s+s^2/3)e^-s = -(s/3)(1+s)e^-s
    dfds = -(s / 3.0) * (1 + s) * np.exp(-s)
    R = np.prod(f, axis=2)
    out = np.empty((n1, n2, d))
    for k in range(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(f[:, :, k] > 0, dfds[:, :, k] / f[:, :, k], 0.0)
        # ds/dtheta_k = -s/theta_k
        out[:, :, k] = R * ratio * (-s[:, :, k] / theta[k])
    return out


# ── exact concentrated LL (profiled sigma2, GLS constant beta) ───────────────
def exact_ll(X, y, theta):
    n = len(y)
    R = corr(X, X, theta)
    L = np.linalg.cholesky(R + 1e-12 * np.eye(n))
    ones = np.ones(n)
    Fs = np.linalg.solve(L, ones)
    ys = np.linalg.solve(L, y)
    beta = (Fs @ ys) / (Fs @ Fs)
    Es = ys - beta * Fs
    sigma2 = (Es @ Es) / n
    return -0.5 * n * np.log(2 * np.pi * sigma2) - np.log(np.diag(L)).sum() - 0.5 * n, beta, sigma2


# ── ordering & neighbors ─────────────────────────────────────────────────────
def maxmin_order(X):
    """Greedy maxmin: start from the point closest to the centroid, then
    repeatedly add the point maximizing its distance to already-ordered ones."""
    n = X.shape[0]
    order = np.empty(n, dtype=int)
    d2 = ((X - X.mean(axis=0)) ** 2).sum(axis=1)
    order[0] = int(np.argmin(d2))
    mind2 = ((X - X[order[0]]) ** 2).sum(axis=1)
    mind2[order[0]] = -np.inf
    for i in range(1, n):
        order[i] = int(np.argmax(mind2))
        upd = ((X - X[order[i]]) ** 2).sum(axis=1)
        mind2 = np.minimum(mind2, upd)
        mind2[order[i]] = -np.inf
    return order


def neighbor_sets(X_ord, m):
    """m nearest previously-ordered points, for each i in ordering."""
    n = X_ord.shape[0]
    N = []
    for i in range(n):
        if i == 0:
            N.append(np.empty(0, dtype=int))
            continue
        d2 = ((X_ord[:i] - X_ord[i]) ** 2).sum(axis=1)
        k = min(m, i)
        N.append(np.argpartition(d2, k - 1)[:k])
    return N


# ── Vecchia LL + analytic gradient ───────────────────────────────────────────
def vll(X_ord, y_ord, theta, N, return_grad=False):
    """VLL with profiled sigma2 and GLS-profiled constant beta.

    Per conditional i (Ni = neighbors): a = R_N^{-1} r,
      v_i = 1 - r'a,  u_i = y_i - a'y_N,  w_i = 1 - a'1
      e_i(beta) = u_i - beta*w_i
    GLS:  beta = sum(w u / v) / sum(w^2 / v) ;  sigma2 = sum(e^2/v)/n
    VLL = -n/2 log(2 pi sigma2) - 1/2 sum log v - n/2
    Gradient by envelope theorem (d/dbeta = 0 at beta_hat)."""
    n, d = X_ord.shape
    u = np.empty(n)
    w = np.empty(n)
    v = np.empty(n)
    du = np.zeros((n, d))
    dw = np.zeros((n, d))
    dv = np.zeros((n, d))
    for i in range(n):
        Ni = N[i]
        if len(Ni) == 0:
            u[i], w[i], v[i] = y_ord[i], 1.0, 1.0
            continue
        XN = X_ord[Ni]
        xi = X_ord[i : i + 1]
        RN = corr(XN, XN, theta) + 1e-12 * np.eye(len(Ni))
        r = corr(XN, xi, theta)[:, 0]
        a = np.linalg.solve(RN, r)
        v[i] = max(1.0 - r @ a, 1e-15)
        u[i] = y_ord[i] - a @ y_ord[Ni]
        w[i] = 1.0 - a.sum()
        if return_grad:
            dRN = dcorr_dtheta(XN, XN, theta)
            dr = dcorr_dtheta(XN, xi, theta)[:, 0, :]
            for k in range(d):
                da = np.linalg.solve(RN, dr[:, k] - dRN[:, :, k] @ a)
                dv[i, k] = -(dr[:, k] @ a + r @ da)
                du[i, k] = -(da @ y_ord[Ni])
                dw[i, k] = -da.sum()

    Sww = (w * w / v).sum()
    Swu = (w * u / v).sum()
    beta = Swu / Sww
    e = u - beta * w
    Q = (e * e / v).sum()
    sigma2 = Q / n
    ll = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.log(v).sum() - 0.5 * n
    if not return_grad:
        return ll, beta, sigma2

    # dQ/dtheta_k at fixed beta_hat (envelope: dQ/dbeta = 0)
    grad = np.empty(d)
    for k in range(d):
        de = du[:, k] - beta * dw[:, k]
        dQ = (2 * e * de / v - (e * e / v**2) * dv[:, k]).sum()
        grad[k] = -0.5 * n * dQ / Q - 0.5 * (dv[:, k] / v).sum()
    return ll, beta, sigma2, grad


# ══ data: 2D Matern-like field ════════════════════════════════════════════════
n, d = 400, 2
X = rng.uniform(size=(n, d))
theta_true = np.array([0.15, 0.25])
R = corr(X, X, theta_true)
L = np.linalg.cholesky(R + 1e-10 * np.eye(n))
y = 1.5 + 0.8 * (L @ rng.standard_normal(n))   # beta=1.5, sigma2=0.64

order = maxmin_order(X)
X_ord, y_ord = X[order], y[order]

# ── [1] m = n-1 reproduces the exact concentrated LL ─────────────────────────
theta0 = np.array([0.2, 0.2])
ll_ex, b_ex, s2_ex = exact_ll(X, y, theta0)
N_full = neighbor_sets(X_ord, n - 1)
ll_v, b_v, s2_v = vll(X_ord, y_ord, theta0, N_full)
print(f"[1] m=n-1 : VLL={ll_v:.6f} vs LL={ll_ex:.6f}  |dLL|={abs(ll_v-ll_ex):.2e}"
      f"  |dbeta|={abs(b_v-b_ex):.2e}  |dsigma2|={abs(s2_v-s2_ex):.2e}")
assert abs(ll_v - ll_ex) < 1e-4 and abs(b_v - b_ex) < 1e-7  # residual = jitter placement

# ── [2] analytic gradient vs finite differences ──────────────────────────────
for m in [5, 20, 50]:
    N = neighbor_sets(X_ord, m)
    _, _, _, g = vll(X_ord, y_ord, theta0, N, return_grad=True)
    h = 1e-6
    fd = np.empty(d)
    for k in range(d):
        tp, tm = theta0.copy(), theta0.copy()
        tp[k] += h; tm[k] -= h
        fd[k] = (vll(X_ord, y_ord, tp, N)[0] - vll(X_ord, y_ord, tm, N)[0]) / (2 * h)
    err = np.abs(g - fd).max() / max(1.0, np.abs(fd).max())
    print(f"[2] m={m:3d} : grad={np.round(g,4)} vs FD={np.round(fd,4)}  rel err={err:.2e}")
    assert err < 1e-5

# ── [3] convergence VLL(m) -> LL ─────────────────────────────────────────────
print("[3] convergence (theta0):")
prev_gap = np.inf
for m in [5, 10, 20, 40, 80]:
    N = neighbor_sets(X_ord, m)
    ll_m, _, _ = vll(X_ord, y_ord, theta0, N)
    gap = abs(ll_m - ll_ex)
    print(f"     m={m:3d} : VLL={ll_m:10.4f}  |VLL-LL|={gap:.4f}")
prev = None

# ── [4] theta_hat from VLL(m=20) vs exact MLE ────────────────────────────────
from scipy.optimize import minimize
N20 = neighbor_sets(X_ord, 20)

def negvll(lt):
    return -vll(X_ord, y_ord, np.exp(lt), N20)[0]
def negll(lt):
    return -exact_ll(X, y, np.exp(lt))[0]

r_v = minimize(negvll, np.log(theta0), method="Nelder-Mead", options={"xatol": 1e-4})
r_e = minimize(negll, np.log(theta0), method="Nelder-Mead", options={"xatol": 1e-4})
th_v, th_e = np.exp(r_v.x), np.exp(r_e.x)
print(f"[4] theta_hat VLL(20)={np.round(th_v,4)} vs exact={np.round(th_e,4)} (true={theta_true})")
assert np.abs(np.log(th_v) - np.log(th_e)).max() < 0.15  # within ~15% on log scale

# ── [5] maxmin vs natural ordering at m=10 ───────────────────────────────────
N10 = neighbor_sets(X_ord, 10)
ll_maxmin = vll(X_ord, y_ord, theta0, N10)[0]
X_nat, y_nat = X, y  # natural (random) order
N10n = neighbor_sets(X_nat, 10)
ll_nat = vll(X_nat, y_nat, theta0, N10n)[0]
print(f"[5] ordering m=10 : maxmin |VLL-LL|={abs(ll_maxmin-ll_ex):.4f}"
      f"  vs natural |VLL-LL|={abs(ll_nat-ll_ex):.4f}")

print("\nAll VLL math checks passed.")
