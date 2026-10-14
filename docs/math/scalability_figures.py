#!/usr/bin/env python3
"""Comparative figures of libKriging's large-n methods (docs/math/images/).

One figure per method (exact, vecchia, nystrom, nested, subsetofdata), each
made of three panels on the same 2D design:

  (a) input space  -- which points a given point x_i is coupled to;
  (b) structure    -- what is actually factorized / kept of the n x n
                      correlation matrix R, in the method's own ordering;
  (c) error        -- |R_approx - R|, where R_approx is the covariance the
                      fitted objective implicitly assumes; same color scale
                      (square-root) and same spatial ordering (k-means
                      clusters) for every method, so that rows compare.

plus one matrix figure stacking the five rows (scalability_matrix.png), and
a block diagram + timing of the Woodbury identity used by Nystrom
(nystrom_woodbury.png; its timings depend on the machine).

The selection rules mirror the C++ implementation (src/lib/Kriging.cpp,
src/lib/LinearAlgebra.cpp) but are re-implemented here in numpy, so that the
figures can be regenerated without a libKriging build:

  - vecchia : greedy maxmin ordering starting from the point closest to the
              centroid, m nearest previously-ordered neighbors;
  - nystrom : k landmarks from greedy pivoted Cholesky on a reference kernel
              whose range is the data extent; R_approx = U U^T + diag(1 -
              diag(U U^T)) (unit diagonal, as in _logLikelihoodNystrom);
  - nested  : k-means partition in p groups, block-diagonal R at fit time
              (the NK aggregation re-couples the groups at predict time,
              which a fit-time covariance cannot show);
  - subsetOfData : k-means centroids snapped to the nearest data point.

Sizes (n, m, k, p, n_max) are deliberately small so that the matrix
structures stay readable; they are not the library defaults.

Usage:  python3 docs/math/scalability_figures.py [output_dir]
Requires numpy and matplotlib.
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402

# --- setup -----------------------------------------------------------------
N = 160          # design size
THETA = 0.15     # Matern 5/2 range (both dimensions)
M = 8            # vecchia neighbors
K = 16           # nystrom landmarks
P = 6            # nested groups
N_MAX = 40       # subsetOfData kept points
SEED = 123

# Sequential one-hue ramps (blue for correlation, orange for error), light->dark.
CMAP_R = LinearSegmentedColormap.from_list(
    "r", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CMAP_ERR = LinearSegmentedColormap.from_list(
    "err", ["#ffffff", "#fbdccd", "#f4a983", "#eb6834", "#b8461c", "#7a2a0c"])
CMAP_R.set_bad("#e6e6e3")
CMAP_ERR.set_bad("#e6e6e3")
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#2b2b29"
MUTED = "#9a9a94"
HILITE = "#eb6834"


def matern52(D, theta=THETA):
    h = np.sqrt(5.0) * D / theta
    return (1.0 + h + h * h / 3.0) * np.exp(-h)


def cdist(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))


def corr(X, theta=THETA):
    return matern52(cdist(X, X), theta)


def kmeans(X, p, rng, iters=50):
    C = X[rng.choice(len(X), p, replace=False)]
    for _ in range(iters):
        lab = cdist(X, C).argmin(1)
        C = np.array([X[lab == g].mean(0) if np.any(lab == g) else C[g] for g in range(p)])
    return cdist(X, C).argmin(1), C


# --- methods -----------------------------------------------------------------
def maxmin_order(X):
    n = len(X)
    order = [int(((X - X.mean(0)) ** 2).sum(1).argmin())]
    mind = ((X - X[order[0]]) ** 2).sum(1)
    mind[order[0]] = -np.inf
    for _ in range(1, n):
        nxt = int(mind.argmax())
        order.append(nxt)
        mind = np.where(np.isinf(mind), mind, np.minimum(mind, ((X - X[nxt]) ** 2).sum(1)))
        mind[nxt] = -np.inf
    return np.array(order)


def vecchia(X, m):
    """Ordering, neighbors (in ordered positions) and implied covariance."""
    order = maxmin_order(X)
    Xo = X[order]
    n = len(X)
    Ro = corr(Xo)
    nbrs = [np.array([], dtype=int)]
    for t in range(1, n):
        d = ((Xo[:t] - Xo[t]) ** 2).sum(1)
        nbrs.append(np.sort(np.argsort(d)[: min(m, t)]))
    # Q = B^T D^-1 B, B unit lower triangular with -kriging weights
    B = np.eye(n)
    Dv = np.ones(n)
    for t in range(1, n):
        Nb = nbrs[t]
        w = np.linalg.solve(Ro[np.ix_(Nb, Nb)], Ro[Nb, t])
        B[t, Nb] = -w
        Dv[t] = 1.0 - Ro[Nb, t] @ w
    Q = B.T @ np.diag(1.0 / Dv) @ B
    Sigma = np.linalg.inv(Q)
    return order, nbrs, B, Ro, Sigma


def nystrom(X, k):
    """Landmarks (pivot order) and implied covariance U U^T + diag residual."""
    n = len(X)
    ref = np.ptp(X, axis=0).max()  # reference range = data extent
    Rref = corr(X, ref)
    d = np.ones(n)
    U = np.zeros((n, k))
    land = []
    for t in range(k):
        piv = int(d.argmax())
        land.append(piv)
        col = Rref[:, piv] - U[:, :t] @ U[piv, :t]
        U[:, t] = col / np.sqrt(d[piv])
        d = np.clip(d - U[:, t] ** 2, 0, None)
        d[piv] = 0
    land = np.array(land)
    R = corr(X)
    Rss = R[np.ix_(land, land)] + 1e-10 * np.eye(k)
    L = np.linalg.cholesky(Rss)
    Uk = np.linalg.solve(L, R[land]).T             # R_ns L_ss^-T
    low = Uk @ Uk.T
    Rapprox = low + np.diag(1.0 - np.diag(low))
    return land, Rapprox


def nested(X, p, rng):
    lab, _ = kmeans(X, p, rng)
    order = np.argsort(lab, kind="stable")
    R = corr(X)
    Rblock = np.where(lab[:, None] == lab[None, :], R, 0.0)
    return lab, order, Rblock


def subset(X, n_max, rng):
    _, C = kmeans(X, n_max, rng)
    idx = np.unique(cdist(C, X).argmin(1))
    return idx


# --- drawing -----------------------------------------------------------------
def style_space(ax, title):
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.6)
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def style_mat(ax, title, n):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.6)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)


def links(ax, x0, pts, color=HILITE, alpha=0.8, lw=0.8):
    for q in pts:
        ax.plot([x0[0], q[0]], [x0[1], q[1]], color=color, alpha=alpha, lw=lw, zorder=1)


def target_marker(ax, x0):
    ax.scatter([x0[0]], [x0[1]], s=70, c=HILITE, edgecolor="white", linewidth=1.5, zorder=5,
               marker="D")


def enorm(c):
    return PowerNorm(gamma=0.5, vmin=0, vmax=c.emax)


class Ctx:
    pass


def build():
    rng = np.random.default_rng(SEED)
    X = rng.random((N, 2))
    c = Ctx()
    c.X = X
    c.R = corr(X)
    c.vo, c.vn, c.vB, c.vRo, c.vS = vecchia(X, M)
    c.land, c.nR = nystrom(X, K)
    c.lab, c.go, c.gR = nested(X, P, np.random.default_rng(SEED + 1))
    c.keep = subset(X, N_MAX, np.random.default_rng(SEED + 2))
    # common target point: the design point closest to (0.62, 0.55)
    c.i0 = int(((X - [0.62, 0.55]) ** 2).sum(1).argmin())
    inv = np.empty(N, dtype=int)
    inv[c.vo] = np.arange(N)
    c.vS = c.vS[np.ix_(inv, inv)]  # implied Vecchia covariance, original row order
    errs = [np.abs(c.vS - c.R), np.abs(c.nR - c.R), np.abs(c.gR - c.R)]
    c.emax = max(e.max() for e in errs)
    return c


def row_exact(axs, c):
    X, i0 = c.X, c.i0
    ax = axs[0]
    links(ax, X[i0], np.delete(X, i0, 0), color=MUTED, alpha=0.25, lw=0.5)
    ax.scatter(X[:, 0], X[:, 1], s=9, c=INK, zorder=3)
    target_marker(ax, X[i0])
    style_space(ax, f"(a) x_i coupled to all n-1 = {N - 1} points")
    o = c.go
    im = axs[1].imshow(c.R[np.ix_(o, o)], cmap=CMAP_R, vmin=0, vmax=1, interpolation="nearest")
    style_mat(axs[1], "(b) dense R, Cholesky O(n³)", N)
    e = axs[2].imshow(np.zeros((N, N)), cmap=CMAP_ERR, norm=enorm(c), interpolation="nearest")
    style_mat(axs[2], "(c) |R̃ - R| = 0", N)
    return im, e


def row_vecchia(axs, c):
    X, order, nbrs = c.X, c.vo, c.vn
    t0 = int(np.where(order == c.i0)[0][0])
    rank = np.empty(N, dtype=int)
    rank[order] = np.arange(N)
    ax = axs[0]
    prev = rank < t0
    ax.scatter(X[~prev, 0], X[~prev, 1], s=9, facecolor="none", edgecolor=MUTED, linewidth=0.6, zorder=2)
    ax.scatter(X[prev, 0], X[prev, 1], s=9, c=INK, zorder=3)
    nb = order[nbrs[t0]]
    links(ax, X[c.i0], X[nb])
    ax.scatter(X[nb, 0], X[nb, 1], s=26, c=HILITE, zorder=4)
    target_marker(ax, X[c.i0])
    style_space(ax, f"(a) x_i (rank {t0} in maxmin order):\n    m={M} nearest among the {t0} previous (filled)")
    # (b) conditioning pattern of B (ordered), faint lower triangle = exact
    ax = axs[1]
    tri = np.tril(np.ones((N, N)), -1)
    pat = np.where(tri > 0, 0.08, np.nan)
    pat[np.abs(c.vB) > 0] = 1.0
    np.fill_diagonal(pat, 1.0)
    im = ax.imshow(pat, cmap=CMAP_R, vmin=0, vmax=1, interpolation="nearest")
    ax.axhline(t0, color=HILITE, lw=0.6, alpha=0.7)
    style_mat(ax, f"(b) row t conditions on N(t), |N(t)| ≤ m (dark);\n    light: dropped vs exact. O(n·m³)", N)
    err = np.abs(c.vS - c.R)
    e = axs[2].imshow(err[np.ix_(c.go, c.go)], cmap=CMAP_ERR, norm=enorm(c), interpolation="nearest")
    style_mat(axs[2], f"(c) |R̃ - R|, max {err.max():.2f}\n    (diffuse, mid-range pairs)", N)
    return im, e


def row_nystrom(axs, c):
    X, land, i = c.X, c.land, c.i0
    # x_j: nearest design point to x_i, to show how their correlation is rebuilt
    j = int(np.argsort(((X - X[i]) ** 2).sum(1))[1])
    ax = axs[0]
    ax.scatter(X[:, 0], X[:, 1], s=9, c=MUTED, zorder=2)
    # links x_i / x_j -> landmarks, width and opacity proportional to correlation
    for p0 in (i, j):
        r = c.R[p0, land]
        for s_, w in zip(land, r):
            if w > 0.02:
                ax.plot([X[p0, 0], X[s_, 0]], [X[p0, 1], X[s_, 1]], color=HILITE,
                        lw=0.4 + 2.5 * w, alpha=0.25 + 0.75 * w, zorder=1)
    ax.scatter(X[land, 0], X[land, 1], s=70, marker="*", c=HILITE, edgecolor="white", linewidth=0.5, zorder=4)
    target_marker(ax, X[i])
    ax.scatter([X[j, 0]], [X[j, 1]], s=45, marker="s", c=INK, edgecolor="white", linewidth=1.2, zorder=5)
    lbl = dict(textcoords="offset points", fontsize=8, color=INK, zorder=6,
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.annotate("x_i", X[i], xytext=(-22, -16), **lbl)
    ax.annotate("x_j", X[j], xytext=(8, 6), **lbl)
    style_space(ax, f"(a) R̃(x_i,x_j) = r_iᵀ R_ss⁻¹ r_j: only through the k={K}\n"
                    f"    landmarks (stars). Here R={c.R[i, j]:.2f}, R̃={c.nR[i, j]:.2f}")
    o = c.go
    im = axs[1].imshow(c.nR[np.ix_(o, o)], cmap=CMAP_R, vmin=0, vmax=1, interpolation="nearest")
    pos = np.where(np.isin(o, land))[0]
    axs[1].scatter(pos, np.full(len(pos), -2.5), marker="v", s=14, c=HILITE, clip_on=False, zorder=5)
    style_mat(axs[1], f"(b) R̃ = R_ns R_ss⁻¹ R_sn + diag (rank k, ▼ landmarks):\n"
                      f"    a blurred R, never formed; Woodbury O(n·k²)", N)
    err = np.abs(c.nR - c.R)
    e = axs[2].imshow(err[np.ix_(o, o)], cmap=CMAP_ERR, norm=enorm(c), interpolation="nearest")
    style_mat(axs[2], f"(c) |R̃ - R|, max {err.max():.2f}: close pairs whose\n"
                      f"    correlation no landmark carries", N)
    return im, e


def row_nested(axs, c):
    X, lab = c.X, c.lab
    ax = axs[0]
    for g in range(P):
        s = lab == g
        ax.scatter(X[s, 0], X[s, 1], s=9, c=GROUP_COLORS[g], zorder=3)
    same = np.where((lab == lab[c.i0]) & (np.arange(N) != c.i0))[0]
    links(ax, X[c.i0], X[same], color=INK, alpha=0.35, lw=0.6)
    target_marker(ax, X[c.i0])
    style_space(ax, f"(a) p={P} groups (k-means): x_i coupled to\n    its own group only (at fit time)")
    o = c.go
    blk = c.gR[np.ix_(o, o)].copy()
    l_o = lab[o]
    blk[l_o[:, None] != l_o[None, :]] = np.nan
    im = axs[1].imshow(blk, cmap=CMAP_R, vmin=0, vmax=1, interpolation="nearest")
    style_mat(axs[1], f"(b) p exact blocks: O(n³/p²)\n    + NK aggregation at predict", N)
    err = np.abs(c.gR - c.R)
    e = axs[2].imshow(err[np.ix_(o, o)], cmap=CMAP_ERR, norm=enorm(c), interpolation="nearest")
    style_mat(axs[2], f"(c) |R̃ - R|, max {err.max():.2f}\n    (cross-group pairs, at boundaries)", N)
    return im, e


def row_subset(axs, c):
    X, keep = c.X, c.keep
    nk = len(keep)
    ax = axs[0]
    drop = np.setdiff1d(np.arange(N), keep)
    ax.scatter(X[drop, 0], X[drop, 1], s=9, facecolor="none", edgecolor=MUTED, linewidth=0.6, zorder=2)
    ax.scatter(X[keep, 0], X[keep, 1], s=18, c=INK, zorder=3)
    style_space(ax, f"(a) n_max={nk} kept (filled), {N - nk} discarded")
    o = np.concatenate([keep, drop])
    Rm = c.R[np.ix_(o, o)].copy()
    Rm[nk:, :] = np.nan
    Rm[:, nk:] = np.nan
    im = axs[1].imshow(Rm, cmap=CMAP_R, vmin=0, vmax=1, interpolation="nearest")
    style_mat(axs[1], f"(b) exact R on the subset only\n    O(n_max³)", N)
    em = np.zeros((N, N))
    em[nk:, :] = np.nan
    em[:, nk:] = np.nan
    e = axs[2].imshow(em, cmap=CMAP_ERR, norm=enorm(c), interpolation="nearest")
    style_mat(axs[2], "(c) no error on the subset,\n    discarded rows/columns not modelled (gray)", N)
    return im, e


ROWS = [
    ("exact", 'method="exact"', row_exact),
    ("vecchia", f'method="vecchia({M})"', row_vecchia),
    ("nystrom", f'method="nystrom({K})"', row_nystrom),
    ("nested", f"NestedKriging(nb_groups={P})", row_nested),
    ("subsetofdata", f"subsetOfData(X, {N_MAX})", row_subset),
]


def colorbars(fig, im, e, axs):
    cb = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK)
    cb.set_label("R", fontsize=7, color=INK)
    cb = fig.colorbar(e, ax=axs[2], fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK)
    cb.set_label("|R̃ - R|", fontsize=7, color=INK)


FOOT = (f"n={N}, d=2, Matérn 5/2, θ={THETA}. (b): each method's own ordering (exact: k-means "
        "clusters, vecchia: maxmin, nystrom: k-means clusters, nested: groups, subset: kept first). "
        "(c): k-means cluster ordering, square-root color scale, same on every row.\n"
        "Generated by docs/math/scalability_figures.py")


def woodbury_figure(outdir):
    """Block structure of the Woodbury identity used by method="nystrom(k)",
    and measured cost of one solve (numpy, machine-dependent timings)."""
    import time
    from matplotlib.patches import Rectangle
    BLUE, DBLUE, OR = "#86b6ef", "#1c5cab", HILITE
    fig = plt.figure(figsize=(13, 8.2), facecolor="white")
    ax = fig.add_axes([0.01, 0.42, 0.98, 0.56]); ax.set_xlim(0, 26); ax.set_ylim(-1.2, 10.5); ax.axis("off")
    
    def box(x, y, w, h, fc, label, sub=None, diag=False):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=INK, lw=1))
        if diag:
            ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=INK, lw=1))
            ax.plot([x, x + w], [y + h, y], color=DBLUE, lw=3)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
        if sub: ax.text(x + w / 2, y - 0.35, sub, ha="center", va="top", fontsize=8.5, color=INK)
    def t(x, y, s, fs=16, **k):
        k.setdefault("ha", "center"); ax.text(x, y, s, va="center", fontsize=fs, color=INK, **k)
    
    N = 4.0; K = 0.8
    # line 1: what must be inverted
    t(0.3, 9.9, "1. The matrix to invert (Nystrom):  R̃ = D + U·Uᵀ", fs=12, ha="left", fontweight="bold")
    y = 4.6
    box(0.5, y, N, N, "white", "R̃", "n × n, dense\n(never formed)")
    ax.add_patch(Rectangle((0.5, y), N, N, fc=BLUE, ec=INK, alpha=0.5))
    t(5.1, y + N/2, "=")
    box(5.7, y, N, N, "white", "", "D: n × n diagonal\n(n numbers)", diag=True); ax.text(5.7+N*0.7, y+N*0.7, "D", fontsize=11, fontweight="bold", color=INK)
    t(10.3, y + N/2, "+")
    box(10.9, y, K, N, OR, "U", "n × k")
    t(12.1, y + N/2, "·", fs=22)
    box(12.6, y + N - K, N, K, OR, "Uᵀ", "k × n")
    ax.text(17.2, y + N/2, "« simple » part (diagonal: inverse\n trivial) + « thin » part (rank k ≪ n)",
            fontsize=10, color=INK, va="center")
    
    # line 2: the identity
    t(0.3, 3.05, "2. Woodbury: the only real inversion is a k × k matrix", fs=12, ha="left", fontweight="bold")
    y = -0.6
    box(0.5, y, 2.6, 2.6, BLUE, "R̃⁻¹", "n × n")
    t(3.6, y + 1.3, "=")
    box(4.1, y, 2.6, 2.6, "white", "", "D⁻¹ (diagonal)", diag=True)
    t(7.2, y + 1.3, "−")
    box(7.7, y, 2.6, 2.6, "white", "", "D⁻¹", diag=True)
    box(10.6, y, 0.55, 2.6, OR, "U", "n × k")
    box(11.5, y + 2.05, 0.55, 0.55, "#fbdccd", "", None)
    ax.add_patch(Rectangle((11.5, y + 2.05), 0.55, 0.55, fc=DBLUE, ec=INK))
    ax.annotate("M⁻¹, M = I + Uᵀ D⁻¹ U\nk × k : the ONLY\nfactorization, O(k³)", (11.78, y + 2.6), xytext=(18.6, 2.6),
                fontsize=9.5, color=DBLUE, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="->", color=DBLUE))
    box(12.4, y + 2.05, 2.6, 0.55, OR, "Uᵀ", None)
    box(15.3, y, 2.6, 2.6, "white", "", "D⁻¹", diag=True)
    ax.text(18.6, y + 0.5, "and  log|R̃| = Σ log Dᵢ + log|M|\n(determinant lemma, same M)",
            fontsize=10, color=INK, va="center")
    
    # bottom: timings
    rng = np.random.default_rng(0)
    ns = [500, 1000, 2000, 3000, 4000, 6000]; k = 50
    td, tw, err = [], [], []
    for n in ns:
        U = rng.standard_normal((n, k)) / np.sqrt(k); D = 0.1 + rng.random(n); b = rng.standard_normal(n)
        A = U @ U.T + np.diag(D)
        t0 = time.perf_counter(); xd = np.linalg.solve(A, b); td.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        Di = 1 / D; DiU = U * Di[:, None]; M = np.eye(k) + U.T @ DiU
        xw = Di * b - DiU @ np.linalg.solve(M, U.T @ (Di * b)); tw.append(time.perf_counter() - t0)
        err.append(np.abs(xd - xw).max() / np.abs(xd).max())
    bx = fig.add_axes([0.07, 0.07, 0.42, 0.28])
    bx.loglog(ns, td, "o-", color=DBLUE, lw=2, label="dense solve of R̃  ~ n³")
    bx.loglog(ns, tw, "o-", color=OR, lw=2, label=f"Woodbury (k={k})  ~ n·k²")
    bx.set_xlabel("n"); bx.set_ylabel("time to solve R̃ x = b (s)"); bx.legend(frameon=False, fontsize=9)
    bx.grid(alpha=0.3, which="both"); bx.set_title("3. Same result, different cost (numpy, one machine; indicative)", fontsize=11, loc="left")
    for s in ("top", "right"): bx.spines[s].set_visible(False)
    fig.text(0.56, 0.21,
      "Why it works, in one sentence: R̃ differs from an easy matrix (D)\n"
      "by a correction that lives in a k-dimensional subspace (columns of U).\n"
      "Woodbury inverts the easy part (free) and corrects only inside that\n"
      "subspace, with a k × k system instead of n × n.\n\n"
      f"Max relative difference dense vs Woodbury here: {max(err):.1e}\n"
      f"At n = {ns[-1]}: {td[-1]/tw[-1]:.0f}× faster; memory n·k instead of n².\n\n"
      "libKriging: LinearAlgebra::woodbury_solve / woodbury_logdet",
      fontsize=10, color=INK, va="center")
    fig.savefig(os.path.join(outdir, "nystrom_woodbury.png"), dpi=110, facecolor="white")
    plt.close(fig)


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "text.color": INK})
    c = build()
    for name, label, fn in ROWS:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.1), facecolor="white")
        im, e = fn(axs, c)
        colorbars(fig, im, e, axs)
        fig.suptitle(label, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
        fig.text(0.01, 0.01, FOOT, fontsize=6.5, color=MUTED)
        fig.subplots_adjust(left=0.01, right=0.955, top=0.80, bottom=0.08, wspace=0.12)
        fig.savefig(os.path.join(outdir, f"scalability_{name}.png"), dpi=130, facecolor="white")
        plt.close(fig)
    fig, axs = plt.subplots(len(ROWS), 3, figsize=(12, 4.0 * len(ROWS)), facecolor="white")
    for r, (name, label, fn) in enumerate(ROWS):
        im, e = fn(axs[r], c)
        colorbars(fig, im, e, axs[r])
        axs[r, 0].text(-0.02, 1.24, label, transform=axs[r, 0].transAxes, fontsize=11,
                       fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.01, 0.003, FOOT, fontsize=7, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.955, top=0.95, bottom=0.02, wspace=0.12, hspace=0.62)
    fig.savefig(os.path.join(outdir, "scalability_matrix.png"), dpi=110, facecolor="white")
    plt.close(fig)
    woodbury_figure(outdir)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "images"))
