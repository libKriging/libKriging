# WarpKriging Optimization Report

Benchmark config: n=400, d=2 (Branin), 3 iterations, Release build.

## Step 0: Baseline (AdamBFGS, 50 Adam iters, 100 BFGS inner iters)

| warping      | fit (ms) | LL       |
|--------------|----------|----------|
| none         |      568 |   449.91 |
| affine       |   18,410 |   449.91 |
| boxcox       |   14,499 |   452.71 |
| kumaraswamy  |   15,083 |   464.61 |
| mlp          |   31,796 |   459.96 |
| neural_mono  |   35,216 |   168.62 |

## Step 1: Reduce inner BFGS iters (100 -> 20)

No improvement — BFGS needs more restarts, total evals similar or worse. **Reverted.**

| warping      | fit (ms) | LL       | vs baseline |
|--------------|----------|----------|-------------|
| affine       |   23,032 |   449.91 |      +25%   |
| boxcox       |   17,831 |   452.71 |      +23%   |
| kumaraswamy  |   15,352 |   464.61 |       +2%   |
| mlp          |   32,587 |   459.96 |       +2%   |
| neural_mono  |   34,839 |   168.62 |       -1%   |


## Step 2: Avoid full R⁻¹ in gradient (Cinv approach)

Replaced `Rinv = solve(Cᵀ, solve(C, I))` with `Cinv = solve(C, I)` then per-dim
`tr(R⁻¹·dR_k) = accu((Cinv·dR_k) % Cinv)`. This is actually **worse** because it
does an O(n³) dense multiply per dimension, while the original precomputes `dLL_dR`
once and reuses it. **Reverted.**

Also tried vectorizing `build_R` and `build_dR_dtheta_k` — created large temporaries
that hurt cache. **Also reverted.**

| warping      | fit (ms) | LL       | vs baseline |
|--------------|----------|----------|-------------|
| none         |      482 |   449.91 |      -15%   |
| affine       |   41,258 |   449.91 |     +124%   |
| boxcox       |   27,046 |   452.71 |      +86%   |
| kumaraswamy  |   34,788 |   464.61 |     +130%   |
| mlp          |   60,247 |   460.38 |      +89%   |
| neural_mono  |   39,239 |   168.62 |      +11%   |


## Step 3: Joint L-BFGS-B on all params (warp + log_theta)

Concatenate `[warp_params ; log_theta]` into a single vector and run L-BFGS-B on
the full joint space. Eliminates the 50× Adam outer loop entirely.

**Speed**: 10-15× faster than baseline. **Quality**: LL much worse for non-trivial
warpings — BFGS gets stuck in bad local optima because the joint landscape is
highly non-convex w.r.t. warp params. Affine/boxcox/kumaraswamy all collapse to
LL≈48 vs 450-465 with Adam+BFGS. **Not usable standalone.**

| warping      | fit (ms) | LL       | vs baseline |
|--------------|----------|----------|-------------|
| none         |      792 |   449.91 |      +39%   |
| affine       |    1,772 |    48.44 |      -90%   |
| boxcox       |    1,356 |    48.44 |      -91%   |
| kumaraswamy  |    1,591 |    48.44 |      -89%   |
| mlp          |    2,236 |  -919.90 |      -93%   |
| neural_mono  |    1,789 | -2001.31 |      -95%   |


## Step 4: Reduce Adam iterations (50 → 10)

Inner BFGS does most of the work; Adam just needs to find the right basin. With 10
Adam iters (vs 50), speed improves 3-5× with minimal LL degradation. Adding a joint
BFGS polish after Adam adds ~500-1000ms overhead with no LL improvement.

**5 Adam iters**: another 2× faster but LL degrades more (kumaraswamy 456 vs 461).
**10 Adam iters**: best trade-off. **Adopted as new default.**

| warping      | fit (ms) | LL       | vs baseline |
|--------------|----------|----------|-------------|
| none         |      491 |   449.91 |      -14%   |
| affine       |    4,815 |   449.91 |      -74%   |
| boxcox       |    2,692 |   452.45 |      -81%   |
| kumaraswamy  |    4,935 |   460.86 |      -67%   |
| mlp          |    6,300 |   459.96 |      -80%   |
| neural_mono  |    7,196 |   163.91 |      -80%   |


## Step 5: Fuse gradient computation (cache pairwise differences)

Instead of calling `build_dR_dtheta_k` d times (each recomputing diff, r2, r over
n(n-1)/2 pairs and allocating an n×n matrix), compute all d gradient components in a
single pass over (i,j) pairs, accumulating `tr(dLL_dR · dR_k)` inline. Eliminates d
temporary n×n matrices and d redundant passes.

| warping      | fit (ms) | LL       | vs Step 4 | vs baseline |
|--------------|----------|----------|-----------|-------------|
| none         |      250 |   449.91 |     -49%  |      -56%   |
| affine       |    2,453 |   449.91 |     -49%  |      -87%   |
| boxcox       |    2,483 |   452.45 |      -8%  |      -83%   |
| kumaraswamy  |    4,745 |   460.86 |      -4%  |      -69%   |
| mlp          |    4,566 |   460.34 |     -28%  |      -86%   |
| neural_mono  |    4,525 |   163.91 |     -37%  |      -87%   |


## Summary

Best results from **Step 4** (reduce Adam 50→10) + **Step 5** (fused gradient).
Combined: **2-8× speedup** across all warpings with minimal LL degradation.

Joint L-BFGS-B on all params (Step 3) is available via `optim="BFGS"` but produces
poor LL for non-trivial warpings.

| warping      | before (ms) | after (ms) | speedup | LL before | LL after |
|--------------|-------------|------------|---------|-----------|----------|
| none         |         568 |        250 |    2.3× |    449.91 |   449.91 |
| affine       |      18,410 |      2,453 |    7.5× |    449.91 |   449.91 |
| boxcox       |      14,499 |      2,483 |    5.8× |    452.71 |   452.45 |
| kumaraswamy  |      15,083 |      4,745 |    3.2× |    464.61 |   460.86 |
| mlp          |      31,796 |      4,566 |    7.0× |    459.96 |   460.34 |
| neural_mono  |      35,216 |      4,525 |    7.8× |    168.62 |   163.91 |
