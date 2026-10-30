# T4. KModel buffers per worker — work log

Task specification: `docs/dev/Rationalization.md`, section T4.

## Status

Done (PR #444).

## Changes

### Kriging (`Kriging::fit_impl`, BFGS multistart)

Before: `preallocated_models(multistart)` allocated R and L (n×n) for every start before the
pool started, the models stayed alive until the end of the fit, and each start deep-copied
L, R, Rinv, Fstar, Rstar, Qstar, Estar, ystar, betahat into its `OptimizationResult`. At the end
of the multistart: 2 × multistart × (R, L) (+ Linv for LOO, kept in the model) alive at once.

After:

- each start builds its `KModel` inside its task (`allocate_KModel()`, or an empty model for
  Vecchia / Nystrom as before): at most min(multistart, pool size) dense models are alive at
  once, the pool bounding the number of running tasks;
- `OptimizationResult` keeps the scalar results only (objective, gamma, theta, extra
  parameter, success, error message);
- when a start finishes, under a mutex, its model is **moved** into `best_model` if it is the
  best so far; the previous best is freed. The order is (objective, start index): the same
  choice as the former scan of the results in index order (first strict minimum), whatever the
  order in which the starts finish. The commit moves `best_model` into `m_T`, `m_R`, `m_M`,
  `m_circ`, `m_star`, `m_Rinv`, `m_beta`, `m_z` (was a copy of the copy). A consistency check
  throws `std::logic_error` if the scan and the mutex-held choice disagree (cannot happen:
  same values, same order).

No matrix of a non-best start is copied; a start that was the best when it finished and is
beaten later was moved, not copied, then freed.

### WarpKriging (`run_parallel_multistart`)

Same pattern, larger state: every clone (`clone_for_thread()`, one per start, built before the
pool) carried a copy of the GP state of `this` (m_X, m_dX = d × n², m_R, m_T, m_Rinv, m_M, ...),
kept it after its start, and each start deep-copied its state (C, R, Rinv, Phi, dPhi, M, ...)
into its result.

After:

- the clones are still built before the pool, in the calling thread, since the warp
  constructors reseed the RNG (`arma_rng::set_seed`; the Armadillo RNG is `thread_local` on
  Linux but not on every platform): building them in the workers would change the RNG state
  left by the fit and race where it is not thread-local;
- each clone's GP state is released right after its construction (`release_gp_state`) and
  again when its start ends: the worker rebuilds all of it (`refresh_cache()`: m_X, m_dX,
  m_maxdX, m_F, then R, T, Rinv, M, circ, beta, z, sigma2, logdet) before reading it. The
  Cholesky update path (`do_update`) is not taken either way (it needs n > m_T.n_rows: false
  with an n×n m_T, false with an empty one, whose memptr is null);
- only the best start's state is kept, moved into `best` under the mutex with the same
  (neg_ll, start index) order, then moved into `this`.

### TaskPool

Not changed. The task suggested exposing a worker index to index per-worker buffers; it is not
needed: allocating the model in the task gives the same bound (one model per running task,
≤ pool size) without a worker-indexed array, and the per-start "preallocation" bought nothing
anyway: `KrigingImpl::populate_Model` move-assigns new matrices into `m.L`, `m.Fstar`,
`m.ystar`, `m.Estar`, ... at every evaluation (`m.L = cholCov(...)`), so the preallocated
buffers were freed at the first evaluation; only R is written in place.

### Refactorization at the optimum: not used

Keeping the best model as the starts finish needs no refactorization (no extra O(n³) and no
question of bit-identity: the committed matrices are those of the start's own final
evaluation, `fit_ofn(best_gamma, nullptr, &m)`, as before).

## Measurements

Machine: 4 cores (Intel Xeon @ 2.10 GHz, cloud container), 15.7 GB, Linux, GCC 13, OpenBLAS
0.3.26 (pthreads), Release. Peak RSS = `getrusage(RUSAGE_SELF).ru_maxrss` at the end of the
process (`/usr/bin/time` is not installed in the container); one run per configuration (the
peak depends on how the starts overlap in time, so ±5 % is expected between runs).

Driver: `y = Σ_j (j+1) sin(2π x_j)`, X ~ U[0,1]^d (seed 123), matern5_2, constant trend,
`normalize = false`; prints theta, LL and 5 predictions at full precision (`%.17g`).
WarpKriging: one `kumaraswamy` warp per input.

| Configuration | Peak RSS before | Peak RSS after | Wall before | Wall after |
|---|---|---|---|---|
| Kriging n = 1000, d = 3, BFGS10, LL | 563.6 MB | 329.7 MB | 13.6 s | 13.0 s |
| Kriging n = 1000, d = 3, BFGS10, LOO | 555.8 MB | 307.0 MB | 18.1 s | 18.0 s |
| Kriging n = 2000, d = 3, BFGS20, LL | 3368.8 MB | 1501.4 MB | 104.7 s | 110.0 s |
| WarpKriging n = 500, d = 2, BFGS10 | 296.3 MB | 151.4 MB | 3.1 s | 3.1 s |
| WarpKriging n = 1000, d = 2, BFGS10 | 1112.2 MB | 578.0 MB | 19.0 s | 18.6 s |

theta, LL and predictions are **bit-identical** (all 17 digits) before / after for every
configuration above. Wall-time differences are within run-to-run noise (single runs).

Remaining peak at n = 2000 (1.5 GB ≈ 47 n×n matrices of 32 MB, 4 workers): m_dX (d × n², 96 MB,
shared) plus, per running start, R, L, Rinv and the temporaries of the LL gradient. It is now
proportional to the pool size, not to multistart; reducing the per-evaluation temporaries is
outside T4 (gradient code, T1).

## Tests

Full `ctest` on the modified tree: see the PR description (same pass / fail set as the base
tree; the multistart tests `KrigingTest` "BFGS20 == best of 20 BFGS1" etc. pass).

## Decisions

- Per-task allocation rather than worker-indexed buffers (see TaskPool above).
- Best kept by move under a mutex, ordered by (objective, start index), rather than a
  refactorization at the optimum.
- WarpKriging clones still built up front (RNG side effects of the warp constructors), but
  without their GP state.
