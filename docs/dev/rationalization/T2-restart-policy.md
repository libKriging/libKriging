# T2. Single restart policy — work log

Task specification: `docs/dev/Rationalization.md`, section T2.

## Status

Done (PR #442, draft).

## Change

`src/lib/RestartPolicy.hpp` (private, not installed): `libKriging::detail::RestartPolicy` holds
the restart criteria, the restart counters, the best value reached and the restart point. After
each L-BFGS-B run the caller calls `record(f_opt)`, then `restart(task, num_iters, at_lower_bound,
long_range)`; on `true` the next run starts from `restart_theta(theta_start, lower, upper)`.

| Criterion | Before: Kriging | Before: WarpKriging joint | Before: AdamBFGS inner | After |
|---|---|---|---|---|
| `ABNORMAL_TERMINATION_IN_LNSRCH` | yes | yes | yes | yes (all) |
| `num_iters <= 2` | yes | yes | yes | yes (all) |
| at lower bound (`< eps`) | theta space, free dims | theta space, all dims | gamma space, all dims | `distance_to_lower` over free dims; space chosen by the caller (unchanged) |
| long range | single start, noise-free, LL | no | no | Kriging only (`max_long_range_restart` = 0 elsewhere) |
| worse than best + `factr * eps * max(1, abs(min(best, f)))` | yes | yes (see below) | no | Kriging, WarpKriging joint; off for the AdamBFGS inner BFGS |
| worse cascade | up to `max_restart` | up to `max_restart` | n/a | **bounded** (below) |
| restart point | `(theta_start + theta_lower) / 2^retry`, clamped to the bounds | same, not clamped | same, clamped | same, clamped (all) |
| resume when the iteration budget is exhausted while improving | no | yes (warp params present) | no | WarpKriging joint, counted by `resume()` |

- `Kriging::fit_impl`: the restart loop now calls the policy; the reparametrization, the noise
  parameter of the restart point and the widening of the local bounds stay in the caller.
- WarpKriging joint BFGS: same; the budget-exhausted resume is kept in the caller and counted as a
  retry (as before, the contraction exponent includes it).
- `AdamBFGS` (public header `libKriging/AdamBFGS.hpp`, cannot include a private header): the
  criteria are removed from it. `inner_restart` now receives the run's `OptimizeResult`, its end
  point and its start point and returns the restart point or an empty vector;
  `AdamBFGS::max_restart` is removed. WarpKriging implements `inner_restart` with a
  `RestartPolicy`. **API change** of an installed header (used in-tree by WarpKriging and
  `AdamBFGSTest` only).

### `worse` bound (the one intended behaviour change)

Before, a run worse than the best one triggered a restart from a point closer to the lower bound;
if that one was worse again, another one, up to `max_restart` = 10. Now a run started **only**
because the previous one was worse (`last_reasons() == Worse`) does not trigger a restart by being
worse; it still does through any other criterion (abnormal stop, <= 2 iterations, lower bound,
long range), and the run after such a restart may again use `worse`. So worse-only restarts can no
longer be consecutive.

Rationale: the restart point depends only on the start point and the retry number, so each
cascaded restart starts closer to (and from the third one on, often at) the lower bound; the best
run is kept, so these restarts only cost time unless one of them finds a better basin, which the
measurements below show to be rare and marginal.

## Measurements

Driver (not committed): 1728 Kriging fits (functions sin/cos d 2, Branin d 2, quadratic + sine
d 4, one active input of 4; n 15 / 40 / 100; gauss, matern3_2, matern5_2, exp; LL, LOO, LMP;
BFGS and BFGS4; noise-free and nugget (LL); 3 seeds) and 216 WarpKriging fits (none,
kumaraswamy, affine, boxcox warps; BFGS (joint), Adam (bi-level), BFGS4; n 20 / 50; 3 seeds),
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, Release, OpenBLAS 0.3.26, Linux x86_64, 4 cores.
The base build is deterministic (two runs: identical output).

| | Base (`90d35b5`) | T2 |
|---|---|---|
| Kriging fits with bit-identical theta and LL | | 1724 / 1728 |
| Kriging restarts (sum over fits and starts) | 1524 | 1341 (−12 %) |
| Kriging fits with fewer restarts | | 43 |
| WarpKriging fits bit-identical (theta, warp params, LL) | | 216 / 216 |
| Wall time, Kriging grid | 50.9 s | 51.2 s (noise level) |
| Wall time, WarpKriging grid | 9.3 s | 9.4 s (noise level) |

The 4 changed Kriging fits (ΔLL = T2 − base):

| Fit | Base LL | T2 LL | ΔLL |
|---|---|---|---|
| sin/cos d 2, n 100, gauss, LL, BFGS, nugget, seed 0 | 87.187200907 | 87.187200568 | −3.4e-7 |
| same, BFGS4 | 87.187200907 | 87.187200858 | −4.9e-8 |
| Branin d 2, n 40, gauss, LL, BFGS, nugget, seed 2 | −138.794911456 | −138.794911422 | +3.4e-8 |
| quadratic + sine d 4, n 40, gauss, LL, BFGS, noise-free, seed 1 | 20.367631306 | 20.367620574 | −1.07e-5 |

In three cases a cascaded worse-only restart of the base had found a marginally better point
(at most 1.07e-5 LL); in the Branin case the T2 value is the higher one: the run kept as best
is chosen on the search objective (with the search jitter, `docs/dev/CholJitter.md`) while the
reported LL is the committed exact model's, so the two orderings can disagree at this level
(hypothesis consistent with the 3.4e-8 gap, not traced further). In all four cases the theta
differences are in the 4th–5th significant digit, inside the L-BFGS-B tolerance (`factr` =
1e7 eps relative on the objective): a flat optimum, not a different basin.

The 39 other fits with fewer restarts gave bit-identical results: the cascaded worse runs never
became the best run.

Restart reasons in the T2 run (bit mask: 1 abnormal, 2 <= 2 iterations, 4 lower bound, 8 long
range, 16 worse): long-range only 926, worse only 77, <= 2 iterations only 66, lower bound only
43, abnormal only 33, combinations 196. The long-range restart makes 69 % of all restarts on this
grid (to be weighed in T3).

`ctest` (Release, same machine): base 226 / 226 passed; T2 228 / 228 passed (the 226 + the 2
`RestartPolicyTest` cases).

## Decisions and critical notes

- **WarpKriging joint `worse` was active, not dead code.** It was computed after updating the
  best value with the run itself; `f > min(best, f) + tol` is equivalent to Kriging's
  `f > best + tol` (and the tolerance uses the same `abs(min(best, f))`). Kept on, so the
  bound applies there too (no change observed on the 216 Warp fits).
- **AdamBFGS inner BFGS: no `worse` criterion**, as before; enabling it would change the Adam
  path's results and is not part of T2. Its at-bound test stays in the inner (gamma = log theta)
  space, as before: with the default reparametrization, `|log θ − log θ_lo| < eps` and
  `|θ − θ_lo| < eps` differ for θ_lo far from 1. Unifying the space would change results; left
  as a documented divergence.
- **Clamping the WarpKriging joint restart point is not a change**: L-BFGS-B projects its
  starting point onto the (fixed) bounds (`active_` in `lbfgsb.c`), and `log` is increasing, so
  clamping in theta space then mapping to gamma gives the same point. Kriging widens its local
  bounds to the restart point, so it needs the clamp (as before).
- **Free dimensions for the at-bound test**: WarpKriging has `theta_upper >= 2 theta_lower`, so no
  pinned dimension: same as the former all-dimensions test.
- **The restart point is questionable but kept**: `(θ_start + θ_lo) / 2^k` is below θ_lo as soon
  as θ_start < (2^k − 1) θ_lo, i.e. from k = 2 often clamped onto the lower bound, where the next
  run tends to stop again (lower-bound criterion). A log-scale contraction would be more
  natural; changing it is a behaviour change outside T2 (candidate for T3 / T7 measurements).
- Logging: the Kriging "Restarting BFGS" line now prints the retry number after the increment
  and the reasons mask, and no longer `sol_to_ub` (unused by any criterion).
- Unit test `tests/RestartPolicyTest.cpp`: criteria, `max_restart` and `max_long_range_restart`
  bounds, tolerance, the worse bound, restart point and free-dimension distance.
