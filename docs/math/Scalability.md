# Scaling past exact Kriging — method inventory

## Idea

Exact Kriging costs O(n³) to fit (Cholesky factorization of the n×n
covariance matrix) and O(n²) per `predict` call (reusing that factor),
which becomes impractical somewhere in the n = 10³–10⁴ range depending
on available memory and time budget. There is no single fix — fit cost,
predict cost, and memory are three separate axes, and libKriging
addresses each with a different, independently-usable method rather
than one do-everything mechanism. This page is the inventory; each
method has its own page with the full derivation.

| Method | Axis | Cost | Approximates | Dimension sensitivity | Doc |
|---|---|---|---|---|---|
| `LLVecchia(m)` | fit objective | O(n·m³)/eval | *local* conditioning (m neighbors) | degrades for d ≳ 5 (nearest neighbors less informative) | [Vecchia.md](Vecchia.md) |
| `LLNystrom(k)` | fit objective | O(n·k²)/eval | *global* low-rank covariance (k landmarks) | dimension-robust | [Nystrom.md](Nystrom.md) |
| `NestedKriging` | fit + predict, whole model | O(n³/p²) fit, O(q·n²/p) or O(q·n²) predict | divide-and-conquer (p groups) + aggregation | dimension-robust (submodels are exact Kriging) | [Nested.md](Nested.md) |
| `subsetOfData` | pre-fit data reduction | O(n_max) k-means pass, then ordinary O(n_max³) fit | *nothing* — exact fit, just on fewer points | none (discards points outright rather than approximating structure) | [SubsetOfData.md](SubsetOfData.md) |
| `predictCG` | predict only | O(n²·iters) mean, +O(n²·iters·q) for stdev | *nothing* — same exact objective, iterative linear algebra instead of a dense factor | none (doesn't touch the covariance structure) | [PredictCG.md](PredictCG.md) |
| OpenMP | fit + predict, cross-cutting | same asymptotic cost, smaller constant | *nothing* — exact, just parallel | none | — (build-time; no dedicated objective/method, always on when available) |

All of these (Vecchia, Nystrom, NestedKriging, subsetOfData, predictCG)
are usable independently and, where noted below, combinable — none of
them require opting out of the others.

## Which one, when

1. **Is the *fit* itself (not just prediction) too slow, and you're
   willing to accept a genuinely different objective — not just a
   cheaper way to evaluate the same one?**
   - **Low-to-moderate dimension (d ≲ 5), spatially local structure**
     → `objective="LLVecchia(m)"`. Local conditioning on m nearest
     neighbors; the approximation degrades as nearest neighbors become
     less informative in higher dimension.
   - **Higher dimension, or no reliable local/spatial structure**
     → `objective="LLNystrom(k)"`. Global low-rank approximation,
     dimension-robust the same way `NestedKriging`'s `NK` aggregation
     is, but as a single model rather than a partition.
   - **n large enough (~10⁴–10⁶) that even Vecchia/Nystrom struggle, and
     the design can be partitioned / prediction can be parallelized**
     → `NestedKriging`. Splits `(X, y)` into p groups, fits exact
     submodels per group, aggregates predictions (`NK` by default —
     converges to the exact/PoE-family predictor as group size grows;
     see [Nested.md](Nested.md) §3 for the aggregation choice).
   - **Willing to lose information rather than approximate it**:
     `subsetOfData(X, n_max)` picks `n_max` representative rows
     (k-means, snapped to real points) and hands them to an ordinary
     exact fit — no new objective, no iterative linear algebra, just
     fewer points. The cheapest option by far, at the cost of
     discarding `n - n_max` observations outright rather than using all
     of them more cheaply the way every other method here does.

2. **Is the fit fine (ordinary `"LL"`/`"LOO"`/`"LMP"`, one-time O(n³) is
   acceptable) but you don't want an O(n²) dense factor resident just
   for `predict` — or it was never computed at all?**
   → `predictCG`. Doesn't change the objective or its accuracy at all;
   just solves each prediction with matrix-free conjugate gradient
   instead of reusing a stored factor. `return_stdev=true` is
   opt-in-expensive (one CG solve *per prediction point*) — cheap only
   for the mean, or for a handful of stdev queries.

3. **Regardless of which of the above you use**: OpenMP parallelizes
   independent work (multi-start `optim`, multi-trajectory `simulate`,
   multi-point `predict`) transparently when built with it — no API
   change, always worth having on for large designs.

Don't reach for any of these by default: for the common case (n in the
hundreds to low thousands), plain exact `Kriging` is both simpler and,
for `NestedKriging`'s `NK` aggregation specifically, a *fallback
target* it converges to as group size grows.

## Combining methods

- **`NestedKriging` + `LLVecchia`**: `NestedKriging`'s common-prior fit
  can estimate the shared (θ, σ², β) via one global light Vecchia fit
  (`objective="LLVecchia(m)"`) instead of a full O(n³) reference fit —
  see [Nested.md](Nested.md)'s "Common prior" section. `LLNystrom` is
  **not** currently wired into `NestedKriging`'s common-prior path.
- **`subsetOfData` before any of the above**: since it's a plain
  pre-fit row selection, not a fit objective, it composes with every
  other method here — reduce `(X, y)` first, then fit with
  `LLVecchia`/`LLNystrom`/`NestedKriging`/exact `"LL"` on the reduced
  design if it's still large enough to warrant one of them.
- **Vecchia/Nystrom + `NoiseModel`**: neither supports a nugget/noise
  channel yet (`NoiseModel::None` only) — see each method's own
  "Current limitations" section.
- **`predictCG` after a light Vecchia fit**: a light Vecchia fit
  (`set_vecchia_exact_commit(false)`) never has a resident dense
  factor, so `predict` on it always routes to the *local* Vecchia
  predictor. `predictCG` is a legitimate alternative there too — same
  fitted θ/β, but a *global* (not m-neighbor-local) prediction, at
  O(n²·iters) instead of Vecchia's O(n³) exact-commit cost. Neither
  path is automatic; call the one you want explicitly.
- **`predictCG` after a Nystrom fit**: works (predictCG only needs
  m_X/m_y/m_F/θ/β, which a Nystrom fit still populates in full), but is
  usually pointless — `predictNystrom` reuses the fit's own committed
  low-rank factors and is cheaper than a fresh CG solve over the full n.

## Not implemented (deferred)

Evaluated and explicitly out of scope for now (see the project's own
scalability roadmap analysis): GPU acceleration (a Bandicoot port would
be the lowest-effort path, since it mirrors Armadillo's API, but
untested on this project's target platforms), and structured/inducing-
point variational methods (KISS-GP/SKI, SVGP) — these would need a
substantially different inference engine (approximate ELBO, automatic
differentiation through arbitrary kernel compositions) rather than an
incremental addition to the current exact-linear-algebra core. See
[libKriging_vs_GPyTorch.ipynb](../comparisons/libKriging_vs_GPyTorch.ipynb)
for a direct comparison against a library that does implement these
(GPyTorch's BBMM/CG + GPU + variational stack).

## References

See each method's own page ([Vecchia.md](Vecchia.md),
[Nystrom.md](Nystrom.md), [Nested.md](Nested.md),
[SubsetOfData.md](SubsetOfData.md), [PredictCG.md](PredictCG.md)) for its
specific references.
