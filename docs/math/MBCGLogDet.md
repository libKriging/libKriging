# Why `LLIterative`'s log-determinant is NOT read off the CG solve (mBCG)

## TL;DR

GPyTorch's BBMM gets its stochastic log-determinant "for free" by reading
it directly off the coefficients of the CG solve it already has to run —
no separate Lanczos pass needed (mBCG, Gardner et al. 2018). `LLIterative`
does not do this: it runs a *dedicated* batched Lanczos recurrence
(`LinearAlgebra::stochasticLogDetBatched` /
`LinearAlgebraCuda::stochasticLogDetBatched`, see [Iterative.md](Iterative.md))
purely to estimate `log|R|`, alongside the CG solves. This was tried once
and reverted before this investigation; this investigation re-examined it
under the codebase's current (much improved) defaults, found the original
symptom hard to reproduce in isolation, then found the *actual* mechanism —
which is not a subtle numerical-precision footnote but a structural
incompatibility with a design already relied on elsewhere in this codebase
(periodic exact-residual CG restarts). The conclusion is: **don't do this
without solving a real, unsolved numerical-analysis problem first.** This
document is the record of that investigation, so it isn't repeated from
scratch next time someone reads the "tried and reverted" comment and
wonders whether today's codebase would fare better.

## Background: what mBCG buys, and what `LLIterative` does instead

For an SPD matrix `A` and starting vector `b`, running CG to solve `Ax = b`
and running the (mathematically equivalent) Lanczos tridiagonalization
process from `v₁ = b/‖b‖` are two views of the *same* underlying Krylov
recurrence. The CG iterates `αᵢ` (step length) and `βᵢ` (search-direction
update coefficient) determine the Lanczos tridiagonal `T` exactly:

```
T(0,0)     = 1/α₀
T(i,i)     = 1/αᵢ + βᵢ₋₁/αᵢ₋₁                  (i > 0)
T(i,i+1) = T(i+1,i) = √βᵢ / αᵢ
```

Feed that `T` into the usual SLQ quadrature (`z' log(A) z ≈ ‖z‖² · e₁'᾿
f(T) e₁`, see [Iterative.md](Iterative.md)'s SLQ section) and the
log-determinant comes out of the CG solve you were already running for
`R⁻¹·probes` — no extra matvecs, no extra kernel launches, nothing. This is
exactly GPyTorch's mBCG: **the log-determinant is "free"** because CG and
SLQ share one Krylov subspace instead of exploring two independent ones
(`LLIterative` currently uses `nprobe` probes for a *dedicated* 40-step (by
default) Lanczos recurrence, entirely separate from the CG solve on those
same probes needed for the gradient's Hutchinson trace — see
`Kriging.cpp:_logLikelihoodIterative`).

The existing code comment (`Kriging.cpp`, at the SLQ dispatch, predating
this investigation) records that this was tried once:

> Raw CG-scalar mBCG fusion was tried and dropped: without full
> reorthogonalization the reconstructed tridiagonal loses accuracy as
> `cond(R)` grows with `n` (verified: d/n 5e-3 -> 0.4 over n=250 -> 2000).

That is: at n=2000, the CG-derived tridiagonal's log-determinant was off by
**40%** relative error, escalating from 0.5% at n=250. The stated
diagnosis was reorthogonalization — full reorthogonalization is what the
*dedicated* Lanczos recurrence pays for (see [Iterative.md](Iterative.md)),
and a raw CG recurrence doesn't get it, so as `cond(R)` grows the implicit
Lanczos vectors lose orthogonality and the reconstructed tridiagonal's
eigenvalues drift from the true spectrum.

## Why this was worth re-examining now

Three things changed since that revert that made it worth checking again
rather than taking the old finding at face value:

1. **`cg_tol` used to be hardcoded at `1e-8`.** It's `1e-4` by default now
   (see [Iterative.md](Iterative.md)'s CG-tolerance section), and the
   gradient's probe solve has its own, even looser `probes_cg_tol` (default
   `1e-2`, see the field's doc comment on `Kriging::m_iterative_probes_cg_tol`).
   Both changes mean today's CG runs to a much looser tolerance than
   whatever was in place for the original mBCG attempt — fewer iterations,
   plausibly less accumulated orthogonality loss.
2. **The probe CG solve and the `[F|y]` solve are now fused into one
   Krylov pass** (mBCG "phase A", this session, see the commit touching
   `LinearAlgebraCuda::conjugateGradient`'s per-column-tolerance overload) —
   the CG whose coefficients you'd read the tridiagonal off is a different,
   more capable piece of machinery than whatever the original attempt used.
3. **The SLQ Lanczos itself is now fully device-resident** (see
   [Iterative.md](Iterative.md) / the commit adding
   `LinearAlgebraCuda::stochasticLogDetBatched`) — eliminating the
   *dedicated* Lanczos pass via mBCG would remove real, now well-understood
   overhead (see the timing numbers in that section), so the payoff for
   getting mBCG working is higher than it used to be.

## Step 1 — a from-scratch reference, no reorthogonalization, no preconditioner

Before touching any CUDA code, the CG↔Lanczos duality was prototyped
directly in Python/NumPy: build the separable Matérn-5/2 `R` exactly as
`Covariance::resolve("matern5_2")` does, run **plain, unpreconditioned,
non-reorthogonalized** CG on Rademacher probe vectors, convert the
`(αᵢ, βᵢ)` sequence to the tridiagonal via the formula above, run the same
SLQ quadrature as production, and compare against `log|R|` from an exact
dense `slogdet`.

d=4, θ=0.15, `sine_sum`-style random design, 30 probes, `tol=1e-4`
(matching `probes_cg_tol`'s magnitude), `max_iter` generous:

| n | exact `log|R|` | mBCG estimate | relative error | CG iterations (med) |
|--:|--:|--:|--:|--:|
| 1000 | -730.28 | -734.13 | 5.3e-3 | 124 |
| 2000 | -2265.95 | -2276.13 | 4.5e-3 | 332 |
| 4000 | -6867.18 | -6870.87 | 5.4e-4 | 745 |

This flatly contradicts the old finding — 0.5% at n=2000, not 40%, and
*shrinking* as n grows, not growing. Pushing further, at n=4000/8000
against the dedicated fixed-40-step Lanczos production actually uses today:

| n | exact | mBCG (no reorthog) | mBCG error | fixed-40-step Lanczos error |
|--:|--:|--:|--:|--:|
| 4000 | -6846.06 | -6884.52 | **5.6e-3** | 4.9e-3 |
| 8000 | -19432.93 | -19408.40 | **1.3e-3** | **4.4e-2** |

At n=8000 the naive mBCG estimate is **~35× more accurate** than what
`LLIterative` ships today. The reason is intuitive once you see the CG
iteration counts: mBCG's implicit Lanczos runs for however many iterations
CG actually needs (hundreds to low thousands here), while the dedicated
Lanczos is capped at a *fixed* 40 steps regardless of `n` — the fixed
budget under-resolves `R`'s spectrum as `n` grows exactly the way
[Iterative.md](Iterative.md)'s own "SLQ bias vs. budget, not method"
discussion describes; mBCG's automatically-scaling step count sidesteps
that same defect for free.

Two more checks to rule out the obvious suspects for why *this* prototype
disagreed with the old finding:

- **`cg_tol` sensitivity** (n=2000, θ=0.15): sweeping the convergence
  tolerance from `1e-4` to `1e-8` (310 → 608 median iterations) left the
  error **completely flat** at 3.7e-3. Once CG has resolved the dominant
  spectral structure, extra iterations refine the *solution* but barely
  move the *log-determinant estimate* — so a tighter historical `cg_tol`
  doesn't explain the old 40% figure either.
- **Conditioning sensitivity** (n=2000, sweeping θ): `cond(R)` from 1.5e4
  (θ=0.15) to 6.8e6 (θ=0.3) — the error stayed at 4-6e-3 throughout, not
  the runaway growth "loses accuracy as cond(R) grows with n" would
  predict if that were the dominant effect at these settings.

Neither of the two parameters the old comment's own reasoning points at
(tolerance, conditioning) reproduces the old 40%-at-n=2000 result. Something
else was different.

## Step 2 — the real mechanism: periodic CG restarts

`LinearAlgebra::conjugateGradientBatched` (and its GPU counterparts) don't
run a textbook, unbroken CG recurrence: every `restart_every = 50`
iterations they recompute the residual **exactly** from scratch
(`r = b - A·x`, discarding the recursively-updated residual) and reset the
search direction to `p = z` — dropping the Fletcher-Reeves β blend for that
one step. This corrects the residual's round-off drift over long runs and
is load-bearing for the *solve*: it's not something this investigation
touched. But it means the sequence of `(αᵢ, βᵢ)` production CG actually
produces is **not** the coefficient sequence of one long, unbroken Lanczos
process — it's several *independent* short Lanczos processes concatenated,
each one starting fresh (in the Krylov sense) at a restart.

The from-scratch prototype above didn't model this at all (it ran a
textbook CG with no restarts). Modeling it exposes exactly the missing
mechanism:

| n=4000, θ=0.15, `tol=1e-4`, ~2270 CG iterations needed | log-determinant relative error |
|---|--:|
| No restart (prototype above) | 1.1e-3 |
| `restart_every = 1000` | 1.1e-3 (identical — no restart actually triggers) |
| `restart_every = 200` | **213%** (wrong order of magnitude) |
| `restart_every = 50` (production default) | **237%**, wrong SIGN of the estimate |

Reconstructing the tridiagonal from *only the segment since the last
restart* (the natural thing to do — a restart truly does start a fresh,
independent Krylov subspace, so earlier segments' `(α, β)` don't belong in
the same tridiagonal) throws away all but the most recent
`iterations_needed mod restart_every` steps. At `restart_every=50` with
~2270 iterations needed, that is somewhere in `[0, 50)` steps — nowhere
near enough to resolve `R`'s spectrum, and the SLQ estimate comes out
essentially uncorrelated with the truth (confirmed at θ=0.3 too: 169%
error). At `restart_every=1000`, the *lucky* alignment (last segment ≈ 270
steps, because 2270 mod 1000 = 270) happens to be long enough — but that
is luck, not a property you can rely on: the same run at a slightly
different `n` (or `cg_tol`, or probe realization) could converge right
after a restart and leave a last segment of a handful of steps.

**This is almost certainly the actual mechanism behind the original
"tried and reverted" finding** — not the reorthogonalization-vs-conditioning
story the comment gives (which this investigation's Step 1 couldn't
reproduce in isolation), but restart-segment fragmentation, which the
original attempt likely hit simply by using the production CG as-is.

## Why this isn't a quick fix

Two ways around it, both rejected:

1. **Combine every segment's quadrature instead of just the last.**
   Mathematically unsound as a drop-in fix: SLQ's unbiasedness relies on
   the starting vector being a genuine Rademacher probe, independent of
   `A`. A restart's starting vector is the *actual CG residual* at that
   point — strongly correlated with `A` and with the probe's own history,
   not a fresh random draw. Averaging per-segment quadratures as if they
   were independent SLQ probes risks introducing **systematic bias**, not
   just extra variance, and there is no off-the-shelf formula for the
   correct correction. This is a real, open numerical-analysis question,
   not an implementation detail.
2. **Raise (or disable) `restart_every`, specifically for the fused
   probe/log-det solve.** Recovers accuracy in every test above (`restart_every
   = 1000` and "disabled" gave identical results to no-restart at all,
   confirming the restart correction wasn't needed for solve *accuracy* in
   these regimes — see the "solve quality" column, unaffected across every
   `restart_every` tested). But this reintroduces the exact round-off-drift
   risk the restart exists to correct, and it's least safe exactly where it
   would matter most: `n = 16000`/`32000` already show the probe CG solve
   failing to converge within its `6n`-iteration budget even *with*
   restarts active (see [Iterative.md](Iterative.md)'s CG iteration-budget
   discussion) — i.e. the regime with the most accumulated iterations
   (tens of thousands) and the least headroom to absorb extra drift is
   precisely the regime this investigation didn't get to validate `restart_every`
   changes in (the sweep above only went up to n=8000/~2270 iterations).

## Conclusion

Phase B of the mBCG idea — reading `log|R|` directly off the fused CG's
own coefficients, eliminating the dedicated Lanczos pass — is **not
implemented**, and shouldn't be without either (a) a principled way to
combine multiple restart-segment SLQ estimates without introducing bias,
which is genuine unsolved numerical-analysis work as far as this
investigation could tell, not a tuning parameter, or (b) validating that
disabling restarts for the probe solve specifically is safe at the `n`
where it matters most (16000+), which is exactly the regime already
flagged elsewhere as running into CG non-convergence.

What *is* implemented and validated, from the same broader effort:

- **Phase A of mBCG** (fusing the `[F|y]` and probe CG solves into one
  Krylov pass, keeping the dedicated Lanczos for `log|R|`) — safe, no new
  numerical assumptions, real measured speedup. See
  [Iterative.md](Iterative.md) and the commit adding the per-column-
  tolerance `conjugateGradient`/`conjugateGradientBatched` overloads.
- **Device-resident SLQ Lanczos** (`LinearAlgebraCuda::stochasticLogDetBatched`)
  — the dedicated Lanczos pass phase B would have eliminated is itself no
  longer paying a host↔device round trip per step. See
  [Iterative.md](Iterative.md).

If this is revisited in the future, the two open questions above (bias-free
multi-segment combination; restart safety at n≥16000) are exactly where to
start — not from scratch on whether mBCG is numerically viable at all,
which this document's Step 1 already answers "yes, comfortably" for the
non-restarted case.
