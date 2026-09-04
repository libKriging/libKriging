---
description: Add new observations to a fitted libKriging model, optionally re-optimizing hyperparameters.
argument-hint: "[model var] [new X/y source] [refit: true|false]"
---

Update a fitted libKriging model with new observations. Context / arguments: $ARGUMENTS

1. Identify the fitted model and the new data `(X_u, y_u)`: `X_u` is
   `n_u × d` with the same `d` and column meaning as the original `X`;
   `y_u` is a plain length-`n_u` vector.

2. Open `skills/libkriging/references/<language>.md` for the exact `update`
   signature. Choose `refit`:
   - `refit=true` (recommended) — re-estimates the hyperparameters on the
     augmented data. Use it unless you have a specific reason not to.
   - `refit=false` — keeps the current hyperparameters and only extends the
     conditioning set. Reasonable for many small sequential updates (e.g.
     Bayesian optimization / active learning inner loop) where a full
     re-optimization each step is too costly; re-fit periodically.

3. `NestedKriging` has no in-place update path and no `noise=` channel —
   if the model is a `NestedKriging`, say so and re-fit instead.

4. After updating, report how the fit changed: new `theta` / `sigma2` /
   `beta` vs. the previous values, and the new objective value. Note if
   any new point lay outside the previous design range (it extends the
   region the model interpolates).
