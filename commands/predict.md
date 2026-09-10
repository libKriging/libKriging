---
description: Predict mean and uncertainty from a fitted libKriging model at new input points.
argument-hint: "[model var] [new-points source] [need: stdev|cov|deriv]"
---

Predict with a fitted libKriging model. Context / arguments: $ARGUMENTS

1. Identify the fitted model object and the new inputs `Xnew`. `Xnew` must
   be `n × d` with the **same `d`** (and same column meaning) as the
   training `X` — never a `d × n` layout.

2. Open the matching `skills/libkriging/references/<language>.md` for the
   exact `predict` signature. Choose what to compute:
   - pointwise mean + `stdev` → the default (`return_stdev=True`).
   - joint predictive covariance across `Xnew` (e.g. for consistent
     multi-point uncertainty or downstream sampling) → also `return_cov=True`.
     Do not request the full covariance for large `Xnew` unless it is needed.
   - gradient of mean / stdev w.r.t. inputs (sensitivity, gradient-based
     optimization) → `return_deriv=True`.

3. Mind the return shape: in Python `predict()` always returns the fixed
   5-tuple `(mean, stdev, cov, mean_deriv, stdev_deriv)` — the flags only
   decide which entries are filled, not the arity. R/Julia/Octave-MATLAB
   return the analogous named fields per their reference file.

4. Report mean ± stdev (and cov/deriv if requested). Flag any `Xnew` rows
   outside the training design's per-column range: Kriging reverts toward
   the trend with inflating variance there — the prediction is valid but
   an extrapolation, call it out.
