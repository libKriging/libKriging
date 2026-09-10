---
description: Fit a libKriging Gaussian-process model to data, choosing the class, kernel, trend, objective and optimizer.
argument-hint: "[data file / description] [language: cpp|python|r|julia|octave|matlab]"
---

Fit a libKriging model. Context / arguments: $ARGUMENTS

1. Read `skills/libkriging/SKILL.md` (bundled with this plugin). Use its §1
   decision tree to pick the class:
   - noise-free or noisy scalar response → `Kriging` (pass `noise="nugget"`
     or a per-observation variance vector if noisy — never `NuggetKriging`
     / `NoiseKriging`).
   - categorical / ordinal / non-linearly-scaled inputs, or a wanted input
     transform → `WarpKriging` (one warp spec per column, §4).
   - single deep joint feature map over all inputs → `MLPKriging`.
   - n ≳ few thousand → keep `Kriging`/`WarpKriging` with
     `objective="LLVecchia(m)"` (d ≲ 5, local structure) or `"LLNystrom(k)"`
     (higher d); n ~ 10⁴–10⁶ and partitionable → `NestedKriging`.

2. Pick options from §2: `kernel="matern5_2"` and `regmodel="constant"` and
   `objective="LL"` and `optim="BFGS"` as defaults; deviate only with a
   stated reason (e.g. `"BFGS10"` for an unstable fit, `"LMP"` for very few
   points, `normalize=True` for very different input scales).

3. Determine the target language from $ARGUMENTS or the surrounding code;
   ask only if there is no signal. Open the matching
   `skills/libkriging/references/<language>.md` for exact call syntax.

4. Check data layout before writing the call: `X` is `n × d` (rows =
   observations), `y` is a plain length-`n` vector (float64 / `Vector{Float64}`
   / `as.numeric(...)`), integer args passed with the language's integer type.

5. Emit the fit call. If the data is available and the binding is built,
   run it and report: chosen class + options (with rationale), fitted
   hyperparameters (`theta`, `sigma2`, `beta`, nugget/noise), and the
   objective value (`logLikelihood()` / `leaveOneOut()` / `logMargPost()`).
   State plainly if anything failed or was skipped.
