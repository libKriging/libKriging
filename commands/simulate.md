---
description: Draw conditional sample paths from a fitted libKriging model at new input points.
argument-hint: "[model var] [new-points source] [nsim] [seed]"
---

Simulate sample paths from a fitted libKriging model. Context / arguments: $ARGUMENTS

1. Identify the fitted model, the points `Xnew` (`n × d`, same `d` as
   training `X`), and `nsim` / `seed`. Simulation is **conditional on the
   training data by default** — the paths interpolate the observed points
   (exactly for noise-free `Kriging`, up to the noise/nugget otherwise).

2. Open `skills/libkriging/references/<language>.md` for the exact
   `simulate` signature. Pass `nsim` and `seed` with the language's integer
   type: `int32(...)` in Octave/MATLAB, `Int`/`Int32` in Julia — a plain
   double raises a low-level error far from the call site.

3. Result is `nsim × npred` (one row per path, or per that file's stated
   orientation). For a smooth path picture, use a dense ordered `Xnew`.

4. Report the result shape and a summary (per-point mean and an empirical
   band across paths). If the user then wants to condition on further
   points, point them at `/libkriging:update` rather than re-fitting.
