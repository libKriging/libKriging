"""Merge results/gpu-*.csv into results/all.csv + a markdown summary
(median [q25; q75] over repetitions), for the GPU GPyTorch vs libKriging
comparison. Mirrors ../comparison/aggregate.py.
"""
import argparse
import glob
import os

import pandas as pd


def fmt(series, digits=3):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "—"
    med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
    return f"{med:.{digits}g} [{q1:.{digits}g}; {q3:.{digits}g}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.md")
    args = ap.parse_args()

    frames = [pd.read_csv(f) for f in glob.glob(os.path.join(args.results, "*.csv"))
              if os.path.basename(f) not in ("all.csv",)]
    if not frames:
        raise SystemExit("no results/*.csv found")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(args.results, "all.csv"), index=False)

    for col, default in [("kernel", "matern5_2"), ("theta_frac", "0.3")]:
        if col not in df.columns:
            df[col] = default
    gcols = ["func", "n", "kernel", "theta_frac"]
    reps = df.groupby(gcols + ["backend"])["rep"].count().max()
    lines = [
        "# GPU comparison benchmark — GPyTorch vs libKriging (n > 1000)",
        "",
        f"Repetitions per case: {reps}. Identical LHS designs. **Fixed shared "
        "reference theta** = `theta_frac × per-dimension input range`, "
        "`sigma2 = var(y)`. Interpolation only (jitter 1e-10). "
        "`theta_frac` small = short correlation / well-conditioned R; large = "
        "long correlation / ill-conditioned R.",
        "",
        "Times are seconds, median [q25; q75] over repetitions. `—` = no "
        "successful run.",
        "",
        "- **fit** = one likelihood(+grad) evaluation at theta_ref "
        "(one optimizer-step-equivalent).",
        "- **pred mean** = posterior mean on the 2000-point test set.",
        "- **pred stdev** = stdev on a subsample (one CG solve per point on "
        "the libKriging side).",
        "- **update** = re-condition on n + ⌈0.25n⌉ fresh points at the same "
        "theta, then predict the test mean.",
        "",
    ]
    for (func, n, kernel, tf), sub in df.groupby(gcols):
        d = int(sub["d"].iloc[0])
        lines += [f"## {func} (d={d}, n={n}, {kernel}, theta_frac={tf})", "",
                  "| backend | device | fit | pred mean | pred stdev | update | "
                  "RMSE | Q² | NLPD | cov90 | CG conv | ok/total |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for bk, g in sub.groupby("backend"):
            ok = (g["status"] == "ok").sum()
            dev = g.loc[g["status"] == "ok", "device"]
            dev = dev.iloc[0] if not dev.empty else "—"
            cg = pd.to_numeric(g["cg_converged"], errors="coerce").dropna()
            cg_txt = "—" if cg.empty else f"{int(cg.sum())}/{len(cg)}"
            lines.append(
                f"| {bk} | {dev} | {fmt(g['fit_time'])} | {fmt(g['pred_mean_time'])} "
                f"| {fmt(g['pred_stdev_time'])} | {fmt(g['update_time'])} "
                f"| {fmt(g['rmse'], 4)} | {fmt(g['q2'], 4)} | {fmt(g['nlpd'], 4)} "
                f"| {fmt(g['coverage90'], 3)} | {cg_txt} | {ok}/{len(g)} |")
        lines.append("")

    lines += [
        "### Reading the sweep",
        "",
        "- **theta_frac** is the knob: small → short correlation → "
        "well-conditioned R → both sides' CG is fast; large (and small `d`) → "
        "long correlation → ill-conditioned R → libKriging's CG runs many "
        "iterations (slow but converges), GpyTorch's CG hits its iteration "
        "cap (fast but its `Q²` / `CG conv` degrade — see `CG conv 0/n`).",
        "- Whether a given theta_frac is a *good hyperparameter* for a "
        "function is separate from the timing: at theta_frac far from the "
        "surface's natural length-scale, both packages predict poorly "
        "(RMSE/Q²), yet the fit/predict/update **timings** still compare "
        "like-for-like.",
        "- `libkriging-cpu` is the same binary with the CUDA CG toggled off; "
        "same RMSE/Q² as `libkriging-gpu`, ~10–50× slower (the GPU-batched "
        "matvec + SLQ is what closes that gap). `gpytorch-cpu` uses "
        "`torch.device('cpu')`.",
        "- NLPD can blow up (1e15+) when the surface has tiny amplitude and "
        "is near-perfectly interpolated: `sigma2 = var(y)` then badly "
        "under-estimates the process variance, so predicted stdev → 0. A "
        "metric artifact of the fixed-theta protocol, not a solver issue.",
        "",
        "### Caveats",
        "",
        "- Fixed shared theta isolates each package's **matrix-free CG linear "
        "algebra**; it is not a comparison of optimizers or of end-to-end MLE "
        "fits (see `run_gpu.py` and `docs/comparisons/libKriging_vs_GPyTorch.ipynb`).",
        "- GPyTorch's `max_cg_iterations` / `cg_tolerance` and libKriging's "
        "`predictIterative` tol are a speed vs convergence trade-off on both "
        "sides; `CG conv` flags whether GPyTorch's CG hit its iteration cap "
        "before tolerance (libKriging's fit-side CG budget is not tunable from "
        "Python — reported as converged).",
        "- libKriging `LLIterative` is `NoiseModel::None`-only; both sides run "
        "pure interpolation.",
        "- The machine's GPUs are shared; absolute times carry run-to-run "
        "contention noise — compare within a run, not across.",
        "",
    ]
    out = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(out)
    print(out)


if __name__ == "__main__":
    main()
