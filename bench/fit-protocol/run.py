#!/usr/bin/env python3
"""Run the fit protocol of the statistical benchmark base (T7) over a profile.

    run.py --bin <build>/bench/bench-fit-protocol --out <dir> [--profile smoke|reference|full]
           [--variant SPEC ...] [--sets N] [--summary]

A profile is a list of chunks (kernel x noise x objective x function x n, with
a number of data sets); each variant runs every chunk in its own process, one
CSV per (variant, chunk) under <dir>/<variant>/. Existing CSVs are kept, so an
interrupted run resumes where it stopped. Chunks run one at a time: the fits
use every core (parallel starts, BLAS threads), and concurrent processes would
distort the times.

A variant SPEC is a whitespace-separated list of tokens:
    NAME=VALUE      environment variable of the run (LK_LONG_RANGE_RESTART=0,
                    LK_CHOL_JITTER=0, LK_NUM_THREADS_PERCENT=25, ...)
    only=NOISE/OBJ  restrict the variant to these chunks (e.g. only=none/LL,
                    several separated by ';'): settings that act on part of
                    the grid only (the long-range restart: noise-free LL)
    OPTIMS          comma-separated optimizers (BFGS,BFGS4,BFGS10, ...)
Default variants: "BFGS,BFGS4,BFGS10" and
"LK_LONG_RANGE_RESTART=0 BFGS only=none/LL".

Then: merge the CSVs (--merge out.csv.gz), write machine.txt (always) and,
with --summary, the tables (summarize.py) in <dir>/summary.md.
"""
import argparse
import gzip
import os
import platform
import re
import subprocess
import sys
import time

KERNELS = ["gauss", "matern5_2"]
FUNCTIONS = ["sin4", "hartmann6", "branin4", "sinhalf8"]
NOISE_OBJ = [("none", "LL"), ("none", "LOO"), ("none", "LMP"), ("nugget", "LL"), ("nugget", "LMP")]


def grid(ns, noise_obj, sets, kernels=KERNELS, functions=FUNCTIONS):
    return [dict(kernel=k, noise=no, objective=ob, function=f, n=n, sets=sets)
            for n in ns for (no, ob) in noise_obj for k in kernels for f in functions]


PROFILES = {
    # about 2 min on 4 cores: every kernel, noise, objective and function at n 100, 5 data sets
    "smoke": grid([100], NOISE_OBJ, 5),
    # committed reference (bench/fit-protocol/results): the full grid up to n 200,
    # the LL objectives at n 500, noise-free LL at n 1000 (the LOO / LMP fits at
    # n 1000 cost 1-6 min per data set on 4 cores)
    "reference": grid([100, 200], NOISE_OBJ, 30)
    + grid([500], [("none", "LL"), ("nugget", "LL")], 30)
    + grid([1000], [("none", "LL")], 30),
    # the whole T7 grid: about 30 h on 4 cores
    "full": grid([100, 200, 500, 1000], NOISE_OBJ, 30),
}

DEFAULT_VARIANTS = ["BFGS,BFGS4,BFGS10", "LK_LONG_RANGE_RESTART=0 BFGS only=none/LL"]


def parse_variant(spec):
    env, only, optims = {}, None, []
    for tok in spec.split():
        if tok.startswith("only="):
            only = {tuple(x.split("/")) for x in tok[5:].split(";")}
        elif "=" in tok:
            k, v = tok.split("=", 1)
            env[k] = v
        else:
            optims += tok.split(",")
    if not optims:
        raise SystemExit("variant without optimizer: " + spec)
    slug = "-".join([",".join(optims)] + ["%s=%s" % kv for kv in sorted(env.items())])
    slug = re.sub(r"[^A-Za-z0-9_.,=-]", "_", slug)
    return dict(env=env, only=only, optims=optims, slug=slug, spec=spec)


def machine_description(binary):
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    lines = ["# " + time.strftime("%Y-%m-%d") + " " + platform.node(), platform.platform(), ""]
    if sys.platform.startswith("linux"):
        lines += [sh("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|Socket|MHz|L[23]'"), "",
                  "nproc: " + sh("nproc"), sh("free -g | head -2"), ""]
        lines.append(sh("ldd %s | grep -i -E 'blas|lapack|gomp|omp'" % binary))
    elif sys.platform == "darwin":
        lines += [sh("sysctl -n machdep.cpu.brand_string"), "ncpu: " + sh("sysctl -n hw.ncpu"),
                  sh("otool -L %s" % binary)]
    lines += ["", sh("c++ --version | head -1"), sh("cmake --version | head -1"),
              "libKriging " + sh("git -C %s rev-parse --short HEAD" % os.path.dirname(os.path.abspath(__file__)))]
    env = {k: v for k, v in os.environ.items() if k.startswith(("LK_", "OMP_", "OPENBLAS_", "MKL_"))}
    lines.append("environment: " + (" ".join("%s=%s" % kv for kv in sorted(env.items())) or "(no LK_/OMP_/BLAS variable)"))
    return "\n".join(lines) + "\n"


def merge(out, dest):
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(out) for f in fs if f.endswith(".csv"))
    with gzip.open(dest, "wt", newline="") as g:
        header = None
        for p in files:
            with open(p) as fh:
                h = fh.readline()
                if header is None:
                    header = h
                    g.write(h)
                elif h != header:
                    raise SystemExit("different CSV header in " + p)
                g.write(fh.read())
    print("merged %d files into %s" % (len(files), dest))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="bench-fit-protocol executable")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--profile", default="smoke", choices=sorted(PROFILES))
    ap.add_argument("--variant", action="append", help="variant SPEC (repeatable), see above")
    ap.add_argument("--sets", type=int, help="data sets per chunk (overrides the profile)")
    ap.add_argument("--merge", help="write every CSV of --out to this .csv.gz")
    ap.add_argument("--summary", action="store_true", help="write <out>/summary.md (summarize.py)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()

    variants = [parse_variant(s) for s in (a.variant or DEFAULT_VARIANTS)]
    chunks = PROFILES[a.profile]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "machine.txt"), "w") as fh:
        fh.write(machine_description(a.bin))
    todo = [(v, c) for c in chunks for v in variants if v["only"] is None or (c["noise"], c["objective"]) in v["only"]]
    t_start = time.time()
    for i, (v, c) in enumerate(todo):
        sets = a.sets or c["sets"]
        d = os.path.join(a.out, v["slug"])
        f = os.path.join(d, "%(kernel)s-%(noise)s-%(objective)s-%(function)s-%(n)d.csv" % c)
        if os.path.exists(f):
            continue
        os.makedirs(d, exist_ok=True)
        cmd = [a.bin, "--optim", ",".join(v["optims"]), "--kernel", c["kernel"], "--noise", c["noise"],
               "--objective", c["objective"], "--function", c["function"], "--n", str(c["n"]), "--sets", str(sets),
               "--variant", " ".join("%s=%s" % kv for kv in sorted(v["env"].items())) or "default"]
        env = dict(os.environ, **v["env"])
        print("[%d/%d %.0f s] %s %s" % (i + 1, len(todo), time.time() - t_start,
                                        " ".join("%s=%s" % kv for kv in v["env"].items()), " ".join(cmd)), flush=True)
        if a.dry_run:
            continue
        with open(f + ".tmp", "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.DEVNULL, env=env)
        if r.returncode != 0:
            raise SystemExit("failed (%d): %s" % (r.returncode, " ".join(cmd)))
        os.replace(f + ".tmp", f)
    if a.dry_run:
        return
    if a.merge:
        merge(a.out, a.merge)
    if a.summary:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(a.out, "summary.md"), "w") as fh:
            subprocess.run([sys.executable, os.path.join(here, "summarize.py"), a.out], stdout=fh, check=True)
        print("wrote " + os.path.join(a.out, "summary.md"))


if __name__ == "__main__":
    main()
