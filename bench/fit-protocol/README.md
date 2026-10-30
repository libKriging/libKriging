Statistical benchmark base of the fit (T7 of [docs/dev/Rationalization.md](../../docs/dev/Rationalization.md);
results and decisions: [docs/dev/rationalization/T7-benchmark-base.md](../../docs/dev/rationalization/T7-benchmark-base.md)).

- `bench-fit-protocol` (built by `bench/CMakeLists.txt`, `bench/bench-fit-protocol.cpp`): fits
  every seeded data set of a grid (kernel x noise x objective x function x n) with each
  optimizer of `--optim`, one CSV row per fit (`--help` for the options and the columns).
  Settings that are not arguments come from the environment as for any libKriging program
  (`LK_LONG_RANGE_RESTART`, `LK_CHOL_JITTER`, `LK_NUM_THREADS_PERCENT`, ...): their effective
  values are in the `settings` column.
- `run.py`: runs a profile (`smoke`, `reference`, `full`) for a list of variants (environment +
  optimizers), one process per chunk, one CSV per (variant, chunk), resumable; writes
  `machine.txt`; `--merge x.csv.gz`, `--summary`.
- `summarize.py`: miss rate (objective more than 10 below the best variant of the data set,
  LOO on the log-density scale (n / 2) log(e / e_best)) with Wilson 95 % intervals, median
  test RMSE and RMSE / best, wall and CPU time, per configuration and variant.
- `results/`: the reference run (`reference.csv.gz`), its machine (`machine.txt`) and tables
  (`summary.md`).

Commands (from the repository root, after building with `cmake --build <build> --target bench-fit-protocol`):

```sh
# tables of the committed reference results
python3 bench/fit-protocol/summarize.py bench/fit-protocol/results/reference.csv.gz --detail

# smoke run (about 2 min on 4 cores: n 100, 5 data sets, every kernel / noise / objective / function)
python3 bench/fit-protocol/run.py --bin <build>/bench/bench-fit-protocol --out /tmp/fp-smoke --profile smoke --summary

# the reference run (about 5 h on 4 cores), then its tables
python3 bench/fit-protocol/run.py --bin <build>/bench/bench-fit-protocol --out /tmp/fp-ref --profile reference \
    --merge /tmp/fp-ref.csv.gz --summary

# other variants, compared on the same data sets (merge the directories or CSVs in summarize.py)
python3 bench/fit-protocol/run.py --bin <build>/bench/bench-fit-protocol --out /tmp/fp-ref --profile reference \
    --variant "LK_NUM_THREADS_PERCENT=25 BFGS,BFGS4,BFGS10" --variant "LK_CHOL_JITTER=0 BFGS only=none/LL"
```

Grid: kernels gauss, matern5_2; noise-free (`none`) and estimated nugget (`nugget`: y + N(0,
(0.05 sd(y))^2)); objectives LL, LOO (noise-free only: Kriging does not support LOO with a
nugget), LMP; functions `sin4` (all 4 inputs active), `hartmann6`, `branin4` (2 inactive inputs),
`sinhalf8` (4 inactive inputs); n 100, 200, 500, 1000; data sets 0-29 (X ~ U[0, 1]^{n x d}
seeded 9000 + set, test set of 500 points seeded 777: the data of
`bench/thread-budget/jitter_n_study.cpp`). The `reference` profile is the grid up to n 200,
the LL objectives at n 500 and noise-free LL at n 1000 (LOO / LMP fits at n 1000 take 1-6 min
per data set on 4 cores); `full` is the whole grid (about 30 h on 4 cores).

Run the chunks one at a time on an otherwise idle machine: the fits use every core (parallel
starts, BLAS threads). A miss is relative to the variants run: add the variant under study to
the same output directory (or merge its CSVs) so that the best of each data set includes the
multistart reference (BFGS10).
