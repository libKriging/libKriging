Drivers of the thread-budget and fixed Cholesky jitter studies
([docs/dev/parallel.md](../../docs/dev/parallel.md), [RESULTS.md](RESULTS.md),
[docs/dev/CholJitter.md](../../docs/dev/CholJitter.md)). Their CSV output is not kept
in the repository (`.gitignore`): rerun them, then the `summarize*.py` scripts in the
output directory. Machine of the published numbers: `machine.txt`.

- Timing sweeps: `bench/bench-threads.sh` / `bench/bench-threads-ops.sh <build-dir> out.csv`
  (built by CMake), `summarize.py out.csv`.
- Jitter versus n (#426): `run_jitter_n.sh <jitter_n_study binary> <out dir>`, then
  `summarize_jitter_n.py <out dir>`; `jitter_n_study` marks each fit on stderr.
- Study drivers (`*_study.cpp`, `*_probe.cpp`, `unstable_ll_profile.cpp`) are not built by
  CMake: compile them with the flags of `bench-fit-profile` from
  `<build-dir>/compile_commands.json`, then link with
  `g++ -O2 x.o -o x -L<build-dir>/src/lib -lKriging -Wl,-rpath,<build-dir>/src/lib -fopenmp`.
- Measure after a warm-up (`LK_BENCH_WARMUP_MS`, 300 ms: OpenBLAS pthreads threads spin
  ~100 ms after loading) and compare fits by work done (iterations, restarts), not only by
  time: a different BLAS thread count can change the optimizer path.
