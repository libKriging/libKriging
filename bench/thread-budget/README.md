Results of the thread-budget measurements (docs/dev/ThreadBudget-TODO.md, step 1).

- `reference-4cores-pthreads.csv`: the 4-core VM where the plan was written
  (Xeon 2.1 GHz, OpenBLAS pthreads, `kOmpMinRows = 800`), for comparison only.
  Note `t_dX_ms` ~ 8 ms with OpenBLAS at 4 threads and OpenMP at 2-4 threads
  versus ~ 0.2 ms otherwise at n = 200: the OpenBLAS / OpenMP contention.
- Files of the dedicated machine go next to it (see the TODO for their names).
