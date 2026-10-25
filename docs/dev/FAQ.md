This page collects questions about libKriging development

# Armadillo uses `-fpic` in Rcpp and `-fPIC` in C++ code, is it important ?

 Options `-fpic` and `-fPIC` are about shared libraries, to generate position independent code. 
 Whether to use -fPIC or -fpic to generate position independent code is target-dependent. 
 The -fPIC choice always works, but may produce larger code than -fpic 
 (mnenomic to remember this is that PIC is in a larger case, so it may produce larger amounts of code). 
 Using -fpic option usually generates smaller and faster code, but will have platform-dependent limitations, 
 such as the number of globally visible symbols or the size of the code. 
 The linker will tell you whether it fits when you create the shared library. 
 When in doubt, choose -fPIC, because it always works.
 
 More info: http://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html

 
 
 
# How many threads does libKriging use?

 One setting, the thread budget, bounds all of them: by default the physical cores of the
 machine (in the process's CPU affinity, within a container's CPU quota). The starts of a
 multistart fit (`"BFGS10"`, ...) get it first, one start per thread; what remains goes to the
 BLAS and the OpenMP loops of each start. Change it with `Optim::set_num_threads_percent(x)` or
 the environment variable `LK_NUM_THREADS_PERCENT=x` (x <= 1: a ratio, x > 1: a percentage of
 the cores, 0: 100 %); `Optim::get_num_threads()` returns the resulting number of threads.

 | | set | read |
 |---|---|---|
 | C++ | `Optim::set_num_threads_percent(50)` | `Optim::get_num_threads()` |
 | Python | `lk.Optim.set_num_threads_percent(50)` | `lk.Optim.get_num_threads()` |
 | R | `rlibkriging:::optim_set_num_threads_percent(50)` | `rlibkriging:::optim_get_num_threads()` |
 | Octave | `Optim.set_num_threads_percent(50)` | `Optim.get_num_threads()` |

 An explicit limit is never exceeded: `OMP_NUM_THREADS` (or `omp_set_num_threads` in the calling
 thread) caps the whole budget, `OPENBLAS_NUM_THREADS` the BLAS only. If you run several fits at
 once yourself (Python `multiprocessing`, R PSOCK / `future` workers, MPI, a job array on one
 node), each process uses the whole machine by default: give each one its part, e.g.
 `set_num_threads_percent(1 / k)` for k concurrent processes, or `OPENBLAS_NUM_THREADS=1
 OMP_NUM_THREADS=1` for many. Children of a `fork()` (R `parallel::mclapply`) already start
 with 1 thread. Details and measurements: [parallel.md](parallel.md).

# Is a fit reproducible with a fixed seed?

 With a fixed seed (`Random::reset_seed`, 123 by default) the multistart starting points are
 always the same, and two fits on the same machine, with the same BLAS library and the same
 BLAS thread count, give bit-identical results.

 The result may change with the BLAS thread count (`OPENBLAS_NUM_THREADS`, the number of cores),
 the BLAS library or its version, or the CPU (OpenBLAS selects AVX2 / AVX-512 kernels at run
 time): a multithreaded BLAS splits its sums by thread, so the rounding of the objective and of
 its gradient changes, by ~1e-15 relative. The OpenMP thread count has no effect: the
 OpenMP loops (covariance matrix, log-likelihood gradient) write independent entries or sum in
 a fixed order. Note that OpenBLAS also reads `OMP_NUM_THREADS` when `OPENBLAS_NUM_THREADS` is
 not set, so that this variable may still change the BLAS thread count. A build without
 OpenMP sums the gradient in another order (from n = 800 on), with the same consequences.

 These rounding differences stay at that level on a well-conditioned covariance matrix, but
 are amplified by its condition number: on noise-free, smooth data the optimum often lies
 at large `theta`, where `cond(R)` reaches 1e14 or more, the objective is then only known to
 ~1e-2 and L-BFGS-B may follow a different path and end at another local optimum, which is
 not necessarily as good: n = 40, d = 2, `"gauss"`, `"BFGS"` (cond(R) ~ 1e15 to 1e18) ends at
 log-likelihood 34.5 or 172.8 depending only on the summation order of the gradient.
 E.g. n = 500, d = 4, `y = sum(sin(3 x))`, `"BFGS20"`: log-likelihood 3376.32 with 1 BLAS thread,
 3375.68 with 2, 3375.63 with 4.

 Since the fixed Cholesky jitter (noise-free `gauss` and `matern5_2` fits), this mostly no longer
 changes the optimum: a small constant (1e-9 for `gauss`, 1e-10 for `matern5_2`) is added to the
 diagonal of the correlation matrix during the optimization, which keeps the objective smooth
 where `cond(R)` explodes; the model at the optimum is then factorized exactly and still
 interpolates. Across BLAS thread counts `theta` then differs only within the optimizer
 tolerance (~1e-8 relative, not bit-identical); on 240 noise-free test cases a different optimum
 was reached in 1.7 % of fits instead of 18 %. `LinearAlgebra::set_chol_jitter(0)` (or the
 environment variable `LK_CHOL_JITTER=0`) restores the former, exact objective; a fitted or
 saved model keeps the jitter it was fitted with. See [parallel.md](parallel.md) 8.6.

 To get the same result on several machines, set the same BLAS thread count everywhere (e.g.
 `OPENBLAS_NUM_THREADS=1`, at the cost of speed) and use the same BLAS build; a nugget or noise
 model, or tighter `theta` bounds, also reduce the sensitivity.

# Are my BLAS / LAPACK results correct? (`check_blas`)

 Some BLAS builds give wrong results on some CPUs, without any error message: fits then fail
 with "Exceed max numerical nugget", or return NaN or nonsense. `LinearAlgebra::check_blas()`
 (C++), `lk.LinearAlgebra.check_blas()` (Python), `rlibkriging:::linalg_check_blas()` (R),
 `LinearAlgebra.check_blas()` (Octave / MATLAB, `[ok, library, report]`), `check_blas()`
 (Julia) checks, in a fraction of a second, the routines libKriging relies on (matrix product,
 Cholesky, triangular and general solves, inverse, rcond, QR, symmetric eigen decomposition)
 against long double references, from one thread then from concurrent threads as the
 multistart fits do (the eigen decomposition, used by `simulate` only, from one thread). It returns `ok`, the BLAS identification (OpenBLAS configuration and
 kernels, MKL or BLIS version) and a report, with the workaround of a known faulty BLAS.

 Known faulty configurations ([#429](https://github.com/libKriging/libKriging/issues/429)),
 all fixed by an environment variable set **before** the process starts (it is read when the
 BLAS is loaded) or by another BLAS:

 | BLAS | Faulty | Workaround |
 |---|---|---|
 | OpenBLAS 0.3.20 `DYNAMIC_ARCH` (Ubuntu 22.04) | `Cooperlake` kernels: Intel Sapphire Rapids, Granite Rapids (Xeon 6), Cooper Lake; any CPU with avx512_bf16 when `OPENBLAS_CORETYPE` holds a value 0.3.20 does not know | `OPENBLAS_CORETYPE=SkylakeX`, or OpenBLAS >= 0.3.21 |
 | OpenBLAS 0.3.19 `DYNAMIC_ARCH` | `Cooperlake` and `SkylakeX` kernels | `OPENBLAS_CORETYPE=Haswell`, or OpenBLAS >= 0.3.21 |
 | MKL 2020 (Ubuntu 22.04 `intel-mkl`) | default Intel threading layer next to the GNU OpenMP runtime of libKriging: NaN or crashes | `MKL_THREADING_LAYER=GNU` (or `SEQUENTIAL`), or a recent MKL |

 Correct in the same tests: OpenBLAS 0.3.18, 0.3.21, 0.3.26, 0.3.34, the reference BLAS,
 BLIS 0.8 / 0.9, MKL 2026. Note that the bundled Armadillo picks its BLAS when libKriging is
 configured (FlexiBLAS, then MKL, then OpenBLAS, then `libblas`): installing MKL before building
 libKriging makes it use MKL.
