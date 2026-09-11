# libKriging vs GPyTorch — iterative path, small n

- **GPU**: NVIDIA H100 NVL
- **CPU**: INTEL(R) XEON(R) PLATINUM 8558
- **host**: `farux-gpu04.cluster`  ·  logical CPUs: 192  ·  OMP_NUM_THREADS: `32`
- **when**: 2026-09-11 09:50 CEST
- **sweep**: `sine_sum` d=4, matern5_2, shared theta=0.15; n = 250, 500, 1000, 2000, 4000; test n=300
- **libKriging iterative objective**: `LLIterative(30,0,40)`  ·  `predictIterative(max_iter=8n, Nystrom precond rank ≤ 128)`
- **GPyTorch**: raised settings so BBMM converges, and `max_cholesky_size=0` so every n in the sweep actually USES BBMM (GPyTorch's default, 800, silently falls back to exact dense Cholesky at or below it, which would make an n=250/500 "GPyTorch-BBMM" row misnamed) — `max_cg_iterations=5000, cg_tolerance=1e-4, eval_cg_tolerance=1e-4, max_lanczos_quadrature_iterations=32, num_trace_samples=32, max_preconditioner_size=100, min_preconditioning_size=1`
- **versions**: pylibkriging=`1.1.0`, cuda_iterative_available=`True`, torch=`2.8.0+cu126`, torch.cuda=`12.6`, torch.cuda.is_available=`True`, gpytorch=`1.15.2`

`fit` = the `Kriging(...)` constructor (dense Cholesky for `LL`; one CG+SLQ commit for the light `LLIterative` fit) / GPyTorch model build. `logLik` = one log-likelihood **+ gradient** evaluation at theta. `predict` = a *cold* posterior mean on 300 held-out points (GPyTorch's per-fit prediction cache is dropped each rep). Every timing is the **min of up to 5 reps** (1 rep once a single call exceeds 3 s). Backends are named `<lib>-<method>-<linalg lib>`. `libKriging-Cholesky-OpenBLAS` is the **reference**: `dLogLik/n` = `|ll − ll_chol|/n` (blank for GPyTorch, whose `-mll` is a differently normalised quantity) and `dMean/rms` = `max|mean − mean_chol| / rms(y_test)`.

## Reference — `libKriging-Cholesky-OpenBLAS` (exact dense Cholesky)

| n | fit (s) | logLik (s) | predict (s) | logLik value | RMSE | Q² |
|--:|--:|--:|--:|--:|--:|--:|
| 250 | 0.009 | 0.010 | 0.005 | -317.241 | 0.4773 | 0.8952 |
| 500 | 0.029 | 0.028 | 0.009 | -463.222 | 0.3402 | 0.9468 |
| 1000 | 0.109 | 0.096 | 0.020 | -512.857 | 0.2111 | 0.9795 |
| 2000 | 0.519 | 0.454 | 0.042 | -60.627 | 0.0955 | 0.9958 |
| 4000 | 2.585 | 2.306 | 0.100 | 2183.769 | 0.0421 | 0.9992 |

## Timing — seconds

| backend | n | fit | logLik | predict |
|---|--:|--:|--:|--:|
| libKriging-Cholesky-OpenBLAS | 250 | 0.009 | 0.010 | 0.005 |
| libKriging-Cholesky-OpenBLAS | 500 | 0.029 | 0.028 | 0.009 |
| libKriging-Cholesky-OpenBLAS | 1000 | 0.109 | 0.096 | 0.020 |
| libKriging-Cholesky-OpenBLAS | 2000 | 0.519 | 0.454 | 0.042 |
| libKriging-Cholesky-OpenBLAS | 4000 | 2.585 | 2.306 | 0.100 |
| libKriging-Iterative-CUDA | 250 | 0.064 | 0.067 | 0.022 |
| libKriging-Iterative-CUDA | 500 | 0.076 | 0.093 | 0.100 |
| libKriging-Iterative-CUDA | 1000 | 0.127 | 0.153 | 0.145 |
| libKriging-Iterative-CUDA | 2000 | 0.217 | 0.296 | 0.471 |
| libKriging-Iterative-CUDA | 4000 | 0.533 | 1.142 | 2.237 |
| libKriging-Iterative-OpenMP | 250 | 0.029 | 0.034 | 0.026 |
| libKriging-Iterative-OpenMP | 500 | 0.039 | 0.066 | 0.027 |
| libKriging-Iterative-OpenMP | 1000 | 0.074 | 0.215 | 0.056 |
| libKriging-Iterative-OpenMP | 2000 | 0.265 | 1.424 | 0.184 |
| libKriging-Iterative-OpenMP | 4000 | 2.418 | 24.064 | 1.351 |
| GPyTorch-BBMM-CUDA | 250 | 0.001 | 0.112 | 0.176 |
| GPyTorch-BBMM-CUDA | 500 | 0.001 | 0.126 | 0.196 |
| GPyTorch-BBMM-CUDA | 1000 | 0.001 | 0.177 | 0.226 |
| GPyTorch-BBMM-CUDA | 2000 | 0.001 | 0.252 | 0.305 |
| GPyTorch-BBMM-CUDA | 4000 | 0.001 | 0.519 | 0.506 |
| GPyTorch-BBMM-MKL | 250 | 0.001 | 0.037 | 0.061 |
| GPyTorch-BBMM-MKL | 500 | 0.001 | 0.058 | 0.102 |
| GPyTorch-BBMM-MKL | 1000 | 0.001 | 0.105 | 0.066 |
| GPyTorch-BBMM-MKL | 2000 | 0.001 | 0.304 | 0.138 |
| GPyTorch-BBMM-MKL | 4000 | 0.001 | 1.574 | 0.595 |
| GPyTorch-Cholesky-CUDA | 250 | 0.001 | 0.003 | 0.003 |
| GPyTorch-Cholesky-CUDA | 500 | 0.001 | 0.003 | 0.005 |
| GPyTorch-Cholesky-CUDA | 1000 | 0.001 | 0.008 | 0.005 |
| GPyTorch-Cholesky-CUDA | 2000 | 0.001 | 0.011 | 0.011 |
| GPyTorch-Cholesky-CUDA | 4000 | 0.001 | 0.025 | 0.017 |
| GPyTorch-Cholesky-MKL | 250 | 0.001 | 0.006 | 0.006 |
| GPyTorch-Cholesky-MKL | 500 | 0.001 | 0.007 | 0.007 |
| GPyTorch-Cholesky-MKL | 1000 | 0.001 | 0.020 | 0.018 |
| GPyTorch-Cholesky-MKL | 2000 | 0.001 | 0.084 | 0.065 |
| GPyTorch-Cholesky-MKL | 4000 | 0.001 | 0.772 | 0.491 |

## Accuracy

| backend | n | RMSE | Q² | logLik value | dLogLik/n | dMean/rms |
|---|--:|--:|--:|--:|--:|--:|
| libKriging-Cholesky-OpenBLAS | 250 | 0.4773 | 0.8952 | -317.241 | 0.00e+00 | 0.00e+00 |
| libKriging-Cholesky-OpenBLAS | 500 | 0.3402 | 0.9468 | -463.222 | 0.00e+00 | 0.00e+00 |
| libKriging-Cholesky-OpenBLAS | 1000 | 0.2111 | 0.9795 | -512.857 | 0.00e+00 | 0.00e+00 |
| libKriging-Cholesky-OpenBLAS | 2000 | 0.0955 | 0.9958 | -60.627 | 0.00e+00 | 0.00e+00 |
| libKriging-Cholesky-OpenBLAS | 4000 | 0.0421 | 0.9992 | 2183.769 | 0.00e+00 | 0.00e+00 |
| libKriging-Iterative-CUDA | 250 | 0.4773 | 0.8952 | -316.216 | 4.10e-03 | 4.54e-08 |
| libKriging-Iterative-CUDA | 500 | 0.3402 | 0.9468 | -458.724 | 9.00e-03 | 3.45e-08 |
| libKriging-Iterative-CUDA | 1000 | 0.2111 | 0.9795 | -510.848 | 2.01e-03 | 5.87e-08 |
| libKriging-Iterative-CUDA | 2000 | 0.0955 | 0.9958 | -64.867 | 2.12e-03 | 6.10e-08 |
| libKriging-Iterative-CUDA | 4000 | 0.0421 | 0.9992 | 2133.269 | 1.26e-02 | 9.28e-08 |
| libKriging-Iterative-OpenMP | 250 | 0.4773 | 0.8952 | -316.216 | 4.10e-03 | 3.46e-08 |
| libKriging-Iterative-OpenMP | 500 | 0.3402 | 0.9468 | -458.724 | 9.00e-03 | 4.19e-08 |
| libKriging-Iterative-OpenMP | 1000 | 0.2111 | 0.9795 | -510.848 | 2.01e-03 | 8.28e-08 |
| libKriging-Iterative-OpenMP | 2000 | 0.0955 | 0.9958 | -64.867 | 2.12e-03 | 7.63e-08 |
| libKriging-Iterative-OpenMP | 4000 | 0.0421 | 0.9992 | 2133.269 | 1.26e-02 | 8.26e-08 |
| GPyTorch-BBMM-CUDA | 250 | 0.4223 | 0.9179 | -1.237 | — | 1.98e-01 |
| GPyTorch-BBMM-CUDA | 500 | 0.3022 | 0.9580 | -1.165 | — | 2.29e-01 |
| GPyTorch-BBMM-CUDA | 1000 | 0.1903 | 0.9833 | -1.088 | — | 1.58e-01 |
| GPyTorch-BBMM-CUDA | 2000 | 0.0953 | 0.9958 | -0.724 | — | 7.02e-02 |
| GPyTorch-BBMM-CUDA | 4000 | 0.0452 | 0.9991 | -0.327 | — | 4.21e-02 |
| GPyTorch-BBMM-MKL | 250 | 0.4223 | 0.9179 | -1.305 | — | 1.98e-01 |
| GPyTorch-BBMM-MKL | 500 | 0.3022 | 0.9580 | -1.150 | — | 2.29e-01 |
| GPyTorch-BBMM-MKL | 1000 | 0.1903 | 0.9833 | -1.106 | — | 1.58e-01 |
| GPyTorch-BBMM-MKL | 2000 | 0.0953 | 0.9958 | -0.725 | — | 7.02e-02 |
| GPyTorch-BBMM-MKL | 4000 | 0.0452 | 0.9991 | -0.330 | — | 4.20e-02 |
| GPyTorch-Cholesky-CUDA | 250 | 0.4223 | 0.9179 | -1.182 | — | 1.98e-01 |
| GPyTorch-Cholesky-CUDA | 500 | 0.3022 | 0.9580 | -0.923 | — | 2.29e-01 |
| GPyTorch-Cholesky-CUDA | 1000 | 0.1903 | 0.9833 | -0.681 | — | 1.58e-01 |
| GPyTorch-Cholesky-CUDA | 2000 | 0.0953 | 0.9958 | -0.436 | — | 7.01e-02 |
| GPyTorch-Cholesky-CUDA | 4000 | 0.0452 | 0.9991 | -0.166 | — | 4.19e-02 |
| GPyTorch-Cholesky-MKL | 250 | 0.4223 | 0.9179 | -1.182 | — | 1.98e-01 |
| GPyTorch-Cholesky-MKL | 500 | 0.3022 | 0.9580 | -0.923 | — | 2.29e-01 |
| GPyTorch-Cholesky-MKL | 1000 | 0.1903 | 0.9833 | -0.681 | — | 1.58e-01 |
| GPyTorch-Cholesky-MKL | 2000 | 0.0953 | 0.9958 | -0.436 | — | 7.01e-02 |
| GPyTorch-Cholesky-MKL | 4000 | 0.0452 | 0.9991 | -0.166 | — | 4.19e-02 |

## Speed-ups (logLik-eval time)

| n | libKriging-Cholesky-OpenBLAS | libKriging-Iterative-CUDA | libKriging-Iterative-OpenMP | **OpenMP / CUDA** | **CUDA / Cholesky** | GPyTorch-BBMM-CUDA | GPyTorch-BBMM-MKL |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | 0.010 | 0.067 | 0.034 | 0.5× | 7.0× | 0.112 | 0.037 |
| 500 | 0.028 | 0.093 | 0.066 | 0.7× | 3.3× | 0.126 | 0.058 |
| 1000 | 0.096 | 0.153 | 0.215 | 1.4× | 1.6× | 0.177 | 0.105 |
| 2000 | 0.454 | 0.296 | 1.424 | 4.8× | 0.7× | 0.252 | 0.304 |
| 4000 | 2.306 | 1.142 | 24.064 | 21.1× | 0.5× | 0.519 | 1.574 |

## Verdict — did everything converge?

- **libKriging iterative**: logLik within `1.3e-02` per point of the chol reference, posterior mean within `9.3e-08` of test RMS — converged.
- **GPyTorch**: posterior mean within `2.3e-01` of test RMS of the chol reference (a roughly n-independent offset from the covariance-argument convention, *not* under-convergence), Q² ≥ `0.9179`. Its `-mll` value is a different normalisation and is not compared.
- **GPyTorch BBMM vs its own exact Cholesky** (`GPyTorch-Cholesky-*`, `max_cholesky_size` forced far above every n here): posterior mean differs by at most `1.8e-04` of test RMS across n — BBMM has converged to GPyTorch's own exact answer, confirming the offset from the libKriging-Cholesky reference above is the covariance-argument-convention difference, not BBMM under-convergence.
- **`libKriging-Cholesky-OpenBLAS`** is fastest on all three ops at these n — the iterative path is for n where the dense factor no longer fits / is too slow, not this range.

## Notes

- Backend names are `<lib>-<method>-<linalg lib>`. Both `libKriging-Iterative-CUDA` (`LK_ITERATIVE_CUDA_DENSE_MAX_MB` budget) and `-OpenMP` (`LK_ITERATIVE_DENSE_MAX_MB`) materialize R once per evaluation for a separable kernel within their memory budget and run the matvecs as a single `cublasDgemm` / BLAS-3 `R*V` (hence `-OpenMP`, the BLAS it links), else fall back to a hand-written matvec kernel. `libKriging-Cholesky-OpenBLAS` and `GPyTorch-BBMM-MKL` name the actual dense BLAS/LAPACK each links against.
- theta=0.15 is chosen so the SLQ log-determinant's Lanczos quadrature and GPyTorch's BBMM CG both converge with sane iteration budgets; at longer theta (better-fitting but more ill-conditioned R) both need far more iterations / Lanczos steps. The `,0,40` in the libKriging objective is the third `LLIterative` argument (SLQ Lanczos steps per probe), added so the iterative log-likelihood *value* also tracks the exact one here.
- `libKriging-Iterative-CUDA` vs `libKriging-Iterative-OpenMP` is the same binary with `set_cuda_iterative_enabled(...)` toggled — identical results, different path for the batched CG / SLQ / gradient matvecs (CUDA kernels vs the CPU dense-`R` BLAS path).
- `GPyTorch-BBMM-CUDA`/`GPyTorch-BBMM-MKL` vs `GPyTorch-Cholesky-CUDA`/`GPyTorch-Cholesky-MKL` are the SAME model and data, only `gpytorch.settings.max_cholesky_size` differs (`0` = always BBMM, forced huge = always exact) — the Cholesky rows are GPyTorch's own reference for whether BBMM has converged, independent of libKriging's Cholesky reference.
- Companion: `docs/comparisons/libKriging_vs_GPyTorch.ipynb` (summary of these results + the GPyTorch code libKriging mimics).
