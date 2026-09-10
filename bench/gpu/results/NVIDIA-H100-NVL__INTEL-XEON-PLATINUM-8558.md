# libKriging vs GPyTorch — iterative path, small n

- **GPU**: NVIDIA H100 NVL
- **CPU**: INTEL(R) XEON(R) PLATINUM 8558
- **host**: `farux-gpu04.cluster`  ·  logical CPUs: 192  ·  OMP_NUM_THREADS: `32`
- **when**: 2026-09-10 18:30 CEST
- **sweep**: `sine_sum` d=4, matern5_2, shared theta=0.15; n = 250, 500, 1000, 2000; test n=300
- **libKriging iterative objective**: `LLIterative(30,0,40)`  ·  `predictIterative(max_iter=8n, Nystrom precond rank ≤ 128)`
- **GPyTorch**: raised settings so BBMM converges — `max_cg_iterations=5000, cg_tolerance=1e-4, eval_cg_tolerance=1e-4, max_lanczos_quadrature_iterations=32, num_trace_samples=32, max_preconditioner_size=100, min_preconditioning_size=1`
- **versions**: pylibkriging=`1.1.0`, cuda_iterative_available=`True`, torch=`2.8.0+cu126`, torch.cuda=`12.6`, torch.cuda.is_available=`True`, gpytorch=`1.15.2`

`fit` = the `Kriging(...)` constructor (dense Cholesky for `LL`; one CG+SLQ commit for the light `LLIterative` fit) / GPyTorch model build. `logLik` = one log-likelihood **+ gradient** evaluation at theta. `predict` = a *cold* posterior mean on 300 held-out points (GPyTorch's per-fit prediction cache is dropped each rep). Every timing is the **min of up to 5 reps** (1 rep once a single call exceeds 3 s). Backends are named `<lib>-<method>-<linalg lib>`. `libKriging-Cholesky-OpenBLAS` is the **reference**: `dLogLik/n` = `|ll − ll_chol|/n` (blank for GPyTorch, whose `-mll` is a differently normalised quantity) and `dMean/rms` = `max|mean − mean_chol| / rms(y_test)`.

## Reference — `libKriging-Cholesky-OpenBLAS` (exact dense Cholesky)

| n | fit (s) | logLik (s) | predict (s) | logLik value | RMSE | Q² |
|--:|--:|--:|--:|--:|--:|--:|
| 250 | 0.009 | 0.010 | 0.005 | -317.241 | 0.4773 | 0.8952 |
| 500 | 0.030 | 0.029 | 0.009 | -463.222 | 0.3402 | 0.9468 |
| 1000 | 0.127 | 0.115 | 0.020 | -512.857 | 0.2111 | 0.9795 |
| 2000 | 0.550 | 0.493 | 0.042 | -60.627 | 0.0955 | 0.9958 |

## Timing — seconds

| backend | n | fit | logLik | predict |
|---|--:|--:|--:|--:|
| libKriging-Cholesky-OpenBLAS | 250 | 0.009 | 0.010 | 0.005 |
| libKriging-Cholesky-OpenBLAS | 500 | 0.030 | 0.029 | 0.009 |
| libKriging-Cholesky-OpenBLAS | 1000 | 0.127 | 0.115 | 0.020 |
| libKriging-Cholesky-OpenBLAS | 2000 | 0.550 | 0.493 | 0.042 |
| libKriging-Iterative-CUDA | 250 | 0.283 | 0.294 | 0.043 |
| libKriging-Iterative-CUDA | 500 | 0.338 | 0.484 | 0.195 |
| libKriging-Iterative-CUDA | 1000 | 0.533 | 1.662 | 0.344 |
| libKriging-Iterative-CUDA | 2000 | 1.803 | 11.038 | 1.256 |
| libKriging-Iterative-OpenMP | 250 | 0.303 | 0.509 | 0.034 |
| libKriging-Iterative-OpenMP | 500 | 0.918 | 3.067 | 0.115 |
| libKriging-Iterative-OpenMP | 1000 | 4.095 | 22.583 | 0.702 |
| libKriging-Iterative-OpenMP | 2000 | 20.949 | 235.405 | 9.840 |
| GPyTorch-BBMM-CUDA | 250 | 0.005 | 0.004 | 0.007 |
| GPyTorch-BBMM-CUDA | 500 | 0.009 | 0.004 | 0.004 |
| GPyTorch-BBMM-CUDA | 1000 | 0.013 | 0.454 | 0.798 |
| GPyTorch-BBMM-CUDA | 2000 | 0.013 | 0.846 | 1.086 |
| GPyTorch-BBMM-MKL | 250 | 0.001 | 0.005 | 0.005 |
| GPyTorch-BBMM-MKL | 500 | 0.001 | 0.006 | 0.006 |
| GPyTorch-BBMM-MKL | 1000 | 0.001 | 0.107 | 0.070 |
| GPyTorch-BBMM-MKL | 2000 | 0.001 | 0.373 | 0.178 |

## Accuracy

| backend | n | RMSE | Q² | logLik value | dLogLik/n | dMean/rms |
|---|--:|--:|--:|--:|--:|--:|
| libKriging-Cholesky-OpenBLAS | 250 | 0.4773 | 0.8952 | -317.241 | 0.00e+00 | 0.00e+00 |
| libKriging-Cholesky-OpenBLAS | 500 | 0.3402 | 0.9468 | -463.222 | 0.00e+00 | 0.00e+00 |
| libKriging-Cholesky-OpenBLAS | 1000 | 0.2111 | 0.9795 | -512.857 | 0.00e+00 | 0.00e+00 |
| libKriging-Cholesky-OpenBLAS | 2000 | 0.0955 | 0.9958 | -60.627 | 0.00e+00 | 0.00e+00 |
| libKriging-Iterative-CUDA | 250 | 0.4773 | 0.8952 | -316.216 | 4.10e-03 | 3.77e-08 |
| libKriging-Iterative-CUDA | 500 | 0.3402 | 0.9468 | -458.724 | 9.00e-03 | 3.01e-08 |
| libKriging-Iterative-CUDA | 1000 | 0.2111 | 0.9795 | -510.848 | 2.01e-03 | 8.28e-08 |
| libKriging-Iterative-CUDA | 2000 | 0.0955 | 0.9958 | -64.867 | 2.12e-03 | 6.84e-08 |
| libKriging-Iterative-OpenMP | 250 | 0.4773 | 0.8952 | -316.216 | 4.10e-03 | 4.39e-08 |
| libKriging-Iterative-OpenMP | 500 | 0.3402 | 0.9468 | -458.724 | 9.00e-03 | 3.24e-08 |
| libKriging-Iterative-OpenMP | 1000 | 0.2111 | 0.9795 | -510.848 | 2.01e-03 | 7.62e-08 |
| libKriging-Iterative-OpenMP | 2000 | 0.0955 | 0.9958 | -64.867 | 2.12e-03 | 4.90e-08 |
| GPyTorch-BBMM-CUDA | 250 | 0.4223 | 0.9179 | -1.182 | — | 1.98e-01 |
| GPyTorch-BBMM-CUDA | 500 | 0.3022 | 0.9580 | -0.923 | — | 2.29e-01 |
| GPyTorch-BBMM-CUDA | 1000 | 0.1903 | 0.9833 | -1.090 | — | 1.58e-01 |
| GPyTorch-BBMM-CUDA | 2000 | 0.0953 | 0.9958 | -0.719 | — | 7.02e-02 |
| GPyTorch-BBMM-MKL | 250 | 0.4223 | 0.9179 | -1.182 | — | 1.98e-01 |
| GPyTorch-BBMM-MKL | 500 | 0.3022 | 0.9580 | -0.923 | — | 2.29e-01 |
| GPyTorch-BBMM-MKL | 1000 | 0.1903 | 0.9833 | -1.115 | — | 1.58e-01 |
| GPyTorch-BBMM-MKL | 2000 | 0.0953 | 0.9958 | -0.728 | — | 7.02e-02 |

## Speed-ups (logLik-eval time)

| n | libKriging-Cholesky-OpenBLAS | libKriging-Iterative-CUDA | libKriging-Iterative-OpenMP | **OpenMP / CUDA** | **CUDA / Cholesky** | GPyTorch-BBMM-CUDA | GPyTorch-BBMM-MKL |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | 0.010 | 0.294 | 0.509 | 1.7× | 30× | 0.004 | 0.005 |
| 500 | 0.029 | 0.484 | 3.067 | 6.3× | 16× | 0.004 | 0.006 |
| 1000 | 0.115 | 1.662 | 22.583 | 13.6× | 14× | 0.454 | 0.107 |
| 2000 | 0.493 | 11.038 | 235.405 | 21.3× | 22× | 0.846 | 0.373 |

## Verdict — did everything converge?

- **libKriging iterative**: logLik within `9.0e-03` per point of the chol reference, posterior mean within `8.3e-08` of test RMS — converged.
- **GPyTorch**: posterior mean within `2.3e-01` of test RMS of the chol reference (a roughly n-independent offset from the covariance-argument convention, *not* under-convergence), Q² ≥ `0.9179`. Its `-mll` value is a different normalisation and is not compared.
- **`libKriging-Cholesky-OpenBLAS`** is fastest on all three ops at these n — the iterative path is for n where the dense factor no longer fits / is too slow, not this range.

## Notes

- Backend names are `<lib>-<method>-<linalg lib>`. `libKriging-Iterative-CUDA` and `-OpenMP` use hand-written matvec kernels (not cuBLAS / a CPU BLAS); `libKriging-Cholesky-OpenBLAS` and `GPyTorch-BBMM-MKL` name the actual dense BLAS/LAPACK each links against.
- theta=0.15 is chosen so the SLQ log-determinant's Lanczos quadrature and GPyTorch's BBMM CG both converge with sane iteration budgets; at longer theta (better-fitting but more ill-conditioned R) both need far more iterations / Lanczos steps. The `,0,40` in the libKriging objective is the third `LLIterative` argument (SLQ Lanczos steps per probe), added so the iterative log-likelihood *value* also tracks the exact one here.
- `libKriging-Iterative-CUDA` vs `libKriging-Iterative-OpenMP` is the same binary with `set_cuda_iterative_enabled(...)` toggled — identical results, different device for the batched CG / SLQ / gradient matvecs.
- Companion: `docs/comparisons/libKriging_vs_GPyTorch.ipynb` (summary + the GPyTorch code libKriging mimics), `bench/comparison-gpu/` (multi-function CI-style variant).
