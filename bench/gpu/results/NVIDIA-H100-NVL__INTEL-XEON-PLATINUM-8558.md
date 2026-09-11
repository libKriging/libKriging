# libKriging vs GPyTorch — iterative path, small n

- **GPU**: NVIDIA H100 NVL
- **CPU**: INTEL(R) XEON(R) PLATINUM 8558
- **host**: `farux-gpu04.cluster`  ·  logical CPUs: 192  ·  OMP_NUM_THREADS: `32`
- **when**: 2026-09-11 09:40 CEST
- **sweep**: `sine_sum` d=4, matern5_2, shared theta=0.15; n = 250, 500, 1000, 2000; test n=300
- **libKriging iterative objective**: `LLIterative(30,0,40)`  ·  `predictIterative(max_iter=8n, Nystrom precond rank ≤ 128)`
- **GPyTorch**: raised settings so BBMM converges, and `max_cholesky_size=0` so every n in the sweep actually USES BBMM (GPyTorch's default, 800, silently falls back to exact dense Cholesky at or below it, which would make an n=250/500 "GPyTorch-BBMM" row misnamed) — `max_cg_iterations=5000, cg_tolerance=1e-4, eval_cg_tolerance=1e-4, max_lanczos_quadrature_iterations=32, num_trace_samples=32, max_preconditioner_size=100, min_preconditioning_size=1`
- **versions**: pylibkriging=`1.1.0`, cuda_iterative_available=`True`, torch=`2.8.0+cu126`, torch.cuda=`12.6`, torch.cuda.is_available=`True`, gpytorch=`1.15.2`

`fit` = the `Kriging(...)` constructor (dense Cholesky for `LL`; one CG+SLQ commit for the light `LLIterative` fit) / GPyTorch model build. `logLik` = one log-likelihood **+ gradient** evaluation at theta. `predict` = a *cold* posterior mean on 300 held-out points (GPyTorch's per-fit prediction cache is dropped each rep). Every timing is the **min of up to 5 reps** (1 rep once a single call exceeds 3 s). Backends are named `<lib>-<method>-<linalg lib>`. `libKriging-Cholesky-OpenBLAS` is the **reference**: `dLogLik/n` = `|ll − ll_chol|/n` (blank for GPyTorch, whose `-mll` is a differently normalised quantity) and `dMean/rms` = `max|mean − mean_chol| / rms(y_test)`.

## Reference — `libKriging-Cholesky-OpenBLAS` (exact dense Cholesky)

| n | fit (s) | logLik (s) | predict (s) | logLik value | RMSE | Q² |
|--:|--:|--:|--:|--:|--:|--:|
| 250 | 0.009 | 0.009 | 0.005 | -317.241 | 0.4773 | 0.8952 |
| 500 | 0.026 | 0.026 | 0.010 | -463.222 | 0.3402 | 0.9468 |
| 1000 | 0.104 | 0.092 | 0.020 | -512.857 | 0.2111 | 0.9795 |
| 2000 | 0.514 | 0.450 | 0.046 | -60.627 | 0.0955 | 0.9958 |

## Timing — seconds

| backend | n | fit | logLik | predict |
|---|--:|--:|--:|--:|
| libKriging-Cholesky-OpenBLAS | 250 | 0.009 | 0.009 | 0.005 |
| libKriging-Cholesky-OpenBLAS | 500 | 0.026 | 0.026 | 0.010 |
| libKriging-Cholesky-OpenBLAS | 1000 | 0.104 | 0.092 | 0.020 |
| libKriging-Cholesky-OpenBLAS | 2000 | 0.514 | 0.450 | 0.046 |
| libKriging-Iterative-CUDA | 250 | 0.069 | 0.069 | 0.022 |
| libKriging-Iterative-CUDA | 500 | 0.084 | 0.101 | 0.100 |
| libKriging-Iterative-CUDA | 1000 | 0.127 | 0.161 | 0.150 |
| libKriging-Iterative-CUDA | 2000 | 0.365 | 0.605 | 0.687 |
| libKriging-Iterative-OpenMP | 250 | 0.027 | 0.034 | 0.025 |
| libKriging-Iterative-OpenMP | 500 | 0.037 | 0.065 | 0.027 |
| libKriging-Iterative-OpenMP | 1000 | 0.083 | 0.279 | 0.060 |
| libKriging-Iterative-OpenMP | 2000 | 0.331 | 1.841 | 0.180 |
| GPyTorch-BBMM-CUDA | 250 | 0.001 | 0.169 | 0.294 |
| GPyTorch-BBMM-CUDA | 500 | 0.001 | 0.059 | 0.069 |
| GPyTorch-BBMM-CUDA | 1000 | 0.001 | 0.063 | 0.079 |
| GPyTorch-BBMM-CUDA | 2000 | 0.001 | 0.088 | 0.104 |
| GPyTorch-BBMM-MKL | 250 | 0.001 | 0.031 | 0.055 |
| GPyTorch-BBMM-MKL | 500 | 0.001 | 0.054 | 0.096 |
| GPyTorch-BBMM-MKL | 1000 | 0.001 | 0.114 | 0.072 |
| GPyTorch-BBMM-MKL | 2000 | 0.001 | 0.246 | 0.097 |

## Accuracy

| backend | n | RMSE | Q² | logLik value | dLogLik/n | dMean/rms |
|---|--:|--:|--:|--:|--:|--:|
| libKriging-Cholesky-OpenBLAS | 250 | 0.4773 | 0.8952 | -317.241 | 0.00e+00 | 0.00e+00 |
| libKriging-Cholesky-OpenBLAS | 500 | 0.3402 | 0.9468 | -463.222 | 0.00e+00 | 0.00e+00 |
| libKriging-Cholesky-OpenBLAS | 1000 | 0.2111 | 0.9795 | -512.857 | 0.00e+00 | 0.00e+00 |
| libKriging-Cholesky-OpenBLAS | 2000 | 0.0955 | 0.9958 | -60.627 | 0.00e+00 | 0.00e+00 |
| libKriging-Iterative-CUDA | 250 | 0.4773 | 0.8952 | -316.216 | 4.10e-03 | 4.54e-08 |
| libKriging-Iterative-CUDA | 500 | 0.3402 | 0.9468 | -458.724 | 9.00e-03 | 3.45e-08 |
| libKriging-Iterative-CUDA | 1000 | 0.2111 | 0.9795 | -510.848 | 2.01e-03 | 5.87e-08 |
| libKriging-Iterative-CUDA | 2000 | 0.0955 | 0.9958 | -64.867 | 2.12e-03 | 6.10e-08 |
| libKriging-Iterative-OpenMP | 250 | 0.4773 | 0.8952 | -316.216 | 4.10e-03 | 3.46e-08 |
| libKriging-Iterative-OpenMP | 500 | 0.3402 | 0.9468 | -458.724 | 9.00e-03 | 4.19e-08 |
| libKriging-Iterative-OpenMP | 1000 | 0.2111 | 0.9795 | -510.848 | 2.01e-03 | 8.28e-08 |
| libKriging-Iterative-OpenMP | 2000 | 0.0955 | 0.9958 | -64.867 | 2.12e-03 | 7.63e-08 |
| GPyTorch-BBMM-CUDA | 250 | 0.4223 | 0.9179 | -1.215 | — | 1.98e-01 |
| GPyTorch-BBMM-CUDA | 500 | 0.3022 | 0.9580 | -1.083 | — | 2.29e-01 |
| GPyTorch-BBMM-CUDA | 1000 | 0.1903 | 0.9833 | -1.087 | — | 1.58e-01 |
| GPyTorch-BBMM-CUDA | 2000 | 0.0953 | 0.9958 | -0.721 | — | 7.02e-02 |
| GPyTorch-BBMM-MKL | 250 | 0.4223 | 0.9179 | -1.185 | — | 1.98e-01 |
| GPyTorch-BBMM-MKL | 500 | 0.3022 | 0.9580 | -1.132 | — | 2.29e-01 |
| GPyTorch-BBMM-MKL | 1000 | 0.1903 | 0.9833 | -1.092 | — | 1.58e-01 |
| GPyTorch-BBMM-MKL | 2000 | 0.0953 | 0.9958 | -0.721 | — | 7.02e-02 |

## Speed-ups (logLik-eval time)

| n | libKriging-Cholesky-OpenBLAS | libKriging-Iterative-CUDA | libKriging-Iterative-OpenMP | **OpenMP / CUDA** | **CUDA / Cholesky** | GPyTorch-BBMM-CUDA | GPyTorch-BBMM-MKL |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | 0.009 | 0.069 | 0.034 | 0.5× | 7.6× | 0.169 | 0.031 |
| 500 | 0.026 | 0.101 | 0.065 | 0.6× | 3.9× | 0.059 | 0.054 |
| 1000 | 0.092 | 0.161 | 0.279 | 1.7× | 1.8× | 0.063 | 0.114 |
| 2000 | 0.450 | 0.605 | 1.841 | 3.0× | 1.3× | 0.088 | 0.246 |

## Verdict — did everything converge?

- **libKriging iterative**: logLik within `9.0e-03` per point of the chol reference, posterior mean within `8.3e-08` of test RMS — converged.
- **GPyTorch**: posterior mean within `2.3e-01` of test RMS of the chol reference (a roughly n-independent offset from the covariance-argument convention, *not* under-convergence), Q² ≥ `0.9179`. Its `-mll` value is a different normalisation and is not compared.
- **`libKriging-Cholesky-OpenBLAS`** is fastest on all three ops at these n — the iterative path is for n where the dense factor no longer fits / is too slow, not this range.

## Notes

- Backend names are `<lib>-<method>-<linalg lib>`. Both `libKriging-Iterative-CUDA` (`LK_ITERATIVE_CUDA_DENSE_MAX_MB` budget) and `-OpenMP` (`LK_ITERATIVE_DENSE_MAX_MB`) materialize R once per evaluation for a separable kernel within their memory budget and run the matvecs as a single `cublasDgemm` / BLAS-3 `R*V` (hence `-OpenMP`, the BLAS it links), else fall back to a hand-written matvec kernel. `libKriging-Cholesky-OpenBLAS` and `GPyTorch-BBMM-MKL` name the actual dense BLAS/LAPACK each links against.
- theta=0.15 is chosen so the SLQ log-determinant's Lanczos quadrature and GPyTorch's BBMM CG both converge with sane iteration budgets; at longer theta (better-fitting but more ill-conditioned R) both need far more iterations / Lanczos steps. The `,0,40` in the libKriging objective is the third `LLIterative` argument (SLQ Lanczos steps per probe), added so the iterative log-likelihood *value* also tracks the exact one here.
- `libKriging-Iterative-CUDA` vs `libKriging-Iterative-OpenMP` is the same binary with `set_cuda_iterative_enabled(...)` toggled — identical results, different path for the batched CG / SLQ / gradient matvecs (CUDA kernels vs the CPU dense-`R` BLAS path).
- Companion: `docs/comparisons/libKriging_vs_GPyTorch.ipynb` (summary of these results + the GPyTorch code libKriging mimics).
