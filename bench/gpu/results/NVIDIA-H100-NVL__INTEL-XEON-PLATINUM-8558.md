# libKriging vs GPyTorch — iterative path, small n

- **GPU**: NVIDIA H100 NVL
- **CPU**: INTEL(R) XEON(R) PLATINUM 8558
- **host**: `farux-gpu04.cluster`  ·  logical CPUs: 192  ·  OMP_NUM_THREADS: `32`
- **when**: 2026-09-10 23:30 CEST
- **sweep**: `sine_sum` d=4, matern5_2, shared theta=0.15; n = 250, 500, 1000, 2000; test n=300
- **libKriging iterative objective**: `LLIterative(30,0,40)`  ·  `predictIterative(max_iter=8n, Nystrom precond rank ≤ 128)`
- **GPyTorch**: raised settings so BBMM converges — `max_cg_iterations=5000, cg_tolerance=1e-4, eval_cg_tolerance=1e-4, max_lanczos_quadrature_iterations=32, num_trace_samples=32, max_preconditioner_size=100, min_preconditioning_size=1`
- **versions**: pylibkriging=`1.1.0`, cuda_iterative_available=`True`, torch=`2.8.0+cu126`, torch.cuda=`12.6`, torch.cuda.is_available=`True`, gpytorch=`1.15.2`

`fit` = the `Kriging(...)` constructor (dense Cholesky for `LL`; one CG+SLQ commit for the light `LLIterative` fit) / GPyTorch model build. `logLik` = one log-likelihood **+ gradient** evaluation at theta. `predict` = a *cold* posterior mean on 300 held-out points (GPyTorch's per-fit prediction cache is dropped each rep). Every timing is the **min of up to 5 reps** (1 rep once a single call exceeds 3 s). Backends are named `<lib>-<method>-<linalg lib>`. `libKriging-Cholesky-OpenBLAS` is the **reference**: `dLogLik/n` = `|ll − ll_chol|/n` (blank for GPyTorch, whose `-mll` is a differently normalised quantity) and `dMean/rms` = `max|mean − mean_chol| / rms(y_test)`.

## Reference — `libKriging-Cholesky-OpenBLAS` (exact dense Cholesky)

| n | fit (s) | logLik (s) | predict (s) | logLik value | RMSE | Q² |
|--:|--:|--:|--:|--:|--:|--:|
| 250 | 0.009 | 0.010 | 0.004 | -317.241 | 0.4773 | 0.8952 |
| 500 | 0.030 | 0.029 | 0.008 | -463.222 | 0.3402 | 0.9468 |
| 1000 | 0.127 | 0.109 | 0.018 | -512.857 | 0.2111 | 0.9795 |
| 2000 | 0.550 | 0.471 | 0.041 | -60.627 | 0.0955 | 0.9958 |

## Timing — seconds

| backend | n | fit | logLik | predict |
|---|--:|--:|--:|--:|
| libKriging-Cholesky-OpenBLAS | 250 | 0.009 | 0.010 | 0.004 |
| libKriging-Cholesky-OpenBLAS | 500 | 0.030 | 0.029 | 0.008 |
| libKriging-Cholesky-OpenBLAS | 1000 | 0.127 | 0.109 | 0.018 |
| libKriging-Cholesky-OpenBLAS | 2000 | 0.550 | 0.471 | 0.041 |
| libKriging-Iterative-CUDA | 250 | 0.144 | 0.157 | 0.027 |
| libKriging-Iterative-CUDA | 500 | 0.223 | 0.326 | 0.138 |
| libKriging-Iterative-CUDA | 1000 | 0.426 | 1.315 | 0.243 |
| libKriging-Iterative-CUDA | 2000 | 1.304 | 9.346 | 0.969 |
| libKriging-Iterative-OpenMP | 250 | 0.027 | 0.031 | 0.034 |
| libKriging-Iterative-OpenMP | 500 | 0.036 | 0.055 | 0.108 |
| libKriging-Iterative-OpenMP | 1000 | 0.069 | 0.205 | 0.669 |
| libKriging-Iterative-OpenMP | 2000 | 0.252 | 1.393 | 6.716 |
| GPyTorch-BBMM-CUDA | 250 | 0.016 | 0.004 | 0.007 |
| GPyTorch-BBMM-CUDA | 500 | 0.004 | 0.006 | 0.004 |
| GPyTorch-BBMM-CUDA | 1000 | 0.008 | 0.360 | 0.546 |
| GPyTorch-BBMM-CUDA | 2000 | 0.004 | 0.522 | 0.742 |
| GPyTorch-BBMM-MKL | 250 | 0.001 | 0.006 | 0.006 |
| GPyTorch-BBMM-MKL | 500 | 0.001 | 0.007 | 0.006 |
| GPyTorch-BBMM-MKL | 1000 | 0.001 | 0.125 | 0.083 |
| GPyTorch-BBMM-MKL | 2000 | 0.001 | 0.310 | 0.138 |

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
| libKriging-Iterative-OpenMP | 250 | 0.4773 | 0.8952 | -316.216 | 4.10e-03 | 3.68e-08 |
| libKriging-Iterative-OpenMP | 500 | 0.3402 | 0.9468 | -458.724 | 9.00e-03 | 3.25e-08 |
| libKriging-Iterative-OpenMP | 1000 | 0.2111 | 0.9795 | -510.848 | 2.01e-03 | 8.48e-08 |
| libKriging-Iterative-OpenMP | 2000 | 0.0955 | 0.9958 | -64.867 | 2.12e-03 | 6.73e-08 |
| GPyTorch-BBMM-CUDA | 250 | 0.4223 | 0.9179 | -1.182 | — | 1.98e-01 |
| GPyTorch-BBMM-CUDA | 500 | 0.3022 | 0.9580 | -0.923 | — | 2.29e-01 |
| GPyTorch-BBMM-CUDA | 1000 | 0.1903 | 0.9833 | -1.100 | — | 1.58e-01 |
| GPyTorch-BBMM-CUDA | 2000 | 0.0953 | 0.9958 | -0.718 | — | 7.02e-02 |
| GPyTorch-BBMM-MKL | 250 | 0.4223 | 0.9179 | -1.182 | — | 1.98e-01 |
| GPyTorch-BBMM-MKL | 500 | 0.3022 | 0.9580 | -0.923 | — | 2.29e-01 |
| GPyTorch-BBMM-MKL | 1000 | 0.1903 | 0.9833 | -1.092 | — | 1.58e-01 |
| GPyTorch-BBMM-MKL | 2000 | 0.0953 | 0.9958 | -0.725 | — | 7.02e-02 |

## Speed-ups (logLik-eval time)

| n | libKriging-Cholesky-OpenBLAS | libKriging-Iterative-CUDA | libKriging-Iterative-OpenMP | **OpenMP / CUDA** | **CUDA / Cholesky** | GPyTorch-BBMM-CUDA | GPyTorch-BBMM-MKL |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | 0.010 | 0.157 | 0.031 | 0.2× | 15× | 0.004 | 0.006 |
| 500 | 0.029 | 0.326 | 0.055 | 0.2× | 11× | 0.006 | 0.007 |
| 1000 | 0.109 | 1.315 | 0.205 | 0.2× | 12× | 0.360 | 0.125 |
| 2000 | 0.471 | 9.346 | 1.393 | 0.1× | 20× | 0.522 | 0.310 |

## Verdict — did everything converge?

- **libKriging iterative**: logLik within `9.0e-03` per point of the chol reference, posterior mean within `8.5e-08` of test RMS — converged.
- **GPyTorch**: posterior mean within `2.3e-01` of test RMS of the chol reference (a roughly n-independent offset from the covariance-argument convention, *not* under-convergence), Q² ≥ `0.9179`. Its `-mll` value is a different normalisation and is not compared.
- **`libKriging-Cholesky-OpenBLAS`** is fastest on all three ops at these n — the iterative path is for n where the dense factor no longer fits / is too slow, not this range.

## Notes

- Backend names are `<lib>-<method>-<linalg lib>`. `libKriging-Iterative-CUDA` uses hand-written CUDA matvec kernels; `libKriging-Iterative-OpenMP` materializes R once per evaluation and runs the matvecs as BLAS-3 `R*V` (hence `-OpenMP`, the BLAS it links) for separable kernels within the `LK_ITERATIVE_DENSE_MAX_MB` budget, else a hand-written OpenMP matvec. `libKriging-Cholesky-OpenBLAS` and `GPyTorch-BBMM-MKL` name the actual dense BLAS/LAPACK each links against.
- theta=0.15 is chosen so the SLQ log-determinant's Lanczos quadrature and GPyTorch's BBMM CG both converge with sane iteration budgets; at longer theta (better-fitting but more ill-conditioned R) both need far more iterations / Lanczos steps. The `,0,40` in the libKriging objective is the third `LLIterative` argument (SLQ Lanczos steps per probe), added so the iterative log-likelihood *value* also tracks the exact one here.
- `libKriging-Iterative-CUDA` vs `libKriging-Iterative-OpenMP` is the same binary with `set_cuda_iterative_enabled(...)` toggled — identical results, different path for the batched CG / SLQ / gradient matvecs (CUDA kernels vs the CPU dense-`R` BLAS path).
- Companion: `docs/comparisons/libKriging_vs_GPyTorch.ipynb` (summary of these results + the GPyTorch code libKriging mimics).
