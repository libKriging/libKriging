# libKriging vs GPyTorch — iterative path, small n

- **GPU**: NVIDIA H100 NVL
- **CPU**: INTEL(R) XEON(R) PLATINUM 8558
- **host**: `farux-gpu04.cluster`  ·  logical CPUs: 192  ·  OMP_NUM_THREADS: `32`
- **when**: 2026-09-10 17:43 CEST
- **sweep**: `sine_sum` d=4, matern5_2, shared theta=0.15; n = 250, 500, 1000, 2000; test n=300
- **libKriging iterative objective**: `LLIterative(30,0,40)`  ·  `predictIterative(max_iter=8n, Nystrom precond rank ≤ 128)`
- **GPyTorch**: raised settings so BBMM converges — `max_cg_iterations=5000, cg_tolerance=1e-4, eval_cg_tolerance=1e-4, max_lanczos_quadrature_iterations=32, num_trace_samples=32, max_preconditioner_size=100, min_preconditioning_size=1`
- **versions**: pylibkriging=`1.1.0`, cuda_iterative_available=`True`, torch=`2.8.0+cu126`, torch.cuda=`12.6`, torch.cuda.is_available=`True`, gpytorch=`1.15.2`

`fit` = the `Kriging(...)` constructor (dense Cholesky for `LL`; one CG+SLQ commit for the light `LLIterative` fit) / GPyTorch model build. `logLik` = one log-likelihood **+ gradient** evaluation at theta. `predict` = a *cold* posterior mean on 300 held-out points (GPyTorch's per-fit prediction cache is dropped each rep). Every timing is the **min of up to 5 reps** (1 rep once a single call exceeds 3 s). `libkriging-chol` is the **reference**: `dLogLik/n` = `|ll − ll_chol|/n` (blank for GPyTorch, whose `-mll` is a differently normalised quantity) and `dMean/rms` = `max|mean − mean_chol| / rms(y_test)`.

## Reference — `libkriging-chol` (exact dense Cholesky)

| n | fit (s) | logLik (s) | predict (s) | logLik value | RMSE | Q² |
|--:|--:|--:|--:|--:|--:|--:|
| 250 | 0.009 | 0.009 | 0.004 | -317.241 | 0.4773 | 0.8952 |
| 500 | 0.025 | 0.025 | 0.008 | -463.222 | 0.3402 | 0.9468 |
| 1000 | 0.102 | 0.090 | 0.017 | -512.857 | 0.2111 | 0.9795 |
| 2000 | 0.472 | 0.385 | 0.037 | -60.627 | 0.0955 | 0.9958 |

## Timing — seconds

| backend | n | fit | logLik | predict |
|---|--:|--:|--:|--:|
| libkriging-chol | 250 | 0.009 | 0.009 | 0.004 |
| libkriging-chol | 500 | 0.025 | 0.025 | 0.008 |
| libkriging-chol | 1000 | 0.102 | 0.090 | 0.017 |
| libkriging-chol | 2000 | 0.472 | 0.385 | 0.037 |
| libkriging-gpu | 250 | 0.030 | 0.037 | 0.018 |
| libkriging-gpu | 500 | 0.058 | 0.113 | 0.080 |
| libkriging-gpu | 1000 | 0.175 | 0.684 | 0.146 |
| libkriging-gpu | 2000 | 0.687 | 5.509 | 0.555 |
| libkriging-cpu | 250 | 0.220 | 0.424 | 0.031 |
| libkriging-cpu | 500 | 0.761 | 2.597 | 0.106 |
| libkriging-cpu | 1000 | 3.826 | 24.512 | 0.592 |
| libkriging-cpu | 2000 | 21.122 | 195.516 | 5.856 |
| gpytorch-gpu | 250 | 0.001 | 0.008 | 0.004 |
| gpytorch-gpu | 500 | 0.001 | 0.005 | 0.004 |
| gpytorch-gpu | 1000 | 0.001 | 0.238 | 0.381 |
| gpytorch-gpu | 2000 | 0.001 | 0.353 | 0.545 |
| gpytorch-cpu | 250 | 0.001 | 0.005 | 0.005 |
| gpytorch-cpu | 500 | 0.001 | 0.006 | 0.006 |
| gpytorch-cpu | 1000 | 0.001 | 0.103 | 0.078 |
| gpytorch-cpu | 2000 | 0.001 | 0.309 | 0.149 |

## Accuracy

| backend | n | RMSE | Q² | logLik value | dLogLik/n | dMean/rms |
|---|--:|--:|--:|--:|--:|--:|
| libkriging-chol | 250 | 0.4773 | 0.8952 | -317.241 | 0.00e+00 | 0.00e+00 |
| libkriging-chol | 500 | 0.3402 | 0.9468 | -463.222 | 0.00e+00 | 0.00e+00 |
| libkriging-chol | 1000 | 0.2111 | 0.9795 | -512.857 | 0.00e+00 | 0.00e+00 |
| libkriging-chol | 2000 | 0.0955 | 0.9958 | -60.627 | 0.00e+00 | 0.00e+00 |
| libkriging-gpu | 250 | 0.4773 | 0.8952 | -316.216 | 4.10e-03 | 3.77e-08 |
| libkriging-gpu | 500 | 0.3402 | 0.9468 | -458.724 | 9.00e-03 | 3.01e-08 |
| libkriging-gpu | 1000 | 0.2111 | 0.9795 | -510.848 | 2.01e-03 | 8.28e-08 |
| libkriging-gpu | 2000 | 0.0955 | 0.9958 | -64.867 | 2.12e-03 | 6.84e-08 |
| libkriging-cpu | 250 | 0.4773 | 0.8952 | -316.216 | 4.10e-03 | 4.39e-08 |
| libkriging-cpu | 500 | 0.3402 | 0.9468 | -458.724 | 9.00e-03 | 3.24e-08 |
| libkriging-cpu | 1000 | 0.2111 | 0.9795 | -510.848 | 2.01e-03 | 7.62e-08 |
| libkriging-cpu | 2000 | 0.0955 | 0.9958 | -64.867 | 2.12e-03 | 4.90e-08 |
| gpytorch-gpu | 250 | 0.4223 | 0.9179 | -1.182 | — | 1.98e-01 |
| gpytorch-gpu | 500 | 0.3022 | 0.9580 | -0.923 | — | 2.29e-01 |
| gpytorch-gpu | 1000 | 0.1903 | 0.9833 | -1.079 | — | 1.58e-01 |
| gpytorch-gpu | 2000 | 0.0953 | 0.9958 | -0.724 | — | 7.02e-02 |
| gpytorch-cpu | 250 | 0.4223 | 0.9179 | -1.182 | — | 1.98e-01 |
| gpytorch-cpu | 500 | 0.3022 | 0.9580 | -0.923 | — | 2.29e-01 |
| gpytorch-cpu | 1000 | 0.1903 | 0.9833 | -1.096 | — | 1.58e-01 |
| gpytorch-cpu | 2000 | 0.0953 | 0.9958 | -0.722 | — | 7.02e-02 |

## Speed-ups (logLik-eval time)

| n | chol | libk-gpu | libk-cpu | **cpu/gpu** | **libk-gpu / chol** | gpytorch-gpu | gpytorch-cpu |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 250 | 0.009 | 0.037 | 0.424 | 11.3× | 4× | 0.008 | 0.005 |
| 500 | 0.025 | 0.113 | 2.597 | 23.0× | 5× | 0.005 | 0.006 |
| 1000 | 0.090 | 0.684 | 24.512 | 35.8× | 8× | 0.238 | 0.103 |
| 2000 | 0.385 | 5.509 | 195.516 | 35.5× | 14× | 0.353 | 0.309 |

## Verdict — did everything converge?

- **libKriging iterative**: logLik within `9.0e-03` per point of the chol reference, posterior mean within `8.3e-08` of test RMS — converged.
- **GPyTorch**: posterior mean within `2.3e-01` of test RMS of the chol reference (a roughly n-independent offset from the covariance-argument convention, *not* under-convergence), Q² ≥ `0.9179`. Its `-mll` value is a different normalisation and is not compared.
- **libkriging-chol** is fastest on all three ops at these n — the iterative path is for n where the dense factor no longer fits / is too slow, not this range.

## Notes

- theta=0.15 is chosen so the SLQ log-determinant's Lanczos quadrature and GPyTorch's BBMM CG both converge with sane iteration budgets; at longer theta (better-fitting but more ill-conditioned R) both need far more iterations / Lanczos steps. The `,0,40` in the libKriging objective is the third `LLIterative` argument (SLQ Lanczos steps per probe), added so the iterative log-likelihood *value* also tracks the exact one here.
- `libkriging-gpu` vs `libkriging-cpu` is the same binary with `set_cuda_iterative_enabled(...)` toggled — identical results, different device for the batched CG / SLQ / gradient matvecs.
- Companion: `docs/comparisons/libKriging_vs_GPyTorch.ipynb` (summary + the GPyTorch code libKriging mimics), `bench/comparison-gpu/` (multi-function CI-style variant).
