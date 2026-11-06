## Fit protocol summary

10320 fits, 3120 data sets, 4 variants; miss: objective more than 10 (log-density units) below the best variant of the data set.

### By kernel, noise, objective, n

| kernel | noise | objective | n | variant | sets | miss | miss % [95 % CI] | RMSE | RMSE / best, median / p90 | wall s | CPU s | wall total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gauss | none | LL | 100 | BFGS:default | 120 | 8 | 7 [3, 13] | 0.638 | 1 / 1.15 | 0.0381 | 0.142 | 4.94 |
| gauss | none | LL | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 120 | 19 | 16 [10, 23] | 0.645 | 1 / 1.37 | 0.0189 | 0.0743 | 2.37 |
| gauss | none | LL | 100 | BFGS4:default | 120 | 3 | 2 [1, 7] | 0.624 | 1 / 1.01 | 0.0358 | 0.141 | 4.62 |
| gauss | none | LL | 100 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.617 | 1 / 1 | 0.0841 | 0.327 | 10.3 |
| gauss | none | LL | 200 | BFGS:default | 120 | 26 | 22 [15, 30] | 0.438 | 1 / 1.42 | 0.501 | 1.96 | 80.5 |
| gauss | none | LL | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 120 | 43 | 36 [28, 45] | 0.461 | 1 / 3.69 | 0.243 | 0.963 | 37.5 |
| gauss | none | LL | 200 | BFGS4:default | 120 | 5 | 4 [2, 9] | 0.411 | 1 / 1.01 | 0.17 | 0.587 | 21.1 |
| gauss | none | LL | 200 | BFGS10:default | 120 | 2 | 2 [0, 6] | 0.411 | 1 / 1 | 0.324 | 1.11 | 38.2 |
| gauss | none | LL | 500 | BFGS:default | 120 | 25 | 21 [15, 29] | 0.124 | 1 / 2.2 | 0.992 | 3.93 | 130 |
| gauss | none | LL | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 120 | 40 | 33 [26, 42] | 0.174 | 1 / 3.91 | 0.532 | 2.12 | 76.8 |
| gauss | none | LL | 500 | BFGS4:default | 120 | 6 | 5 [2, 10] | 0.0985 | 1 / 1 | 1.04 | 3.11 | 152 |
| gauss | none | LL | 500 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.0918 | 1 / 1 | 2.53 | 8.24 | 290 |
| gauss | none | LL | 1000 | BFGS:default | 120 | 23 | 19 [13, 27] | 0.0258 | 1 / 1.85 | 3.19 | 12.7 | 558 |
| gauss | none | LL | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 120 | 52 | 43 [35, 52] | 0.0734 | 1 / 8.29 | 2.27 | 8.79 | 334 |
| gauss | none | LL | 1000 | BFGS4:default | 120 | 2 | 2 [0, 6] | 0.0258 | 1 / 1 | 6.28 | 18.5 | 951 |
| gauss | none | LL | 1000 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.0254 | 1 / 1 | 14 | 47.6 | 1.92e+03 |
| gauss | none | LMP | 100 | BFGS:default | 120 | 9 | 8 [4, 14] | 0.644 | 1 / 1.13 | 0.0406 | 0.16 | 5.52 |
| gauss | none | LMP | 100 | BFGS4:default | 120 | 0 | 0 [0, 3] | 0.631 | 1 / 1 | 0.0949 | 0.376 | 12.3 |
| gauss | none | LMP | 100 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.63 | 1 / 1 | 0.186 | 0.652 | 24.4 |
| gauss | none | LMP | 200 | BFGS:default | 120 | 35 | 29 [22, 38] | 0.447 | 1 / 2.24 | 0.36 | 1.42 | 45.8 |
| gauss | none | LMP | 200 | BFGS4:default | 120 | 3 | 2 [1, 7] | 0.41 | 1 / 1 | 0.388 | 1.2 | 51.3 |
| gauss | none | LMP | 200 | BFGS10:default | 120 | 1 | 1 [0, 5] | 0.41 | 1 / 1 | 0.88 | 2.99 | 113 |
| gauss | none | LOO | 100 | BFGS:default | 120 | 59 | 49 [40, 58] | 0.716 | 1.02 / 2.69 | 0.0325 | 0.126 | 4.41 |
| gauss | none | LOO | 100 | BFGS4:default | 120 | 12 | 10 [6, 17] | 0.622 | 1 / 1.09 | 0.057 | 0.225 | 8.17 |
| gauss | none | LOO | 100 | BFGS10:default | 120 | 1 | 1 [0, 5] | 0.629 | 1 / 1 | 0.133 | 0.51 | 20.5 |
| gauss | none | LOO | 200 | BFGS:default | 120 | 59 | 49 [40, 58] | 0.482 | 1.04 / 4.13 | 0.351 | 1.38 | 47.8 |
| gauss | none | LOO | 200 | BFGS4:default | 120 | 25 | 21 [15, 29] | 0.41 | 1 / 1.14 | 0.239 | 0.757 | 32.1 |
| gauss | none | LOO | 200 | BFGS10:default | 120 | 1 | 1 [0, 5] | 0.41 | 1 / 1 | 0.533 | 1.78 | 72.8 |
| gauss | nugget | LL | 100 | BFGS:default | 120 | 5 | 4 [2, 9] | 0.625 | 1 / 1.11 | 0.0481 | 0.188 | 6.45 |
| gauss | nugget | LL | 100 | BFGS4:default | 120 | 0 | 0 [0, 3] | 0.626 | 1 / 1 | 0.0672 | 0.263 | 8.56 |
| gauss | nugget | LL | 100 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.622 | 1 / 1 | 0.107 | 0.416 | 13.6 |
| gauss | nugget | LL | 200 | BFGS:default | 120 | 16 | 13 [8, 21] | 0.429 | 1 / 1.21 | 0.768 | 3.01 | 106 |
| gauss | nugget | LL | 200 | BFGS4:default | 120 | 7 | 6 [3, 12] | 0.425 | 1 / 1.04 | 0.543 | 2.11 | 78.7 |
| gauss | nugget | LL | 200 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.41 | 1 / 1 | 0.729 | 2.72 | 104 |
| gauss | nugget | LL | 500 | BFGS:default | 120 | 11 | 9 [5, 16] | 0.147 | 1 / 1.01 | 1.43 | 5.69 | 199 |
| gauss | nugget | LL | 500 | BFGS4:default | 120 | 3 | 2 [1, 7] | 0.127 | 1 / 1 | 1.97 | 6.52 | 253 |
| gauss | nugget | LL | 500 | BFGS10:default | 120 | 3 | 2 [1, 7] | 0.127 | 1 / 1 | 2.91 | 10 | 381 |
| gauss | nugget | LMP | 100 | BFGS:default | 120 | 9 | 8 [4, 14] | 0.676 | 1 / 1.17 | 0.0698 | 0.273 | 8.73 |
| gauss | nugget | LMP | 100 | BFGS4:default | 120 | 1 | 1 [0, 5] | 0.676 | 1 / 1 | 0.118 | 0.467 | 15.1 |
| gauss | nugget | LMP | 100 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.649 | 1 / 1 | 0.199 | 0.709 | 24.7 |
| gauss | nugget | LMP | 200 | BFGS:default | 120 | 22 | 18 [12, 26] | 0.435 | 1 / 1.21 | 0.824 | 3.24 | 109 |
| gauss | nugget | LMP | 200 | BFGS4:default | 120 | 4 | 3 [1, 8] | 0.426 | 1 / 1.02 | 1.02 | 3.31 | 132 |
| gauss | nugget | LMP | 200 | BFGS10:default | 120 | 3 | 2 [1, 7] | 0.43 | 1 / 1 | 1.54 | 5.51 | 200 |
| matern5_2 | none | LL | 100 | BFGS:default | 120 | 16 | 13 [8, 21] | 0.576 | 1 / 2.12 | 0.0638 | 0.25 | 9.01 |
| matern5_2 | none | LL | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 120 | 26 | 22 [15, 30] | 0.587 | 1 / 2.42 | 0.0337 | 0.133 | 4.11 |
| matern5_2 | none | LL | 100 | BFGS4:default | 120 | 8 | 7 [3, 13] | 0.436 | 1 / 1.01 | 0.0519 | 0.2 | 6.39 |
| matern5_2 | none | LL | 100 | BFGS10:default | 120 | 3 | 2 [1, 7] | 0.265 | 1 / 1 | 0.125 | 0.482 | 15 |
| matern5_2 | none | LL | 200 | BFGS:default | 120 | 16 | 13 [8, 21] | 0.0693 | 1 / 2.54 | 0.633 | 2.49 | 88.1 |
| matern5_2 | none | LL | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 120 | 27 | 22 [16, 31] | 0.0693 | 1 / 3.37 | 0.223 | 0.879 | 31.2 |
| matern5_2 | none | LL | 200 | BFGS4:default | 120 | 5 | 4 [2, 9] | 0.0376 | 1 / 1 | 0.226 | 0.683 | 27.5 |
| matern5_2 | none | LL | 200 | BFGS10:default | 120 | 1 | 1 [0, 5] | 0.0185 | 1 / 1 | 0.447 | 1.53 | 53.3 |
| matern5_2 | none | LL | 500 | BFGS:default | 120 | 9 | 8 [4, 14] | 0.00491 | 1 / 1 | 0.922 | 3.67 | 132 |
| matern5_2 | none | LL | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 120 | 16 | 13 [8, 21] | 0.00491 | 1 / 1.14 | 0.378 | 1.5 | 61.5 |
| matern5_2 | none | LL | 500 | BFGS4:default | 120 | 0 | 0 [0, 3] | 0.00491 | 1 / 1 | 1.14 | 2.91 | 139 |
| matern5_2 | none | LL | 500 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.00491 | 1 / 1 | 2.54 | 8.77 | 304 |
| matern5_2 | none | LL | 1000 | BFGS:default | 120 | 10 | 8 [5, 15] | 0.00111 | 1 / 1 | 4.04 | 16 | 881 |
| matern5_2 | none | LL | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 120 | 21 | 18 [12, 25] | 0.00184 | 1 / 1.86 | 1.76 | 7.01 | 254 |
| matern5_2 | none | LL | 1000 | BFGS4:default | 120 | 5 | 4 [2, 9] | 0.00149 | 1 / 1 | 6.15 | 16 | 899 |
| matern5_2 | none | LL | 1000 | BFGS10:default | 120 | 1 | 1 [0, 5] | 0.0011 | 1 / 1 | 14.5 | 47.3 | 2.37e+03 |
| matern5_2 | none | LMP | 100 | BFGS:default | 120 | 21 | 18 [12, 25] | 0.474 | 1 / 2.04 | 0.047 | 0.186 | 6.16 |
| matern5_2 | none | LMP | 100 | BFGS4:default | 120 | 9 | 8 [4, 14] | 0.356 | 1 / 1 | 0.0718 | 0.284 | 10 |
| matern5_2 | none | LMP | 100 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.149 | 1 / 1 | 0.181 | 0.645 | 23.5 |
| matern5_2 | none | LMP | 200 | BFGS:default | 120 | 17 | 14 [9, 22] | 0.0372 | 1 / 1.42 | 0.283 | 1.1 | 41.7 |
| matern5_2 | none | LMP | 200 | BFGS4:default | 120 | 1 | 1 [0, 5] | 0.0178 | 1 / 1 | 0.392 | 1.17 | 52.2 |
| matern5_2 | none | LMP | 200 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.0178 | 1 / 1 | 0.846 | 2.88 | 111 |
| matern5_2 | none | LOO | 100 | BFGS:default | 120 | 34 | 28 [21, 37] | 0.545 | 1 / 4.68 | 0.0551 | 0.214 | 10.8 |
| matern5_2 | none | LOO | 100 | BFGS4:default | 120 | 12 | 10 [6, 17] | 0.153 | 1 / 1.26 | 0.0764 | 0.297 | 15.1 |
| matern5_2 | none | LOO | 100 | BFGS10:default | 120 | 3 | 2 [1, 7] | 0.0869 | 1 / 1 | 0.234 | 0.785 | 37.6 |
| matern5_2 | none | LOO | 200 | BFGS:default | 120 | 21 | 18 [12, 25] | 0.0124 | 1 / 5.2 | 0.323 | 1.28 | 83 |
| matern5_2 | none | LOO | 200 | BFGS4:default | 120 | 5 | 4 [2, 9] | 0.0124 | 1 / 1 | 0.352 | 1.07 | 79.3 |
| matern5_2 | none | LOO | 200 | BFGS10:default | 120 | 3 | 2 [1, 7] | 0.0124 | 1 / 1 | 0.798 | 2.55 | 138 |
| matern5_2 | nugget | LL | 100 | BFGS:default | 120 | 16 | 13 [8, 21] | 0.587 | 1 / 2.06 | 0.094 | 0.371 | 12 |
| matern5_2 | nugget | LL | 100 | BFGS4:default | 120 | 8 | 7 [3, 13] | 0.536 | 1 / 1.16 | 0.119 | 0.469 | 15.1 |
| matern5_2 | nugget | LL | 100 | BFGS10:default | 120 | 1 | 1 [0, 5] | 0.413 | 1 / 1 | 0.177 | 0.687 | 22.3 |
| matern5_2 | nugget | LL | 200 | BFGS:default | 120 | 31 | 26 [19, 34] | 0.184 | 1 / 3.23 | 0.83 | 3.27 | 124 |
| matern5_2 | nugget | LL | 200 | BFGS4:default | 120 | 13 | 11 [6, 18] | 0.117 | 1 / 1.11 | 0.724 | 2.68 | 108 |
| matern5_2 | nugget | LL | 200 | BFGS10:default | 120 | 6 | 5 [2, 10] | 0.103 | 1 / 1 | 1.03 | 3.74 | 144 |
| matern5_2 | nugget | LL | 500 | BFGS:default | 120 | 47 | 39 [31, 48] | 0.0387 | 1 / 1.33 | 1.54 | 6.1 | 230 |
| matern5_2 | nugget | LL | 500 | BFGS4:default | 120 | 7 | 6 [3, 12] | 0.0381 | 1 / 1.01 | 2.03 | 7.15 | 304 |
| matern5_2 | nugget | LL | 500 | BFGS10:default | 120 | 2 | 2 [0, 6] | 0.0381 | 1 / 1 | 4.09 | 14.3 | 530 |
| matern5_2 | nugget | LMP | 100 | BFGS:default | 120 | 19 | 16 [10, 23] | 0.598 | 1 / 1.38 | 0.116 | 0.458 | 14.5 |
| matern5_2 | nugget | LMP | 100 | BFGS4:default | 120 | 6 | 5 [2, 10] | 0.579 | 1 / 1.08 | 0.164 | 0.642 | 20.3 |
| matern5_2 | nugget | LMP | 100 | BFGS10:default | 120 | 1 | 1 [0, 5] | 0.544 | 1 / 1 | 0.277 | 0.964 | 33.8 |
| matern5_2 | nugget | LMP | 200 | BFGS:default | 120 | 35 | 29 [22, 38] | 0.177 | 1 / 3.86 | 0.883 | 3.44 | 127 |
| matern5_2 | nugget | LMP | 200 | BFGS4:default | 120 | 17 | 14 [9, 22] | 0.12 | 1 / 1.18 | 1.07 | 3.55 | 154 |
| matern5_2 | nugget | LMP | 200 | BFGS10:default | 120 | 2 | 2 [0, 6] | 0.112 | 1 / 1 | 1.75 | 6.31 | 229 |

### By noise, objective, function (kernels and n pooled)

| noise | objective | function | variant | sets | miss | miss % [95 % CI] | RMSE | RMSE / best, median / p90 | wall s | CPU s | wall total s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | LL | branin4 | BFGS:default | 240 | 3 | 1 [0, 4] | 0.00015 | 1 / 1 | 0.651 | 2.57 | 385 |
| none | LL | branin4 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 14 | 6 [4, 10] | 0.00017 | 1 / 1 | 0.211 | 0.823 | 104 |
| none | LL | branin4 | BFGS4:default | 240 | 1 | 0 [0, 2] | 0.00015 | 1 / 1 | 0.427 | 1.1 | 358 |
| none | LL | branin4 | BFGS10:default | 240 | 1 | 0 [0, 2] | 0.00015 | 1 / 1 | 0.717 | 2.29 | 862 |
| none | LL | hartmann6 | BFGS:default | 240 | 19 | 8 [5, 12] | 0.389 | 1 / 1.02 | 0.698 | 2.77 | 357 |
| none | LL | hartmann6 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 53 | 22 [17, 28] | 0.443 | 1 / 1.73 | 0.487 | 1.91 | 308 |
| none | LL | hartmann6 | BFGS4:default | 240 | 1 | 0 [0, 2] | 0.377 | 1 / 1 | 0.501 | 1.62 | 691 |
| none | LL | hartmann6 | BFGS10:default | 240 | 0 | 0 [0, 2] | 0.377 | 1 / 1 | 1.11 | 3.65 | 1.45e+03 |
| none | LL | sin4 | BFGS:default | 240 | 28 | 12 [8, 16] | 0.0227 | 1 / 1.16 | 0.782 | 3.09 | 615 |
| none | LL | sin4 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 73 | 30 [25, 37] | 0.0468 | 1 / 5.7 | 0.294 | 1.16 | 157 |
| none | LL | sin4 | BFGS4:default | 240 | 14 | 6 [4, 10] | 0.0191 | 1 / 1 | 0.459 | 1.42 | 594 |
| none | LL | sin4 | BFGS10:default | 240 | 2 | 1 [0, 3] | 0.0189 | 1 / 1 | 1.23 | 3.79 | 1.46e+03 |
| none | LL | sinhalf8 | BFGS:default | 240 | 83 | 35 [29, 41] | 0.318 | 1 / 4.14 | 1.06 | 4.16 | 526 |
| none | LL | sinhalf8 | BFGS:LK_LONG_RANGE_RESTART=0 | 240 | 104 | 43 [37, 50] | 0.535 | 1.02 / 5.61 | 0.45 | 1.79 | 232 |
| none | LL | sinhalf8 | BFGS4:default | 240 | 18 | 8 [5, 12] | 0.142 | 1 / 1.12 | 0.525 | 1.61 | 558 |
| none | LL | sinhalf8 | BFGS10:default | 240 | 4 | 2 [1, 4] | 0.125 | 1 / 1 | 1.44 | 4.77 | 1.23e+03 |
| none | LMP | branin4 | BFGS:default | 120 | 7 | 6 [3, 12] | 0.00281 | 1 / 1 | 0.106 | 0.411 | 18.6 |
| none | LMP | branin4 | BFGS4:default | 120 | 0 | 0 [0, 3] | 0.00183 | 1 / 1 | 0.159 | 0.554 | 23 |
| none | LMP | branin4 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.00183 | 1 / 1 | 0.287 | 0.949 | 39.2 |
| none | LMP | hartmann6 | BFGS:default | 120 | 2 | 2 [0, 6] | 0.557 | 1 / 1 | 0.133 | 0.527 | 19.7 |
| none | LMP | hartmann6 | BFGS4:default | 120 | 0 | 0 [0, 3] | 0.556 | 1 / 1 | 0.198 | 0.71 | 26.8 |
| none | LMP | hartmann6 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.556 | 1 / 1 | 0.446 | 1.63 | 67.5 |
| none | LMP | sin4 | BFGS:default | 120 | 26 | 22 [15, 30] | 0.208 | 1 / 1.74 | 0.121 | 0.477 | 25.3 |
| none | LMP | sin4 | BFGS4:default | 120 | 6 | 5 [2, 10] | 0.166 | 1 / 1 | 0.179 | 0.591 | 29 |
| none | LMP | sin4 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.098 | 1 / 1 | 0.439 | 1.43 | 58.4 |
| none | LMP | sinhalf8 | BFGS:default | 120 | 47 | 39 [31, 48] | 0.712 | 1.04 / 2.71 | 0.209 | 0.833 | 35.7 |
| none | LMP | sinhalf8 | BFGS4:default | 120 | 7 | 6 [3, 12] | 0.446 | 1 / 1.04 | 0.339 | 1.11 | 47.1 |
| none | LMP | sinhalf8 | BFGS10:default | 120 | 1 | 1 [0, 5] | 0.399 | 1 / 1 | 0.684 | 2.34 | 107 |
| none | LOO | branin4 | BFGS:default | 120 | 38 | 32 [24, 40] | 0.00239 | 1 / 7.4 | 0.239 | 0.939 | 61.4 |
| none | LOO | branin4 | BFGS4:default | 120 | 15 | 12 [8, 20] | 0.00183 | 1 / 1.01 | 0.205 | 0.657 | 51.2 |
| none | LOO | branin4 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.00184 | 1 / 1 | 0.508 | 1.36 | 96 |
| none | LOO | hartmann6 | BFGS:default | 120 | 11 | 9 [5, 16] | 0.562 | 1 / 1.11 | 0.128 | 0.498 | 19.7 |
| none | LOO | hartmann6 | BFGS4:default | 120 | 3 | 2 [1, 7] | 0.546 | 1 / 1 | 0.138 | 0.526 | 19 |
| none | LOO | hartmann6 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.546 | 1 / 1 | 0.367 | 1.14 | 45.8 |
| none | LOO | sin4 | BFGS:default | 120 | 35 | 29 [22, 38] | 0.167 | 1 / 2.85 | 0.149 | 0.59 | 37.6 |
| none | LOO | sin4 | BFGS4:default | 120 | 8 | 7 [3, 13] | 0.131 | 1 / 1.02 | 0.146 | 0.549 | 33.5 |
| none | LOO | sin4 | BFGS10:default | 120 | 1 | 1 [0, 5] | 0.0845 | 1 / 1 | 0.424 | 1.22 | 58.2 |
| none | LOO | sinhalf8 | BFGS:default | 120 | 89 | 74 [66, 81] | 0.841 | 1.31 / 6.85 | 0.162 | 0.623 | 27.3 |
| none | LOO | sinhalf8 | BFGS4:default | 120 | 28 | 23 [17, 32] | 0.662 | 1 / 1.62 | 0.207 | 0.657 | 30.9 |
| none | LOO | sinhalf8 | BFGS10:default | 120 | 7 | 6 [3, 12] | 0.456 | 1 / 1 | 0.529 | 1.62 | 69.4 |
| nugget | LL | branin4 | BFGS:default | 180 | 28 | 16 [11, 22] | 0.0302 | 1 / 1.28 | 0.773 | 3.05 | 164 |
| nugget | LL | branin4 | BFGS4:default | 180 | 0 | 0 [0, 2] | 0.0302 | 1 / 1 | 0.628 | 2.48 | 170 |
| nugget | LL | branin4 | BFGS10:default | 180 | 0 | 0 [0, 2] | 0.0302 | 1 / 1 | 0.722 | 2.68 | 236 |
| nugget | LL | hartmann6 | BFGS:default | 180 | 0 | 0 [0, 2] | 0.469 | 1 / 1 | 0.625 | 2.46 | 132 |
| nugget | LL | hartmann6 | BFGS4:default | 180 | 0 | 0 [0, 2] | 0.469 | 1 / 1 | 0.559 | 2.13 | 164 |
| nugget | LL | hartmann6 | BFGS10:default | 180 | 0 | 0 [0, 2] | 0.469 | 1 / 1 | 0.942 | 3.37 | 299 |
| nugget | LL | sin4 | BFGS:default | 180 | 44 | 24 [19, 31] | 0.193 | 1 / 2.28 | 0.731 | 2.88 | 157 |
| nugget | LL | sin4 | BFGS4:default | 180 | 19 | 11 [7, 16] | 0.153 | 1 / 1.1 | 0.504 | 1.86 | 180 |
| nugget | LL | sin4 | BFGS10:default | 180 | 7 | 4 [2, 8] | 0.126 | 1 / 1 | 0.765 | 2.82 | 277 |
| nugget | LL | sinhalf8 | BFGS:default | 180 | 54 | 30 [24, 37] | 0.565 | 1 / 2.17 | 1.04 | 4.11 | 225 |
| nugget | LL | sinhalf8 | BFGS4:default | 180 | 19 | 11 [7, 16] | 0.297 | 1 / 1.21 | 0.923 | 3.61 | 253 |
| nugget | LL | sinhalf8 | BFGS10:default | 180 | 5 | 3 [1, 6] | 0.268 | 1 / 1 | 1.25 | 4.64 | 382 |
| nugget | LMP | branin4 | BFGS:default | 120 | 2 | 2 [0, 6] | 0.0364 | 1 / 1.02 | 0.285 | 1.14 | 57.8 |
| nugget | LMP | branin4 | BFGS4:default | 120 | 0 | 0 [0, 3] | 0.0364 | 1 / 1 | 0.348 | 1.24 | 62.7 |
| nugget | LMP | branin4 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.0364 | 1 / 1 | 0.648 | 2.26 | 99.1 |
| nugget | LMP | hartmann6 | BFGS:default | 120 | 0 | 0 [0, 3] | 0.561 | 1 / 1 | 0.269 | 1.07 | 53 |
| nugget | LMP | hartmann6 | BFGS4:default | 120 | 0 | 0 [0, 3] | 0.561 | 1 / 1 | 0.5 | 1.63 | 78.3 |
| nugget | LMP | hartmann6 | BFGS10:default | 120 | 0 | 0 [0, 3] | 0.56 | 1 / 1 | 0.919 | 3.14 | 131 |
| nugget | LMP | sin4 | BFGS:default | 120 | 46 | 38 [30, 47] | 0.557 | 1.05 / 6.89 | 0.32 | 1.27 | 70.2 |
| nugget | LMP | sin4 | BFGS4:default | 120 | 21 | 18 [12, 25] | 0.264 | 1 / 1.34 | 0.389 | 1.37 | 82 |
| nugget | LMP | sin4 | BFGS10:default | 120 | 2 | 2 [0, 6] | 0.223 | 1 / 1 | 0.604 | 2.17 | 120 |
| nugget | LMP | sinhalf8 | BFGS:default | 120 | 37 | 31 [23, 40] | 0.785 | 1.06 / 1.61 | 0.341 | 1.36 | 78.7 |
| nugget | LMP | sinhalf8 | BFGS4:default | 120 | 7 | 6 [3, 12] | 0.65 | 1 / 1.14 | 0.591 | 1.73 | 98.5 |
| nugget | LMP | sinhalf8 | BFGS10:default | 120 | 4 | 3 [1, 8] | 0.629 | 1 / 1 | 0.789 | 2.67 | 138 |

### Per configuration

| kernel | noise | objective | function | n | variant | sets | miss | miss % [95 % CI] | RMSE | RMSE / best, median / p90 | wall s | CPU s | wall total s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gauss | none | LL | branin4 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.00374 | 1 / 1 | 0.0369 | 0.14 | 1.15 |
| gauss | none | LL | branin4 | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 3 | 10 [3, 26] | 0.00391 | 1 / 1.16 | 0.0165 | 0.0618 | 0.522 |
| gauss | none | LL | branin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.00374 | 1 / 1 | 0.0311 | 0.117 | 0.98 |
| gauss | none | LL | branin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00374 | 1 / 1 | 0.0599 | 0.232 | 1.95 |
| gauss | none | LL | branin4 | 200 | BFGS:default | 30 | 2 | 7 [2, 21] | 0.000295 | 1 / 1 | 0.667 | 2.59 | 24.6 |
| gauss | none | LL | branin4 | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 10 | 33 [19, 51] | 0.000419 | 1 / 796 | 0.196 | 0.775 | 8.49 |
| gauss | none | LL | branin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.000295 | 1 / 1 | 0.158 | 0.561 | 4.99 |
| gauss | none | LL | branin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.000295 | 1 / 1 | 0.213 | 0.785 | 6.67 |
| gauss | none | LL | branin4 | 500 | BFGS:default | 30 | 0 | 0 [0, 11] | 3.45e-05 | 1 / 1 | 1.06 | 4.23 | 34.7 |
| gauss | none | LL | branin4 | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 0 | 0 [0, 11] | 3.45e-05 | 1 / 1 | 0.363 | 1.44 | 12.2 |
| gauss | none | LL | branin4 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 3.45e-05 | 1 / 1 | 0.775 | 2.1 | 23.1 |
| gauss | none | LL | branin4 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 3.45e-05 | 1 / 1 | 1.31 | 4.48 | 39.9 |
| gauss | none | LL | branin4 | 1000 | BFGS:default | 30 | 0 | 0 [0, 11] | 9e-06 | 1 / 1 | 2.58 | 10.2 | 82.9 |
| gauss | none | LL | branin4 | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 0 | 0 [0, 11] | 9e-06 | 1 / 1 | 0.999 | 3.95 | 31 |
| gauss | none | LL | branin4 | 1000 | BFGS4:default | 30 | 0 | 0 [0, 11] | 9e-06 | 1 / 1 | 3.38 | 9.95 | 110 |
| gauss | none | LL | branin4 | 1000 | BFGS10:default | 30 | 0 | 0 [0, 11] | 9e-06 | 1 / 1 | 6.51 | 22.4 | 195 |
| gauss | none | LL | hartmann6 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.643 | 1 / 1 | 0.0415 | 0.158 | 1.24 |
| gauss | none | LL | hartmann6 | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 6 | 20 [10, 37] | 0.645 | 1 / 1.36 | 0.0176 | 0.069 | 0.602 |
| gauss | none | LL | hartmann6 | 100 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.645 | 1 / 1 | 0.0367 | 0.146 | 1.16 |
| gauss | none | LL | hartmann6 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.645 | 1 / 1 | 0.0851 | 0.336 | 2.69 |
| gauss | none | LL | hartmann6 | 200 | BFGS:default | 30 | 1 | 3 [1, 17] | 0.472 | 1 / 1 | 0.386 | 1.52 | 17 |
| gauss | none | LL | hartmann6 | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 4 | 13 [5, 30] | 0.481 | 1 / 1.15 | 0.239 | 0.948 | 9.42 |
| gauss | none | LL | hartmann6 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.473 | 1 / 1 | 0.18 | 0.593 | 5.8 |
| gauss | none | LL | hartmann6 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.473 | 1 / 1 | 0.359 | 1.21 | 10.8 |
| gauss | none | LL | hartmann6 | 500 | BFGS:default | 30 | 4 | 13 [5, 30] | 0.294 | 1 / 1.03 | 1.09 | 4.36 | 34.7 |
| gauss | none | LL | hartmann6 | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 9 | 30 [17, 48] | 0.3 | 1 / 2.88 | 0.791 | 3.13 | 27.5 |
| gauss | none | LL | hartmann6 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.291 | 1 / 1 | 2.02 | 3.93 | 60.7 |
| gauss | none | LL | hartmann6 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.291 | 1 / 1 | 2.94 | 9.22 | 95.3 |
| gauss | none | LL | hartmann6 | 1000 | BFGS:default | 30 | 5 | 17 [7, 34] | 0.193 | 1 / 1.14 | 5.53 | 22 | 198 |
| gauss | none | LL | hartmann6 | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 12 | 40 [25, 58] | 0.205 | 1 / 3.51 | 4.66 | 18.5 | 163 |
| gauss | none | LL | hartmann6 | 1000 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.191 | 1 / 1 | 15.4 | 34.2 | 446 |
| gauss | none | LL | hartmann6 | 1000 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.191 | 1 / 1 | 24.7 | 82.8 | 830 |
| gauss | none | LL | sin4 | 100 | BFGS:default | 30 | 1 | 3 [1, 17] | 0.748 | 1 / 1.16 | 0.0243 | 0.095 | 0.982 |
| gauss | none | LL | sin4 | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 2 | 7 [2, 21] | 0.756 | 1 / 1.18 | 0.0249 | 0.0961 | 0.672 |
| gauss | none | LL | sin4 | 100 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.737 | 1 / 1 | 0.0401 | 0.151 | 1.28 |
| gauss | none | LL | sin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.733 | 1 / 1 | 0.091 | 0.353 | 2.83 |
| gauss | none | LL | sin4 | 200 | BFGS:default | 30 | 7 | 23 [12, 41] | 0.37 | 1 / 1.43 | 0.358 | 1.41 | 14.8 |
| gauss | none | LL | sin4 | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 11 | 37 [22, 54] | 0.449 | 1 / 2.48 | 0.296 | 1.18 | 11.3 |
| gauss | none | LL | sin4 | 200 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.311 | 1 / 1 | 0.179 | 0.599 | 5.45 |
| gauss | none | LL | sin4 | 200 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.311 | 1 / 1 | 0.346 | 1.17 | 10.5 |
| gauss | none | LL | sin4 | 500 | BFGS:default | 30 | 3 | 10 [3, 26] | 0.0965 | 1 / 1.01 | 0.664 | 2.64 | 21.8 |
| gauss | none | LL | sin4 | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 11 | 37 [22, 54] | 0.136 | 1 / 3.91 | 0.516 | 2.06 | 16.5 |
| gauss | none | LL | sin4 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0965 | 1 / 1 | 1.04 | 3.11 | 32.4 |
| gauss | none | LL | sin4 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0965 | 1 / 1 | 2.55 | 8.57 | 75.2 |
| gauss | none | LL | sin4 | 1000 | BFGS:default | 30 | 2 | 7 [2, 21] | 0.00878 | 1 / 1 | 2.82 | 11.2 | 81.1 |
| gauss | none | LL | sin4 | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 19 | 63 [46, 78] | 0.0703 | 4.27 / 28.4 | 1.26 | 5.02 | 45.6 |
| gauss | none | LL | sin4 | 1000 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.00749 | 1 / 1 | 5.56 | 14.8 | 170 |
| gauss | none | LL | sin4 | 1000 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00749 | 1 / 1 | 12.8 | 44.6 | 383 |
| gauss | none | LL | sinhalf8 | 100 | BFGS:default | 30 | 7 | 23 [12, 41] | 0.808 | 1.04 / 1.5 | 0.0523 | 0.201 | 1.56 |
| gauss | none | LL | sinhalf8 | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 8 | 27 [14, 44] | 0.811 | 1.08 / 1.75 | 0.0189 | 0.0747 | 0.574 |
| gauss | none | LL | sinhalf8 | 100 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.779 | 1 / 1.16 | 0.0377 | 0.147 | 1.2 |
| gauss | none | LL | sinhalf8 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.724 | 1 / 1.01 | 0.0927 | 0.365 | 2.87 |
| gauss | none | LL | sinhalf8 | 200 | BFGS:default | 30 | 16 | 53 [36, 70] | 0.679 | 1.19 / 2.15 | 0.713 | 2.83 | 24.1 |
| gauss | none | LL | sinhalf8 | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 18 | 60 [42, 75] | 0.708 | 1.27 / 2.15 | 0.231 | 0.895 | 8.33 |
| gauss | none | LL | sinhalf8 | 200 | BFGS4:default | 30 | 4 | 13 [5, 30] | 0.539 | 1 / 1.11 | 0.166 | 0.586 | 4.91 |
| gauss | none | LL | sinhalf8 | 200 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.522 | 1 / 1 | 0.333 | 1.14 | 10.2 |
| gauss | none | LL | sinhalf8 | 500 | BFGS:default | 30 | 18 | 60 [42, 75] | 0.265 | 1.21 / 7.28 | 1.32 | 5.23 | 39 |
| gauss | none | LL | sinhalf8 | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 20 | 67 [49, 81] | 0.366 | 1.23 / 7.28 | 0.604 | 2.39 | 20.5 |
| gauss | none | LL | sinhalf8 | 500 | BFGS4:default | 30 | 6 | 20 [10, 37] | 0.0985 | 1 / 2.22 | 1.09 | 3.37 | 35.8 |
| gauss | none | LL | sinhalf8 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0918 | 1 / 1 | 2.65 | 8.49 | 79.9 |
| gauss | none | LL | sinhalf8 | 1000 | BFGS:default | 30 | 16 | 53 [36, 70] | 0.0666 | 1.01 / 7.12 | 6.64 | 26 | 196 |
| gauss | none | LL | sinhalf8 | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 21 | 70 [52, 83] | 0.0738 | 1.15 / 7.12 | 2.81 | 11.1 | 93.9 |
| gauss | none | LL | sinhalf8 | 1000 | BFGS4:default | 30 | 2 | 7 [2, 21] | 0.0477 | 1 / 1 | 6.91 | 21.1 | 225 |
| gauss | none | LL | sinhalf8 | 1000 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0476 | 1 / 1 | 16 | 55.2 | 515 |
| gauss | none | LMP | branin4 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.00367 | 1 / 1 | 0.0333 | 0.129 | 1.08 |
| gauss | none | LMP | branin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.00367 | 1 / 1 | 0.076 | 0.3 | 2.39 |
| gauss | none | LMP | branin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00367 | 1 / 1 | 0.137 | 0.514 | 4.33 |
| gauss | none | LMP | branin4 | 200 | BFGS:default | 30 | 7 | 23 [12, 41] | 0.000311 | 1 / 621 | 0.348 | 1.37 | 11.2 |
| gauss | none | LMP | branin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.000294 | 1 / 1 | 0.339 | 0.97 | 10.9 |
| gauss | none | LMP | branin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.000294 | 1 / 1 | 0.561 | 2.03 | 18.1 |
| gauss | none | LMP | hartmann6 | 100 | BFGS:default | 30 | 2 | 7 [2, 21] | 0.669 | 1 / 1.04 | 0.0396 | 0.155 | 1.26 |
| gauss | none | LMP | hartmann6 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.664 | 1 / 1 | 0.0853 | 0.335 | 2.97 |
| gauss | none | LMP | hartmann6 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.663 | 1 / 1 | 0.205 | 0.734 | 6.19 |
| gauss | none | LMP | hartmann6 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.477 | 1 / 1 | 0.296 | 1.17 | 8.97 |
| gauss | none | LMP | hartmann6 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.477 | 1 / 1 | 0.368 | 1.23 | 11.2 |
| gauss | none | LMP | hartmann6 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.475 | 1 / 1 | 0.934 | 3.16 | 28.4 |
| gauss | none | LMP | sin4 | 100 | BFGS:default | 30 | 1 | 3 [1, 17] | 0.747 | 1 / 1.07 | 0.0373 | 0.141 | 1.14 |
| gauss | none | LMP | sin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.746 | 1 / 1 | 0.0775 | 0.303 | 2.49 |
| gauss | none | LMP | sin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.746 | 1 / 1 | 0.161 | 0.56 | 4.81 |
| gauss | none | LMP | sin4 | 200 | BFGS:default | 30 | 9 | 30 [17, 48] | 0.419 | 1 / 1.57 | 0.369 | 1.46 | 11.4 |
| gauss | none | LMP | sin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.309 | 1 / 1 | 0.339 | 1.12 | 10.6 |
| gauss | none | LMP | sin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.309 | 1 / 1 | 0.793 | 2.65 | 23.6 |
| gauss | none | LMP | sinhalf8 | 100 | BFGS:default | 30 | 6 | 20 [10, 37] | 0.816 | 1 / 1.4 | 0.0664 | 0.262 | 2.05 |
| gauss | none | LMP | sinhalf8 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.809 | 1 / 1.09 | 0.15 | 0.528 | 4.45 |
| gauss | none | LMP | sinhalf8 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.726 | 1 / 1 | 0.305 | 1.07 | 9.07 |
| gauss | none | LMP | sinhalf8 | 200 | BFGS:default | 30 | 19 | 63 [46, 78] | 0.755 | 1.4 / 2.29 | 0.485 | 1.91 | 14.2 |
| gauss | none | LMP | sinhalf8 | 200 | BFGS4:default | 30 | 3 | 10 [3, 26] | 0.516 | 1 / 1.03 | 0.598 | 1.92 | 18.6 |
| gauss | none | LMP | sinhalf8 | 200 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.519 | 1 / 1 | 1.43 | 4.9 | 42.8 |
| gauss | none | LOO | branin4 | 100 | BFGS:default | 30 | 17 | 57 [39, 73] | 0.00549 | 1.13 / 62.5 | 0.0241 | 0.095 | 0.819 |
| gauss | none | LOO | branin4 | 100 | BFGS4:default | 30 | 4 | 13 [5, 30] | 0.0038 | 1 / 1.01 | 0.0516 | 0.203 | 2.61 |
| gauss | none | LOO | branin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00386 | 1 / 1 | 0.131 | 0.507 | 6.94 |
| gauss | none | LOO | branin4 | 200 | BFGS:default | 30 | 21 | 70 [52, 83] | 0.000609 | 2.5 / 324 | 0.338 | 1.32 | 12.4 |
| gauss | none | LOO | branin4 | 200 | BFGS4:default | 30 | 11 | 37 [22, 54] | 0.00032 | 1 / 1.08 | 0.205 | 0.658 | 7 |
| gauss | none | LOO | branin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.000317 | 1 / 1 | 0.413 | 1.4 | 15.9 |
| gauss | none | LOO | hartmann6 | 100 | BFGS:default | 30 | 8 | 27 [14, 44] | 0.7 | 1 / 1.37 | 0.0279 | 0.107 | 0.878 |
| gauss | none | LOO | hartmann6 | 100 | BFGS4:default | 30 | 2 | 7 [2, 21] | 0.665 | 1 / 1 | 0.0482 | 0.192 | 1.62 |
| gauss | none | LOO | hartmann6 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.675 | 1 / 1 | 0.121 | 0.468 | 4.14 |
| gauss | none | LOO | hartmann6 | 200 | BFGS:default | 30 | 1 | 3 [1, 17] | 0.485 | 1 / 1 | 0.302 | 1.19 | 9.22 |
| gauss | none | LOO | hartmann6 | 200 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.482 | 1 / 1 | 0.213 | 0.693 | 6.56 |
| gauss | none | LOO | hartmann6 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.482 | 1 / 1 | 0.481 | 1.75 | 15 |
| gauss | none | LOO | sin4 | 100 | BFGS:default | 30 | 12 | 40 [25, 58] | 0.764 | 1.01 / 1.4 | 0.036 | 0.14 | 1.28 |
| gauss | none | LOO | sin4 | 100 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.763 | 1 / 1 | 0.0512 | 0.194 | 1.56 |
| gauss | none | LOO | sin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.765 | 1 / 1 | 0.103 | 0.405 | 3.37 |
| gauss | none | LOO | sin4 | 200 | BFGS:default | 30 | 13 | 43 [27, 61] | 0.545 | 1.01 / 2.85 | 0.432 | 1.7 | 15.1 |
| gauss | none | LOO | sin4 | 200 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.297 | 1 / 1 | 0.26 | 0.767 | 8.06 |
| gauss | none | LOO | sin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.301 | 1 / 1 | 0.533 | 1.72 | 19.8 |
| gauss | none | LOO | sinhalf8 | 100 | BFGS:default | 30 | 22 | 73 [56, 86] | 0.958 | 1.11 / 2.8 | 0.0461 | 0.18 | 1.43 |
| gauss | none | LOO | sinhalf8 | 100 | BFGS4:default | 30 | 5 | 17 [7, 34] | 0.829 | 1 / 1.3 | 0.0712 | 0.276 | 2.37 |
| gauss | none | LOO | sinhalf8 | 100 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.789 | 1 / 1.01 | 0.191 | 0.638 | 6.05 |
| gauss | none | LOO | sinhalf8 | 200 | BFGS:default | 30 | 24 | 80 [63, 90] | 0.793 | 1.36 / 2.8 | 0.375 | 1.48 | 11.1 |
| gauss | none | LOO | sinhalf8 | 200 | BFGS4:default | 30 | 12 | 40 [25, 58] | 0.735 | 1.02 / 1.62 | 0.296 | 0.946 | 10.4 |
| gauss | none | LOO | sinhalf8 | 200 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.588 | 1 / 1 | 0.709 | 2.42 | 22.2 |
| gauss | nugget | LL | branin4 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.0434 | 1 / 1 | 0.0456 | 0.171 | 1.55 |
| gauss | nugget | LL | branin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0434 | 1 / 1 | 0.0622 | 0.247 | 2.03 |
| gauss | nugget | LL | branin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0434 | 1 / 1 | 0.0921 | 0.363 | 3 |
| gauss | nugget | LL | branin4 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.0301 | 1 / 1 | 0.855 | 3.37 | 27.7 |
| gauss | nugget | LL | branin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0301 | 1 / 1 | 0.629 | 2.48 | 20.2 |
| gauss | nugget | LL | branin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0301 | 1 / 1 | 0.659 | 2.49 | 22.1 |
| gauss | nugget | LL | branin4 | 500 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.0182 | 1 / 1 | 2.04 | 8.09 | 61.7 |
| gauss | nugget | LL | branin4 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0182 | 1 / 1 | 2.26 | 7.85 | 68.9 |
| gauss | nugget | LL | branin4 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0182 | 1 / 1 | 2.87 | 10.1 | 88.5 |
| gauss | nugget | LL | hartmann6 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.649 | 1 / 1 | 0.046 | 0.183 | 1.52 |
| gauss | nugget | LL | hartmann6 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.649 | 1 / 1 | 0.0732 | 0.292 | 2.27 |
| gauss | nugget | LL | hartmann6 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.649 | 1 / 1 | 0.121 | 0.477 | 3.86 |
| gauss | nugget | LL | hartmann6 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.48 | 1 / 1 | 0.553 | 2.18 | 24.3 |
| gauss | nugget | LL | hartmann6 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.486 | 1 / 1 | 0.423 | 1.65 | 17.9 |
| gauss | nugget | LL | hartmann6 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.486 | 1 / 1 | 0.692 | 2.5 | 26 |
| gauss | nugget | LL | hartmann6 | 500 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.295 | 1 / 1 | 1.35 | 5.34 | 39.8 |
| gauss | nugget | LL | hartmann6 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.295 | 1 / 1 | 1.8 | 6.43 | 57.7 |
| gauss | nugget | LL | hartmann6 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.295 | 1 / 1 | 3.58 | 11.5 | 111 |
| gauss | nugget | LL | sin4 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.714 | 1 / 1.11 | 0.0377 | 0.141 | 1.41 |
| gauss | nugget | LL | sin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.735 | 1 / 1.09 | 0.0556 | 0.213 | 1.87 |
| gauss | nugget | LL | sin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.716 | 1 / 1 | 0.0897 | 0.343 | 2.95 |
| gauss | nugget | LL | sin4 | 200 | BFGS:default | 30 | 2 | 7 [2, 21] | 0.339 | 1 / 1.02 | 0.553 | 2.18 | 19 |
| gauss | nugget | LL | sin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.339 | 1 / 1 | 0.401 | 1.57 | 14.5 |
| gauss | nugget | LL | sin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.339 | 1 / 1 | 0.592 | 2.16 | 19.7 |
| gauss | nugget | LL | sin4 | 500 | BFGS:default | 30 | 1 | 3 [1, 17] | 0.147 | 1 / 1 | 1.09 | 4.31 | 43.6 |
| gauss | nugget | LL | sin4 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.147 | 1 / 1 | 1.68 | 5.04 | 60.4 |
| gauss | nugget | LL | sin4 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.147 | 1 / 1 | 2.21 | 8 | 78.1 |
| gauss | nugget | LL | sinhalf8 | 100 | BFGS:default | 30 | 5 | 17 [7, 34] | 0.762 | 1 / 1.39 | 0.0682 | 0.272 | 1.97 |
| gauss | nugget | LL | sinhalf8 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.702 | 1 / 1.11 | 0.0795 | 0.31 | 2.39 |
| gauss | nugget | LL | sinhalf8 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.695 | 1 / 1 | 0.118 | 0.461 | 3.83 |
| gauss | nugget | LL | sinhalf8 | 200 | BFGS:default | 30 | 14 | 47 [30, 64] | 0.641 | 1.07 / 1.7 | 1.16 | 4.6 | 35.5 |
| gauss | nugget | LL | sinhalf8 | 200 | BFGS4:default | 30 | 7 | 23 [12, 41] | 0.624 | 1 / 1.35 | 0.787 | 3.08 | 26.2 |
| gauss | nugget | LL | sinhalf8 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.567 | 1 / 1 | 1.15 | 4.13 | 36 |
| gauss | nugget | LL | sinhalf8 | 500 | BFGS:default | 30 | 10 | 33 [19, 51] | 0.177 | 1 / 4.22 | 1.85 | 7.34 | 54.3 |
| gauss | nugget | LL | sinhalf8 | 500 | BFGS4:default | 30 | 3 | 10 [3, 26] | 0.121 | 1 / 1 | 2.13 | 7.59 | 65.7 |
| gauss | nugget | LL | sinhalf8 | 500 | BFGS10:default | 30 | 3 | 10 [3, 26] | 0.125 | 1 / 1.08 | 3.49 | 12.5 | 104 |
| gauss | nugget | LMP | branin4 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.0429 | 1 / 1 | 0.0713 | 0.274 | 2.11 |
| gauss | nugget | LMP | branin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0429 | 1 / 1 | 0.106 | 0.416 | 3.47 |
| gauss | nugget | LMP | branin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0429 | 1 / 1 | 0.189 | 0.674 | 5.85 |
| gauss | nugget | LMP | branin4 | 200 | BFGS:default | 30 | 1 | 3 [1, 17] | 0.0299 | 1 / 1 | 0.884 | 3.49 | 27.5 |
| gauss | nugget | LMP | branin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0299 | 1 / 1 | 0.937 | 3.36 | 29.6 |
| gauss | nugget | LMP | branin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0299 | 1 / 1 | 1.29 | 4.79 | 45.3 |
| gauss | nugget | LMP | hartmann6 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.676 | 1 / 1 | 0.071 | 0.283 | 2.24 |
| gauss | nugget | LMP | hartmann6 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.681 | 1 / 1 | 0.135 | 0.53 | 4.05 |
| gauss | nugget | LMP | hartmann6 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.681 | 1 / 1 | 0.232 | 0.806 | 6.97 |
| gauss | nugget | LMP | hartmann6 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.484 | 1 / 1 | 0.796 | 3.13 | 26.2 |
| gauss | nugget | LMP | hartmann6 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.49 | 1 / 1 | 1.17 | 3.37 | 37.1 |
| gauss | nugget | LMP | hartmann6 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.49 | 1 / 1 | 1.92 | 6.65 | 58 |
| gauss | nugget | LMP | sin4 | 100 | BFGS:default | 30 | 5 | 17 [7, 34] | 0.757 | 1 / 1.29 | 0.0536 | 0.201 | 1.94 |
| gauss | nugget | LMP | sin4 | 100 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.747 | 1 / 1.04 | 0.103 | 0.404 | 3.18 |
| gauss | nugget | LMP | sin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.704 | 1 / 1 | 0.175 | 0.64 | 5.67 |
| gauss | nugget | LMP | sin4 | 200 | BFGS:default | 30 | 2 | 7 [2, 21] | 0.337 | 1 / 1.08 | 0.704 | 2.78 | 22.9 |
| gauss | nugget | LMP | sin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.337 | 1 / 1.01 | 0.766 | 2.65 | 24.3 |
| gauss | nugget | LMP | sin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.337 | 1 / 1 | 1.33 | 4.58 | 39.3 |
| gauss | nugget | LMP | sinhalf8 | 100 | BFGS:default | 30 | 4 | 13 [5, 30] | 0.889 | 1.03 / 1.38 | 0.0796 | 0.309 | 2.44 |
| gauss | nugget | LMP | sinhalf8 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.843 | 1 / 1.17 | 0.133 | 0.528 | 4.36 |
| gauss | nugget | LMP | sinhalf8 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.818 | 1 / 1.01 | 0.198 | 0.697 | 6.23 |
| gauss | nugget | LMP | sinhalf8 | 200 | BFGS:default | 30 | 19 | 63 [46, 78] | 0.835 | 1.17 / 1.78 | 1.03 | 4.06 | 32.3 |
| gauss | nugget | LMP | sinhalf8 | 200 | BFGS4:default | 30 | 4 | 13 [5, 30] | 0.626 | 1 / 1.2 | 1.24 | 4.03 | 41.4 |
| gauss | nugget | LMP | sinhalf8 | 200 | BFGS10:default | 30 | 3 | 10 [3, 26] | 0.636 | 1 / 1.09 | 1.77 | 6.21 | 57.9 |
| matern5_2 | none | LL | branin4 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.00552 | 1 / 1 | 0.0349 | 0.129 | 1.12 |
| matern5_2 | none | LL | branin4 | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 0 | 0 [0, 11] | 0.00552 | 1 / 1 | 0.0111 | 0.0444 | 0.354 |
| matern5_2 | none | LL | branin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.00552 | 1 / 1 | 0.0283 | 0.107 | 0.899 |
| matern5_2 | none | LL | branin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00552 | 1 / 1 | 0.0589 | 0.234 | 1.88 |
| matern5_2 | none | LL | branin4 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.000888 | 1 / 1 | 0.392 | 1.53 | 13.8 |
| matern5_2 | none | LL | branin4 | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 0 | 0 [0, 11] | 0.000888 | 1 / 1 | 0.183 | 0.72 | 6.36 |
| matern5_2 | none | LL | branin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.000888 | 1 / 1 | 0.133 | 0.513 | 5 |
| matern5_2 | none | LL | branin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.000888 | 1 / 1 | 0.24 | 0.835 | 7.59 |
| matern5_2 | none | LL | branin4 | 500 | BFGS:default | 30 | 0 | 0 [0, 11] | 8.71e-05 | 1 / 1 | 0.612 | 2.42 | 21.8 |
| matern5_2 | none | LL | branin4 | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 0 | 0 [0, 11] | 8.71e-05 | 1 / 1 | 0.196 | 0.775 | 5.9 |
| matern5_2 | none | LL | branin4 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 8.71e-05 | 1 / 1 | 0.519 | 1.56 | 18.8 |
| matern5_2 | none | LL | branin4 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 8.71e-05 | 1 / 1 | 1.28 | 4.36 | 41.5 |
| matern5_2 | none | LL | branin4 | 1000 | BFGS:default | 30 | 1 | 3 [1, 17] | 2.2e-05 | 1 / 1 | 2.95 | 11.7 | 205 |
| matern5_2 | none | LL | branin4 | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 1 | 3 [1, 17] | 2.2e-05 | 1 / 1 | 0.878 | 3.47 | 38.6 |
| matern5_2 | none | LL | branin4 | 1000 | BFGS4:default | 30 | 1 | 3 [1, 17] | 2.19e-05 | 1 / 1 | 3.98 | 10 | 194 |
| matern5_2 | none | LL | branin4 | 1000 | BFGS10:default | 30 | 1 | 3 [1, 17] | 2.2e-05 | 1 / 1 | 9.44 | 27.5 | 567 |
| matern5_2 | none | LL | hartmann6 | 100 | BFGS:default | 30 | 1 | 3 [1, 17] | 0.625 | 1 / 1 | 0.0638 | 0.252 | 2.28 |
| matern5_2 | none | LL | hartmann6 | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 4 | 13 [5, 30] | 0.639 | 1 / 1.08 | 0.0432 | 0.17 | 1.24 |
| matern5_2 | none | LL | hartmann6 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.625 | 1 / 1 | 0.0629 | 0.24 | 1.88 |
| matern5_2 | none | LL | hartmann6 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.625 | 1 / 1 | 0.142 | 0.522 | 4.35 |
| matern5_2 | none | LL | hartmann6 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.453 | 1 / 1 | 0.479 | 1.9 | 15.4 |
| matern5_2 | none | LL | hartmann6 | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 4 | 13 [5, 30] | 0.453 | 1 / 1.11 | 0.24 | 0.95 | 8.86 |
| matern5_2 | none | LL | hartmann6 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.453 | 1 / 1 | 0.237 | 0.703 | 7.15 |
| matern5_2 | none | LL | hartmann6 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.458 | 1 / 1 | 0.475 | 1.66 | 14.7 |
| matern5_2 | none | LL | hartmann6 | 500 | BFGS:default | 30 | 4 | 13 [5, 30] | 0.273 | 1 / 1.03 | 0.87 | 3.45 | 26.9 |
| matern5_2 | none | LL | hartmann6 | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 9 | 30 [17, 48] | 0.274 | 1 / 1.72 | 0.892 | 3.53 | 26.4 |
| matern5_2 | none | LL | hartmann6 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.267 | 1 / 1 | 0.968 | 2.78 | 34.6 |
| matern5_2 | none | LL | hartmann6 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.267 | 1 / 1 | 2.61 | 9.18 | 81.5 |
| matern5_2 | none | LL | hartmann6 | 1000 | BFGS:default | 30 | 4 | 13 [5, 30] | 0.167 | 1 / 1 | 1.33 | 5.27 | 61.7 |
| matern5_2 | none | LL | hartmann6 | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 5 | 17 [7, 34] | 0.168 | 1 / 1.01 | 1.72 | 6.81 | 70.9 |
| matern5_2 | none | LL | hartmann6 | 1000 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.166 | 1 / 1 | 3.54 | 11.8 | 134 |
| matern5_2 | none | LL | hartmann6 | 1000 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.166 | 1 / 1 | 11.9 | 40.3 | 411 |
| matern5_2 | none | LL | sin4 | 100 | BFGS:default | 30 | 8 | 27 [14, 44] | 0.09 | 1 / 14.5 | 0.0562 | 0.221 | 2.26 |
| matern5_2 | none | LL | sin4 | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 12 | 40 [25, 58] | 0.345 | 1.02 / 15.2 | 0.0371 | 0.145 | 1.02 |
| matern5_2 | none | LL | sin4 | 100 | BFGS4:default | 30 | 7 | 23 [12, 41] | 0.0403 | 1 / 14.8 | 0.0474 | 0.181 | 1.5 |
| matern5_2 | none | LL | sin4 | 100 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.0377 | 1 / 1 | 0.111 | 0.435 | 3.46 |
| matern5_2 | none | LL | sin4 | 200 | BFGS:default | 30 | 3 | 10 [3, 26] | 0.00696 | 1 / 4.66 | 0.936 | 3.7 | 31.7 |
| matern5_2 | none | LL | sin4 | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 7 | 23 [12, 41] | 0.00696 | 1 / 6.46 | 0.157 | 0.62 | 6.44 |
| matern5_2 | none | LL | sin4 | 200 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.00696 | 1 / 1 | 0.198 | 0.654 | 6.29 |
| matern5_2 | none | LL | sin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00696 | 1 / 1 | 0.414 | 1.38 | 12.4 |
| matern5_2 | none | LL | sin4 | 500 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.00155 | 1 / 1 | 0.967 | 3.85 | 34 |
| matern5_2 | none | LL | sin4 | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 0 | 0 [0, 11] | 0.00155 | 1 / 1 | 0.29 | 1.15 | 9.22 |
| matern5_2 | none | LL | sin4 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.00155 | 1 / 1 | 1.14 | 2.95 | 37.6 |
| matern5_2 | none | LL | sin4 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00155 | 1 / 1 | 2.45 | 8.06 | 74.6 |
| matern5_2 | none | LL | sin4 | 1000 | BFGS:default | 30 | 4 | 13 [5, 30] | 0.000368 | 1 / 1 | 7.58 | 30 | 428 |
| matern5_2 | none | LL | sin4 | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 11 | 37 [22, 54] | 0.000486 | 1 / 2.1 | 1.63 | 6.47 | 66.1 |
| matern5_2 | none | LL | sin4 | 1000 | BFGS4:default | 30 | 4 | 13 [5, 30] | 0.000363 | 1 / 1.01 | 8.67 | 24.1 | 339 |
| matern5_2 | none | LL | sin4 | 1000 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00036 | 1 / 1 | 23.1 | 72.8 | 903 |
| matern5_2 | none | LL | sinhalf8 | 100 | BFGS:default | 30 | 7 | 23 [12, 41] | 0.742 | 1 / 2.2 | 0.102 | 0.405 | 3.36 |
| matern5_2 | none | LL | sinhalf8 | 100 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 10 | 33 [19, 51] | 0.79 | 1.09 / 2.42 | 0.0447 | 0.178 | 1.49 |
| matern5_2 | none | LL | sinhalf8 | 100 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.577 | 1 / 1.11 | 0.0697 | 0.275 | 2.12 |
| matern5_2 | none | LL | sinhalf8 | 100 | BFGS10:default | 30 | 2 | 7 [2, 21] | 0.628 | 1 / 1 | 0.179 | 0.641 | 5.32 |
| matern5_2 | none | LL | sinhalf8 | 200 | BFGS:default | 30 | 13 | 43 [27, 61] | 0.47 | 1.02 / 4.28 | 0.809 | 3.17 | 27.2 |
| matern5_2 | none | LL | sinhalf8 | 200 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 16 | 53 [36, 70] | 0.661 | 1.09 / 5.72 | 0.29 | 1.15 | 9.53 |
| matern5_2 | none | LL | sinhalf8 | 200 | BFGS4:default | 30 | 4 | 13 [5, 30] | 0.174 | 1 / 1.42 | 0.306 | 0.995 | 9.05 |
| matern5_2 | none | LL | sinhalf8 | 200 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.162 | 1 / 1 | 0.595 | 2.03 | 18.6 |
| matern5_2 | none | LL | sinhalf8 | 500 | BFGS:default | 30 | 5 | 17 [7, 34] | 0.0138 | 1 / 14.9 | 1.59 | 6.32 | 49.3 |
| matern5_2 | none | LL | sinhalf8 | 500 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 7 | 23 [12, 41] | 0.0138 | 1 / 24.2 | 0.657 | 2.61 | 20 |
| matern5_2 | none | LL | sinhalf8 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0134 | 1 / 1 | 1.6 | 4.3 | 48.5 |
| matern5_2 | none | LL | sinhalf8 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0134 | 1 / 1 | 3.58 | 12.1 | 107 |
| matern5_2 | none | LL | sinhalf8 | 1000 | BFGS:default | 30 | 1 | 3 [1, 17] | 0.00415 | 1 / 1 | 6.29 | 24.8 | 186 |
| matern5_2 | none | LL | sinhalf8 | 1000 | BFGS:LK_LONG_RANGE_RESTART=0 | 30 | 4 | 13 [5, 30] | 0.00516 | 1 / 2.41 | 2.32 | 9.12 | 78.1 |
| matern5_2 | none | LL | sinhalf8 | 1000 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.00415 | 1 / 1 | 7.84 | 21 | 231 |
| matern5_2 | none | LL | sinhalf8 | 1000 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00415 | 1 / 1 | 16 | 51.9 | 493 |
| matern5_2 | none | LMP | branin4 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.00537 | 1 / 1 | 0.0212 | 0.0817 | 0.657 |
| matern5_2 | none | LMP | branin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.00537 | 1 / 1 | 0.0463 | 0.183 | 1.41 |
| matern5_2 | none | LMP | branin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00537 | 1 / 1 | 0.0935 | 0.371 | 2.85 |
| matern5_2 | none | LMP | branin4 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.000884 | 1 / 1 | 0.189 | 0.741 | 5.61 |
| matern5_2 | none | LMP | branin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.000884 | 1 / 1 | 0.231 | 0.711 | 8.26 |
| matern5_2 | none | LMP | branin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.000884 | 1 / 1 | 0.454 | 1.55 | 13.9 |
| matern5_2 | none | LMP | hartmann6 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.64 | 1 / 1 | 0.048 | 0.188 | 1.47 |
| matern5_2 | none | LMP | hartmann6 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.64 | 1 / 1 | 0.0708 | 0.277 | 2.28 |
| matern5_2 | none | LMP | hartmann6 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.64 | 1 / 1 | 0.197 | 0.686 | 6.09 |
| matern5_2 | none | LMP | hartmann6 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.455 | 1 / 1 | 0.257 | 1.02 | 7.97 |
| matern5_2 | none | LMP | hartmann6 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.455 | 1 / 1 | 0.334 | 1.09 | 10.3 |
| matern5_2 | none | LMP | hartmann6 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.455 | 1 / 1 | 0.883 | 2.99 | 26.8 |
| matern5_2 | none | LMP | sin4 | 100 | BFGS:default | 30 | 10 | 33 [19, 51] | 0.0396 | 1 / 13.1 | 0.0441 | 0.176 | 1.35 |
| matern5_2 | none | LMP | sin4 | 100 | BFGS4:default | 30 | 6 | 20 [10, 37] | 0.0354 | 1 / 12.8 | 0.0734 | 0.288 | 2.3 |
| matern5_2 | none | LMP | sin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0354 | 1 / 1 | 0.168 | 0.572 | 5.17 |
| matern5_2 | none | LMP | sin4 | 200 | BFGS:default | 30 | 6 | 20 [10, 37] | 0.00782 | 1 / 1.45 | 0.339 | 1.3 | 11.4 |
| matern5_2 | none | LMP | sin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.00691 | 1 / 1 | 0.439 | 1.27 | 13.6 |
| matern5_2 | none | LMP | sin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00691 | 1 / 1 | 0.834 | 2.81 | 24.9 |
| matern5_2 | none | LMP | sinhalf8 | 100 | BFGS:default | 30 | 11 | 37 [22, 54] | 0.744 | 1.02 / 2.71 | 0.0875 | 0.34 | 2.68 |
| matern5_2 | none | LMP | sinhalf8 | 100 | BFGS4:default | 30 | 3 | 10 [3, 26] | 0.465 | 1 / 1.6 | 0.132 | 0.509 | 4.01 |
| matern5_2 | none | LMP | sinhalf8 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.362 | 1 / 1 | 0.306 | 1.08 | 9.44 |
| matern5_2 | none | LMP | sinhalf8 | 200 | BFGS:default | 30 | 11 | 37 [22, 54] | 0.164 | 1 / 4.04 | 0.533 | 2.08 | 16.7 |
| matern5_2 | none | LMP | sinhalf8 | 200 | BFGS4:default | 30 | 1 | 3 [1, 17] | 0.151 | 1 / 1 | 0.666 | 2.15 | 20 |
| matern5_2 | none | LMP | sinhalf8 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.151 | 1 / 1 | 1.5 | 5.02 | 45.6 |
| matern5_2 | none | LOO | branin4 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.0048 | 1 / 1 | 0.0579 | 0.226 | 5.77 |
| matern5_2 | none | LOO | branin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0048 | 1 / 1 | 0.131 | 0.489 | 7.77 |
| matern5_2 | none | LOO | branin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0048 | 1 / 1 | 0.559 | 1.36 | 17.2 |
| matern5_2 | none | LOO | branin4 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.000894 | 1 / 1 | 0.44 | 1.7 | 42.4 |
| matern5_2 | none | LOO | branin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.000893 | 1 / 1 | 0.585 | 1.15 | 33.8 |
| matern5_2 | none | LOO | branin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.000893 | 1 / 1 | 1.71 | 3.1 | 56 |
| matern5_2 | none | LOO | hartmann6 | 100 | BFGS:default | 30 | 2 | 7 [2, 21] | 0.644 | 1 / 1.05 | 0.0505 | 0.198 | 1.58 |
| matern5_2 | none | LOO | hartmann6 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.636 | 1 / 1 | 0.0678 | 0.264 | 2.16 |
| matern5_2 | none | LOO | hartmann6 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.63 | 1 / 1 | 0.208 | 0.707 | 6.38 |
| matern5_2 | none | LOO | hartmann6 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.457 | 1 / 1 | 0.261 | 1.02 | 8.04 |
| matern5_2 | none | LOO | hartmann6 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.454 | 1 / 1 | 0.291 | 0.897 | 8.67 |
| matern5_2 | none | LOO | hartmann6 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.454 | 1 / 1 | 0.665 | 2.27 | 20.3 |
| matern5_2 | none | LOO | sin4 | 100 | BFGS:default | 30 | 10 | 33 [19, 51] | 0.0419 | 1 / 21.3 | 0.0392 | 0.157 | 1.34 |
| matern5_2 | none | LOO | sin4 | 100 | BFGS4:default | 30 | 6 | 20 [10, 37] | 0.0325 | 1 / 17.3 | 0.0599 | 0.234 | 2.23 |
| matern5_2 | none | LOO | sin4 | 100 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.0325 | 1 / 1 | 0.154 | 0.547 | 5.96 |
| matern5_2 | none | LOO | sin4 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.00618 | 1 / 1 | 0.278 | 1.09 | 19.9 |
| matern5_2 | none | LOO | sin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.00618 | 1 / 1 | 0.263 | 0.818 | 21.7 |
| matern5_2 | none | LOO | sin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.00618 | 1 / 1 | 0.602 | 1.86 | 29.1 |
| matern5_2 | none | LOO | sinhalf8 | 100 | BFGS:default | 30 | 22 | 73 [56, 86] | 0.84 | 1.38 / 4.68 | 0.0687 | 0.269 | 2.08 |
| matern5_2 | none | LOO | sinhalf8 | 100 | BFGS4:default | 30 | 6 | 20 [10, 37] | 0.774 | 1 / 2.53 | 0.0932 | 0.359 | 2.92 |
| matern5_2 | none | LOO | sinhalf8 | 100 | BFGS10:default | 30 | 2 | 7 [2, 21] | 0.335 | 1 / 1 | 0.25 | 0.881 | 8.05 |
| matern5_2 | none | LOO | sinhalf8 | 200 | BFGS:default | 30 | 21 | 70 [52, 83] | 0.706 | 4.05 / 10.7 | 0.438 | 1.71 | 12.7 |
| matern5_2 | none | LOO | sinhalf8 | 200 | BFGS4:default | 30 | 5 | 17 [7, 34] | 0.109 | 1 / 2.93 | 0.464 | 1.46 | 15.1 |
| matern5_2 | none | LOO | sinhalf8 | 200 | BFGS10:default | 30 | 3 | 10 [3, 26] | 0.11 | 1 / 1.06 | 1.07 | 3.66 | 33.1 |
| matern5_2 | nugget | LL | branin4 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.0471 | 1 / 1 | 0.0619 | 0.245 | 2.29 |
| matern5_2 | nugget | LL | branin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0471 | 1 / 1 | 0.0837 | 0.333 | 2.85 |
| matern5_2 | nugget | LL | branin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0471 | 1 / 1 | 0.127 | 0.501 | 3.99 |
| matern5_2 | nugget | LL | branin4 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.031 | 1 / 1.01 | 0.681 | 2.71 | 22.1 |
| matern5_2 | nugget | LL | branin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0306 | 1 / 1 | 0.599 | 2.36 | 19.7 |
| matern5_2 | nugget | LL | branin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0306 | 1 / 1 | 0.805 | 2.98 | 26.1 |
| matern5_2 | nugget | LL | branin4 | 500 | BFGS:default | 30 | 28 | 93 [79, 98] | 0.0229 | 1.3 / 1.39 | 1.45 | 5.75 | 48.3 |
| matern5_2 | nugget | LL | branin4 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.018 | 1 / 1 | 1.7 | 5.95 | 56.3 |
| matern5_2 | nugget | LL | branin4 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.018 | 1 / 1 | 3 | 10.2 | 92.5 |
| matern5_2 | nugget | LL | hartmann6 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.623 | 1 / 1 | 0.0922 | 0.368 | 2.62 |
| matern5_2 | nugget | LL | hartmann6 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.623 | 1 / 1 | 0.127 | 0.501 | 3.81 |
| matern5_2 | nugget | LL | hartmann6 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.623 | 1 / 1 | 0.183 | 0.704 | 5.76 |
| matern5_2 | nugget | LL | hartmann6 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.463 | 1 / 1 | 0.699 | 2.78 | 23.4 |
| matern5_2 | nugget | LL | hartmann6 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.463 | 1 / 1 | 0.665 | 2.41 | 22.3 |
| matern5_2 | nugget | LL | hartmann6 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.463 | 1 / 1 | 1.01 | 3.63 | 30.4 |
| matern5_2 | nugget | LL | hartmann6 | 500 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.274 | 1 / 1 | 1.31 | 5.2 | 40.9 |
| matern5_2 | nugget | LL | hartmann6 | 500 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.274 | 1 / 1 | 1.99 | 7.05 | 60.1 |
| matern5_2 | nugget | LL | hartmann6 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.274 | 1 / 1 | 4.05 | 14 | 122 |
| matern5_2 | nugget | LL | sin4 | 100 | BFGS:default | 30 | 9 | 30 [17, 48] | 0.684 | 1.04 / 5.53 | 0.0818 | 0.325 | 2.86 |
| matern5_2 | nugget | LL | sin4 | 100 | BFGS4:default | 30 | 6 | 20 [10, 37] | 0.579 | 1 / 3.98 | 0.104 | 0.407 | 3.28 |
| matern5_2 | nugget | LL | sin4 | 100 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.147 | 1 / 1 | 0.16 | 0.604 | 4.95 |
| matern5_2 | nugget | LL | sin4 | 200 | BFGS:default | 30 | 21 | 70 [52, 83] | 0.0974 | 1.37 / 11.8 | 1.03 | 4.06 | 42.6 |
| matern5_2 | nugget | LL | sin4 | 200 | BFGS4:default | 30 | 8 | 27 [14, 44] | 0.0696 | 1.03 / 1.29 | 0.732 | 2.44 | 32.3 |
| matern5_2 | nugget | LL | sin4 | 200 | BFGS10:default | 30 | 4 | 13 [5, 30] | 0.0691 | 1 / 1.13 | 0.956 | 3.5 | 42.5 |
| matern5_2 | nugget | LL | sin4 | 500 | BFGS:default | 30 | 11 | 37 [22, 54] | 0.0288 | 1 / 1.06 | 1.29 | 5.13 | 47.8 |
| matern5_2 | nugget | LL | sin4 | 500 | BFGS4:default | 30 | 5 | 17 [7, 34] | 0.0283 | 1 / 1.02 | 1.93 | 6.6 | 67.5 |
| matern5_2 | nugget | LL | sin4 | 500 | BFGS10:default | 30 | 2 | 7 [2, 21] | 0.0279 | 1 / 1 | 4.28 | 14.3 | 129 |
| matern5_2 | nugget | LL | sinhalf8 | 100 | BFGS:default | 30 | 7 | 23 [12, 41] | 0.716 | 1.06 / 2.08 | 0.132 | 0.525 | 4.21 |
| matern5_2 | nugget | LL | sinhalf8 | 100 | BFGS4:default | 30 | 2 | 7 [2, 21] | 0.673 | 1 / 1.4 | 0.163 | 0.641 | 5.13 |
| matern5_2 | nugget | LL | sinhalf8 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.419 | 1 / 1 | 0.246 | 0.91 | 7.58 |
| matern5_2 | nugget | LL | sinhalf8 | 200 | BFGS:default | 30 | 10 | 33 [19, 51] | 0.281 | 1 / 3.1 | 1.19 | 4.69 | 35.9 |
| matern5_2 | nugget | LL | sinhalf8 | 200 | BFGS4:default | 30 | 5 | 17 [7, 34] | 0.2 | 1 / 1.67 | 1.06 | 4.06 | 33.7 |
| matern5_2 | nugget | LL | sinhalf8 | 200 | BFGS10:default | 30 | 2 | 7 [2, 21] | 0.187 | 1 / 1 | 1.39 | 5.14 | 45.1 |
| matern5_2 | nugget | LL | sinhalf8 | 500 | BFGS:default | 30 | 8 | 27 [14, 44] | 0.0588 | 1 / 1.09 | 3.12 | 12.4 | 92.8 |
| matern5_2 | nugget | LL | sinhalf8 | 500 | BFGS4:default | 30 | 2 | 7 [2, 21] | 0.0588 | 1 / 1.02 | 3.92 | 14.5 | 120 |
| matern5_2 | nugget | LL | sinhalf8 | 500 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0587 | 1 / 1 | 6.09 | 21.6 | 186 |
| matern5_2 | nugget | LMP | branin4 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.0459 | 1 / 1.06 | 0.0848 | 0.334 | 2.9 |
| matern5_2 | nugget | LMP | branin4 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0459 | 1 / 1 | 0.115 | 0.456 | 3.78 |
| matern5_2 | nugget | LMP | branin4 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0459 | 1 / 1 | 0.218 | 0.776 | 6.89 |
| matern5_2 | nugget | LMP | branin4 | 200 | BFGS:default | 30 | 1 | 3 [1, 17] | 0.0305 | 1 / 1.06 | 0.773 | 3.06 | 25.2 |
| matern5_2 | nugget | LMP | branin4 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.0304 | 1 / 1 | 0.85 | 3.05 | 25.8 |
| matern5_2 | nugget | LMP | branin4 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.0304 | 1 / 1 | 1.33 | 4.62 | 41.1 |
| matern5_2 | nugget | LMP | hartmann6 | 100 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.636 | 1 / 1 | 0.115 | 0.452 | 3.46 |
| matern5_2 | nugget | LMP | hartmann6 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.64 | 1 / 1 | 0.175 | 0.671 | 5.44 |
| matern5_2 | nugget | LMP | hartmann6 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.64 | 1 / 1 | 0.317 | 1.1 | 9.68 |
| matern5_2 | nugget | LMP | hartmann6 | 200 | BFGS:default | 30 | 0 | 0 [0, 11] | 0.459 | 1 / 1 | 0.654 | 2.6 | 21.1 |
| matern5_2 | nugget | LMP | hartmann6 | 200 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.462 | 1 / 1 | 1.01 | 3.08 | 31.7 |
| matern5_2 | nugget | LMP | hartmann6 | 200 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.462 | 1 / 1 | 1.9 | 6.56 | 56.9 |
| matern5_2 | nugget | LMP | sin4 | 100 | BFGS:default | 30 | 13 | 43 [27, 61] | 0.614 | 1.14 / 5.19 | 0.111 | 0.437 | 3.44 |
| matern5_2 | nugget | LMP | sin4 | 100 | BFGS4:default | 30 | 6 | 20 [10, 37] | 0.331 | 1 / 4.65 | 0.155 | 0.605 | 4.57 |
| matern5_2 | nugget | LMP | sin4 | 100 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.128 | 1 / 1 | 0.253 | 0.908 | 7.66 |
| matern5_2 | nugget | LMP | sin4 | 200 | BFGS:default | 30 | 26 | 87 [70, 95] | 0.109 | 1.74 / 11.3 | 0.986 | 3.91 | 41.9 |
| matern5_2 | nugget | LMP | sin4 | 200 | BFGS4:default | 30 | 14 | 47 [30, 64] | 0.0689 | 1.08 / 1.36 | 1.11 | 3.54 | 49.9 |
| matern5_2 | nugget | LMP | sin4 | 200 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.0657 | 1 / 1 | 1.71 | 6.1 | 67 |
| matern5_2 | nugget | LMP | sinhalf8 | 100 | BFGS:default | 30 | 6 | 20 [10, 37] | 0.845 | 1.09 / 1.34 | 0.152 | 0.598 | 4.73 |
| matern5_2 | nugget | LMP | sinhalf8 | 100 | BFGS4:default | 30 | 0 | 0 [0, 11] | 0.75 | 1 / 1 | 0.206 | 0.82 | 6.49 |
| matern5_2 | nugget | LMP | sinhalf8 | 100 | BFGS10:default | 30 | 0 | 0 [0, 11] | 0.738 | 1 / 1 | 0.315 | 1.16 | 9.61 |
| matern5_2 | nugget | LMP | sinhalf8 | 200 | BFGS:default | 30 | 8 | 27 [14, 44] | 0.342 | 1 / 3.13 | 1.17 | 4.63 | 39.2 |
| matern5_2 | nugget | LMP | sinhalf8 | 200 | BFGS4:default | 30 | 3 | 10 [3, 26] | 0.177 | 1 / 1.06 | 1.52 | 5.1 | 46.3 |
| matern5_2 | nugget | LMP | sinhalf8 | 200 | BFGS10:default | 30 | 1 | 3 [1, 17] | 0.177 | 1 / 1 | 2.09 | 7.88 | 64.1 |

